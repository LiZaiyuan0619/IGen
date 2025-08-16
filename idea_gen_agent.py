"""
Idea Generation 多智能体系统（骨架定义）

脚本目标:
    - 在不改动现有 Survey Gen 代码的前提下，基于其产物(final_result 与 enriched_outline)与可检索数据库，
      提供“机会图谱→想法生成→新颖性/可行性评审→精炼迭代”的多智能体架构骨架。

上下文:
    - 上游: `ma_gen.generate_survey` 已产出 `final_result`(Markdown整合正文等) 与 `enriched_outline`(JSON大纲)。
    - 下游: 本文件只提供类与函数签名及实现指引，便于工程师逐步填充具体实现。
    - 运行环境: 复用 `multi_agent.py` 中的 LLMFactory、AcademicPaperDatabase 等基础设施。

输入:
    - final_result: Dict，至少包含 `full_document`(Markdown正文完整字符串)、可选 `bibliography/figures/tables/equations/statistics` 等。
    - enriched_outline: Dict，章节级 `keywords/research_focus/content_guide/subsections` 等写作指引。
    - db: AcademicPaperDatabase，可通过 `db.search_content(query, content_type="texts", n_results=K)` 进行RAG检索。

执行步骤(高层):
    1) IdeaMiner 构建语义机会图谱 SemanticOpportunityGraph（抽取实体与关系，定位空洞/冲突）。
    2) IdeaGenerator 在图谱上触发模式并应用策略模板，生成 CandidateIdea 列表（默认并发/数量=6）。
    3) NoveltyCritic 基于 RAG 的两阶段检索与重排，给出新颖性评审。
    4) FeasibilityCritic 结合图谱资源约束，给出可行性评审。
    5) IdeaRefiner 汇总批判并下达可执行的精炼指令，进入迭代辩论闭环，直至收敛或达到上限。

输出:
    - 结构化的结果对象，包括：机会图谱、候选想法、评审记录、精炼指令、迭代后的最终想法集与轨迹。

注意事项:
    - 本文件仅为“骨架”，不包含实际算法与LLM提示词；请在各方法内按docstring提供的实现思路补充。
    - 保持数据可追溯：所有结论尽可能绑定 `provenance` 证据锚点（source/loc/paper_id/quote）。
    - 并发建议: 默认并发=6（与既有Writer并发习惯一致）。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import networkx as nx
from datetime import datetime

# 复用既有基础设施
from multi_agent import LLMFactory, AcademicPaperDatabase, AgentConfig, ModelType


# =========================
# 数据结构定义（骨架）
# =========================

class GraphNodeType(Enum):
    """语义机会图谱中的节点类型。

    - Method: 方法/模型/算法
    - Task: 任务/问题定义
    - Dataset: 数据集
    - Metric: 评价指标
    - Paper: 论文/证据源
    - Problem: 研究难题/挑战
    - Domain: 领域/子领域
    """

    METHOD = "Method"
    TASK = "Task"
    DATASET = "Dataset"
    METRIC = "Metric"
    PAPER = "Paper"
    PROBLEM = "Problem"
    DOMAIN = "Domain"


@dataclass
class GraphNode:
    """图谱节点。

    字段:
        id: 唯一ID，建议前缀编码，如 "M:Transformer"、"T:MT"。
        type: 节点类型 GraphNodeType。
        name: 规范化主名。
        aliases: 别名/缩写列表，便于消歧与检索。
        evidence: 证据列表，元素形如 {source, loc, quote, paper_id}。
        salience: 节点重要性(0~1)，可由频次/中心性/引用密度综合估计。
    """

    id: str
    type: GraphNodeType
    name: str
    aliases: List[str] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    salience: float = 0.0


class GraphEdgeRelation(Enum):
    """图谱语义关系类型。"""

    SOLVES = "solves"
    IMPROVES_ON = "improves_on"
    USES_DATASET = "uses_dataset"
    EVALUATED_BY = "evaluated_by"
    CRITIQUES = "critiques"
    CONTRADICTS = "contradicts"
    SIMILAR_TO = "similar_to"
    EXTENSION_OF = "extension_of"
    CAUSES = "causes"
    MITIGATES = "mitigates"


@dataclass
class GraphEdge:
    """图谱边。"""

    src: str
    dst: str
    relation: GraphEdgeRelation
    weight: float = 0.5
    confidence: float = 0.5
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    notes: Optional[str] = None


# 使用 NetworkX 作为语义机会图谱的底层实现
# 节点属性包含: type, name, aliases, evidence, salience
# 边属性包含: relation, weight, confidence, evidence, notes
# 图属性包含: gaps, provenance, indices
SemanticOpportunityGraph = nx.DiGraph


def create_semantic_graph() -> SemanticOpportunityGraph:
    """创建空的语义机会图谱。
    
    返回:
        SemanticOpportunityGraph: 初始化的有向图，包含空的全局属性。
    """
    graph = nx.DiGraph()
    # 初始化图的全局属性
    graph.graph['gaps'] = []
    graph.graph['provenance'] = {
        'created_at': datetime.now().isoformat(),
        'source': 'IdeaMinerAgent',
        'version': '1.0'
    }
    graph.graph['indices'] = {
        'by_type': {},
        'by_token': {}
    }
    return graph


def add_graph_node(graph: SemanticOpportunityGraph, node: GraphNode) -> None:
    """向图中添加节点。
    
    参数:
        graph: 目标图谱。
        node: 要添加的节点对象。
    """
    graph.add_node(
        node.id,
        type=node.type.value,
        name=node.name,
        aliases=node.aliases,
        evidence=node.evidence,
        salience=node.salience
    )
    
    # 更新类型索引
    node_type = node.type.value
    if node_type not in graph.graph['indices']['by_type']:
        graph.graph['indices']['by_type'][node_type] = []
    graph.graph['indices']['by_type'][node_type].append(node.id)
    
    # 更新token索引
    tokens = [node.name.lower()] + [alias.lower() for alias in node.aliases]
    for token in tokens:
        if token not in graph.graph['indices']['by_token']:
            graph.graph['indices']['by_token'][token] = []
        if node.id not in graph.graph['indices']['by_token'][token]:
            graph.graph['indices']['by_token'][token].append(node.id)


def add_graph_edge(graph: SemanticOpportunityGraph, edge: GraphEdge) -> None:
    """向图中添加边。
    
    参数:
        graph: 目标图谱。
        edge: 要添加的边对象。
    """
    graph.add_edge(
        edge.src,
        edge.dst,
        relation=edge.relation.value,
        weight=edge.weight,
        confidence=edge.confidence,
        evidence=edge.evidence,
        notes=edge.notes
    )


def find_nodes_by_type(graph: SemanticOpportunityGraph, node_type: GraphNodeType) -> List[str]:
    """根据类型查找节点。
    
    参数:
        graph: 目标图谱。
        node_type: 节点类型。
        
    返回:
        List[str]: 匹配的节点ID列表。
    """
    return graph.graph['indices']['by_type'].get(node_type.value, [])


def find_nodes_by_token(graph: SemanticOpportunityGraph, token: str) -> List[str]:
    """根据名称/别名token查找节点。
    
    参数:
        graph: 目标图谱。
        token: 搜索token（不区分大小写）。
        
    返回:
        List[str]: 匹配的节点ID列表。
    """
    return graph.graph['indices']['by_token'].get(token.lower(), [])


def add_opportunity_gap(graph: SemanticOpportunityGraph, gap: Dict[str, Any]) -> None:
    """向图中添加机会缺口。
    
    参数:
        graph: 目标图谱。
        gap: 机会缺口描述字典。
    """
    graph.graph['gaps'].append(gap)


@dataclass
class CandidateIdea:
    """候选想法对象。"""

    id: str
    title: str
    core_hypothesis: str
    initial_innovation_points: List[str] = field(default_factory=list)
    source_trigger_nodes: List[str] = field(default_factory=list)
    expected_contribution: List[str] = field(default_factory=list)
    required_assets: List[Dict[str, Any]] = field(default_factory=list)
    preliminary_experiments: List[Dict[str, Any]] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)
    version: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为可序列化的字典。"""
        return {
            'id': self.id,
            'title': self.title,
            'core_hypothesis': self.core_hypothesis,
            'initial_innovation_points': self.initial_innovation_points,
            'source_trigger_nodes': self.source_trigger_nodes,
            'expected_contribution': self.expected_contribution,
            'required_assets': self.required_assets,
            'preliminary_experiments': self.preliminary_experiments,
            'risks': self.risks,
            'provenance': self.provenance,
            'version': self.version
        }


@dataclass
class NoveltyCritique:
    """新颖性评审结果。"""

    idea_id: str
    novelty_score: float
    facet_scores: Dict[str, float] = field(default_factory=dict)
    similar_works: List[Dict[str, Any]] = field(default_factory=list)
    difference_claims: List[Dict[str, Any]] = field(default_factory=list)
    method: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为可序列化的字典。"""
        return {
            'idea_id': self.idea_id,
            'novelty_score': self.novelty_score,
            'facet_scores': self.facet_scores,
            'similar_works': self.similar_works,
            'difference_claims': self.difference_claims,
            'method': self.method
        }


@dataclass
class FeasibilityCritique:
    """可行性评审结果。"""

    idea_id: str
    feasibility_score: float
    relevance: str = ""
    required_assets: List[Dict[str, Any]] = field(default_factory=list)
    potential_risks: List[Dict[str, Any]] = field(default_factory=list)
    graph_checks: Dict[str, Any] = field(default_factory=dict)
    dimension_scores: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为可序列化的字典。"""
        return {
            'idea_id': self.idea_id,
            'feasibility_score': self.feasibility_score,
            'relevance': self.relevance,
            'required_assets': self.required_assets,
            'potential_risks': self.potential_risks,
            'graph_checks': self.graph_checks,
            'dimension_scores': self.dimension_scores
        }


@dataclass
class RefinementPrompt:
    """精炼指令。"""

    idea_id: str
    decision: str  # revise | split | merge | discard | accept
    instructions: List[str] = field(default_factory=list)
    rationale: str = ""
    acceptance_criteria: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为可序列化的字典。"""
        return {
            'idea_id': self.idea_id,
            'decision': self.decision,
            'instructions': self.instructions,
            'rationale': self.rationale,
            'acceptance_criteria': self.acceptance_criteria
        }


# =========================
# 基类与智能体（骨架）
# =========================


class BaseIdeaAgent:
    """Idea生成子系统的智能体基类。

    目标:
        - 统一注入 LLMFactory 与 AcademicPaperDatabase，提供基本工具/配置的持有。

    输入:
        - llm_factory: 统一的LLM调用工厂。
        - db: 学术数据库实例。
        - config: 智能体本地配置（温度、检索K、并发等）。

    注意事项:
        - 只定义接口与依赖，不实现特定业务逻辑。
    """

    def __init__(self, name: str, llm_factory: LLMFactory, db: AcademicPaperDatabase, config: Optional[AgentConfig] = None):
        self.name = name
        self.llm = llm_factory
        self.db = db
        self.config = config


class IdeaMinerAgent(BaseIdeaAgent):
    """第一阶段：机会图谱构建。

    核心职责:
        - 从 `final_result.full_document` 与 `enriched_outline` 抽取实体与关系，构建 `SemanticOpportunityGraph`。
        - 识别"结构空洞/冲突/断裂群集"等机会信号，写入 `gaps`。

    关键实现思路:
        - 实体抽取: 结合规则/LLM对正文与关键词进行标注；对同义/别名进行规范化。
        - 关系抽取: 按模板识别 solves/improves_on/uses_dataset/evaluated_by/critiques/...；记录证据锚点与置信度。
        - 图谱整理: 去重合并、构造索引(byType/byToken)、计算salience(频次/中心性/引用密度)。
        - 机会检测: 基于相似度与结构搜索定位迁移/组合/反转等触发模式缺口。

    注意事项:
        - 严格保留 `provenance` 以便追溯；冲突边不应被清洗。
        - 可按章节/子章节批量化处理，降低上下文窗口压力。
    """

    async def build_opportunity_graph(self, final_result: Dict[str, Any], enriched_outline: Dict[str, Any]) -> SemanticOpportunityGraph:
        """构建语义机会图谱。

        输入:
            - final_result: 上游整合结果，至少包含 `full_document`。
            - enriched_outline: 丰富大纲（含章节关键词、研究重点等）。

        输出:
            - SemanticOpportunityGraph: 含 nodes/edges/indices/gaps/provenance 的完整图谱对象。

        实现步骤建议:
            1) 解析 `full_document`，分章节/段落切片，进行实体候选抽取。
            2) 结合 `enriched_outline` 的关键词/研究重点，做实体规范化与类型判定。
            3) 基于句法/模板/LLM判定关系，并聚合证据形成边，估计 weight/confidence。
            4) 构建 indices 与 provenance，执行机会检测，填充 gaps。
        """
        # 创建空图谱
        graph = create_semantic_graph()
        
        # 分步骤构建
        await self._extract_entities(graph, final_result, enriched_outline)
        await self._extract_relations(graph, final_result, enriched_outline)
        await self._compute_salience(graph)
        await self._detect_opportunities(graph)
        
        return graph

    async def _extract_entities(self, graph: SemanticOpportunityGraph, final_result: Dict[str, Any], enriched_outline: Dict[str, Any]) -> None:
        """第一步：实体抽取与规范化。
        
        输入:
            - graph: 目标图谱（将被就地修改）。
            - final_result: 综述正文等内容。
            - enriched_outline: 丰富大纲。
            
        实现思路:
            1) 从 `final_document` 按章节切片，使用 LLM 识别方法/任务/数据集/指标/问题等实体。
            2) 结合 `enriched_outline` 的关键词进行种子扩展与类型标注。
            3) 实体规范化：处理别名、缩写、同义词合并，避免重复节点。
            4) 调用 `add_graph_node` 将标准化实体加入图谱。
        
        注意事项:
            - 保留实体出现的文档位置作为 `evidence`。
            - 预设实体类型优先级：Method > Task > Dataset > Metric > Problem。
        """
        print("  🔍 开始实体抽取与规范化")
        
        # 步骤1：收集种子关键词
        seed_keywords = self._collect_seed_keywords(enriched_outline)
        print(f"    📋 收集到 {len(seed_keywords)} 个种子关键词")
        
        # 步骤2：从综述正文中切分章节并抽取实体
        full_document = final_result.get("full_document", "")
        chapters = self._split_document_by_chapters(full_document)
        print(f"    📄 将文档切分为 {len(chapters)} 个章节")
        
        # 步骤3：并行章节实体抽取
        import asyncio
        all_entities = []
        
        # 创建所有章节的并行任务
        tasks = []
        for chapter_num, chapter_content in chapters.items():
            task = self._extract_entities_from_chapter(
                chapter_num, chapter_content, seed_keywords
            )
            tasks.append(task)
        
        # 并行执行所有LLM调用
        if tasks:
            print(f"    🚀 启动 {len(tasks)} 个章节的并行实体抽取")
            chapter_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理结果
            for i, result in enumerate(chapter_results):
                if isinstance(result, Exception):
                    print(f"      ❌ 章节 {list(chapters.keys())[i]} 抽取失败: {str(result)}")
                else:
                    all_entities.extend(result)
        
        print(f"    🎯 共抽取 {len(all_entities)} 个原始实体")
        
        # 步骤4：实体规范化与去重
        normalized_entities = await self._normalize_and_deduplicate_entities(all_entities)
        print(f"    ✨ 规范化后保留 {len(normalized_entities)} 个实体")
        
        # 步骤5：添加到图谱
        for entity in normalized_entities:
            # 字符串到枚举的映射
            type_mapping = {
                "Method": GraphNodeType.METHOD,
                "Task": GraphNodeType.TASK,
                "Dataset": GraphNodeType.DATASET,
                "Metric": GraphNodeType.METRIC,
                "Paper": GraphNodeType.PAPER,
                "Problem": GraphNodeType.PROBLEM,
                "Domain": GraphNodeType.DOMAIN
            }
            
            entity_type_str = entity["type"]
            entity_type = type_mapping.get(entity_type_str, GraphNodeType.METHOD)  # 默认为METHOD
            
            node = GraphNode(
                id=entity["id"],
                type=entity_type,
                name=entity["name"],
                aliases=entity.get("aliases", []),
                evidence=entity.get("evidence", []),
                salience=entity.get("salience", 0.0)
            )
            add_graph_node(graph, node)
        
        print(f"    ✅ 成功添加 {len(normalized_entities)} 个节点到图谱")

    async def _extract_relations(self, graph: SemanticOpportunityGraph, final_result: Dict[str, Any], enriched_outline: Dict[str, Any]) -> None:
        """第二步：关系抽取与证据绑定。
        
        输入:
            - graph: 已有实体的图谱。
            - final_result: 综述正文等内容。
            - enriched_outline: 丰富大纲。
            
        实现思路:
            1) 为每对实体生成关系候选，基于距离、共现、句法模式等启发式过滤。
            2) 使用 LLM 或规则模板判定关系类型（solves/improves_on/uses_dataset/...）。
            3) 估计关系权重（基于证据强度/频次）与置信度（基于模型输出）。
            4) 调用 `add_graph_edge` 将关系边加入图谱。
        
        注意事项:
            - 记录每条边的文档证据（quote/loc）；冲突边（contradicts/critiques）需特别保留。
            - 可按章节并行处理，避免上下文窗口超限。
        """
        print("  🔗 开始关系抽取与证据绑定")
        
        # 获取已有的节点
        nodes = list(graph.nodes())
        print(f"    📊 图谱中共有 {len(nodes)} 个节点")
        
        # 步骤2：基于LLM的并行关系抽取
        pattern_relations = await self._extract_pattern_relations_parallel(graph, final_result)
        print(f"    📝 基于LLM抽取 {len(pattern_relations)} 个关系")
        
        # 步骤3：基于数据库检索的关系验证
        verified_relations = await self._verify_relations_with_database(pattern_relations)
        print(f"    ✅ 验证后保留 {len(verified_relations)} 个关系")
        
        # 步骤4：添加关系到图谱
        for relation in verified_relations:
            # 字符串到枚举的安全转换
            relation_str = relation["relation"]
            relation_mapping = {
                "improves_on": GraphEdgeRelation.IMPROVES_ON,
                "uses_dataset": GraphEdgeRelation.USES_DATASET,
                "evaluated_by": GraphEdgeRelation.EVALUATED_BY,
                "applies_to": GraphEdgeRelation.SOLVES,  # 映射到现有的solves
                "requires": GraphEdgeRelation.SIMILAR_TO,  # 映射到相似关系
                "enables": GraphEdgeRelation.EXTENSION_OF,  # 映射到扩展关系
                "addresses": GraphEdgeRelation.SOLVES,  # 映射到解决关系
                "similar_to": GraphEdgeRelation.SIMILAR_TO
            }
            
            relation_enum = relation_mapping.get(relation_str, GraphEdgeRelation.SIMILAR_TO)  # 默认相似关系
            
            edge = GraphEdge(
                src=relation["src"],
                dst=relation["dst"], 
                relation=relation_enum,
                weight=relation.get("weight", 0.5),
                confidence=relation.get("confidence", 0.5),
                evidence=relation.get("evidence", []),
                notes=relation.get("notes", "")
            )
            add_graph_edge(graph, edge)
        
        print(f"    ✅ 成功添加 {len(verified_relations)} 条边到图谱")

    async def _compute_salience(self, graph: SemanticOpportunityGraph) -> None:
        """第三步：计算节点重要性。
        
        输入:
            - graph: 已有节点和边的图谱。
            
        实现思路:
            1) 利用 NetworkX 计算中心性指标：度中心性、接近中心性、介数中心性、PageRank等。
            2) 结合节点频次（evidence数量）、引用密度等启发式指标。
            3) 加权融合得到 `salience` 分数，更新节点属性。
        
        注意事项:
            - 不同类型节点可使用不同权重：Method/Task 偏向中心性，Dataset/Metric 偏向频次。
        """
        print("  📊 开始计算节点重要性")
        
        if graph.number_of_nodes() == 0:
            print("    ⚠️ 图谱为空，跳过重要性计算")
            return
        
        # 计算各种中心性指标
        centrality_scores = {}
        
        if graph.number_of_edges() > 0:
            # 度中心性
            degree_centrality = nx.degree_centrality(graph)
            centrality_scores['degree'] = degree_centrality
            
            # PageRank
            try:
                pagerank = nx.pagerank(graph, weight='weight')
                centrality_scores['pagerank'] = pagerank
            except:
                centrality_scores['pagerank'] = {node: 0.0 for node in graph.nodes()}
            
            # 介数中心性（对于大图可能很慢，这里简化）
            if graph.number_of_nodes() < 100:
                betweenness = nx.betweenness_centrality(graph, weight='weight')
                centrality_scores['betweenness'] = betweenness
            else:
                centrality_scores['betweenness'] = {node: 0.0 for node in graph.nodes()}
        else:
            # 没有边的情况，所有中心性为0
            for metric in ['degree', 'pagerank', 'betweenness']:
                centrality_scores[metric] = {node: 0.0 for node in graph.nodes()}
        
        # 更新每个节点的salience
        for node_id in graph.nodes():
            node_data = graph.nodes[node_id]
            node_type = node_data.get('type', 'Method')
            evidence_count = len(node_data.get('evidence', []))
            
            # 基础分数：证据数量
            evidence_score = min(evidence_count * 0.1, 1.0)
            
            # 中心性分数
            degree_score = centrality_scores['degree'].get(node_id, 0.0)
            pagerank_score = centrality_scores['pagerank'].get(node_id, 0.0) * 10  # 放大PageRank
            betweenness_score = centrality_scores['betweenness'].get(node_id, 0.0)
            
            # 根据节点类型调整权重
            if node_type in ['Method', 'Task']:
                # 方法和任务偏向中心性
                salience = (
                    0.3 * evidence_score + 
                    0.3 * degree_score + 
                    0.3 * pagerank_score + 
                    0.1 * betweenness_score
                )
            elif node_type in ['Dataset', 'Metric']:
                # 数据集和指标偏向频次
                salience = (
                    0.6 * evidence_score + 
                    0.2 * degree_score + 
                    0.2 * pagerank_score
                )
            else:
                # 其他类型使用平均权重
                salience = (
                    0.4 * evidence_score + 
                    0.3 * degree_score + 
                    0.3 * pagerank_score
                )
            
            # 更新节点属性
            graph.nodes[node_id]['salience'] = min(salience, 1.0)
        
        print(f"    ✅ 完成 {graph.number_of_nodes()} 个节点的重要性计算")

    async def _detect_opportunities(self, graph: SemanticOpportunityGraph) -> None:
        """第四步：机会检测与gap识别。
        
        输入:
            - graph: 完整构建的图谱。
            
        实现思路:
            1) 迁移机会：高相似Task间缺少Method迁移边。
            2) 组合机会：多个Method解决同Task的不同子问题。
            3) 反转机会：存在critique关系，但缺少改进方案。
            4) 评测空缺：Task缺少合适Metric，或Metric过时。
            5) 数据增强：Task有Method但缺少足够Dataset。
        
        注意事项:
            - 利用 NetworkX 的路径查找、子图检测等算法辅助gap发现。
            - 每个gap记录触发模式、相关节点、置信度评估。
        """
        print("  🔍 开始机会检测与gap识别")
        
        gaps = []
        
        # 获取不同类型的节点
        methods = find_nodes_by_type(graph, GraphNodeType.METHOD)
        tasks = find_nodes_by_type(graph, GraphNodeType.TASK)
        datasets = find_nodes_by_type(graph, GraphNodeType.DATASET)
        metrics = find_nodes_by_type(graph, GraphNodeType.METRIC)
        
        print(f"    📊 节点统计: {len(methods)}个方法, {len(tasks)}个任务, {len(datasets)}个数据集, {len(metrics)}个指标")
        
        # 机会1：迁移机会检测
        transfer_gaps = self._detect_transfer_opportunities(graph, methods, tasks)
        gaps.extend(transfer_gaps)
        print(f"    🔄 发现 {len(transfer_gaps)} 个迁移机会")
        
        # 机会2：组合机会检测
        composition_gaps = self._detect_composition_opportunities(graph, methods, tasks)
        gaps.extend(composition_gaps)
        print(f"    🔗 发现 {len(composition_gaps)} 个组合机会")
        
        # 机会3：反转机会检测
        reverse_gaps = self._detect_reverse_opportunities(graph)
        gaps.extend(reverse_gaps)
        print(f"    🔄 发现 {len(reverse_gaps)} 个反转机会")
        
        # 机会4：评测空缺检测
        evaluation_gaps = self._detect_evaluation_gaps(graph, tasks, metrics)
        gaps.extend(evaluation_gaps)
        print(f"    📏 发现 {len(evaluation_gaps)} 个评测空缺")
        
        # 机会5：数据增强机会检测
        data_gaps = self._detect_data_enhancement_opportunities(graph, tasks, datasets)
        gaps.extend(data_gaps)
        print(f"    📊 发现 {len(data_gaps)} 个数据增强机会")
        
        # 将gaps添加到图谱
        for gap in gaps:
            add_opportunity_gap(graph, gap)
        
        print(f"    ✅ 总共识别出 {len(gaps)} 个机会gap")

    def _collect_seed_keywords(self, enriched_outline: Dict[str, Any]) -> List[str]:
        """从丰富大纲中收集种子关键词。"""
        keywords = []
        
        # 从章节结构中提取关键词
        # enriched_outline本身就是parsed_structure，不需要再嵌套获取
        chapters = enriched_outline.get("chapters", {})
        
        for chapter_id, chapter in chapters.items():
            # 章节关键词
            chapter_keywords = chapter.get("keywords", [])
            keywords.extend(chapter_keywords)
            
            # 研究重点中的关键概念
            research_focus = chapter.get("research_focus", [])
            for focus in research_focus:
                # 简单提取：使用常见分隔符分割
                focus_keywords = self._extract_keywords_from_text(focus)
                keywords.extend(focus_keywords)
            
            # 子章节关键词
            subsections = chapter.get("subsections", {})
            for subsection_id, subsection in subsections.items():
                sub_keywords = subsection.get("key_points", [])
                for point in sub_keywords:
                    point_keywords = self._extract_keywords_from_text(point)
                    keywords.extend(point_keywords)
        
        # 去重并过滤
        unique_keywords = list(set(keywords))
        filtered_keywords = [kw for kw in unique_keywords if len(kw.strip()) > 2]
        
        return filtered_keywords[:100]  # 限制数量
    
    def _extract_keywords_from_text(self, text: str) -> List[str]:
        """从文本中提取关键术语。"""
        import re
        
        # 简单的关键词提取规则
        keywords = []
        
        # 提取专有名词和技术术语（大写字母开头的词组）
        patterns = [
            r'\b[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*\b',  # 专有名词
            r'\b[A-Z]{2,}\b',  # 缩写
            r'\b\w*(?:Model|Network|Transformer|BERT|GPT|LLM)\w*\b',  # 模型相关
            r'\b\w*(?:Dataset|Benchmark|Task|Metric)\w*\b',  # 数据相关
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            keywords.extend(matches)
        
        return [kw.strip() for kw in keywords if len(kw.strip()) > 2]
    
    def _split_document_by_chapters(self, full_document: str) -> Dict[str, str]:
        """将完整文档按章节切分。"""
        import re
        
        chapters = {}
        
        # 使用正则表达式匹配章节标题
        chapter_pattern = r'^#+\s*(\d+(?:\.\d+)*)\s+(.+)$'
        lines = full_document.split('\n')
        
        current_chapter = None
        current_content = []
        
        for line in lines:
            match = re.match(chapter_pattern, line.strip())
            if match:
                # 保存前一章节
                if current_chapter and current_content:
                    chapters[current_chapter] = '\n'.join(current_content)
                
                # 开始新章节
                chapter_num = match.group(1)
                if '.' not in chapter_num:  # 只处理一级章节
                    current_chapter = chapter_num
                    current_content = [line]
                else:
                    if current_chapter:
                        current_content.append(line)
            else:
                if current_chapter:
                    current_content.append(line)
        
        # 保存最后一章
        if current_chapter and current_content:
            chapters[current_chapter] = '\n'.join(current_content)
        
        return chapters
    
    async def _extract_entities_from_chapter(self, chapter_num: str, chapter_content: str, seed_keywords: List[str]) -> List[Dict[str, Any]]:
        """从单个章节中抽取实体。"""
        entities = []
        
        # 使用规则 + LLM 的方式抽取实体
        
        # 规则1：从种子关键词中匹配
        for keyword in seed_keywords:
            if keyword.lower() in chapter_content.lower():
                entity_type = self._classify_entity_type(keyword)
                entities.append({
                    "name": keyword,
                    "type": entity_type,
                    "source": "seed_keyword",
                    "chapter": chapter_num,
                    "confidence": 0.8,
                    "evidence": [{
                        "source": "survey",
                        "loc": f"ch{chapter_num}",
                        "quote": self._extract_context_around_keyword(chapter_content, keyword)
                    }]
                })
        
        # 规则2：模式匹配常见实体类型
        pattern_entities = self._extract_entities_by_patterns(chapter_content, chapter_num)
        entities.extend(pattern_entities)
        
        # 规则3：使用 LLM 进一步抽取（简化版，实际可调用 self.llm）
        # TODO: 这里可以调用 LLM 进行更精确的实体识别
        llm_entities = await self._extract_entities_with_llm(chapter_content, chapter_num)
        entities.extend(llm_entities)
        
        return entities
    
    def _classify_entity_type(self, keyword: str) -> str:
        """根据关键词分类实体类型。"""
        keyword_lower = keyword.lower()
        
        # 任务相关 - 扩展关键词
        if any(term in keyword_lower for term in [
            'task', 'generation', 'translation', 'classification', 'qa', 'question answering',
            'summarization', 'reasoning', 'understanding', 'inference', 'prediction',
            'fine-tuning', 'training', 'learning', 'alignment', 'evaluation',
            'processing', 'analysis', 'synthesis', 'optimization', 'improvement',
            'enhancement', 'adaptation', 'personalization', 'customization'
        ]):
            return "Task"
        
        # 评价指标相关 - 扩展关键词
        if any(term in keyword_lower for term in [
            'metric', 'score', 'bleu', 'rouge', 'accuracy', 'perplexity',
            'precision', 'recall', 'f1', 'performance', 'quality', 'effectiveness',
            'efficiency', 'benchmarking', 'evaluation', 'assessment', 'measurement'
        ]):
            return "Metric"
        
        # 数据集相关
        if any(term in keyword_lower for term in [
            'dataset', 'corpus', 'benchmark', 'data', 'collection', 'training data'
        ]):
            return "Dataset"
        
        # 问题/挑战相关
        if any(term in keyword_lower for term in [
            'problem', 'challenge', 'issue', 'bias', 'hallucination',
            'limitation', 'difficulty', 'barrier', 'obstacle'
        ]):
            return "Problem"
        
        # 方法/模型相关 - 放在最后，作为更具体的分类
        if any(term in keyword_lower for term in [
            'model', 'transformer', 'bert', 'gpt', 'network', 'algorithm', 
            'attention', 'embedding', 'architecture', 'framework', 'approach',
            'technique', 'method', 'mechanism', 'strategy', 'procedure'
        ]):
            return "Method"
        
        # 检查是否可能是任务（基于动词模式）
        if any(keyword_lower.endswith(suffix) for suffix in ['ing', 'ion', 'ment', 'ance', 'ence']):
            return "Task"
        
        # 默认为方法类型
        return "Method"
    
    def _extract_context_around_keyword(self, text: str, keyword: str, context_size: int = 100) -> str:
        """提取关键词周围的上下文。"""
        import re
        
        # 找到关键词的位置
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        match = pattern.search(text)
        
        if match:
            start = max(0, match.start() - context_size)
            end = min(len(text), match.end() + context_size)
            context = text[start:end].strip()
            return f"...{context}..."
        
        return keyword
    
    def _extract_entities_by_patterns(self, chapter_content: str, chapter_num: str) -> List[Dict[str, Any]]:
        """使用模式匹配抽取实体。"""
        import re
        entities = []
        
        # 模式1：模型名称（通常是大写或首字母大写）
        model_patterns = [
            r'\b(?:GPT|BERT|T5|RoBERTa|ELECTRA|DeBERTa|ALBERT|DistilBERT|XLNet|Transformer|LLaMA|PaLM|ChatGPT|GPT-\d+|BERT-\w+)(?:-\w+)*\b',
            r'\b[A-Z][a-zA-Z]*(?:Model|Net|Former|Bert|GPT)\b'
        ]
        
        for pattern in model_patterns:
            matches = re.finditer(pattern, chapter_content, re.IGNORECASE)
            for match in matches:
                name = match.group()
                entities.append({
                    "name": name,
                    "type": "Method",
                    "source": "pattern_match",
                    "chapter": chapter_num,
                    "confidence": 0.7,
                    "evidence": [{
                        "source": "survey",
                        "loc": f"ch{chapter_num}",
                        "quote": self._extract_context_around_keyword(chapter_content, name)
                    }]
                })
        
        # 模式2：数据集名称
        dataset_patterns = [
            r'\b(?:GLUE|SuperGLUE|SQuAD|MNLI|CoLA|SST|QNLI|RTE|WNLI|MultiNLI|ImageNet|COCO|WikiText|BookCorpus|Common Crawl)\b',
            r'\b[A-Z][a-zA-Z]*(?:Dataset|Corpus|Benchmark|DB)\b'
        ]
        
        for pattern in dataset_patterns:
            matches = re.finditer(pattern, chapter_content, re.IGNORECASE)
            for match in matches:
                name = match.group()
                entities.append({
                    "name": name,
                    "type": "Dataset",
                    "source": "pattern_match",
                    "chapter": chapter_num,
                    "confidence": 0.7,
                    "evidence": [{
                        "source": "survey",
                        "loc": f"ch{chapter_num}",
                        "quote": self._extract_context_around_keyword(chapter_content, name)
                    }]
                })
        
        return entities
    
    async def _extract_entities_with_llm(self, chapter_content: str, chapter_num: str) -> List[Dict[str, Any]]:
        """使用 LLM 抽取实体。"""
        try:
            # 构造实体抽取提示词
            prompt = f"""
从以下学术文本中识别和抽取关键实体。请按照以下分类标准：

**实体类型**：
- Method: 方法、模型、算法、技术（如GPT、Transformer、BERT等）
- Task: 任务、应用、问题（如文本生成、机器翻译、情感分析等）
- Dataset: 数据集、语料库（如GLUE、SQuAD、ImageNet等）
- Metric: 评价指标（如BLEU、ROUGE、准确率、困惑度等）
- Problem: 挑战、问题、限制（如偏见、幻觉、泛化等）

**文本内容**：
{chapter_content[:30000]}  

请以JSON格式返回抽取的实体，每个实体包含：
- name: 实体名称
- type: 实体类型（Method/Task/Dataset/Metric/Problem）
- context: 实体在文本中的上下文（简短）
- confidence: 置信度（0-1）

示例格式：
{{
  "entities": [
    {{
      "name": "BERT",
      "type": "Method",
      "context": "BERT模型在多个NLP任务上表现出色",
      "confidence": 0.9
    }}
  ]
}}
"""

            # 调用LLM
            response_data = await self.llm.generate(
                model_name=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=30000,
                agent_name=self.name,
                task_type="entity_extraction"
            )
            response = response_data.get("content", "")
            
            # 解析LLM响应
            import json
            import re
            try:
                # 提取JSON代码块
                if "```json" in response:
                    # 使用正则表达式提取JSON内容
                    json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
                    if json_match:
                        json_content = json_match.group(1).strip()
                    else:
                        # 如果没找到完整的代码块，尝试从```json开始到最后
                        json_start = response.find("```json") + 7
                        json_content = response[json_start:].strip()
                        if json_content.endswith("```"):
                            json_content = json_content[:-3].strip()
                else:
                    json_content = response.strip()
                
                result = json.loads(json_content)
                llm_entities = []
                
                for entity_data in result.get("entities", []):
                    entity = {
                        "name": entity_data.get("name", "").strip(),
                        "type": entity_data.get("type", "Method"),
                        "confidence": entity_data.get("confidence", 0.5),
                        "evidence": [{
                            "source": "llm_extraction",
                            "loc": f"ch{chapter_num}",
                            "quote": entity_data.get("context", ""),
                            "confidence": entity_data.get("confidence", 0.5)
                        }]
                    }
                    
                    if entity["name"] and len(entity["name"]) > 2:
                        llm_entities.append(entity)
                
                print(f"      🤖 LLM抽取到 {len(llm_entities)} 个实体")
                return llm_entities
                
            except json.JSONDecodeError:
                print(f"      ⚠️ LLM响应解析失败，跳过章节 {chapter_num}")
                return []
                
        except Exception as e:
            print(f"      ❌ LLM实体抽取失败: {str(e)}")
            return []
    
    async def _normalize_and_deduplicate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """实体规范化与去重。"""
        # 按名称分组
        entity_groups = {}
        for entity in entities:
            name = entity["name"].strip()
            normalized_name = self._normalize_entity_name(name)
            
            if normalized_name not in entity_groups:
                entity_groups[normalized_name] = []
            entity_groups[normalized_name].append(entity)
        
        # 合并同类实体
        normalized_entities = []
        entity_id_counter = 1
        
        for normalized_name, group in entity_groups.items():
            if not group:
                continue
            
            # 选择最佳代表实体
            representative = max(group, key=lambda x: x.get("confidence", 0))
            
            # 收集所有别名
            aliases = set()
            all_evidence = []
            
            for entity in group:
                aliases.add(entity["name"])
                all_evidence.extend(entity.get("evidence", []))
            
            aliases.discard(normalized_name)  # 移除主名称
            
            # 确定实体类型（取最常见的类型）
            type_counts = {}
            for entity in group:
                entity_type = entity["type"]
                type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
            
            most_common_type = max(type_counts.items(), key=lambda x: x[1])[0]
            
            # 构造规范化实体
            type_prefix = most_common_type[0]  # Method -> M, Task -> T, etc.
            entity_id = f"{type_prefix}:{normalized_name.replace(' ', '_')}"
            
            normalized_entity = {
                "id": entity_id,
                "name": normalized_name,
                "type": most_common_type,
                "aliases": list(aliases),
                "evidence": all_evidence,
                "salience": len(group) * 0.1,  # 简单的重要性计算
                "source_count": len(group)
            }
            
            normalized_entities.append(normalized_entity)
            entity_id_counter += 1
        
        return normalized_entities
    
    def _normalize_entity_name(self, name: str) -> str:
        """规范化实体名称。"""
        # 基本清理
        name = name.strip()
        
        # 处理常见的缩写和变体
        normalizations = {
            "GPT-3": "GPT-3",
            "GPT-4": "GPT-4", 
            "BERT": "BERT",
            "Transformer": "Transformer",
            "Large Language Model": "Large Language Model",
            "LLM": "Large Language Model",
            "Natural Language Processing": "Natural Language Processing",
            "NLP": "Natural Language Processing",
        }
        
        return normalizations.get(name, name)

    # ====== 关系抽取相关方法 ======
    
    async def _extract_cooccurrence_relations(self, graph: SemanticOpportunityGraph, full_document: str) -> List[Dict[str, Any]]:
        """基于共现模式抽取关系。"""
        relations = []
        nodes = list(graph.nodes())
        
        # 简化实现：检查节点对在文档中的共现
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i+1:]:
                node1_name = graph.nodes[node1]['name']
                node2_name = graph.nodes[node2]['name']
                
                # 检查两个实体是否在同一段落中共现
                if self._check_cooccurrence(full_document, node1_name, node2_name):
                    relation_type = self._infer_relation_type(
                        graph.nodes[node1]['type'], 
                        graph.nodes[node2]['type']
                    )
                    
                    if relation_type:
                        relations.append({
                            "src": node1,
                            "dst": node2,
                            "relation": relation_type,
                            "weight": 0.5,
                            "confidence": 0.6,
                            "evidence": [{
                                "source": "survey",
                                "loc": "cooccurrence",
                                "quote": f"{node1_name} and {node2_name} co-occur"
                            }]
                        })
        
        return relations
    
    def _check_cooccurrence(self, document: str, entity1: str, entity2: str, window_size: int = 200) -> bool:
        """检查两个实体是否在指定窗口内共现。"""
        import re
        
        # 忽略大小写查找实体位置
        pattern1 = re.compile(re.escape(entity1), re.IGNORECASE)
        pattern2 = re.compile(re.escape(entity2), re.IGNORECASE)
        
        matches1 = list(pattern1.finditer(document))
        matches2 = list(pattern2.finditer(document))
        
        # 检查是否有匹配在窗口内
        for match1 in matches1:
            for match2 in matches2:
                if abs(match1.start() - match2.start()) <= window_size:
                    return True
        
        return False
    
    def _infer_relation_type(self, type1: str, type2: str) -> Optional[str]:
        """根据实体类型推断可能的关系类型。"""
        # 方法 -> 任务
        if type1 == "Method" and type2 == "Task":
            return "solves"
        
        # 任务 -> 方法 (反向)
        if type1 == "Task" and type2 == "Method":
            return "solves"
        
        # 方法 -> 数据集
        if type1 == "Method" and type2 == "Dataset":
            return "uses_dataset"
        
        # 任务 -> 指标
        if type1 == "Task" and type2 == "Metric":
            return "evaluated_by"
        
        # 方法 -> 方法
        if type1 == "Method" and type2 == "Method":
            return "similar_to"
        
        return None
    
    async def _extract_pattern_relations(self, graph: SemanticOpportunityGraph, full_document: str) -> List[Dict[str, Any]]:
        """基于LLM的关系抽取。"""
        relations = []
        
        try:
            # 获取图谱中的所有节点
            nodes = list(graph.nodes())
            if len(nodes) < 2:
                return relations
            
            # 限制处理的节点数量，避免上下文过长
            node_names = [graph.nodes[node_id].get('name', node_id) for node_id in nodes[:50]]
            
            # 构造关系抽取提示词
            prompt = f"""
从以下学术文本中识别实体间的关系。已知实体列表：
{', '.join(node_names[:30])}

**关系类型**：
- improves_on: A改进了B、A优于B
- uses_dataset: A使用了数据集B进行训练/评估  
- evaluated_by: A通过指标B进行评估
- applies_to: A应用于任务B
- requires: A需要/依赖B
- enables: A使得B成为可能
- addresses: A解决了问题B

**文本片段**（前5000字符）：
{full_document[:30000]}

请以JSON格式返回识别的关系，格式如下：
{{
  "relations": [
    {{
      "src": "实体A名称", 
      "dst": "实体B名称",
      "relation": "关系类型",
      "evidence": "支撑该关系的文本证据",
      "confidence": 0.8
    }}
  ]
}}

只返回置信度>0.6且在已知实体列表中的关系。
"""

            # 调用LLM
            response_data = await self.llm.generate(
                model_name=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=15000,
                agent_name=self.name,
                task_type="relation_extraction"
            )
            response = response_data.get("content", "")
            
            # 解析LLM响应
            import json
            import re
            try:
                # 提取JSON代码块
                if "```json" in response:
                    # 使用正则表达式提取JSON内容
                    json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
                    if json_match:
                        json_content = json_match.group(1).strip()
                    else:
                        # 如果没找到完整的代码块，尝试从```json开始到最后
                        json_start = response.find("```json") + 7
                        json_content = response[json_start:].strip()
                        if json_content.endswith("```"):
                            json_content = json_content[:-3].strip()
                else:
                    json_content = response.strip()
                
                result = json.loads(json_content)
                
                # 创建节点名称到ID的映射
                name_to_id = {}
                for node_id in nodes:
                    node_data = graph.nodes[node_id]
                    node_name = node_data.get('name', node_id)
                    name_to_id[node_name.lower()] = node_id
                    
                    # 也映射别名
                    aliases = node_data.get('aliases', [])
                    for alias in aliases:
                        name_to_id[alias.lower()] = node_id
                
                for rel_data in result.get("relations", []):
                    src_name = rel_data.get("src", "").strip().lower()
                    dst_name = rel_data.get("dst", "").strip().lower()
                    relation_type = rel_data.get("relation", "")
                    confidence = rel_data.get("confidence", 0.0)
                    evidence = rel_data.get("evidence", "")
                    
                    # 检查实体是否在图谱中
                    src_id = name_to_id.get(src_name)
                    dst_id = name_to_id.get(dst_name)
                    
                    if src_id and dst_id and src_id != dst_id and confidence > 0.6:
                        relation = {
                            "src": src_id,
                            "dst": dst_id,
                            "relation": relation_type,
                            "confidence": confidence,
                            "evidence": [{
                                "source": "llm_extraction",
                                "quote": evidence,
                                "confidence": confidence
                            }]
                        }
                        relations.append(relation)
                
                print(f"      🤖 LLM抽取到 {len(relations)} 个关系")
                return relations
                
            except json.JSONDecodeError:
                print(f"      ⚠️ LLM关系抽取响应解析失败")
                return []
                
        except Exception as e:
            print(f"      ❌ LLM关系抽取失败: {str(e)}")
            return []
    
    async def _extract_pattern_relations_parallel(self, graph: SemanticOpportunityGraph, final_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """基于LLM的并行关系抽取。按章节分别处理，提高覆盖率和并行度。"""
        import asyncio
        
        # 获取文档章节
        full_document = final_result.get("full_document", "")
        chapters = self._split_document_by_chapters(full_document)
        
        if not chapters:
            return []
        
        # 创建所有章节的并行任务
        tasks = []
        for chapter_num, chapter_content in chapters.items():
            task = self._extract_relations_from_chapter(graph, chapter_num, chapter_content)
            tasks.append(task)
        
        # 并行执行所有LLM调用
        all_relations = []
        if tasks:
            print(f"    🚀 启动 {len(tasks)} 个章节的并行关系抽取")
            chapter_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理结果
            for i, result in enumerate(chapter_results):
                if isinstance(result, Exception):
                    print(f"      ❌ 章节 {list(chapters.keys())[i]} 关系抽取失败: {str(result)}")
                else:
                    all_relations.extend(result)
        
        return all_relations
    
    async def _extract_relations_from_chapter(self, graph: SemanticOpportunityGraph, chapter_num: str, chapter_content: str) -> List[Dict[str, Any]]:
        """从单个章节抽取关系。"""
        relations = []
        
        try:
            # 获取图谱中的所有节点
            nodes = list(graph.nodes())
            if len(nodes) < 2:
                return relations
            
            # 处理更多节点，不再限制为50个
            node_names = [graph.nodes[node_id].get('name', node_id) for node_id in nodes[:100]]
            
            # 构造关系抽取提示词
            prompt = f"""
从以下学术文本章节中识别实体间的关系。已知实体列表：
{', '.join(node_names[:100])}

**关系类型**：
- improves_on: A改进了B、A优于B
- uses_dataset: A使用了数据集B进行训练/评估  
- evaluated_by: A通过指标B进行评估
- applies_to: A应用于任务B
- requires: A需要/依赖B
- enables: A使得B成为可能
- addresses: A解决了问题B

**章节 {chapter_num} 内容**：
{chapter_content[:20000]}

请以JSON格式返回识别的关系，格式如下：
{{
  "relations": [
    {{
      "src": "实体A名称", 
      "dst": "实体B名称",
      "relation": "关系类型",
      "evidence": "支撑该关系的文本证据",
      "confidence": 0.8
    }}
  ]
}}

只返回置信度>0.6且在已知实体列表中的关系。
"""

            # 调用LLM
            response_data = await self.llm.generate(
                model_name=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=15000,
                agent_name=self.name,
                task_type="relation_extraction"
            )
            response = response_data.get("content", "")
            
            # 解析LLM响应
            import json
            import re
            try:
                # 提取JSON代码块
                if "```json" in response:
                    json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
                    if json_match:
                        json_content = json_match.group(1).strip()
                    else:
                        json_start = response.find("```json") + 7
                        json_content = response[json_start:].strip()
                        if json_content.endswith("```"):
                            json_content = json_content[:-3].strip()
                else:
                    json_content = response.strip()
                
                result = json.loads(json_content)
                
                # 创建节点名称到ID的映射
                name_to_id = {}
                for node_id in nodes:
                    node_data = graph.nodes[node_id]
                    node_name = node_data.get('name', node_id)
                    name_to_id[node_name.lower()] = node_id
                    
                    # 也映射别名
                    aliases = node_data.get('aliases', [])
                    for alias in aliases:
                        name_to_id[alias.lower()] = node_id
                
                for rel_data in result.get("relations", []):
                    src_name = rel_data.get("src", "").strip().lower()
                    dst_name = rel_data.get("dst", "").strip().lower()
                    relation_type = rel_data.get("relation", "")
                    confidence = rel_data.get("confidence", 0.0)
                    evidence = rel_data.get("evidence", "")
                    
                    # 检查实体是否在图谱中
                    src_id = name_to_id.get(src_name)
                    dst_id = name_to_id.get(dst_name)
                    
                    if src_id and dst_id and src_id != dst_id and confidence > 0.6:
                        relation = {
                            "src": src_id,
                            "dst": dst_id,
                            "relation": relation_type,
                            "confidence": confidence,
                            "evidence": [{
                                "source": "llm_extraction",
                                "quote": evidence,
                                "confidence": confidence
                            }]
                        }
                        relations.append(relation)
                
                return relations
                
            except json.JSONDecodeError:
                print(f"      ⚠️ 章节 {chapter_num} LLM关系抽取响应解析失败")
                return []
                
        except Exception as e:
            print(f"      ❌ 章节 {chapter_num} LLM关系抽取失败: {str(e)}")
            return []
    
    async def _verify_relations_with_database(self, relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """使用数据库验证关系。"""
        verified_relations = []
        
        for relation in relations:
            # 简化实现：目前只进行基本过滤
            if relation.get("confidence", 0) > 0.3:
                verified_relations.append(relation)
        
        return verified_relations
    
    # ====== 机会检测相关方法 ======
    
    def _detect_transfer_opportunities(self, graph: SemanticOpportunityGraph, methods: List[str], tasks: List[str]) -> List[Dict[str, Any]]:
        """检测迁移机会。"""
        gaps = []
        
        # 简化实现：寻找方法与任务之间缺失的连接
        for method in methods:
            connected_tasks = []
            for task in tasks:
                if graph.has_edge(method, task) or graph.has_edge(task, method):
                    connected_tasks.append(task)
            
            # 如果方法连接的任务少于总任务的一半，可能存在迁移机会
            if len(connected_tasks) < len(tasks) / 2 and len(connected_tasks) > 0:
                unconnected_tasks = [t for t in tasks if t not in connected_tasks]
                for unconnected_task in unconnected_tasks[:2]:  # 限制数量
                    gaps.append({
                        "pattern": "transfer",
                        "from_method": method,
                        "to_task": unconnected_task,
                        "explanation": f"方法{graph.nodes[method]['name']}可能适用于任务{graph.nodes[unconnected_task]['name']}",
                        "confidence": 0.7,
                        "related_nodes": [method, unconnected_task]
                    })
        
        return gaps
    
    def _detect_composition_opportunities(self, graph: SemanticOpportunityGraph, methods: List[str], tasks: List[str]) -> List[Dict[str, Any]]:
        """检测组合机会。"""
        gaps = []
        
        # 寻找解决同一任务的多个方法，可能可以组合
        for task in tasks:
            connected_methods = []
            for method in methods:
                if graph.has_edge(method, task) or graph.has_edge(task, method):
                    connected_methods.append(method)
            
            # 如果有多个方法解决同一任务，考虑组合机会
            if len(connected_methods) >= 2:
                for i, method1 in enumerate(connected_methods):
                    for method2 in connected_methods[i+1:]:
                        gaps.append({
                            "pattern": "composition",
                            "method1": method1,
                            "method2": method2,
                            "target_task": task,
                            "explanation": f"组合{graph.nodes[method1]['name']}和{graph.nodes[method2]['name']}可能改进{graph.nodes[task]['name']}",
                            "confidence": 0.6,
                            "related_nodes": [method1, method2, task]
                        })
                        break  # 限制每个任务只生成一个组合机会
        
        return gaps
    
    def _detect_reverse_opportunities(self, graph: SemanticOpportunityGraph) -> List[Dict[str, Any]]:
        """检测反转机会（基于critique关系）。"""
        gaps = []
        
        # 寻找critique边，生成改进机会
        for edge in graph.edges(data=True):
            src, dst, edge_data = edge
            if edge_data.get('relation') == 'critiques':
                gaps.append({
                    "pattern": "reverse",
                    "critiqued_method": dst,
                    "critique_source": src,
                    "explanation": f"针对{graph.nodes[dst]['name']}的批评，寻找改进方案",
                    "confidence": 0.8,
                    "related_nodes": [src, dst]
                })
        
        return gaps
    
    def _detect_evaluation_gaps(self, graph: SemanticOpportunityGraph, tasks: List[str], metrics: List[str]) -> List[Dict[str, Any]]:
        """检测评测空缺。"""
        gaps = []
        
        # 寻找缺少评价指标的任务
        for task in tasks:
            connected_metrics = []
            for metric in metrics:
                if graph.has_edge(task, metric) or graph.has_edge(metric, task):
                    connected_metrics.append(metric)
            
            # 如果任务没有或很少评价指标
            if len(connected_metrics) == 0:
                gaps.append({
                    "pattern": "evaluation",
                    "task": task,
                    "explanation": f"任务{graph.nodes[task]['name']}缺少合适的评价指标",
                    "confidence": 0.7,
                    "related_nodes": [task]
                })
        
        return gaps
    
    def _detect_data_enhancement_opportunities(self, graph: SemanticOpportunityGraph, tasks: List[str], datasets: List[str]) -> List[Dict[str, Any]]:
        """检测数据增强机会。"""
        gaps = []
        
        # 寻找数据集不足的任务
        for task in tasks:
            connected_datasets = []
            for dataset in datasets:
                if graph.has_edge(task, dataset) or graph.has_edge(dataset, task):
                    connected_datasets.append(dataset)
            
            # 如果任务缺少数据集
            if len(connected_datasets) <= 1:
                gaps.append({
                    "pattern": "data_enhancement",
                    "task": task,
                    "explanation": f"任务{graph.nodes[task]['name']}可能需要更多数据集支持",
                    "confidence": 0.6,
                    "related_nodes": [task]
                })
        
        return gaps


class IdeaGeneratorAgent(BaseIdeaAgent):
    """第二阶段：想法萌生与策略性生成。

    核心职责:
        - 在 `SemanticOpportunityGraph` 上扫描触发模式（迁移/组合/反转/评测重构/数据增强等）。
        - 应用可配置的策略模板与 Chain-of-Ideas 思维链，生成结构化 `CandidateIdea` 列表。

    注意事项:
        - 支持批量生成：每批10个，最多5批次，总共50个idea
        - 每个想法需绑定触发节点来源 `source_trigger_nodes` 与 `provenance`。
    """

    def __init__(self, name: str, llm_factory: LLMFactory, db: AcademicPaperDatabase, config: Optional[AgentConfig] = None):
        super().__init__(name, llm_factory, db, config)
        self.strategy_templates = self._load_strategy_templates()

    def _load_strategy_templates(self) -> Dict[str, Dict[str, Any]]:
        """加载策略模板库。
        
        返回:
            Dict[str, Dict]: 策略ID -> 策略配置的映射。
            
        实现思路:
            1) 预定义常见策略模板：transfer_across_tasks、compose_methods、reverse_critique等。
            2) 每个策略包含：触发条件、提示词槽位、检查项、示例等。
        """
        return {
            "transfer_across_tasks": {
                "trigger_pattern": "Method->Task_A strong; Task_B similar_to Task_A; no edge Method->Task_B",
                "prompt_slots": ["Method", "Task_A", "Task_B"],
                "checks": ["数据规模对齐", "评价指标可比", "归纳偏置合理性"],
                "example": "将Transformer从机器翻译迁移到代码生成"
            },
            "compose_methods": {
                "trigger_pattern": "Method_X solves SubZ1; Method_Y solves SubZ2; both under Task_Z",
                "prompt_slots": ["Method_X", "Method_Y", "Task_Z"],
                "checks": ["接口/表征可兼容", "训练/推理复杂度"],
                "example": "结合检索与生成改进问答系统"
            },
            "reverse_critique": {
                "trigger_pattern": "Paper critiques Method w/ facet=inefficiency",
                "prompt_slots": ["Method", "Critique_Facet"],
                "checks": ["等价替换或近似推断", "压缩/蒸馏/剪枝"],
                "example": "针对Transformer计算复杂度批评提出高效变种"
            }
        }

    async def generate_candidates(self, graph: SemanticOpportunityGraph, 
                                 max_ideas: int = 50, 
                                 ideas_per_generation: int = 10, 
                                 num_generations: int = 5) -> Dict[str, Any]:
        """策略性生成候选想法：从大量机会中智能筛选。

        输入:
            - graph: 已构建的 `SemanticOpportunityGraph`。
            - max_ideas: 生成数量上限（默认50）。
            - ideas_per_generation: 每次生成的想法数量（默认10）。
            - num_generations: 生成次数（默认5次）。

        输出:
            - Dict: {
                'all_candidates': List[CandidateIdea],  # 所有候选想法
                'batches': List[List[CandidateIdea]],   # 按生成批次组织的想法
                'generation_info': Dict                  # 生成过程信息
              }

        实现步骤:
            1) 识别大量机会触发点（目标50+个机会点）。
            2) 策略性生成：5次LLM调用，每次从50个机会中筛选10个最佳想法。
            3) 每次LLM调用都包含完整的机会分析和智能筛选过程。
            4) 保持批次结构，供后续批量评审使用。
        """
        print(f"🧠 开始策略性想法生成：目标{max_ideas}个想法，{num_generations}次生成，每次挑选{ideas_per_generation}个")
        
        # 步骤1：收集大量机会触发点
        all_opportunities = await self._collect_comprehensive_opportunities(graph)
        print(f"🎯 识别出 {len(all_opportunities)} 个机会触发点")
        
        if len(all_opportunities) < 30:
            print(f"⚠️ 机会点数量偏少({len(all_opportunities)})，可能影响策略性选择的质量")
        
        # 步骤2：将机会分批分配给不同轮次的生成
        opportunity_batches = self._allocate_opportunities_to_batches(all_opportunities, num_generations)
        print(f"🎯 机会分配：{len(all_opportunities)}个机会分为{len(opportunity_batches)}批，每批平均{len(all_opportunities)//num_generations}个")
        
        all_candidates = []
        idea_batches = []  # 保持批次结构
        
        # 步骤3：并发策略性生成 - 5个Agent同时从不同批次的机会中智能挑选
        print(f"🚀 启动{num_generations}个并发IdeaGenerator，每个处理不同的机会批次")
        
        # 创建并发任务
        import asyncio
        tasks = []
        valid_batches = []
        
        for generation_round in range(num_generations):
            current_opportunities = opportunity_batches[generation_round] if generation_round < len(opportunity_batches) else []
            
            if not current_opportunities:
                print(f"⚠️ 第{generation_round + 1}轮没有可用机会，跳过")
                continue
            
            print(f"🧠 第{generation_round + 1}轮：准备从第{generation_round + 1}批{len(current_opportunities)}个机会中智能挑选{ideas_per_generation}个")
            
            # 创建异步任务
            task = asyncio.create_task(
                self._strategic_idea_generation(current_opportunities, ideas_per_generation, generation_round, graph)
            )
            tasks.append(task)
            valid_batches.append(generation_round + 1)
        
        if not tasks:
            print("❌ 没有有效的生成任务，无法继续")
            idea_batches = []
        else:
            print(f"⚡ 开始并发执行{len(tasks)}个IdeaGenerator任务...")
            
            # 并发执行所有生成任务
            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # 处理并发结果
                for i, result in enumerate(results):
                    batch_num = valid_batches[i]
                    if isinstance(result, Exception):
                        print(f"⚠️ 第{batch_num}轮生成失败: {str(result)}")
                        idea_batches.append([])  # 添加空批次保持索引对应
                    else:
                        batch_candidates = result
                        idea_batches.append(batch_candidates)
                        all_candidates.extend(batch_candidates)
                        print(f"✅ 第{batch_num}轮完成，生成了{len(batch_candidates)}个高质量想法")
                
                print(f"🎉 并发生成完成！总共{len(tasks)}个任务，成功{len([r for r in results if not isinstance(r, Exception)])}个")
                
            except Exception as e:
                print(f"❌ 并发执行过程中出现错误: {str(e)}")
                idea_batches = []
        
        print(f"🎯 策略性生成完成：总共生成{len(all_candidates)}个想法，分为{len(idea_batches)}批")
        
        # 步骤3：去重和最终排序（但保持批次结构）
        final_candidates = await self._rank_and_filter(all_candidates, max_ideas)
        
        return {
            'all_candidates': final_candidates,
            'batches': idea_batches,
            'generation_info': {
                'num_generations': len(idea_batches),
                'ideas_per_generation': ideas_per_generation,
                'total_generated': len(all_candidates),
                'final_count': len(final_candidates),
                'opportunities_used': len(all_opportunities)
            }
        }

    async def _identify_triggers(self, graph: SemanticOpportunityGraph) -> List[Dict[str, Any]]:
        """识别图谱中的触发模式。
        
        输入:
            - graph: 语义机会图谱。
            
        输出:
            - List[Dict]: 触发信号列表，每个包含 pattern、nodes、strategy、confidence。
            
        实现思路:
            1) 遍历 graph.graph['gaps']，提取已检测的机会缺口。
            2) 补充实时模式匹配：利用 NetworkX 查找特定子图结构。
            3) 为每个trigger关联合适的策略模板，估计触发置信度。
        """
        triggers = []
        
        # 步骤1：从已检测的gaps中提取触发信号
        gaps = graph.graph.get('gaps', [])
        print(f"    🎯 从 {len(gaps)} 个gaps中提取触发信号")
        
        for gap in gaps:
            pattern = gap.get('pattern', 'unknown')
            confidence = gap.get('confidence', 0.5)
            
            # 根据gap模式映射到策略模板
            strategy = self._map_gap_to_strategy(pattern)
            
            if strategy:
                trigger = {
                    "pattern": pattern,
                    "nodes": gap.get('related_nodes', []),
                    "strategy": strategy,
                    "confidence": confidence,
                    "source": "gap_detection",
                    "gap_data": gap
                }
                triggers.append(trigger)
        
        # 步骤2：补充实时模式匹配
        additional_triggers = await self._find_additional_patterns(graph)
        triggers.extend(additional_triggers)
        
        # 步骤3：按置信度排序并限制数量
        triggers.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        return triggers
    
    async def _collect_comprehensive_opportunities(self, graph: SemanticOpportunityGraph) -> List[Dict[str, Any]]:
        """收集全面的机会点，为策略性生成做准备。
        
        输入:
            - graph: 语义机会图谱。
            
        输出:
            - List[Dict[str, Any]]: 详细的机会点列表。
        """
        opportunities = []
        
        # 1. 基础触发模式识别
        basic_triggers = await self._identify_triggers(graph)
        opportunities.extend(basic_triggers)
        
        # 2. 扩展机会识别 - 基于图谱结构深入挖掘
        extended_opportunities = await self._identify_extended_opportunities(graph)
        opportunities.extend(extended_opportunities)
        
        # 3. 组合机会 - 发现多个节点/边的复杂组合
        combination_opportunities = await self._identify_combination_opportunities(graph)
        opportunities.extend(combination_opportunities)
        
        # 4. 反向工程机会 - 从现有批评/问题中发现机会
        reverse_opportunities = await self._identify_reverse_engineering_opportunities(graph)
        opportunities.extend(reverse_opportunities)
        
        # 5. 跨领域迁移机会
        cross_domain_opportunities = await self._identify_cross_domain_opportunities(graph)
        opportunities.extend(cross_domain_opportunities)
        
        # 去重并丰富机会信息
        unique_opportunities = self._deduplicate_and_enrich_opportunities(opportunities, graph)
        
        # 用实际节点名称增强机会信息
        enhanced_opportunities = self._enhance_opportunities_with_node_names(unique_opportunities, graph)
        
        return enhanced_opportunities
    
    def _allocate_opportunities_to_batches(self, opportunities: List[Dict[str, Any]], 
                                         num_batches: int) -> List[List[Dict[str, Any]]]:
        """将机会智能分配到不同批次中，确保每批次的多样性和质量。"""
        if not opportunities:
            return [[] for _ in range(num_batches)]
        
        # 按重要性和类型对机会进行排序
        sorted_opportunities = sorted(opportunities, 
                                   key=lambda x: (x.get('confidence', 0), 
                                                x.get('opportunity_type', 'zzz')), 
                                   reverse=True)
        
        # 智能分配：轮流分配而不是简单切片，确保每批都有高质量机会
        batches = [[] for _ in range(num_batches)]
        
        for i, opportunity in enumerate(sorted_opportunities):
            batch_index = i % num_batches
            batches[batch_index].append(opportunity)
        
        # 确保每批次都有足够的机会（至少20个，如果总数允许）
        min_per_batch = max(20, len(opportunities) // (num_batches * 2))
        
        # 如果某些批次太小，从大批次中重新分配
        for i in range(num_batches):
            if len(batches[i]) < min_per_batch:
                # 找到最大的批次
                largest_batch_idx = max(range(num_batches), key=lambda x: len(batches[x]))
                if len(batches[largest_batch_idx]) > min_per_batch:
                    # 移动一些机会到小批次
                    need = min_per_batch - len(batches[i])
                    can_move = min(need, len(batches[largest_batch_idx]) - min_per_batch)
                    if can_move > 0:
                        moved_items = batches[largest_batch_idx][-can_move:]
                        batches[largest_batch_idx] = batches[largest_batch_idx][:-can_move]
                        batches[i].extend(moved_items)
        
        print(f"    📊 机会分配详情: {[len(batch) for batch in batches]}")
        return batches
    
    async def _strategic_idea_generation(self, opportunities: List[Dict[str, Any]], 
                                       num_ideas: int, generation_round: int, 
                                       graph: SemanticOpportunityGraph) -> List[CandidateIdea]:
        """策略性想法生成：让LLM从大量机会中智能挑选并生成想法。
        
        输入:
            - opportunities: 所有可用的机会点。
            - num_ideas: 要生成的想法数量。
            - generation_round: 当前生成轮次。
            - graph: 语义机会图谱。
            
        输出:
            - List[CandidateIdea]: 生成的候选想法列表。
        """
        print(f"    🎯 第{generation_round + 1}轮：从{len(opportunities)}个机会中策略性挑选{num_ideas}个idea")
        
        # 准备机会摘要，让LLM能够全面了解所有选择
        opportunities_summary = self._prepare_opportunities_summary(opportunities)
        
        # 构造策略性生成的prompt
        prompt = f"""
你是一位资深的科研想法策略师。现在需要你从以下第{generation_round + 1}批共 {len(opportunities)} 个研究机会中，策略性地挑选出 {num_ideas} 个最有前景的想法并进行详细设计。

**背景说明**：
- 这是第{generation_round + 1}轮想法生成
- 机会已经过预筛选和智能分配
- 需要从当前批次中发现最优质的研究方向

**你的任务**：
1. **批次分析**：深入分析当前批次机会的特点和潜力
2. **策略性筛选**：基于新颖性、可行性、影响力潜力进行intelligent selection
3. **详细设计**：为选中的想法提供完整的idea设计

**可用研究机会**：
{opportunities_summary}

**筛选标准**：
- **新颖性**: 是否填补了重要的研究空白？是否提供了新的视角？
- **可行性**: 所需资源是否合理？技术路径是否可行？
- **影响力**: 是否能推动领域发展？是否有实际应用价值？
- **科学价值**: 是否有深刻的理论贡献？
- **实现潜力**: 在当前技术条件下是否可实现？

**生成要求**：
- 请挑选 {num_ideas} 个最有前景的机会
- 为每个选中的想法提供详细的设计
- 解释为什么选择这个机会而非其他
- 确保想法之间有一定的多样性

请以JSON格式返回，包含选中的想法：
{{
  "strategy_analysis": "你的全局分析和选择策略",
  "selected_ideas": [
    {{
      "rank": 1,
      "title": "想法标题",
      "core_hypothesis": "核心假设",
      "selection_rationale": "为什么选择这个机会的详细理由",
      "opportunity_source": "基于哪个/哪些机会点",
      "innovation_points": ["创新点1", "创新点2", "创新点3"],
      "expected_contributions": ["贡献1", "贡献2"],
      "required_assets": [
        {{"type": "dataset", "name": "数据集名称", "availability": "可获得性"}},
        {{"type": "method", "name": "方法名称", "status": "现有/需开发"}}
      ],
      "preliminary_experiments": [
        {{"name": "实验1", "purpose": "验证目的", "expected_outcome": "预期结果"}}
      ],
      "potential_risks": ["风险1", "风险2"],
      "uniqueness_vs_existing": "与现有工作的区别和优势"
    }}
  ],
  "rejected_opportunities": [
    {{
      "opportunity": "机会描述",
      "rejection_reason": "为什么不选择的理由"
    }}
  ]
}}
"""

        try:
            # 调用LLM进行策略性生成
            response_data = await self.llm.generate(
                model_name=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=15000,
                agent_name=self.name,
                task_type="strategic_idea_generation"
            )
            response = response_data.get('content', '')
            
            # 解析LLM响应
            import json
            import re
            try:
                if "```json" in response:
                    json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
                    if json_match:
                        json_content = json_match.group(1).strip()
                    else:
                        json_start = response.find("```json") + 7
                        json_content = response[json_start:].strip()
                        if json_content.endswith("```"):
                            json_content = json_content[:-3].strip()
                else:
                    json_content = response.strip()
                
                result = json.loads(json_content)
                
                # 转换为CandidateIdea对象
                candidates = []
                for i, idea_data in enumerate(result.get("selected_ideas", [])):
                    idea_id = f"strategic-gen-{generation_round+1}-{i+1:02d}"
                    
                    candidate = CandidateIdea(
                        id=idea_id,
                        title=idea_data.get("title", f"策略性想法{i+1}"),
                        core_hypothesis=idea_data.get("core_hypothesis", ""),
                        initial_innovation_points=idea_data.get("innovation_points", []),
                        source_trigger_nodes=idea_data.get("opportunity_source", []),
                        expected_contribution=idea_data.get("expected_contributions", []),
                        required_assets=idea_data.get("required_assets", []),
                        preliminary_experiments=idea_data.get("preliminary_experiments", []),
                        risks=idea_data.get("potential_risks", []),
                        provenance={
                            "generation_round": generation_round + 1,
                            "selection_rationale": idea_data.get("selection_rationale", ""),
                            "uniqueness_analysis": idea_data.get("uniqueness_vs_existing", ""),
                            "rank_in_batch": idea_data.get("rank", i+1),
                            "strategy_analysis": result.get("strategy_analysis", ""),
                            "generation_method": "strategic_selection"
                        }
                    )
                    candidates.append(candidate)
                
                print(f"    ✅ 成功生成{len(candidates)}个策略性想法")
                return candidates
                
            except json.JSONDecodeError as e:
                print(f"    ⚠️ JSON解析失败: {str(e)}")
                print(f"    🔧 尝试修复JSON格式...")
                
                # 尝试修复常见的JSON格式错误
                fixed_candidates = self._try_fix_json_and_parse(json_content, generation_round, e)
                if fixed_candidates:
                    print(f"    ✅ JSON修复成功，生成{len(fixed_candidates)}个策略性想法")
                    return fixed_candidates
                else:
                    print(f"    ❌ JSON修复失败，保存响应用于调试")
                    # 保存原始响应用于调试
                    self._save_failed_response(response, generation_round, e)
                    return []
                
        except Exception as e:
            print(f"    ❌ 策略性生成失败: {str(e)}")
            return []
    
    def _try_fix_json_and_parse(self, json_content: str, generation_round: int, original_error: Exception) -> List[CandidateIdea]:
        """尝试修复JSON格式错误并重新解析。"""
        import json
        import re
        
        # 常见的JSON修复策略
        fixes = [
            # 1. 移除末尾的不完整内容
            lambda x: re.sub(r',\s*[}\]]\s*[^}\]]*$', '}', x.rsplit('}', 1)[0] + '}'),
            
            # 2. 修复缺失的引号
            lambda x: re.sub(r'(\w+):', r'"\1":', x),
            
            # 3. 修复末尾缺失的括号
            lambda x: x.strip() + ('}' if x.count('{') > x.count('}') else ''),
            
            # 4. 移除末尾多余的逗号
            lambda x: re.sub(r',(\s*[}\]])', r'\1', x),
            
            # 5. 尝试截取到最后一个完整的想法
            lambda x: self._truncate_to_last_complete_idea(x)
        ]
        
        for i, fix_func in enumerate(fixes):
            try:
                fixed_content = fix_func(json_content)
                result = json.loads(fixed_content)
                
                print(f"    🔧 JSON修复策略 {i+1} 成功")
                
                # 转换为CandidateIdea对象
                candidates = []
                for j, idea_data in enumerate(result.get("selected_ideas", [])):
                    idea_id = f"strategic-gen-{generation_round+1}-{j+1:02d}"
                    
                    candidate = CandidateIdea(
                        id=idea_id,
                        title=idea_data.get("title", f"策略性想法{j+1}"),
                        core_hypothesis=idea_data.get("core_hypothesis", ""),
                        initial_innovation_points=idea_data.get("innovation_points", []),
                        source_trigger_nodes=idea_data.get("opportunity_source", []),
                        expected_contribution=idea_data.get("expected_contributions", []),
                        required_assets=idea_data.get("required_assets", []),
                        preliminary_experiments=idea_data.get("preliminary_experiments", []),
                        risks=idea_data.get("potential_risks", []),
                        provenance={
                            "generation_round": generation_round + 1,
                            "selection_rationale": idea_data.get("selection_rationale", ""),
                            "uniqueness_analysis": idea_data.get("uniqueness_vs_existing", ""),
                            "rank_in_batch": idea_data.get("rank", j+1),
                            "strategy_analysis": result.get("strategy_analysis", ""),
                            "generation_method": "strategic_selection_fixed",
                            "json_fix_strategy": i+1
                        }
                    )
                    candidates.append(candidate)
                
                return candidates
                
            except (json.JSONDecodeError, Exception) as e:
                continue
        
        return []
    
    def _truncate_to_last_complete_idea(self, json_content: str) -> str:
        """截取到最后一个完整的想法。"""
        import re
        
        # 查找所有想法的开始位置
        idea_pattern = r'"title":\s*"[^"]*"'
        matches = list(re.finditer(idea_pattern, json_content))
        
        if len(matches) < 2:
            return json_content
        
        # 找到倒数第二个想法的位置，截取到那里
        second_last_match = matches[-2]
        truncate_pos = second_last_match.start()
        
        # 向前查找这个想法的开始 "{"
        bracket_count = 0
        for i in range(truncate_pos, -1, -1):
            if json_content[i] == '}':
                bracket_count += 1
            elif json_content[i] == '{':
                bracket_count -= 1
                if bracket_count == 0:
                    # 找到这个想法的开始，截取到前一个想法结束
                    truncated = json_content[:i]
                    if truncated.rstrip().endswith(','):
                        truncated = truncated.rstrip()[:-1]  # 移除末尾逗号
                    return truncated + '\n  ]\n}'
        
        return json_content
    
    def _save_failed_response(self, response: str, generation_round: int, error: Exception) -> None:
        """保存失败的响应用于调试。"""
        import os
        from datetime import datetime
        
        # 确保日志目录存在
        log_dir = "./logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"failed_json_response_round{generation_round}_{timestamp}.txt"
        filepath = os.path.join(log_dir, filename)
        
        # 保存响应内容
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"Generation Round: {generation_round}\n")
            f.write(f"Error: {str(error)}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write("="*80 + "\n")
            f.write("Raw Response:\n")
            f.write("="*80 + "\n")
            f.write(response)
        
        print(f"    💾 失败响应已保存到: {filepath}")
    
    def _enhance_opportunities_with_node_names(self, opportunities: List[Dict[str, Any]], 
                                             graph: SemanticOpportunityGraph) -> List[Dict[str, Any]]:
        """用实际的节点名称增强机会信息。"""
        enhanced_opportunities = []
        
        for opp in opportunities:
            enhanced_opp = opp.copy()
            
            # 获取节点的实际名称
            nodes = opp.get('nodes', [])
            if nodes:
                node_names = []
                for node_id in nodes:
                    if node_id in graph.nodes:
                        node_name = graph.nodes[node_id].get('name', str(node_id))
                        node_names.append(node_name)
                    else:
                        node_names.append(str(node_id))
                enhanced_opp['node_names'] = node_names
                enhanced_opp['nodes'] = node_names  # 替换原始的节点ID
            
            # 获取相关节点的实际名称
            related_nodes = opp.get('related_nodes', [])
            if related_nodes:
                related_node_names = []
                for node_id in related_nodes:
                    if node_id in graph.nodes:
                        node_name = graph.nodes[node_id].get('name', str(node_id))
                        related_node_names.append(node_name)
                    else:
                        related_node_names.append(str(node_id))
                enhanced_opp['related_node_names'] = related_node_names
            
            enhanced_opportunities.append(enhanced_opp)
        
        return enhanced_opportunities
    
    async def _identify_extended_opportunities(self, graph: SemanticOpportunityGraph) -> List[Dict[str, Any]]:
        """识别扩展机会 - 基于图谱结构深入挖掘。"""
        opportunities = []
        
        # 1. 高度连接但缺乏某种关系的节点对
        nodes = list(graph.nodes())
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i+1:]:
                # 检查两个节点是否在图中有很多共同邻居但没有直接连接
                common_neighbors = set(graph.neighbors(node1)) & set(graph.neighbors(node2))
                if len(common_neighbors) >= 2 and not graph.has_edge(node1, node2):
                    opportunities.append({
                        "pattern": "missing_connection",
                        "nodes": [node1, node2],
                        "explanation": f"节点{graph.nodes[node1]['name']}和{graph.nodes[node2]['name']}有{len(common_neighbors)}个共同邻居但缺乏直接连接",
                        "confidence": min(len(common_neighbors) * 0.15, 0.8),
                        "related_nodes": [node1, node2] + list(common_neighbors)[:3],
                        "opportunity_type": "connection_gap"
                    })
        
        # 2. 关键节点的未开发潜力
        degree_centrality = nx.degree_centrality(graph)
        high_degree_nodes = [node for node, centrality in degree_centrality.items() if centrality > 0.1]
        
        for node in high_degree_nodes:
            node_edges = list(graph.edges(node, data=True))
            edge_types = [edge[2].get('relation', 'unknown') for edge in node_edges]
            
            # 如果这个重要节点缺少某些常见关系类型
            common_relations = ['applies_to', 'improves', 'evaluates_with', 'requires']
            missing_relations = [rel for rel in common_relations if rel not in edge_types]
            
            if missing_relations:
                opportunities.append({
                    "pattern": "underexplored_potential",
                    "nodes": [node], 
                    "explanation": f"重要节点{graph.nodes[node]['name']}缺少{missing_relations}等关系",
                    "confidence": 0.6,
                    "related_nodes": [node],
                    "missing_relations": missing_relations,
                    "opportunity_type": "node_potential"
                })
        
        return opportunities[:20]  # 限制数量
    
    async def _identify_combination_opportunities(self, graph: SemanticOpportunityGraph) -> List[Dict[str, Any]]:
        """识别组合机会 - 发现多个节点/边的复杂组合。"""
        opportunities = []
        
        # 1. 方法组合机会
        method_nodes = [node for node, data in graph.nodes(data=True) if data.get('type') == 'Method']
        
        for i, method1 in enumerate(method_nodes):
            for method2 in method_nodes[i+1:]:
                # 检查两个方法是否适合组合
                method1_tasks = [neighbor for neighbor in graph.neighbors(method1) 
                               if graph.nodes[neighbor].get('type') == 'Task']
                method2_tasks = [neighbor for neighbor in graph.neighbors(method2) 
                               if graph.nodes[neighbor].get('type') == 'Task']
                
                # 如果两个方法处理相关但不同的任务
                if len(set(method1_tasks) & set(method2_tasks)) == 0 and len(method1_tasks) > 0 and len(method2_tasks) > 0:
                    opportunities.append({
                        "pattern": "method_combination",
                        "nodes": [method1, method2],
                        "explanation": f"方法{graph.nodes[method1]['name']}和{graph.nodes[method2]['name']}可能具有互补性",
                        "confidence": 0.65,
                        "related_nodes": [method1, method2] + method1_tasks[:2] + method2_tasks[:2],
                        "opportunity_type": "method_synergy"
                    })
        
        # 2. 数据集-评价指标新组合
        dataset_nodes = [node for node, data in graph.nodes(data=True) if data.get('type') == 'Dataset']
        metric_nodes = [node for node, data in graph.nodes(data=True) if data.get('type') == 'Metric']
        
        for dataset in dataset_nodes[:10]:  # 限制数量
            dataset_metrics = [neighbor for neighbor in graph.neighbors(dataset) 
                             if graph.nodes[neighbor].get('type') == 'Metric']
            
            # 寻找未与此数据集关联的相关评价指标
            for metric in metric_nodes:
                if metric not in dataset_metrics:
                    opportunities.append({
                        "pattern": "dataset_metric_combination",
                        "nodes": [dataset, metric],
                        "explanation": f"数据集{graph.nodes[dataset]['name']}与评价指标{graph.nodes[metric]['name']}的新组合",
                        "confidence": 0.5,
                        "related_nodes": [dataset, metric],
                        "opportunity_type": "evaluation_innovation"
                    })
        
        return opportunities[:15]  # 限制数量
    
    async def _identify_reverse_engineering_opportunities(self, graph: SemanticOpportunityGraph) -> List[Dict[str, Any]]:
        """识别反向工程机会 - 从现有批评/问题中发现机会。"""
        opportunities = []
        
        # 1. 从问题节点反向推导解决方案
        problem_nodes = [node for node, data in graph.nodes(data=True) if data.get('type') == 'Problem']
        
        for problem in problem_nodes:
            problem_name = graph.nodes[problem]['name']
            
            # 寻找可能解决此问题的现有方法
            related_methods = []
            for node in graph.nodes():
                if graph.nodes[node].get('type') == 'Method':
                    # 检查是否有路径连接这个方法和问题
                    try:
                        if nx.has_path(graph, node, problem) and nx.shortest_path_length(graph, node, problem) <= 3:
                            related_methods.append(node)
                    except:
                        continue
            
            if related_methods:
                opportunities.append({
                    "pattern": "problem_solving",
                    "nodes": [problem] + related_methods[:2],
                    "explanation": f"针对问题'{problem_name}'的改进解决方案",
                    "confidence": 0.7,
                    "related_nodes": [problem] + related_methods[:3],
                    "opportunity_type": "problem_driven"
                })
        
        # 2. 从负面边（批评关系）中发现改进机会
        negative_edges = [(u, v, data) for u, v, data in graph.edges(data=True) 
                         if data.get('relation', '').lower() in ['critiques', 'improves', 'addresses']]
        
        for src, dst, edge_data in negative_edges[:10]:
            if edge_data.get('relation') == 'critiques':
                opportunities.append({
                    "pattern": "improvement_opportunity",
                    "nodes": [src, dst],
                    "explanation": f"基于对{graph.nodes[dst]['name']}的批评，开发改进方案",
                    "confidence": 0.75,
                    "related_nodes": [src, dst],
                    "opportunity_type": "critique_driven"
                })
        
        return opportunities[:10]  # 限制数量
    
    async def _identify_cross_domain_opportunities(self, graph: SemanticOpportunityGraph) -> List[Dict[str, Any]]:
        """识别跨领域迁移机会。"""
        opportunities = []
        
        # 1. 跨任务迁移
        task_nodes = [node for node, data in graph.nodes(data=True) if data.get('type') == 'Task']
        
        for i, task1 in enumerate(task_nodes):
            for task2 in task_nodes[i+1:]:
                # 检查两个任务是否有相似的方法但没有直接关联
                task1_methods = [neighbor for neighbor in graph.neighbors(task1) 
                               if graph.nodes[neighbor].get('type') == 'Method']
                task2_methods = [neighbor for neighbor in graph.neighbors(task2) 
                               if graph.nodes[neighbor].get('type') == 'Method']
                
                common_methods = set(task1_methods) & set(task2_methods)
                if len(common_methods) >= 1 and not graph.has_edge(task1, task2):
                    opportunities.append({
                        "pattern": "cross_task_transfer",
                        "nodes": [task1, task2],
                        "explanation": f"任务{graph.nodes[task1]['name']}和{graph.nodes[task2]['name']}间的方法迁移",
                        "confidence": len(common_methods) * 0.2,
                        "related_nodes": [task1, task2] + list(common_methods)[:2],
                        "opportunity_type": "domain_transfer"
                    })
        
        return opportunities[:12]  # 限制数量
    
    def _deduplicate_and_enrich_opportunities(self, opportunities: List[Dict[str, Any]], 
                                            graph: SemanticOpportunityGraph) -> List[Dict[str, Any]]:
        """去重并丰富机会信息。"""
        # 简单去重 - 基于节点组合
        seen_combinations = set()
        unique_opportunities = []
        
        for opp in opportunities:
            nodes_key = tuple(sorted(opp.get('nodes', [])))
            if nodes_key not in seen_combinations:
                seen_combinations.add(nodes_key)
                
                # 丰富机会信息
                opp['detailed_description'] = self._generate_opportunity_description(opp, graph)
                opp['complexity_estimate'] = self._estimate_opportunity_complexity(opp, graph)
                
                unique_opportunities.append(opp)
        
        # 按置信度排序
        unique_opportunities.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        return unique_opportunities
    
    def _generate_opportunity_description(self, opportunity: Dict[str, Any], 
                                        graph: SemanticOpportunityGraph) -> str:
        """生成机会的详细描述。"""
        pattern = opportunity.get('pattern', 'unknown')
        nodes = opportunity.get('nodes', [])
        
        if not nodes:
            return opportunity.get('explanation', '未知机会')
        
        node_names = [graph.nodes[node]['name'] for node in nodes if node in graph.nodes]
        
        if pattern == 'transfer':
            return f"将{node_names[0]}的成功经验迁移到{node_names[1]}领域"
        elif pattern == 'method_combination':
            return f"结合{node_names[0]}和{node_names[1]}的互补优势创造新方法"
        elif pattern == 'problem_solving':
            return f"针对{node_names[0]}问题开发创新解决方案"
        else:
            return opportunity.get('explanation', '研究机会')
    
    def _estimate_opportunity_complexity(self, opportunity: Dict[str, Any], 
                                       graph: SemanticOpportunityGraph) -> str:
        """估算机会的复杂程度。"""
        nodes = opportunity.get('nodes', [])
        
        if len(nodes) <= 1:
            return "低"
        elif len(nodes) <= 3:
            return "中"
        else:
            return "高"
    
    def _prepare_opportunities_summary(self, opportunities: List[Dict[str, Any]]) -> str:
        """准备机会摘要供LLM分析。"""
        summary_lines = []
        
        for i, opp in enumerate(opportunities[:50], 1):  # 限制最多50个
            pattern = opp.get('pattern', 'unknown')
            confidence = opp.get('confidence', 0)
            complexity = opp.get('complexity_estimate', '未知')
            opportunity_type = opp.get('opportunity_type', 'general')
            
            # 生成具体的机会描述
            detailed_description = self._generate_detailed_opportunity_description(opp)
            
            summary_lines.append(
                f"{i}. [{pattern.upper()}] {detailed_description}\n"
                f"   - 类型: {opportunity_type}\n"
                f"   - 复杂度: {complexity}\n"
                f"   - 置信度: {confidence:.2f}\n"
            )
        
        return "\n".join(summary_lines)
    
    def _generate_detailed_opportunity_description(self, opportunity: Dict[str, Any]) -> str:
        """根据机会结构生成详细的研究机会描述。"""
        pattern = opportunity.get('pattern', 'unknown')
        nodes = opportunity.get('nodes', [])
        related_nodes = opportunity.get('related_nodes', [])
        gap_data = opportunity.get('gap_data', {})
        
        # 优先使用已有的详细描述
        if opportunity.get('detailed_description'):
            return opportunity['detailed_description']
        
        # 优先使用已有的explanation（如果不为空）
        explanation = opportunity.get('explanation', '').strip()
        if explanation and explanation != '':
            return explanation
        
        # 根据不同模式生成具体描述
        if pattern == 'transfer':
            return self._generate_transfer_description(opportunity, gap_data)
        elif pattern == 'missing_connection':
            return self._generate_missing_connection_description(opportunity)
        elif pattern == 'cross_task_transfer':
            return self._generate_cross_task_description(opportunity)
        elif pattern == 'method_combination':
            return self._generate_method_combination_description(opportunity)
        elif pattern == 'dataset_metric_combination':
            return self._generate_dataset_metric_description(opportunity)
        elif pattern == 'problem_solving':
            return self._generate_problem_solving_description(opportunity)
        elif pattern == 'improvement_opportunity':
            return self._generate_improvement_description(opportunity)
        elif pattern == 'underexplored_potential':
            return self._generate_underexplored_description(opportunity)
        else:
            # 通用描述生成
            return self._generate_generic_description(opportunity, pattern)
    
    def _generate_transfer_description(self, opportunity: Dict[str, Any], gap_data: Dict[str, Any]) -> str:
        """生成迁移机会的具体描述。"""
        # 从gap_data中提取具体信息
        source_method = gap_data.get('source_method', '未知方法')
        source_task = gap_data.get('source_task', '未知任务')
        target_task = gap_data.get('target_task', '未知目标任务')
        
        if source_method != '未知方法' and target_task != '未知目标任务':
            return f"将{source_method}方法从{source_task}迁移到{target_task}任务"
        else:
            # 从节点信息推断
            nodes = opportunity.get('nodes', [])
            if len(nodes) >= 2:
                return f"探索{nodes[0]}与{nodes[1]}之间的迁移学习机会"
            else:
                return "探索跨领域方法迁移的新机会"
    
    def _generate_missing_connection_description(self, opportunity: Dict[str, Any]) -> str:
        """生成缺失连接的具体描述。"""
        nodes = opportunity.get('nodes', [])
        if len(nodes) >= 2:
            return f"探索{nodes[0]}和{nodes[1]}之间的潜在关联性研究"
        else:
            return "发现关键概念间的缺失连接"
    
    def _generate_cross_task_description(self, opportunity: Dict[str, Any]) -> str:
        """生成跨任务迁移的具体描述。"""
        nodes = opportunity.get('nodes', [])
        if len(nodes) >= 2:
            return f"开发从{nodes[0]}到{nodes[1]}的跨任务迁移方法"
        else:
            return "探索跨任务知识迁移的新方法"
    
    def _generate_method_combination_description(self, opportunity: Dict[str, Any]) -> str:
        """生成方法组合的具体描述。"""
        nodes = opportunity.get('nodes', [])
        if len(nodes) >= 2:
            return f"研究{nodes[0]}与{nodes[1]}方法的创新性组合"
        else:
            return "探索多种方法的协同组合策略"
    
    def _generate_dataset_metric_description(self, opportunity: Dict[str, Any]) -> str:
        """生成数据集-指标组合的具体描述。"""
        nodes = opportunity.get('nodes', [])
        if len(nodes) >= 2:
            return f"开发{nodes[0]}数据集上{nodes[1]}指标的评估体系"
        else:
            return "建立新的数据集评估指标体系"
    
    def _generate_problem_solving_description(self, opportunity: Dict[str, Any]) -> str:
        """生成问题解决的具体描述。"""
        nodes = opportunity.get('nodes', [])
        if len(nodes) >= 1:
            return f"针对{nodes[0]}问题开发创新解决方案"
        else:
            return "开发针对关键技术问题的创新解决方案"
    
    def _generate_improvement_description(self, opportunity: Dict[str, Any]) -> str:
        """生成改进机会的具体描述。"""
        nodes = opportunity.get('nodes', [])
        if len(nodes) >= 2:
            return f"基于{nodes[0]}的观察，改进{nodes[1]}的性能和效率"
        else:
            return "基于现有批评和观察，开发性能改进方案"
    
    def _generate_underexplored_description(self, opportunity: Dict[str, Any]) -> str:
        """生成未充分探索的具体描述。"""
        nodes = opportunity.get('nodes', [])
        missing_relations = opportunity.get('missing_relations', [])
        
        if nodes and missing_relations:
            return f"深入挖掘{nodes[0]}在{', '.join(missing_relations[:2])}方面的应用潜力"
        elif nodes:
            return f"全面探索{nodes[0]}的未开发应用潜力"
        else:
            return "识别和开发关键技术的未探索应用领域"
    
    def _generate_generic_description(self, opportunity: Dict[str, Any], pattern: str) -> str:
        """生成通用描述。"""
        nodes = opportunity.get('nodes', [])
        opportunity_type = opportunity.get('opportunity_type', 'general')
        
        if nodes:
            if len(nodes) == 1:
                return f"探索{nodes[0]}的创新应用和发展机会（{pattern}类型）"
            elif len(nodes) == 2:
                return f"研究{nodes[0]}与{nodes[1]}的协同创新机会（{pattern}类型）"
            else:
                return f"开发涉及{nodes[0]}、{nodes[1]}等多要素的综合创新方案（{pattern}类型）"
        else:
            return f"探索{opportunity_type}领域的{pattern}研究机会"
    
    def _map_gap_to_strategy(self, gap_pattern: str) -> Optional[str]:
        """将gap模式映射到策略模板。"""
        pattern_to_strategy = {
            "transfer": "transfer_across_tasks",
            "composition": "compose_methods", 
            "reverse": "reverse_critique",
            "evaluation": "evaluation_reconstruction",
            "data_enhancement": "data_augmentation"
        }
        return pattern_to_strategy.get(gap_pattern)
    
    async def _find_additional_patterns(self, graph: SemanticOpportunityGraph) -> List[Dict[str, Any]]:
        """查找图谱中的额外模式。"""
        additional_triggers = []
        
        # 模式1：高度中心的节点可能适合迁移
        high_salience_nodes = []
        for node_id, node_data in graph.nodes(data=True):
            if node_data.get('salience', 0) > 0.7:
                high_salience_nodes.append(node_id)
        
        if len(high_salience_nodes) > 0:
            for node in high_salience_nodes[:3]:  # 限制数量
                additional_triggers.append({
                    "pattern": "high_impact_extension",
                    "nodes": [node],
                    "strategy": "transfer_across_tasks",
                    "confidence": 0.6,
                    "source": "salience_analysis"
                })
        
        # 模式2：孤立节点可能需要连接
        isolated_nodes = [node for node in graph.nodes() if graph.degree(node) == 0]
        if isolated_nodes:
            for node in isolated_nodes[:2]:  # 限制数量
                additional_triggers.append({
                    "pattern": "isolated_integration",
                    "nodes": [node],
                    "strategy": "compose_methods",
                    "confidence": 0.4,
                    "source": "topology_analysis"
                })
        
        return additional_triggers

    async def _generate_from_trigger(self, trigger: Dict[str, Any], graph: SemanticOpportunityGraph) -> Optional[CandidateIdea]:
        """从单个触发信号生成候选想法。
        
        输入:
            - trigger: 触发信号字典。
            - graph: 语义机会图谱。
            
        输出:
            - CandidateIdea: 生成的候选想法，失败时返回None。
            
        实现思路:
            1) 根据 trigger['strategy'] 选择对应的策略模板。
            2) 使用 Chain-of-Ideas 提示链：触发模式 → 核心假设 → 机制设想 → 实验草案 → 预期收益 → 风险评估。
            3) 调用 LLM 生成结构化内容，解析为 CandidateIdea 对象。
        """
        try:
            strategy_name = trigger.get('strategy')
            if not strategy_name or strategy_name not in self.strategy_templates:
                return None
            
            strategy_template = self.strategy_templates[strategy_name]
            trigger_nodes = trigger.get('nodes', [])
            
            # 收集节点信息
            node_info = {}
            for node_id in trigger_nodes:
                if node_id in graph.nodes():
                    node_info[node_id] = {
                        "name": graph.nodes[node_id].get('name', node_id),
                        "type": graph.nodes[node_id].get('type', 'Unknown'),
                        "salience": graph.nodes[node_id].get('salience', 0.0)
                    }
            
            # 基于策略模板生成想法内容
            idea_content = await self._apply_strategy_template(strategy_template, trigger, node_info, graph)
            
            if not idea_content:
                return None
            
            # 生成想法ID
            idea_id = f"IDEA-{len(trigger_nodes):02d}-{strategy_name[:8]}-{hash(str(trigger_nodes)) % 10000:04d}"
            
            # 构造CandidateIdea对象
            candidate_idea = CandidateIdea(
                id=idea_id,
                title=idea_content.get('title', f'基于{strategy_name}的想法'),
                core_hypothesis=idea_content.get('hypothesis', ''),
                initial_innovation_points=idea_content.get('innovation_points', []),
                source_trigger_nodes=trigger_nodes,
                expected_contribution=idea_content.get('contributions', []),
                required_assets=idea_content.get('assets', []),
                preliminary_experiments=idea_content.get('experiments', []),
                risks=idea_content.get('risks', []),
                provenance={
                    "trigger": trigger,
                    "strategy": strategy_name,
                    "generated_at": datetime.now().isoformat(),
                    "confidence": trigger.get('confidence', 0.5)
                },
                version=1
            )
            
            return candidate_idea
            
        except Exception as e:
            print(f"    ⚠️ 从触发器生成想法失败: {e}")
            return None
    
    async def _apply_strategy_template(self, template: Dict[str, Any], trigger: Dict[str, Any], 
                                     node_info: Dict[str, Any], graph: SemanticOpportunityGraph) -> Optional[Dict[str, Any]]:
        """应用策略模板生成想法内容。"""
        strategy_name = trigger.get('strategy')
        
        # 根据不同策略生成内容
        if strategy_name == "transfer_across_tasks":
            return await self._generate_transfer_idea(template, trigger, node_info, graph)
        elif strategy_name == "compose_methods":
            return await self._generate_composition_idea(template, trigger, node_info, graph)
        elif strategy_name == "reverse_critique":
            return await self._generate_reverse_idea(template, trigger, node_info, graph)
        else:
            # 通用模板
            return await self._generate_generic_idea(template, trigger, node_info, graph)
    
    async def _generate_transfer_idea(self, template: Dict[str, Any], trigger: Dict[str, Any], 
                                    node_info: Dict[str, Any], graph: SemanticOpportunityGraph) -> Dict[str, Any]:
        """生成迁移类型的想法。"""
        trigger_nodes = trigger.get('nodes', [])
        gap_data = trigger.get('gap_data', {})
        
        # 识别方法和任务
        method_node = None
        task_node = None
        
        for node_id in trigger_nodes:
            if node_id in node_info:
                if node_info[node_id]['type'] == 'Method':
                    method_node = node_id
                elif node_info[node_id]['type'] == 'Task':
                    task_node = node_id
        
        if not method_node:
            method_node = gap_data.get('from_method')
        if not task_node:
            task_node = gap_data.get('to_task')
        
        method_name = graph.nodes[method_node]['name'] if method_node in graph.nodes() else "Unknown Method"
        task_name = graph.nodes[task_node]['name'] if task_node in graph.nodes() else "Unknown Task"
        
        # 收集相关上下文信息
        context_info = self._collect_context_info(graph, [method_node, task_node])
        
        # 使用LLM生成具体的研究想法
        prompt = f"""
基于语义机会图谱分析，设计一个创新的研究想法，将方法"{method_name}"迁移应用到任务"{task_name}"中。

**背景信息**：
- 源方法: {method_name}
- 目标任务: {task_name}
- 发现的机会gap: {gap_data.get('description', '跨任务迁移机会')}

**上下文信息**：
{context_info}

请设计一个具体的、有创新性的研究想法，包含以下要素：

1. **标题**: 简洁明确的研究题目
2. **核心假设**: 科学的、可验证的假设
3. **创新点**: 3-4个具体的技术创新点
4. **预期贡献**: 对学术界和实践的具体贡献
5. **实验设计**: 具体的实验方案和评估指标
6. **潜在风险**: 可能遇到的技术挑战和解决方案

请以JSON格式返回：
{{
  "title": "研究题目",
  "hypothesis": "核心研究假设",
  "innovation_points": ["创新点1", "创新点2", "创新点3"],
  "contributions": ["贡献1", "贡献2"],
  "experiments": [
    {{
      "name": "实验名称",
      "description": "实验描述", 
      "metrics": ["评估指标1", "评估指标2"],
      "datasets": ["数据集1", "数据集2"]
    }}
  ],
  "risks": ["风险1", "风险2"],
  "technical_approach": "具体的技术实现路径"
}}
"""

        try:
            response_data = await self.llm.generate(
                model_name=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=15000,
                agent_name=self.name,
                task_type="idea_generation"
            )
            response = response_data.get("content", "")
            
            # 解析LLM响应
            import json
            import re
            try:
                if "```json" in response:
                    json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
                    if json_match:
                        json_content = json_match.group(1).strip()
                    else:
                        json_start = response.find("```json") + 7
                        json_content = response[json_start:].strip()
                        if json_content.endswith("```"):
                            json_content = json_content[:-3].strip()
                else:
                    json_content = response.strip()
                
                result = json.loads(json_content)
                
                # 补充资产信息
                result['assets'] = [
                    {'type': 'Method', 'id': method_node, 'availability': 'adaptation_required'},
                    {'type': 'Dataset', 'id': f'{task_name}_dataset', 'availability': 'to_be_identified'}
                ]
                
                print(f"    💡 LLM生成想法: {result.get('title', '未知标题')}")
                return result
                
            except json.JSONDecodeError:
                print(f"    ⚠️ LLM想法生成响应解析失败，使用备用模板")
                # 备用模板
                return self._generate_fallback_transfer_idea(method_name, task_name, method_node)
                
        except Exception as e:
            print(f"    ❌ LLM想法生成失败: {str(e)}，使用备用模板")
            return self._generate_fallback_transfer_idea(method_name, task_name, method_node)
    
    def _generate_fallback_transfer_idea(self, method_name: str, task_name: str, method_node: str) -> Dict[str, Any]:
        """备用的想法生成模板（当LLM调用失败时使用）。"""
        return {
            'title': f'将{method_name}迁移应用于{task_name}',
            'hypothesis': f'将在其他任务中表现优异的{method_name}方法适配到{task_name}任务中，可能获得显著性能提升',
            'innovation_points': [
                f'跨任务迁移{method_name}的核心机制',
                f'针对{task_name}的特定适配策略',
                '创新的评估与对比基准'
            ],
            'contributions': ['方法学贡献', '实证验证'],
            'assets': [
                {'type': 'Method', 'id': method_node, 'availability': 'adaptation_required'},
                {'type': 'Dataset', 'id': f'{task_name}_dataset', 'availability': 'to_be_identified'}
            ],
            'experiments': [
                {'name': f'{task_name}基线对比', 'metric': 'domain_specific_metrics', 'dataset': f'{task_name}_benchmark'}
            ],
            'risks': [
                '跨任务的方法适配可能存在兼容性问题',
                f'{task_name}领域的评估标准可能需要重新设计'
            ]
        }
    
    def _collect_context_info(self, graph: SemanticOpportunityGraph, node_ids: List[str]) -> str:
        """收集节点的上下文信息用于LLM提示。"""
        context_lines = []
        
        for node_id in node_ids:
            if node_id and node_id in graph.nodes():
                node_data = graph.nodes[node_id]
                context_lines.append(f"- {node_data.get('name', node_id)}: {node_data.get('type', 'Unknown')} (重要性: {node_data.get('salience', 0.0):.2f})")
                
                # 添加相关边信息
                edges = list(graph.edges(node_id, data=True))
                if edges:
                    context_lines.append(f"  相关关系: {len(edges)}个")
                    for src, dst, edge_data in edges[:3]:  # 只显示前3个
                        rel_type = edge_data.get('relation', 'unknown')
                        dst_name = graph.nodes[dst].get('name', dst) if dst in graph.nodes() else dst
                        context_lines.append(f"    → {dst_name} ({rel_type})")
        
        return "\n".join(context_lines) if context_lines else "无相关上下文信息"
    
    async def _generate_composition_idea(self, template: Dict[str, Any], trigger: Dict[str, Any], 
                                       node_info: Dict[str, Any], graph: SemanticOpportunityGraph) -> Dict[str, Any]:
        """生成组合类型的想法。"""
        trigger_nodes = trigger.get('nodes', [])
        gap_data = trigger.get('gap_data', {})
        
        method_nodes = [node for node in trigger_nodes if node_info.get(node, {}).get('type') == 'Method']
        target_task = gap_data.get('target_task')
        
        if len(method_nodes) >= 2:
            method1_name = graph.nodes[method_nodes[0]]['name'] if method_nodes[0] in graph.nodes() else "Method1"
            method2_name = graph.nodes[method_nodes[1]]['name'] if method_nodes[1] in graph.nodes() else "Method2"
        else:
            method1_name, method2_name = "Method1", "Method2"
        
        task_name = graph.nodes[target_task]['name'] if target_task and target_task in graph.nodes() else "Target Task"
        
        # 收集上下文信息
        context_info = self._collect_context_info(graph, method_nodes + ([target_task] if target_task else []))
        
        # 使用LLM生成组合想法
        prompt = f"""
基于语义机会图谱分析，设计一个创新的研究想法，将多个方法"{method1_name}"和"{method2_name}"组合应用到任务"{task_name}"中。

**背景信息**：
- 方法1: {method1_name}
- 方法2: {method2_name}
- 目标任务: {task_name}
- 发现的机会gap: {gap_data.get('description', '方法组合机会')}

**上下文信息**：
{context_info}

请设计一个具体的、创新的方法融合研究想法，包含以下要素：

1. **标题**: 简洁明确的研究题目
2. **核心假设**: 为什么这两个方法的组合会有效
3. **创新点**: 3-4个具体的技术融合创新点
4. **预期贡献**: 对学术界和实践的具体贡献
5. **实验设计**: 具体的实验方案，包括消融研究
6. **潜在风险**: 方法组合可能遇到的挑战

请以JSON格式返回：
{{
  "title": "研究题目",
  "hypothesis": "核心研究假设",
  "innovation_points": ["创新点1", "创新点2", "创新点3"],
  "contributions": ["贡献1", "贡献2"],
  "experiments": [
    {{
      "name": "实验名称",
      "description": "实验描述", 
      "metrics": ["评估指标1", "评估指标2"],
      "datasets": ["数据集1", "数据集2"]
    }}
  ],
  "risks": ["风险1", "风险2"],
  "technical_approach": "具体的融合技术路径"
}}
"""

        try:
            response_data = await self.llm.generate(
                model_name=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=15000,
                agent_name=self.name,
                task_type="idea_generation"
            )
            response = response_data.get("content", "")
            
            # 解析LLM响应
            import json
            import re
            try:
                if "```json" in response:
                    json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
                    if json_match:
                        json_content = json_match.group(1).strip()
                    else:
                        json_start = response.find("```json") + 7
                        json_content = response[json_start:].strip()
                        if json_content.endswith("```"):
                            json_content = json_content[:-3].strip()
                else:
                    json_content = response.strip()
                
                result = json.loads(json_content)
                
                # 补充资产信息
                result['assets'] = [
                    {'type': 'Method', 'id': method_nodes[0] if method_nodes else 'method1', 'availability': 'public'},
                    {'type': 'Method', 'id': method_nodes[1] if len(method_nodes) > 1 else 'method2', 'availability': 'public'}
                ]
                
                print(f"    💡 LLM生成组合想法: {result.get('title', '未知标题')}")
                return result
                
            except json.JSONDecodeError:
                print(f"    ⚠️ LLM组合想法生成响应解析失败，使用备用模板")
                return self._generate_fallback_composition_idea(method1_name, method2_name, task_name, method_nodes)
                
        except Exception as e:
            print(f"    ❌ LLM组合想法生成失败: {str(e)}，使用备用模板")
            return self._generate_fallback_composition_idea(method1_name, method2_name, task_name, method_nodes)
    
    def _generate_fallback_composition_idea(self, method1_name: str, method2_name: str, task_name: str, method_nodes: List[str]) -> Dict[str, Any]:
        """备用的组合想法生成模板。"""
        return {
            'title': f'融合{method1_name}与{method2_name}改进{task_name}',
            'hypothesis': f'通过创新性地组合{method1_name}和{method2_name}的优势，可以在{task_name}上实现性能突破',
            'innovation_points': [
                f'新颖的{method1_name}与{method2_name}融合架构',
                '互补优势的协同机制设计',
                '统一的端到端训练策略'
            ],
            'contributions': ['架构创新', '性能提升', '方法学贡献'],
            'assets': [
                {'type': 'Method', 'id': method_nodes[0] if method_nodes else 'method1', 'availability': 'public'},
                {'type': 'Method', 'id': method_nodes[1] if len(method_nodes) > 1 else 'method2', 'availability': 'public'}
            ],
            'experiments': [
                {'name': f'{task_name}性能对比', 'metric': 'accuracy,efficiency', 'dataset': f'{task_name}_standard_benchmark'},
                {'name': '消融研究', 'metric': 'component_contribution', 'dataset': 'same_as_main'}
            ],
            'risks': [
                '方法组合可能增加计算复杂度',
                '不同方法的训练策略可能存在冲突',
                '集成效果的可解释性可能降低'
            ]
        }
    
    async def _generate_reverse_idea(self, template: Dict[str, Any], trigger: Dict[str, Any], 
                                   node_info: Dict[str, Any], graph: SemanticOpportunityGraph) -> Dict[str, Any]:
        """生成反转类型的想法（针对批评的改进）。"""
        gap_data = trigger.get('gap_data', {})
        critiqued_method = gap_data.get('critiqued_method')
        
        method_name = graph.nodes[critiqued_method]['name'] if critiqued_method and critiqued_method in graph.nodes() else "Target Method"
        
        return {
            'title': f'针对{method_name}缺陷的改进方案',
            'hypothesis': f'通过分析{method_name}的已知局限性，设计针对性的改进策略，实现性能和效率的双重提升',
            'innovation_points': [
                f'识别并解决{method_name}的核心瓶颈',
                '创新的优化算法或架构改进',
                '保持原有优势的同时消除劣势'
            ],
            'contributions': ['方法改进', '理论分析', '实证验证'],
            'assets': [
                {'type': 'Method', 'id': critiqued_method, 'availability': 'baseline_reference'},
                {'type': 'Analysis', 'id': 'critique_analysis', 'availability': 'literature_based'}
            ],
            'experiments': [
                {'name': '改进效果验证', 'metric': 'performance_improvement', 'dataset': 'original_benchmark'},
                {'name': '效率分析', 'metric': 'computational_efficiency', 'dataset': 'efficiency_benchmark'}
            ],
            'risks': [
                '改进可能引入新的问题',
                '对原有优势的保持存在不确定性',
                '改进的泛化能力需要验证'
            ]
        }
    
    async def _generate_generic_idea(self, template: Dict[str, Any], trigger: Dict[str, Any], 
                                   node_info: Dict[str, Any], graph: SemanticOpportunityGraph) -> Dict[str, Any]:
        """生成通用类型的想法。"""
        trigger_nodes = trigger.get('nodes', [])
        pattern = trigger.get('pattern', 'unknown')
        gap_data = trigger.get('gap_data', {})
        
        node_names = [node_info.get(node, {}).get('name', node) for node in trigger_nodes]
        context_info = self._collect_context_info(graph, trigger_nodes)
        
        # 使用LLM生成通用想法
        prompt = f"""
基于语义机会图谱分析，发现了一个"{pattern}"类型的研究机会，涉及以下实体：{", ".join(node_names)}。

**背景信息**：
- 模式类型: {pattern}
- 相关实体: {", ".join(node_names)}
- 机会描述: {gap_data.get('description', '发现的研究机会')}

**上下文信息**：
{context_info}

请设计一个具体的、创新的研究想法，包含以下要素：

1. **标题**: 简洁明确的研究题目
2. **核心假设**: 基于发现的机会提出的研究假设
3. **创新点**: 3-4个具体的技术或方法创新点
4. **预期贡献**: 对学术界和实践的具体贡献
5. **实验设计**: 具体的验证方案和评估方法
6. **潜在风险**: 可能遇到的挑战和解决思路

请以JSON格式返回：
{{
  "title": "研究题目",
  "hypothesis": "核心研究假设",
  "innovation_points": ["创新点1", "创新点2", "创新点3"],
  "contributions": ["贡献1", "贡献2"],
  "experiments": [
    {{
      "name": "实验名称",
      "description": "实验描述", 
      "metrics": ["评估指标1", "评估指标2"],
      "datasets": ["数据集1", "数据集2"]
    }}
  ],
  "risks": ["风险1", "风险2"],
  "technical_approach": "具体的技术实现路径"
}}
"""

        try:
            response_data = await self.llm.generate(
                model_name=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,  # 通用想法可以更有创造性
                max_tokens=15000,
                agent_name=self.name,
                task_type="idea_generation"
            )
            response = response_data.get("content", "")
            
            # 解析LLM响应
            import json
            import re
            try:
                if "```json" in response:
                    json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
                    if json_match:
                        json_content = json_match.group(1).strip()
                    else:
                        json_start = response.find("```json") + 7
                        json_content = response[json_start:].strip()
                        if json_content.endswith("```"):
                            json_content = json_content[:-3].strip()
                else:
                    json_content = response.strip()
                
                result = json.loads(json_content)
                
                # 补充资产信息
                result['assets'] = [
                    {'type': 'Unknown', 'id': 'to_be_determined', 'availability': 'unknown'}
                ]
                
                print(f"    💡 LLM生成通用想法: {result.get('title', '未知标题')}")
                return result
                
            except json.JSONDecodeError:
                print(f"    ⚠️ LLM通用想法生成响应解析失败，使用备用模板")
                return self._generate_fallback_generic_idea(pattern, node_names)
                
        except Exception as e:
            print(f"    ❌ LLM通用想法生成失败: {str(e)}，使用备用模板")
            return self._generate_fallback_generic_idea(pattern, node_names)
    
    def _generate_fallback_generic_idea(self, pattern: str, node_names: List[str]) -> Dict[str, Any]:
        """备用的通用想法生成模板。"""
        return {
            'title': f'基于{pattern}模式的创新研究',
            'hypothesis': f'通过分析{", ".join(node_names)}之间的关系，探索新的研究机会',
            'innovation_points': [
                f'新颖的{pattern}应用',
                '跨领域的方法迁移',
                '创新的评估框架'
            ],
            'contributions': ['探索性研究', '方法学贡献'],
            'assets': [
                {'type': 'Unknown', 'id': 'to_be_determined', 'availability': 'unknown'}
            ],
            'experiments': [
                {'name': '可行性验证', 'metric': 'proof_of_concept', 'dataset': 'pilot_study'}
            ],
            'risks': [
                '研究方向的不确定性',
                '预期效果的不可预测性'
            ]
        }

    async def _rank_and_filter(self, candidates: List[CandidateIdea], max_ideas: int) -> List[CandidateIdea]:
        """对候选想法进行排序与过滤。
        
        输入:
            - candidates: 原始候选想法列表。
            - max_ideas: 返回数量上限。
            
        输出:
            - List[CandidateIdea]: 排序后的top-k想法。
            
        实现思路:
            1) 去重：基于 core_hypothesis 相似度检测重复想法。
            2) 预评估：根据触发置信度、节点重要性、风险评估等计算初步得分。
            3) 排序返回top-k。
        """
        if not candidates:
            return []
        
        print(f"    📊 对 {len(candidates)} 个候选想法进行排序与过滤")
        
        # 步骤1：去重
        unique_candidates = await self._deduplicate_ideas(candidates)
        print(f"    🔄 去重后保留 {len(unique_candidates)} 个想法")
        
        # 步骤2：预评估
        scored_candidates = []
        for candidate in unique_candidates:
            score = await self._calculate_preliminary_score(candidate)
            scored_candidates.append((candidate, score))
        
        # 步骤3：排序
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # 步骤4：返回top-k
        final_candidates = [candidate for candidate, score in scored_candidates[:max_ideas]]
        
        print(f"    ✅ 最终保留 {len(final_candidates)} 个高质量想法")
        return final_candidates
    
    async def _deduplicate_ideas(self, candidates: List[CandidateIdea]) -> List[CandidateIdea]:
        """对想法进行去重。"""
        if len(candidates) <= 1:
            return candidates
        
        unique_candidates = []
        seen_hypotheses = set()
        
        for candidate in candidates:
            # 简化的去重：基于核心假设的关键词重叠
            hypothesis_keywords = set(candidate.core_hypothesis.lower().split())
            
            # 检查是否与已有想法重复
            is_duplicate = False
            for seen_hypothesis in seen_hypotheses:
                seen_keywords = set(seen_hypothesis.split())
                overlap = len(hypothesis_keywords & seen_keywords)
                
                # 如果重叠度超过50%，认为是重复
                if overlap > 0 and overlap / len(hypothesis_keywords | seen_keywords) > 0.5:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_candidates.append(candidate)
                seen_hypotheses.add(candidate.core_hypothesis.lower())
        
        return unique_candidates
    
    async def _calculate_preliminary_score(self, candidate: CandidateIdea) -> float:
        """计算想法的初步得分。"""
        score = 0.0
        
        # 因子1：触发置信度 (30%)
        trigger_confidence = candidate.provenance.get('confidence', 0.5)
        score += trigger_confidence * 0.3
        
        # 因子2：创新点数量和质量 (25%)
        innovation_count = len(candidate.initial_innovation_points)
        innovation_score = min(innovation_count / 3.0, 1.0)  # 最多3个创新点得满分
        score += innovation_score * 0.25
        
        # 因子3：预期贡献多样性 (20%)
        contribution_count = len(candidate.expected_contribution)
        contribution_score = min(contribution_count / 2.0, 1.0)  # 最多2个贡献得满分
        score += contribution_score * 0.20
        
        # 因子4：风险评估 (15%) - 风险越少得分越高
        risk_count = len(candidate.risks)
        risk_penalty = min(risk_count / 5.0, 1.0)  # 最多5个风险全扣分
        score += (1.0 - risk_penalty) * 0.15
        
        # 因子5：实验设计完整性 (10%)
        experiment_count = len(candidate.preliminary_experiments)
        experiment_score = min(experiment_count / 2.0, 1.0)  # 最多2个实验得满分
        score += experiment_score * 0.10
        
        return min(score, 1.0)  # 确保得分不超过1.0

    async def refine_idea(self, idea: CandidateIdea, refinement_prompt: RefinementPrompt, graph: SemanticOpportunityGraph) -> CandidateIdea:
        """根据精炼指令生成想法的新版本。
        
        输入:
            - idea: 当前想法版本。
            - refinement_prompt: 精炼指令。
            - graph: 语义机会图谱。
            
        输出:
            - CandidateIdea: 精炼后的新版本想法（version+1）。
            
        实现思路:
            1) 解析 refinement_prompt.instructions，识别保留/替换/补充的具体项目。
            2) 调用 LLM 重新生成对应字段，保持其他字段不变或局部调整。
            3) 更新 version、provenance，记录变更轨迹。
        """
        print(f"    🔄 精炼想法 {idea.id} (当前版本: {idea.version})")
        
        # 创建新版本的想法，基于原想法
        refined_idea = CandidateIdea(
            id=idea.id,
            title=idea.title,
            core_hypothesis=idea.core_hypothesis,
            initial_innovation_points=idea.initial_innovation_points.copy(),
            source_trigger_nodes=idea.source_trigger_nodes.copy(),
            expected_contribution=idea.expected_contribution.copy(),
            required_assets=idea.required_assets.copy(),
            preliminary_experiments=idea.preliminary_experiments.copy(),
            risks=idea.risks.copy(),
            provenance=idea.provenance.copy(),
            version=idea.version + 1
        )
        
        # 解析精炼指令并应用修改
        instructions = refinement_prompt.instructions
        
        for instruction in instructions:
            await self._apply_refinement_instruction(refined_idea, instruction, graph)
        
        # 更新provenance记录变更轨迹
        refined_idea.provenance.update({
            "refined_at": datetime.now().isoformat(),
            "refinement_reason": refinement_prompt.rationale,
            "previous_version": idea.version,
            "applied_instructions": instructions
        })
        
        print(f"    ✨ 完成想法精炼，新版本: {refined_idea.version}")
        return refined_idea
    
    async def _apply_refinement_instruction(self, idea: CandidateIdea, instruction: str, graph: SemanticOpportunityGraph) -> None:
        """应用单个精炼指令。"""
        instruction_lower = instruction.lower()
        
        # 指令类型1：替换资产
        if "替换" in instruction and ("资产" in instruction or "dataset" in instruction_lower or "method" in instruction_lower):
            await self._refine_assets(idea, instruction, graph)
        
        # 指令类型2：强化差异点
        elif "强化" in instruction and "差异" in instruction:
            await self._enhance_innovation_points(idea, instruction)
        
        # 指令类型3：缓解风险
        elif "缓解" in instruction and "风险" in instruction:
            await self._mitigate_risks(idea, instruction)
        
        # 指令类型4：改进标题或假设
        elif "改进" in instruction and ("标题" in instruction or "假设" in instruction):
            await self._improve_core_content(idea, instruction)
        
        # 指令类型5：补充实验
        elif "补充" in instruction and "实验" in instruction:
            await self._supplement_experiments(idea, instruction)
        
        else:
            # 通用指令处理
            await self._apply_generic_instruction(idea, instruction)
    
    async def _refine_assets(self, idea: CandidateIdea, instruction: str, graph: SemanticOpportunityGraph) -> None:
        """精炼所需资产。"""
        # 简化实现：添加替代资产
        if "dataset" in instruction.lower():
            # 替换数据集
            new_asset = {
                'type': 'Dataset',
                'id': 'alternative_dataset',
                'availability': 'public'
            }
            idea.required_assets.append(new_asset)
        
        elif "method" in instruction.lower():
            # 替换方法
            new_asset = {
                'type': 'Method',
                'id': 'alternative_method',
                'availability': 'open_source'
            }
            idea.required_assets.append(new_asset)
    
    async def _enhance_innovation_points(self, idea: CandidateIdea, instruction: str) -> None:
        """强化创新点。"""
        # 添加差异化的创新点
        enhanced_points = [
            "引入新颖的技术机制",
            "建立创新的评估框架",
            "提出独特的理论分析视角"
        ]
        
        # 避免重复添加
        for point in enhanced_points:
            if point not in idea.initial_innovation_points:
                idea.initial_innovation_points.append(point)
                break  # 只添加一个新的创新点
    
    async def _mitigate_risks(self, idea: CandidateIdea, instruction: str) -> None:
        """缓解风险。"""
        # 简化实现：为现有风险添加缓解策略
        risk_mitigations = [
            "通过渐进式实验验证减少不确定性",
            "建立多重验证机制确保结果可靠性",
            "设计备选方案应对潜在失败"
        ]
        
        # 将缓解策略添加到创新点或者修改风险描述
        if risk_mitigations[0] not in idea.initial_innovation_points:
            idea.initial_innovation_points.append(risk_mitigations[0])
    
    async def _improve_core_content(self, idea: CandidateIdea, instruction: str) -> None:
        """改进核心内容。"""
        if "标题" in instruction:
            # 改进标题，使其更具体和吸引力
            if "基于" not in idea.title:
                idea.title = f"基于创新方法的{idea.title}"
        
        elif "假设" in instruction:
            # 改进核心假设，使其更明确
            if "预期" not in idea.core_hypothesis:
                idea.core_hypothesis += "，预期能够取得显著的性能提升和理论贡献"
    
    async def _supplement_experiments(self, idea: CandidateIdea, instruction: str) -> None:
        """补充实验设计。"""
        # 添加更全面的实验设计
        additional_experiments = [
            {'name': '消融研究', 'metric': 'component_analysis', 'dataset': 'validation_set'},
            {'name': '泛化能力测试', 'metric': 'cross_domain_performance', 'dataset': 'diverse_testset'},
            {'name': '效率对比分析', 'metric': 'computational_efficiency', 'dataset': 'benchmark_suite'}
        ]
        
        # 避免重复添加相同的实验
        existing_names = {exp.get('name', '') for exp in idea.preliminary_experiments}
        for exp in additional_experiments:
            if exp['name'] not in existing_names:
                idea.preliminary_experiments.append(exp)
                break  # 只添加一个新实验
    
    async def _apply_generic_instruction(self, idea: CandidateIdea, instruction: str) -> None:
        """应用通用指令。"""
        # 通用改进：增加一个综合性的创新点
        generic_improvement = "结合多种先进技术实现综合性能提升"
        
        if generic_improvement not in idea.initial_innovation_points:
            idea.initial_innovation_points.append(generic_improvement)


class NoveltyCriticAgent(BaseIdeaAgent):
    """第三阶段：新颖性批判（并行之一）。

    核心职责:
        - 采用 RAG 两阶段(召回→重排)对 `CandidateIdea` 进行多分面新颖性评审。

    注意事项:
        - 召回构造需融合 idea.title/core_hypothesis/trigger_nodes 的关键词及别名。
        - 重排提示需覆盖"概念/方法/应用/评测/证据"分面，并产出差异性主张 `difference_claims`。
    """

    def __init__(self, name: str, llm_factory: LLMFactory, db: AcademicPaperDatabase, config: Optional[AgentConfig] = None):
        super().__init__(name, llm_factory, db, config)
        self.novelty_facets = ["conceptual", "methodological", "application", "evaluation"]

    async def assess_novelty(self, idea: CandidateIdea, retrieve_k: int = 30) -> NoveltyCritique:
        """评估单个想法的新颖性。"""
        results = await self.assess_novelty_batch([idea], retrieve_k)
        return results[0]
    
    async def assess_novelty_batch(self, ideas: List[CandidateIdea], retrieve_k: int = 30) -> List[NoveltyCritique]:
        """批量评估想法的新颖性。"""
        import asyncio
        
        print(f"🔬 开始批量新颖性评估：{len(ideas)}个想法")
        
        # 创建并发任务
        tasks = []
        for idea in ideas:
            task = asyncio.create_task(self._assess_single_novelty(idea, retrieve_k))
            tasks.append(task)
        
        # 等待所有任务完成
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 过滤成功的结果
        critiques = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"⚠️ 想法{i}新颖性评估失败: {str(result)}")
                # 创建默认的critique
                critiques.append(NoveltyCritique(
                    idea_id=ideas[i].id,
                    novelty_score=5.0,
                    facet_scores={"概念": 5.0, "方法": 5.0, "应用": 5.0, "评测": 5.0},
                    similar_works=[],
                    difference_claims=[],
                    method={"error": f"评估失败: {str(result)}"}
                ))
            else:
                critiques.append(result)
        
        return critiques
    
    async def assess_batch_comprehensive(self, ideas: List[CandidateIdea], graph: SemanticOpportunityGraph = None, retrieve_k: int = 30) -> List[NoveltyCritique]:
        """一次LLM调用对整批想法进行综合新颖性评估。"""
        print(f"🔬 开始批量综合新颖性评估：{len(ideas)}个想法")
        
        # 为整批想法构建综合评估prompt
        ideas_summary = "\n".join([
            f"{i+1}. {idea.title}\n   核心假设: {idea.core_hypothesis}\n   创新点: {', '.join(idea.initial_innovation_points)}"
            for i, idea in enumerate(ideas)
        ])
        
        prompt = f"""
作为资深学术专家，请对以下{len(ideas)}个研究想法进行综合新颖性评估。

**待评估想法列表**：
{ideas_summary}

**评估任务**：
1. 对每个想法从多个维度评估新颖性（概念、方法、应用、评测）
2. 识别想法间的相互关系和差异化程度
3. 给出每个想法的新颖性分数(1-10分)
4. 提供评估理由和建议

请以JSON格式返回：
{{
  "batch_analysis": "对整批想法的综合分析",
  "novelty_assessments": [
    {{
      "idea_index": 1,
      "novelty_score": X.X,
      "facet_scores": {{
        "conceptual": X.X,
        "methodological": X.X, 
        "application": X.X,
        "evaluation": X.X
      }},
      "reasoning": "详细评估理由",
      "strengths": ["优势1", "优势2"],
      "concerns": ["关注点1", "关注点2"],
      "differentiation": "与其他想法的差异化程度"
    }}
  ]
}}
"""
        
        try:
            response_data = await self.llm.generate(
                model_name=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=15000,
                agent_name=self.name,
                task_type="batch_novelty_assessment"
            )
            response = response_data.get("content", "")
            
            # 解析JSON响应
            import json
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_content = json_match.group(1).strip()
            else:
                json_content = response.strip()
            
            result = json.loads(json_content)
            assessments = result.get("novelty_assessments", [])
            
            # 为每个想法创建NoveltyCritique对象
            critiques = []
            for i, idea in enumerate(ideas):
                if i < len(assessments):
                    assessment = assessments[i]
                    critique = NoveltyCritique(
                        idea_id=idea.id,
                        novelty_score=assessment.get("novelty_score", 5.0),
                        facet_scores=assessment.get("facet_scores", {"概念": 5.0, "方法": 5.0, "应用": 5.0, "评测": 5.0}),
                        similar_works=[],
                        difference_claims=[assessment.get("differentiation", "适度创新")],
                        method={"type": "batch_comprehensive", "reasoning": assessment.get("reasoning", "")}
                    )
                else:
                    # 默认评估
                    critique = NoveltyCritique(
                        idea_id=idea.id,
                        novelty_score=5.0,
                        facet_scores={"概念": 5.0, "方法": 5.0, "应用": 5.0, "评测": 5.0},
                        similar_works=[],
                        difference_claims=["待进一步评估"],
                        method={"type": "batch_comprehensive", "error": "批量评估解析失败"}
                    )
                critiques.append(critique)
            
            print(f"✅ 批量综合新颖性评估完成")
            return critiques
            
        except Exception as e:
            print(f"❌ 批量综合新颖性评估失败: {str(e)}")
            # 返回默认评估
            return [NoveltyCritique(
                idea_id=idea.id,
                novelty_score=5.0,
                facet_scores={"概念": 5.0, "方法": 5.0, "应用": 5.0, "评测": 5.0},
                similar_works=[],
                difference_claims=["评估失败"],
                method={"type": "batch_comprehensive", "error": str(e)}
            ) for idea in ideas]
    
    async def _assess_single_novelty(self, idea: CandidateIdea, retrieve_k: int = 30) -> NoveltyCritique:
        """对单个想法进行新颖性评审。

        输入:
            - idea: 候选想法。
            - retrieve_k: RAG召回量（默认30）。

        输出:
            - NoveltyCritique: 包含分面得分、相似工作清单、差异主张与方法说明。

        实现步骤建议:
            1) 构造多样化查询（主术语+别名+触发节点+任务/指标）。
            2) `db.search_content` 召回并去重；按分面用 LLM 重排与对比。
            3) 汇总 facet_scores/novelty_score，生成 difference_claims 与 similar_works。
        """
        # 阶段1：召回相关文献
        retrieved_papers = await self._retrieve_similar_works(idea, retrieve_k)
        
        # 阶段2：多分面重排与对比
        facet_results = await self._rerank_by_facets(idea, retrieved_papers)
        
        # 阶段3：综合评估与差异性分析
        return await self._synthesize_novelty_critique(idea, facet_results)

    async def _retrieve_similar_works(self, idea: CandidateIdea, k: int) -> List[Dict[str, Any]]:
        """RAG第一阶段：召回相似工作。
        
        输入:
            - idea: 候选想法。
            - k: 召回数量。
            
        输出:
            - List[Dict]: 召回的文献列表，每个包含 paper_id、content、metadata。
            
        实现思路:
            1) 从 idea.title/core_hypothesis/source_trigger_nodes 抽取关键词。
            2) 构造多样化查询：核心概念+方法名+任务名+触发节点名称。
            3) 调用 `db.search_content` 进行向量检索。
            4) 去重（同一论文的多个片段）与质量过滤。
        """
        print(f"    🔍 开始召回与想法 '{idea.title}' 相似的文献")
        
        # 步骤1：构造多样化查询
        queries = await self._construct_search_queries(idea)
        print(f"    📝 构造了 {len(queries)} 个查询")
        
        # 步骤2：执行检索
        all_results = []
        for query in queries:
            try:
                # 使用数据库搜索，每个查询召回k/2个结果
                results = self.db.search_content(
                    query, 
                    content_type="texts", 
                    n_results=max(k // len(queries), 5)
                )
                
                # 标准化检索结果格式
                for result in results:
                    processed_result = {
                        "paper_id": result.get("paper_id", "unknown"),
                        "content": result.get("content", ""),
                        "metadata": {
                            "title": result.get("title", ""),
                            "authors": result.get("authors", []),
                            "venue": result.get("venue", ""),
                            "year": result.get("year", ""),
                            "query_source": query,
                            "relevance_score": result.get("score", 0.0)
                        }
                    }
                    all_results.append(processed_result)
                    
            except Exception as e:
                print(f"    ⚠️ 查询 '{query}' 失败: {e}")
                continue
        
        # 步骤3：去重处理
        deduplicated_results = await self._deduplicate_papers(all_results)
        print(f"    🔄 去重后保留 {len(deduplicated_results)} 个文献片段")
        
        # 步骤4：质量过滤和排序
        filtered_results = await self._filter_and_rank_papers(deduplicated_results, k)
        print(f"    ✅ 最终召回 {len(filtered_results)} 个高质量文献")
        
        return filtered_results
    
    async def _construct_search_queries(self, idea: CandidateIdea) -> List[str]:
        """构造多样化的搜索查询。"""
        queries = []
        
        # 查询1：基于标题的核心概念
        title_keywords = self._extract_core_concepts(idea.title)
        if title_keywords:
            queries.append(" ".join(title_keywords))
        
        # 查询2：基于核心假设的方法和任务
        hypothesis_keywords = self._extract_core_concepts(idea.core_hypothesis)
        if hypothesis_keywords:
            queries.append(" ".join(hypothesis_keywords))
        
        # 查询3：基于创新点的技术关键词
        innovation_keywords = []
        for point in idea.initial_innovation_points:
            innovation_keywords.extend(self._extract_core_concepts(point))
        if innovation_keywords:
            # 取前3个最重要的创新关键词
            queries.append(" ".join(innovation_keywords[:3]))
        
        # 查询4：基于触发节点的专门术语
        if hasattr(idea, 'source_trigger_nodes') and idea.source_trigger_nodes:
            # 简化处理：直接使用节点ID作为查询
            trigger_query = " ".join(idea.source_trigger_nodes[:2])  # 最多2个节点
            if trigger_query.strip():
                queries.append(trigger_query)
        
        # 查询5：组合查询（标题+方法）
        if len(title_keywords) > 0 and len(hypothesis_keywords) > 0:
            combined_query = f"{title_keywords[0]} {hypothesis_keywords[0]}"
            queries.append(combined_query)
        
        # 去除空查询和过短查询
        filtered_queries = [q for q in queries if len(q.strip()) > 3]
        
        return filtered_queries[:5]  # 最多5个查询，避免过度检索
    
    def _extract_core_concepts(self, text: str) -> List[str]:
        """从文本中提取核心概念。"""
        import re
        
        # 简化的关键词提取：基于常见学术术语模式
        patterns = [
            r'\b[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*\b',  # 大写开头的专有名词
            r'\b(?:learning|model|algorithm|method|approach|technique|framework)\b',  # 核心方法词
            r'\b(?:classification|detection|generation|optimization|prediction|analysis)\b',  # 任务词
            r'\b(?:neural|deep|machine|artificial|intelligence|network)\b',  # AI相关词
            r'\b(?:transformer|attention|lstm|cnn|bert|gpt)\b',  # 具体模型名
        ]
        
        keywords = []
        text_lower = text.lower()
        
        for pattern in patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            keywords.extend(matches)
        
        # 去重并保持顺序
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw.lower() not in seen:
                seen.add(kw.lower())
                unique_keywords.append(kw)
        
        return unique_keywords[:5]  # 最多5个关键词
    
    async def _deduplicate_papers(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """对检索结果进行去重。"""
        seen_papers = {}
        deduplicated = []
        
        for result in results:
            paper_id = result["paper_id"]
            
            if paper_id in seen_papers:
                # 如果已存在，保留相关性分数更高的
                existing_score = seen_papers[paper_id]["metadata"]["relevance_score"]
                current_score = result["metadata"]["relevance_score"]
                
                if current_score > existing_score:
                    # 替换为分数更高的结果
                    seen_papers[paper_id] = result
            else:
                seen_papers[paper_id] = result
        
        return list(seen_papers.values())
    
    async def _filter_and_rank_papers(self, results: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        """过滤和排序检索结果。"""
        # 过滤：移除内容过短或质量过低的结果
        filtered = []
        for result in results:
            content = result["content"]
            score = result["metadata"]["relevance_score"]
            
            # 基本质量过滤
            if len(content) >= 50 and score > 0.1:  # 最低质量门槛
                filtered.append(result)
        
        # 排序：按相关性分数降序
        filtered.sort(key=lambda x: x["metadata"]["relevance_score"], reverse=True)
        
        # 返回top-k
        return filtered[:k]

    async def _rerank_by_facets(self, idea: CandidateIdea, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """RAG第二阶段：分面重排与对比。
        
        输入:
            - idea: 候选想法。
            - papers: 召回的文献列表。
            
        输出:
            - Dict: 分面分析结果，包含每个facet的相似工作与差异分析。
            
        实现思路:
            1) 对每个novelty facet（概念/方法/应用/评测），用专门提示词让LLM对比idea与papers。
            2) 识别最相似的工作，计算相似度分数，提取关键差异点。
            3) 生成分面得分(1-10)与差异性主张（claim+evidence）。
        """
        print(f"    🔬 开始多分面新颖性分析")
        
        facet_results = {}
        
        # 对每个新颖性分面进行分析
        for facet in self.novelty_facets:
            print(f"    📊 分析 {facet} 分面")
            
            try:
                # 分面特定的分析
                facet_analysis = await self._analyze_single_facet(idea, papers, facet)
                facet_results[facet] = facet_analysis
                
            except Exception as e:
                print(f"    ⚠️ {facet} 分面分析失败: {e}")
                # 设置默认分析结果
                facet_results[facet] = {
                    "score": 5.0,  # 中等分数
                    "most_similar_papers": [],
                    "differences": [f"{facet}分面分析失败"],
                    "analysis_method": "fallback"
                }
        
        print(f"    ✅ 完成 {len(facet_results)} 个分面的分析")
        return facet_results
    
    async def _analyze_single_facet(self, idea: CandidateIdea, papers: List[Dict[str, Any]], facet: str) -> Dict[str, Any]:
        """分析单个新颖性分面。"""
        
        # 选择最相关的论文进行详细对比（最多5篇）
        top_papers = papers[:5]
        
        if not top_papers:
            return {
                "score": 8.0,  # 如果没有相似工作，认为较新颖
                "most_similar_papers": [],
                "differences": [f"未发现{facet}分面的直接相似工作"],
                "analysis_method": "no_similar_works"
            }
        
        # 根据分面类型进行专门分析
        if facet == "conceptual":
            return await self._analyze_conceptual_novelty(idea, top_papers)
        elif facet == "methodological":
            return await self._analyze_methodological_novelty(idea, top_papers)
        elif facet == "application":
            return await self._analyze_application_novelty(idea, top_papers)
        elif facet == "evaluation":
            return await self._analyze_evaluation_novelty(idea, top_papers)
        else:
            # 通用分析
            return await self._analyze_generic_novelty(idea, top_papers, facet)
    
    async def _analyze_conceptual_novelty(self, idea: CandidateIdea, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """基于LLM的概念新颖性分析。"""
        
        # 准备相似文献的上下文
        papers_context = ""
        for i, paper in enumerate(papers[:3]):  # 只分析最相似的3篇
            papers_context += f"\n论文 {i+1}: {paper['metadata']['title']}\n"
            papers_context += f"内容摘要: {paper['content'][:500]}...\n"
        
        if not papers_context.strip():
            papers_context = "未找到直接相关的已有工作。"
        
        # 构造LLM提示词
        prompt = f"""
作为学术专家，请从**概念创新**角度评估以下研究想法的新颖性。

**待评估想法**：
- 标题: {idea.title}
- 核心假设: {idea.core_hypothesis}
- 创新点: {', '.join(idea.initial_innovation_points)}

**已有相关工作**：
{papers_context}

请从以下维度进行概念新颖性分析：

1. **核心概念创新性**: 想法引入的核心概念是否新颖？与已有概念框架的区别？
2. **理论贡献**: 是否提出了新的理论观点或概念模型？
3. **概念整合**: 是否创新性地结合了不同领域的概念？
4. **问题定义**: 是否以新的角度重新定义了问题？

请基于你的分析给出评分，并以JSON格式返回分析结果：
{{
  "score": X.X,  // 概念新颖性评分 (1-10分，10分最新颖，请基于实际分析给出)
  "analysis": {{
    "core_innovation": "详细分析核心概念的创新性",
    "theoretical_contribution": "分析理论贡献",
    "concept_integration": "分析概念整合的创新性",
    "problem_redefinition": "分析问题重新定义的程度"
  }},
  "most_similar_papers": [
    {{
      "title": "最相似论文标题",
      "similarity_reason": "相似性分析",
      "key_differences": "关键差异点"
    }}
  ],
  "differences": [
    "差异点1：具体的概念创新差异",
    "差异点2：理论框架的差异"
  ],
  "strengths": ["概念新颖性的优势1", "概念新颖性的优势2"],
  "weaknesses": ["可能的概念局限性1", "可能的概念局限性2"]
}}
"""

        try:
            response_data = await self.llm.generate(
                model_name=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=15000,
                agent_name=self.name,
                task_type="novelty_assessment"
            )
            response = response_data.get("content", "")
            
            # 解析LLM响应
            import json
            import re
            try:
                if "```json" in response:
                    json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
                    if json_match:
                        json_content = json_match.group(1).strip()
                    else:
                        json_start = response.find("```json") + 7
                        json_content = response[json_start:].strip()
                        if json_content.endswith("```"):
                            json_content = json_content[:-3].strip()
                else:
                    json_content = response.strip()
                
                result = json.loads(json_content)
                
                # 标准化输出格式
                return {
                    "score": float(result.get("score", 5.0)),
                    "most_similar_papers": result.get("most_similar_papers", [])[:3],
                    "differences": result.get("differences", ["LLM分析的概念差异"]),
                    "analysis_method": "llm_conceptual_analysis",
                    "detailed_analysis": result.get("analysis", {}),
                    "strengths": result.get("strengths", []),
                    "weaknesses": result.get("weaknesses", [])
                }
                
            except json.JSONDecodeError:
                print(f"    ⚠️ LLM概念新颖性分析响应解析失败，使用备用评估")
                return self._fallback_conceptual_analysis(idea, papers)
                
        except Exception as e:
            print(f"    ❌ LLM概念新颖性分析失败: {str(e)}，使用备用评估")
            return self._fallback_conceptual_analysis(idea, papers)
    
    def _fallback_conceptual_analysis(self, idea: CandidateIdea, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """备用的概念新颖性分析。"""
        return {
            "score": 6.0,  # 中等偏上分数
            "most_similar_papers": [{"title": p["metadata"]["title"], "similarity_reason": "关键词匹配"} for p in papers[:2]],
            "differences": ["概念框架存在一定创新性", "理论贡献需要进一步验证"],
            "analysis_method": "fallback_analysis",
            "detailed_analysis": {"core_innovation": "分析失败，使用备用评估"},
            "strengths": ["想法结构完整"],
            "weaknesses": ["需要更深入的概念分析"]
        }
    
    async def _analyze_methodological_novelty(self, idea: CandidateIdea, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """基于LLM的方法新颖性分析。"""
        
        # 准备相似文献的上下文
        papers_context = ""
        for i, paper in enumerate(papers[:3]):
            papers_context += f"\n论文 {i+1}: {paper['metadata']['title']}\n"
            papers_context += f"内容摘要: {paper['content'][:500]}...\n"
        
        if not papers_context.strip():
            papers_context = "未找到直接相关的已有工作。"
        
        prompt = f"""
作为学术专家，请从**方法创新**角度评估以下研究想法的新颖性。

**待评估想法**：
- 标题: {idea.title}
- 核心假设: {idea.core_hypothesis}
- 创新点: {', '.join(idea.initial_innovation_points)}

**已有相关工作**：
{papers_context}

请从以下维度进行方法新颖性分析：

1. **技术方法创新**: 是否提出了新的算法、模型或技术框架？
2. **方法组合创新**: 是否创新性地组合了现有方法？
3. **实现创新**: 在具体实现层面是否有技术突破？
4. **优化创新**: 是否在效率、准确性或资源使用上有方法学改进？

请以JSON格式返回分析结果：
{{
  "score": 7.5,  // 方法新颖性评分 (1-10分，10分最新颖)
  "analysis": {{
    "technical_innovation": "技术方法创新分析",
    "combination_innovation": "方法组合创新分析",
    "implementation_innovation": "实现层面创新分析",
    "optimization_innovation": "优化改进分析"
  }},
  "most_similar_papers": [
    {{
      "title": "最相似论文标题",
      "method_similarity": "方法相似性分析",
      "key_differences": "关键技术差异"
    }}
  ],
  "differences": [
    "差异点1：具体的技术方法差异",
    "差异点2：算法架构的创新"
  ],
  "strengths": ["方法创新的优势1", "方法创新的优势2"],
  "potential_issues": ["可能的技术挑战1", "可能的技术挑战2"]
}}
"""

        try:
            response_data = await self.llm.generate(
                model_name=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=15000,
                agent_name=self.name,
                task_type="novelty_assessment"
            )
            response = response_data.get("content", "")
            
            # 解析LLM响应
            import json
            import re
            try:
                if "```json" in response:
                    json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
                    if json_match:
                        json_content = json_match.group(1).strip()
                    else:
                        json_start = response.find("```json") + 7
                        json_content = response[json_start:].strip()
                        if json_content.endswith("```"):
                            json_content = json_content[:-3].strip()
                else:
                    json_content = response.strip()
                
                result = json.loads(json_content)
                
                return {
                    "score": float(result.get("score", 5.0)),
                    "most_similar_papers": result.get("most_similar_papers", [])[:3],
                    "differences": result.get("differences", ["LLM分析的方法差异"]),
                    "analysis_method": "llm_methodological_analysis",
                    "detailed_analysis": result.get("analysis", {}),
                    "strengths": result.get("strengths", []),
                    "potential_issues": result.get("potential_issues", [])
                }
                
            except json.JSONDecodeError:
                print(f"    ⚠️ LLM方法新颖性分析响应解析失败，使用备用评估")
                return {"score": 6.0, "most_similar_papers": [], "differences": ["方法分析失败"], "analysis_method": "fallback"}
                
        except Exception as e:
            print(f"    ❌ LLM方法新颖性分析失败: {str(e)}，使用备用评估")
            return {"score": 6.0, "most_similar_papers": [], "differences": ["方法分析失败"], "analysis_method": "fallback"}
    
    async def _analyze_application_novelty(self, idea: CandidateIdea, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析应用新颖性。"""
        
        # 提取应用领域信息
        idea_domains = self._extract_application_domains(idea.title + " " + idea.core_hypothesis)
        
        similar_papers = []
        max_similarity = 0.0
        
        for paper in papers:
            paper_domains = self._extract_application_domains(paper["content"])
            
            if len(idea_domains) > 0:
                overlap = len(set(idea_domains) & set(paper_domains))
                similarity = overlap / max(len(set(idea_domains) | set(paper_domains)), 1)
                
                similar_papers.append({
                    "paper_id": paper["paper_id"],
                    "title": paper["metadata"]["title"],
                    "similarity": similarity,
                    "shared_domains": list(set(idea_domains) & set(paper_domains))
                })
                
                max_similarity = max(max_similarity, similarity)
        
        similar_papers.sort(key=lambda x: x["similarity"], reverse=True)
        
        # 基于相似度计算应用新颖性评分（如果没有LLM评估结果）
        novelty_score = max(1.0, 8.5 - max_similarity * 7.5)
        
        differences = []
        if max_similarity < 0.3:
            differences.append("开拓了全新的应用领域")
        elif max_similarity < 0.6:
            differences.append("在现有应用基础上扩展到新的场景")
        else:
            differences.append("应用场景与现有工作重叠较多")
            
        return {
            "score": novelty_score,
            "most_similar_papers": similar_papers[:3],
            "differences": differences,
            "analysis_method": "application_analysis"
        }
    
    async def _analyze_evaluation_novelty(self, idea: CandidateIdea, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析评估新颖性。"""
        
        # 提取实验和评估相关信息
        idea_evaluations = []
        for exp in idea.preliminary_experiments:
            idea_evaluations.append(exp.get('metric', ''))
            idea_evaluations.append(exp.get('dataset', ''))
        
        similar_papers = []
        max_similarity = 0.0
        
        for paper in papers:
            paper_evaluations = self._extract_evaluation_terms(paper["content"])
            
            if len(idea_evaluations) > 0:
                overlap = len(set(idea_evaluations) & set(paper_evaluations))
                similarity = overlap / max(len(set(idea_evaluations) | set(paper_evaluations)), 1)
                
                similar_papers.append({
                    "paper_id": paper["paper_id"],
                    "title": paper["metadata"]["title"],
                    "similarity": similarity,
                    "shared_evaluations": list(set(idea_evaluations) & set(paper_evaluations))
                })
                
                max_similarity = max(max_similarity, similarity)
        
        similar_papers.sort(key=lambda x: x["similarity"], reverse=True)
        
        # 基于相似度计算评估新颖性评分（如果没有LLM评估结果）
        novelty_score = max(1.0, 9.0 - max_similarity * 8.0)
        
        differences = []
        if max_similarity < 0.25:
            differences.append("设计了创新的评估框架和指标")
        elif max_similarity < 0.55:
            differences.append("在现有评估基础上引入了新的评估维度")
        else:
            differences.append("评估方法与现有工作相似度较高")
            
        return {
            "score": novelty_score,
            "most_similar_papers": similar_papers[:3],
            "differences": differences,
            "analysis_method": "evaluation_analysis"
        }
    
    async def _analyze_generic_novelty(self, idea: CandidateIdea, papers: List[Dict[str, Any]], facet: str) -> Dict[str, Any]:
        """通用新颖性分析。"""
        
        # 基本的文本相似度分析
        idea_text = idea.title + " " + idea.core_hypothesis
        
        similarities = []
        for paper in papers:
            # 简化的相似度计算
            similarity = self._calculate_text_similarity(idea_text, paper["content"])
            similarities.append({
                "paper_id": paper["paper_id"],
                "title": paper["metadata"]["title"],
                "similarity": similarity
            })
        
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        max_similarity = similarities[0]["similarity"] if similarities else 0.0
        
        # 基于相似度计算通用新颖性评分（如果没有LLM评估结果）
        novelty_score = max(1.0, 8.0 - max_similarity * 7.0)
        
        return {
            "score": novelty_score,
            "most_similar_papers": similarities[:3],
            "differences": [f"{facet}分面显示适度的新颖性"],
            "analysis_method": "generic_similarity"
        }
    
    def _extract_method_terms(self, text: str) -> List[str]:
        """提取方法相关术语。"""
        import re
        
        method_patterns = [
            r'\b(?:algorithm|approach|method|technique|framework|model|architecture)\b',
            r'\b(?:neural|deep|machine|learning|training|optimization)\b',
            r'\b(?:transformer|attention|lstm|cnn|bert|gpt|resnet)\b',
            r'\b(?:supervised|unsupervised|reinforcement|transfer|meta)\b'
        ]
        
        methods = []
        text_lower = text.lower()
        
        for pattern in method_patterns:
            matches = re.findall(pattern, text_lower)
            methods.extend(matches)
        
        return list(set(methods))
    
    def _extract_application_domains(self, text: str) -> List[str]:
        """提取应用领域术语。"""
        import re
        
        domain_patterns = [
            r'\b(?:computer vision|natural language|speech|robotics|healthcare|finance)\b',
            r'\b(?:classification|detection|generation|translation|recommendation)\b',
            r'\b(?:image|text|video|audio|graph|code|medical)\b'
        ]
        
        domains = []
        text_lower = text.lower()
        
        for pattern in domain_patterns:
            matches = re.findall(pattern, text_lower)
            domains.extend(matches)
        
        return list(set(domains))
    
    def _extract_evaluation_terms(self, text: str) -> List[str]:
        """提取评估相关术语。"""
        import re
        
        eval_patterns = [
            r'\b(?:accuracy|precision|recall|f1|auc|bleu|rouge|meteor)\b',
            r'\b(?:benchmark|dataset|evaluation|metric|measure|score)\b',
            r'\b(?:test|validation|experiment|ablation|comparison)\b'
        ]
        
        evaluations = []
        text_lower = text.lower()
        
        for pattern in eval_patterns:
            matches = re.findall(pattern, text_lower)
            evaluations.extend(matches)
        
        return list(set(evaluations))
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度的简化实现。"""
        
        # 基于词汇重叠的简单相似度计算
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if len(words1) == 0 or len(words2) == 0:
            return 0.0
        
        overlap = len(words1 & words2)
        union = len(words1 | words2)
        
        return overlap / union if union > 0 else 0.0

    async def _synthesize_novelty_critique(self, idea: CandidateIdea, facet_results: Dict[str, Any]) -> NoveltyCritique:
        """第三阶段：综合新颖性评审。
        
        输入:
            - idea: 候选想法。
            - facet_results: 分面分析结果。
            
        输出:
            - NoveltyCritique: 完整的新颖性评审结果。
            
        实现思路:
            1) 汇总各分面得分，计算加权平均作为总体新颖性分数。
            2) 整理最相似工作列表，按相似度降序排列。
            3) 合并各分面的差异性主张，形成结构化的difference_claims。
            4) 记录评审方法与参数供复现。
        """
        print(f"    🎯 综合新颖性评审结果")
        
        # 步骤1：计算加权总体新颖性分数
        novelty_score = await self._calculate_weighted_novelty_score(facet_results)
        
        # 步骤2：整理最相似工作列表
        similar_works = await self._compile_similar_works(facet_results)
        
        # 步骤3：合并差异性主张
        difference_claims = await self._compile_difference_claims(facet_results)
        
        # 步骤4：生成分面得分详情
        facet_scores = {facet: result["score"] for facet, result in facet_results.items()}
        
        # 创建NoveltyCritique对象
        critique = NoveltyCritique(
            idea_id=idea.id,
            novelty_score=novelty_score,
            facet_scores=facet_scores,
            similar_works=similar_works,
            difference_claims=difference_claims,
            method={
                "approach": "multi_facet_rag",
                "facets": list(self.novelty_facets),
                "weights": self._get_facet_weights(),
                "retrieval_k": len(similar_works),
                "timestamp": datetime.now().isoformat()
            }
        )
        
        print(f"    ✅ 新颖性评分: {novelty_score:.2f}/10.0")
        print(f"    📋 发现 {len(similar_works)} 个相似工作")
        print(f"    💡 生成 {len(difference_claims)} 条差异性主张")
        
        return critique
    
    async def _calculate_weighted_novelty_score(self, facet_results: Dict[str, Any]) -> float:
        """计算加权新颖性分数。"""
        
        # 分面权重设置
        weights = self._get_facet_weights()
        
        total_score = 0.0
        total_weight = 0.0
        
        for facet, result in facet_results.items():
            weight = weights.get(facet, 0.25)  # 默认权重
            score = result.get("score", 5.0)
            
            total_score += score * weight
            total_weight += weight
        
        # 归一化
        if total_weight > 0:
            final_score = total_score / total_weight
        else:
            final_score = 5.0  # 默认中等分数
        
        # 确保分数在合理范围内
        return max(1.0, min(10.0, final_score))
    
    def _get_facet_weights(self) -> Dict[str, float]:
        """获取分面权重配置。"""
        return {
            "conceptual": 0.30,      # 概念新颖性最重要
            "methodological": 0.35,  # 方法新颖性次之
            "application": 0.20,     # 应用新颖性
            "evaluation": 0.15       # 评估新颖性
        }
    
    async def _compile_similar_works(self, facet_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """整理最相似工作列表。"""
        
        all_similar_works = {}  # 使用字典去重
        
        # 收集所有分面的相似工作
        for facet, result in facet_results.items():
            similar_papers = result.get("most_similar_papers", [])
            
            for paper in similar_papers:
                paper_id = paper.get("paper_id", "unknown")
                
                if paper_id in all_similar_works:
                    # 如果已存在，更新最高相似度和增加相关分面
                    existing = all_similar_works[paper_id]
                    current_similarity = paper.get("similarity", 0.0)
                    
                    if current_similarity > existing.get("max_similarity", 0.0):
                        existing["max_similarity"] = current_similarity
                    
                    existing["relevant_facets"].append(facet)
                else:
                    # 新的相似工作
                    all_similar_works[paper_id] = {
                        "paper_id": paper_id,
                        "title": paper.get("title", "Unknown Title"),
                        "max_similarity": paper.get("similarity", 0.0),
                        "relevant_facets": [facet],
                        "details": paper
                    }
        
        # 转换为列表并按相似度排序
        similar_works_list = list(all_similar_works.values())
        similar_works_list.sort(key=lambda x: x["max_similarity"], reverse=True)
        
        # 返回前10个最相似的工作
        return similar_works_list[:10]
    
    async def _compile_difference_claims(self, facet_results: Dict[str, Any]) -> List[str]:
        """合并差异性主张。"""
        
        all_claims = []
        
        for facet, result in facet_results.items():
            differences = result.get("differences", [])
            
            # 为每个差异点添加分面标识
            for diff in differences:
                claim = f"[{facet.upper()}] {diff}"
                all_claims.append(claim)
        
        # 去重并保持顺序
        unique_claims = []
        seen = set()
        
        for claim in all_claims:
            if claim not in seen:
                unique_claims.append(claim)
                seen.add(claim)
        
        # 如果没有特定的差异点，添加通用主张
        if not unique_claims:
            unique_claims.append("[GENERAL] 该想法在多个维度上显示了适度的创新性")
        
        return unique_claims


class FeasibilityCriticAgent(BaseIdeaAgent):
    """第三阶段：可行性批判（并行之二）。

    核心职责:
        - 结合机会图谱的资源节点与关系边，对想法的相关性与可行性做多维评估。

    注意事项:
        - 明确 required_assets 的可得性与替代路径；冲突边应降低置信度但不直接否定。
    """

    def __init__(self, name: str, llm_factory: LLMFactory, db: AcademicPaperDatabase, config: Optional[AgentConfig] = None):
        super().__init__(name, llm_factory, db, config)
        self.feasibility_dimensions = ["relevance", "asset_availability", "complexity", "risk_assessment"]

    async def assess_batch_comprehensive(self, ideas: List[CandidateIdea], graph: SemanticOpportunityGraph) -> List[FeasibilityCritique]:
        """一次LLM调用对整批想法进行综合可行性评估。"""
        print(f"🔬 开始批量综合可行性评估：{len(ideas)}个想法")
        
        # 为整批想法构建综合评估prompt
        ideas_summary = "\n".join([
            f"{i+1}. {idea.title}\n   核心假设: {idea.core_hypothesis}\n   所需资产: {', '.join([asset.get('name', asset.get('type', 'unknown')) for asset in getattr(idea, 'required_assets', [])])}"
            for i, idea in enumerate(ideas)
        ])
        
        # 从图谱中获取可用资源概况
        available_methods = len([n for n, d in graph.nodes(data=True) if d.get('type') == 'Method'])
        available_datasets = len([n for n, d in graph.nodes(data=True) if d.get('type') == 'Dataset'])
        available_metrics = len([n for n, d in graph.nodes(data=True) if d.get('type') == 'Metric'])
        
        prompt = f"""
作为技术可行性专家，请对以下{len(ideas)}个研究想法进行综合可行性评估。

**待评估想法列表**：
{ideas_summary}

**可用资源概况**：
- 方法库: {available_methods}个可用方法
- 数据集: {available_datasets}个可用数据集  
- 评估指标: {available_metrics}个可用指标

**评估维度**：
1. **相关性** (0-3分): 想法与现有技术生态的契合度
2. **资产可获得性** (0-3分): 所需数据集、方法、工具的可获得性
3. **实现复杂度** (0-2分): 技术实现的难度（分数越低越复杂）
4. **风险评估** (0-2分): 技术风险和不确定性（分数越低风险越高）

请以JSON格式返回：
{{
  "batch_analysis": "对整批想法的综合可行性分析",
  "feasibility_assessments": [
    {{
      "idea_index": 1,
      "feasibility_score": X.X,
      "dimension_scores": {{
        "relevance": X.X,
        "asset_availability": X.X,
        "complexity": X.X, 
        "risk_assessment": X.X
      }},
      "reasoning": "详细可行性分析",
      "required_resources": ["资源1", "资源2"],
      "potential_risks": ["风险1", "风险2"],
      "implementation_suggestions": ["建议1", "建议2"]
    }}
  ]
}}
"""
        
        try:
            response_data = await self.llm.generate(
                model_name=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=15000,
                agent_name=self.name,
                task_type="batch_feasibility_assessment"
            )
            response = response_data.get("content", "")
            
            # 解析JSON响应
            import json
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_content = json_match.group(1).strip()
            else:
                json_content = response.strip()
            
            result = json.loads(json_content)
            assessments = result.get("feasibility_assessments", [])
            
            # 为每个想法创建FeasibilityCritique对象
            critiques = []
            for i, idea in enumerate(ideas):
                if i < len(assessments):
                    assessment = assessments[i]
                    dimension_scores = assessment.get("dimension_scores", {})
                    feasibility_score = sum(dimension_scores.values())
                    
                    critique = FeasibilityCritique(
                        idea_id=idea.id,
                        feasibility_score=feasibility_score,
                        potential_risks =assessment.get("potential_risks", []),
                        graph_checks={"type": "batch_comprehensive", "reasoning": assessment.get("reasoning", "")}
                    )
                else:
                    # 默认评估
                    critique = FeasibilityCritique(
                        idea_id=idea.id,
                        feasibility_score=6.0,
                        required_assets={"relevance": 1.5, "asset_availability": 1.5, "complexity": 1.5, "risk_assessment": 1.5},
                        potential_risks=["待进一步评估"],
                        graph_checks={"type": "batch_comprehensive", "error": "批量评估解析失败"}
                    )
                critiques.append(critique)
            
            print(f"✅ 批量综合可行性评估完成")
            return critiques
            
        except Exception as e:
            print(f"❌ 批量综合可行性评估失败: {str(e)}")
            # 返回默认评估
            return [FeasibilityCritique(
                idea_id=idea.id,
                feasibility_score=6.0,
                dimension_scores={"relevance": 1.5, "asset_availability": 1.5, "complexity": 1.5, "risk_assessment": 1.5},
                required_resources=[],
                risks=["评估失败"],
                graph_checks={"type": "batch_comprehensive", "error": str(e)}
            ) for idea in ideas]

    async def assess_feasibility(self, idea: CandidateIdea, graph: SemanticOpportunityGraph) -> FeasibilityCritique:
        """评估单个想法的可行性。"""
        results = await self.assess_feasibility_batch([idea], graph)
        return results[0]
    
    async def assess_feasibility_batch(self, ideas: List[CandidateIdea], graph: SemanticOpportunityGraph) -> List[FeasibilityCritique]:
        """批量评估想法的可行性。"""
        import asyncio
        
        print(f"🔬 开始批量可行性评估：{len(ideas)}个想法")
        
        # 创建并发任务
        tasks = []
        for idea in ideas:
            task = asyncio.create_task(self._assess_single_feasibility(idea, graph))
            tasks.append(task)
        
        # 等待所有任务完成
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        critiques = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"⚠️ 想法{i+1}可行性评估失败: {str(result)}")
                # 创建默认评估结果
                critiques.append(FeasibilityCritique(
                    idea_id=ideas[i].id,
                    feasibility_score=5.0,
                    relevance="评估失败",
                    required_assets=[],
                    potential_risks=[{"risk": "评估失败", "impact": "unknown", "probability": "unknown", "mitigation": "重新评估"}],
                    graph_checks={"error": str(result)}
                ))
            else:
                critiques.append(result)
        
        print(f"✅ 批量可行性评估完成：{len(critiques)}个结果")
        return critiques
    
    async def _assess_single_feasibility(self, idea: CandidateIdea, graph: SemanticOpportunityGraph) -> FeasibilityCritique:
        """基于LLM的智能可行性评审。"""
        
        print(f"    🔬 开始对想法 '{idea.title}' 进行全面可行性分析")
        
        # 收集图谱上下文信息
        graph_context = self._collect_graph_context(graph)
        
        # 收集想法的完整信息
        idea_details = self._prepare_idea_details(idea)
        
        prompt = f"""
作为研究可行性专家，请对以下研究想法进行全面的可行性评估。

**待评估想法**：
{idea_details}

**研究领域背景**：
{graph_context}

请从以下维度进行深入的可行性分析：

### 1. 技术可行性 (Technical Feasibility)
- 所需技术是否成熟可靠？
- 是否存在技术实现的关键挑战？
- 技术路径是否清晰可行？

### 2. 资源可行性 (Resource Feasibility)  
- 所需数据集是否可获得？
- 计算资源需求是否合理？
- 人力和时间成本是否可控？

### 3. 方法可行性 (Methodological Feasibility)
- 研究方法是否科学合理？
- 实验设计是否可执行？
- 评估指标是否恰当？

### 4. 风险可控性 (Risk Manageability)
- 主要风险点是什么？
- 风险是否可以预防或缓解？
- 是否有备选方案？

### 5. 领域相关性 (Domain Relevance)
- 想法是否契合当前研究热点？
- 是否有实际应用价值？
- 学术影响力潜力如何？

请基于你的分析给出评分，并以JSON格式返回详细结果：
{{
  "feasibility_score": X.X,  // 总体可行性评分 (1-10分，10分最可行)
  "dimension_scores": {{
    "technical": X.X,      // 技术可行性评分
    "resource": X.X,       // 资源可行性评分  
    "methodological": X.X, // 方法可行性评分
    "risk": X.X,          // 风险可控性评分
    "relevance": X.X      // 领域相关性评分
  }},
  "detailed_analysis": {{
    "technical_analysis": "技术可行性详细分析",
    "resource_analysis": "资源可行性详细分析",
    "methodological_analysis": "方法可行性详细分析",
    "risk_analysis": "风险评估详细分析",
    "relevance_analysis": "相关性详细分析"
  }},
  "required_assets": [
    {{
      "type": "dataset",
      "name": "所需数据集名称",
      "availability": "public/restricted/need_creation",
      "difficulty": "获取/创建难度评估"
    }},
    {{
      "type": "compute",
      "name": "计算资源需求",
      "specification": "具体规格要求",
      "cost_estimate": "成本估算"
    }}
  ],
  "potential_risks": [
    {{
      "risk": "风险描述",
      "impact": "high/medium/low",
      "probability": "high/medium/low",
      "mitigation": "缓解策略"
    }}
  ],
  "strengths": ["可行性优势1", "可行性优势2"],
  "weaknesses": ["可行性挑战1", "可行性挑战2"],
  "recommendations": [
    "改进建议1：具体建议",
    "改进建议2：具体建议"
  ],
  "timeline_estimate": "预估研究周期",
  "success_probability": "成功概率评估"
}}
"""

        try:
            response_data = await self.llm.generate(
                model_name=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=15000,
                agent_name=self.name,
                task_type="feasibility_assessment"
            )
            response = response_data.get("content", "")
            
            # 解析LLM响应
            import json
            import re
            try:
                if "```json" in response:
                    json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
                    if json_match:
                        json_content = json_match.group(1).strip()
                    else:
                        json_start = response.find("```json") + 7
                        json_content = response[json_start:].strip()
                        if json_content.endswith("```"):
                            json_content = json_content[:-3].strip()
                else:
                    json_content = response.strip()
                
                result = json.loads(json_content)
                
                # 创建FeasibilityCritique对象
                critique = FeasibilityCritique(
                    idea_id=idea.id,
                    feasibility_score=float(result.get("feasibility_score", 6.0)),
                    required_assets=result.get("required_assets", []),
                    potential_risks=result.get("potential_risks", []),
                    graph_checks={
                        "approach": "llm_comprehensive_analysis",
                        "dimensions": list(self.feasibility_dimensions),
                        "detailed_analysis": result.get("detailed_analysis", {}),
                        "dimension_scores": result.get("dimension_scores", {}),
                        "strengths": result.get("strengths", []),
                        "weaknesses": result.get("weaknesses", []),
                        "recommendations": result.get("recommendations", []),
                        "timeline_estimate": result.get("timeline_estimate", "未估算"),
                        "success_probability": result.get("success_probability", "未评估")
                    }
                )
                
                print(f"    ✅ 可行性评分: {critique.feasibility_score:.2f}/10.0")
                return critique
                
            except json.JSONDecodeError:
                print(f"    ⚠️ LLM可行性分析响应解析失败，使用备用评估")
                return self._create_fallback_critique(idea)
                
        except Exception as e:
            print(f"    ❌ LLM可行性分析失败: {str(e)}，使用备用评估")
            return self._create_fallback_critique(idea)
    
    def _prepare_idea_details(self, idea: CandidateIdea) -> str:
        """准备想法的详细信息用于LLM分析。"""
        details = []
        details.append(f"- 标题: {idea.title}")
        details.append(f"- 核心假设: {idea.core_hypothesis}")
        details.append(f"- 创新点: {', '.join(idea.initial_innovation_points)}")
        
        if hasattr(idea, 'expected_contribution') and idea.expected_contribution:
            details.append(f"- 预期贡献: {', '.join(idea.expected_contribution)}")
        
        if hasattr(idea, 'preliminary_experiments') and idea.preliminary_experiments:
            experiments = [exp.get('name', str(exp)) if isinstance(exp, dict) else str(exp) for exp in idea.preliminary_experiments]
            details.append(f"- 初步实验设计: {', '.join(experiments)}")
        
        if hasattr(idea, 'required_assets') and idea.required_assets:
            assets = [asset.get('type', str(asset)) if isinstance(asset, dict) else str(asset) for asset in idea.required_assets]
            details.append(f"- 所需资产: {', '.join(assets)}")
        
        if hasattr(idea, 'risks') and idea.risks:
            details.append(f"- 已识别风险: {', '.join(idea.risks)}")
        
        return "\n".join(details)
    
    def _create_fallback_critique(self, idea: CandidateIdea) -> FeasibilityCritique:
        """创建备用的可行性评审结果。"""
        return FeasibilityCritique(
            idea_id=idea.id,
            feasibility_score=6.5,  # 中等偏上分数
            required_assets=[{"type": "unknown", "availability": "需要进一步评估"}],
            potential_risks=[{"risk": "评估失败，需要人工复核", "impact": "medium"}],
            graph_checks={
                "approach": "fallback_analysis",
                "note": "LLM分析失败，使用简化评估"
            }
        )

    async def _assess_relevance(self, idea: CandidateIdea, graph: SemanticOpportunityGraph) -> float:
        """基于LLM的领域相关性评估。"""
        print(f"    🎯 评估想法 '{idea.title}' 的领域相关性")
        
        # 收集图谱上下文信息
        graph_context = self._collect_graph_context(graph)
        
        prompt = f"""
作为学术专家，请评估以下研究想法与当前研究领域的相关性。

**待评估想法**：
- 标题: {idea.title}
- 核心假设: {idea.core_hypothesis}
- 创新点: {', '.join(idea.initial_innovation_points)}
- 触发节点: {', '.join(idea.source_trigger_nodes) if hasattr(idea, 'source_trigger_nodes') and idea.source_trigger_nodes else '无'}

**当前研究领域背景**：
{graph_context}

请从以下维度评估想法的领域相关性：

1. **研究主题匹配度**: 想法是否符合当前领域的研究热点和趋势？
2. **技术栈契合度**: 所涉及的技术和方法是否与领域主流技术相符？
3. **问题重要性**: 想法解决的问题是否是领域内的重要问题？
4. **学术价值**: 想法是否具有重要的学术意义和影响潜力？

请以JSON格式返回评估结果：
{{
  "relevance_score": 8.2,  // 相关性评分 (1-10分，10分最相关)
  "analysis": {{
    "topic_match": "研究主题匹配度分析",
    "tech_alignment": "技术栈契合度分析", 
    "problem_importance": "问题重要性分析",
    "academic_value": "学术价值分析"
  }},
  "strengths": ["相关性优势1", "相关性优势2"],
  "concerns": ["可能的相关性问题1", "可能的相关性问题2"],
  "recommendations": ["改进建议1", "改进建议2"]
}}
"""

        try:
            response_data = await self.llm.generate(
                model_name=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=15000,
                agent_name=self.name,
                task_type="feasibility_assessment"
            )
            response = response_data.get("content", "")
            
            # 解析LLM响应
            import json
            import re
            try:
                if "```json" in response:
                    json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
                    if json_match:
                        json_content = json_match.group(1).strip()
                    else:
                        json_start = response.find("```json") + 7
                        json_content = response[json_start:].strip()
                        if json_content.endswith("```"):
                            json_content = json_content[:-3].strip()
                else:
                    json_content = response.strip()
                
                result = json.loads(json_content)
                final_score = float(result.get("relevance_score", 6.0))
                
                print(f"    ✅ 相关性评分: {final_score:.2f}/10.0")
                return final_score
                
            except json.JSONDecodeError:
                print(f"    ⚠️ LLM相关性评估响应解析失败，使用默认评分")
                return 6.0
                
        except Exception as e:
            print(f"    ❌ LLM相关性评估失败: {str(e)}，使用默认评分")
            return 6.0
    
    def _collect_graph_context(self, graph: SemanticOpportunityGraph) -> str:
        """收集图谱的领域背景信息。"""
        context_lines = []
        
        # 统计节点类型分布
        methods = find_nodes_by_type(graph, GraphNodeType.METHOD)
        tasks = find_nodes_by_type(graph, GraphNodeType.TASK)
        datasets = find_nodes_by_type(graph, GraphNodeType.DATASET)
        metrics = find_nodes_by_type(graph, GraphNodeType.METRIC)
        
        context_lines.append(f"研究领域统计: {len(methods)}个方法, {len(tasks)}个任务, {len(datasets)}个数据集, {len(metrics)}个指标")
        
        # 主要方法
        if methods:
            top_methods = []
            for method_id in methods[:5]:
                if method_id in graph.nodes():
                    method_name = graph.nodes[method_id].get('name', method_id)
                    salience = graph.nodes[method_id].get('salience', 0.0)
                    top_methods.append(f"{method_name}(重要性:{salience:.2f})")
            context_lines.append(f"主要方法: {', '.join(top_methods)}")
        
        # 主要任务
        if tasks:
            top_tasks = []
            for task_id in tasks[:5]:
                if task_id in graph.nodes():
                    task_name = graph.nodes[task_id].get('name', task_id)
                    salience = graph.nodes[task_id].get('salience', 0.0)
                    top_tasks.append(f"{task_name}(重要性:{salience:.2f})")
            context_lines.append(f"主要任务: {', '.join(top_tasks)}")
        
        return "\n".join(context_lines) if context_lines else "无法获取图谱背景信息"
    
    async def _evaluate_node_relevance(self, idea: CandidateIdea, graph: SemanticOpportunityGraph) -> float:
        """评估触发节点的图谱相关性。"""
        
        if not hasattr(idea, 'source_trigger_nodes') or not idea.source_trigger_nodes:
            return 5.0  # 默认中等相关性
        
        node_scores = []
        
        for node_id in idea.source_trigger_nodes:
            if node_id in graph.nodes():
                # 节点存在，评估其重要性
                node_data = graph.nodes[node_id]
                salience = node_data.get('salience', 0.5)
                node_type = node_data.get('type', 'Unknown')
                
                # 根据节点类型调整权重
                type_weights = {
                    'Method': 1.0,
                    'Task': 0.9,
                    'Dataset': 0.8,
                    'Metric': 0.7,
                    'Paper': 0.6,
                    'Domain': 0.8
                }
                
                type_weight = type_weights.get(node_type, 0.5)
                node_score = salience * type_weight * 10.0
                node_scores.append(node_score)
            else:
                # 节点不存在，降低相关性
                node_scores.append(2.0)
        
        if node_scores:
            return sum(node_scores) / len(node_scores)
        else:
            return 5.0
    
    async def _evaluate_domain_relevance(self, idea: CandidateIdea, graph: SemanticOpportunityGraph) -> float:
        """评估领域匹配度。"""
        
        # 从图谱中提取主导领域
        domain_nodes = find_nodes_by_type(graph, GraphNodeType.DOMAIN)
        
        if not domain_nodes:
            return 7.0  # 如果没有明确领域节点，给予较高相关性
        
        # 从想法中提取领域信息
        idea_text = idea.title + " " + idea.core_hypothesis
        idea_domains = self._extract_domains_from_text(idea_text)
        
        # 计算领域重叠度
        graph_domains = []
        for domain_id in domain_nodes:
            domain_name = graph.nodes[domain_id].get('name', domain_id)
            graph_domains.append(domain_name.lower())
        
        if not idea_domains or not graph_domains:
            return 6.0
        
        # 简单的关键词匹配
        overlap_count = 0
        for idea_domain in idea_domains:
            for graph_domain in graph_domains:
                if idea_domain in graph_domain or graph_domain in idea_domain:
                    overlap_count += 1
                    break
        
        # 计算匹配度分数
        if len(idea_domains) > 0:
            match_ratio = overlap_count / len(idea_domains)
            return 4.0 + match_ratio * 6.0  # 4-10分范围
        else:
            return 6.0
    
    def _extract_domains_from_text(self, text: str) -> List[str]:
        """从文本中提取领域关键词。"""
        import re
        
        domain_patterns = [
            r'\b(?:computer vision|natural language|speech|robotics|healthcare|finance|security)\b',
            r'\b(?:machine learning|deep learning|reinforcement learning|transfer learning)\b',
            r'\b(?:neural networks|optimization|data mining|information retrieval)\b',
            r'\b(?:classification|detection|generation|translation|recommendation|prediction)\b'
        ]
        
        domains = []
        text_lower = text.lower()
        
        for pattern in domain_patterns:
            matches = re.findall(pattern, text_lower)
            domains.extend(matches)
        
        return list(set(domains))
    
    async def _evaluate_academic_relevance(self, idea: CandidateIdea) -> float:
        """评估学术价值和研究意义。"""
        
        # 简化实现：基于想法结构的完整性评估
        score = 5.0  # 基础分
        
        # 因子1：创新点数量和质量
        innovation_count = len(idea.initial_innovation_points)
        if innovation_count >= 3:
            score += 1.5
        elif innovation_count >= 2:
            score += 1.0
        elif innovation_count >= 1:
            score += 0.5
        
        # 因子2：预期贡献的多样性
        contribution_count = len(idea.expected_contribution)
        if contribution_count >= 2:
            score += 1.0
        elif contribution_count >= 1:
            score += 0.5
        
        # 因子3：实验设计的完整性
        experiment_count = len(idea.preliminary_experiments)
        if experiment_count >= 2:
            score += 1.0
        elif experiment_count >= 1:
            score += 0.5
        
        # 因子4：核心假设的清晰度
        hypothesis_length = len(idea.core_hypothesis.strip())
        if hypothesis_length >= 100:
            score += 1.0
        elif hypothesis_length >= 50:
            score += 0.5
        
        return min(10.0, score)

    async def _analyze_required_assets(self, idea: CandidateIdea, graph: SemanticOpportunityGraph) -> Dict[str, Any]:
        """分析所需资产的可得性。
        
        输入:
            - idea: 候选想法。
            - graph: 语义机会图谱。
            
        输出:
            - Dict: 资产分析结果，包含可得资产列表、缺失资产、替代方案。
            
        实现思路:
            1) 从idea.required_assets和图谱中识别所需的数据集、算力、代码等资源。
            2) 检查资源在图谱中的可得性标记，查询外部数据库确认公开性。
            3) 为缺失资源提出替代方案：合成数据、代理任务、开源实现等。
        """
        print(f"    📋 分析所需资产的可得性")
        
        # 收集所有所需资产
        all_required_assets = []
        
        # 来源1：想法中明确列出的资产
        if hasattr(idea, 'required_assets') and idea.required_assets:
            all_required_assets.extend(idea.required_assets)
        
        # 来源2：从想法内容推断的隐含资产
        inferred_assets = await self._infer_assets_from_idea(idea)
        all_required_assets.extend(inferred_assets)
        
        # 分析每个资产的可得性
        available_assets = []
        missing_assets = []
        
        for asset in all_required_assets:
            availability_analysis = await self._analyze_single_asset(asset, graph)
            
            if availability_analysis['available']:
                available_assets.append(availability_analysis)
            else:
                missing_assets.append(availability_analysis)
        
        # 为缺失资产生成替代方案
        alternatives = await self._generate_asset_alternatives(missing_assets)
        
        # 计算整体可得性分数
        overall_availability = await self._calculate_asset_availability_score(available_assets, missing_assets, alternatives)
        
        print(f"    ✅ 资产分析完成: {len(available_assets)} 可得, {len(missing_assets)} 缺失")
        
        return {
            "available_assets": available_assets,
            "missing_assets": missing_assets,
            "alternatives": alternatives,
            "overall_availability_score": overall_availability,
            "total_assets_analyzed": len(all_required_assets)
        }
    
    async def _infer_assets_from_idea(self, idea: CandidateIdea) -> List[Dict[str, Any]]:
        """从想法内容推断隐含的所需资产。"""
        
        inferred_assets = []
        
        # 从实验设计中推断数据集需求
        for exp in idea.preliminary_experiments:
            dataset = exp.get('dataset')
            if dataset and dataset != 'to_be_determined':
                inferred_assets.append({
                    'type': 'Dataset',
                    'id': dataset,
                    'availability': 'to_be_verified',
                    'source': 'experiment_design'
                })
        
        # 从创新点中推断方法和技术需求
        for point in idea.initial_innovation_points:
            # 简化推断：寻找常见的技术术语
            tech_terms = self._extract_technology_requirements(point)
            for term in tech_terms:
                inferred_assets.append({
                    'type': 'Technology',
                    'id': term,
                    'availability': 'to_be_verified',
                    'source': 'innovation_points'
                })
        
        # 基于想法复杂度推断计算资源需求
        complexity_score = self._estimate_computational_complexity(idea)
        if complexity_score > 7:
            inferred_assets.append({
                'type': 'ComputeResource',
                'id': 'high_performance_gpu',
                'availability': 'institutional_dependent',
                'source': 'complexity_analysis'
            })
        elif complexity_score > 4:
            inferred_assets.append({
                'type': 'ComputeResource',
                'id': 'standard_gpu',
                'availability': 'widely_available',
                'source': 'complexity_analysis'
            })
        
        return inferred_assets
    
    def _extract_technology_requirements(self, text: str) -> List[str]:
        """从文本中提取技术需求。"""
        import re
        
        tech_patterns = [
            r'\b(?:transformer|attention|lstm|cnn|bert|gpt|resnet|vgg|alexnet)\b',
            r'\b(?:pytorch|tensorflow|keras|huggingface|openai|anthropic)\b',
            r'\b(?:cuda|gpu|distributed|parallel|cloud)\b'
        ]
        
        technologies = []
        text_lower = text.lower()
        
        for pattern in tech_patterns:
            matches = re.findall(pattern, text_lower)
            technologies.extend(matches)
        
        return list(set(technologies))
    
    def _estimate_computational_complexity(self, idea: CandidateIdea) -> float:
        """估算想法的计算复杂度。"""
        
        complexity_score = 3.0  # 基础分
        
        # 因子1：方法复杂度
        complex_methods = ['transformer', 'attention', 'neural', 'deep', 'reinforcement', 'meta']
        idea_text = (idea.title + " " + idea.core_hypothesis + " " + 
                    " ".join(idea.initial_innovation_points)).lower()
        
        for method in complex_methods:
            if method in idea_text:
                complexity_score += 1.0
        
        # 因子2：数据规模
        large_data_indicators = ['large-scale', 'massive', 'big data', 'billion', 'million']
        for indicator in large_data_indicators:
            if indicator in idea_text:
                complexity_score += 1.5
        
        # 因子3：多模态或多任务
        multi_indicators = ['multi-modal', 'multi-task', 'cross-domain', 'transfer']
        for indicator in multi_indicators:
            if indicator in idea_text:
                complexity_score += 1.0
        
        return min(10.0, complexity_score)
    
    async def _analyze_single_asset(self, asset: Dict[str, Any], graph: SemanticOpportunityGraph) -> Dict[str, Any]:
        """分析单个资产的可得性。"""
        
        asset_type = asset.get('type', 'Unknown')
        asset_id = asset.get('id', 'unknown')
        stated_availability = asset.get('availability', 'unknown')
        
        analysis = {
            'type': asset_type,
            'id': asset_id,
            'stated_availability': stated_availability,
            'available': False,
            'confidence': 0.5,
            'notes': []
        }
        
        # 检查图谱中的资产信息
        graph_availability = await self._check_asset_in_graph(asset_id, graph)
        if graph_availability:
            analysis['available'] = True
            analysis['confidence'] = 0.8
            analysis['notes'].append(f"在图谱中找到相关节点")
        
        # 基于声明的可得性进行评估
        availability_scores = {
            'public': (True, 0.9),
            'open_source': (True, 0.9),
            'widely_available': (True, 0.8),
            'institutional_available': (True, 0.7),
            'commercial': (True, 0.6),
            'restricted': (False, 0.3),
            'proprietary': (False, 0.2),
            'unavailable': (False, 0.1),
            'to_be_determined': (False, 0.4),
            'unknown': (False, 0.5)
        }
        
        if stated_availability in availability_scores:
            available, confidence = availability_scores[stated_availability]
            analysis['available'] = available
            analysis['confidence'] = max(analysis['confidence'], confidence)
        
        # 基于资产类型的默认评估
        type_defaults = {
            'Dataset': (True, 0.7),  # 大多数数据集相对可得
            'Method': (True, 0.8),   # 方法通常有开源实现
            'Technology': (True, 0.6), # 技术可得性变化较大
            'ComputeResource': (True, 0.5), # 计算资源依赖机构
            'Tool': (True, 0.8),     # 工具通常开源
            'Library': (True, 0.9)   # 库文件高度可得
        }
        
        if asset_type in type_defaults and analysis['confidence'] < 0.6:
            default_available, default_confidence = type_defaults[asset_type]
            if not analysis['available']:  # 只在当前评估为不可得时应用默认值
                analysis['available'] = default_available
                analysis['confidence'] = default_confidence
        
        return analysis
    
    async def _check_asset_in_graph(self, asset_id: str, graph: SemanticOpportunityGraph) -> bool:
        """检查资产是否在图谱中存在。"""
        
        # 直接ID匹配
        if asset_id in graph.nodes():
            return True
        
        # 名称匹配
        for node_id, node_data in graph.nodes(data=True):
            node_name = node_data.get('name', '').lower()
            if asset_id.lower() in node_name or node_name in asset_id.lower():
                return True
        
        # 别名匹配
        for node_id, node_data in graph.nodes(data=True):
            aliases = node_data.get('aliases', [])
            for alias in aliases:
                if asset_id.lower() in alias.lower() or alias.lower() in asset_id.lower():
                    return True
        
        return False
    
    async def _generate_asset_alternatives(self, missing_assets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """为缺失资产生成替代方案。"""
        
        alternatives = []
        
        for asset in missing_assets:
            asset_type = asset['type']
            asset_id = asset['id']
            
            if asset_type == 'Dataset':
                # 数据集替代方案
                alternatives.extend([
                    {
                        'for_asset': asset_id,
                        'alternative_type': 'synthetic_dataset',
                        'description': f'为{asset_id}生成合成数据集',
                        'feasibility': 0.7,
                        'effort': 'medium'
                    },
                    {
                        'for_asset': asset_id,
                        'alternative_type': 'similar_dataset',
                        'description': f'使用与{asset_id}相似的公开数据集',
                        'feasibility': 0.8,
                        'effort': 'low'
                    }
                ])
            
            elif asset_type == 'Method':
                # 方法替代方案
                alternatives.append({
                    'for_asset': asset_id,
                    'alternative_type': 'open_source_implementation',
                    'description': f'寻找{asset_id}的开源实现',
                    'feasibility': 0.8,
                    'effort': 'low'
                })
            
            elif asset_type == 'ComputeResource':
                # 计算资源替代方案
                alternatives.extend([
                    {
                        'for_asset': asset_id,
                        'alternative_type': 'cloud_platform',
                        'description': f'使用云计算平台提供{asset_id}',
                        'feasibility': 0.9,
                        'effort': 'low'
                    },
                    {
                        'for_asset': asset_id,
                        'alternative_type': 'model_optimization',
                        'description': f'通过模型优化减少对{asset_id}的需求',
                        'feasibility': 0.6,
                        'effort': 'high'
                    }
                ])
            
            else:
                # 通用替代方案
                alternatives.append({
                    'for_asset': asset_id,
                    'alternative_type': 'equivalent_alternative',
                    'description': f'寻找{asset_id}的等价替代品',
                    'feasibility': 0.6,
                    'effort': 'medium'
                })
        
        return alternatives
    
    async def _calculate_asset_availability_score(self, available_assets: List[Dict[str, Any]], 
                                                missing_assets: List[Dict[str, Any]], 
                                                alternatives: List[Dict[str, Any]]) -> float:
        """计算整体资产可得性分数。"""
        
        total_assets = len(available_assets) + len(missing_assets)
        
        if total_assets == 0:
            return 8.0  # 如果没有明确资产需求，给予较高分数
        
        # 基础分数：直接可得的资产
        available_score = 0.0
        for asset in available_assets:
            confidence = asset.get('confidence', 0.5)
            available_score += confidence
        
        # 替代方案的分数
        alternative_score = 0.0
        for alternative in alternatives:
            feasibility = alternative.get('feasibility', 0.5)
            effort_multiplier = {'low': 1.0, 'medium': 0.8, 'high': 0.6}.get(alternative.get('effort', 'medium'), 0.7)
            alternative_score += feasibility * effort_multiplier
        
        # 综合计算
        total_effective_score = available_score + alternative_score * 0.7  # 替代方案权重降低
        max_possible_score = total_assets * 1.0
        
        if max_possible_score > 0:
            normalized_score = (total_effective_score / max_possible_score) * 10.0
        else:
            normalized_score = 8.0
        
        return max(0.0, min(10.0, normalized_score))

    async def _assess_risks_and_complexity(self, idea: CandidateIdea, graph: SemanticOpportunityGraph) -> Dict[str, Any]:
        """评估实现风险与复杂度。
        
        输入:
            - idea: 候选想法。
            - graph: 语义机会图谱。
            
        输出:
            - Dict: 风险分析结果，包含风险类型、严重程度、缓解策略。
            
        实现思路:
            1) 技术风险：方法复杂度、实现难度、调参敏感性。
            2) 数据风险：数据质量、分布偏移、标注成本。
            3) 资源风险：计算开销、时间预算、人力需求。
            4) 结果风险：可重现性、泛化能力、评估公平性。
        """
        print(f"    ⚠️ 评估实现风险与复杂度")
        
        # 简化实现：基于想法内容的风险评估
        overall_risk_score = await self._calculate_simplified_risk_score(idea)
        
        # 识别主要风险类型
        risk_types = await self._identify_main_risk_types(idea)
        
        # 生成风险缓解建议
        mitigation_strategies = await self._generate_basic_mitigation_strategies(risk_types)
        
        risk_analysis = {
            "overall_risk_score": overall_risk_score,
            "identified_risks": risk_types,
            "mitigation_strategies": mitigation_strategies,
            "risk_level": self._risk_level_from_score(overall_risk_score)
        }
        
        print(f"    ✅ 风险评估完成，整体风险等级: {risk_analysis['risk_level']}")
        
        return risk_analysis
    
    async def _calculate_simplified_risk_score(self, idea: CandidateIdea) -> float:
        """计算简化的风险分数。"""
        
        risk_score = 3.0  # 基础风险分数
        
        # 因子1：计算复杂度
        complexity = self._estimate_computational_complexity(idea)
        if complexity > 8:
            risk_score += 2.0
        elif complexity > 6:
            risk_score += 1.0
        
        # 因子2：创新程度（创新越高，风险越大）
        innovation_count = len(idea.initial_innovation_points)
        if innovation_count > 3:
            risk_score += 1.5
        elif innovation_count > 2:
            risk_score += 1.0
        
        # 因子3：实验复杂度
        experiment_count = len(idea.preliminary_experiments)
        if experiment_count > 3:
            risk_score += 1.0
        
        # 因子4：已知风险
        existing_risks = len(idea.risks) if hasattr(idea, 'risks') and idea.risks else 0
        risk_score += existing_risks * 0.5
        
        return max(0.0, min(10.0, risk_score))
    
    async def _identify_main_risk_types(self, idea: CandidateIdea) -> List[Dict[str, Any]]:
        """识别主要风险类型。"""
        
        risks = []
        idea_text = (idea.title + " " + idea.core_hypothesis + " " + 
                    " ".join(idea.initial_innovation_points)).lower()
        
        # 技术风险
        if any(term in idea_text for term in ['complex', 'novel', 'advanced', 'sophisticated']):
            risks.append({
                "type": "technical_complexity",
                "severity": "medium",
                "description": "技术实现复杂度较高"
            })
        
        # 数据风险
        if any(term in idea_text for term in ['data', 'dataset', 'training']):
            risks.append({
                "type": "data_dependency",
                "severity": "medium", 
                "description": "对数据质量和可得性有较高依赖"
            })
        
        # 资源风险
        complexity = self._estimate_computational_complexity(idea)
        if complexity > 6:
            risks.append({
                "type": "resource_intensive",
                "severity": "medium",
                "description": "计算资源需求较高"
            })
        
        # 结果风险
        if len(idea.preliminary_experiments) < 2:
            risks.append({
                "type": "limited_validation",
                "severity": "low",
                "description": "验证实验相对有限"
            })
        
        return risks
    
    async def _generate_basic_mitigation_strategies(self, risk_types: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """生成基础的风险缓解策略。"""
        
        strategies = []
        
        for risk in risk_types:
            risk_type = risk["type"]
            
            if risk_type == "technical_complexity":
                strategies.append({
                    "target_risk": risk_type,
                    "strategy": "分阶段实现",
                    "description": "将复杂技术分解为可管理的子模块",
                    "priority": "high"
                })
            
            elif risk_type == "data_dependency":
                strategies.append({
                    "target_risk": risk_type,
                    "strategy": "数据备选方案",
                    "description": "准备多个数据源和合成数据方案",
                    "priority": "medium"
                })
            
            elif risk_type == "resource_intensive":
                strategies.append({
                    "target_risk": risk_type,
                    "strategy": "资源优化",
                    "description": "采用模型压缩和云计算资源",
                    "priority": "medium"
                })
            
            elif risk_type == "limited_validation":
                strategies.append({
                    "target_risk": risk_type,
                    "strategy": "扩展验证",
                    "description": "增加更多评估维度和测试场景",
                    "priority": "low"
                })
        
        return strategies
    
    def _risk_level_from_score(self, score: float) -> str:
        """根据分数确定风险等级。"""
        if score >= 7:
            return "高风险"
        elif score >= 4:
            return "中等风险"
        else:
            return "低风险"

    async def _verify_graph_consistency(self, idea: CandidateIdea, graph: SemanticOpportunityGraph) -> Dict[str, Any]:
        """检查想法与图谱的一致性。
        
        输入:
            - idea: 候选想法。
            - graph: 语义机会图谱。
            
        输出:
            - Dict: 图谱检查结果，包含验证边数、冲突检测、一致性评分。
            
        实现思路:
            1) 检查想法假设是否与图谱中的冲突边（contradicts/critiques）一致。
            2) 验证想法涉及的方法-任务-数据-指标组合在图谱中的支持度。
            3) 利用NetworkX的图分析算法检测潜在的逻辑不一致。
        """
        print(f"    🔍 检查想法与图谱的一致性")
        
        # 检查触发节点的存在性和连接性
        node_verification = await self._verify_trigger_nodes(idea, graph)
        
        # 检查冲突边
        conflict_analysis = await self._detect_graph_conflicts(idea, graph)
        
        # 验证方法-任务组合的支持度
        combination_support = await self._verify_method_task_combinations(idea, graph)
        
        # 计算整体一致性分数
        consistency_score = await self._calculate_consistency_score(node_verification, conflict_analysis, combination_support)
        
        graph_checks = {
            "node_verification": node_verification,
            "conflict_analysis": conflict_analysis,
            "combination_support": combination_support,
            "consistency_score": consistency_score,
            "is_consistent": consistency_score >= 6.0
        }
        
        print(f"    ✅ 图谱一致性检查完成，一致性分数: {consistency_score:.2f}/10.0")
        
        return graph_checks
    
    async def _verify_trigger_nodes(self, idea: CandidateIdea, graph: SemanticOpportunityGraph) -> Dict[str, Any]:
        """验证触发节点在图谱中的存在性。"""
        
        if not hasattr(idea, 'source_trigger_nodes') or not idea.source_trigger_nodes:
            return {
                "nodes_checked": 0,
                "nodes_found": 0,
                "missing_nodes": [],
                "verification_score": 7.0  # 如果没有明确节点，给予中等分数
            }
        
        nodes_found = 0
        missing_nodes = []
        
        for node_id in idea.source_trigger_nodes:
            if node_id in graph.nodes():
                nodes_found += 1
            else:
                missing_nodes.append(node_id)
        
        total_nodes = len(idea.source_trigger_nodes)
        verification_score = (nodes_found / total_nodes) * 10.0 if total_nodes > 0 else 7.0
        
        return {
            "nodes_checked": total_nodes,
            "nodes_found": nodes_found,
            "missing_nodes": missing_nodes,
            "verification_score": verification_score
        }
    
    async def _detect_graph_conflicts(self, idea: CandidateIdea, graph: SemanticOpportunityGraph) -> Dict[str, Any]:
        """检测与图谱中冲突边的矛盾。"""
        
        conflicts = []
        conflict_score = 0.0
        
        # 检查图谱中的所有冲突边
        for src, dst, edge_data in graph.edges(data=True):
            relation = edge_data.get('relation', '')
            
            if relation in ['contradicts', 'critiques', 'conflicts_with']:
                # 检查想法是否涉及这些冲突的概念
                src_name = graph.nodes[src].get('name', src)
                dst_name = graph.nodes[dst].get('name', dst)
                
                idea_text = (idea.title + " " + idea.core_hypothesis).lower()
                
                if (src_name.lower() in idea_text and dst_name.lower() in idea_text):
                    conflicts.append({
                        "source": src_name,
                        "target": dst_name,
                        "relation": relation,
                        "confidence": edge_data.get('confidence', 0.5)
                    })
                    conflict_score += edge_data.get('confidence', 0.5)
        
        # 计算冲突影响分数（冲突越多，分数越低）
        if len(conflicts) == 0:
            final_score = 9.0
        elif conflict_score < 0.5:
            final_score = 7.0
        elif conflict_score < 1.0:
            final_score = 5.0
        else:
            final_score = 3.0
        
        return {
            "conflicts_detected": conflicts,
            "conflict_count": len(conflicts),
            "total_conflict_confidence": conflict_score,
            "conflict_score": final_score
        }
    
    async def _verify_method_task_combinations(self, idea: CandidateIdea, graph: SemanticOpportunityGraph) -> Dict[str, Any]:
        """验证方法-任务组合在图谱中的支持度。"""
        
        # 从想法中提取方法和任务信息
        extracted_methods = self._extract_methods_from_idea(idea)
        extracted_tasks = self._extract_tasks_from_idea(idea)
        
        if not extracted_methods or not extracted_tasks:
            return {
                "combinations_checked": 0,
                "supported_combinations": 0,
                "support_score": 7.0
            }
        
        supported_combinations = 0
        total_combinations = 0
        
        # 检查每个方法-任务组合在图谱中的支持度
        for method in extracted_methods:
            for task in extracted_tasks:
                total_combinations += 1
                
                # 在图谱中寻找支持这个组合的证据
                if await self._check_method_task_support(method, task, graph):
                    supported_combinations += 1
        
        support_ratio = supported_combinations / total_combinations if total_combinations > 0 else 0.7
        support_score = support_ratio * 10.0
        
        return {
            "combinations_checked": total_combinations,
            "supported_combinations": supported_combinations,
            "support_ratio": support_ratio,
            "support_score": support_score,
            "extracted_methods": extracted_methods,
            "extracted_tasks": extracted_tasks
        }
    
    def _extract_methods_from_idea(self, idea: CandidateIdea) -> List[str]:
        """从想法中提取方法名称。"""
        
        methods = []
        idea_text = idea.title + " " + idea.core_hypothesis + " " + " ".join(idea.initial_innovation_points)
        
        # 常见方法关键词
        method_keywords = [
            'transformer', 'attention', 'lstm', 'cnn', 'bert', 'gpt',
            'neural network', 'deep learning', 'machine learning',
            'reinforcement learning', 'transfer learning', 'meta learning'
        ]
        
        for keyword in method_keywords:
            if keyword.lower() in idea_text.lower():
                methods.append(keyword)
        
        return list(set(methods))
    
    def _extract_tasks_from_idea(self, idea: CandidateIdea) -> List[str]:
        """从想法中提取任务名称。"""
        
        tasks = []
        idea_text = idea.title + " " + idea.core_hypothesis + " " + " ".join(idea.initial_innovation_points)
        
        # 常见任务关键词
        task_keywords = [
            'classification', 'detection', 'generation', 'translation',
            'prediction', 'recommendation', 'optimization', 'clustering',
            'regression', 'segmentation', 'recognition', 'synthesis'
        ]
        
        for keyword in task_keywords:
            if keyword.lower() in idea_text.lower():
                tasks.append(keyword)
        
        return list(set(tasks))
    
    async def _check_method_task_support(self, method: str, task: str, graph: SemanticOpportunityGraph) -> bool:
        """检查图谱中是否支持特定的方法-任务组合。"""
        
        # 寻找方法节点
        method_nodes = []
        for node_id, node_data in graph.nodes(data=True):
            node_name = node_data.get('name', '').lower()
            if method.lower() in node_name or node_name in method.lower():
                if node_data.get('type') == 'Method':
                    method_nodes.append(node_id)
        
        # 寻找任务节点
        task_nodes = []
        for node_id, node_data in graph.nodes(data=True):
            node_name = node_data.get('name', '').lower()
            if task.lower() in node_name or node_name in task.lower():
                if node_data.get('type') == 'Task':
                    task_nodes.append(node_id)
        
        # 检查是否存在连接
        for method_node in method_nodes:
            for task_node in task_nodes:
                if graph.has_edge(method_node, task_node) or graph.has_edge(task_node, method_node):
                    return True
        
        # 如果没有直接连接，检查是否有间接支持
        # 简化实现：如果两个节点都存在，认为有一定支持
        return len(method_nodes) > 0 and len(task_nodes) > 0
    
    async def _calculate_consistency_score(self, node_verification: Dict[str, Any], 
                                         conflict_analysis: Dict[str, Any], 
                                         combination_support: Dict[str, Any]) -> float:
        """计算整体一致性分数。"""
        
        # 权重分配
        node_weight = 0.4
        conflict_weight = 0.3
        support_weight = 0.3
        
        node_score = node_verification.get("verification_score", 7.0)
        conflict_score = conflict_analysis.get("conflict_score", 7.0)
        support_score = combination_support.get("support_score", 7.0)
        
        total_score = (node_score * node_weight + 
                      conflict_score * conflict_weight + 
                      support_score * support_weight)
        
        return max(0.0, min(10.0, total_score))

    async def _synthesize_feasibility_critique(self, idea: CandidateIdea, relevance_score: float, 
                                             asset_analysis: Dict[str, Any], risk_analysis: Dict[str, Any], 
                                             graph_checks: Dict[str, Any]) -> FeasibilityCritique:
        """综合可行性评审结果。
        
        输入:
            - idea: 候选想法。
            - relevance_score: 相关性得分。
            - asset_analysis: 资产分析结果。
            - risk_analysis: 风险分析结果。
            - graph_checks: 图谱检查结果。
            
        输出:
            - FeasibilityCritique: 完整的可行性评审结果。
            
        实现思路:
            1) 根据各维度得分计算综合可行性分数。
            2) 整理所需资产清单与潜在风险列表。
            3) 生成可行性评估总结与改进建议。
        """
        print(f"    🎯 综合可行性评审结果")
        
        # 步骤1：计算综合可行性分数
        feasibility_score = await self._calculate_weighted_feasibility_score(
            relevance_score, asset_analysis, risk_analysis, graph_checks
        )
        
        # 步骤2：整理详细的可行性分析
        dimension_scores = {
            "relevance": relevance_score,
            "asset_availability": asset_analysis.get("overall_availability_score", 7.0),
            "risk_assessment": 10.0 - risk_analysis.get("overall_risk_score", 3.0),  # 风险越低可行性越高
            "graph_consistency": graph_checks.get("consistency_score", 7.0)
        }
        
        # 步骤3：整理所需资产和潜在风险
        required_assets = await self._compile_required_assets(asset_analysis)
        potential_risks = await self._compile_potential_risks(risk_analysis)
        
        # 步骤4：生成改进建议
        improvement_suggestions = await self._generate_improvement_suggestions(
            dimension_scores, asset_analysis, risk_analysis, graph_checks
        )
        
        # 创建FeasibilityCritique对象
        critique = FeasibilityCritique(
            idea_id=idea.id,
            feasibility_score=feasibility_score,
            required_assets=required_assets,
            potential_risks=potential_risks,
            graph_checks={
                "approach": "multi_dimensional_analysis",
                "dimensions": list(self.feasibility_dimensions),
                "weights": self._get_feasibility_weights(),
                "timestamp": datetime.now().isoformat()
            }
        )
        
        print(f"    ✅ 可行性评分: {feasibility_score:.2f}/10.0")
        print(f"    📋 识别 {len(required_assets)} 项资产需求")
        print(f"    ⚠️ 发现 {len(potential_risks)} 项潜在风险")
        
        return critique
    
    async def _calculate_weighted_feasibility_score(self, relevance_score: float, 
                                                  asset_analysis: Dict[str, Any], 
                                                  risk_analysis: Dict[str, Any], 
                                                  graph_checks: Dict[str, Any]) -> float:
        """计算加权可行性分数。"""
        
        # 维度权重配置
        weights = self._get_feasibility_weights()
        
        # 收集各维度分数
        scores = {
            "relevance": relevance_score,
            "asset_availability": asset_analysis.get("overall_availability_score", 7.0),
            "risk_assessment": 10.0 - risk_analysis.get("overall_risk_score", 3.0),  # 风险分转可行性分
            "graph_consistency": graph_checks.get("consistency_score", 7.0)
        }
        
        # 加权计算
        total_score = 0.0
        total_weight = 0.0
        
        for dimension, weight in weights.items():
            if dimension in scores:
                score = scores[dimension]
                total_score += score * weight
                total_weight += weight
        
        # 归一化
        if total_weight > 0:
            final_score = total_score / total_weight
        else:
            final_score = 6.0  # 默认中等可行性
        
        return max(0.0, min(10.0, final_score))
    
    def _get_feasibility_weights(self) -> Dict[str, float]:
        """获取可行性维度权重配置。"""
        return {
            "relevance": 0.25,          # 相关性
            "asset_availability": 0.35, # 资产可得性最重要
            "risk_assessment": 0.25,    # 风险评估
            "graph_consistency": 0.15   # 图谱一致性
        }
    
    async def _compile_required_assets(self, asset_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """整理所需资产清单。"""
        
        required_assets = []
        
        # 可得资产
        available_assets = asset_analysis.get("available_assets", [])
        for asset in available_assets:
            required_assets.append({
                "type": asset.get("type", "Unknown"),
                "id": asset.get("id", "unknown"),
                "status": "available",
                "confidence": asset.get("confidence", 0.8),
                "notes": asset.get("notes", [])
            })
        
        # 缺失资产
        missing_assets = asset_analysis.get("missing_assets", [])
        for asset in missing_assets:
            required_assets.append({
                "type": asset.get("type", "Unknown"),
                "id": asset.get("id", "unknown"),
                "status": "missing",
                "confidence": asset.get("confidence", 0.3),
                "notes": ["需要寻找替代方案"]
            })
        
        # 添加替代方案信息
        alternatives = asset_analysis.get("alternatives", [])
        alternative_map = {}
        for alt in alternatives:
            asset_id = alt.get("for_asset", "unknown")
            if asset_id not in alternative_map:
                alternative_map[asset_id] = []
            alternative_map[asset_id].append({
                "type": alt.get("alternative_type", "unknown"),
                "description": alt.get("description", ""),
                "feasibility": alt.get("feasibility", 0.5)
            })
        
        # 为缺失资产添加替代方案
        for asset in required_assets:
            if asset["status"] == "missing" and asset["id"] in alternative_map:
                asset["alternatives"] = alternative_map[asset["id"]]
        
        return required_assets
    
    async def _compile_potential_risks(self, risk_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """整理潜在风险清单。"""
        
        potential_risks = []
        
        # 从风险分析中提取风险
        identified_risks = risk_analysis.get("identified_risks", [])
        for risk in identified_risks:
            potential_risks.append({
                "type": risk.get("type", "unknown"),
                "severity": risk.get("severity", "medium"),
                "description": risk.get("description", ""),
                "likelihood": "medium"  # 简化设置
            })
        
        # 添加缓解策略
        mitigation_strategies = risk_analysis.get("mitigation_strategies", [])
        strategy_map = {}
        for strategy in mitigation_strategies:
            target_risk = strategy.get("target_risk", "unknown")
            if target_risk not in strategy_map:
                strategy_map[target_risk] = []
            strategy_map[target_risk].append({
                "strategy": strategy.get("strategy", ""),
                "description": strategy.get("description", ""),
                "priority": strategy.get("priority", "medium")
            })
        
        # 为风险添加缓解策略
        for risk in potential_risks:
            risk_type = risk["type"]
            if risk_type in strategy_map:
                risk["mitigation_strategies"] = strategy_map[risk_type]
        
        return potential_risks
    
    async def _generate_improvement_suggestions(self, dimension_scores: Dict[str, float], 
                                              asset_analysis: Dict[str, Any], 
                                              risk_analysis: Dict[str, Any], 
                                              graph_checks: Dict[str, Any]) -> List[str]:
        """生成改进建议。"""
        
        suggestions = []
        
        # 基于维度分数的建议
        for dimension, score in dimension_scores.items():
            if score < 6.0:
                if dimension == "relevance":
                    suggestions.append("建议加强想法与领域主流研究的关联性，明确学术价值定位")
                elif dimension == "asset_availability":
                    suggestions.append("建议确认关键资产的可得性，制定备选资源获取方案")
                elif dimension == "risk_assessment":
                    suggestions.append("建议详细分析实现风险，制定相应的风险缓解策略")
                elif dimension == "graph_consistency":
                    suggestions.append("建议检查想法与现有知识的一致性，避免逻辑冲突")
        
        # 基于资产分析的建议
        missing_count = len(asset_analysis.get("missing_assets", []))
        if missing_count > 2:
            suggestions.append(f"建议优先解决 {missing_count} 项缺失资产，或寻找可行的替代方案")
        
        # 基于风险分析的建议
        high_risk_count = len([r for r in risk_analysis.get("identified_risks", []) if r.get("severity") == "high"])
        if high_risk_count > 0:
            suggestions.append(f"建议重点关注 {high_risk_count} 项高风险因素，制定详细的应对预案")
        
        # 基于图谱检查的建议
        if not graph_checks.get("is_consistent", True):
            conflict_count = graph_checks.get("conflict_analysis", {}).get("conflict_count", 0)
            if conflict_count > 0:
                suggestions.append(f"建议解决与图谱知识的 {conflict_count} 项冲突，或重新评估想法的可行性")
        
        # 通用改进建议
        if not suggestions:
            suggestions.append("建议进一步细化实现方案，加强实验设计的完整性")
        
        return suggestions


class IdeaRefinerAgent(BaseIdeaAgent):
    """第四阶段：引导式精炼与迭代决策。

    核心职责:
        - 汇总新颖性/可行性批判，生成可执行 `RefinementPrompt`，并决定是否继续迭代。

    注意事项:
        - 指令需具体可执行并附验收标准(目标分阈值/边界条件)。
    """

    def __init__(self, name: str, llm_factory: LLMFactory, db: AcademicPaperDatabase, config: Optional[AgentConfig] = None):
        super().__init__(name, llm_factory, db, config)
        self.default_thresholds = {
            "novelty_threshold": 8.0,
            "feasibility_threshold": 7.0,
            "combined_threshold": 15.0
        }

    async def make_refinement_prompt(self, idea: CandidateIdea, novelty: NoveltyCritique, feasibility: FeasibilityCritique) -> RefinementPrompt:
        """生成精炼指令。

        输入:
            - idea: 当前想法版本。
            - novelty: 新颖性批判结果。
            - feasibility: 可行性批判结果。

        输出:
            - RefinementPrompt: 决策(decision)与具体修改说明(instructions)、验收标准(acceptance_criteria)。

        实现步骤建议:
            1) 识别"像谁/缺什么/难在哪"，以差异-风险对齐生成修改路径。
            2) 给出明确保留/替换/补充项与可验证的验收标准。
        """
        # 分析当前状态
        decision = await self._analyze_refinement_decision(idea, novelty, feasibility)
        
        # 生成具体指令
        instructions = await self._generate_refinement_instructions(idea, novelty, feasibility, decision)
        
        # 设定验收标准
        acceptance_criteria = self._define_acceptance_criteria(novelty, feasibility, decision)
        
        return RefinementPrompt(
            idea_id=idea.id,
            decision=decision,
            instructions=instructions,
            rationale=await self._generate_rationale(novelty, feasibility, decision),
            acceptance_criteria=acceptance_criteria
        )

    async def make_refinement_decisions_batch(self, idea_critique_pairs: List[Tuple[CandidateIdea, NoveltyCritique, FeasibilityCritique]]) -> List[Dict[str, Any]]:
        """批量分析精炼决策类型：接受/修订/拆分/合并/丢弃。
        
        输入:
            - idea_critique_pairs: (想法, 新颖性评审, 可行性评审)的元组列表。
            
        输出:
            - List[Dict]: 决策结果列表，每个包含decision、confidence、reasoning等。
            
        实现思路:
            使用LLM一次性分析多个想法的双方批判，智能判断最佳决策路径。
        """
        print(f"🤔 开始批量精炼决策分析：{len(idea_critique_pairs)}个想法")
        
        # 构建批量分析prompt
        ideas_summary = []
        for i, (idea, novelty, feasibility) in enumerate(idea_critique_pairs, 1):
            ideas_summary.append(f"""**想法{i}**：
- ID: {idea.id}
- 标题: {idea.title}
- 核心假设: {idea.core_hypothesis}
- 创新点: {', '.join(idea.initial_innovation_points)}
- 当前版本: {idea.version}
- 新颖性评分: {novelty.novelty_score:.1f}/10.0 (相似工作数量: {len(novelty.similar_works)}, 差异性主张: {len(novelty.difference_claims)} 条)
- 可行性评分: {feasibility.feasibility_score:.1f}/10.0 (所需资源: {len(feasibility.required_assets)} 项, 潜在风险: {len(feasibility.potential_risks)} 项)""")
        
        prompt = f"""作为研究想法精炼专家和会议主席，请对以下{len(idea_critique_pairs)}个想法的评审结果进行批量分析，为每个想法给出最佳决策建议。

**待分析想法列表**：
{chr(10).join(ideas_summary)}

**决策选项说明**：
1. **accept**: 想法已足够成熟，可以进入实施阶段
2. **revise**: 想法有潜力但需要特定改进（如替换方法、调整范围等）
3. **split**: 想法过于复杂，应拆分为多个独立的子想法
4. **merge**: 想法过于单薄，需要与其他想法合并
5. **discard**: 想法存在根本性问题，改进潜力有限

请基于每个想法的研究价值、技术可行性、改进潜力等角度综合分析，给出最佳决策。

输出格式：
```json
{{
    "batch_analysis": "对整批想法的总体分析和决策思路",
    "decisions": [
        {{
            "idea_id": "想法ID",
            "decision": "accept|revise|split|merge|discard",
            "confidence": 0.95,
            "reasoning": "详细的决策理由，说明为什么选择这个决策",
            "key_factors": ["影响决策的关键因素1", "关键因素2", "关键因素3"]
        }}
    ]
}}
```"""

        try:
            llm = self.llm
            
            response_data = await llm.generate(
                model_name=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20000,
                temperature=0.3,
                agent_name=self.name,
                task_type="batch_refinement_decisions"
            )
            response = response_data.get('content', '')
            
            # 解析LLM响应
            response_text = response.strip()
            
            # 提取JSON部分
            import re
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                import json
                result = json.loads(json_match.group(1))
                decisions = result.get('decisions', [])
                batch_analysis = result.get('batch_analysis', '')
                
                print(f"✅ 批量决策分析完成：{len(decisions)}个决策")
                if batch_analysis:
                    print(f"📊 总体分析: {batch_analysis[:100]}...")
                
                # 确保决策数量匹配
                if len(decisions) != len(idea_critique_pairs):
                    print(f"⚠️ 决策数量不匹配，预期{len(idea_critique_pairs)}个，实际{len(decisions)}个，使用备用逻辑")
                    return await self._fallback_batch_decisions(idea_critique_pairs)
                
                return decisions
            else:
                print(f"⚠️ LLM批量决策响应解析失败，使用备用决策逻辑")
                return await self._fallback_batch_decisions(idea_critique_pairs)
                
        except Exception as e:
            print(f"❌ LLM批量决策分析失败: {e}")
            return await self._fallback_batch_decisions(idea_critique_pairs)

    async def _fallback_batch_decisions(self, idea_critique_pairs: List[Tuple[CandidateIdea, NoveltyCritique, FeasibilityCritique]]) -> List[Dict[str, Any]]:
        """批量决策的备用逻辑。"""
        print(f"🔄 执行批量决策备用逻辑")
        results = []
        for idea, novelty, feasibility in idea_critique_pairs:
            decision = await self._fallback_decision(idea, novelty, feasibility)
            results.append({
                "idea_id": idea.id,
                "decision": decision,
                "confidence": 0.7,
                "reasoning": f"备用决策逻辑：基于新颖性{novelty.novelty_score:.1f}和可行性{feasibility.feasibility_score:.1f}的综合判断",
                "key_factors": ["备用逻辑", "分数评估", "阈值比较"]
            })
        return results

    async def _analyze_refinement_decision(self, idea: CandidateIdea, novelty: NoveltyCritique, feasibility: FeasibilityCritique) -> str:
        """基于LLM分析决策类型：接受/修订/拆分/合并/丢弃。
        
        输入:
            - idea: 当前想法。
            - novelty: 新颖性评审。
            - feasibility: 可行性评审。
            
        输出:
            - str: 决策类型 (accept | revise | split | merge | discard)。
            
        实现思路:
            使用LLM分析双方批判，智能判断最佳决策路径。
        """
        print(f"    🤔 分析精炼决策 - 新颖性: {novelty.novelty_score:.1f}, 可行性: {feasibility.feasibility_score:.1f}")
        
        prompt = f"""作为研究想法精炼专家和会议主席，请分析以下想法的评审结果，并给出最佳决策建议。

**待分析想法**：
- ID: {idea.id}
- 标题: {idea.title}
- 核心假设: {idea.core_hypothesis}
- 创新点: {', '.join(idea.initial_innovation_points)}
- 当前版本: {idea.version}

**新颖性评审结果**：
- 评分: {novelty.novelty_score:.1f}/10.0
- 相似工作数量: {len(novelty.similar_works)}
- 差异性主张: {len(novelty.difference_claims)} 条
- 关键评价: {novelty.similar_works[0].get('summary', '暂无') if novelty.similar_works else '暂无相似工作'}

**可行性评审结果**：
- 评分: {feasibility.feasibility_score:.1f}/10.0
- 所需资源: {len(feasibility.required_assets)} 项
- 潜在风险: {len(feasibility.potential_risks)} 项
- 关键评价: {feasibility.relevance[:200] if feasibility.relevance else '暂无'}

**决策选项说明**：
1. **accept**: 想法已足够成熟，可以进入实施阶段
2. **revise**: 想法有潜力但需要特定改进（如替换方法、调整范围等）
3. **split**: 想法过于复杂，应拆分为多个独立的子想法
4. **merge**: 想法过于单薄，需要与其他想法合并
5. **discard**: 想法存在根本性问题，改进潜力有限

请基于以上信息，从研究价值、技术可行性、改进潜力等角度综合分析，给出最佳决策。

输出格式：
```json
{{
    "decision": "accept|revise|split|merge|discard",
    "confidence": 0.95,
    "reasoning": "详细的决策理由，说明为什么选择这个决策",
    "key_factors": ["影响决策的关键因素1", "关键因素2", "关键因素3"]
}}
```"""

        try:
            llm = self.llm
            
            response_data = await llm.generate(
                model_name=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=15000,
                temperature=0.3,
                agent_name=self.name,
                task_type="refinement_decision"
            )
            response = response_data.get('content', '')
            
            # 解析LLM响应
            response_text = response.strip()
            
            # 提取JSON部分
            import re
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                import json
                decision_result = json.loads(json_match.group(1))
                decision = decision_result.get('decision', 'revise')
                reasoning = decision_result.get('reasoning', '')
                confidence = decision_result.get('confidence', 0.5)
                
                print(f"    🎯 LLM决策: {decision} (置信度: {confidence:.2f})")
                if reasoning:
                    print(f"    💭 决策理由: {reasoning[:100]}...")
                
                return decision
            else:
                print(f"    ⚠️ LLM响应解析失败，使用备用决策逻辑")
                return await self._fallback_decision(idea, novelty, feasibility)
                
        except Exception as e:
            print(f"    ❌ LLM决策分析失败: {e}")
            return await self._fallback_decision(idea, novelty, feasibility)
    
    async def _fallback_decision(self, idea: CandidateIdea, novelty: NoveltyCritique, feasibility: FeasibilityCritique) -> str:
        """备用决策逻辑（简化版），当LLM调用失败时使用。"""
        combined_score = novelty.novelty_score + feasibility.feasibility_score
        
        if combined_score >= 15.0:
            return "accept"
        elif combined_score < 8.0:
            return "discard"
        elif len(idea.initial_innovation_points) > 4:
            return "split"
        elif len(idea.initial_innovation_points) < 2:
            return "merge"
        else:
            return "revise"
    
    async def _assess_improvement_potential(self, idea: CandidateIdea, novelty: NoveltyCritique, feasibility: FeasibilityCritique) -> float:
        """评估想法的改进潜力。"""
        
        potential_score = 0.0
        
        # 新颖性改进潜力
        if novelty.novelty_score < 6.0 and len(novelty.similar_works) > 0:
            # 如果有明确的相似工作，可以针对性改进
            potential_score += 0.3
        
        # 可行性改进潜力  
        if feasibility.feasibility_score < 6.0:
            # 检查是否有替代方案
            alternatives_count = sum(1 for asset in feasibility.required_assets 
                                   if asset.get('alternatives', []))
            if alternatives_count > 0:
                potential_score += 0.4
            
            # 检查风险是否可缓解
            mitigatable_risks = sum(1 for risk in feasibility.potential_risks 
                                  if risk.get('mitigation_strategies', []))
            if mitigatable_risks > 0:
                potential_score += 0.3
        
        # 想法结构完整性
        if len(idea.initial_innovation_points) >= 2 and len(idea.preliminary_experiments) >= 1:
            potential_score += 0.2
        
        return min(1.0, potential_score)
    
    async def _should_split_idea(self, idea: CandidateIdea, novelty: NoveltyCritique, feasibility: FeasibilityCritique) -> bool:
        """判断是否应该拆分想法。"""
        
        # 拆分条件1：创新点过多
        if len(idea.initial_innovation_points) > 4:
            return True
        
        # 拆分条件2：实验设计复杂度过高
        if len(idea.preliminary_experiments) > 3:
            return True
        
        # 拆分条件3：可行性风险过于分散
        if len(feasibility.potential_risks) > 5:
            risk_types = set(risk.get('type', '') for risk in feasibility.potential_risks)
            if len(risk_types) > 3:  # 风险类型过于多样
                return True
        
        # 拆分条件4：想法标题或假设表明多个独立方向
        idea_text = idea.title + " " + idea.core_hypothesis
        multi_indicators = ['and', '以及', '同时', 'both', 'multiple', '多个', '多种']
        multi_count = sum(1 for indicator in multi_indicators if indicator in idea_text.lower())
        if multi_count > 2:
            return True
        
        return False
    
    async def _should_merge_idea(self, idea: CandidateIdea, novelty: NoveltyCritique, feasibility: FeasibilityCritique) -> bool:
        """判断是否应该合并想法。"""
        
        # 合并条件1：创新点过少且单薄
        if len(idea.initial_innovation_points) < 2:
            return True
        
        # 合并条件2：实验设计过于简单
        if len(idea.preliminary_experiments) < 1:
            return True
        
        # 合并条件3：核心假设过于简短
        if len(idea.core_hypothesis.strip()) < 50:
            return True
        
        # 合并条件4：新颖性和可行性都中等但不突出
        if 4.0 <= novelty.novelty_score <= 6.0 and 4.0 <= feasibility.feasibility_score <= 6.0:
            return True
        
        return False

    async def _generate_refinement_instructions(self, idea: CandidateIdea, novelty: NoveltyCritique, 
                                               feasibility: FeasibilityCritique, decision: str) -> List[str]:
        """基于LLM生成具体的修改指令。
        
        输入:
            - idea: 当前想法。
            - novelty: 新颖性评审。
            - feasibility: 可行性评审。
            - decision: 决策类型。
            
        输出:
            - List[str]: 具体修改指令列表。
            
        实现思路:
            使用LLM根据评审结果生成具体可执行的修改指令。
        """
        print(f"    📝 生成 {decision} 类型的精炼指令")
        
        if decision in ["accept", "discard"]:
            if decision == "accept":
                return ["想法已达到验收标准，无需进一步修改"]
            else:
                return ["想法质量不足且无明显改进潜力，建议丢弃"]
        
        # 构建详细的上下文信息（添加类型检查）
        similar_works_summary = []
        for work in novelty.similar_works[:3]:  # 只取前3个最相似的工作
            if isinstance(work, dict):
                similar_works_summary.append(f"- {work.get('title', 'Unknown')}: {work.get('summary', 'No summary')}")
            else:
                # 如果是字符串，直接使用
                similar_works_summary.append(f"- {str(work)}")
        
        risks_summary = []
        for risk in feasibility.potential_risks[:3]:  # 只取前3个主要风险
            if isinstance(risk, dict):
                risks_summary.append(f"- {risk.get('description', 'Unknown risk')}")
            else:
                # 如果是字符串，直接使用
                risks_summary.append(f"- {str(risk)}")
        
        assets_summary = []
        for asset in feasibility.required_assets[:3]:  # 只取前3个主要资源
            if isinstance(asset, dict):
                assets_summary.append(f"- {asset.get('name', 'Unknown asset')}: {asset.get('availability', 'Unknown')}")
            else:
                # 如果是字符串，直接使用
                assets_summary.append(f"- {str(asset)}")

        prompt = f"""作为研究想法精炼专家，请根据评审结果为以下想法生成具体可执行的修改指令。

**待精炼想法**：
- ID: {idea.id}
- 标题: {idea.title}
- 核心假设: {idea.core_hypothesis}
- 创新点: {idea.initial_innovation_points}
- 版本: {idea.version}

**决策类型**: {decision}

**新颖性评审详情**：
- 评分: {novelty.novelty_score:.1f}/10.0
- 主要相似工作:
{chr(10).join(similar_works_summary) if similar_works_summary else '- 无相似工作'}
- 差异性主张数量: {len(novelty.difference_claims)}

**可行性评审详情**：
- 评分: {feasibility.feasibility_score:.1f}/10.0
- 主要风险:
{chr(10).join(risks_summary) if risks_summary else '- 无明显风险'}
- 关键资源:
{chr(10).join(assets_summary) if assets_summary else '- 无特殊资源需求'}

**任务要求**：
根据决策类型"{decision}"，生成3-5条具体可执行的修改指令。每条指令应该：
1. 具体明确，避免模糊表述
2. 可直接操作，指明需要修改的具体部分（标题/假设/创新点/实验设计等）
3. 提供改进方向和预期效果
4. 考虑新颖性和可行性的平衡

**指令类型说明**：
- revise: 针对性改进，保持想法主体结构不变
- split: 将复杂想法拆分为多个独立且聚焦的子想法
- merge: 与其他想法合并，或内部整合增强内容深度

输出格式：
```json
{{
    "instructions": [
        "具体指令1：详细描述需要做什么改动",
        "具体指令2：详细描述需要做什么改动",
        "具体指令3：详细描述需要做什么改动"
    ],
    "rationale": "生成这些指令的总体思路和预期改进效果"
}}
```"""

        try:
            llm = self.llm
            
            response_data = await llm.generate(
                model_name=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=15000,
                temperature=0.4,
                agent_name=self.name,
                task_type="refinement_instructions"
            )
            response = response_data.get('content', '')
            
            # 解析LLM响应
            response_text = response.strip()
            
            # 提取JSON部分
            import re
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                import json
                result = json.loads(json_match.group(1))
                instructions = result.get('instructions', [])
                rationale = result.get('rationale', '')
                
                print(f"    ✅ 生成 {len(instructions)} 条具体指令")
                if rationale:
                    print(f"    💡 指令思路: {rationale[:100]}...")
                
                return instructions
            else:
                print(f"    ⚠️ LLM指令生成解析失败，使用备用方案")
                return await self._fallback_instructions(idea, novelty, feasibility, decision)
                
        except Exception as e:
            print(f"    ❌ LLM指令生成失败: {e}")
            return await self._fallback_instructions(idea, novelty, feasibility, decision)

    async def _generate_refinement_instructions_batch(self, batch_data: List[Tuple[CandidateIdea, NoveltyCritique, FeasibilityCritique, str]]) -> List[List[str]]:
        """批量生成精炼指令。
        
        输入:
            - batch_data: [(想法, 新颖性评审, 可行性评审, 决策类型)] 的列表。
            
        输出:
            - List[List[str]]: 每个想法对应的指令列表。
            
        实现思路:
            使用LLM一次性为多个想法生成精炼指令。
        """
        print(f"📝 开始批量生成精炼指令：{len(batch_data)}个想法")
        
        # 准备批量prompt
        ideas_summary = []
        for i, (idea, novelty, feasibility, decision) in enumerate(batch_data, 1):
            # 构建相似工作和风险摘要（添加类型检查）
            similar_works_summary = []
            for work in novelty.similar_works[:3]:
                if isinstance(work, dict):
                    similar_works_summary.append(f"  - {work.get('title', 'Unknown')}")
                else:
                    # 如果是字符串，直接使用
                    similar_works_summary.append(f"  - {str(work)}")
            
            risks_summary = []
            for risk in feasibility.potential_risks[:3]:
                if isinstance(risk, dict):
                    risks_summary.append(f"  - {risk.get('description', 'Unknown risk')}")
                else:
                    # 如果是字符串，直接使用
                    risks_summary.append(f"  - {str(risk)}")
            
            assets_summary = []
            for asset in feasibility.required_assets[:3]:
                if isinstance(asset, dict):
                    assets_summary.append(f"  - {asset.get('name', 'Unknown asset')}")
                else:
                    # 如果是字符串，直接使用
                    assets_summary.append(f"  - {str(asset)}")

            ideas_summary.append(f"""**想法{i}**：
- ID: {idea.id}
- 标题: {idea.title}
- 核心假设: {idea.core_hypothesis}
- 创新点: {idea.initial_innovation_points}
- 版本: {idea.version}
- 决策类型: {decision}
- 新颖性评分: {novelty.novelty_score:.1f}/10.0
- 可行性评分: {feasibility.feasibility_score:.1f}/10.0
- 主要相似工作:
{chr(10).join(similar_works_summary) if similar_works_summary else '  - 无相似工作'}
- 主要风险:
{chr(10).join(risks_summary) if risks_summary else '  - 无明显风险'}
- 关键资源:
{chr(10).join(assets_summary) if assets_summary else '  - 无特殊资源需求'}""")

        prompt = f"""作为研究想法精炼专家，请根据评审结果为以下{len(batch_data)}个想法批量生成具体可执行的修改指令。

**待精炼想法列表**：
{chr(10).join(ideas_summary)}

**任务要求**：
为每个想法根据其决策类型生成3-5条具体可执行的修改指令。每条指令应该：
1. 具体明确，避免模糊表述
2. 可直接操作，指明需要修改的具体部分（标题/假设/创新点/实验设计等）
3. 提供改进方向和预期效果
4. 考虑新颖性和可行性的平衡

**指令类型说明**：
- accept: 想法已达到验收标准，无需进一步修改
- discard: 想法质量不足且无明显改进潜力，建议丢弃
- revise: 针对性改进，保持想法主体结构不变
- split: 将复杂想法拆分为多个独立且聚焦的子想法
- merge: 与其他想法合并，或内部整合增强内容深度

输出格式：
```json
{{
    "batch_analysis": "对整批想法的总体指令生成思路",
    "instructions_list": [
        {{
            "idea_id": "想法1的ID", 
            "instructions": [
                "具体指令1：详细描述需要做什么改动",
                "具体指令2：详细描述需要做什么改动",
                "具体指令3：详细描述需要做什么改动"
            ],
            "rationale": "生成这些指令的思路"
        }}
    ]
}}
```"""

        try:
            llm = self.llm
            
            response_data = await llm.generate(
                model_name=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=25000,
                temperature=0.4,
                agent_name=self.name,
                task_type="batch_refinement_instructions"
            )
            response = response_data.get('content', '')
            
            # 解析LLM响应
            response_text = response.strip()
            
            # 提取JSON部分
            import re
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                import json
                result = json.loads(json_match.group(1))
                instructions_list = result.get('instructions_list', [])
                batch_analysis = result.get('batch_analysis', '')
                
                print(f"✅ 批量指令生成完成：{len(instructions_list)}个想法的指令")
                if batch_analysis:
                    print(f"💡 批量分析: {batch_analysis[:100]}...")
                
                # 确保指令数量匹配并提取指令
                final_instructions = []
                for i, (idea, _, _, decision) in enumerate(batch_data):
                    if i < len(instructions_list):
                        instruction_data = instructions_list[i]
                        # 添加类型检查，确保 instruction_data 是字典
                        if isinstance(instruction_data, dict):
                            instructions = instruction_data.get('instructions', [])
                        elif isinstance(instruction_data, list):
                            instructions = instruction_data  # 如果已经是列表，直接使用
                        else:
                            instructions = []  # 其他情况使用空列表
                        
                        # 对于accept和discard类型，提供默认指令
                        if decision == "accept" and not instructions:
                            instructions = ["想法已达到验收标准，无需进一步修改"]
                        elif decision == "discard" and not instructions:
                            instructions = ["想法质量不足且无明显改进潜力，建议丢弃"]
                        
                        final_instructions.append(instructions)
                        print(f"    ✅ 想法 {idea.id}: {len(instructions)} 条指令")
                    else:
                        print(f"    ⚠️ 想法 {idea.id} 缺少指令，使用备用逻辑")
                        fallback_instructions = await self._fallback_instructions(idea, batch_data[i][1], batch_data[i][2], decision)
                        final_instructions.append(fallback_instructions)
                
                return final_instructions
            else:
                print(f"⚠️ LLM批量指令响应解析失败，使用备用方案")
                return await self._fallback_instructions_batch(batch_data)
                
        except Exception as e:
            print(f"❌ LLM批量指令生成失败: {e}")
            return await self._fallback_instructions_batch(batch_data)

    async def _fallback_instructions_batch(self, batch_data: List[Tuple[CandidateIdea, NoveltyCritique, FeasibilityCritique, str]]) -> List[List[str]]:
        """批量指令生成的备用逻辑。"""
        print(f"🔄 执行批量指令备用逻辑")
        results = []
        for idea, novelty, feasibility, decision in batch_data:
            instructions = await self._fallback_instructions(idea, novelty, feasibility, decision)
            results.append(instructions)
        return results

    async def make_refinement_prompts_batch(self, decisions: List[Dict[str, Any]], 
                                          idea_critique_pairs: List[Tuple[CandidateIdea, NoveltyCritique, FeasibilityCritique]]) -> List[RefinementPrompt]:
        """批量生成精炼指令。
        
        输入:
            - decisions: 批量决策结果列表。
            - idea_critique_pairs: (想法, 新颖性评审, 可行性评审)的元组列表。
            
        输出:
            - List[RefinementPrompt]: 精炼指令列表。
            
        实现思路:
            基于批量决策结果，使用批量LLM调用为所有想法生成对应的精炼指令。
        """
        print(f"📝 开始批量生成精炼指令：{len(decisions)}个决策（真正的批量模式）")
        
        try:
            # 准备批量数据：[(想法, 新颖性评审, 可行性评审, 决策类型)]
            batch_data = []
            for decision_info, (idea, novelty, feasibility) in zip(decisions, idea_critique_pairs):
                # 添加类型检查
                if isinstance(decision_info, dict):
                    decision = decision_info.get('decision', 'revise')
                elif isinstance(decision_info, str):
                    decision = decision_info  # 如果是字符串，直接使用
                else:
                    decision = 'revise'  # 默认值
                batch_data.append((idea, novelty, feasibility, decision))
            
            # 🚀 批量生成所有想法的指令（一次LLM调用）
            print(f"    🤖 启动批量指令生成LLM调用...")
            all_instructions = await self._generate_refinement_instructions_batch(batch_data)
            
            # 构建最终的RefinementPrompt对象列表
            refinement_prompts = []
            for i, (decision_info, (idea, novelty, feasibility)) in enumerate(zip(decisions, idea_critique_pairs)):
                # 添加类型检查
                if isinstance(decision_info, dict):
                    decision = decision_info.get('decision', 'revise')
                    reasoning = decision_info.get('reasoning', '')
                elif isinstance(decision_info, str):
                    decision = decision_info  # 如果是字符串，直接使用
                    reasoning = ''
                else:
                    decision = 'revise'  # 默认值
                    reasoning = ''
                
                try:
                    # 获取批量生成的指令
                    instructions = all_instructions[i] if i < len(all_instructions) else []
                    
                    # 设定验收标准
                    acceptance_criteria = self._define_acceptance_criteria(novelty, feasibility, decision)
                    
                    # 生成理由（直接使用决策理由）
                    rationale = reasoning if reasoning else await self._generate_rationale(novelty, feasibility, decision)
                    
                    refinement_prompt = RefinementPrompt(
                        idea_id=idea.id,
                        decision=decision,
                        instructions=instructions,
                        rationale=rationale,
                        acceptance_criteria=acceptance_criteria
                    )
                    
                    refinement_prompts.append(refinement_prompt)
                    print(f"    ✅ 想法 {idea.id}: {decision} 指令组装完成（{len(instructions)}条指令）")
                    
                except Exception as e:
                    print(f"    ❌ 想法 {idea.id} 指令组装失败: {e}")
                    # 创建默认的精炼指令
                    fallback_instructions = await self._fallback_instructions(idea, novelty, feasibility, decision)
                    refinement_prompt = RefinementPrompt(
                        idea_id=idea.id,
                        decision=decision,
                        instructions=fallback_instructions,
                        rationale=f"指令组装失败，使用备用逻辑: {str(e)}",
                        acceptance_criteria=self._define_acceptance_criteria(novelty, feasibility, decision)
                    )
                    refinement_prompts.append(refinement_prompt)
            
            print(f"✅ 批量精炼指令生成完成：{len(refinement_prompts)}个指令（批量LLM模式）")
            return refinement_prompts
            
        except Exception as e:
            print(f"❌ 批量指令生成流程失败: {e}，回退到逐个生成模式")
            # 完全失败时回退到原来的逐个生成模式
            return await self._fallback_prompts_batch_generation(decisions, idea_critique_pairs)
   
    async def _fallback_prompts_batch_generation(self, decisions: List[Dict[str, Any]], 
                                            idea_critique_pairs: List[Tuple[CandidateIdea, NoveltyCritique, FeasibilityCritique]]) -> List[RefinementPrompt]:
        """批量指令生成的完全备用方案：回退到逐个生成模式。"""
        print(f"🔄 执行完全备用方案：逐个生成指令模式")
        
        refinement_prompts = []
        for i, (decision_info, (idea, novelty, feasibility)) in enumerate(zip(decisions, idea_critique_pairs)):
            # 添加类型检查，确保 decision_info 是字典
            if isinstance(decision_info, dict):
                decision = decision_info.get('decision', 'revise')
                reasoning = decision_info.get('reasoning', '')
            elif isinstance(decision_info, str):
                decision = decision_info  # 如果是字符串，直接使用
                reasoning = ''
            else:
                decision = 'revise'  # 默认值
                reasoning = ''
            
            try:
                # 逐个生成具体指令
                instructions = await self._generate_refinement_instructions(idea, novelty, feasibility, decision)
                acceptance_criteria = self._define_acceptance_criteria(novelty, feasibility, decision)
                rationale = reasoning if reasoning else await self._generate_rationale(novelty, feasibility, decision)
                
                refinement_prompt = RefinementPrompt(
                    idea_id=idea.id,
                    decision=decision,
                    instructions=instructions,
                    rationale=rationale,
                    acceptance_criteria=acceptance_criteria
                )
                refinement_prompts.append(refinement_prompt)
                
            except Exception as e:
                print(f"    ❌ 想法 {idea.id} 备用指令生成失败: {e}")
                fallback_instructions = await self._fallback_instructions(idea, novelty, feasibility, decision)
                refinement_prompt = RefinementPrompt(
                    idea_id=idea.id,
                    decision=decision,
                    instructions=fallback_instructions,
                    rationale=f"备用指令生成失败: {str(e)}",
                    acceptance_criteria=self._define_acceptance_criteria(novelty, feasibility, decision)
                )
                refinement_prompts.append(refinement_prompt)
        
        return refinement_prompts
    
    async def _fallback_instructions(self, idea: CandidateIdea, novelty: NoveltyCritique, 
                                   feasibility: FeasibilityCritique, decision: str) -> List[str]:
        """备用指令生成逻辑，当LLM调用失败时使用。"""
        instructions = []
        
        if decision == "revise":
            if novelty.novelty_score < 7.0:
                instructions.append("增强想法的新颖性：在标题中突出独特的技术路径或应用场景")
                instructions.append("强化与现有工作的差异性：明确说明核心方法的创新之处")
            
            if feasibility.feasibility_score < 7.0:
                instructions.append("降低实现难度：简化技术方案或提供可行的替代方案")
                instructions.append("补充实验设计：添加具体的验证方法和评价指标")
        
        elif decision == "split":
            instructions.append("拆分为多个独立想法：将核心假设分解为2-3个可独立研究的子问题")
            instructions.append("简化单个想法的复杂度：每个子想法专注一个核心创新点")
        
        elif decision == "merge":
            instructions.append("增强想法深度：补充更多创新点和技术细节")
            instructions.append("拓展应用范围：考虑与相关领域的结合应用")
        
        return instructions if instructions else ["进行常规优化：完善想法的表述和技术描述"]
    
    async def _generate_novelty_instructions(self, idea: CandidateIdea, novelty: NoveltyCritique) -> List[str]:
        """生成新颖性改进指令。"""
        
        instructions = []
        
        if novelty.novelty_score < self.default_thresholds["novelty_threshold"]:
            # 策略1：强化与最相似工作的差异
            if novelty.similar_works:
                most_similar = max(novelty.similar_works, key=lambda x: x.get('max_similarity', 0))
                similar_title = most_similar.get('title', 'unknown work')
                
                instructions.append(
                    f"强化与《{similar_title}》的差异性：在标题和核心假设中明确突出本想法的独特价值主张"
                )
            
            # 策略2：基于差异性主张的具体改进
            if novelty.difference_claims:
                top_claims = novelty.difference_claims[:2]  # 取前2个最重要的差异点
                for i, claim in enumerate(top_claims):
                    instructions.append(
                        f"根据差异点{i+1}（{claim}），在创新点中补充相应的技术细节和实现机制"
                    )
            
            # 策略3：补充创新机制
            if len(idea.initial_innovation_points) < 3:
                instructions.append(
                    "补充至少1个新的创新点，重点描述与现有方法的技术差异和优势"
                )
            
            # 策略4：增强方法新颖性
            facet_scores = novelty.facet_scores
            if facet_scores.get('methodological', 7) < 7:
                instructions.append(
                    "改进方法新颖性：引入新的技术组件或创新的组合方式，避免直接复用现有方法"
                )
            
            # 策略5：增强应用新颖性
            if facet_scores.get('application', 7) < 7:
                instructions.append(
                    "扩展应用新颖性：明确指出在新应用场景下的独特挑战和解决方案"
                )
        
        return instructions
    
    async def _generate_feasibility_instructions(self, idea: CandidateIdea, feasibility: FeasibilityCritique) -> List[str]:
        """生成可行性改进指令。"""
        
        instructions = []
        
        if feasibility.feasibility_score < self.default_thresholds["feasibility_threshold"]:
            # 策略1：解决资产可得性问题
            missing_assets = [asset for asset in feasibility.required_assets 
                            if asset.get('status') == 'missing']
            
            for asset in missing_assets[:2]:  # 处理前2个最关键的缺失资产
                asset_type = asset.get('type', 'Unknown')
                asset_id = asset.get('id', 'unknown')
                
                alternatives = asset.get('alternatives', [])
                if alternatives:
                    best_alt = max(alternatives, key=lambda x: x.get('feasibility', 0))
                    instructions.append(
                        f"替换缺失的{asset_type} '{asset_id}'为替代方案：{best_alt.get('description', '寻找可得的替代资源')}"
                    )
                else:
                    instructions.append(
                        f"为缺失的{asset_type} '{asset_id}'寻找开源实现或公开数据集替代"
                    )
            
            # 策略2：缓解主要风险
            high_priority_risks = [risk for risk in feasibility.potential_risks 
                                 if risk.get('severity') in ['high', 'medium']]
            
            for risk in high_priority_risks[:2]:  # 处理前2个高优先级风险
                risk_type = risk.get('type', 'unknown')
                mitigation_strategies = risk.get('mitigation_strategies', [])
                
                if mitigation_strategies:
                    strategy = mitigation_strategies[0]  # 采用第一个策略
                    instructions.append(
                        f"缓解{risk_type}风险：{strategy.get('description', '制定详细的风险应对计划')}"
                    )
                else:
                    instructions.append(
                        f"制定{risk_type}风险的具体缓解策略，降低实现难度"
                    )
            
            # 策略3：改善维度分数
            dimension_scores = feasibility.dimension_scores
            
            if dimension_scores.get('relevance', 7) < 6:
                instructions.append(
                    "增强想法的领域相关性：明确与当前研究热点的联系，强化学术价值"
                )
            
            if dimension_scores.get('asset_availability', 7) < 6:
                instructions.append(
                    "改善资产可得性：优先使用公开数据集和开源工具，避免依赖专有资源"
                )
            
            if dimension_scores.get('graph_consistency', 7) < 6:
                instructions.append(
                    "解决与现有知识的一致性问题：重新评估方法-任务组合的合理性"
                )
            
            # 策略4：简化复杂度
            if len(feasibility.potential_risks) > 4:
                instructions.append(
                    "简化实现复杂度：将复杂的技术方案分解为更容易实现的子步骤"
                )
        
        return instructions
    
    async def _generate_split_instructions(self, idea: CandidateIdea, novelty: NoveltyCritique, feasibility: FeasibilityCritique) -> List[str]:
        """生成拆分指令。"""
        
        instructions = []
        
        # 拆分策略1：按创新点拆分
        if len(idea.initial_innovation_points) > 3:
            instructions.append(
                f"将 {len(idea.initial_innovation_points)} 个创新点拆分为2-3个独立的子想法，每个想法专注于1-2个核心创新"
            )
        
        # 拆分策略2：按实验维度拆分
        if len(idea.preliminary_experiments) > 2:
            instructions.append(
                f"将 {len(idea.preliminary_experiments)} 个实验拆分为不同的研究阶段，每个阶段形成独立的验证想法"
            )
        
        # 拆分策略3：按风险类型拆分
        if len(feasibility.potential_risks) > 4:
            risk_types = set(risk.get('type', '') for risk in feasibility.potential_risks)
            if len(risk_types) > 2:
                instructions.append(
                    f"按风险类型拆分：将涉及 {len(risk_types)} 种不同风险的想法分解为风险更集中的子方案"
                )
        
        # 拆分策略4：按应用场景拆分
        idea_text = idea.title + " " + idea.core_hypothesis
        if any(indicator in idea_text.lower() for indicator in ['multi', '多', 'various', '各种']):
            instructions.append(
                "按应用场景拆分：将多场景应用的想法拆分为专注于单一场景的具体实现"
            )
        
        return instructions
    
    async def _generate_merge_instructions(self, idea: CandidateIdea, novelty: NoveltyCritique, feasibility: FeasibilityCritique) -> List[str]:
        """生成合并指令。"""
        
        instructions = []
        
        # 合并策略1：补充创新点
        if len(idea.initial_innovation_points) < 2:
            instructions.append(
                "补充创新点：与相关的技术创新或应用扩展进行合并，形成更丰富的研究内容"
            )
        
        # 合并策略2：扩展实验设计
        if len(idea.preliminary_experiments) < 2:
            instructions.append(
                "扩展实验验证：与相关的评估维度或数据集验证进行合并，形成更全面的验证方案"
            )
        
        # 合并策略3：丰富核心假设
        if len(idea.core_hypothesis.strip()) < 50:
            instructions.append(
                "丰富核心假设：结合相关的理论基础或技术机制，形成更深入的研究假设"
            )
        
        # 合并策略4：提升整体价值
        combined_score = novelty.novelty_score + feasibility.feasibility_score
        if combined_score < 12:
            instructions.append(
                f"提升整体价值：与互补的研究方向合并，目标是将综合分从 {combined_score:.1f} 提升到 >12"
            )
        
        return instructions

        
    def _define_acceptance_criteria(self, novelty: NoveltyCritique, feasibility: FeasibilityCritique, decision: str) -> List[str]:
        """定义验收标准。
        
        输入:
            - novelty: 新颖性评审。
            - feasibility: 可行性评审。
            - decision: 决策类型。
            
        输出:
            - List[str]: 验收标准列表。
            
        实现思路:
            基于评审结果和决策类型，生成具体的验收标准。
        """
        criteria = []
        
        # 基础质量标准
        criteria.append("想法表述清晰完整，逻辑自洽")
        criteria.append("核心假设明确且具有可验证性")
        criteria.append("创新点描述具体且与现有工作有明确差异")
        
        if decision in ["accept", "revise"]:
            # 新颖性具体标准
            if novelty.novelty_score < self.default_thresholds["novelty_threshold"]:
                criteria.append("新颖性评分达到阈值要求")
                
                if len(novelty.similar_works) > 0:
                    criteria.append("与最相似工作的差异性已明确阐述并在标题/假设中体现")
                
                # 添加类型检查
                facet_scores = novelty.facet_scores
                if isinstance(facet_scores, dict):
                    if facet_scores.get('methodological', 7) < 7:
                        criteria.append("方法新颖性分面得分 >= 7.0")
                    if facet_scores.get('conceptual', 7) < 7:
                        criteria.append("概念新颖性分面得分 >= 7.0")
            
            # 可行性具体标准
            if feasibility.feasibility_score < self.default_thresholds["feasibility_threshold"]:
                # 资产可得性标准 - 添加类型检查
                missing_assets = []
                for asset in feasibility.required_assets:
                    if isinstance(asset, dict) and asset.get('status') == 'missing':
                        missing_assets.append(asset)
                if missing_assets:
                    criteria.append("所有标记为'缺失'的关键资产已被替换或提供替代方案")
                
                # 风险缓解标准 - 添加类型检查
                high_risks = []
                for risk in feasibility.potential_risks:
                    if isinstance(risk, dict) and risk.get('severity') == 'high':
                        high_risks.append(risk)
                if high_risks:
                    criteria.append("所有高风险项已制定具体的缓解策略")
                
                # 维度改进标准 - 添加类型检查
                dimension_scores = feasibility.dimension_scores
                if isinstance(dimension_scores, dict):
                    if dimension_scores.get('asset_availability', 7) < 6:
                        criteria.append("资产可得性维度得分 >= 6.0")
                    if dimension_scores.get('relevance', 7) < 6:
                        criteria.append("领域相关性维度得分 >= 6.0")
        
        elif decision == "split":
            criteria.append("已成功拆分为2-3个独立的聚焦子想法")
            criteria.append("每个子想法都有明确的研究边界和核心问题")
            criteria.append("拆分后的想法复杂度适中且各自具有研究价值")
        
        elif decision == "merge":
            criteria.append("已与其他相关想法成功合并或内容深度显著增强")
            criteria.append("合并后的想法内容更加丰富且逻辑连贯")
            criteria.append("创新点更加突出且研究价值明显提升")
        
        elif decision == "discard":
            criteria.append("已确认想法无法通过合理修改达到质量要求")
        
        return criteria

    async def _generate_rationale(self, novelty: NoveltyCritique, feasibility: FeasibilityCritique, decision: str) -> str:
        """基于LLM生成决策理由。
        
        输入:
            - novelty: 新颖性评审。
            - feasibility: 可行性评审。
            - decision: 决策类型。
            
        输出:
            - str: 决策理由说明。
        """
        combined_score = novelty.novelty_score + feasibility.feasibility_score
        
        prompt = f"""作为研究想法评审专家，请为以下决策生成简洁有力的理由说明。

**评审摘要**：
- 新颖性评分: {novelty.novelty_score:.1f}/10.0
- 可行性评分: {feasibility.feasibility_score:.1f}/10.0
- 综合评分: {combined_score:.1f}/20.0
- 相似工作数量: {len(novelty.similar_works)}
- 风险点数量: {len(feasibility.potential_risks)}

**做出的决策**: {decision}

请用1-2句话解释这个决策的合理性，重点说明：
1. 决策的主要依据（分数水平、关键问题、改进潜力等）
2. 预期的后续行动方向

要求：
- 语言简洁专业，避免冗余
- 突出关键数据和事实依据
- 体现专业判断的逻辑性

直接输出理由说明文本，无需格式化。"""

        try:
            llm = self.llm
            
            response_data = await llm.generate(
                model_name=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=15000,
                temperature=0.2,
                agent_name=self.name,
                task_type="refinement_rationale"
            )
            response = response_data.get('content', '')
            
            rationale = response.strip()
            return rationale if rationale else self._fallback_rationale(novelty, feasibility, decision)
            
        except Exception as e:
            print(f"    ⚠️ LLM理由生成失败: {e}")
            return self._fallback_rationale(novelty, feasibility, decision)
    
    def _fallback_rationale(self, novelty: NoveltyCritique, feasibility: FeasibilityCritique, decision: str) -> str:
        """备用决策理由生成，当LLM调用失败时使用。"""
        combined_score = novelty.novelty_score + feasibility.feasibility_score
        
        if decision == "accept":
            return f"想法达到验收标准：新颖性{novelty.novelty_score:.1f}/10.0，可行性{feasibility.feasibility_score:.1f}/10.0，综合表现优秀"
        elif decision == "revise":
            return f"想法有潜力但需改进：新颖性{novelty.novelty_score:.1f}，可行性{feasibility.feasibility_score:.1f}，通过针对性修改可提升质量"
        elif decision == "split":
            return f"想法过于复杂（综合分{combined_score:.1f}），建议拆分为多个聚焦的子问题"
        elif decision == "merge":
            return f"想法较为单薄（综合分{combined_score:.1f}），建议与其他想法合并或增强内容深度"
        elif decision == "discard":
            return f"想法质量不足（综合分{combined_score:.1f}）且改进潜力有限，建议丢弃"
        else:
            return f"基于评审结果做出{decision}决策，综合评分{combined_score:.1f}/20.0"


# =========================
# 协调器（骨架）
# =========================


class IdeaGenCoordinator:
    """Idea Generation 子系统的总协调器。

    目标:
        - 管理 5 个智能体的调用顺序与并发，提供端到端的想法生成与迭代流程。

    输入:
        - llm_factory: 统一LLM工厂。
        - db: 学术数据库。
        - config: 运行时配置（并发、阈值、最大迭代轮次等）。

    输出:
        - 结构化字典，包括机会图谱、候选想法、各轮评审与精炼记录、最终想法集。

    注意事项:
        - 并发上限默认=6；可通过 config 覆盖。
        - 各阶段的产出需包含 `provenance` 以便追溯。
    """

    def __init__(self, llm_factory: LLMFactory, db: AcademicPaperDatabase, config: Optional[Dict[str, Any]] = None):
        self.llm_factory = llm_factory
        self.db = db
        self.config = config or {}
        
        # 配置参数解析
        self.concurrency: int = int(self.config.get("idea_concurrency", 6))
        self.max_rounds: int = int(self.config.get("max_rounds", 3))
        self.novelty_threshold: float = float(self.config.get("novelty_threshold", 8.0))
        self.feasibility_threshold: float = float(self.config.get("feasibility_threshold", 7.0))
        self.max_initial_ideas: int = int(self.config.get("max_initial_ideas", 6))
        
        # 组装智能体（可在外部以依赖注入方式覆盖）
        # 创建各智能体的配置
        miner_config = AgentConfig(
            model_name=self.config.get("miner_model", ModelType.GEMINI.value),
            temperature=0.7,
            max_tokens=15000,
            role_description="机会图谱构建专家",
            system_message="你是机会图谱构建专家，擅长从学术文献中抽取实体关系并识别研究机会。"
        )
        
        generator_config = AgentConfig(
            model_name=self.config.get("generator_model", ModelType.GEMINI.value),
            temperature=0.8,
            max_tokens=15000,
            role_description="研究想法生成专家",
            system_message="你是研究想法生成专家，擅长基于机会图谱生成创新的研究想法。"
        )
        
        novelty_critic_config = AgentConfig(
            model_name=self.config.get("novelty_critic_model", ModelType.GEMINI.value),
            temperature=0.3,
            max_tokens=15000,
            role_description="新颖性评审专家",
            system_message="你是新颖性评审专家，擅长评估研究想法的创新性和独特性。"
        )
        
        feasibility_critic_config = AgentConfig(
            model_name=self.config.get("feasibility_critic_model", ModelType.GEMINI.value),
            temperature=0.3,
            max_tokens=15000,
            role_description="可行性评审专家",
            system_message="你是可行性评审专家，擅长评估研究想法的技术可行性和实现难度。"
        )
        
        refiner_config = AgentConfig(
            model_name=self.config.get("refiner_model", ModelType.GEMINI.value),
            temperature=0.5,
            max_tokens=15000,
            role_description="想法精炼专家",
            system_message="你是想法精炼专家，擅长根据评审意见优化和改进研究想法。"
        )
        
        self.miner = IdeaMinerAgent("IdeaMiner", llm_factory, db, miner_config)
        self.generator = IdeaGeneratorAgent("IdeaGenerator", llm_factory, db, generator_config)
        self.novelty_critic = NoveltyCriticAgent("NoveltyCritic", llm_factory, db, novelty_critic_config)
        self.feasibility_critic = FeasibilityCriticAgent("FeasibilityCritic", llm_factory, db, feasibility_critic_config)
        self.refiner = IdeaRefinerAgent("IdeaRefiner", llm_factory, db, refiner_config)
        
        print(f"🏗️ 协调器初始化完成 - 并发度: {self.concurrency}, 最大轮次: {self.max_rounds}")

    async def run_pipeline(self, final_result: Dict[str, Any], enriched_outline: Dict[str, Any]) -> Dict[str, Any]:
        """端到端执行：图谱→生成→评审→精炼→收敛。

        输入:
            - final_result: 上游 Survey Gen 的整合结果。
            - enriched_outline: 上游丰富大纲。

        输出:
            - Dict: 结构化产出，包括 graph/candidates/iterations/final_ideas 等。

        实现步骤建议:
            1) 调用 miner 构建 SemanticOpportunityGraph。
            2) 调用 generator 生成初始候选想法（≤并发上限）。
            3) 对每个想法并行执行 debate_loop（新颖性/可行性→精炼→重生）。
            4) 汇总各想法的版本链与终止状态，生成最终报表。
        """
        print("🔍 第一阶段：构建语义机会图谱")
        start_time = datetime.now()
        
        try:
            # 阶段1：构建机会图谱
            graph = await self.miner.build_opportunity_graph(final_result, enriched_outline)
            stage1_time = (datetime.now() - start_time).total_seconds()
            print(f"   ✅ 图谱构建完成 - {graph.number_of_nodes()}节点, {graph.number_of_edges()}边, 耗时{stage1_time:.1f}秒")
        except Exception as e:
            print(f"   ❌ 图谱构建失败: {e}")
            raise RuntimeError(f"图谱构建阶段失败: {e}") from e
        
        print("💡 第二阶段：生成候选想法")
        stage2_start = datetime.now()
        
        try:
            # 阶段2：生成初始候选想法
            generation_result = await self.generator.generate_candidates(graph, max_ideas=self.max_initial_ideas)
            initial_candidates = generation_result['all_candidates']
            stage2_time = (datetime.now() - stage2_start).total_seconds()
            print(f"   ✅ 候选想法生成完成 - {len(initial_candidates)}个想法, 耗时{stage2_time:.1f}秒")
            
            if not initial_candidates:
                print("   ⚠️ 未生成任何候选想法，流程提前结束")
                return {
                    "opportunity_graph": {
                        "node_count": graph.number_of_nodes(),
                        "edge_count": graph.number_of_edges(),
                        "gaps_count": len(graph.graph.get('gaps', [])),
                        "provenance": graph.graph.get('provenance', {})
                    },
                    "initial_candidates": {"count": 0, "ideas": []},
                    "final_ideas": {"accepted": {"count": 0, "ideas": [], "avg_scores": {}}, 
                                  "discarded": {"count": 0, "ideas": [], "reasons": []},
                                  "failed": {"count": 0, "errors": []}},
                    "statistics": {"total_candidates": 0, "success_rate": 0.0},
                    "timestamp": datetime.now().isoformat(),
                    "status": "no_candidates_generated"
                }
        except Exception as e:
            print(f"   ❌ 候选想法生成失败: {e}")
            raise RuntimeError(f"想法生成阶段失败: {e}") from e
        
        print(f"🏛️ 第三阶段：辩论竞技场（{len(initial_candidates)}个想法，批量评审模式）")
        stage3_start = datetime.now()
        
        # 阶段3：批量评审模式的辩论循环
        debate_results = await self._batch_debate_arena(generation_result, graph)
        
        # 处理异常结果
        processed_results = []
        for i, result in enumerate(debate_results):
            if isinstance(result, Exception):
                processed_results.append({
                    "status": "failed",
                    "final_idea": initial_candidates[i],
                    "history": [],
                    "error": str(result)
                })
            else:
                processed_results.append(result)
        
        stage3_time = (datetime.now() - stage3_start).total_seconds()
        print(f"   ✅ 辩论竞技场完成，耗时{stage3_time:.1f}秒")
        
        # 阶段4：整理最终结果
        print("📊 第四阶段：整理最终结果")
        final_results = await self._synthesize_final_results(graph, initial_candidates, processed_results)
        
        total_time = (datetime.now() - start_time).total_seconds()
        final_results["execution_details"] = {
            "stage_times": {
                "graph_construction": stage1_time,
                "idea_generation": stage2_time,
                "debate_arena": stage3_time,
                "result_synthesis": (datetime.now() - start_time).total_seconds() - stage1_time - stage2_time - stage3_time
            },
            "total_time": total_time
        }
        
        print(f"✅ 流水线执行完成，总耗时{total_time:.1f}秒")
        return final_results

    async def debate_loop(
        self,
        idea: CandidateIdea,
        graph: SemanticOpportunityGraph,
        max_rounds: Optional[int] = None,
        novelty_threshold: Optional[float] = None,
        feasibility_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """辩论竞技场：针对单个想法的迭代闭环。

        输入:
            - idea: 初始候选想法。
            - graph: 语义机会图谱。
            - max_rounds: 最大迭代轮数（默认3）。
            - novelty_threshold/feasibility_threshold: 接受阈值。

        输出:
            - Dict: {final_idea, history: [{novelty, feasibility, refinement, version}, ...], status}

        实现步骤建议:
            1) 同步/并行获取 NoveltyCritique 与 FeasibilityCritique。
            2) 交由 IdeaRefiner 生成 RefinementPrompt 并决策（revise/split/merge/discard/accept）。
            3) 若 revise，则回到 IdeaGenerator 生成新版本 idea(version+1) 并继续；达成阈值或上限则终止。
        """
        # 使用配置参数的默认值
        max_rounds = max_rounds or self.max_rounds
        novelty_threshold = novelty_threshold or self.novelty_threshold
        feasibility_threshold = feasibility_threshold or self.feasibility_threshold
        
        current_idea = idea
        history = []
        
        for round_num in range(max_rounds):
            print(f"  🔄 想法 {idea.id} 第 {round_num + 1} 轮评审（版本 {current_idea.version}）")
            
            try:
                # 并行执行新颖性与可行性评审
                novelty_task = self.novelty_critic.assess_novelty(current_idea)
                feasibility_task = self.feasibility_critic.assess_feasibility(current_idea, graph)
                
                novelty_critique, feasibility_critique = await asyncio.gather(novelty_task, feasibility_task)
                
                print(f"    📈 评审完成 - 新颖性: {novelty_critique.novelty_score:.1f}, 可行性: {feasibility_critique.feasibility_score:.1f}")
                
            except Exception as e:
                print(f"    ❌ 评审失败: {e}")
                return {
                    "status": "failed",
                    "final_idea": current_idea,
                    "history": history,
                    "error": f"评审阶段失败: {str(e)}"
                }
            
            # 生成精炼指令
            refinement_prompt = await self.refiner.make_refinement_prompt(
                current_idea, novelty_critique, feasibility_critique
            )
            
            # 记录本轮历史
            round_record = {
                "round": round_num + 1,
                "idea_version": current_idea.version,
                "novelty": novelty_critique,
                "feasibility": feasibility_critique,
                "refinement": refinement_prompt,
                "timestamp": datetime.now().isoformat()
            }
            history.append(round_record)
            
            # 根据决策确定下一步
            decision = refinement_prompt.decision
            print(f"    🤔 决策结果: {decision}")
            
            if decision == "accept":
                print(f"    ✅ 想法 {idea.id} 被接受！")
                return {
                    "status": "accepted",
                    "final_idea": current_idea,
                    "history": history,
                    "final_scores": {
                        "novelty": novelty_critique.novelty_score,
                        "feasibility": feasibility_critique.feasibility_score,
                        "combined": novelty_critique.novelty_score + feasibility_critique.feasibility_score
                    },
                    "acceptance_round": round_num + 1
                }
            elif decision == "discard":
                print(f"    ❌ 想法 {idea.id} 被丢弃: {refinement_prompt.rationale}")
                return {
                    "status": "discarded",
                    "final_idea": current_idea,
                    "history": history,
                    "discard_reason": refinement_prompt.rationale,
                    "discard_round": round_num + 1
                }
            elif decision == "revise":
                # 生成新版本
                try:
                    current_idea = await self.generator.refine_idea(current_idea, refinement_prompt, graph)
                    print(f"    ✨ 生成想法 {idea.id} 的第 {current_idea.version} 版本")
                except Exception as e:
                    return {
                        "status": "failed",
                        "final_idea": current_idea,
                        "history": history,
                        "error": str(e)
                    }
            elif decision == "split":
                # 处理拆分决策
                print(f"    ✂️ 想法 {idea.id} 需要拆分: {refinement_prompt.rationale}")
                return {
                    "status": "split_required",
                    "final_idea": current_idea,
                    "history": history,
                    "split_instructions": refinement_prompt.instructions,
                    "rationale": refinement_prompt.rationale,
                    "split_round": round_num + 1
                }
            elif decision == "merge":
                # 处理合并决策
                print(f"    🔗 想法 {idea.id} 需要合并: {refinement_prompt.rationale}")
                return {
                    "status": "merge_required",
                    "final_idea": current_idea,
                    "history": history,
                    "merge_instructions": refinement_prompt.instructions,
                    "rationale": refinement_prompt.rationale,
                    "merge_round": round_num + 1
                }
            else:
                # 处理未知决策类型
                print(f"    ⚠️ 未支持的决策类型: {decision}")
                return {
                    "status": "unsupported_decision",
                    "final_idea": current_idea,
                    "history": history,
                    "decision": decision,
                    "rationale": refinement_prompt.rationale
                }
        
        # 达到最大轮数
        print(f"    ⏰ 想法 {idea.id} 达到最大轮数 {max_rounds}，迭代结束")
        final_novelty = history[-1]["novelty"].novelty_score if history else 0.0
        final_feasibility = history[-1]["feasibility"].feasibility_score if history else 0.0
        
        return {
            "status": "max_rounds_reached",
            "final_idea": current_idea,
            "history": history,
            "final_scores": {
                "novelty": final_novelty,
                "feasibility": final_feasibility,
                "combined": final_novelty + final_feasibility
            },
            "rounds_completed": len(history)
        }
    
    async def _batch_debate_arena(self, generation_result: Dict[str, Any], 
                                graph: SemanticOpportunityGraph) -> List[Dict[str, Any]]:
        """批量评审模式的辩论竞技场。
        
        实现用户描述的架构：
        1. Generator生成5批，每批10个想法
        2. 每批10个想法一起送给两类评审Agent并行评分
        3. 每个想法再单独经过refiner处理
        """
        idea_batches = generation_result.get('batches', [])
        total_ideas = sum(len(batch) for batch in idea_batches)
        all_results = []
        
        print(f"🏛️ 第三阶段：辩论竞技场（{total_ideas}个想法，{len(idea_batches)}批次评审模式）")
        
        # 按照Generator的原始批次结构处理
        for batch_num, batch_ideas in enumerate(idea_batches, 1):
            print(f"   📦 第{batch_num}批评审：{len(batch_ideas)}个想法（Generator第{batch_num}批）")
            
            try:
                # 真正的批量评审：一次LLM调用评估整批10个想法
                print(f"      🔍 批量新颖性与可行性评审（真正的批量LLM调用）...")
                novelty_task = self.novelty_critic.assess_batch_comprehensive(batch_ideas, graph)
                feasibility_task = self.feasibility_critic.assess_batch_comprehensive(batch_ideas, graph)
                
                novelty_critiques, feasibility_critiques = await asyncio.gather(
                    novelty_task, feasibility_task
                )
                
                print(f"      ✅ 批量评审完成")
                
                # 检查评审结果数量是否匹配
                if len(novelty_critiques) != len(batch_ideas) or len(feasibility_critiques) != len(batch_ideas):
                    print(f"      ⚠️ 评审结果数量不匹配，预期{len(batch_ideas)}个，实际新颖性{len(novelty_critiques)}个，可行性{len(feasibility_critiques)}个")
                    # 为所有想法创建失败结果
                    batch_results = [{
                        "status": "failed",
                        "final_idea": idea,
                        "history": [],
                        "error": "批量评审结果数量不匹配"
                    } for idea in batch_ideas]
                else:
                    # 显示每个想法的评审分数
                    for i, idea in enumerate(batch_ideas):
                        novelty = novelty_critiques[i]
                        feasibility = feasibility_critiques[i]
                        print(f"      💡 想法 {idea.id}: 新颖性{novelty.novelty_score:.1f}, 可行性{feasibility.feasibility_score:.1f}")
                    
                    # 🚀 批量并行精炼处理（替代逐个处理）
                    print(f"      🔄 启动批量精炼处理（{len(batch_ideas)}个想法）...")
                    batch_results = await self._batch_idea_refinement(
                        batch_ideas, novelty_critiques, feasibility_critiques, graph
                    )
                
                all_results.extend(batch_results)
                print(f"      ✅ 第{batch_num}批处理完成（批量精炼模式）")
                
            except Exception as e:
                print(f"      ❌ 第{batch_num}批处理失败: {e}")
                # 为这批的所有想法创建失败结果
                for idea in batch_ideas:
                    all_results.append({
                        "status": "failed",
                        "final_idea": idea,
                        "history": [],
                        "error": f"批量评审失败: {str(e)}"
                    })
        
        return all_results
    
    async def _single_idea_refinement(self, idea: CandidateIdea, 
                                    novelty: 'NoveltyCritique', 
                                    feasibility: 'FeasibilityCritique',
                                    graph: SemanticOpportunityGraph) -> Dict[str, Any]:
        """针对单个想法的精炼处理（已有评审结果）。"""
        current_idea = idea
        history = []
        
        # 初始评审记录
        round_record = {
            "round": 1,
            "idea_version": current_idea.version,
            "novelty": novelty,
            "feasibility": feasibility,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # 生成精炼指令
            refinement_prompt = await self.refiner.make_refinement_prompt(
                current_idea, novelty, feasibility
            )
            round_record["refinement"] = refinement_prompt
            history.append(round_record)
            
            decision = refinement_prompt.decision
            print(f"        🤔 想法 {idea.id} 决策: {decision}")
            
            if decision == "accept":
                return {
                    "status": "accepted",
                    "final_idea": current_idea,
                    "history": history,
                    "final_scores": {
                        "novelty": novelty.novelty_score,
                        "feasibility": feasibility.feasibility_score,
                        "combined": novelty.novelty_score + feasibility.feasibility_score
                    },
                    "acceptance_round": 1
                }
            elif decision == "discard":
                return {
                    "status": "discarded",
                    "final_idea": current_idea,
                    "history": history,
                    "discard_reason": refinement_prompt.rationale,
                    "discard_round": 1
                }
            elif decision == "revise":
                # 对于需要修订的想法，可以进行一轮改进
                try:
                    current_idea = await self.generator.refine_idea(current_idea, refinement_prompt, graph)
                    return {
                        "status": "revised",
                        "final_idea": current_idea,
                        "history": history,
                        "revision_round": 1
                    }
                except Exception as e:
                    return {
                        "status": "failed",
                        "final_idea": current_idea,
                        "history": history,
                        "error": f"修订失败: {str(e)}"
                    }
            elif decision == "split":
                return {
                    "status": "split_required",
                    "final_idea": current_idea,
                    "history": history,
                    "split_instructions": refinement_prompt.instructions,
                    "rationale": refinement_prompt.rationale,
                    "split_round": 1
                }
            elif decision == "merge":
                return {
                    "status": "merge_required",
                    "final_idea": current_idea,
                    "history": history,
                    "merge_instructions": refinement_prompt.instructions,
                    "rationale": refinement_prompt.rationale,
                    "merge_round": 1
                }
            else:
                return {
                    "status": "unsupported_decision",
                    "final_idea": current_idea,
                    "history": history,
                    "decision": decision,
                    "rationale": refinement_prompt.rationale
                }
                
        except Exception as e:
            round_record["error"] = str(e)
            history.append(round_record)
            return {
                "status": "failed",
                "final_idea": current_idea,
                "history": history,
                "error": f"精炼处理失败: {str(e)}"
            }

    async def _batch_idea_refinement(self, batch_ideas: List[CandidateIdea], 
                                   novelty_critiques: List['NoveltyCritique'], 
                                   feasibility_critiques: List['FeasibilityCritique'],
                                   graph: SemanticOpportunityGraph) -> List[Dict[str, Any]]:
        """批量并行处理想法精炼。
        
        输入:
            - batch_ideas: 想法列表。
            - novelty_critiques: 新颖性评审结果列表。
            - feasibility_critiques: 可行性评审结果列表。
            - graph: 语义机会图谱。
            
        输出:
            - List[Dict]: 精炼结果列表。
            
        实现思路:
            1. 批量分析精炼决策
            2. 批量生成精炼指令
            3. 并行处理想法精炼
        """
        print(f"🔄 开始批量想法精炼：{len(batch_ideas)}个想法")
        
        try:
            # 步骤1：准备批量数据
            idea_critique_pairs = list(zip(batch_ideas, novelty_critiques, feasibility_critiques))
            
            # 步骤2：批量分析精炼决策（一次LLM调用）
            print(f"    🤔 批量分析精炼决策...")
            decisions = await self.refiner.make_refinement_decisions_batch(idea_critique_pairs)
            
            # 步骤3：批量生成精炼指令（并行处理）
            print(f"    📝 批量生成精炼指令...")
            refinement_prompts = await self.refiner.make_refinement_prompts_batch(decisions, idea_critique_pairs)
            
            # 步骤4：并行处理想法精炼
            print(f"    🚀 启动并行精炼处理...")
            import asyncio
            
            # 创建并发任务
            refinement_tasks = []
            for i, (idea, decision_info, refinement_prompt) in enumerate(zip(batch_ideas, decisions, refinement_prompts)):
                task = asyncio.create_task(
                    self._process_single_refinement(
                        idea, decision_info, refinement_prompt, 
                        novelty_critiques[i], feasibility_critiques[i], graph
                    )
                )
                refinement_tasks.append(task)
            
            # 等待所有任务完成
            refinement_results = await asyncio.gather(*refinement_tasks, return_exceptions=True)
            
            # 处理结果
            batch_results = []
            for i, result in enumerate(refinement_results):
                if isinstance(result, Exception):
                    print(f"    ❌ 想法 {batch_ideas[i].id} 精炼失败: {str(result)}")
                    batch_results.append({
                        "status": "failed",
                        "final_idea": batch_ideas[i],
                        "history": [],
                        "error": f"并行精炼失败: {str(result)}"
                    })
                else:
                    batch_results.append(result)
                    print(f"    ✅ 想法 {batch_ideas[i].id} 精炼完成: {result.get('status', 'unknown')}")
            
            print(f"✅ 批量精炼完成：{len(batch_results)}个结果")
            return batch_results
            
        except Exception as e:
            print(f"❌ 批量精炼处理失败: {e}")
            # 返回所有想法的失败结果
            return [{
                "status": "failed",
                "final_idea": idea,
                "history": [],
                "error": f"批量精炼失败: {str(e)}"
            } for idea in batch_ideas]

    async def _process_single_refinement(self, idea: CandidateIdea, 
                                       decision_info: Dict[str, Any],
                                       refinement_prompt: 'RefinementPrompt',
                                       novelty: 'NoveltyCritique', 
                                       feasibility: 'FeasibilityCritique',
                                       graph: SemanticOpportunityGraph) -> Dict[str, Any]:
        """处理单个想法的精炼（在批量处理的并行任务中）。"""
        current_idea = idea
        history = []
        # 添加类型检查，确保 decision_info 是字典
        if isinstance(decision_info, dict):
            decision = decision_info.get('decision', 'revise')
        elif isinstance(decision_info, str):
            decision = decision_info  # 如果是字符串，直接使用
        else:
            decision = 'revise'  # 默认值
        
        # 初始评审记录
        round_record = {
            "round": 1,
            "idea_version": current_idea.version,
            "novelty": novelty,
            "feasibility": feasibility,
            "refinement": refinement_prompt,
            "timestamp": datetime.now().isoformat()
        }
        history.append(round_record)
        
        try:
            if decision == "accept":
                return {
                    "status": "accepted",
                    "final_idea": current_idea,
                    "history": history,
                    "final_scores": {
                        "novelty": novelty.novelty_score,
                        "feasibility": feasibility.feasibility_score,
                        "combined": novelty.novelty_score + feasibility.feasibility_score
                    },
                    "acceptance_round": 1
                }
            elif decision == "discard":
                return {
                    "status": "discarded",
                    "final_idea": current_idea,
                    "history": history,
                    "discard_reason": refinement_prompt.rationale,
                    "discard_round": 1
                }
            elif decision == "revise":
                # 对于需要修订的想法，进行一轮改进
                try:
                    current_idea = await self.generator.refine_idea(current_idea, refinement_prompt, graph)
                    return {
                        "status": "revised",
                        "final_idea": current_idea,
                        "history": history,
                        "revision_round": 1
                    }
                except Exception as e:
                    return {
                        "status": "failed",
                        "final_idea": current_idea,
                        "history": history,
                        "error": f"修订失败: {str(e)}"
                    }
            elif decision == "split":
                return {
                    "status": "split_required",
                    "final_idea": current_idea,
                    "history": history,
                    "split_instructions": refinement_prompt.instructions,
                    "rationale": refinement_prompt.rationale,
                    "split_round": 1
                }
            elif decision == "merge":
                return {
                    "status": "merge_required",
                    "final_idea": current_idea,
                    "history": history,
                    "merge_instructions": refinement_prompt.instructions,
                    "rationale": refinement_prompt.rationale,
                    "merge_round": 1
                }
            else:
                return {
                    "status": "unsupported_decision",
                    "final_idea": current_idea,
                    "history": history,
                    "decision": decision,
                    "rationale": refinement_prompt.rationale
                }
                
        except Exception as e:
            return {
                "status": "failed",
                "final_idea": current_idea,
                "history": history,
                "error": f"精炼处理失败: {str(e)}"
            }

    async def _synthesize_final_results(self, graph: SemanticOpportunityGraph, 
                                       initial_candidates: List[CandidateIdea], 
                                       debate_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """整合最终结果报表。
        
        输入:
            - graph: 构建的语义机会图谱。
            - initial_candidates: 初始候选想法。
            - debate_results: 辩论循环结果。
            
        输出:
            - Dict: 完整的系统产出报表。
        """
        # 统计各种结果状态
        accepted_ideas = [r for r in debate_results if isinstance(r, dict) and r.get("status") == "accepted"]
        discarded_ideas = [r for r in debate_results if isinstance(r, dict) and r.get("status") == "discarded"]
        failed_ideas = [r for r in debate_results if isinstance(r, dict) and r.get("status") == "failed"]
        split_ideas = [r for r in debate_results if isinstance(r, dict) and r.get("status") == "split_required"]
        merge_ideas = [r for r in debate_results if isinstance(r, dict) and r.get("status") == "merge_required"]
        max_rounds_ideas = [r for r in debate_results if isinstance(r, dict) and r.get("status") == "max_rounds_reached"]
        other_ideas = [r for r in debate_results if isinstance(r, dict) and r.get("status") not in 
                      ["accepted", "discarded", "failed", "split_required", "merge_required", "max_rounds_reached"]]
        
        return {
            "opportunity_graph": {
                "node_count": graph.number_of_nodes(),
                "edge_count": graph.number_of_edges(),
                "gaps_count": len(graph.graph.get('gaps', [])),
                "provenance": graph.graph.get('provenance', {})
            },
            "initial_candidates": {
                "count": len(initial_candidates),
                "ideas": [{"id": idea.id, "title": idea.title} for idea in initial_candidates]
            },
            "final_ideas": {
                "accepted": {
                    "count": len(accepted_ideas),
                    "ideas": [r["final_idea"] for r in accepted_ideas],
                    "avg_scores": self._calculate_average_scores(accepted_ideas)
                },
                "discarded": {
                    "count": len(discarded_ideas),
                    "ideas": [r["final_idea"] for r in discarded_ideas],
                    "reasons": [r.get("discard_reason", "") for r in discarded_ideas]
                },
                "split_required": {
                    "count": len(split_ideas),
                    "ideas": [r["final_idea"] for r in split_ideas],
                    "instructions": [r.get("split_instructions", []) for r in split_ideas]
                },
                "merge_required": {
                    "count": len(merge_ideas),
                    "ideas": [r["final_idea"] for r in merge_ideas],
                    "instructions": [r.get("merge_instructions", []) for r in merge_ideas]
                },
                "max_rounds_reached": {
                    "count": len(max_rounds_ideas),
                    "ideas": [r["final_idea"] for r in max_rounds_ideas],
                    "final_scores": [r.get("final_scores", {}) for r in max_rounds_ideas]
                },
                "failed": {
                    "count": len(failed_ideas),
                    "errors": [r.get("error", "") for r in failed_ideas]
                },
                "other": {
                    "count": len(other_ideas),
                    "statuses": [r.get("status", "") for r in other_ideas]
                }
            },
            "iteration_history": [r.get("history", []) for r in debate_results if isinstance(r, dict)],
            "statistics": {
                "total_candidates": len(initial_candidates),
                "total_rounds": sum(len(r.get("history", [])) for r in debate_results if isinstance(r, dict)),
                "success_rate": len(accepted_ideas) / len(initial_candidates) if initial_candidates else 0,
                "avg_rounds_per_idea": sum(len(r.get("history", [])) for r in debate_results if isinstance(r, dict)) / len(debate_results) if debate_results else 0
            },
            "timestamp": datetime.now().isoformat()
        }

    def _calculate_average_scores(self, accepted_ideas: List[Dict[str, Any]]) -> Dict[str, float]:
        """计算已接受想法的平均分数。"""
        if not accepted_ideas:
            return {"novelty": 0.0, "feasibility": 0.0, "combined": 0.0}
        
        scores = [idea.get("final_scores", {}) for idea in accepted_ideas]
        novelty_scores = [s.get("novelty", 0) for s in scores]
        feasibility_scores = [s.get("feasibility", 0) for s in scores]
        
        return {
            "novelty": sum(novelty_scores) / len(novelty_scores),
            "feasibility": sum(feasibility_scores) / len(feasibility_scores),
            "combined": sum(novelty_scores) / len(novelty_scores) + sum(feasibility_scores) / len(feasibility_scores)
        }
    
    async def handle_split_merge_operations(self, split_ideas: List[Dict[str, Any]], 
                                          merge_ideas: List[Dict[str, Any]], 
                                          graph: SemanticOpportunityGraph) -> Dict[str, Any]:
        """处理需要拆分和合并的想法（可选的后处理步骤）。
        
        输入:
            - split_ideas: 需要拆分的想法列表。
            - merge_ideas: 需要合并的想法列表。
            - graph: 语义机会图谱。
            
        输出:
            - Dict: 拆分和合并操作的结果。
            
        注意:
            这是一个可选的扩展功能，用于处理复杂的想法重构操作。
            当前版本只提供框架，具体实现可根据需要扩展。
        """
        print(f"🔄 后处理阶段：处理 {len(split_ideas)} 个拆分请求和 {len(merge_ideas)} 个合并请求")
        
        split_results = []
        merge_results = []
        
        # 处理拆分操作
        for split_request in split_ideas:
            idea = split_request["final_idea"]
            instructions = split_request.get("split_instructions", [])
            
            # 这里可以实现具体的拆分逻辑
            # 例如：基于创新点、实验设计等维度拆分想法
            split_results.append({
                "original_idea_id": idea.id,
                "split_method": "instruction_based",
                "sub_ideas_count": 2,  # 默认拆分为2个子想法
                "status": "pending_implementation",
                "instructions": instructions
            })
        
        # 处理合并操作
        for merge_request in merge_ideas:
            idea = merge_request["final_idea"]
            instructions = merge_request.get("merge_instructions", [])
            
            # 这里可以实现具体的合并逻辑
            # 例如：寻找相关想法进行合并
            merge_results.append({
                "original_idea_id": idea.id,
                "merge_method": "instruction_based",
                "merge_candidates": [],  # 可合并的想法候选
                "status": "pending_implementation",
                "instructions": instructions
            })
        
        return {
            "split_operations": {
                "count": len(split_results),
                "results": split_results
            },
            "merge_operations": {
                "count": len(merge_results),
                "results": merge_results
            },
            "status": "framework_ready"  # 表示框架已就绪，等待具体实现
        }


# =========================
# 对外便捷入口（骨架）
# =========================


async def run_idea_generation(
    final_result: Dict[str, Any],
    enriched_outline: Dict[str, Any],
    llm_factory: LLMFactory,
    db: AcademicPaperDatabase,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """便捷入口：一行调用完成想法生成全流程。

    输入:
        - final_result: Survey Gen 产出字典。
        - enriched_outline: 丰富大纲字典。
        - llm_factory: 统一LLM工厂。
        - db: 学术数据库。
        - config: 运行时配置。

    输出:
        - Dict: 结构化产物（见协调器说明）。

    实现步骤建议:
        1) 实例化 `IdeaGenCoordinator`。
        2) 调用 `run_pipeline` 并返回结果。
        
    使用示例:
        ```python
        # 假设已有 Survey Gen 的产物
        result = await run_idea_generation(
            final_result=survey_result,
            enriched_outline=outline_data,
            llm_factory=llm_factory,
            db=database,
            config={"idea_concurrency": 6}
        )
        
        print(f"生成了 {result['final_ideas']['accepted']['count']} 个被接受的想法")
        for idea in result['final_ideas']['accepted']['ideas']:
            print(f"- {idea.title}")
        ```
    """
    print("🚀 启动 Idea Generation 多智能体系统")
    
    # 创建协调器
    coordinator = IdeaGenCoordinator(llm_factory, db, config)
    
    # 执行完整流水线
    start_time = datetime.now()
    try:
        result = await coordinator.run_pipeline(final_result, enriched_outline)
        
        duration = (datetime.now() - start_time).total_seconds()
        result["execution_time_seconds"] = duration
        
        print(f"✅ Idea Generation 完成，耗时 {duration:.2f} 秒")
        print(f"📊 结果统计：")
        print(f"   - 机会图谱: {result['opportunity_graph']['node_count']} 节点, {result['opportunity_graph']['edge_count']} 边")
        print(f"   - 初始候选: {result['initial_candidates']['count']} 个")
        print(f"   - 最终接受: {result['final_ideas']['accepted']['count']} 个")
        print(f"   - 成功率: {result['statistics']['success_rate']:.1%}")
        
        return result
        
    except Exception as e:
        print(f"❌ Idea Generation 执行失败: {str(e)}")
        return {
            "status": "failed",
            "error": str(e),
            "execution_time_seconds": (datetime.now() - start_time).total_seconds(),
            "timestamp": datetime.now().isoformat()
        }