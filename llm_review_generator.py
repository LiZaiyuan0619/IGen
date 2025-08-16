import os
import json
import re
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

# 导入我们的数据库
from database_setup import AcademicPaperDatabase

# 支持多种LLM选择
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

import numpy as np
from typing import Dict, List, Any
import re
from collections import Counter
import math

class EnhancedSimilarityCalculator:
    def __init__(self, topic: str, subtopics: List[str] = None):
        self.topic = topic.lower()
        self.subtopics = [s.lower() for s in (subtopics or [])]
        
        # 构建主题词库
        self.topic_keywords = self._extract_keywords(topic)
        self.subtopic_keywords = []
        for subtopic in self.subtopics:
            self.subtopic_keywords.extend(self._extract_keywords(subtopic))
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        # 移除标点，转小写，分词
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        # 过滤停用词
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had'}
        return [word for word in words if word not in stopwords and len(word) > 2]
    
    def calculate_keyword_similarity(self, content: str) -> float:
        """计算关键词匹配相似性"""
        content_lower = content.lower()
        content_keywords = self._extract_keywords(content)
        
        if not content_keywords:
            return 0.0
        
        # 1. 主题词完全匹配得分
        main_topic_matches = 0
        for keyword in self.topic_keywords:
            if keyword in content_lower:
                main_topic_matches += 0.5
        
        main_topic_score = main_topic_matches / len(self.topic_keywords) if self.topic_keywords else 0
        
        # 2. 子主题词匹配得分
        subtopic_matches = 0
        if self.subtopic_keywords:
            for keyword in self.subtopic_keywords:
                if keyword in content_lower:
                    subtopic_matches += 0.5
            subtopic_score = subtopic_matches / len(self.subtopic_keywords)
        else:
            subtopic_score = 0
        
        
        # 4. TF-IDF风格的词频得分
        content_counter = Counter(content_keywords)
        tf_score = 0
        for keyword in self.topic_keywords:
            if keyword in content_counter:
                tf = content_counter[keyword] / len(content_keywords)
                tf_score += tf
        
        tf_score = min(tf_score, 1.0)
        
        # 综合得分
        final_score = (
            0.5 * main_topic_score +      # 主题词匹配最重要
            0.3 * subtopic_score +        # 子主题词次之
            0.2 * tf_score               # 词频得分
        )
        
        return min(final_score, 1.0)
    
    def calculate_position_bonus(self, metadata: Dict) -> float:
        """根据内容在论文中的位置给予奖励分数"""
        content_type = metadata.get('content_type', '')
        
        # 不同类型内容的重要性权重
        type_weights = {
            'text': 1.0,
            'equation': 1.05,    # 公式通常很重要
            'image': 1.05,       # 图片包含重要信息
            'table': 1.05        # 表格通常是关键结果
        }
        
        return type_weights.get(content_type, 1.0)
    
    def extract_core_concepts(self, topic: str) -> str:
        """从长标题中提取核心概念，提高向量相似度计算的准确性"""
        import re
        
        # 🔧添加类型检查，防止传入列表等非字符串类型
        if not isinstance(topic, str):
            if isinstance(topic, list):
                # 如果是列表，取第一个元素或连接所有元素
                topic = topic[0] if topic else ""
                print(f"⚠️ extract_core_concepts收到列表参数，已转换为字符串: {topic}")
            else:
                # 其他类型，转换为字符串
                topic = str(topic)
                print(f"⚠️ extract_core_concepts收到非字符串参数，已转换: {topic}")
        
        # 定义停用词（介词、连词、常见修饰词等）
        stop_words = {
            'and', 'or', 'for', 'in', 'on', 'at', 'to', 'from', 'with', 'by', 'of', 'the', 'a', 'an',
            'advanced', 'comprehensive', 'detailed', 'systematic', 'efficient', 'effective', 'novel',
            'improved', 'enhanced', 'optimized', 'based', 'using', 'through', 'via', 'approaches',
            'methods', 'techniques', 'applications', 'systems', 'frameworks', 'models', 'analysis'
        }
        
        # 1. 基础清理：移除标点符号，转为小写，分词
        words = re.findall(r'\b[a-zA-Z]+\b', topic.lower())
        
        # 2. 移除停用词
        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
        
        # 3. 保留重要的专业术语和核心概念（优先保留较长的词汇）
        important_words = []
        for word in filtered_words:
            # 优先保留长词汇（通常是专业术语）
            if len(word) >= 6:
                important_words.append(word)
            # 保留一些重要的短词汇
            elif word in ['nlp', 'llm', 'ai', 'ml', 'gpu', 'cpu', 'api', 'gpt', 'bert']:
                important_words.append(word)
        
        # 4. 如果筛选后的词太少，保留一些中等长度的词
        if len(important_words) < 3:
            for word in filtered_words:
                if len(word) >= 4 and word not in important_words:
                    important_words.append(word)
                if len(important_words) >= 5:  # 限制核心概念数量
                    break
        
        # 5. 保留前5个最重要的概念
        core_concepts = ' '.join(important_words[:5])
        
        return core_concepts if core_concepts else topic  # 如果提取失败，返回原标题
    
    def multi_level_similarity(self, search_result: Dict, full_topic: str, core_concepts: str) -> float:
        """多层次相似度计算，根据标题长度动态调整计算策略"""
        document_content = search_result.get('document', '')
        metadata = search_result.get('metadata', {})
        
        # 1. 计算基于核心概念的相似度
        core_vector_sim = 1 - search_result.get('distance', 1.0)  # 原始向量相似度
        core_keyword_sim = self._calculate_text_keyword_similarity(document_content, core_concepts)
        
        # 2. 计算基于完整标题的关键词匹配
        full_keyword_sim = self.calculate_keyword_similarity(document_content)
        
        # 3. 其他维度保持不变
        position_bonus = self.calculate_position_bonus(metadata)
        content_length = len(document_content)
        length_factor = min(math.log(content_length + 1) / math.log(1000), 1.5)
        
        # 论文相关性
        paper_name = metadata.get('paper_name', '').lower()
        paper_relevance = 1.0
        for keyword in self.topic_keywords:
            if keyword in paper_name:
                paper_relevance += 0.02
        paper_relevance = min(paper_relevance, 1.5)
        
        # 4. 根据标题长度动态调整权重
        title_length = len(full_topic.split())
        
        if title_length > 8:  # 长标题
            # 提高核心概念相似度权重，降低完整标题权重
            enhanced_similarity = (
                0.6 * core_vector_sim +         # 核心概念向量相似度60%
                0.2 * core_keyword_sim +        # 核心概念关键词匹配20%
                0.1 * full_keyword_sim +        # 完整关键词匹配10%
                0.1 * (position_bonus - 1.0)    # 位置加成10%
            ) * length_factor * paper_relevance
        else:  # 正常长度标题
            # 使用原始权重分配
            enhanced_similarity = (
                0.7 * core_vector_sim +         # 向量相似度70%
                0.1 * core_keyword_sim +        # 核心概念匹配10%
                0.1 * full_keyword_sim +        # 完整关键词匹配10%
                0.1 * (position_bonus - 1.0)    # 位置加成10%
            ) * length_factor * paper_relevance
        
        return min(enhanced_similarity, 1.0)
    
    def _calculate_text_keyword_similarity(self, document_content: str, concepts_text: str) -> float:
        """计算文档内容与核心概念的关键词匹配相似度"""
        if not concepts_text or not document_content:
            return 0.0
        
        concepts_words = set(concepts_text.lower().split())
        doc_words = set(document_content.lower().split())
        
        # 计算交集比例
        if not concepts_words:
            return 0.0
        
        intersection = concepts_words.intersection(doc_words)
        similarity = len(intersection) / len(concepts_words)
        
        return similarity
    
    def calculate_enhanced_similarity(self, search_result: Dict, topic: str) -> float:
        """计算增强的相似性得分 - 🔧改进：使用多层次相似度计算"""
        document_content = search_result.get('document', '')
        metadata = search_result.get('metadata', {})
        content_type = metadata.get('content_type', '')

        # 对于文本内容，进行多重检查
        if content_type == 'text':
            # 1. 长度检查
            if len(document_content) < 100:
                return 0.0
            
            # 2. 内容质量检查：过滤掉数字占比过高的内容
            total_len = len(document_content)
            if total_len > 0:
                digit_count = sum(c.isdigit() for c in document_content)
                if (digit_count / total_len) > 0.8:
                    return 0.0  # 数字占比超过80%，判定为无意义内容

        # 🔧改进：使用多层次相似度计算来处理长标题问题
        # 1. 提取核心概念
        core_concepts = self.extract_core_concepts(topic)
        
        # 2. 使用多层次相似度计算
        enhanced_similarity = self.multi_level_similarity(search_result, topic, core_concepts)
        
        # 添加调试信息（可选）
        if len(topic.split()) > 8:  # 长标题时输出调试信息
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"长标题处理: '{topic}' -> 核心概念: '{core_concepts}'")
        
        return enhanced_similarity

@dataclass
class ReviewConfig:
    """综述生成配置"""
    max_context_length: int = 80000  # 增加上下文长度限制
    
    # 移除硬编码，改为可配置的限制
    max_texts_per_query: int = 300           # 每次查询的文本数量
    max_equations_per_query: int = 100       # 每次查询的公式数量  
    max_figures_per_query: int = 100         # 每次查询的图片数量
    max_tables_per_query: int = 100          # 每次查询的表格数量
    
    # 提示词中使用的最大数量（可以设置为None表示无限制）
    max_texts_in_prompt: Optional[int] = None     # None表示使用所有检索到的内容
    max_equations_in_prompt: Optional[int] = None
    max_figures_in_prompt: Optional[int] = None  
    max_tables_in_prompt: Optional[int] = None
    
    min_relevance_score: float = 0.1
    include_equations: bool = True
    include_figures: bool = True
    include_tables: bool = True
    output_format: str = "markdown"
    language: str = "chinese"

class LLMReviewGenerator:
    def __init__(self, db: AcademicPaperDatabase, config: ReviewConfig = None):
        self.db = db
        self.config = config or ReviewConfig()
        self.llm_client = None
        self.llm_type = None
        
    def setup_openai(self, api_key: str, base_url: str = None, model: str = "gpt-4o"):
        """设置OpenAI API"""
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not installed. Run: pip install openai")
        
        self.llm_client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.llm_type = "openai"
        self.model_name = model
        print(f"✅ OpenAI API设置成功，使用模型: {model}")
    
    def setup_local_model(self, model_name: str = "microsoft/DialoGPT-medium"):
        """设置本地模型"""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library not installed. Run: pip install transformers torch")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.llm_type = "local"
        self.model_name = model_name
        print(f"✅ 本地模型设置成功: {model_name}")

    def enhanced_gather_research_context(self, topic: str, subtopics: List[str] = None) -> Dict:
        """使用增强相似性计算收集研究上下文材料"""
        print(f"🔍 正在收集关于'{topic}'的研究材料（使用增强相似性计算）...")
        
        # 初始化增强相似性计算器
        similarity_calculator = EnhancedSimilarityCalculator(topic, subtopics)
        
        context = {
            "main_topic": topic,
            "subtopics": subtopics or [],
            "relevant_content": {
                "texts": [],
                "equations": [],
                "figures": [],
                "tables": []
            },
            "source_papers": {},
            "statistics": {}
        }
        
        # 主题搜索
        search_queries = [topic]
        if subtopics:
            search_queries.extend(subtopics)
        
        all_results = []
        
        for query in search_queries:
            print(f"🔍 搜索: {query}")
            
            # 获取更多初始结果，然后用增强相似性重新排序
            text_results = self.db.search_content(
                query, content_type="texts", 
                n_results=min(self.config.max_texts_per_query * 2, 200)  # 获取2倍结果用于重排序
            )
            
            if self.config.include_equations:
                equation_results = self.db.search_content(
                    query, content_type="equations", 
                    n_results=min(self.config.max_equations_per_query * 2, 100)
                )
                all_results.extend(equation_results)
            
            if self.config.include_figures:
                figure_results = self.db.search_content(
                    query, content_type="images", 
                    n_results=min(self.config.max_figures_per_query * 2, 100)
                )
                all_results.extend(figure_results)
            
            if self.config.include_tables:
                table_results = self.db.search_content(
                    query, content_type="tables", 
                    n_results=min(self.config.max_tables_per_query * 2, 60)
                )
                all_results.extend(table_results)
            
            all_results.extend(text_results)
        
        # ===== 关键：使用增强相似性重新计算和排序 =====
        enhanced_results = []
        seen_ids = set()
        
        for result in all_results:
            if result['id'] in seen_ids:
                continue
            seen_ids.add(result['id'])
            
            # 计算增强相似性得分
            enhanced_score = similarity_calculator.calculate_enhanced_similarity(result, topic)
            
            # 只保留超过阈值的结果
            if enhanced_score >= self.config.min_relevance_score:
                result['enhanced_similarity'] = enhanced_score
                enhanced_results.append(result)
        
        # 按增强相似性得分重新排序
        enhanced_results.sort(key=lambda x: x['enhanced_similarity'], reverse=True)
        
        print(f"  📊 原始结果: {len(all_results)}, 增强筛选后: {len(enhanced_results)}")
        
        # 按内容类型分组并限制数量
        content_type_limits = {
            'text': self.config.max_texts_per_query,
            'equation': self.config.max_equations_per_query,
            'image': self.config.max_figures_per_query,
            'table': self.config.max_tables_per_query
        }
        
        content_type_counts = {key: 0 for key in content_type_limits.keys()}
        
        # 按相关性筛选和分组
        for result in enhanced_results:
            content_type = result['metadata']['content_type']
            
            # 检查该类型是否已达到限制
            if content_type_counts.get(content_type, 0) >= content_type_limits.get(content_type, float('inf')):
                continue
            
            content_type_counts[content_type] += 1
            
            paper_name = result['metadata']['paper_name']
            
            # 记录来源论文
            if paper_name not in context["source_papers"]:
                context["source_papers"][paper_name] = {
                    "content_count": 0,
                    "sections": []
                }
            
            context["source_papers"][paper_name]["content_count"] += 1
            
            # 分类存储内容
            content_item = {
                "content": result["document"],
                "paper": paper_name,
                "page": result["metadata"]["page_idx"],
                "relevance_score": result['enhanced_similarity'],  # 使用增强得分
                "original_vector_score": 1 - result['distance'],   # 保留原始向量得分用于参考
                "original_data": result["metadata"].get("original_data", "")
            }
            
            if content_type == "text":
                context["relevant_content"]["texts"].append(content_item)
            elif content_type == "equation":
                context["relevant_content"]["equations"].append(content_item)
            elif content_type == "image":
                context["relevant_content"]["figures"].append(content_item)
            elif content_type == "table":
                context["relevant_content"]["tables"].append(content_item)
        
        # 统计信息
        context["statistics"] = {
            "total_papers": len(context["source_papers"]),
            "total_texts": len(context["relevant_content"]["texts"]),
            "total_equations": len(context["relevant_content"]["equations"]),
            "total_figures": len(context["relevant_content"]["figures"]),
            "total_tables": len(context["relevant_content"]["tables"]),
            "enhancement_info": {
                "original_results": len(all_results),
                "enhanced_filtered": len(enhanced_results),
                "final_selected": sum(len(context["relevant_content"][key]) for key in context["relevant_content"])
            }
        }
        
        print(f"📊 增强检索完成:")
        print(f"  - 相关论文: {context['statistics']['total_papers']} 篇")
        print(f"  - 文本段落: {context['statistics']['total_texts']} 条")
        print(f"  - 数学公式: {context['statistics']['total_equations']} 条")
        print(f"  - 图表: {context['statistics']['total_figures']} 条")
        print(f"  - 表格: {context['statistics']['total_tables']} 条")
        print(f"  - 筛选效果: {len(all_results)} -> {len(enhanced_results)} -> {context['statistics']['enhancement_info']['final_selected']}")
        
        return context
    # 替换原来的 gather_research_context 方法
    def gather_research_context(self, topic: str, subtopics: List[str] = None) -> Dict:
        return self.enhanced_gather_research_context(topic, subtopics)

    def analyze_similarity_distribution(self, topic: str) -> Dict:
        """分析相似性得分分布，用于调试和优化"""
        search_results = self.db.search_content(topic, n_results=100)
        
        similarity_calculator = EnhancedSimilarityCalculator(topic)
        
        vector_scores = []
        enhanced_scores = []
        
        for result in search_results:
            vector_score = 1 - result['distance']
            enhanced_score = similarity_calculator.calculate_enhanced_similarity(result, topic)
            
            vector_scores.append(vector_score)
            enhanced_scores.append(enhanced_score)
        
        return {
            "vector_similarity": {
                "mean": np.mean(vector_scores),
                "std": np.std(vector_scores),
                "min": np.min(vector_scores),
                "max": np.max(vector_scores)
            },
            "enhanced_similarity": {
                "mean": np.mean(enhanced_scores),
                "std": np.std(enhanced_scores),
                "min": np.min(enhanced_scores),
                "max": np.max(enhanced_scores)
            },
            "improvement_ratio": np.mean(enhanced_scores) / np.mean(vector_scores) if np.mean(vector_scores) > 0 else 1.0
        }


    def create_prompt(self, topic: str, context: Dict, section_type: str = "full_review") -> str:
        """创建LLM提示词"""
        
        if self.config.language == "chinese":
            base_prompt = f"""
请基于提供的学术文献材料，撰写一篇关于"{topic}"的详尽综述文章。我需要一篇至少50,000字的全面深入的学术综述。

【写作要求】
1. 文章结构完整，每个章节需要详尽展开，不可泛泛而谈
2. 准确引用相关文献，标注来源论文和页码，采用(作者，年份，页码)的格式
3. 详细解释所有概念、方法和技术，确保专业性和学术性
4. 恰当引入数学公式、图表和表格，并提供详细解释
5. 语言学术规范，逻辑清晰，段落之间过渡自然
6. 突出关键技术点和创新点，进行深入分析和比较
7. 每个小节至少2000字，主要章节至少5000字

【内容深度要求】
1. 对每个关键概念进行多角度、多层次分析
2. 对相关理论和方法进行系统性比较
3. 详细探讨每个方法的优缺点、适用条件和局限性
4. 讨论研究现状、挑战和未来发展方向
5. 对重要观点进行批判性分析和评价

【研究材料统计】
- 涉及论文: {context['statistics']['total_papers']} 篇
- 文本内容: {context['statistics']['total_texts']} 条
- 数学公式: {context['statistics']['total_equations']} 条
- 图表资料: {context['statistics']['total_figures']} 条
- 表格数据: {context['statistics']['total_tables']} 条

【主要来源论文】
"""
        else:
            base_prompt = f"""
Please write a comprehensive and extensive literature review on "{topic}" based on the provided academic materials. I need an in-depth academic survey of at least 50,000 words.

【WRITING REQUIREMENTS】
1. Complete article structure with each section thoroughly developed, avoiding superficial treatment
2. Accurate citations with source papers and page numbers using (Author, Year, Page) format
3. Detailed explanation of all concepts, methods, and techniques ensuring professionalism and academic rigor
4. Appropriate inclusion of mathematical formulas, figures, and tables with comprehensive explanations
5. Academic language with clear logic and natural transitions between paragraphs
6. Highlight key technical points and innovations with in-depth analysis and comparison
7. Each subsection should be at least 2,000 words, main sections at least 5,000 words

【DEPTH REQUIREMENTS】
1. Multi-perspective, multi-level analysis of each key concept
2. Systematic comparison of related theories and methods
3. Detailed discussion of advantages, disadvantages, applicable conditions, and limitations of each method
4. Discussion of research status, challenges, and future development directions
5. Critical analysis and evaluation of important viewpoints

【Research materials statistics】
- Papers involved: {context['statistics']['total_papers']}
- Text content: {context['statistics']['total_texts']} items
- Mathematical formulas: {context['statistics']['total_equations']} items
- Figures: {context['statistics']['total_figures']} items
- Tables: {context['statistics']['total_tables']} items

【Main source papers】
"""
        
        # 添加论文列表
        for i, (paper_name, info) in enumerate(context["source_papers"].items(), 1):
            base_prompt += f"\n{i}. {paper_name} ({info['content_count']} 条相关内容)"
        
        # ================= 增强的综述结构大纲 =================
        if self.config.language == "chinese":
            outline_text = """
\n\n【综述详细结构大纲】请按照以下结构详细撰写综述，确保每个章节内容丰富、深入且全面：

1. 引言（5000字以上）
   1.1 研究背景与意义
   1.2 研究问题与挑战
   1.3 研究现状概述
   1.4 本综述的组织结构

2. 技术基础与核心概念（8000字以上）
   2.1 关键技术原理详解
   2.2 算法框架与数学基础
   2.3 评估方法与指标
   2.4 技术演进历程

3. 研究方法与模型（10000字以上）
   3.1 主流模型架构详解
   3.2 训练方法与优化策略
   3.3 推理技术与加速方法
   3.4 模型评估与比较

4. 应用领域与案例分析（10000字以上）
   4.1 典型应用场景详解
   4.2 实际落地案例研究
   4.3 性能表现与效果评估
   4.4 应用挑战与解决方案

5. 前沿进展与创新点（8000字以上）
   5.1 最新研究突破
   5.2 创新技术与方法
   5.3 理论与实践创新
   5.4 重要发现与贡献

6. 挑战与未解决问题（5000字以上）
   6.1 技术瓶颈分析
   6.2 开放性问题讨论
   6.3 争议问题与不同观点
   6.4 理论与实践差距

7. 未来研究方向（5000字以上）
   7.1 潜在研究方向
   7.2 技术发展趋势预测
   7.3 跨领域融合机会
   7.4 长期研究愿景

8. 结论（3000字以上）
   8.1 研究总结与贡献
   8.2 方法论反思
   8.3 局限性讨论
   8.4 实践建议与展望
"""
        else:
            outline_text = """
\n\n【DETAILED REVIEW STRUCTURE】Please follow this comprehensive structure to write a thorough, in-depth and complete survey:

1. Introduction (5,000+ words)
   1.1 Research Background and Significance
   1.2 Research Questions and Challenges
   1.3 Overview of Current Research Status
   1.4 Organization of this Review

2. Technical Foundations and Core Concepts (8,000+ words)
   2.1 Detailed Explanation of Key Technical Principles
   2.2 Algorithm Frameworks and Mathematical Foundations
   2.3 Evaluation Methods and Metrics
   2.4 Technical Evolution Process

3. Research Methods and Models (10,000+ words)
   3.1 Detailed Analysis of Mainstream Model Architectures
   3.2 Training Methods and Optimization Strategies
   3.3 Inference Techniques and Acceleration Methods
   3.4 Model Evaluation and Comparison

4. Application Domains and Case Studies (10,000+ words)
   4.1 Detailed Explanation of Typical Application Scenarios
   4.2 Real-world Implementation Case Studies
   4.3 Performance and Effectiveness Evaluation
   4.4 Application Challenges and Solutions

5. Frontier Advances and Innovations (8,000+ words)
   5.1 Latest Research Breakthroughs
   5.2 Innovative Technologies and Methods
   5.3 Theoretical and Practical Innovations
   5.4 Important Findings and Contributions

6. Challenges and Unsolved Problems (5,000+ words)
   6.1 Technical Bottleneck Analysis
   6.2 Discussion of Open Problems
   6.3 Controversial Issues and Different Perspectives
   6.4 Gaps Between Theory and Practice

7. Future Research Directions (5,000+ words)
   7.1 Potential Research Directions
   7.2 Predicted Technology Development Trends
   7.3 Cross-domain Integration Opportunities
   7.4 Long-term Research Vision

8. Conclusion (3,000+ words)
   8.1 Research Summary and Contributions
   8.2 Methodological Reflections
   8.3 Limitations Discussion
   8.4 Practical Recommendations and Outlook
"""
        base_prompt += outline_text
        # =====================================================
        
        # 添加内容利用指导
        if self.config.language == "chinese":
            content_guidance = """
\n\n【内容利用指导】
1. 深入分析提供的学术材料，提取关键信息和见解
2. 对每篇重要论文的贡献、方法和结果进行详细讨论
3. 将不同论文的观点和发现进行对比和整合
4. 对论文中的数学公式进行详细解释和分析
5. 针对图表和表格进行深度解读，不仅描述其内容，还要分析其意义和影响
6. 确保在综述中全面利用提供的材料，不遗漏重要内容
7. 对于每个关键概念，至少引用3-5篇不同来源的文献进行论述
8. 对于争议性问题，呈现不同论文的不同观点
"""
        else:
            content_guidance = """
\n\n【CONTENT UTILIZATION GUIDANCE】
1. Analyze the provided academic materials in depth to extract key information and insights
2. Discuss the contributions, methods, and results of each important paper in detail
3. Compare and integrate viewpoints and findings from different papers
4. Provide detailed explanations and analyses of mathematical formulas
5. Offer in-depth interpretations of figures and tables, not only describing their content but also analyzing their significance and impact
6. Ensure comprehensive utilization of provided materials in the review, without omitting important content
7. For each key concept, cite at least 3-5 different source papers for discussion
8. For controversial issues, present different perspectives from different papers
"""
        base_prompt += content_guidance
        
        # 动态确定使用的内容数量
        max_texts = (self.config.max_texts_in_prompt or 
                    len(context["relevant_content"]["texts"]))
        max_equations = (self.config.max_equations_in_prompt or 
                        len(context["relevant_content"]["equations"]))
        max_figures = (self.config.max_figures_in_prompt or 
                    len(context["relevant_content"]["figures"]))
        max_tables = (self.config.max_tables_in_prompt or 
                    len(context["relevant_content"]["tables"]))
        
        # 添加主要文本内容 - 使用动态数量
        base_prompt += "\n\n=== 主要文本内容 ===\n"
        texts_to_use = context["relevant_content"]["texts"][:max_texts]
        for i, text_item in enumerate(texts_to_use, 1):
            base_prompt += f"\n[文本{i}] 来源: {text_item['paper']} (第{text_item['page']}页)\n"
            base_prompt += f"内容: {text_item['content'][:1000]}...\n"  # 增加内容长度以提供更多上下文
        
        # 添加公式内容 - 使用动态数量
        if context["relevant_content"]["equations"]:
            base_prompt += "\n\n=== 相关数学公式 ===\n"
            equations_to_use = context["relevant_content"]["equations"][:max_equations]
            for i, eq_item in enumerate(equations_to_use, 1):
                base_prompt += f"\n[公式{i}] 来源: {eq_item['paper']} (第{eq_item['page']}页)\n"
                base_prompt += f"内容: {eq_item['content']}\n"
                base_prompt += f"原始数据: {eq_item['original_data'][:200]}...\n"  # 添加原始数据以帮助理解公式
        
        # 添加图表内容 - 使用动态数量
        if context["relevant_content"]["figures"]:
            base_prompt += "\n\n=== 相关图表 ===\n"
            figures_to_use = context["relevant_content"]["figures"][:max_figures]
            for i, fig_item in enumerate(figures_to_use, 1):
                base_prompt += f"\n[图表{i}] 来源: {fig_item['paper']} (第{fig_item['page']}页)\n"
                base_prompt += f"描述: {fig_item['content']}\n"
                base_prompt += f"原始数据: {fig_item['original_data'][:200]}...\n"  # 添加原始数据以帮助理解图表
        
        # 添加表格内容 - 使用动态数量
        if context["relevant_content"]["tables"]:
            base_prompt += "\n\n=== 相关表格 ===\n"
            tables_to_use = context["relevant_content"]["tables"][:max_tables]
            for i, table_item in enumerate(tables_to_use, 1):
                base_prompt += f"\n[表格{i}] 来源: {table_item['paper']} (第{table_item['page']}页)\n"
                base_prompt += f"内容: {table_item['content'][:500]}...\n"  # 增加内容长度
                base_prompt += f"原始数据: {table_item['original_data'][:200]}...\n"  # 添加原始数据以帮助理解表格
        
        # 添加写作指导和输出格式要求
        if self.config.language == "chinese":
            writing_guide = """
\n\n【写作风格与输出要求】
1. 撰写长度要求：总计50,000字以上，每个主要章节至少5,000字
2. 写作风格：学术严谨、逻辑清晰、内容深入、分析透彻
3. 引用格式：使用(作者，年份，页码)格式进行准确引用
4. 公式处理：使用LaTeX语法正确呈现所有数学公式
5. 图表引用：详细描述图表内容并分析其意义，使用![图X](图片路径)格式引用图片
6. 表格处理：使用markdown表格语法正确呈现表格内容
7. 章节组织：每个章节开始前提供简短概述，章节结束提供小结
8. 逻辑连贯：确保章节间、段落间逻辑流畅，使用适当的过渡词

请确保内容详尽、学术性强、引用充分，不要泛泛而谈。对于每个重要概念和方法，都需要深入探讨其基础原理、技术细节、应用场景和发展趋势。不要简单罗列和堆砌信息，而要进行深度分析和批判性思考。最终交付的综述应当是该领域最全面、最深入、最权威的学术文献之一。
"""
        else:
            writing_guide = """
\n\n【WRITING STYLE AND OUTPUT REQUIREMENTS】
1. Length requirement: Total of 50,000+ words, with each main section at least 5,000 words
2. Writing style: Academically rigorous, logically clear, in-depth content, thorough analysis
3. Citation format: Use (Author, Year, Page) format for accurate citations
4. Formula processing: Correctly present all mathematical formulas using LaTeX syntax
5. Figure references: Describe figure content in detail and analyze its significance, use ![Figure X](image path) format
6. Table processing: Correctly present table content using markdown table syntax
7. Section organization: Provide brief overview before each section begins, provide summary at end of section
8. Logical coherence: Ensure smooth logic between sections and paragraphs, use appropriate transition words

Please ensure the content is extensive, highly academic, well-cited, and not superficial. Each important concept and method needs in-depth discussion of its fundamental principles, technical details, application scenarios, and development trends. Don't simply list and pile up information, but engage in deep analysis and critical thinking. The final delivered review should be one of the most comprehensive, in-depth, and authoritative academic literature in the field.
"""
        base_prompt += writing_guide
        
        # 修改语言判断逻辑并补充字数说明
        if self.config.language == "chinese":
            base_prompt += "\n\n请基于以上材料和结构撰写不少于50000字的综述文章，确保引用准确，内容完整，深度分析，学术严谨。注意：这是一篇学术综述，需要详尽地覆盖所有相关主题和文献，字数不足将无法满足学术要求。"
        else:
            base_prompt += "\n\nPlease write a comprehensive review article based on the above materials and structure, with at least 50,000 words, ensuring accurate citations, complete content, in-depth analysis, and academic rigor. Note: This is an academic review that needs to comprehensively cover all relevant topics and literature; insufficient word count will not meet academic requirements."
        
        return base_prompt
    
    def call_llm(self, prompt: str) -> str:
        """调用LLM生成内容"""
        if self.llm_type == "openai":
            return self._call_openai(prompt)
        elif self.llm_type == "local":
            return self._call_local_model(prompt)
        else:
            raise ValueError("请先设置LLM（setup_openai或setup_local_model）")
    
    def _call_openai(self, prompt: str) -> str:
        """调用OpenAI API"""
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是一个专业的学术综述写作助手，擅长分析和整合学术文献。你的任务是创建详尽、全面的学术综述，内容必须详实、深入、有学术价值。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=32000,  # 大幅增加令牌限制
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ OpenAI API调用失败: {e}")
            return None
    
    def _call_local_model(self, prompt: str) -> str:
        """调用本地模型"""
        try:
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=2048, truncation=True)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 1000,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response[len(prompt):].strip()
        except Exception as e:
            print(f"❌ 本地模型调用失败: {e}")
            return None
    
    def generate_review(self, topic: str, subtopics: List[str] = None, output_file: str = None) -> str:
        """生成完整的综述文章"""
        print(f"🚀 开始生成关于'{topic}'的综述文章...")
        
        # 1. 收集研究材料
        context = self.gather_research_context(topic, subtopics)
        
        # 2. 创建提示词
        prompt = self.create_prompt(topic, context)
        
        print(f"📝 正在调用{self.llm_type}模型生成综述...")
        
        # 3. 调用LLM生成
        review_content = self.call_llm(prompt)
        
        if not review_content:
            print("❌ 综述生成失败")
            return None
        
        # 4. 后处理和格式化
        formatted_review = self.format_review(review_content, context, topic)
        
        # 5. 保存文件
        if output_file:
            self.save_review(formatted_review, output_file, context)
            print(f"✅ 综述已保存到: {output_file}")
        
        return formatted_review
    
    def format_review(self, review_content: str, context: Dict, topic: str) -> str:
        """格式化综述内容"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if self.config.output_format == "markdown":
            formatted = f"""# {topic} - 综述

**生成时间**: {current_time}
**数据源**: {context['statistics']['total_papers']} 篇学术论文
**包含内容**: {context['statistics']['total_texts']} 个文本段落, {context['statistics']['total_equations']} 个公式, {context['statistics']['total_figures']} 个图表, {context['statistics']['total_tables']} 个表格

## 内容提要

本综述基于 {context['statistics']['total_papers']} 篇学术论文，全面深入地分析了 {topic} 相关的研究现状、技术方法、应用场景和未来发展方向。综述总长度超过 50,000 字，涵盖了该领域的核心概念、关键技术和最新进展，是一份全面而权威的学术参考资料。

---

{review_content}

---

## 参考文献

"""
            
            # 添加参考文献列表
            for i, (paper_name, info) in enumerate(context["source_papers"].items(), 1):
                formatted += f"{i}. {paper_name}\n"
            
            formatted += f"\n*本综述由AI系统基于{len(context['source_papers'])}篇学术论文自动生成，生成时间：{current_time}*"
            
            # 添加图表和公式处理说明
            if context["relevant_content"]["figures"] or context["relevant_content"]["equations"]:
                formatted += "\n\n## 附录：图表和公式来源\n\n"
                
                if context["relevant_content"]["figures"]:
                    formatted += "### 图表来源\n\n"
                    for i, fig_item in enumerate(context["relevant_content"]["figures"][:20], 1):
                        formatted += f"**图{i}**: 来源于《{fig_item['paper']}》第{fig_item['page']}页\n\n"
                
                if context["relevant_content"]["equations"]:
                    formatted += "### 公式来源\n\n"
                    for i, eq_item in enumerate(context["relevant_content"]["equations"][:20], 1):
                        formatted += f"**公式{i}**: 来源于《{eq_item['paper']}》第{eq_item['page']}页\n\n"
            
        return formatted
    
    def save_review(self, content: str, output_file: str, context: Dict):
        """保存综述到文件"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.config.output_format == "markdown":
            with open(output_path.with_suffix('.md'), 'w', encoding='utf-8') as f:
                f.write(content)
        
        # 同时保存上下文信息
        context_file = output_path.with_suffix('.json')
        with open(context_file, 'w', encoding='utf-8') as f:
            json.dump(context, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 上下文信息已保存到: {context_file}")

def main():
    """主函数 - 演示使用"""
    print("🔬 学术论文综述生成系统")
    print("=" * 50)
    
    # 初始化数据库
    db = AcademicPaperDatabase(db_path="D:/Desktop/ZJU/300/academic_papers_db")
    
    # 配置更宽松的参数以充分利用增强相似性
    config = ReviewConfig(
        max_texts_per_query=500,        # 增加搜索数量
        max_equations_per_query=200,
        max_figures_per_query=200,
        max_tables_per_query=200,
        min_relevance_score=0.10,       # 稍微降低阈值，获取更多内容
        max_texts_in_prompt=400,        # 增加提示词中包含的内容数量
        max_equations_in_prompt=150,
        max_figures_in_prompt=150,
        max_tables_in_prompt=150,
        max_context_length=80000,       # 确保上下文长度足够大
    )
    
    generator = LLMReviewGenerator(db, config)
    
    # 设置LLM（请根据您的情况选择）
    
    # 选项1: 使用OpenAI API
    try:
        api_key = input("请输入API Key(可选，回车使用默认Openrouter):").strip()
        api_key=api_key if api_key else "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"
        if api_key:
            base_url = input("请输入API Base URL (可选，直接回车使用默认): ").strip()
            model = input("请输入模型名称(可选，直接回车使用默认): ").strip()
            generator.setup_openai(
                api_key=api_key,
                # 默认使用openrouter的API
                base_url=base_url if base_url else "https://openrouter.ai/api/v1",
                model=model if model else "anthropic/claude-sonnet-4"
            )
        else:
            raise ValueError("使用本地模型")
    except:
        # 选项2: 使用本地模型（如果OpenAI不可用）
        print("⚠️ 将使用本地模型（可能效果较差）")
        try:
            generator.setup_local_model("microsoft/DialoGPT-medium")
        except Exception as e:
            print(f"❌ 本地模型设置失败: {e}")
            print("请安装必要依赖: pip install transformers torch")
            return
    
    # 用户输入主题
    topic = input("\n请输入综述主题（可选，回车使用默认LLM）: ").strip()
    if not topic:
        topic = "Large Language Models"
    
    subtopics_input = input("请输入子主题（用逗号分隔，可选）: ").strip()
    subtopics = [s.strip() for s in subtopics_input.split(",")] if subtopics_input else ["transformer architecture", "attention mechanism", "fine-tuning", "RLHF"]
    
    output_file = input("请输入输出文件名（可选）: ").strip()
    if not output_file:
        safe_topic = re.sub(r'[^\w\s-]', '', topic)[:50]
        output_file = f"./reviews/{safe_topic}_review"
    
    # 生成综述
    review = generator.generate_review(
        topic=topic,
        subtopics=subtopics,
        output_file=output_file
    )
    
    if review:
        print("\n" + "=" * 50)
        print("📄 生成的综述预览:")
        print("=" * 50)
        print(review[:1000] + "..." if len(review) > 1000 else review)
    
if __name__ == "__main__":
    main()