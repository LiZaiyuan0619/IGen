from __future__ import annotations  # 支持前向引用
import copy
from datetime import datetime
import imp
import os
import json
import asyncio
import re
from openai import AsyncOpenAI
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from tenacity import retry, stop_after_attempt, wait_exponential
import hashlib  # 新增：用于生成唯一ID

# 从现有代码导入数据库和辅助功能
from database_setup import AcademicPaperDatabase
from llm_review_generator import EnhancedSimilarityCalculator, ReviewConfig
# 导入工具文件中的utils函数
from utils import Citation, CitationManager, _format_enrichment_for_analysis, _parse_interpretation_response, _format_global_outline_for_prompt, _extract_quality_evaluation_writing, _extract_difference_analysis, _extract_iteration_decision, _clean_numeric_content, _format_materials_for_writing_prompt, _format_content_for_analysis, _format_global_context_for_analysis, _parse_writing_refinement_response, _format_materials_for_enrichment, _extract_scientific_enrichment_decision
from utils import *

class ModelType(Enum):
    """支持的模型类型"""
    CLAUDE = "anthropic/claude-sonnet-4"
    GPT = "openai/gpt-4o"
    # GEMINI = "google/gemini-2.5-pro"
    GEMINI = "google/gemini-2.5-flash"
    QWEN = "qwen/qwen2.5-vl-72b-instruct"
    DS = "deepseek/deepseek-chat-v3-0324"
    GLM = "z-ai/glm-4.5"

# 修改 LLMFactory 类

class LLMFactory:
    """统一的LLM调用接口，支持多种模型和参数配置"""
    
    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1", log_dir: str = "./logs"):
        """
        初始化LLM工厂
        
        Args:
            api_key: OpenRouter API密钥
            base_url: API基础URL，默认为OpenRouter
            log_dir: 日志目录
        """
        self.api_key = api_key
        self.base_url = base_url
        # 使用异步客户端，确保在事件循环中不会阻塞
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
        # 初始化日志记录器
        self.logger = LLMLogger(log_dir=log_dir)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate(self, 
                     model_name: str, 
                     messages: List[Dict], 
                     temperature: float = 0.7,
                     max_tokens: int = 8000,
                     stream: bool = False,
                     agent_name: str = "未知智能体",
                     task_type: str = None) -> Dict:
        """
        统一的生成接口，支持所有模型，并记录日志
        
        Args:
            model_name: 要使用的模型名称（使用ModelType枚举）
            messages: 对话消息列表
            temperature: 温度参数，控制创造性
            max_tokens: 最大生成令牌数
            stream: 是否使用流式输出
            agent_name: 调用的智能体名称，用于日志
            task_type: 任务类型，用于日志
            
        Returns:
            生成的响应
        """
        try:
            response = await self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )

            if stream:
                # 流式输出不记录日志（如需流式，可在调用处处理异步流）
                return response
            else:
                # 返回完整响应并记录日志
                result = {
                    "content": response.choices[0].message.content,
                    "model": response.model,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens if getattr(response, "usage", None) else 0,
                        "completion_tokens": response.usage.completion_tokens if getattr(response, "usage", None) else 0,
                        "total_tokens": response.usage.total_tokens if getattr(response, "usage", None) else 0
                    }
                }

                # 记录到日志
                self.logger.log_call(
                    agent_name=agent_name,
                    model_name=model_name,
                    messages=messages,
                    response=result,
                    task_type=task_type
                )

                return result

        except Exception as e:
            print(f"LLM调用错误: {e}")
            # 记录错误
            error_response = {
                "error": str(e),
                "content": f"❌ LLM调用失败: {str(e)}",  # 提供有意义的错误内容而不是None
                "model": model_name,
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }
            self.logger.log_call(
                agent_name=agent_name,
                model_name=model_name,
                messages=messages,
                response=error_response,
                task_type=task_type
            )
            return error_response

@dataclass
class AgentConfig:
    """智能体配置类"""
    model_name: str  # 智能体使用的LLM模型，从ModelType中选择
    temperature: float = 0.7  # 创造性参数，默认0.7
    max_tokens: int = 8000  # 生成长度，默认10000
    role_description: str = ""  # 智能体角色描述
    system_message: str = ""  # 系统提示词
    # 特定任务的提示词模板，键为任务名，值为提示词模板
    task_templates: Dict[str, str] = field(default_factory=dict)
    # 模型特定的参数
    model_params: Dict[str, Any] = field(default_factory=dict)

class BaseAgent:
    """所有智能体的基类，提供基本通信和执行接口"""
    
    def __init__(self, name: str, config: AgentConfig, llm_factory: LLMFactory):
        """
        初始化基础智能体
        
        Args:
            name: 智能体名称
            config: 智能体配置
            llm_factory: LLM工厂实例
        """
        self.name = name
        self.config = config
        self.llm = llm_factory
        
        # 确保系统消息设置正确
        if not self.config.system_message:
            self.config.system_message = f"你是{self.name}，一个{self.config.role_description}。"
    
    async def execute(self, task: Dict) -> Dict:
        """
        执行分配的任务，由子类实现
        
        Args:
            task: 任务描述，包含任务类型和相关参数
            
        Returns:
            任务执行结果
        """
        raise NotImplementedError("子类必须实现此方法")
    

    
    # 修改 BaseAgent 类的 call_llm 方法

    async def call_llm(self, prompt: str, task_type: str = None) -> Dict:
        """调用LLM生成内容"""
        messages = []
        
        # 添加系统消息
        if self.config.system_message:
            messages.append({"role": "system", "content": self.config.system_message})
        
        # 如果有特定任务类型的模板，则使用模板格式化提示词
        task_template = self.config.task_templates.get(task_type, "")
        if task_template:
            prompt = task_template.format(prompt=prompt)
        
        # 添加当前提示
        messages.append({"role": "user", "content": prompt})
        
        try:
            # LLM调用
            response = await self.llm.generate(
                model_name=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                agent_name=self.name,
                task_type=task_type,
                **self.config.model_params
            )
            
            # 检查响应有效性
            if not isinstance(response, dict):
                print(f"⚠️ LLM响应格式异常: {type(response)}")
                return {"error": "响应格式不是字典", "content": None}
            
            # 检查响应内容
            if response.get("error"):
                print(f"⚠️ LLM调用出现错误: {response.get('error')}")
            elif not response.get("content"):
                print(f"⚠️ LLM响应中无有效content字段: {list(response.keys())}")
            
            return response
            
        except Exception as e:
            print(f"❌ LLM调用异常: {e}")
            return {"error": str(e), "content": None}
    

class PlannerAgent(BaseAgent):
    """规划智能体：负责综述整体结构规划"""
    
    def __init__(self, name: str, config: AgentConfig, llm_factory: LLMFactory, db: AcademicPaperDatabase, 
                 citation_manager: CitationManager = None):
        super().__init__(name, config, llm_factory)
        self.db = db
        self.similarity_calculator = None
        self.citation_manager = citation_manager or CitationManager()
    
    async def setup_similarity_calculator(self, topic: str, subtopics: List[str] = None):
        """设置相似度计算器"""
        self.similarity_calculator = EnhancedSimilarityCalculator(topic, subtopics)
    
    async def search_relevant_content(self, topic: str, subtopics: List[str] = None) -> Dict:
        """搜索相关内容并计算增强相似度"""
        # 初始化相似度计算器
        if not self.similarity_calculator:
            await self.setup_similarity_calculator(topic, subtopics)
        
        # 调用utils中的函数
        return await search_relevant_content(
            self.db, 
            self.similarity_calculator, 
            topic, 
            subtopics, 
            purpose="规划",
            llm_call_func=self.call_llm  # 🆕 传入LLM调用函数用于翻译
        )
    
    async def create_outline(self, topic: str, subtopics: List[str], context: Dict) -> Dict:
        """创建综述大纲，使用1+3迭代优化方法"""
        print(f"📝 正在为'{topic}'创建综述大纲...")
        
        # 检查上下文数据
        if not context or not context.get("relevant_content"):
            print(f"⚠️ 警告: 上下文数据为空或缺少相关内容")
            return {}
        
        text_count = len(context.get("relevant_content", {}).get("texts", []))
        print(f"📊 共获取研究材料: {text_count} 条文本")
        
        # 准备所有材料
        all_materials = self._prepare_materials(context)
        
        # 1. 生成初始大纲
        initial_outline = await self._create_initial_outline(topic, subtopics, all_materials, context)
        
        # 2. 迭代优化大纲
        final_outline = await self._refine_outline_iteratively(
            initial_outline, all_materials, topic, context
        )
        
        return final_outline
    
    def _prepare_materials(self, context: Dict) -> List:
        """准备和排序所有材料"""
        return sorted(context["relevant_content"]["texts"], 
                      key=lambda x: x["relevance_score"], reverse=True)
    
    async def _create_initial_outline(self, topic: str, subtopics: List[str], all_materials: List, context: Dict) -> Dict:
        """生成初始大纲（使用前100个材料）"""
        print(f"📝 第0轮：生成初始大纲（使用前150个材料）")
        
        # 使用前100个材料生成初始大纲
        initial_materials = all_materials[:150]
        
        # 构建CoT提示词，引导LLM像学术专家一样思考综述结构
        cot_prompt = f"""你是一位资深的学术综述写作专家，需要为主题"{topic}"设计一个高质量的学术综述大纲框架。关键词领域包括：{subtopics}。请基于提供的研究材料，模仿顶级期刊(如Nature Reviews, ACM Computing Surveys, IEEE Surveys等)的综述写作标准来设计大纲。

⚠️ 核心结构约束：大纲必须严格遵循6章节固定结构
- 第1章：引言与背景
- 第2-5章：核心技术内容章节（4个章节涵盖该领域的主要技术方向）
- 第6章：结论与展望 

【学术综述写作原则】
1. 结构严谨性：遵循学术论文的经典逻辑脉络，确保每个章节都有明确的学术目的
2. 内容全面性：覆盖该领域的核心理论、关键技术、重要应用和前沿发展
3. 专业准确性：使用规范的学术术语，章节标题体现专业深度
4. 逻辑连贯性：章节间形成递进或互补关系，构建完整的知识体系
5. 结构控制性：通过丰富的子章节来承载内容，严禁超过6个一级章节，每一章可以有多个子章节

【6章节固定结构设计策略】
请严格按照以下6个步骤进行深度分析，确保在6章节框架内完整覆盖研究内容：

步骤1: 研究领域全景分析与4核心维度识别
思考要点：
- 基于提供的研究材料，这个领域可以划分为哪些核心技术方向？
- 这些方向如何能够完整覆盖该领域的主要研究内容？
- 每个技术方向的重要性、发展阶段和学术价值如何？
- 如何确保这些核心方向之间既有区别又相互补充，形成完整的知识体系？

步骤2: 固定6章节内容分配策略
思考要点：
- 第1章(引言)：如何系统介绍研究背景、核心问题、发展历程、综述范围？
- 第2章(核心方向1)：承载哪个最重要的技术方向？包含哪些关键子领域？
- 第3章(核心方向2)：聚焦哪个技术维度？与第2章如何形成互补？
- 第4章(核心方向3)：涵盖哪些技术内容？在整体结构中的作用是什么？
- 第5章(核心方向4)：处理哪些研究方向？如何与前面章节衔接？
- 第6章(结论)：如何总结发现、分析挑战、展望未来发展？

步骤3: 核心章节主题确定与内容边界
思考要点：
- 第2-5章分别应该聚焦哪个核心技术主题？主题表述要足够专业和精确
- 每个章节的学术定位和论述目标是什么？
- 如何确保4个章节既有明确的内容边界，又能完整覆盖该领域？
- 章节间的逻辑递进关系如何设计？（从基础到前沿、从理论到应用等）

步骤4: 子章节详细架构设计
思考要点：
- 每个核心章节(第2-5章)需要设置多少个子章节来充分展开内容？
- 子章节的划分原则是什么？（按技术类型、发展阶段、应用场景等）
- 如何通过丰富的子章节结构来承载大量技术内容，避免需要增加一级章节？
- 子章节的深度和覆盖度如何平衡，确保既全面又有重点？

步骤5: 专业术语与标题规范化
思考要点：
- 6个章节的标题如何使用该领域的标准学术术语？
- 章节标题是否准确反映该部分的核心技术内容和学术价值？
- 子章节标题如何体现具体的技术方向和研究重点？
- 标题的专业性和准确性是否符合国际期刊的发表标准？

步骤6: 内容完整性与结构合理性验证
思考要点：
- 固定的4个核心章节是否系统覆盖了该领域的所有重要研究方向？
- 是否存在重要技术或前沿发展无法放入这4个章节的情况？
- 通过子章节的细分是否能够充分展开技术内容而不需要增加一级章节？
- 6章节结构是否符合国际期刊综述论文的学术规范和篇幅要求？

【输出格式要求】
请严格按照以下格式输出，必须遵循6章节固定结构：

===大纲开始===

【综述概述】
(描述整个综述的研究范围、核心内容和学术价值，体现综述的专业定位)

【章节结构】
1. 引言与背景
(简述引言部分的内容安排，包括研究背景、问题定义、综述范围等)

1.1 子章节标题
(描述子章节的内容安排)

1.2 子章节标题
(描述子章节的内容安排)

1.3 子章节标题
(描述子章节的内容安排)

2. [第1个核心技术方向的专业标题]
(描述本章的学术目标、主要技术内容和在整体结构中的作用)

2.1 子章节标题
(描述子章节的内容安排)
...

3. [第2个核心技术方向的专业标题]
(描述本章的学术目标、主要技术内容和与第2章的关系)
(子章节自拟)

4. [第3个核心技术方向的专业标题]
(描述本章的学术目标、主要技术内容和在整体中的地位)
(子章节自拟)

5. [第4个核心技术方向的专业标题]
(描述本章的学术目标、主要技术内容和与前面章节的衔接)
(子章节自拟)

6. 结论与展望
(说明结论章节的内容安排，包括主要发现总结、挑战分析、未来方向)
(子章节自拟)

===大纲结束===

【思考过程】
步骤1分析: [研究领域分析的具体思考过程]
步骤2分析: [结构模式选择的具体思考过程]
步骤3分析: [章节主题确定的具体思考过程]
步骤4分析: [术语标题规范化的具体思考过程]
步骤5分析: [子章节架构的具体思考过程]
步骤6分析: [完整性验证的具体思考过程]

【学术规范要求】
1. 严格遵循6章节固定结构：1章引言 + 4章核心内容 + 1章结论
2. 所有章节标题必须使用该领域的标准学术术语
3. 章节结构应符合国际期刊综述论文的写作规范
4. 内容安排应体现从基础到前沿、从理论到应用的学术逻辑
5. 确保综述的学术深度和专业水准
6. 核心章节(2-5章)的子节数量不限，通过丰富子节来承载内容，给出子章节标题的同时需要给出对应描述
7. 禁止超过6个一级章节，所有内容必须合理分配到固定的6章节结构中"""

        # 添加部分高相关度内容作为参考
        cot_prompt += "\n【高相关度的研究材料】\n"
        for i, text_item in enumerate(initial_materials, 1):
            if i <= 150:  
                if(len(text_item['content']) > 2000):
                    cot_prompt += f"材料{i} (相关度: {text_item['relevance_score']:.2f}): {text_item['content'][:2000]}...\n\n"
                else:
                    cot_prompt += f"材料{i} (相关度: {text_item['relevance_score']:.2f}): {text_item['content']}\n\n"

        # 调用LLM生成初始大纲
        response = await self.call_llm(cot_prompt, task_type="initial_outline_creation")
        
        content = response.get("content")
        if not content or content.startswith("❌ LLM调用失败"):
            error_msg = f"初始大纲生成失败 - LLM响应异常"
            if "error" in response:
                error_msg += f": {response['error']}"
            print(f"❌ {error_msg}")
            raise ValueError(error_msg)
        
        # 解析初始大纲
        subtopics = context.get('subtopics', [])
        outline = parse_outline_response(content, topic, subtopics)
        
        print(f"✅ 初始大纲生成完成，使用了{len(initial_materials)}条材料")
        return outline
    
    async def _refine_outline_iteratively(self, initial_outline: Dict, 
                                        all_materials: List, topic: str, context: Dict) -> Dict:
        """迭代优化大纲的核心逻辑"""
        current_outline = initial_outline
        used_materials = 150  # 已使用的材料数量
        max_iterations = 1
        
        for iteration in range(1, max_iterations + 1):
            print(f"📝 第{iteration}轮：大纲优化迭代")
            
            # 获取下一批材料
            start_idx = used_materials
            end_idx = min(start_idx + 150, len(all_materials))
            
            if start_idx >= len(all_materials):
                print(f"⚠️ 材料已用完，停止迭代")
                break
                
            current_materials = all_materials[start_idx:end_idx]
            print(f"📊 使用材料范围：{start_idx+1}-{end_idx} ({len(current_materials)}条)")
            
            # 执行大纲优化和决策（合并在一次LLM调用中）
            refinement_result = await self._refine_and_decide(
                current_outline, current_materials, topic, iteration
            )
            
            if not refinement_result:
                print(f"❌ 第{iteration}轮优化失败，使用当前大纲")
                break
            
            # 更新大纲
            current_outline = refinement_result.get("outline")
            should_continue = refinement_result.get("should_continue", False)
            
            used_materials = end_idx
            print(f"✅ 第{iteration}轮优化完成，已使用材料：{used_materials}条")
            
            if not should_continue:
                break
        
        print(f"🎯 大纲优化完成，共进行了{min(iteration, max_iterations)}轮迭代")
        return current_outline
    
    async def _refine_and_decide(self, current_outline: Dict, new_materials: List, 
                               topic: str, iteration: int) -> Dict:
        """
        基于新材料优化大纲并决策是否继续迭代
        一次LLM调用完成：分析新材料、对比当前大纲、优化大纲、给出是否继续的决策
        """
        from utils import _format_outline_for_refinement, _format_materials_for_refinement
        refinement_prompt = f"""你是资深的学术综述质量评估与优化专家，需要采用科学严谨的方法评估当前大纲质量，并基于新材料进行精准的查漏补缺。请参考Self-Refine等先进的迭代优化框架，确保评估的客观性和改进的有效性。

⚠️ 核心结构约束：在迭代优化过程中，必须严格维持6章节固定结构
- 第1章：引言与背景
- 第2-5章：核心技术内容章节（4个章节，不允许增加或减少，但是可以调整子章节）
- 第6章：结论与展望
- 禁止增加一级章节数量，只能通过修改和补充现有章节内容和调整子章节来改进大纲，每一章可以有多个子章节

【核心任务】
1. 多维度质量评估：客观评估当前大纲的学术质量
2. 新材料缺陷识别：基于新材料发现当前大纲的不足
3. 增量精准改进：保持6个章节，通过内容优化和调整来改进
4. 科学迭代决策：基于量化指标决定是否继续迭代

【当前大纲】
{_format_outline_for_refinement(current_outline)}

【科学评估与优化框架】
请严格按照以下5个步骤进行分析：

步骤1: 多维度质量评估
评估维度：
- 学术完整性：当前大纲是否涵盖了该领域的核心理论、关键技术、重要方法和前沿发展？(评分: 1-10)
- 结构逻辑性：章节安排是否符合学术逻辑？从基础到前沿、从理论到应用的脉络是否清晰？(评分: 1-10)  
- 术语专业性：章节标题和描述是否使用了该领域的标准学术术语？专业深度是否适当？(评分: 1-10)
- 内容平衡性：各章节的重要性分配是否合理？是否存在过重或过轻的部分？(评分: 1-10)
- 国际规范性：大纲结构是否符合国际期刊综述论文的写作标准？(评分: 1-10)

步骤2: 新材料内容分析
分析要点：
- 核心概念提取：从新材料中识别出哪些重要的学术概念、技术方法、研究方向？
- 研究热点识别：新材料反映了该领域哪些最新的研究热点和发展趋势？
- 方法技术归类：新材料中的技术方法可以归类到哪些学术类别中？
- 应用场景扩展：新材料是否涉及了新的应用领域或场景？

步骤3: 缺陷与遗漏识别
对比分析（在6章节固定结构内进行）：
- 概念遗漏：新材料中的重要概念有哪些在当前大纲中完全缺失？应该放在哪个现有章节中？
- 章节缺陷：基于新材料，当前大纲的哪些章节存在内容薄弱、表述不准确或逻辑缺陷？
- 子章节不足：哪些现有章节需要通过增加或调整子章节来承载新材料的内容？
- 深度不够：哪些重要主题在当前大纲中涉及深度不足，需要在现有章节框架内加强？
- 内容分配：如何在固定的6章节结构内重新分配和组织内容以更好地反映新材料？

步骤4: 改进效果量化评估
量化指标：
- 新增重要内容占比：基于新材料可以新增的重要学术内容占当前大纲内容的百分比
- 结构优化程度：针对识别的结构缺陷，优化后能提升多少大纲的逻辑性？
- 专业性提升度：通过术语和表述的改进，能提升多少大纲的学术专业性？
- 完整性改善率：补充遗漏内容后，大纲完整性的提升程度
- 综合改进评分：基于上述四个维度，预估整体改进能带来的质量提升(1-10分)

步骤5: 科学迭代决策
决策标准：
- 如果综合改进评分 ≥ 7分，或新增重要内容占比 ≥ 25%，或发现严重结构缺陷，则继续迭代
- 如果综合改进评分 < 5分，且新增重要内容占比 < 15%，且无严重缺陷，则停止迭代
- 如果介于两者之间，综合考虑材料质量和改进必要性做决策

【输出格式要求】
⚠️ 严格格式要求：以下标记是系统解析的核心依据，任何遗漏都会导致系统解析失败！
⚠️ 重要提醒：必须严格保持6章节固定结构，禁止增加或减少一级章节数量！

请严格按照以下格式输出，确保所有标记完整：

===优化结果开始===

【优化后大纲】

===大纲开始===

【综述概述】
(基于质量评估和缺陷识别的结果，提供优化后的综述概述，3-5句学术化描述)

【章节结构】
1. 引言与背景
(章节描述，融入新材料的相关背景)

1.1 子章节标题
(描述子章节的内容安排)

1.2 子章节标题
(描述子章节的内容安排)

1.3 子章节标题
(描述子章节的内容安排)

2. [第1个核心技术方向的优化标题]
(描述本章的学术目标、主要技术内容和在整体结构中的作用)

2.1 子章节标题
(描述子章节的内容安排)

...

3. [第2个核心技术方向的优化标题]
(描述本章的学术目标、主要技术内容和与第2章的关系)
(子章节自拟)

4. [第3个核心技术方向的优化标题]
(描述本章的学术目标、主要技术内容和在整体中的地位)
(子章节自拟)

5. [第4个核心技术方向的优化标题]
(描述本章的学术目标、主要技术内容和与前面章节的衔接)
(子章节自拟)

6. 结论与展望
(基于新材料的优化结论安排)
(子章节自拟)

===大纲结束===

【是否继续迭代】
是/否

【科学决策依据】
决策量化指标:
- 综合改进评分: [评分]/10
- 新增重要内容占比: [百分比]%
- 是否发现严重结构缺陷: 是/否
- 是否有重要概念遗漏: 是/否

决策逻辑: 说明为什么做出这个决策

【多维度质量评估】
学术完整性: [评分]/10
结构逻辑性: [评分]/10 
术语专业性: [评分]/10
内容平衡性: [评分]/10 
国际规范性: [评分]/10 
综合质量评分: [平均分]/10


===优化结果结束===

【格式验证清单】
✓ ===优化结果开始=== (优化结果开头标记)
✓ ===优化结果结束=== (优化结果结尾标记)
✓ ===大纲开始=== (大纲开头标记)
✓ ===大纲结束=== (大纲结尾标记)
✓ 【是否继续迭代】(决策标记)
✓ 所有评估维度都有量化评分
✓ 决策有明确的科学依据

【学术规范要求】
1. 严格维持6章节固定结构：1章引言 + 4章核心内容 + 1章结论
2. 禁止增加或减少一级章节数量，只能通过子章节调整来承载新内容
3. 所有章节标题必须使用该领域的标准学术术语
4. 章节结构应符合国际期刊综述论文的写作规范
5. 内容安排应体现从基础到前沿、从理论到应用的学术逻辑
6. 确保综述的学术深度和专业水准
7. 核心章节(2-5章)的子节数量可灵活调整，通过子节扩展来承载内容
8. 所有"===XXX==="标记必须完整，不能有遗漏

【新参考材料】
{_format_materials_for_refinement(new_materials)}
"""

        response = await self.call_llm(refinement_prompt, task_type=f"outline_refinement_iter{iteration}")
        
        content = response.get("content")
        if not content or content.startswith("❌ LLM调用失败"):
            print(f"⚠️ 第{iteration}轮优化LLM调用失败")
            return None
        from utils import _parse_refinement_response
        # 解析优化结果
        return _parse_refinement_response(content, topic, iteration)
    

    async def execute(self, task: Dict) -> Dict:
        """执行规划任务"""
        if task.get("action") == "create_outline":
            topic = task.get("topic", "")
            subtopics = task.get("subtopics", [])
            
            # 1. 搜索相关内容
            context = await self.search_relevant_content(topic, subtopics)
            
            # 2. 创建大纲

            outline = await self.create_outline(topic, subtopics,context)
            
            return {
                "status": "success",
                "outline": outline,
                "context": context
            }
        
        elif task.get("action") == "integrate":
            chapter_contents = task.get("chapter_contents", [])
            topic = task.get("topic", "")
            enriched_outline = task.get("enriched_outline", {})  # 获取详细大纲
            subtopics = task.get("subtopics", [])
            result = await self.integrate_final_result(chapter_contents, topic, enriched_outline, subtopics)
            return {
                "status": "success",
                "result": result
            }
        
        else:
            return {
                "status": "error",
                "message": "未知的任务类型"
            }
    
    async def integrate_final_result(self, chapter_contents: List[Dict], topic: str = "", enriched_outline: Dict = None, subtopics: List[str] = None) -> Dict:
        """整合所有章节内容并生成摘要，使用CoT技术优化处理"""
        
        # 确保章节按ID排序
        sorted_chapters = sorted(chapter_contents, key=lambda x: x.get("id", "0"))
        
        
        # 🆕 先构建章节内容（用于生成目录）
        chapters_content = ""
        for chapter in sorted_chapters:
            chapter_title = chapter.get('title', '未命名章节')
            chapter_content = chapter.get('content', '')
            
            # 🆕 过滤掉可能包含的重复主标题、摘要和目录，避免重复
            if chapter_content:
                lines = chapter_content.split('\n')
                filtered_lines = []
                in_abstract_section = False
                in_toc_section = False
                abstract_lines_count = 0
                
                for i, line in enumerate(lines):
                    line_stripped = line.strip()
                    
                    # 跳过与主标题相同的行
                    if line_stripped == f"# {topic}" or line_stripped == f"#{topic}":
                        continue
                    
                    # 检测摘要标题
                    if line_stripped == "# 摘要" or line_stripped == "#摘要":
                        in_abstract_section = True
                        abstract_lines_count = 0
                        continue
                    
                    # 检测目录标题
                    if line_stripped == "# 目录" or line_stripped == "#目录":
                        in_toc_section = True
                        continue
                    
                    # 如果在摘要部分，跳过摘要内容（检测摘要特征或关键词行）
                    if in_abstract_section:
                        abstract_lines_count += 1
                        # 跳过摘要内容，直到遇到下一个主要标题
                        if (line_stripped.startswith("# ") and 
                            not line_stripped.startswith("# 摘要") and
                            abstract_lines_count > 3):  # 确保不是摘要标题本身
                            in_abstract_section = False
                            # 这一行是下个章节标题，需要保留
                        else:
                            if line_stripped.startswith("**关键词"):
                                pass
                            continue
                    
                    # 如果在目录部分，跳过目录内容
                    if in_toc_section:
                        # 跳过目录内容，直到遇到下一个主要标题
                        if line_stripped.startswith("# ") and not line_stripped.startswith("# 目录"):
                            in_toc_section = False
                            # 这一行是下个章节标题，需要保留
                        else:
                            continue
                    
                    # 只有不在摘要和目录部分的内容才保留
                    if not in_abstract_section and not in_toc_section:
                        filtered_lines.append(line)
                
                chapter_content = '\n'.join(filtered_lines)
                
                # 🆕 最后清理：移除多余的空行
                if chapter_content:
                    lines = chapter_content.split('\n')
                    cleaned_lines = []
                    prev_empty = False
                    for line in lines:
                        if line.strip():  # 非空行
                            cleaned_lines.append(line)
                            prev_empty = False
                        elif not prev_empty:  # 空行但前一行不是空行
                            cleaned_lines.append(line)
                            prev_empty = True
                        # 跳过连续的空行
                    chapter_content = '\n'.join(cleaned_lines)
            
            # 检查章节内容是否已经包含标题，避免重复
            chapter_id = chapter.get('id', '')
            expected_title_patterns = [
                f"# {chapter_id} {chapter_title}",  # 完整格式
                f"# {chapter_title}",               # 只有标题
                f"#{chapter_id} {chapter_title}",   # 无空格格式
            ]
            
            has_title = False
            if chapter_content:
                content_start = chapter_content.strip()
                # 检查多种可能的标题格式
                for pattern in expected_title_patterns:
                    if content_start.startswith(pattern):
                        has_title = True
                        break
                
                # 如果没有找到精确匹配，检查是否以任何标题开头
                if not has_title and content_start.startswith("#"):
                    has_title = True
            
            if has_title:
                # 内容已包含标题，直接使用
                chapters_content += f"{chapter_content}\n\n"
            else:
                # 内容没有标题，添加标题
                print(f"📝 为章节 {chapter_id} 添加标题")
                chapters_content += f"# {chapter_id} {chapter_title}\n\n"
                chapters_content += f"{chapter_content}\n\n"
        
        # 🆕 生成目录
        table_of_contents = generate_table_of_contents(chapters_content, topic)
        
        # 2. subtopics信息（如果存在）
        subtopics_section = format_subtopics_section(subtopics)
        
        # 先保存章节内容，稍后添加到文档中
        document_chapters = chapters_content
        
        # 统计信息（基于章节内容计算）
        total_words = len(document_chapters.split())
        chapter_count = len(sorted_chapters)
        
        # 创建摘要提示词，使用CoT技术，首先系统性地展示完整的综述规划
        abstract_prompt = f"""你是一位专业学术综述编辑，需要为一篇关于"{topic}"的综述论文创建一个全面的摘要。

                            ===========================================
                            【完整综述规划信息】
                            ===========================================
                            """

        # 使用utils中的函数生成详细规划内容
        abstract_prompt += build_detailed_planning_section(enriched_outline)

        # 添加CoT思考指引 - 重新组织逻辑
        abstract_prompt += """
                        ===========================================
                        【学术综述摘要生成专家指南】
                        ===========================================
                        
                        你是一位资深的学术期刊编辑和综述写作专家，专门为国际顶级期刊撰写高质量学术综述摘要。请基于上述完整的综述规划信息，按照国际学术期刊标准生成一份专业的综述摘要。

                        ⚠️ 重要提醒：在撰写摘要时，请不要引用具体的材料编号，直接描述学术观点和研究发现即可。

                        【学术综述摘要的核心要求】
                        1. **自包含性**: 读者仅通过摘要就能理解综述的核心价值和主要贡献
                        2. **简洁精准**: 每句话都承载关键信息
                        3. **学术规范**: 使用标准的学术表达和专业术语
                        4. **逻辑清晰**: 遵循问题→方法→发现→贡献→展望的清晰脉络
                        5. **突出价值**: 明确阐述综述对该研究领域的独特贡献

                        【摘要写作的6步分析框架】
                        请严格按照以下步骤进行深入分析：

                        步骤1: 研究背景与问题识别
                        分析要点：
                        - 该综述所针对的核心研究问题或技术挑战是什么？
                        - 为什么这个问题在当前具有重要的学术价值和实践意义？
                        - 现有研究在解决这个问题上存在哪些不足或空白？
                        - 综述的主题在整个学科领域中的地位和重要性如何？

                        步骤2: 综述范围与方法论分析
                        分析要点：
                        - 综述覆盖了该领域的哪些核心方面和研究方向？
                        - 采用了什么样的系统性方法来组织和分析现有研究？
                        - 文献来源的覆盖范围和时间跨度如何？
                        - 综述的分析框架和评估标准是什么？

                        步骤3: 核心发现与技术洞察提炼 
                        分析要点：
                        - 通过综合分析现有研究，发现了哪些重要的技术趋势和规律？
                        - 不同技术路线、方法或理论之间的比较分析得出了什么结论？
                        - 在技术原理、算法机制、系统设计等方面有哪些深入洞察？
                        - 各章节的关键词和重点研究领域反映出哪些核心技术特征？

                        步骤4: 学术贡献与创新价值识别 
                        分析要点：
                        - 综述在知识整合和理论构建方面的主要贡献是什么？
                        - 提出了哪些新的分类体系、评估框架或分析视角？
                        - 对该领域未来发展方向提供了哪些有价值的指导？
                        - 综述的系统性分析为研究者和实践者提供了什么独特价值？

                        步骤5: 挑战与局限性认知 
                        分析要点：
                        - 当前研究中存在哪些技术瓶颈、理论空白或方法局限？
                        - 不同研究之间存在哪些争议或不一致的地方？
                        - 哪些重要问题仍然缺乏深入研究或有效解决方案？
                        - 技术实现、实际应用中面临的主要挑战有哪些？

                        步骤6: 未来方向与发展展望
                        分析要点：
                        - 基于综述分析，该领域最有前景的发展方向是什么？
                        - 哪些新兴技术、交叉领域或研究方法值得重点关注？
                        - 综述的分析结果如何指导未来的研究议程和技术发展？
                        - 对学术界和产业界的实践有什么重要启示？

                        【输出格式要求】
                        请严格按照以下格式输出，确保使用精确的标记分隔符：
                         
                        ===摘要开始===
                        
                        [按照上述5段标准结构撰写摘要正文]
                        
                        **关键词:** [关键词1], [关键词2], [关键词3], [关键词4], [关键词5]
                        (关键词不能与{topic}相同)
                        
                        ===摘要结束===
                         
                        ===专家分析过程开始===
                        **步骤1 - 背景问题分析:** [详述对研究背景和核心问题的分析]
                        **步骤2 - 方法范围分析:** [说明对综述范围和方法的理解]  
                        **步骤3 - 核心发现提炼:** [阐述从章节内容中提炼的重要发现]
                        **步骤4 - 贡献价值识别:** [说明综述的独特学术贡献和价值]
                        **步骤5 - 挑战局限认知:** [指出当前存在的主要挑战和局限]
                        **步骤6 - 未来展望构建:** [描述对未来发展方向的思考]
                        ===专家分析过程结束===

                        【摘要质量控制清单】
                        ✓ 每句话都承载关键信息，无冗余表述
                        ✓ 使用精准的学术术语和专业表达
                        ✓ 逻辑结构清晰，段落间有明确的递进关系
                        ✓ 突出综述的独特价值和学术贡献  
                        ✓ 关键词准确反映综述的核心技术和方法
                        ✓ 整体表述具有国际期刊发表水准
                        """

        # 调用LLM生成摘要
        response = await self.call_llm(abstract_prompt, task_type="abstract_generation")
        
        content = response.get("content")
        if not content or content.startswith("❌ LLM调用失败"):
            error_msg = f"摘要生成失败: {response.get('error', '未知错误')}"
            raise ValueError(error_msg)
        
        raw_response = content
        
        # 解析响应，提取摘要部分
        abstract_text, keywords = parse_abstract_response(raw_response)
        
        # 🆕 从引用JSON文件生成真实的参考文献列表
        bibliography = generate_bibliography_from_citations(topic)
        
        # 🆕 从引用JSON文件生成参考公式列表
        equations_section = generate_equations_from_citations(topic)
        
        # 🆕 从引用JSON文件生成参考图片列表
        figures_section = generate_figures_from_citations(topic)
        
        # 🆕 从引用JSON文件生成参考表格列表
        tables_section = generate_tables_from_citations(topic)
        
        # 🆕 重新构建完整文档结构
        # 1. 主标题
        full_document = ""
        
        # 3. subtopics信息（如果存在）
        if subtopics_section:
            full_document += subtopics_section
        else:
            print(f"🆕 没有subtopics信息!")
        
        # 4. 目录
        full_document += table_of_contents
        
        # 5. 章节内容
        full_document += document_chapters
        
        # 6. 参考文献
        if bibliography and bibliography.strip() != "# 参考文献\n\n无引用文献。\n":
            full_document += f"\n{bibliography}"
        
        # 7. 参考公式
        if equations_section and equations_section.strip() != "# 参考公式\n\n无引用公式。\n":
            full_document += f"\n{equations_section}"
        
        # 8. 参考图片
        if figures_section and figures_section.strip() != "# 参考图片\n\n无引用图片。\n":
            full_document += f"\n{figures_section}"
        
        # 9. 参考表格
        if tables_section and tables_section.strip() != "# 参考表格\n\n无引用表格。\n":
            full_document += f"\n{tables_section}"
        
        # 🆕 在文档正文中插入图片（在第一次引用位置）
        full_document = insert_figures_into_document(full_document, topic)
        
        # 🆕 在文档正文中插入表格图像（在第一次引用位置）
        full_document = insert_tables_into_document(full_document, topic)
        
        # 重新计算总字数（包含所有内容）
        total_words = len(full_document.split())
        
        from utils import _count_actual_citations

        # 构建最终结果
        result = {
            "full_document": full_document,
            "abstract": abstract_text,
            "keywords": keywords,
            "bibliography": bibliography,
            "equations": equations_section,
            "figures": figures_section,
            "tables": tables_section,
            "statistics": {
                "chapter_count": chapter_count,
                "total_words": total_words,
                "total_citations": _count_actual_citations(topic)
            }
        }
        
        return result
    
class EnricherAgent(BaseAgent):
    """丰富智能体：负责丰富大纲内容，添加编写指引"""
    
    def __init__(self, name: str, config: AgentConfig, llm_factory: LLMFactory, db: AcademicPaperDatabase,
                 citation_manager: CitationManager = None):
        super().__init__(name, config, llm_factory)
        self.db = db
        self.similarity_calculator = None
        self.citation_manager = citation_manager or CitationManager()
    
    async def setup_similarity_calculator(self, topic: str, subtopics: List[str] = None):
        """设置相似度计算器"""
        self.similarity_calculator = EnhancedSimilarityCalculator(topic, subtopics)
    
    async def enrich_outline(self, outline: Dict, context: Dict) -> Dict:
        """丰富大纲内容，使用1+3迭代优化方法"""
        print(f"📝 正在丰富综述大纲: {outline.get('topic', '未知主题')}")
        
        # 确保有相似度计算器
        if not self.similarity_calculator:
            await self.setup_similarity_calculator(outline.get('topic', ''), 
                                              context.get('subtopics', []))
        
        # 如果context中没有relevant_content，则需要先搜索相关内容
        if not context.get("relevant_content"):
            context = await self.search_relevant_content(outline.get('topic', ''), context.get('subtopics', []))
        
        # 检查材料数据
        text_count = len(context.get("relevant_content", {}).get("texts", []))
        print(f"📊 共获取研究材料: {text_count} 条文本")
        
        # 准备所有材料
        all_materials = self._prepare_materials(context)
        print(f"🔄 开始1+3迭代优化：初始丰富 + 最多3轮迭代")
        
        # 1. 生成初始丰富大纲
        initial_enriched = await self._create_initial_enrichment(outline, all_materials, context)
        
        # 2. 迭代优化丰富内容
        final_enriched = await self._refine_enrichment_iteratively(
            initial_enriched, all_materials, context
        )
        
        # 🧹 清洗材料引用信息
        cleaned_enriched = clean_material_references_enriched(final_enriched)
        return cleaned_enriched
    
    def _prepare_materials(self, context: Dict) -> List:
        """准备和排序所有材料"""
        return sorted(context["relevant_content"]["texts"], 
                      key=lambda x: x["relevance_score"], reverse=True)
    
    async def _create_initial_enrichment(self, outline: Dict, all_materials: List, context: Dict) -> Dict:
        """第0轮：生成初始丰富大纲（使用前100个材料）"""
        print(f"📝 第0轮：生成初始丰富大纲（使用前100个材料）")
        
        # 使用前100个材料生成初始丰富大纲
        initial_materials = all_materials[:100]
        
        # 创建临时context用于初始丰富
        temp_context = copy.deepcopy(context)
        temp_context["relevant_content"]["texts"] = initial_materials
        
        # 创建丰富后的大纲副本
        enriched_outline = copy.deepcopy(outline)
        
        # 构建初始丰富提示词
        full_prompt = self._build_full_enrichment_prompt(outline, {}, temp_context)
        
        # 调用LLM生成初始丰富大纲
        print("🧠 正在调用LLM丰富整个大纲...")
        response = await self.call_llm(full_prompt, task_type="initial_enrichment")
        
        content = response.get("content")
        if not content or content.startswith("❌ LLM调用失败"):
            error_msg = response.get("error", "未知错误")
            print(f"⚠️ 初始大纲丰富失败: {error_msg}")
            return enriched_outline
        
        # 解析LLM响应，获取丰富后的大纲
        from utils import parse_full_enrichment
        parsed_enrichment = parse_full_enrichment(content, enriched_outline)
        
        print(f"✅ 初始丰富完成，使用了{len(initial_materials)}条材料")
        return parsed_enrichment
    
    async def _refine_enrichment_iteratively(self, initial_enriched: Dict, 
                                           all_materials: List, context: Dict) -> Dict:
        """迭代优化丰富内容的核心逻辑"""
        current_enriched = initial_enriched
        used_materials = 100  # 已使用的材料数量
        max_iterations = 2
        
        for iteration in range(1, max_iterations + 1):
            print(f"📝 第{iteration}轮：丰富内容优化迭代")
            
            # 获取下一批材料
            start_idx = used_materials
            end_idx = min(start_idx + 100, len(all_materials))
            
            if start_idx >= len(all_materials):
                print(f"⚠️ 材料已用完，停止迭代")
                break
                
            current_materials = all_materials[start_idx:end_idx]
            print(f"📊 使用材料范围：{start_idx+1}-{end_idx} ({len(current_materials)}条)")
            
            # 执行丰富内容分析和优化（合并在一次LLM调用中）
            refinement_result = await self._analyze_and_refine(
                current_enriched, current_materials, context, iteration
            )
            
            if not refinement_result:
                print(f"❌ 第{iteration}轮优化失败，使用当前丰富内容")
                break
            
            # 更新丰富内容
            new_enrichment = refinement_result.get("enrichment")
            should_continue = refinement_result.get("should_continue", False)
            
            if isinstance(new_enrichment, dict):
                current_enriched = new_enrichment
                print(f"✓ 成功更新丰富内容，主题: {new_enrichment.get('topic', '未知')}")
            else:
                print(f"⚠️ new_enrichment类型异常，保持当前丰富内容。异常值: {str(new_enrichment)[:200] if new_enrichment else 'None'}")
                # 保持 current_enriched 不变
            
            used_materials = end_idx
            print(f"✅ 第{iteration}轮优化完成，已使用材料：{used_materials}条")
            
            if not should_continue:
                break
        
        print(f"🎯 丰富内容优化完成，共进行了{min(iteration, max_iterations)}轮迭代")
        
        # 记录最终丰富后的结构
        if hasattr(self.llm, "logger"):
            self.llm.logger.log_parsed_structure(
                agent_name=self.name, 
                task_type="enrichment_final",
                parsed_structure=current_enriched
            )
        
        return current_enriched
    
    async def _analyze_and_refine(self, current_enriched: Dict, new_materials: List, 
                                 context: Dict, iteration: int) -> Dict:
        """
        基于新材料科学分析并优化丰富内容，采用多维度评估与精准改进机制
        """
        
        refinement_prompt = f"""你是资深的学术综述内容质量评估与优化专家，需要采用科学严谨的方法评估当前丰富大纲的质量，并基于新材料进行精准的内容优化。请参考国际期刊编辑标准，确保评估的客观性和改进的有效性。

⚠️ 重要提醒：在分析和优化过程中，请不要引用具体的材料编号（如"材料1"、"材料3"、"结合材料NUM"等），直接描述发现的技术内容、学术观点和改进建议即可。

【核心任务】
1. 多维度内容质量评估：客观评估当前丰富大纲的学术指导价值
2. 新材料价值挖掘：深度分析新材料对内容规划的补充潜力
3. 关键词优化策略：确保关键词精准适配数据库检索需求
4. 增量精准改进：保持整体框架，针对性优化和补充
5. 科学迭代决策：基于量化指标决定是否继续迭代
6. 严格遵循6章节固定结构：1章引言 + 4章核心内容 + 1章结论，禁止修改章节数量，但是可以调整子章节。不需要给出字数建议

【当前丰富大纲】
{_format_enrichment_for_analysis(current_enriched)}

【科学评估与优化框架】
请严格按照以下6个步骤进行分析：

步骤1: 多维度内容质量评估
评估维度：
- 写作指导完整性：当前大纲是否为Writer提供了充分的写作指导？每个子章节的任务是否明确？(评分: 1-10)
- 关键词检索精准性：章节关键词是否能够精准检索到相关文献？是否避免了过泛或过窄的词汇？(评分: 1-10)
- 学术深度适宜性：内容规划的学术深度是否匹配目标期刊标准？技术性是否充分？(评分: 1-10)
- 结构逻辑合理性：章节安排和子章节组织是否符合学术逻辑？内容衔接是否流畅？(评分: 1-10)
- 实用指导价值：写作建议是否具体可操作？能否真正指导Writer完成高质量写作？(评分: 1-10)

步骤2: 新材料价值分析
分析要点：
- 技术贡献识别：新材料中包含哪些当前大纲未涵盖的重要技术方法、算法或理论？
- 应用场景扩展：新材料是否揭示了新的应用领域或使用场景？
- 研究热点发现：新材料反映了哪些最新的研究趋势和发展方向？
- 评估标准更新：新材料是否提供了新的评估指标或性能标准？

步骤3: 关键词优化潜力评估
评估要点：
- 检索效果分析：当前关键词在学术数据库中的检索效果如何？是否存在遗漏重要文献的风险？
- 术语更新需求：新材料是否引入了更精准或更前沿的技术术语？
- 覆盖度优化：关键词是否充分覆盖了该章节的核心技术概念？
- 差异化程度：章节间关键词是否有足够的差异化，避免检索重复？

步骤4: 内容缺陷与改进机会识别
对比分析：
- 写作指导缺陷：当前写作建议有哪些不够具体或缺失的部分？
- 内容规划遗漏：基于新材料，当前大纲在哪些重要方向上存在内容薄弱？
- 关键词策略不足：哪些重要的技术术语未被纳入关键词体系？
- 学术价值提升：如何通过内容优化提升大纲的学术指导价值？

步骤5: 改进效果量化评估
量化指标：
- 新增指导价值占比：基于新材料可以新增的重要指导内容占当前大纲内容的百分比
- 关键词优化程度：关键词优化后能提升多少检索精准度和文献覆盖率？
- 写作指导增强度：通过改进能提升多少Writer的写作效率和质量？
- 学术价值提升率：内容优化后大纲的学术指导价值能提升多少？
- 综合改进评分：基于上述四个维度，预估整体改进能带来的价值提升(1-10分)

步骤6: 科学迭代决策
决策标准：
- 如果综合改进评分 ≥ 7分，或新增指导价值占比 ≥ 25%，或发现重要内容缺陷，则继续迭代
- 如果综合改进评分 < 5分，且新增指导价值 < 15%，且无重要缺陷，则停止迭代
- 如果介于两者之间，综合考虑材料质量和改进必要性做决策

【输出格式要求】
⚠️ 严格格式要求：以下标记是系统解析的核心依据，任何格式错误都会导致后续流程失败！
⚠️ 内容要求：在所有分析和优化内容中，不要使用"材料1"、"材料NUM"、"结合材料X"等具体材料编号引用！

请严格按照以下格式输出，确保所有标记完整：

===优化结果开始===

【优化后丰富大纲】
===内容规划开始===
【章节内容指引】
# 第1章：章节标题
章节内容指引:基于上述6步分析，详细描述章节的学术目标、核心内容、研究范围和期望贡献，融入新材料的相关发现

## 本章节关键词
(基于关键词优化分析，精选6-8该章节特有的技术术语，优先选择能精确检索相关文献的专业术语，尽量不同章节之间不重复，确保数据库搜索的有效性)

## 重点研究领域
(基于新材料价值分析，识别该章节需要深入分析的核心研究方向)
1. 研究方向1: 结合新材料发现，明确该方向的研究现状、主要挑战和发展趋势
2. 研究方向2: 基于技术贡献识别，阐述该方向的核心技术、代表性工作和学术价值
3. 研究方向3: 融入前沿发现，分析该方向的应用前景、局限性和未来机遇
...

### 1.1 子章节标题
内容概要:基于内容缺陷识别的结果，明确定义该子章节的学术范围、主要论点和预期成果

#### 关键要点
(基于新材料分析的核心学术观点，每个要点都应有充分的文献支撑)
1. 核心观点1: 融入新技术贡献的具体学术结论或技术特点
2. 核心观点2: 基于前沿发现的重要方法论贡献或实验结果
...

#### 写作建议
(基于写作指导缺陷分析，为Writer提供更具体的写作策略和结构安排)
优化后的写作指导，包括具体的论证逻辑、文献使用策略、技术深度要求等，确保Writer能够高效完成高质量写作

### 1.2 子章节标题
内容概要:优化后的子章节学术任务和内容边界
#### 关键要点
#### 写作建议
...

[继续为所有章节提供完整的优化内容规划，确保每个章节都融入了新材料的价值发现]

===内容规划结束===

【是否继续迭代】
是/否

【科学决策依据】
决策量化指标:
- 综合改进评分: [评分]/10
- 新增指导价值占比: [百分比]%
- 是否发现重要内容缺陷: 是/否
- 是否有重要关键词遗漏: 是/否

决策逻辑: [基于步骤6的科学决策标准，明确说明为什么做出这个决策]


【多维度内容质量评估】
写作指导完整性: [评分]/10
关键词检索精准性: [评分]/10
学术深度适宜性: [评分]/10 
结构逻辑合理性: [评分]/10 
实用指导价值: [评分]/10 
综合质量评分: [平均分]/10

===优化结果结束===

【必须出现下面的格式】
✓ ===优化结果开始=== (开头标记)
✓ ===优化结果结束=== (结尾标记)
✓ ===内容规划开始=== 和 ===内容规划结束=== (内容标记)
✓ 【是否继续迭代】(决策标记)
✓ 所有评估维度都有量化评分

【新参考材料】
{_format_materials_for_enrichment(new_materials)}
"""

        response = await self.call_llm(refinement_prompt, task_type=f"enrichment_refinement_iter{iteration}")
        
        content = response.get("content")
        if not content or content.startswith("❌ LLM调用失败"):
            print(f"⚠️ 第{iteration}轮优化LLM调用失败")
            return None
        from utils import  _parse_enrichment_refinement_response
        # 解析优化结果，传递当前丰富内容以保留topic和overview信息
        return _parse_enrichment_refinement_response(content, iteration, current_enriched)
    

    async def search_relevant_content(self, topic: str, subtopics: List[str] = None) -> Dict:
        """搜索相关内容并计算增强相似度"""
        # 初始化相似度计算器
        if not self.similarity_calculator:
            await self.setup_similarity_calculator(topic, subtopics)
        
        # 调用utils中的函数
        return await search_relevant_content(
            self.db, 
            self.similarity_calculator, 
            topic, 
            subtopics, 
            purpose="内容丰富",
            llm_call_func=self.call_llm  # 🆕 传入LLM调用函数用于翻译
        )
    
    def _build_full_enrichment_prompt(self, outline: Dict, global_materials: Dict, context: Dict) -> str:
        """构建一次性丰富整个大纲的CoT提示词"""
        topic = outline.get('topic', '')
        overview = outline.get('overview', '')
        subtopics = context.get('subtopics', [])
        subtopics_str = ", ".join(subtopics) if subtopics else "无"
        
        prompt = f"""你是一位资深的学术综述内容规划专家，专门为学术期刊级别的综述论文制定详细的写作指引。你需要为"{topic}"综述的每个章节制定精准的内容规划，确保Writer能够高效完成高质量的学术写作。

⚠️ 重要提醒：在生成内容时，请不要引用具体的材料编号（如"材料1"、"材料3"、"结合材料NUM"等），直接描述技术内容和学术观点即可。

【核心使命】
为每个章节的Writer提供：
1. 明确的学术任务定位和目标
2. 精准的章节特定关键词(用于检索相关材料)  
3. 详细的内容结构和写作指导
4. 符合国际期刊标准的学术规范要求

【综述背景信息】
核心主题: {topic}
相关领域: {subtopics_str}
综述概述: {overview}

【学术综述内容规划策略】
请严格按照以下6个步骤进行深度分析：

步骤1: 学术综述整体框架解读与任务分解
思考要点：
- 这篇综述旨在解决该领域的哪些核心学术问题？
- 每个章节在整个综述论证逻辑中承担什么角色？
- 如何确保各章节内容的连贯性和递进性？
- 读者(学术同行)期望从每个章节获得什么价值？

步骤2: 章节学术定位与研究范围界定
思考要点：
- 每个章节应该覆盖该领域的哪些核心子方向？
- 章节内容的学术深度和广度如何平衡？
- 该章节需要回答哪些具体的研究问题？
- 如何体现该章节的学术创新点和贡献？

步骤3: 章节特定关键词策略设计
思考要点：
- 每个章节的Writer需要检索哪些特定的技术术语和概念？
- 如何选择能够精准定位相关文献的关键词？
- 章节关键词如何深化与全局关键词({subtopics_str})？
- 哪些专业术语能最好地代表该章节的技术特色？

步骤4: 研究脉络与文献组织策略
思考要点：
- 每个章节应该如何组织相关研究的发展脉络？
- 需要重点分析哪些代表性工作和里程碑成果？
- 如何处理不同研究路径之间的对比和评价？
- 章节内容如何体现研究领域的最新进展？

步骤5: 写作结构与论证逻辑设计
思考要点：
- 每个子章节的论证逻辑如何支撑主章节的目标？
- 如何安排内容的呈现顺序以增强说服力？
- 章节内部的技术细节和概念阐述如何平衡？
- 如何确保内容的学术严谨性和可读性？

步骤6: Writer任务指导与质量控制
思考要点：
- 如何为Writer提供明确的写作任务和期望？
- 每个章节的写作重点和难点是什么？
- 如何确保输出内容符合目标期刊的发表标准？
- 章节写作的评价标准和质量控制要点是什么？

【输出格式要求】
⚠️ 严格格式要求：以下标记是系统解析的核心依据，任何格式错误都会导致后续流程失败！

请严格按照以下格式输出，确保每个标记都完整准确：

===内容规划开始===
【章节内容指引】
# 第1章：章节标题
章节内容指引:基于上述6步分析，详细描述章节的学术目标、核心内容、研究范围和期望贡献，确保Writer明确理解任务定位，体现学术专业性

## 本章节关键词
(精选5-6个该章节特有的技术术语，用于Writer检索章节相关材料，避免与全局关键词重复，优先选择能精确定位相关文献的专业术语)

## 重点研究领域
(识别该章节需要深入分析的核心研究方向)
1. 研究方向1: 明确该方向的研究现状、主要挑战和发展趋势
2. 研究方向2: 阐述该方向的核心技术、代表性工作和学术价值
3. 研究方向3: 分析该方向的应用前景、局限性和未来机遇
...

### 1.1 子章节标题
内容概要:明确定义该子章节的学术范围、主要论点和预期成果，为Writer提供清晰的写作目标

#### 关键要点
(该子章节必须阐述的核心学术观点，每个要点都应有充分的文献支撑)
1. 核心观点1: 具体的学术结论或技术特点，需要在写作中重点论证
2. 核心观点2: 重要的方法论贡献或实验发现，需要详细分析
...

#### 写作建议
(为Writer提供具体的写作策略和结构安排)
采用[问题提出→方法分析→效果评估→局限讨论]的逻辑结构，重点突出该子章节的技术创新点，确保引用权威文献支撑论点。注意与相邻子章节的内容衔接，避免重复。

### 1.2 子章节标题  
内容概要:该子章节的具体学术任务和内容边界

#### 关键要点
1. 核心观点1: 需要深入分析的技术要点或理论贡献
2. 核心观点2: 需要对比评价的方法或实验结果
...

#### 写作建议
具体的写作指导，包括结构安排、重点强调、文献使用策略等

### 1.3 子章节标题
内容概要:该子章节的学术价值和预期输出

#### 关键要点
1. 核心观点1: 该子章节的独特贡献或重要发现
2. 核心观点2: 需要特别关注的技术细节或方法论问题
...

#### 写作建议
针对该子章节的具体写作指导和质量要求

# 第2章：章节标题
章节内容指引:基于6步分析的该章节学术定位和具体任务

## 本章节关键词
(该章节特有的5-6个专业术语，用于精准检索)

## 重点研究领域
(该章节的核心研究方向，)
1. 研究方向1的具体描述和学术价值
2. 研究方向2的技术特点和发展现状
...

### 2.1 子章节标题
内容概要:明确的学术任务定义
#### 关键要点
#### 写作建议
...

### 2.2 子章节标题
内容概要:具体的内容范围和目标
#### 关键要点
#### 写作建议
...

[继续为所有章节提供完整的内容规划，确保每个章节都有明确的学术定位和详细的写作指导]

===内容规划结束===

⚠️ 重要提醒：在生成章节内容指引、关键要点、写作建议等所有内容时，请不要使用"基于材料1"、"结合材料NUM"、"材料X显示"等具体材料编号引用，直接描述技术内容和学术观点即可。

【学术规范要求】
1. 严格遵循6章节固定结构：1章引言 + 4章核心内容 + 1章结论
2. 每个章节的关键词必须具有高度的技术针对性，能够精准检索相关文献
3. 内容规划必须体现国际期刊级别的学术深度和专业性
4. 写作建议必须具体可操作，包含明确的结构安排和质量标准
5. 确保各章节内容的逻辑连贯性和学术价值的递进性
6. 为Writer提供的指导应该具有足够的专业深度和可执行性
7. 核心章节(2-5章)的子节数量可灵活调整，通过子节扩展来承载内容。不需要给出字数建议
"""
         
        # 添加当前大纲结构信息
        prompt += "\n\n【当前综述大纲结构】\n"
        for chapter in outline.get("chapters", []):
            chapter_id = chapter.get("id", "")
            chapter_title = chapter.get("title", "")
            chapter_desc = chapter.get("description", "")
            
            prompt += f"第{chapter_id}章: {chapter_title}\n"
            prompt += f"章节定位: {chapter_desc}\n"
            
            # 添加子章节信息
            subsections = chapter.get("subsections", [])
            if subsections:
                prompt += "子章节:\n"
                for subsection in subsections:
                    subsection_id = subsection.get("id", "")
                    subsection_title = subsection.get("title", "")
                    subsection_desc = subsection.get("description", "")
                    
                    prompt += f"  {subsection_id} {subsection_title}\n"
                    if subsection_desc:
                        prompt += f"    任务: {subsection_desc}\n"
            prompt += "\n"
        
        # 添加高质量研究材料用于内容规划参考
        prompt += "\n【高质量研究材料参考】\n"
        prompt += "以下材料仅供理解研究现状和技术发展，请基于这些材料的学术内容来规划各章节的写作重点。\n"
        prompt += "⚠️ 注意：材料编号仅用于展示，在生成内容时材料编号对应可能不匹配！\n\n"
        
        # 检查context是否有relevant_content
        if context and "relevant_content" in context and "texts" in context["relevant_content"]:
            # 提取最相关的内容用于提示词，优化材料呈现
            top_texts = sorted(context["relevant_content"]["texts"], 
                              key=lambda x: x["relevance_score"], reverse=True)[:100]  # 取前100条
            
            for i, text_item in enumerate(top_texts, 1):
                relevance_score = text_item.get('relevance_score', 0.0)
                content = text_item.get('content', '')
                
                # 优化材料呈现，重点突出学术价值
                if len(content) > 2000:
                    prompt += f"📚 研究材料{i} (相关度: {relevance_score:.2f}):\n{content[:2000]}...\n\n"
                else:
                    prompt += f"📚 研究材料{i} (相关度: {relevance_score:.2f}):\n{content}\n\n"
        else:
            prompt += "⚠️ 当前未获取到相关研究材料，请基于已有的学术知识进行内容规划。\n"
        
        # 添加最终提醒
        prompt += "\n【最终提醒】\n"
        prompt += "请确保为每个章节和子章节都提供：\n"
        prompt += "1. 明确的学术任务定位和预期贡献\n"
        prompt += "2. 精准的章节特定关键词(5-6个)\n"
        prompt += "3. 具体可操作的写作建议和结构安排\n"
        prompt += "4. 符合国际期刊标准的质量要求\n"
        prompt += "严格按照===内容规划开始===到===内容规划结束===的格式输出！"
        
        return prompt
    
    async def execute(self, task: Dict) -> Dict:
        """执行丰富任务"""
        if task.get("action") == "enrich_outline":
            outline = task.get("outline", {})
            context = task.get("context", {})
            
            enriched_outline = await self.enrich_outline(outline, context)
            
            return {
                "status": "success",
                "enriched_outline": enriched_outline
            }
        
        elif task.get("action") == "analyze_section":
            section_info = task.get("section_info", {})
            context = task.get("context", {})
            
            analysis = await self.analyze_section_needs(section_info, context)
            
            return {
                "status": "success",
                "analysis": analysis
            }
        
        else:
            return {
                "status": "error",
                "message": "未知的任务类型"
            }

class WriterAgent(BaseAgent):
    """撰写智能体：负责具体章节内容撰写"""
    
    def __init__(self, name: str, config: AgentConfig, llm_factory: LLMFactory, db: AcademicPaperDatabase, 
                 section_id: str, section_guidance: Dict = None, citation_manager: CitationManager = None):
        super().__init__(name, config, llm_factory)
        self.db = db
        self.section_id = section_id  # 负责的章节ID
        self.section_guidance = section_guidance or {}  # 章节指引
        self.similarity_calculator = None
        self.citation_manager = citation_manager or CitationManager()
    
    async def setup_similarity_calculator(self, topic: str, subtopics: List[str] = None):
        """设置相似度计算器"""
        self.similarity_calculator = EnhancedSimilarityCalculator(topic, subtopics)
    
    async def write_section_iteratively(self, section_info: Dict, main_topic: str, subtopics: List[str] = None, global_outline_summary: Dict = None) -> Dict:
        """撰写特定章节内容，使用迭代优化机制，参考多批次材料逐步完善内容"""
        print(f"✍️ 开始迭代撰写章节: {section_info.get('title', '未命名章节')}")
        
        # 确保有相似度计算器
        if not self.similarity_calculator:
            await self.setup_similarity_calculator(main_topic, subtopics)
        
        # 🆕 一次性获取所有章节相关材料（现在返回分类格式）
        categorized_materials = await gather_section_materials(
            section_info, self.db, main_topic, EnhancedSimilarityCalculator, 
            self.call_llm, self.citation_manager,
            max_texts=180, max_equations=30, max_figures=60, max_tables=60  # 增加材料数量以支持迭代
        )
        
        # 🆕 【修复】简化方案：只为实际要使用的材料创建编号映射
        # 先选择写作需要的材料，再为这些材料创建编号，确保数量精确匹配
        
        # 选择总材料池（用于初始写作 + 迭代优化）
        # 初始：60文本+10公式+20图片+20表格 = 110条
        # 迭代：每轮60文本+10公式+20图片+20表格 = 110条/轮 × 3轮 = 330条
        # 总计：440条材料
        from utils import _select_materials_proportionally
        total_selected_materials = _select_materials_proportionally(
            categorized_materials, 
            target_texts=240,   # 60 + 60*3 = 240
            target_equations=40,  # 10 + 10*3 = 40
            target_figures=80,   # 20 + 20*3 = 80
            target_tables=80     # 20 + 20*3 = 80
        )
        
        
        # 🆕 只为实际选中的材料创建编号映射，确保数量精确匹配
        all_numbered_materials = create_numbered_materials_mapping(total_selected_materials, section_info)
        
        # 🆕 重新构建分类材料，基于实际选中的材料
        updated_categorized_materials = {
            "texts": [m for m in total_selected_materials if m.get("content_type") in ["text", "texts"] or m.get("content_type") is None],
            "equations": [m for m in total_selected_materials if m.get("content_type") in ["equation", "equations"]], 
            "figures": [m for m in total_selected_materials if m.get("content_type") in ["figure", "figures", "image", "images"]],
            "tables": [m for m in total_selected_materials if m.get("content_type") in ["table", "tables"]]
        }
        
        # 🆕 记录材料统计信息
        total_count = sum(len(materials) for materials in categorized_materials.values())
        selected_count = sum(len(materials) for materials in updated_categorized_materials.values())
        print(f"📊 实际使用材料: {selected_count} 条")
        print(f"📊 使用分类统计: 文本{len(updated_categorized_materials['texts'])}, 公式{len(updated_categorized_materials['equations'])}, 图片{len(updated_categorized_materials['figures'])}, 表格{len(updated_categorized_materials['tables'])}")
        
        # 1. 生成初始章节内容（使用精确选择的材料）
        initial_content = await self._create_initial_writing(
            section_info, updated_categorized_materials, all_numbered_materials, main_topic, subtopics, global_outline_summary
        )
        
        # 2. 迭代优化内容（使用精确选择的材料）
        final_content = await self._refine_writing_iteratively(
            initial_content, updated_categorized_materials, all_numbered_materials, section_info, main_topic, subtopics, global_outline_summary
        )
        
        # 🧹 清洗最终内容
        final_generated_content = clean_generated_content(final_content.get("content", ""))
        
        # 🆕 处理最终的引用信息
        final_citation_mapping = extract_citation_mapping(final_generated_content, all_numbered_materials)
        
        # 🆕 生成章节引用JSON文件
        section_citation_file = write_section_citations(
            section_info, final_citation_mapping, all_numbered_materials, main_topic
        )
        
        # 构建最终结果
        result = {
            "id": section_info.get("id", ""),
            "title": section_info.get("title", "未命名章节"),
            "content": final_generated_content,
            "status": "success",
            "subsections": section_info.get("subsections", []),
            "statistics": {
                "word_count": len(final_generated_content.split()),
                "material_count": len(total_selected_materials),
                "citations_used": len(final_citation_mapping),
                "iterations_completed": final_content.get("iterations_completed", 1)
            },
            # 🆕 新增引用相关信息
            "citation_info": {
                "citation_file": section_citation_file,
                "citations_count": len(final_citation_mapping),
                "materials_referenced": list(final_citation_mapping.keys())
            }
        }
        
        print(f"✅ 章节 '{section_info.get('title', '未命名章节')}' ")
        return result

    async def write_section(self, section_info: Dict, main_topic: str, subtopics: List[str] = None, global_outline_summary: Dict = None) -> Dict:
        """撰写特定章节内容的统一入口，默认使用迭代优化机制"""
        return await self.write_section_iteratively(section_info, main_topic, subtopics, global_outline_summary)
    
    def _build_citation_aware_prompt(self, section_info: Dict, numbered_materials: Dict, main_topic: str, subtopics: List[str] = None, global_outline_summary: Dict = None) -> str:
        """构建带引用指导的CoT提示词"""
        # 章节标识信息
        chapter_id = section_info.get("id", "")
        chapter_title = section_info.get("title", "未命名章节")
        chapter_desc = section_info.get("description", "")
        
        # 优先从section_info中获取丰富的章节信息，如果没有则从section_guidance中获取
        content_guide = section_info.get("content_guide", "") or self.section_guidance.get("content_guide", "")
        keywords = section_info.get("keywords", []) or self.section_guidance.get("keywords", [])
        research_focus = section_info.get("research_focus", []) or self.section_guidance.get("research_focus", [])
        
        # 开始构建提示词
        prompt = f"""你是一位资深的学术研究者，正在撰写一篇高水平的学术综述。请基于提供的研究材料和写作指引，为章节"{chapter_id} {chapter_title}"撰写符合国际学术期刊标准的综述内容。

        【综述写作核心理念】
        请牢记，综述不是文献的简单堆砌，而是对研究领域的深度分析和理论建构。你需要：
        - 系统梳理该领域的研究发展脉络和理论演进
        - 客观评价不同研究的贡献、局限和争议点
        - 综合多方观点，形成更深层的学术理解
        - 识别研究空白，指出未来发展方向
        - 用流畅的学术语言构建连贯的理论叙事

                【章节基本信息】
                章节编号: {chapter_id}
                章节标题: {chapter_title}
                章节思路: {chapter_desc}

                【章节内容指引】
                {content_guide}

                【关键词】
                {', '.join(keywords) if keywords else '无特定关键词'}

                【重点研究领域】
                """
        
        # 添加研究重点
        if research_focus:
            for i, focus in enumerate(research_focus, 1):
                prompt += f"{i}. {focus}\n"
        else:
            prompt += "无特定研究重点要求\n"
        
        # 使用utils中的函数生成子章节详细内容指引
        prompt += build_subsection_guidance(section_info, self.section_guidance)

        # 添加全局综述上下文信息
        if main_topic:
            prompt += f"\n【全局综述上下文】\n"
            prompt += f"综述主题: {main_topic}\n"
            if subtopics:
                prompt += f"综述子主题: {', '.join(subtopics)}\n"
        
        # 添加全局结构概览信息
        if global_outline_summary:
            prompt += f"""
        【全局综述结构概览】
        请注意，你正在撰写的是一个完整综述的一部分。以下是整个综述的结构概览，请确保你的内容与整体结构保持一致，避免重复其他章节的内容，并适当引用或呼应相关章节：

        综述总体框架：
        {_format_global_outline_for_prompt(global_outline_summary)}
        
        【写作提醒】
        1. 注意你当前撰写的章节在整体结构中的位置和作用
        2. 避免与其他章节内容重复，保持内容边界清晰
        3. 适当提及与其他章节的逻辑关系（如"如第X章所述"、"将在第Y章详细讨论"等）
        4. 确保内容深度和详细程度与整个综述的学术水平保持一致
        """

        # 使用CoT技术指导撰写过程
        prompt += f"""
                【综述写作思考步骤】作为一篇学术综述的撰写者，你需要按照以下专业的综述写作逻辑进行深度思考，然后撰写出符合综述规范的高质量章节内容：

                步骤1: 确立综述视角和理论框架
                思考: 这个章节在整个综述中承担什么理论功能？需要建立怎样的分析框架？应该从哪个学术视角来审视和组织现有文献？如何确保内容符合综述的系统性和批判性要求？

                步骤2: 梳理文献发展脉络和研究演进
                思考: 该领域的研究是如何发展演进的？有哪些重要的理论转折点和方法论突破？不同时期的研究重点和方法有何变化？如何构建清晰的发展时间线？

                步骤3: 识别和归纳核心观点流派
                思考: 现有研究中存在哪些主要的理论观点和学术流派？这些观点之间有何异同？如何对不同观点进行客观的归纳和分类？哪些是主流观点，哪些是新兴或争议性观点？

                步骤4: 深度分析图表数据和实证证据
                思考: 提供的图片、表格、公式等材料蕴含了哪些关键信息？这些数据如何支撑或质疑现有理论？不同研究的实验设计和数据表现有何差异？如何从这些图表中提炼出有价值的学术洞察？

                步骤5: 进行批判性综合和理论建构
                思考: 如何对现有研究进行客观评价？存在哪些研究空白、方法论局限或理论争议？如何综合不同观点形成更深层的理论理解？这个章节应该为整个综述贡献怎样的学术价值？

                步骤6: 构建连贯的学术叙事
                思考: 如何用流畅的学术语言将复杂的理论观点和研究发现串联成连贯的叙事？如何确保每个段落都承载明确的学术功能？如何在保持客观性的同时体现批判性思维？

                【综述写作核心要求】
                1. 综述本质: 这是一篇学术综述，不是教科书或工作报告。必须体现对现有研究的系统梳理、批判分析和理论建构
                2. 文献综合: 不仅要描述各项研究，更要分析研究间的关联、矛盾和互补关系，构建知识的整体图景
                3. 批判视角: 对每项研究都要进行客观评价，指出其贡献、局限和争议，避免简单的文献堆砌
                4. 理论深度: 从具体研究中抽象出理论规律，识别研究趋势，提出有见地的学术观点
                5. 发展脉络: 清晰呈现研究领域的历史发展、现状分析和未来展望
                6. 学术严谨: 确保内容准确、客观，逻辑严密，术语使用规范
                7. 自然表达: 采用流畅的学术写作风格，每个段落围绕一个核心学术观点展开，段落间逻辑关系清晰自然
                8. 🔥【强制】多模态材料深度融合: 必须深度分析和解读所有提供的图片、表格、公式等材料，将其作为论证的核心证据融入学术论述中，不得遗漏任何类型的材料
                9. 🔥【强制】均衡引用策略: 引用必须覆盖文本、公式、图片、表格所有类型，每种类型都要有充分的引用，避免过度依赖文本材料，确保每个引用都有明确的学术功能
                10. 内容丰富: 撰写详尽的内容（至少8000字），深度挖掘每个主题的学术内涵，避免浅尝辄止

                【输出格式与写作风格】
                请严格按照以下学术综述的标准格式和写作风格来输出内容：
                
                "
                # {chapter_id} {chapter_title}

                [章节开篇段落：用2-3个自然段介绍本章节的主题背景、研究意义和结构安排，体现综述的整体性和系统性]

                ## {chapter_id}.1 第一个二级子章节标题
                
                [引言段落：介绍本子章节的核心问题和分析框架]
                
                ### {chapter_id}.1.1 第一个三级子章节标题
                
                ...

                ### {chapter_id}.1.2 第二个三级子章节标题
                
                ...
                
                ## {chapter_id}.2 第二个二级子章节标题
                
                [保持相同的写作风格和结构...]
                "
                
                【关键写作风格要求】
                1. **自然段落结构**：每个段落围绕一个明确的学术观点或研究发现展开，长度适中
                2. **流畅的学术叙事**：段落间用过渡性语句连接，如"基于以上分析"、"与此形成对比的是"、"进一步的研究表明"等
                3. **有机的引用融合**：引用不是生硬插入，而是自然地嵌入学术论述中。应该避免集中式引用，尽量分散，目的是支撑多元观点
                4. **批判性的语言表达**：使用"然而"、"值得注意的是"、"尽管如此"等表达来体现学术思辨
                5. **严禁条目化表达**：绝对不使用"1.、2.、3.","首先、其次、最后","- 内容1 - 内容2 - 内容3"或"* 内容1 * 内容2 * 内容3"等明显的列举形式
                6. **连贯的逻辑线索**：确保读者能清晰地跟随你的学术论证脉络
                
                注意：不需要在章节末尾添加参考文献列表
                """

        # 🆕 添加强制性多模态材料引用指导
        prompt += f"""
        
        ⚠️⚠️【强制要求：多模态材料引用与深度分析】⚠️⚠️
        这不是可选要求，而是必须严格执行的写作标准！你必须在综述内容中积极引用和深度分析所有类型的研究材料。
        
        ✅ 引用分布应该均匀，避免过度集中在文本材料

        【根据参考材料来确定引用格式】
        📝 文本材料：[{chapter_id}-文本1]、[{chapter_id}-文本2] ...
        🧮 公式材料：[{chapter_id}-公式1]、[{chapter_id}-公式2] ...
        📊 图片材料：[{chapter_id}-图片1]、[{chapter_id}-图片2] ...
        📋 表格材料：[{chapter_id}-表格1]、[{chapter_id}-表格2] ...
        
        ✅ 正确示例（多模态综合引用）：
        "深度学习的理论基础建立在反向传播算法[{chapter_id}-公式1]和梯度下降优化[{chapter_id}-公式2]之上。如[{chapter_id}-图片1]所示的神经网络架构演进历程，从简单的多层感知机发展到复杂的Transformer结构，体现了该领域的技术进步轨迹。[{chapter_id}-表格1]的性能对比数据清晰展现了不同模型在标准数据集上的表现差异，其中Transformer在BLEU得分上的优势特别明显。然而，正如[{chapter_id}-文本1]指出的计算复杂度挑战，这些先进模型的实际部署仍面临诸多限制。"

        【引用质量检查清单】
        在完成写作后，请自检以下要点：
        ☑️ 是否引用了所有类型的材料？
        ☑️ 公式引用是否用于阐述理论原理？
        ☑️ 图片引用是否用于说明架构、流程或结果？
        ☑️ 表格引用是否用于数据分析和性能对比？
        ☑️ 引用分布是否均匀，避免过度依赖某类材料？

        ⚠️ 最终警告：如果你的综述内容缺乏对公式、图片、表格的充分引用和分析，将被视为不合格的学术写作！

        """
        
        # 添加相关研究材料，按类型和来源分组展示
        prompt += "\n【相关研究材料】\n"
        
        # 从numbered_materials中按类型分组
        text_materials = []
        equation_materials = []
        figure_materials = []
        table_materials = []
        
        print(f"🔍 开始按类型分组 {len(numbered_materials)} 条材料...")
        
        # 统计信息
        type_counts = {"text": 0, "equation": 0, "figure": 0, "table": 0}
        
        for material_id, material_info in numbered_materials.items():
            material_type = material_info.get("type", "text")
            type_counts[material_type] = type_counts.get(material_type, 0) + 1
            
            # 重构材料对象以适配原有显示逻辑
            material_obj = {
                "content": material_info["content"],
                "paper": material_info["paper"],
                "relevance_score": material_info["relevance_score"],
                "source": "章节特定搜索",  # 默认来源
                "material_id": material_id  # 添加材料标识符
            }
            
            if material_type == "text":
                text_materials.append(material_obj)
            elif material_type == "equation":
                equation_materials.append(material_obj)
            elif material_type == "figure":
                figure_materials.append(material_obj)
            elif material_type == "table":
                table_materials.append(material_obj)
        
        # 显示文本材料
        if text_materials:
            prompt += "\n **文本材料**\n"
            for i, material in enumerate(text_materials, 1):
                material_id = material.get("material_id", f"文本{i}")
                relevance = material.get("relevance_score", 0)
                paper = material.get("paper", "未知来源")
                content = material.get("content", "")
                # 文本材料：使用前1000字符
                display_content = content[:1000]
                if len(content) > 1000:
                    display_content += "..."
                prompt += f"{material_id} (相关度: {relevance:.2f}, 来源ID: {paper[:3]}):\n{display_content}\n\n"
        
        # 显示公式材料
        if equation_materials:
            prompt += "\n **相关公式**\n"
            for i, material in enumerate(equation_materials, 1):
                material_id = material.get("material_id", f"公式{i}")
                relevance = material.get("relevance_score", 0)
                paper = material.get("paper", "未知来源")
                content = material.get("content", "")
                # 公式材料：使用完整内容（完整上文+公式内容+完整下文）
                display_content = content  # 不进行字符限制，使用完整内容
                prompt += f"{material_id} (相关度: {relevance:.2f}, 来源ID: {paper[:3]}):\n{display_content}\n\n"
        
        # 显示图表材料
        if figure_materials:
            prompt += "\n **图片资料**\n"
            for i, material in enumerate(figure_materials, 1):
                material_id = material.get("material_id", f"图片{i}")
                relevance = material.get("relevance_score", 0)
                paper = material.get("paper", "未知来源")
                content = material.get("content", "")
                # 图片材料：使用前1000字符
                display_content = content[:1000]
                if len(content) > 1000:
                    display_content += "..."
                prompt += f"{material_id} (相关度: {relevance:.2f}, 来源ID: {paper[:3]}):\n{display_content}\n\n"
        
        # 显示表格材料
        if table_materials:
            prompt += "\n **表格数据**\n"
            for i, material in enumerate(table_materials, 1):
                material_id = material.get("material_id", f"表{i}")
                relevance = material.get("relevance_score", 0)
                paper = material.get("paper", "未知来源")
                content = material.get("content", "")
                # 表格材料：使用前1000字符，并清理大量数字
                display_content = content[:1000]
                if len(content) > 1000:
                    display_content += "..."
                # 🆕 清理包含大量数字的表格内容
                display_content = _clean_numeric_content(display_content)
                prompt += f"{material_id} (相关度: {relevance:.2f}, 来源ID: {paper[:3]}):\n{display_content}\n\n"
        
        # 材料统计信息
        total_materials = len(text_materials) + len(equation_materials) + len(figure_materials) + len(table_materials)
        prompt += f"\n📊 **材料统计**: 共{total_materials}条材料 (文本:{len(text_materials)}, 公式:{len(equation_materials)}, 图:{len(figure_materials)}, 表格:{len(table_materials)})\n"
        
        return prompt
    
    def _prepare_materials_for_writing(self, all_materials: List) -> List:
        """准备和排序所有写作材料，类似Enricher的材料准备逻辑"""
        return sorted(all_materials, key=lambda x: x.get("relevance_score", 0), reverse=True)
    
    async def _create_initial_writing(self, section_info: Dict, categorized_materials: Dict, all_numbered_materials: Dict, 
                                    main_topic: str, subtopics: List[str] = None, global_outline_summary: Dict = None) -> Dict:
        """第0轮：生成初始章节内容（按比例从各类型材料中取材料）"""
        print(f"📝 第0轮：生成初始章节内容（按比例从各类型材料中取材料）")
        
        # 🆕 按比例从各类型材料中取材料
        from utils import _select_materials_proportionally
        initial_materials = _select_materials_proportionally(categorized_materials, 
                                                                  target_texts=60, target_equations=10, 
                                                                  target_figures=20, target_tables=20)
        
        # 创建初始材料的编号映射（从全局映射中提取对应材料）
        initial_numbered_materials = {}
        
        # 🔧 使用内容匹配来找到对应的numbered材料
        initial_contents = set(mat.get("content", "")[:100] for mat in initial_materials)  # 使用前100字符作为唯一标识
        
        
        for material_id, material_info in all_numbered_materials.items():
            material_content_prefix = material_info.get("content", "")[:100]
            if material_content_prefix in initial_contents:
                initial_numbered_materials[material_id] = material_info
        
        
        # 🔍 统计匹配到的材料类型分布
        matched_types = {}
        for material_id, material_info in initial_numbered_materials.items():
            material_type = material_info.get("type", "unknown")
            matched_types[material_type] = matched_types.get(material_type, 0) + 1
        
        
        # 构建初始写作提示词
        initial_prompt = self._build_initial_writing_prompt(
            section_info, initial_numbered_materials, main_topic, subtopics, global_outline_summary
        )
        
        # 调用LLM生成初始内容
        print("🧠 正在调用LLM生成初始章节内容...")
        response = await self.call_llm(initial_prompt, task_type="initial_writing")
        
        content = response.get("content")
        if not content or content.startswith("❌ LLM调用失败"):
            error_msg = response.get("error", "未知错误")
            print(f"⚠️ 初始章节内容生成失败: {error_msg}")
            return {
                "content": f"内容生成失败: {error_msg}",
                "status": "error",
                "materials_used": 0,
                "iterations_completed": 0
            }
        
        # 清洗生成的内容
        initial_content = clean_generated_content(content)
        
        result = {
            "content": initial_content,
            "status": "success",
            "materials_used": len(initial_materials),
            "iterations_completed": 0,
            "quality_scores": {
                "academic_rigor": 7.0,  # 初始默认评分
                "content_completeness": 7.0,
                "literature_integration": 7.0,
                "argument_depth": 7.0,
                "expression_quality": 7.0,
                "overall_quality": 7.0
            }
        }
        
        print(f"✅ 初始内容生成完成，使用了{len(initial_materials)}条材料")
        return result
    
    def _build_initial_writing_prompt(self, section_info: Dict, numbered_materials: Dict, 
                                    main_topic: str, subtopics: List[str] = None, 
                                    global_outline_summary: Dict = None) -> str:
        """构建初始写作的提示词，直接使用_build_citation_aware_prompt的逻辑"""
        # 直接调用现有的_build_citation_aware_prompt函数，避免重复代码
        return self._build_citation_aware_prompt(section_info, numbered_materials, main_topic, subtopics, global_outline_summary)

    async def _refine_writing_iteratively(self, initial_content: Dict, categorized_materials: Dict, all_numbered_materials: Dict, 
                                        section_info: Dict, main_topic: str, subtopics: List[str] = None, 
                                        global_outline_summary: Dict = None) -> Dict:
        """迭代优化写作内容的核心逻辑"""
        current_content = initial_content
        max_iterations = 2
        
        # 计算总材料数量
        total_materials = sum(len(materials) for materials in categorized_materials.values())
        
        # 记录已使用的材料数量（按类型）
        used_materials_count = {
            "texts": 60,
            "equations": 10, 
            "figures": 20,
            "tables": 20
        }
        
        for iteration in range(1, max_iterations + 1):
            print(f"📝 第{iteration}轮：内容优化迭代")
            
            # 🆕 计算下一批材料的数量分配
            # 每轮使用约100个材料，按比例分配
            next_batch_targets = {
                "texts": 60,
                "equations": 10,
                "figures": 20, 
                "tables": 20
            }
            
            # 检查是否还有足够的材料
            remaining_materials = {
                material_type: len(materials) - used_materials_count[material_type]
                for material_type, materials in categorized_materials.items()
            }
            
            total_remaining = sum(remaining_materials.values())
            if total_remaining <= 0:
                print(f"⚠️ 材料已用完，停止迭代")
                break
            
            # 🆕 按比例从剩余材料中选择下一批
            from utils import _select_next_batch_materials
            current_materials = _select_next_batch_materials(
                categorized_materials, used_materials_count, next_batch_targets
            )
            
            if not current_materials:
                print(f"⚠️ 无法获取更多材料，停止迭代")
                break
                
            # 创建当前批次材料的编号映射
            current_numbered_materials = {}
            
            # 🔧 使用内容匹配来找到对应的numbered材料（与初始写作相同的逻辑）
            current_contents = set(mat.get("content", "")[:100] for mat in current_materials)
            
            for material_id, material_info in all_numbered_materials.items():
                material_content_prefix = material_info.get("content", "")[:100]
                if material_content_prefix in current_contents:
                    current_numbered_materials[material_id] = material_info
                        
            # 执行内容分析和优化
            refinement_result = await self._analyze_and_refine_content(
                current_content, current_materials, current_numbered_materials, section_info, main_topic, iteration, subtopics, global_outline_summary
            )
            
            if not refinement_result:
                print(f"❌ 第{iteration}轮优化失败，使用当前内容")
                break
            
            # 更新内容
            new_content = refinement_result.get("content")
            should_continue = refinement_result.get("should_continue", False)
            
            # 🔧 类型检查和更新
            if isinstance(new_content, dict):
                current_content = new_content
                current_content["iterations_completed"] = iteration
            else:
                print(f"⚠️ 内容更新异常，保持当前内容")
                # 保持 current_content 不变，但更新迭代次数
                current_content["iterations_completed"] = iteration
            
            # 🆕 更新已使用的材料计数
            for material_type, target_count in next_batch_targets.items():
                materials = categorized_materials.get(material_type, [])
                current_used = used_materials_count.get(material_type, 0)
                available = len(materials) - current_used
                actual_used = min(available, target_count)
                used_materials_count[material_type] += actual_used
            
            total_used = sum(used_materials_count.values())
            print(f"✅ 第{iteration}轮优化完成，累计已使用材料：{total_used}条")
            
            if not should_continue:
                print(f"🎯 达到优化目标，停止迭代")
                break
        
        final_iteration = min(iteration if 'iteration' in locals() else 0, max_iterations)
        print(f"🎯 内容优化完成，共进行了{final_iteration}轮迭代")
        
        # 确保最终结果包含迭代信息
        if not current_content.get("iterations_completed"):
            current_content["iterations_completed"] = final_iteration
        
        return current_content
    
    async def _analyze_and_refine_content(self, current_content: Dict, new_materials: List, new_numbered_materials: Dict,
                                        section_info: Dict, main_topic: str, iteration: int, 
                                        subtopics: List[str] = None, global_outline_summary: Dict = None) -> Dict:
        """
        基于新材料科学分析并优化写作内容，采用多维度评估与精准改进机制
        """
        chapter_id = section_info.get("chapter_id", "")
        # 🆕 获取重新编号的材料字典，用于后续的引用映射
        _, renumbered_materials = _format_materials_for_writing_prompt(new_numbered_materials, iteration)        
        # 构建内容分析优化提示词
        refinement_prompt = f"""你是资深的学术综述内容质量评估与优化专家，需要采用科学严谨的方法评估当前章节内容的质量，并基于新材料进行**增量式精准改进**。请参考国际期刊编辑标准，确保评估的客观性和改进的有效性。

【核心任务】
1. 多维度内容质量评估：客观评估当前章节内容的学术水平
2. 新材料价值挖掘：深度分析新材料对内容完善的**补充**潜力（不是替换）
3. 差异度分析：量化评估新材料融入后内容的**增量**变化程度
4. 增量精准改进：在**完全保持**现有内容基础上，针对性补充和增强
5. 科学迭代决策：基于量化指标决定是否继续迭代

【当前章节内容】
{_format_content_for_analysis(current_content)}

【章节写作指引】
{self._format_section_guidance_for_analysis(section_info)}

【全局综述框架】
{_format_global_context_for_analysis(main_topic, subtopics, global_outline_summary)}

【科学评估与优化框架】
请严格按照以下6个步骤进行分析：

步骤1: 多维度内容质量评估
评估维度：
- 学术严谨性：论证逻辑、引用准确性、观点客观性 (评分: 1-10)
- 内容完整性：章节结构完整度、核心要点覆盖度 (评分: 1-10)
- 文献融合度：引用自然性、材料利用充分性 (评分: 1-10)
- 🔥多模态材料引用：公式、图片、表格引用的充分性和均衡性 (评分: 1-10)
- 🔥图表分析深度：对多模态材料的分析解读深度和学术价值 (评分: 1-10)
- 论述深度：技术分析深度、批判性思维体现 (评分: 1-10)
- 表达质量：学术语言规范性、逻辑连贯性 (评分: 1-10)

步骤2: 新材料价值分析
分析要点：
- 技术贡献补充：新材料中包含哪些当前内容未涵盖的重要技术方法、算法或理论？
- 论证支撑增强：新材料如何为现有观点提供更强的证据支撑？
- 观点平衡优化：新材料是否提供了不同的学术观点，有助于平衡论述？
- 前沿发现融入：新材料反映了哪些最新的研究趋势？

步骤3: 内容差异度分析
分析要点：
- 核心观点补充：新材料为现有学术论点提供了哪些额外支撑和补充？
- 结构扩展程度：在保持原有结构基础上，新增了哪些段落或论述？
- **引用增量统计：仅统计新增引用的数量和类型**（严禁替换现有引用）
- 论述深度提升：技术分析深度的增强和丰富程度

步骤4: 内容缺陷与改进机会识别
增量改进分析（基于保持原有内容的前提）：
- **论证支撑加强**：当前内容哪些观点需要补充更多证据支撑？
- **内容覆盖补充**：基于新材料，发现哪些重要方向完全缺失需要新增？
- 🔥**多模态引用增强**：可以新增哪些公式、图片、表格等材料的引用来丰富内容？
- 🔥**图表分析深化**：哪些现有或新增的图表、公式需要补充更深入的学术解读？
- **引用密度优化**：哪些段落缺乏引用支撑，需要新增相关材料引用？
- **学术价值增值**：如何通过**新增内容**来提升章节的学术价值（保持原有价值）

步骤5: 增量改进效果量化评估 
量化指标（基于保持原有内容的增量改进）：
- **内容增量度**：新增内容占原有内容的比例 (百分比，通常应<30%)
- **质量增值程度**：通过新增内容能为整体学术水平带来多少提升？
- **新增价值密度**：新材料贡献的重要新观点和证据的价值密度
- **综合增值评分**：基于增量改进维度，预估整体价值提升(1-10分)

步骤6: 科学迭代决策
决策标准：
- 如果综合改进评分 ≥ 7分，或内容差异度 ≥ 30%，或发现重要内容缺陷，则继续迭代
- 如果综合改进评分 < 5分，且内容差异度 < 25%，且无重要缺陷，则停止迭代
- 如果介于两者之间，综合考虑材料质量和改进必要性做决策

【输出格式要求】
⚠️ 严格格式要求：以下标记是系统解析的核心依据，任何格式错误都会导致后续流程失败！

🚨【增量改进核心原则】
⚠️ 这是增量优化，不是重写！请严格遵循以下原则：

1. **引用保护机制**：
   - 🔒 绝对禁止删除或替换任何现有的引用，如 [{chapter_id}-文本12, {chapter_id}-文本26]、[{chapter_id}-公式5]、[{chapter_id}-图片1] 等
   - ✅ 只能新增引用，在合适位置补充新的材料引用，如 [{chapter_id}-文本61]、[{chapter_id}-公式11]、[{chapter_id}-图片21] 等
   - 🔒 保持引用格式，严格保持现有引用的格式和编号

2. **内容增量改进**：
   - ✅ 补充遗漏观点，发现内容缺口时，在合适位置添加新段落或句子
   - ✅ 增强论证支撑，为现有观点增加更多证据和引用支持
   - ✅ 丰富技术细节，在现有基础上补充更深入的技术分析
   - 🔒 保持原有结构，维持章节的组织逻辑和段落结构

请严格按照以下格式输出，确保所有标记完整：

===写作优化结果开始===

【增量优化后章节内容】
===章节内容开始===
[完整输出原有章节内容，并在合适位置插入基于新材料的补充内容。
===章节内容结束===

✅ **正确示例（增量模式）**：
保留: "通过增加低资源语言的数据量来缓解数据不平衡问题 [{chapter_id}-文本12, {chapter_id}-文本26]。"
新增: "此外，[{chapter_id}-文本61] 进一步指出，通过动态词汇调整策略可以实现更精准的资源分配。"

【是否继续迭代】
是/否

【增量改进分析】
内容增量度: [百分比]% (新增内容占原有内容的比例)
主要新增点: [具体描述新增的内容和观点]
质量增值度: [描述通过新增内容带来的质量提升]
新增引用统计: [文本X条, 公式X条, 图片X条, 表格X条]

【多维度内容质量评估】
学术严谨性: [评分]/10
内容完整性: [评分]/10
文献融合度: [评分]/10
多模态材料引用: [评分]/10
图表分析深度: [评分]/10
论述深度: [评分]/10
表达质量: [评分]/10
综合质量: [平均分]/10

===写作优化结果结束===

【必须出现的格式标记】
✓ ===写作优化结果开始=== (开头标记)
✓ ===写作优化结果结束=== (结尾标记)
✓ ===章节内容开始=== 和 ===章节内容结束=== (内容标记)
✓ 【是否继续迭代】(决策标记)
✓ 所有评估维度都有量化评分

【新增研究材料】
{_format_materials_for_writing_prompt(new_numbered_materials, iteration)[0]}
"""

        response = await self.call_llm(refinement_prompt, task_type=f"writing_refinement_iter{iteration}")
        
        content = response.get("content")
        if not content or content.startswith("❌ LLM调用失败"):
            print(f"⚠️ 第{iteration}轮内容优化LLM调用失败")
            return None
        
        # 解析优化结果，传递重新编号的材料字典用于引用映射
        return _parse_writing_refinement_response(content, iteration, current_content, renumbered_materials)
    
   
    def _format_section_guidance_for_analysis(self, section_info: Dict) -> str:
        """格式化章节写作指引用于分析"""
        # 获取章节指引信息
        content_guide = section_info.get("content_guide", "") or self.section_guidance.get("content_guide", "")
        keywords = section_info.get("keywords", []) or self.section_guidance.get("keywords", [])
        research_focus = section_info.get("research_focus", []) or self.section_guidance.get("research_focus", [])
        
        formatted = f"""
章节编号: {section_info.get("id", "")}
章节标题: {section_info.get("title", "未命名章节")}
章节思路: {section_info.get("description", "")}

【内容指引】
{content_guide}

【关键词】
{', '.join(keywords) if keywords else '无特定关键词'}

【重点研究领域】
"""
        
        if research_focus:
            for i, focus in enumerate(research_focus, 1):
                formatted += f"{i}. {focus}\n"
        else:
            formatted += "无特定研究重点要求\n"
        
        # 使用utils中的函数生成子章节详细内容指引
        from utils import build_subsection_guidance
        formatted += build_subsection_guidance(section_info, self.section_guidance)
        
        return formatted
    
    
    async def execute(self, task: Dict) -> Dict:
        """执行撰写任务"""
        if task.get("action") == "write_section":
            section_info = task.get("section_info", {})
            subtopics = task.get("subtopics", [])
            main_topic = task.get("main_topic", "")
            global_outline_summary = task.get("global_outline_summary", {})
            # 将section_info与self.section_guidance合并，确保最新的指引被使用
            if not self.section_guidance and section_info:
                self.section_guidance = {
                    "content_guide": section_info.get("content_guide", ""),
                    "keywords": section_info.get("keywords", []),
                    "research_focus": section_info.get("research_focus", []),
                    "subsections": {}
                }
                
                # 处理子章节
                for subsection in section_info.get("subsections", []):
                    subsection_id = subsection.get("id", "")
                    if subsection_id:
                        self.section_guidance["subsections"][subsection_id] = {
                            "content_guide": subsection.get("content_guide", ""),
                            "key_points": subsection.get("key_points", []),
                            "writing_guide": subsection.get("writing_guide", "")
                        }
            
            result = await self.write_section(section_info, main_topic, subtopics, global_outline_summary)
            return {
                "status": "success",
                "result": result
            }
        
        else:
            return {
                "status": "error",
                "message": "未知的任务类型"
            }


class InterpreterAgent(BaseAgent):
    """
    解释器智能体：负责解析和标准化用户输入的主题和次要主题
    
    功能：
    - 将中文输入翻译为英文学术术语
    - 将句子形式的输入提取为核心关键词
    - 标准化不规范的输入格式
    - 生成适合数据库查询的关键词（限制10个以内）
    """
    
    def __init__(self, name: str, config: AgentConfig, llm_factory: LLMFactory, db: AcademicPaperDatabase):
        """
        初始化解释器智能体
        
        Args:
            name: 智能体名称
            config: 智能体配置
            llm_factory: LLM工厂实例
            db: 学术论文数据库实例
        """
        super().__init__(name, config, llm_factory)
        self.db = db
    
    async def execute(self, task: Dict) -> Dict:
        """执行解释器任务"""
        if task.get("action") == "interpret_topic":
            topic = task.get("topic", "")
            subtopics = task.get("subtopics", [])
            
            # 解析和标准化用户输入
            result = await self.interpret_user_input(topic, subtopics)
            
            return {
                "status": "success",
                "standardized_topic": result.get("standardized_topic"),
                "standardized_subtopics": result.get("standardized_subtopics"),
                "analysis": result.get("analysis"),
                "original_input": {
                    "topic": topic,
                    "subtopics": subtopics
                }
            }
        
        else:
            return {
                "status": "error",
                "message": "未知的任务类型"
            }
    
    async def interpret_user_input(self, topic: str, subtopics: List[str] = None) -> Dict:
        """
        解析和标准化用户输入的主题和次要主题
        
        Args:
            topic: 用户输入的主要主题
            subtopics: 用户输入的次要主题列表
            
        Returns:
            标准化后的主题和关键词
        """
        print(f"🔄 正在解析用户输入: 主题='{topic}', 次要主题={subtopics}")
        
        # 构建CoT提示词，引导LLM分步分析用户输入
        subtopics_str = ", ".join(subtopics) if subtopics else "无"
        
        cot_prompt = f"""你是一个专业的学术综述关键词策略专家，专门为生成高质量学术综述设计关键词检索策略。你的任务是将用户输入转换为一套完整的、适合综述写作的英文学术关键词体系。

【核心使命】
为学术综述生成全面、准确、具有良好检索效果的关键词集合，确保能够覆盖该研究领域的核心内容、前沿发展、相关方法和交叉应用。

【用户输入】
主要主题: "{topic}"
次要主题: {subtopics_str}

【综述关键词生成策略】
请严格按照以下5个步骤进行深度分析：

步骤1: 综述课题领域定位与边界分析
思考要点：
- 这个课题属于哪个主要学科领域？（如计算机科学、材料科学、生物医学等）
- 该领域在近5年的主要研究热点和发展趋势是什么？
- 课题的研究边界在哪里？涉及哪些相关的交叉学科？
- 从综述的角度看，需要覆盖该领域的哪些核心维度？
- 用户输入的表述是否准确反映了真实的学术研究方向？

步骤2: 综述内容框架构建与关键词维度规划
思考要点：
- 写这个主题的综述通常需要涵盖哪些核心内容板块？
- 每个板块对应的核心概念和技术术语是什么？
- 需要包含哪些基础理论、核心方法、技术路径和应用场景？
- 该领域的标志性技术、重要算法、关键评估指标是什么？
- 有哪些重要的研究方法、实验设计、数据集需要关注？
- 相关的前沿技术、新兴方向、未来趋势关键词是什么？

步骤3: 学术术语标准化与国际化表达
思考要点：
- 该领域在顶级期刊（如Nature、Science、Cell等）和顶会中的标准英文表述是什么？
- IEEE、ACM、Springer等主要学术机构的官方术语如何？
- 避免使用过于宽泛（如"AI"、"machine learning"）或过于狭窄的术语
- 优先选择在Web of Science、Scopus等数据库中检索效果好的术语
- 确保术语的时效性，优先使用当前主流的表达方式

步骤4: 检索覆盖度优化与关键词组合策略
思考要点：
- 主题词应该是该领域最核心、最具代表性的术语
- 10个次要关键词应该形成完整的知识图谱，覆盖：
  * 核心技术和方法（3-4个）
  * 重要应用领域（2-3个）
  * 评估指标和标准（1-2个）
  * 相关技术和交叉领域（2-3个）
- 关键词之间应该有逻辑关联，能够相互补充和强化检索效果
- 考虑同义词和相关词的检索覆盖，避免重要文献遗漏

步骤5: 综述质量保证与关键词验证
思考要点：
- 这套关键词能否检索到该领域的奠基性论文和最新进展？
- 是否能够涵盖该主题综述应该包含的所有重要子方向？
- 关键词的学术层次和专业深度是否匹配目标综述的水平？
- 是否避免了可能导致大量无关文献的泛化词汇？
- 这套关键词在主流学术数据库中的检索效果预期如何？

【输出格式要求】
请严格按照以下格式输出结果：

===解析结果开始===

【综述核心主题】
[一个精确的英文学术术语，代表该综述的核心研究方向和主要焦点]

【综述关键词矩阵】
核心技术方法: [3-4个核心技术/方法术语，逗号分隔]
重要应用领域: [2-3个主要应用场景/领域术语，逗号分隔] 
评估与标准: [1-2个评估指标/标准术语，逗号分隔]
交叉与前沿: [2-3个相关技术/前沿方向术语，逗号分隔]

【综述策略分析】
领域定位与边界: [说明该课题的学科归属和研究范围边界]
内容框架规划: [描述综述应涵盖的主要内容板块和逻辑结构]
检索策略优化: [解释关键词选择的检索覆盖度和精确性考虑]
质量保证机制: [说明这套关键词如何确保综述的完整性和权威性]

===解析结果结束===

【思考过程】
步骤1分析: [领域定位与边界分析的具体思考]
步骤2分析: [内容框架构建的具体思考]
步骤3分析: [术语标准化的具体思考]
步骤4分析: [检索优化的具体思考]
步骤5分析: [质量验证的具体思考]

【关键要求】
1. 所有关键词必须是英文学术标准术语
2. 关键词总数严格控制为11个（1个主题 + 10个分类关键词）
3. 关键词应具备国际期刊/会议级别的学术规范性
4. 优先使用在Scopus、Web of Science等主流数据库中检索效果好的术语
5. 关键词之间应形成完整的知识图谱，覆盖综述写作的全部必要维度
6. 避免过于泛化（如"technology"、"research"）和过于具体的术语
7. 确保关键词的时效性，反映该领域的当前发展状态"""

        # 调用LLM进行解析，并记录任务类型
        response = await self.call_llm(cot_prompt, task_type="topic_interpretation")
        
        content = response.get("content")
        if not content or content.startswith("❌ LLM调用失败"):
            error_msg = f"主题解析失败: {response.get('error', '未知错误')}"
            raise ValueError(error_msg)
        
        # 解析响应
        try:
            result = _parse_interpretation_response(content)
            
            # 验证结果
            standardized_topic = result.get("standardized_topic", "").strip()
            standardized_subtopics = result.get("standardized_subtopics", [])
            
            if not standardized_topic:
                print(f"⚠️ 警告: 主题标准化失败，使用原始输入")
                standardized_topic = topic
            
            # 确保关键词总数不超过10个
            if len(standardized_subtopics) > 10:
                print(f"⚠️ 警告: 次要主题过多({len(standardized_subtopics)}个)，截取前10个")
                standardized_subtopics = standardized_subtopics[:10]
            
            return {
                "standardized_topic": standardized_topic,
                "standardized_subtopics": standardized_subtopics,
                "analysis": result.get("analysis", ""),
                "raw_response": content
            }
            
        except Exception as e:
            print(f"⚠️ 解析响应时出错: {e}，使用原始输入")
            return {
                "standardized_topic": topic,
                "standardized_subtopics": subtopics or [],
                "analysis": f"解析失败，使用原始输入: {str(e)}",
                "raw_response": content
            }
