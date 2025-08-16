# -*- coding: utf-8 -*-
"""
学术综述多智能体生成系统

脚本目标任务：使用多智能体架构自动生成学术综述

上下文：
- 基于ChromaDB向量数据库中已有的学术论文内容
- 通过多个专业智能体协作完成复杂的学术综述生成任务
- 支持交互模式和命令行参数模式两种运行方式

输入：
- 用户提供的主题和子主题
- API配置（密钥、模型选择等）
- 向量数据库路径和输出路径

执行步骤：
1. 初始化LLM工厂和数据库连接
2. 解析和标准化用户输入的主题
3. 创建并协调多个智能体（解释器、规划、丰富、撰写）
4. 按阶段生成综述（大纲创建→大纲丰富→内容撰写→内容整合）
5. 保存结果并记录生成耗时

输出：
- 学术综述Markdown文件
- 元数据JSON文件
- Word文档（如果可用）
- 生成统计信息和总耗时
"""

import os
import json
import asyncio
import argparse
from typing import List, Dict
from datetime import datetime
import traceback
import sys
from pathlib import Path
import re
import time  # 添加时间模块用于计时
try:
    from docx import Document
    from docx.shared import Inches, Pt
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.style import WD_STYLE_TYPE
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    print("⚠️ python-docx未安装，无法生成Word文档。可使用 'pip install python-docx' 安装。")
from multi_agent import (
    EnricherAgent,
    LLMFactory, 
    AcademicPaperDatabase, 
    ModelType,
    AgentConfig,
    PlannerAgent,
    WriterAgent,
    InterpreterAgent
)
from md_to_word_converter import convert_markdown_to_word

class MultiAgentCoordinator:
    """多智能体系统协调器，管理智能体间交互与任务分配"""
    
    def __init__(self, llm_factory: LLMFactory, db: AcademicPaperDatabase, config: Dict = None):
        """初始化多智能体协调器"""
        self.llm_factory = llm_factory
        self.db = db
        self.config = config or {}
        self.planner = None
        self.enricher = None
        self.interpreter = None  # 🆕 添加解释器智能体
        self.writers = {}
        self.topic = ""
        self.subtopics = []
        # 章节并发上限（章间并发、章内顺序），默认 6，可通过传入 config['writer_concurrency'] 调整
        self.writer_concurrency = int(self.config.get("writer_concurrency", 6))
        
    async def initialize_agents(self, topic: str, subtopics: List[str] = None):
        """初始化并配置所有智能体"""
        print(f"🚀 初始化智能体系统，主题：'{topic}'")
        self.topic = topic
        self.subtopics = subtopics or []
        
        # 创建规划智能体
        planner_config = AgentConfig(
            model_name=self.config.get("planner_model", ModelType.CLAUDE.value),
            temperature=0.7,
            max_tokens=15000,
            role_description="学术综述规划专家",
            system_message="你是学术综述规划专家，擅长组织和规划复杂的学术文献综述结构。"
        )
        self.planner = await self.create_planner(planner_config)
        
        # 创建丰富智能体
        enricher_config = AgentConfig(
            model_name=self.config.get("enricher_model", ModelType.CLAUDE.value),
            temperature=0.7,
            max_tokens=15000,
            role_description="学术综述编辑专家",
            system_message="你是学术综述编辑专家，擅长丰富大纲内容，为下游LLM提供详细的章节编写指引。"
        )
        self.enricher = await self.create_enricher(enricher_config)
            
    async def create_planner(self, config: AgentConfig) -> PlannerAgent:
        """创建规划智能体"""
        planner = PlannerAgent(
            name="规划智能体",
            config=config,
            llm_factory=self.llm_factory,
            db=self.db
        )
        return planner
    
    async def create_enricher(self, config: AgentConfig) -> EnricherAgent:
        """创建丰富智能体"""
        enricher = EnricherAgent(
            name="丰富智能体",
            config=config,
            llm_factory=self.llm_factory,
            db=self.db
        )
        return enricher
    
    async def create_interpreter(self, config: AgentConfig) -> InterpreterAgent:
        """创建解释器智能体"""
        interpreter = InterpreterAgent(
            name="解释器智能体",
            config=config,
            llm_factory=self.llm_factory,
            db=self.db
        )
        return interpreter
    
    async def create_writers(self, outline: Dict, writer_config: AgentConfig) -> Dict[str, WriterAgent]:
        """根据大纲创建撰写智能体，并分配对应章节的内容指引"""
        writers = {}
        
        # 检查outline结构，确保只处理顶层章节
        chapters = outline.get("chapters", {})
        
        # 处理两种可能的数据结构：列表或字典
        if isinstance(chapters, dict):
            # 如果chapters是字典，则直接使用字典的值
            main_chapters = list(chapters.values())
            print(f"🔍 识别出 {len(main_chapters)} 个一级章节（字典格式），为每个章节创建一个撰写智能体")
        elif isinstance(chapters, list):
            # 如果chapters是列表，筛选出真正的一级章节
            main_chapters = [c for c in chapters if c.get("id", "").isdigit() or len(c.get("id", "").split(".")) == 1]
            print(f"🔍 识别出 {len(main_chapters)} 个一级章节（列表格式），为每个章节创建一个撰写智能体")
        else:
            print("❌ 无法识别章节数据结构")
            return {}
        
        for chapter in main_chapters:
            chapter_id = chapter.get("id", "")
            chapter_title = chapter.get("title", "")
            
            # 收集章节的内容指引和其他信息
            chapter_guidance = {
                "content_guide": chapter.get("content_guide", ""),
                "keywords": chapter.get("keywords", []),
                "research_focus": chapter.get("research_focus", []),
                "subsections": {}
            }
            
            # 收集子章节的内容指引
            subsections = chapter.get("subsections", {})
            if isinstance(subsections, dict):
                # 如果subsections是字典，遍历其值
                for subsection_id, subsection in subsections.items():
                    if subsection_id:
                        chapter_guidance["subsections"][subsection_id] = {
                            "content_guide": subsection.get("content_guide", ""),
                            "key_points": subsection.get("key_points", []),
                            "writing_guide": subsection.get("writing_guide", "")
                        }
            elif isinstance(subsections, list):
                # 如果subsections是列表，保持原有逻辑
                for subsection in subsections:
                    subsection_id = subsection.get("id", "")
                    if subsection_id:
                        chapter_guidance["subsections"][subsection_id] = {
                            "content_guide": subsection.get("content_guide", ""),
                            "key_points": subsection.get("key_points", []),
                            "writing_guide": subsection.get("writing_guide", "")
                        }
            
            # 为每个一级章节创建一个撰写智能体
            writer = WriterAgent(
                name=f"撰写智能体-{chapter_id}",
                config=writer_config,
                llm_factory=self.llm_factory,
                db=self.db,
                section_id=chapter_id,
                section_guidance=chapter_guidance  # 传递章节指引
            )
            writers[chapter_id] = writer
            print(f"✅ 创建撰写智能体：章节 {chapter_id} - {chapter_title}")
            
        return writers
    
    def extract_global_outline_summary(self, enriched_outline: Dict) -> Dict:
        """
        从完整的enriched_outline中提取全局概览信息
        只保留章节标题、子章节标题和content_guide，控制token数量
        """
        global_summary = {
            "chapters": {}
        }
        
        chapters = enriched_outline.get("chapters", {})
        
        # 处理两种可能的数据结构：列表或字典
        if isinstance(chapters, dict):
            # 如果chapters是字典格式
            for chapter_id, chapter in chapters.items():
                chapter_summary = {
                    "id": chapter.get("id", chapter_id),
                    "title": chapter.get("title", ""),
                    "content_guide": chapter.get("content_guide", ""),
                    "subsections": {}
                }
                
                # 提取子章节标题
                subsections = chapter.get("subsections", {})
                if isinstance(subsections, dict):
                    for subsection_id, subsection in subsections.items():
                        chapter_summary["subsections"][subsection_id] = {
                            "id": subsection.get("id", subsection_id),
                            "title": subsection.get("title", "")
                        }
                elif isinstance(subsections, list):
                    for subsection in subsections:
                        subsection_id = subsection.get("id", "")
                        if subsection_id:
                            chapter_summary["subsections"][subsection_id] = {
                                "id": subsection_id,
                                "title": subsection.get("title", "")
                            }
                
                global_summary["chapters"][chapter_id] = chapter_summary
                
        elif isinstance(chapters, list):
            # 如果chapters是列表格式
            for chapter in chapters:
                chapter_id = chapter.get("id", "")
                if chapter_id:
                    chapter_summary = {
                        "id": chapter_id,
                        "title": chapter.get("title", ""),
                        "content_guide": chapter.get("content_guide", ""),
                        "subsections": {}
                    }
                    
                    # 提取子章节标题
                    subsections = chapter.get("subsections", {})
                    if isinstance(subsections, dict):
                        for subsection_id, subsection in subsections.items():
                            chapter_summary["subsections"][subsection_id] = {
                                "id": subsection.get("id", subsection_id),
                                "title": subsection.get("title", "")
                            }
                    elif isinstance(subsections, list):
                        for subsection in subsections:
                            subsection_id = subsection.get("id", "")
                            if subsection_id:
                                chapter_summary["subsections"][subsection_id] = {
                                    "id": subsection_id,
                                    "title": subsection.get("title", "")
                                }
                    
                    global_summary["chapters"][chapter_id] = chapter_summary
        
        return global_summary
    
    async def generate_survey(self, topic: str, subtopics: List[str] = None, output_path: str = None) -> Dict:
        """
        生成完整综述的主流程
        
        Args:
            topic: 综述主题
            subtopics: 子主题列表（可选）
            output_path: 输出路径（可选）
            
        Returns:
            生成的综述结果
        """
        # 🆕 第零阶段：解析和标准化用户输入
        print(f"🔄 第零阶段：解析用户课题")
        
        # 先创建解释器智能体
        interpreter_config = AgentConfig(
            model_name=self.config.get("interpreter_model", ModelType.CLAUDE.value),
            temperature=0.3,  # 较低的温度确保标准化的一致性
            max_tokens=15000,
            role_description="学术主题解释和标准化专家",
            system_message="你是学术主题解释和标准化专家，擅长将用户输入转换为标准的学术检索关键词。"
        )
        self.interpreter = await self.create_interpreter(interpreter_config)
        
        # 执行主题解析
        interpreter_result = await self.interpreter.execute({
            "action": "interpret_topic",
            "topic": topic,
            "subtopics": subtopics or []
        })
        
        if interpreter_result.get("status") != "success":
            print(f"⚠️ 课题解析失败，使用原始输入继续执行")
            standardized_topic = topic
            standardized_subtopics = subtopics or []
        else:
            standardized_topic = interpreter_result.get("standardized_topic")
            standardized_subtopics = interpreter_result.get("standardized_subtopics", [])
            print(f"✅ 课题解析完成：")
            print(f"   标准化主题: {standardized_topic}")
            print(f"   标准化次要主题: {standardized_subtopics}")

        # 1. 初始化智能体（使用标准化后的主题）
        await self.initialize_agents(standardized_topic, standardized_subtopics)
        
        # 2. 使用规划智能体创建大纲（使用标准化后的主题）
        print(f"📝 第一阶段：创建综述大纲")
        planning_result = await self.planner.execute({
            "action": "create_outline",
            "topic": standardized_topic,
            "subtopics": standardized_subtopics
        })
        
        if planning_result.get("status") != "success":
            raise RuntimeError("大纲创建失败")
        
        outline = planning_result.get("outline")
        context = planning_result.get("context")
        
        # 3. 使用丰富智能体丰富大纲
        print(f"📚 第二阶段：丰富综述大纲")
        enrichment_result = await self.enricher.execute({
            "action": "enrich_outline",
            "outline": outline,
            "context": context
        })
        
        if enrichment_result.get("status") != "success":
            raise RuntimeError("大纲丰富失败")
        
        enriched_outline = enrichment_result.get("enriched_outline")
        
        # 4. 创建撰写智能体
        writer_config = AgentConfig(
            model_name=self.config.get("writer_model", ModelType.CLAUDE.value),
            temperature=0.7,
            max_tokens=15000,  # 需要更大的上下文窗口来生成长内容
            role_description="学术综述撰写专家",
            system_message="你是学术综述撰写专家，擅长根据材料撰写专业、深入的学术内容。"
        )
        self.writers = await self.create_writers(enriched_outline, writer_config)
        
        # 提取全局概览信息
        global_outline_summary = self.extract_global_outline_summary(enriched_outline)
        
        # 在generate_survey方法中的并行撰写部分
        print(f"✍️ 第三阶段：撰写章节内容（共 {len(self.writers)} 个一级章节，并发上限 {self.writer_concurrency}）")
        semaphore = asyncio.Semaphore(self.writer_concurrency)

        async def run_writer_with_limit(writer_agent: WriterAgent, payload: Dict) -> Dict:
            async with semaphore:
                return await writer_agent.execute(payload)

        chapter_writing_tasks = []
        for chapter_id, writer in self.writers.items():
            # 找到对应的章节信息
            chapters = enriched_outline.get("chapters", {})
            if isinstance(chapters, dict):
                # 如果chapters是字典，直接通过键获取
                chapter_info = chapters.get(chapter_id, {})
            elif isinstance(chapters, list):
                # 如果chapters是列表，使用原来的查找方法
                chapter_info = next((c for c in chapters if c.get("id") == chapter_id), {})
            else:
                chapter_info = {}
            
            if not chapter_info:  # 跳过找不到对应章节的智能体
                print(f"⚠️ 未找到章节 {chapter_id} 的信息，跳过")
                continue
            
            payload = {
                "action": "write_section",
                "section_info": chapter_info,
                "main_topic": standardized_topic,
                "subtopics": standardized_subtopics,  # 使用标准化的次要主题
                "global_outline_summary": global_outline_summary  # 传递全局概览信息
            }

            # 包裹并发控制
            chapter_writing_tasks.append(run_writer_with_limit(writer, payload))

        # 等待所有章节完成写作（如需更稳健，可使用 return_exceptions=True）
        chapter_results = await asyncio.gather(*chapter_writing_tasks)
        
        # 处理写作结果
        chapter_contents = []
        for result in chapter_results:
            if result.get("status") == "success":
                chapter_contents.append(result.get("result"))
        
        # 6. 使用规划智能体整合结果
        print(f"📄 第四阶段：整合综述内容")
        integration_result = await self.planner.execute({
            "action": "integrate",
            "chapter_contents": chapter_contents,
            "topic": standardized_topic,  # 使用标准化主题
            "enriched_outline": enriched_outline,  # 传递详细大纲用于摘要生成
            "subtopics": standardized_subtopics  # 使用标准化次要主题
        })
        
        if integration_result.get("status") != "success":
            raise RuntimeError("内容整合失败")
        
        final_result = integration_result.get("result")
        
        # 🆕 在结果中保存原始输入和标准化结果
        final_result["interpretation_info"] = {
            "original_topic": topic,
            "original_subtopics": subtopics or [],
            "standardized_topic": standardized_topic,
            "standardized_subtopics": standardized_subtopics,
            "interpretation_analysis": interpreter_result.get("analysis", "") if interpreter_result.get("status") == "success" else "解析失败"
        }
        
        # 7. 保存结果
        if output_path:
            await self.save_results(final_result, output_path)
        
        print(f"🎉 综述生成完成: 共 {final_result['statistics']['chapter_count']} 章 ")
        
        return final_result


    async def save_results(self, survey: Dict, output_path: str):
        """
        保存生成结果
        
        Args:
            survey: 生成的综述结果
            output_path: 输出路径
        """
        # 确保目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存Markdown文件
        md_path = f"{output_path}_{timestamp}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            # 写入主标题
            f.write(f"# {self.topic}\n\n")
            
            # 处理摘要部分，避免重复标题
            abstract_content = survey.get('abstract', '')
            if abstract_content:
                # 检查摘要是否已经包含标题
                if abstract_content.strip().startswith("# 摘要") or abstract_content.strip().startswith("## 摘要"):
                    # 摘要已包含标题，直接写入
                    f.write(f"{abstract_content}\n\n")
                else:
                    # 摘要没有标题，添加标题
                    f.write("# 摘要\n\n")
                    f.write(f"{abstract_content}\n\n")
            
            # 写入关键词
            keywords = survey.get("keywords", [])
            if keywords:
                f.write("**关键词**: " + ", ".join(keywords) + "\n\n")
            
            # 写入正文
            f.write(survey.get("full_document", ""))
        
        # 保存元数据JSON
        meta_path = f"{output_path}_{timestamp}_meta.json"
        meta_data = {
            "topic": self.topic,
            "subtopics": self.subtopics,
            "timestamp": timestamp,
            "statistics": survey.get("statistics", {}),
            "keywords": survey.get("keywords", [])
        }
        
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, ensure_ascii=False, indent=2)
        
        print(f"📁 综述已保存到: {md_path}")
        print(f"📁 元数据已保存到: {meta_path}")
        # 生成Word文档
        if DOCX_AVAILABLE:
            word_path = f"{output_path}_{timestamp}.docx"
            
            # 构建完整的文档内容
            full_content = f"# {self.topic}\n\n"
            
            # 添加摘要部分
            if abstract_content:
                if not (abstract_content.strip().startswith("# 摘要") or abstract_content.strip().startswith("## 摘要")):
                    full_content += "# 摘要\n\n"
                full_content += f"{abstract_content}\n\n"
            
            # 添加关键词
            if keywords:
                full_content += "**关键词**: " + ", ".join(keywords) + "\n\n"
            
            # 添加正文
            full_content += survey.get("full_document", "")
            
            # 转换为Word文档（使用新的转换器）
            success = convert_markdown_to_word(full_content, word_path, self.topic)
            if success:
                print(f"📄 Word文档已保存到: {word_path}")
        else:
            print("⚠️ 无法生成Word文档，请安装python-docx: pip install python-docx")


async def generate_survey(
    topic: str,
    subtopics: List[str] = None,
    output_path: str = None,
    api_key: str = None,
    base_url: str = "https://openrouter.ai/api/v1",
    db_path: str = "./chroma_db",
    models: Dict[str, str] = None,
    log_dir: str = "./logs",
    verbose: bool = True
) -> Dict:
    """
    生成学术综述的便捷函数
    
    参数:
        topic: 综述主题
        subtopics: 子主题列表（可选）
        output_path: 输出路径（可选，默认为'./ma_output/{topic}')
        api_key: API密钥（可选，默认使用环境变量OPENROUTER_API_KEY）
        base_url: API基础URL（默认使用OpenRouter）
        db_path: 向量数据库路径（默认为'./chroma_db'）
        models: 各智能体使用的模型配置（可选）
        verbose: 是否显示详细日志
        
    返回:
        生成的综述结果字典
    """
    # 参数检查和默认值设置
    if not api_key:
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise ValueError("需要提供API密钥（通过参数或环境变量OPENROUTER_API_KEY）")
    
    if not output_path:
        # 创建安全的文件名
        safe_topic = "".join(c for c in topic if c.isalnum() or c in [' ', '_']).rstrip()
        safe_topic = safe_topic.replace(' ', '_')
        output_path = f"./ma_output/{safe_topic}"
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 默认模型配置
    if not models:
        models = {
            "interpreter_model": ModelType.CLAUDE.value,  # 🆕 新增解释器模型配置
            "planner_model": ModelType.CLAUDE.value,
            "enricher_model": ModelType.CLAUDE.value,
            "writer_model": ModelType.CLAUDE.value,
        }
    
    if verbose:
        print(f"📝 综述主题: {topic}")
        print(f"📌 子主题: {', '.join(subtopics) if subtopics else '无'}")
        print(f"💾 输出路径: {output_path}")
        print(f"🗄️ 数据库路径: {db_path}")
        print(f"🤖 使用模型: {models}")
    
    try:
        # 初始化LLM工厂
        llm_factory = LLMFactory(api_key=api_key, base_url=base_url, log_dir=log_dir)
        
        # 初始化向量数据库
        db = AcademicPaperDatabase(db_path=db_path)
        
        # 创建多智能体协调器
        coordinator = MultiAgentCoordinator(
            llm_factory=llm_factory,
            db=db,
            config=models
        )
        
        # 生成综述
        start_time = datetime.now()
        if verbose:
            print(f"⏱️ 开始生成: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        result = await coordinator.generate_survey(topic, subtopics, output_path)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60  # 转换为分钟
        
        if verbose:
            print(f"✅ 生成完成: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"⏱️ 总耗时: {duration:.2f} 分钟")
            print(f"🔑 关键词: {', '.join(result['keywords'])}")
        
        return result
    
    except Exception as e:
        if verbose:
            print(f"❌ 生成失败: {str(e)}")
            traceback.print_exc()
        raise e

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="多智能体学术综述生成系统")
    
    # parser.add_argument("--topic", "-t", type=str, default="Multimodal", help="综述主题")
    # parser.add_argument("--subtopics", "-s", type=str, default="Cross-modal Alignment, Multimodal Reasoning, Efficient Multimodal Training", help="子主题，用逗号分隔")
    parser.add_argument("--topic", "-t", type=str, default="Diffusion Models", help="综述主题")
    parser.add_argument("--subtopics", "-s", type=str, default="image generation, text-to-image, video generation, Image Synthesis, Style Transfer", help="子主题，用逗号分隔")
    parser.add_argument("--output", "-o", type=str, default="./ma_output/", help="输出文件路径")
    parser.add_argument("--api-key", "-k", type=str, default="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", help="OpenRouter API密钥")
    parser.add_argument("--base-url", "-u", type=str, default="https://openrouter.ai/api/v1", help="API基础URL")
    parser.add_argument("--db-path", "-d", type=str, default="D:/desktop/ZJU/acl300/academic_papers_db", help="向量数据库路径")
    parser.add_argument("--interpreter-model", type=str, default=ModelType.GEMINI.value, help="解释器智能体使用的模型")  # 🆕 新增
    parser.add_argument("--planner-model", type=str, default=ModelType.GEMINI.value, help="规划智能体使用的模型")
    parser.add_argument("--enricher-model", type=str, default=ModelType.GEMINI.value, help="丰富智能体使用的模型")
    parser.add_argument("--writer-model", type=str, default=ModelType.GEMINI.value, help="撰写智能体使用的模型")
    parser.add_argument("--log-dir", type=str, default="./logs", help="日志目录路径")

    return parser.parse_args()

async def interactive_mode():
    """交互模式，通过命令行与用户交互"""
    print("=" * 60)
    print("📚 多智能体学术综述生成系统")
    print("=" * 60)
    
    # 收集用户输入
    topic = input("请输入综述主题: ").strip()
    if not topic:
        print("❌ 错误: 主题不能为空")
        return
    
    subtopics_input = input("请输入子主题（用逗号分隔，可选）: ").strip()
    subtopics = [s.strip() for s in subtopics_input.split(",")] if subtopics_input else []
    
    output_path = input("请输入输出文件路径（可选，直接回车使用默认路径）: ").strip()
    
    api_key = input("请输入API密钥（可选，直接回车使用环境变量）: ").strip()
    
    db_path = input("请输入向量数据库路径（可选，直接回车使用默认路径）: ").strip()
    if not db_path:
        db_path = "./chroma_db"
    
    # 配置模型
    print("\n选择智能体使用的模型:")
    print("1. Claude (anthropic/claude-sonnet-4)")
    print("2. GPT-4o (openai/gpt-4o)")
    print("3. Gemini (google/gemini-2.5-pro)")
    print("4. Qwen (qwen/qwen2.5-vl-72b-instruct)")
    print("5. DeepSeek (deepseek/deepseek-chat-v3-0324)")
    
    model_map = {
        "1": ModelType.CLAUDE.value,
        "2": ModelType.GPT.value,
        "3": ModelType.GEMINI.value,
        "4": ModelType.QWEN.value,
        "5": ModelType.DS.value
    }
    
    interpreter_model = model_map.get(input("解释器智能体模型 (1-5，默认1): ").strip() or "1", ModelType.CLAUDE.value)  # 🆕 新增
    planner_model = model_map.get(input("规划智能体模型 (1-5，默认1): ").strip() or "1", ModelType.CLAUDE.value)
    enricher_model = model_map.get(input("丰富智能体模型 (1-5，默认1): ").strip() or "1", ModelType.CLAUDE.value)
    writer_model = model_map.get(input("撰写智能体模型 (1-5，默认1): ").strip() or "1", ModelType.CLAUDE.value)
    
    models = {
        "interpreter_model": interpreter_model,  # 🆕 新增
        "planner_model": planner_model,
        "enricher_model": enricher_model,
        "writer_model": writer_model,
    }
    
    # 确认生成
    print("\n" + "=" * 60)
    print(f"主题: {topic}")
    print(f"子主题: {', '.join(subtopics) if subtopics else '无'}")
    print(f"输出路径: {output_path or '默认'}")
    print(f"数据库路径: {db_path}")
    print(f"解释器模型: {interpreter_model}")  # 🆕 新增
    print(f"规划模型: {planner_model}")
    print(f"丰富模型: {enricher_model}")
    print(f"撰写模型: {writer_model}")
    print("=" * 60)
    
    # 记录交互开始时间（用于取消时显示耗时）
    interaction_start_time = time.time()
    
    confirm = input("\n确认开始生成? (y/n): ").strip().lower()
    if confirm != 'y':
        cancel_time = time.time()
        interaction_duration = cancel_time - interaction_start_time
        print(f"已取消生成（交互耗时{interaction_duration:.2f}秒）")
        return
    
    # 记录开始时间
    start_time = time.time()
    
    # 开始生成
    try:
        await generate_survey(
            topic=topic,
            subtopics=subtopics if subtopics else None,
            output_path=output_path,
            api_key=api_key,
            db_path=db_path,
            models=models
        )
        
        # 记录结束时间并计算总耗时（正常完成）
        end_time = time.time()
        total_time = end_time - start_time
        
        # 格式化时间显示（时分秒）
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = total_time % 60
        
        print(f"\n=== 综述生成完成 ===")
        print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
        print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
        
        if hours > 0:
            print(f"综述生成耗时{hours}小时{minutes}分钟{seconds:.2f}秒")
        elif minutes > 0:
            print(f"综述生成耗时{minutes}分钟{seconds:.2f}秒")
        else:
            print(f"综述生成耗时{seconds:.2f}秒")
            
    except Exception as e:
        # 记录结束时间并计算总耗时（异常情况）
        end_time = time.time()
        total_time = end_time - start_time
        
        # 格式化时间显示（时分秒）
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = total_time % 60
        
        print(f"\n=== 综述生成异常结束 ===")
        print(f"错误信息: {str(e)}")
        print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
        print(f"异常时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
        
        if hours > 0:
            print(f"运行耗时{hours}小时{minutes}分钟{seconds:.2f}秒")
        elif minutes > 0:
            print(f"运行耗时{minutes}分钟{seconds:.2f}秒")
        else:
            print(f"运行耗时{seconds:.2f}秒")
            
        print(f"❌ 生成过程中发生错误: {str(e)}")
        traceback.print_exc()

async def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 如果没有提供主题，进入交互模式
    if not args.topic:
        await interactive_mode()
        return
    
    # 使用命令行参数
    subtopics = [s.strip() for s in args.subtopics.split(",")] if args.subtopics else None
    
    models = {
        "interpreter_model": args.interpreter_model,  # 🆕 新增
        "planner_model": args.planner_model,
        "enricher_model": args.enricher_model,
        "writer_model": args.writer_model,
    }
    
    # 记录开始时间
    start_time = time.time()
    
    try:
        await generate_survey(
            topic=args.topic,
            subtopics=subtopics,
            output_path=args.output,
            api_key=args.api_key,
            base_url=args.base_url,
            db_path=args.db_path,
            models=models,
            log_dir=args.log_dir
        )
        
        # 记录结束时间并计算总耗时（正常完成）
        end_time = time.time()
        total_time = end_time - start_time
        
        # 格式化时间显示（时分秒）
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = total_time % 60
        
        print(f"\n=== 综述生成完成 ===")
        print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
        print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
        
        if hours > 0:
            print(f"综述生成耗时{hours}小时{minutes}分钟{seconds:.2f}秒")
        elif minutes > 0:
            print(f"综述生成耗时{minutes}分钟{seconds:.2f}秒")
        else:
            print(f"综述生成耗时{seconds:.2f}秒")
            
    except Exception as e:
        # 记录结束时间并计算总耗时（异常情况）
        end_time = time.time()
        total_time = end_time - start_time
        
        # 格式化时间显示（时分秒）
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = total_time % 60
        
        print(f"\n=== 综述生成异常结束 ===")
        print(f"错误信息: {str(e)}")
        print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
        print(f"异常时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
        
        if hours > 0:
            print(f"运行耗时{hours}小时{minutes}分钟{seconds:.2f}秒")
        elif minutes > 0:
            print(f"运行耗时{minutes}分钟{seconds:.2f}秒")
        else:
            print(f"运行耗时{seconds:.2f}秒")
            
        print(f"❌ 生成过程中发生错误: {str(e)}")
        traceback.print_exc()
        raise e  # 重新抛出异常以保持原有行为

if __name__ == "__main__":
    # 设置事件循环策略，以支持Windows上的asyncio
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main())