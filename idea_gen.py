# -*- coding: utf-8 -*-
"""
学术想法生成多智能体系统

脚本目标任务：基于已生成的综述文档，使用多智能体架构自动生成研究想法

上下文：
- 基于Survey Gen产出的markdown文件和enriched outline JSON文件
- 通过多个专业智能体协作完成复杂的想法生成任务
- 支持交互模式和命令行参数模式两种运行方式

输入：
- survey_md_dir: Survey Gen产出的包含多个md文件的目录路径
- logs_dir: 包含LLM调用日志的目录路径（用来提取enriched_outline）
- API配置、模型选择和生成参数
- 向量数据库路径和输出路径

执行步骤：
1. 从指定目录找到最新的md和json文件
2. 解析文件获取final_result和enriched_outline数据
3. 初始化LLM工厂和数据库连接
4. 启动多智能体系统：构建机会图谱→生成idea→评判→优化
5. 保存结果并记录生成耗时

输出：
- 结构化的研究想法集JSON文件
- 可读的想法摘要Markdown文件
- 生成统计信息和总耗时
"""

import os
import json
import asyncio
import argparse
import glob
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import traceback
import sys
from pathlib import Path
import re
import time  # 添加时间模块用于计时

# 导入必要的模块
from multi_agent import (
    LLMFactory,
    AcademicPaperDatabase,
    ModelType,
)
from idea_gen_agent import run_idea_generation


# =========================
# 序列化工具函数
# =========================

def convert_to_serializable(obj: Any) -> Any:
    """递归转换对象为可JSON序列化的格式。"""
    if hasattr(obj, 'to_dict'):
        return obj.to_dict()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        # 对于其他类型，尝试转换为字符串
        return str(obj)


# =========================
# 文件工具函数
# =========================

def find_latest_file_by_pattern(directory: str, pattern: str) -> Optional[str]:
    """在指定目录中查找符合模式的最新文件
    
    Args:
        directory: 目录路径
        pattern: 文件模式（如 "*.md", "*_meta.json"）
        
    Returns:
        最新文件的完整路径，如果没有找到则返回None
    """
    if not os.path.exists(directory):
        print(f"⚠️ 目录不存在: {directory}")
        return None
    
    search_pattern = os.path.join(directory, pattern)
    files = glob.glob(search_pattern)
    
    if not files:
        print(f"⚠️ 在目录 {directory} 中未找到符合模式 {pattern} 的文件")
        return None
    
    # 按修改时间排序，返回最新的文件
    latest_file = max(files, key=os.path.getmtime)
    print(f"✅ 找到最新文件: {latest_file}")
    return latest_file


def find_latest_survey_files(survey_md_dir: str, json_dir: str) -> Tuple[Optional[str], Optional[str]]:
    """查找最新的综述md文件和enriched outline json文件
    
    Args:
        survey_md_dir: Survey Gen产出的md文件目录
        json_dir: 包含JSON文件的目录（实际上应该是logs目录）
        
    Returns:
        (最新的md文件路径, 最新的LLM调用日志文件路径)
    """
    print(f"🔍 在目录中查找最新文件...")
    print(f"   Markdown目录: {survey_md_dir}")
    print(f"   日志目录: {json_dir}")
    
    # 查找最新的md文件（排除test_开头的文件）
    md_files = []
    if os.path.exists(survey_md_dir):
        for file in glob.glob(os.path.join(survey_md_dir, "*.md")):
            if not os.path.basename(file).startswith("test_"):
                md_files.append(file)
    
    latest_md = max(md_files, key=os.path.getmtime) if md_files else None
    
    # 查找最新的LLM调用日志文件（包含enriched outline信息）
    log_files = []
    possible_log_dirs = [json_dir, "./logs", os.path.join(json_dir, "logs")]
    
    for log_dir in possible_log_dirs:
        if os.path.exists(log_dir):
            for file in glob.glob(os.path.join(log_dir, "llm_calls_*.json")):
                log_files.append(file)
    
    latest_json = max(log_files, key=os.path.getmtime) if log_files else None
    
    if latest_json:
        print(f"✅ 找到最新日志文件: {latest_json}")
    else:
        print(f"⚠️ 未找到LLM调用日志文件")
    
    return latest_md, latest_json


def extract_timestamp_from_filename(filename: str) -> Optional[str]:
    """从文件名中提取时间戳
    
    Args:
        filename: 文件名
        
    Returns:
        时间戳字符串，如果未找到则返回None
    """
    # 匹配形如 YYYYMMDD_HHMMSS 的时间戳
    pattern = r'(\d{8}_\d{6})'
    match = re.search(pattern, filename)
    return match.group(1) if match else None


# =========================
# 数据解析函数
# =========================

def parse_survey_markdown(md_file_path: str) -> Dict[str, Any]:
    """解析综述markdown文件，提取final_result格式的数据
    
    Args:
        md_file_path: markdown文件路径
        
    Returns:
        类似于ma_gen产出的final_result结构的字典
    """
    if not os.path.exists(md_file_path):
        raise FileNotFoundError(f"Markdown文件不存在: {md_file_path}")
    
    print(f"📖 解析综述文件: {md_file_path}")
    
    with open(md_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取基本信息
    lines = content.split('\n')
    title = ""
    abstract = ""
    keywords = []
    
    # 解析标题（第一个#）
    for line in lines:
        if line.strip().startswith('# ') and not title:
            title = line.strip()[2:].strip()
            break
    
    # 查找摘要部分
    in_abstract = False
    abstract_lines = []
    for line in lines:
        if line.strip().startswith('# 摘要') or line.strip().startswith('## 摘要'):
            in_abstract = True
            continue
        elif line.strip().startswith('#') and in_abstract:
            break
        elif in_abstract and line.strip():
            if line.strip().startswith('**关键词'):
                # 提取关键词
                keywords_text = line.replace('**关键词**:', '').replace('**关键词', '').strip()
                keywords = [k.strip() for k in keywords_text.split(',') if k.strip()]
                break
            else:
                abstract_lines.append(line)
    
    abstract = '\n'.join(abstract_lines).strip()
    
    # 构建final_result结构
    final_result = {
        "title": title,
        "abstract": abstract,
        "keywords": keywords,
        "full_document": content,
        "statistics": {
            "word_count": len(content.split()),
            "chapter_count": len([line for line in lines if line.strip().startswith('# ') and not line.startswith('# 摘要')]),
            "character_count": len(content)
        },
        "timestamp": extract_timestamp_from_filename(md_file_path) or datetime.now().strftime("%Y%m%d_%H%M%S"),
        "source_file": md_file_path
    }
    
    print(f"✅ 解析完成: 标题='{title}', 关键词={len(keywords)}个, 字数={final_result['statistics']['word_count']}")
    return final_result


def parse_enriched_outline_json(json_file_path: str) -> Optional[Dict[str, Any]]:
    """解析LLM调用日志文件，提取enriched outline数据
    
    Args:
        json_file_path: LLM调用日志文件路径
        
    Returns:
        enriched_outline数据，如果无法解析则返回None
    """
    if not os.path.exists(json_file_path):
        print(f"⚠️ 日志文件不存在: {json_file_path}")
        return None
    
    print(f"📄 解析LLM调用日志: {json_file_path}")
    
    try:
        enriched_outline = None
        
        # 逐行读取大文件，查找特定的记录
        with open(json_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 尝试解析整个JSON
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # 如果不是标准JSON格式，尝试按行解析
            print("📄 文件不是标准JSON格式，尝试按行解析...")
            lines = content.strip().split('\n')
            data = []
            for line in lines:
                try:
                    if line.strip():
                        data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        
        # 确保data是列表格式
        if not isinstance(data, list):
            data = [data] if isinstance(data, dict) else []
        
        # 查找符合条件的记录
        for record in data:
            if isinstance(record, dict):
                # 查找agent_name为"丰富智能体"且task_type为"enrichment_final"的记录
                if (record.get("agent_name") == "丰富智能体" and 
                    record.get("task_type") == "enrichment_final"):
                    
                    print(f"✅ 找到丰富智能体的enrichment_final记录")
                    
                    # 提取parsed_structure
                    parsed_structure = record.get("parsed_structure")
                    if parsed_structure:
                        print(f"✅ 成功提取enriched_outline数据")
                        return parsed_structure
                    else:
                        # 尝试从其他字段提取
                        response_data = record.get("response_data", {})
                        if isinstance(response_data, dict) and response_data.get("parsed_structure"):
                            return response_data["parsed_structure"]
                        
                        # 尝试从result字段提取
                        result = record.get("result", {})
                        if isinstance(result, dict):
                            if result.get("enriched_outline"):
                                return result["enriched_outline"]
                            elif result.get("parsed_structure"):
                                return result["parsed_structure"]
        
        print(f"⚠️ 未找到丰富智能体的enrichment_final记录，将构造基本结构")
        
        # 如果没有找到，尝试从任何包含chapters的记录中提取
        for record in data:
            if isinstance(record, dict):
                # 检查各种可能的字段
                for field in ["parsed_structure", "response_data", "result"]:
                    field_data = record.get(field, {})
                    if isinstance(field_data, dict) and field_data.get("chapters"):
                        print(f"🔄 从{field}字段找到章节数据")
                        return field_data
        
        return None
        
    except Exception as e:
        print(f"❌ 日志文件解析错误: {e}")
        return None


# =========================
# 便捷入口函数
# =========================

async def generate_ideas(
    survey_md_dir: str,
    logs_dir: str, 
    output_path: str = None,
    api_key: str = None,
    base_url: str = "https://openrouter.ai/api/v1",
    db_path: str = "./chroma_db",
    models: Dict[str, str] = None,
    config: Dict[str, Any] = None,
    log_dir: str = "./logs",
    verbose: bool = True
) -> Dict[str, Any]:
    """
    生成研究想法的便捷函数
    
    参数:
        survey_md_dir: Survey Gen产出的md文件目录
        logs_dir: LLM调用日志目录（包含enriched outline数据）
        output_path: 输出路径（可选）
        api_key: API密钥（可选，默认使用环境变量OPENROUTER_API_KEY）
        base_url: API基础URL（默认使用OpenRouter）
        db_path: 向量数据库路径（默认为'./chroma_db'）
        models: 各智能体使用的模型配置（可选）
        config: 想法生成配置（并发数、阈值等）
        log_dir: 日志目录
        verbose: 是否显示详细日志
        
    返回:
        生成的想法结果字典
    """
    # 参数检查和默认值设置
    if not api_key:
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise ValueError("需要提供API密钥（通过参数或环境变量OPENROUTER_API_KEY）")
    
    # 查找最新文件
    latest_md, latest_json = find_latest_survey_files(survey_md_dir, logs_dir)
    
    if not latest_md:
        raise FileNotFoundError(f"在目录 {survey_md_dir} 中未找到markdown文件")
    
    if not latest_json:
        print("⚠️ 未找到JSON文件，将使用基本配置")
    
    # 解析数据
    final_result = parse_survey_markdown(latest_md)
    enriched_outline = parse_enriched_outline_json(latest_json) if latest_json else None
    
    if not enriched_outline:
        # 构造基本的enriched_outline
        enriched_outline = {
            "topic": final_result.get("title", "未知主题"),
            "chapters": {
                "1": {
                    "id": "1",
                    "title": "引言", 
                    "keywords": final_result.get("keywords", [])[:3],
                    "content_guide": "研究背景介绍"
                }
            }
        }
        print("🔧 使用构造的基本大纲结构")
    
    # 设置输出路径
    if not output_path:
        safe_title = "".join(c for c in final_result.get("title", "ideas") if c.isalnum() or c in [' ', '_']).rstrip()
        safe_title = safe_title.replace(' ', '_')
        output_path = f"./idea_output/{safe_title}"
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else output_path, exist_ok=True)
    
    # 默认模型配置
    if not models:
        models = {
            "miner_model": ModelType.CLAUDE.value,
            "generator_model": ModelType.CLAUDE.value,
            "novelty_critic_model": ModelType.CLAUDE.value,
            "feasibility_critic_model": ModelType.CLAUDE.value,
            "refiner_model": ModelType.CLAUDE.value,
        }
    
    # 默认配置
    if not config:
        config = {
            "idea_concurrency": 6,
            "max_rounds": 3,
            "novelty_threshold": 8.0,
            "feasibility_threshold": 7.0,
            "max_initial_ideas": 6
        }
    
    if verbose:
        print(f"📝 综述标题: {final_result.get('title')}")
        print(f"📊 统计信息: {final_result.get('statistics')}")
        print(f"💾 输出路径: {output_path}")
        print(f"🗄️ 数据库路径: {db_path}")
        print(f"🤖 使用模型: {models}")
        print(f"⚙️ 配置参数: {config}")
    
    try:
        # 初始化LLM工厂
        llm_factory = LLMFactory(api_key=api_key, base_url=base_url, log_dir=log_dir)
        
        # 初始化向量数据库
        db = AcademicPaperDatabase(db_path=db_path)
        
        # 生成想法
        start_time = datetime.now()
        if verbose:
            print(f"⏱️ 开始生成: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 调用核心想法生成函数
        result = await run_idea_generation(
            final_result=final_result,
            enriched_outline=enriched_outline,
            llm_factory=llm_factory,
            db=db,
            config=config
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60  # 转换为分钟
        
        # 保存结果
        await save_idea_results(result, output_path, final_result.get("title", "研究想法"))
        
        if verbose:
            print(f"✅ 生成完成: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"⏱️ 总耗时: {duration:.2f} 分钟")
            
            # 显示生成结果统计
            if result.get("status") != "failed":
                stats = result.get("statistics", {})
                print(f"📈 生成统计:")
                print(f"   - 成功率: {stats.get('success_rate', 0):.1%}")
                print(f"   - 最终想法数: {result.get('final_ideas', {}).get('accepted', {}).get('count', 0)}")
        
        return result
    
    except Exception as e:
        if verbose:
            print(f"❌ 生成失败: {str(e)}")
            traceback.print_exc()
        raise e


async def save_idea_results(result: Dict[str, Any], output_path: str, title: str):
    """
    保存想法生成结果
    
    Args:
        result: 生成的想法结果
        output_path: 输出路径
        title: 标题
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    # 时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存详细结果JSON
    result_path = f"{output_path}_{timestamp}_ideas.json"
    
    # 转换对象为可序列化的字典
    serializable_result = convert_to_serializable(result)
    
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(serializable_result, f, ensure_ascii=False, indent=2)
    
    # 保存可读的想法摘要
    summary_path = f"{output_path}_{timestamp}_summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"# {title} - 研究想法生成报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        if result.get("status") == "failed":
            f.write(f"## ❌ 生成失败\n\n")
            f.write(f"错误信息: {result.get('error', '未知错误')}\n\n")
            return
        
        # 统计信息
        stats = result.get("statistics", {})
        f.write(f"## 📊 生成统计\n\n")
        f.write(f"- 成功率: {stats.get('success_rate', 0):.1%}\n")
        f.write(f"- 执行时间: {result.get('execution_time_seconds', 0):.2f} 秒\n\n")
        
        # 机会图谱
        opportunity_graph = result.get("opportunity_graph", {})
        f.write(f"## 🗺️ 机会图谱\n\n")
        f.write(f"- 节点数: {opportunity_graph.get('node_count', 0)}\n")
        f.write(f"- 边数: {opportunity_graph.get('edge_count', 0)}\n\n")
        
        # 最终想法
        final_ideas = result.get("final_ideas", {}).get("accepted", {})
        ideas = final_ideas.get("ideas", [])
        f.write(f"## 💡 最终接受的想法 ({len(ideas)} 个)\n\n")
        
        for i, idea in enumerate(ideas, 1):
            # CandidateIdea是dataclass，直接访问属性
            if hasattr(idea, 'title'):
                f.write(f"### {i}. {idea.title}\n\n")
                f.write(f"**核心假设**: {idea.core_hypothesis}\n\n")
                f.write(f"**创新点**: {', '.join(idea.initial_innovation_points)}\n\n")
            else:
                # 如果是字典格式（已转换过的）
                f.write(f"### {i}. {idea.get('title', '未命名想法')}\n\n")
                f.write(f"**描述**: {idea.get('description', '无描述')}\n\n")
                f.write(f"**新颖性评分**: {idea.get('novelty_score', 'N/A')}\n")
                f.write(f"**可行性评分**: {idea.get('feasibility_score', 'N/A')}\n\n")
                
                if idea.get('rationale'):
                    f.write(f"**评审理由**: {idea.get('rationale')}\n\n")
            
            f.write("---\n\n")
    
    print(f"📁 想法结果已保存到: {result_path}")
    print(f"📄 想法摘要已保存到: {summary_path}")


# =========================
# 命令行接口
# =========================

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="多智能体研究想法生成系统")
    
    parser.add_argument("--survey-md-dir", "-md", type=str, default="./ma_output/", 
                       help="Survey Gen产出的markdown文件目录")
    parser.add_argument("--logs-dir", "-ld", type=str, default="./logs/", 
                       help="LLM调用日志目录（包含enriched outline数据）")
    parser.add_argument("--output", "-o", type=str, default="./idea_output/", 
                       help="输出文件路径")
    parser.add_argument("--api-key", "-k", type=str, 
                       default="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", 
                       help="OpenRouter API密钥")
    parser.add_argument("--base-url", "-u", type=str, default="https://openrouter.ai/api/v1", 
                       help="API基础URL")
    parser.add_argument("--db-path", "-d", type=str, default="D:/Desktop/ZJU/final_test/db/", 
                       help="向量数据库路径")
    
    # 智能体模型配置
    parser.add_argument("--miner-model", type=str, default=ModelType.GEMINI.value, 
                       help="机会挖掘智能体使用的模型")
    parser.add_argument("--generator-model", type=str, default=ModelType.GEMINI.value, 
                       help="想法生成智能体使用的模型")
    parser.add_argument("--novelty-critic-model", type=str, default=ModelType.GEMINI.value, 
                       help="新颖性评审智能体使用的模型")
    parser.add_argument("--feasibility-critic-model", type=str, default=ModelType.GEMINI.value, 
                       help="可行性评审智能体使用的模型")
    parser.add_argument("--refiner-model", type=str, default=ModelType.GEMINI.value, 
                       help="想法精炼智能体使用的模型")
    
    # 想法生成配置
    parser.add_argument("--idea-concurrency", type=int, default=6, 
                       help="想法生成并发数")
    parser.add_argument("--max-rounds", type=int, default=2, 
                       help="最大迭代轮数")
    parser.add_argument("--novelty-threshold", type=float, default=8.0, 
                       help="新颖性阈值")
    parser.add_argument("--feasibility-threshold", type=float, default=7.0, 
                       help="可行性阈值")
    parser.add_argument("--max-initial-ideas", type=int, default=50, 
                       help="初始想法生成数量")
    
    parser.add_argument("--log-dir", type=str, default="./logs", help="日志目录路径")

    return parser.parse_args()


async def interactive_mode():
    """交互模式，通过命令行与用户交互"""
    print("=" * 60)
    print("💡 多智能体研究想法生成系统")
    print("=" * 60)
    
    # 收集用户输入
    survey_md_dir = input("请输入Survey Gen产出的Markdown文件目录路径: ").strip()
    if not survey_md_dir:
        survey_md_dir = "./ma_output/"
    
    logs_dir = input("请输入LLM调用日志目录路径（包含enriched outline数据）: ").strip()
    if not logs_dir:
        logs_dir = "./logs/"  # 默认logs目录
    
    output_path = input("请输入输出文件路径（可选，直接回车使用默认路径）: ").strip()
    
    api_key = input("请输入API密钥（可选，直接回车使用环境变量）: ").strip()
    
    db_path = input("请输入向量数据库路径（可选，直接回车使用默认路径）: ").strip()
    if not db_path:
        db_path = "./chroma_db"
    
    # 配置模型
    print("\n选择智能体使用的模型:")
    print("1. Claude (anthropic/claude-sonnet-4)")
    print("2. GPT-4o (openai/gpt-4o)")
    print("3. Gemini (google/gemini-2.5-flash)")
    print("4. Qwen (qwen/qwen2.5-vl-72b-instruct)")
    print("5. DeepSeek (deepseek/deepseek-chat-v3-0324)")
    
    model_map = {
        "1": ModelType.CLAUDE.value,
        "2": ModelType.GPT.value,
        "3": ModelType.GEMINI.value,
        "4": ModelType.QWEN.value,
        "5": ModelType.DS.value
    }
    
    miner_model = model_map.get(input("机会挖掘智能体模型 (1-5，默认1): ").strip() or "1", ModelType.CLAUDE.value)
    generator_model = model_map.get(input("想法生成智能体模型 (1-5，默认1): ").strip() or "1", ModelType.CLAUDE.value)
    novelty_critic_model = model_map.get(input("新颖性评审智能体模型 (1-5，默认1): ").strip() or "1", ModelType.CLAUDE.value)
    feasibility_critic_model = model_map.get(input("可行性评审智能体模型 (1-5，默认1): ").strip() or "1", ModelType.CLAUDE.value)
    refiner_model = model_map.get(input("想法精炼智能体模型 (1-5，默认1): ").strip() or "1", ModelType.CLAUDE.value)
    
    models = {
        "miner_model": miner_model,
        "generator_model": generator_model,
        "novelty_critic_model": novelty_critic_model,
        "feasibility_critic_model": feasibility_critic_model,
        "refiner_model": refiner_model,
    }
    
    # 配置参数
    print("\n配置想法生成参数:")
    idea_concurrency = int(input("想法生成并发数 (默认6): ").strip() or "6")
    max_rounds = int(input("最大迭代轮数 (默认3): ").strip() or "3")
    novelty_threshold = float(input("新颖性阈值 (默认8.0): ").strip() or "8.0")
    feasibility_threshold = float(input("可行性阈值 (默认7.0): ").strip() or "7.0")
    max_initial_ideas = int(input("初始想法生成数量 (默认6): ").strip() or "6")
    
    config = {
        "idea_concurrency": idea_concurrency,
        "max_rounds": max_rounds,
        "novelty_threshold": novelty_threshold,
        "feasibility_threshold": feasibility_threshold,
        "max_initial_ideas": max_initial_ideas
    }
    
    # 确认生成
    print("\n" + "=" * 60)
    print(f"Markdown目录: {survey_md_dir}")
    print(f"日志目录: {logs_dir}")
    print(f"输出路径: {output_path or '默认'}")
    print(f"数据库路径: {db_path}")
    print(f"机会挖掘模型: {miner_model}")
    print(f"想法生成模型: {generator_model}")
    print(f"新颖性评审模型: {novelty_critic_model}")
    print(f"可行性评审模型: {feasibility_critic_model}")
    print(f"想法精炼模型: {refiner_model}")
    print(f"配置参数: {config}")
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
        await generate_ideas(
            survey_md_dir=survey_md_dir,
            logs_dir=logs_dir,
            output_path=output_path,
            api_key=api_key,
            db_path=db_path,
            models=models,
            config=config
        )
        
        # 记录结束时间并计算总耗时（正常完成）
        end_time = time.time()
        total_time = end_time - start_time
        
        # 格式化时间显示（时分秒）
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = total_time % 60
        
        print(f"\n=== 想法生成完成 ===")
        print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
        print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
        
        if hours > 0:
            print(f"生成idea耗时{hours}小时{minutes}分钟{seconds:.2f}秒")
        elif minutes > 0:
            print(f"生成idea耗时{minutes}分钟{seconds:.2f}秒")
        else:
            print(f"生成idea耗时{seconds:.2f}秒")
            
    except Exception as e:
        # 记录结束时间并计算总耗时（异常情况）
        end_time = time.time()
        total_time = end_time - start_time
        
        # 格式化时间显示（时分秒）
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = total_time % 60
        
        print(f"\n=== 想法生成异常结束 ===")
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
    
    # 如果没有提供markdown目录，进入交互模式
    if not args.survey_md_dir or args.survey_md_dir == "./ma_output/":
        # 检查是否存在默认目录
        if not os.path.exists(args.survey_md_dir):
            await interactive_mode()
            return
    
    # 使用命令行参数
    models = {
        "miner_model": args.miner_model,
        "generator_model": args.generator_model,
        "novelty_critic_model": args.novelty_critic_model,
        "feasibility_critic_model": args.feasibility_critic_model,
        "refiner_model": args.refiner_model,
    }
    
    config = {
        "idea_concurrency": args.idea_concurrency,
        "max_rounds": args.max_rounds,
        "novelty_threshold": args.novelty_threshold,
        "feasibility_threshold": args.feasibility_threshold,
        "max_initial_ideas": args.max_initial_ideas
    }
    
    # 记录开始时间
    start_time = time.time()
    
    try:
        await generate_ideas(
            survey_md_dir=args.survey_md_dir,
            logs_dir=args.logs_dir,
            output_path=args.output,
            api_key=args.api_key,
            base_url=args.base_url,
            db_path=args.db_path,
            models=models,
            config=config,
            log_dir=args.log_dir
        )
        
        # 记录结束时间并计算总耗时（正常完成）
        end_time = time.time()
        total_time = end_time - start_time
        
        # 格式化时间显示（时分秒）
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = total_time % 60
        
        print(f"\n=== 想法生成完成 ===")
        print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
        print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
        
        if hours > 0:
            print(f"生成idea耗时{hours}小时{minutes}分钟{seconds:.2f}秒")
        elif minutes > 0:
            print(f"生成idea耗时{minutes}分钟{seconds:.2f}秒")
        else:
            print(f"生成idea耗时{seconds:.2f}秒")
            
    except Exception as e:
        # 记录结束时间并计算总耗时（异常情况）
        end_time = time.time()
        total_time = end_time - start_time
        
        # 格式化时间显示（时分秒）
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = total_time % 60
        
        print(f"\n=== 想法生成异常结束 ===")
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