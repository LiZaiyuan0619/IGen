# -*- coding: utf-8 -*-
"""
学术综述生成系统工具函数库

脚本目标: 提供多智能体系统共用的工具函数
上下文: 从multi_agent.py中提取的可复用工具函数
输入: 各种数据结构和参数
执行步骤: 按需调用不同的工具函数
输出: 处理后的数据结果
"""

import re
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional
from llm_review_generator import EnhancedSimilarityCalculator
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional

async def search_relevant_content(
    db, 
    similarity_calculator: EnhancedSimilarityCalculator,
    topic: str, 
    subtopics: List[str] = None,
    purpose: str = "检索",
    llm_call_func = None  # 新增：LLM调用函数用于翻译
    ) -> Dict:
    """
    搜索相关内容并计算增强相似度
    
    Args:
        db: 数据库对象 (AcademicPaperDatabase)
        similarity_calculator: 增强相似度计算器
        topic: 主题
        subtopics: 子主题列表
        purpose: 搜索目的，用于日志显示
        llm_call_func: LLM调用函数，用于翻译关键词
        
    Returns:
        包含分类内容和统计信息的上下文字典
    """
    print(f"🔍 正在检索关于'{topic}'的研究资料用于{purpose}...")
    
    search_queries = [topic]
    if subtopics:
        search_queries.extend(subtopics)
    
    # 🆕 翻译搜索查询为英文
    if llm_call_func:
        print("🔤 正在翻译搜索查询为英文...")
        english_queries = await translate_keywords_batch(search_queries, llm_call_func)
        # 同时保留原始查询作为备用
        final_queries = english_queries 
        print(f"📝 英文查询: {final_queries}")
    else:
        print("⚠️ 未提供LLM函数，直接使用原始查询（可能影响搜索效果）")
        final_queries = search_queries
    
    # 🔧修复：english_topic应该是字符串，不是列表
    if llm_call_func:
        english_topic_list = await translate_keywords_batch([topic], llm_call_func)
        if english_topic_list:
            english_topic = english_topic_list[0]  # 取翻译后的主题字符串
        else:
            english_topic = topic  # 翻译失败时使用原始主题
        print(f"🎯 相似度计算使用英文主题: {english_topic}")
    else:
        # 如果没有LLM函数，使用原始主题
        english_topic = topic
        print(f"🎯 相似度计算使用原始主题: {english_topic}")
    
    all_results = []
    for query in final_queries:
        # 检索文本内容
        text_results = db.search_content(
            query, content_type="texts", n_results=500
        )
        all_results.extend(text_results)
        
    # 使用增强相似度重新排序
    enhanced_results = []
    seen_ids = set()
    
    # 添加过滤统计
    total_raw_results = len(all_results)
    duplicate_count = 0
    content_type_filtered_count = 0  # 新增：内容类型过滤统计
    short_content_count = 0
    low_similarity_count = 0
    
    for result in all_results:
        if result['id'] in seen_ids:
            duplicate_count += 1
            continue
        seen_ids.add(result['id'])
        
        # 🆕 智能内容类型过滤：根据搜索来源决定过滤策略
        content_type = result.get('metadata', {}).get('content_type', '')
        collection_name = result.get('collection', '')
        
        # 只过滤来自texts collection的图片和表格描述文本
        # 来自images/tables collection的内容保留（这些是我们专门搜索的图片/表格）
        if content_type in ['image_text', 'table_text'] and collection_name == 'texts':
            content_type_filtered_count += 1
            continue  # 只跳过存储在texts collection中的图片和表格描述文本
        
        # 🆕 长度过滤：删除少于200字符的短内容
        content = result.get("document", "")
        if len(content) < 200:
            short_content_count += 1
            continue
            
        # 🆕 清理学术引用，提高内容质量
        content = clean_academic_citations(content)
        result["document"] = content  # 更新清理后的内容
        
        # 🆕 使用英文主题计算相似度
        enhanced_score = similarity_calculator.calculate_enhanced_similarity(result, english_topic)
        if enhanced_score >= 0.05:  # 相关性阈值
            result['enhanced_similarity'] = enhanced_score
            enhanced_results.append(result)
        else:
            low_similarity_count += 1
    
    # 按增强相似度排序
    enhanced_results.sort(key=lambda x: x['enhanced_similarity'], reverse=True)
    
    # 🆕 详细的过滤统计
    print(f"📊 函数search_relevant_content过滤统计:")
    print(f"  原始结果: {total_raw_results} 条，去重过滤: -{duplicate_count} 条，内容类型过滤: -{content_type_filtered_count} 条 (来自texts的图片/表格描述)，长度过滤: -{short_content_count} 条，相似度过滤: -{low_similarity_count} 条")
    print(f"  最终结果: {len(enhanced_results)} 条")
    
    # 按内容类型分组
    context = {
        "main_topic": topic,
        "subtopics": subtopics or [],
        "relevant_content": {
            "texts": [],
        },
        "source_papers": {},
        "statistics": {}
    }
    
    # 分类存储内容  
    for result in enhanced_results:  # 🆕 移除数量限制，返回所有相关结果
        content_type = result['metadata']['content_type']
        paper_name = result['metadata']['paper_name']
        
        if paper_name not in context["source_papers"]:
            context["source_papers"][paper_name] = {
                "content_count": 0,
                "sections": []
            }
        
        context["source_papers"][paper_name]["content_count"] += 1
        
        content_item = {
            "content": result["document"],
            "paper": paper_name,
            "page": result["metadata"]["page_idx"],
            "relevance_score": result['enhanced_similarity'],
            "metadata": result["metadata"]
        }
        
        context["relevant_content"]["texts"].append(content_item)

    # 统计信息
    context["statistics"] = {
        "total_papers": len(context["source_papers"]),
        "total_texts": len(context["relevant_content"]["texts"])
    }
    
    print(f"📊 函数search_relevant_content检索完成: {context['statistics']['total_papers']} 篇论文, "
          f"{context['statistics']['total_texts']} 条文本"
          )
    
    return context

async def search_section_specific_materials(
    section_info: Dict, 
    db, 
    similarity_calculator_class, 
    llm_call_func,
    chapter_title_english: str,  # 🆕 新增：翻译后的章节标题，用作相似度计算的主要依据
    existing_materials: List[Dict] = None,  # 新增：已有的通用材料
    logger = None,  # 新增：日志记录器
    max_texts: int = 50,
    max_equations: int = 20,
    max_figures: int = 20,
    max_tables: int = 20
    ) -> Dict:
    """
    根据章节关键词搜索特定的相关材料，使用章节标题作为相似度计算的主要依据
    
    脚本目标: 为特定章节搜索相关的研究材料
    上下文: 从multi_agent.py中提取的章节特定材料搜索逻辑
    输入: 
    - section_info: 章节信息
    - db: 数据库对象
    - similarity_calculator_class: 相似度计算器类
    - llm_call_func: LLM调用函数
    - chapter_title_english: 翻译后的章节标题，用作相似度计算的主要依据
    - existing_materials: 已有的通用材料（用于去重）
    - logger: 日志记录器（可选）
    - max_texts: 文本材料最大数量
    - max_equations: 公式材料最大数量
    - max_figures: 图表材料最大数量
    - max_tables: 表格材料最大数量
    执行步骤:
    1. 提取章节关键词和标题
    2. 构建搜索查询并翻译为英文
    3. 使用数据库搜索各类型内容
    4. 🔧改进：使用章节标题计算增强相似度并去重
    5. 与已有通用材料去重（仅对文本类型）
    6. 按内容类型分组返回
    输出: 按类型分类的材料字典（已去除与通用材料重复的内容）
    
    Args:
        section_info: 包含章节信息的字典
        db: 数据库对象，需要有search_content方法
        similarity_calculator_class: 相似度计算器类(EnhancedSimilarityCalculator)
        llm_call_func: LLM调用函数，用于翻译关键词
        chapter_title_english (str): 翻译后的章节标题，用作相似度计算的主要依据
        existing_materials (List[Dict], optional): 已有的通用材料列表，用于去重. Defaults to None.
        logger (optional): 日志记录器，用于记录过程信息. Defaults to None.
        max_texts (int, optional): 文本材料最大数量. Defaults to 50.
        max_equations (int, optional): 公式材料最大数量. Defaults to 20.
        max_figures (int, optional): 图表材料最大数量. Defaults to 20.
        max_tables (int, optional): 表格材料最大数量. Defaults to 20.
        
    Returns:
        按内容类型分类的材料字典（已去除与通用材料重复的内容）
    """
    # 设置日志记录器
    if logger is None:
        # 如果没有传入logger，创建一个简单的控制台日志器
        import logging
        logger = logging.getLogger("section_materials_search")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
    
    logger.info(f"🔍 为章节 '{section_info.get('title', '未命名章节')}' 搜索特定材料...")
    
    
    # 获取章节关键词
    keywords = section_info.get("keywords", [])
    
    # 构建搜索查询
    search_queries = []
    
    # 添加关键词作为主要搜索词
    if keywords:
        search_queries.extend(keywords)
    
    # 使用批量翻译功能，提高效率
    search_queries = await translate_keywords_batch(search_queries, llm_call_func)
    logger.info(f"🔍 章节特定搜索查询关键词: {search_queries}")
    
    # 存储所有搜索结果
    all_results = []
    seen_ids = set()
    
    # 对每个查询词进行搜索
    for query in search_queries:
        if not query.strip():
            continue
            
        # 搜索文本内容
        text_results = db.search_content(
            query, content_type="texts", n_results=max_texts*10
        )
        all_results.extend(text_results)
        
        # 搜索公式
        equation_results = db.search_content(
            query, content_type="equations", n_results=max_equations*10
        )
        all_results.extend(equation_results)
        
        # 搜索图表
        figure_results = db.search_content(
            query, content_type="images", n_results=max_figures*10
        )
        all_results.extend(figure_results)
        
        # 搜索表格
        table_results = db.search_content(
            query, content_type="tables", n_results=max_tables*10
        )
        all_results.extend(table_results)
    

    # 📊 开始详细的筛选统计
    total_raw_results = len(all_results)
    logger.info(f"🔍 原始搜索结果总数: {total_raw_results} 条")
    
    # 去重并计算增强相似度
    enhanced_results = []
    duplicate_count = 0
    content_type_filtered_count = 0  # 新增：内容类型过滤统计
    short_content_count = 0
    low_similarity_count = 0
    
    for result in all_results:
        if result['id'] in seen_ids:
            duplicate_count += 1
            continue
        seen_ids.add(result['id'])
        
        # 🆕 智能内容类型过滤：根据搜索来源决定过滤策略
        content_type = result.get('metadata', {}).get('content_type', '')
        collection_name = result.get('collection', '')
        
        # 只过滤来自texts collection的图片和表格描述文本
        # 来自images/tables collection的内容保留（这些是我们专门搜索的图片/表格）
        if content_type in ['image_text', 'table_text'] and collection_name == 'texts':
            content_type_filtered_count += 1
            continue  # 只跳过存储在texts collection中的图片和表格描述文本
        
        # 🆕 长度过滤：删除少于200字符的短内容
        content = result.get("document", "")
        if len(content) < 200:
            short_content_count += 1
            continue
            
        # 🆕 清理学术引用，提高内容质量
        content = clean_academic_citations(content)
        result["document"] = content  # 更新清理后的内容
        
        # 🔧改进：使用章节标题作为相似度计算的主要依据
        if similarity_calculator_class and chapter_title_english:
            # 使用真正的章节标题创建相似度计算器
            english_keywords = search_queries[:6] if search_queries else []  # 限制关键词数量
            temp_calculator = similarity_calculator_class(chapter_title_english, english_keywords)
            enhanced_score = temp_calculator.calculate_enhanced_similarity(result, chapter_title_english)
            
            # 🆕 添加关键词匹配加分，提高材料多样性
            keyword_bonus = 0
            content_lower = content.lower()
            for keyword in search_queries:
                if keyword and keyword.lower() in content_lower:
                    keyword_bonus += 0.005  # 较小的加分，保持章节标题为主导
            enhanced_score += keyword_bonus
        else:
            # 回退方案：简单字符串匹配
            enhanced_score = 0.01  # 默认分数
            content_lower = content.lower()
            # 优先匹配章节标题
            if chapter_title_english and chapter_title_english.lower() in content_lower:
                enhanced_score += 0.02
            # 然后匹配关键词
            for english_query in search_queries:
                if english_query and english_query.lower() in content_lower:
                    enhanced_score += 0.005
        
        if enhanced_score >= 0.05:  # 章节特定的相关性阈值
            result['enhanced_similarity'] = enhanced_score
            enhanced_results.append(result)
        else:
            low_similarity_count += 1
    
    # 📊 输出详细的筛选统计
    logger.info(f"📊 函数search_section_specific_materials章节特定材料筛选统计:")
    logger.info(f"  原始结果: {total_raw_results} 条，去重过滤: -{duplicate_count} 条 (保留: {total_raw_results - duplicate_count} 条)，内容类型过滤: -{content_type_filtered_count} 条，长度过滤: -{short_content_count} 条 (保留: {total_raw_results - duplicate_count - content_type_filtered_count - short_content_count} 条)，相似度过滤: -{low_similarity_count} 条 (阈值 >= 0.05)，最终通过: {len(enhanced_results)} 条，总过滤率: {((total_raw_results - len(enhanced_results)) / total_raw_results * 100):.1f}%" if total_raw_results > 0 else "  总过滤率: 0%")
    
    # 按相似度排序
    enhanced_results.sort(key=lambda x: x['enhanced_similarity'], reverse=True)
    
    # 按内容类型分组
    section_materials = {
        "texts": [],
        "equations": [],
        "figures": [],
        "tables": []
    }
    
    # 分类存储内容 - 增强版本，增加调试信息
    logger.info(f"🔍 开始分类 {len(enhanced_results)} 条搜索结果...")
    
    # 用于统计各种内容类型
    type_counts_before_limit = {}
    
    for i, result in enumerate(enhanced_results):  
        content_type = result['metadata'].get('content_type', 'unknown')
        paper_name = result['metadata'].get('paper_name', '未知论文')
        
        # 统计内容类型 (限制前)
        type_counts_before_limit[content_type] = type_counts_before_limit.get(content_type, 0) + 1
        
        content_item = {
            "content": result["document"],
            "paper": paper_name,
            "page": result["metadata"].get("page_idx", -1),
            "relevance_score": result['enhanced_similarity'],
            "metadata": result["metadata"],
            "source": "章节特定搜索",
            "content_type": content_type  # 🔧 添加材料类型字段
        }
        
        # 改进的内容类型判断逻辑
        if content_type in ["text", "texts"]:
            section_materials["texts"].append(content_item)
        elif content_type in ["equation", "equations"]:
            section_materials["equations"].append(content_item)
        elif content_type in ["image", "images", "figure", "figures", "image_text"]:
            # 🆕 包含 image_text 类型，这是图片的文本描述
            section_materials["figures"].append(content_item)
        elif content_type in ["table", "tables", "table_text", "table_image"]:
            # 🆕 包含 table_text 和 table_image 类型，分别是表格的文本内容和图片形式
            section_materials["tables"].append(content_item)
        else:
            # 默认归类为文本，但记录警告，并进行去重检查
            logger.warning(f"⚠️ 未知内容类型 '{content_type}'，归类为文本材料")
            text_content = content_item.get("content", "").strip()

    
    # 📊 记录限制前的数量
    texts_before = len(section_materials["texts"])
    equations_before = len(section_materials["equations"])
    figures_before = len(section_materials["figures"])
    tables_before = len(section_materials["tables"])
    
    # 新增：限制每种类型材料的数量
    section_materials["texts"] = section_materials["texts"][:max_texts]
    section_materials["equations"] = section_materials["equations"][:max_equations]
    section_materials["figures"] = section_materials["figures"][:max_figures]
    section_materials["tables"] = section_materials["tables"][:max_tables]
    
    # 📊 显示详细的内容类型分布和限制效果
    logger.info(f"📊 内容类型原始分布: {type_counts_before_limit}")
    logger.info(f"📊 内容分类和数量限制统计:")
    logger.info(f"  文本 (texts): {texts_before} → {len(section_materials['texts'])} (限制: {max_texts}, 删除: {max(0, texts_before - max_texts)})")
    logger.info(f"  公式 (equations): {equations_before} → {len(section_materials['equations'])} (限制: {max_equations}, 删除: {max(0, equations_before - max_equations)})")
    logger.info(f"  图表 (figures): {figures_before} → {len(section_materials['figures'])} (限制: {max_figures}, 删除: {max(0, figures_before - max_figures)})")
    logger.info(f"  表格 (tables): {tables_before} → {len(section_materials['tables'])} (限制: {max_tables}, 删除: {max(0, tables_before - max_tables)})")
    
    # 📊 计算最终统计
    final_total = sum(len(materials) for materials in section_materials.values())
    total_deleted_by_limit = (texts_before + equations_before + figures_before + tables_before) - final_total
    logger.info(f"📊 数量限制影响: 删除了 {total_deleted_by_limit} 条特定材料")
    logger.info(f"📊 最终材料总数: {final_total} 条")
    
    return section_materials


async def gather_section_materials(
    section_info: Dict, 
    db, 
    main_topic,
    similarity_calculator_class, 
    llm_call_func,
    citation_manager,
    logger = None,  # 新增：日志记录器参数
    max_texts: int = 50,
    max_equations: int = 10,
    max_figures: int = 10,
    max_tables: int = 10
    ) -> List[Dict]:
    """
    收集章节相关材料，结合通用材料和章节特定材料
    
    脚本目标: 为特定章节收集和整合相关的研究材料
    上下文: 从multi_agent.py中提取的章节材料收集逻辑
    输入:
    - section_info: 章节信息
    - db: 数据库对象
    - similarity_calculator_class: 相似度计算器类
    - llm_call_func: LLM调用函数
    - citation_manager: 引用管理器
    执行步骤:
    1. 提取章节标识信息
    2. 从上下文中获取通用相关材料
    3. 计算材料与章节的相关度
    4. 补充数据库搜索结果
    5. 获取章节特定材料
    6. 合并所有材料并处理引用
    输出: 整合后的材料列表
    
    Args:
        section_info: 包含章节信息的字典
        db: 数据库对象，需要有search_content方法
        main_topic: 主题信息
        similarity_calculator_class: 相似度计算器类
        llm_call_func: LLM调用函数
        citation_manager: 引用管理器对象
        logger (optional): 日志记录器，用于记录过程信息. Defaults to None.
        
    Returns:
        整合后的材料列表
    """
    # 设置日志记录器
    if logger is None:
        # 如果没有传入logger，创建一个简单的控制台日志器
        import logging
        logger = logging.getLogger("gather_section_materials")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
    
    # 章节标识信息
    chapter_title = section_info.get("title", "")
    
    # 翻译章节标题
    translated_titles = await translate_keywords_batch([chapter_title], llm_call_func)
    chapter_title = translated_titles[0] if translated_titles else chapter_title

    # 🆕 处理 Introduction 的特殊情况
    if chapter_title:
        # 获取主题信息，作为 Introduction 的替换内容
        topic = main_topic
        
        # 情况1：如果 chapter_title 只包含 "Introduction"（需要注意大小写的变化，可能是INTRODUCTION，introduction等等
        if chapter_title.lower() == "introduction":
            if topic:
                chapter_title = topic
                logger.info(f"🔄 检测到单独的 'Introduction' 章节，已替换为主题: {topic}")
            else:
                logger.warning("⚠️ 检测到 'Introduction' 章节但未找到主题信息")
        
        # 情况2：如果包含 "Introduction" 但还有其他内容，则删除 "Introduction"
        elif "introduction" in chapter_title.lower():
            # 使用正则表达式删除 "Introduction" 及其前后的分隔符
            # 删除 "Introduction" 以及可能的分隔符（冒号、破折号、空格等）
            cleaned_title = re.sub(r'\bIntroduction\b[:\-\s]*', '', chapter_title)
            cleaned_title = re.sub(r'[:\-\s]*\bIntroduction\b', '', cleaned_title)
            # 清理多余的空格和标点
            cleaned_title = re.sub(r'\s+', ' ', cleaned_title).strip()
            cleaned_title = re.sub(r'^[:\-\s]+|[:\-\s]+$', '', cleaned_title)
            
            if cleaned_title and cleaned_title != chapter_title:
                chapter_title = cleaned_title
                logger.info(f"🔄 从章节标题中删除了 'Introduction'，保留内容: {chapter_title}")

    logger.info(f"📝 处理后的章节标题: {chapter_title}")


    # 2. 🔧改进：获取章节特定材料（使用翻译后的章节标题作为相似度计算的主要依据）
    logger.info(f"🎯 使用章节标题 '{chapter_title}' 作为相似度计算的主要依据")
    section_specific_materials = await search_section_specific_materials(
        section_info, db, similarity_calculator_class, llm_call_func,
        chapter_title_english=chapter_title,  # 🔧改进：使用翻译后的章节标题作为相似度计算依据
        logger=logger,  # 🆕 传入日志记录器
        max_texts=max_texts, max_equations=max_equations, max_figures=max_figures, max_tables=max_tables
    )

    
    # 🆕 分类返回材料，而不是混合在一起
    categorized_materials = {
        "texts": section_specific_materials.get("texts", []),
        "equations": section_specific_materials.get("equations", []),
        "figures": section_specific_materials.get("figures", []),
        "tables": section_specific_materials.get("tables", [])
    }
    
    # 为所有材料处理引用
    all_materials_for_citation = []
    for material_type, materials in categorized_materials.items():
        logger.info(f"📊 章节特定{material_type}材料数量: {len(materials)} 条")
        all_materials_for_citation.extend(materials)
    
    # 处理材料引用，为每个材料添加到引用管理器中
    citation_ids = citation_manager.process_materials_for_citations(all_materials_for_citation)
    
    # 📊 输出完整的材料收集统计
    total_count = sum(len(materials) for materials in categorized_materials.values())
    logger.info(f"📚 章节 '{chapter_title}' 材料收集完成:")
    logger.info(f"  材料总数: {total_count} 条")
    logger.info(f"  分类统计: 文本{len(categorized_materials['texts'])}, 公式{len(categorized_materials['equations'])}, 图片{len(categorized_materials['figures'])}, 表格{len(categorized_materials['tables'])}")
    
    return categorized_materials



def clean_generated_content(content: str) -> str:
    """
    清理生成的内容，移除可能的指令性文本
    
    Args:
        content: 待清理的文本内容
        
    Returns:
        清理后的文本内容
    """
    # 移除可能的前导说明文本
    content = re.sub(r"^(好的|下面是|以下是|这是|我将|我会|根据要求|基于提供的材料).*?\n\n", "", content, flags=re.DOTALL)
    
    # 移除可能的结尾说明文本
    content = re.sub(r"\n\n(以上是|这就是|希望这个|我已经完成).*?$", "", content, flags=re.DOTALL)
    
    return content.strip()


def extract_authors_from_source(source: str) -> str:
    """
    从来源字符串中提取作者信息
    
    Args:
        source: 来源字符串（论文名称等）
        
    Returns:
        提取的作者信息
    """
    # 简单的作者提取逻辑
    if "brown" in source.lower():
        return "Brown et al."
    elif "vaswani" in source.lower():
        return "Vaswani et al."
    elif "devlin" in source.lower():
        return "Devlin et al."
    elif "radford" in source.lower():
        return "Radford et al."
    else:
        return "未知作者"


def clean_academic_citations(text: str) -> str:
    """
    清理学术文本中的引用信息
    
    脚本目标: 去除文本中的学术引用，提高LLM对核心内容的关注度
    上下文: 在搜索相关材料时，去除无意义的引用信息
    输入: 包含学术引用的文本
    执行步骤:
    1. 使用正则表达式匹配各种引用格式
    2. 去除匹配到的引用内容
    3. 清理多余的空格和标点符号
    4. 返回清理后的文本
    输出: 去除引用后的干净文本
    
    Args:
        text (str): 需要清理的文本
        
    Returns:
        str: 去除引用后的文本
        
    Examples:
        >>> text = "LLMs are powerful (Tang et al., 2024). They solve tasks (Liu et al., 2024; Li et al., 2024a)."
        >>> clean_academic_citations(text)
        "LLMs are powerful. They solve tasks."
    """
    if not text or not isinstance(text, str):
        return text
    
    import re
    
    # 保存原始文本长度用于统计
    original_length = len(text)
    
    # 1. 匹配标准学术引用格式：包括圆括号引用和方括号数字引用
    # 支持多种格式：
    # - (Tang et al., 2024)
    # - (Liu et al., 2024; Li et al., 2024a; Wang, 2023)
    # - (Smith & Jones, 2023)
    # - (Brown et al., 2022a,b)
    # - [14, 37, 40, 48, 63, 75, 76, 90, 96]
    # - [1-5, 10, 15-20]
    # - [14a, 37b]
    citation_patterns = [
        # 匹配方括号数字引用（最常见的格式）
        # [14, 37, 40, 48, 63, 75, 76, 90, 96]
        # [1-5, 10, 15-20]
        # [14a, 37b]
        r'\[\s*\d+[a-z]?(?:\s*[-–]\s*\d+[a-z]?)?(?:\s*[,;]\s*\d+[a-z]?(?:\s*[-–]\s*\d+[a-z]?)?)*\s*\]',
        
        # 匹配包含 "et al." 的圆括号引用
        r'\([^)]*et al\.[^)]*\d{4}[a-z]?[^)]*\)',
        
        # 匹配标准的作者-年份格式，包括多作者用分号分隔的情况
        r'\([^)]*[A-Z][a-z]+(?:\s+(?:&|and)\s+[A-Z][a-z]+)*\s*,\s*\d{4}[a-z]?[^)]*\)',
        
        # 匹配包含多个引用的复杂格式（用分号分隔）
        r'\([^)]*\d{4}[a-z]?(?:\s*[;,]\s*[^)]*\d{4}[a-z]?)*[^)]*\)',
        
        # 匹配简单的年份引用
        r'\(\s*\d{4}[a-z]?\s*\)',
        
        # 匹配 ibid., op. cit. 等学术引用
        r'\([^)]*(?:ibid\.|op\.\s*cit\.|loc\.\s*cit\.)[^)]*\)',
        
        # 匹配单个方括号数字引用（防止遗漏）
        r'\[\s*\d+[a-z]?\s*\]',
    ]
    
    # 应用所有引用模式
    for pattern in citation_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # 2. 清理由于删除引用导致的多余空格和标点符号
    # 删除多个连续空格
    text = re.sub(r'\s+', ' ', text)
    
    # 删除多个连续的逗号或分号
    text = re.sub(r'[,;]\s*[,;]+', ',', text)
    
    # 删除句号前的多余空格
    text = re.sub(r'\s+\.', '.', text)
    
    # 删除句子开头的逗号或分号
    text = re.sub(r'^\s*[,;]\s*', '', text)
    text = re.sub(r'\.\s*[,;]\s*', '. ', text)
    
    # 删除多个连续的句号
    text = re.sub(r'\.{2,}', '.', text)
    
    # 🆕 处理删除方括号引用后的特殊情况
    # 删除单词间的多余空格（如 "field [14, 37] applications" 变成 "field  applications"）
    text = re.sub(r'([a-zA-Z])\s{2,}([a-zA-Z])', r'\1 \2', text)
    
    # 删除句末引用后遗留的空格和标点问题
    # 如 "applications . " 改为 "applications."
    text = re.sub(r'\s+([.!?;,])', r'\1', text)
    
    # 删除段落开头的空格和奇怪字符
    text = re.sub(r'^\s*[.;,]\s*', '', text)
    
    # 处理如 "within the medical field . They can" 这样的情况
    text = re.sub(r'([a-zA-Z])\s+\.\s+([A-Z])', r'\1. \2', text)
    
    # 清理首尾空格
    text = text.strip()
    
    # 统计清理效果
    cleaned_length = len(text)
    if original_length > cleaned_length:
        reduction = original_length - cleaned_length
        reduction_percent = (reduction / original_length) * 100
        # 这里使用简单的print，实际使用时可以替换为logger
        # print(f"📝 引用清理: 删除 {reduction} 字符 ({reduction_percent:.1f}%)")
    
    return text


def extract_title_from_source(source: str) -> str:
    """
    从来源字符串中提取标题
    
    Args:
        source: 来源字符串（文件名等）
        
    Returns:
        提取的标题
    """
    # 移除文件扩展名和路径
    title = source.replace(".pdf_result", "").replace("_", " ")
    # 清理特殊字符
    title = re.sub(r'[^\w\s-]', '', title)
    return title.strip()


class LLMLogger:
    """日志记录器，用于记录LLM调用的输入和输出"""
    
    def __init__(self, log_dir="./logs"):
        """初始化日志记录器"""
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # 创建带时间戳的日志文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = f"{log_dir}/llm_calls_{timestamp}.log"
        self.json_log_file = f"{log_dir}/llm_calls_{timestamp}.json"
        
        # 初始化JSON日志
        self.json_logs = []
        
        # 设置文本日志
        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger("LLM_Logger")
        
        # 添加控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
    
    def log_call(self, agent_name: str, model_name: str, messages: list, response: dict, task_type: str = None):
        """记录一次LLM调用"""
        # 创建日志条目
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "agent_name": agent_name,
            "model_name": model_name,
            "task_type": task_type,
            "messages": messages,
            "response": response
        }
        
        # 保存到JSON日志
        self.json_logs.append(log_entry)
        self.save_json_logs()
        
        # 记录到文本日志
        self.logger.info(f"Agent: {agent_name} | Model: {model_name} | Task: {task_type}")
        self.logger.info(f"Input messages count: {len(messages)}")
        
        # 记录最后一条输入消息
        if messages and len(messages) > 0:
            last_msg = messages[-1].get("content", "")
            self.logger.info(f"Last input message (truncated): {last_msg[:80]}...")
        
        # 记录响应内容
        response_content = response.get("content", "")
        if response_content:
            self.logger.info(f"Response (truncated): {response_content[:80]}...")
            
        # 记录使用情况
        if "usage" in response:
            usage = response["usage"]
            self.logger.info(f"Usage: {usage}")
        
        self.logger.info("-" * 50)
    
    def log_parsed_structure(self, agent_name: str, task_type: str, parsed_structure: dict):
        """记录解析后的结构化数据"""
        # 创建日志条目
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "agent_name": agent_name,
            "task_type": task_type,
            "parsed_structure": parsed_structure
        }
        
        # 保存到JSON日志
        self.json_logs.append(log_entry)
        self.save_json_logs()
        
        # 记录到文本日志
        self.logger.info(f"Agent: {agent_name} | Task: {task_type} | Parsed Structure")
        self.logger.info(f"Parsed structure keys: {list(parsed_structure.keys())}")
        
        # 记录解析结果的基本统计信息
        if "chapters" in parsed_structure:
            chapters_count = len(parsed_structure["chapters"])
            self.logger.info(f"Parsed {chapters_count} chapters")

            # 统计子章节数量
            subsections_count = 0
            # 检查chapters是列表还是字典
            if isinstance(parsed_structure["chapters"], list):
                for chapter in parsed_structure["chapters"]:
                    if "subsections" in chapter:
                        subsections_count += len(chapter["subsections"])
            elif isinstance(parsed_structure["chapters"], dict):
                for chapter_id, chapter in parsed_structure["chapters"].items():
                    if "subsections" in chapter:
                        subsections_count += len(chapter["subsections"])

            self.logger.info(f"Parsed {subsections_count} subsections")
        
        self.logger.info("-" * 50)
    
    def save_json_logs(self):
        """保存JSON日志到文件"""
        try:
            with open(self.json_log_file, 'w', encoding='utf-8') as f:
                json.dump(self.json_logs, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存JSON日志失败: {e}")


async def translate_keywords_batch(keywords: List[str], llm_call_func) -> List[str]:
    """
    批量翻译关键词，提高效率
    
    Args:
        keywords: 需要翻译的关键词列表
        llm_call_func: LLM调用函数，应该接受prompt和task_type参数
        
    Returns:
        翻译后的关键词列表
    """
    if not keywords:
        return []
    
    # 分离需要翻译和不需要翻译的词汇
    english_keywords = []
    need_translation = []
    need_translation_indices = []
    
    for i, keyword in enumerate(keywords):
        if not keyword or keyword.strip().isascii():
            english_keywords.append(keyword.strip() if keyword else "")
        else:
            # 这里可以添加缓存检查的逻辑
            english_keywords.append("")  # 占位符
            need_translation.append(keyword.strip())
            need_translation_indices.append(i)
    
    # 批量翻译需要翻译的词汇
    if need_translation:
        try:
            batch_prompt = f"""你是一位专业的计算机领域的学术翻译专家。请将以下中文学术术语精确翻译为英文学术术语。

    【翻译要求】
    1. 必须使用标准的学术英文术语
    2. 保持术语的专业性和准确性
    3. 严格按照给定顺序逐一翻译
    4. 只返回翻译结果，不要任何解释或描述
    5. 每个术语输出一个最贴近的翻译结果

    【输入术语】
    {chr(10).join(f"{i+1}. {term}" for i, term in enumerate(need_translation))}

    【输出格式】
    请严格按照以下格式逐行输出翻译结果，不要添加任何其他内容：
    1. [第一个术语的英文翻译]
    2. [第二个术语的英文翻译]
    3. [第三个术语的英文翻译]
    ...

    开始翻译："""
            
            response = await llm_call_func(batch_prompt, task_type="batch_translation")
            
            if response.get("content"):
                translated_lines = response["content"].strip().split('\n')
                
                # 处理翻译结果 - 改进的解析逻辑
                successful_translations = 0
                for i, line in enumerate(translated_lines):
                    if i < len(need_translation):
                        # 清理翻译结果
                        clean_translation = line.strip()
                        
                        # 移除可能的编号和格式标记
                        clean_translation = re.sub(r'^\d+[.)\s]+', '', clean_translation)
                        clean_translation = re.sub(r'^[-*]\s+', '', clean_translation)
                        clean_translation = re.sub(r'^\[', '', clean_translation)
                        clean_translation = re.sub(r'\]$', '', clean_translation)
                        clean_translation = clean_translation.strip('"\'')
                        clean_translation = clean_translation.strip()
                        
                        # 验证翻译质量
                        is_valid_translation = (
                            clean_translation and 
                            len(clean_translation) > 0 and
                            not clean_translation.lower().startswith("抱歉") and
                            not clean_translation.lower().startswith("对不起") and
                            not clean_translation.lower().startswith("翻译") and
                            not "以下是" in clean_translation and
                            not "按照" in clean_translation and
                            clean_translation != need_translation[i]  # 确保不是原文
                        )
                        
                        if is_valid_translation:
                            # 更新结果列表
                            index = need_translation_indices[i]
                            english_keywords[index] = clean_translation
                            successful_translations += 1
                        else:
                            # 翻译失败，使用原文
                            index = need_translation_indices[i]
                            english_keywords[index] = need_translation[i]
                            print(f"⚠️ 翻译质量不符合要求，使用原文: {need_translation[i]} (LLM输出: '{clean_translation}')")
                
                print(f"📊 批量翻译统计: 成功 {successful_translations}/{len(need_translation)} 个术语")
            
        except Exception as e:
            print(f"⚠️ 批量翻译错误: {e}，使用原文")
            # 翻译失败，使用原文
            for i, original in enumerate(need_translation):
                index = need_translation_indices[i]
                english_keywords[index] = original
    
    # 过滤空字符串
    result = [kw for kw in english_keywords if kw.strip()]
    return result


def parse_outline_response(response_content: str, topic: str, subtopics: List[str] = None, found_start_marker: str = None) -> Dict:
    """
    解析Planner生成的大纲响应，提取结构化的大纲信息
    
    脚本目标: 将LLM生成的大纲响应解析为结构化的数据格式
    上下文: 从multi_agent.py中提取的大纲解析逻辑
    输入: LLM响应内容、主题和子主题
    执行步骤: 
    1. 检查响应有效性
    2. 提取大纲标记之间的内容  
    3. 解析概述和章节结构
    4. 构建结构化大纲对象
    5. 备用解析方案处理
    输出: 结构化的大纲字典
    
    Args:
        response_content: LLM的响应内容
        topic: 综述主题
        subtopics: 子主题列表（可选）
        
    Returns:
        结构化的大纲字典，包含topic、subtopics、overview、chapters等字段
        
    Raises:
        ValueError: 当响应内容为空或无效时
    """
    if not response_content:
        raise ValueError("大纲生成失败 - LLM响应内容为空")
    
    # 🆕 放宽开头标记限制 - 支持多种开头标记
    outline_text = response_content
    
    # 先尝试找到完整的标记对
    outline_match = re.search(r"===大纲开始===(.*?)===大纲结束===", response_content, re.DOTALL)
    if outline_match:
        outline_text = outline_match.group(1)
        print("✓ 找到完整的===大纲开始===和===大纲结束===标记")
    else:
        # 尝试多种开头标记
        start_markers = ["【优化后大纲】", "===大纲开始===", "【综述概述】", "===优化结果开始==="]
        
        # 如果传入了found_start_marker，优先使用它
        if found_start_marker:
            start_markers = [found_start_marker] + [m for m in start_markers if m != found_start_marker]
        
        found_start = False
        for marker in start_markers:
            start_pos = response_content.find(marker)
            if start_pos != -1:
                # 找到开头标记，查找结束标记
                end_pos = response_content.find("===大纲结束===", start_pos)
                if end_pos != -1:
                    # 找到了结束标记
                    outline_text = response_content[start_pos + len(marker):end_pos]
                    print(f"✓ 找到完整标记对: {marker} ... ===大纲结束===")
                else:
                    # 没有结束标记，使用从开头标记到结尾的内容
                    outline_text = response_content[start_pos + len(marker):]
                    print(f"⚠️ 只找到开头标记: {marker}，使用到结尾的内容")
                found_start = True
                break
        
        if not found_start:
            print("⚠️ 未找到任何有效的开头标记，使用完整响应解析")
    
    # 构建结构化大纲
    outline = {
        "topic": topic,
        "subtopics": subtopics or [],  # 🆕 包含子主题信息
        "overview": "",  # 将从响应中提取
        "chapters": []   # 将从响应中提取
    }
    
    # 提取概述 - 精确匹配【综述概述】标签之后的内容，直到【章节结构】或【优化后大纲】标签
    overview_match = re.search(r"【综述概述】\s*(.*?)(?=【章节结构】|【优化后大纲】|$)", outline_text, re.DOTALL)
    if overview_match:
        outline["overview"] = overview_match.group(1).strip()
    else:
        # 备用方案：尝试查找"概述"或"摘要"关键词
        alt_overview = re.search(r"(?:概述|摘要)[:：](.*?)(?=\d+\.\s|\n\d+\.|\[\[|$)", outline_text, re.DOTALL)
        if alt_overview:
            outline["overview"] = alt_overview.group(1).strip()
    
    # 提取所有一级章节
    chapter_matches = []
    chapter_pattern = re.compile(r"(\d+)\.\s+(.*?)(?:\n|$)((?:(?!\d+\.\s+).)*?)(?=\d+\.\s+|\Z)", re.DOTALL)
    for match in chapter_pattern.finditer(outline_text):
        chapter_id = match.group(1)
        chapter_title = match.group(2).strip()
        chapter_content = match.group(3).strip()
        chapter_matches.append((chapter_id, chapter_title, chapter_content))

    # 处理每个章节
    for chapter_id, chapter_title, chapter_content in chapter_matches:
        # 初始化章节数据
        chapter = {
            "id": chapter_id,
            "title": chapter_title,
            "description": "",
            "subsections": []
        }
        
        # 提取章节描述和子章节
        lines = chapter_content.split('\n')
        description_lines = []
        current_subsection = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 检查是否为子章节
            subsection_match = re.match(r"(\d+\.\d+)\s+(.*?)$", line)
            if subsection_match:
                # 如果有当前子章节，保存它
                if current_subsection:
                    chapter["subsections"].append(current_subsection)
                    
                # 创建新子章节
                subsection_id = subsection_match.group(1)
                subsection_title = subsection_match.group(2).strip()
                current_subsection = {
                    "id": subsection_id,
                    "title": subsection_title,
                    "description": ""
                }
            # 处理子章节描述
            elif current_subsection is not None:
                # 如果不是以数字开头的新子章节，就是当前子章节的描述
                current_subsection["description"] += line + " "
            # 处理章节描述
            else:
                # 如果不是子章节，也不是空行，就是章节描述
                description_lines.append(line)
        
        # 添加最后一个子章节
        if current_subsection:
            chapter["subsections"].append(current_subsection)
        
        # 设置章节描述
        chapter["description"] = " ".join(description_lines).strip()
        
        # 清理所有描述（移除多余空格）
        chapter["description"] = re.sub(r'\s+', ' ', chapter["description"]).strip()
        for subsection in chapter["subsections"]:
            subsection["description"] = re.sub(r'\s+', ' ', subsection["description"]).strip()
        
        # 添加到大纲
        outline["chapters"].append(chapter)
    
    # 处理可能的不完整格式
    if not outline["chapters"]:
        # 备用解析方案：尝试查找任何数字编号的标题
        simple_chapters = re.findall(r"(\d+)[\.、]\s*(.*?)(?:\n|$)((?:(?!\d+[\.、]\s+).)*?)(?=\d+[\.、]\s+|\Z)", 
                                outline_text, re.DOTALL)
        
        for chap_id, chap_title, chap_content in simple_chapters:
            chapter = {
                "id": chap_id,
                "title": chap_title.strip(),
                "description": chap_content.strip(),
                "subsections": []
            }
            
            # 尝试提取子章节
            subsections = re.findall(r"(\d+\.\d+)[\.、]?\s*(.*?)(?:\n|$)((?:(?!\d+\.?\d+[\.、]?\s+).)*?)(?=\d+\.?\d+[\.、]?\s+|\Z)", 
                                chap_content, re.DOTALL)
            
            for sub_id, sub_title, sub_content in subsections:
                subsection = {
                    "id": sub_id,
                    "title": sub_title.strip(),
                    "description": sub_content.strip()
                }
                chapter["subsections"].append(subsection)
                
            # 如果找到子章节，从章节描述中移除
            if chapter["subsections"]:
                # 提取子章节之前的内容作为章节描述
                pre_subsection = re.match(r"(.*?)(?=\d+\.\d+[\.、]?\s+)", chap_content, re.DOTALL)
                if pre_subsection:
                    chapter["description"] = pre_subsection.group(1).strip()
                    
            outline["chapters"].append(chapter)
    
    return outline


def build_detailed_planning_section(enriched_outline: Dict) -> str:
    """
    构建详细的章节规划内容，用于摘要生成
    
    脚本目标: 将enriched_outline中的详细规划信息格式化为可读的文本
    上下文: 从multi_agent.py中提取的摘要生成部分的详细规划逻辑
    输入: enriched_outline - 包含详细章节规划的字典
    执行步骤:
    1. 检查输入数据的有效性
    2. 处理章节数据结构（字典或列表）
    3. 格式化章节信息（标题、内容指引、关键词等）
    4. 处理子章节的详细信息
    5. 构建完整的规划文本
    输出: 格式化的详细规划文本字符串
    
    Args:
        enriched_outline: 包含详细章节规划信息的字典
        
    Returns:
        格式化的详细规划内容字符串
    """
    planning_content = ""
    
    # 检查输入数据的有效性
    if not enriched_outline or not enriched_outline.get("chapters"):
        return planning_content
    
    planning_content += "\n【详细章节规划结构】\n"
    
    chapters_data = enriched_outline.get("chapters", {})
    
    # 处理两种可能的数据结构：字典或列表
    if isinstance(chapters_data, dict):
        # 按章节ID排序
        sorted_chapter_items = sorted(chapters_data.items(), key=lambda x: x[0])
        
        for chapter_id, chapter_data in sorted_chapter_items:
            planning_content += _format_chapter_details(chapter_id, chapter_data)
    
    elif isinstance(chapters_data, list):
        # 保持向后兼容的列表格式处理
        for chapter_data in chapters_data:
            chapter_id = chapter_data.get("id", "")
            planning_content += _format_chapter_details(chapter_id, chapter_data)
    
    return planning_content


def _format_chapter_details(chapter_id: str, chapter_data: Dict) -> str:
    """
    格式化单个章节的详细信息
    
    Args:
        chapter_id: 章节ID
        chapter_data: 章节数据字典
        
    Returns:
        格式化的章节详细信息字符串
    """
    chapter_content = ""
    
    chapter_title = chapter_data.get("title", "未命名章节")
    content_guide = chapter_data.get("content_guide", "")
    keywords = chapter_data.get("keywords", [])
    research_focus = chapter_data.get("research_focus", [])
    
    chapter_content += f"\n==== 第{chapter_id}章: {chapter_title} ====\n"
    
    # 章节内容指引（完整显示）
    if content_guide:
        chapter_content += f"【章节内容指引】\n{content_guide}\n\n"
    
    # 章节关键词
    if keywords:
        chapter_content += f"【章节关键词】\n{', '.join(keywords)}\n\n"
    
    # 重点研究领域
    if research_focus:
        chapter_content += f"【重点研究领域】\n"
        for i, focus in enumerate(research_focus, 1):
            chapter_content += f"{i}. {focus}\n"
        chapter_content += "\n"
    
    # 详细子章节信息
    subsections = chapter_data.get("subsections", {})
    if subsections:
        chapter_content += f"【子章节详细规划】\n"
        
        if isinstance(subsections, dict):
            # 按子章节ID排序
            sorted_subsection_items = sorted(subsections.items(), key=lambda x: x[0])
            
            for sub_id, sub_data in sorted_subsection_items:
                chapter_content += _format_subsection_details(sub_id, sub_data)
        
        elif isinstance(subsections, list):
            for sub_data in subsections:
                sub_id = sub_data.get("id", "")
                chapter_content += _format_subsection_details(sub_id, sub_data)
    
    chapter_content += "=" * 60 + "\n"
    
    return chapter_content


def _format_subsection_details(sub_id: str, sub_data: Dict) -> str:
    """
    格式化单个子章节的详细信息
    
    Args:
        sub_id: 子章节ID
        sub_data: 子章节数据字典
        
    Returns:
        格式化的子章节详细信息字符串
    """
    subsection_content = ""
    
    sub_title = sub_data.get("title", "")
    sub_content_guide = sub_data.get("content_guide", "")
    key_points = sub_data.get("key_points", [])
    writing_guide = sub_data.get("writing_guide", "")
    
    subsection_content += f"\n  ◆ {sub_id} {sub_title}\n"
    
    if sub_content_guide:
        subsection_content += f"    【内容概要】{sub_content_guide}\n"
    
    if key_points:
        subsection_content += f"    【关键要点】\n"
        for i, point in enumerate(key_points, 1):
            subsection_content += f"      • {point}\n"
    
    if writing_guide:
        subsection_content += f"    【写作建议】{writing_guide}\n"
    
    subsection_content += "\n"
    
    return subsection_content


def parse_abstract_response(raw_response: str) -> tuple[str, list]:
    """
    解析LLM摘要响应，提取摘要内容和关键词，过滤掉思考过程
    
    Args:
        raw_response: LLM的原始响应文本
        
    Returns:
        tuple: (摘要文本, 关键词列表)
    """
    # 首先尝试提取===摘要开始===和===摘要结束===之间的内容
    abstract_match = re.search(r"===摘要开始===(.*?)===摘要结束===", raw_response, re.DOTALL)
    
    if abstract_match:
        # 找到了标记，提取摘要部分
        abstract_section = abstract_match.group(1).strip()
        print("✅ 成功提取标记内的摘要内容")
    else:
        # 没找到标记，尝试备用解析方案
        print("⚠️ 未找到摘要标记，尝试备用解析方案...")
        
        # 检查是否有思考过程标记，如果有则截取之前的内容
        if "===思考过程开始===" in raw_response:
            abstract_section = raw_response.split("===思考过程开始===")[0].strip()
            print("✅ 根据思考过程标记截取摘要内容")
        elif "思考过程记录" in raw_response:
            # 更宽松的匹配
            abstract_section = raw_response.split("思考过程记录")[0].strip()
            print("✅ 根据思考过程关键词截取摘要内容")
        elif "---" in raw_response:
            # 如果有分隔符，取分隔符前的内容
            abstract_section = raw_response.split("---")[0].strip()
            print("✅ 根据分隔符截取摘要内容")
        else:
            # 最后的兜底方案，使用全部内容
            abstract_section = raw_response
            print("⚠️ 未找到任何分隔标记，使用完整响应")
    
    # 从摘要部分提取关键词
    keywords = []
    keywords_pattern = r"(?:\*\*关键词[:：]\*\*|关键词[:：])(.*?)(?=\n\n|\n#|\n\*\*|$)"
    keywords_match = re.search(keywords_pattern, abstract_section, re.DOTALL)
    
    if keywords_match:
        keywords_text = keywords_match.group(1).strip()
        # 移除可能的markdown格式标记
        keywords_text = re.sub(r'\*\*', '', keywords_text)
        # 分割关键词
        keywords = [k.strip() for k in re.split(r"[,，;；、\s]+", keywords_text) if k.strip()]
        print(f"✅ 提取到 {len(keywords)} 个关键词")
    else:
        print("⚠️ 未找到关键词")
    
    # 清理摘要文本，移除关键词部分
    abstract_text = re.sub(r"(?:\*\*关键词[:：]\*\*|关键词[:：]).*?(?=\n\n|\n#|$)", "", abstract_section, flags=re.DOTALL)
    
    # 清理多余的空行和首尾空白
    abstract_text = re.sub(r'\n\s*\n\s*\n', '\n\n', abstract_text)  # 合并多个空行
    abstract_text = abstract_text.strip()
    
    # 确保摘要有合适的标题格式
    if not abstract_text.startswith("#"):
        # 没有任何标题，添加标题
        abstract_text = "# 摘要\n\n" + abstract_text
    elif not re.match(r"^#\s*摘要", abstract_text):
        # 有标题但不是"摘要"标题，检查是否需要调整
        if abstract_text.startswith("##"):
            # 如果是二级标题，改为一级标题
            abstract_text = re.sub(r"^##\s*", "# ", abstract_text)
    # 如果已经是正确的"# 摘要"格式，保持原样
        
    return abstract_text, keywords


def _deduplicate_keywords(keywords: List[str], outline: Dict, context: Dict = None) -> List[str]:
    """
    去除与topic和subtopics重复的关键词
    
    Args:
        keywords: 原始关键词列表
        outline: 包含topic信息的大纲字典
        context: 包含subtopics等上下文信息的字典（可选）
        
    Returns:
        去重后的关键词列表
    """
    if not keywords:
        return keywords
    
    # 🔧 添加类型检查防护措施
    if not isinstance(outline, dict):
        print(f"⚠️ _deduplicate_keywords: outline参数类型异常，期望dict，实际{type(outline)}，跳过去重")
        return keywords
    
    # 获取需要去重的词汇
    duplicate_terms = set()
    
    # 从outline中获取主题和子主题
    topic = outline.get("topic", "")
    if topic:
        # 将topic拆分为单词并添加到去重集合
        topic_words = re.split(r'[,，;；、\s\-_]+', topic.lower())
        duplicate_terms.update([word.strip() for word in topic_words if word.strip()])
        # 也添加完整的topic
        duplicate_terms.add(topic.lower().strip())
    
    # 处理subtopics（优先从context中获取，再从outline中获取）
    subtopics = []
    if context:
        subtopics = context.get("subtopics", [])
        # 如果context中有main_topic，也可以作为补充
        main_topic = context.get("main_topic", "")
        if main_topic and main_topic != topic:
            subtopics.append(main_topic)
    
    # 如果context中没有找到，从outline中查找
    if not subtopics:
        subtopics = outline.get("subtopics", [])
        if not subtopics:
            # 尝试其他可能的字段名
            subtopics = outline.get("sub_topics", [])
    
    if subtopics:
        for subtopic in subtopics:
            if subtopic:
                # 将每个subtopic拆分为单词并添加到去重集合
                subtopic_words = re.split(r'[,，;；、\s\-_]+', subtopic.lower())
                duplicate_terms.update([word.strip() for word in subtopic_words if word.strip()])
                # 也添加完整的subtopic
                duplicate_terms.add(subtopic.lower().strip())
    
    # 从章节标题中提取可能重复的词汇
    chapters = outline.get("chapters", [])
    if isinstance(chapters, list):
        for chapter in chapters:
            chapter_title = chapter.get("title", "")
            if chapter_title:
                # 提取章节标题中的关键词
                title_words = re.split(r'[,，;；、\s\-_]+', chapter_title.lower())
                duplicate_terms.update([word.strip() for word in title_words if word.strip() and len(word.strip()) > 2])
    
    # 添加一些通用的学术词汇到去重列表
    common_academic_terms = {
        "研究", "方法", "技术", "系统", "模型", "算法", "应用", "分析", "设计", "实现",
        "research", "method", "technique", "system", "model", "algorithm", "application", 
        "analysis", "design", "implementation", "approach", "framework", "study", "evaluation"
    }
    duplicate_terms.update(common_academic_terms)
    
    # 进行去重处理
    deduplicated_keywords = []
    for keyword in keywords:
        keyword_clean = keyword.strip()
        if not keyword_clean:
            continue
            
        # 检查是否与去重词汇重复（不区分大小写）
        keyword_lower = keyword_clean.lower()
        
        # 检查完全匹配
        if keyword_lower in duplicate_terms:
            continue
            
        # 检查是否为去重词汇的子字符串或父字符串
        is_duplicate = False
        for term in duplicate_terms:
            if len(term) < 2:  # 跳过太短的词
                continue
            # 如果关键词包含在去重词汇中，或去重词汇包含在关键词中
            if (term in keyword_lower and len(term) > len(keyword_lower) * 0.7) or \
               (keyword_lower in term and len(keyword_lower) > len(term) * 0.7):
                is_duplicate = True
                break
        
        if not is_duplicate:
            deduplicated_keywords.append(keyword_clean)
    
    # 记录去重情况
    removed_count = len(keywords) - len(deduplicated_keywords)
    if removed_count > 0:
        removed_keywords = [kw for kw in keywords if kw.strip().lower() not in [dk.lower() for dk in deduplicated_keywords]]
        print(f"🧹 关键词去重: 移除了 {removed_count} 个重复关键词")
        print(f"   移除的关键词: {', '.join(removed_keywords[:5])}{'...' if len(removed_keywords) > 5 else ''}")
    
    return deduplicated_keywords


def  parse_full_enrichment(response_text: str, outline: Dict) -> Dict:
    """解析LLM返回的完整大纲丰富内容，适应更新后的输出格式"""
    # 🔧 添加输入参数类型检查
    if not isinstance(response_text, str):
        print(f"❌ parse_full_enrichment: response_text参数类型异常，期望str，实际{type(response_text)}")
        return {"topic": "未知", "overview": "无概述", "chapters": {}}
    
    if not isinstance(outline, dict):
        print(f"❌ parse_full_enrichment: outline参数类型异常，期望dict，实际{type(outline)}，使用默认值")
        outline = {"topic": "未知", "overview": "无概述", "chapters": {}}
    
    # 🔧 修改：从原始大纲中保留 topic 和 overview 信息
    enrichment = {
        "topic": outline.get("topic", "未知"),
        "overview": outline.get("overview", "无概述"),
        "chapters": {}
    }
    
    # 🆕 更宽松的内容提取逻辑 - 支持多种格式
    content_text = None
    
    # 方案1: 尝试标准的内容规划标记
    content_match = re.search(r"===内容规划开始===(.*?)===内容规划结束===", 
                            response_text, re.DOTALL)
    if content_match:
        content_text = content_match.group(1).strip()
        print("✅ 成功提取完整的标准内容规划")
    
    # 方案2: 尝试优化结果标记格式
    if not content_text:
        # 查找===优化结果开始===到===内容规划结束===之间的内容
        opt_match = re.search(r"===优化结果开始===(.*?)===内容规划结束===", 
                            response_text, re.DOTALL)
        if opt_match:
            content_text = opt_match.group(1).strip()
            print("✅ 成功提取优化结果格式的内容规划")
    
    # 方案3: 尝试从【章节内容指引】开始到===内容规划结束===
    if not content_text:
        guide_match = re.search(r"【章节内容指引】(.*?)===内容规划结束===", 
                              response_text, re.DOTALL)
        if guide_match:
            content_text = guide_match.group(1).strip()
            print("✅ 成功从章节内容指引开始提取")
    
    # 方案4: 从第一个章节标题开始到任意结束标记
    if not content_text:
        chapter_to_end = re.search(r"(# 第\d+章.*?)(?:===.*?结束===|【是否继续迭代】|写作指导完整性:|$)", 
                                 response_text, re.DOTALL)
        if chapter_to_end:
            content_text = chapter_to_end.group(1).strip()
            print("✅ 成功从第一个章节标题开始提取")
    
    # 方案5: 最宽松模式 - 只要找到章节内容就尝试提取
    if not content_text:
        # 查找所有可能的开始位置
        start_patterns = [
            r"===内容规划开始===",
            r"===优化结果开始===",
            r"【优化后丰富大纲】",
            r"【章节内容指引】",
            r"# 第\d+章"
        ]
        
        start_pos = -1
        start_pattern_found = ""
        
        for pattern in start_patterns:
            match = re.search(pattern, response_text)
            if match:
                start_pos = match.start()
                start_pattern_found = pattern
                break
        
        if start_pos >= 0:
            # 查找可能的结束位置
            remaining_text = response_text[start_pos:]
            end_patterns = [
                r"===内容规划结束===",
                r"===优化结果结束===",
                r"【是否继续迭代】",
                r"写作指导完整性:",
                r"综合质量评分:"
            ]
            
            end_pos = len(remaining_text)  # 默认到文本结尾
            
            for pattern in end_patterns:
                match = re.search(pattern, remaining_text)
                if match:
                    end_pos = match.start()
                    break
            
            # 如果从优化结果开始，尝试去掉前面的标记
            extracted_text = remaining_text[:end_pos].strip()
            if start_pattern_found in [r"===优化结果开始===", r"【优化后丰富大纲】"]:
                # 查找实际章节内容开始位置
                chapter_start = re.search(r"# 第\d+章", extracted_text)
                if chapter_start:
                    content_text = extracted_text[chapter_start.start():].strip()
                else:
                    content_text = extracted_text
            else:
                content_text = extracted_text
            
            print(f"✅ 宽松模式成功提取，从 {start_pattern_found} 开始")
    
    # 如果还是没有提取到内容
    if not content_text:
        print("❌ 所有解析方案都失败，使用原始内容进行章节匹配")
        content_text = response_text
        
    # 清理提取的内容
    if content_text:
        # 移除可能的思考过程部分
        if "【思考过程记录】" in content_text:
            content_text = content_text.split("【思考过程记录】")[0].strip()
        
        # 移除可能残留的标记
        content_text = re.sub(r"^(===.*?===|【.*?】)\s*", "", content_text, flags=re.MULTILINE)
        content_text = content_text.strip()
    
    # 🆕 更宽松的章节匹配逻辑
    chapter_matches = []
    
    # 方案1: 标准的章节模式
    standard_pattern = r"#\s*第(\d+)章[：:]\s*(.+?)\s*\n(.*?)(?=\n#\s*第\d+章[：:]|\Z)"
    chapter_matches = re.findall(standard_pattern, content_text, re.DOTALL)
    
    if chapter_matches:
        print(f"✅ 标准模式找到 {len(chapter_matches)} 个章节")
    else:
        # 方案2: 更宽松的章节模式（支持不同的分隔符）
        loose_patterns = [
            r"#\s*第(\d+)章[：:：]\s*(.+?)\s*\n(.*?)(?=\n#\s*第\d+章|\Z)",  # 支持中文冒号
            r"#\s*(\d+)[\.、\s]*(.+?)\s*\n(.*?)(?=\n#\s*\d+|\Z)",  # 数字开头
            r"#\s*第(\d+)章\s*(.+?)\s*\n(.*?)(?=\n#|\Z)",  # 无冒号
        ]
        
        for i, pattern in enumerate(loose_patterns):
            chapter_matches = re.findall(pattern, content_text, re.DOTALL)
            if chapter_matches:
                print(f"✅ 宽松模式{i+1}找到 {len(chapter_matches)} 个章节")
                break
    
    # 方案3: 如果还是没找到，尝试最宽松的匹配
    if not chapter_matches:
        print("🔄 使用最宽松的匹配模式...")
        # 查找所有以#开头的行作为章节标题
        lines = content_text.split('\n')
        chapter_lines = []
        
        for i, line in enumerate(lines):
            if re.match(r'^\s*#\s*', line) and ('章' in line or '第' in line):
                chapter_lines.append((i, line.strip()))
        
        print(f"🔍 找到 {len(chapter_lines)} 个可能的章节标题行")
        
        # 构建章节匹配
        for j, (line_idx, title_line) in enumerate(chapter_lines):
            # 提取章节号和标题
            title_match = re.search(r'#\s*(?:第)?(\d+)章?[：:：]?\s*(.+)', title_line)
            if title_match:
                chapter_num = title_match.group(1)
                chapter_title = title_match.group(2).strip()
            else:
                # 备用提取
                chapter_num = str(j + 1)
                chapter_title = re.sub(r'^\s*#+\s*', '', title_line)
            
            # 提取章节内容（从当前行到下一个章节标题行）
            start_line = line_idx + 1
            if j + 1 < len(chapter_lines):
                end_line = chapter_lines[j + 1][0]
            else:
                end_line = len(lines)
            
            chapter_content = '\n'.join(lines[start_line:end_line]).strip()
            
            if chapter_content:  # 只有当有内容时才添加
                chapter_matches.append((chapter_num, chapter_title, chapter_content))
        
        print(f"✅ 最宽松模式构建了 {len(chapter_matches)} 个章节")
    
    # 如果完全没有匹配到，给出详细的调试信息（但更简洁）
    if not chapter_matches:
        print("⚠️ 未找到任何章节内容，显示前300字符用于调试:")
        print(content_text[:300] + "..." if len(content_text) > 300 else content_text)
        print("⚠️ 查找所有 # 开头的行:")
        lines = content_text.split('\n')
        hash_lines = [line for line in lines[:10] if line.strip().startswith('#')]
        for line in hash_lines:
            print(f"  找到: {line.strip()}")
    else:
        print(f"📊 最终成功解析 {len(chapter_matches)} 个章节")
    
    # 处理每个章节
    for chapter_num, chapter_title, chapter_content in chapter_matches:
        chapter_id = chapter_num
        
        # 初始化章节数据
        chapter_data = {
            "id": chapter_id,
            "title": chapter_title,
            "content_guide": "",
            "keywords": [],
            "research_focus": [],
            "subsections": {}
        }
        
        # 提取章节内容指引 - 适应新格式 "章节内容指引:"
        content_guide_match = re.search(r"章节内容指引[：:](.*?)(?=\n##|\n#|\Z)", 
                                    chapter_content, re.DOTALL)
        if not content_guide_match:
            # 尝试旧格式
            content_guide_match = re.search(r"##\s*章节内容指引\s*\n(.*?)(?=\n##|$)", 
                                    chapter_content, re.DOTALL)
        
        if content_guide_match:
            chapter_data["content_guide"] = content_guide_match.group(1).strip()
        
        # 提取章节关键词
        keywords_match = re.search(r"##\s*本章节关键词.*?\n(.*?)(?=\n##|\n###|$)", 
                                chapter_content, re.DOTALL)
        if keywords_match:
            keywords_text = keywords_match.group(1).strip()
            # 提取引号中的关键词或用逗号分隔的关键词
            keywords = re.findall(r'"([^"]+)"', keywords_text)
            if not keywords:
                keywords = [k.strip() for k in re.split(r'[,，;；、\s]+', keywords_text) if k.strip()]
            
            # 去重处理：移除与topic和subtopics重复的关键词
            # 注意：context参数未传递，subtopics将从outline中获取（如果存在）
            keywords = _deduplicate_keywords(keywords, outline)
            chapter_data["keywords"] = keywords
        
        # 提取重点研究领域
        research_focus_match = re.search(r"##\s*重点研究领域\s*\n(.*?)(?=\n##|\n###|$)", 
                                    chapter_content, re.DOTALL)
        if research_focus_match:
            research_text = research_focus_match.group(1).strip()
            research_points = []
            for line in research_text.split("\n"):
                line = line.strip()
                if line and re.match(r"^\d+\.", line):
                    research_points.append(re.sub(r"^\d+\.\s*", "", line))
            chapter_data["research_focus"] = research_points
        
        # 使用更精确的子章节匹配模式 - 修复版本
        # 这个模式会正确匹配到下一个子章节的开始，或者到内容结束
        subsection_pattern = r"###\s*(\d+\.\d+)\s+(.+?)\s*\n(.*?)(?=\n###\s*\d+\.\d+|\n#\s*第\d+章|\Z)"
        subsection_matches = re.findall(subsection_pattern, chapter_content, re.DOTALL)
        
        for subsection_id, subsection_title, subsection_content in subsection_matches:
            # 初始化子章节数据结构
            subsection_data = {
                "id": subsection_id,
                "title": subsection_title,
                "content_guide": "",
                "key_points": [],
                "writing_guide": ""
            }
            
            # 处理子章节 - 调试信息已移除，保持简洁输出
            
            # 提取内容概要 - 适应新格式 "内容概要:"
            content_overview = re.search(r"内容概要[：:](.*?)(?=\n####|\n###|\Z)", 
                                    subsection_content, re.DOTALL)
            if not content_overview:
                # 尝试旧格式
                content_overview = re.search(r"####\s*内容概要\s*\n(.*?)(?=\n####|$)", 
                                    subsection_content, re.DOTALL)
            
            if content_overview:
                subsection_data["content_guide"] = content_overview.group(1).strip()
            
            # 提取关键要点 - 修复版本，支持多种格式
            key_points_match = re.search(r"####\s*关键要点\s*\n(.*?)(?=\n####|\n###|\Z)", 
                                        subsection_content, re.DOTALL)
            
            if key_points_match:
                key_points_text = key_points_match.group(1).strip()
                points = []
                
                for line in key_points_text.split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    
                    # 匹配编号格式：1. 2. 3. 等
                    if re.match(r"^\d+\.\s+", line):
                        # 移除编号前缀，保留内容
                        clean_point = re.sub(r"^\d+\.\s+", "", line)
                        if clean_point:
                            points.append(clean_point)
                    # 匹配其他编号格式：-, *, • 等
                    elif re.match(r"^[-*•]\s+", line):
                        clean_point = re.sub(r"^[-*•]\s+", "", line)
                        if clean_point:
                            points.append(clean_point)
                
                subsection_data["key_points"] = points
            else:
                print(f"⚠️ 子章节 {subsection_id} 未找到关键要点")
            
            # 提取写作建议 - 修复版本，支持多种格式  
            writing_guide_match = re.search(r"####\s*写作建议\s*\n(.*?)(?=\n####|\n###|\Z)", 
                                            subsection_content, re.DOTALL)
            
            if writing_guide_match:
                writing_guide_text = writing_guide_match.group(1).strip()
                
                # 清理文本，保持段落结构但移除多余换行
                writing_guide_text = re.sub(r'\n+', ' ', writing_guide_text)
                writing_guide_text = re.sub(r'\s+', ' ', writing_guide_text)
                
                subsection_data["writing_guide"] = writing_guide_text
            else:
                print(f"⚠️ 子章节 {subsection_id} 未找到写作建议")
            
            # 🆕 简化的调试信息
            total_items = len(subsection_data["key_points"]) + (1 if subsection_data["writing_guide"] else 0)
            if total_items == 0:
                print(f"⚠️ 子章节 {subsection_id} 解析不完整 (要点:{len(subsection_data['key_points'])}, 指导:{'有' if subsection_data['writing_guide'] else '无'})")
            else:
                pass
            # 将子章节数据添加到章节中
            chapter_data["subsections"][subsection_id] = subsection_data
        
        # 将章节数据添加到结果中
        enrichment["chapters"][chapter_id] = chapter_data
    
    # 为了更好的分配给写作智能体，确保所有章节ID都有对应的内容
    original_chapters = outline.get("chapters", [])
    
    # 🔧 修复：处理chapters可能是字典或列表的情况
    if isinstance(original_chapters, dict):
        # 如果是字典格式，转换为列表
        chapter_list = list(original_chapters.values())
        print(f"🔧 检测到chapters是字典格式，包含 {len(chapter_list)} 个章节")
    elif isinstance(original_chapters, list):
        # 如果是列表格式，直接使用
        chapter_list = original_chapters
    else:
        # 如果既不是字典也不是列表，使用空列表
        chapter_list = []
        print(f"⚠️ chapters格式异常，类型: {type(original_chapters)}，使用空列表")
    
    for chapter in chapter_list:
        # 🔧 添加类型检查
        if not isinstance(chapter, dict):
            print(f"⚠️ 跳过非字典类型的章节: {type(chapter)} - {str(chapter)[:50]}")
            continue
            
        chapter_id = chapter.get("id", "")
        if chapter_id and chapter_id not in enrichment["chapters"]:
            print(f"⚠️ 为未解析的章节 {chapter_id} 添加默认结构")
            # 为未解析到的章节添加一个空结构
            enrichment["chapters"][chapter_id] = {
                "id": chapter_id,
                "title": chapter.get("title", ""),
                "content_guide": "请根据大纲内容撰写本章节。",
                "research_focus": [],
                "keywords": [],
                "subsections": {}
            }
            
            # 添加子章节
            subsections = chapter.get("subsections", [])
            # 🔧 同样处理subsections可能是字典的情况
            if isinstance(subsections, dict):
                subsections = list(subsections.values())
            elif not isinstance(subsections, list):
                subsections = []
                
            for subsection in subsections:
                if not isinstance(subsection, dict):
                    continue
                subsection_id = subsection.get("id", "")
                if subsection_id:
                    enrichment["chapters"][chapter_id]["subsections"][subsection_id] = {
                        "id": subsection_id,
                        "title": subsection.get("title", ""),
                        "content_guide": "请根据大纲内容撰写本小节。",
                        "key_points": [],
                        "writing_guide": ""
                    }
    
    # 最终解析结果统计
    parsed_chapters = len(enrichment["chapters"])
    expected_chapters = len(original_chapters)
    
    if parsed_chapters > 0:
        # 统计子章节数量
        total_subsections = sum(len(ch.get("subsections", {})) for ch in enrichment["chapters"].values())
        print(f"📊 子章节统计: 共 {total_subsections} 个子章节")
        
    return enrichment

def create_numbered_materials_mapping(materials: List[Dict], section_info: Dict) -> Dict:
    """
    为材料创建带编号的映射表，类似llm_review_generator的方式
    
    说明：根据database_setup的处理逻辑，不同类型的材料需要包含不同的元数据字段：
    - 文本材料：基础字段 + text_level
    - 公式材料：基础字段 + equation_text, text_format, context_before/after
    - 图片材料：基础字段 + img_path, img_caption, img_footnote, modality等
    - 表格材料：基础字段 + img_path, table_caption, table_footnote, has_table_body等
    """
    
    numbered_materials = {}
    
    # 按类型分组材料
    text_materials = []
    equation_materials = []
    figure_materials = []
    table_materials = []

    chapter_id = section_info.get("id", "")
    
    # 🔍 详细分析每个材料的类型
    type_analysis = {}
    
    for i, material in enumerate(materials):
        # 检查多个可能的content_type字段
        content_type_from_metadata = material.get("metadata", {}).get("content_type", None)
        content_type_direct = material.get("content_type", None)
        type_field = material.get("type", None)
        
        # 优先级：metadata.content_type > content_type > type > default
        if content_type_from_metadata:
            content_type = content_type_from_metadata
            type_source = "metadata.content_type"
        elif content_type_direct:
            content_type = content_type_direct
            type_source = "content_type"
        elif type_field:
            content_type = type_field
            type_source = "type"
        else:
            content_type = "text"
            type_source = "default"
        
        # 记录类型分析
        type_key = f"{content_type}({type_source})"
        type_analysis[type_key] = type_analysis.get(type_key, 0) + 1
        
        
        if content_type in ["text", "texts"]:
            text_materials.append(material)
        elif content_type in ["equation", "equations"]:
            equation_materials.append(material)
        elif content_type in ["image", "images", "figure", "figures", "image_text"]:  # ✅ 包含了image_text
            figure_materials.append(material)
        elif content_type in ["table", "tables", "table_text", "table_image"]:        # ✅ 包含了table_text
            table_materials.append(material)
        else:
            text_materials.append(material)
        
    # 为每种类型的材料分配编号
    material_counter = 1
    
    # 文本材料 [文本1], [文本2], ...
    for i, material in enumerate(text_materials, 1):
        temp_id = f"{chapter_id}-文本{i}"
        metadata = material.get("metadata", {})
        
        numbered_materials[temp_id] = {
            "type": "text",
            "global_id": material_counter,
            "content": material.get("content", ""),
            "paper": material.get("paper", "未知来源"),
            "page": metadata.get("page_idx", "-1"),
            "relevance_score": material.get("relevance_score", 0.0),
            "material_id": material.get("id", f"material_{material_counter}"),
            # 文本特有字段
            "text_level": metadata.get("text_level", 0),
            "order_in_paper": metadata.get("order_in_paper", -1),
            "original_data": metadata.get("original_data", "")
        }
        material_counter += 1
    
    # 公式材料 [公式1], [公式2], ...
    for i, material in enumerate(equation_materials, 1):
        temp_id = f"{chapter_id}-公式{i}"
        metadata = material.get("metadata", {})
        
        numbered_materials[temp_id] = {
            "type": "equation",
            "global_id": material_counter,
            "content": material.get("content", ""),
            "paper": material.get("paper", "未知来源"),
            "page": metadata.get("page_idx", "-1"),
            "relevance_score": material.get("relevance_score", 0.0),
            "material_id": material.get("id", f"material_{material_counter}"),
            # 公式特有字段
            "equation_text": metadata.get("equation_text", material.get("content", "")),
            "text_format": metadata.get("text_format", "unknown"),
            "context_before": metadata.get("context_before", ""),
            "context_after": metadata.get("context_after", ""),
            "has_context": metadata.get("has_context", False),
            "order_in_paper": metadata.get("order_in_paper", -1),
            "original_data": metadata.get("original_data", "")
        }
        material_counter += 1
    
    # 图片材料 [图片1], [图片2], ...
    for i, material in enumerate(figure_materials, 1):
        temp_id = f"{chapter_id}-图片{i}"
        metadata = material.get("metadata", {})
        
        numbered_materials[temp_id] = {
            "type": "figure",
            "global_id": material_counter,
            "content": material.get("content", ""),
            "paper": material.get("paper", "未知来源"),
            "page": metadata.get("page_idx", "-1"),
            "relevance_score": material.get("relevance_score", 0.0),
            "material_id": material.get("id", f"material_{material_counter}"),
            # 图片特有字段 ✅ 添加了缺失的关键信息
            "img_path": metadata.get("img_path", ""),  # 图片路径 - 关键信息
            "img_caption": metadata.get("img_caption", ""),
            "img_footnote": metadata.get("img_footnote", ""),
            "modality": metadata.get("modality", "image"),  # image 或 text
            "related_image_id": metadata.get("related_image_id", ""),
            "reference_texts": metadata.get("reference_texts", ""),
            "has_references": metadata.get("has_references", False),
            "search_key_used": metadata.get("search_key_used", ""),
            "order_in_paper": metadata.get("order_in_paper", -1),
            "original_data": metadata.get("original_data", "")
        }
        material_counter += 1
    
    # 表格材料 [表格1], [表格2], ...
    for i, material in enumerate(table_materials, 1):
        temp_id = f"{chapter_id}-表格{i}"
        metadata = material.get("metadata", {})
        
        numbered_materials[temp_id] = {
            "type": "table",
            "global_id": material_counter,
            "content": material.get("content", ""),
            "paper": material.get("paper", "未知来源"),
            "page": metadata.get("page_idx", "-1"),
            "relevance_score": material.get("relevance_score", 0.0),
            "material_id": material.get("id", f"material_{material_counter}"),
            # 表格特有字段 ✅ 添加了缺失的关键信息
            "img_path": metadata.get("img_path", ""),  # 表格图片路径 - 关键信息
            "table_caption": metadata.get("table_caption", ""),
            "table_footnote": metadata.get("table_footnote", ""),
            "has_table_body": metadata.get("has_table_body", False),
            "has_table_image": metadata.get("has_table_image", False),
            "modality": metadata.get("modality", "text"),  # image 或 text
            "reference_texts": metadata.get("reference_texts", ""),
            "has_references": metadata.get("has_references", False),
            "search_key_used": metadata.get("search_key_used", ""),
            "order_in_paper": metadata.get("order_in_paper", -1),
            "original_data": metadata.get("original_data", "")
        }
        material_counter += 1
    
    return numbered_materials


def extract_citation_mapping(content: str, numbered_materials: Dict) -> Dict:
    """
    从生成的内容中提取引用映射
    
    说明：根据不同材料类型，完整保留所有特有字段信息，确保最终JSON文件包含完整信息
    - 图片材料：包含img_path, img_caption, modality等图片特有信息
    - 表格材料：包含table_caption, has_table_body, has_table_image等表格特有信息  
    - 公式材料：包含equation_text, text_format, context_before/after等公式特有信息
    - 文本材料：包含text_level, order_in_paper等文本特有信息
    """
    import re
    
    citation_mapping = {}
    
    # 查找所有的引用标识符，如[文本1], [公式2], [图片1], [表格1]等
    # 先找到所有方括号内容，然后分割处理连续引用
    pattern = r'\[([^\]]+)\]'
    bracket_contents = re.findall(pattern, content)
    
    # 收集所有单独的引用
    all_citations = []
    for bracket_content in bracket_contents:
        # 分割连续引用：[6-文本11, 6-公式6] -> ['6-文本11', '6-公式6']
        # 处理逗号分隔的引用
        individual_citations = [cite.strip() for cite in bracket_content.split(',')]
        all_citations.extend(individual_citations)
    
    # 去重但保持顺序
    unique_citations = []
    for citation in all_citations:
        if citation and citation not in unique_citations:
            unique_citations.append(citation)
    
    for citation in unique_citations:
        # 检查是否是我们的材料标识符
        if citation in numbered_materials:
            material_info = numbered_materials[citation]
            material_type = material_info["type"]
            
            # 基础信息（所有类型都有）
            citation_data = {
                "material_id": material_info["material_id"],
                "type": material_type,
                "paper": material_info["paper"],
                "page": material_info["page"],
                "content_preview": material_info["content"][:200] + "..." if len(material_info["content"]) > 200 else material_info["content"],
                "relevance_score": material_info["relevance_score"],
                "full_content": material_info["content"],
                "original_data": material_info["original_data"]
            }
            
            # 根据材料类型添加特有字段 ✅
            if material_type == "figure":
                # 图片材料特有字段
                citation_data.update({
                    "img_path": material_info.get("img_path", ""),           # 图片路径 - 关键信息
                    "img_caption": material_info.get("img_caption", ""),
                    "img_footnote": material_info.get("img_footnote", ""),
                    "modality": material_info.get("modality", "image"),
                    "related_image_id": material_info.get("related_image_id", ""),
                    "reference_texts": material_info.get("reference_texts", ""),
                    "has_references": material_info.get("has_references", False),
                    "search_key_used": material_info.get("search_key_used", ""),
                    "order_in_paper": material_info.get("order_in_paper", -1)
                })
                
            elif material_type == "table":
                # 表格材料特有字段
                citation_data.update({
                    "img_path": material_info.get("img_path", ""),           # 表格图片路径 - 关键信息
                    "table_caption": material_info.get("table_caption", ""),
                    "table_footnote": material_info.get("table_footnote", ""),
                    "has_table_body": material_info.get("has_table_body", False),
                    "has_table_image": material_info.get("has_table_image", False),
                    "modality": material_info.get("modality", "text"),
                    "reference_texts": material_info.get("reference_texts", ""),
                    "has_references": material_info.get("has_references", False),
                    "search_key_used": material_info.get("search_key_used", ""),
                    "order_in_paper": material_info.get("order_in_paper", -1)
                })
                
            elif material_type == "equation":
                # 公式材料特有字段
                citation_data.update({
                    "equation_text": material_info.get("equation_text", ""),
                    "text_format": material_info.get("text_format", "unknown"),
                    "context_before": material_info.get("context_before", ""),
                    "context_after": material_info.get("context_after", ""),
                    "has_context": material_info.get("has_context", False),
                    "order_in_paper": material_info.get("order_in_paper", -1)
                })
                
            elif material_type == "text":
                # 文本材料特有字段
                citation_data.update({
                    "text_level": material_info.get("text_level", 0),
                    "order_in_paper": material_info.get("order_in_paper", -1)
                })
            
            citation_mapping[citation] = citation_data
        
    # 统计各类型引用数量
    type_counts = {}
    for citation_data in citation_mapping.values():
        material_type = citation_data["type"]
        type_counts[material_type] = type_counts.get(material_type, 0) + 1
    
    if type_counts:
        type_summary = ", ".join([f"{t}:{c}个" for t, c in type_counts.items()])
        print(f"📊 引用类型分布: {type_summary}")
    
    return citation_mapping


def write_section_citations(section_info: Dict, citation_mapping: Dict, numbered_materials: Dict, topic: str = "综述") -> str:
    """
    将章节引用信息写入统一的JSON文件
    
    说明：采用统一文件管理策略，所有writer的引用信息都写入同一个JSON文件
    - 文件命名格式：{topic}_citations_{timestamp}.json
    - 支持多个章节的引用信息累积
    - 确保所有原始信息完整保存，包括img_path等关键字段
    - 支持并发写入保护
    """
    import json
    import os
    import threading
    from datetime import datetime
    from pathlib import Path
    
    # 创建引用目录
    citations_dir = "./chapter_citations"
    os.makedirs(citations_dir, exist_ok=True)
    
    # 查找或生成统一文件名：{topic}_citations_{timestamp}.json
    date_str = datetime.now().strftime("%Y%m%d")
    time_str = datetime.now().strftime("%H%M%S")
    
    # 清理主题名称中的特殊字符
    safe_topic = "".join(c for c in topic if c.isalnum() or c in [' ', '_', '-']).rstrip()
    safe_topic = safe_topic.replace(' ', '_') if safe_topic else "综述"
    
    # 🆕 根据章节ID判断是否为第一个writer
    section_id = section_info.get("id", "")
    is_first_writer = section_id in ["1", "1.1", "01", "001"] or section_id.startswith("1.")
    
    import glob
    pattern = f"{safe_topic}_citations_*.json"
    pattern_path = os.path.join(citations_dir, pattern)
    existing_files = glob.glob(pattern_path)
    
    if is_first_writer:
        # 第一个writer：直接创建新文件（即使已有文件也创建新的，避免冲突）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_topic}_citations_{timestamp}.json"
        filepath = os.path.join(citations_dir, filename)
        print(f"📄 第一个writer创建新引用文件: {filename}")
    else:
        # 后续writer：查找并使用已存在的文件
        if existing_files:
            # 使用已存在的文件（如果有多个，使用最新的）
            filepath = max(existing_files, key=os.path.getmtime)
            print(f"📄 后续writer使用已存在的引用文件: {os.path.basename(filepath)}")
        else:
            # 如果是后续writer但找不到文件，可能第一个writer还没执行
            print(f"⚠️ 章节{section_id}未找到已存在的引用文件，第一个writer可能还未执行")
            # 等待一段时间再重试
            import time
            time.sleep(2)
            existing_files = glob.glob(pattern_path)
            if existing_files:
                filepath = max(existing_files, key=os.path.getmtime)
                print(f"📄 重试后找到引用文件: {os.path.basename(filepath)}")
            else:
                # 仍然找不到，创建临时文件（但使用特殊标记）
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{safe_topic}_citations_temp_{section_id}_{timestamp}.json"
                filepath = os.path.join(citations_dir, filename)
                print(f"⚠️ 创建临时引用文件: {filename} （建议检查writer执行顺序）")
    
    # 使用文件锁确保并发写入安全
    lock_file = f"{filepath}.lock"
    
    # 构建当前章节的引用数据
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    section_data = {
        "section_info": {
            "id": section_info.get("id", ""),
            "title": section_info.get("title", ""),
            "description": section_info.get("description", ""),
            "content_guide": section_info.get("content_guide", ""),
            "subsections": section_info.get("subsections", []),
            "timestamp": timestamp,
            "write_time": time_str
        },
        "citation_summary": {
            "total_materials_available": len(numbered_materials),
            "total_citations_used": len(citation_mapping),
            "usage_rate": round(len(citation_mapping) / len(numbered_materials), 3) if numbered_materials else 0,
            "citations_by_type": {}
        },
        "detailed_citations": citation_mapping,  # ✅ 包含所有类型特有字段的完整引用信息
        "available_materials": numbered_materials,  # ✅ 包含所有原始材料信息
        "material_statistics": {
            "by_type": {},
            "by_paper": {},
            "by_relevance": []
        }
    }
    
    # 统计引用类型
    for citation_id, citation_info in citation_mapping.items():
        material_type = citation_info["type"]
        section_data["citation_summary"]["citations_by_type"][material_type] = \
            section_data["citation_summary"]["citations_by_type"].get(material_type, 0) + 1
    
    # 统计材料类型
    for material_id, material_info in numbered_materials.items():
        material_type = material_info["type"]
        section_data["material_statistics"]["by_type"][material_type] = \
            section_data["material_statistics"]["by_type"].get(material_type, 0) + 1
        
        # 统计论文来源
        paper = material_info["paper"]
        section_data["material_statistics"]["by_paper"][paper] = \
            section_data["material_statistics"]["by_paper"].get(paper, 0) + 1
    
    # 按相关度排序
    relevance_list = []
    for material_id, material_info in numbered_materials.items():
        relevance_list.append({
            "material_id": material_id,
            "relevance_score": material_info["relevance_score"],
            "used": material_id in citation_mapping,
            "type": material_info["type"],
            "paper": material_info["paper"]
        })
    
    section_data["material_statistics"]["by_relevance"] = sorted(
        relevance_list, key=lambda x: x["relevance_score"], reverse=True
    )
    
    # 线程安全的文件写入
    try:
        # 简单的文件锁机制
        while os.path.exists(lock_file):
            import time
            time.sleep(0.1)
        
        # 创建锁文件
        with open(lock_file, 'w') as f:
            f.write(str(os.getpid()))
        
        # 读取现有数据或创建新的数据结构
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                try:
                    citation_data = json.load(f)
                except json.JSONDecodeError:
                    print(f"⚠️ JSON文件损坏，创建新文件: {filepath}")
                    citation_data = {"meta": {}, "sections": {}}
        else:
            citation_data = {
                "meta": {
                    "topic": topic,
                    "created_date": date_str,
                    "last_updated": timestamp,
                    "total_sections": 0,
                    "file_version": "2.0"  # 标记为新版本格式
                },
                "sections": {}
            }
        
        # 更新元信息
        citation_data["meta"]["last_updated"] = timestamp
        citation_data["meta"]["total_sections"] = len(citation_data["sections"]) + 1
        
        # 添加当前章节数据
        section_key = f"section_{section_info.get('id', 'unknown')}"
        citation_data["sections"][section_key] = section_data
        
        # 写入文件
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(citation_data, f, ensure_ascii=False, indent=2)
        
        print(f"📄 章节引用信息已写入: {filepath}")
        print(f"📊 当前文件包含 {citation_data['meta']['total_sections']} 个章节的引用信息")
        
    finally:
        # 清理锁文件
        if os.path.exists(lock_file):
            os.remove(lock_file)
    
    return filepath


def generate_bibliography_from_citations(topic: str = "综述") -> str:
    """
    从统一的引用JSON文件中生成参考文献列表
    
    说明：读取write_section_citations保存的JSON文件，提取所有文本类型引用，
    按论文分组整理，生成格式化的参考文献列表
    
    Args:
        topic: 主题名称，用于定位对应的引用文件
        
    Returns:
        格式化的参考文献字符串
    """
    import json
    import os
    from datetime import datetime
    from collections import defaultdict
    
    # 🆕 查找已存在的引用文件
    citations_dir = "./chapter_citations"
    safe_topic = "".join(c for c in topic if c.isalnum() or c in [' ', '_', '-']).rstrip()
    safe_topic = safe_topic.replace(' ', '_') if safe_topic else "综述"
    
    # 🆕 智能查找引用文件（优先使用正式文件，避免临时文件）
    import glob
    
    # 先查找正式的引用文件（不包含temp标记）
    formal_pattern = f"{safe_topic}_citations_[0-9]*_[0-9]*.json"
    formal_path = os.path.join(citations_dir, formal_pattern)
    formal_files = glob.glob(formal_path)
    
    # 查找所有引用文件（包括临时文件）
    all_pattern = f"{safe_topic}_citations*.json"
    all_path = os.path.join(citations_dir, all_pattern)
    all_files = glob.glob(all_path)
    
    if formal_files:
        # 优先使用正式文件
        filepath = max(formal_files, key=os.path.getmtime)
        print(f"📚 使用正式引用文件: {os.path.basename(filepath)}")
    elif all_files:
        # 如果没有正式文件，使用临时文件
        temp_files = [f for f in all_files if 'temp' in f]
        if temp_files:
            filepath = max(temp_files, key=os.path.getmtime)
            print(f"📚 使用临时引用文件: {os.path.basename(filepath)} （建议等待正式文件生成）")
        else:
            filepath = max(all_files, key=os.path.getmtime)
            print(f"📚 使用引用文件: {os.path.basename(filepath)}")
    else:
        print(f"⚠️ 未找到任何引用文件: {safe_topic}_citations*.json")
        return "# 参考文献\n\n无引用文献。\n"
    
    try:
        # 读取引用数据
        with open(filepath, 'r', encoding='utf-8') as f:
            citation_data = json.load(f)
        
        # 收集所有文本类型的引用
        text_citations = {}  # {citation_key: citation_info}
        
        # 遍历所有章节的详细引用
        sections = citation_data.get("sections", {})
        for section_key, section_data in sections.items():
            detailed_citations = section_data.get("detailed_citations", {})
            
            for citation_key, citation_info in detailed_citations.items():
                if citation_info.get("type") == "text":
                    text_citations[citation_key] = citation_info
        
        if not text_citations:
            print("📚 未找到文本类型引用")
            return "# 参考文献\n\n无引用文献。\n"
        
        # 按论文分组整理引用
        papers_citations = defaultdict(list)  # {paper_name: [citation_info_list]}
        
        for citation_key, citation_info in text_citations.items():
            paper_name = citation_info.get("paper", "未知来源")
            page = citation_info.get("page", "-1")
            
            # 构建引用条目信息
            citation_entry = {
                "citation_key": citation_key,  # 如 "1-文本2"
                "page": page,
                "relevance_score": citation_info.get("relevance_score", 0.0)
            }
            
            papers_citations[paper_name].append(citation_entry)
        
        # 生成格式化的参考文献
        bibliography = "# 参考文献\n\n"
        
        # 按论文名称排序
        sorted_papers = sorted(papers_citations.keys())
        
        for paper_name in sorted_papers:
            citations = papers_citations[paper_name]
            
            # 按相关性评分排序引用
            citations.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            # 添加论文名称
            bibliography += f"**{paper_name}**\n"
            
            # 生成相关文本列表
            citation_list = []
            for i, citation in enumerate(citations, 1):
                citation_key = citation["citation_key"]
                page = citation["page"]
                page_text = f"第{page}页" 
                citation_list.append(f"{i}. 章节{citation_key}，{page_text}")
            
            # 添加相关文本
            bibliography += f"（相关文本）{'; '.join(citation_list)}\n\n"
        
        # 添加统计信息
        total_papers = len(sorted_papers)
        total_citations = len(text_citations)
        bibliography += f"*总计：{total_papers} 篇论文，{total_citations} 个文本引用，页码为-1表明PDF解析问题，请参考引用JSON*\n"
        
        print(f"📚 参考文献生成完成: {total_papers} 篇论文，{total_citations} 个引用")
        return bibliography
        
    except Exception as e:
        print(f"❌ 生成参考文献时出错: {e}")
        return "# 参考文献\n\n参考文献生成失败。\n"


def generate_equations_from_citations(topic: str = "综述") -> str:
    """
    从统一的引用JSON文件中生成参考公式列表
    
    说明：读取write_section_citations保存的JSON文件，提取所有公式类型引用，
    按论文分组整理，生成格式化的参考公式列表
    
    Args:
        topic: 主题名称，用于定位对应的引用文件
        
    Returns:
        格式化的参考公式字符串
    """
    import json
    import os
    from datetime import datetime
    from collections import defaultdict
    
    # 🆕 查找已存在的引用文件
    citations_dir = "./chapter_citations"
    safe_topic = "".join(c for c in topic if c.isalnum() or c in [' ', '_', '-']).rstrip()
    safe_topic = safe_topic.replace(' ', '_') if safe_topic else "综述"
    
    # 🆕 智能查找引用文件（优先使用正式文件，避免临时文件）
    import glob
    
    # 先查找正式的引用文件（不包含temp标记）
    formal_pattern = f"{safe_topic}_citations_[0-9]*_[0-9]*.json"
    formal_path = os.path.join(citations_dir, formal_pattern)
    formal_files = glob.glob(formal_path)
    
    # 查找所有引用文件（包括临时文件）
    all_pattern = f"{safe_topic}_citations*.json"
    all_path = os.path.join(citations_dir, all_pattern)
    all_files = glob.glob(all_path)
    
    if formal_files:
        # 优先使用正式文件
        filepath = max(formal_files, key=os.path.getmtime)
        print(f"🔢 使用正式引用文件生成参考公式: {os.path.basename(filepath)}")
    elif all_files:
        # 如果没有正式文件，使用临时文件
        temp_files = [f for f in all_files if 'temp' in f]
        if temp_files:
            filepath = max(temp_files, key=os.path.getmtime)
            print(f"🔢 使用临时引用文件生成参考公式: {os.path.basename(filepath)} （建议等待正式文件生成）")
        else:
            filepath = max(all_files, key=os.path.getmtime)
            print(f"🔢 使用引用文件生成参考公式: {os.path.basename(filepath)}")
    else:
        print(f"⚠️ 未找到任何引用文件: {safe_topic}_citations*.json")
        return "# 参考公式\n\n无引用公式。\n"
    
    try:
        # 读取引用数据
        with open(filepath, 'r', encoding='utf-8') as f:
            citation_data = json.load(f)
        
        # 收集所有公式类型的引用
        equation_citations = {}  # {citation_key: citation_info}
        
        # 遍历所有章节的详细引用
        sections = citation_data.get("sections", {})
        for section_key, section_data in sections.items():
            detailed_citations = section_data.get("detailed_citations", {})
            
            for citation_key, citation_info in detailed_citations.items():
                if citation_info.get("type") == "equation":
                    equation_citations[citation_key] = citation_info
        
        if not equation_citations:
            print("🔢 未找到公式类型引用")
            return "# 参考公式\n\n无引用公式。\n"
        
        # 按论文分组整理引用
        papers_equations = defaultdict(list)  # {paper_name: [equation_info_list]}
        
        for citation_key, citation_info in equation_citations.items():
            paper_name = citation_info.get("paper", "未知来源")
            page = citation_info.get("page", "-1")
            
            # 构建公式引用条目信息
            equation_entry = {
                "citation_key": citation_key,  # 如 "1-公式1"
                "page": page,
                "equation_text": citation_info.get("equation_text", ""),
                "context_before": citation_info.get("context_before", ""),
                "context_after": citation_info.get("context_after", ""),
                "text_format": citation_info.get("text_format", "latex"),
                "relevance_score": citation_info.get("relevance_score", 0.0)
            }
            
            papers_equations[paper_name].append(equation_entry)
        
        # 生成格式化的参考公式
        equations_section = "# 参考公式\n\n"
        
        # 按论文名称排序
        sorted_papers = sorted(papers_equations.keys())
        
        for paper_name in sorted_papers:
            equations = papers_equations[paper_name]
            
            # 按相关性评分排序公式
            equations.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            # 添加论文名称
            equations_section += f"**{paper_name}(2025)**\n"
            
            # 生成公式列表
            for i, equation in enumerate(equations, 1):
                citation_key = equation["citation_key"]
                page = equation["page"]
                page_text = f"第{page}页"
                
                # 添加基本信息行
                equations_section += f"{i}. 章节{citation_key}，{page_text}\n"
                
                # 添加前文（如果存在）
                if equation["context_before"]:
                    equations_section += f"前文：{equation['context_before']}\n"
                
                # 添加公式
                equation_text = equation["equation_text"]
                if equation_text:
                    equations_section += f"公式：\n{equation_text}\n"
                
                # 添加后文（如果存在）
                if equation["context_after"]:
                    equations_section += f"后文：{equation['context_after']}\n"
                
                # 添加格式信息
                text_format = equation["text_format"]
                equations_section += f"格式：{text_format}\n\n"
            
            equations_section += "\n"
        
        # 添加统计信息
        total_papers = len(sorted_papers)
        total_equations = len(equation_citations)
        equations_section += f"*总计：{total_papers} 篇论文，{total_equations} 个公式引用，页码为-1表明PDF解析问题，请参考引用JSON*\n"
        
        print(f"🔢 参考公式生成完成: {total_papers} 篇论文，{total_equations} 个公式引用")
        return equations_section
        
    except Exception as e:
        print(f"❌ 生成参考公式时出错: {e}")
        return "# 参考公式\n\n参考公式生成失败。\n"


def generate_figures_from_citations(topic: str = "综述") -> str:
    """
    从统一的引用JSON文件中生成参考图片列表
    
    说明：读取write_section_citations保存的JSON文件，提取所有图片类型引用，
    按论文分组整理，生成格式化的参考图片列表，包含图片路径和页码信息
    
    输出格式：
    **(论文名)**
    1. 章节{chapter_ID}-图片NUM，第{page}页，章节{chapter_ID}-图片NUM，第{page}页
    ({full_content})
    ![章节{chapter_ID}](root_image_path/paper_name/img_path)
    
    Args:
        topic: 主题名称，用于定位对应的引用文件
        
    Returns:
        格式化的参考图片字符串，包含页码信息和正确的图片路径结构
    """
    import json
    import os
    from datetime import datetime
    from collections import defaultdict
    
    # 设置根图片路径
    root_image_path = r"D:\Desktop\ZJU\download\dl3\direct_crawler\results"
    
    # 查找已存在的引用文件
    citations_dir = "./chapter_citations"
    safe_topic = "".join(c for c in topic if c.isalnum() or c in [' ', '_', '-']).rstrip()
    safe_topic = safe_topic.replace(' ', '_') if safe_topic else "综述"
    
    # 智能查找引用文件（优先使用正式文件，避免临时文件）
    import glob
    
    # 先查找正式的引用文件（不包含temp标记）
    formal_pattern = f"{safe_topic}_citations_[0-9]*_[0-9]*.json"
    formal_path = os.path.join(citations_dir, formal_pattern)
    formal_files = glob.glob(formal_path)
    
    # 查找所有引用文件（包括临时文件）
    all_pattern = f"{safe_topic}_citations*.json"
    all_path = os.path.join(citations_dir, all_pattern)
    all_files = glob.glob(all_path)
    
    if formal_files:
        # 优先使用正式文件
        filepath = max(formal_files, key=os.path.getmtime)
        print(f"🖼️ 使用正式引用文件生成参考图片: {os.path.basename(filepath)}")
    elif all_files:
        # 如果没有正式文件，使用临时文件
        temp_files = [f for f in all_files if 'temp' in f]
        if temp_files:
            filepath = max(temp_files, key=os.path.getmtime)
            print(f"🖼️ 使用临时引用文件生成参考图片: {os.path.basename(filepath)} （建议等待正式文件生成）")
        else:
            filepath = max(all_files, key=os.path.getmtime)
            print(f"🖼️ 使用引用文件生成参考图片: {os.path.basename(filepath)}")
    else:
        print(f"⚠️ 未找到任何引用文件: {safe_topic}_citations*.json")
        return "# 参考图片\n\n无引用图片。\n"
    
    try:
        # 读取引用数据
        with open(filepath, 'r', encoding='utf-8') as f:
            citation_data = json.load(f)
        
        # 收集所有图片类型的引用
        figure_citations = {}  # {citation_key: citation_info}
        
        # 遍历所有章节的详细引用
        sections = citation_data.get("sections", {})
        for section_key, section_data in sections.items():
            detailed_citations = section_data.get("detailed_citations", {})
            
            for citation_key, citation_info in detailed_citations.items():
                if citation_info.get("type") == "figure":
                    figure_citations[citation_key] = citation_info
        
        if not figure_citations:
            print("🖼️ 未找到图片类型引用")
            return "# 参考图片\n\n无引用图片。\n"
        
        # 按论文分组整理引用，并根据full_content去重
        papers_figures = defaultdict(dict)  # {paper_name: {full_content: [citation_key_list]}}
        
        for citation_key, citation_info in figure_citations.items():
            paper_name = citation_info.get("paper", "未知来源")
            full_content = citation_info.get("full_content", "")
            
            # 如果full_content不存在，则创建新条目
            if full_content not in papers_figures[paper_name]:
                papers_figures[paper_name][full_content] = {
                    "citation_keys": [],
                    "info": citation_info  # 保存引用信息
                }
            
            # 添加章节引用键
            papers_figures[paper_name][full_content]["citation_keys"].append(citation_key)
        
        # 生成格式化的参考图片
        figures_section = "# 参考图片\n\n"
        
        # 按论文名称排序
        sorted_papers = sorted(papers_figures.keys())
        
        for paper_name in sorted_papers:
            figures_dict = papers_figures[paper_name]
            
            # 添加论文名称
            figures_section += f"**({paper_name})(2025)**\n"
            
            # 按full_content组织图片
            figure_num = 1
            for full_content, figure_data in figures_dict.items():
                citation_keys = figure_data["citation_keys"]
                citation_info = figure_data["info"]
                
                # 生成章节引用键列表（包含页码信息）
                citation_keys_with_pages = []
                for key in citation_keys:
                    # 从原始引用数据中获取页码信息
                    key_citation_info = figure_citations.get(key, {})
                    page = key_citation_info.get("page", "-1")
                    page_text = f"第{page}页" if page != "-1" else "页码未知"
                    citation_keys_with_pages.append(f"章节{key}，{page_text}")
                
                citation_keys_text = "，".join(citation_keys_with_pages)
                
                # 添加图片项
                figures_section += f"{figure_num}. {citation_keys_text}\n"
                
                # 添加full_content
                if full_content:
                    figures_section += f"({full_content})\n"
                
                # 添加图片路径
                img_path = citation_info.get("img_path", "")
                paper_name = citation_info.get("paper", "")
                if img_path and paper_name:
                    # 构建完整路径：root_image_path/paper/img_path
                    full_img_path = os.path.join(root_image_path, paper_name, img_path).replace("\\", "/")
                    # 使用第一个章节ID作为图片的alt文本
                    first_citation_key = citation_keys[0] if citation_keys else "图片"
                    figures_section += f"\n\n![章节{first_citation_key}]({full_img_path})\n"
                
                figure_num += 1
                figures_section += "\n"
        
        # 添加统计信息
        total_papers = len(sorted_papers)
        total_figures = len(figure_citations)
        figures_section += f"*总计：{total_papers} 篇论文，{total_figures} 个图片引用，页码为-1表明PDF解析问题，请参考引用JSON*\n"
        
        print(f"🖼️ 参考图片生成完成: {total_papers} 篇论文，{total_figures} 个图片引用")
        return figures_section
        
    except Exception as e:
        print(f"❌ 生成参考图片时出错: {e}")
        return "# 参考图片\n\n参考图片生成失败。\n"


def generate_tables_from_citations(topic: str = "综述") -> str:
    """
    从统一的引用JSON文件中生成参考表格列表
    
    说明：读取write_section_citations保存的JSON文件，提取所有表格类型引用，
    按论文分组整理，生成格式化的参考表格列表，包含表格路径和详细信息
    
    输出格式：
    **(论文名)**
    1. 章节{chapter_ID}-表格NUM，第{page}页
    {table_caption}+{table_footnote}
    Reference: {reference_texts}
    {table_body from original_data}
    ![章节{chapter_ID}](root_image_path/paper_name/img_path)
    
    Args:
        topic: 主题名称，用于定位对应的引用文件
        
    Returns:
        格式化的参考表格字符串，包含表格详细信息和正确的图片路径结构
    """
    import json
    import os
    from datetime import datetime
    from collections import defaultdict
    
    # 设置根图片路径
    root_image_path = r"D:\Desktop\ZJU\download\dl3\direct_crawler\results"
    
    # 查找已存在的引用文件
    citations_dir = "./chapter_citations"
    safe_topic = "".join(c for c in topic if c.isalnum() or c in [' ', '_', '-']).rstrip()
    safe_topic = safe_topic.replace(' ', '_') if safe_topic else "综述"
    
    # 智能查找引用文件（优先使用正式文件，避免临时文件）
    import glob
    
    # 先查找正式的引用文件（不包含temp标记）
    formal_pattern = f"{safe_topic}_citations_[0-9]*_[0-9]*.json"
    formal_path = os.path.join(citations_dir, formal_pattern)
    formal_files = glob.glob(formal_path)
    
    # 查找所有引用文件（包括临时文件）
    all_pattern = f"{safe_topic}_citations*.json"
    all_path = os.path.join(citations_dir, all_pattern)
    all_files = glob.glob(all_path)
    
    if formal_files:
        # 优先使用正式文件
        filepath = max(formal_files, key=os.path.getmtime)
        print(f"📊 使用正式引用文件生成参考表格: {os.path.basename(filepath)}")
    elif all_files:
        # 如果没有正式文件，使用临时文件
        temp_files = [f for f in all_files if 'temp' in f]
        if temp_files:
            filepath = max(temp_files, key=os.path.getmtime)
            print(f"📊 使用临时引用文件生成参考表格: {os.path.basename(filepath)} （建议等待正式文件生成）")
        else:
            filepath = max(all_files, key=os.path.getmtime)
            print(f"📊 使用引用文件生成参考表格: {os.path.basename(filepath)}")
    else:
        print(f"⚠️ 未找到任何引用文件: {safe_topic}_citations*.json")
        return "# 参考表格\n\n无引用表格。\n"
    
    try:
        # 读取引用数据
        with open(filepath, 'r', encoding='utf-8') as f:
            citation_data = json.load(f)
        
        # 收集所有表格类型的引用
        table_citations = {}  # {citation_key: citation_info}
        
        # 遍历所有章节的详细引用
        sections = citation_data.get("sections", {})
        for section_key, section_data in sections.items():
            detailed_citations = section_data.get("detailed_citations", {})
            
            for citation_key, citation_info in detailed_citations.items():
                if citation_info.get("type") == "table":
                    table_citations[citation_key] = citation_info
        
        if not table_citations:
            print("📊 未找到表格类型引用")
            return "# 参考表格\n\n无引用表格。\n"
        
        # 按论文分组整理引用，并根据table_caption去重
        papers_tables = defaultdict(dict)  # {paper_name: {table_caption: [citation_key_list]}}
        
        for citation_key, citation_info in table_citations.items():
            paper_name = citation_info.get("paper", "未知来源")
            table_caption = citation_info.get("table_caption", "")
            
            # 如果table_caption不存在，则创建新条目
            if table_caption not in papers_tables[paper_name]:
                papers_tables[paper_name][table_caption] = {
                    "citation_keys": [],
                    "info": citation_info  # 保存引用信息
                }
            
            # 添加章节引用键
            papers_tables[paper_name][table_caption]["citation_keys"].append(citation_key)
        
        # 生成格式化的参考表格
        tables_section = "# 参考表格\n\n"
        
        # 按论文名称排序
        sorted_papers = sorted(papers_tables.keys())
        
        for paper_name in sorted_papers:
            tables_dict = papers_tables[paper_name]
            
            # 添加论文名称
            tables_section += f"**({paper_name})(2025)**\n"
            
            # 按table_caption组织表格
            table_num = 1
            for table_caption, table_data in tables_dict.items():
                citation_keys = table_data["citation_keys"]
                citation_info = table_data["info"]
                
                # 生成章节引用键列表（包含页码信息）
                citation_keys_with_pages = []
                for key in citation_keys:
                    # 从原始引用数据中获取页码信息
                    key_citation_info = table_citations.get(key, {})
                    page = key_citation_info.get("page", "-1")
                    page_text = f"第{page}页" if page != "-1" else "页码未知"
                    citation_keys_with_pages.append(f"章节{key}，{page_text}")
                
                citation_keys_text = "，".join(citation_keys_with_pages)
                
                # 添加表格项
                tables_section += f"{table_num}. {citation_keys_text}\n"
                
                # 添加table_caption和table_footnote
                table_caption_text = citation_info.get("table_caption", "")
                table_footnote = citation_info.get("table_footnote", "")
                caption_and_footnote = f"{table_caption_text}+{table_footnote}" if table_footnote else table_caption_text
                if caption_and_footnote:
                    tables_section += f"{caption_and_footnote}\n"
                
                # 添加reference_texts
                reference_texts = citation_info.get("reference_texts", "")
                if reference_texts:
                    tables_section += f"Reference: {reference_texts}\n"
                
                # 从original_data中提取table_body
                original_data = citation_info.get("original_data", "")
                table_body = ""
                if original_data:
                    try:
                        # 尝试解析original_data（可能是JSON字符串）
                        if isinstance(original_data, str):
                            original_dict = json.loads(original_data)
                        else:
                            original_dict = original_data
                        
                        table_body = original_dict.get("table_body", "")
                    except (json.JSONDecodeError, AttributeError):
                        # 如果解析失败，尝试直接作为字典处理
                        if isinstance(original_data, dict):
                            table_body = original_data.get("table_body", "")
                        else:
                            table_body = str(original_data)
                
                if table_body:
                    tables_section += f"{table_body}\n"
                
                # 添加表格图片路径
                img_path = citation_info.get("img_path", "")
                if img_path and paper_name:
                    # 构建完整路径：root_image_path/paper/img_path
                    full_img_path = os.path.join(root_image_path, paper_name, img_path).replace("\\", "/")
                    # 使用第一个章节ID作为图片的alt文本
                    first_citation_key = citation_keys[0] if citation_keys else "表格"
                    tables_section += f"\n\n![章节{first_citation_key}]({full_img_path})\n"
                
                table_num += 1
                tables_section += "\n"
        
        # 添加统计信息
        total_papers = len(sorted_papers)
        total_tables = len(table_citations)
        tables_section += f"*总计：{total_papers} 篇论文，{total_tables} 个表格引用，页码为-1表明PDF解析问题，请参考引用JSON*\n"
        
        print(f"📊 参考表格生成完成: {total_papers} 篇论文，{total_tables} 个表格引用")
        return tables_section
        
    except Exception as e:
        print(f"❌ 生成参考表格时出错: {e}")
        return "# 参考表格\n\n参考表格生成失败。\n"


def insert_figures_into_document(full_document: str, topic: str = "综述") -> str:
    """
    在完整文档中自动插入图片
    
    说明：检索全文中的图片引用，在第一次引用位置插入图片链接，
    确保每个图片只插入一次
    
    Args:
        full_document: 完整的文档内容
        topic: 主题名称，用于定位对应的引用文件
        
    Returns:
        插入图片后的文档内容
    """
    import json
    import os
    import re
    from collections import defaultdict
    
    # 设置根图片路径
    root_image_path = r"D:\Desktop\ZJU\download\dl3\direct_crawler\results"
    
    # 查找已存在的引用文件
    citations_dir = "./chapter_citations"
    safe_topic = "".join(c for c in topic if c.isalnum() or c in [' ', '_', '-']).rstrip()
    safe_topic = safe_topic.replace(' ', '_') if safe_topic else "综述"
    
    # 智能查找引用文件（优先使用正式文件，避免临时文件）
    import glob
    
    # 先查找正式的引用文件（不包含temp标记）
    formal_pattern = f"{safe_topic}_citations_[0-9]*_[0-9]*.json"
    formal_path = os.path.join(citations_dir, formal_pattern)
    formal_files = glob.glob(formal_path)
    
    # 查找所有引用文件（包括临时文件）
    all_pattern = f"{safe_topic}_citations*.json"
    all_path = os.path.join(citations_dir, all_pattern)
    all_files = glob.glob(all_path)
    
    if formal_files:
        # 优先使用正式文件
        filepath = max(formal_files, key=os.path.getmtime)
        print(f"🖼️ 使用正式引用文件插入图片: {os.path.basename(filepath)}")
    elif all_files:
        # 如果没有正式文件，使用临时文件
        temp_files = [f for f in all_files if 'temp' in f]
        if temp_files:
            filepath = max(temp_files, key=os.path.getmtime)
            print(f"🖼️ 使用临时引用文件插入图片: {os.path.basename(filepath)} （建议等待正式文件生成）")
        else:
            filepath = max(all_files, key=os.path.getmtime)
            print(f"🖼️ 使用引用文件插入图片: {os.path.basename(filepath)}")
    else:
        print(f"⚠️ 未找到任何引用文件，跳过图片插入")
        return full_document
    
    try:
        # 读取引用数据
        with open(filepath, 'r', encoding='utf-8') as f:
            citation_data = json.load(f)
        
        # 收集所有图片类型的引用
        figure_citations = {}  # {citation_key: citation_info}
        
        # 遍历所有章节的详细引用
        sections = citation_data.get("sections", {})
        for section_key, section_data in sections.items():
            detailed_citations = section_data.get("detailed_citations", {})
            
            for citation_key, citation_info in detailed_citations.items():
                if citation_info.get("type") == "figure":
                    figure_citations[citation_key] = citation_info
        
        if not figure_citations:
            print("🖼️ 未找到图片类型引用，跳过图片插入")
            return full_document
        
        # 追踪已插入的图片，避免重复插入
        inserted_figures = set()
        
        # 按段落分割文档
        paragraphs = full_document.split('\n\n')
        result_paragraphs = []
        
        for paragraph in paragraphs:
            # 添加当前段落
            result_paragraphs.append(paragraph)
            
            # 从段落中提取所有图片引用模式
            # 匹配 "数字-图片数字" 格式的引用
            figure_pattern = r'(\d+-图片\d+)'
            found_references = re.findall(figure_pattern, paragraph)
            
            # 去重并保持顺序
            unique_references = []
            for ref in found_references:
                if ref not in unique_references:
                    unique_references.append(ref)
            
            # 对每个找到的图片引用进行处理
            for citation_key in unique_references:
                # 检查该引用是否在已知的图片引用中
                if citation_key not in figure_citations:
                    continue
                    
                citation_info = figure_citations[citation_key]
                
                # 构建图片唯一标识（基于img_path和paper）
                img_path = citation_info.get("img_path", "")
                paper_name = citation_info.get("paper", "")
                figure_id = f"{paper_name}_{img_path}"
                
                # 如果图片已经插入过，跳过
                if figure_id in inserted_figures:
                    continue
                
                if img_path and paper_name:
                    # 构建完整图片路径
                    full_img_path = os.path.join(root_image_path, paper_name, img_path).replace("\\", "/")
                    
                    # 插入图片（带标题）
                    figure_markdown = f"\n\n章节{citation_key}\n\n![章节{citation_key}]({full_img_path})\n"
                    result_paragraphs.append(figure_markdown)
                    
                    # 标记该图片已插入
                    inserted_figures.add(figure_id)
                    
                    print(f"🖼️ 已插入图片: 章节{citation_key}")
        
        # 重新组合文档
        modified_document = '\n\n'.join(result_paragraphs)
        
        # 清理多余的空行
        modified_document = re.sub(r'\n{4,}', '\n\n\n', modified_document)
        
        total_inserted = len(inserted_figures)
        print(f"🖼️ 图片插入完成: 共插入 {total_inserted} 张图片")
        
        return modified_document
        
    except Exception as e:
        print(f"❌ 插入图片时出错: {e}")
        return full_document


def insert_tables_into_document(full_document: str, topic: str = "综述") -> str:
    """
    在完整文档中自动插入表格图像
    
    说明：检索全文中的表格引用，在第一次引用位置插入表格图像链接，
    确保每个表格图像只插入一次
    
    Args:
        full_document: 完整的文档内容
        topic: 主题名称，用于定位对应的引用文件
        
    Returns:
        插入表格图像后的文档内容
    """
    import json
    import os
    import re
    from collections import defaultdict
    
    # 设置根图片路径
    root_image_path = r"D:\Desktop\ZJU\download\dl3\direct_crawler\results"
    
    # 查找已存在的引用文件
    citations_dir = "./chapter_citations"
    safe_topic = "".join(c for c in topic if c.isalnum() or c in [' ', '_', '-']).rstrip()
    safe_topic = safe_topic.replace(' ', '_') if safe_topic else "综述"
    
    # 智能查找引用文件（优先使用正式文件，避免临时文件）
    import glob
    
    # 先查找正式的引用文件（不包含temp标记）
    formal_pattern = f"{safe_topic}_citations_[0-9]*_[0-9]*.json"
    formal_path = os.path.join(citations_dir, formal_pattern)
    formal_files = glob.glob(formal_path)
    
    # 查找所有引用文件（包括临时文件）
    all_pattern = f"{safe_topic}_citations*.json"
    all_path = os.path.join(citations_dir, all_pattern)
    all_files = glob.glob(all_path)
    
    if formal_files:
        # 优先使用正式文件
        filepath = max(formal_files, key=os.path.getmtime)
        print(f"📊 使用正式引用文件插入表格: {os.path.basename(filepath)}")
    elif all_files:
        # 如果没有正式文件，使用临时文件
        temp_files = [f for f in all_files if 'temp' in f]
        if temp_files:
            filepath = max(temp_files, key=os.path.getmtime)
            print(f"📊 使用临时引用文件插入表格: {os.path.basename(filepath)} （建议等待正式文件生成）")
        else:
            filepath = max(all_files, key=os.path.getmtime)
            print(f"📊 使用引用文件插入表格: {os.path.basename(filepath)}")
    else:
        print(f"⚠️ 未找到任何引用文件，跳过表格插入")
        return full_document
    
    try:
        # 读取引用数据
        with open(filepath, 'r', encoding='utf-8') as f:
            citation_data = json.load(f)
        
        # 收集所有表格类型的引用
        table_citations = {}  # {citation_key: citation_info}
        
        # 遍历所有章节的详细引用
        sections = citation_data.get("sections", {})
        for section_key, section_data in sections.items():
            detailed_citations = section_data.get("detailed_citations", {})
            
            for citation_key, citation_info in detailed_citations.items():
                if citation_info.get("type") == "table":
                    table_citations[citation_key] = citation_info
        
        if not table_citations:
            print("📊 未找到表格类型引用，跳过表格插入")
            return full_document
        
        # 追踪已插入的表格，避免重复插入
        inserted_tables = set()
        
        # 按段落分割文档
        paragraphs = full_document.split('\n\n')
        result_paragraphs = []
        
        for paragraph in paragraphs:
            # 添加当前段落
            result_paragraphs.append(paragraph)
            
            # 从段落中提取所有表格引用模式
            # 匹配 "数字-表格数字" 格式的引用
            table_pattern = r'(\d+-表格\d+)'
            found_references = re.findall(table_pattern, paragraph)
            
            # 去重并保持顺序
            unique_references = []
            for ref in found_references:
                if ref not in unique_references:
                    unique_references.append(ref)
            
            # 对每个找到的表格引用进行处理
            for citation_key in unique_references:
                # 检查该引用是否在已知的表格引用中
                if citation_key not in table_citations:
                    continue
                    
                citation_info = table_citations[citation_key]
                
                # 构建表格唯一标识（基于img_path和paper）
                img_path = citation_info.get("img_path", "")
                paper_name = citation_info.get("paper", "")
                table_id = f"{paper_name}_{img_path}"
                
                # 如果表格已经插入过，跳过
                if table_id in inserted_tables:
                    continue
                
                if img_path and paper_name:
                    # 构建完整表格图像路径
                    full_img_path = os.path.join(root_image_path, paper_name, img_path).replace("\\", "/")
                    
                    # 插入表格图像（带标题）
                    table_markdown = f"\n\n章节{citation_key}\n\n![章节{citation_key}]({full_img_path})\n"
                    result_paragraphs.append(table_markdown)
                    
                    # 标记该表格已插入
                    inserted_tables.add(table_id)
                    
                    print(f"📊 已插入表格: 章节{citation_key}")
        
        # 重新组合文档
        modified_document = '\n\n'.join(result_paragraphs)
        
        # 清理多余的空行
        modified_document = re.sub(r'\n{4,}', '\n\n\n', modified_document)
        
        total_inserted = len(inserted_tables)
        print(f"📊 表格插入完成: 共插入 {total_inserted} 张表格图像")
        
        return modified_document
        
    except Exception as e:
        print(f"❌ 插入表格时出错: {e}")
        return full_document


def generate_table_of_contents(full_document: str, topic: str = "") -> str:
    """
    从完整文档中提取标题并生成目录
    
    Args:
        full_document: 完整的文档内容
        topic: 主题名称，用于过滤重复的主标题
        
    Returns:
        格式化的目录字符串
    """
    import re
    
    # 提取所有标题（支持多级标题）
    headings = []
    lines = full_document.split('\n')
    
    for line in lines:
        line = line.strip()
        if line.startswith('#'):
            # 计算标题级别
            level = 0
            for char in line:
                if char == '#':
                    level += 1
                else:
                    break
            
            # 提取标题文本
            title_text = line[level:].strip()
            
            # 跳过摘要部分、参考文献、参考公式、目录和主标题
            skip_titles = ['摘要', '参考文献', '参考公式', '目录']
            if topic:
                skip_titles.append(topic)  # 添加主标题到跳过列表
                
            if title_text in skip_titles:
                continue
                
            headings.append({
                'level': level,
                'text': title_text,
                'original': line
            })
    
    if not headings:
        return "# 目录\n\n暂无标题结构。\n\n"
    
    # 生成目录
    toc = "# 目录\n\n"
    
    for heading in headings:
        level = heading['level']
        text = heading['text']
        
        # 根据级别添加缩进
        indent = '  ' * (level - 1)  # 第一级不缩进，第二级缩进2空格，以此类推
        
        # 为一级标题添加序号
        if level == 1:
            toc += f"{indent}- **{text}**\n"
        else:
            toc += f"{indent}- {text}\n"
    
    toc += "\n"
    return toc


def format_subtopics_section(subtopics: List[str]) -> str:
    """
    格式化subtopics信息
    
    Args:
        subtopics: subtopics列表
        
    Returns:
        格式化的subtopics字符串
    """
    if not subtopics:
        return ""
    
    subtopics_text = "**研究子主题：**\n\n"
    
    for i, subtopic in enumerate(subtopics, 1):
        subtopics_text += f"{i}. {subtopic}\n"
    
    subtopics_text += "\n"
    return subtopics_text


def build_subsection_guidance(section_info: Dict, section_guidance: Dict = None) -> str:
    """
    构建子章节详细内容指引
    
    脚本目标: 为章节撰写生成详细的子章节内容指引文本
    上下文: 从multi_agent.py中提取的子章节指引生成逻辑
    输入: 
    - section_info: 章节信息字典
    - section_guidance: 章节指引字典（可选）
    执行步骤:
    1. 检查是否有子章节信息
    2. 处理字典或列表两种数据结构
    3. 为每个子章节提取详细信息
    4. 格式化生成指引文本
    5. 处理无子章节的情况
    输出: 格式化的子章节指引文本
    
    Args:
        section_info: 包含章节和子章节信息的字典
        section_guidance: 章节指引信息字典，用于补充子章节详细信息
        
    Returns:
        格式化的子章节详细内容指引字符串
    """
    guidance_text = ""
    
    # 添加子章节信息 - 优先从section_info获取完整信息
    subsections = section_info.get("subsections", {})
    if subsections:
        guidance_text += "\n【子章节详细内容指引】\n"
        
        # 处理两种可能的数据结构：字典或列表
        if isinstance(subsections, dict):
            # 如果subsections是字典，遍历其项
            for subsection_id, subsection in subsections.items():
                subsection_title = subsection.get("title", "")
                
                # 优先从section_info的子章节信息中获取，如果没有则从section_guidance获取
                subsection_guide = {}
                if isinstance(subsection, dict):
                    # 如果subsection本身包含详细信息，直接使用
                    subsection_guide = subsection
                else:
                    # 否则从section_guidance中获取
                    if section_guidance:
                        subsection_guide = section_guidance.get("subsections", {}).get(subsection_id, {})
                
                sub_content_guide = subsection_guide.get("content_guide", "")
                key_points = subsection_guide.get("key_points", [])
                writing_guide = subsection_guide.get("writing_guide", "")
                
                guidance_text += f"\n【{subsection_id} {subsection_title}】\n"
                guidance_text += f"内容概要: {sub_content_guide}\n"
                
                if key_points:
                    guidance_text += "关键要点:\n"
                    for i, point in enumerate(key_points, 1):
                        guidance_text += f"- {point}\n"
                
                if writing_guide:
                    guidance_text += f"写作建议: {writing_guide}\n"
                    
        elif isinstance(subsections, list):
            # 如果subsections是列表，保持原有逻辑
            for subsection in subsections:
                subsection_id = subsection.get("id", "")
                subsection_title = subsection.get("title", "")
                
                # 从section_guidance中获取子章节指引
                subsection_guide = {}
                if section_guidance:
                    subsection_guide = section_guidance.get("subsections", {}).get(subsection_id, {})
                
                sub_content_guide = subsection_guide.get("content_guide", "")
                key_points = subsection_guide.get("key_points", [])
                writing_guide = subsection_guide.get("writing_guide", "")
                
                guidance_text += f"\n【{subsection_id} {subsection_title}】\n"
                guidance_text += f"内容概要: {sub_content_guide}\n"
                
                if key_points:
                    guidance_text += "关键要点:\n"
                    for i, point in enumerate(key_points, 1):
                        guidance_text += f"- {point}\n"
                
                if writing_guide:
                    guidance_text += f"写作建议: {writing_guide}\n"
    else:
        guidance_text += "\n【注意】本章节无明确子章节划分，请根据章节主题和内容指引进行合理的内容组织。\n"
    
    return guidance_text


def _deduplicate_materials(materials: List[Dict]) -> List[Dict]:
    """
    去除重复材料，保留相关度更高的版本
    
    脚本目标: 去除通用材料和特定材料之间的重复项
    上下文: 在合并通用材料和特定材料后调用
    输入: 材料列表，每个材料包含content, paper, page, relevance_score等字段
    执行步骤:
    1. 基于内容hash进行初步去重
    2. 基于paper+page组合进行二次去重
    3. 对重复项保留相关度更高的版本
    4. 输出去重统计信息
    输出: 去重后的材料列表
    
    Args:
        materials: 包含重复项的材料列表
        
    Returns:
        去重后的材料列表
    """
    if not materials:
        return materials
    
    print(f"🔄 开始材料去重，原始材料数量: {len(materials)} 条")
    
    # 用于跟踪已见过的内容和来源
    content_hash_map = {}  # content_hash -> material_with_highest_score
    paper_page_map = {}    # (paper, page) -> material_with_highest_score
    
    deduplicated = []
    content_duplicates = 0
    paper_page_duplicates = 0
    
    for material in materials:
        content = material.get("content", "")
        paper = material.get("paper", "")
        page = material.get("page", 0)
        relevance_score = material.get("relevance_score", 0.0)
        
        # 生成内容的hash值（用于快速比较）
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest() if content else "empty"
        
        # 生成paper+page的组合键
        paper_page_key = (paper, page)
        
        is_duplicate = False
        
        # 1. 检查内容是否重复
        if content_hash in content_hash_map:
            existing_material = content_hash_map[content_hash]
            existing_score = existing_material.get("relevance_score", 0.0)
            
            if relevance_score > existing_score:
                # 新材料相关度更高，替换旧的
                content_hash_map[content_hash] = material
                # 从deduplicated中移除旧材料
                deduplicated = [m for m in deduplicated if hashlib.md5(m.get("content", "").encode('utf-8')).hexdigest() != content_hash]
                deduplicated.append(material)
            
            content_duplicates += 1
            is_duplicate = True
        
        # 2. 检查paper+page组合是否重复
        elif paper_page_key in paper_page_map and paper and page > 0:
            existing_material = paper_page_map[paper_page_key]
            existing_score = existing_material.get("relevance_score", 0.0)
            
            if relevance_score > existing_score:
                # 新材料相关度更高，替换旧的
                paper_page_map[paper_page_key] = material
                # 从deduplicated中移除旧材料
                deduplicated = [m for m in deduplicated if not (m.get("paper") == paper and m.get("page") == page)]
                deduplicated.append(material)
            
            paper_page_duplicates += 1
            is_duplicate = True
        
        # 3. 如果不是重复项，直接添加
        if not is_duplicate:
            content_hash_map[content_hash] = material
            if paper and page > 0:
                paper_page_map[paper_page_key] = material
            deduplicated.append(material)
    
    # 📊 输出去重统计
    total_removed = len(materials) - len(deduplicated)
    print(f"📊 材料去重统计:")
    print(f"  原始材料: {len(materials)} 条")
    print(f"  内容重复: {content_duplicates} 条")
    print(f"  来源重复: {paper_page_duplicates} 条")
    print(f"  总计去除: {total_removed} 条")
    print(f"  最终保留: {len(deduplicated)} 条")
    print(f"  去重率: {(total_removed / len(materials) * 100):.1f}%" if materials else "0%")
    
    return deduplicated


@dataclass
class Citation:
    """单个引用的数据结构"""
    id: str  # 唯一标识符
    authors: str  # 作者
    year: str  # 年份
    title: str  # 标题
    source: str  # 来源（论文名或材料编号）
    page: Optional[int] = None  # 页码
    relevance_score: Optional[float] = None  # 相关度评分


class CitationManager:
    """统一的引用管理系统"""
    
    def __init__(self):
        self.citations: Dict[str, Citation] = {}  # 存储所有引用
        self.citation_counter = 1  # 引用编号计数器
        self.used_citations: List[str] = []  # 已使用的引用ID列表
    
    def add_citation(self, title: str, authors: str = None, year: str = "2025", 
                    source: str = None, page: int = None, relevance_score: float = None) -> str:
        """
        添加一个新引用并返回其引用ID
        
        Args:
            title: 论文标题
            authors: 作者列表
            year: 发表年份
            source: 来源标识
            page: 页码
            relevance_score: 相关度评分
            
        Returns:
            引用ID (如 "1", "2", "3"...)
        """
        # 生成唯一ID基于标题的哈希
        content_hash = hashlib.md5(title.encode('utf-8')).hexdigest()[:8]
        citation_id = str(self.citation_counter)
        
        # 处理作者名称
        if not authors:
            # 从标题或来源中尝试提取作者信息
            if source and any(keyword in source.lower() for keyword in ['et al', 'brown', 'vaswani', 'devlin']):
                authors = extract_authors_from_source(source)
            else:
                authors = "未知作者"
        
        # 处理标题
        if not title and source:
            title = extract_title_from_source(source)
        
        citation = Citation(
            id=citation_id,
            authors=authors,
            year=year,
            title=title,
            source=source or title,
            page=page,
            relevance_score=relevance_score
        )
        
        self.citations[citation_id] = citation
        self.citation_counter += 1
        
        return citation_id
    
    def get_citation_text(self, citation_id: str) -> str:
        """
        获取规范的引用文本
        
        Args:
            citation_id: 引用ID
            
        Returns:
            格式化的引用文本，如 "[1]" 或 "(Smith et al., 2023)"
        """
        if citation_id not in self.citations:
            return f"[未知引用:{citation_id}]"
        
        citation = self.citations[citation_id]
        
        # 使用数字引用格式
        return f"[{citation.id}]"
    
    def get_full_citation(self, citation_id: str) -> str:
        """
        获取完整的引用格式用于参考文献列表
        
        Args:
            citation_id: 引用ID
            
        Returns:
            完整的引用格式
        """
        if citation_id not in self.citations:
            return f"[{citation_id}] 未知引用"
        
        citation = self.citations[citation_id]
        
        # 格式：[编号] 作者. (年份). 标题.
        return f"[{citation.id}] {citation.authors} ({citation.year}). {citation.title}."
    
    def mark_citation_used(self, citation_id: str):
        """标记引用为已使用"""
        if citation_id not in self.used_citations:
            self.used_citations.append(citation_id)
    
    def get_bibliography(self) -> str:
        """
        生成完整的参考文献列表
        
        Returns:
            格式化的参考文献列表
        """
        if not self.used_citations:
            return "# 参考文献\n\n无引用文献。\n"
        
        bibliography = "# 参考文献\n\n"
        
        # 按引用ID排序
        sorted_ids = sorted(self.used_citations, key=lambda x: int(x) if x.isdigit() else float('inf'))
        
        for citation_id in sorted_ids:
            if citation_id in self.citations:
                bibliography += f"{self.get_full_citation(citation_id)}\n\n"
        
        return bibliography
    
    def process_materials_for_citations(self, materials: List[Dict]) -> List[str]:
        """
        处理材料列表，为每个材料添加引用并返回引用ID列表
        
        Args:
            materials: 材料列表
            
        Returns:
            引用ID列表
        """
        citation_ids = []
        
        for material in materials:
            paper_name = material.get("paper", "未知论文")
            relevance_score = material.get("relevance_score", 0.0)
            page = material.get("page", None)
            
            # 添加引用
            citation_id = self.add_citation(
                title=paper_name,
                source=paper_name,
                page=page,
                relevance_score=relevance_score
            )
            citation_ids.append(citation_id)
            self.mark_citation_used(citation_id)
        
        return citation_ids
    
    def get_citation_stats(self) -> Dict:
        """
        获取引用统计信息
        
        Returns:
            包含引用统计的字典
        """
        return {
            "total_citations": len(self.citations),
            "used_citations": len(self.used_citations),
            "unused_citations": len(self.citations) - len(self.used_citations),
            "usage_rate": len(self.used_citations) / len(self.citations) if self.citations else 0
        }
    
    def clear_unused_citations(self):
        """清理未使用的引用"""
        unused_ids = [cid for cid in self.citations.keys() if cid not in self.used_citations]
        for cid in unused_ids:
            del self.citations[cid]
    
    def export_citations_json(self) -> Dict:
        """
        导出引用数据为JSON格式
        
        Returns:
            包含所有引用信息的字典
        """
        return {
            "citations": {
                cid: {
                    "id": citation.id,
                    "authors": citation.authors,
                    "year": citation.year,
                    "title": citation.title,
                    "source": citation.source,
                    "page": citation.page,
                    "relevance_score": citation.relevance_score
                }
                for cid, citation in self.citations.items()
            },
            "used_citations": self.used_citations,
            "stats": self.get_citation_stats()
        } 



def clean_material_references_enriched(enriched: Dict) -> Dict:
    """
    清洗丰富大纲中的材料引用信息
    删除所有形如"相关材料"、"材料NUM"、"(材料X, 材料Y)"等引用内容
    """
    import re
    import copy
    
    if not enriched or not isinstance(enriched, dict):
        return enriched
    
    # 深拷贝以避免修改原始数据
    cleaned = copy.deepcopy(enriched)
    
    def clean_text(text: str) -> str:
        """清洗单个文本中的材料引用"""
        if not isinstance(text, str):
            return text
        
        # 定义需要清理的模式
        patterns = [
            r'（相关材料[：:][^）]*）',           # （相关材料：材料36, 材料8, 材料98, 材料89）
            r'\(相关材料[：:][^)]*\)',           # (相关材料：材料36, 材料8, 材料98, 材料89)
            r'相关材料[：:][^。，\n]*[。，]?',      # 相关材料：材料36, 材料8, 材料98, 材料89
            r'参考材料[：:][^。，\n]*[。，]?',      # 参考材料：材料1, 材料2
            r'（材料\d+[，,、\s]*[材料\d+，,、\s]*[^）]*）',  # （材料36, 材料8, 材料98）
            r'\(材料\d+[，,、\s]*[材料\d+，,、\s]*[^)]*\)',  # (材料36, 材料8, 材料98)
            r'材料\d+[，,、\s]*(?:材料\d+[，,、\s]*)*',     # 材料36, 材料8, 材料98
            r'参见材料\d+',                        # 参见材料1
            r'见材料\d+',                         # 见材料1
            r'\[材料\d+\]',                      # [材料1]
        ]
        
        cleaned_text = text
        for pattern in patterns:
            cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)
        
        # 清理多余的空格、标点符号
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # 多个空格变成单个
        cleaned_text = re.sub(r'[，,]\s*[，,]', '，', cleaned_text)  # 重复逗号
        cleaned_text = re.sub(r'[。]\s*[。]', '。', cleaned_text)  # 重复句号
        cleaned_text = re.sub(r'\s*[，,]\s*$', '', cleaned_text)  # 结尾的逗号
        cleaned_text = re.sub(r'^\s*[，,]\s*', '', cleaned_text)  # 开头的逗号
        cleaned_text = cleaned_text.strip()
        
        return cleaned_text
    
    def clean_list(items: list) -> list:
        """清洗列表中的每个字符串项"""
        if not isinstance(items, list):
            return items
        return [clean_text(item) if isinstance(item, str) else item for item in items]
    
    # 清洗顶级字段
    if 'topic' in cleaned:
        cleaned['topic'] = clean_text(cleaned['topic'])
    if 'overview' in cleaned:
        cleaned['overview'] = clean_text(cleaned['overview'])
    
    # 清洗章节数据
    chapters = cleaned.get('chapters', {})
    if isinstance(chapters, dict):
        # 字典格式的章节
        for chapter_id, chapter in chapters.items():
            if isinstance(chapter, dict):
                # 清洗章节级别的字段
                if 'title' in chapter:
                    chapter['title'] = clean_text(chapter['title'])
                if 'content_guide' in chapter:
                    chapter['content_guide'] = clean_text(chapter['content_guide'])
                if 'keywords' in chapter:
                    chapter['keywords'] = clean_list(chapter['keywords'])
                if 'research_focus' in chapter:
                    chapter['research_focus'] = clean_list(chapter['research_focus'])
                
                # 清洗子章节
                subsections = chapter.get('subsections', {})
                if isinstance(subsections, dict):
                    # 字典格式的子章节
                    for subsection_id, subsection in subsections.items():
                        if isinstance(subsection, dict):
                            if 'title' in subsection:
                                subsection['title'] = clean_text(subsection['title'])
                            if 'content_guide' in subsection:
                                subsection['content_guide'] = clean_text(subsection['content_guide'])
                            if 'key_points' in subsection:
                                subsection['key_points'] = clean_list(subsection['key_points'])
                            if 'writing_guide' in subsection:
                                subsection['writing_guide'] = clean_text(subsection['writing_guide'])
                elif isinstance(subsections, list):
                    # 列表格式的子章节
                    for subsection in subsections:
                        if isinstance(subsection, dict):
                            if 'title' in subsection:
                                subsection['title'] = clean_text(subsection['title'])
                            if 'content_guide' in subsection:
                                subsection['content_guide'] = clean_text(subsection['content_guide'])
                            if 'key_points' in subsection:
                                subsection['key_points'] = clean_list(subsection['key_points'])
                            if 'writing_guide' in subsection:
                                subsection['writing_guide'] = clean_text(subsection['writing_guide'])
    
    elif isinstance(chapters, list):
        # 列表格式的章节
        for chapter in chapters:
            if isinstance(chapter, dict):
                # 清洗章节级别的字段
                if 'title' in chapter:
                    chapter['title'] = clean_text(chapter['title'])
                if 'content_guide' in chapter:
                    chapter['content_guide'] = clean_text(chapter['content_guide'])
                if 'keywords' in chapter:
                    chapter['keywords'] = clean_list(chapter['keywords'])
                if 'research_focus' in chapter:
                    chapter['research_focus'] = clean_list(chapter['research_focus'])
                
                # 清洗子章节
                subsections = chapter.get('subsections', [])
                if isinstance(subsections, list):
                    for subsection in subsections:
                        if isinstance(subsection, dict):
                            if 'title' in subsection:
                                subsection['title'] = clean_text(subsection['title'])
                            if 'content_guide' in subsection:
                                subsection['content_guide'] = clean_text(subsection['content_guide'])
                            if 'key_points' in subsection:
                                subsection['key_points'] = clean_list(subsection['key_points'])
                            if 'writing_guide' in subsection:
                                subsection['writing_guide'] = clean_text(subsection['writing_guide'])
                elif isinstance(subsections, dict):
                    for subsection_id, subsection in subsections.items():
                        if isinstance(subsection, dict):
                            if 'title' in subsection:
                                subsection['title'] = clean_text(subsection['title'])
                            if 'content_guide' in subsection:
                                subsection['content_guide'] = clean_text(subsection['content_guide'])
                            if 'key_points' in subsection:
                                subsection['key_points'] = clean_list(subsection['key_points'])
                            if 'writing_guide' in subsection:
                                subsection['writing_guide'] = clean_text(subsection['writing_guide'])
    
    print("🧹 已清洗丰富大纲中的材料引用信息")
    return cleaned



def _parse_interpretation_response(response: str) -> Dict:
    """
    解析LLM的综述主题解释响应
    
    Args:
        response: LLM的原始响应
        
    Returns:
        解析后的结果字典
    """
    result = {
        "standardized_topic": "",
        "standardized_subtopics": [],
        "analysis": ""
    }
    
    try:
        # 查找解析结果部分
        start_marker = "===解析结果开始==="
        end_marker = "===解析结果结束==="
        
        start_idx = response.find(start_marker)
        end_idx = response.find(end_marker)
        
        if start_idx == -1 or end_idx == -1:
            raise ValueError("未找到解析结果标记")
        
        result_section = response[start_idx + len(start_marker):end_idx].strip()
        
        # 解析综述核心主题
        topic_match = re.search(r'【综述核心主题】\s*\n?(.+?)(?=\n【|$)', result_section, re.DOTALL)
        if topic_match:
            result["standardized_topic"] = topic_match.group(1).strip()
        
        # 解析综述关键词矩阵
        subtopics_list = []
        
        # 提取关键词矩阵部分
        matrix_match = re.search(r'【综述关键词矩阵】\s*\n?(.*?)(?=\n【|$)', result_section, re.DOTALL)
        if matrix_match:
            matrix_content = matrix_match.group(1).strip()
            
            # 解析各个维度的关键词
            categories = [
                (r'核心技术方法:\s*(.+?)(?=\n|$)', "core_methods"),
                (r'重要应用领域:\s*(.+?)(?=\n|$)', "applications"),
                (r'评估与标准:\s*(.+?)(?=\n|$)', "evaluation"),
                (r'交叉与前沿:\s*(.+?)(?=\n|$)', "frontiers")
            ]
            
            for pattern, category in categories:
                cat_match = re.search(pattern, matrix_content, re.MULTILINE)
                if cat_match:
                    keywords_text = cat_match.group(1).strip()
                    if keywords_text:
                        # 分割并清理关键词
                        keywords = [k.strip() for k in keywords_text.split(',') if k.strip()]
                        subtopics_list.extend(keywords)
        
        # 如果矩阵解析失败，尝试解析旧格式作为fallback
        if not subtopics_list:
            subtopics_match = re.search(r'【标准化次要主题】\s*\n?(.+?)(?=\n【|$)', result_section, re.DOTALL)
            if subtopics_match:
                subtopics_text = subtopics_match.group(1).strip()
                if subtopics_text and subtopics_text != "无":
                    subtopics_list = [s.strip() for s in subtopics_text.split(',') if s.strip()]
        
        result["standardized_subtopics"] = subtopics_list
        
        # 解析综述策略分析
        analysis_match = re.search(r'【综述策略分析】\s*\n?(.*?)(?=\n===|$)', result_section, re.DOTALL)
        if analysis_match:
            result["analysis"] = analysis_match.group(1).strip()
        
        # 如果没有找到策略分析，尝试旧格式
        if not result["analysis"]:
            old_analysis_match = re.search(r'【解析分析】\s*\n?(.*?)(?=\n===|$)', result_section, re.DOTALL)
            if old_analysis_match:
                result["analysis"] = old_analysis_match.group(1).strip()
        
    except Exception as e:
        print(f"⚠️ 解析响应格式时出错: {e}")
        # 尝试简单的文本提取作为fallback
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('===') and not line.startswith('【'):
                if not result["standardized_topic"] and len(line.split()) <= 5:
                    result["standardized_topic"] = line
                    break
    
    return result


def _format_enrichment_for_analysis( enriched: Dict) -> str:
    """格式化当前丰富大纲用于分析"""
    if not enriched:
        return "当前丰富大纲为空"
    
    
    formatted = f"主题: {enriched.get('topic', '未知')}\n"
    formatted += f"概述: {enriched.get('overview', '无概述')}\n\n"
    
    chapters = enriched.get("chapters", {})
    
    # 🔧 修改：同时支持列表和字典格式的章节数据
    if isinstance(chapters, list):
        # 处理列表格式的章节 (如果存在)
        for chapter in chapters:
            chapter_id = chapter.get('id', '?')
            chapter_title = chapter.get('title', '未命名章节')
            content_guide = chapter.get('content_guide', '无指引')
            
            formatted += f"# 第{chapter_id}章：{chapter_title}\n"
            formatted += f"章节内容指引: {content_guide}\n"
            
            # 添加关键词
            keywords = chapter.get("keywords", [])
            if keywords:
                formatted += f"关键词: {', '.join(keywords)}\n"
            
            # 添加研究领域
            research_focus = chapter.get("research_focus", [])
            if research_focus:
                formatted += f"研究领域: {', '.join(research_focus)}\n"
            
            # 处理子章节
            subsections = chapter.get("subsections", [])
            if isinstance(subsections, list):
                for subsection in subsections:
                    sub_id = subsection.get('id', '?')
                    sub_title = subsection.get('title', '未命名子章节')
                    sub_content_guide = subsection.get('content_guide', '无概要')
                    
                    formatted += f"## {sub_id} {sub_title}\n"
                    formatted += f"内容概要: {sub_content_guide}\n"
                    
                    # 关键要点
                    key_points = subsection.get('key_points', [])
                    if key_points:
                        formatted += f"关键要点: {', '.join(key_points)}\n"
                    
                    # 写作建议
                    writing_guide = subsection.get('writing_guide', '')
                    if writing_guide:
                        formatted += f"写作建议: {writing_guide}\n"
            elif isinstance(subsections, dict):
                # 处理字典格式的子章节
                for sub_id, subsection in subsections.items():
                    formatted += f"## {sub_id} {subsection.get('title', '未命名子章节')}\n"
                    formatted += f"内容概要: {subsection.get('content_guide', '无概要')}\n"
                    
                    # 关键要点
                    key_points = subsection.get('key_points', [])
                    if key_points:
                        formatted += f"关键要点: {', '.join(key_points)}\n"
                    
                    # 写作建议
                    writing_guide = subsection.get('writing_guide', '')
                    if writing_guide:
                        formatted += f"写作建议: {writing_guide}\n"
            
            formatted += "\n"
            
    elif isinstance(chapters, dict):
        # 处理字典格式的章节 (原有逻辑)
        for chapter_id, chapter in chapters.items():
            formatted += f"# 第{chapter_id}章：{chapter.get('title', '未命名章节')}\n"
            formatted += f"章节内容指引: {chapter.get('content_guide', '无指引')}\n"
            
            # 添加关键词
            keywords = chapter.get("keywords", [])
            if keywords:
                formatted += f"关键词: {', '.join(keywords)}\n"
            
            # 添加研究领域
            research_focus = chapter.get("research_focus", [])
            if research_focus:
                formatted += f"研究领域: {', '.join(research_focus)}\n"
            
            # 添加子章节
            subsections = chapter.get("subsections", {})
            if isinstance(subsections, dict):
                for sub_id, subsection in subsections.items():
                    formatted += f"## {sub_id} {subsection.get('title', '未命名子章节')}\n"
                    formatted += f"内容概要: {subsection.get('content_guide', '无概要')}\n"
                    
                    # 关键要点
                    key_points = subsection.get('key_points', [])
                    if key_points:
                        formatted += f"关键要点: {', '.join(key_points)}\n"
                    
                    # 写作建议
                    writing_guide = subsection.get('writing_guide', '')
                    if writing_guide:
                        formatted += f"写作建议: {writing_guide}\n"
            elif isinstance(subsections, list):
                # 处理列表格式的子章节
                for subsection in subsections:
                    sub_id = subsection.get('id', '?')
                    sub_title = subsection.get('title', '未命名子章节')
                    sub_content_guide = subsection.get('content_guide', '无概要')
                    
                    formatted += f"## {sub_id} {sub_title}\n"
                    formatted += f"内容概要: {sub_content_guide}\n"
                    
                    # 关键要点
                    key_points = subsection.get('key_points', [])
                    if key_points:
                        formatted += f"关键要点: {', '.join(key_points)}\n"
                    
                    # 写作建议
                    writing_guide = subsection.get('writing_guide', '')
                    if writing_guide:
                        formatted += f"写作建议: {writing_guide}\n"
            
            formatted += "\n"
    else:
        formatted += "章节信息格式异常\n"
    
    return formatted


def _format_global_outline_for_prompt(global_summary: Dict) -> str:
    """
    将全局概览格式化为适合LLM理解的文本
    控制输出长度，突出结构关系
    """
    if not global_summary or not global_summary.get("chapters"):
        return "无全局结构信息"
    
    formatted_text = ""
    chapters = global_summary.get("chapters", {})
    
    # 按章节ID排序
    sorted_chapters = sorted(chapters.items(), key=lambda x: str(x[0]))
    
    for chapter_id, chapter_info in sorted_chapters:
        chapter_title = chapter_info.get("title", "")
        content_guide = chapter_info.get("content_guide", "")
        
        # 添加章节信息
        formatted_text += f"第{chapter_id}章: {chapter_title}\n"
        
        # 添加简化的内容指引（截取前200字符）
        if content_guide:
            short_guide = content_guide[:200] + "..." if len(content_guide) > 200 else content_guide
            formatted_text += f"  核心内容: {short_guide}\n"
        
        # 添加子章节标题
        subsections = chapter_info.get("subsections", {})
        if subsections:
            formatted_text += "  子章节: "
            subsection_titles = []
            for subsection_id, subsection_info in sorted(subsections.items(), key=lambda x: str(x[0])):
                subsection_title = subsection_info.get("title", "")
                if subsection_title:
                    subsection_titles.append(f"{subsection_id} {subsection_title}")
            formatted_text += " | ".join(subsection_titles) + "\n"
        
        formatted_text += "\n"
    
    return formatted_text.strip()


def _extract_figure_caption(content: str) -> str:
    """从图片内容中提取caption部分"""
    if not content:
        return ""
    
    # 图片content格式: "... Content: ..."我们只需要Content之前的部分
    if " Content:" in content:
        return content.split(" Content:")[0].strip()
    else:
        return content.strip()

def _extract_table_caption(content: str) -> str:
    """从表格内容中提取caption部分"""
    if not content:
        return ""
    
    # 图片content格式: "... Content: ..."我们只需要Content之前的部分
    if " Content:" in content:
        return content.split(" Content:")[0].strip()
    else:
        return content.strip()

def _extract_quality_evaluation_writing(content: str) -> Dict:
    """提取内容质量评估信息"""
    evaluation = {"scores": {}}
    
    # 评估维度映射
    dimensions = {
        "学术严谨性": "academic_rigor",
        "内容完整性": "content_completeness", 
        "文献融合度": "literature_integration",
        "多模态材料引用": "multimodal_material_citation",
        "图表分析深度": "chart_analysis_depth",
        "论述深度": "argument_depth",
        "表达质量": "expression_quality"
    }
    
    # 提取各维度评分
    for chinese_name, english_key in dimensions.items():
        pattern = rf"{chinese_name}:\s*(\d+(?:\.\d+)?)/10"
        match = re.search(pattern, content)
        if match:
            evaluation["scores"][english_key] = float(match.group(1))
    
    # 提取综合质量评分
    overall_pattern = r"综合质量:\s*(\d+(?:\.\d+)?)/10"
    match = re.search(overall_pattern, content)
    if match:
        evaluation["scores"]["overall_quality"] = float(match.group(1))
    
    return evaluation

def _extract_difference_analysis(content: str) -> Dict:
    """提取增量改进分析"""
    analysis = {}
    
    # 提取内容增量度百分比
    increment_pattern = r"内容增量度:\s*(\d+(?:\.\d+)?)%"
    match = re.search(increment_pattern, content)
    if match:
        analysis["increment_rate"] = float(match.group(1)) / 100
    
    # 兼容旧格式的差异度
    diff_pattern = r"内容差异度:\s*(\d+(?:\.\d+)?)%"
    match = re.search(diff_pattern, content)
    if match:
        analysis["difference_rate"] = float(match.group(1)) / 100
    
    # 提取主要新增点
    additions_pattern = r"主要新增点:\s*([^\n]+)"
    match = re.search(additions_pattern, content)
    if match:
        analysis["main_additions"] = match.group(1).strip()
    
    # 兼容旧格式的主要变化点
    changes_pattern = r"主要变化点:\s*([^\n]+)"
    match = re.search(changes_pattern, content)
    if match:
        analysis["main_changes"] = match.group(1).strip()
    
    # 提取质量增值度
    value_pattern = r"质量增值度:\s*([^\n]+)"
    match = re.search(value_pattern, content)
    if match:
        analysis["quality_value_added"] = match.group(1).strip()
    
    # 兼容旧格式的质量提升度
    quality_pattern = r"质量提升度:\s*([^\n]+)"
    match = re.search(quality_pattern, content)
    if match:
        analysis["quality_improvement"] = match.group(1).strip()
    
    # 提取新增引用统计
    citation_pattern = r"新增引用统计:\s*([^\n]+)"
    match = re.search(citation_pattern, content)
    if match:
        analysis["new_citations"] = match.group(1).strip()
    
    return analysis

def _extract_iteration_decision(content: str) -> bool:
    """提取迭代决策"""
    decision_pattern = r"【是否继续迭代】\s*(是|否)"
    match = re.search(decision_pattern, content)
    if match:
        decision = match.group(1).strip()
        return decision == "是"
    return False

def _clean_numeric_content(content: str, threshold: float = 0.1) -> str:
    """
    清理包含大量数字的内容
    
    Args:
        content: 要检查的内容
        threshold: 数字占比阈值，超过此值将清理数字（默认30%）
    
    Returns:
        清理后的内容
    """
    if not content:
        return content
    
    # 统计数字字符数量
    digit_count = sum(1 for char in content if char.isdigit())
    total_chars = len(content)
    
    if total_chars == 0:
        return content
    
    digit_ratio = digit_count / total_chars
    
    # 如果数字占比超过阈值，则智能清理数字和相关无意义符号
    if digit_ratio > threshold:
        import re
        
        # 步骤1: 移除数字
        cleaned_content = re.sub(r'\d', '', content)
        
        # 步骤2: 清理由数字移除导致的无意义标点符号
        # 移除连续的点、逗号、破折号等标点符号
        cleaned_content = re.sub(r'[.,\-_]+\s*[.,\-_]*', ' ', cleaned_content)
        
        # 步骤3: 清理多余的标点符号和空格组合
        # 移除单独的标点符号（被空格包围的）
        cleaned_content = re.sub(r'\s+[.,\-_:;]+\s+', ' ', cleaned_content)
        
        # 步骤4: 清理行首行尾的标点符号
        lines = cleaned_content.split('\n')
        cleaned_lines = []
        for line in lines:
            # 移除行首行尾的标点符号和空格
            line = re.sub(r'^[.,\-_:;\s]+|[.,\-_:;\s]+$', '', line.strip())
            # 只保留有实际内容的行（至少包含一个字母）
            if line and re.search(r'[a-zA-Z]', line):
                cleaned_lines.append(line)
        
        # 步骤5: 重新组合并最终清理
        cleaned_content = '\n'.join(cleaned_lines)
        # 清理多余的空格和换行
        cleaned_content = re.sub(r'\s+', ' ', cleaned_content).strip()
        # 移除剩余的孤立标点符号
        cleaned_content = re.sub(r'\s+[.,\-_:;]\s+', ' ', cleaned_content)
        
        return cleaned_content
    
    return content


def _format_materials_for_writing_prompt(numbered_materials: Dict, iteration: int = 0) -> tuple:
    """
    格式化材料用于写作提示词，基于迭代次数重新编号确保连续性
    
    Args:
        numbered_materials: 编号材料字典
        iteration: 当前迭代次数（0表示初始写作，1、2、3表示后续迭代）
        
    Returns:
        tuple: (格式化的材料字符串, 重新编号的材料字典)
    """
    
    
    if not numbered_materials:
        return "⚠️ 当前没有可用的研究材料\n", {}
    
    # 🆕 定义每轮迭代的材料数量配置
    materials_per_iteration = {
        "text": 60,      # 每轮文本材料数量
        "equation": 10,  # 每轮公式材料数量  
        "figure": 20,    # 每轮图片材料数量
        "table": 20      # 每轮表格材料数量
    }
    
    # 按类型分组
    text_materials = []
    equation_materials = []
    figure_materials = []
    table_materials = []
    
    # 🔍 分析每个编号材料的类型
    type_distribution = {}
    
    for material_id, material_info in numbered_materials.items():
        material_type = material_info.get("type", "text")
        type_distribution[material_type] = type_distribution.get(material_type, 0) + 1
        
        if material_type == "text":
            text_materials.append((material_id, material_info))
        elif material_type == "equation":
            equation_materials.append((material_id, material_info))
        elif material_type == "figure":
            figure_materials.append((material_id, material_info))
        elif material_type == "table":
            table_materials.append((material_id, material_info))

    # 🆕 计算每种材料类型的起始编号
    text_start_num = iteration * materials_per_iteration["text"] + 1
    equation_start_num = iteration * materials_per_iteration["equation"] + 1
    figure_start_num = iteration * materials_per_iteration["figure"] + 1
    table_start_num = iteration * materials_per_iteration["table"] + 1
    
    
    formatted = ""
    # 🆕 创建重新编号的材料字典，用于后续的引用映射
    renumbered_materials = {}
    
    # 显示文本材料（使用连续编号）
    if text_materials:
        formatted += "\n **文本材料**\n"
        for i, (material_id, material_info) in enumerate(text_materials):
            new_material_num = text_start_num + i
            new_material_key = f"文本{new_material_num}"
            relevance = material_info.get("relevance_score", 0)
            paper = material_info.get("paper", "未知来源")
            content = material_info.get("content", "")
            display_content = content[:2000] + "..." if len(content) > 2000 else content
            formatted += f"{new_material_key} (相关度: {relevance:.2f}, 来源ID: {paper[:3]}):\n{display_content}\n\n"
            
            # 🆕 添加到重新编号的字典中
            renumbered_materials[new_material_key] = material_info
    
    # 显示公式材料（使用连续编号）
    if equation_materials:
        formatted += "\n **相关公式**\n"
        for i, (material_id, material_info) in enumerate(equation_materials):
            new_material_num = equation_start_num + i
            new_material_key = f"公式{new_material_num}"
            relevance = material_info.get("relevance_score", 0)
            paper = material_info.get("paper", "未知来源")
            content = material_info.get("content", "")
            formatted += f"{new_material_key} (相关度: {relevance:.2f}, 来源ID: {paper[:3]}):\n{content}\n\n"
            
            # 🆕 添加到重新编号的字典中
            renumbered_materials[new_material_key] = material_info
    
    # 显示图片材料（使用连续编号）
    if figure_materials:
        formatted += "\n **图片资料**\n"
        for i, (material_id, material_info) in enumerate(figure_materials):
            new_material_num = figure_start_num + i
            new_material_key = f"图片{new_material_num}"
            relevance = material_info.get("relevance_score", 0)
            paper = material_info.get("paper", "未知来源")
            content = material_info.get("content", "")
            display_content = content[:1000] + "..." if len(content) > 1000 else content
            formatted += f"{new_material_key} (相关度: {relevance:.2f}, 来源ID: {paper[:3]}):\n{display_content}\n\n"
            
            # 🆕 添加到重新编号的字典中
            renumbered_materials[new_material_key] = material_info
    
    # 显示表格材料（使用连续编号）
    if table_materials:
        formatted += "\n **表格数据**\n"
        for i, (material_id, material_info) in enumerate(table_materials):
            new_material_num = table_start_num + i
            new_material_key = f"表格{new_material_num}"
            relevance = material_info.get("relevance_score", 0)
            paper = material_info.get("paper", "未知来源")
            content = material_info.get("content", "")
            display_content = content[:1000] + "..." if len(content) > 1000 else content
            # 清理大量数字的表格内容
            display_content = _clean_numeric_content(display_content)
            formatted += f"{new_material_key} (相关度: {relevance:.2f}, 来源ID: {paper[:3]}):\n{display_content}\n\n"
            
            # 🆕 添加到重新编号的字典中
            renumbered_materials[new_material_key] = material_info
    
    # 材料统计信息
    total_materials = len(text_materials) + len(equation_materials) + len(figure_materials) + len(table_materials)
    
    # 🆕 显示编号范围信息
    ranges_info = []
    if text_materials:
        ranges_info.append(f"文本{text_start_num}-{text_start_num + len(text_materials) - 1}")
    if equation_materials:
        ranges_info.append(f"公式{equation_start_num}-{equation_start_num + len(equation_materials) - 1}")
    if figure_materials:
        ranges_info.append(f"图片{figure_start_num}-{figure_start_num + len(figure_materials) - 1}")
    if table_materials:
        ranges_info.append(f"表格{table_start_num}-{table_start_num + len(table_materials) - 1}")
            
    formatted += f"\n📊 **材料统计**: 共{total_materials}条材料 (文本:{len(text_materials)}, 公式:{len(equation_materials)}, 图:{len(figure_materials)}, 表格:{len(table_materials)})\n"
    formatted += f"📊 **编号范围**: {', '.join(ranges_info)}\n"
    
    return formatted, renumbered_materials

def _format_content_for_analysis( current_content: Dict) -> str:
    """格式化当前内容用于分析，类似Enricher的_format_enrichment_for_analysis"""
    if not current_content:
        return "⚠️ 当前没有内容"
    
    content_text = current_content.get("content", "")
    if not content_text:
        return "⚠️ 内容为空"
    
    # 显示内容基本信息
    word_count = len(content_text.split())
    status = current_content.get("status", "unknown")
    materials_used = current_content.get("materials_used", 0)
    iterations = current_content.get("iterations_completed", 0)
    
    formatted = f"""

【当前章节内容】
{content_text}

【质量评分历史】
"""
    
    quality_scores = current_content.get("quality_scores", {})
    if quality_scores:
        for dimension, score in quality_scores.items():
            formatted += f"{dimension}: {score}/10\n"
    else:
        formatted += "无历史评分记录\n"
    
    return formatted

def _format_global_context_for_analysis(main_topic: str, subtopics: List[str] = None, global_outline_summary: Dict = None) -> str:
    """格式化全局综述框架用于分析"""
    formatted = ""
    
    # 添加全局上下文信息
    if main_topic:
        formatted += f"综述主题: {main_topic}\n"
        if subtopics:
            formatted += f"综述子主题: {', '.join(subtopics)}\n"
    
    # 添加全局结构概览
    if global_outline_summary:
        formatted += f"""
【全局综述结构概览】
请注意，你正在优化的是一个完整综述的一部分。以下是整个综述的结构概览，请确保你的内容与整体结构保持一致：

综述总体框架：
{_format_global_outline_for_prompt(global_outline_summary)}

【结构一致性要求】
1. 避免与其他章节内容重复，保持内容边界清晰
2. 适当提及与其他章节的逻辑关系
3. 确保内容深度和详细程度与整个综述的学术水平保持一致
4. 注意当前章节在整体结构中的位置和作用
"""
    else:
        formatted += "无全局结构信息\n"
    
    return formatted

def _parse_writing_refinement_response(response: str, iteration: int, current_content: Dict, renumbered_materials: Dict = None) -> Dict:
    """解析写作优化响应，提取优化后内容和决策信息"""
    try:
        # 查找优化结果部分
        start_marker = "===写作优化结果开始==="
        end_marker = "===写作优化结果结束==="
        
        start_idx = response.find(start_marker)
        if start_idx != -1:
            end_idx = response.find(end_marker)
            if end_idx != -1:
                result_section = response[start_idx + len(start_marker):end_idx].strip()
                print(f"✅ 第{iteration}轮：找到完整的优化结果标记")
            else:
                result_section = response[start_idx + len(start_marker):].strip()
                print(f"⚠️ 第{iteration}轮：只找到开始标记，从此处开始解析")
        else:
            print(f"ℹ️ 第{iteration}轮：未找到优化结果标记，直接解析整个响应")
            result_section = response
        
        # 提取优化后的章节内容
        content_start = "===章节内容开始==="
        content_end = "===章节内容结束==="
        
        content_start_idx = result_section.find(content_start)
        if content_start_idx != -1:
            content_end_idx = result_section.find(content_end)
            if content_end_idx != -1:
                optimized_content = result_section[content_start_idx + len(content_start):content_end_idx].strip()
                print(f"✅ 第{iteration}轮：成功提取优化后内容")
            else:
                print(f"⚠️ 第{iteration}轮：缺少内容结束标记")
                optimized_content = current_content.get("content", "")
        else:
            print(f"⚠️ 第{iteration}轮：缺少内容开始标记")
            optimized_content = current_content.get("content", "")
        
        # 提取是否继续迭代的决策
        should_continue = _extract_iteration_decision(result_section)
        
        # 提取差异度分析
        difference_analysis = _extract_difference_analysis(result_section)
        
        # 提取质量评估
        quality_evaluation = _extract_quality_evaluation_writing(result_section)
        
        # 🆕 如果提供了重新编号的材料字典，进行引用映射
        citation_mapping = {}
        if renumbered_materials and optimized_content:
            from utils import extract_citation_mapping
            citation_mapping = extract_citation_mapping(optimized_content, renumbered_materials)
        
        # 构建结果
        result = {
            "content": {
                "content": optimized_content,
                "status": "success",
                "materials_used": current_content.get("materials_used", 0) + 100,  # 增加100个新材料
                "iterations_completed": iteration,
                "quality_scores": quality_evaluation.get("scores", current_content.get("quality_scores", {})),
                "citation_mapping": citation_mapping  # 🆕 添加引用映射信息
            },
            "should_continue": should_continue,
            "difference_analysis": difference_analysis,
            "quality_evaluation": quality_evaluation
        }
        
        return result
        
    except Exception as e:
        print(f"❌ 第{iteration}轮优化结果解析失败: {str(e)}")
        return None


def _extract_scientific_enrichment_decision(response: str, iteration: int) -> Dict:
    """提取科学决策依据"""
    decision_info = {}
    try:
        # 提取是否继续迭代的决策
        decision_match = re.search(r'【是否继续迭代】\s*\n?(.+?)(?=\n【|\n\n|$)', response, re.DOTALL)
        should_continue = False
        if decision_match:
            decision_text = decision_match.group(1).strip()
            should_continue = "是" in decision_text
            decision_info["decision_text"] = decision_text
            pass
        
        # 提取科学决策依据
        basis_match = re.search(r'【科学决策依据】\s*(.*?)(?=\n===|$)', response, re.DOTALL)
        if basis_match:
            basis_content = basis_match.group(1).strip()
            print(f"✓ 找到科学决策依据部分，长度: {len(basis_content)}字符")
            
            # 提取量化指标
            metrics_section = re.search(r'决策量化指标:\s*(.*?)(?=决策逻辑:|$)', basis_content, re.DOTALL)
            if metrics_section:
                metrics_content = metrics_section.group(1).strip()
                print(f"✓ 找到量化指标部分: {metrics_content[:100]}...")
                
                # 提取具体指标 - 🔧 修复：支持小数评分
                improvement_score = re.search(r'综合改进评分:\s*(\d+\.?\d*)/10', metrics_content)
                if improvement_score:
                    decision_info["improvement_score"] = float(improvement_score.group(1))
                    print(f"✓ 成功提取综合改进评分: {improvement_score.group(1)}")
                else:
                    print(f"❌ 在量化指标中未找到综合改进评分模式")
                    print(f"   量化指标内容: {metrics_content}")
                
                guidance_ratio = re.search(r'新增指导价值占比:\s*(\d+\.?\d*)%', metrics_content)
                if guidance_ratio:
                    decision_info["guidance_ratio"] = float(guidance_ratio.group(1))
                    print(f"✓ 成功提取指导价值占比: {guidance_ratio.group(1)}%")
            else:
                print(f"❌ 未找到'决策量化指标:'部分")
                print(f"   决策依据内容: {basis_content[:200]}...")
            
            # 提取决策逻辑
            logic_match = re.search(r'决策逻辑:\s*(.*?)(?=主要改进点:|$)', basis_content, re.DOTALL)
            if logic_match:
                decision_info["logic"] = logic_match.group(1).strip()
        else:
            print(f"❌ 未找到【科学决策依据】标记")
            print(f"   响应长度: {len(response)}字符")
            print(f"   响应内容预览: {response[:300]}...")
        
        decision_info["should_continue"] = should_continue
        
    except Exception as e:
        print(f"⚠️ 第{iteration}轮：解析科学决策时出错: {e}")
        # 回退到简单的决策提取
        if "【是否继续迭代】" in response:
            decision_info["should_continue"] = "是" in response[response.find("【是否继续迭代】"):response.find("【是否继续迭代】")+20]
    
    return decision_info



def _format_materials_for_enrichment(materials: List) -> str:
    """格式化材料用于丰富分析"""
    if not materials:
        return "无新材料"
    
    # 🆕 使用max(100, len(materials))来决定返回的材料数量，尽可能多返回参考资料
    material_count = max(100, len(materials))
    actual_materials = materials[:material_count]
    
    formatted = f"共{len(actual_materials)}条新材料:\n"
    formatted += "⚠️ 注意：材料编号仅用于展示，在分析和优化内容时请不要引用具体的材料编号！\n\n"
    
    # 🆕 与初始大纲生成保持一致的格式化方式
    for i, material in enumerate(actual_materials, 1):
        content = material.get('content', '')
        relevance_score = material.get('relevance_score', 0)
        
        # 🆕 与之前保持一致：长度超过1000则只取前1000，不超过则直接使用
        if len(content) > 2000:
            formatted += f"材料{i} (相关度: {relevance_score:.2f}): {content[:2000]}...\n\n"
        else:
            formatted += f"材料{i} (相关度: {relevance_score:.2f}): {content}\n\n"
    
    return formatted

def _extract_content_quality_evaluation(response: str, iteration: int) -> Dict:
    """提取多维度内容质量评估信息"""
    evaluation = {}
    try:
        # 查找质量评估部分
        eval_match = re.search(r'【多维度内容质量评估】\s*(.*?)(?=\n【|$)', response, re.DOTALL)
        if eval_match:
            eval_content = eval_match.group(1).strip()
            
            # 提取各维度评分
            dimensions = ["写作指导完整性", "关键词检索精准性", "学术深度适宜性", "结构逻辑合理性", "实用指导价值"]
            for dim in dimensions:
                score_match = re.search(rf'{dim}:\s*(\d+)/10', eval_content)
                if score_match:
                    evaluation[dim] = int(score_match.group(1))
            
            # 提取综合评分
            overall_match = re.search(r'综合质量评分:\s*(\d+\.?\d*)/10', eval_content)
            if overall_match:
                evaluation["overall_score"] = float(overall_match.group(1))
        
    except Exception as e:
        print(f"⚠️ 第{iteration}轮：解析内容质量评估时出错: {e}")
    
    return evaluation

def _extract_material_value_analysis(response: str, iteration: int) -> Dict:
    """提取新材料价值分析"""
    analysis = {}
    try:
        # 查找新材料价值分析部分
        analysis_match = re.search(r'【新材料价值分析】\s*(.*?)(?=\n【|$)', response, re.DOTALL)
        if analysis_match:
            analysis_content = analysis_match.group(1).strip()
            
            # 提取各个分析维度
            fields = ["技术贡献识别", "应用场景扩展", "研究热点发现", "评估标准更新"]
            for field in fields:
                field_match = re.search(rf'{field}:\s*(.*?)(?=\n[^\s]|\n{field}|\n技术|\n应用|\n研究|\n评估|$)', analysis_content, re.DOTALL)
                if field_match:
                    analysis[field] = field_match.group(1).strip()
        
    except Exception as e:
        print(f"⚠️ 第{iteration}轮：解析新材料价值分析时出错: {e}")
    
    return analysis

def _extract_keyword_optimization_analysis( response: str, iteration: int) -> Dict:
    """提取关键词优化潜力评估"""
    optimization = {}
    try:
        # 查找关键词优化潜力评估部分
        opt_match = re.search(r'【关键词优化潜力评估】\s*(.*?)(?=\n【|$)', response, re.DOTALL)
        if opt_match:
            opt_content = opt_match.group(1).strip()
            
            # 提取各个优化维度
            fields = ["检索效果分析", "术语更新需求", "覆盖度优化", "差异化程度"]
            for field in fields:
                field_match = re.search(rf'{field}:\s*(.*?)(?=\n[^\s]|\n{field}|\n检索|\n术语|\n覆盖|\n差异|$)', opt_content, re.DOTALL)
                if field_match:
                    optimization[field] = field_match.group(1).strip()
        
    except Exception as e:
        print(f"⚠️ 第{iteration}轮：解析关键词优化分析时出错: {e}")
    
    return optimization

def _extract_improvement_opportunities(response: str, iteration: int) -> Dict:
    """提取内容缺陷与改进机会识别"""
    opportunities = {}
    try:
        # 查找改进机会识别部分
        opp_match = re.search(r'【内容缺陷与改进机会识别】\s*(.*?)(?=\n【|$)', response, re.DOTALL)
        if opp_match:
            opp_content = opp_match.group(1).strip()
            
            # 提取各个改进维度
            fields = ["写作指导缺陷", "内容规划遗漏", "关键词策略不足", "学术价值提升"]
            for field in fields:
                field_match = re.search(rf'{field}:\s*(.*?)(?=\n[^\s]|\n{field}|\n写作|\n内容|\n关键|\n学术|$)', opp_content, re.DOTALL)
                if field_match:
                    opportunities[field] = field_match.group(1).strip()
        
    except Exception as e:
        print(f"⚠️ 第{iteration}轮：解析改进机会时出错: {e}")
    
    return opportunities

def _extract_enrichment_improvement_assessment(response: str, iteration: int) -> Dict:
    """提取改进效果量化评估"""
    assessment = {}

    return assessment

def _count_actual_citations(topic: str) -> int:
    """
    从引用JSON文件中统计实际使用的引用数量
    
    Args:
        topic: 主题名称，用于定位引用文件
        
    Returns:
        实际引用的总数量
    """
    import json
    import os
    from datetime import datetime
    
    try:
        # 🆕 查找已存在的引用文件
        citations_dir = "./chapter_citations"
        safe_topic = "".join(c for c in topic if c.isalnum() or c in [' ', '_', '-']).rstrip()
        safe_topic = safe_topic.replace(' ', '_') if safe_topic else "综述"
        
        # 🆕 智能查找引用文件（优先使用正式文件，避免临时文件）
        import glob
        
        # 先查找正式的引用文件
        formal_pattern = f"{safe_topic}_citations_[0-9]*_[0-9]*.json"
        formal_path = os.path.join(citations_dir, formal_pattern)
        formal_files = glob.glob(formal_path)
        
        # 查找所有引用文件
        all_pattern = f"{safe_topic}_citations*.json"
        all_path = os.path.join(citations_dir, all_pattern)
        all_files = glob.glob(all_path)
        
        if formal_files:
            # 优先使用正式文件
            filepath = max(formal_files, key=os.path.getmtime)
        elif all_files:
            # 如果没有正式文件，使用其他文件
            filepath = max(all_files, key=os.path.getmtime)
        else:
            return 0
        
        # 读取引用数据并统计
        with open(filepath, 'r', encoding='utf-8') as f:
            citation_data = json.load(f)
        
        total_citations = 0
        sections = citation_data.get("sections", {})
        
        for section_key, section_data in sections.items():
            detailed_citations = section_data.get("detailed_citations", {})
            total_citations += len(detailed_citations)
        
        return total_citations
        
    except Exception as e:
        print(f"⚠️ 统计引用数量时出错: {e}")
        return 0

def _debug_response_structure(response: str, iteration: int):
    """调试响应结构，输出关键标记的位置信息"""
    # markers = [
    #     "===优化结果开始===",
    #     "===优化结果结束===", 
    #     "===大纲开始===",
    #     "===大纲结束===",
    #     "【多维度质量评估】",
    #     "【新材料内容分析】",
    #     "【改进效果量化评估】",
    #     "【是否继续迭代】",
    #     "【科学决策依据】"
    # ]
    
    # 简化调试输出，只在需要时启用
    # print(f"🔍 第{iteration}轮：响应结构调试")
    # print(f"   响应总长度: {len(response)} 字符")
    
    # for marker in markers:
    #     pos = response.find(marker)
    #     if pos != -1:
    #         print(f"   ✅ '{marker}' 位置: {pos}")
    #     else:
    #         print(f"   ❌ '{marker}' 未找到")
    return {}


def _extract_scientific_decision(response: str, iteration: int) -> Dict:
    """提取科学决策依据"""
    decision_info = {}
    try:
        # 提取是否继续迭代的决策
        decision_match = re.search(r'【是否继续迭代】\s*\n?(.+?)(?=\n【|\n\n|$)', response, re.DOTALL)
        should_continue = False
        if decision_match:
            decision_text = decision_match.group(1).strip()
            should_continue = "是" in decision_text
            decision_info["decision_text"] = decision_text
        
        # 提取科学决策依据
        basis_match = re.search(r'【科学决策依据】\s*(.*?)(?=\n===|$)', response, re.DOTALL)
        if basis_match:
            basis_content = basis_match.group(1).strip()
            
            # 提取量化指标
            metrics_section = re.search(r'决策量化指标:\s*(.*?)(?=决策逻辑:|$)', basis_content, re.DOTALL)
            if metrics_section:
                metrics_content = metrics_section.group(1).strip()
                
                # 提取具体指标 - 🔧 修复：支持小数评分
                improvement_score = re.search(r'综合改进评分:\s*(\d+\.?\d*)/10', metrics_content)
                if improvement_score:
                    decision_info["improvement_score"] = float(improvement_score.group(1))
                
                content_ratio = re.search(r'新增重要内容占比:\s*(\d+\.?\d*)%', metrics_content)
                if content_ratio:
                    decision_info["content_ratio"] = float(content_ratio.group(1))
            
            # 提取决策逻辑
            logic_match = re.search(r'决策逻辑:\s*(.*?)(?=主要改进点:|$)', basis_content, re.DOTALL)
            if logic_match:
                decision_info["logic"] = logic_match.group(1).strip()
        
        decision_info["should_continue"] = should_continue
        
    except Exception as e:
        print(f"⚠️ 第{iteration}轮：解析科学决策时出错: {e}")
        # 回退到简单的决策提取
        if "【是否继续迭代】" in response:
            decision_info["should_continue"] = "是" in response[response.find("【是否继续迭代】"):response.find("【是否继续迭代】")+20]
    
    return decision_info
    

def _extract_material_analysis(response: str, iteration: int) -> Dict:
    """提取新材料内容分析"""
    analysis = {}
    try:
        # 查找新材料分析部分
        analysis_match = re.search(r'【新材料内容分析】\s*(.*?)(?=\n【|$)', response, re.DOTALL)
        if analysis_match:
            analysis_content = analysis_match.group(1).strip()
            
            # 提取各个分析维度
            fields = ["核心概念提取", "研究热点识别", "方法技术归类", "应用场景扩展"]
            for field in fields:
                field_match = re.search(rf'{field}:\s*(.*?)(?=\n[^\s]|\n{field}|\n核心|\n研究|\n方法|\n应用|$)', analysis_content, re.DOTALL)
                if field_match:
                    analysis[field] = field_match.group(1).strip()
        
    except Exception as e:
        print(f"⚠️ 第{iteration}轮：解析材料分析时出错: {e}")
    
    return analysis

def _extract_improvement_assessment(response: str, iteration: int) -> Dict:
    """提取改进效果量化评估"""
    assessment = {}
    try:
        # 查找改进效果评估部分
        improve_match = re.search(r'【改进效果量化评估】\s*(.*?)(?=\n【|$)', response, re.DOTALL)
        if improve_match:
            improve_content = improve_match.group(1).strip()
            
            # 提取新增内容占比
            content_match = re.search(r'新增重要内容占比:\s*(\d+\.?\d*)%', improve_content)
            if content_match:
                assessment["new_content_ratio"] = float(content_match.group(1))
            
            # 提取综合改进评分 - 🔧 修复：支持小数评分
            overall_match = re.search(r'综合改进评分:\s*(\d+\.?\d*)/10', improve_content)
            if overall_match:
                assessment["overall_improvement"] = float(overall_match.group(1))
                
            # 提取其他改进描述
            fields = ["结构优化程度", "专业性提升度", "完整性改善率"]
            for field in fields:
                field_match = re.search(rf'{field}:\s*(.*?)(?=\n[^\s]|$)', improve_content, re.DOTALL)
                if field_match:
                    assessment[field] = field_match.group(1).strip()
        
    except Exception as e:
        print(f"⚠️ 第{iteration}轮：解析改进评估时出错: {e}")
    
    return assessment
    
def _extract_quality_evaluation(response: str, iteration: int) -> Dict:
    """提取多维度质量评估信息"""
    evaluation = {}
    try:
        # 查找质量评估部分
        eval_match = re.search(r'【多维度质量评估】\s*(.*?)(?=\n【|$)', response, re.DOTALL)
        if eval_match:
            eval_content = eval_match.group(1).strip()
            
            # 提取各维度评分
            dimensions = ["学术完整性", "结构逻辑性", "术语专业性", "内容平衡性", "国际规范性"]
            for dim in dimensions:
                score_match = re.search(rf'{dim}:\s*(\d+)/10', eval_content)
                if score_match:
                    evaluation[dim] = int(score_match.group(1))
            
            # 提取综合评分
            overall_match = re.search(r'综合质量评分:\s*(\d+\.?\d*)/10', eval_content)
            if overall_match:
                evaluation["overall_score"] = float(overall_match.group(1))
        
    except Exception as e:
        print(f"⚠️ 第{iteration}轮：解析质量评估时出错: {e}")
    
    return evaluation
    
def _format_materials_for_refinement(materials: List) -> str:
    """格式化材料用于优化分析"""
    if not materials:
        return "无新材料"
    
    # 🆕 使用max(100, len(materials))来决定返回的材料数量，尽可能多返回参考资料
    material_count = max(100, len(materials))
    actual_materials = materials[:material_count]
    
    formatted = f"共{len(actual_materials)}条新材料:\n\n"
    
    # 🆕 与初始大纲生成保持一致的格式化方式
    for i, material in enumerate(actual_materials, 1):
        content = material.get('content', '')
        relevance_score = material.get('relevance_score', 0)
        
        # 🆕 与之前保持一致：长度超过1000则只取前1000，不超过则直接使用
        if len(content) > 2000:
            formatted += f"材料{i} (相关度: {relevance_score:.2f}): {content[:2000]}...\n\n"
        else:
            formatted += f"材料{i} (相关度: {relevance_score:.2f}): {content}\n\n"
    
    return formatted

def _parse_refinement_response( response: str, topic: str, iteration: int) -> Dict:
    """解析科学优化响应，提取评估信息、大纲和决策"""
    try:
        from utils import _debug_response_structure
        # 🆕 输出调试信息
        _debug_response_structure(response, iteration)
        
        # 🆕 更鲁棒的解析逻辑：首先尝试完整的标记匹配
        result_section = response
        
        # 查找优化结果部分（如果存在）
        start_marker = "===优化结果开始==="
        end_marker = "===优化结果结束==="
        
        start_idx = response.find(start_marker)
        if start_idx != -1:
            end_idx = response.find(end_marker)
            if end_idx != -1:
                result_section = response[start_idx + len(start_marker):end_idx].strip()
                print(f"✅ 第{iteration}轮：找到完整的优化结果标记")
            else:
                # 如果只有开始标记，从开始标记后面开始解析
                result_section = response[start_idx + len(start_marker):].strip()
                print(f"⚠️ 第{iteration}轮：只找到开始标记，从此处开始解析")
        else:
            print(f"ℹ️ 第{iteration}轮：未找到优化结果标记，直接解析整个响应")
        
        from utils import _extract_quality_evaluation, _extract_material_analysis, _extract_improvement_assessment

        # 🆕 解析多维度质量评估
        evaluation_info = _extract_quality_evaluation(result_section, iteration)
        
        # 🆕 解析新材料分析
        material_analysis = _extract_material_analysis(result_section, iteration)
        
        # 🆕 解析改进效果评估
        improvement_assessment = _extract_improvement_assessment(result_section, iteration)
        
        # 🆕 提取优化后的大纲 - 放宽开头标记限制
        # 支持多种开头标记：【优化后大纲】、===大纲开始===、【综述概述】、===优化结果开始===
        start_markers = ["【优化后大纲】", "===大纲开始===", "【综述概述】", "===优化结果开始==="]
        outline_start = -1
        found_marker = None
        
        for marker in start_markers:
            pos = result_section.find(marker)
            if pos != -1:
                outline_start = pos
                found_marker = marker
                break
        
        if outline_start == -1:
            print(f"❌ 第{iteration}轮：未找到任何有效的开头标记，响应格式可能不正确")
            print(f"调试信息：响应开头300字符: {result_section[:300]}")
            return None
        
        # 查找结束标记（可选）
        outline_end = result_section.find("===大纲结束===")
        
        if outline_end != -1:
            # 找到了结束标记
            outline_content = result_section[outline_start:outline_end + len("===大纲结束===")]
        else:
            # 没有找到结束标记，使用从开头标记到响应结尾的内容
            outline_content = result_section[outline_start:]
            print(f"⚠️ 第{iteration}轮：只找到开头标记 ({found_marker})，使用到结尾的内容")
        
        # 解析大纲 - 传递找到的开头标记信息
        from utils import parse_outline_response
        outline = parse_outline_response(outline_content, topic, [], found_start_marker=found_marker)
        
        from utils import _extract_scientific_decision

        # 🆕 更科学的决策提取：解析科学决策依据
        decision_info = _extract_scientific_decision(response, iteration)
        should_continue = decision_info.get("should_continue", False)
        
        # 🆕 添加解析成功的详细日志
        print(f"✅ 第{iteration}轮：Planner优化解析成功,质量评估: 综合评分 {evaluation_info.get('overall_score', 'N/A')}/10")
        # 🔧 修复：从决策信息中获取综合改进评分，优先使用decision_info中的数据
        improvement_score = decision_info.get('improvement_score', improvement_assessment.get('overall_improvement', 'N/A'))
        print(f"   改进评估: 综合改进评分 {improvement_score}/10, 决策结果: {'继续' if should_continue else '停止'}")
        
        return {
            "outline": outline,
            "should_continue": should_continue,
            "evaluation_info": evaluation_info,
            "material_analysis": material_analysis,
            "improvement_assessment": improvement_assessment,
            "decision_info": decision_info
        }
        
    except Exception as e:
        print(f"❌ 第{iteration}轮：解析优化响应时出错: {e}")
        print(f"调试信息：响应长度: {len(response)}, 开头200字符: {response[:200]}")
        return None


def _format_outline_for_refinement(outline: Dict) -> str:
    """格式化大纲用于优化分析"""
    if not outline:
        return "当前大纲为空"
    
    formatted = f"主题: {outline.get('topic', '未知')}\n"
    formatted += f"概述: {outline.get('overview', '无概述')}\n\n"
    
    chapters = outline.get("chapters", [])
    
    # 🔧 修改：同时支持列表和字典格式的章节数据
    if isinstance(chapters, list):
        # 处理列表格式的章节 (parse_outline_response 返回的格式)
        for chapter in chapters:
            chapter_id = chapter.get('id', '?')
            chapter_title = chapter.get('title', '未命名章节')
            chapter_desc = chapter.get('description', '无描述')
            
            formatted += f"{chapter_id}. {chapter_title}\n"
            formatted += f"描述: {chapter_desc}\n"
            
            # 处理子章节（也是列表格式）
            subsections = chapter.get("subsections", [])
            if isinstance(subsections, list):
                for subsection in subsections:
                    sub_id = subsection.get('id', '?')
                    sub_title = subsection.get('title', '未命名子章节')
                    sub_desc = subsection.get('description', '')
                    
                    formatted += f"  {sub_id} {sub_title}\n"
                    if sub_desc:
                        formatted += f"    描述: {sub_desc}\n"
            formatted += "\n"
            
    elif isinstance(chapters, dict):
        # 保持原有的字典格式处理逻辑（向后兼容）
        for chapter_id, chapter in chapters.items():
            formatted += f"{chapter_id}. {chapter.get('title', '未命名章节')}\n"
            formatted += f"描述: {chapter.get('description', '无描述')}\n"
            
            subsections = chapter.get("subsections", {})
            if isinstance(subsections, dict):
                for sub_id, subsection in subsections.items():
                    formatted += f"  {sub_id} {subsection.get('title', '未命名子章节')}\n"
                    if subsection.get('description'):
                        formatted += f"    描述: {subsection['description']}\n"
            elif isinstance(subsections, list):
                # 处理字典格式章节中的列表格式子章节
                for subsection in subsections:
                    sub_id = subsection.get('id', '?')
                    sub_title = subsection.get('title', '未命名子章节')
                    sub_desc = subsection.get('description', '')
                    
                    formatted += f"  {sub_id} {sub_title}\n"
                    if sub_desc:
                        formatted += f"    描述: {sub_desc}\n"
            formatted += "\n"
    else:
        formatted += "章节信息格式异常\n"
    
    return formatted


def _parse_enrichment_refinement_response( response: str, iteration: int, current_enriched: Dict = None) -> Dict:
    """解析科学优化响应，提取评估信息、丰富内容和决策"""
    try:
        # 🆕 更鲁棒的解析逻辑：首先尝试完整的标记匹配
        result_section = response
        
        # 查找优化结果部分（如果存在）
        start_marker = "===优化结果开始==="
        end_marker = "===优化结果结束==="
        
        start_idx = response.find(start_marker)
        if start_idx != -1:
            end_idx = response.find(end_marker)
            if end_idx != -1:
                result_section = response[start_idx + len(start_marker):end_idx].strip()
                print(f"✅ 第{iteration}轮：找到完整的优化结果标记")
            else:
                result_section = response[start_idx + len(start_marker):].strip()
                print(f"⚠️ 第{iteration}轮：只找到开始标记，从此处开始解析")
        else:
            print(f"ℹ️ 第{iteration}轮：未找到优化结果标记，直接解析整个响应")
        
        from utils import _extract_content_quality_evaluation, _extract_material_value_analysis, _extract_keyword_optimization_analysis, _extract_improvement_opportunities, _extract_enrichment_improvement_assessment
        # 🆕 解析多维度内容质量评估
        quality_evaluation = _extract_content_quality_evaluation(result_section, iteration)
        
        # 🆕 解析新材料价值分析
        material_value_analysis = _extract_material_value_analysis(result_section, iteration)
        
        # 🆕 解析关键词优化潜力评估
        keyword_optimization = _extract_keyword_optimization_analysis(result_section, iteration)
        
        # 🆕 解析内容缺陷与改进机会识别
        improvement_opportunities = _extract_improvement_opportunities(result_section, iteration)
        
        # 🆕 解析改进效果量化评估
        improvement_assessment = _extract_enrichment_improvement_assessment(result_section, iteration)
        
        # 🆕 提取优化后的丰富内容 - 保留完整标记供parse_full_enrichment使用
        enrichment_start = result_section.find("===内容规划开始===")
        enrichment_end = result_section.find("===内容规划结束===")
        
        if enrichment_start == -1:
            # 如果没有找到开始标记，使用整个result_section
            enrichment_content = result_section
            print(f"⚠️ 第{iteration}轮：未找到===内容规划开始===标记，使用全部内容解析")
        elif enrichment_end == -1:
            # 只有开始标记，没有结束标记，保留标记
            enrichment_content = result_section[enrichment_start:]
            print(f"⚠️ 第{iteration}轮：只找到开始标记，使用到结尾的内容")
        else:
            # 🔧 修复：找到了完整的标记对，保留完整的标记供parse_full_enrichment解析
            enrichment_content = result_section[enrichment_start:enrichment_end + len("===内容规划结束===")]
            print(f"✅ 第{iteration}轮：找到完整的内容规划标记")
        
        # 解析丰富内容（复用现有的解析函数）
        from utils import parse_full_enrichment
        # 🔧 修改：使用当前的丰富内容作为模板，保留topic和overview信息
        if current_enriched and isinstance(current_enriched, dict):
            template_outline = {
                "topic": current_enriched.get("topic", "未知"),
                "overview": current_enriched.get("overview", "无概述"),
                "chapters": current_enriched.get("chapters", {})
            }
            print(f"✓ 使用当前丰富内容作为模板: {current_enriched.get('topic', '未知')}")
        else:
            # 备用方案：创建临时大纲
            template_outline = {"topic": "temp", "chapters": {}}
            print(f"⚠️ current_enriched类型异常，使用临时模板。类型: {type(current_enriched)}, 值: {str(current_enriched)[:100] if current_enriched else 'None'}")
        
        enriched = parse_full_enrichment(enrichment_content, template_outline)
        
        # 🆕 更科学的决策提取：解析科学决策依据
        decision_info = _extract_scientific_enrichment_decision(response, iteration)
        should_continue = decision_info.get("should_continue", False)
        
        # 🆕 添加解析成功的详细日志
        print(f"✅ 第{iteration}轮：Enricher优化解析成功,质量评估: 综合评分 {quality_evaluation.get('overall_score', 'N/A')}/10")
        # 🔧 修复：从决策信息中获取综合改进评分，优先使用decision_info中的数据
        improvement_score = decision_info.get('improvement_score', improvement_assessment.get('overall_improvement', 'N/A'))
        print(f"   改进评估: 综合改进评分 {improvement_score}/10, 关键词优化: {len(keyword_optimization.get('术语更新需求', ''))} 个术语需要更新, 决策结果: {'继续' if should_continue else '停止'}")
        
        # 🔧 添加类型检查，确保返回的enriched是字典类型
        if not isinstance(enriched, dict):
            print(f"⚠️ 警告：解析结果不是字典类型，类型: {type(enriched)}, 值: {str(enriched)[:200] if enriched else 'None'}")
            return None
        
        return {
            "enrichment": enriched,
            "should_continue": should_continue,
            "quality_evaluation": quality_evaluation,
            "material_value_analysis": material_value_analysis,
            "keyword_optimization": keyword_optimization,
            "improvement_opportunities": improvement_opportunities,
            "improvement_assessment": improvement_assessment,
            "decision_info": decision_info
        }
        
    except Exception as e:
        print(f"❌ 第{iteration}轮：解析丰富优化响应时出错: {e}")
        import traceback
        print(f"完整错误堆栈：\n{traceback.format_exc()}")
        return None


def _select_materials_proportionally( categorized_materials: Dict, 
                                    target_texts: int = 75, target_equations: int = 8, 
                                    target_figures: int = 10, target_tables: int = 12,
                                    skip_count: int = 0) -> List:
    """按比例从各类型材料中选择材料"""

    # 🔍 详细检查输入的分类材料
    for material_type, materials in categorized_materials.items():
        if len(materials) > 0:
            # 检查第一个材料的结构
            first_material = materials[0]
            content_type_from_metadata = first_material.get("metadata", {}).get("content_type", "unknown")
            content_type_direct = first_material.get("content_type", "unknown")
    
    selected_materials = []
    
    # 按相关度排序各类型材料
    sorted_texts = sorted(categorized_materials.get("texts", []), 
                            key=lambda x: x.get("relevance_score", 0), reverse=True)
    sorted_equations = sorted(categorized_materials.get("equations", []), 
                                key=lambda x: x.get("relevance_score", 0), reverse=True)
    sorted_figures = sorted(categorized_materials.get("figures", []), 
                            key=lambda x: x.get("relevance_score", 0), reverse=True)
    sorted_tables = sorted(categorized_materials.get("tables", []), 
                            key=lambda x: x.get("relevance_score", 0), reverse=True)
    
    
    # 🆕 按目标数量选择材料，支持跳过前面已选的材料
    selected_texts = sorted_texts[skip_count:skip_count + target_texts]
    selected_equations = sorted_equations[skip_count//4:skip_count//4 + target_equations]  # 公式材料较少，按比例跳过
    selected_figures = sorted_figures[skip_count//3:skip_count//3 + target_figures]      # 图片材料中等，按比例跳过
    selected_tables = sorted_tables[skip_count//3:skip_count//3 + target_tables]        # 表格材料中等，按比例跳过
    
    
    # 合并所有选中的材料，并为每个材料标记正确的类型
    initial_count = len(selected_materials)
    
    # 添加文本材料并标记类型
    for material in selected_texts:
        material['content_type'] = 'text'  # 🔧 为材料设置正确的类型标识
    selected_materials.extend(selected_texts)
    
    # 添加公式材料并标记类型
    initial_count = len(selected_materials)
    for material in selected_equations:
        material['content_type'] = 'equation'  # 🔧 为材料设置正确的类型标识
    selected_materials.extend(selected_equations)
    
    # 添加图片材料并标记类型
    initial_count = len(selected_materials)
    for material in selected_figures:
        material['content_type'] = 'figure'  # 🔧 为材料设置正确的类型标识
    selected_materials.extend(selected_figures)
    
    # 添加表格材料并标记类型
    initial_count = len(selected_materials)
    for material in selected_tables:
        material['content_type'] = 'table'  # 🔧 为材料设置正确的类型标识
    selected_materials.extend(selected_tables)
    
    # 🔍 基于实际选择数量统计最终的材料类型分布
    final_type_counts = {
        "text": len(selected_texts),
        "equation": len(selected_equations), 
        "figure": len(selected_figures),
        "table": len(selected_tables)
    }
    
    # 记录实际选择的数量
    actual_counts = {
        "texts": len(selected_texts),
        "equations": len(selected_equations), 
        "figures": len(selected_figures),
        "tables": len(selected_tables)
    }
    
    
    return selected_materials


def _select_next_batch_materials(categorized_materials: Dict, used_materials_count: Dict, 
                                target_counts: Dict) -> List:
    """从剩余材料中选择下一批材料"""
    selected_materials = []
    
    for material_type, target_count in target_counts.items():
        materials = categorized_materials.get(material_type, [])
        used_count = used_materials_count.get(material_type, 0)
        
        # 按相关度排序材料
        sorted_materials = sorted(materials, key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        # 获取下一批材料
        start_idx = used_count
        end_idx = min(start_idx + target_count, len(sorted_materials))
        
        if start_idx < len(sorted_materials):
            next_batch = sorted_materials[start_idx:end_idx]
            
            # 🔧 为选中的材料设置正确的类型标识
            content_type_mapping = {
                "texts": "text",
                "equations": "equation", 
                "figures": "figure",
                "tables": "table"
            }
            content_type = content_type_mapping.get(material_type, "text")
            for material in next_batch:
                material['content_type'] = content_type
                
            selected_materials.extend(next_batch)
            
    # 记录实际选择的数量统计
    actual_counts = {}
    for material_type in target_counts.keys():
        materials = categorized_materials.get(material_type, [])
        used_count = used_materials_count.get(material_type, 0)
        available = len(materials) - used_count
        target = target_counts[material_type]
        actual = min(available, target)
        actual_counts[material_type] = actual
        
    print(f"📊 选择下一批材料: 文本{actual_counts.get('texts', 0)}, 公式{actual_counts.get('equations', 0)}, 图片{actual_counts.get('figures', 0)}, 表格{actual_counts.get('tables', 0)}, 总计{len(selected_materials)}条")
    
    return selected_materials
