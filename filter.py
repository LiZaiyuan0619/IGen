# -*- coding: utf-8 -*-

# # 任务目标
# 本脚本旨在处理由 MinerU API 解析生成的 JSON 文件。
# 主要功能包括：
# 1. 移除 JSON 文件中 "REFERENCES" 部分及其之后的所有内容
# 2. 清理所有文本内容中的学术引用信息（如引用标记、作者年份等）
# 以生成一个不包含参考文献、附录和引用标记的干净版 JSON 文件。

# # 上下文
# 前置脚本 `api.py` 会从 MinerU API 下载并解压每个PDF的解析结果。
# 每个PDF的结果都存储在一个名为 `[论文名].pdf_result` 的目录中。
# 此脚本将遍历这些结果目录，处理其中的 `*_content_list.json` 文件。

# # 输入
# - `results_base_dir`: 存储所有 `..._result` 文件夹的根目录。
# - 每个 `..._result` 文件夹中都包含一个 `*_content_list.json` 文件。

# # 执行步骤
# 1. 设置 `results_base_dir` 变量，指向包含所有结果文件夹的目录。
# 2. 遍历 `results_base_dir` 下的所有 `..._result` 目录。
# 3. 在每个结果目录中，查找 `*_content_list.json` 文件。
# 4. 读取并解析该 JSON 文件，它是一个包含多个字典的列表。
# 5. 精确删除参考文献内容：
#    a) 查找 "REFERENCES" 部分的起始位置（具有 text_level=1 的文本项）
#    b) 查找 REFERENCES 之后第一个具有 text_level 的文本项（通常是 appendix）
#    c) 删除这两个位置之间的所有内容，保留 appendix 等有用部分
# 6. 对剩余内容中所有 "type"=="text" 的项进行引用清理，去除学术引用标记。
# 7. 从结果目录的名称中提取论文名。
# 8. 将处理后的新列表保存为一个新的 JSON 文件，命名为 `[论文名]_filter.json`，并存储在原结果目录中。
# 9. 如果在文件中未找到参考文献部分，则打印一条警告信息。

# # 输出
# - 在每个 `..._result` 文件夹内生成一个新的 `[论文名]_filter.json` 文件，其中：
#   1. 精确删除了参考文献列表部分，但保留了 appendix 等有用内容
#   2. 所有文本内容已清理学术引用标记

import os
import json
import glob
import re
import time  # 添加时间模块用于计时

# --- 配置区 ---
# 请将此路径设置为包含所有 `..._result` 文件夹的根目录
# 这个路径应该和您 api.py 脚本中的 output_dir 一致
results_base_dir = "D:/Desktop/ZJU/final_test/pdf/result/set1/"
# --- 配置区结束 ---


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


def clean_text_content(content_list):
    """
    清理内容列表中所有text类型项的引用信息
    
    Args:
        content_list: JSON内容列表
        
    Returns:
        tuple: (处理后的内容列表, 处理统计信息)
    """
    if not content_list:
        return content_list, {"total_items": 0, "text_items": 0, "cleaned_items": 0}
    
    total_items = len(content_list)
    text_items = 0
    cleaned_items = 0
    
    print(f"  - 开始清理文本内容中的引用信息...")
    
    for item in content_list:
        if isinstance(item, dict) and item.get("type") == "text":
            text_items += 1
            original_text = item.get("text", "")
            
            if original_text and isinstance(original_text, str):
                cleaned_text = clean_academic_citations(original_text)
                
                # 检查是否有实际的清理发生
                if len(cleaned_text) < len(original_text):
                    cleaned_items += 1
                    item["text"] = cleaned_text
    
    stats = {
        "total_items": total_items,
        "text_items": text_items,
        "cleaned_items": cleaned_items
    }
    
    print(f"  - 引用清理完成: 共处理 {text_items} 个文本项，其中 {cleaned_items} 个包含引用并已清理")
    
    return content_list, stats

def find_references_start(content_list):
    """
    在内容列表中查找 "REFERENCES" 部分的起始索引。
    """
    for i, item in enumerate(content_list):
        # 检查是否为符合条件的文本块
        if (item.get("type") == "text" and
            item.get("text", "").strip().upper() == "REFERENCES" and
            item.get("text_level") == 1):
            return i
    return -1  # 未找到


def find_next_text_level_item(content_list, start_index):
    """
    从指定索引开始，查找下一个包含text_level的text类型项
    
    Args:
        content_list: JSON内容列表
        start_index: 开始搜索的索引（通常是REFERENCES的索引）
        
    Returns:
        int: 下一个有text_level的text项的索引，如果未找到返回-1
    """
    # 从start_index的下一项开始搜索
    for i in range(start_index + 1, len(content_list)):
        item = content_list[i]
        # 查找type==text且存在text_level的项
        if (item.get("type") == "text" and 
            "text_level" in item and 
            item.get("text_level") is not None):
            return i
    return -1  # 未找到


def remove_references_content(content_list):
    """
    精确删除参考文献内容，保留appendix等有用部分
    
    执行步骤:
    1. 查找REFERENCES部分的起始位置
    2. 查找REFERENCES之后第一个有text_level的文本项
    3. 删除这两个位置之间的所有内容（包括REFERENCES本身）
    4. 保留后续的appendix等内容
    
    Args:
        content_list: JSON内容列表
        
    Returns:
        tuple: (处理后的内容列表, 删除统计信息)
    """
    if not content_list:
        return content_list, {"found_references": False, "removed_count": 0}
    
    # 1. 查找REFERENCES起始位置
    ref_start_index = find_references_start(content_list)
    
    if ref_start_index == -1:
        print("  - 警告: 未在本文件中找到 'REFERENCES' 部分，将不进行参考文献删除。")
        return content_list, {"found_references": False, "removed_count": 0}
    
    print(f"  - 在索引 {ref_start_index} 处找到 'REFERENCES' 部分")
    
    # 2. 查找REFERENCES之后下一个有text_level的项
    next_section_index = find_next_text_level_item(content_list, ref_start_index)
    
    if next_section_index == -1:
        # 如果没找到下一个section，说明references后面就没有其他章节了，删除到最后
        print(f"  - 未找到REFERENCES后的下一个章节，删除从索引 {ref_start_index} 到文件末尾的所有内容")
        filtered_list = content_list[:ref_start_index]
        removed_count = len(content_list) - ref_start_index
    else:
        # 找到了下一个section，只删除references和下一个section之间的内容
        next_section_text = content_list[next_section_index].get("text", "").strip()
        print(f"  - 在索引 {next_section_index} 处找到下一个章节: '{next_section_text}'")
        print(f"  - 删除索引 {ref_start_index} 到 {next_section_index-1} 之间的参考文献内容")
        
        # 构建新的内容列表：保留references前的内容 + 保留下一个section及其后的内容
        filtered_list = content_list[:ref_start_index] + content_list[next_section_index:]
        removed_count = next_section_index - ref_start_index
    
    print(f"  - 参考文献删除完成: 删除了 {removed_count} 个条目，保留 {len(filtered_list)} 个条目")
    
    return filtered_list, {"found_references": True, "removed_count": removed_count}

def process_result_directory(dir_path):
    """
    处理单个结果目录：查找、读取、过滤并保存新的JSON文件。
    """
    print(f"--- 正在处理目录: {os.path.basename(dir_path)} ---")

    # 1. 查找需要处理的 JSON 文件
    # 新逻辑：查找以 '_content_list.json' 结尾的JSON文件
    all_json_paths = glob.glob(os.path.join(dir_path, '*.json'))
    
    content_list_files = [
        path for path in all_json_paths 
        if os.path.basename(path).endswith('_content_list.json')
    ]

    if len(content_list_files) == 0:
        print("  ✗ 错误: 在此目录中未找到以 '_content_list.json' 结尾的文件。")
        return
    elif len(content_list_files) > 1:
        print(f"  ✗ 错误: 找到多个以 '_content_list.json' 结尾的文件: {[os.path.basename(p) for p in content_list_files]}。")
        print("  每个目录应该只有一个 _content_list.json 文件。")
        return
    
    # 找到唯一的 _content_list.json 文件
    input_json_path = content_list_files[0]
    print(f"  ✓ 找到_content_list.json文件: {os.path.basename(input_json_path)}")

    # 2. 读取和解析JSON
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            content_data = json.load(f)
    except Exception as e:
        print(f"  ✗ 读取或解析JSON文件失败: {e}")
        return

    # 3. 精确删除参考文献内容，保留appendix等有用部分
    filtered_data, ref_removal_stats = remove_references_content(content_data)

    # 4. 清理文本内容中的引用信息
    filtered_data, citation_stats = clean_text_content(filtered_data)

    # 5. 准备输出
    # 从目录名 '.../paper.pdf_result' 中提取 'paper.pdf'
    base_name_with_ext = os.path.basename(dir_path).replace('.pdf_result', '.pdf')
    # 移除 '.pdf' 后缀得到 'paper'
    paper_name = os.path.splitext(base_name_with_ext)[0]
    output_filename = f"{paper_name}_filter.json"
    output_json_path = os.path.join(dir_path, output_filename)

    # 6. 保存新的JSON文件
    try:
        # 为支持 Windows 长路径，对路径进行处理
        final_output_path = output_json_path
        if os.name == 'nt':
            abs_path = os.path.abspath(output_json_path)
            # 添加长路径前缀 `\\?\`
            if not abs_path.startswith('\\\\?\\'):
                final_output_path = '\\\\?\\' + abs_path
                
        with open(final_output_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, indent=4, ensure_ascii=False)
        print(f"  ✓ 成功保存过滤后的文件到: {os.path.basename(output_json_path)}")
    except Exception as e:
        print(f"  ✗ 保存新的JSON文件失败: {e}")


def main():
    """
    主函数，执行整个处理流程。
    """
    # 记录脚本开始时间
    script_start_time = time.time()
    
    # 检查根目录是否存在
    if not os.path.isdir(results_base_dir):
        print(f"错误: 根目录不存在 -> {results_base_dir}")
        
        # 即使出错也显示运行时间
        end_time = time.time()
        total_time = end_time - script_start_time
        print(f"\n脚本运行耗时{total_time:.2f}秒")
        return

    # 获取results_base_dir下面所有子目录
    result_dirs = [d for d in os.listdir(results_base_dir) if os.path.isdir(os.path.join(results_base_dir, d))]

    if not result_dirs:
        print(f"在 '{results_base_dir}' 中未找到任何子目录。")
        
        # 即使没有找到目录也显示运行时间
        end_time = time.time()
        total_time = end_time - script_start_time
        print(f"\n脚本运行耗时{total_time:.2f}秒")
        return

    print(f"在根目录中找到 {len(result_dirs)} 个结果目录。开始处理...\n")
    
    # 记录开始时间
    start_time = time.time()
    
    try:
        for dir_name in result_dirs:
            dir_path = os.path.join(results_base_dir, dir_name)
            process_result_directory(dir_path)
            print("-" * 50)

        # 记录结束时间并计算总耗时
        end_time = time.time()
        total_time = end_time - start_time
        
        # 格式化时间显示（时分秒）
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = total_time % 60
        
        print(f"\n=== 数据过滤完成 ===")
        print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
        print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
        
        if hours > 0:
            print(f"数据过滤耗时{hours}小时{minutes}分钟{seconds:.2f}秒")
        elif minutes > 0:
            print(f"数据过滤耗时{minutes}分钟{seconds:.2f}秒")
        else:
            print(f"数据过滤耗时{seconds:.2f}秒")

        print("\n所有目录处理完毕。")
        
    except KeyboardInterrupt:
        # 用户中断处理
        end_time = time.time()
        total_time = end_time - start_time
        
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = total_time % 60
        
        print(f"\n=== 用户中断处理 ===")
        print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
        print(f"中断时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
        
        if hours > 0:
            print(f"已运行{hours}小时{minutes}分钟{seconds:.2f}秒")
        elif minutes > 0:
            print(f"已运行{minutes}分钟{seconds:.2f}秒")
        else:
            print(f"已运行{seconds:.2f}秒")
            
    except Exception as e:
        # 其他异常处理
        end_time = time.time()
        total_time = end_time - start_time
        
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = total_time % 60
        
        print(f"\n=== 处理异常结束 ===")
        print(f"错误信息: {e}")
        print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
        print(f"异常时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
        
        if hours > 0:
            print(f"运行耗时{hours}小时{minutes}分钟{seconds:.2f}秒")
        elif minutes > 0:
            print(f"运行耗时{minutes}分钟{seconds:.2f}秒")
        else:
            print(f"运行耗时{seconds:.2f}秒")
            
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()