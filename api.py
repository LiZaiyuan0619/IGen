import requests
import time
import os
import glob
from pathlib import Path
import zipfile
import re

"""
脚本目标任务：批量上传PDF文件并通过MinerU API进行文档解析和处理

上下文：
- 使用MinerU在线API服务解析PDF文档
- 支持OCR、公式识别、表格识别等功能
- 自动处理文件命名、上传、解析、下载和解压

输入：
- PDF文件目录路径（todo目录）
- 每个PDF文件会被自动预处理（编号、长度检查）

执行步骤：
1. 扫描和预处理PDF文件（编号分配、文件名长度处理）
2. 准备批量上传请求数据
3. 申请上传URL并上传所有PDF文件
4. 监控处理状态直到所有文件完成
5. 下载解析结果ZIP文件
6. 自动解压所有ZIP文件并清理

输出：
- 解析完成的文档文件（包含文本、图片、表格等）
- 处理统计信息和总耗时
"""

# 设置API参数
url = 'https://mineru.net/api/v4/file-urls/batch'
header = {
    'Content-Type': 'application/json',
    'Authorization': ''
}

# 自动扫描目录下的所有PDF文件
def get_pdf_files(directory_path):
    """扫描指定目录下的所有PDF文件并返回完整路径列表"""
    pdf_list = []
    for file_path in glob.glob(f"{directory_path}/**/*.pdf", recursive=True):
        pdf_list.append(os.path.abspath(file_path))
    return pdf_list

def check_has_number_prefix(filename):
    """
    检查文件名是否以三位数字+下划线开头（XXX_格式）
    
    Args:
        filename: 文件名（包含扩展名）
    
    Returns:
        bool: 如果有三位数字编号前缀返回True，否则返回False
    """
    # 使用正则表达式检查是否以三位数字+下划线开头
    pattern = r'^\d{3}_'
    return bool(re.match(pattern, filename))


def extract_existing_numbers(pdf_files):
    """
    提取已存在的编号，用于分配新编号时避免冲突
    
    Args:
        pdf_files: PDF文件路径列表
    
    Returns:
        set: 已使用的编号集合
    """
    used_numbers = set()
    pattern = r'^(\d{3})_'
    
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        match = re.match(pattern, filename)
        if match:
            used_numbers.add(int(match.group(1)))
    
    return used_numbers


def get_next_available_number(used_numbers):
    """
    获取下一个可用的三位数字编号
    
    Args:
        used_numbers: 已使用的编号集合
    
    Returns:
        str: 三位数字字符串格式的编号（如"001"）
    """
    for i in range(1, 1000):  # 从001到999
        if i not in used_numbers:
            return f"{i:03d}"
    
    # 如果所有编号都被使用，从1000开始（虽然不是三位数，但确保唯一）
    i = 1000
    while i in used_numbers:
        i += 1
    return str(i)


def truncate_filename(filename, max_length=100):
    """
    截断文件名以确保不超过指定长度
    保持文件扩展名不变
    """
    # 分离文件名和扩展名
    name_without_ext = os.path.splitext(filename)[0]
    extension = os.path.splitext(filename)[1]
    
    # 如果文件名（包含扩展名）已经不超过最大长度，直接返回
    if len(filename) <= max_length:
        return filename
    
    # 计算可用于文件名主体的最大长度（要为扩展名留空间）
    available_length = max_length - len(extension)
    
    # 截断文件名主体
    truncated_name = name_without_ext[:available_length]
    
    # 重新组合文件名
    truncated_filename = truncated_name + extension
    
    return truncated_filename

def preprocess_pdf_files(directory_path, max_length=100):
    """
    预处理指定目录下的所有PDF文件
    1. 首先检查并处理编号前缀（XXX_格式）
    2. 然后检查文件名长度，如果超过指定长度则重命名文件
    """
    print(f"\n[PDF文件预处理开始]")
    print(f"扫描目录: {directory_path}")
    print(f"最大文件名长度: {max_length} 字符")
    print("-" * 60)
    
    # 获取所有PDF文件
    pdf_files = []
    for file_path in glob.glob(f"{directory_path}/**/*.pdf", recursive=True):
        pdf_files.append(os.path.abspath(file_path))
    
    if not pdf_files:
        print("❌ 未找到任何PDF文件")
        return []
    
    print(f"找到 {len(pdf_files)} 个PDF文件")
    print("\n[第一步: 编号前缀检查与处理]")
    print("-" * 40)
    
    # 提取已存在的编号
    used_numbers = extract_existing_numbers(pdf_files)
    print(f"已使用的编号: {sorted(used_numbers) if used_numbers else '无'}")
    
    # 检查编号前缀并处理
    files_without_prefix = []
    files_with_prefix = []
    
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        if check_has_number_prefix(filename):
            print(f"✓ {filename} - 已有编号前缀")
            files_with_prefix.append(pdf_path)
        else:
            print(f"⚠ {filename} - 缺少编号前缀")
            files_without_prefix.append(pdf_path)
    
    # 为缺少编号的文件分配编号并重命名
    number_assign_count = 0
    current_pdf_files = files_with_prefix.copy()  # 保存已有编号的文件
    
    for pdf_path in files_without_prefix:
        filename = os.path.basename(pdf_path)
        directory = os.path.dirname(pdf_path)
        
        # 获取下一个可用编号
        next_number = get_next_available_number(used_numbers)
        used_numbers.add(int(next_number))
        
        # 构建新文件名
        new_filename = f"{next_number}_{filename}"
        new_pdf_path = os.path.join(directory, new_filename)
        
        try:
            # 重命名文件
            os.rename(pdf_path, new_pdf_path)
            number_assign_count += 1
            
            print(f"✓ 编号分配成功:")
            print(f"  原名称: {filename}")
            print(f"  新名称: {new_filename}")
            print(f"  分配编号: {next_number}")
            
            current_pdf_files.append(new_pdf_path)
            
        except Exception as e:
            print(f"❌ 编号分配失败: {filename}")
            print(f"  错误信息: {e}")
            # 如果重命名失败，仍然使用原文件
            current_pdf_files.append(pdf_path)
    
    print("-" * 40)
    if number_assign_count > 0:
        print(f"✓ 编号处理完成! 共为 {number_assign_count} 个文件分配了编号")
    else:
        print("✓ 编号检查完成! 所有文件都已有编号前缀")
    
    print(f"\n[第二步: 文件名长度检查与处理]")
    print("-" * 40)
    
    # 现在进行文件名长度检查和处理
    length_rename_count = 0
    processed_files = []
    
    for pdf_path in current_pdf_files:
        filename = os.path.basename(pdf_path)
        
        # 检查文件名长度
        if len(filename) <= max_length:
            print(f"✓ {filename} ({len(filename)} 字符) - 长度符合要求")
            processed_files.append(pdf_path)
        else:
            # 需要重命名
            truncated_filename = truncate_filename(filename, max_length)
            
            # 构建新的完整路径
            directory = os.path.dirname(pdf_path)
            new_pdf_path = os.path.join(directory, truncated_filename)
            
            try:
                # 重命名文件
                os.rename(pdf_path, new_pdf_path)
                length_rename_count += 1
                
                print(f"⚠ 文件长度已处理:")
                print(f"  原名称: {filename} ({len(filename)} 字符)")
                print(f"  新名称: {truncated_filename} ({len(truncated_filename)} 字符)")
                
                processed_files.append(new_pdf_path)
                
            except Exception as e:
                print(f"❌ 长度处理失败: {filename}")
                print(f"  错误信息: {e}")
                # 如果重命名失败，仍然使用原文件
                processed_files.append(pdf_path)
    
    print("-" * 60)
    print(f"✓ 预处理完成!")
    print(f"  编号分配: {number_assign_count} 个文件")
    print(f"  长度处理: {length_rename_count} 个文件")
    print(f"  最终文件数: {len(processed_files)} 个")
    print("")
    
    return processed_files

# 设置输入和输出目录
input_dir = "D:/Desktop/ZJU/final_test/pdf/todo/set1"
output_dir = "D:/Desktop/ZJU/final_test/pdf/result/set1"

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 预处理PDF文件（包含扫描、检查长度、重命名等操作）
processed_pdf_files = preprocess_pdf_files(input_dir)

# 如果预处理后没有可用文件，退出程序
if not processed_pdf_files:
    print("预处理后没有可用的PDF文件，退出程序")
    exit()

# 准备请求参数
files_info = []
print(f"[准备上传数据]")
for pdf_path in processed_pdf_files:
    filename = os.path.basename(pdf_path)
    files_info.append({
        "name": filename,
        "is_ocr": True,  # 使用OCR
        "data_id": filename  # 经过预处理，文件名长度已符合要求
    })

print(f"✓ 准备上传 {len(files_info)} 个文件\n")

data = {
    "enable_formula": True,  # 启用公式识别
    "language": "en",        # 设置语言为英语
    "enable_table": True,    # 启用表格识别
    "files": files_info
}

# 记录开始时间（在开始上传PDF之前）
start_time = time.time()

# 上传文件并获取批处理ID
try:
    print("正在申请上传URL...")
    response = requests.post(url, headers=header, json=data)
    
    if response.status_code == 200:
        result = response.json()
        print('申请上传URL成功')
        
        if result["code"] == 0:
            batch_id = result["data"]["batch_id"]
            urls = result["data"]["file_urls"]
            
            print(f'批次ID: {batch_id}, 获得 {len(urls)} 个上传URL')
            
            # 上传文件
            for i in range(len(urls)):
                print(f"正在上传第 {i+1}/{len(urls)} 个文件: {os.path.basename(processed_pdf_files[i])}")
                with open(processed_pdf_files[i], 'rb') as f:
                    res_upload = requests.put(urls[i], data=f)
                    if res_upload.status_code == 200:
                        print(f"✓ 文件上传成功")
                    else:
                        print(f"✗ 文件上传失败: {res_upload.status_code}")
            
            print("\n所有文件上传完成，等待处理...\n")
            
            # 查询处理状态
            status_url = f'https://mineru.net/api/v4/extract-results/batch/{batch_id}'
            completed_files = []
            failed_files = []
            
            while True:
                time.sleep(10)  # 每10秒检查一次
                status_response = requests.get(status_url, headers=header)
                
                if status_response.status_code != 200:
                    print(f"查询状态失败: {status_response.status_code} - {status_response.text}")
                    time.sleep(30)
                    continue
                
                status = status_response.json()
                
                # 检查是否有extract_result字段
                if "extract_result" not in status["data"]:
                    print("等待处理启动...")
                    continue
                
                # 显示每个文件的当前状态
                print("\n当前处理状态:")
                print("-" * 50)
                
                all_done = True
                
                for file_result in status["data"]["extract_result"]:
                    file_state = file_result.get("state", "")
                    file_name = file_result.get("file_name", "未知文件")
                    
                    # 如果文件状态是 done 并且还没有下载过
                    if file_state == "done" and file_name not in completed_files:
                        print(f"✓ 文件 '{file_name}' - 处理完成，准备下载")
                        
                        # 使用正确的字段名下载结果 (full_zip_url 而不是 result_url)
                        if "full_zip_url" in file_result:
                            print(f"  下载结果中...")
                            download_response = requests.get(file_result["full_zip_url"])
                            
                            if download_response.status_code == 200:
                                # 保存到本地文件
                                safe_filename = file_name.replace(":", "_").replace("/", "_").replace("\\", "_")
                                output_path = f"{output_dir}/{safe_filename}_result.zip"
                                with open(output_path, "wb") as f:
                                    f.write(download_response.content)
                                print(f"  ✓ 结果已保存到 {output_path}")
                                completed_files.append(file_name)
                                time.sleep(1)  # 下载间隔，避免请求过于频繁
                            else:
                                print(f"  ✗ 下载失败: {download_response.status_code}")
                        else:
                            print(f"  ✗ 找不到下载链接")
                    
                    elif file_state == "failed" and file_name not in failed_files:
                        print(f"✗ 文件 '{file_name}' - 处理失败: {file_result.get('err_msg', '未知错误')}")
                        failed_files.append(file_name)
                    
                    elif file_name not in completed_files and file_name not in failed_files:
                        print(f"⟳ 文件 '{file_name}' - 状态: {file_state}")
                        all_done = False
                
                print("-" * 50)
                print(f"处理进度: {len(completed_files)}/{len(processed_pdf_files)} 完成, {len(failed_files)} 失败")
                
                # 如果所有文件都处理完了，或者都已失败，就退出循环
                if len(completed_files) + len(failed_files) == len(processed_pdf_files):
                    print("\n✓ 所有文件处理和下载已完成!")
                    break
                
                if not all_done:
                    print("\n继续等待其他文件处理...")
                    print("(每10秒检查一次，请耐心等待)")
            
            # --- 新增功能：自动解压所有下载的ZIP文件 ---
            print("\n[自动解压任务开始]")
            
            # 查找输出目录中所有的 .zip 文件
            zip_files_to_extract = glob.glob(f"{output_dir}/*.zip")

            if not zip_files_to_extract:
                print("在输出目录中没有找到需要解压的 .zip 文件。")
            else:
                print(f"找到 {len(zip_files_to_extract)} 个ZIP文件，准备解压...")
                
                successfully_extracted = []  # 记录成功解压的ZIP文件
                
                for zip_path in zip_files_to_extract:
                    try:
                        # 创建一个与ZIP文件同名的目录来存放解压后的文件 (移除 .zip 后缀)
                        extract_dir = os.path.splitext(zip_path)[0]
                        
                        # 如果目录名以 "_result" 结尾，则去掉这部分 (对应论文名.pdf_result)
                        if extract_dir.endswith("_result"):
                            extract_dir = extract_dir[:-11]  # 移除 ".pdf_result"
                        
                        os.makedirs(extract_dir, exist_ok=True)
                        
                        print(f"  解压: {os.path.basename(zip_path)} -> {os.path.basename(extract_dir)}")
                        
                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                            zip_ref.extractall(extract_dir)
                        
                        print(f"  ✓ 解压成功")
                        successfully_extracted.append(zip_path)

                    except zipfile.BadZipFile:
                        print(f"  ✗ 错误: {os.path.basename(zip_path)} 不是一个有效的ZIP文件或文件已损坏。")
                    except Exception as e:
                        print(f"  ✗ 解压文件 {os.path.basename(zip_path)} 时发生未知错误: {e}")
                
                print(f"\n✓ 解压完成! {len(successfully_extracted)}/{len(zip_files_to_extract)} 个文件成功解压")
                
                # 删除所有成功解压的ZIP文件
                if successfully_extracted:
                    print("\n[清理ZIP文件]")
                    for zip_path in successfully_extracted:
                        try:
                            os.remove(zip_path)
                            print(f"  ✓ 已删除: {os.path.basename(zip_path)}")
                        except Exception as e:
                            print(f"  ✗ 删除失败 {os.path.basename(zip_path)}: {e}")
                    
                    print(f"\n✓ ZIP文件清理完成! 共删除 {len(successfully_extracted)} 个文件")
                else:
                    print("\n⚠ 没有成功解压的文件，不执行清理操作")
                
                print("\n✓ 所有解压和清理任务完成!")
            
            # 记录结束时间并计算总耗时
            end_time = time.time()
            total_time = end_time - start_time
            
            # 格式化时间显示（时分秒）
            hours = int(total_time // 3600)
            minutes = int((total_time % 3600) // 60)
            seconds = total_time % 60
            
            print(f"\n=== PDF解析完成 ===")
            print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
            print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
            
            if hours > 0:
                print(f"解析耗时{hours}小时{minutes}分钟{seconds:.2f}秒")
            elif minutes > 0:
                print(f"解析耗时{minutes}分钟{seconds:.2f}秒")
            else:
                print(f"解析耗时{seconds:.2f}秒")
                
        else:
            print(f'申请上传URL失败，原因: {result.get("msg", "未知错误")}')
            
            # 在API调用失败时也输出耗时信息
            end_time = time.time()
            total_time = end_time - start_time
            
            # 格式化时间显示（时分秒）
            hours = int(total_time // 3600)
            minutes = int((total_time % 3600) // 60)
            seconds = total_time % 60
            
            print(f"\n=== 处理失败 ===")
            print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
            print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
            
            if hours > 0:
                print(f"运行耗时{hours}小时{minutes}分钟{seconds:.2f}秒")
            elif minutes > 0:
                print(f"运行耗时{minutes}分钟{seconds:.2f}秒")
            else:
                print(f"运行耗时{seconds:.2f}秒")
                
    else:
        print(f'请求失败. 状态码: {response.status_code}, 结果: {response.text}')
        
        # 在请求失败时也输出耗时信息
        end_time = time.time()
        total_time = end_time - start_time
        
        # 格式化时间显示（时分秒）
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = total_time % 60
        
        print(f"\n=== 请求失败 ===")
        print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
        print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
        
        if hours > 0:
            print(f"运行耗时{hours}小时{minutes}分钟{seconds:.2f}秒")
        elif minutes > 0:
            print(f"运行耗时{minutes}分钟{seconds:.2f}秒")
        else:
            print(f"运行耗时{seconds:.2f}秒")

except Exception as err:
    print(f"发生错误: {err}")
    import traceback
    traceback.print_exc()
    
    # 即使发生错误也输出耗时信息
    try:
        end_time = time.time()
        total_time = end_time - start_time
        
        # 格式化时间显示（时分秒）
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = total_time % 60
        
        print(f"\n=== 程序异常结束 ===")
        print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
        print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
        
        if hours > 0:
            print(f"运行耗时{hours}小时{minutes}分钟{seconds:.2f}秒")
        elif minutes > 0:
            print(f"运行耗时{minutes}分钟{seconds:.2f}秒")
        else:
            print(f"运行耗时{seconds:.2f}秒")
    except:
        pass  # 如果计时也出错，就不输出时间信息


print("\n程序执行完毕.")
