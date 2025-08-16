import chromadb
import json
import os
import torch
import time  # 添加时间模块用于计时
from pathlib import Path
from sentence_transformers import SentenceTransformer
import hashlib
from typing import Dict, List, Any, Optional
import base64
import re
from chromadb.utils import embedding_functions
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader  # 添加这个import


class AcademicPaperDatabase:
    def __init__(self, db_path: str = "./chroma_db"):
        """初始化学术论文数据库"""
        self.client = chromadb.PersistentClient(path=db_path)

        # 确定设备
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Embedding model is running on: {device}")

        # 使用OpenCLIPEmbeddingFunction并指定设备
        # default model
        # embedding_function = OpenCLIPEmbeddingFunction(
        #     model_name="ViT-H-14", 
        #     checkpoint="laion2b_s32b_b79k",
        #     device=device
        # )
        embedding_function = OpenCLIPEmbeddingFunction(
            model_name="ViT-B-32", 
            checkpoint="laion2b_s34b_b79k",
            device=device
        )
        # 创建图片数据加载器
        image_loader = ImageLoader()

# options:
# model_name: ViT-B-32, ViT-L-14, ViT-H-14, ViT-g-14
# checkpoint: laion2b_s34b_b79k, datacomp_l_s13b_b90k, laion2b_s32b_b79k, laion2b_s34b_b88k

        # 为不同类型的内容创建Collection
        self.collections = {
            'texts': self.client.get_or_create_collection(
                name="academic_texts",
                embedding_function=embedding_function,
                metadata={"description": "学术论文文本内容"}
            ),
            'equations': self.client.get_or_create_collection(
                name="academic_equations", 
                embedding_function=embedding_function,
                metadata={"description": "学术论文公式"}
            ),
            'images': self.client.get_or_create_collection(
                name="academic_images",
                embedding_function=embedding_function,
                data_loader=image_loader,  # 添加图片加载器
                metadata={"description": "学术论文图片和表格"}
            ),
            'tables': self.client.get_or_create_collection(
                name="academic_tables",
                embedding_function=embedding_function,
                data_loader=image_loader,  # 表格也可能有图片
                metadata={"description": "学术论文表格"}
            )
        }
    
    def clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """清理metadata，确保所有值都是Chroma支持的类型"""
        cleaned = {}
        for key, value in metadata.items():
            if value is None:
                # 将None值转换为空字符串
                cleaned[key] = ""
            elif isinstance(value, (bool, int, float, str)):
                # 直接支持的类型
                cleaned[key] = value
            elif isinstance(value, list):
                # 将列表转换为字符串
                cleaned[key] = str(value) if value else ""
            elif isinstance(value, dict):
                # 将字典转换为JSON字符串
                cleaned[key] = json.dumps(value, ensure_ascii=False) if value else ""
            else:
                # 其他类型转换为字符串
                cleaned[key] = str(value) if value is not None else ""
        
        return cleaned
    
    def safe_get_value(self, item: Dict, key: str, default: Any = "") -> Any:
        """安全获取字典值，避免None"""
        value = item.get(key, default)
        return value if value is not None else default
    
    def generate_unique_id(self, paper_name: str, content_type: str, index: int) -> str:
        """生成唯一ID"""
        content = f"{paper_name}_{content_type}_{index}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def process_single_paper(self, paper_dir: Path):
        """处理单篇论文的所有内容"""
        paper_name = paper_dir.name
        filter_json_path = None
        image_dir = paper_dir  
        
        # 查找filter.json文件
        for file in paper_dir.glob("*_filter.json"):
            filter_json_path = file
            break
        
        if not filter_json_path:
            print(f"未找到{paper_name}的filter.json文件")
            return
                    
        # 读取JSON数据
        try:
            # 准备路径，以支持Windows长路径
            path_to_open = filter_json_path
            if os.name == 'nt':
                # Path.resolve() 获取绝对路径
                abs_path_str = str(filter_json_path.resolve())
                # 添加长路径前缀
                if not abs_path_str.startswith('\\\\?\\'):
                    path_to_open = '\\\\?\\' + abs_path_str

            with open(path_to_open, 'r', encoding='utf-8') as f:
                paper_data = json.load(f)
        except Exception as e:
            print(f"读取JSON文件失败: {e}")
            return
        
        # 按类型处理数据 - 分别处理，避免一个模块的错误影响其他模块
        try:
            self._process_texts(paper_data, paper_name, image_dir)
        except Exception as e:
            print(f"处理文本内容时出错: {e}")
            import traceback
            traceback.print_exc()
            
        try:
            self._process_equations(paper_data, paper_name, image_dir)
        except Exception as e:
            print(f"处理公式内容时出错: {e}")
            import traceback
            traceback.print_exc()
            
        try:
            self._process_images(paper_data, paper_name, image_dir)
        except Exception as e:
            print(f"处理图片内容时出错: {e}")
            import traceback
            traceback.print_exc()
            
        try:
            self._process_tables(paper_data, paper_name, image_dir)
        except Exception as e:
            print(f"处理表格内容时出错: {e}")
            import traceback
            traceback.print_exc()
    
    def _process_texts(self, paper_data: List[Dict], paper_name: str, image_dir: Path):
        """处理文本内容"""
        text_items = [item for item in paper_data if item.get('type') == 'text']
        
        if not text_items:
            print("未找到文本内容")
            return
            
        ids = []
        documents = []
        metadatas = []
        
        for idx, item in enumerate(text_items):
            unique_id = self.generate_unique_id(paper_name, 'text', idx)
            text_content = self.safe_get_value(item, 'text', '')
            
            # 跳过空文本
            if not text_content.strip():
                continue
                
            ids.append(unique_id)
            documents.append(text_content)
            
            # 创建metadata并清理
            raw_metadata = {
                'paper_name': paper_name,
                'content_type': 'text',
                'page_idx': self.safe_get_value(item, 'page_idx', -1),
                'text_level': self.safe_get_value(item, 'text_level', 0),
                'order_in_paper': idx,
                'original_data': json.dumps(item, ensure_ascii=False)
            }
            
            cleaned_metadata = self.clean_metadata(raw_metadata)
            metadatas.append(cleaned_metadata)
        
        if ids:
            try:
                self.collections['texts'].add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )
                print(f"添加了 {len(ids)} 个文本项")
            except Exception as e:
                print(f"添加文本项时出错: {e}")
                # 调试：打印第一个metadata
                if metadatas:
                    print(f"第一个metadata示例: {metadatas[0]}")
    
    def _find_context_text(self, paper_data: List[Dict], target_index: int, direction: str = 'up') -> List[str]:
        """
        查找公式的上下文文本
        
        Args:
            paper_data: 完整的论文数据
            target_index: 目标公式在paper_data中的索引
            direction: 查找方向，'up' 向上查找，'down' 向下查找
        
        Returns:
            找到的文本内容列表
        """
        context_texts = []
        step = -1 if direction == 'up' else 1
        current_index = target_index + step
        
        while 0 <= current_index < len(paper_data):
            item = paper_data[current_index]
            item_type = item.get('type', '')
            
            if item_type == 'text':
                # 找到text类型，检查内容是否为空
                text_content = self.safe_get_value(item, 'text', '').strip()
                if text_content:  # 如果文本不为空，添加到上下文并停止查找
                    context_texts.append(text_content)
                    break
                # 如果文本为空，继续查找下一个
                current_index += step
                continue
            elif item_type == 'equation':
                # 如果是公式，继续向前/后查找
                current_index += step
                continue
            elif item_type in ['image', 'table']:
                # 忽略图片和表格，继续查找
                current_index += step
                continue
            else:
                # 其他类型也继续查找
                current_index += step
                continue
        
        return context_texts
    
    def _find_table_caption_in_nearby_text(self, paper_data: List[Dict], table_index: int) -> str:
        """
        在表格前后的text元素中查找以'Table NUM'格式开头的标题
        
        Args:
            paper_data: 完整的论文数据
            table_index: 表格在paper_data中的实际索引（不是table_items中的索引）
        
        Returns:
            找到的表格标题，如果没找到则返回空字符串
        """
        import re
        

        # 检查下一个元素
        if table_index < len(paper_data) - 1:
            next_item = paper_data[table_index + 1]
            if next_item.get('type') == 'text':
                next_text = self.safe_get_value(next_item, 'text', '').strip()
                # 检查是否以"Table NUM"格式开头（支持中英文）
                if re.match(r'^Table\s+\d+|^表\s*\d+', next_text, re.IGNORECASE):
                    return next_text
                # 检查是否以Algorithm NUM格式开头
                if re.match(r'^Algorithm\s+\d+|^算法\s*\d+', next_text, re.IGNORECASE):
                    return next_text
            elif next_item.get('type') == 'table':
                next_caption = self.safe_get_value(next_item, 'table_caption', [])
                # 确保caption是列表，然后转换为字符串
                if isinstance(next_caption, str):
                    next_caption = [next_caption]
                next_text = ' '.join(next_caption) if next_caption else ""
                if next_text.strip():
                    return next_text.strip()
                # 检查上一个元素
        if table_index > 0:
            prev_item = paper_data[table_index - 1]
            if prev_item.get('type') == 'text':
                prev_text = self.safe_get_value(prev_item, 'text', '').strip()
                # 检查是否以"Table NUM"格式开头（支持中英文）
                if re.match(r'^Table\s+\d+|^表\s*\d+', prev_text, re.IGNORECASE):
                    return prev_text
                # 检查是否以Algorithm NUM格式开头
                if re.match(r'^Algorithm\s+\d+|^算法\s*\d+', prev_text, re.IGNORECASE):
                    return prev_text
            elif prev_item.get('type') == 'table':
                prev_caption = self.safe_get_value(prev_item, 'table_caption', [])
                # 确保caption是列表，然后转换为字符串
                if isinstance(prev_caption, str):
                    prev_caption = [prev_caption]
                prev_text = ' '.join(prev_caption) if prev_caption else ""
                if prev_text.strip():
                    return prev_text.strip()
        
        return ""
    
    def _find_image_references(self, paper_data: List[Dict], search_key: str) -> List[str]:
        """
        在论文文本中查找对图片的引用
        
        Args:
            paper_data: 完整的论文数据
            search_key: 搜索关键词（通常是图片标题的前8个字符，如"Figure 1"）
        
        Returns:
            包含引用的文本段落列表
        """
        reference_texts = []
        
        # 遍历所有文本项
        for item in paper_data:
            if item.get('type') == 'text':
                text_content = self.safe_get_value(item, 'text', '').strip()
                
                # 检查文本中是否包含搜索关键词
                if search_key.lower() in text_content.lower():
                    # 找到引用，添加到结果中
                    # 限制文本长度，避免过长的描述
                    if len(text_content) > 500:
                        # 找到包含关键词的句子或段落
                        sentences = text_content.split('.')
                        for sentence in sentences:
                            if search_key.lower() in sentence.lower():
                                # 取包含引用的句子及其前后句子（如果存在）
                                sentence_index = sentences.index(sentence)
                                context_sentences = []
                                
                                # 添加前一句（如果存在且不为空）
                                if sentence_index > 0 and sentences[sentence_index - 1].strip():
                                    context_sentences.append(sentences[sentence_index - 1].strip())
                                
                                # 添加当前句子
                                context_sentences.append(sentence.strip())
                                
                                # 添加后一句（如果存在且不为空）
                                if sentence_index < len(sentences) - 1 and sentences[sentence_index + 1].strip():
                                    context_sentences.append(sentences[sentence_index + 1].strip())
                                
                                reference_text = '. '.join(context_sentences)
                                # 确保引用文本不超过300字符
                                if len(reference_text) > 1000:
                                    reference_text = reference_text[:1000] + "..."
                                reference_texts.append(reference_text)
                                break  # 找到一个引用就够了，避免重复
                    else:
                        # 文本较短，直接使用整个文本
                        reference_texts.append(text_content)
                    
                    # 限制引用数量，避免过多重复信息
                    if len(reference_texts) >= 3:
                        break
        
        return reference_texts
    
    def _process_equations(self, paper_data: List[Dict], paper_name: str, image_dir: Path):
        """处理公式内容"""
        equation_items = [item for item in paper_data if item.get('type') == 'equation']
        
        if not equation_items:
            print("未找到公式内容")
            return
            
        ids = []
        documents = []
        metadatas = []
        
        # 创建一个映射，从equation item到在paper_data中的索引
        equation_to_index = {}
        for i, item in enumerate(paper_data):
            if item.get('type') == 'equation':
                equation_to_index[id(item)] = i
        
        for idx, item in enumerate(equation_items):
            unique_id = self.generate_unique_id(paper_name, 'equation', idx)
            
            # 构建可搜索的文档内容
            equation_text = self.safe_get_value(item, 'text', '')
            text_format = self.safe_get_value(item, 'text_format', 'unknown')
            
            # 获取当前公式在paper_data中的索引
            paper_data_index = equation_to_index.get(id(item), -1)
            
            # 查找上下文
            context_before = []
            context_after = []
            
            if paper_data_index != -1:
                # 向上查找上下文
                context_before = self._find_context_text(paper_data, paper_data_index, 'up')
                # 向下查找上下文  
                context_after = self._find_context_text(paper_data, paper_data_index, 'down')
            
            # 构建带上下文的可搜索内容
            searchable_content_parts = []
            
            # 添加上文
            if context_before:
                searchable_content_parts.append(f"Context before: {' '.join(context_before)}")
            
            # 添加公式主体
            equation_part = f"Mathematical equation: {equation_text}"
            if text_format == 'latex':
                equation_part += f" (LaTeX format)"
            searchable_content_parts.append(equation_part)
            
            # 添加下文
            if context_after:
                searchable_content_parts.append(f"Context after: {' '.join(context_after)}")
            
            # 组合所有部分
            searchable_content = " | ".join(searchable_content_parts)
            
            ids.append(unique_id)
            documents.append(searchable_content)
            
            # 创建metadata并清理（增加上下文信息）
            raw_metadata = {
                'paper_name': paper_name,
                'content_type': 'equation',
                'page_idx': self.safe_get_value(item, 'page_idx', -1),
                'order_in_paper': idx,
                'equation_text': equation_text,
                'text_format': text_format,
                'context_before': ' '.join(context_before) if context_before else "",
                'context_after': ' '.join(context_after) if context_after else "",
                'has_context': bool(context_before or context_after),
                'original_data': json.dumps(item, ensure_ascii=False)
            }
            
            cleaned_metadata = self.clean_metadata(raw_metadata)
            metadatas.append(cleaned_metadata)
        
        if ids:
            try:
                self.collections['equations'].add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )
                print(f"成功添加了 {len(ids)} 个公式项")
            except Exception as e:
                print(f"添加公式项时出错: {e}")
    
    def _process_images(self, paper_data: List[Dict], paper_name: str, image_dir: Path):
        """处理图片内容 - 使用真正的多模态方法"""
        image_items = [item for item in paper_data if item.get('type') == 'image']
        
        if not image_items:
            print("未找到图片内容")
            return
            
        # 分别处理图片和文本描述
        image_ids = []
        image_uris = []
        image_metadatas = []
        
        text_ids = []
        text_documents = []
        text_metadatas = []
        
        for idx, item in enumerate(image_items):
            img_path = self.safe_get_value(item, 'img_path', '')
            
            if img_path and image_dir.exists():
                full_img_path = image_dir / img_path
                if full_img_path.exists():
                    # 处理图片：使用 uris 让ChromaDB自动加载和嵌入
                    image_unique_id = self.generate_unique_id(paper_name, 'image', idx)
                    image_ids.append(image_unique_id)
                    image_uris.append(str(full_img_path))  # 使用绝对路径
                    
                    # 图片的metadata
                    img_caption = self.safe_get_value(item, 'img_caption', [])
                    img_footnote = self.safe_get_value(item, 'img_footnote', [])
                    
                    if isinstance(img_caption, str):
                        img_caption = [img_caption]
                    if isinstance(img_footnote, str):
                        img_footnote = [img_footnote]
                    
                    caption_text = ' '.join(img_caption) if img_caption else ""
                    footnote_text = ' '.join(img_footnote) if img_footnote else ""
                    
                    img_metadata = {
                        'paper_name': paper_name,
                        'content_type': 'image',
                        'page_idx': self.safe_get_value(item, 'page_idx', -1),
                        'order_in_paper': idx,
                        'img_path': img_path,
                        'img_caption': caption_text,
                        'img_footnote': footnote_text,
                        'modality': 'image',  # 标记为图片模态
                        'original_data': json.dumps(item, ensure_ascii=False)
                    }
                    
                    image_metadatas.append(self.clean_metadata(img_metadata))
                    # caption 与 footnote 添加
                    caption_text = caption_text + footnote_text 
                    # 同时添加图片的文本描述到同一个collection
                    if caption_text:  # 只有当有说明文字时才添加
                        text_unique_id = self.generate_unique_id(paper_name, 'image_text', idx)
                        text_ids.append(text_unique_id)
                        
                        # 构建改进的文本描述
                        searchable_content = f"{caption_text}"
                        
                        # 查找图片引用文本并添加到描述中
                        reference_texts = []
                        if caption_text:
                            # 取caption前8个字符作为搜索关键词（通常是"Figure NUM"格式）
                            search_key = caption_text[:8].strip()
                            if search_key:
                                reference_texts = self._find_image_references(paper_data, search_key)
                                if reference_texts:
                                    # 将引用文本添加到描述中
                                    reference_content = " ".join(reference_texts)
                                    searchable_content += f" Content: {reference_content}"
                        
                        text_documents.append(searchable_content)
                        
                        text_metadata = {
                            'paper_name': paper_name,
                            'content_type': 'image_text',
                            'page_idx': self.safe_get_value(item, 'page_idx', -1),
                            'order_in_paper': idx,
                            'img_path': img_path,
                            'img_caption': caption_text,
                            'img_footnote': footnote_text,
                            'reference_texts': ' | '.join(reference_texts) if reference_texts else "",
                            'has_references': bool(reference_texts),
                            'search_key_used': search_key if caption_text else "",
                            'modality': 'text',  # 标记为文本模态
                            'related_image_id': image_unique_id,  # 关联到对应的图片
                            'original_data': json.dumps(item, ensure_ascii=False)
                        }
                        
                        text_metadatas.append(self.clean_metadata(text_metadata))
                else:
                    print(f"未找到图片文件: {full_img_path}")
            else:
                print(f"图片路径无效: {img_path}")
        
        # 添加图片（使用uris）
        if image_ids:
            try:
                self.collections['images'].add(
                    ids=image_ids,
                    uris=image_uris,  # 使用uris让ChromaDB自动处理图片
                    metadatas=image_metadatas
                )
                print(f"成功添加了 {len(image_ids)} 个图片项")
            except Exception as e:
                print(f"添加图片项时出错: {e}")
                import traceback
                traceback.print_exc()
        
        # 添加图片的文本描述（使用documents）
        if text_ids:
            try:
                self.collections['images'].add(
                    ids=text_ids,
                    documents=text_documents,  # 图片说明的文本
                    metadatas=text_metadatas
                )
            except Exception as e:
                print(f"添加图片文本描述时出错: {e}")
                import traceback
                traceback.print_exc()
    
    def _process_tables(self, paper_data: List[Dict], paper_name: str, image_dir: Path):
        """处理表格内容 - 支持多模态（表格图片+文本内容）"""
        table_items = [item for item in paper_data if item.get('type') == 'table']
        
        if not table_items:
            print("未找到表格内容")
            return
            
        # 分别处理表格图片和文本内容
        table_image_ids = []
        table_image_uris = []
        table_image_metadatas = []
        
        table_text_ids = []
        table_text_documents = []
        table_text_metadatas = []
        
        for idx, item in enumerate(table_items):
            table_caption = self.safe_get_value(item, 'table_caption', [])
            table_footnote = self.safe_get_value(item, 'table_footnote', [])
            table_body = self.safe_get_value(item, 'table_body', '')
            img_path = self.safe_get_value(item, 'img_path', '')
            
            # 确保caption和footnote是列表
            if isinstance(table_caption, str):
                table_caption = [table_caption]
            if isinstance(table_footnote, str):
                table_footnote = [table_footnote]
            
            caption_text = ' '.join(table_caption) if table_caption else ""
            footnote_text = ' '.join(table_footnote) if table_footnote else ""
            
            # 如果caption和footnote都为空，尝试从前后的text元素中查找表头
            if not caption_text and not footnote_text:
                # 找到当前表格在paper_data中的实际索引
                table_index_in_paper = -1
                for i, paper_item in enumerate(paper_data):
                    if paper_item is item:  # 使用对象引用比较
                        table_index_in_paper = i
                        break
                
                if table_index_in_paper != -1:
                    nearby_caption = self._find_table_caption_in_nearby_text(paper_data, table_index_in_paper)
                    if nearby_caption:
                        caption_text = nearby_caption
                        print(f"在附近文本中找到表格标题: {nearby_caption[:50]}...")
                    else:
                        print(f"!!!在附近文本中也未找到表格标题: paper={paper_name}, idx={idx}!!!")
            
            # 处理表格图片（如果存在）
            if img_path and image_dir.exists():
                full_img_path = image_dir / img_path
                if full_img_path.exists():
                    # 使用uris添加表格图片
                    table_img_id = self.generate_unique_id(paper_name, 'table_image', idx)
                    table_image_ids.append(table_img_id)
                    table_image_uris.append(str(full_img_path))
                    
                    table_img_metadata = {
                        'paper_name': paper_name,
                        'content_type': 'table_image',
                        'page_idx': self.safe_get_value(item, 'page_idx', -1),
                        'order_in_paper': idx,
                        'table_caption': caption_text,
                        'table_footnote': footnote_text,
                        'has_table_body': bool(table_body),
                        'img_path': img_path,
                        'modality': 'image',  # 标记为图片模态
                        'original_data': json.dumps(item, ensure_ascii=False)
                    }
                    
                    table_image_metadatas.append(self.clean_metadata(table_img_metadata))
                else:
                    print(f"未找到表格图片文件: {full_img_path}")
            # 组合caption和footnote文本
            caption_text = caption_text + footnote_text 
            
            # 构建表格的文本描述
            searchable_content = ""
            if caption_text.strip():
                searchable_content = f"{caption_text.strip()}"
            else:
                print(f"!!!空表格内容，没有表头: paper={paper_name}, idx={idx}!!!")
            
            # 简化HTML表格为文本
            clean_table = re.sub(r'<[^>]+>', ' ', table_body)
            clean_table = re.sub(r'\s+', ' ', clean_table).strip()
            
            # 添加表格内容
            if clean_table:
                if searchable_content:
                    searchable_content += f" Content: {clean_table}"
                else:
                    searchable_content = f"Content: {clean_table}"
            
            # 只有当有实际内容时才添加到数据库
            if searchable_content.strip():
                table_text_id = self.generate_unique_id(paper_name, 'table_text', idx)
                table_text_ids.append(table_text_id)
                
                # 查找表格引用文本并添加到描述中
                reference_texts = []
                if caption_text:
                    # 取caption前7个字符作为搜索关键词（通常是"Table NUM"格式）
                    search_key = caption_text[:7].strip()
                    if search_key:
                        reference_texts = self._find_image_references(paper_data, search_key)
                        if reference_texts:
                            # 将引用文本添加到描述中
                            reference_content = " ".join(reference_texts)
                            searchable_content += f" Referenced in text: {reference_content}"
                
                table_text_documents.append(searchable_content)
                
                table_text_metadata = {
                    'paper_name': paper_name,
                    'content_type': 'table_text',
                    'page_idx': self.safe_get_value(item, 'page_idx', -1),
                    'order_in_paper': idx,
                    'table_caption': caption_text,
                    'table_footnote': footnote_text,
                    'has_table_body': bool(table_body),
                    'img_path': img_path,
                    'has_table_image': bool(img_path and (image_dir / img_path).exists()),
                    'reference_texts': ' | '.join(reference_texts) if reference_texts else "",
                    'has_references': bool(reference_texts),
                    'search_key_used': search_key if caption_text else "",
                    'modality': 'text',  # 标记为文本模态
                    'original_data': json.dumps(item, ensure_ascii=False)
                }
                
                table_text_metadatas.append(self.clean_metadata(table_text_metadata))
            else:
                print(f"跳过空表格内容: paper={paper_name}, idx={idx}")
        
        # 添加表格图片（使用uris）
        if table_image_ids:
            try:
                self.collections['tables'].add(
                    ids=table_image_ids,
                    uris=table_image_uris,  # 使用uris让ChromaDB自动处理表格图片
                    metadatas=table_image_metadatas
                )
                print(f"成功添加了 {len(table_image_ids)} 个表格图片")
            except Exception as e:
                print(f"添加表格图片时出错: {e}.调试信息 - 图片IDs数量: {len(table_image_ids)}, URIs数量: {len(table_image_uris)}, 元数据数量: {len(table_image_metadatas)}")
        
        # 添加表格文本内容（使用documents）
        if table_text_ids:
            try:
                # 确保数组长度一致
                if len(table_text_ids) == len(table_text_documents) == len(table_text_metadatas):
                    self.collections['tables'].add(
                        ids=table_text_ids,
                        documents=table_text_documents,  # 表格的文本内容
                        metadatas=table_text_metadatas
                    )
                    print(f"成功添加了 {len(table_text_ids)} 个表格文本项")
                else:
                    print(f"错误：数组长度不匹配！IDs: {len(table_text_ids)}, documents: {len(table_text_documents)}, metadatas: {len(table_text_metadatas)}")
                    
            except Exception as e:
                print(f"添加表格文本时出错: {e}.调试信息 - 文本IDs数量: {len(table_text_ids)}, 文档数量: {len(table_text_documents)}, 元数据数量: {len(table_text_metadatas)}")
                # 打印第一个示例用于调试
                if table_text_ids:
                    print(f"第一个ID示例: {table_text_ids[0]}")
                if table_text_documents:
                    print(f"第一个文档示例: {table_text_documents[0][:100]}...")
                if table_text_metadatas:
                    print(f"第一个元数据示例: {list(table_text_metadatas[0].keys())}")
    
    def batch_process_papers(self, papers_root_dir: str, start_from: int = 0, max_papers: int = None):
        """批量处理所有论文"""
        papers_dir = Path(papers_root_dir)
        
        if not papers_dir.exists():
            raise FileNotFoundError(f"目录不存在: {papers_root_dir}")
        
        # 遍历所有论文子目录并排序
        paper_dirs = sorted([d for d in papers_dir.iterdir() if d.is_dir()], key=lambda x: x.name)
        
        # 应用起始位置和最大数量限制
        if start_from > 0:
            paper_dirs = paper_dirs[start_from:]
        if max_papers:
            paper_dirs = paper_dirs[:max_papers]
        
        print(f"找到 {len(paper_dirs)} 篇论文需要处理 (从索引 {start_from} 开始)")
        
        success_count = 0
        failed_papers = []
        
        for i, paper_dir in enumerate(paper_dirs):
            try:
                print(f"\n[{i+1}/{len(paper_dirs)}] 正在处理: {paper_dir.name}")
                self.process_single_paper(paper_dir)
                success_count += 1
                print(f"✓ 成功处理: {paper_dir.name}")
            except KeyboardInterrupt:
                print(f"\n用户中断，已处理 {success_count} 篇论文")
                break
            except Exception as e:
                print(f"✗ 处理论文 {paper_dir.name} 时出错: {e}")
                failed_papers.append((paper_dir.name, str(e)))
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\n=== 批量处理完成 ===")
        print(f"成功处理: {success_count}/{len(paper_dirs)} 篇论文")
        if failed_papers:
            print(f"失败论文数量: {len(failed_papers)}")
            print("失败的论文:")
            for paper_name, error in failed_papers[:10]:  # 只显示前10个失败案例
                print(f"  - {paper_name}: {error}")
            if len(failed_papers) > 10:
                print(f"  ... 还有 {len(failed_papers) - 10} 个失败案例")
        
        self.print_statistics()
    
    def print_statistics(self):
        """打印数据库统计信息"""
        print("\n=== 数据库统计信息 ===")
        for name, collection in self.collections.items():
            count = collection.count()
            print(f"{name}: {count} 条记录")
    
    def search_content(self, query: str, content_type: str = None, n_results: int = 10, **filters):
        """搜索内容 - 传统的文本搜索方法"""
        if content_type and content_type in self.collections:
            collections_to_search = [self.collections[content_type]]
        else:
            collections_to_search = list(self.collections.values())
        
        all_results = []
        
        for collection in collections_to_search:
            try:
                results = collection.query(
                    query_texts=[query],
                    n_results=n_results,
                    where=filters if filters else None,
                    include=['documents', 'metadatas', 'distances']
                )
                
                # 添加collection信息到结果中
                for i in range(len(results['ids'][0])):
                    all_results.append({
                        'id': results['ids'][0][i],
                        'document': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i],
                        'collection': collection.name
                    })
            except Exception as e:
                print(f"搜索collection {collection.name}时出错: {e}")
        
        # 按距离排序
        all_results.sort(key=lambda x: x['distance'])
        return all_results[:n_results]
    
    def search_multimodal(self, 
                         query_texts: List[str] = None, 
                         query_images: List[str] = None,  # 图片路径列表
                         content_type: str = None, 
                         n_results: int = 10, 
                         include_data: bool = False,
                         **filters):
        """多模态搜索方法 - 支持文本、图像和URI查询"""
        if content_type and content_type in self.collections:
            collections_to_search = [self.collections[content_type]]
        else:
            # 只搜索支持多模态的collections（有data_loader的）
            collections_to_search = [
                self.collections['images'], 
                self.collections['tables']
            ]
        
        all_results = []
        
        for collection in collections_to_search:
            try:
                query_params = {
                    'n_results': n_results,
                    'where': filters if filters else None,
                    'include': ['documents', 'metadatas', 'distances']
                }
                
                if include_data:
                    query_params['include'].extend(['data', 'uris'])
                
                # 根据查询类型选择查询方法
                if query_texts and query_images:
                    # 同时搜索文本和图像
                    print(f"执行多模态搜索: 文本={query_texts}, 图像={query_images}")
                    # 分别查询然后合并结果
                    text_results = collection.query(query_texts=query_texts, **query_params)
                    
                    # 对于图像查询，需要图片的numpy数组，这里先跳过
                    # 实际使用时需要加载图片并转换为numpy数组
                    results = text_results
                    
                elif query_texts:
                    # 只搜索文本
                    print(f"执行文本搜索: {query_texts}")
                    results = collection.query(query_texts=query_texts, **query_params)
                    
                elif query_images:
                    # 只搜索图像 - 需要将图片路径转换为numpy数组
                    print(f"执行图像搜索: {query_images}")
                    try:
                        from PIL import Image
                        import numpy as np
                        
                        query_image_arrays = []
                        for img_path in query_images:
                            if os.path.exists(img_path):
                                img = Image.open(img_path).convert('RGB')
                                img_array = np.array(img)
                                query_image_arrays.append(img_array)
                            else:
                                print(f"查询图片不存在: {img_path}")
                        
                        if query_image_arrays:
                            results = collection.query(query_images=query_image_arrays, **query_params)
                        else:
                            continue
                            
                    except Exception as e:
                        print(f"处理查询图像时出错: {e}")
                        continue
                else:
                    print("未提供查询内容")
                    continue
                
                # 添加collection信息到结果中
                for i in range(len(results['ids'][0])):
                    result_item = {
                        'id': results['ids'][0][i],
                        'document': results['documents'][0][i] if results['documents'][0][i] else None,
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i],
                        'collection': collection.name
                    }
                    
                    # 如果包含数据和URI
                    if include_data and 'data' in results:
                        result_item['data'] = results['data'][0][i] if i < len(results['data'][0]) else None
                    if 'uris' in results:
                        result_item['uri'] = results['uris'][0][i] if i < len(results['uris'][0]) else None
                    
                    all_results.append(result_item)
                    
            except Exception as e:
                print(f"多模态搜索collection {collection.name}时出错: {e}")
                import traceback
                traceback.print_exc()
        
        # 按距离排序
        all_results.sort(key=lambda x: x['distance'])
        return all_results[:n_results]

if __name__ == "__main__":

# 脚本目标任务：批量处理学术论文并构建向量数据库

# 上下文：
# - 处理包含文本、公式、图片、表格的学术论文数据
# - 使用ChromaDB构建多模态向量数据库
# - 支持CUDA加速的嵌入模型

# 输入：
# - 论文目录路径（包含多个论文子目录）
# - 每个论文目录包含filter.json文件和图片资源

# 执行步骤：
# 1. 检查CUDA可用性
# 2. 初始化数据库连接和嵌入模型
# 3. 批量处理论文（文本、公式、图片、表格）
# 4. 统计处理结果和耗时

# 输出：
# - 构建完成的ChromaDB向量数据库
# - 处理统计信息和总耗时

    
    # 记录开始时间
    start_time = time.time()
    print("=== 开始构建数据库 ===")    
    # 首先判断cuda是否可用
    import torch
    if torch.cuda.is_available():
        print("CUDA is available")
    else:
        print("CUDA is not available")
        exit()
    
    # 初始化数据库
    db = AcademicPaperDatabase(db_path="D:/Desktop/ZJU/final_test/db/")
    
    # 批量处理所有论文 - 支持分批处理
    # papers_directory = "D:/desktop/ZJU/download/dl3/direct_crawler/results"
    papers_directory = "D:/desktop/ZJU/final_test/pdf/result/set1/"
    
    # 处理论文（可根据需要调整参数）
    db.batch_process_papers(papers_directory, start_from=0, max_papers=258)
    
    # 如果需要处理所有论文，取消下面注释
    # print("\n=== 处理所有论文 ===")
    # db.batch_process_papers(papers_directory)
    
    # 记录结束时间并计算总耗时
    end_time = time.time()
    total_time = end_time - start_time
    
    # 格式化时间显示（时分秒）
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = total_time % 60
    
    print(f"\n=== 数据库构建完成 ===")
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    
    if hours > 0:
        print(f"构建数据库总耗时{hours}小时{minutes}分钟{seconds:.2f}秒")
    elif minutes > 0:
        print(f"构建数据库总耗时{minutes}分钟{seconds:.2f}秒")
    else:
        print(f"构建数据库总耗时{seconds:.2f}秒")