# README

## 概述

本项目主要包括10个文件，其中，5个为需要运行的主文件，另外5个为辅助文件，总览如下图所示。

![image-20250815114658712](https://lizaiyuan0804.oss-cn-chengdu.aliyuncs.com/tempPics/image-20250815114658712.png)

## 主文件解析

### api.py

**目标：**

批量上传PDF文件并通过MinerU API进行文档解析和处理

**上下文：**

- 使用MinerU在线API服务解析PDF文档
- 支持OCR、公式识别、表格识别等功能
- 自动处理文件命名、上传、解析、下载和解压

**输入：**
- PDF文件目录路径（todo目录）
- 每个PDF文件会被自动预处理（编号、长度检查）

**执行步骤：**

1. 扫描和预处理PDF文件（编号分配、文件名长度处理）
2. 准备批量上传请求数据
3. 申请上传URL并上传所有PDF文件
4. 监控处理状态直到所有文件完成
5. 下载解析结果ZIP文件
6. 自动解压所有ZIP文件并清理

**输出：**

- 解析完成的文档文件（包含文本、图片、表格等）
- 处理统计信息和总耗时

**参数设置：**

1. MinerU API的相关参数包括如下两个部分，每个key有一定的时间限制，可在https://mineru.net/上免费申请

```python
# 设置API参数
url = 'https://mineru.net/api/v4/file-urls/batch'
header = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer eyJ0eXBlIjoiSldUIiwiYWxnIjoiSFM1MTIifQ.eyJqdGkiOiI2MjQwMjI4MiIsInJvbCI6IlJPTEVfUkVHSVNURVIiLCJpc3MiOiJPcGVuWExhYiIsImlhdCI6MTc1NTE1NzI0MywiY2xpZW50SWQiOiJsa3pkeDU3bnZ5MjJqa3BxOXgydyIsInBob25lIjoiIiwib3BlbklkIjpudWxsLCJ1dWlkIjoiM2Q1NTA2ZmYtMWQwMy00YTllLWFmZGUtMjg4N2RmY2VmYTFiIiwiZW1haWwiOiIiLCJleHAiOjE3NTYzNjY4NDN9.WbeV3ADCfmj7Qk-IWPkt77H41_qnS25GtHFWUFT5b8g_KykMKH7ojyicvn7hpbakdDwBdouUHNYLf8cEy4u7Mg'
}
```

2. 输入和输出目录，分别是PDF保存目录和解析结果保存目录，例如

```python
# 设置输入和输出目录
input_dir = "D:/Desktop/ZJU/final_test/pdf/todo"
output_dir = "D:/Desktop/ZJU/final_test/pdf/result"
```

**注意事项：**

一次性上传100+文件可能导致API解析速度偏慢，分批上传和解析会更快，建议单位50，排队时间较少

### filter.py

**目标**

本脚本旨在处理由 MinerU API 解析生成的 JSON 文件。
主要功能包括：
1. 移除 JSON 文件中 "REFERENCES" 部分
2. 清理所有文本内容中的学术引用信息（如引用标记、作者年份等）
以生成一个不包含参考文献、附录和引用标记的干净版 JSON 文件。

**上下文**

前置脚本 `api.py` 会从 MinerU API 下载并解压每个PDF的解析结果。每个PDF的结果都存储在一个名为 `[论文名].pdf_result` 的目录中。此脚本将遍历这些结果目录，处理其中的 `*_content_list.json` 文件。

**输入**

- `results_base_dir`: 存储所有 `..._result` 文件夹的根目录。
- 每个 `..._result` 文件夹中都包含一个 `*_content_list.json` 文件。

**执行步骤**

1. 设置 `results_base_dir` 变量，指向包含所有结果文件夹的目录。
2. 遍历 `results_base_dir` 下的所有 `..._result` 目录。
3. 在每个结果目录中，查找 `*_content_list.json` 文件。
4. 读取并解析该 JSON 文件，它是一个包含多个字典的列表。
5. 精确删除参考文献内容：
   a) 查找 "REFERENCES" 部分的起始位置（具有 text_level=1 的文本项）
   b) 查找 REFERENCES 之后第一个具有 text_level 的文本项（通常是 appendix）
   c) 删除这两个位置之间的所有内容，保留 appendix 等有用部分
6. 对剩余内容中所有 "type"=="text" 的项进行引用清理，去除学术引用标记。
7. 从结果目录的名称中提取论文名。
8. 将处理后的新列表保存为一个新的 JSON 文件，命名为 `[论文名]_filter.json`，并存储在原结果目录中。
9. 如果在文件中未找到参考文献部分，则打印一条警告信息。

 **输出**

- 在每个 `..._result` 文件夹内生成一个新的 `[论文名]_filter.json` 文件，其中：
  1. 精确删除了参考文献列表部分，但保留了 appendix 等有用内容
  2. 所有文本内容已清理学术引用标记

**参数设置**

```python
# 请将此路径设置为包含所有 `..._result` 文件夹的根目录
# 这个路径应该和您 api.py 脚本中的 output_dir 一致
results_base_dir = "D:/Desktop/ZJU/final_test/pdf/result"
```

### dataset_setup.py

**目标：**

批量处理学术论文并构建向量数据库

**上下文：**

- 处理包含文本、公式、图片、表格的学术论文数据
- 使用ChromaDB构建多模态向量数据库
- 支持CUDA加速的嵌入模型

**输入：**

- 论文目录路径（包含多个论文子目录）
- 每个论文目录包含filter.json文件和图片资源

**执行步骤：**

1. 检查CUDA可用性
2. 初始化数据库连接和嵌入模型
3. 批量处理论文（文本、公式、图片、表格）
4. 统计处理结果和耗时

**输出：**

- 构建完成的ChromaDB向量数据库
- 处理统计信息和总耗时

**参数设置：**

需要设置两个参数，分别是api.py的输出目录，以及DB存储目录

```python
    db = AcademicPaperDatabase(db_path="D:/desktop/ZJU/acl300/academic_papers_db")
    
    # 批量处理所有论文 - 支持分批处理
    papers_directory = "D:/desktop/ZJU/final_test/pdf/result"
```

**注意事项：**

建议启用CUDA

### ma_gen.py

**目标：**

使用多智能体架构自动生成学术综述

**上下文：**

- 基于ChromaDB向量数据库中已有的学术论文内容
- 通过多个专业智能体协作完成复杂的学术综述生成任务
- 支持交互模式和命令行参数模式两种运行方式

**输入：**

- 用户提供的主题和子主题
- API配置（密钥、模型选择等）
- 向量数据库路径和输出路径

**执行步骤：**

1. 初始化LLM工厂和数据库连接
2. 解析和标准化用户输入的主题
3. 创建并协调多个智能体（解释器、规划、丰富、撰写）
4. 按阶段生成综述（大纲创建→大纲丰富→内容撰写→内容整合）
5. 保存结果并记录生成耗时

**输出：**

- 学术综述Markdown文件
- 元数据JSON文件
- Word文档
- 生成统计信息和总耗时

**参数设置：**

主要参数

```python
    parser.add_argument("--topic", "-t", type=str, default="Large Language Models", help="综述主题")
    parser.add_argument("--subtopics", "-s", type=str, default="fine-tuning, Post-training, Reasoning, Alignment, Multimodal", help="子主题，用逗号分隔")
    parser.add_argument("--output", "-o", type=str, default="./ma_output/", help="输出文件路径")
    parser.add_argument("--api-key", "-k", type=str, default="", help="OpenRouter API密钥")
    parser.add_argument("--base-url", "-u", type=str, default="https://openrouter.ai/api/v1", help="API基础URL")
    parser.add_argument("--db-path", "-d", type=str, default="D:/desktop/ZJU/acl300/academic_papers_db", help="向量数据库路径")
    parser.add_argument("--interpreter-model", type=str, default=ModelType.GEMINI.value, help="解释器智能体使用的模型") 
    parser.add_argument("--planner-model", type=str, default=ModelType.GEMINI.value, help="规划智能体使用的模型")
    parser.add_argument("--enricher-model", type=str, default=ModelType.GEMINI.value, help="丰富智能体使用的模型")
    parser.add_argument("--writer-model", type=str, default=ModelType.GEMINI.value, help="撰写智能体使用的模型")
    parser.add_argument("--log-dir", type=str, default="./logs", help="日志目录路径")
```

**注意事项：**

OpenRouter API和OpenAI一样，可以更换base_url和key为对应内容

该文件中基础的Agent定义在multi_agent中，用到的工具函数大多在utils中，同时，少量使用了llm_review_generator.py中的类（该文件是最初单个模型生成综述），最后转换md到word过程中使用了md_to_word_converter.py

### idea_gen.py

**目标：**

基于已生成的综述文档，使用多智能体架构自动生成研究想法

**上下文：**

- 基于Survey Gen产出的markdown文件和enriched outline JSON文件
- 通过多个专业智能体协作完成复杂的想法生成任务
- 支持交互模式和命令行参数模式两种运行方式

**输入：**

- survey_md_dir: Survey Gen产出的包含多个md文件的目录路径
- logs_dir: 包含LLM调用日志的目录路径（用来提取enriched_outline）
- API配置、模型选择和生成参数
- 向量数据库路径和输出路径

**执行步骤：**

1. 从指定目录找到最新的md和json文件
2. 解析文件获取final_result和enriched_outline数据
3. 初始化LLM工厂和数据库连接
4. 启动多智能体系统：构建机会图谱→生成idea→评判→优化
5. 保存结果并记录生成耗时

**输出：**

- 结构化的研究想法集JSON文件
- 可读的想法摘要Markdown文件
- 生成统计信息和总耗时

**参数设置**

```python
    
    parser.add_argument("--survey-md-dir", "-md", type=str, default="./ma_output/", 
                       help="Survey Gen产出的markdown文件目录")
    parser.add_argument("--logs-dir", "-ld", type=str, default="./logs/", 
                       help="LLM调用日志目录（包含enriched outline数据）")
    parser.add_argument("--output", "-o", type=str, default="./idea_output/", 
                       help="输出文件路径")
    parser.add_argument("--api-key", "-k", type=str, 
                       default="", 
                       help="OpenRouter API密钥")
    parser.add_argument("--base-url", "-u", type=str, default="https://openrouter.ai/api/v1", 
                       help="API基础URL")
    parser.add_argument("--db-path", "-d", type=str, default="D:/desktop/ZJU/acl300/academic_papers_db", 
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
```

**注意事项：**

该脚本主要用到了idea_gen_agent.py中的内容

由于时间紧张，该脚本并未完全调优，生产的idea质量一般，尽管异常处理机制不够完善，但目前运行正常

## 辅助文件

### multi_agent.py

**目标：**

自动化生成高质量学术综述论文，通过多智能体协作完成从主题分析到内容撰写的全流程

**上下文：**

- 基于多智能体架构，包含解释、规划、丰富、撰写四个专业智能体
- 支持多种LLM模型（Claude、GPT-4、Gemini、Qwen、DeepSeek等）
- 集成学术数据库检索和相似度计算功能
- 采用CoT（思维链）技术指导综述写作流程

**输入：**

- 研究主题描述（中文或英文）
- 可选的次要关键词列表
- LLM配置参数（模型选择、API密钥等）

**执行步骤：**

1. 主题解析与标准化（生成学术关键词矩阵）
2. 综述大纲结构规划（固定6章节框架）
3. 章节内容指引丰富（添加写作指导和关键要点）
4. 分章节并行撰写（基于材料检索和引文管理）
5. 内容整合与格式化输出

**输出：**

- 结构化的学术综述文档（Markdown格式）
- 详细的执行日志和统计信息
- 智能体协作过程的完整记录

### md_to_word_converter.py

**目标：**

将Markdown格式文档转换为格式规范的Word文档，解决标题对齐、字体设置、公式渲染等格式化问题

**上下文：**

- 基于python-docx库进行Word文档操作和样式设置
- 集成Pandoc支持LaTeX数学公式转换为Word原生格式
- 支持PIL图片处理和插入功能
- 自动处理HTML内容清理和文档结构优化
- 统一中英文字体配置（Times New Roman + 宋体/黑体）

**输入：**

- Markdown格式的文本内容
- Word文档输出路径
- 可选的文档标题参数

**执行步骤：**

1. 预处理清理HTML块和转义字符
2. 解析Markdown结构（标题、段落、列表、代码块、图片、公式）
3. 设置文档基础样式（字体、行距、页码、页眉）
4. 按层级处理标题对齐方式（一级标题居中，其他左对齐）
5. 处理LaTeX公式转换为Word原生数学格式
6. 插入图片并设置合适尺寸和说明
7. 应用列表缩进和格式化文本样式

**输出：**

- 格式化的Word文档文件（.docx）
- 处理统计报告（图片插入、公式转换、HTML清理等信息）

### utils

**目标：**

提供学术综述生成系统的核心工具函数库，支持材料检索、引用管理、内容解析和文本处理等功能

**上下文：**

- 集成学术数据库检索和相似度计算功能
- 支持多种LLM模型的调用日志记录和管理
- 提供完整的学术引用管理系统（Citation/CitationManager）
- 包含响应解析器处理各种格式的LLM输出
- 实现文档结构化处理和内容清理功能

**输入：**

- 搜索关键词和主题信息
- LLM模型响应内容
- 学术材料数据（文本、公式、图表等）
- 引用信息和元数据

**执行步骤：**

1. 材料检索与相关性计算（基于主题和关键词匹配）
2. LLM调用日志记录和响应处理
3. 学术引用自动生成和格式化管理
4. 文档结构解析（大纲、摘要、章节等）
5. 内容清理和格式标准化处理
6. 图表和公式的提取与插入处理

**输出：**

- 结构化的材料检索结果
- 标准格式的学术引用和参考文献
- 清理后的文档内容和解析结果
- 完整的操作日志和统计信息

### idea_gen_agent.py

**目标：**

基于学术文献自动生成高质量研究想法，通过多智能体协作完成从机会挖掘到想法精炼的全流程

**上下文：**

- 采用多智能体架构（挖掘、生成、新颖性评估、可行性评估、精炼智能体）
- 构建语义机会图谱识别研究空白和潜在机会点
- 支持多种想法生成策略（迁移、组合、逆向工程、跨领域融合）
- 集成新颖性和可行性双重评估机制
- 实现迭代辩论和精炼流程确保想法质量

**输入：**

- 综述文档内容和富化大纲信息
- LLM工厂实例和学术数据库连接
- 生成配置参数（想法数量、评估阈值、迭代轮次等）

**执行步骤：**

1. 语义图谱构建（实体抽取、关系识别、显著性计算）
2. 研究机会挖掘（空白检测、迁移机会、组合潜力分析）
3. 多策略候选想法生成（基于触发器和模板的批量生成）
4. 新颖性评估（概念、方法、应用、评估四维度分析）
5. 可行性评估（相关性、资源需求、风险分析、图谱一致性）
6. 迭代精炼和辩论（基于评估结果的想法改进和筛选）

**输出：**

- 经过筛选的高质量研究想法列表（包含假设、创新点、实验设计）
- 详细的新颖性和可行性评估报告
- 语义机会图谱和研究空白分析
- 完整的想法演化历史和精炼记录

### llm_review_generator.py

**目标：**

基于学术数据库自动生成高质量的学术综述文章，使用大语言模型整合和分析相关研究文献

**上下文：**

- 支持多种LLM模型调用（OpenAI API、本地Transformers模型）
- 集成增强相似性计算算法，提高文献检索精度和相关性
- 支持多类型学术内容处理（文本、公式、图表、表格）
- 提供可配置的综述生成参数和格式控制
- 基于向量数据库进行语义检索和内容匹配

**输入：**

- 综述研究主题和相关子主题列表
- LLM配置参数（API密钥、模型选择、生成参数）
- 输出文件路径和格式配置

**执行步骤：**

1. 多维度学术材料检索（基于主题和子主题）
2. 增强相似性计算和内容质量筛选
3. 按相关性对检索结果重新排序和分类
4. 构建结构化的LLM提示词模板（包含写作指导）
5. 调用LLM生成详细综述内容
6. 格式化输出并添加参考文献和元数据

**输出：**

- 结构化学术综述文档（Markdown格式，50000字以上）
- 完整的参考文献列表和引用信息
- 材料来源统计和上下文数据（JSON格式）
- 图表公式来源附录和说明文档

## 其他

acl300为示例数据库，由来自ACL 2025的300篇论文构成


环境参考requirements
