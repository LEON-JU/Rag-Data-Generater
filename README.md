# Rag-Data-Generater

面向 Retrieval-Augmented Generation (RAG) 的数据生成工具箱。仓库提供两套可运行示例：  
1) 完整 system prompt + 中断式工具调用管线；
2) 基于 LangChain 的多 Agent 协作式对话。  
所有运行入口都对应一个 YAML 配置文件，可以集中管理 LLM/Elasticsearch/API Key 等参数。

## 目录结构

```text
Rag-Data-Generater
├── configs/
│   ├── full_prompt_pipeline.yaml      # 完整 prompt 管线配置（LLM、ES、数据抽样等）
│   └── multi_agent_langchain.yaml     # LangChain 多 Agent 配置（LLM、问题列表、输出路径）
├── docs/
│   └── data_formats.md                # 当前支持的数据格式说明
├── examples/
│   ├── full_prompt_pipeline/
│   │   └── run_pipeline.py            # 中断式工具调用演示入口
│   └── multi_agent_langchain/
│       └── demo.py                    # LangChain 多 Agent SFT 数据生成入口
├── rag_data_generator/
│   ├── datasets/prepare.py            # HotpotQA/ASearcher 等数据加载与抽样
│   ├── llm/client.py                  # OpenAI/SiliconFlow 客户端封装
│   ├── pipeline/interruption.py       # 中断式对话 Orchestrator
│   ├── prompts/prompts.py             # 系统 prompt 模板和工具 prompt
│   ├── tooling/{registry,tools}.py    # 工具注册表 & Wiki_RAG 工具实现
│   └── search/{elastic,wiki}.py       # ES 搜索辅助函数
├── data/                              # 示例数据（可替换成自己的 Parquet）
├── multi_agent_sft_dataset.jsonl      # 运行示例生成的样例输出
└── pyproject.toml                     # 依赖与安装配置
```

## 配置文件说明

- `env`: 会在脚本启动时写入 `os.environ`，覆盖历史写法（如 `ELASTIC_PASSWORD`、`GENERATOR_LLM_API_KEY` 等），可统一管理 API Key。
- `llm`: 指定模型名称、推理服务 URL、温度等；`full_prompt_pipeline.yaml` 支持 `client=openai|siliconflow` 两种客户端。
- `dataset`: 为示例脚本提供默认数据来源（如 ASearcher 子集、样本索引、SFT 问题列表），也可在命令行覆盖。
- `pipeline`/`output`: 控制最大推理轮数、最终 JSONL 输出路径。

## 运行模式 1：完整 prompt + 工具中断管线

脚本：`examples/full_prompt_pipeline/run_pipeline.py`

按需修改 `configs/full_prompt_pipeline.yaml`

脚本流程：读取配置 → 设置系统 prompt → 运行 `InterruptionOrchestrator` → 打印每一轮思考、工具调用结果和最终回答，并输出 token 统计。

## 运行模式 2：LangChain 多 Agent 协作

脚本：`examples/multi_agent_langchain/demo.py`

按需编辑 `configs/multi_agent_langchain.yaml`：


脚本会组装 5 个 Agent（Reasoning/Search/Summary/Backtrack/Answer），循环调用 Wiki_RAG 工具并将完整标签序列展平为 SFT 所需的 `messages`，写入 JSONL。

# 安装说明
按照ArtSearch的指导进行安装

需要下载wikipedia数据

注意下载elasticsearch后需要设置密码：
```
cd ArtSearch/data/elasticsearch-8.17.3/bin

# 启动服务(建议用tmux挂起服务)
./elasticsearch

# 设置密码，稍后将密码填入configs中的yaml或export ELASTIC_PASSWORD=密码
./elasticsearch-reset-password -u elastic