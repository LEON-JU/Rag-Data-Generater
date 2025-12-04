# Rag-Data-Generater

面向 Retrieval-Augmented Generation (RAG) 的数据生成工具箱。仓库提供两套可运行示例：  
1) 完整 system prompt + 中断式工具调用管线；2) 基于 LangChain 的多 Agent 协作式对话。  
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

1. 按需修改 `configs/full_prompt_pipeline.yaml`：
   - `env` 中填入 Elasticsearch、LLM 服务 地址 + 密码；  
   - `llm.client` 选择 `openai` 或 `siliconflow`，补充各自的 `model/base_url/api_key`；  
   - `dataset` 的 subset/split/index 控制默认数据抽样。
2. 安装依赖：`pip install -e .`
3. 调用示例（命令行参数会覆盖配置）：

```bash
python examples/full_prompt_pipeline/run_pipeline.py \
    --config configs/full_prompt_pipeline.yaml \
    --subset hotpotqa_rand1000 \
    --split test \
    --index 0 \
    --client openai \
    --max-rounds 4
```

脚本流程：读取配置 → 设置系统 prompt → 运行 `InterruptionOrchestrator` → 打印每一轮思考、工具调用结果和最终回答，并输出 token 统计。

常用可选参数：
- `--sample-size`: 指定读取 parquet 时的采样大小；
- `--client`: 临时切换为 `siliconflow/openai`；
- `--config`: 指向自定义 YAML。

## 运行模式 2：LangChain 多 Agent 协作

脚本：`examples/multi_agent_langchain/demo.py`

1. 编辑 `configs/multi_agent_langchain.yaml`：
   - `env` 保持与 Elasticsearch/Wiki_RAG 相关的变量一致；
   - `llm` 区块内填写 LangChain `ChatOpenAI` 需要的 `model/base_url/api_key/temperature`；
   - `dataset.questions` 列表写入待生成的多跳问题；
   - `output.path` 指定 JSONL 结果输出路径。
2. 运行命令：

```bash
python examples/multi_agent_langchain/demo.py \
    --config configs/multi_agent_langchain.yaml \
    --output multi_agent_sft_dataset.jsonl \
    --max-rounds 5 \
    --verbose
```

脚本会组装 5 个 Agent（Reasoning/Search/Summary/Verification/Answer），循环调用 Wiki_RAG 工具并将完整标签序列展平为 SFT 所需的 `messages`，写入 JSONL。

常用可选参数：
- `--max-rounds`: 限制 search→backtrack 循环次数；
- `--verbose`: 打印每轮中间结果，便于排查；
- `--output`: 指定新的 JSONL 路径。

## 其它提示

- `docs/data_formats.md` 描述了现成的数据格式，可作为构建自己数据集的参考。
- 将新的工具接入管线时，可在 `rag_data_generator/tooling` 内实现 `Tool` 并注册至 `ToolRegistry`，两套示例都会自动拾取。
- 若需批量生成数据，可在上述脚本的基础上编写额外的 CLI，只需指向对应配置文件即可复用所有环境参数。*** End Patch
