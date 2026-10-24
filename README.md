# 企业办公 Agent

一个基于 **LangGraph + RAG + Function Calling** 的企业办公智能体。员工用自然语言即可：

- **问制度**：报销标准、年假病假、发票与审批流程、VPN/密码/打印机等 IT 问题 → 走 **RAG 知识库**检索并标注来源回答；
- **办事务**：查询 IT 工单进度、计算年休假天数 → Agent 自动**调用工具**执行。

项目用一个场景完整覆盖 Agent 核心能力：**ReAct 推理-行动循环、工具调用（Function Calling）、RAG 检索增强、多轮记忆（Memory）**。

---

## 技术栈

| 模块 | 选型 |
|---|---|
| Agent 编排 | LangGraph（StateGraph 手写 ReAct：节点 / 条件边 / 循环） |
| 大模型 | DeepSeek（`deepseek-chat`，OpenAI 兼容，便宜稳定；可换通义/Kimi/智谱/Ollama） |
| RAG 检索 | 文档切分 + 本地特征哈希 Embedding + **混合检索（向量余弦 + 关键词 IDF）** Top-K（零下载、零额外 Key） |
| 记忆 | LangGraph MemorySaver + thread_id（同一会话多轮上下文） |
| 界面 | Streamlit（实时展示思考与工具调用过程） |
| 数据 | 本地 JSON（向量索引、模拟工单） |

---

## 工作流程（ReAct）

```
用户提问
   │
   ▼
[agent 节点] LLM 判断：需要工具吗？
   │                   │
   │ 有 tool_calls      │ 不需要
   ▼                   ▼
[tools 节点] 执行工具   结束并回答
   │  （RAG检索 / 查工单 / 算年假）
   ▲
   └──────── 工具结果回传，回到 agent 继续推理（可多轮）
```

---

## 快速开始（Windows，约 10 分钟）

> 以下命令在**项目根目录**的 PowerShell 中执行。推荐直接用虚拟环境里的 python，可省去"激活/执行策略"的麻烦。

### 1. 创建虚拟环境并安装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. 申请 DeepSeek API Key 并配置

- 打开 https://platform.deepseek.com/ 注册，在「API Keys」创建一个 Key；充值 5～10 元即可大量测试。
- 复制环境变量模板并编辑：

```powershell
copy .env.example .env
notepad .env
```

在 `.env` 中填入你的 Key：

```env
DEEPSEEK_API_KEY=sk-你的真实key
```

> 想换其他 OpenAI 兼容模型，只需改 `.env` 里的 `OPENAI_BASE_URL` 和 `LLM_MODEL`。

### 3. 构建知识库

```powershell
.\.venv\Scripts\python.exe -m knowledge.build_index
```

看到「共 N 个文本块」即成功（知识库文档在 `knowledge/docs/`，可自行增删后重新构建）。

### 4. 启动界面

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

浏览器自动打开 `http://localhost:8501`，即可开始对话。

---

## 功能演示示例

- `去上海出差，住宿费最多能报多少？` → 调用 `search_knowledge`，依据《差旅报销制度》回答并标注来源。
- `我累计工作 8 年，有几天年假？` → 调用 `calc_annual_leave`，返回 5 天。
- `帮我查一下工单 IT-1001 的进度` → 调用 `get_ticket_status`，返回状态/负责人/更新时间。
- 追问 `那 IT-1003 呢？` → 依靠多轮记忆理解"查工单"意图。

---

## 评测检索召回（推荐，能产出真实数据）

每次改完检索逻辑或知识库，用一个固定问题集回归，确认"该召回的文档块有没有进 Top-K"：

```powershell
.\.venv\Scripts\python.exe test\retrieval_eval.py
```

- 默认混合检索（`RETRIEVAL_KEYWORD_WEIGHT=0.8`）应输出 **`Top-4 召回命中率：9/9`**；
- 把 `.env` 里该项设为 `0` 即退回纯向量检索，可对比基线（**7/9**）：混合检索修复了纯向量检索下"住宿费金额块漏召回"的问题。

---

## 目录结构

```
office-agent/
├── app.py                     # Streamlit 界面入口
├── requirements.txt
├── .env.example
├── agent/
│   ├── config.py              # 配置（读取 .env）
│   ├── graph.py               # LangGraph ReAct 智能体
│   └── tools/
│       ├── knowledge.py       # 工具：RAG 知识库检索
│       └── business.py        # 工具：查工单、算年假
├── knowledge/
│   ├── vector_store.py        # 本地 Embedding + 向量存储/检索
│   ├── build_index.py         # 知识库构建脚本
│   └── docs/                  # 知识库种子文档（可替换为自己的业务文档）
├── data/
│   ├── tickets.json           # 模拟工单数据
│   └── vector_store.json      # 构建后生成的向量索引
├── test/
│   └── retrieval_eval.py      # 检索召回命中率评测（改检索/文档后一键回归）
└── docs/
    └── 架构流程图.html         # 系统架构图（ReAct + RAG 流程）
```

---

## 进阶：切换语义 Embedding / Chroma（可选）

默认的本地哈希 Embedding 胜在**零成本、离线、能讲清原理**，但语义理解有限。要获得更好的召回效果：

- **云端语义 Embedding**：在 `.env` 配置 `EMBEDDING_PROVIDER` 与对应 Key（如阿里云 DashScope 的 `text-embedding-v4`、硅基流动的 `bge-m3`），把 `HashingEmbeddings` 替换为调用该接口的实现（接口 `embed_documents / embed_query` 保持不变）。
- **Chroma 向量库**：`pip install langchain-chroma chromadb`，用 `langchain_chroma.Chroma` 替换 `VectorStore`，检索接口一致。

> 这是一个典型的技术取舍：本地轻量方案用于开发与原理验证，生产可平滑升级到语义 Embedding + 专业向量库。

---

## 常见报错排查

| 报错现象 | 原因与处理 |
|---|---|
| `未配置大模型 API Key` / `api_key ... must be set` | `.env` 不在项目根目录，或没填 `DEEPSEEK_API_KEY`；填好后重启 |
| `知识库为空` | 没运行构建脚本，执行 `python -m knowledge.build_index` |
| `streamlit 不是内部或外部命令` | 用 `python -m streamlit run app.py` 调用 |
| 模型不调用工具 | 确认用的是 `deepseek-chat`（支持 Function Calling），不要用 `deepseek-reasoner` |
| 请求超时 / 连接失败 | 检查网络；公司/校园网可能拦截，换网络或稍后重试 |
| Activate.ps1 无法加载 | 不用激活，直接用 `.\.venv\Scripts\python.exe` 运行即可 |

遇到报错时，先根据上表的现象定位；仍无法解决时，核对配置、网络环境后重试。
