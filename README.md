# semantic-code-mcp

自建代码语义检索 MCP Server —— 自然语言查询 → 相关代码片段。

「效果优先」技术栈：**Tree-sitter AST 切分 + voyage-code-3 embedding（int8 量化）+ sqlite-vec 向量库 + BM25 全文 + RRF 融合 + Voyage rerank-2.5 重排 + Call Graph 图扩展**。一个 `VOYAGE_API_KEY` 跑通 embedding + rerank 全链路。

工程特性：**watchdog 文件实时监控 + 多工作区 LRU 管理 + MCP progress notification + cancel signal**。

## 架构

```
codebase_search(query, dir)
        │
        ▼
  WorkspaceManager ── LRU(max 8) ── FileWatcher(watchdog + 2s debounce)
        │
        ▼
  Indexer ── 全量扫描/watcher增量(O(变更数)) ── Chunker(Tree-sitter AST) ── Embedder(voyage-code-3)
        │                                                                        │
        │                                                          Store(sqlite-vec + FTS5 + 文件hash)
        ▼
  Retriever ── 向量召回 + BM25召回 ── RRF融合 ── rerank（Voyage/Cohere）── Call Graph扩展 ── top_n 代码片段
        │
        ▼
  MCP Server ── async tools + progress notification + cancel signal
```

| 模块 | 职责 |
|---|---|
| `chunker.py` | Tree-sitter 按函数/类切分（16 语言 + md/yaml/json 等文档配置按行切），提取调用关系构建 call graph |
| `store.py` | sqlite-vec 向量（float/int8）+ FTS5 全文 + 文件指纹 + call graph 边表 |
| `embedder.py` | voyage-code-3，document/query 双模式，int8 量化，限流重试 |
| `retriever.py` | 向量 + BM25 → RRF → rerank → 意图感知打分（测试/文档/调用链意图）→ 文件多样性截断 → 图扩展 |
| `indexer.py` | 目录扫描 + 文件 hash 增量同步 + watcher dirty 增量同步 |
| `watcher.py` | watchdog 文件监控 + 2s debounce + 50 变更立即 flush |
| `workspace.py` | 多工作区 LRU 管理器（创建/淘汰/状态追踪） |
| `server.py` | MCP 封装，async 工具 + progress + cancel + WorkspaceManager |

## 安装

发布包名为 `semcode-mcp`（PyPI 上 `semantic-code-mcp` 已被同名项目占用）：

```bash
# 方式一（推荐）：uv 环境零安装，MCP 客户端配置里直接 uvx 启动
uvx semcode-mcp

# 方式二：npx（要求已装 uv，该 npm 包是透传 uvx 的薄壳）
npx -y semcode-mcp

# 方式三：pip
pip install semcode-mcp

# 源码开发
pip install -e .
```

## 配置

复制 `.env.example` 为 `.env` 并填入 key：

| 变量 | 必需 | 说明 |
|---|---|---|
| `VOYAGE_API_KEY` | 是 | Voyage embedding + rerank（同一个 key 双用），申请：https://dash.voyageai.com/ |
| `COHERE_API_KEY` | 否 | 可选 rerank 后备后端；默认用 Voyage rerank-2.5，无需此 key |

### 高级配置

| 变量 | 默认值 | 说明 |
|---|---|---|
| `SCM_EMBED_MODEL` | `voyage-code-3` | Embedding 模型 |
| `SCM_EMBED_DTYPE` | `float` | 向量精度：`float`（最高精度）/ `int8`（省 71% 存储） |
| `SCM_EMBED_MAX_RETRIES` | `4` | API 限流/错误时自动重试次数 |
| `SCM_EMBED_MIN_INTERVAL` | `0` | 请求最小间隔秒数（未绑卡 3 RPM 限制时设 `21`） |
| `SCM_EMBED_CONCURRENCY` | voyage `3` / local `1` | 索引时 embedding 并发请求数（实测 3 路≈提速 3 倍） |
| `SCM_EMBED_CHAR_BUDGET` | `200000` | 单批最大字符数（未绑卡时可调小） |
| `SCM_RERANK_BACKEND` | `auto` | rerank 后端：`auto`（有 VOYAGE key 用 Voyage，其次 Cohere）/ `voyage` / `cohere` / `off` |
| `SCM_RERANK_MODEL` | 按后端 | voyage → `rerank-2.5`，cohere → `rerank-v3.5` |
| `SCM_DB_DIR` | `~/.semantic-code-mcp` | 索引数据库存放目录 |
| `SCM_TOP_K` | `50` | 向量召回候选数 |
| `SCM_TOP_N` | `10` | 最终返回结果数 |
| `HTTPS_PROXY` | — | Cohere API 代理（国内需配） |
| `SCM_MAX_WORKSPACES` | `8` | 最大同时管理的工作区数（LRU 淘汰） |
| `SCM_WATCH` | `1` | 文件监控：`1` 启用（默认），`0` 禁用 |

## 在 MCP 客户端注册

```json
{
  "mcpServers": {
    "semantic-code": {
      "command": "uvx",
      "args": ["semcode-mcp"],
      "env": {
        "VOYAGE_API_KEY": "your-voyage-key"
      }
    }
  }
}
```

npx 用法把 `command/args` 换成 `"npx", ["-y", "semcode-mcp"]`；
源码开发用 `pip install -e .` 后配 `"semcode-mcp", []`（包采用相对 import，
不支持 `python server.py` 直接跑，需用 console script 或 `python -m semantic_code_mcp.server`）。

## 工具

### codebase_search

| 参数 | 类型 | 说明 |
|---|---|---|
| `information_request` | string | 自然语言查询，如"用户登录鉴权逻辑在哪里" |
| `directory_path` | string | 代码库绝对路径 |

唯一工具，首次调用自动全量索引，之后 watcher 驱动增量同步（O(变更数)，极速）。支持 MCP progress notification 实时回报索引进度。需要强制重建索引时删除 `~/.semantic-code-mcp/` 下对应 `.db` 文件即可。

## 工作原理

1. **切分**：Tree-sitter 解析 AST，按函数/类边界切块，保留 `file_path / symbol / 行号` 元数据。
2. **向量化**：voyage-code-3 对代码块生成 1024 维向量（`input_type=document`）。
3. **存储**：向量入 sqlite-vec（vec0 虚拟表），同时入 FTS5 全文索引；`blob_hash = SHA256(path+code)` 实现内容寻址去重。
4. **检索**：查询同时走向量 KNN 与 BM25，用 RRF 融合排名，再用 rerank（默认 Voyage rerank-2.5）精排出 top_n。
5. **图扩展**：主结果沿 call graph 扩展 1 跳，把调用者/被调用者连带召回（附 `relation` 标记）。
6. **实时监控**：watchdog 监控文件变更（2s debounce），**后台主动增量同步**（O(变更数)），查询时零等待；后台失败不丢变更，下次查询前兜底。
7. **LRU 管理**：多工作区并发（最多 8 个），超限自动淘汰最久未用的 workspace。

## 索引存储

每个工作区按目录路径哈希存为 `~/.semantic-code-mcp/<hash>.db`（可用 `SCM_DB_DIR` 改）。删除该目录即清空所有索引。

## 隐私与数据边界

本工具的完整数据流向，用于评估能否用于敏感/闭源仓库：

### 会出网的数据（默认配置）

| 接收方 | 发送内容 | 时机 |
|---|---|---|
| Voyage AI | 代码块全文（embedding + rerank 候选）+ 查询文本 | 索引时 + 每次检索 |
| Cohere（可选后端） | 查询文本 + top-50 候选代码块全文 | 仅 `SCM_RERANK_BACKEND=cohere` 时；默认零发送 |
| 自配 LLM 端点（可选，默认关） | 仅查询文本（不含代码） | `SCM_QUERY_EXPANSION=on` 时 |

### ⚠️ Voyage 数据条款（重要，索引闭源代码前必读）

按 Voyage [TOS](https://www.voyageai.com/tos) Section 3(iii)（2026-07 复核）：

- **默认（opt-out 前）提交内容可被 Voyage 用于训练和改进模型**
- Opt-out 需要：账户绑定支付方式 + 组织 Admin 身份，dashboard → Terms of Service → 底部 toggle 切到 Opted Out
- Opt-out 后为 zero-day retention（处理完即删），但**不追溯**之前提交的数据，且免费额度可能被作废
- Batch API 上传的文件会保留 30 天（本工具不使用 Batch API）

**用于公司/闭源代码前，先完成 opt-out 或使用全本地模式。** Cohere（可选 rerank 后端）条款请自行评估。

### 始终留在本地的

- 索引库全部落盘本机 `~/.semantic-code-mcp/`（向量 + 代码原文 + call graph），不上传任何服务
- 无遥测、无统计上报，除上表 API 外零网络请求

### 索引范围控制

- 全层级遵守 `.gitignore`（含否定模式）：`.env`、构建产物等被忽略的文件不会进索引
- 例外：`AGENTS.md`/`.windsurfrules` 等 agent 指导文件即使被 gitignore 也会索引（有意设计，它们对检索架构问题价值极高）
- ⚠️ **硬编码在代码里的密钥会随代码块发往上表供应商**，敏感仓库索引前请先清理或用全本地模式

### 全本地模式（代码零出网）

```bash
pip install sentence-transformers
SCM_EMBED_BACKEND=local        # 本地 embedding（默认 Qwen/Qwen3-Embedding-0.6B）
# 不配 COHERE_API_KEY         # rerank 自动跳过，用 RRF 融合排序
```

检索质量会下降（参考公开评测中自托管 0.6B 方案的差距），但数据完全不离开本机。

## 路线

- **阶段 1** ✅：Tree-sitter + voyage-code-3 + hybrid + rerank —— 第一梯队 RAG
- **阶段 2** ✅：Call Graph 图扩展（caller/callee 连带召回）+ int8 量化向量（存储省 71%，一致率 95%）
- **阶段 3** ✅：工程化 —— watchdog 文件监控 + 多工作区 LRU + progress + cancel
- **阶段 4** ✅（部分）：意图感知排序（调用链直查 edges / 测试意图 / 文档意图）+ 文档配置索引（AGENTS.md 豁免 gitignore）+ 50 条分类评测集
- **阶段 5** ✅：单工具化 + 后台主动增量同步 + 分阶段进度条日志 + Java 调用链（接收者边/构造类型名/泛型剥离）+ 配置意图（yaml 反转 boost）
  + 并发 embedding 流水线（worker 只跑网络调用，sqlite 写回留在调用线程，顺序写回；CLIProxyAPI 9067 块全量 77s，≈串行 3 倍）
- **HyDE 已评测，结论默认关**：Top-1 86%→82%、延迟 2.9s→10.4s、Recall@5 98%→100%；架构类 +40 分但配置/SDK 类 -40（变体稀释精确标识符信号）。
  改进线：选择性扩展——仅对不含精确标识符的模糊查询开 HyDE
- **后续候选**：选择性 HyDE、本地后端对照评测、超大仓库 ANN 索引

## 评测（eval_golden.py，CLIProxyAPI 778 文件 / 9067 块）

50 条 query × 10 类（符号定位/调用链/架构职责/配置schema/同名干扰/SDK边界/协议契约/测试/中文/细节），
机械化打分可复现（期望文件 + must_contain 关键字）：

| 指标 | Voyage rerank-2.5（默认） | Cohere v3.5 基线 | 参考（自托管 0.6B 方案公开数据） |
|---|---|---|---|
| Top-1 精确命中 | **88%** | 84% | 32% |
| Recall@5 | 98% | 100% | 54% |
| 答案可用性 | 100% | 100% | 80% |
| MRR | **0.916** | 0.897 | — |
| 平均延迟 | **0.96s** | 6.2s（试用 key 节流） | — |

附加 **K 类私有业务仓库探针**（不计入主总分）：在真实业务仓库上验证复合意图拆分/
子查询分数混合/保底、trigram 中文召回、实现意图降权、`expand_graph=True` 全链路。
探针定义在 gitignore 的 `eval_local_probes.py`（`PROBE_TARGET` + `PROBE_GOLDEN`，
写法见 `eval_golden.py` 的 K 类加载段），不存在自动跳过（`--skip-java` 显式跳过）——
你可以对自己的仓库建一组同格式探针作为私有回归安全网。

复测：`python eval_golden.py --json`；回归测试：`python test_fixes.py`

### 模型选型说明

- **embedding 为什么是 voyage-code-3 而非 Voyage 4 系列**：4 系列（2026-01）是通用文本模型，
  **没有 voyage-code-4**；官方文档代码检索场景仍推荐 code-3。想试通用旗舰可
  `SCM_EMBED_MODEL=voyage-4-large`（需重建索引），与 code-3 的对比评测列入后续。
- **rerank 为什么默认 Voyage**：与 embedding 同一个 key 零额外配置；rerank-2.5 公开基准
  优于 Cohere v3.5（官方数据 +7.94%，本项目评测 Top-1 +4）；免费额度 200M token 且无
  试用 key 10 次/分钟限速（评测提速 6.5×）。Cohere 保留为可选后端。

### 工程边界（已知成本）

- **rerank 限速**：未绑卡/试用 key 限速低（Cohere 试用 10 次/分钟）。密集调用设
  `SCM_RERANK_MIN_INTERVAL` 节流；429 自动退避重试兜底。
- **复合 query 额外延迟**：中文双动作查询（"生成…并清理…"）每个子查询多发一次
  rerank（+200~400ms/次，最多 2 次）。仅复合措辞触发，普通查询零开销。
- **阈值标定范围**：子查询分数门槛 0.2、两段拆分等参数按 **Cohere 分数分布**在中文
  Java 仓库 + Go 仓库双评测集上标定；切换 rerank 后端（分数分布不同）或其它语言/
  仓库表现异常时，优先复核 `_COMPOUND_SUB_MIN_SCORE`。

## 性能数据（42 块 / 9 文件）

| 指标 | float32 | int8 |
|---|---|---|
| 索引耗时 | 2.9s | 2.4s |
| DB 大小 | 4328 KB | 1252 KB（节省 71%） |
| 检索延迟 | 0.3–0.8s | 0.3–0.8s |
| Top5 一致率 | — | 95%（19/20） |
| Rerank 排序 | score 0.1–0.7 | — |
