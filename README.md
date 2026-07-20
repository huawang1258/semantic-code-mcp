# semantic-code-mcp

自建代码语义检索 MCP Server —— 自然语言查询 → 相关代码片段。

「效果优先」技术栈：**Tree-sitter AST 切分 + voyage-code-3 embedding（int8 量化）+ sqlite-vec 向量库 + BM25 全文 + RRF 融合 + Cohere rerank + Call Graph 图扩展**。

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
  Retriever ── 向量召回 + BM25召回 ── RRF融合 ── Cohere rerank ── Call Graph扩展 ── top_n 代码片段
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

```bash
pip install -r requirements.txt
```

## 配置

复制 `.env.example` 为 `.env` 并填入 key：

| 变量 | 必需 | 说明 |
|---|---|---|
| `VOYAGE_API_KEY` | 是 | Voyage embedding，申请：https://dash.voyageai.com/ |
| `COHERE_API_KEY` | 否 | Cohere rerank，不配则跳过重排仅用 RRF |

### 高级配置

| 变量 | 默认值 | 说明 |
|---|---|---|
| `SCM_EMBED_MODEL` | `voyage-code-3` | Embedding 模型 |
| `SCM_EMBED_DTYPE` | `float` | 向量精度：`float`（最高精度）/ `int8`（省 71% 存储） |
| `SCM_EMBED_MAX_RETRIES` | `4` | API 限流/错误时自动重试次数 |
| `SCM_EMBED_MIN_INTERVAL` | `0` | 请求最小间隔秒数（未绑卡 3 RPM 限制时设 `21`） |
| `SCM_EMBED_CONCURRENCY` | voyage `3` / local `1` | 索引时 embedding 并发请求数（实测 3 路≈提速 3 倍） |
| `SCM_EMBED_CHAR_BUDGET` | `200000` | 单批最大字符数（未绑卡时可调小） |
| `SCM_RERANK_MODEL` | `rerank-v3.5` | Rerank 模型 |
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
      "command": "python",
      "args": ["D:/project/main/MCP/semantic-code-mcp/server.py"],
      "env": {
        "VOYAGE_API_KEY": "your-voyage-key",
        "COHERE_API_KEY": "your-cohere-key",
        "HTTPS_PROXY": "http://127.0.0.1:7897"
      }
    }
  }
}
```

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
4. **检索**：查询同时走向量 KNN 与 BM25，用 RRF 融合排名，再用 Cohere rerank 精排出 top_n。
5. **图扩展**：主结果沿 call graph 扩展 1 跳，把调用者/被调用者连带召回（附 `relation` 标记）。
6. **实时监控**：watchdog 监控文件变更（2s debounce），**后台主动增量同步**（O(变更数)），查询时零等待；后台失败不丢变更，下次查询前兜底。
7. **LRU 管理**：多工作区并发（最多 8 个），超限自动淘汰最久未用的 workspace。

## 索引存储

每个工作区按目录路径哈希存为 `~/.semantic-code-mcp/<hash>.db`（可用 `SCM_DB_DIR` 改）。删除该目录即清空所有索引。

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
机械化打分可复现（期望文件 + must_contain 关键字），rerank 节流至试用限额内：

| 指标 | 成绩 | 参考（自托管 0.6B 方案公开数据） |
|---|---|---|
| Top-1 精确命中 | 84% | 32% |
| Recall@5 | 100% | 54% |
| 答案可用性 | 100% | 80% |
| MRR | 0.897 | — |

附加 **K 类私有业务仓库探针**（不计入主总分）：在真实业务仓库上验证复合意图拆分/
子查询分数混合/保底、trigram 中文召回、实现意图降权、`expand_graph=True` 全链路。
探针定义在 gitignore 的 `eval_local_probes.py`（`PROBE_TARGET` + `PROBE_GOLDEN`，
写法见 `eval_golden.py` 的 K 类加载段），不存在自动跳过（`--skip-java` 显式跳过）——
你可以对自己的仓库建一组同格式探针作为私有回归安全网。

复测：`python eval_golden.py --json`；回归测试：`python test_fixes.py`

### 工程边界（已知成本）

- **Cohere rerank 限速**：试用 key 10 次/分钟。密集调用（评测、批量脚本）设
  `SCM_RERANK_MIN_INTERVAL` 节流；生产建议换付费 key（429 自动退避重试兜底）。
- **复合 query 额外延迟**：中文双动作查询（"生成…并清理…"）每个子查询多发一次
  rerank（+200~400ms/次，最多 2 次）。仅复合措辞触发，普通查询零开销。
- **阈值标定范围**：子查询分数门槛 0.2、两段拆分等参数在中文 Java 仓库 + Go 仓库
  双评测集上标定；其它语言/仓库如表现异常，优先复核 `_COMPOUND_SUB_MIN_SCORE`
  （Cohere 中文短查询分数整体偏低，0.2 是"字面噪声/真业务"分界线）。

## 性能数据（42 块 / 9 文件）

| 指标 | float32 | int8 |
|---|---|---|
| 索引耗时 | 2.9s | 2.4s |
| DB 大小 | 4328 KB | 1252 KB（节省 71%） |
| 检索延迟 | 0.3–0.8s | 0.3–0.8s |
| Top5 一致率 | — | 95%（19/20） |
| Rerank 排序 | score 0.1–0.7 | — |

## Code Review 摘要

全部 ~1600 行核心代码（含 watcher/workspace），无严重 bug。架构分层清晰，每层职责单一：

- **降级策略完善**：AST 失败→行切分，rerank 失败→RRF，int8 不支持→float
- **增量一致性**：embedding 成功后才记录 hash，崩溃后下次自动重试
- **实时增量**：watchdog 文件监控 + dirty 增量同步（O(变更数)），解决大项目全量扫描延迟
- **资源管理**：LRU 淘汰超限工作区，watcher 随 workspace 生命周期自动清理
