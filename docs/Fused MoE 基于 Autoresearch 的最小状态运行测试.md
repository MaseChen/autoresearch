# Fused MoE 基于 Autoresearch 的最小状态运行测试

## 1. 执行摘要

项目进入 MetaX C500 上的自动算子迭代优化阶段。

**当前示例解读**

![image](https://origin.picgo.net/2026/07/30/Screenshot-2026-07-30-at-03.00.06658a465ace56b2d9.png)

以下是 2026-07-30 采样时正式 run 处于 `RUNNING` 的历史快照；该 run 后续已
结束，控制器和正式无人值守闭环均完成验证，当前 accepted kernel 为
`88b9eb6f612d…`。本版本还支持以独立 run 选择 V4 Pro 或 Flash。

- iteration 1 的模型输出格式不合规，在 Proposal 解析阶段被拒绝，没有生成
  候选，也没有接触 GPU。
- iteration 2 生成了候选，但其共享内存需求为 128 KiB，超过 C500 的
  64 KiB 上限；该结果被正确记为 `COMPILE_ERROR`，随后流水线继续运行。
- iteration 3 的候选已经通过 smoke 和 quick，当前正在执行
  `FULL_PRIMARY`，即与已接受基线进行四个生产规模 case 的交错性能对比。
- 当前使用了 2/5 个有效候选额度。iteration 1 的格式错误不是有效候选，
  因而不占候选额度。

这组结果说明控制闭环已生效：Agent 提出候选，可信控制器会拒绝格式错误，evaluator 会把不可编译配置作为科学失败返回，后续候选仍可继续；第一个通过小规模正确性门禁的候选已经进入性能主测。

## 2. 概况

### 2.1 算子背景

优化对象是运行在 MetaX C500 上的 Fused MoE W8A8 Triton kernel。

MoE（Mixture of Experts，混合专家）模型包含许多“专家”权重，但每个 token 只路由给少数专家。本项目固定 `topk=8`：一个 token 在逻辑上会产生 8 条专家路径。上游已经把输入展开并排成 routed-row 顺序，kernel 不需要再次根据 `token_ids` 搬运输入。

W8A8 表示激活 `a` 和专家权重 `b_col_major` 都是 INT8。INT8 乘法结果累加
到 INT32，再通过 FP32 scale 和 MoE 权重缩放，最终原地写入 BF16 输出。

对 routed row `r` 和输出列 `n`，目标语义为：

```text
expert(r) = expert_ids[r // 128]

out[r, n] =
    sum_k(a[r, k] * b_col_major[expert(r), n, k])
    * scale_a[r]
    * scale_b[expert(r), n]
    * moe_weights[r]
```

每连续 128 行构成一个 expert tile，同一个 tile 只能读取一个 expert ID。

### 2.2 固定候选接口

Agent 只能给出完整的 `kernel.py`，其中必须实现：

```python
def run_kernel(
    a, b_col_major, scale_a, scale_b, moe_weights,
    token_ids, expert_ids, topk, out,
) -> None:
    ...
```

Agent 可以优化 block size、program grid、加载顺序、warp 数和 pipeline stage 等实现细节，但不能改变输入输出语义、评测框架、数据、oracle、计时或晋级标准。

### 2.3 Full suite 的四个生产规模 case

| Case | EM | N | K | Experts | 路由 |
|---|---:|---:|---:|---:|---|
| `full_decode_gate_up` | 4,096 | 4,096 | 7,168 | 256 | uniform |
| `full_prefill_gate_up` | 32,768 | 4,096 | 7,168 | 256 | Zipf α=1.2 |
| `full_decode_down` | 4,096 | 7,168 | 2,048 | 256 | uniform |
| `full_prefill_down` | 32,768 | 7,168 | 2,048 | 256 | Zipf α=1.2 |

固定数据生成器估算每个 full case 的数据规模约为 3.63–7.75 GiB。full case 将逐个运行，不会同时把四个 full case 全部保留在内存中。

## 3. 系统总览

```text
OpenCode 1.17.7 + DeepSeek V4 Pro
无工具、无仓库、无 GPU
        │
        │ ProposalV1：假设、理由、父 hash、完整 kernel.py
        ▼
可信宿主控制器 kernel_research/autorun
配置校验、策略检查、预算、状态机、审计、checkpoint
        │
        │ 固定 argv + 只读候选文件 + accepted baseline
        ▼
C500 evaluator 容器
断网、单 GPU、只读框架、候选隔离缓存、分阶段验证
        │
        ├── smoke → quick → full primary → full confirmation
        │
        ├── History SQLite：实验和逐 case 证据
        └── SHA-256 artifacts：每一版候选源码
```



## 4. 模块分工

### 4.1 OpenCode + DeepSeek：无工具提案器

当前 Agent 组合为：

- 调用外壳：固定 OpenCode 1.17.7 容器。
- 模型：`deepseek/deepseek-v4-pro`。
- 推理设置：thinking enabled、`reasoningEffort=max`。
- 会话方式：每个 iteration 都启动全新、无状态、单 step 会话。

控制器给 Agent 的上下文包括 accepted kernel、`program.md`、当前 C500 环境、accepted 四个 p50、最近实验摘要以及失败反馈。Agent 只能返回严格的 `ProposalV1`：

```json
{
  "schema_version": 1,
  "parent_candidate_hash": "<accepted kernel SHA-256>",
  "hypothesis": "<单一可证伪假设>",
  "rationale": "<依据和预期影响>",
  "kernel_source": "<完整 kernel.py>"
}
```

提案器拥有访问 DeepSeek API 所需的网络，但没有：

- 仓库挂载；
- GPU 设备；
- Docker socket；
- controller/history 数据库；
- shell、读写、搜索、MCP、插件或子任务工具。

Agent 只能“提交答卷”，不能自己运行实验或修改可信状态。

### 4.2 可信宿主控制器

`kernel-autoresearch start` 启动的是可信 Python 控制器。它由服务器用户
`mx` 运行，是唯一允许调用 Docker 和更新实验状态的项目代码。

它负责：

1. 运行 doctor，核对 Git commit、kernel hash、镜像 digest、设备、SDK、
   driver 和 mcTriton compile probe。
2. 组装本轮 prompt，调用一个新的 OpenCode proposer 容器。
3. 解析 ProposalV1，拒绝额外文本、未知字段、父 hash 错误和重复候选。
4. 在受限子进程中进行 research policy 检查。
5. 按状态机顺序调用 evaluator。
6. 读取 evaluator 的版本化 JSON，写入 History SQLite。
7. 根据结果进入下一 stage、下一个候选或终止 run。
8. 记录 prompt、原始 NDJSON、stderr、候选、事件和停止原因。
9. 通过 checkpoint 备份数据库、artifact 和 SHA-256 manifest。

控制器不会把晋级候选复制到真实 `kernel.py`，也不会执行 Git commit/push。正式晋级后仍需要可信操作者检查 artifact，才会物化到仓库。

### 4.3 Research policy

候选源码最大 256 KiB。策略检查在短生命周期子进程中运行，限制为：

- CPU 时间 2 秒；
- 墙钟 5 秒；
- 内存 256 MiB；
- 输出 64 KiB；
- 最多 100 条策略错误。

检查内容包括固定函数签名、危险 import/builtin、文件/网络/进程访问、dunder、模块级副作用、B expert 基址 int64 数据流，以及 C500 的 `num_warps ∈ {1,2,4,8,16}` 约束。

可信控制器检查一次，evaluator 的 `evaluate-raw` 在导入候选前再检查一次。

### 4.4 C500 evaluator

evaluator 使用固定 digest 的厂商 Torch/mcTriton 镜像。容器：

- `--network none`；
- 非 root、只读根文件系统；
- `cap-drop ALL`、`no-new-privileges`；
- 只挂载只读框架快照、单个候选和 accepted baseline；
- 只透传获授权 GPU1 的三个设备节点；
- 看不到真实仓库、controller DB、history DB、DeepSeek key 或 Docker socket。

编译缓存按：

```text
evaluator image digest / framework commit / candidate hash
```

隔离。同一个候选可以在 smoke、quick、primary、confirmation 之间复用缓存，
不同候选不能共享可写缓存。

### 4.5 History、artifact 与 controller state

系统有两类可信记录：

- History SQLite：GPU 实验、逐 case 正确率、原始计时样本、p20/p50/p80、
  环境指纹和 promotion 结果。
- Controller SQLite：run、iteration、stage、Proposal、事件、容器名和停止
  原因。

候选源码按 SHA-256 内容寻址保存。即使候选失败，源码和证据也不会被覆盖。
checkpoint 使用 SQLite backup API，并对备份数据库执行 integrity check，
再用 manifest 校验文件数量和 SHA-256。
