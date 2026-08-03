# Fused MoE Autoresearch 项目改造说明

> 组会版，更新于 2026-07-28

## 一句话概括

我们把原本面向语言模型训练的 `autoresearch`，改造成了一个面向 MetaX
C500 的 Fused MoE Triton 算子自优化平台：固定框架负责正确性、真实性能、
实验历史和晋级，外部 Agent 只提出完整的 `kernel.py` 候选，不能直接控制
仓库、GPU、Docker 或实验数据库。

## 改造前后

| 方面 | 原始 autoresearch 思路 | 当前项目 |
|---|---|---|
| 优化对象 | 训练代码和训练指标 | 单一 Fused MoE INT8 TN Triton kernel |
| 可变范围 | Agent 修改训练程序 | 候选只能替换 `kernel.py` |
| 正确性 | 训练结果间接反映 | NumPy INT32/FP32/BF16 oracle，逐 case 比对 |
| 性能证据 | 单次训练指标 | C500 原始计时样本、p20/p50/p80、交错基线 |
| 搜索方式 | 程序内模型或人工循环 | 外部 OpenCode + DeepSeek 无工具提案器 |
| 实验记录 | 文本日志为主 | SQLite WAL、源码 SHA-256、环境指纹和 checkpoint |
| 故障处理 | 普通进程运行 | worker、watchdog、容器、硬故障停止和可恢复状态机 |
| 安全边界 | 依赖操作者约束 | 提案器、可信控制器、离线 evaluator 三层隔离 |

## 改造后的系统

```text
OpenCode + DeepSeek
无工具、无仓库、无 GPU
        │ ProposalV1（完整候选源码）
        ▼
可信宿主控制器
策略检查、预算、状态机、审计、checkpoint
        │ 只读候选文件 + 固定参数
        ▼
C500 evaluator 容器
断网、单 GPU、只读框架、分阶段验证
        │
        ├── smoke → quick → full primary → full confirmation
        └── History SQLite + 内容寻址 artifacts
```

系统分为三层：

1. **固定评测框架 `kernel_research/`**
   提供确定性 case、NumPy oracle、mock/C500 backend、子进程故障隔离、
   评分、SQLite 历史和 CLI。mock 只验证控制面，不生成伪延迟，也永不晋级。

2. **唯一候选文件 `kernel.py`**
   接口固定为 `run_kernel(...)`。二维 grid 按 128 行 expert tile 和 N tile
   分工，INT8 dot 累加到 INT32，FP32 epilogue，原地写 BF16。Full shape 的
   B 权重线性偏移超过 int32，因此 expert 基址必须在乘 stride 前转为
   `tl.int64`。

3. **受控 Agent 控制层 `kernel_research/autorun/`**
   OpenCode 只输出严格的 `ProposalV1`；可信控制器执行
   `PROPOSE → POLICY → SMOKE → QUICK → FULL_PRIMARY → CONFIRMATION → DONE`。
   默认最多 5 个有效候选、6 小时、连续 3 次控制故障，首次确认晋级即停止。
   控制器不修改真实仓库、不提交 Git。

## 科学评测与晋级

- 固定随机种子、均匀与 Zipf 路由分布，并覆盖边界和异常输入。
- mock 不导入候选、不测真实性能、不允许晋级。
- C500 每个 case 先 warmup 10 次，再进行 3 轮 × 10 次测量，保留全部样本。
- 所有 case 的 matched ratio 必须不低于 0.99。
- Full 四个 case 等权计算归一化几何平均 speedup。
- 候选至少提升 1%，且任何 case 不得回退超过 3%。
- Primary 通过后，必须用完全相同的源码 hash 再做一次 confirmation。
- `CRASH`、`TIMEOUT`、`UNSUPPORTED_ENV`、ATU/Xnack、illegal address 或
  容器 exit 137 会终止整个无人值守会话，不自动重试。

## 已解决的关键技术问题

### 1. Full shape 非法地址

最初 seed 在 full suite 触发 C500 ATU/Xnack。根因是 B 权重最大线性偏移
达到约 75 亿个元素，`expert_id * stride_be` 的 int32 运算溢出。现在仅将
B expert 基址提升到 int64，在不扩大其他地址计算开销的前提下通过了四个
full case。

### 2. 第一个有效优化假设

在原 C500 环境中，我们把 `BLOCK_SIZE_K` 从 32 单变量提升到 64。确认实验
相对 BLOCK_K=32 基线的几何平均 speedup 为 **1.722×**，四个 case 的确认
speedup 分别约为 **1.658×、1.769×、1.651×、1.818×**，全部正确且无回退，
随后晋级为当前 seed：

```text
kernel SHA-256:
88b9eb6f612dbe47e2e59498fd8c45df305e832524155dafb310cc98b26cf9b9
```

上述 1.722× 是原服务器同环境交错基线结果。服务器资源丢失后，新服务器已
重新完成 doctor、smoke、quick 和 full；后续 128×128 tile 候选又通过
full primary 与 confirmation，晋级为当前 accepted baseline。不能把旧环境的
speedup 直接当作新环境的对比结果。

提案器现在允许在 run 配置中选择 `deepseek/deepseek-v4-pro` 或
`deepseek/deepseek-v4-flash`。模型是不可变实验身份，同一 run 的 resume 不得
切换；网络或 API 失败也不会自动回退到另一个模型。

## Agent 安全设计

### 提案器：不给工具和资产

- OpenCode 容器不挂载仓库、GPU 或 Docker socket。
- 只允许访问模型 API；关闭工具权限、插件、MCP、LSP、formatter、
  autoupdate、sharing 和 snapshot。
- 使用非 root、只读根、`cap-drop ALL`、`no-new-privileges`，并限制
  CPU、内存、PID、运行时间和输出大小。
- DeepSeek key 通过 mode 600 的只读文件注入，不进入环境变量、prompt 或
  日志；原始输出保存前执行脱敏。
- 每轮是全新无状态会话，Agent 只能返回一个严格 schema 的完整源码提案。

### 候选代码：GPU 前做两次约束检查

- 源码最大 256 KiB；策略解析在短生命周期子进程执行，限制 CPU 2 秒、
  墙钟 5 秒、内存 256 MiB、输出 64 KiB。
- 只允许 Torch/Triton import；拒绝危险 builtin、文件/进程/网络模块、
  dunder 访问和模块级副作用。
- 数据流检查要求 expert ID 在乘 `stride_be` 前转为 `tl.int64`，且乘积
  实际流向 B 的加载地址。
- 可信控制器检查一次，`evaluate-raw` 在导入候选前再次检查。
- 这些检查被明确称为 **research policy**，不冒充恶意 Python 安全沙箱。

### evaluator：最小化候选可见范围

- evaluator 断网、只读根、非 root、capabilities 全部移除。
- 只透传获授权的单张 GPU；不挂载仓库、controller 状态、history DB 或
  Docker socket。
- 固定框架和 candidate/baseline 都以单独只读文件挂载。
- 编译缓存按 `evaluator digest / framework commit / candidate hash`
  隔离，不同候选不能共享可写缓存。
- stdout/stderr 流式限流，超限立即精确 kill/rm 容器，不继续等待输出。

### 可信控制器：控制循环而不是让 Agent 控制系统

- 只有宿主控制器可调用 Docker；固定 argv、`shell=False`，候选内容不进入
  命令行。
- 容器使用完整 run ID 命名和标记，只清理本次 run 的精确对象。
- SIGINT/SIGTERM/SIGQUIT 会记录停止原因、更新 iteration，并执行清理；
  异常退出可由 `resume` 对账恢复。
- Controller DB 使用受约束状态枚举、事务化 transition + event；History
  DB 与 Controller DB 分离，并针对跨库提交窗口做幂等对账。
- checkpoint 使用 SQLite backup API，随后执行 integrity check、数量核对
  和 SHA-256 manifest 校验。
- `acknowledge_gpu_passthrough_risk` 必须显式为 true 才能启动 GPU 候选；
  false 时仍可 doctor 和 proposal-only。

## 安全边界与诚实声明

本设计能有效防止常见的 Agent 越界、误改仓库、泄露 key、污染实验历史、
跨候选缓存污染和失控进程，但它不是针对恶意 GPU kernel 的强隔离：

- Docker 用户组本身等价于高权限，Agent 因此永远不能接触 Docker socket。
- 直通 GPU 设备仍可能让恶意或严重错误的 kernel 影响驱动。
- 在没有专用 VM、IOMMU/VFIO 或独占物理机时，无法宣称对抗性 GPU 沙箱。
- 当前威胁模型是“Agent 可能犯错，但不是主动攻击者”；硬件级对抗隔离属于
  后续基础设施工作。

## 主要入口

- [README](../README.md)：安装、命令和服务器验收步骤
- [program.md](../program.md)：人工和 staged-candidate 科研协议
- [kernel.py](../kernel.py)：当前已验证候选
- [控制器](../kernel_research/autorun/controller.py)
- [运行时隔离](../kernel_research/autorun/runtime.py)
- [候选策略](../kernel_research/research_policy.py)
- [评分与晋级](../kernel_research/scoring.py)
