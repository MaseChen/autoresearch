# Fused MoE Autoresearch 当前进度报告

> 2026-07-28｜组会一页版；2026-08-02 更新当前状态

## 当前结论

**算子评测基础设施、C500 accepted baseline、受控 Agent 控制层和正式
无人值守优化会话均已完成端到端验证。** 当前 accepted kernel 为
`88b9eb6f…`，并已具备按 run 配置选择 DeepSeek V4 Pro 或 Flash 的能力。

## 进度总览

| 工作项 | 状态 | 证据 |
|---|---|---|
| Fused MoE evaluator MVP | ✅ 完成 | mock/C500、oracle、历史、评分、CLI |
| Full shape int64 寻址 | ✅ 完成 | 四个 full case 不再出现 ATU/Xnack |
| 当前 accepted kernel | ✅ 完成 | hash `88b9eb6f…`，128×128 tile |
| 原服务器性能确认 | ✅ 完成 | 相对 BLOCK_K=32 确认 speedup 1.722× |
| 新服务器 C500 基线重建 | ✅ 完成 | doctor、smoke、quick、full 均成功，matched ratio 1.0 |
| 受控 Agent 控制层 | ✅ 代码完成 | OpenCode adapter、状态机、policy、raw evaluator、checkpoint |
| 安全加固与恢复机制 | ✅ 本地完成 | 输出限流、信号清理、缓存隔离、DB v2 与对账 |
| 本地自动化验证 | ✅ 通过 | 118 项测试；总 branch-aware coverage 83.31% |
| 远程部署控制层 | ✅ 完成 | 固定 proposer/evaluator digest 并通过 doctor |
| OpenCode + DeepSeek live proposal | ✅ 完成 | proposal-only 与格式重试均已验证 |
| 单候选 GPU dry run | ✅ 完成 | smoke/quick/full 与残留检查通过 |
| 正式 5 候选会话 | ✅ 已运行 | 已产生可审计的编译失败和 full 科学拒绝证据 |
| Pro / Flash 切换 | ✅ 本版本完成 | 配置级选择、run 内不可变、禁止静默回退 |

## 当前固定身份

```text
Git 分支：codex/fused-moe-autoresearch
控制器提交：部署时填写本版本 `git rev-parse HEAD`
当前 kernel：
88b9eb6f612dbe47e2e59498fd8c45df305e832524155dafb310cc98b26cf9b9

已验证 evaluator image：
registry.cn-shanghai.aliyuncs.com/kcr-3rd/kesci_kernel_lab
@sha256:5f1da890360acc5a81438d0e35a80079b34f30d1fa64fc954fef2e9a1ae45b64
```

新服务器此前记录的 C500 环境为 driver 3.3.12、MXMACA/Torch 3.5.3.9、
mcTriton 3.0.0。正式会话前必须重新运行 doctor，以当次环境指纹为准。

## 已经具备的闭环

1. Agent 根据 accepted kernel、实验摘要和失败反馈提出一个单一假设。
2. Agent 只返回严格 `ProposalV1`，不能查看或修改仓库。
3. 控制器做资源受限 policy 检查，重复候选不占 GPU。
4. 候选依次通过 smoke、quick、full primary。
5. 同一 hash 再做 full confirmation。
6. 正确性、整体提升和单 case 回退门槛全部满足才晋级。
7. 每次状态转换、提案、原始输出、结果和环境都可审计、恢复和 checkpoint。
8. 首次确认晋级或达到候选/时间/故障预算时自动停止。

## 本轮设计最重要的安全结论

- **Agent 没有系统工具。** 它看不到仓库、GPU、Docker socket 和数据库。
- **Agent 不直接运行候选。** 只有可信宿主控制器能以固定参数启动容器。
- **候选不接触可信状态。** evaluator 不挂载真实仓库、history 或 controller DB。
- **故障默认收紧。** 非法地址、ATU/Xnack、崩溃、超时、OOM 等终止会话，
  不进行无人值守重试。
- **密钥不进入 Agent 上下文。** 使用只读 mode 600 文件和日志脱敏。
- **缓存不跨候选。** 降低编译缓存污染对科学结论和后续候选的影响。
- **GPU 风险必须显式确认。** 未设置
  `acknowledge_gpu_passthrough_risk: true` 时禁止 GPU 评测。
- **不夸大隔离能力。** Docker 设备直通不能防御恶意 GPU kernel；真正的
  对抗性隔离仍需要 VM/IOMMU/VFIO 或独占主机。

## 下一步执行清单

- [ ] 在服务器拉取并核对 `11a5654`、分支、clean worktree 和 kernel hash。
- [ ] 备份现有 runtime/history，并用 controller doctor 验证 schema 与路径。
- [ ] 构建固定 OpenCode 1.17.7 proposer image，记录不可变 digest。
- [ ] 创建独立、限额的 DeepSeek key；文件 owner 正确且 mode 600。
- [ ] 生成服务器 config，先保持 GPU 风险确认关闭，完成 doctor。
- [ ] 运行一次 proposal-only，核对无仓库/GPU/socket 挂载和输出脱敏。
- [ ] 将候选上限设为 1，显式确认 GPU 风险，完成 smoke/quick/full dry run。
- [ ] 注入安全 compile failure、precision failure、输出超限和 SIGTERM。
- [ ] 核对无容器、进程、跨候选缓存和状态残留。
- [ ] 从 checkpoint 重建 controller/history/artifacts 决策链。
- [ ] 门禁全部通过后，启动最多 5 候选、6 小时、首次晋级即停的正式会话。

## 组会需要确认的事项

1. 是否接受当前威胁模型：Agent 非恶意，但候选可能错误；GPU 直通风险由授权
   操作者显式确认。
2. DeepSeek 专用 key 的消费上限和保管人。
3. proposer image 的镜像仓库与 digest 固定流程。
4. GPU1 的正式验收时间窗口，以及发生驱动故障时的人工联系人。

## 60 秒口头汇报

我们把 autoresearch 从训练脚本搜索改造成了 C500 Fused MoE Triton 算子
自优化系统。固定框架掌握数据、oracle、真实计时、历史和晋级，Agent 只提出
完整 kernel，不直接接触仓库或 GPU。工程上已经解决 full 大张量 int32 地址
溢出，并把 BLOCK_K 从 32 提升到 64，在原服务器确认获得约 1.72 倍整体加速；
新服务器也重新通过了 smoke、quick 和 full，当前 seed 正确率为 1.0。

现在控制层代码和本地 118 项测试已经完成，重点做了三层安全隔离：提案器无
工具无资产，可信控制器限制预算并审计状态，evaluator 断网、单 GPU、只读挂载，
候选前后两次策略检查，硬故障立即停机。同时我们明确承认 GPU 直通不是恶意
kernel 沙箱。远程 proposer、单候选 dry run 和正式 5 候选无人值守会话都已
执行；本版本进一步允许用独立新 run 对比 DeepSeek V4 Pro 与 Flash，同时在
网络失败时保持原模型失败关闭，不做不可审计的自动回退。
