# C500 服务器运维自动化

`kernel-autoresearch-admin` 是可信宿主操作者使用的独立入口。它不会进入
Agent prompt，不会暴露给 OpenCode，也不负责提案或 GPU 研究循环。其目标是把
每次 Git 更新后的重复人工操作固化为可审计、失败关闭的命令。

## 首次迁移

先确认服务器仓库位于目标分支、工作树完全干净，Pro/Flash 两份现有配置都能
被 `kernel-autoresearch doctor` 读取，并且两份配置只在 `opencode_model` 上
不同。随后执行：

```bash
cd /home/mx/workspace/autoresearch
python3 -m pip install --user --no-deps -e .

kernel-autoresearch-admin bootstrap \
  --manifest /home/mx/autoresearch-runtime/gpu1/admin.json \
  --pro-config /home/mx/autoresearch-runtime/gpu1/autorun.pro.json \
  --flash-config /home/mx/autoresearch-runtime/gpu1/autorun.flash.json

source /home/mx/autoresearch-runtime/gpu1/env.sh
kernel-autoresearch-admin verify \
  --manifest "$AUTORESEARCH_MANIFEST" \
  --level doctor
```

`bootstrap` 固定当前 branch、tracking upstream 和宿主 Python 绝对路径；生成
的 manifest、五份配置与 `env.sh` 均为 0600。`env.sh` 只含路径、commit 和
kernel hash，不包含 API key 内容。以后每个 tmux pane 只需要 source 该文件。

## 三档验证

```bash
kernel-autoresearch-admin verify --manifest "$AUTORESEARCH_MANIFEST" --level static
kernel-autoresearch-admin verify --manifest "$AUTORESEARCH_MANIFEST" --level cpu
kernel-autoresearch-admin verify --manifest "$AUTORESEARCH_MANIFEST" --level doctor
```

- `static` 检查 Git、baseline/History/artifact、配置关系、secret 元数据、镜像
  digest、本地镜像、设备节点、OpenCode 模型映射、GPU 锁和遗留容器。
- `cpu` 在固定 evaluator digest 内，以断网、无 GPU、只读仓库方式增加 Python
  3.10 全量测试。
- `doctor` 再分别执行 Pro/Flash controller doctor，要求 C500 compile probe
  为 `PASSED`。它不会调用 DeepSeek API。

每次命令输出 `schema_version: 1` 的 JSON，并在
`$AUTORESEARCH_RUNTIME/admin-reports/` 留下 mode 0600 报告。

## 后续 Git 更新

```bash
kernel-autoresearch-admin update \
  --manifest "$AUTORESEARCH_MANIFEST" \
  --doctor
```

更新器非阻塞获取 GPU1 flock；活动 run、遗留项目容器、dirty worktree、分支或
upstream 漂移都会立即拒绝。它执行固定 argv 的 fetch，证明目标是当前 HEAD 的
后代后才 `merge --ff-only`。Git 前进后由更新后源码的子进程执行 CPU 测试、
editable host 安装、临时配置同步和可选 doctor，最后才原子发布配置组。

如果 post-update 失败，工具不会 reset 或回滚 Git；正式配置仍固定旧 commit，
因此研究控制器会安全拒绝启动。修正临时问题后重复同一 `update` 即可继续完成。

## Baseline 采纳

普通 `sync` 与 `update` 永远不会自动改变 `expected_kernel_hash`。确认后的候选
必须先由人工审阅、写入真实 `kernel.py` 并提交，然后显式执行：

```bash
kernel-autoresearch-admin adopt-baseline \
  --manifest "$AUTORESEARCH_MANIFEST" \
  --candidate-hash 64位小写SHA256 \
  --doctor
```

只有当参数 hash 同时匹配工作树、HEAD Git blob、accepted C500 full
confirmation 和内容寻址 artifact 时才会发布新 pin。未确认实验、缺失 artifact、
dirty tree、活动 run 或任一 hash 不一致都会失败关闭。

## Flash 65K canary

本版本把 Pro/Flash 的项目侧 output token ceiling 都提高到 65,536，仍保持
thinking enabled、max reasoning、三步、20 分钟与 2 MiB 宿主输出限制。首次更新
后先验证 Flash 不再出现纯 reasoning 截断：

```bash
kernel-autoresearch start \
  --config "$AUTORESEARCH_FLASH_CONFIG" \
  --proposal-only \
  --format compact
```

期望状态为 `PROPOSAL_READY`，审计 NDJSON 中应出现 text event 且 output token
大于 0，不应产生 GPU history。若 65K 仍以 `reason=length` 结束，停止继续扩容；
保留 raw NDJSON，下一批单独评估 Flash 的 `reasoningEffort=high`。
