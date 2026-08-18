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
的 manifest、五份配置与 `env.sh` 均为 0600。`env.sh` 只含路径、deployment
commit、framework commit 和 kernel hash，不包含 API key 内容。以后每个 tmux
pane 只需要 source 该文件。

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

部署 Git 身份与 evaluator framework 身份永久分离。`expected_git_commit` 固定
完整工作树和 `kernel.py`；`framework_git_commit` 固定由 evaluator 容器实际挂载
的 `kernel_research` Git tree、编译缓存和 ExecutionEnvironment。普通 update/sync
保留现有 framework commit，不会把控制面更新伪装成新的科学环境。框架升级必须
另行重新验证 baseline，不得通过修改配置字段绕过。

如果 post-update 失败，工具不会 reset 或回滚 Git；正式配置仍固定旧 commit，
因此研究控制器会安全拒绝启动。修正临时问题后重复同一 `update` 即可继续完成。

## Baseline 采纳

普通 `sync` 与 `update` 永远不会自动改变 `expected_kernel_hash`。确认后的候选
必须先由人工审阅、写入真实 `kernel.py` 并提交，然后显式执行：

```bash
kernel-autoresearch-admin adopt-baseline \
  --manifest "$AUTORESEARCH_MANIFEST" \
  --candidate-hash 64位小写SHA256 \
  --namespace 精确的内置namespace_id \
  --doctor
```

只有当参数 hash 同时匹配工作树、HEAD Git blob、accepted C500 full
confirmation 和内容寻址 artifact 时才会发布新 pin。未确认实验、缺失 artifact、
dirty tree、活动 run 或任一 hash 不一致都会失败关闭。

从 V2 迁移到 V3 的旧实验没有可证明的执行环境，身份会保持
`LEGACY_UNKNOWN`。这类实验不能直接作为 V2 baseline 采纳依据，也不得通过修改
数据库或配置给旧证据补标签。若一个旧环境下已确认的候选仍要采纳，先在候选尚未
写入 Git 时执行一次受信重新资格评测：

```bash
kernel-autoresearch-admin requalify-adoption \
  --manifest "$AUTORESEARCH_MANIFEST" \
  --candidate-hash 64位小写SHA256 \
  --candidate-path /绝对路径/候选.py \
  --namespace 精确的legacy内置namespace_id
```

该命令先在当前固定镜像、framework、toolchain、ABI 和 build flags 下，将已部署
baseline 与自身做一次完整 full 评测，生成新的 resolved baseline seed；随后让指定
候选依次经过 POLICY、SMOKE、QUICK、FULL_PRIMARY 和 CONFIRMATION。它持有 Admin
maintenance fence 与 GPU1 锁，拒绝活动 Campaign/Run、候选字节漂移和非 Docker
Evaluator。每次外部动作均先写 Controller intent；GPU 启动后没有完整持久证据时
结果固定为 `UNKNOWN_OUTCOME`，不会自动重试。已有完整证据时只按 UID 幂等对账。

`requalify-adoption` 只生成新证据，不修改 `kernel.py`、Git、正式配置或
`deployment-baseline.json`。只有它返回 `PROMOTED`、新 Run checkpoint 已验证且人工
审阅仍通过后，才把同一候选字节提交为唯一的 `kernel.py` 变更，再运行上面的严格
`adopt-baseline`。旧 primary/confirmation 继续只读保留，不会被重写。

首次采纳还要求候选 commit 是当前已部署 commit 的唯一直接子提交，并且 Git diff
只包含 `kernel.py`。候选 commit 不改变 `framework_git_commit`；evaluator framework
从该冻结 commit 的 Git blobs 重新物化，而不是从当前工作树复制。这使实际顺序
`评测未提交候选 → 人工审查 → 仅提交 kernel.py → adopt` 保持同一科学环境，同时
继续拒绝夹带框架、配置或文档改动的候选提交。

## CURRENT baseline bootstrap

完成一次 legacy→resolved deployment adoption 后，不得把已有 LEGACY History
记录改写或重新标记为 CURRENT。先通过上面的 `update --doctor` 部署包含 ADR-008
能力的独立控制面提交，确认工作树干净、正式配置已固定该提交，并再次执行
`verify --level static` 与 `verify --level doctor`。随后才运行：

```bash
kernel-autoresearch-admin bootstrap-current-baseline \
  --manifest "$AUTORESEARCH_MANIFEST" \
  --candidate-hash 64位小写的当前kernel.py_SHA256
```

该入口没有候选路径、namespace、baseline、protocol、image 或 timeout 参数。它只
接受正式仓库中已部署的 `kernel.py`，并从 resolved LEGACY deployment pin 精确导出
不可变的父候选。父候选先在内置 CURRENT protocol 下做 full qualification，当前
部署候选再依次执行 SMOKE、八个 QUICK（含 shadow/holdout）、FULL_PRIMARY 和
CONFIRMATION。五个 GPU action 均先写 Controller intent；未知结果不可自动重放。

成功输出必须为 `PROMOTED`，并回显精确内置 `namespace_id`、protocol、run ID、
qualification ID 和 resolved environment。使用产生该 Run 的 Pro 配置创建 checkpoint，
再以 `verify_checkpoint()` 完整验证。确认 History/Controller 五个 attempt 均成功链接、
LEGACY 证据与旧 deployment pin 均未变化后，才允许人工执行：

```bash
kernel-autoresearch-admin adopt-baseline \
  --manifest "$AUTORESEARCH_MANIFEST" \
  --candidate-hash 64位小写的当前kernel.py_SHA256 \
  --namespace bootstrap输出的精确namespace_id \
  --doctor
```

bootstrap 本身只产生证据，不修改 Git、配置或 deployment pin。上述人工 adoption
允许在同一个已部署控制面 commit 上把 pin 从 LEGACY 切换到 CURRENT，因为
`kernel.py` 字节未变化；不得为此重建或 amend 已部署提交。CURRENT pin 完整验证和
归档完成前，不得开始 noise、Campaign soak 或 profiling collection。

## MetaX bounded profiler 镜像

Profiler 使用独立私有镜像，不得在服务器上把 evaluator tag 临时当作 profiler。
提交 A 的 activation profile 固定为 `active=false`；它只提供可审核的
Dockerfile、worker、host 接口和测试。build profile 不含 activation 位或最终镜像
RepoDigest，因此提交 A 构建的 worker 在提交 B 激活后仍能回显同一个 build digest。
已完成的 A4 构建身份为：

- source commit：`92469041b4786d970ef93c05c14ef09ece8628a2`
- image ID：`sha256:d6b214ade63ba37db7bea8c97f2d2e5c296178ce444654fc8af14566b99800e0`
- RepoDigest：`ghcr.io/masechen/autoresearch-metax-profiler@sha256:9d5516991a89945e7ad008c33ee847667831f5f7fc39fccf7030745f4ffa9acb`
- build profile digest：`sha256:a122359bc9c7d13356587964f51a4bdb68676f0841856d005cb380f1bd5facc7`
- activation profile digest：`sha256:bfd57223ce8bac17fdb38c63c29df8400b9bf3dc8f3d280530f62542b8435e13`
- 服务器证据：`/home/mx/autoresearch-evidence/profiler-image-a4-20260817T080230Z`
- 最终证据清单 SHA-256：`7013e3d4ab34468f4fe8b6eee164e78206e379f4bc0ff948bbed937cf480a1e8`

该证据包的最终 `sha256sum -c SHA256SUMS` 为成功；旧清单作为
`SHA256SUMS.pre-quiescence` 保留。以下命令是可重复构建记录，不应在部署 Submit B
时重新构建镜像：

```bash
export SOURCE_REVISION="$(git rev-parse HEAD)"
export PROFILER_LOCAL_TAG="ghcr.io/masechen/autoresearch-metax-profiler:${SOURCE_REVISION}"

PYTHONPATH=. python -c '
from kernel_research.profiler_contract import PROFILER_BUILD_PROFILE_DIGEST
print(PROFILER_BUILD_PROFILE_DIGEST)
' | tee profiler-build-profile-digest.txt

DOCKER_BUILDKIT=0 docker build \
  --platform linux/amd64 \
  --network=none \
  --pull=false \
  --no-cache \
  --build-arg "SOURCE_REVISION=$SOURCE_REVISION" \
  --file containers/profiler/Dockerfile \
  --tag "$PROFILER_LOCAL_TAG" \
  .

docker image inspect "$PROFILER_LOCAL_TAG" >profiler-image-local-inspect.json
```

本地 native build 阶段不得登录或推送 GHCR。必须先验证 image platform、OCI
revision、`USER 1000:1000`、entrypoint、`WORKDIR /output`，容器内 entrypoint 的
mode/owner 必须为 `555 0:0`，并以 `--pull=never` 运行 `verify-toolchain`。其 worker
runtime 必须精确使用 `--user 1000:1000`，并挂载
`/tmp:rw,nosuid,nodev,size=64m,mode=700,uid=1000,gid=1000`。build digest 必须为
A4 冻结值
`sha256:a122359bc9c7d13356587964f51a4bdb68676f0841856d005cb380f1bd5facc7`。

`SOURCE_REVISION` 必须是独立的 runtime-identity 修正提交 A4；不得从已知身份耦合
错误的 `e19cecc`、classic builder 无法解析的 A2 `559e8e9`，或仅完成root构建探针
但runtime资格失败的 A3 `1339fcc` 构建，也不得 amend任何旧提交。构建前确认该提交
不修改 `kernel.py`，且工作树干净。A4 Dockerfile 使用普通 `COPY`，随后执行
`chmod 0555`并验证`555 0:0`；toolchain probe必须位于`USER 1000:1000`之后，并先
证明当前euid/egid和私有HOME均正确。任何验证失败均停止构建。

把 build profile digest、build 输出的 RepoDigest、source commit、base RepoDigest、
image ID、worker revision、mcTracer 与两个 MetaX 库 hash 归档。提交 B 只能把该精确
`ghcr.io/masechen/autoresearch-metax-profiler@sha256:...` 写入轻量 contract 并将
`active=true`；不得同时修改 build profile 的任何字段、worker、recipe 或
`kernel.py`。提交 B 的测试必须证明 build digest 与提交 A 归档值完全相同，同时
activation digest 已绑定最终 RepoDigest。worker 输出只允许回显 build digest；镜像
RepoDigest 由宿主的 activation profile 和精确 Docker argv 验证。

只有本地检查全部通过且操作员批准约 12.71 GiB 上传后，才使用临时 write token
执行 `docker push`，从成功输出解析唯一 registry digest，并按该 digest 再次 pull、
inspect 和运行 `verify-toolchain`。然后立即 logout 并删除临时 `DOCKER_CONFIG`。
凭证阶段必须独立开始，且使用受限目录：

```bash
export DOCKER_CONFIG="$(mktemp -d /tmp/autoresearch-ghcr.XXXXXX)"
chmod 0700 "$DOCKER_CONFIG"

printf '%s' "$GHCR_WRITE_TOKEN" | docker login ghcr.io \
  --username masechen --password-stdin

docker push "$PROFILER_LOCAL_TAG"

docker logout ghcr.io
rm -rf "$DOCKER_CONFIG"
unset DOCKER_CONFIG GHCR_WRITE_TOKEN
```

服务器后续运行仅使用临时只读凭证拉取精确 digest。随后通过
`kernel-autoresearch-admin update --doctor` 部署提交 B，
完成 static、doctor 和完整主机回归。运行唯一公开 canary 入口：

```bash
kernel-research profile image-doctor \
  --config "$AUTORESEARCH_PRO_CONFIG" \
  --database "$AUTORESEARCH_RUNTIME/campaign/campaign.sqlite3" \
  --campaign-id "profile-image-canary-$(date -u +%Y%m%dT%H%M%SZ)"
```

该命令不接受 image、candidate、case、device、timeout 或 mcTracer 参数。它创建一个
无 child-run 的 CURRENT Campaign，固定预留 1800 秒 wall 与 900 秒 GPU，按
maintenance fence → `gpu1.lock` → lease 的顺序执行 compile manifest 和固定
`quick_decode_gate_up` mctx。成功 Campaign 自动结束；原始 trace 只进入私有 CAS，
两个 recipe 的 raw trace 合计不得超过 64 MiB，白名单摘要不具 promotion 或
baseline 权限。

UNKNOWN 或 hard failure 会保留 reservation 并 quarantine；不得重放。新版宿主会在
每个 recipe 启动前记录不可变 intent，并在临时目录清理前把限长 stdout、stderr、
`outcome.json`、已有 sentinel 和文件清单写入 controller 私有 CAS；Campaign status
输出会列出 intent、诊断对象和安全分类。修复若改变 worker、recipe 或镜像，必须产生
新镜像和新激活提交；仅补宿主诊断或受信恢复边界时不得重建同一个已验证镜像。

对已经进入 `PAUSED_UNKNOWN_OUTCOME` 或 `PAUSED_HARD_FAILURE` 的 image-doctor，先部署
包含恢复能力的代码会被活动 Campaign 管理锁拒绝。这种情况下只能从该修复提交的干净
detached recovery worktree 运行以下唯一入口，仍然读取正式配置和 canonical Campaign
数据库：

```bash
PYTHONPATH="$RECOVERY_WORKTREE" "$HOST_PYTHON" -m kernel_research \
  profile image-doctor-abandon \
  --config "$AUTORESEARCH_PRO_CONFIG" \
  --database "$AUTORESEARCH_RUNTIME/campaign/campaign.sqlite3" \
  --campaign-id "profile-image-canary-20260818T030324Z"
```

该入口不接受 doctor digest、资源、image、case、timeout 或 reason。它在 maintenance
fence 和 `gpu1.lock` 内运行一次新的受信 C500 doctor，精确绑定旧 quarantine fence，
随后把旧 Campaign 终态化为 `CANCELLED` 并将旧 lease 标为 `RELEASED`。它不会重放旧
action，不会结算或删除原 `RESERVED` 预算，也不会写 History、promotion、baseline、
Git 或部署 pin。旧版 canary 若尚无诊断对象，会显式记录
`diagnostic_unavailable=true`，不得伪造失败原因。

操作完成后，先归档 Campaign status、attempt/diagnostic 列表、abandonment、doctor、
lease 和 budget action，再走正常 Admin update 部署控制面修复并完成 static、doctor 和
完整主机回归。下一次 image-doctor 必须使用全新的 Campaign ID。Soak 只会在上述完整
不可变证明存在时忽略旧 canary 保留下来的 reservation；证明缺失或被修改会使观测
UNAVAILABLE。新 canary READY 后才能创建全新 Discovery Campaign，从零累计
24/72/168 小时 soak。正式 `profile collect` 仍须等三阶段全部合格后执行；所有数值
counter 可以是 `UNAVAILABLE/COUNTER_NOT_EXPOSED`，不得填零。

## Flash 384K/max canary

Pro/Flash 的模型声明 output 和 OpenCode 请求 cap 均固定为 384,000，thinking 均
保持 enabled，reasoning effort 均为 max。OpenCode 1.17.7 单独读取
`limit.output` 仍会把请求截在 32K，因此可信控制器会向 proposer 容器显式注入
`OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX=384000`。该值不能由普通配置或宿主同名
环境变量覆盖。OpenCode 三步、20 分钟与 2 MiB 宿主输出限制保持不变。更新后应
连续运行五次全新 proposal-only：

```bash
kernel-autoresearch start \
  --config "$AUTORESEARCH_FLASH_CONFIG" \
  --proposal-only \
  --format compact
```

至少四次应达到 `PROPOSAL_READY`。doctor/compact 输出应显示 Flash/max、声明 output
384000 和请求 cap 384000；生成的 `opencode.json` 应显示
`reasoningEffort=max`、`limit.output=384000`。不应产生 GPU history 或遗留容器。
若仍精确在 32K 结束，检查 Docker argv 中的实验变量；若 token 已超过 32K 后发生
20 分钟 timeout，说明 cap 已解除，应另行评估超时；若精确耗尽 384K，则停止扩容。
任何情形都不得自动关闭 thinking、切换模型或 fallback。

正常 `reason=stop` 的裸JSON若仅缺最外层最后一个 `}`，或在完整对象后精确多出
`"}`，控制器会执行唯一允许的尾部结构恢复。原始NDJSON不会被覆盖，并会增加
`PROPOSER_TRANSPORT_RECOVERY`审计事件。统计canary时应区分strict success与
transport recovery；其他JSON错误、fence或`reason=length`仍必须失败关闭。
