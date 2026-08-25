import type { ConsoleRow, ConsoleSnapshot } from './types'

export const statusLabels: Record<string, string> = {
  STABLE: '稳定',
  CHANGING: '正在变化',
  RUNNING: '运行中',
  CREATED: '待启动',
  SUCCESS: '成功',
  SUCCEEDED: '成功',
  COMPLETED: '已完成',
  PROMOTED: '已晋级',
  STOPPED: '已停止',
  FAILED: '失败',
  KNOWN_FAILURE: '已知失败',
  HARD_FAILED: '严重失败',
  PAUSED_OPERATOR: '已由操作员暂停',
  PAUSED_DATA_INTEGRITY: '数据完整性异常，已暂停',
  PAUSED_HARD_FAILURE: '严重失败，已暂停',
  UNKNOWN: '结果未知',
  UNKNOWN_OUTCOME: '结果未知',
  UNKNOWN_GPU_OUTCOME: 'GPU 结果未知',
  PAUSED_UNKNOWN_OUTCOME: '结果未知，已隔离',
  UNAVAILABLE: '数据不可用',
  BUDGET_EXHAUSTED: '预算已耗尽',
  PROPOSAL_READY: '候选已就绪',
  ACTIVE: '使用中',
  RELEASED: '已释放',
  QUARANTINED: '已隔离',
  RESERVED: '已预留',
  SETTLED: '已结算',
  CANCELLED: '已取消',
  ABANDONED: '已终止并保留证据',
  PENDING: '等待执行',
  QUALIFIED: '已通过资格验证',
}

export const stageLabels: Record<string, string> = {
  PROPOSE: '生成候选',
  POLICY: '策略检查',
  SMOKE: '冒烟评测',
  QUICK: '快速评测',
  FULL_PRIMARY: '完整主评测',
  CONFIRMATION: '确认评测',
  BASELINE_QUALIFICATION: '基准资格验证',
  DONE: '生成证据',
  smoke: '冒烟评测',
  quick: '快速评测',
  full_primary: '完整主评测',
  confirmation: '确认评测',
  baseline_qualification: '基准资格验证',
}

export function statusLabel(value?: string): string {
  if (!value) return statusLabels.UNAVAILABLE
  return statusLabels[value] ?? value
}

export function stageLabel(value?: unknown): string {
  const key = String(value ?? '')
  return stageLabels[key] ?? (key || '等待开始')
}

export function shortIdentity(value: unknown, visible = 12): string {
  const text = String(value ?? '')
  if (!text) return '—'
  return text.length > visible + 4 ? `${text.slice(0, visible)}…` : text
}

export function formatTime(value: unknown): string {
  if (typeof value !== 'string' || !value) return '—'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return new Intl.DateTimeFormat('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  }).format(date)
}

export interface TaskSummary {
  id: string
  kind: 'LONG' | 'BENCHMARK' | 'RUN'
  title: string
  subtitle: string
  status: string
  updatedAt: unknown
  row: ConsoleRow
}

export function taskSummaries(snapshot: ConsoleSnapshot): TaskSummary[] {
  const childRunIds = new Set(
    snapshot.data.child_runs
      .map((row) => String(row.controller_run_id ?? ''))
      .filter(Boolean),
  )
  const campaigns = snapshot.data.campaigns.map((row): TaskSummary => {
    const benchmark = row.mode === 'BENCHMARK'
    return {
      id: String(row.id),
      kind: benchmark ? 'BENCHMARK' : 'LONG',
      title: benchmark ? '模型策略对照实验' : '长期持续优化',
      subtitle: benchmark ? '冻结 Pro / Flash 对照组' : '多轮候选探索与人工谱系复证',
      status: String(row.status ?? 'UNAVAILABLE'),
      updatedAt: row.updated_at ?? row.created_at,
      row,
    }
  })
  const runs = snapshot.data.runs
    .filter((row) => !childRunIds.has(String(row.id)))
    .map((row): TaskSummary => ({
      id: String(row.id),
      kind: 'RUN',
      title: String(row.id).startsWith('console-manual-') ? '已有代码评测' : '单次自主优化',
      subtitle: String(row.id).startsWith('console-manual-')
        ? '一个候选的完整 CURRENT 评测链'
        : '受信模型生成并评测候选',
      status: String(row.status ?? 'UNAVAILABLE'),
      updatedAt: row.updated_at ?? row.created_at,
      row,
    }))
  return [...campaigns, ...runs].sort((left, right) => (
    String(right.updatedAt ?? '').localeCompare(String(left.updatedAt ?? ''))
  ))
}
