import type { ConsoleRow, ConsoleSnapshot } from './types'

export const statusLabels: Record<string, string> = {
  STABLE: '稳定', CHANGING: '正在变化', RUNNING: '运行中', CREATED: '待启动',
  SUCCESS: '成功', SUCCEEDED: '成功', COMPLETED: '已完成', PROMOTED: '已晋级',
  STOPPED: '已停止', FAILED: '失败', KNOWN_FAILURE: '已知失败', HARD_FAILED: '严重失败',
  PAUSED_OPERATOR: '已暂停', PAUSED_DATA_INTEGRITY: '数据异常，已暂停',
  PAUSED_HARD_FAILURE: '严重失败，已暂停', UNKNOWN: '结果未知',
  UNKNOWN_OUTCOME: '结果未知', UNKNOWN_GPU_OUTCOME: 'GPU 结果未知',
  PAUSED_UNKNOWN_OUTCOME: '结果未知，已隔离', UNAVAILABLE: '数据不可用',
  BUDGET_EXHAUSTED: '预算已用完', PROPOSAL_READY: '候选已就绪', ACTIVE: '使用中',
  RELEASED: '已释放', QUARANTINED: '已隔离', RESERVED: '已预留', SETTLED: '已结算',
  CANCELLED: '已取消', ABANDONED: '已终止', PENDING: '等待执行', QUALIFIED: '已确认',
}

export const stageLabels: Record<string, string> = {
  PROPOSE: '生成候选', POLICY: '策略检查', SMOKE: '快速验证', QUICK: '快速评测',
  FULL_PRIMARY: '完整评测', CONFIRMATION: '重复确认', BASELINE_QUALIFICATION: '基准确认',
  DONE: '结果归档', smoke: '快速验证', quick: '快速评测', full_primary: '完整评测',
  confirmation: '重复确认', baseline_qualification: '基准确认', validation: '验证测试',
  qualification: '资格测试', primary: '主评测', noise: '稳定性采样', full: '完整测试',
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
    month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false,
  }).format(date)
}

export function numberValue(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0
}

export function formatDuration(milliseconds: unknown): string {
  const value = numberValue(milliseconds)
  if (value <= 0) return '0 秒'
  if (value >= 3_600_000) return `${(value / 3_600_000).toFixed(2)} 小时`
  if (value >= 60_000) return `${(value / 60_000).toFixed(1)} 分钟`
  return `${(value / 1000).toFixed(1)} 秒`
}

export function scoreText(value: number | null): string {
  return value == null ? 'UNAVAILABLE' : value.toLocaleString('zh-CN', { maximumFractionDigits: 6 })
}

export function taskKindLabel(task: Pick<TaskSummary, 'kind'>): string {
  return task.kind === 'LONG' ? '长期优化' : task.kind === 'BENCHMARK' ? '策略对照' : '短期任务'
}

export function taskNeedsAttention(status: string): boolean {
  return status.includes('UNKNOWN') || status.includes('HARD') || status.includes('DATA_INTEGRITY')
}

export interface TaskSummary {
  id: string
  kind: 'LONG' | 'BENCHMARK' | 'RUN'
  title: string
  subtitle: string
  status: string
  updatedAt: unknown
  createdAt: unknown
  row: ConsoleRow
}

export interface EvaluationStageView {
  key: string
  label: string
  status: string
  completed: number
  total: number
  latest?: ConsoleRow
}

export interface OptimizationRoundView {
  id: string
  index: number
  status: string
  stage: string
  candidateHash: string
  outcome: string
  error: string
  updatedAt: unknown
  attempts: ConsoleRow[]
  experiments: ConsoleRow[]
  bestScore: number | null
  bestArtifactId: string
}

export interface ScorePointView {
  iteration: number
  score: number
  experimentUid: string
}

export interface TaskDetailView {
  task: TaskSummary
  children: ConsoleRow[]
  runs: ConsoleRow[]
  iterations: ConsoleRow[]
  attempts: ConsoleRow[]
  experiments: ConsoleRow[]
  relations: ConsoleRow[]
  budgets: ConsoleRow[]
  rounds: OptimizationRoundView[]
  scoreSeries: ScorePointView[]
  stages: EvaluationStageView[]
  candidateCount: number
  completedAttempts: number
  progress: number
  bestScore: number | null
  currentStage: string
  actualGpuMs: number
  reservedGpuMs: number
  actualWallMs: number
  actualTokens: number
  createdAt: unknown
  updatedAt: unknown
}

export function taskSummaries(snapshot: ConsoleSnapshot): TaskSummary[] {
  const childRunIds = new Set(snapshot.data.child_runs.map((row) => String(row.controller_run_id ?? '')).filter(Boolean))
  const campaigns = snapshot.data.campaigns.map((row): TaskSummary => {
    const benchmark = row.mode === 'BENCHMARK'
    return {
      id: String(row.id), kind: benchmark ? 'BENCHMARK' : 'LONG',
      title: benchmark ? '模型策略对照实验' : '长期持续优化',
      subtitle: benchmark ? '比较 Pro 和 Flash 两种模型策略' : '连续生成、测试并筛选候选代码',
      status: String(row.status ?? 'UNAVAILABLE'), updatedAt: row.updated_at ?? row.created_at,
      createdAt: row.created_at, row,
    }
  })
  const runs = snapshot.data.runs.filter((row) => !childRunIds.has(String(row.id))).map((row): TaskSummary => ({
    id: String(row.id), kind: 'RUN',
    title: String(row.id).startsWith('console-manual-') ? '已有代码评测' : '单次自主优化',
    subtitle: String(row.id).startsWith('console-manual-') ? '测试一份已有 kernel.py' : '模型生成并测试候选代码',
    status: String(row.status ?? 'UNAVAILABLE'), updatedAt: row.updated_at ?? row.created_at,
    createdAt: row.created_at, row,
  }))
  return [...campaigns, ...runs].sort((left, right) => String(right.updatedAt ?? '').localeCompare(String(left.updatedAt ?? '')))
}

function attemptsForIteration(attempts: ConsoleRow[], iteration: ConsoleRow): ConsoleRow[] {
  return attempts.filter((row) => String(row.iteration_id ?? '') === String(iteration.id ?? ''))
}

function experimentsForAttempts(experiments: ConsoleRow[], attempts: ConsoleRow[]): ConsoleRow[] {
  const uids = new Set(attempts.map((row) => String(row.experiment_uid ?? '')).filter(Boolean))
  return experiments.filter((row) => uids.has(String(row.experiment_uid ?? '')))
}

const stageOrder = ['POLICY', 'SMOKE', 'QUICK', 'FULL_PRIMARY', 'CONFIRMATION'] as const

export function taskDetail(snapshot: ConsoleSnapshot, task: TaskSummary): TaskDetailView {
  const children = task.kind === 'RUN' ? [] : snapshot.data.child_runs.filter((row) => String(row.campaign_id) === task.id)
  const runIds = new Set(task.kind === 'RUN' ? [task.id] : children.map((row) => String(row.controller_run_id ?? '')).filter(Boolean))
  const runs = snapshot.data.runs.filter((row) => runIds.has(String(row.id ?? '')))
  const iterations = snapshot.data.iterations.filter((row) => runIds.has(String(row.run_id ?? '')))
  const iterationIds = new Set(iterations.map((row) => String(row.id ?? '')))
  const attempts = snapshot.data.evaluation_attempts.filter((row) => runIds.has(String(row.run_id ?? '')) || iterationIds.has(String(row.iteration_id ?? '')))
  const experiments = experimentsForAttempts(snapshot.data.experiments, attempts)
  const experimentUids = new Set(experiments.map((row) => String(row.experiment_uid ?? '')).filter(Boolean))
  const relations = snapshot.data.experiment_relations.filter((row) => experimentUids.has(String(row.source_experiment_uid ?? '')) || experimentUids.has(String(row.target_experiment_uid ?? '')))
  const budgets = snapshot.data.budget_actions.filter((row) => String(row.campaign_id ?? '') === task.id)
  const candidateHashes = new Set(iterations.map((row) => String(row.candidate_hash ?? '')).filter(Boolean))
  const completedAttempts = attempts.filter((row) => ['SUCCESS', 'SUCCEEDED'].includes(String(row.status))).length
  const scores = experiments.map((row) => row.aggregate_score).filter((value): value is number => typeof value === 'number' && Number.isFinite(value))
  const sortedAttempts = attempts.slice().sort((left, right) => numberValue(left.id) - numberValue(right.id))
  const latestAttempt = sortedAttempts.at(-1)
  const currentStageKey = String(latestAttempt?.stage ?? iterations[0]?.stage ?? '').toUpperCase()
  const currentStageIndex = stageOrder.indexOf(currentStageKey as (typeof stageOrder)[number])
  const stages = stageOrder.map((stage): EvaluationStageView => {
    const matching = sortedAttempts.filter((row) => String(row.stage ?? '').toUpperCase() === stage)
    const completed = matching.filter((row) => ['SUCCESS', 'SUCCEEDED'].includes(String(row.status))).length
    const latest = matching.at(-1)
    return { key: stage, label: stageLabel(stage), status: latest ? String(latest.status ?? 'UNAVAILABLE') : 'PENDING', completed, total: matching.length, latest }
  })
  const rounds = iterations.slice().sort((left, right) => numberValue(right.iteration_index ?? right.id) - numberValue(left.iteration_index ?? left.id)).map((iteration): OptimizationRoundView => {
    const roundAttempts = attemptsForIteration(attempts, iteration)
    const roundExperiments = experimentsForAttempts(experiments, roundAttempts)
    const roundScores = roundExperiments.map((row) => row.aggregate_score).filter((value): value is number => typeof value === 'number' && Number.isFinite(value))
    return {
      id: String(iteration.id), index: numberValue(iteration.iteration_index) + 1,
      status: String(iteration.status ?? 'UNAVAILABLE'), stage: stageLabel(iteration.stage),
      candidateHash: String(iteration.candidate_hash ?? ''), outcome: String(iteration.outcome ?? ''),
      error: String(iteration.error ?? ''), updatedAt: iteration.updated_at,
      attempts: roundAttempts, experiments: roundExperiments,
      bestScore: roundScores.length ? Math.max(...roundScores) : null,
      bestArtifactId: String(
        roundExperiments
          .filter((row) => typeof row.aggregate_score === 'number' && Number.isFinite(row.aggregate_score))
          .sort((left, right) => Number(right.aggregate_score) - Number(left.aggregate_score))[0]?.artifact_id ?? '',
      ),
    }
  })
  const scoreSeries = rounds
    .filter((round): round is OptimizationRoundView & { bestScore: number } => round.bestScore !== null)
    .map((round) => ({
      iteration: round.index,
      score: round.bestScore,
      experimentUid: String(
        round.experiments.find((row) => row.aggregate_score === round.bestScore)?.experiment_uid ?? '',
      ),
    }))
    .sort((left, right) => left.iteration - right.iteration)
  return {
    task, children, runs, iterations, attempts: sortedAttempts, experiments, relations, budgets, rounds, scoreSeries, stages,
    candidateCount: candidateHashes.size || numberValue(task.row.valid_candidates), completedAttempts,
    progress: currentStageIndex >= 0 ? Math.round((currentStageIndex + 1) / stageOrder.length * 100) : 0,
    bestScore: scores.length ? Math.max(...scores) : null,
    currentStage: stageLabel(currentStageKey),
    actualGpuMs: budgets.reduce((total, row) => total + numberValue(row.actual_gpu_ms), 0),
    reservedGpuMs: budgets.reduce((total, row) => total + numberValue(row.reserved_gpu_ms), 0),
    actualWallMs: budgets.reduce((total, row) => total + numberValue(row.actual_wall_ms), 0),
    actualTokens: budgets.reduce((total, row) => total + numberValue(row.actual_tokens), 0),
    createdAt: task.createdAt ?? runs[0]?.created_at, updatedAt: task.updatedAt,
  }
}
