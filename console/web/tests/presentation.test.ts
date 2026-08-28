import { describe, expect, it } from 'vitest'
import { formatDuration, formatTime, numberValue, scoreText, shortIdentity, stageLabel, statusLabel, taskDetail, taskNeedsAttention, taskSummaries } from '../src/presentation'
import type { ConsoleSnapshot } from '../src/types'

const snapshot = {
  status: 'STABLE',
  data: {
    campaigns: [{ id: 'campaign-1', mode: 'DISCOVERY', status: 'RUNNING', updated_at: '2026-08-25T00:00:00Z' }],
    child_runs: [{ id: 1, campaign_id: 'campaign-1', controller_run_id: 'child-1' }],
    runs: [
      { id: 'child-1', status: 'RUNNING' },
      { id: 'console-manual-one', status: 'SUCCEEDED', updated_at: '2026-08-24T00:00:00Z' },
    ],
    iterations: [{ id: 11, run_id: 'console-manual-one', iteration_index: 0, status: 'DONE', stage: 'CONFIRMATION', candidate_hash: 'c'.repeat(64), updated_at: '2026-08-24T00:03:00Z' }],
    evaluation_attempts: [
      { id: 21, run_id: 'console-manual-one', iteration_id: 11, experiment_uid: 'exp-smoke', stage: 'SMOKE', status: 'SUCCEEDED' },
      { id: 22, run_id: 'console-manual-one', iteration_id: 11, experiment_uid: 'exp-full', stage: 'FULL_PRIMARY', status: 'SUCCEEDED' },
    ],
    experiments: [
      { id: 1, experiment_uid: 'exp-smoke', aggregate_score: 1.5, candidate_hash: 'c'.repeat(64) },
      { id: 2, experiment_uid: 'exp-full', aggregate_score: 2.5, candidate_hash: 'c'.repeat(64), artifact_id: 'source-bundle-v1:full' },
    ],
    experiment_relations: [{ id: 1, source_experiment_uid: 'exp-smoke', target_experiment_uid: 'exp-full' }],
    resource_leases: [], budget_actions: [], soak_generations: [], soak_violations: [],
  },
} as unknown as ConsoleSnapshot

describe('Chinese presentation model', () => {
  it('translates common states and stages without hiding unknown raw values', () => {
    expect(statusLabel('SUCCEEDED')).toBe('成功')
    expect(statusLabel('CUSTOM')).toBe('CUSTOM')
    expect(statusLabel()).toBe('数据不可用')
    expect(stageLabel('FULL_PRIMARY')).toBe('完整评测')
    expect(stageLabel('CUSTOM_STAGE')).toBe('CUSTOM_STAGE')
    expect(stageLabel()).toBe('等待开始')
  })

  it('formats identities and timestamps safely', () => {
    expect(shortIdentity('abcdefghijklmnop', 6)).toBe('abcdef…')
    expect(shortIdentity('short')).toBe('short')
    expect(shortIdentity('')).toBe('—')
    expect(formatTime(undefined)).toBe('—')
    expect(formatTime('not-a-date')).toBe('not-a-date')
    expect(formatTime('2026-08-25T00:00:00Z')).not.toContain('2026-08-25T00:00:00Z')
  })

  it('combines campaigns and top-level runs while excluding child runs', () => {
    const tasks = taskSummaries(snapshot)
    expect(tasks.map((task) => task.id)).toEqual(['campaign-1', 'console-manual-one'])
    expect(tasks[0].title).toBe('长期持续优化')
    expect(tasks[1].title).toBe('已有代码评测')
  })

  it('builds a readable task detail from linked controller and history rows', () => {
    const task = taskSummaries(snapshot).find((value) => value.id === 'console-manual-one')!
    const detail = taskDetail(snapshot, task)
    expect(detail.candidateCount).toBe(1)
    expect(detail.completedAttempts).toBe(2)
    expect(detail.progress).toBe(80)
    expect(detail.bestScore).toBe(2.5)
    expect(detail.currentStage).toBe('完整评测')
    expect(detail.rounds[0].stage).toBe('重复确认')
    expect(detail.rounds[0].bestArtifactId).toBe('source-bundle-v1:full')
    expect(detail.scoreSeries).toEqual([{ iteration: 1, score: 2.5, experimentUid: 'exp-full' }])
    expect(detail.relations).toHaveLength(1)
    expect(detail.stages.find((stage) => stage.key === 'SMOKE')?.status).toBe('SUCCEEDED')
  })

  it('formats bounded durations without inventing unavailable usage', () => {
    expect(formatDuration(undefined)).toBe('0 秒')
    expect(formatDuration(1_500)).toBe('1.5 秒')
    expect(formatDuration(90_000)).toBe('1.5 分钟')
    expect(formatDuration(7_200_000)).toBe('2.00 小时')
    expect(numberValue(4)).toBe(4)
    expect(numberValue(Number.NaN)).toBe(0)
    expect(numberValue('4')).toBe(0)
    expect(scoreText(null)).toBe('UNAVAILABLE')
    expect(taskNeedsAttention('PAUSED_UNKNOWN_OUTCOME')).toBe(true)
    expect(taskNeedsAttention('SUCCEEDED')).toBe(false)
  })

  it('covers benchmark and autonomous task labels with safe fallbacks', () => {
    const varied = structuredClone(snapshot) as ConsoleSnapshot
    varied.data.campaigns.push({ id: 'benchmark-1', mode: 'BENCHMARK', created_at: '2026-08-22T00:00:00Z' })
    varied.data.runs.push({ id: 'autonomous-1', created_at: '2026-08-21T00:00:00Z' })
    const tasks = taskSummaries(varied)
    expect(tasks.find((task) => task.id === 'benchmark-1')).toMatchObject({ kind: 'BENCHMARK', title: '模型策略对照实验', status: 'UNAVAILABLE' })
    expect(tasks.find((task) => task.id === 'autonomous-1')).toMatchObject({ title: '单次自主优化', subtitle: '模型生成并测试候选代码' })
  })

  it('builds empty and campaign details without fabricating results', () => {
    const varied = structuredClone(snapshot) as ConsoleSnapshot
    varied.data.campaigns = [{ id: 'campaign-empty', mode: 'DISCOVERY', status: 'CREATED', created_at: '2026-08-20T00:00:00Z', valid_candidates: 3 }]
    varied.data.child_runs = [{ id: 30, campaign_id: 'campaign-empty', controller_run_id: 'child-empty' }]
    varied.data.runs = [{ id: 'child-empty', status: 'CREATED', created_at: '2026-08-20T00:00:00Z' }]
    varied.data.iterations = [{ id: 31, run_id: 'child-empty' }]
    varied.data.evaluation_attempts = [{ id: 32, run_id: 'different-run', iteration_id: 31 }]
    varied.data.experiments = []
    varied.data.experiment_relations = [{ id: 1, source_experiment_uid: 'unlinked', target_experiment_uid: 'unlinked' }]
    varied.data.budget_actions = [{ id: 1, campaign_id: 'campaign-empty', actual_gpu_ms: 2_000, reserved_gpu_ms: 4_000, actual_wall_ms: 5_000, actual_tokens: 12 }, { id: 2, campaign_id: 'other', actual_gpu_ms: 9_000 }]
    const task = taskSummaries(varied)[0]
    const detail = taskDetail(varied, task)
    expect(detail.children).toHaveLength(1)
    expect(detail.runs).toHaveLength(1)
    expect(detail.attempts).toHaveLength(1)
    expect(detail.candidateCount).toBe(3)
    expect(detail.progress).toBe(0)
    expect(detail.bestScore).toBeNull()
    expect(detail.currentStage).toBe('等待开始')
    expect(detail.rounds[0]).toMatchObject({ index: 1, status: 'UNAVAILABLE', stage: '等待开始', candidateHash: '', bestScore: null })
    expect(detail.scoreSeries).toEqual([])
    expect(detail.stages.every((stage) => stage.status === 'PENDING')).toBe(true)
    expect(detail.relations).toHaveLength(0)
    expect(detail.actualGpuMs).toBe(2_000)
    expect(detail.reservedGpuMs).toBe(4_000)
    expect(detail.actualWallMs).toBe(5_000)
    expect(detail.actualTokens).toBe(12)
  })
})
