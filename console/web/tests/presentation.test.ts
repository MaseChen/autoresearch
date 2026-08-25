import { describe, expect, it } from 'vitest'
import { formatTime, shortIdentity, stageLabel, statusLabel, taskSummaries } from '../src/presentation'
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
    iterations: [], evaluation_attempts: [], experiments: [], experiment_relations: [],
    resource_leases: [], budget_actions: [], soak_generations: [], soak_violations: [],
  },
} as unknown as ConsoleSnapshot

describe('Chinese presentation model', () => {
  it('translates common states and stages without hiding unknown raw values', () => {
    expect(statusLabel('SUCCEEDED')).toBe('成功')
    expect(statusLabel('CUSTOM')).toBe('CUSTOM')
    expect(stageLabel('FULL_PRIMARY')).toBe('完整主评测')
    expect(stageLabel()).toBe('等待开始')
  })

  it('formats identities and timestamps safely', () => {
    expect(shortIdentity('abcdefghijklmnop', 6)).toBe('abcdef…')
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
})
