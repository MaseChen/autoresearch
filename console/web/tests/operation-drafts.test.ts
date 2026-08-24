import { describe, expect, it } from 'vitest'
import { buildOperationParameters, canMutate } from '../src/operationDrafts'

const defaults = {
  source: 'print("candidate")\n',
  profile: 'pro' as const,
  candidateBudget: 1,
  proposalOnly: false,
  repetitions: 2,
}

describe('operation draft safety logic', () => {
  it('requires stable live desktop state for writes', () => {
    expect(canMutate('STABLE', 1280, 'live')).toBe(true)
    expect(canMutate('CHANGING', 1440, 'live')).toBe(false)
    expect(canMutate(undefined, 1440, 'live')).toBe(false)
    expect(canMutate('STABLE', 1279, 'live')).toBe(false)
    expect(canMutate('STABLE', Number.NaN, 'live')).toBe(false)
    expect(canMutate('STABLE', 1440, 'stale')).toBe(false)
  })

  it('builds all four fixed task DTOs', () => {
    const manual = buildOperationParameters({ ...defaults, kind: 'MANUAL_EVALUATION_START' })
    expect((manual.candidate as { format: string }).format).toBe('source-bundle-v1')
    expect(buildOperationParameters({ ...defaults, kind: 'RUN_START', proposalOnly: true })).toEqual({ profile: 'pro', proposal_only: true })
    expect(buildOperationParameters({ ...defaults, kind: 'CAMPAIGN_CREATE' })).toMatchObject({ mode: 'DISCOVERY', profile: 'pro' })
    expect(buildOperationParameters({ ...defaults, kind: 'BENCHMARK_INIT' })).toMatchObject({ arms: ['pro', 'flash'], repetitions: 2 })
  })

  it('rejects unsafe bounds, profiles and candidates', () => {
    for (const candidateBudget of [0, 6, 1.5]) {
      expect(() => buildOperationParameters({ ...defaults, kind: 'RUN_START', candidateBudget })).toThrow('candidate budget')
    }
    expect(() => buildOperationParameters({ ...defaults, kind: 'RUN_START', profile: 'other' as 'pro' })).toThrow('profile')
    expect(() => buildOperationParameters({ ...defaults, kind: 'MANUAL_EVALUATION_START', source: '  ' })).toThrow('candidate source')
    expect(() => buildOperationParameters({ ...defaults, kind: 'MANUAL_EVALUATION_START', source: 'x'.repeat(262_145) })).toThrow('candidate source')
    for (const repetitions of [0, 101, 1.5]) {
      expect(() => buildOperationParameters({ ...defaults, kind: 'BENCHMARK_INIT', repetitions })).toThrow('repetitions')
    }
  })
})
