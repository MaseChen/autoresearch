import type { SnapshotStatus, TaskKind } from './types'

export type TrustedProfile = 'pro' | 'flash'

export const defaultBudget = Object.freeze({
  candidates: 25,
  wall_ms: 86_400_000,
  gpu_ms: 21_600_000,
  tokens: 5_000_000,
  cost_microusd: 100_000_000,
})

export function canMutate(
  status: SnapshotStatus | undefined,
  viewportWidth: number,
  streamStatus: 'live' | 'stale',
): boolean {
  return status === 'STABLE'
    && Number.isFinite(viewportWidth)
    && viewportWidth >= 1280
    && streamStatus === 'live'
}

export function buildOperationParameters({
  kind,
  source,
  profile,
  candidateBudget,
  proposalOnly,
  repetitions,
}: {
  kind: TaskKind
  source: string
  profile: TrustedProfile
  candidateBudget: number
  proposalOnly: boolean
  repetitions: number
}): Record<string, unknown> {
  if (!Number.isInteger(candidateBudget) || candidateBudget < 1 || candidateBudget > 5) {
    throw new Error('candidate budget must be an integer from 1 to 5')
  }
  if (profile !== 'pro' && profile !== 'flash') {
    throw new Error('profile is not trusted')
  }
  if (kind === 'MANUAL_EVALUATION_START') {
    const bytes = new TextEncoder().encode(source).length
    if (!source.trim() || bytes > 262_144) throw new Error('candidate source is empty or too large')
    return { candidate: {
      format: 'source-bundle-v1',
      entrypoint: 'kernel.py',
      files: [{ path: 'kernel.py', media_type: 'text/x-python', content: source }],
    } }
  }
  if (kind === 'RUN_START') return { profile, proposal_only: proposalOnly }
  if (kind === 'CAMPAIGN_CREATE') {
    return { mode: 'DISCOVERY', profile, budget: { ...defaultBudget, candidates: Math.max(5, candidateBudget) } }
  }
  if (!Number.isInteger(repetitions) || repetitions < 1 || repetitions > 100) {
    throw new Error('benchmark repetitions must be an integer from 1 to 100')
  }
  return {
    profile,
    arms: ['pro', 'flash'],
    repetitions,
    budget: { ...defaultBudget, candidates: Math.max(10, candidateBudget) },
  }
}
