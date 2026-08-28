import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import { DeepEvidencePanel } from '../src/components/DeepEvidencePanel'
import { exposedEvidenceCount, notExposedIterationEvidence, type IterationDeepEvidenceView } from '../src/deepEvidence'

const bounded = (text: string) => ({ text, byteCount: text.length, sha256: `sha256:${'a'.repeat(64)}`, truncated: false })

function availableEvidence(): IterationDeepEvidenceView {
  return {
    writerOutputs: { state: 'AVAILABLE', value: [{
      phase: 'initial', model: 'deepseek-v4-pro', responseId: 'response-1', finishReason: 'stop', usage: { total_tokens: 42 },
      reasoning: { state: 'AVAILABLE', value: bounded('reasoning text') },
      content: { state: 'AVAILABLE', value: bounded('content text') },
    }] },
    planner: { state: 'AVAILABLE', value: {
      status: 'SUCCESS', policy: 'planner-v2', triggerReasons: ['连续三轮未晋级'], avoid: ['重复方向'], elapsedSeconds: 2.5, error: null,
      directions: [{ id: 'direction-1', coreMechanism: '重排分块', measuredProblem: '访存受限', cheapestTest: 'quick', continueIf: '评分提升', stopIf: '正确性失败', implementationRisk: 'MEDIUM', selected: true }],
    } },
    proposalDiagnostic: { state: 'AVAILABLE', value: {
      status: 'SUCCESS', algorithmFamily: 'tiling', searchDirection: 'reduce-loads', bottleneck: 'memory', profileUse: 'used', profileEvidence: ['counter-a'], hypothesis: '减少访存', codeChange: '调整分块', expectedCounterDelta: ['loads down'], historyAvoidance: ['old-path'], confidence: 'high', outcome: 'promoted', conclusion: 'supported', parseError: null,
    } },
    compileRepair: { state: 'AVAILABLE', value: {
      policy: 'compile-repair-v1', initialStatus: 'FAILED', initialCategory: 'compile', initialError: 'syntax error', finalStatus: 'SUCCESS',
      attempts: [{ attempt: 1, status: 'SUCCESS', inputSha256: null, candidateSha256: `sha256:${'b'.repeat(64)}`, writerStatus: 'SUCCESS', screenStatus: 'SUCCESS', description: '修复括号', error: null, writerSeconds: 1, screenSeconds: 2 }],
    } },
    candidateDiff: { state: 'AVAILABLE', value: { parentSha256: `sha256:${'c'.repeat(64)}`, candidateSha256: `sha256:${'d'.repeat(64)}`, diff: bounded('- old\n+ new') } },
    parentProfiler: { state: 'AVAILABLE', value: {
      status: 'SUCCESS', profileId: 'profile-1', parentSha256: `sha256:${'c'.repeat(64)}`, cacheStatus: 'reused', promotionAuthority: false, error: null,
      metrics: [{ name: 'registers', value: null, unit: null, unavailableReason: 'COUNTER_NOT_EXPOSED' }, { name: 'cycles', value: 12, unit: 'cycles', unavailableReason: null }],
    } },
  }
}

describe('DeepEvidencePanel', () => {
  it('collapses all not-exposed capabilities into one honest production notice', async () => {
    const user = userEvent.setup()
    const evidence = notExposedIterationEvidence()
    expect(exposedEvidenceCount(evidence)).toBe(0)
    render(<DeepEvidencePanel evidence={evidence} />)
    expect(screen.getAllByText(/当前协议未接入/)).toHaveLength(1)
    expect(screen.queryByText('Writer reasoning')).not.toBeInTheDocument()
    await user.click(screen.getByText(/当前协议未接入/))
    expect(screen.getByText('当前 Agent 协议尚未提供逐轮深度证据')).toBeInTheDocument()
    expect(screen.getByText(/未取得可信数据前不会显示模拟内容/)).toBeInTheDocument()
  })

  it('renders all six future evidence surfaces from bounded fixture data', async () => {
    const user = userEvent.setup()
    const evidence = availableEvidence()
    expect(exposedEvidenceCount(evidence)).toBe(6)
    render(<DeepEvidencePanel evidence={evidence} />)
    for (const label of ['Writer 输出 (1)', 'Planner 方向与触发原因', 'Proposal 结构化诊断', 'Compile repair (1)', '候选与精确 writer parent 的 diff', '每轮父版本 Profiler 指标']) {
      await user.click(screen.getByText(label))
    }
    await user.click(screen.getByText('Writer reasoning'))
    await user.click(screen.getByText('Writer content'))
    expect(await screen.findByText('reasoning text')).toBeInTheDocument()
    expect(await screen.findByText('连续三轮未晋级')).toBeInTheDocument()
    expect(await screen.findByText('重复方向')).toBeInTheDocument()
    expect(await screen.findByText('减少访存')).toBeInTheDocument()
    expect(await screen.findByText('counter-a')).toBeInTheDocument()
    expect(await screen.findByText('loads down')).toBeInTheDocument()
    expect(await screen.findByText('old-path')).toBeInTheDocument()
    expect(await screen.findByText(/修复括号/)).toBeInTheDocument()
    expect(await screen.findByText(/Writer 1.00 秒 · Screen 2.00 秒/)).toBeInTheDocument()
    expect(await screen.findByText(/\+ new/)).toBeInTheDocument()
    expect(await screen.findByText(/COUNTER_NOT_EXPOSED/)).toBeInTheDocument()
    expect(await screen.findByText('12 cycles')).toBeInTheDocument()
  })

  it('fails closed when an available diff lacks its exact parent identity', async () => {
    const user = userEvent.setup()
    const evidence = notExposedIterationEvidence()
    evidence.candidateDiff = { state: 'AVAILABLE', value: { parentSha256: '', candidateSha256: `sha256:${'d'.repeat(64)}`, diff: bounded('unsafe diff') } }
    render(<DeepEvidencePanel evidence={evidence} />)
    await user.click(screen.getByText('候选与精确 writer parent 的 diff'))
    expect(screen.getByText('精确父本身份缺失，diff 已拒绝显示')).toBeInTheDocument()
    expect(screen.queryByText('unsafe diff')).not.toBeInTheDocument()
  })

  it('renders sparse structured evidence as unavailable instead of inventing defaults', async () => {
    const user = userEvent.setup()
    const evidence = availableEvidence()
    if (evidence.planner.state === 'AVAILABLE') evidence.planner.value = {
      ...evidence.planner.value,
      policy: null,
      triggerReasons: [],
      avoid: [],
      elapsedSeconds: null,
      error: 'planner stopped',
      directions: evidence.planner.value.directions.map((direction) => ({ ...direction, selected: false })),
    }
    if (evidence.proposalDiagnostic.state === 'AVAILABLE') evidence.proposalDiagnostic.value = {
      ...evidence.proposalDiagnostic.value,
      algorithmFamily: null,
      searchDirection: null,
      bottleneck: null,
      hypothesis: null,
      codeChange: null,
      confidence: null,
      outcome: null,
      profileEvidence: [],
      expectedCounterDelta: [],
      historyAvoidance: [],
      parseError: 'invalid diagnostic',
    }
    if (evidence.compileRepair.state === 'AVAILABLE') evidence.compileRepair.value = {
      ...evidence.compileRepair.value,
      initialCategory: null,
      initialError: null,
      attempts: [{
        ...evidence.compileRepair.value.attempts[0],
        description: null,
        inputSha256: null,
        candidateSha256: null,
        writerSeconds: null,
        screenSeconds: null,
        error: 'repair failed',
      }],
    }
    if (evidence.parentProfiler.state === 'AVAILABLE') evidence.parentProfiler.value = {
      ...evidence.parentProfiler.value,
      error: 'profile failed',
      metrics: [{ name: 'cycles', value: null, unit: null, unavailableReason: null }],
    }
    render(<DeepEvidencePanel evidence={evidence} />)
    for (const label of ['Planner 方向与触发原因', 'Proposal 结构化诊断', 'Compile repair (1)', '每轮父版本 Profiler 指标']) {
      await user.click(screen.getByText(label))
    }
    expect(screen.getByText('planner stopped')).toBeInTheDocument()
    expect(screen.getByText('invalid diagnostic')).toBeInTheDocument()
    expect(screen.getByText('repair failed')).toBeInTheDocument()
    expect(screen.getByText('profile failed')).toBeInTheDocument()
    expect(screen.getAllByText('未记录')).toHaveLength(5)
    expect(screen.getAllByText('UNAVAILABLE').length).toBeGreaterThan(2)
    expect(screen.getByText(/Writer UNAVAILABLE · Screen UNAVAILABLE/)).toBeInTheDocument()
    expect(screen.getByText(/COUNTER_NOT_EXPOSED/)).toBeInTheDocument()
  })

  it('shows fixed reasons for every unavailable evidence family without inventing content', async () => {
    const user = userEvent.setup()
    const evidence = notExposedIterationEvidence()
    evidence.writerOutputs = { state: 'UNAVAILABLE', reason: 'WRITER_OUTPUT_NOT_RECORDED' }
    evidence.planner = { state: 'UNAVAILABLE', reason: 'PLANNER_NOT_RECORDED' }
    evidence.proposalDiagnostic = { state: 'UNAVAILABLE', reason: 'PROPOSAL_NOT_RECORDED' }
    evidence.compileRepair = { state: 'UNAVAILABLE', reason: 'REPAIR_NOT_ATTEMPTED' }
    evidence.candidateDiff = { state: 'UNAVAILABLE', reason: 'PARENT_IDENTITY_NOT_EXPOSED' }
    evidence.parentProfiler = { state: 'UNAVAILABLE', reason: 'PROFILE_NOT_COLLECTED' }
    render(<DeepEvidencePanel evidence={evidence} />)
    const labels = ['Writer 输出', 'Planner 方向与触发原因', 'Proposal 结构化诊断', 'Compile repair', '候选与精确 writer parent 的 diff', '每轮父版本 Profiler 指标']
    for (const label of labels) await user.click(screen.getByText(label))
    for (const reason of ['WRITER_OUTPUT_NOT_RECORDED', 'PLANNER_NOT_RECORDED', 'PROPOSAL_NOT_RECORDED', 'REPAIR_NOT_ATTEMPTED', 'PARENT_IDENTITY_NOT_EXPOSED', 'PROFILE_NOT_COLLECTED']) {
      expect(screen.getByText(new RegExp(reason))).toBeInTheDocument()
    }
  })
})
