import type { Meta, StoryObj } from '@storybook/react-vite'
import { notExposedIterationEvidence, type IterationDeepEvidenceView } from '../deepEvidence'
import { DeepEvidencePanel } from './DeepEvidencePanel'

const text = (value: string) => ({ text: value, byteCount: value.length, sha256: `sha256:${'a'.repeat(64)}`, truncated: false })

const completeFixture: IterationDeepEvidenceView = {
  writerOutputs: { state: 'AVAILABLE', value: [{ phase: 'initial', model: 'deepseek-v4-pro', responseId: 'response-1', finishReason: 'stop', usage: { total_tokens: 2048 }, reasoning: { state: 'AVAILABLE', value: text('分析访存与分块关系。') }, content: { state: 'AVAILABLE', value: text('建议调整 BLOCK_N 并保持正确性门禁。') } }] },
  planner: { state: 'AVAILABLE', value: { status: 'SUCCESS', policy: 'planner-v2', triggerReasons: ['同一冠军连续三轮未晋级'], directions: [{ id: 'direction-1', coreMechanism: '重排分块', measuredProblem: '访存受限', cheapestTest: 'quick', continueIf: '评分提升', stopIf: '正确性失败', implementationRisk: 'MEDIUM', selected: true }], avoid: ['重复已失败方向'], elapsedSeconds: 2.5, error: null } },
  proposalDiagnostic: { state: 'AVAILABLE', value: { status: 'SUCCESS', algorithmFamily: 'tiling', searchDirection: 'reduce-loads', bottleneck: 'memory', profileUse: 'used', profileEvidence: ['counter-a'], hypothesis: '减少访存', codeChange: '调整分块', expectedCounterDelta: ['loads down'], historyAvoidance: ['old-path'], confidence: 'high', outcome: 'promoted', conclusion: 'supported', parseError: null } },
  compileRepair: { state: 'AVAILABLE', value: { policy: 'compile-repair-v1', initialStatus: 'FAILED', initialCategory: 'compile', initialError: 'syntax error', attempts: [{ attempt: 1, status: 'SUCCESS', inputSha256: null, candidateSha256: `sha256:${'b'.repeat(64)}`, writerStatus: 'SUCCESS', screenStatus: 'SUCCESS', description: '修复括号', error: null, writerSeconds: 1, screenSeconds: 2 }], finalStatus: 'SUCCESS' } },
  candidateDiff: { state: 'AVAILABLE', value: { parentSha256: `sha256:${'c'.repeat(64)}`, candidateSha256: `sha256:${'d'.repeat(64)}`, diff: text('- old\n+ new') } },
  parentProfiler: { state: 'AVAILABLE', value: { status: 'SUCCESS', profileId: 'profile-1', parentSha256: `sha256:${'c'.repeat(64)}`, cacheStatus: 'reused', promotionAuthority: false, metrics: [{ name: 'registers', value: null, unit: null, unavailableReason: 'COUNTER_NOT_EXPOSED' }, { name: 'cycles', value: 12, unit: 'cycles', unavailableReason: null }], error: null } },
}

const meta = {
  title: 'Console/DeepEvidencePanel',
  component: DeepEvidencePanel,
  tags: ['autodocs'],
} satisfies Meta<typeof DeepEvidencePanel>

export default meta
type Story = StoryObj<typeof meta>

export const ProtocolNotExposed: Story = { args: { evidence: notExposedIterationEvidence() } }
export const CompleteFutureFixture: Story = { args: { evidence: completeFixture } }
