export type EvidenceSlot<T> =
  | { state: 'NOT_EXPOSED' }
  | { state: 'UNAVAILABLE'; reason: string }
  | { state: 'AVAILABLE'; value: T }

export interface BoundedTextView {
  text: string
  byteCount: number
  sha256: string
  truncated: boolean
}

export interface WriterOutputView {
  phase: string
  model: string | null
  responseId: string | null
  finishReason: string | null
  usage: Record<string, number>
  reasoning: EvidenceSlot<BoundedTextView>
  content: EvidenceSlot<BoundedTextView>
}

export interface PlannerDirectionView {
  id: string
  coreMechanism: string
  measuredProblem: string
  cheapestTest: string
  continueIf: string
  stopIf: string
  implementationRisk: 'LOW' | 'MEDIUM' | 'HIGH' | 'UNAVAILABLE'
  selected: boolean
}

export interface PlannerEvidenceView {
  status: string
  policy: string | null
  triggerReasons: string[]
  directions: PlannerDirectionView[]
  avoid: string[]
  elapsedSeconds: number | null
  error: string | null
}

export interface ProposalDiagnosticView {
  status: string
  algorithmFamily: string | null
  searchDirection: string | null
  bottleneck: string | null
  profileUse: string
  profileEvidence: string[]
  hypothesis: string | null
  codeChange: string | null
  expectedCounterDelta: string[]
  historyAvoidance: string[]
  confidence: string | null
  outcome: string | null
  conclusion: string
  parseError: string | null
}

export interface CompileRepairAttemptView {
  attempt: number
  status: string
  inputSha256: string | null
  candidateSha256: string | null
  writerStatus: string
  screenStatus: string
  description: string | null
  error: string | null
  writerSeconds: number | null
  screenSeconds: number | null
}

export interface CompileRepairEvidenceView {
  policy: string | null
  initialStatus: string
  initialCategory: string | null
  initialError: string | null
  attempts: CompileRepairAttemptView[]
  finalStatus: string
}

export interface CandidateDiffView {
  parentSha256: string
  candidateSha256: string
  diff: BoundedTextView
}

export interface ProfilerMetricView {
  name: string
  value: number | null
  unit: string | null
  unavailableReason: string | null
}

export interface ParentProfilerView {
  status: string
  profileId: string
  parentSha256: string
  cacheStatus: string
  promotionAuthority: false
  metrics: ProfilerMetricView[]
  error: string | null
}

export interface IterationDeepEvidenceView {
  writerOutputs: EvidenceSlot<WriterOutputView[]>
  planner: EvidenceSlot<PlannerEvidenceView>
  proposalDiagnostic: EvidenceSlot<ProposalDiagnosticView>
  compileRepair: EvidenceSlot<CompileRepairEvidenceView>
  candidateDiff: EvidenceSlot<CandidateDiffView>
  parentProfiler: EvidenceSlot<ParentProfilerView>
}

export const notExposedEvidence = <T>(): EvidenceSlot<T> => ({ state: 'NOT_EXPOSED' })

export function notExposedIterationEvidence(): IterationDeepEvidenceView {
  return {
    writerOutputs: notExposedEvidence(),
    planner: notExposedEvidence(),
    proposalDiagnostic: notExposedEvidence(),
    compileRepair: notExposedEvidence(),
    candidateDiff: notExposedEvidence(),
    parentProfiler: notExposedEvidence(),
  }
}

export function exposedEvidenceCount(evidence: IterationDeepEvidenceView): number {
  return Object.values(evidence).filter((slot) => slot.state !== 'NOT_EXPOSED').length
}
