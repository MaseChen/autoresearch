export type SnapshotStatus = 'STABLE' | 'CHANGING' | 'UNAVAILABLE'

export interface RuntimeIdentity {
  schema_version: 1
  git_commit: string
  expected_git_commit: string
  config_digest: string
  deployment_evidence_digest: string
  namespace_id: string
  execution_environment_digest: string
  profiler_activation_profile_digest: string
  scoring_shadow_profile_digest: string
  controller_schema_version: number
  history_schema_version: number
  campaign_schema_version: number
  agent_protocol_digest: string
  runtime_identity_digest: string
}

export interface ConsoleRow {
  id?: number | string
  status?: string
  stage?: string
  [key: string]: unknown
}

export interface SnapshotData {
  runs: ConsoleRow[]
  iterations: ConsoleRow[]
  evaluation_attempts: ConsoleRow[]
  experiments: ConsoleRow[]
  experiment_relations: ConsoleRow[]
  campaigns: ConsoleRow[]
  child_runs: ConsoleRow[]
  resource_leases: ConsoleRow[]
  budget_actions: ConsoleRow[]
  soak_generations: ConsoleRow[]
  soak_violations: ConsoleRow[]
}

export interface ConsoleSnapshot {
  schema_version: 1
  status: SnapshotStatus
  runtime_identity: RuntimeIdentity
  cursor: Record<string, unknown>
  observed_at: string
  source_digests: Record<string, string>
  data: SnapshotData
}

export interface ApiEnvelope<T> {
  schema_version: 1
  request_id: string
  data: T
}

export type TaskKind =
  | 'RUN_START'
  | 'MANUAL_EVALUATION_START'
  | 'CAMPAIGN_CREATE'
  | 'BENCHMARK_INIT'

export interface PreparedOperation {
  schema_version: 1
  operation_id: string
  kind: string
  operation_digest: string
  runtime_identity_digest: string
  prepared_at: string
  expires_epoch: number
  confirmation_phrase: string
  impact: Record<string, unknown>
}

export interface OperationReceipt {
  schema_version: 1
  operation_id: string
  kind: string
  operation_digest: string
  status: 'PREPARED' | 'EXECUTING' | 'SUCCEEDED' | 'FAILED' | 'UNKNOWN_OUTCOME' | 'EXPIRED'
  observed_at: string
  domain_identity: Record<string, unknown>
  result: Record<string, unknown> | null
  problem: Record<string, unknown> | null
}
