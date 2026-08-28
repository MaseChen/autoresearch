import type { Meta, StoryObj } from '@storybook/react-vite'
import { taskDetail, taskSummaries } from '../presentation'
import type { ConsoleSnapshot } from '../types'
import { TaskDetailPage } from './TaskDetail'

const candidate = 'c'.repeat(64)
const snapshot = {
  schema_version: 1,
  status: 'STABLE',
  runtime_identity: {
    schema_version: 1,
    git_commit: 'a'.repeat(40),
    expected_git_commit: 'a'.repeat(40),
    config_digest: `sha256:${'1'.repeat(64)}`,
    deployment_evidence_digest: `sha256:${'2'.repeat(64)}`,
    namespace_id: `sha256:${'3'.repeat(64)}`,
    execution_environment_digest: `sha256:${'4'.repeat(64)}`,
    profiler_activation_profile_digest: `sha256:${'5'.repeat(64)}`,
    controller_schema_version: 3,
    history_schema_version: 3,
    campaign_schema_version: 1,
    agent_protocol_digest: `sha256:${'6'.repeat(64)}`,
    runtime_identity_digest: `sha256:${'7'.repeat(64)}`,
  },
  cursor: {}, observed_at: '2026-08-25T00:00:00Z', source_digests: {},
  data: {
    runs: [{ id: 'run-dense-fixture', status: 'RUNNING', valid_candidates: 2, created_at: '2026-08-25T00:00:00Z', updated_at: '2026-08-25T00:15:00Z' }],
    iterations: [
      { id: 1, run_id: 'run-dense-fixture', iteration_index: 0, status: 'SUCCEEDED', stage: 'CONFIRMATION', candidate_hash: candidate, outcome: 'PROMOTED', updated_at: '2026-08-25T00:08:00Z' },
      { id: 2, run_id: 'run-dense-fixture', iteration_index: 1, status: 'RUNNING', stage: 'QUICK', candidate_hash: 'd'.repeat(64), updated_at: '2026-08-25T00:15:00Z' },
    ],
    evaluation_attempts: [
      { id: 1, run_id: 'run-dense-fixture', iteration_id: 1, experiment_uid: 'smoke-1', stage: 'SMOKE', suite: 'smoke', status: 'SUCCEEDED' },
      { id: 2, run_id: 'run-dense-fixture', iteration_id: 1, experiment_uid: 'quick-1', stage: 'QUICK', suite: 'quick', status: 'SUCCEEDED' },
      { id: 3, run_id: 'run-dense-fixture', iteration_id: 1, experiment_uid: 'full-1', stage: 'FULL_PRIMARY', suite: 'full', status: 'SUCCEEDED' },
      { id: 4, run_id: 'run-dense-fixture', iteration_id: 1, experiment_uid: 'confirm-1', stage: 'CONFIRMATION', suite: 'full', status: 'SUCCEEDED' },
      { id: 5, run_id: 'run-dense-fixture', iteration_id: 2, experiment_uid: 'smoke-2', stage: 'SMOKE', suite: 'smoke', status: 'SUCCEEDED' },
    ],
    experiments: [
      { id: 1, experiment_uid: 'smoke-1', status: 'SUCCESS', aggregate_score: 1.82, candidate_hash: candidate, artifact_id: 'source-bundle-v1:fixture', suite: 'smoke', backend: 'metax-c500', created_at: '2026-08-25T00:02:00Z' },
      { id: 2, experiment_uid: 'quick-1', status: 'SUCCESS', aggregate_score: 1.91, candidate_hash: candidate, artifact_id: 'source-bundle-v1:fixture', suite: 'quick', backend: 'metax-c500', created_at: '2026-08-25T00:04:00Z' },
      { id: 3, experiment_uid: 'full-1', status: 'SUCCESS', aggregate_score: 2.14, candidate_hash: candidate, artifact_id: 'source-bundle-v1:fixture', suite: 'full', backend: 'metax-c500', created_at: '2026-08-25T00:06:00Z' },
      { id: 4, experiment_uid: 'confirm-1', status: 'SUCCESS', aggregate_score: 2.12, candidate_hash: candidate, artifact_id: 'source-bundle-v1:fixture', suite: 'full', backend: 'metax-c500', created_at: '2026-08-25T00:08:00Z' },
      { id: 5, experiment_uid: 'smoke-2', status: 'SUCCESS', aggregate_score: null, candidate_hash: 'd'.repeat(64), artifact_id: 'source-bundle-v1:fixture-2', suite: 'smoke', backend: 'metax-c500', created_at: '2026-08-25T00:14:00Z' },
    ],
    experiment_relations: [], campaigns: [], child_runs: [], resource_leases: [],
    budget_actions: [{ id: 1, campaign_id: 'other', actual_gpu_ms: 120_000, actual_wall_ms: 600_000, actual_tokens: 88_000, reserved_gpu_ms: 900_000 }],
    soak_generations: [], soak_violations: [],
  },
} as unknown as ConsoleSnapshot

const detail = taskDetail(snapshot, taskSummaries(snapshot)[0])

const meta = {
  title: 'Console/TaskDetail',
  component: TaskDetailPage,
  parameters: { layout: 'fullscreen' },
  args: { detail, snapshot, canWrite: false, mode: 'light', onBack: () => undefined },
} satisfies Meta<typeof TaskDetailPage>

export default meta
type Story = StoryObj<typeof meta>

export const DenseScientificView: Story = {}
export const DarkDenseScientificView: Story = { args: { mode: 'dark' }, decorators: [(Story) => <div className="theme-dark" style={{ padding: 24, background: '#0e1928' }}><Story /></div>] }
