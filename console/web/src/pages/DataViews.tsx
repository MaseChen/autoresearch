import { Card, Descriptions, Space, Table, Tabs, Typography } from 'antd'
import { lazy, type ReactNode, Suspense } from 'react'
import type { ConsoleRow, ConsoleSnapshot } from '../types'
import { StatusBadge } from '../components/StatusBadge'
import { OperationButton } from '../components/OperationButton'

const LineageGraph = lazy(() => import('../components/LineageGraph').then((module) => ({ default: module.LineageGraph })))

const { Text, Title } = Typography

function GenericTable({ rows, identity = 'id', actions }: { rows: ConsoleRow[]; identity?: string; actions?: (row: ConsoleRow) => ReactNode }) {
  const keys = Array.from(new Set(rows.flatMap((row) => Object.keys(row)))).slice(0, 8)
  return (
    <Table
      size="small"
      scroll={{ x: true }}
      rowKey={(row, index) => String(row[identity] ?? index)}
      dataSource={rows}
      pagination={{ pageSize: 20, hideOnSinglePage: true }}
      columns={[...keys.map((key) => ({
        title: key,
        dataIndex: key,
        render: (value: unknown, row: ConsoleRow) => key === 'status'
          ? <StatusBadge value={String(value)} reason={String(row.reason_code ?? row.reason ?? '')} />
          : <Text code={key.includes('id') || key.includes('digest')}>{value == null ? '—' : String(value)}</Text>,
      })), ...(actions ? [{ title: '操作', key: 'actions', fixed: 'right' as const, render: (_: unknown, row: ConsoleRow) => actions(row) }] : [])]}
    />
  )
}

export function TasksPage({ snapshot, canWrite = false }: { snapshot: ConsoleSnapshot; canWrite?: boolean }) {
  const runtime = snapshot.runtime_identity.runtime_identity_digest
  return <section><Title level={2}>观察过程</Title><Tabs items={[
    { key: 'runs', label: `Runs (${snapshot.data.runs.length})`, children: <Card><GenericTable rows={snapshot.data.runs} actions={(row) => <Space>
      {row.status === 'RUNNING' && <OperationButton kind="RUN_STOP" parameters={{ profile: 'pro', run_id: row.id }} runtimeIdentityDigest={runtime} canWrite={canWrite} label="停止" danger />}
      {!['PROMOTED', 'STOPPED', 'FAILED', 'HARD_FAILED', 'BUDGET_EXHAUSTED', 'PROPOSAL_READY'].includes(String(row.status)) && row.status !== 'RUNNING' && <OperationButton kind="RUN_RESUME" parameters={{ profile: 'pro', run_id: row.id }} runtimeIdentityDigest={runtime} canWrite={canWrite} label="恢复" />}
    </Space>} /></Card> },
    { key: 'iterations', label: `Iterations (${snapshot.data.iterations.length})`, children: <Card><GenericTable rows={snapshot.data.iterations} /></Card> },
    { key: 'attempts', label: `Evaluation attempts (${snapshot.data.evaluation_attempts.length})`, children: <Card><GenericTable rows={snapshot.data.evaluation_attempts} /></Card> },
  ]} /></section>
}

export function ExperimentsPage({ snapshot }: { snapshot: ConsoleSnapshot }) {
  return <section><Title level={2}>实验与 History</Title><Card title="XPU-OJ 对齐评分（shadow-only）"><Table
    size="small"
    scroll={{ x: true }}
    rowKey={(row) => String(row.experiment_uid)}
    dataSource={snapshot.data.experiments}
    pagination={{ pageSize: 20, hideOnSinglePage: true }}
    columns={[
      { title: 'Experiment', dataIndex: 'id' },
      { title: '状态', dataIndex: 'status', render: (value: string) => <StatusBadge value={value} /> },
      { title: 'Suite', dataIndex: 'suite' },
      { title: '候选', dataIndex: 'candidate_hash', render: (value: string) => <Text code copyable>{value}</Text> },
      { title: 'XPU-OJ proxy 状态', dataIndex: 'xpuoj_proxy_status', render: (value: string, row) => <StatusBadge value={value} reason={String(row.xpuoj_proxy_reason ?? '')} /> },
      { title: 'TH0 proxy', dataIndex: 'xpuoj_proxy_score', render: (value: unknown) => value == null ? 'UNAVAILABLE' : Number(value).toFixed(4) },
      { title: '配对 speedup', dataIndex: 'xpuoj_proxy_paired_speedup', render: (value: unknown) => value == null ? 'UNAVAILABLE' : `${Number(value).toFixed(4)}×` },
      { title: '最坏 case 回退', dataIndex: 'xpuoj_proxy_worst_case_regression', render: (value: unknown) => value == null ? 'UNAVAILABLE' : `${(Number(value) * 100).toFixed(2)}%` },
      { title: 'Promotion authority', dataIndex: 'xpuoj_proxy_promotion_authority', render: () => 'false' },
      { title: 'Legacy aggregate', dataIndex: 'aggregate_score', render: (value: unknown) => value == null ? 'UNAVAILABLE' : String(value) },
    ]}
  /></Card><Card title="完整 History 身份" className="section-card"><GenericTable rows={snapshot.data.experiments} identity="experiment_uid" /></Card></section>
}

export function CampaignsPage({ snapshot, canWrite = false }: { snapshot: ConsoleSnapshot; canWrite?: boolean }) {
  const revisionIds = snapshot.data.child_runs.map((row) => String(row.baseline_revision_id ?? row.id))
  const runtime = snapshot.runtime_identity.runtime_identity_digest
  return <section><Title level={2}>Campaign 与 Baseline lineage</Title><Card title="Lineage"><Suspense fallback={<div className="lineage" aria-label="正在加载谱系图" />}><LineageGraph ids={revisionIds} /></Suspense></Card><Card className="section-card"><GenericTable rows={snapshot.data.campaigns} actions={(row) => <Space wrap>
    {row.status === 'CREATED' && <OperationButton kind="CAMPAIGN_START" parameters={{ campaign_id: row.id }} runtimeIdentityDigest={runtime} canWrite={canWrite} label="启动" />}
    {row.status === 'RUNNING' && row.mode !== 'BENCHMARK' && <OperationButton kind="CAMPAIGN_CHILD_EXECUTE" parameters={{ campaign_id: row.id, profile: 'pro', max_candidates: 5, max_wall_seconds: 21600 }} runtimeIdentityDigest={runtime} canWrite={canWrite} label="执行 Child" />}
    {row.status === 'RUNNING' && row.mode === 'BENCHMARK' && <OperationButton kind="BENCHMARK_EXECUTE" parameters={{ campaign_id: row.id, profile: 'pro' }} runtimeIdentityDigest={runtime} canWrite={canWrite} label="执行 Benchmark" />}
    {row.status === 'RUNNING' && <OperationButton kind="CAMPAIGN_PAUSE" parameters={{ campaign_id: row.id, reason: 'Console operator pause' }} runtimeIdentityDigest={runtime} canWrite={canWrite} label="暂停" danger />}
    {String(row.status).startsWith('PAUSED_') && row.status !== 'PAUSED_DATA_INTEGRITY' && <OperationButton kind="CAMPAIGN_RESUME" parameters={{ campaign_id: row.id, profile: 'pro' }} runtimeIdentityDigest={runtime} canWrite={canWrite} label="恢复" />}
  </Space>} /></Card><Card title="Child 与 promotion" className="section-card"><GenericTable rows={snapshot.data.child_runs} actions={(row) => row.status === 'PROMOTED' ? <OperationButton kind="CAMPAIGN_LINEAGE_ADVANCE" parameters={{ campaign_id: row.campaign_id, child_id: row.id }} runtimeIdentityDigest={runtime} canWrite={canWrite} label="复证并推进谱系" danger /> : null} /></Card></section>
}

export function ResourcesPage({ snapshot }: { snapshot: ConsoleSnapshot }) {
  return <section><Title level={2}>资源、Lease 与 Budget</Title><Tabs items={[
    { key: 'leases', label: 'Resource leases', children: <Card><GenericTable rows={snapshot.data.resource_leases} identity="fencing_epoch" /></Card> },
    { key: 'budgets', label: 'Budget actions', children: <Card><GenericTable rows={snapshot.data.budget_actions} /></Card> },
  ]} /></section>
}

export function SoakPage({ snapshot }: { snapshot: ConsoleSnapshot }) {
  return <section><Title level={2}>Soak 门禁</Title><Card><GenericTable rows={snapshot.data.soak_generations} /></Card><Card title="违规记录" className="section-card"><GenericTable rows={snapshot.data.soak_violations} /></Card></section>
}

export function AuditPage({ snapshot }: { snapshot: ConsoleSnapshot }) {
  return <section><Title level={2}>审计与身份</Title><Card><Descriptions column={1}>
    {Object.entries(snapshot.runtime_identity).map(([key, value]) => <Descriptions.Item key={key} label={key}><Text code copyable>{String(value)}</Text></Descriptions.Item>)}
  </Descriptions></Card></section>
}

export function SettingsPage() {
  return <section><Title level={2}>本地设置</Title><Card><Descriptions column={1}>
    <Descriptions.Item label="部署拓扑">Mac localhost Gateway → fixed SSH → standard-library Agent</Descriptions.Item>
    <Descriptions.Item label="认证">一次性 bootstrap + HttpOnly SameSite session</Descriptions.Item>
    <Descriptions.Item label="Secrets">永不展示或编辑，仅显示 redacted credential reference</Descriptions.Item>
    <Descriptions.Item label="移动端">只读健康状态；所有写操作禁用</Descriptions.Item>
  </Descriptions></Card></section>
}
