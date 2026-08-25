import { Alert, Card, Col, Descriptions, Empty, List, Row, Select, Space, Statistic, Table, Tabs, Tag, Timeline, Typography } from 'antd'
import { useMemo, useState, type ReactNode } from 'react'
import type { ConsoleRow, ConsoleSnapshot } from '../types'
import { StatusBadge } from '../components/StatusBadge'
import { OperationButton } from '../components/OperationButton'
import { formatTime, shortIdentity, stageLabel, taskSummaries, type TaskSummary } from '../presentation'

const { Paragraph, Text, Title } = Typography

function GenericTable({ rows, identity = 'id' }: { rows: ConsoleRow[]; identity?: string }) {
  const keys = Array.from(new Set(rows.flatMap((row) => Object.keys(row)))).slice(0, 10)
  return (
    <Table
      size="small"
      scroll={{ x: true }}
      rowKey={(row) => String(row[identity] ?? row.id ?? JSON.stringify(row))}
      dataSource={rows}
      pagination={{ pageSize: 20, hideOnSinglePage: true }}
      locale={{ emptyText: '当前表没有记录' }}
      columns={keys.map((key) => ({
        title: key,
        dataIndex: key,
        render: (value: unknown, row: ConsoleRow) => key === 'status'
          ? <StatusBadge value={String(value)} reason={String(row.reason_code ?? row.reason ?? '')} />
          : <Text code={key.includes('id') || key.includes('digest')}>{value == null ? '—' : String(value)}</Text>,
      }))}
    />
  )
}

function taskActions(task: TaskSummary, runtime: string, canWrite: boolean): ReactNode {
  const row = task.row
  if (task.kind === 'RUN') {
    return <Space wrap>
      {row.status === 'RUNNING' && <OperationButton kind="RUN_STOP" parameters={{ profile: 'pro', run_id: row.id }} runtimeIdentityDigest={runtime} canWrite={canWrite} label="停止任务" danger />}
      {!['PROMOTED', 'STOPPED', 'FAILED', 'HARD_FAILED', 'BUDGET_EXHAUSTED', 'PROPOSAL_READY'].includes(String(row.status)) && row.status !== 'RUNNING' && <OperationButton kind="RUN_RESUME" parameters={{ profile: 'pro', run_id: row.id }} runtimeIdentityDigest={runtime} canWrite={canWrite} label="恢复任务" />}
    </Space>
  }
  return <Space wrap>
    {row.status === 'CREATED' && <OperationButton kind="CAMPAIGN_START" parameters={{ campaign_id: row.id }} runtimeIdentityDigest={runtime} canWrite={canWrite} label="启动长期任务" />}
    {row.status === 'RUNNING' && task.kind === 'LONG' && <OperationButton kind="CAMPAIGN_CHILD_EXECUTE" parameters={{ campaign_id: row.id, profile: 'pro', max_candidates: 5, max_wall_seconds: 21600 }} runtimeIdentityDigest={runtime} canWrite={canWrite} label="执行下一子任务" />}
    {row.status === 'RUNNING' && task.kind === 'BENCHMARK' && <OperationButton kind="BENCHMARK_EXECUTE" parameters={{ campaign_id: row.id, profile: 'pro' }} runtimeIdentityDigest={runtime} canWrite={canWrite} label="执行对照批次" />}
    {row.status === 'RUNNING' && <OperationButton kind="CAMPAIGN_PAUSE" parameters={{ campaign_id: row.id, reason: 'Console operator pause' }} runtimeIdentityDigest={runtime} canWrite={canWrite} label="暂停任务" danger />}
    {String(row.status).startsWith('PAUSED_') && row.status !== 'PAUSED_DATA_INTEGRITY' && <OperationButton kind="CAMPAIGN_RESUME" parameters={{ campaign_id: row.id, profile: 'pro' }} runtimeIdentityDigest={runtime} canWrite={canWrite} label="可信恢复" />}
  </Space>
}

function TaskDetail({ task, snapshot, canWrite }: { task: TaskSummary; snapshot: ConsoleSnapshot; canWrite: boolean }) {
  const runtime = snapshot.runtime_identity.runtime_identity_digest
  const campaignChildren = task.kind === 'RUN' ? [] : snapshot.data.child_runs.filter((row) => String(row.campaign_id) === task.id)
  const runIds = new Set(task.kind === 'RUN' ? [task.id] : campaignChildren.map((row) => String(row.controller_run_id ?? '')))
  const iterations = snapshot.data.iterations.filter((row) => runIds.has(String(row.run_id ?? '')))
  const iterationIds = new Set(iterations.map((row) => String(row.id)))
  const attempts = snapshot.data.evaluation_attempts.filter((row) => iterationIds.has(String(row.iteration_id ?? '')) || runIds.has(String(row.run_id ?? '')))
  const experimentIds = new Set(attempts.map((row) => String(row.experiment_uid ?? '')).filter(Boolean))
  const experiments = snapshot.data.experiments.filter((row) => experimentIds.has(String(row.experiment_uid ?? '')))
  const budgetRows = snapshot.data.budget_actions.filter((row) => String(row.campaign_id ?? row.run_id ?? '').includes(task.id))
  const stageRows = attempts.length > 0 ? attempts : iterations

  return <div className="task-detail">
    <div className="task-detail-heading">
      <div><Text className="eyebrow">{task.kind === 'LONG' ? '长期任务' : task.kind === 'BENCHMARK' ? '策略对照' : '短期任务'}</Text><Title level={3}>{task.title}</Title><Paragraph>{task.subtitle}</Paragraph></div>
      <StatusBadge value={task.status} />
    </div>
    <Descriptions size="small" column={{ xs: 1, md: 2 }} className="task-facts">
      <Descriptions.Item label="任务标识"><Text code copyable>{task.id}</Text></Descriptions.Item>
      <Descriptions.Item label="最近更新">{formatTime(task.updatedAt)}</Descriptions.Item>
      <Descriptions.Item label="目标算子">Fused MoE I8 TN</Descriptions.Item>
      <Descriptions.Item label="实现技术栈">Triton · MetaX C500</Descriptions.Item>
    </Descriptions>
    <div className="task-action-strip">{taskActions(task, runtime, canWrite)}{!canWrite && <Text type="secondary">连接、快照或屏幕门禁未满足，写操作已禁用。</Text>}</div>

    <Tabs items={[
      { key: 'process', label: '执行过程', children: <Card className="detail-card" variant="borderless">
        {stageRows.length === 0 ? <Empty description="任务已创建，但还没有进入评测阶段。" /> : <Timeline items={stageRows.map((row) => ({
          color: String(row.status).includes('FAIL') ? 'red' : row.status === 'SUCCEEDED' || row.status === 'SUCCESS' ? 'green' : 'blue',
          children: <div className="timeline-entry"><strong>{stageLabel(row.stage)}</strong><StatusBadge value={String(row.status ?? 'UNAVAILABLE')} /><Text type="secondary">{formatTime(row.updated_at ?? row.created_at)}</Text>{row.error ? <Paragraph type="danger">{String(row.error)}</Paragraph> : null}</div>,
        }))} />}
      </Card> },
      { key: 'results', label: '结果与证据', children: <Card className="detail-card" variant="borderless">
        {experiments.length === 0 ? <Empty description="尚未形成可展示的科学记录；没有数据不会显示为零。" /> : <List dataSource={experiments} renderItem={(row) => <List.Item extra={<StatusBadge value={String(row.status ?? 'UNAVAILABLE')} />}><List.Item.Meta title={`${stageLabel(row.stage)} · 科学记录 ${row.id ?? '—'}`} description={<Space wrap><Text>综合评分：{row.aggregate_score == null ? 'UNAVAILABLE' : String(row.aggregate_score)}</Text><Text code>{shortIdentity(row.experiment_uid)}</Text></Space>} /></List.Item>} />}
      </Card> },
      { key: 'limits', label: '预算与权限', children: <Card className="detail-card" variant="borderless">
        <Alert type="info" showIcon title="任务不能自行修改部署基准" description={task.kind === 'LONG' ? '长期任务的候选晋级仍需要从完整 primary / confirmation 证据复证。' : '短期任务只生成科学证据，不会改变部署基准或长期谱系。'} />
        <Row gutter={[12, 12]} className="authority-grid compact-grid">
          <Col span={12}><div><small>预算流水</small><strong>{budgetRows.length ? `${budgetRows.length} 条受信记录` : '尚无记录'}</strong></div></Col>
          <Col span={12}><div><small>子任务</small><strong>{campaignChildren.length}</strong></div></Col>
          <Col span={12}><div><small>结果未知</small><strong>禁止自动重放</strong></div></Col>
          <Col span={12}><div><small>部署基准</small><strong>无修改权限</strong></div></Col>
        </Row>
      </Card> },
    ]} />
  </div>
}

export function TaskCenterPage({ snapshot, canWrite = false }: { snapshot: ConsoleSnapshot; canWrite?: boolean }) {
  const tasks = useMemo(() => taskSummaries(snapshot), [snapshot])
  const [filter, setFilter] = useState('ALL')
  const filtered = tasks.filter((task) => filter === 'ALL' || (filter === 'ACTIVE' ? task.status === 'RUNNING' : task.kind === filter))
  const [selectedId, setSelectedId] = useState(tasks[0]?.id ?? '')
  const selected = filtered.find((task) => task.id === selectedId) ?? filtered[0]

  return <section aria-labelledby="tasks-title">
    <div className="page-heading"><div><Text className="eyebrow">创建 → 运行 → 结果</Text><Title id="tasks-title" level={2}>任务中心</Title><Paragraph type="secondary">把阶段、资源和科学证据汇集到每个任务下，不要求用户理解数据库表关系。</Paragraph></div><Select aria-label="筛选任务" value={filter} onChange={(value) => { setFilter(value); setSelectedId('') }} options={[{ value: 'ALL', label: '全部任务' }, { value: 'ACTIVE', label: '正在运行' }, { value: 'LONG', label: '长期任务' }, { value: 'RUN', label: '短期任务' }, { value: 'BENCHMARK', label: '策略对照' }]} /></div>
    <Row gutter={[16, 16]}>
      <Col xs={24} xl={8}>
        <Card className="task-list-panel" title={`任务列表 · ${filtered.length}`}>
          {filtered.length === 0 ? <Empty description="当前筛选条件下没有任务" /> : <div className="task-list">{filtered.map((task) => <button type="button" key={`${task.kind}-${task.id}`} className={`task-list-item ${selected?.id === task.id ? 'selected' : ''}`} onClick={() => setSelectedId(task.id)}><span><strong>{task.title}</strong><small>{shortIdentity(task.id, 18)}</small></span><StatusBadge value={task.status} /></button>)}</div>}
        </Card>
      </Col>
      <Col xs={24} xl={16}>{selected ? <Card className="task-detail-panel"><TaskDetail task={selected} snapshot={snapshot} canWrite={canWrite} /></Card> : <Card><Empty description="选择一项任务查看过程和结果" /></Card>}</Col>
    </Row>
  </section>
}

function DataDiagnostics({ snapshot }: { snapshot: ConsoleSnapshot }) {
  const tables = [
    ['runs', '单次研究任务', snapshot.data.runs],
    ['iterations', '优化轮次', snapshot.data.iterations],
    ['attempts', '评测步骤', snapshot.data.evaluation_attempts],
    ['experiments', '科学实验记录', snapshot.data.experiments],
    ['relations', '实验关系', snapshot.data.experiment_relations],
    ['campaigns', '长期研究任务', snapshot.data.campaigns],
    ['children', '子任务', snapshot.data.child_runs],
    ['leases', 'GPU 资源锁', snapshot.data.resource_leases],
    ['budgets', '预算流水', snapshot.data.budget_actions],
    ['soak', '稳定性观察', snapshot.data.soak_generations],
    ['violations', '稳定性违规', snapshot.data.soak_violations],
  ] as const
  return <Card title="高级：数据与诊断" extra={<Tag>只读</Tag>}><Alert type="warning" showIcon title="这里是唯一直接展示原始账本的页面" description="用于工程诊断与逐字段复核；日常工作请使用任务中心。技术字段保留英文原名。" /><Tabs className="diagnostic-tabs" tabPlacement="start" items={tables.map(([key, label, rows]) => ({ key, label: `${label} (${rows.length})`, children: <GenericTable rows={rows} identity={key === 'experiments' ? 'experiment_uid' : 'id'} /> }))} /></Card>
}

export function SystemPage({ snapshot }: { snapshot: ConsoleSnapshot }) {
  const activeLeases = snapshot.data.resource_leases.filter((row) => row.status === 'ACTIVE')
  const quarantined = snapshot.data.resource_leases.filter((row) => row.status === 'QUARANTINED')
  const latestSoak = snapshot.data.soak_generations.at(0)
  return <section aria-labelledby="system-title">
    <div className="page-heading"><div><Text className="eyebrow">运行边界与稳定性</Text><Title id="system-title" level={2}>系统与门禁</Title><Paragraph type="secondary">先给出能否继续工作的结论，再提供必要证据；原始账本收纳在最后一个高级页签。</Paragraph></div><StatusBadge value={snapshot.status} /></div>
    <Tabs items={[
      { key: 'health', label: '系统健康', children: <Row gutter={[16, 16]}>
        <Col xs={24} lg={8}><Card className="gate-card"><Statistic title="快照一致性" value={snapshot.status === 'STABLE' ? '稳定' : '正在变化'} /><Paragraph type="secondary">三库高水位前后对账</Paragraph></Card></Col>
        <Col xs={24} lg={8}><Card className="gate-card"><Statistic title="活动 GPU 资源锁" value={activeLeases.length} /><Paragraph type="secondary">任何重叠都会阻止新任务</Paragraph></Card></Col>
        <Col xs={24} lg={8}><Card className="gate-card"><Statistic title="隔离资源" value={quarantined.length} /><Paragraph type="secondary">隔离资源不能由页面直接释放</Paragraph></Card></Col>
        <Col span={24}><Card title="冻结运行身份"><Descriptions column={{ xs: 1, md: 2 }} size="small">{Object.entries(snapshot.runtime_identity).map(([key, value]) => <Descriptions.Item key={key} label={key}><Text code copyable>{String(value)}</Text></Descriptions.Item>)}</Descriptions></Card></Col>
      </Row> },
      { key: 'resources', label: '资源与预算', children: <Row gutter={[16, 16]}>
        <Col xs={24} xl={12}><Card title="GPU 资源状态">{snapshot.data.resource_leases.length === 0 ? <Empty description="当前没有 GPU 资源锁记录" /> : <List dataSource={snapshot.data.resource_leases} renderItem={(row) => <List.Item extra={<StatusBadge value={String(row.status ?? 'UNAVAILABLE')} />}><List.Item.Meta title={`资源 ${String(row.resource_id ?? row.id ?? 'gpu1')}`} description={`Fencing epoch ${String(row.fencing_epoch ?? '—')} · ${String(row.owner ?? row.campaign_id ?? '所有者未记录')}`} /></List.Item>} />}</Card></Col>
        <Col xs={24} xl={12}><Card title="预算状态">{snapshot.data.budget_actions.length === 0 ? <Empty description="当前没有预算流水" /> : <List dataSource={snapshot.data.budget_actions.slice(0, 12)} renderItem={(row) => <List.Item extra={<StatusBadge value={String(row.status ?? 'UNAVAILABLE')} />}><List.Item.Meta title={String(row.action ?? row.action_key ?? '预算动作')} description={`任务 ${shortIdentity(row.campaign_id ?? row.run_id)} · ${String(row.gpu_ms ?? row.wall_ms ?? '数值不可用')}`} /></List.Item>} />}</Card></Col>
      </Row> },
      { key: 'soak', label: '稳定性观察', children: <Card title="24 / 72 / 168 小时门禁">{latestSoak ? <><Descriptions column={{ xs: 1, md: 2 }}><Descriptions.Item label="当前阶段">{String(latestSoak.stage ?? latestSoak.target_stage ?? '等待识别')}</Descriptions.Item><Descriptions.Item label="状态"><StatusBadge value={String(latestSoak.status ?? 'UNAVAILABLE')} /></Descriptions.Item><Descriptions.Item label="累计时间">{String(latestSoak.accumulated_seconds ?? 'UNAVAILABLE')}</Descriptions.Item><Descriptions.Item label="最近观测">{formatTime(latestSoak.updated_at ?? latestSoak.last_heartbeat)}</Descriptions.Item></Descriptions>{snapshot.data.soak_violations.length > 0 && <Alert type="error" showIcon title={`${snapshot.data.soak_violations.length} 条稳定性违规`} description="违规记录不会被隐藏或自动清除。" />}</> : <Empty description="稳定性观察尚未启动；这不代表已经通过门禁。" />}</Card> },
      { key: 'diagnostics', label: '数据与诊断', children: <DataDiagnostics snapshot={snapshot} /> },
    ]} />
  </section>
}
