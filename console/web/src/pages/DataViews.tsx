import { Alert, Card, Col, Descriptions, Empty, Progress, Row, Select, Space, Statistic, Table, Tabs, Tag, Timeline, Typography } from 'antd'
import { useMemo, useState, type ReactNode } from 'react'
import type { ConsoleRow, ConsoleSnapshot } from '../types'
import { StatusBadge } from '../components/StatusBadge'
import { OperationButton } from '../components/OperationButton'
import { formatTime, shortIdentity, stageLabel, taskSummaries, type TaskSummary } from '../presentation'

const { Paragraph, Text, Title } = Typography

function numberValue(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0
}

function formatDuration(milliseconds: unknown): string {
  const value = numberValue(milliseconds)
  if (value <= 0) return '0 秒'
  if (value >= 3_600_000) return `${(value / 3_600_000).toFixed(2)} 小时`
  if (value >= 60_000) return `${(value / 60_000).toFixed(1)} 分钟`
  return `${(value / 1000).toFixed(1)} 秒`
}

function GenericTable({ rows, identity = 'id' }: { rows: ConsoleRow[]; identity?: string }) {
  const keys = Array.from(new Set(rows.flatMap((row) => Object.keys(row)))).slice(0, 10)
  return <Table size="small" scroll={{ x: true }} rowKey={(row) => String(row[identity] ?? row.id ?? JSON.stringify(row))} dataSource={rows} pagination={{ pageSize: 20, hideOnSinglePage: true }} locale={{ emptyText: '暂无记录' }} columns={keys.map((key) => ({
    title: key,
    dataIndex: key,
    render: (value: unknown, row: ConsoleRow) => key === 'status'
      ? <StatusBadge value={String(value)} reason={String(row.reason_code ?? row.reason ?? '')} />
      : <Text code={key.includes('id') || key.includes('digest')}>{value == null ? '—' : String(value)}</Text>,
  }))} />
}

function taskActions(task: TaskSummary, runtime: string, canWrite: boolean): ReactNode {
  const row = task.row
  if (task.kind === 'RUN') return <Space wrap>
    {row.status === 'RUNNING' && <OperationButton kind="RUN_STOP" parameters={{ profile: 'pro', run_id: row.id }} runtimeIdentityDigest={runtime} canWrite={canWrite} label="停止任务" danger />}
    {!['PROMOTED', 'STOPPED', 'FAILED', 'HARD_FAILED', 'BUDGET_EXHAUSTED', 'PROPOSAL_READY'].includes(String(row.status)) && row.status !== 'RUNNING' && <OperationButton kind="RUN_RESUME" parameters={{ profile: 'pro', run_id: row.id }} runtimeIdentityDigest={runtime} canWrite={canWrite} label="恢复任务" />}
  </Space>
  return <Space wrap>
    {row.status === 'CREATED' && <OperationButton kind="CAMPAIGN_START" parameters={{ campaign_id: row.id }} runtimeIdentityDigest={runtime} canWrite={canWrite} label="启动任务" />}
    {row.status === 'RUNNING' && task.kind === 'LONG' && <OperationButton kind="CAMPAIGN_CHILD_EXECUTE" parameters={{ campaign_id: row.id, profile: 'pro', max_candidates: 5, max_wall_seconds: 21600 }} runtimeIdentityDigest={runtime} canWrite={canWrite} label="执行下一轮" />}
    {row.status === 'RUNNING' && task.kind === 'BENCHMARK' && <OperationButton kind="BENCHMARK_EXECUTE" parameters={{ campaign_id: row.id, profile: 'pro' }} runtimeIdentityDigest={runtime} canWrite={canWrite} label="执行对照测试" />}
    {row.status === 'RUNNING' && <OperationButton kind="CAMPAIGN_PAUSE" parameters={{ campaign_id: row.id, reason: 'Console operator pause' }} runtimeIdentityDigest={runtime} canWrite={canWrite} label="暂停任务" danger />}
    {String(row.status).startsWith('PAUSED_') && row.status !== 'PAUSED_DATA_INTEGRITY' && <OperationButton kind="CAMPAIGN_RESUME" parameters={{ campaign_id: row.id, profile: 'pro' }} runtimeIdentityDigest={runtime} canWrite={canWrite} label="检查并恢复" />}
  </Space>
}

function TaskDetail({ task, snapshot, canWrite }: { task: TaskSummary; snapshot: ConsoleSnapshot; canWrite: boolean }) {
  const runtime = snapshot.runtime_identity.runtime_identity_digest
  const children = task.kind === 'RUN' ? [] : snapshot.data.child_runs.filter((row) => String(row.campaign_id) === task.id)
  const runIds = new Set(task.kind === 'RUN' ? [task.id] : children.map((row) => String(row.controller_run_id ?? '')))
  const iterations = snapshot.data.iterations.filter((row) => runIds.has(String(row.run_id ?? '')))
  const iterationIds = new Set(iterations.map((row) => String(row.id)))
  const attempts = snapshot.data.evaluation_attempts.filter((row) => iterationIds.has(String(row.iteration_id ?? '')) || runIds.has(String(row.run_id ?? '')))
  const experimentIds = new Set(attempts.map((row) => String(row.experiment_uid ?? '')).filter(Boolean))
  const experiments = snapshot.data.experiments.filter((row) => experimentIds.has(String(row.experiment_uid ?? '')))
  const budgets = snapshot.data.budget_actions.filter((row) => String(row.campaign_id ?? '') === task.id)
  const completedAttempts = attempts.filter((row) => ['SUCCESS', 'SUCCEEDED'].includes(String(row.status))).length
  const candidateHashes = new Set(iterations.map((row) => String(row.candidate_hash ?? '')).filter(Boolean))
  const latestAttempt = attempts[0]
  const scored = experiments.filter((row) => typeof row.aggregate_score === 'number')
  const bestScore = scored.length ? Math.max(...scored.map((row) => Number(row.aggregate_score))) : null
  const actualGpu = budgets.reduce((total, row) => total + numberValue(row.actual_gpu_ms), 0)
  const reservedGpu = budgets.reduce((total, row) => total + numberValue(row.reserved_gpu_ms), 0)
  const progress = attempts.length ? Math.round(completedAttempts / attempts.length * 100) : 0

  return <div className="task-detail">
    <div className="task-detail-heading"><div><Text className="eyebrow">{task.kind === 'LONG' ? '长期任务' : task.kind === 'BENCHMARK' ? '策略对照' : '短期任务'}</Text><Title level={3}>{task.title}</Title><Paragraph>{task.subtitle}</Paragraph></div><StatusBadge value={task.status} /></div>
    <div className="task-action-strip">{taskActions(task, runtime, canWrite)}{!canWrite && <Text type="secondary">当前连接只允许查看。</Text>}</div>

    <Tabs items={[
      { key: 'overview', label: '任务概览', children: <>
        <Row gutter={[12, 12]} className="task-metrics">
          <Col xs={12} lg={6}><Card size="small"><Statistic title="完成进度" value={progress} suffix="%" /></Card></Col>
          <Col xs={12} lg={6}><Card size="small"><Statistic title="候选代码" value={candidateHashes.size || numberValue(task.row.valid_candidates)} suffix="个" /></Card></Col>
          <Col xs={12} lg={6}><Card size="small"><Statistic title="已完成评测" value={completedAttempts} suffix={`/ ${attempts.length}`} /></Card></Col>
          <Col xs={12} lg={6}><Card size="small"><Statistic title="最佳评分" value={bestScore ?? '暂无'} /></Card></Col>
        </Row>
        <Card className="detail-card" variant="borderless" title="当前进展">
          <Progress percent={progress} status={task.status.includes('FAIL') ? 'exception' : task.status === 'RUNNING' ? 'active' : 'normal'} />
          <Descriptions size="small" column={{ xs: 1, md: 2 }} className="task-facts">
            <Descriptions.Item label="当前步骤">{stageLabel(latestAttempt?.stage ?? iterations[0]?.stage)}</Descriptions.Item>
            <Descriptions.Item label="最近更新">{formatTime(task.updatedAt)}</Descriptions.Item>
            <Descriptions.Item label="算子">Fused MoE I8 TN</Descriptions.Item>
            <Descriptions.Item label="运行环境">Triton · MetaX C500</Descriptions.Item>
            <Descriptions.Item label="任务编号"><Text code copyable>{task.id}</Text></Descriptions.Item>
            <Descriptions.Item label="子任务数量">{children.length}</Descriptions.Item>
          </Descriptions>
        </Card>
      </> },
      { key: 'process', label: '评测进度', children: <Card className="detail-card" variant="borderless">
        {attempts.length === 0 ? <Empty description="任务还没有开始评测" /> : <Timeline items={attempts.slice().reverse().map((row) => ({
          color: String(row.status).includes('FAIL') ? 'red' : ['SUCCEEDED', 'SUCCESS'].includes(String(row.status)) ? 'green' : 'blue',
          children: <div className="evaluation-step"><div><strong>{stageLabel(row.stage)}</strong><StatusBadge value={String(row.status ?? 'UNAVAILABLE')} /></div><Space wrap><Text>测试集：{String(row.suite ?? '—')}</Text><Text>类型：{String(row.replicate_kind ?? '—')}</Text>{row.history_experiment_id ? <Text>结果编号：{String(row.history_experiment_id)}</Text> : null}</Space>{row.error ? <Alert type="error" title={String(row.error)} /> : null}</div>,
        }))} />}
      </Card> },
      { key: 'results', label: '性能结果', children: <Card className="detail-card" variant="borderless">
        {experiments.length === 0 ? <Empty description="评测完成后，这里会显示评分和候选代码信息" /> : <div className="record-list">{experiments.map((row) => <div className="record-row" key={String(row.experiment_uid ?? row.id)}><div className="record-row-main"><Space wrap><strong>{stageLabel(row.replicate_kind ?? row.suite)}</strong><Tag>{row.aggregate_score == null ? '评分暂不可用' : `评分 ${String(row.aggregate_score)}`}</Tag></Space><div className="result-details"><span>候选代码 <Text code copyable={{ text: String(row.candidate_hash ?? '') }}>{shortIdentity(row.candidate_hash, 14)}</Text></span><span>测试集 {String(row.suite ?? '—')}</span><span>评测后端 {String(row.backend ?? '—')}</span><span>完成时间 {formatTime(row.created_at)}</span></div></div><StatusBadge value={String(row.status ?? 'UNAVAILABLE')} /></div>)}</div>}
      </Card> },
      { key: 'usage', label: '资源用量', children: <Card className="detail-card" variant="borderless">
        {budgets.length === 0 ? <Empty description="当前任务没有单独的长期预算记录" /> : <>
          <Row gutter={[12, 12]}><Col xs={24} md={8}><Statistic title="已使用 GPU 时间" value={formatDuration(actualGpu)} /></Col><Col xs={24} md={8}><Statistic title="已预留 GPU 时间" value={formatDuration(reservedGpu)} /></Col><Col xs={24} md={8}><Statistic title="预算记录" value={budgets.length} suffix="条" /></Col></Row>
          <Progress className="usage-progress" percent={reservedGpu ? Math.min(100, Math.round(actualGpu / reservedGpu * 100)) : 0} format={(value) => `已使用 ${value}%`} />
          <div className="record-list">{budgets.map((row) => <div className="record-row" key={String(row.id)}><div className="record-row-main"><strong>{String(row.action_kind ?? '预算记录')}</strong><Text type="secondary">GPU {formatDuration(row.actual_gpu_ms)} · 运行 {formatDuration(row.actual_wall_ms)} · 候选 {numberValue(row.actual_candidates)} 个</Text></div><StatusBadge value={String(row.status ?? 'UNAVAILABLE')} /></div>)}</div>
        </>}
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
    <div className="page-heading"><div><Text className="eyebrow">任务中心</Text><Title id="tasks-title" level={2}>运行记录与结果</Title><Paragraph type="secondary">选择一个任务，查看进度、性能结果和资源用量。</Paragraph></div><Select aria-label="筛选任务" value={filter} onChange={(value) => { setFilter(value); setSelectedId('') }} options={[{ value: 'ALL', label: '全部任务' }, { value: 'ACTIVE', label: '正在运行' }, { value: 'LONG', label: '长期任务' }, { value: 'RUN', label: '短期任务' }, { value: 'BENCHMARK', label: '策略对照' }]} /></div>
    <Row gutter={[16, 16]}><Col xs={24} xl={8}><Card className="task-list-panel" title={`任务列表 · ${filtered.length}`}>{filtered.length === 0 ? <Empty description="没有符合条件的任务" /> : <div className="task-list">{filtered.map((task) => <button type="button" key={`${task.kind}-${task.id}`} className={`task-list-item ${selected?.id === task.id ? 'selected' : ''}`} onClick={() => setSelectedId(task.id)}><span><strong>{task.title}</strong><small>{formatTime(task.updatedAt)} · {shortIdentity(task.id, 14)}</small></span><StatusBadge value={task.status} /></button>)}</div>}</Card></Col><Col xs={24} xl={16}>{selected ? <Card className="task-detail-panel"><TaskDetail task={selected} snapshot={snapshot} canWrite={canWrite} /></Card> : <Card><Empty description="请选择一个任务" /></Card>}</Col></Row>
  </section>
}

function DataDiagnostics({ snapshot }: { snapshot: ConsoleSnapshot }) {
  const tables = [['runs', '任务', snapshot.data.runs], ['iterations', '优化轮次', snapshot.data.iterations], ['attempts', '评测步骤', snapshot.data.evaluation_attempts], ['experiments', '评测结果', snapshot.data.experiments], ['relations', '结果关系', snapshot.data.experiment_relations], ['campaigns', '长期任务', snapshot.data.campaigns], ['children', '子任务', snapshot.data.child_runs], ['leases', 'GPU 资源记录', snapshot.data.resource_leases], ['budgets', '预算记录', snapshot.data.budget_actions], ['soak', '稳定性测试', snapshot.data.soak_generations], ['violations', '异常记录', snapshot.data.soak_violations]] as const
  return <Card title="原始数据" extra={<Tag>高级功能</Tag>}><Paragraph type="secondary">用于排查问题。日常查看请使用任务中心和系统状态。</Paragraph><Tabs className="diagnostic-tabs" tabPlacement="start" items={tables.map(([key, label, rows]) => ({ key, label: `${label} (${rows.length})`, children: <GenericTable rows={rows} identity={key === 'experiments' ? 'experiment_uid' : 'id'} /> }))} /></Card>
}

function RuntimeStatus({ snapshot }: { snapshot: ConsoleSnapshot }) {
  const identity = snapshot.runtime_identity
  const commitMatches = identity.git_commit === identity.expected_git_commit
  const checks = [
    { title: '控制面代码', healthy: commitMatches, value: identity.git_commit, description: commitMatches ? '正在运行的版本与配置一致' : '运行版本与配置不一致' },
    { title: '评测环境', healthy: Boolean(identity.execution_environment_digest), value: identity.execution_environment_digest, description: 'GPU、框架和工具链环境已识别' },
    { title: '性能分析工具', healthy: Boolean(identity.profiler_activation_profile_digest), value: identity.profiler_activation_profile_digest, description: 'Profiler 配置已加载' },
    { title: '服务器连接', healthy: snapshot.status === 'STABLE', value: identity.runtime_identity_digest, description: snapshot.status === 'STABLE' ? '数据读取正常' : '数据正在同步' },
  ]
  return <Row gutter={[16, 16]}>{checks.map((check) => <Col xs={24} md={12} key={check.title}><Card className="health-check-card"><div className="health-check-title"><strong>{check.title}</strong><Tag color={check.healthy ? 'green' : 'red'}>{check.healthy ? '正常' : '异常'}</Tag></div><Paragraph>{check.description}</Paragraph><Text code copyable={{ text: check.value }}>{shortIdentity(check.value, 22)}</Text></Card></Col>)}</Row>
}

function ResourceStatus({ snapshot }: { snapshot: ConsoleSnapshot }) {
  const leases = snapshot.data.resource_leases
  const latest = leases[0]
  const active = leases.find((row) => row.status === 'ACTIVE')
  const quarantined = leases.find((row) => row.status === 'QUARANTINED')
  const gpuStatus = quarantined ? '需要检查' : active ? '使用中' : '空闲'
  const budgets = snapshot.data.budget_actions
  const actualGpu = budgets.reduce((total, row) => total + numberValue(row.actual_gpu_ms), 0)
  const reservedGpu = budgets.reduce((total, row) => total + numberValue(row.reserved_gpu_ms), 0)
  const actualWall = budgets.reduce((total, row) => total + numberValue(row.actual_wall_ms), 0)
  return <>
    <Row gutter={[16, 16]} className="resource-overview">
      <Col xs={24} md={12} xl={6}><Card><Statistic title="GPU 1" value={gpuStatus} /><StatusBadge value={quarantined ? 'QUARANTINED' : active ? 'ACTIVE' : 'RELEASED'} /></Card></Col>
      <Col xs={24} md={12} xl={6}><Card><Statistic title="当前任务" value={active ? shortIdentity(active.campaign_id, 12) : '无'} /><Text type="secondary">{active ? '正在占用 GPU' : '没有任务占用 GPU'}</Text></Card></Col>
      <Col xs={24} md={12} xl={6}><Card><Statistic title="累计 GPU 时间" value={formatDuration(actualGpu)} /><Text type="secondary">来自已记录任务</Text></Card></Col>
      <Col xs={24} md={12} xl={6}><Card><Statistic title="累计运行时间" value={formatDuration(actualWall)} /><Text type="secondary">来自已记录任务</Text></Card></Col>
    </Row>
    <Row gutter={[16, 16]}>
      <Col xs={24} xl={12}><Card title="GPU 使用记录">{latest ? <Descriptions column={1} size="small"><Descriptions.Item label="最近状态"><StatusBadge value={String(latest.status ?? 'UNAVAILABLE')} /></Descriptions.Item><Descriptions.Item label="所属任务">{String(latest.campaign_id ?? '无')}</Descriptions.Item><Descriptions.Item label="开始时间">{formatTime(latest.acquired_at)}</Descriptions.Item><Descriptions.Item label="释放时间">{formatTime(latest.released_at)}</Descriptions.Item><Descriptions.Item label="说明">{String(latest.reason ?? '—')}</Descriptions.Item></Descriptions> : <Empty description="暂无 GPU 使用记录" />}</Card></Col>
      <Col xs={24} xl={12}><Card title="长期任务预算"><Row gutter={[12, 12]}><Col span={12}><Statistic title="已使用" value={formatDuration(actualGpu)} /></Col><Col span={12}><Statistic title="已预留" value={formatDuration(reservedGpu)} /></Col></Row><Progress className="usage-progress" percent={reservedGpu ? Math.min(100, Math.round(actualGpu / reservedGpu * 100)) : 0} format={(value) => `已使用 ${value}%`} /><Text type="secondary">仅统计系统已经记录的长期任务。</Text></Card></Col>
    </Row>
  </>
}

function ProfilerStatus({ snapshot }: { snapshot: ConsoleSnapshot }) {
  const digest = snapshot.runtime_identity.profiler_activation_profile_digest
  return <Row gutter={[16, 16]}><Col xs={24} xl={10}><Card className="profiler-card"><div className="health-check-title"><Title level={4}>性能分析工具</Title><Tag color={digest ? 'green' : 'red'}>{digest ? '配置已加载' : '不可用'}</Tag></div><Paragraph>Profiler 用于查看编译信息和硬件计数器。它不会自动修改候选代码或部署版本。</Paragraph><Descriptions column={1} size="small"><Descriptions.Item label="运行设备">MetaX C500</Descriptions.Item><Descriptions.Item label="编译信息">metax-compile-metadata-v1</Descriptions.Item><Descriptions.Item label="硬件计数器">metax-hardware-counters-v1</Descriptions.Item><Descriptions.Item label="配置标识"><Text code copyable>{digest}</Text></Descriptions.Item></Descriptions></Card></Col><Col xs={24} xl={14}><Card title="可生成的报告"><Row gutter={[12, 12]}><Col xs={24} md={12}><div className="profiler-capability"><strong>编译信息</strong><span>记录工具链、候选代码和固定测试用例。</span><Tag color="blue">无需 GPU</Tag></div></Col><Col xs={24} md={12}><div className="profiler-capability"><strong>硬件计数器</strong><span>运行固定用例并保存 MetaX trace。</span><Tag color="orange">需要 GPU</Tag></div></Col></Row></Card></Col></Row>
}

export function SystemPage({ snapshot }: { snapshot: ConsoleSnapshot }) {
  const latestSoak = snapshot.data.soak_generations[0]
  return <section aria-labelledby="system-title">
    <div className="page-heading"><div><Text className="eyebrow">系统状态</Text><Title id="system-title" level={2}>服务器与评测环境</Title><Paragraph type="secondary">查看运行版本、GPU 状态、Profiler 和稳定性测试。</Paragraph></div><Tag color={snapshot.status === 'STABLE' ? 'green' : 'orange'}>{snapshot.status === 'STABLE' ? '运行正常' : '正在同步'}</Tag></div>
    <Tabs items={[
      { key: 'health', label: '运行环境', children: <RuntimeStatus snapshot={snapshot} /> },
      { key: 'resources', label: '服务器资源', children: <ResourceStatus snapshot={snapshot} /> },
      { key: 'profiler', label: 'Profiler', children: <ProfilerStatus snapshot={snapshot} /> },
      { key: 'soak', label: '稳定性测试', children: <Card title="24 / 72 / 168 小时稳定性测试">{latestSoak ? <><Row gutter={[16, 16]}><Col xs={24} md={8}><Statistic title="当前阶段" value={String(latestSoak.stage ?? '等待开始')} /></Col><Col xs={24} md={8}><Statistic title="已累计" value={formatDuration(numberValue(latestSoak.accumulated_seconds) * 1000)} /></Col><Col xs={24} md={8}><Statistic title="目标时长" value={formatDuration(numberValue(latestSoak.required_seconds) * 1000)} /></Col></Row><Progress percent={numberValue(latestSoak.required_seconds) ? Math.min(100, Math.round(numberValue(latestSoak.accumulated_seconds) / numberValue(latestSoak.required_seconds) * 100)) : 0} /><Descriptions column={{ xs: 1, md: 2 }}><Descriptions.Item label="状态"><StatusBadge value={String(latestSoak.status ?? 'UNAVAILABLE')} /></Descriptions.Item><Descriptions.Item label="最近记录">{latestSoak.last_heartbeat_epoch ? new Date(numberValue(latestSoak.last_heartbeat_epoch) * 1000).toLocaleString('zh-CN') : '—'}</Descriptions.Item></Descriptions>{snapshot.data.soak_violations.length > 0 && <Alert type="error" showIcon title={`${snapshot.data.soak_violations.length} 条异常记录`} />}</> : <Empty description="稳定性测试尚未开始" />}</Card> },
      { key: 'diagnostics', label: '原始数据', children: <DataDiagnostics snapshot={snapshot} /> },
    ]} />
  </section>
}
