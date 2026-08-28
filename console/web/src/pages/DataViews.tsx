import { Alert, Button, Card, Col, Descriptions, Empty, Input, Progress, Row, Select, Statistic, Table, Tabs, Tag, Typography } from 'antd'
import { lazy, Suspense, useMemo, useState } from 'react'
import type { ConsoleRow, ConsoleSnapshot } from '../types'
import { StatusBadge } from '../components/StatusBadge'
import { formatDuration, formatTime, numberValue, scoreText, shortIdentity, taskDetail, taskKindLabel, taskNeedsAttention, taskSummaries, type TaskSummary } from '../presentation'

const { Paragraph, Text, Title } = Typography
const TaskDetailPage = lazy(() => import('../components/TaskDetail').then((module) => ({ default: module.TaskDetailPage })))

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

export function RunRecordsPage({ snapshot, canWrite = false, mode = 'light', selectedTaskId, onOpenTask, onBack, onCreate }: { snapshot: ConsoleSnapshot; canWrite?: boolean; mode?: 'dark' | 'light'; selectedTaskId?: string; onOpenTask: (id: string) => void; onBack: () => void; onCreate: () => void }) {
  const tasks = useMemo(() => taskSummaries(snapshot), [snapshot])
  const details = useMemo(() => new Map(tasks.map((task) => [task.id, taskDetail(snapshot, task)])), [snapshot, tasks])
  const [filter, setFilter] = useState('ALL')
  const [search, setSearch] = useState('')
  const selected = selectedTaskId ? details.get(selectedTaskId) : undefined
  if (selected) return <Suspense fallback={<Card loading />}><TaskDetailPage detail={selected} snapshot={snapshot} canWrite={canWrite} mode={mode} onBack={onBack} /></Suspense>
  const normalizedSearch = search.trim().toLowerCase()
  const filtered = tasks.filter((task) => {
    const matchesFilter = filter === 'ALL' || (filter === 'ACTIVE' ? task.status === 'RUNNING' : filter === 'ATTENTION' ? taskNeedsAttention(task.status) : filter === 'DONE' ? ['COMPLETED', 'PROMOTED', 'SUCCEEDED', 'SUCCESS', 'STOPPED'].includes(task.status) : task.kind === filter)
    const matchesSearch = !normalizedSearch || `${task.title} ${task.id}`.toLowerCase().includes(normalizedSearch)
    return matchesFilter && matchesSearch
  })
  const active = tasks.filter((task) => task.status === 'RUNNING').length
  const attention = tasks.filter((task) => taskNeedsAttention(task.status)).length
  const completed = tasks.filter((task) => ['COMPLETED', 'PROMOTED', 'SUCCEEDED', 'SUCCESS', 'STOPPED'].includes(task.status)).length
  const lease = snapshot.data.resource_leases.find((row) => row.status === 'ACTIVE' || row.status === 'QUARANTINED')
  const columns = [
    { title: '状态', key: 'status', width: 130, render: (_: unknown, task: TaskSummary) => <StatusBadge value={task.status} /> },
    { title: '任务', key: 'task', render: (_: unknown, task: TaskSummary) => <div className="run-name-cell"><strong>{task.title}</strong><small>{taskKindLabel(task)} · {shortIdentity(task.id, 18)}</small></div> },
    { title: '算子与语言', key: 'operator', width: 185, render: () => <div className="run-fact-cell"><strong>Fused MoE I8 TN</strong><small>Triton</small></div> },
    { title: '当前阶段', key: 'stage', width: 140, render: (_: unknown, task: TaskSummary) => details.get(task.id)?.currentStage ?? '等待开始' },
    { title: '最佳结果', key: 'result', width: 130, render: (_: unknown, task: TaskSummary) => scoreText(details.get(task.id)?.bestScore ?? null) },
    { title: '进度', key: 'progress', width: 150, render: (_: unknown, task: TaskSummary) => <Progress aria-label={`${task.title} 完成进度`} size="small" percent={details.get(task.id)?.progress ?? 0} /> },
    { title: '最近更新', key: 'updated', width: 150, render: (_: unknown, task: TaskSummary) => formatTime(task.updatedAt) },
  ]
  return <section aria-labelledby="runs-title">
    <div className="page-heading"><div><Text className="eyebrow">算子优化</Text><Title id="runs-title" level={2}>运行记录</Title><Paragraph type="secondary">查看每项任务的进度、最佳结果和完整优化过程。</Paragraph></div><Button type="primary" onClick={onCreate}>新建任务</Button></div>
    <Row gutter={[16, 16]} className="run-overview">
      <Col xs={12} xl={6}><Card><Statistic title="正在运行" value={active} suffix="项" /></Card></Col>
      <Col xs={12} xl={6}><Card><Statistic title="需要处理" value={attention} suffix="项" /></Card></Col>
      <Col xs={12} xl={6}><Card><Statistic title="最近完成" value={completed} suffix="项" /></Card></Col>
      <Col xs={12} xl={6}><Card><Statistic title="GPU 状态" value={lease?.status === 'QUARANTINED' ? '需要检查' : lease ? '使用中' : '空闲'} /></Card></Col>
    </Row>
    <Card className="run-records-card">
      <div className="run-toolbar"><Input.Search allowClear aria-label="搜索运行记录" placeholder="搜索任务名称或编号" value={search} onChange={(event) => setSearch(event.target.value)} /><Select aria-label="筛选运行记录" value={filter} onChange={setFilter} options={[{ value: 'ALL', label: '全部任务' }, { value: 'ACTIVE', label: '正在运行' }, { value: 'ATTENTION', label: '需要处理' }, { value: 'DONE', label: '已完成' }, { value: 'LONG', label: '长期优化' }, { value: 'RUN', label: '短期任务' }, { value: 'BENCHMARK', label: '策略对照' }]} /></div>
      <Table rowKey={(task) => `${task.kind}-${task.id}`} dataSource={filtered} columns={columns} scroll={{ x: 1080 }} pagination={{ pageSize: 12, hideOnSinglePage: true }} locale={{ emptyText: <Empty description="没有符合条件的任务"><Button type="primary" onClick={onCreate}>新建任务</Button></Empty> }} onRow={(task) => ({ onClick: () => onOpenTask(task.id), onKeyDown: (event) => { if (event.key === 'Enter' || event.key === ' ') onOpenTask(task.id) }, tabIndex: 0, role: 'button', 'aria-label': `查看任务 ${task.title}` })} />
    </Card>
  </section>
}

function DataDiagnostics({ snapshot }: { snapshot: ConsoleSnapshot }) {
  const tables = [['runs', '任务', snapshot.data.runs], ['iterations', '优化轮次', snapshot.data.iterations], ['attempts', '评测步骤', snapshot.data.evaluation_attempts], ['experiments', '评测结果', snapshot.data.experiments], ['relations', '结果关系', snapshot.data.experiment_relations], ['campaigns', '长期任务', snapshot.data.campaigns], ['children', '子任务', snapshot.data.child_runs], ['leases', 'GPU 资源记录', snapshot.data.resource_leases], ['budgets', '预算记录', snapshot.data.budget_actions], ['soak', '稳定性测试', snapshot.data.soak_generations], ['violations', '异常记录', snapshot.data.soak_violations]] as const
  return <Card title="原始数据" extra={<Tag>高级功能</Tag>}><Paragraph type="secondary">用于排查问题。日常查看请使用运行记录和系统状态。</Paragraph><Tabs className="diagnostic-tabs" tabPlacement="start" items={tables.map(([key, label, rows]) => ({ key, label: `${label} (${rows.length})`, children: <GenericTable rows={rows} identity={key === 'experiments' ? 'experiment_uid' : 'id'} /> }))} /></Card>
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
      <Col xs={24} xl={12}><Card title="长期任务预算"><Row gutter={[12, 12]}><Col span={12}><Statistic title="已使用" value={formatDuration(actualGpu)} /></Col><Col span={12}><Statistic title="已预留" value={formatDuration(reservedGpu)} /></Col></Row><Progress aria-label="长期任务预算使用进度" className="usage-progress" percent={reservedGpu ? Math.min(100, Math.round(actualGpu / reservedGpu * 100)) : 0} format={(value) => `已使用 ${value}%`} /><Text type="secondary">仅统计系统已经记录的长期任务。</Text></Card></Col>
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
      { key: 'soak', label: '稳定性测试', children: <Card title="24 / 72 / 168 小时稳定性测试">{latestSoak ? <><Row gutter={[16, 16]}><Col xs={24} md={8}><Statistic title="当前阶段" value={String(latestSoak.stage ?? '等待开始')} /></Col><Col xs={24} md={8}><Statistic title="已累计" value={formatDuration(numberValue(latestSoak.accumulated_seconds) * 1000)} /></Col><Col xs={24} md={8}><Statistic title="目标时长" value={formatDuration(numberValue(latestSoak.required_seconds) * 1000)} /></Col></Row><Progress aria-label="稳定性测试完成进度" percent={numberValue(latestSoak.required_seconds) ? Math.min(100, Math.round(numberValue(latestSoak.accumulated_seconds) / numberValue(latestSoak.required_seconds) * 100)) : 0} /><Descriptions column={{ xs: 1, md: 2 }}><Descriptions.Item label="状态"><StatusBadge value={String(latestSoak.status ?? 'UNAVAILABLE')} /></Descriptions.Item><Descriptions.Item label="最近记录">{latestSoak.last_heartbeat_epoch ? new Date(numberValue(latestSoak.last_heartbeat_epoch) * 1000).toLocaleString('zh-CN') : '—'}</Descriptions.Item></Descriptions>{snapshot.data.soak_violations.length > 0 && <Alert type="error" showIcon title={`${snapshot.data.soak_violations.length} 条异常记录`} />}</> : <Empty description="稳定性测试尚未开始" />}</Card> },
      { key: 'diagnostics', label: '原始数据', children: <DataDiagnostics snapshot={snapshot} /> },
    ]} />
  </section>
}
