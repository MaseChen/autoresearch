import { Alert, Button, Card, Col, Descriptions, Empty, Row, Space, Statistic, Table, Typography } from 'antd'
import { lazy, Suspense, useMemo } from 'react'
import type { ConsoleSnapshot } from '../types'
import { StatusBadge } from '../components/StatusBadge'
import { formatTime, shortIdentity, taskSummaries } from '../presentation'
import {
  aggregateScoreChartSummary,
  aggregateScoreText,
  aggregateScoreValue,
} from '../chartAccessibility'

const EChart = lazy(() => import('../components/EChart').then((module) => ({ default: module.EChart })))

const { Paragraph, Text, Title } = Typography

export function Dashboard({ snapshot, onNavigate }: { snapshot: ConsoleSnapshot; onNavigate: (page: 'create' | 'tasks' | 'system') => void }) {
  const data = snapshot.data
  const tasks = taskSummaries(snapshot)
  const activeTasks = tasks.filter((task) => task.status === 'RUNNING')
  const activeLeases = data.resource_leases.filter((row) => row.status === 'ACTIVE').length
  const blocked = tasks.filter((task) => task.status.includes('UNKNOWN') || task.status.includes('HARD')).length
  const recentExperiments = useMemo(() => data.experiments.slice(0, 12).reverse(), [data.experiments])
  const option = useMemo(() => ({
    tooltip: {},
    xAxis: { type: 'category', data: recentExperiments.map((row) => row.id) },
    yAxis: { type: 'value', name: '科学评分' },
    series: [{ type: 'line', data: recentExperiments.map((row) => aggregateScoreValue(row.aggregate_score)), symbolSize: 8 }],
  }), [recentExperiments])
  const chartRows = recentExperiments.map((row) => ({ id: row.id, aggregate_score: row.aggregate_score, status: row.status }))
  const chartSummary = aggregateScoreChartSummary(recentExperiments)
  const latestSoak = data.soak_generations.at(0)

  return (
    <section aria-labelledby="dashboard-title">
      <div className="page-heading hero-heading">
        <div>
          <Text className="eyebrow">研究工作台</Text>
          <Title id="dashboard-title" level={2}>从任务目标出发，而不是从数据库出发</Title>
          <Paragraph type="secondary">创建任务、观察关键阶段、理解结果。原始账本只保留在系统诊断区。</Paragraph>
        </div>
        <Space wrap>
          <Button onClick={() => onNavigate('tasks')}>查看任务</Button>
          <Button type="primary" onClick={() => onNavigate('create')}>创建优化任务</Button>
        </Space>
      </div>

      {snapshot.status !== 'STABLE' && <Alert type="warning" showIcon title="三库快照正在变化，所有写操作已禁用。" />}
      {blocked > 0 && <Alert className="section-card" type="error" showIcon title={`${blocked} 个任务需要人工处理`} description="UNKNOWN 结果禁止自动重放；请进入任务中心查看已保存证据。" />}

      <Row gutter={[16, 16]} className="metric-grid">
        <Col xs={24} md={12} xl={6}><Card className="metric-card"><Statistic title="正在运行" value={activeTasks.length} suffix="项任务" /><Text type="secondary">单次与长期任务合计</Text></Card></Col>
        <Col xs={24} md={12} xl={6}><Card className="metric-card"><Statistic title="可信科学记录" value={data.experiments.length} suffix="条" /><Text type="secondary">包含评测步骤与确认关系</Text></Card></Col>
        <Col xs={24} md={12} xl={6}><Card className="metric-card"><Statistic title="GPU 资源锁" value={activeLeases} suffix="个活动" /><Text type="secondary">同一时刻不得重叠</Text></Card></Col>
        <Col xs={24} md={12} xl={6}><Card className="metric-card"><Statistic title="稳定性观察" value={latestSoak ? String(latestSoak.status ?? '进行中') : '尚未开始'} /><Text type="secondary">24 / 72 / 168 小时门禁</Text></Card></Col>
      </Row>

      <Row gutter={[16, 16]}>
        <Col xs={24} xl={15}>
          <Card title="最近任务" extra={<Button type="link" onClick={() => onNavigate('tasks')}>进入任务中心</Button>}>
            {tasks.length === 0 ? (
              <Empty description="还没有任务"><Button type="primary" onClick={() => onNavigate('create')}>创建第一个任务</Button></Empty>
            ) : (
              <div className="task-summary-list">
                {tasks.slice(0, 6).map((task) => (
                  <button className="task-summary-row" type="button" key={`${task.kind}-${task.id}`} onClick={() => onNavigate('tasks')}>
                    <span className="task-kind-mark">{task.kind === 'LONG' ? '长' : task.kind === 'BENCHMARK' ? '对' : '短'}</span>
                    <span className="task-summary-main"><strong>{task.title}</strong><small>{task.subtitle}</small></span>
                    <StatusBadge value={task.status} />
                    <Text type="secondary">{formatTime(task.updatedAt)}</Text>
                  </button>
                ))}
              </div>
            )}
          </Card>
        </Col>
        <Col xs={24} xl={9}>
          <Card title="当前运行身份" extra={<Button type="link" onClick={() => onNavigate('system')}>查看门禁</Button>}>
            <Descriptions column={1} size="small" className="identity-summary">
              <Descriptions.Item label="控制面版本"><Text code copyable={{ text: snapshot.runtime_identity.git_commit }}>{shortIdentity(snapshot.runtime_identity.git_commit, 12)}</Text></Descriptions.Item>
              <Descriptions.Item label="科学命名空间"><Text code copyable={{ text: snapshot.runtime_identity.namespace_id }}>{shortIdentity(snapshot.runtime_identity.namespace_id)}</Text></Descriptions.Item>
              <Descriptions.Item label="执行环境"><Text code copyable={{ text: snapshot.runtime_identity.execution_environment_digest }}>{shortIdentity(snapshot.runtime_identity.execution_environment_digest)}</Text></Descriptions.Item>
              <Descriptions.Item label="快照状态"><StatusBadge value={snapshot.status} /></Descriptions.Item>
            </Descriptions>
            <Alert type="info" showIcon title="身份变化会立即禁用写操作" description="页面必须重新握手，避免把任务提交到已经变化的运行环境。" />
          </Card>
        </Col>
      </Row>

      <Card title="科学结果趋势" className="section-card" extra={<Text type="secondary">UNAVAILABLE 不会被画成 0</Text>}>
        {recentExperiments.length === 0 ? <Empty description="完成评测后，这里会展示可信评分趋势。" /> : (
          <>
            <Suspense fallback={<div className="chart" role="status" aria-label="正在加载图表" />}><EChart option={option} label={chartSummary} /></Suspense>
            <details className="chart-data">
              <summary>查看无障碍数据表</summary>
              <Table size="small" rowKey={(row) => String(row.id)} dataSource={chartRows} pagination={false} columns={[
                { title: '科学记录编号', dataIndex: 'id' },
                { title: '综合评分', dataIndex: 'aggregate_score', render: aggregateScoreText },
                { title: '状态', dataIndex: 'status', render: (value: string) => <StatusBadge value={value} /> },
              ]} />
            </details>
          </>
        )}
      </Card>
    </section>
  )
}
