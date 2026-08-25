import { Alert, Button, Card, Col, Descriptions, Empty, Row, Space, Statistic, Table, Typography } from 'antd'
import { lazy, Suspense, useMemo } from 'react'
import type { ConsoleSnapshot } from '../types'
import { StatusBadge } from '../components/StatusBadge'
import { formatTime, shortIdentity, statusLabel, taskSummaries } from '../presentation'
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
  const quarantinedLeases = data.resource_leases.filter((row) => row.status === 'QUARANTINED').length
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
          <Text className="eyebrow">工作台</Text>
          <Title id="dashboard-title" level={2}>算子优化概览</Title>
          <Paragraph type="secondary">查看正在运行的任务、最近结果和服务器状态。</Paragraph>
        </div>
        <Space wrap>
          <Button onClick={() => onNavigate('tasks')}>查看任务</Button>
          <Button type="primary" onClick={() => onNavigate('create')}>创建优化任务</Button>
        </Space>
      </div>

      {snapshot.status !== 'STABLE' && <Alert type="warning" showIcon title="数据正在同步，暂时不能提交新任务。" />}
      {blocked > 0 && <Alert className="section-card" type="error" showIcon title={`${blocked} 个任务需要人工处理`} description="请进入任务中心查看失败阶段和已保存的运行记录。" />}

      <Row gutter={[16, 16]} className="metric-grid">
        <Col xs={24} md={12} xl={6}><Card className="metric-card"><Statistic title="正在运行" value={activeTasks.length} suffix="项" /><Text type="secondary">所有优化任务</Text></Card></Col>
        <Col xs={24} md={12} xl={6}><Card className="metric-card"><Statistic title="已完成评测" value={data.experiments.length} suffix="次" /><Text type="secondary">可在任务中心查看结果</Text></Card></Col>
        <Col xs={24} md={12} xl={6}><Card className="metric-card"><Statistic title="GPU 状态" value={quarantinedLeases ? '已隔离' : activeLeases ? '使用中' : '空闲'} /><Text type="secondary">{activeLeases ? '正在执行任务' : quarantinedLeases ? '需要人工检查' : '可以接收新任务'}</Text></Card></Col>
        <Col xs={24} md={12} xl={6}><Card className="metric-card"><Statistic title="稳定性测试" value={latestSoak ? statusLabel(String(latestSoak.status ?? 'RUNNING')) : '未开始'} /><Text type="secondary">24 / 72 / 168 小时</Text></Card></Col>
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
          <Card title="当前运行环境" extra={<Button type="link" onClick={() => onNavigate('system')}>查看详情</Button>}>
            <Descriptions column={1} size="small" className="identity-summary">
              <Descriptions.Item label="控制面版本"><Text code copyable={{ text: snapshot.runtime_identity.git_commit }}>{shortIdentity(snapshot.runtime_identity.git_commit, 12)}</Text></Descriptions.Item>
              <Descriptions.Item label="评测环境"><StatusBadge value={snapshot.runtime_identity.execution_environment_digest ? 'QUALIFIED' : 'UNAVAILABLE'} /></Descriptions.Item>
              <Descriptions.Item label="Profiler 配置"><StatusBadge value={snapshot.runtime_identity.profiler_activation_profile_digest ? 'ACTIVE' : 'UNAVAILABLE'} /></Descriptions.Item>
              <Descriptions.Item label="数据状态"><StatusBadge value={snapshot.status} /></Descriptions.Item>
            </Descriptions>
            <Alert type="success" showIcon title="关键配置已加载" description="版本、评测环境和性能分析配置均已识别。" />
          </Card>
        </Col>
      </Row>

      <Card title="近期评分变化" className="section-card">
        {recentExperiments.length === 0 ? <Empty description="完成评测后，这里会显示评分变化。" /> : (
          <>
            <Suspense fallback={<div className="chart" role="status" aria-label="正在加载图表" />}><EChart option={option} label={chartSummary} /></Suspense>
            <details className="chart-data">
              <summary>查看详细数据</summary>
              <Table size="small" rowKey={(row) => String(row.id)} dataSource={chartRows} pagination={false} columns={[
                { title: '评测编号', dataIndex: 'id' },
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
