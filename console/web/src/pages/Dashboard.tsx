import { Alert, Card, Col, Descriptions, Row, Statistic, Table, Typography } from 'antd'
import { lazy, Suspense, useMemo } from 'react'
import type { ConsoleSnapshot } from '../types'
import { StatusBadge } from '../components/StatusBadge'

const EChart = lazy(() => import('../components/EChart').then((module) => ({ default: module.EChart })))

const { Text, Title } = Typography

export function Dashboard({ snapshot }: { snapshot: ConsoleSnapshot }) {
  const data = snapshot.data
  const activeRuns = data.runs.filter((row) => row.status === 'RUNNING').length
  const activeCampaigns = data.campaigns.filter((row) => row.status === 'RUNNING').length
  const activeLeases = data.resource_leases.filter((row) => row.status === 'ACTIVE').length
  const blocked = [...data.runs, ...data.campaigns].filter((row) =>
    String(row.status ?? '').includes('UNKNOWN') || String(row.status ?? '').includes('HARD'),
  ).length
  const option = useMemo(() => ({
    tooltip: {},
    xAxis: { type: 'category', data: data.experiments.slice(0, 12).reverse().map((row) => row.id) },
    yAxis: { type: 'value', name: 'aggregate score' },
    series: [{
      type: 'line',
      data: data.experiments.slice(0, 12).reverse().map((row) => row.aggregate_score ?? null),
      symbolSize: 8,
    }],
  }), [data.experiments])

  return (
    <section aria-labelledby="dashboard-title">
      <div className="page-heading">
        <div>
          <Title id="dashboard-title" level={2}>运行总览</Title>
          <Text type="secondary">创建任务 → 观察过程 → 查看结果</Text>
        </div>
        <StatusBadge value={snapshot.status} />
      </div>
      {snapshot.status !== 'STABLE' && (
        <Alert type="warning" showIcon title="三库快照正在变化，所有写操作已禁用。" />
      )}
      <Row gutter={[16, 16]} className="metric-grid">
        <Col xs={24} md={12} xl={6}><Card><Statistic title="活动 Run" value={activeRuns} /></Card></Col>
        <Col xs={24} md={12} xl={6}><Card><Statistic title="活动 Campaign" value={activeCampaigns} /></Card></Col>
        <Col xs={24} md={12} xl={6}><Card><Statistic title="GPU Active Lease" value={activeLeases} /></Card></Col>
        <Col xs={24} md={12} xl={6}><Card><Statistic title="需人工处理" value={blocked} /></Card></Col>
      </Row>
      <Row gutter={[16, 16]}>
        <Col xs={24} xl={15}>
          <Card title="近期科学评分">
            <Suspense fallback={<div className="chart" aria-label="正在加载图表" />}><EChart option={option} label="近期实验 aggregate score 折线图" /></Suspense>
          </Card>
        </Col>
        <Col xs={24} xl={9}>
          <Card title="冻结运行身份">
            <Descriptions column={1} size="small">
              <Descriptions.Item label="Git"><Text code copyable>{snapshot.runtime_identity.git_commit}</Text></Descriptions.Item>
              <Descriptions.Item label="Namespace"><Text code copyable>{snapshot.runtime_identity.namespace_id}</Text></Descriptions.Item>
              <Descriptions.Item label="Environment"><Text code copyable>{snapshot.runtime_identity.execution_environment_digest}</Text></Descriptions.Item>
              <Descriptions.Item label="Agent"><Text code copyable>{snapshot.runtime_identity.agent_protocol_digest}</Text></Descriptions.Item>
            </Descriptions>
          </Card>
        </Col>
      </Row>
      <Card title="最近任务" className="section-card">
        <Table
          size="small"
          rowKey={(row) => String(row.id)}
          dataSource={data.runs.slice(0, 8)}
          pagination={false}
          columns={[
            { title: 'Run ID', dataIndex: 'id', render: (value: string) => <Text code>{value}</Text> },
            { title: '状态', dataIndex: 'status', render: (value: string) => <StatusBadge value={value} /> },
            { title: '候选', dataIndex: 'valid_candidates' },
            { title: '更新时间', dataIndex: 'updated_at' },
          ]}
        />
      </Card>
    </section>
  )
}
