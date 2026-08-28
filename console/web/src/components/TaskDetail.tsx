import { Alert, Button, Card, Col, Collapse, Descriptions, Empty, Modal, Progress, Row, Space, Statistic, Table, Tag, Typography } from 'antd'
import { useMemo, useState, type ReactNode } from 'react'
import type { EChartsCoreOption } from 'echarts/core'
import { fetchScientificArtifact } from '../api'
import { notExposedIterationEvidence } from '../deepEvidence'
import {
  formatDuration,
  formatTime,
  numberValue,
  scoreText,
  shortIdentity,
  stageLabel,
  statusLabel,
  taskKindLabel,
  taskNeedsAttention,
  type OptimizationRoundView,
  type TaskDetailView,
  type TaskSummary,
} from '../presentation'
import type { ConsoleRow, ConsoleSnapshot, ScientificArtifact } from '../types'
import { CodeEditor } from './CodeEditor'
import { DeepEvidencePanel } from './DeepEvidencePanel'
import { EChart } from './EChart'
import { OperationButton } from './OperationButton'
import { StatusBadge } from './StatusBadge'

const { Paragraph, Text, Title } = Typography

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

function finiteScore(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function ExperimentTable({ experiments, compact = false }: { experiments: ConsoleRow[]; compact?: boolean }) {
  if (experiments.length === 0) return <Empty description="没有关联的正式评测记录" />
  return <Table
    className={compact ? 'dense-experiment-table' : undefined}
    size="small"
    rowKey={(row) => String(row.experiment_uid ?? row.id)}
    dataSource={experiments}
    pagination={false}
    scroll={{ x: 760 }}
    columns={[
      { title: '阶段', key: 'stage', width: 120, render: (_value, row) => stageLabel(row.replicate_kind ?? row.suite) },
      { title: '状态', dataIndex: 'status', width: 112, render: (value, row) => <StatusBadge value={String(value ?? 'UNAVAILABLE')} reason={String(row.reason_code ?? row.reason ?? '')} /> },
      { title: '综合评分', dataIndex: 'aggregate_score', width: 126, render: (value) => scoreText(finiteScore(value)) },
      { title: '测试集', dataIndex: 'suite', width: 110, render: (value) => String(value ?? '—') },
      { title: '候选', dataIndex: 'candidate_hash', render: (value) => <Text code copyable={{ text: String(value ?? '') }}>{shortIdentity(value, 16)}</Text> },
      { title: '完成时间', dataIndex: 'created_at', width: 150, render: formatTime },
    ]}
  />
}

function AttemptStrip({ attempts }: { attempts: ConsoleRow[] }) {
  if (attempts.length === 0) return <Text type="secondary">尚无评测 attempt</Text>
  return <div className="attempt-strip">{attempts
    .slice()
    .sort((left, right) => numberValue(left.id) - numberValue(right.id))
    .map((attempt) => <div className="attempt-step" key={String(attempt.id)}>
      <span>{stageLabel(attempt.stage)}</span>
      <StatusBadge value={String(attempt.status ?? 'UNAVAILABLE')} reason={String(attempt.error ?? '')} />
      <small>{String(attempt.suite ?? attempt.replicate_kind ?? '—')}</small>
    </div>)}</div>
}

function RoundPanel({ round, fallbackIndex, onShowArtifact }: { round: OptimizationRoundView; fallbackIndex: number; onShowArtifact: (artifactId: string) => void }) {
  const evidence = notExposedIterationEvidence()
  return <div className="round-detail-grid">
    <div className="round-identity-bar">
      <div><span>候选 SHA-256</span><Text code copyable={{ text: round.candidateHash }}>{shortIdentity(round.candidateHash, 28)}</Text></div>
      <div><span>运行结果</span><strong>{round.outcome ? statusLabel(round.outcome) : '等待结果'}</strong></div>
      <div><span>到达阶段</span><strong>{round.stage}</strong></div>
      <div><span>最近更新</span><strong>{formatTime(round.updatedAt)}</strong></div>
    </div>
    {round.error && <Alert type="error" showIcon title={`第 ${round.index || fallbackIndex} 轮失败`} description={round.error} />}
    <section className="round-subsection">
      <div className="dense-section-heading"><strong>评测链</strong><Text type="secondary">{round.attempts.length} 个 attempt</Text></div>
      <AttemptStrip attempts={round.attempts} />
    </section>
    <section className="round-subsection">
      <div className="dense-section-heading"><strong>正式 History 结果</strong>{round.bestArtifactId && <Button size="small" onClick={() => onShowArtifact(round.bestArtifactId)}>查看本轮候选</Button>}</div>
      <ExperimentTable experiments={round.experiments} compact />
    </section>
    <DeepEvidencePanel evidence={evidence} />
  </div>
}

function scoreChartOption(detail: TaskDetailView): EChartsCoreOption {
  return {
    animation: false,
    grid: { left: 54, right: 20, top: 26, bottom: 42 },
    tooltip: { trigger: 'axis' },
    xAxis: { type: 'category', name: '轮次', data: detail.scoreSeries.map((point) => `#${point.iteration}`) },
    yAxis: { type: 'value', name: '综合评分', scale: true },
    series: [{
      type: 'line',
      name: '正式综合评分',
      data: detail.scoreSeries.map((point) => point.score),
      symbolSize: 7,
      lineStyle: { width: 2 },
      itemStyle: { color: '#2563eb' },
    }],
  }
}

export function TaskDetailPage({
  detail,
  snapshot,
  canWrite,
  mode = 'light',
  onBack,
}: {
  detail: TaskDetailView
  snapshot: ConsoleSnapshot
  canWrite: boolean
  mode?: 'dark' | 'light'
  onBack: () => void
}) {
  const [artifact, setArtifact] = useState<ScientificArtifact>()
  const [artifactLoading, setArtifactLoading] = useState(false)
  const [artifactError, setArtifactError] = useState('')
  const { task } = detail
  const bestExperiment = detail.experiments
    .filter((row) => finiteScore(row.aggregate_score) !== null)
    .sort((left, right) => Number(right.aggregate_score) - Number(left.aggregate_score))[0] ?? detail.experiments[0]
  const bestCandidate = String(bestExperiment?.candidate_hash ?? detail.rounds.find((round) => round.candidateHash)?.candidateHash ?? '')
  const bestArtifactId = String(bestExperiment?.artifact_id ?? '')
  const profilerReady = Boolean(snapshot.runtime_identity.profiler_activation_profile_digest)
  const option = useMemo(() => scoreChartOption(detail), [detail])

  const showArtifact = async (artifactId = bestArtifactId) => {
    if (!artifactId) return
    setArtifactError('')
    setArtifactLoading(true)
    try {
      setArtifact(await fetchScientificArtifact(artifactId))
    } catch (error) {
      setArtifactError(String(error))
    } finally {
      setArtifactLoading(false)
    }
  }
  const downloadArtifact = () => {
    if (!artifact) return
    const url = URL.createObjectURL(new Blob([artifact.source], { type: artifact.media_type }))
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = artifact.entrypoint
    anchor.click()
    URL.revokeObjectURL(url)
  }

  return <section aria-labelledby="task-detail-title" className="task-detail-page dense-task-detail">
    <div className="detail-breadcrumb"><Button onClick={onBack}>返回运行记录</Button><Text type="secondary">运行记录 / 任务详情</Text></div>
    <Card className="task-hero-card dense-hero-card">
      <div className="task-detail-heading"><div><Space wrap><StatusBadge value={task.status} /><Text className="eyebrow">{taskKindLabel(task)}</Text></Space><Title id="task-detail-title" level={2}>{task.title}</Title><Paragraph>{task.subtitle}</Paragraph><Text code copyable>{task.id}</Text></div><div className="task-hero-actions">{taskActions(task, snapshot.runtime_identity.runtime_identity_digest, canWrite)}{!canWrite && <Text type="secondary">当前连接只允许查看</Text>}</div></div>
      <Descriptions className="task-identity-strip" size="small" column={4}>
        <Descriptions.Item label="算子">Fused MoE I8 TN</Descriptions.Item>
        <Descriptions.Item label="实现语言">Triton</Descriptions.Item>
        <Descriptions.Item label="运行设备">MetaX C500</Descriptions.Item>
        <Descriptions.Item label="最近更新">{formatTime(detail.updatedAt)}</Descriptions.Item>
      </Descriptions>
    </Card>

    {taskNeedsAttention(task.status) && <Alert className="section-card" type="error" showIcon title="这项任务需要人工检查" description="运行结果尚不能安全确认，请查看下方失败阶段和已保存的错误信息。禁止根据 UNKNOWN 状态重放任务。" />}

    <div className="dense-metric-strip">
      <div><span>最佳综合评分</span><strong>{scoreText(detail.bestScore)}</strong></div>
      <div><span>候选代码</span><strong>{detail.candidateCount} 个</strong></div>
      <div><span>完成评测</span><strong>{detail.completedAttempts} / {detail.attempts.length}</strong></div>
      <div><span>当前步骤</span><strong>{detail.currentStage}</strong></div>
      <div><span>GPU 时间</span><strong>{formatDuration(detail.actualGpuMs)}</strong></div>
      <div><span>Token</span><strong>{detail.actualTokens.toLocaleString('zh-CN')}</strong></div>
    </div>

    <Card className="detail-section dense-card" title="执行进度" extra={<Text strong>{detail.progress}%</Text>}>
      <Progress aria-label={`任务完成进度 ${detail.progress}%`} percent={detail.progress} status={taskNeedsAttention(task.status) ? 'exception' : task.status === 'RUNNING' ? 'active' : 'normal'} showInfo={false} />
      <div className="dense-stage-flow">{detail.stages.map((stage, index) => <div className={`dense-stage-node stage-${stage.status.toLowerCase()}`} key={stage.key}>
        <span>{String(index + 1).padStart(2, '0')}</span>
        <div><strong>{stage.label}</strong><small>{stage.total ? `${stage.completed}/${stage.total} 完成` : '尚未执行'}</small></div>
        <StatusBadge value={stage.status} />
      </div>)}</div>
    </Card>

    <Card className="detail-section dense-card" title="实时优化轮次" extra={<Text type="secondary">共 {detail.rounds.length} 轮 · 展开查看完整评测链</Text>}>
      {detail.rounds.length === 0 ? <Empty description="任务还没有生成候选代码" /> : <Collapse className="dense-rounds" items={detail.rounds.map((round, index) => ({
        key: round.id,
        label: <div className="dense-round-summary">
          <strong>第 {round.index || detail.rounds.length - index} 轮</strong>
          <span>{round.stage}</span>
          <span>{shortIdentity(round.candidateHash, 14)}</span>
          <span>{round.bestScore == null ? '评分 UNAVAILABLE' : `最佳 ${scoreText(round.bestScore)}`}</span>
          <span>{round.attempts.length} attempts</span>
          <StatusBadge value={round.status} />
        </div>,
        children: <RoundPanel round={round} fallbackIndex={detail.rounds.length - index} onShowArtifact={(artifactId) => void showArtifact(artifactId)} />,
      }))} />}
    </Card>

    <Row gutter={[12, 12]} className="detail-section-row dense-results-row">
      <Col xs={24} xl={10}><Card className="detail-section dense-card" title="正式评分走势" extra={<Text type="secondary">只包含有限 aggregate score</Text>}>
        {detail.scoreSeries.length
          ? <><EChart option={option} label={`正式综合评分走势：${detail.scoreSeries.length} 个有界数据点，详细数值见正式评测表。`} /><div className="score-point-list">{detail.scoreSeries.map((point) => <span key={`${point.iteration}-${point.experimentUid}`}>#{point.iteration} <strong>{scoreText(point.score)}</strong></span>)}</div></>
          : <Empty description="综合评分 UNAVAILABLE；不会用零替代缺失数据" />}
      </Card></Col>
      <Col xs={24} xl={14}><Card className="detail-section dense-card" title="性能结果" extra={<Text type="secondary">已保存的正式 History 记录</Text>}><ExperimentTable experiments={detail.experiments} /></Card></Col>
    </Row>

    <Row gutter={[12, 12]} className="detail-section-row dense-results-row">
      <Col xs={24} xl={12}><Card className="detail-section dense-card" title="最佳候选代码">{bestCandidate ? <>
        <div className="candidate-identity"><Text code copyable={{ text: bestCandidate }}>{bestCandidate}</Text></div>
        <Space wrap><Button onClick={() => void showArtifact()} loading={artifactLoading} disabled={!bestArtifactId}>查看源代码</Button><Button disabled>与精确 writer parent 比较</Button></Space>
        {!bestArtifactId && <Text className="capability-note" type="secondary">当前记录没有可读取的源码制品。</Text>}
        {artifactError && <Alert className="inline-feedback" type="error" showIcon title="无法读取候选代码" description={artifactError} />}
      </> : <Empty description="当前还没有候选代码" />}</Card></Col>
      <Col xs={24} xl={12}><Card className="detail-section dense-card" title="Profiler 分析"><div className="profiler-task-state"><Tag color={profilerReady ? 'green' : 'red'}>{profilerReady ? '工具已就绪' : '工具不可用'}</Tag><Title level={4}>逐轮父版本指标尚未接入</Title><Paragraph>当前只确认 Profiler 激活身份。每轮 parent 指标将在后续受信只读接口提供后进入深度证据区。</Paragraph></div></Card></Col>
    </Row>

    <Card className="detail-section dense-card" title="资源消耗">
      {detail.budgets.length === 0 ? <Empty description="这项任务没有独立的长期预算记录" /> : <><Row gutter={[12, 12]}><Col span={6}><Statistic title="GPU 时间" value={formatDuration(detail.actualGpuMs)} /></Col><Col span={6}><Statistic title="总运行时间" value={formatDuration(detail.actualWallMs)} /></Col><Col span={6}><Statistic title="Token" value={detail.actualTokens.toLocaleString('zh-CN')} /></Col><Col span={6}><Statistic title="已预留 GPU" value={formatDuration(detail.reservedGpuMs)} /></Col></Row><Progress aria-label="GPU 预算使用进度" className="usage-progress" percent={detail.reservedGpuMs ? Math.min(100, Math.round(detail.actualGpuMs / detail.reservedGpuMs * 100)) : 0} format={(value) => `已使用 ${value}%`} /></>}
    </Card>

    <Collapse className="advanced-evidence" size="small" items={[{ key: 'evidence', label: '高级技术身份', children: <Descriptions column={2} size="small"><Descriptions.Item label="任务编号"><Text code copyable>{task.id}</Text></Descriptions.Item><Descriptions.Item label="运行环境"><Text code copyable>{snapshot.runtime_identity.execution_environment_digest}</Text></Descriptions.Item><Descriptions.Item label="命名空间"><Text code copyable>{snapshot.runtime_identity.namespace_id}</Text></Descriptions.Item><Descriptions.Item label="Profiler 配置"><Text code copyable>{snapshot.runtime_identity.profiler_activation_profile_digest}</Text></Descriptions.Item><Descriptions.Item label="评测关系">{detail.relations.length} 条</Descriptions.Item><Descriptions.Item label="子任务">{detail.children.length} 个</Descriptions.Item></Descriptions> }]} />

    <Modal open={Boolean(artifact)} onCancel={() => setArtifact(undefined)} footer={<Space><Button onClick={() => setArtifact(undefined)}>关闭</Button><Button type="primary" onClick={downloadArtifact}>下载 {artifact?.entrypoint ?? ''}</Button></Space>} width={1040} title="候选源代码">
      {artifact && <><Descriptions size="small" column={2}><Descriptions.Item label="文件">{artifact.entrypoint}</Descriptions.Item><Descriptions.Item label="制品"><Text code copyable>{artifact.artifact_id}</Text></Descriptions.Item></Descriptions><CodeEditor value={artifact.source} onChange={() => undefined} readOnly theme={mode} /></>}
    </Modal>
  </section>
}
