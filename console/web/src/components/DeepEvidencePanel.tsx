import { Alert, Card, Collapse, Descriptions, Empty, Space, Tag, Typography } from 'antd'
import type {
  BoundedTextView,
  EvidenceSlot,
  IterationDeepEvidenceView,
  ParentProfilerView,
  WriterOutputView,
} from '../deepEvidence'
import { exposedEvidenceCount } from '../deepEvidence'
import { shortIdentity } from '../presentation'
import { StatusBadge } from './StatusBadge'

const { Paragraph, Text } = Typography

function BoundedText({ value, empty }: { value: EvidenceSlot<BoundedTextView>; empty: string }) {
  if (value.state === 'NOT_EXPOSED') return <Text type="secondary">{empty}</Text>
  if (value.state === 'UNAVAILABLE') return <StatusBadge value="UNAVAILABLE" reason={value.reason} />
  return <div className="bounded-evidence-text">
    <div className="evidence-text-meta">
      <Text code copyable={{ text: value.value.sha256 }}>{shortIdentity(value.value.sha256, 18)}</Text>
      <Text type="secondary">{value.value.byteCount.toLocaleString('zh-CN')} bytes</Text>
      {value.value.truncated && <Tag color="orange">已截断</Tag>}
    </div>
    <pre>{value.value.text}</pre>
  </div>
}

function WriterOutputs({ outputs }: { outputs: WriterOutputView[] }) {
  if (outputs.length === 0) return <Empty description="没有 Writer 输出" />
  return <div className="writer-output-list">{outputs.map((output, index) => <Card size="small" key={`${output.phase}-${output.responseId ?? index}`} title={<Space wrap><strong>{output.phase}</strong><Tag>{output.model ?? '模型未记录'}</Tag></Space>}>
    <Descriptions size="small" column={3}>
      <Descriptions.Item label="Response ID"><Text code copyable>{output.responseId ?? '—'}</Text></Descriptions.Item>
      <Descriptions.Item label="结束原因">{output.finishReason ?? '—'}</Descriptions.Item>
      <Descriptions.Item label="Token">{Object.values(output.usage).reduce((total, value) => total + value, 0).toLocaleString('zh-CN')}</Descriptions.Item>
    </Descriptions>
    <Collapse size="small" items={[
      { key: 'reasoning', label: 'Writer reasoning', children: <BoundedText value={output.reasoning} empty="Provider 未暴露 reasoning" /> },
      { key: 'content', label: 'Writer content', children: <BoundedText value={output.content} empty="Provider 未返回 content" /> },
    ]} />
  </Card>)}</div>
}

function ParentProfiler({ profile }: { profile: ParentProfilerView }) {
  return <>
    <Descriptions size="small" column={2}>
      <Descriptions.Item label="状态"><StatusBadge value={profile.status} /></Descriptions.Item>
      <Descriptions.Item label="缓存">{profile.cacheStatus}</Descriptions.Item>
      <Descriptions.Item label="父版本"><Text code copyable>{profile.parentSha256}</Text></Descriptions.Item>
      <Descriptions.Item label="Profile"><Text code copyable>{profile.profileId}</Text></Descriptions.Item>
      <Descriptions.Item label="晋级权威">否，仅用于方向判断</Descriptions.Item>
      {profile.error && <Descriptions.Item label="错误"><Text type="danger">{profile.error}</Text></Descriptions.Item>}
    </Descriptions>
    <div className="profiler-metric-grid">{profile.metrics.map((metric) => <div key={metric.name}>
      <Text type="secondary">{metric.name}</Text>
      {metric.value == null
        ? <StatusBadge value="UNAVAILABLE" reason={metric.unavailableReason ?? 'COUNTER_NOT_EXPOSED'} />
        : <strong>{metric.value.toLocaleString('zh-CN')} {metric.unit ?? ''}</strong>}
    </div>)}</div>
  </>
}

export function DeepEvidencePanel({ evidence }: { evidence: IterationDeepEvidenceView }) {
  const exposed = exposedEvidenceCount(evidence)
  if (exposed === 0) {
    return <Collapse className="deep-evidence-placeholder" size="small" items={[{
      key: 'future-evidence',
      label: '深度研究证据 · 当前协议未接入',
      children: <Alert
        type="info"
        showIcon
        title="当前 Agent 协议尚未提供逐轮深度证据"
        description="已为 Writer reasoning/content、Planner、Proposal 诊断、Compile repair、精确父本 diff 和父版本 Profiler 预留受控展示接口。未取得可信数据前不会显示模拟内容。"
      />,
    }]} />
  }

  const items = []
  if (evidence.writerOutputs.state === 'AVAILABLE') items.push({
    key: 'writer', label: `Writer 输出 (${evidence.writerOutputs.value.length})`,
    children: <WriterOutputs outputs={evidence.writerOutputs.value} />,
  })
  else if (evidence.writerOutputs.state === 'UNAVAILABLE') items.push({ key: 'writer', label: 'Writer 输出', children: <StatusBadge value="UNAVAILABLE" reason={evidence.writerOutputs.reason} /> })

  if (evidence.planner.state === 'AVAILABLE') {
    const planner = evidence.planner.value
    items.push({ key: 'planner', label: 'Planner 方向与触发原因', children: <>
      <Descriptions size="small" column={3}>
        <Descriptions.Item label="状态"><StatusBadge value={planner.status} /></Descriptions.Item>
        <Descriptions.Item label="策略">{planner.policy ?? '—'}</Descriptions.Item>
        <Descriptions.Item label="耗时">{planner.elapsedSeconds == null ? 'UNAVAILABLE' : `${planner.elapsedSeconds.toFixed(2)} 秒`}</Descriptions.Item>
      </Descriptions>
      <div className="evidence-list"><strong>触发原因</strong>{planner.triggerReasons.length ? <ul>{planner.triggerReasons.map((reason) => <li key={reason}>{reason}</li>)}</ul> : <Text type="secondary">未记录</Text>}</div>
      <div className="evidence-list"><strong>本轮避免方向</strong>{planner.avoid.length ? <ul>{planner.avoid.map((item) => <li key={item}>{item}</li>)}</ul> : <Text type="secondary">未记录</Text>}</div>
      <div className="planner-direction-grid">{planner.directions.map((direction) => <Card size="small" key={direction.id} className={direction.selected ? 'selected-direction' : ''} title={<Space><Text code>{direction.id}</Text>{direction.selected && <Tag color="blue">已选择</Tag>}</Space>}>
        <Descriptions size="small" column={1}>
          <Descriptions.Item label="核心机制">{direction.coreMechanism}</Descriptions.Item>
          <Descriptions.Item label="测量问题">{direction.measuredProblem}</Descriptions.Item>
          <Descriptions.Item label="最小测试">{direction.cheapestTest}</Descriptions.Item>
          <Descriptions.Item label="继续条件">{direction.continueIf}</Descriptions.Item>
          <Descriptions.Item label="停止条件">{direction.stopIf}</Descriptions.Item>
          <Descriptions.Item label="实现风险">{direction.implementationRisk}</Descriptions.Item>
        </Descriptions>
      </Card>)}</div>
      {planner.error && <Alert type="error" showIcon title="Planner 失败" description={planner.error} />}
    </> })
  } else if (evidence.planner.state === 'UNAVAILABLE') items.push({ key: 'planner', label: 'Planner 方向与触发原因', children: <StatusBadge value="UNAVAILABLE" reason={evidence.planner.reason} /> })

  if (evidence.proposalDiagnostic.state === 'AVAILABLE') {
    const proposal = evidence.proposalDiagnostic.value
    items.push({ key: 'proposal', label: 'Proposal 结构化诊断', children: <>
      <Descriptions size="small" column={2}>
        <Descriptions.Item label="状态"><StatusBadge value={proposal.status} /></Descriptions.Item>
        <Descriptions.Item label="置信度">{proposal.confidence ?? 'UNAVAILABLE'}</Descriptions.Item>
        <Descriptions.Item label="算法族">{proposal.algorithmFamily ?? 'UNAVAILABLE'}</Descriptions.Item>
        <Descriptions.Item label="搜索方向">{proposal.searchDirection ?? 'UNAVAILABLE'}</Descriptions.Item>
        <Descriptions.Item label="瓶颈">{proposal.bottleneck ?? 'UNAVAILABLE'}</Descriptions.Item>
        <Descriptions.Item label="Profiler 使用">{proposal.profileUse}</Descriptions.Item>
        <Descriptions.Item label="假设" span={2}>{proposal.hypothesis ?? 'UNAVAILABLE'}</Descriptions.Item>
        <Descriptions.Item label="代码变化" span={2}>{proposal.codeChange ?? 'UNAVAILABLE'}</Descriptions.Item>
        <Descriptions.Item label="实际结果">{proposal.outcome ?? 'UNAVAILABLE'}</Descriptions.Item>
        <Descriptions.Item label="结论">{proposal.conclusion}</Descriptions.Item>
      </Descriptions>
      <div className="proposal-evidence-grid">
        <div className="evidence-list"><strong>Profiler 证据</strong>{proposal.profileEvidence.length ? <ul>{proposal.profileEvidence.map((item) => <li key={item}>{item}</li>)}</ul> : <Text type="secondary">未记录</Text>}</div>
        <div className="evidence-list"><strong>预期指标变化</strong>{proposal.expectedCounterDelta.length ? <ul>{proposal.expectedCounterDelta.map((item) => <li key={item}>{item}</li>)}</ul> : <Text type="secondary">未记录</Text>}</div>
        <div className="evidence-list"><strong>历史方向规避</strong>{proposal.historyAvoidance.length ? <ul>{proposal.historyAvoidance.map((item) => <li key={item}>{item}</li>)}</ul> : <Text type="secondary">未记录</Text>}</div>
      </div>
      {proposal.parseError && <Alert type="error" showIcon title="Proposal 解析失败" description={proposal.parseError} />}
    </> })
  } else if (evidence.proposalDiagnostic.state === 'UNAVAILABLE') items.push({ key: 'proposal', label: 'Proposal 结构化诊断', children: <StatusBadge value="UNAVAILABLE" reason={evidence.proposalDiagnostic.reason} /> })

  if (evidence.compileRepair.state === 'AVAILABLE') {
    const repair = evidence.compileRepair.value
    items.push({ key: 'repair', label: `Compile repair (${repair.attempts.length})`, children: <>
      <Descriptions size="small" column={3}>
        <Descriptions.Item label="初始状态"><StatusBadge value={repair.initialStatus} /></Descriptions.Item>
        <Descriptions.Item label="初始分类">{repair.initialCategory ?? '—'}</Descriptions.Item>
        <Descriptions.Item label="最终状态"><StatusBadge value={repair.finalStatus} /></Descriptions.Item>
      </Descriptions>
      {repair.initialError && <Paragraph type="danger">{repair.initialError}</Paragraph>}
      <div className="repair-timeline">{repair.attempts.map((attempt) => <div className="repair-attempt-row" key={attempt.attempt}>
        <span className="repair-attempt-index">{attempt.attempt}</span>
        <strong>尝试 {attempt.attempt} · {attempt.description ?? attempt.status}</strong>
        <Space wrap><StatusBadge value={attempt.writerStatus} /><StatusBadge value={attempt.screenStatus} /></Space>
        <div className="repair-attempt-detail">
          <Text type="secondary">Writer {attempt.writerSeconds == null ? 'UNAVAILABLE' : `${attempt.writerSeconds.toFixed(2)} 秒`} · Screen {attempt.screenSeconds == null ? 'UNAVAILABLE' : `${attempt.screenSeconds.toFixed(2)} 秒`}</Text>
          <Text code copyable={{ text: attempt.inputSha256 ?? '' }}>输入 {attempt.inputSha256 ? shortIdentity(attempt.inputSha256, 18) : 'UNAVAILABLE'}</Text>
          <Text code copyable={{ text: attempt.candidateSha256 ?? '' }}>候选 {attempt.candidateSha256 ? shortIdentity(attempt.candidateSha256, 18) : 'UNAVAILABLE'}</Text>
        </div>
        {attempt.error && <Text type="danger">{attempt.error}</Text>}
      </div>)}</div>
    </> })
  } else if (evidence.compileRepair.state === 'UNAVAILABLE') items.push({ key: 'repair', label: 'Compile repair', children: <StatusBadge value="UNAVAILABLE" reason={evidence.compileRepair.reason} /> })

  if (evidence.candidateDiff.state === 'AVAILABLE') {
    const diff = evidence.candidateDiff.value
    const identityComplete = Boolean(diff.parentSha256 && diff.candidateSha256)
    items.push({ key: 'diff', label: '候选与精确 writer parent 的 diff', children: identityComplete ? <>
      <Descriptions size="small" column={2}>
        <Descriptions.Item label="Writer parent"><Text code copyable>{diff.parentSha256}</Text></Descriptions.Item>
        <Descriptions.Item label="候选"><Text code copyable>{diff.candidateSha256}</Text></Descriptions.Item>
      </Descriptions>
      <BoundedText value={{ state: 'AVAILABLE', value: diff.diff }} empty="没有文本差异" />
    </> : <Alert type="error" showIcon title="精确父本身份缺失，diff 已拒绝显示" /> })
  } else if (evidence.candidateDiff.state === 'UNAVAILABLE') items.push({ key: 'diff', label: '候选与精确 writer parent 的 diff', children: <StatusBadge value="UNAVAILABLE" reason={evidence.candidateDiff.reason} /> })

  if (evidence.parentProfiler.state === 'AVAILABLE') items.push({ key: 'profile', label: '每轮父版本 Profiler 指标', children: <ParentProfiler profile={evidence.parentProfiler.value} /> })
  else if (evidence.parentProfiler.state === 'UNAVAILABLE') items.push({ key: 'profile', label: '每轮父版本 Profiler 指标', children: <StatusBadge value="UNAVAILABLE" reason={evidence.parentProfiler.reason} /> })

  return <Collapse className="deep-evidence-panel" size="small" items={items} />
}
