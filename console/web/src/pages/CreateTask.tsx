import {
  Alert,
  Button,
  Card,
  Checkbox,
  Col,
  Descriptions,
  Form,
  Input,
  InputNumber,
  Radio,
  Row,
  Segmented,
  Select,
  Space,
  Tag,
  Tooltip,
  Typography,
  Upload,
} from 'antd'
import { lazy, Suspense, useMemo, useState } from 'react'
import type { PreparedOperation, TaskKind } from '../types'
import { confirmOperation, prepareOperation } from '../api'
import { buildOperationParameters, defaultBudget, type TrustedProfile } from '../operationDrafts'

const { Paragraph, Text, Title } = Typography
const CodeEditor = lazy(() => import('../components/CodeEditor').then((module) => ({ default: module.CodeEditor })))

type TaskCycle = 'LONG' | 'SHORT'

const shortTasks: { label: string; value: TaskKind; description: string; badge: string }[] = [
  { label: 'AI 生成并评测', value: 'RUN_START', description: '由受信模型生成候选，完成一次有界研究任务。', badge: '自主' },
  { label: '评测已有代码', value: 'MANUAL_EVALUATION_START', description: '提交 kernel.py，产生一条完整 CURRENT 科学证据链。', badge: '推荐' },
  { label: '模型策略对照', value: 'BENCHMARK_INIT', description: '冻结 Pro / Flash 对照组，生成可信比较报告。', badge: '对照' },
]

const longTasks = [
  { label: '持续自主优化', value: 'CAMPAIGN_CREATE' as TaskKind, description: '在总预算内运行多个子任务，支持暂停、恢复与 checkpoint。', badge: '长期' },
]

const operators = [
  { name: 'Fused MoE I8 TN', code: 'MOE', description: 'MetaX C500 · 当前受信算子', enabled: true },
  { name: 'FlashInfer Ragged Prefill', code: 'FI-RP', description: 'Ragged KV prefill', enabled: false },
  { name: 'FlashInfer Paged Prefill', code: 'FI-PP', description: 'Paged KV prefill', enabled: false },
  { name: 'FlashInfer Paged Decode', code: 'FI-PD', description: 'Paged KV decode', enabled: false },
  { name: 'FlashInfer Paged MLA Attention', code: 'FI-MLA', description: 'Paged MLA attention', enabled: false },
  { name: 'FlashAttention KV Cache Decode', code: 'FA-KV', description: 'KV cache decode', enabled: false },
]

const implementations = [
  { name: 'Triton', code: 'TR', description: '当前唯一受信实现技术栈', enabled: true },
  { name: 'TileLang', code: 'TL', description: '尚无受信 profile', enabled: false },
  { name: 'MACA CUDA', code: 'MC', description: '尚无受信 profile', enabled: false },
]

const chains: Record<TaskKind, string[]> = {
  RUN_START: ['冻结任务身份', '受信模型生成候选', '策略检查', '多级评测', '生成科学证据'],
  MANUAL_EVALUATION_START: ['候选字节与制品校验', '策略检查', '冒烟评测', '快速评测', '完整主评测', '确认评测', '生成科学证据'],
  CAMPAIGN_CREATE: ['冻结长期任务与预算', '创建子任务', '生成并筛选候选', '完整评测与 checkpoint', '等待人工谱系复证'],
  BENCHMARK_INIT: ['冻结反馈与对照组', '执行 Pro / Flash arms', '重复评测', '用量与证据对账', '生成可信比较报告'],
}

const labels: Record<TaskKind, string> = {
  RUN_START: '单次自主优化',
  MANUAL_EVALUATION_START: '已有代码评测',
  CAMPAIGN_CREATE: '长期持续优化',
  BENCHMARK_INIT: '模型策略对照',
}

function redactedPreview(value: Record<string, unknown> | null, sourceBytes: number): Record<string, unknown> {
  if (!value) return { status: '等待补全必填配置' }
  if (!('candidate' in value)) return value
  return {
    candidate: {
      format: 'source-bundle-v1',
      entrypoint: 'kernel.py',
      files: [{ path: 'kernel.py', media_type: 'text/x-python', content: `<${sourceBytes.toLocaleString()} bytes>` }],
    },
  }
}

export function CreateTask({ canWrite, runtimeIdentityDigest }: { canWrite: boolean; runtimeIdentityDigest: string }) {
  const [cycle, setCycle] = useState<TaskCycle>('SHORT')
  const [kind, setKind] = useState<TaskKind>('MANUAL_EVALUATION_START')
  const [source, setSource] = useState('')
  const [profile, setProfile] = useState<TrustedProfile>('pro')
  const [candidateBudget, setCandidateBudget] = useState(1)
  const [proposalOnly, setProposalOnly] = useState(false)
  const [repetitions, setRepetitions] = useState(2)
  const [prepared, setPrepared] = useState<PreparedOperation | null>(null)
  const [phrase, setPhrase] = useState('')
  const [message, setMessage] = useState('')
  const [busy, setBusy] = useState(false)
  const sourceBytes = useMemo(() => new TextEncoder().encode(source).length, [source])
  const taskOptions = cycle === 'LONG' ? longTasks : shortTasks

  const parameters = useMemo(() => {
    try {
      return buildOperationParameters({ kind, source, profile, candidateBudget, proposalOnly, repetitions })
    } catch {
      return null
    }
  }, [candidateBudget, kind, profile, proposalOnly, repetitions, source])

  const switchCycle = (value: TaskCycle) => {
    setCycle(value)
    setKind(value === 'LONG' ? 'CAMPAIGN_CREATE' : 'MANUAL_EVALUATION_START')
    setPrepared(null)
    setPhrase('')
    setMessage('')
  }

  const prepare = async () => {
    setBusy(true)
    try {
      const payload = buildOperationParameters({ kind, source, profile, candidateBudget, proposalOnly, repetitions })
      setPrepared(await prepareOperation(kind, payload, runtimeIdentityDigest))
      setPhrase('')
      setMessage('')
    } catch (error: unknown) {
      setMessage(String(error))
    } finally {
      setBusy(false)
    }
  }

  const confirm = async () => {
    if (!prepared) return
    setBusy(true)
    try {
      const receipt = await confirmOperation(prepared, phrase)
      setMessage(`任务已提交，远端状态：${receipt.status}。`)
    } catch (error: unknown) {
      setMessage(String(error))
    } finally {
      setBusy(false)
    }
  }

  const locked = prepared !== null
  const canPrepare = canWrite && parameters !== null && sourceBytes <= 262_144
  const modelLabel = kind === 'MANUAL_EVALUATION_START'
    ? '无需模型 · 直接评测候选'
    : profile === 'pro' ? 'DeepSeek V4 Pro' : 'DeepSeek V4 Flash'

  return (
    <section aria-labelledby="create-title" className="create-page">
      <div className="page-heading">
        <div>
          <Title id="create-title" level={2}>创建优化任务</Title>
          <Paragraph type="secondary">选择目标，系统会把配置冻结成不可变请求；右侧始终展示实际执行方式与权限边界。</Paragraph>
        </div>
        <Tag color="blue">Fused MoE · Triton · CURRENT</Tag>
      </div>
      {!canWrite && <Alert type="warning" showIcon title="当前快照不可写、连接滞后或屏幕宽度不足；可浏览配置，但不能提交。" />}

      <Row gutter={[20, 20]} align="top">
        <Col xs={24} xxl={14}>
          <Card className="configuration-panel">
            <div className="section-title"><span className="step-number">01</span><div><strong>任务周期</strong><Text type="secondary">先选择持续优化，或一次有明确终点的实验</Text></div></div>
            <Segmented
              block
              size="large"
              disabled={locked}
              value={cycle}
              onChange={(value) => switchCycle(value as TaskCycle)}
              options={[
                { value: 'LONG', label: '长期任务 · 持续优化' },
                { value: 'SHORT', label: '短期任务 · 单次或批次验证' },
              ]}
            />

            <div className="section-title"><span className="step-number">02</span><div><strong>实验目的</strong><Text type="secondary">底层可信操作仍保持独立</Text></div></div>
            <Radio.Group
              value={kind}
              disabled={locked}
              onChange={(event) => {
                setKind(event.target.value as TaskKind)
                setPrepared(null)
                setMessage('')
              }}
              className="scenario-grid"
            >
              {taskOptions.map((item) => (
                <Radio.Button value={item.value} key={item.value}>
                  <span><strong>{item.label}</strong><Tag>{item.badge}</Tag></span>
                  <small>{item.description}</small>
                </Radio.Button>
              ))}
            </Radio.Group>

            <div className="section-title"><span className="step-number">03</span><div><strong>目标算子</strong><Text type="secondary">规划中的算子仅作目录展示，不能提交</Text></div></div>
            <div className="catalog-grid operator-grid">
              {operators.map((operator) => (
                <Tooltip key={operator.name} title={operator.enabled ? '当前受信算子' : '尚无受信 profile，暂不可创建任务'}>
                  <button type="button" className={`catalog-card ${operator.enabled ? 'selected' : ''}`} disabled={!operator.enabled || locked}>
                    <span className="catalog-code">{operator.code}</span>
                    <span><strong>{operator.name}</strong><small>{operator.description}</small></span>
                    <Tag color={operator.enabled ? 'blue' : 'default'}>{operator.enabled ? '可用' : '规划中'}</Tag>
                  </button>
                </Tooltip>
              ))}
            </div>

            <div className="section-title"><span className="step-number">04</span><div><strong>实现技术栈</strong><Text type="secondary">设备、镜像和工具链由系统冻结</Text></div></div>
            <div className="catalog-grid technology-grid">
              {implementations.map((implementation) => (
                <Tooltip key={implementation.name} title={implementation.enabled ? '当前受信技术栈' : '尚无受信 profile，暂不可创建任务'}>
                  <button type="button" className={`catalog-card ${implementation.enabled ? 'selected' : ''}`} disabled={!implementation.enabled || locked}>
                    <span className="catalog-code">{implementation.code}</span>
                    <span><strong>{implementation.name}</strong><small>{implementation.description}</small></span>
                    <Tag color={implementation.enabled ? 'orange' : 'default'}>{implementation.enabled ? '已冻结' : '规划中'}</Tag>
                  </button>
                </Tooltip>
              ))}
            </div>

            <div className="section-title"><span className="step-number">05</span><div><strong>模型与研究策略</strong><Text type="secondary">这里只展示已接入的受信模型</Text></div></div>
            {kind === 'MANUAL_EVALUATION_START' ? (
              <Alert type="info" showIcon title="无需模型：系统直接评测你提交的候选代码。" />
            ) : kind === 'BENCHMARK_INIT' ? (
              <Alert type="info" showIcon title="固定对照组：DeepSeek V4 Pro + DeepSeek V4 Flash，调用方不能替换 arms。" />
            ) : (
              <Form layout="vertical">
                <Form.Item label="候选生成模型">
                  <Select
                    disabled={locked}
                    value={profile}
                    onChange={(value) => setProfile(value)}
                    options={[
                      { value: 'pro', label: 'DeepSeek V4 Pro · 质量优先' },
                      { value: 'flash', label: 'DeepSeek V4 Flash · 速度优先' },
                    ]}
                  />
                </Form.Item>
              </Form>
            )}

            <div className="section-title"><span className="step-number">06</span><div><strong>输入与预算</strong><Text type="secondary">所有上限都必须处于受信边界内</Text></div></div>
            <Form layout="vertical" className="budget-form">
              {(kind === 'CAMPAIGN_CREATE' || kind === 'BENCHMARK_INIT') && (
                <Form.Item label="界面候选预算（1–5）">
                  <InputNumber disabled={locked} min={1} max={5} value={candidateBudget} onChange={(value) => setCandidateBudget(value ?? 1)} />
                </Form.Item>
              )}
              {kind === 'RUN_START' && (
                <Form.Item>
                  <Checkbox disabled={locked} checked={proposalOnly} onChange={(event) => setProposalOnly(event.target.checked)}>
                    只生成候选并执行策略检查，不启动 GPU
                  </Checkbox>
                </Form.Item>
              )}
              {kind === 'BENCHMARK_INIT' && (
                <Form.Item label="每个模型策略的重复次数">
                  <InputNumber disabled={locked} min={1} max={100} value={repetitions} onChange={(value) => setRepetitions(value ?? 2)} />
                </Form.Item>
              )}
              {kind === 'MANUAL_EVALUATION_START' && (
                <>
                  <Form.Item label="上传或编辑 kernel.py">
                    <Upload
                      disabled={locked}
                      accept=".py,text/x-python"
                      maxCount={1}
                      beforeUpload={async (file) => {
                        setSource(await file.text())
                        return false
                      }}
                      showUploadList={false}
                    >
                      <Button disabled={locked}>选择 kernel.py</Button>
                    </Upload>
                    <Text type={sourceBytes > 262_144 ? 'danger' : 'secondary'}> {sourceBytes.toLocaleString()} / 262,144 bytes</Text>
                  </Form.Item>
                  <Suspense fallback={<div className="code-editor" aria-label="正在加载候选编辑器" />}>
                    <CodeEditor value={source} onChange={setSource} readOnly={locked} />
                  </Suspense>
                </>
              )}
            </Form>

            {message && <Alert className="inline-feedback" type={message.startsWith('任务已提交') ? 'success' : 'error'} showIcon title={message} />}
          </Card>
        </Col>

        <Col xs={24} xxl={10}>
          <div className="preview-sticky">
            <Card className="preview-panel" title={<div className="section-title compact"><span className="step-number">07</span><div><strong>任务预览</strong><Text type="secondary">{prepared ? '远端已冻结，等待确认' : '本地草稿，尚未提交'}</Text></div></div>} extra={<Tag color={prepared ? 'green' : 'default'}>{prepared ? '已冻结' : '草稿'}</Tag>}>
              <Descriptions className="preview-summary" layout="vertical" column={2} size="small">
                <Descriptions.Item label="任务类型">{labels[kind]}</Descriptions.Item>
                <Descriptions.Item label="目标算子">Fused MoE I8 TN</Descriptions.Item>
                <Descriptions.Item label="实现技术栈">Triton</Descriptions.Item>
                <Descriptions.Item label="模型策略">{modelLabel}</Descriptions.Item>
                <Descriptions.Item label="优化目标">系统综合评分 · 正确性优先</Descriptions.Item>
                <Descriptions.Item label="运行环境">MetaX C500 · CURRENT</Descriptions.Item>
              </Descriptions>

              <div className="request-preview">
                <div><Text>受信请求预览</Text><Text type="secondary">参数确认后不可修改</Text></div>
                <pre>{JSON.stringify({ kind, parameters: redactedPreview(parameters, sourceBytes) }, null, 2)}</pre>
              </div>

              <div className="execution-preview">
                <div className="preview-block-title"><strong>计划执行链</strong><Text type="secondary">根据任务目的自动生成</Text></div>
                <ol className="execution-chain">
                  {chains[kind].map((item, index) => <li key={item}><span>{String(index + 1).padStart(2, '0')}</span><div><strong>{item}</strong><small>{index === chains[kind].length - 1 ? '完成后提供可审计结果' : '由受信控制面执行'}</small></div></li>)}
                </ol>
              </div>

              <div className="authority-grid">
                <div><small>部署基准</small><strong>不会自动修改</strong></div>
                <div><small>基准演进链</small><strong>{kind === 'CAMPAIGN_CREATE' ? '需要人工复证' : '无修改权限'}</strong></div>
                <div><small>结果未知</small><strong>禁止自动重放</strong></div>
                <div><small>预算</small><strong>{kind === 'CAMPAIGN_CREATE' || kind === 'BENCHMARK_INIT' ? `${defaultBudget.gpu_ms / 3_600_000} GPU 小时上限` : '由正式 profile 冻结'}</strong></div>
              </div>

              {!prepared ? (
                <Button type="primary" size="large" block disabled={!canPrepare} loading={busy} onClick={() => void prepare()}>
                  冻结任务配置并预检
                </Button>
              ) : (
                <Space direction="vertical" className="confirmation-box">
                  <Alert type="warning" showIcon title="远端已冻结请求。确认不会修改参数；网络重试不得创建第二个 GPU 动作。" />
                  <Text code copyable>{prepared.operation_digest}</Text>
                  <Input value={phrase} onChange={(event) => setPhrase(event.target.value)} placeholder={prepared.confirmation_phrase} />
                  <Button danger type="primary" size="large" block disabled={phrase !== prepared.confirmation_phrase || Boolean(message)} loading={busy} onClick={() => void confirm()}>
                    {kind === 'CAMPAIGN_CREATE' ? '创建长期优化任务' : '确认并启动任务'}
                  </Button>
                  {!message && <Button block onClick={() => { setPrepared(null); setPhrase('') }}>返回修改草稿</Button>}
                </Space>
              )}
            </Card>
          </div>
        </Col>
      </Row>
    </section>
  )
}
