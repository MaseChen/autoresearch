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
  { label: 'AI 生成并评测', value: 'RUN_START', description: '让模型生成候选代码，并完成一次评测。', badge: '自主' },
  { label: '评测已有代码', value: 'MANUAL_EVALUATION_START', description: '上传或编辑 kernel.py，测试正确性和性能。', badge: '推荐' },
  { label: '模型策略对照', value: 'BENCHMARK_INIT', description: '比较 Pro 和 Flash 两种模型策略。', badge: '对照' },
]

const longTasks = [
  { label: '持续自主优化', value: 'CAMPAIGN_CREATE' as TaskKind, description: '让系统连续尝试多个候选，直到预算用完或手动停止。', badge: '长期' },
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
  { name: 'TileLang', code: 'TL', description: '暂未开放', enabled: false },
  { name: 'MACA CUDA', code: 'MC', description: '暂未开放', enabled: false },
]

const chains: Record<TaskKind, string[]> = {
  RUN_START: ['确认任务配置', '模型生成候选代码', '检查代码规则', '测试正确性与性能', '保存结果'],
  MANUAL_EVALUATION_START: ['读取 kernel.py', '检查代码规则', '快速试跑', '完整性能测试', '再次确认结果', '保存报告'],
  CAMPAIGN_CREATE: ['确认目标与预算', '创建第一轮优化', '生成并筛选候选', '测试最佳候选', '等待下一轮或人工确认'],
  BENCHMARK_INIT: ['确认对照设置', '运行 Pro 模型策略', '运行 Flash 模型策略', '重复测试', '生成对比报告'],
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

export function CreateTask({ canWrite, runtimeIdentityDigest, mode }: { canWrite: boolean; runtimeIdentityDigest: string; mode: 'dark' | 'light' }) {
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
          <Paragraph type="secondary">选择任务类型、算子和模型，右侧会同步显示执行步骤。</Paragraph>
        </div>
        <Tag color="blue">Fused MoE · Triton · MetaX C500</Tag>
      </div>
      {!canWrite && <Alert type="warning" showIcon title="当前连接暂时不能提交任务，但可以继续查看和编辑配置。" />}

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

            <div className="section-title"><span className="step-number">02</span><div><strong>要做什么</strong></div></div>
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

            <div className="section-title"><span className="step-number">03</span><div><strong>选择算子</strong><Text type="secondary">灰色项目正在开发中</Text></div></div>
            <div className="catalog-grid operator-grid">
              {operators.map((operator) => (
                <Tooltip key={operator.name} title={operator.enabled ? '当前可用' : '暂未开放'}>
                  <button type="button" className={`catalog-card ${operator.enabled ? 'selected' : ''}`} disabled={!operator.enabled || locked}>
                    <span className="catalog-code">{operator.code}</span>
                    <span><strong>{operator.name}</strong><small>{operator.description}</small></span>
                    <Tag color={operator.enabled ? 'blue' : 'default'}>{operator.enabled ? '可用' : '规划中'}</Tag>
                  </button>
                </Tooltip>
              ))}
            </div>

            <div className="section-title"><span className="step-number">04</span><div><strong>实现语言</strong></div></div>
            <div className="catalog-grid technology-grid">
              {implementations.map((implementation) => (
                <Tooltip key={implementation.name} title={implementation.enabled ? '当前可用' : '暂未开放'}>
                  <button type="button" className={`catalog-card ${implementation.enabled ? 'selected' : ''}`} disabled={!implementation.enabled || locked}>
                    <span className="catalog-code">{implementation.code}</span>
                    <span><strong>{implementation.name}</strong><small>{implementation.description}</small></span>
                    <Tag color={implementation.enabled ? 'orange' : 'default'}>{implementation.enabled ? '可用' : '规划中'}</Tag>
                  </button>
                </Tooltip>
              ))}
            </div>

            <div className="section-title"><span className="step-number">05</span><div><strong>选择模型</strong><Text type="secondary">只显示当前可用模型</Text></div></div>
            {kind === 'MANUAL_EVALUATION_START' ? (
              <Alert type="info" showIcon title="这个任务不需要模型，系统会直接评测你提交的代码。" />
            ) : kind === 'BENCHMARK_INIT' ? (
              <Alert type="info" showIcon title="将自动比较 DeepSeek V4 Pro 和 DeepSeek V4 Flash。" />
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

            <div className="section-title"><span className="step-number">06</span><div><strong>代码与运行次数</strong></div></div>
            <Form layout="vertical" className="budget-form">
              {(kind === 'CAMPAIGN_CREATE' || kind === 'BENCHMARK_INIT') && (
                <Form.Item label="本次最多保留多少个候选（1–5）">
                  <InputNumber disabled={locked} min={1} max={5} value={candidateBudget} onChange={(value) => setCandidateBudget(value ?? 1)} />
                </Form.Item>
              )}
              {kind === 'RUN_START' && (
                <Form.Item>
                  <Checkbox disabled={locked} checked={proposalOnly} onChange={(event) => setProposalOnly(event.target.checked)}>
                    只生成并检查代码，不运行 GPU 测试
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
                  <Suspense fallback={<div className="code-editor-loading" role="status">正在加载代码编辑器…</div>}>
                    <CodeEditor value={source} onChange={setSource} readOnly={locked} theme={mode} />
                  </Suspense>
                </>
              )}
            </Form>

            {message && <Alert className="inline-feedback" type={message.startsWith('任务已提交') ? 'success' : 'error'} showIcon title={message} />}
          </Card>
        </Col>

        <Col xs={24} xxl={10}>
          <div className="preview-sticky">
            <Card className="preview-panel" title={<div className="section-title compact"><span className="step-number">07</span><div><strong>任务预览</strong><Text type="secondary">{prepared ? '配置已确认，等待启动' : '尚未提交'}</Text></div></div>} extra={<Tag color={prepared ? 'green' : 'default'}>{prepared ? '待启动' : '草稿'}</Tag>}>
              <Descriptions className="preview-summary" layout="vertical" column={2} size="small">
                <Descriptions.Item label="任务类型">{labels[kind]}</Descriptions.Item>
                <Descriptions.Item label="目标算子">Fused MoE I8 TN</Descriptions.Item>
                <Descriptions.Item label="实现技术栈">Triton</Descriptions.Item>
                <Descriptions.Item label="模型策略">{modelLabel}</Descriptions.Item>
                <Descriptions.Item label="优化目标">系统综合评分 · 正确性优先</Descriptions.Item>
                <Descriptions.Item label="运行环境">MetaX C500 · 正式评测环境</Descriptions.Item>
              </Descriptions>

              <div className="request-preview">
                <div><Text>提交内容</Text><Text type="secondary">确认后不能修改</Text></div>
                <pre>{JSON.stringify({ kind, parameters: redactedPreview(parameters, sourceBytes) }, null, 2)}</pre>
              </div>

              <div className="execution-preview">
                <div className="preview-block-title"><strong>执行步骤</strong></div>
                <ol className="execution-chain">
                  {chains[kind].map((item, index) => <li key={item}><span>{String(index + 1).padStart(2, '0')}</span><div><strong>{item}</strong><small>{index === chains[kind].length - 1 ? '完成后可在运行记录中查看' : '系统自动执行'}</small></div></li>)}
                </ol>
              </div>

              <div className="authority-grid environment-grid">
                <div><small>GPU</small><strong>MetaX C500</strong></div>
                <div><small>开发语言</small><strong>Triton</strong></div>
                <div><small>模型</small><strong>{modelLabel}</strong></div>
                <div><small>GPU 时间上限</small><strong>{kind === 'CAMPAIGN_CREATE' || kind === 'BENCHMARK_INIT' ? `${defaultBudget.gpu_ms / 3_600_000} 小时` : '按任务配置'}</strong></div>
                <div><small>编译与运行框架</small><strong>PyTorch · Triton</strong></div>
                <div><small>代码大小</small><strong>{kind === 'MANUAL_EVALUATION_START' ? `${sourceBytes.toLocaleString()} bytes` : '由模型生成'}</strong></div>
              </div>

              {!prepared ? (
                <Button type="primary" size="large" block disabled={!canPrepare} loading={busy} onClick={() => void prepare()}>
                  检查配置并继续
                </Button>
              ) : (
                <Space orientation="vertical" className="confirmation-box">
                  <Alert type="warning" showIcon title="请再次核对配置。确认后任务会在服务器上创建。" />
                  <Text code copyable>{prepared.operation_digest}</Text>
                  <div className="confirmation-phrase">
                    <Text>请输入以下确认短语：</Text>
                    <Text code copyable>{prepared.confirmation_phrase}</Text>
                  </div>
                  <Space.Compact block>
                    <Input
                      aria-label="确认短语"
                      value={phrase}
                      onChange={(event) => setPhrase(event.target.value)}
                      placeholder="在此输入或粘贴上方短语"
                      status={phrase && phrase !== prepared.confirmation_phrase ? 'error' : undefined}
                    />
                    <Button onClick={() => setPhrase(prepared.confirmation_phrase)}>填入确认短语</Button>
                  </Space.Compact>
                  {phrase && phrase !== prepared.confirmation_phrase && (
                    <Text type="danger">确认短语必须与上方文字逐字一致。</Text>
                  )}
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
