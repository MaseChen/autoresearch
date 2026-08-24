import { Alert, Button, Card, Checkbox, Form, Input, InputNumber, Radio, Select, Space, Steps, Typography, Upload } from 'antd'
import { lazy, Suspense, useMemo, useState } from 'react'
import type { PreparedOperation, TaskKind } from '../types'
import { confirmOperation, prepareOperation } from '../api'
import { buildOperationParameters, type TrustedProfile } from '../operationDrafts'

const { Paragraph, Text, Title } = Typography
const CodeEditor = lazy(() => import('../components/CodeEditor').then((module) => ({ default: module.CodeEditor })))

const taskKinds: { label: string; value: TaskKind; description: string }[] = [
  { label: '自治 Run', value: 'RUN_START', description: '由受信 proposer 生成并评测候选。' },
  { label: '手工 CandidateBundle', value: 'MANUAL_EVALUATION_START', description: '提交 kernel.py 并产生 CURRENT 科学证据。' },
  { label: 'Discovery Campaign', value: 'CAMPAIGN_CREATE', description: '创建有界研究 Campaign。' },
  { label: 'Benchmark Campaign', value: 'BENCHMARK_INIT', description: '执行冻结 cohort 对照。' },
]

export function CreateTask({ canWrite, runtimeIdentityDigest }: { canWrite: boolean; runtimeIdentityDigest: string }) {
  const [step, setStep] = useState(0)
  const [kind, setKind] = useState<TaskKind>('MANUAL_EVALUATION_START')
  const [source, setSource] = useState('')
  const [profile, setProfile] = useState<TrustedProfile>('pro')
  const [candidateBudget, setCandidateBudget] = useState(1)
  const [proposalOnly, setProposalOnly] = useState(false)
  const [repetitions, setRepetitions] = useState(2)
  const [prepared, setPrepared] = useState<PreparedOperation | null>(null)
  const [phrase, setPhrase] = useState('')
  const [message, setMessage] = useState('')
  const sourceBytes = useMemo(() => new TextEncoder().encode(source).length, [source])

  const prepare = async () => {
    try {
      const payload = buildOperationParameters({ kind, source, profile, candidateBudget, proposalOnly, repetitions })
      const operation = await prepareOperation(kind, payload, runtimeIdentityDigest)
      setPrepared(operation)
      setStep(2)
      setMessage('')
    } catch (error: unknown) {
      setMessage(String(error))
    }
  }

  const confirm = async () => {
    if (!prepared) return
    try {
      const receipt = await confirmOperation(prepared, phrase)
      setMessage(`操作 ${prepared.operation_id} 已提交，远端状态：${receipt.status}。`)
      setStep(3)
    } catch (error: unknown) {
      setMessage(String(error))
    }
  }

  return (
    <section aria-labelledby="create-title">
      <Title id="create-title" level={2}>创建任务</Title>
      <Paragraph type="secondary">所有参数先形成草稿；确认后冻结为不可变快照。</Paragraph>
      {!canWrite && <Alert type="warning" showIcon title="当前快照不可写、屏幕过窄或会话未完成可信握手。" />}
      <Steps current={step} items={[{ title: '类型' }, { title: '配置' }, { title: '确认' }, { title: '完成' }]} />
      <Card className="wizard-card">
        {step === 0 && (
          <Radio.Group value={kind} onChange={(event) => setKind(event.target.value as TaskKind)} className="task-types">
            {taskKinds.map((item) => (
              <Radio.Button key={item.value} value={item.value}>
                <strong>{item.label}</strong><span>{item.description}</span>
              </Radio.Button>
            ))}
          </Radio.Group>
        )}
        {step === 1 && (
          <Form layout="vertical">
            {kind !== 'MANUAL_EVALUATION_START' && (
              <Form.Item label="受信 Proposer profile">
                <Select value={profile} onChange={setProfile} options={[
                  { value: 'pro', label: 'DeepSeek V4 Pro · OpenCode' },
                  { value: 'flash', label: 'DeepSeek V4 Flash · OpenCode' },
                ]} />
              </Form.Item>
            )}
            <Form.Item label="最大候选数（受信边界 1–5）">
              <InputNumber min={1} max={5} value={candidateBudget} onChange={(value) => setCandidateBudget(value ?? 1)} />
            </Form.Item>
            {kind === 'RUN_START' && <Form.Item><Checkbox checked={proposalOnly} onChange={(event) => setProposalOnly(event.target.checked)}>仅生成并进行 POLICY 校验，不使用 GPU</Checkbox></Form.Item>}
            {kind === 'BENCHMARK_INIT' && <Form.Item label="每个 arm 重复次数"><InputNumber min={1} max={100} value={repetitions} onChange={(value) => setRepetitions(value ?? 2)} /></Form.Item>}
            {kind === 'MANUAL_EVALUATION_START' && (
              <>
                <Form.Item label="上传 kernel.py">
                  <Upload
                    accept=".py,text/x-python"
                    maxCount={1}
                    beforeUpload={async (file) => {
                      setSource(await file.text())
                      return false
                    }}
                    showUploadList={false}
                  >
                    <Button>选择文件</Button>
                  </Upload>
                  <Text type={sourceBytes > 262144 ? 'danger' : 'secondary'}> {sourceBytes.toLocaleString()} / 262,144 bytes</Text>
                </Form.Item>
                <Suspense fallback={<div className="code-editor" aria-label="正在加载候选编辑器" />}><CodeEditor value={source} onChange={setSource} /></Suspense>
              </>
            )}
          </Form>
        )}
        {step === 2 && prepared && (
          <Space direction="vertical" size="middle" className="confirm-panel">
            <Alert type="warning" showIcon title="这是受信写操作。页面刷新或网络重试不得创建第二个 GPU action。" />
            <Text code copyable>{prepared.operation_digest}</Text>
            <pre>{JSON.stringify(prepared.impact, null, 2)}</pre>
            <Input value={phrase} onChange={(event) => setPhrase(event.target.value)} placeholder={prepared.confirmation_phrase} />
            <Button danger type="primary" disabled={phrase !== prepared.confirmation_phrase} onClick={() => void confirm()}>确认并执行</Button>
          </Space>
        )}
        {step === 3 && <Alert type="success" showIcon title={message} />}
        {message && step !== 3 && <Alert className="section-card" type="error" showIcon title={message} />}
        <div className="wizard-actions">
          {step > 0 && step < 3 && <Button onClick={() => setStep(step - 1)}>上一步</Button>}
          {step === 0 && <Button type="primary" onClick={() => setStep(1)}>下一步</Button>}
          {step === 1 && <Button type="primary" disabled={!canWrite || sourceBytes > 262144 || (kind === 'MANUAL_EVALUATION_START' && !source.trim())} onClick={() => void prepare()}>冻结并预览</Button>}
        </div>
      </Card>
    </section>
  )
}
