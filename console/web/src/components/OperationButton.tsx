import { Alert, Button, Input, Modal, Space, Typography } from 'antd'
import { useState } from 'react'
import { confirmOperation, prepareOperation } from '../api'
import type { PreparedOperation, TaskKind } from '../types'

const { Text } = Typography

export function OperationButton({
  kind,
  parameters,
  runtimeIdentityDigest,
  canWrite,
  label,
  danger = false,
}: {
  kind: TaskKind | 'RUN_STOP' | 'RUN_RESUME' | 'CAMPAIGN_START' | 'CAMPAIGN_PAUSE' | 'CAMPAIGN_RESUME' | 'CAMPAIGN_CHILD_EXECUTE' | 'CAMPAIGN_LINEAGE_ADVANCE' | 'BENCHMARK_EXECUTE'
  parameters: Record<string, unknown>
  runtimeIdentityDigest: string
  canWrite: boolean
  label: string
  danger?: boolean
}) {
  const [prepared, setPrepared] = useState<PreparedOperation | null>(null)
  const [phrase, setPhrase] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const prepare = async () => {
    setBusy(true)
    setError('')
    try {
      setPrepared(await prepareOperation(kind, parameters, runtimeIdentityDigest))
    } catch (reason: unknown) {
      setError(String(reason))
    } finally {
      setBusy(false)
    }
  }

  const confirm = async () => {
    if (!prepared) return
    setBusy(true)
    setError('')
    try {
      await confirmOperation(prepared, phrase)
      setPrepared(null)
      setPhrase('')
    } catch (reason: unknown) {
      setError(String(reason))
    } finally {
      setBusy(false)
    }
  }

  return <>
    <Button size="small" danger={danger} disabled={!canWrite} loading={busy} onClick={() => void prepare()}>{label}</Button>
    <Modal
      title={`确认：${label}`}
      open={prepared !== null || Boolean(error)}
      okText="确认并执行"
      cancelText="取消"
      okButtonProps={{ danger, disabled: !prepared || phrase !== prepared.confirmation_phrase, loading: busy }}
      onOk={() => void confirm()}
      onCancel={() => { setPrepared(null); setError(''); setPhrase('') }}
    >
      <Space direction="vertical" className="confirm-panel">
        {error && <Alert type="error" showIcon title={error} />}
        {prepared && <>
          <Alert type="warning" showIcon title="请核对操作内容。确认后将立即提交到服务器。" />
          <Text code copyable>{prepared.operation_digest}</Text>
          <pre>{JSON.stringify(prepared.impact, null, 2)}</pre>
          <Input value={phrase} onChange={(event) => setPhrase(event.target.value)} placeholder={prepared.confirmation_phrase} />
        </>}
      </Space>
    </Modal>
  </>
}
