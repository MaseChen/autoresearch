import type { ApiEnvelope, ConsoleSnapshot, OperationReceipt, PreparedOperation } from './types'

let csrfToken = ''

async function decode<T>(response: Response): Promise<T> {
  const value = (await response.json()) as unknown
  if (!response.ok) {
    const detail = typeof value === 'object' && value !== null && 'detail' in value
      ? String(value.detail)
      : `HTTP ${response.status}`
    throw new Error(detail)
  }
  return value as T
}

export async function bootstrapSession(token: string): Promise<void> {
  const response = await fetch('/api/v1/session/bootstrap', {
    method: 'POST',
    credentials: 'same-origin',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ schema_version: 1, token }),
  })
  const envelope = await decode<ApiEnvelope<{ csrf_token: string }>>(response)
  csrfToken = envelope.data.csrf_token
}

export function bootstrapTokenFromFragment(): string | null {
  const fragment = new URLSearchParams(window.location.hash.slice(1))
  const token = fragment.get('bootstrap')
  if (token) {
    history.replaceState(null, '', `${location.pathname}${location.search}`)
  }
  return token
}

export async function fetchSnapshot(): Promise<ConsoleSnapshot> {
  const response = await fetch('/api/v1/runtime', { credentials: 'same-origin' })
  const envelope = await decode<ApiEnvelope<{ snapshot: ConsoleSnapshot }>>(response)
  return envelope.data.snapshot
}

export async function prepareOperation(
  kind: string,
  parameters: Record<string, unknown>,
  runtimeIdentityDigest: string,
  operationId: string = crypto.randomUUID(),
): Promise<PreparedOperation> {
  const response = await fetch(`/api/v1/operations/${encodeURIComponent(kind)}/prepare`, {
    method: 'POST',
    credentials: 'same-origin',
    headers: {
      'Content-Type': 'application/json',
      'X-CSRF-Token': csrfToken,
    },
    body: JSON.stringify({
      schema_version: 1,
      operation_id: operationId,
      runtime_identity_digest: runtimeIdentityDigest,
      parameters,
    }),
  })
  const envelope = await decode<ApiEnvelope<{ prepared_operation: PreparedOperation }>>(response)
  return envelope.data.prepared_operation
}

export async function confirmOperation(
  operation: PreparedOperation,
  phrase: string,
): Promise<OperationReceipt> {
  const response = await fetch(`/api/v1/operations/${operation.operation_id}/confirm`, {
    method: 'POST',
    credentials: 'same-origin',
    headers: {
      'Content-Type': 'application/json',
      'X-CSRF-Token': csrfToken,
    },
    body: JSON.stringify({
      schema_version: 1,
      operation_digest: operation.operation_digest,
      confirmation_phrase: phrase,
    }),
  })
  const envelope = await decode<ApiEnvelope<{ operation_receipt: OperationReceipt }>>(response)
  return envelope.data.operation_receipt
}

export async function fetchOperation(operationId: string): Promise<Record<string, unknown>> {
  const response = await fetch(`/api/v1/operations/${encodeURIComponent(operationId)}`, {
    credentials: 'same-origin',
  })
  const envelope = await decode<ApiEnvelope<{ operation: Record<string, unknown> }>>(response)
  return envelope.data.operation
}

export async function fetchAudit(): Promise<Record<string, unknown>[]> {
  const response = await fetch('/api/v1/audit', { credentials: 'same-origin' })
  const envelope = await decode<ApiEnvelope<{ audit: Record<string, unknown>[] }>>(response)
  return envelope.data.audit
}

export function subscribeEvents(
  onSnapshot: (snapshot: ConsoleSnapshot) => void,
  onStatus: (status: 'live' | 'stale') => void,
): () => void {
  const source = new EventSource('/api/v1/events/stream', { withCredentials: true })
  source.addEventListener('snapshot', (event) => {
    const message = JSON.parse((event as MessageEvent<string>).data) as {
      snapshot: ConsoleSnapshot
    }
    onStatus('live')
    onSnapshot(message.snapshot)
  })
  source.addEventListener('reset', () => onStatus('stale'))
  source.onerror = () => onStatus('stale')
  return () => source.close()
}
