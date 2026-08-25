import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  bootstrapSession,
  bootstrapTokenFromFragment,
  confirmOperation,
  fetchAudit,
  fetchOperation,
  fetchScientificArtifact,
  fetchSnapshot,
  prepareOperation,
  subscribeEvents,
} from '../src/api'
import type { PreparedOperation } from '../src/types'

const envelope = (data: unknown) => ({ schema_version: 1, request_id: 'request', data })

function response(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), { status, headers: { 'Content-Type': 'application/json' } })
}

afterEach(() => {
  vi.restoreAllMocks()
  history.replaceState(null, '', '/')
})

describe('Console API', () => {
  it('bootstraps once and removes the fragment', async () => {
    const fetch = vi.spyOn(globalThis, 'fetch').mockResolvedValue(response(envelope({ csrf_token: 'csrf' })))
    history.replaceState(null, '', '/#bootstrap=one-time')
    expect(bootstrapTokenFromFragment()).toBe('one-time')
    expect(location.hash).toBe('')
    expect(bootstrapTokenFromFragment()).toBeNull()
    await bootstrapSession('one-time')
    expect(fetch).toHaveBeenCalledWith('/api/v1/session/bootstrap', expect.objectContaining({ method: 'POST' }))
  })

  it('reads snapshots, operations and audit envelopes', async () => {
    const prepared: PreparedOperation = {
      schema_version: 1,
      operation_id: crypto.randomUUID(),
      kind: 'RUN_START',
      operation_digest: `sha256:${'1'.repeat(64)}`,
      runtime_identity_digest: `sha256:${'2'.repeat(64)}`,
      prepared_at: 'now',
      expires_epoch: 9999999999,
      confirmation_phrase: '启动自治 Run',
      impact: {},
    }
    const snapshot = { status: 'STABLE' }
    vi.spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(response(envelope({ snapshot })))
      .mockResolvedValueOnce(response(envelope({ prepared_operation: prepared })))
      .mockResolvedValueOnce(response(envelope({ operation_receipt: { status: 'EXECUTING' } })))
      .mockResolvedValueOnce(response(envelope({ operation: { status: 'EXECUTING' } })))
      .mockResolvedValueOnce(response(envelope({ audit: [{ sequence: 1 }] })))
      .mockResolvedValueOnce(response(envelope({ artifact: { schema_version: 1, artifact_id: 'source-bundle-v1:fixture', manifest: {}, entrypoint: 'kernel.py', media_type: 'text/x-python', source: 'def run(): pass\n' } })))
    expect((await fetchSnapshot()).status).toBe('STABLE')
    expect((await prepareOperation('RUN_START', { profile: 'pro' }, prepared.runtime_identity_digest, prepared.operation_id)).operation_id).toBe(prepared.operation_id)
    expect((await confirmOperation(prepared, prepared.confirmation_phrase)).status).toBe('EXECUTING')
    expect((await fetchOperation(prepared.operation_id)).status).toBe('EXECUTING')
    expect(await fetchAudit()).toEqual([{ sequence: 1 }])
    expect((await fetchScientificArtifact('source-bundle-v1:fixture')).entrypoint).toBe('kernel.py')
  })

  it('surfaces bounded server errors with and without detail', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce(response({ detail: 'stale' }, 409))
      .mockResolvedValueOnce(response({}, 500))
    await expect(fetchSnapshot()).rejects.toThrow('stale')
    await expect(fetchSnapshot()).rejects.toThrow('HTTP 500')
  })

  it('streams snapshots, reset/error status and closes cleanly', () => {
    const listeners = new Map<string, (event: Event) => void>()
    let closed = false
    class EventSourceFixture {
      static readonly instances: EventSourceFixture[] = []
      onerror: (() => void) | null = null
      constructor(public readonly url: string, public readonly options: EventSourceInit) {
        EventSourceFixture.instances.push(this)
      }
      addEventListener(name: string, callback: EventListenerOrEventListenerObject) {
        listeners.set(name, callback as (event: Event) => void)
      }
      close() { closed = true }
    }
    vi.stubGlobal('EventSource', EventSourceFixture)
    const snapshots: unknown[] = []
    const statuses: string[] = []
    const close = subscribeEvents((value) => snapshots.push(value), (value) => statuses.push(value))
    listeners.get('snapshot')?.({ data: JSON.stringify({ snapshot: { status: 'STABLE' } }) } as MessageEvent)
    listeners.get('reset')?.(new Event('reset'))
    EventSourceFixture.instances[0].onerror?.()
    close()
    expect(snapshots).toEqual([{ status: 'STABLE' }])
    expect(statuses).toEqual(['live', 'stale', 'stale'])
    expect(closed).toBe(true)
    expect(EventSourceFixture.instances[0].options).toEqual({ withCredentials: true })
  })
})
