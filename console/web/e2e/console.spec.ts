import { expect, test } from '@playwright/test'
import AxeBuilder from '@axe-core/playwright'

const runtimeDigest = `sha256:${'7'.repeat(64)}`
const manualConfirmationPhrase = '启动手工 CURRENT 评测'
const preparedManualOperation = {
  schema_version: 1,
  operation_id: '00000000-0000-4000-8000-000000000111',
  kind: 'MANUAL_EVALUATION_START',
  operation_digest: `sha256:${'8'.repeat(64)}`,
  runtime_identity_digest: runtimeDigest,
  prepared_at: '2026-08-25T00:00:00Z',
  expires_epoch: 1_800_000_000,
  confirmation_phrase: manualConfirmationPhrase,
  impact: {},
}
const snapshot = {
  schema_version: 1,
  status: 'STABLE',
  runtime_identity: {
    schema_version: 1,
    git_commit: 'a'.repeat(40),
    expected_git_commit: 'a'.repeat(40),
    config_digest: `sha256:${'1'.repeat(64)}`,
    deployment_evidence_digest: `sha256:${'2'.repeat(64)}`,
    namespace_id: `sha256:${'3'.repeat(64)}`,
    execution_environment_digest: `sha256:${'4'.repeat(64)}`,
    profiler_activation_profile_digest: `sha256:${'5'.repeat(64)}`,
    controller_schema_version: 3,
    history_schema_version: 3,
    campaign_schema_version: 1,
    agent_protocol_digest: `sha256:${'6'.repeat(64)}`,
    runtime_identity_digest: runtimeDigest,
  },
  cursor: { controller_event_id: 1 },
  observed_at: '2026-08-24T00:00:00Z',
  source_digests: {},
  data: {
    runs: [{ id: 'run-1', status: 'RUNNING', valid_candidates: 1, updated_at: '2026-08-24T00:02:00Z' }],
    iterations: [{ id: 11, run_id: 'run-1', status: 'RUNNING', stage: 'QUICK', candidate_hash: 'c'.repeat(64) }],
    evaluation_attempts: [{ id: 21, run_id: 'run-1', iteration_id: 11, experiment_uid: 'experiment-one', stage: 'SMOKE', suite: 'smoke', replicate_kind: 'validation', status: 'SUCCEEDED', history_experiment_id: 1 }], experiments: [
      { id: 1, experiment_uid: 'experiment-one', status: 'SUCCESS', aggregate_score: 2.5, candidate_hash: 'c'.repeat(64), artifact_id: 'source-bundle-v1:fixture', suite: 'smoke', backend: 'metax-c500', created_at: '2026-08-24T00:01:00Z' },
      { id: 2, status: 'SUCCESS', aggregate_score: null },
    ], experiment_relations: [],
    campaigns: [{ id: 'campaign-1', mode: 'DISCOVERY', status: 'RUNNING' }],
    child_runs: [], resource_leases: [], budget_actions: [], soak_generations: [], soak_violations: [],
  },
}

test.beforeEach(async ({ page }, testInfo) => {
  page.on('console', (message) => {
    if (message.type() === 'warning' || message.type() === 'error') {
      throw new Error(`browser console ${message.type()}: ${message.text()}`)
    }
  })
  if (testInfo.project.name === 'desktop-textarea-fallback') {
    await page.addInitScript(() => {
      Reflect.deleteProperty(window, 'EditContext')
    })
  }
  await page.addInitScript((snapshotValue) => {
    class StableEventSource extends EventTarget {
      static readonly CONNECTING = 0
      static readonly OPEN = 1
      static readonly CLOSED = 2
      readonly CONNECTING = 0
      readonly OPEN = 1
      readonly CLOSED = 2
      readonly url = '/api/v1/events/stream'
      readonly withCredentials = true
      readyState = StableEventSource.OPEN
      onerror: ((event: Event) => unknown) | null = null
      onmessage: ((event: MessageEvent) => unknown) | null = null
      onopen: ((event: Event) => unknown) | null = null

      constructor() {
        super()
        setTimeout(() => {
          this.dispatchEvent(new MessageEvent('snapshot', {
            data: JSON.stringify({ snapshot: snapshotValue }),
            lastEventId: 'one',
          }))
        }, 0)
      }

      close() {
        this.readyState = StableEventSource.CLOSED
      }
    }
    Object.defineProperty(window, 'EventSource', {
      configurable: true,
      value: StableEventSource,
      writable: true,
    })
  }, snapshot)
  page.on('pageerror', (error) => { throw error })
  await page.route('**/api/v1/session/bootstrap', (route) => route.fulfill({
    contentType: 'application/json',
    body: JSON.stringify({ schema_version: 1, request_id: 'bootstrap', data: { csrf_token: 'csrf' } }),
  }))
  await page.route('**/api/v1/runtime', (route) => route.fulfill({
    contentType: 'application/json',
    body: JSON.stringify({ schema_version: 1, request_id: 'runtime', data: { snapshot } }),
  }))
  await page.route('**/api/v1/artifacts/**', (route) => route.fulfill({
    contentType: 'application/json',
    body: JSON.stringify({ schema_version: 1, request_id: 'artifact', data: { artifact: { schema_version: 1, artifact_id: 'source-bundle-v1:fixture', manifest: {}, entrypoint: 'kernel.py', media_type: 'text/x-python', source: 'def run():\n    return 1\n' } } }),
  }))
  await page.route('**/api/v1/operations/MANUAL_EVALUATION_START/prepare', (route) => route.fulfill({
    contentType: 'application/json',
    body: JSON.stringify({ schema_version: 1, request_id: 'prepare', data: { prepared_operation: preparedManualOperation } }),
  }))
})

test('desktop follows create, run records, and full task detail workflow', async ({ page }, testInfo) => {
  test.skip(!testInfo.project.name.startsWith('desktop'))
  await page.goto('/#bootstrap=test-bootstrap-token')
  await expect(page.getByRole('heading', { name: '运行记录' })).toBeVisible({ timeout: 15_000 })
  for (const item of ['创建任务', '运行记录', '系统状态']) {
    await expect(page.getByRole('menuitem', { name: item })).toBeVisible()
  }
  await expect(page.getByRole('menuitem')).toHaveCount(3)
  const results = await new AxeBuilder({ page }).analyze()
  expect(results.violations).toEqual([])
  await expect(page.locator('.console-layout')).toHaveClass(/theme-light/)
  await expect(page.locator('[aria-label*="NaN"]')).toHaveCount(0)
  await expect(page.locator('body')).not.toContainText('NaN')
  await expect(page.getByText('单次自主优化').first()).toBeVisible()
  await expect(page.getByText('快速验证').first()).toBeVisible()
  await expect(page.getByText('2.5').first()).toBeVisible()

  await page.getByRole('button', { name: '查看任务 单次自主优化' }).click()
  await expect(page.getByRole('heading', { name: '单次自主优化' })).toBeVisible()
  await expect(page.getByText('执行进度')).toBeVisible()
  await expect(page.getByText('实时优化轮次')).toBeVisible()
  await expect(page.getByText('正式评分走势')).toBeVisible()
  await expect(page.getByText('性能结果')).toBeVisible()
  await expect(page.getByText('Profiler 分析')).toBeVisible()
  await expect(page.getByText('最佳 2.5')).toBeVisible()
  await page.getByText('第 1 轮').click()
  await expect(page.getByText('评测链', { exact: true })).toBeVisible()
  await expect(page.getByText(/深度研究证据 · 当前协议未接入/)).toBeVisible()
  await page.getByText(/深度研究证据 · 当前协议未接入/).click()
  await expect(page.getByText('当前 Agent 协议尚未提供逐轮深度证据')).toBeVisible()
  await page.getByRole('button', { name: '查看源代码' }).click()
  await expect(page.getByRole('dialog', { name: '候选源代码' })).toBeVisible()
  await expect(page.locator('.ant-modal .monaco-editor .view-lines')).toContainText('def run():')
  await page.getByRole('button', { name: /关\s*闭/ }).click()
  await page.getByRole('button', { name: '返回运行记录' }).click()
  await expect(page.getByRole('heading', { name: '运行记录' })).toBeVisible()

  await page.getByRole('menuitem', { name: '创建任务' }).click()
  await expect(page.getByRole('heading', { name: '创建优化任务' })).toBeVisible()
  await expect(page.getByText('Fused MoE I8 TN', { exact: true }).first()).toBeVisible()
  await expect(page.getByRole('button', { name: /FlashInfer Ragged Prefill/ })).toBeDisabled()
  await expect(page.getByRole('button', { name: /TileLang/ })).toBeDisabled()
  await expect(page.getByText('执行步骤', { exact: true })).toBeVisible()
  await expect(page.getByText('编译与运行框架')).toBeVisible()
  const editor = page.getByRole('textbox', { name: 'kernel.py 代码编辑器' })
  await page.locator('.monaco-editor .view-lines').click()
  await page.keyboard.type('import triton')
  await page.keyboard.press('Enter')
  await page.keyboard.type('import numpy')
  await expect(editor).toHaveAttribute('aria-roledescription', 'editor')
  await expect(page.locator('.monaco-editor .view-lines')).toContainText('import triton')
  await expect(page.locator('.monaco-editor')).toBeVisible()
  const visibleCursor = page.locator('.monaco-editor .cursor').first()
  await expect(visibleCursor).toBeVisible()
  await expect(visibleCursor).toHaveCSS('background-color', 'rgb(23, 79, 178)')
  const activeLineNumber = page.locator('.monaco-editor .margin-view-overlays .line-numbers.active-line-number')
  await expect(activeLineNumber).toHaveCount(1)
  await expect(activeLineNumber).toHaveText('2')
  await expect(activeLineNumber).toHaveCSS('color', 'rgb(11, 63, 145)')
  await expect(page.getByText('2 行')).toBeVisible()
  await expect(page.getByText('空格: 4')).toBeVisible()
  const lineTops = await page.locator('.monaco-editor .view-lines .view-line').evaluateAll((nodes) => nodes.slice(0, 2).map((node) => node.getBoundingClientRect().top))
  const gutterTops = await page.locator('.monaco-editor .margin-view-overlays .line-numbers').evaluateAll((nodes) => nodes.filter((node) => ['1', '2'].includes(node.textContent?.trim() ?? '')).slice(0, 2).map((node) => node.getBoundingClientRect().top))
  expect(lineTops).toHaveLength(2)
  expect(gutterTops).toHaveLength(2)
  expect(Math.abs(lineTops[0] - gutterTops[0])).toBeLessThan(2)
  expect(Math.abs(lineTops[1] - gutterTops[1])).toBeLessThan(2)
  const imeInput = page.locator('.monaco-editor textarea.inputarea')
  if (testInfo.project.name === 'desktop-textarea-fallback') {
    await imeInput.dispatchEvent('compositionstart', { data: '候选' })
    await expect(imeInput).toHaveClass(/ime-input/)
    await expect(imeInput).toHaveCSS('font-size', '14px')
    await expect(imeInput).toHaveCSS('line-height', '22px')
    await expect(imeInput).toHaveCSS('box-sizing', 'content-box')
    const compositionTop = await imeInput.evaluate((node) => node.getBoundingClientRect().top)
    const currentLineTop = await page.locator('.monaco-editor .view-lines .view-line').last().evaluate((node) => node.getBoundingClientRect().top)
    expect(Math.abs(compositionTop - currentLineTop)).toBeLessThan(2)
    await imeInput.dispatchEvent('compositionend', { data: '候选' })
    await expect(imeInput).not.toHaveClass(/ime-input/)
  } else {
    await expect(page.locator('.monaco-editor .native-edit-context')).toHaveCount(1)
  }

  await page.getByRole('button', { name: '检查配置并继续' }).click()
  await expect(page.getByText('请输入以下确认短语：')).toBeVisible()
  await expect(page.getByText(manualConfirmationPhrase, { exact: true })).toBeVisible()
  const confirmationInput = page.getByRole('textbox', { name: '确认短语' })
  const launchButton = page.getByRole('button', { name: '确认并启动任务' })
  await expect(launchButton).toBeDisabled()
  await confirmationInput.fill('错误短语')
  await expect(page.getByText('确认短语必须与上方文字逐字一致。')).toBeVisible()
  await expect(launchButton).toBeDisabled()
  await page.getByRole('button', { name: '填入确认短语' }).click()
  await expect(confirmationInput).toHaveValue(manualConfirmationPhrase)
  await expect(launchButton).toBeEnabled()

  await page.getByRole('menuitem', { name: '系统状态' }).click()
  await expect(page.getByRole('heading', { name: '服务器与评测环境' })).toBeVisible()
  await page.getByRole('tab', { name: 'Profiler' }).click()
  await expect(page.getByRole('heading', { name: '性能分析工具' })).toBeVisible()
  await page.getByRole('tab', { name: '原始数据' }).click()
  await expect(page.getByText('用于排查问题。日常查看请使用运行记录和系统状态。')).toBeVisible()
})

test('tablet and mobile keep every write control disabled', async ({ page }, testInfo) => {
  test.skip(testInfo.project.name.startsWith('desktop'))
  await page.goto('/#bootstrap=test-bootstrap-token')
  if (testInfo.project.name === 'mobile') {
    await expect(page.getByRole('heading', { name: '运行健康状态' })).toBeVisible()
    await expect(page.getByText('手机模式只显示健康状态，不能执行任务操作。')).toBeVisible()
    await expect(page.locator('.ant-menu')).toHaveCount(0)
    await expect(page.locator('button:not([disabled])').filter({ hasText: /确认|执行|启动|冻结/ })).toHaveCount(0)
    return
  }
  await page.getByRole('menuitem', { name: '创建任务' }).click()
  await expect(page.getByRole('heading', { name: '创建优化任务' })).toBeVisible()
  await expect(page.getByRole('button', { name: '检查配置并继续' })).toBeDisabled()
})
