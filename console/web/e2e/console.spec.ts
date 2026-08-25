import { expect, test } from '@playwright/test'
import AxeBuilder from '@axe-core/playwright'

const runtimeDigest = `sha256:${'7'.repeat(64)}`
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

test.beforeEach(async ({ page }) => {
  page.on('pageerror', (error) => { throw error })
  await page.route('**/api/v1/session/bootstrap', (route) => route.fulfill({
    contentType: 'application/json',
    body: JSON.stringify({ schema_version: 1, request_id: 'bootstrap', data: { csrf_token: 'csrf' } }),
  }))
  await page.route('**/api/v1/runtime', (route) => route.fulfill({
    contentType: 'application/json',
    body: JSON.stringify({ schema_version: 1, request_id: 'runtime', data: { snapshot } }),
  }))
  await page.route('**/api/v1/events/stream', (route) => route.fulfill({
    contentType: 'text/event-stream',
    body: `id: one\nevent: snapshot\ndata: ${JSON.stringify({ snapshot })}\n\n`,
  }))
  await page.route('**/api/v1/artifacts/**', (route) => route.fulfill({
    contentType: 'application/json',
    body: JSON.stringify({ schema_version: 1, request_id: 'artifact', data: { artifact: { schema_version: 1, artifact_id: 'source-bundle-v1:fixture', manifest: {}, entrypoint: 'kernel.py', media_type: 'text/x-python', source: 'def run():\n    return 1\n' } } }),
  }))
})

test('desktop follows create, run records, and full task detail workflow', async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== 'desktop')
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
  await expect(page.getByText('优化轮次')).toBeVisible()
  await expect(page.getByText('性能结果')).toBeVisible()
  await expect(page.getByText('Profiler 分析')).toBeVisible()
  await expect(page.getByText('综合评分 2.5')).toBeVisible()
  await page.getByRole('button', { name: '查看源代码' }).click()
  await expect(page.getByRole('dialog', { name: '候选源代码' })).toBeVisible()
  await expect(page.getByText('def run():')).toBeVisible()
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
  await editor.click()
  await editor.pressSequentially('import triton')
  await page.keyboard.press('Enter')
  await editor.pressSequentially('import numpy')
  await expect(editor).toContainText('import triton')
  await expect(page.locator('.cm-cursor-primary')).toBeVisible()
  const lineTops = await page.locator('.cm-line').evaluateAll((nodes) => nodes.slice(0, 2).map((node) => node.getBoundingClientRect().top))
  const gutterTops = await page.locator('.cm-lineNumbers .cm-gutterElement').evaluateAll((nodes) => nodes.filter((node) => ['1', '2'].includes(node.textContent ?? '')).slice(0, 2).map((node) => node.getBoundingClientRect().top))
  expect(lineTops).toHaveLength(2)
  expect(gutterTops).toHaveLength(2)
  expect(Math.abs(lineTops[0] - gutterTops[0])).toBeLessThan(2)
  expect(Math.abs(lineTops[1] - gutterTops[1])).toBeLessThan(2)

  await page.getByRole('menuitem', { name: '系统状态' }).click()
  await expect(page.getByRole('heading', { name: '服务器与评测环境' })).toBeVisible()
  await page.getByRole('tab', { name: 'Profiler' }).click()
  await expect(page.getByRole('heading', { name: '性能分析工具' })).toBeVisible()
  await page.getByRole('tab', { name: '原始数据' }).click()
  await expect(page.getByText('用于排查问题。日常查看请使用运行记录和系统状态。')).toBeVisible()
})

test('tablet and mobile keep every write control disabled', async ({ page }, testInfo) => {
  test.skip(testInfo.project.name === 'desktop')
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
