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
    runs: [{ id: 'run-1', status: 'RUNNING', valid_candidates: 1 }],
    iterations: [], evaluation_attempts: [], experiments: [
      { id: 1, status: 'SUCCESS', aggregate_score: 2.5 },
      { id: 2, status: 'SUCCESS', aggregate_score: null },
    ], experiment_relations: [],
    campaigns: [{ id: 'campaign-1', mode: 'DISCOVERY', status: 'RUNNING' }],
    child_runs: [], resource_leases: [], budget_actions: [], soak_generations: [], soak_violations: [],
  },
}

test.beforeEach(async ({ page }) => {
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
})

test('desktop presents a four-entry workflow instead of database pages', async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== 'desktop')
  await page.goto('/#bootstrap=test-bootstrap-token')
  await expect(page.getByRole('heading', { name: '从任务目标出发，而不是从数据库出发' })).toBeVisible()
  for (const item of ['工作台', '创建任务', '任务中心', '系统与门禁']) {
    await expect(page.getByRole('menuitem', { name: item })).toBeVisible()
  }
  await expect(page.getByRole('menuitem')).toHaveCount(4)
  const results = await new AxeBuilder({ page }).analyze()
  expect(results.violations).toEqual([])
  await expect(page.locator('.console-layout')).toHaveClass(/theme-light/)
  await expect(page.getByText('查看无障碍数据表')).toBeVisible()
  const chart = page.getByRole('img', { name: /近期实验 aggregate score 折线图/ })
  await expect(chart).toHaveAttribute('aria-label', /1 个 UNAVAILABLE/)
  await expect(chart).not.toHaveAttribute('aria-label', /NaN/)
  await expect(page.locator('[aria-label*="NaN"]')).toHaveCount(0)
  await expect(page.locator('body')).not.toContainText('NaN')
  await page.getByText('查看无障碍数据表').click()
  await expect(page.getByRole('cell', { name: 'UNAVAILABLE' })).toBeVisible()

  await page.getByRole('menuitem', { name: '创建任务' }).click()
  await expect(page.getByRole('heading', { name: '创建优化任务' })).toBeVisible()
  await expect(page.getByText('Fused MoE I8 TN', { exact: true }).first()).toBeVisible()
  await expect(page.getByRole('button', { name: /FlashInfer Ragged Prefill/ })).toBeDisabled()
  await expect(page.getByRole('button', { name: /TileLang/ })).toBeDisabled()
  await expect(page.getByText('计划执行链')).toBeVisible()

  await page.getByRole('menuitem', { name: '任务中心' }).click()
  await expect(page.getByRole('heading', { name: '任务中心' })).toBeVisible()
  await expect(page.getByRole('heading', { name: '长期持续优化' })).toBeVisible()

  await page.getByRole('menuitem', { name: '系统与门禁' }).click()
  await expect(page.getByRole('heading', { name: '系统与门禁' })).toBeVisible()
  await page.getByRole('tab', { name: '数据与诊断' }).click()
  await expect(page.getByText('这里是唯一直接展示原始账本的页面')).toBeVisible()
})

test('tablet and mobile keep every write control disabled', async ({ page }, testInfo) => {
  test.skip(testInfo.project.name === 'desktop')
  await page.goto('/#bootstrap=test-bootstrap-token')
  if (testInfo.project.name === 'mobile') {
    await expect(page.getByRole('heading', { name: '运行健康状态' })).toBeVisible()
    await expect(page.getByText('手机模式仅显示健康状态；所有写操作和任务详情均已禁用。')).toBeVisible()
    await expect(page.locator('.ant-menu')).toHaveCount(0)
    await expect(page.locator('button:not([disabled])').filter({ hasText: /确认|执行|启动|冻结/ })).toHaveCount(0)
    return
  }
  await page.getByRole('menuitem', { name: '创建任务' }).click()
  await expect(page.getByRole('heading', { name: '创建优化任务' })).toBeVisible()
  await expect(page.getByRole('button', { name: '冻结任务配置并预检' })).toBeDisabled()
})
