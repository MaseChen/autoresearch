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
    scoring_shadow_profile_digest: `sha256:${'6'.repeat(64)}`,
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
    iterations: [], evaluation_attempts: [], experiments: [], experiment_relations: [],
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

test('desktop shows create-observe-results information architecture', async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== 'desktop')
  await page.goto('/#bootstrap=test-bootstrap-token')
  await expect(page.getByRole('heading', { name: '运行总览' })).toBeVisible()
  await expect(page.locator('body')).not.toContainText('NaN')
  await expect(page.getByText('创建任务 → 观察过程 → 查看结果')).toBeVisible()
  const results = await new AxeBuilder({ page }).analyze()
  expect(results.violations).toEqual([])
  await page.getByText('浅色').click()
  await expect(page.locator('.console-layout')).toHaveClass(/theme-light/)
  await expect(page.getByText('查看图表数据表')).toBeVisible()
})

test('tablet and mobile keep every write control disabled', async ({ page }, testInfo) => {
  test.skip(testInfo.project.name === 'desktop')
  await page.goto('/#bootstrap=test-bootstrap-token')
  if (testInfo.project.name === 'mobile') {
    await expect(page.getByRole('heading', { name: '运行总览' })).toBeVisible()
    await expect(page.getByText('手机模式仅显示健康状态；所有写操作和任务详情均已禁用。')).toBeVisible()
    await expect(page.locator('.ant-menu')).toHaveCount(0)
    await expect(page.locator('button:not([disabled])').filter({ hasText: /确认|执行|启动|冻结/ })).toHaveCount(0)
    return
  }
  await page.locator('li[data-menu-id$="-create"]').click()
  await expect(page.getByRole('button', { name: '下一步' })).toBeVisible()
  await page.getByRole('button', { name: '下一步' }).click()
  await expect(page.getByRole('button', { name: '冻结并预览' })).toBeDisabled()
})
