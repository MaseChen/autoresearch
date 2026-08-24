import { lazy, Suspense, useEffect, useMemo, useState } from 'react'
import { App as AntApp, Button, ConfigProvider, Layout, Menu, Result, Segmented, Spin, Typography, theme } from 'antd'
import { QueryClient, QueryClientProvider, useQuery } from '@tanstack/react-query'
import { bootstrapSession, bootstrapTokenFromFragment, fetchSnapshot, subscribeEvents } from './api'
import './styles.css'
import { canMutate } from './operationDrafts'

const Dashboard = lazy(() => import('./pages/Dashboard').then((module) => ({ default: module.Dashboard })))
const CreateTask = lazy(() => import('./pages/CreateTask').then((module) => ({ default: module.CreateTask })))
const dataViews = () => import('./pages/DataViews')
const TasksPage = lazy(() => dataViews().then((module) => ({ default: module.TasksPage })))
const ExperimentsPage = lazy(() => dataViews().then((module) => ({ default: module.ExperimentsPage })))
const CampaignsPage = lazy(() => dataViews().then((module) => ({ default: module.CampaignsPage })))
const ResourcesPage = lazy(() => dataViews().then((module) => ({ default: module.ResourcesPage })))
const SoakPage = lazy(() => dataViews().then((module) => ({ default: module.SoakPage })))
const AuditPage = lazy(() => dataViews().then((module) => ({ default: module.AuditPage })))
const SettingsPage = lazy(() => dataViews().then((module) => ({ default: module.SettingsPage })))

const { Header, Sider, Content } = Layout
const { Text } = Typography
const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
const initialBootstrapToken = bootstrapTokenFromFragment()

type Page = 'dashboard' | 'create' | 'tasks' | 'experiments' | 'campaigns' | 'resources' | 'soak' | 'audit' | 'settings'

function ConsoleApp({ mode, setMode }: { mode: 'dark' | 'light'; setMode: (mode: 'dark' | 'light') => void }) {
  const [page, setPage] = useState<Page>('dashboard')
  const [sessionReady, setSessionReady] = useState(false)
  const [sessionError, setSessionError] = useState(
    initialBootstrapToken ? '' : '缺少一次性 bootstrap token。请从 kernel-autoresearch-console 输出的 URL 打开页面。',
  )
  const [streamStatus, setStreamStatus] = useState<'live' | 'stale'>('stale')
  const [viewportWidth, setViewportWidth] = useState(() => window.innerWidth)
  const query = useQuery({ queryKey: ['snapshot'], queryFn: fetchSnapshot, enabled: sessionReady, refetchInterval: 15000 })

  useEffect(() => {
    if (!initialBootstrapToken) return
    void bootstrapSession(initialBootstrapToken)
      .then(() => setSessionReady(true))
      .catch((error: unknown) => setSessionError(String(error)))
  }, [])

  useEffect(() => {
    if (!sessionReady) return
    return subscribeEvents(
      (snapshot) => queryClient.setQueryData(['snapshot'], snapshot),
      setStreamStatus,
    )
  }, [sessionReady])

  useEffect(() => {
    const update = () => setViewportWidth(window.innerWidth)
    window.addEventListener('resize', update)
    return () => window.removeEventListener('resize', update)
  }, [])

  const snapshot = query.data
  const canWrite = canMutate(snapshot?.status, viewportWidth, streamStatus)
  const pageContent = useMemo(() => {
    if (!snapshot) return null
    const pages: Record<Page, React.ReactNode> = {
      dashboard: <Dashboard snapshot={snapshot} />,
      create: <CreateTask canWrite={canWrite} runtimeIdentityDigest={snapshot.runtime_identity.runtime_identity_digest} />,
      tasks: <TasksPage snapshot={snapshot} canWrite={canWrite} />,
      experiments: <ExperimentsPage snapshot={snapshot} />,
      campaigns: <CampaignsPage snapshot={snapshot} canWrite={canWrite} />,
      resources: <ResourcesPage snapshot={snapshot} />,
      soak: <SoakPage snapshot={snapshot} />,
      audit: <AuditPage snapshot={snapshot} />,
      settings: <SettingsPage />,
    }
    return pages[page]
  }, [canWrite, page, snapshot])

  if (sessionError) return <Result status="error" title="Console 会话未建立" subTitle={sessionError} />
  if (!sessionReady || query.isLoading) return <div className="center"><Spin size="large" /><Text>正在进行可信握手…</Text></div>
  if (query.error || !snapshot) return <Result status="error" title="远端快照不可用" subTitle={String(query.error)} extra={<Button onClick={() => void query.refetch()}>重试只读请求</Button>} />

  return (
    <Layout className="console-layout">
      <Header className="topbar">
        <div className="brand"><span className="brand-mark">AR</span><span><h1>Autoresearch Console</h1><small>可信算子研究控制台</small></span></div>
        <div className="topbar-status"><span className={`live-dot ${streamStatus}`} />{streamStatus === 'live' ? 'LIVE' : 'STALE'}<Text code>{snapshot.runtime_identity.git_commit.slice(0, 10)}</Text><Segmented size="small" value={mode} onChange={(value) => setMode(value as 'dark' | 'light')} options={[{ label: '暗色', value: 'dark' }, { label: '浅色', value: 'light' }]} /></div>
      </Header>
      <Layout>
        <Sider width={220} breakpoint="lg" collapsedWidth={72} theme="dark">
          <Menu
            theme="dark"
            mode="inline"
            selectedKeys={[page]}
            onClick={({ key }) => setPage(key as Page)}
            items={[
              { key: 'dashboard', label: '总览' },
              { key: 'create', label: '创建任务' },
              { key: 'tasks', label: '观察过程' },
              { key: 'experiments', label: '实验与 History' },
              { key: 'campaigns', label: 'Campaign' },
              { key: 'resources', label: '资源与 Budget' },
              { key: 'soak', label: 'Soak 门禁' },
              { key: 'audit', label: '审计' },
              { key: 'settings', label: '设置' },
            ]}
          />
        </Sider>
        <Content className="content"><Suspense fallback={<div className="center"><Spin /></div>}>{pageContent}</Suspense></Content>
      </Layout>
    </Layout>
  )
}

export default function App() {
  const nonce = document.querySelector<HTMLMetaElement>('meta[name="csp-nonce"]')?.content
  const [mode, setMode] = useState<'dark' | 'light'>('dark')
  return (
    <QueryClientProvider client={queryClient}>
      <ConfigProvider csp={nonce && nonce !== '__CSP_NONCE__' ? { nonce } : undefined} theme={{ algorithm: mode === 'dark' ? theme.darkAlgorithm : theme.defaultAlgorithm, token: { colorPrimary: '#087f70', borderRadius: 6, fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif' } }}>
        <AntApp><ConsoleApp mode={mode} setMode={setMode} /></AntApp>
      </ConfigProvider>
    </QueryClientProvider>
  )
}
