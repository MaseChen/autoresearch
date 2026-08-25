import { lazy, Suspense, useEffect, useMemo, useState } from 'react'
import { App as AntApp, Button, ConfigProvider, Layout, Menu, Result, Segmented, Spin, Tooltip, Typography, theme } from 'antd'
import { QueryClient, QueryClientProvider, useQuery } from '@tanstack/react-query'
import { bootstrapSession, bootstrapTokenFromFragment, fetchSnapshot, subscribeEvents } from './api'
import './styles.css'
import { canMutate } from './operationDrafts'

const CreateTask = lazy(() => import('./pages/CreateTask').then((module) => ({ default: module.CreateTask })))
const dataViews = () => import('./pages/DataViews')
const RunRecordsPage = lazy(() => dataViews().then((module) => ({ default: module.RunRecordsPage })))
const SystemPage = lazy(() => dataViews().then((module) => ({ default: module.SystemPage })))

const { Header, Sider, Content } = Layout
const { Text } = Typography
const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
const initialBootstrapToken = bootstrapTokenFromFragment()

type Page = 'create' | 'runs' | 'system'

function ConsoleApp({ mode, setMode }: { mode: 'dark' | 'light'; setMode: (mode: 'dark' | 'light') => void }) {
  const [page, setPage] = useState<Page>('runs')
  const [selectedTaskId, setSelectedTaskId] = useState<string>()
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
      create: <CreateTask canWrite={canWrite} runtimeIdentityDigest={snapshot.runtime_identity.runtime_identity_digest} />,
      runs: <RunRecordsPage snapshot={snapshot} canWrite={canWrite} selectedTaskId={selectedTaskId} onOpenTask={setSelectedTaskId} onBack={() => setSelectedTaskId(undefined)} onCreate={() => { setSelectedTaskId(undefined); setPage('create') }} />,
      system: <SystemPage snapshot={snapshot} />,
    }
    return pages[page]
  }, [canWrite, page, selectedTaskId, snapshot])

  const navigate = (next: Page) => {
    setSelectedTaskId(undefined)
    setPage(next)
  }

  if (sessionError) return <Result status="error" title="控制台会话未建立" subTitle={sessionError} />
  if (!sessionReady || query.isLoading) return <div className="center"><Spin size="large" /><Text>正在连接服务器…</Text></div>
  if (query.error || !snapshot) return <Result status="error" title="暂时无法读取服务器状态" subTitle={String(query.error)} extra={<Button onClick={() => void query.refetch()}>重新连接</Button>} />
  if (viewportWidth < 768) {
    const activeRuns = snapshot.data.runs.filter((row) => row.status === 'RUNNING').length
    const activeCampaigns = snapshot.data.campaigns.filter((row) => row.status === 'RUNNING').length
    return (
      <Layout className={`console-layout theme-${mode} mobile-health`}>
        <Header className="topbar"><div className="brand"><h1>算子优化控制台</h1></div></Header>
        <Content className="content">
          <section aria-labelledby="mobile-health-title">
            <Typography.Title id="mobile-health-title" level={2}>运行健康状态</Typography.Title>
            <Result status={snapshot.status === 'STABLE' ? 'success' : 'warning'} title={snapshot.status === 'STABLE' ? '系统稳定' : '数据正在变化'} subTitle="手机模式只显示健康状态，不能执行任务操作。" />
            <Typography.Paragraph>活动单次任务：{activeRuns} · 活动长期任务：{activeCampaigns}</Typography.Paragraph>
          </section>
        </Content>
      </Layout>
    )
  }

  return (
    <Layout className={`console-layout theme-${mode}`}>
      <Header className="topbar">
        <div className="brand"><span><h1>算子优化控制台</h1><small>Autoresearch Console</small></span></div>
        <div className="topbar-status"><span className={`live-dot ${streamStatus}`} />{streamStatus === 'live' ? '实时' : '连接滞后'}<Tooltip title="当前服务器正在运行的控制面代码版本"><Text className="version-chip">版本 {snapshot.runtime_identity.git_commit.slice(0, 8)}</Text></Tooltip><Segmented size="small" value={mode} onChange={(value) => setMode(value as 'dark' | 'light')} options={[{ label: '暗色', value: 'dark' }, { label: '浅色', value: 'light' }]} /></div>
      </Header>
      <Layout>
        <Sider width={208} breakpoint="lg" collapsedWidth={68} className="console-sider">
          <Menu
            mode="inline"
            selectedKeys={[page]}
            onClick={({ key }) => navigate(key as Page)}
            items={[
              { key: 'create', label: '创建任务' },
              { key: 'runs', label: '运行记录' },
              { key: 'system', label: '系统状态' },
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
  const [mode, setMode] = useState<'dark' | 'light'>('light')
  return (
    <QueryClientProvider client={queryClient}>
      <ConfigProvider csp={nonce && nonce !== '__CSP_NONCE__' ? { nonce } : undefined} theme={{ algorithm: mode === 'dark' ? theme.darkAlgorithm : theme.defaultAlgorithm, token: { colorPrimary: '#2563eb', colorLink: '#174fb2', colorTextSecondary: mode === 'dark' ? '#b3c0d2' : '#58677c', colorTextDescription: mode === 'dark' ? '#b3c0d2' : '#58677c', borderRadius: 12, fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif' } }}>
        <AntApp><ConsoleApp mode={mode} setMode={setMode} /></AntApp>
      </ConfigProvider>
    </QueryClientProvider>
  )
}
