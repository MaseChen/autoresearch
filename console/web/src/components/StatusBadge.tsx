import { Tag } from 'antd'
import { statusLabel } from '../presentation'

const colors: Record<string, string> = {
  RUNNING: 'blue',
  SUCCESS: 'green',
  SUCCEEDED: 'green',
  COMPLETED: 'green',
  PROMOTED: 'green',
  FAILED: 'red',
  KNOWN_FAILURE: 'red',
  HARD_FAILED: 'volcano',
  PAUSED_HARD_FAILURE: 'volcano',
  UNKNOWN: 'purple',
  UNKNOWN_OUTCOME: 'purple',
  UNKNOWN_GPU_OUTCOME: 'purple',
  PAUSED_UNKNOWN_OUTCOME: 'purple',
  PAUSED_OPERATOR: 'gold',
  UNAVAILABLE: 'default',
  CHANGING: 'gold',
  STABLE: 'green',
}

export function StatusBadge({ value, reason }: { value?: string; reason?: string }) {
  const selected = value ?? 'UNAVAILABLE'
  const suffix = selected.includes('UNKNOWN')
    ? ' · 禁止重放'
    : selected === 'UNAVAILABLE'
      ? ` · ${reason?.trim() || '原因待记录'}`
      : ''
  return (
    <Tag
      className={selected === 'UNAVAILABLE' ? 'status-unavailable' : undefined}
      color={colors[selected] ?? 'default'}
      title={`原始状态：${selected}`}
    >
      {statusLabel(selected)}{suffix}
    </Tag>
  )
}
