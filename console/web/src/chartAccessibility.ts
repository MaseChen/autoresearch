import type { ConsoleRow } from './types'

export function aggregateScoreValue(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

export function aggregateScoreText(value: unknown): string {
  const selected = aggregateScoreValue(value)
  return selected === null ? '数据不可用' : String(selected)
}

export function aggregateScoreChartSummary(rows: ConsoleRow[]): string {
  const available = rows.filter((row) => aggregateScoreValue(row.aggregate_score) !== null).length
  const unavailable = rows.length - available
  return [
    `近期评测评分折线图：共 ${rows.length} 次评测，`,
    `${available} 次有评分，${unavailable} 次数据不可用。`,
    '详细数据见图表下方。',
  ].join('')
}
