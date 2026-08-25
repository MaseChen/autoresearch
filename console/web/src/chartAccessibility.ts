import type { ConsoleRow } from './types'

export function aggregateScoreValue(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

export function aggregateScoreText(value: unknown): string {
  const selected = aggregateScoreValue(value)
  return selected === null ? 'UNAVAILABLE' : String(selected)
}

export function aggregateScoreChartSummary(rows: ConsoleRow[]): string {
  const available = rows.filter((row) => aggregateScoreValue(row.aggregate_score) !== null).length
  const unavailable = rows.length - available
  return [
    `近期实验 aggregate score 折线图：共 ${rows.length} 个实验，`,
    `${available} 个可用，${unavailable} 个 UNAVAILABLE。`,
    '详细数值和状态见后续数据表。',
  ].join('')
}
