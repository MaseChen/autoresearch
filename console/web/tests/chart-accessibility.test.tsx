import { render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
  aggregateScoreChartSummary,
  aggregateScoreText,
  aggregateScoreValue,
} from '../src/chartAccessibility'

const chart = vi.hoisted(() => ({
  dispose: vi.fn(),
  resize: vi.fn(),
  setOption: vi.fn(),
}))

vi.mock('echarts/core', () => ({
  init: vi.fn(() => chart),
  use: vi.fn(),
}))

import { EChart } from '../src/components/EChart'

class ResizeObserverFixture {
  observe = vi.fn()
  disconnect = vi.fn()
}

describe('chart accessibility', () => {
  beforeEach(() => {
    vi.stubGlobal('ResizeObserver', ResizeObserverFixture)
    chart.dispose.mockClear()
    chart.resize.mockClear()
    chart.setOption.mockClear()
  })

  afterEach(() => vi.unstubAllGlobals())

  it('normalizes missing and non-finite aggregate scores', () => {
    expect(aggregateScoreValue(1.25)).toBe(1.25)
    expect(aggregateScoreValue(null)).toBeNull()
    expect(aggregateScoreValue(undefined)).toBeNull()
    expect(aggregateScoreValue('1.25')).toBeNull()
    expect(aggregateScoreValue(Number.NaN)).toBeNull()
    expect(aggregateScoreValue(Number.POSITIVE_INFINITY)).toBeNull()
    expect(aggregateScoreText(Number.NaN)).toBe('UNAVAILABLE')
    expect(aggregateScoreText(1.25)).toBe('1.25')
  })

  it('builds a controlled summary with explicit unavailable counts', () => {
    const summary = aggregateScoreChartSummary([
      { id: 1, aggregate_score: 2.5 },
      { id: 2, aggregate_score: null },
      { id: 3, aggregate_score: Number.NaN },
    ])
    expect(summary).toContain('1 个可用')
    expect(summary).toContain('2 个 UNAVAILABLE')
    expect(summary).not.toContain('NaN')
    expect(aggregateScoreChartSummary([])).toContain('0 个 UNAVAILABLE')
  })

  it('disables automatic numeric aria and preserves the controlled label', () => {
    const label = '近期实验 aggregate score 折线图：1 个 UNAVAILABLE。详细数值见后续数据表。'
    const { unmount } = render(
      <EChart
        option={{ series: [{ type: 'line', data: [1, null] }] }}
        label={label}
      />,
    )

    expect(chart.setOption).toHaveBeenCalledWith(expect.objectContaining({
      aria: { enabled: false },
    }))
    const image = screen.getByRole('img', { name: label })
    expect(image).toHaveAttribute('aria-label', expect.stringContaining('UNAVAILABLE'))
    expect(image).not.toHaveAttribute('aria-label', expect.stringContaining('NaN'))

    unmount()
    expect(chart.dispose).toHaveBeenCalledOnce()
  })
})
