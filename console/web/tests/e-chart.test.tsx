import { render } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

const setOption = vi.fn()
const dispose = vi.fn()
const resize = vi.fn()

vi.mock('echarts/core', () => ({
  init: () => ({ setOption, dispose, resize }),
  use: vi.fn(),
}))
vi.mock('echarts/charts', () => ({
  BarChart: {}, BoxplotChart: {}, HeatmapChart: {}, LineChart: {},
}))
vi.mock('echarts/components', () => ({
  DatasetComponent: {}, GridComponent: {}, LegendComponent: {}, TooltipComponent: {},
}))
vi.mock('echarts/renderers', () => ({ CanvasRenderer: {} }))

class ResizeObserverStub {
  observe() {}
  disconnect() {}
}
vi.stubGlobal('ResizeObserver', ResizeObserverStub)

import { EChart } from '../src/components/EChart'

describe('EChart', () => {
  it('uses the controlled summary and disables generated NaN narration', () => {
    const { container } = render(
      <EChart
        label="性能数据含 UNAVAILABLE；完整数值见下方数据表"
        option={{ series: [{ type: 'line', data: [1, null, 2] }] }}
      />,
    )

    expect(setOption).toHaveBeenCalledWith(expect.objectContaining({ aria: { enabled: false } }))
    expect(container.textContent).not.toContain('NaN')
    expect(container.querySelector('[role="img"]')).toHaveAttribute(
      'aria-label',
      '性能数据含 UNAVAILABLE；完整数值见下方数据表',
    )
  })
})
