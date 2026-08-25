import { useEffect, useRef } from 'react'
import * as echarts from 'echarts/core'
import { BarChart, BoxplotChart, HeatmapChart, LineChart } from 'echarts/charts'
import {
  AriaComponent,
  DatasetComponent,
  GridComponent,
  LegendComponent,
  TooltipComponent,
} from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import type { EChartsCoreOption } from 'echarts/core'

echarts.use([
  AriaComponent,
  BarChart,
  BoxplotChart,
  CanvasRenderer,
  DatasetComponent,
  GridComponent,
  HeatmapChart,
  LegendComponent,
  LineChart,
  TooltipComponent,
])

export function EChart({ option, label }: { option: EChartsCoreOption; label: string }) {
  const element = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!element.current) return
    const container = element.current
    const chart = echarts.init(container, undefined, { renderer: 'canvas' })
    chart.setOption({ ...option, aria: { enabled: false } })
    container.setAttribute('role', 'img')
    container.setAttribute('aria-label', label)
    const observer = new ResizeObserver(() => chart.resize())
    observer.observe(container)
    return () => {
      observer.disconnect()
      chart.dispose()
    }
  }, [label, option])

  return <div ref={element} className="chart" role="img" aria-label={label} />
}
