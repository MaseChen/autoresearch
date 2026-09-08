import { useEffect, useRef } from 'react'
import * as echarts from 'echarts/core'
import { BarChart, BoxplotChart, HeatmapChart, LineChart } from 'echarts/charts'
import {
  DatasetComponent,
  GridComponent,
  LegendComponent,
  TooltipComponent,
} from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import type { EChartsCoreOption } from 'echarts/core'

echarts.use([
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
    const chart = echarts.init(element.current, undefined, { renderer: 'canvas' })
    // ECharts' generated numeric narration renders null samples as "NaN".
    // The caller owns the bounded human-readable summary and data table.
    chart.setOption({ ...option, aria: { enabled: false } })
    const observer = new ResizeObserver(() => chart.resize())
    observer.observe(element.current)
    return () => {
      observer.disconnect()
      chart.dispose()
    }
  }, [option])

  return <div ref={element} className="chart" role="img" aria-label={label} />
}
