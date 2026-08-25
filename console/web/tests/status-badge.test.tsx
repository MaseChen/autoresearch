import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { StatusBadge } from '../src/components/StatusBadge'

describe('StatusBadge', () => {
  it('labels unknown outcomes as no-replay', () => {
    render(<StatusBadge value="UNKNOWN_OUTCOME" />)
    expect(screen.getByText(/禁止重放/)).toBeInTheDocument()
  })

  it('does not turn unavailable into a numeric value', () => {
    render(<StatusBadge value="UNAVAILABLE" reason="COUNTER_NOT_EXPOSED" />)
    expect(screen.getByText(/COUNTER_NOT_EXPOSED/)).toBeInTheDocument()
    expect(screen.queryByText('0')).not.toBeInTheDocument()
  })

  it('uses safe defaults for missing and unfamiliar states', () => {
    const { rerender } = render(<StatusBadge />)
    expect(screen.getByText(/原因待记录/)).toBeInTheDocument()
    rerender(<StatusBadge value="CUSTOM_TERMINAL" />)
    expect(screen.getByText('CUSTOM_TERMINAL')).toBeInTheDocument()
    rerender(<StatusBadge value="SUCCEEDED" />)
    expect(screen.getByText('成功')).toBeInTheDocument()
    expect(screen.getByTitle('原始状态：SUCCEEDED')).toBeInTheDocument()
  })
})
