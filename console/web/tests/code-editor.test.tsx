import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { CodeEditor } from '../src/components/CodeEditor'

describe('CodeEditor', () => {
  it('keeps code aligned and preserves the document when read-only changes', () => {
    const onChange = vi.fn()
    const { rerender } = render(<CodeEditor value={'import triton\nimport numpy\n'} onChange={onChange} />)
    const editor = screen.getByRole('textbox', { name: 'kernel.py 代码编辑器' })
    expect(editor).toHaveTextContent('import triton')
    expect(editor).toHaveAttribute('contenteditable', 'true')

    rerender(<CodeEditor value={'import triton\nimport numpy\n'} onChange={onChange} readOnly />)
    expect(editor).toHaveTextContent('import triton')
    expect(editor).toHaveAttribute('contenteditable', 'false')
  })

  it('shows a first-line prompt for an empty candidate', () => {
    render(<CodeEditor value="" onChange={vi.fn()} />)
    expect(screen.getByText('从第一行开始输入或粘贴 kernel.py')).toBeInTheDocument()
  })
})
