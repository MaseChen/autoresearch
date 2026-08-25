import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

vi.mock('../src/monacoSetup', () => ({}))
vi.mock('@monaco-editor/react', () => ({
  default: ({
    language,
    onChange,
    options,
    theme,
    value,
  }: {
    language: string
    onChange: (value: string) => void
    options: { ariaLabel: string; readOnly: boolean }
    theme: string
    value: string
  }) => (
    <textarea
      aria-label={options.ariaLabel}
      data-language={language}
      data-theme={theme}
      onChange={(event) => onChange(event.target.value)}
      readOnly={options.readOnly}
      value={value}
    />
  ),
}))

import { CodeEditor } from '../src/components/CodeEditor'

describe('CodeEditor', () => {
  it('preserves the document and exposes the expected editor contract', () => {
    const onChange = vi.fn()
    const { rerender } = render(<CodeEditor value={'import triton\nimport numpy\n'} onChange={onChange} />)
    const editor = screen.getByRole('textbox', { name: 'kernel.py 代码编辑器' })
    expect(editor).toHaveValue('import triton\nimport numpy\n')
    expect(editor).toHaveAttribute('data-language', 'python')
    expect(editor).toHaveAttribute('data-theme', 'vs')
    expect(editor).not.toHaveAttribute('readonly')
    expect(screen.getByText('3 行')).toBeInTheDocument()
    expect(screen.getByText('空格: 4')).toBeInTheDocument()
    expect(screen.getByText('UTF-8')).toBeInTheDocument()

    fireEvent.change(editor, { target: { value: 'import triton\n' } })
    expect(onChange).toHaveBeenCalledWith('import triton\n')

    rerender(<CodeEditor value={'import triton\nimport numpy\n'} onChange={onChange} readOnly theme="dark" />)
    expect(editor).toHaveValue('import triton\nimport numpy\n')
    expect(editor).toHaveAttribute('readonly')
    expect(editor).toHaveAttribute('data-theme', 'vs-dark')
    expect(screen.getByText('只读')).toBeInTheDocument()
  })

  it('counts an empty candidate as the first editable line', () => {
    render(<CodeEditor value="" onChange={vi.fn()} />)
    expect(screen.getByText('1 行')).toBeInTheDocument()
  })
})
