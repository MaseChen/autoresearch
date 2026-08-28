import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { applyTabEdit } from '../src/codeEditorModel'
import { CodeEditor } from '../src/components/CodeEditor'

describe('CodeEditor', () => {
  it('uses the stable final-handoff textarea contract', () => {
    const onChange = vi.fn()
    const { rerender } = render(<CodeEditor value={'import triton\nimport numpy\n'} onChange={onChange} />)
    const editor = screen.getByRole('textbox', { name: 'kernel.py 代码编辑器' })
    expect(editor).toHaveValue('import triton\nimport numpy\n')
    expect(editor).toHaveClass('source-editor', 'code-editor-input')
    expect(editor).toHaveAttribute('data-theme', 'light')
    expect(editor).toHaveAttribute('wrap', 'off')
    expect(editor).toHaveAttribute('spellcheck', 'false')
    expect(editor).not.toHaveAttribute('readonly')
    expect(screen.getByText('3 行')).toBeInTheDocument()
    expect(screen.getByText('第 1 行，第 1 列')).toBeInTheDocument()
    expect(screen.getByText('空格: 4')).toBeInTheDocument()
    expect(screen.getByText('UTF-8')).toBeInTheDocument()

    fireEvent.change(editor, { target: { value: 'import triton\n' } })
    expect(onChange).toHaveBeenCalledWith('import triton\n')

    rerender(<CodeEditor value={'import triton\nimport numpy\n'} onChange={onChange} readOnly theme="dark" />)
    expect(editor).toHaveValue('import triton\nimport numpy\n')
    expect(editor).toHaveAttribute('readonly')
    expect(editor).toHaveAttribute('data-theme', 'dark')
    expect(screen.getByText('只读')).toBeInTheDocument()
  })

  it('tracks the native caret without a synthetic cursor layer', () => {
    render(<CodeEditor value={'one\ntwo'} onChange={vi.fn()} />)
    const editor = screen.getByRole('textbox', { name: 'kernel.py 代码编辑器' }) as HTMLTextAreaElement
    editor.setSelectionRange(6, 6)
    fireEvent.select(editor)
    expect(screen.getByText('第 2 行，第 3 列')).toBeInTheDocument()
  })

  it('inserts and removes four-space indentation with Tab and Shift+Tab', () => {
    expect(applyTabEdit('ab', 1, 1, false)).toEqual({ value: 'a    b', selectionStart: 5, selectionEnd: 5 })
    expect(applyTabEdit('    ab', 6, 6, true)).toEqual({ value: 'ab', selectionStart: 2, selectionEnd: 2 })
    expect(applyTabEdit('a\nb', 0, 3, false)).toEqual({ value: '    a\n    b', selectionStart: 4, selectionEnd: 11 })
    expect(applyTabEdit('plain', 2, 2, true)).toEqual({ value: 'plain', selectionStart: 2, selectionEnd: 2 })
  })

  it('counts an empty candidate as the first editable line', () => {
    render(<CodeEditor value="" onChange={vi.fn()} />)
    expect(screen.getByText('1 行')).toBeInTheDocument()
  })
})
