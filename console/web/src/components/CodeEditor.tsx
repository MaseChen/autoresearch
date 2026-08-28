import { useRef, useState, type ChangeEvent, type KeyboardEvent, type SyntheticEvent } from 'react'
import { applyTabEdit } from '../codeEditorModel'

function caretPosition(value: string, offset: number): { line: number; column: number } {
  const prefix = value.slice(0, Math.max(0, Math.min(offset, value.length)))
  const lines = prefix.split('\n')
  return { line: lines.length, column: (lines.at(-1)?.length ?? 0) + 1 }
}

export function CodeEditor({
  value,
  onChange,
  readOnly = false,
  theme = 'light',
}: {
  value: string
  onChange: (value: string) => void
  readOnly?: boolean
  theme?: 'dark' | 'light'
}) {
  const editorRef = useRef<HTMLTextAreaElement>(null)
  const [caret, setCaret] = useState({ line: 1, column: 1 })
  const lineCount = value.length === 0 ? 1 : value.split('\n').length

  const updateCaret = (event: SyntheticEvent<HTMLTextAreaElement>) => {
    setCaret(caretPosition(value, event.currentTarget.selectionStart))
  }

  const change = (event: ChangeEvent<HTMLTextAreaElement>) => {
    const nextValue = event.currentTarget.value
    onChange(nextValue)
    setCaret(caretPosition(nextValue, event.currentTarget.selectionStart))
  }

  const keyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (readOnly || event.key !== 'Tab') return
    event.preventDefault()
    const result = applyTabEdit(value, event.currentTarget.selectionStart, event.currentTarget.selectionEnd, event.shiftKey)
    onChange(result.value)
    setCaret(caretPosition(result.value, result.selectionStart))
    requestAnimationFrame(() => {
      editorRef.current?.focus()
      editorRef.current?.setSelectionRange(result.selectionStart, result.selectionEnd)
    })
  }

  return (
    <section className="code-editor-shell" aria-label="kernel.py 编辑区域">
      <header className="code-editor-toolbar">
        <strong>kernel.py</strong>
        <span>Python</span>
        {readOnly && <span className="code-editor-readonly">只读</span>}
      </header>
      <textarea
        ref={editorRef}
        className="source-editor code-editor-input"
        aria-label="kernel.py 代码编辑器"
        data-theme={theme}
        value={value}
        onChange={change}
        onKeyDown={keyDown}
        onSelect={updateCaret}
        onClick={updateCaret}
        onKeyUp={updateCaret}
        readOnly={readOnly}
        spellCheck={false}
        autoCapitalize="off"
        autoCorrect="off"
        wrap="off"
      />
      <footer className="code-editor-status" aria-label="编辑器状态">
        <span>{lineCount} 行</span>
        <span>第 {caret.line} 行，第 {caret.column} 列</span>
        <span>空格: 4</span>
        <span>UTF-8</span>
      </footer>
    </section>
  )
}
