import { useEffect, useRef } from 'react'
import { basicSetup } from 'codemirror'
import { python } from '@codemirror/lang-python'
import { Compartment, EditorState } from '@codemirror/state'
import { EditorView, placeholder } from '@codemirror/view'

const editorTheme = EditorView.theme({
  '&': {
    height: '420px',
    fontSize: '14px',
    backgroundColor: 'transparent',
  },
  '&.cm-focused': { outline: 'none' },
  '.cm-scroller': {
    overflow: 'auto',
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace',
    lineHeight: '1.65',
  },
  '.cm-content': {
    minHeight: '100%',
    padding: '12px 0',
    caretColor: '#2563eb',
  },
  '.cm-line': { padding: '0 14px' },
  '.cm-gutters': {
    backgroundColor: '#f5f7fb',
    borderRight: '1px solid #dbe3ee',
  },
  '.cm-gutterElement': {
    minWidth: '40px',
    padding: '0 10px 0 6px',
    color: '#66758a',
  },
  '.cm-cursor': { borderLeftWidth: '2px', borderLeftColor: '#2563eb' },
  '.cm-placeholder': { color: '#77869a', fontStyle: 'normal' },
  '.cm-activeLine': { backgroundColor: '#f5f8ff' },
  '.cm-activeLineGutter': { backgroundColor: '#eaf1ff', color: '#174fb2' },
})

export function CodeEditor({
  value,
  onChange,
  readOnly = false,
}: {
  value: string
  onChange: (value: string) => void
  readOnly?: boolean
}) {
  const container = useRef<HTMLDivElement>(null)
  const callback = useRef(onChange)
  const viewRef = useRef<EditorView | null>(null)
  const access = useRef(new Compartment())
  const initialValue = useRef(value)
  const initialReadOnly = useRef(readOnly)

  useEffect(() => {
    callback.current = onChange
  }, [onChange])

  useEffect(() => {
    if (!container.current) return
    const view = new EditorView({
      parent: container.current,
      state: EditorState.create({
        doc: initialValue.current,
        extensions: [
          basicSetup,
          python(),
          editorTheme,
          placeholder('从第一行开始输入或粘贴 kernel.py'),
          EditorView.contentAttributes.of({ 'aria-label': 'kernel.py 代码编辑器' }),
          access.current.of([
            EditorState.readOnly.of(initialReadOnly.current),
            EditorView.editable.of(!initialReadOnly.current),
          ]),
          EditorView.updateListener.of((update) => {
            if (update.docChanged) callback.current(update.state.doc.toString())
          }),
        ],
      }),
    })
    viewRef.current = view
    return () => {
      viewRef.current = null
      view.destroy()
    }
  }, [])

  useEffect(() => {
    const view = viewRef.current
    if (!view) return
    view.dispatch({
      effects: access.current.reconfigure([
        EditorState.readOnly.of(readOnly),
        EditorView.editable.of(!readOnly),
      ]),
    })
  }, [readOnly])

  useEffect(() => {
    const view = viewRef.current
    if (!view || view.state.doc.toString() === value) return
    view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: value } })
  }, [value])

  return <div ref={container} className="code-editor" />
}
