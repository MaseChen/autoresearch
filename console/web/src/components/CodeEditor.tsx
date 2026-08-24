import { useEffect, useRef } from 'react'
import { basicSetup } from 'codemirror'
import { python } from '@codemirror/lang-python'
import { EditorState } from '@codemirror/state'
import { EditorView } from '@codemirror/view'

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
  const initialValue = useRef(value)

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
          EditorState.readOnly.of(readOnly),
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
  }, [readOnly])

  useEffect(() => {
    const view = viewRef.current
    if (!view || view.state.doc.toString() === value) return
    view.dispatch({
      changes: { from: 0, to: view.state.doc.length, insert: value },
    })
  }, [value])

  return <div ref={container} className="code-editor" aria-label="kernel.py 编辑器" />
}
