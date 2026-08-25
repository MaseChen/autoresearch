import Editor from '@monaco-editor/react'
import '../monacoSetup'

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
  const lineCount = value.length === 0 ? 1 : value.split('\n').length

  return (
    <section className="code-editor-shell" aria-label="kernel.py 编辑区域">
      <header className="code-editor-toolbar">
        <strong>kernel.py</strong>
        <span>Python</span>
        {readOnly && <span className="code-editor-readonly">只读</span>}
      </header>
      <div className="code-editor-frame">
        <Editor
          height="520px"
          path="file:///candidate/kernel.py"
          language="python"
          theme={theme === 'dark' ? 'vs-dark' : 'vs'}
          value={value}
          onChange={(nextValue) => onChange(nextValue ?? '')}
          loading={<div className="code-editor-loading" role="status">正在加载代码编辑器…</div>}
          keepCurrentModel={false}
          options={{
            ariaLabel: 'kernel.py 代码编辑器',
            accessibilitySupport: 'auto',
            automaticLayout: true,
            bracketPairColorization: { enabled: true },
            contextmenu: true,
            cursorBlinking: 'blink',
            detectIndentation: false,
            folding: true,
            fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace',
            fontSize: 14,
            glyphMargin: false,
            guides: { bracketPairs: true, indentation: true },
            hideCursorInOverviewRuler: true,
            insertSpaces: true,
            lineHeight: 22,
            lineNumbers: 'on',
            lineNumbersMinChars: 3,
            minimap: { enabled: false },
            occurrencesHighlight: 'singleFile',
            overviewRulerLanes: 0,
            padding: { top: 14, bottom: 14 },
            readOnly,
            renderLineHighlight: 'all',
            renderWhitespace: 'selection',
            scrollBeyondLastLine: false,
            selectionHighlight: true,
            smoothScrolling: false,
            stickyScroll: { enabled: false },
            tabSize: 4,
            wordWrap: 'on',
          }}
        />
      </div>
      <footer className="code-editor-status" aria-label="编辑器状态">
        <span>{lineCount} 行</span>
        <span>空格: 4</span>
        <span>UTF-8</span>
      </footer>
    </section>
  )
}
