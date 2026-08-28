import { loader } from '@monaco-editor/react'
import * as monaco from 'monaco-editor/editor/editor.api'
import EditorWorker from 'monaco-editor/editor/editor.worker?worker'
import 'monaco-editor/languages/definitions/python/register'

self.MonacoEnvironment = {
  getWorker() {
    return new EditorWorker()
  },
}

monaco.editor.defineTheme('kernel-research-light', {
  base: 'vs',
  inherit: true,
  rules: [],
  colors: {
    'editor.background': '#ffffff',
    'editor.foreground': '#172033',
    'editorCursor.foreground': '#174fb2',
    'editorGutter.background': '#ffffff',
    'editorLineNumber.foreground': '#52657e',
    'editorLineNumber.activeForeground': '#0b3f91',
    'editor.lineHighlightBackground': '#edf4ff',
    'editor.lineHighlightBorder': '#c8dcff',
  },
})

monaco.editor.defineTheme('kernel-research-dark', {
  base: 'vs-dark',
  inherit: true,
  rules: [],
  colors: {
    'editor.background': '#1e1e1e',
    'editor.foreground': '#e6edf7',
    'editorCursor.foreground': '#7fb3ff',
    'editorGutter.background': '#1e1e1e',
    'editorLineNumber.foreground': '#9dacbf',
    'editorLineNumber.activeForeground': '#ffffff',
    'editor.lineHighlightBackground': '#253650',
    'editor.lineHighlightBorder': '#41618f',
  },
})

loader.config({ monaco })
