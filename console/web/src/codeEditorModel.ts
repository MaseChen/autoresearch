const INDENT = '    '

export interface EditResult {
  value: string
  selectionStart: number
  selectionEnd: number
}

export function applyTabEdit(value: string, selectionStart: number, selectionEnd: number, outdent: boolean): EditResult {
  if (selectionStart === selectionEnd && !outdent) {
    return {
      value: value.slice(0, selectionStart) + INDENT + value.slice(selectionEnd),
      selectionStart: selectionStart + INDENT.length,
      selectionEnd: selectionStart + INDENT.length,
    }
  }

  const lineStart = value.lastIndexOf('\n', Math.max(0, selectionStart - 1)) + 1
  const effectiveEnd = selectionEnd > selectionStart && value[selectionEnd - 1] === '\n' ? selectionEnd - 1 : selectionEnd
  const nextLine = value.indexOf('\n', effectiveEnd)
  const blockEnd = nextLine === -1 ? value.length : nextLine
  const lines = value.slice(lineStart, blockEnd).split('\n')

  if (outdent) {
    const removed = lines.map((line) => line.match(/^( {1,4}|\t)/)?.[0].length ?? 0)
    if (removed.every((count) => count === 0)) return { value, selectionStart, selectionEnd }
    const replacement = lines.map((line, index) => line.slice(removed[index])).join('\n')
    const totalRemoved = removed.reduce((total, count) => total + count, 0)
    const firstAdjustment = selectionStart > lineStart ? removed[0] : 0
    return {
      value: value.slice(0, lineStart) + replacement + value.slice(blockEnd),
      selectionStart: Math.max(lineStart, selectionStart - firstAdjustment),
      selectionEnd: Math.max(lineStart, selectionEnd - totalRemoved),
    }
  }

  const replacement = lines.map((line) => INDENT + line).join('\n')
  return {
    value: value.slice(0, lineStart) + replacement + value.slice(blockEnd),
    selectionStart: selectionStart + INDENT.length,
    selectionEnd: selectionEnd + INDENT.length * lines.length,
  }
}
