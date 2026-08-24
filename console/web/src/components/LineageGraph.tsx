import { Background, Controls, ReactFlow, type Edge, type Node } from '@xyflow/react'
import '@xyflow/react/dist/style.css'

export function LineageGraph({ ids }: { ids: string[] }) {
  const nodes: Node[] = ids.slice(0, 8).map((id, index) => ({
    id,
    position: { x: index * 190, y: index % 2 === 0 ? 30 : 120 },
    data: { label: id.length > 18 ? `${id.slice(0, 16)}…` : id },
    draggable: false,
    selectable: true,
  }))
  const edges: Edge[] = nodes.slice(1).map((node, index) => ({
    id: `${nodes[index].id}-${node.id}`,
    source: nodes[index].id,
    target: node.id,
  }))
  return (
    <div className="lineage" aria-label="Baseline lineage 图">
      <ReactFlow nodes={nodes} edges={edges} fitView nodesDraggable={false}>
        <Background />
        <Controls showInteractive={false} />
      </ReactFlow>
    </div>
  )
}
