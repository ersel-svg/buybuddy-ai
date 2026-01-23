"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { useParams, useRouter } from "next/navigation";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  Panel,
  Handle,
  Position,
  addEdge,
  applyNodeChanges,
  applyEdgeChanges,
  type Connection,
  type Node,
  type Edge,
  type NodeChange,
  type EdgeChange,
  ReactFlowProvider,
  useReactFlow,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import { apiClient } from "@/lib/api-client";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  NodeConfigDrawer,
  WorkflowParameters,
  type NodeData,
  type WorkflowParameter,
  type NodeInfo,
  type EdgeInfo,
} from "@/components/workflows";
import { Separator } from "@/components/ui/separator";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  ArrowLeft,
  Save,
  Play,
  Loader2,
  Layers,
  Cpu,
  ScanLine,
  Box,
  GitBranch,
  Image,
  Filter,
  Grid,
  FileJson,
  Scissors,
  EyeOff,
  PenTool,
  Search,
  GripVertical,
  Variable,
} from "lucide-react";

// Node data type
interface WorkflowNodeData extends Record<string, unknown> {
  label: string;
  type: string;
  category: string;
  config?: Record<string, unknown>;
  model_id?: string;
  model_source?: "pretrained" | "trained";
}

// Custom node type
type WorkflowNode = Node<WorkflowNodeData>;

// Block category colors
const categoryColors: Record<string, string> = {
  input: "#3b82f6",
  model: "#8b5cf6",
  transform: "#10b981",
  logic: "#f59e0b",
  output: "#ef4444",
};

// Block icons
const blockIcons: Record<string, React.ReactNode> = {
  image_input: <Image className="h-4 w-4" />,
  detection: <ScanLine className="h-4 w-4" />,
  classification: <Layers className="h-4 w-4" />,
  embedding: <Cpu className="h-4 w-4" />,
  similarity_search: <Search className="h-4 w-4" />,
  crop: <Scissors className="h-4 w-4" />,
  blur_region: <EyeOff className="h-4 w-4" />,
  draw_boxes: <PenTool className="h-4 w-4" />,
  segmentation: <Box className="h-4 w-4" />,
  condition: <GitBranch className="h-4 w-4" />,
  filter: <Filter className="h-4 w-4" />,
  grid_builder: <Grid className="h-4 w-4" />,
  json_output: <FileJson className="h-4 w-4" />,
};

// Custom node component
function WorkflowNodeComponent({
  data,
  selected,
}: {
  data: WorkflowNodeData;
  selected?: boolean;
}) {
  const color = categoryColors[data.category] || "#6b7280";
  const icon = blockIcons[data.type] || <Box className="h-4 w-4" />;

  // Determine if node has input/output based on type
  const hasInput = data.type !== "image_input" && data.type !== "parameter_input";
  const hasOutput = data.type !== "json_output" && data.type !== "grid_builder";

  return (
    <div
      className={`px-4 py-3 rounded-lg border-2 bg-card shadow-md min-w-[160px] transition-all relative ${
        selected ? "ring-2 ring-primary ring-offset-2" : ""
      }`}
      style={{ borderColor: color }}
    >
      {/* Input Handle (left side) */}
      {hasInput && (
        <Handle
          type="target"
          position={Position.Left}
          id="image"
          className="!w-3 !h-3 !bg-gray-400 hover:!bg-blue-500 !border-2 !border-white"
        />
      )}

      <div className="flex items-center gap-2">
        <div
          className="p-1.5 rounded"
          style={{ backgroundColor: `${color}20`, color }}
        >
          {icon}
        </div>
        <span className="font-medium text-sm">{data.label}</span>
      </div>

      {/* Output Handle (right side) */}
      {hasOutput && (
        <Handle
          type="source"
          position={Position.Right}
          id="image"
          className="!w-3 !h-3 !bg-gray-400 hover:!bg-green-500 !border-2 !border-white"
        />
      )}
    </div>
  );
}

const nodeTypes = {
  workflowNode: WorkflowNodeComponent,
};

// Block definition type
interface BlockDef {
  type: string;
  display_name: string;
  description: string;
  category: string;
}

// Block palette component
function BlockPalette({
  blocks,
  onDragStart,
}: {
  blocks: BlockDef[];
  onDragStart: (event: React.DragEvent, block: BlockDef) => void;
}) {
  const categories = ["input", "model", "transform", "logic", "output"];
  const categoryLabels: Record<string, string> = {
    input: "Input",
    model: "Models",
    transform: "Transform",
    logic: "Logic",
    output: "Output",
  };

  return (
    <div className="w-64 border-r bg-muted/30 flex flex-col">
      <div className="p-3 border-b">
        <h3 className="font-semibold text-sm">Blocks</h3>
        <p className="text-xs text-muted-foreground mt-0.5">
          Drag blocks to canvas
        </p>
      </div>
      <ScrollArea className="flex-1">
        <div className="p-2 space-y-3">
          {categories.map((category) => {
            const safeBlocks = Array.isArray(blocks) ? blocks : [];
            const categoryBlocks = safeBlocks.filter((b) => b.category === category);
            if (categoryBlocks.length === 0) return null;

            return (
              <div key={category}>
                <div
                  className="text-xs font-medium uppercase tracking-wide mb-1.5 px-1"
                  style={{ color: categoryColors[category] }}
                >
                  {categoryLabels[category]}
                </div>
                <div className="space-y-1">
                  {categoryBlocks.map((block) => (
                    <div
                      key={block.type}
                      draggable
                      onDragStart={(e) => onDragStart(e, block)}
                      className="flex items-center gap-2 p-2 rounded-md border bg-card hover:bg-accent cursor-grab active:cursor-grabbing transition-colors"
                    >
                      <GripVertical className="h-3 w-3 text-muted-foreground" />
                      <div
                        className="p-1 rounded"
                        style={{
                          backgroundColor: `${categoryColors[category]}20`,
                          color: categoryColors[category],
                        }}
                      >
                        {blockIcons[block.type] || <Box className="h-3 w-3" />}
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="text-xs font-medium truncate">
                          {block.display_name}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      </ScrollArea>
    </div>
  );
}

// Main editor component
function WorkflowEditorContent() {
  const params = useParams();
  const router = useRouter();
  const queryClient = useQueryClient();
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const { screenToFlowPosition } = useReactFlow();

  const workflowId = params.id as string;

  // State with proper types - use useState instead of useNodesState for better control
  const [nodes, setNodes] = useState<WorkflowNode[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);
  const [parameters, setParameters] = useState<WorkflowParameter[]>([]);
  const [selectedNode, setSelectedNode] = useState<WorkflowNode | null>(null);
  const [parametersOpen, setParametersOpen] = useState(false);
  const [hasChanges, setHasChanges] = useState(false);

  // Fetch workflow
  const { data: workflow, isLoading: workflowLoading } = useQuery({
    queryKey: ["workflow", workflowId],
    queryFn: () => apiClient.getWorkflow(workflowId),
    enabled: !!workflowId,
  });

  // Fetch blocks registry
  const { data: blocks } = useQuery({
    queryKey: ["workflow-blocks"],
    queryFn: () => apiClient.getWorkflowBlocks(),
  });

  // Get block category
  const getBlockCategory = useCallback(
    (type: string): string => {
      const block = blocks?.find((b) => b.type === type);
      return block?.category || "logic";
    },
    [blocks]
  );

  // Load workflow definition into React Flow
  useEffect(() => {
    if (workflow?.definition) {
      const defNodes = workflow.definition.nodes as Array<{
        id: string;
        type: string;
        position: { x: number; y: number };
        data?: {
          label?: string;
          config?: Record<string, unknown>;
          model_id?: string;
          model_source?: "pretrained" | "trained";
        };
      }>;
      const defEdges = workflow.definition.edges as Array<{
        id: string;
        source: string;
        target: string;
        sourceHandle?: string;
        targetHandle?: string;
      }>;
      const defParams = ((workflow.definition as Record<string, unknown>).parameters as WorkflowParameter[]) || [];

      const flowNodes: WorkflowNode[] = (defNodes || []).map((node) => ({
        id: node.id,
        type: "workflowNode",
        position: node.position,
        data: {
          label: node.data?.label || node.type,
          type: node.type,
          category: getBlockCategory(node.type),
          config: node.data?.config || {},
          model_id: node.data?.model_id,
          model_source: node.data?.model_source,
        },
      }));

      const flowEdges: Edge[] = (defEdges || []).map((edge) => ({
        id: edge.id,
        source: edge.source,
        target: edge.target,
        sourceHandle: edge.sourceHandle,
        targetHandle: edge.targetHandle,
      }));

      setNodes(flowNodes);
      setEdges(flowEdges);
      setParameters(defParams);
    }
  }, [workflow, getBlockCategory]);

  // Save workflow mutation
  const saveMutation = useMutation({
    mutationFn: () => {
      const definition = {
        nodes: nodes.map((node) => ({
          id: node.id,
          type: node.data.type,
          position: node.position,
          data: {
            label: node.data.label,
            config: node.data.config || {},
            model_id: node.data.model_id,
            model_source: node.data.model_source,
          },
        })),
        edges: edges.map((edge) => ({
          id: edge.id,
          source: edge.source,
          target: edge.target,
          sourceHandle: edge.sourceHandle,
          targetHandle: edge.targetHandle,
        })),
        parameters: parameters,
      };
      return apiClient.updateWorkflow(workflowId, { definition });
    },
    onSuccess: () => {
      toast.success("Workflow saved");
      setHasChanges(false);
      queryClient.invalidateQueries({ queryKey: ["workflow", workflowId] });
    },
    onError: (error: Error) => {
      toast.error(`Failed to save: ${error.message}`);
    },
  });

  // Execute workflow mutation
  const executeMutation = useMutation({
    mutationFn: () => apiClient.executeWorkflow(workflowId, {}),
    onSuccess: (data) => {
      toast.success("Workflow execution started");
      router.push(`/workflows/executions?execution_id=${data.id}`);
    },
    onError: (error: Error) => {
      toast.error(`Failed to execute: ${error.message}`);
    },
  });

  // Handle node changes
  const onNodesChange = useCallback(
    (changes: NodeChange<WorkflowNode>[]) => {
      setNodes((nds) => applyNodeChanges(changes, nds));
      setHasChanges(true);
    },
    []
  );

  // Handle edge changes
  const onEdgesChange = useCallback(
    (changes: EdgeChange[]) => {
      setEdges((eds) => applyEdgeChanges(changes, eds));
      setHasChanges(true);
    },
    []
  );

  // Handle edge connection
  const onConnect = useCallback(
    (connection: Connection) => {
      setEdges((eds) => addEdge({ ...connection, id: `e-${Date.now()}` }, eds));
      setHasChanges(true);
    },
    []
  );

  // Handle node selection
  const onNodeClick = useCallback((_: React.MouseEvent, node: WorkflowNode) => {
    setSelectedNode(node);
  }, []);

  // Handle drag start from palette
  const onDragStart = (event: React.DragEvent, block: BlockDef) => {
    event.dataTransfer.setData("application/reactflow", JSON.stringify(block));
    event.dataTransfer.effectAllowed = "move";
  };

  // Handle drop on canvas
  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();

      const data = event.dataTransfer.getData("application/reactflow");
      if (!data) return;

      const block = JSON.parse(data) as BlockDef;
      const position = screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });

      const newNode: WorkflowNode = {
        id: `node-${Date.now()}`,
        type: "workflowNode",
        position,
        data: {
          label: block.display_name,
          type: block.type,
          category: block.category,
          config: {},
        },
      };

      setNodes((nds) => [...nds, newNode]);
      setHasChanges(true);
    },
    [screenToFlowPosition]
  );

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
  }, []);

  // Status badge
  const statusConfig: Record<string, { label: string; color: string }> = {
    draft: { label: "Draft", color: "bg-yellow-100 text-yellow-800" },
    active: { label: "Active", color: "bg-green-100 text-green-800" },
    archived: { label: "Archived", color: "bg-gray-100 text-gray-800" },
  };

  if (workflowLoading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <Loader2 className="h-8 w-8 animate-spin" />
      </div>
    );
  }

  if (!workflow) {
    return (
      <div className="flex flex-col items-center justify-center h-screen gap-4">
        <p className="text-muted-foreground">Workflow not found</p>
        <Button variant="outline" onClick={() => router.push("/workflows")}>
          <ArrowLeft className="h-4 w-4 mr-2" />
          Back to Workflows
        </Button>
      </div>
    );
  }

  const status = statusConfig[workflow.status] || statusConfig.draft;

  return (
    <div className="h-screen flex flex-col">
      {/* Header */}
      <div className="h-14 border-b bg-background flex items-center justify-between px-4 shrink-0">
        <div className="flex items-center gap-3">
          <Button
            variant="ghost"
            size="icon"
            onClick={() => router.push("/workflows")}
          >
            <ArrowLeft className="h-4 w-4" />
          </Button>
          <Separator orientation="vertical" className="h-6" />
          <div>
            <div className="flex items-center gap-2">
              <h1 className="font-semibold">{workflow.name}</h1>
              <Badge className={status.color}>{status.label}</Badge>
              {hasChanges && (
                <Badge variant="outline" className="text-orange-600 border-orange-300">
                  Unsaved
                </Badge>
              )}
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="icon"
            onClick={() => setParametersOpen(true)}
            className="relative"
          >
            <Variable className="h-4 w-4" />
            {parameters.length > 0 && (
              <span className="absolute -top-1 -right-1 h-4 w-4 rounded-full bg-primary text-[10px] text-primary-foreground flex items-center justify-center">
                {parameters.length}
              </span>
            )}
          </Button>
          <Separator orientation="vertical" className="h-6" />
          <Button
            variant="outline"
            onClick={() => saveMutation.mutate()}
            disabled={!hasChanges || saveMutation.isPending}
          >
            {saveMutation.isPending ? (
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            ) : (
              <Save className="h-4 w-4 mr-2" />
            )}
            Save
          </Button>
          <Button
            onClick={() => executeMutation.mutate()}
            disabled={executeMutation.isPending || nodes.length === 0}
          >
            {executeMutation.isPending ? (
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            ) : (
              <Play className="h-4 w-4 mr-2" />
            )}
            Execute
          </Button>
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Block palette */}
        <BlockPalette blocks={(blocks as BlockDef[]) || []} onDragStart={onDragStart} />

        {/* Canvas */}
        <div className="flex-1" ref={reactFlowWrapper}>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onNodeClick={onNodeClick}
            onDrop={onDrop}
            onDragOver={onDragOver}
            nodeTypes={nodeTypes}
            fitView
            snapToGrid
            snapGrid={[16, 16]}
            deleteKeyCode={["Backspace", "Delete"]}
          >
            <Background gap={16} />
            <Controls />
            <MiniMap
              nodeColor={(node) => {
                const nodeData = node.data as WorkflowNodeData | undefined;
                return categoryColors[nodeData?.category || ""] || "#6b7280";
              }}
              maskColor="rgba(0, 0, 0, 0.1)"
              className="!bg-card border"
            />
            <Panel position="bottom-left" className="text-xs text-muted-foreground">
              {nodes.length} nodes, {edges.length} connections
            </Panel>
          </ReactFlow>
        </div>

        {/* Node Configuration Drawer */}
        <NodeConfigDrawer
          open={!!selectedNode}
          onClose={() => setSelectedNode(null)}
          node={selectedNode ? { id: selectedNode.id, data: selectedNode.data as NodeData } : null}
          onNodeChange={(nodeId, updates) => {
            setNodes((nds) =>
              nds.map((n) =>
                n.id === nodeId
                  ? { ...n, data: { ...n.data, ...updates } }
                  : n
              )
            );
            setHasChanges(true);
          }}
          onNodeDelete={(nodeId) => {
            setNodes((nds) => nds.filter((n) => n.id !== nodeId));
            setEdges((eds) =>
              eds.filter((e) => e.source !== nodeId && e.target !== nodeId)
            );
            setSelectedNode(null);
            setHasChanges(true);
          }}
          onNodeDuplicate={(nodeId) => {
            const nodeToDuplicate = nodes.find((n) => n.id === nodeId);
            if (!nodeToDuplicate) return;

            const newNode: WorkflowNode = {
              id: `${nodeToDuplicate.data.type}_${Date.now()}`,
              type: "workflowNode",
              position: {
                x: nodeToDuplicate.position.x + 50,
                y: nodeToDuplicate.position.y + 50,
              },
              data: {
                ...nodeToDuplicate.data,
                label: `${nodeToDuplicate.data.label} (Copy)`,
              },
            };

            setNodes((nds) => [...nds, newNode]);
            setHasChanges(true);
            toast.success("Node duplicated");
          }}
          allNodes={nodes.map((n): NodeInfo => ({
            id: n.id,
            label: (n.data as WorkflowNodeData).label,
            type: (n.data as WorkflowNodeData).type,
            outputPorts: [], // Will be filled from BLOCK_PORTS in drawer
          }))}
          edges={edges.map((e): EdgeInfo => ({
            id: e.id,
            source: e.source,
            target: e.target,
            sourceHandle: e.sourceHandle || undefined,
            targetHandle: e.targetHandle || undefined,
          }))}
          onEdgeChange={(sourceId, sourceHandle, targetId, targetHandle) => {
            // Remove existing edge to this target handle
            setEdges((eds) => {
              const filtered = eds.filter(
                (e) => !(e.target === targetId && e.targetHandle === targetHandle)
              );
              // Add new edge
              const newEdge: Edge = {
                id: `e_${sourceId}_${targetId}_${Date.now()}`,
                source: sourceId,
                target: targetId,
                sourceHandle,
                targetHandle,
              };
              return [...filtered, newEdge];
            });
            setHasChanges(true);
          }}
        />

        {/* Workflow Parameters Panel */}
        <WorkflowParameters
          open={parametersOpen}
          onClose={() => setParametersOpen(false)}
          parameters={parameters}
          onParametersChange={(newParams) => {
            setParameters(newParams);
            setHasChanges(true);
          }}
        />
      </div>
    </div>
  );
}

// Wrap with ReactFlowProvider
export default function WorkflowEditorPage() {
  return (
    <ReactFlowProvider>
      <WorkflowEditorContent />
    </ReactFlowProvider>
  );
}
