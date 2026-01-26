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
  WorkflowTestPanel,
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
  Maximize2,
  LayoutGrid,
  Combine,
  RotateCw,
  Sparkles,
  Undo2,
  Redo2,
  ZoomIn,
  ZoomOut,
  Maximize,
  Command,
  ChevronDown,
  ChevronRight,
  X,
  Keyboard,
  Info,
  Repeat,
  ListPlus,
  Shuffle,
} from "lucide-react";
import { Input } from "@/components/ui/input";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

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
  visualization: "#ec4899",
  output: "#ef4444",
};

// Block icons
const blockIcons: Record<string, React.ReactNode> = {
  image_input: <Image className="h-4 w-4" />,
  parameter_input: <Variable className="h-4 w-4" />,
  detection: <ScanLine className="h-4 w-4" />,
  classification: <Layers className="h-4 w-4" />,
  embedding: <Cpu className="h-4 w-4" />,
  similarity_search: <Search className="h-4 w-4" />,
  segmentation: <Box className="h-4 w-4" />,
  crop: <Scissors className="h-4 w-4" />,
  resize: <Maximize2 className="h-4 w-4" />,
  tile: <LayoutGrid className="h-4 w-4" />,
  stitch: <Combine className="h-4 w-4" />,
  rotate_flip: <RotateCw className="h-4 w-4" />,
  normalize: <Sparkles className="h-4 w-4" />,
  blur_region: <EyeOff className="h-4 w-4" />,
  draw_boxes: <PenTool className="h-4 w-4" />,
  condition: <GitBranch className="h-4 w-4" />,
  filter: <Filter className="h-4 w-4" />,
  foreach: <Repeat className="h-4 w-4" />,
  collect: <ListPlus className="h-4 w-4" />,
  map: <Shuffle className="h-4 w-4" />,
  grid_builder: <Grid className="h-4 w-4" />,
  json_output: <FileJson className="h-4 w-4" />,
};

// Custom node component with enhanced visuals
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
  const hasOutput = data.type !== "json_output";

  // Check if node is configured (has model selected for model blocks, etc)
  const isConfigured = data.model_id ||
    (data.config && Object.keys(data.config).length > 0) ||
    data.type === "image_input" ||
    data.type === "parameter_input";

  return (
    <div
      className={`group relative rounded-xl transition-all duration-200 ${
        selected
          ? "scale-105"
          : "hover:scale-[1.02]"
      }`}
      style={{
        filter: selected ? `drop-shadow(0 0 12px ${color}40)` : undefined,
      }}
    >
      {/* Main node card */}
      <div
        className={`px-4 py-3 rounded-xl border-2 bg-card shadow-lg min-w-[180px] transition-all relative overflow-hidden ${
          selected ? "border-opacity-100" : "border-opacity-60 hover:border-opacity-100"
        }`}
        style={{ borderColor: color }}
      >
        {/* Subtle gradient background */}
        <div
          className="absolute inset-0 opacity-[0.03]"
          style={{
            background: `linear-gradient(135deg, ${color} 0%, transparent 60%)`
          }}
        />

        {/* Input Handle (left side) */}
        {hasInput && (
          <div className="absolute left-0 top-1/2 -translate-y-1/2 -translate-x-1/2">
            <Handle
              type="target"
              position={Position.Left}
              className="!w-3.5 !h-3.5 !bg-gradient-to-r !from-blue-400 !to-blue-500 !border-2 !border-white !shadow-md hover:!scale-125 !transition-transform"
              style={{ position: 'relative', left: 0, transform: 'none' }}
            />
            {/* Port label on hover */}
            <span className="absolute left-4 top-1/2 -translate-y-1/2 text-[10px] text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none">
              input
            </span>
          </div>
        )}

        {/* Node content */}
        <div className="flex items-center gap-3 relative z-10">
          <div
            className="p-2 rounded-lg shrink-0 transition-all group-hover:scale-110"
            style={{ backgroundColor: `${color}15`, color }}
          >
            {icon}
          </div>
          <div className="flex-1 min-w-0">
            <div className="font-medium text-sm truncate">{data.label}</div>
            <div className="text-[10px] text-muted-foreground capitalize">
              {data.type.replace(/_/g, " ")}
            </div>
          </div>
          {/* Config status indicator */}
          {!isConfigured && (
            <div className="shrink-0">
              <div className="w-2 h-2 rounded-full bg-orange-400 animate-pulse" title="Needs configuration" />
            </div>
          )}
        </div>

        {/* Output Handle (right side) */}
        {hasOutput && (
          <div className="absolute right-0 top-1/2 -translate-y-1/2 translate-x-1/2">
            <Handle
              type="source"
              position={Position.Right}
              className="!w-3.5 !h-3.5 !bg-gradient-to-r !from-green-400 !to-green-500 !border-2 !border-white !shadow-md hover:!scale-125 !transition-transform"
              style={{ position: 'relative', right: 0, transform: 'none' }}
            />
            {/* Port label on hover */}
            <span className="absolute right-4 top-1/2 -translate-y-1/2 text-[10px] text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none">
              output
            </span>
          </div>
        )}

        {/* Category indicator line at bottom */}
        <div
          className="absolute bottom-0 left-4 right-4 h-0.5 rounded-full opacity-30"
          style={{ backgroundColor: color }}
        />
      </div>

      {/* Selection ring */}
      {selected && (
        <div
          className="absolute -inset-1 rounded-xl border-2 border-dashed opacity-40 pointer-events-none animate-pulse"
          style={{ borderColor: color }}
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

// Block palette component with search and collapsible categories
function BlockPalette({
  blocks,
  onDragStart,
  onQuickAdd,
}: {
  blocks: BlockDef[];
  onDragStart: (event: React.DragEvent, block: BlockDef) => void;
  onQuickAdd?: (block: BlockDef) => void;
}) {
  const [searchQuery, setSearchQuery] = useState("");
  const [collapsedCategories, setCollapsedCategories] = useState<Set<string>>(new Set());

  const categories = ["input", "model", "transform", "logic", "visualization", "output"];
  const categoryLabels: Record<string, string> = {
    input: "Input",
    model: "Models",
    transform: "Transform",
    logic: "Logic",
    visualization: "Visualization",
    output: "Output",
  };

  const safeBlocks = Array.isArray(blocks) ? blocks : [];

  // Filter blocks by search query
  const filteredBlocks = searchQuery.trim()
    ? safeBlocks.filter(
        (b) =>
          b.display_name.toLowerCase().includes(searchQuery.toLowerCase()) ||
          b.type.toLowerCase().includes(searchQuery.toLowerCase()) ||
          b.description?.toLowerCase().includes(searchQuery.toLowerCase())
      )
    : safeBlocks;

  const toggleCategory = (category: string) => {
    setCollapsedCategories((prev) => {
      const next = new Set(prev);
      if (next.has(category)) {
        next.delete(category);
      } else {
        next.add(category);
      }
      return next;
    });
  };

  return (
    <TooltipProvider delayDuration={300}>
      <div className="w-64 border-r bg-muted/30 flex flex-col h-full overflow-hidden">
        {/* Header with search */}
        <div className="p-3 border-b space-y-2 shrink-0">
          <div className="flex items-center justify-between">
            <h3 className="font-semibold text-sm">Blocks</h3>
            <span className="text-[10px] text-muted-foreground bg-muted px-1.5 py-0.5 rounded">
              {filteredBlocks.length}
            </span>
          </div>
          <div className="relative">
            <Search className="absolute left-2 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
            <Input
              placeholder="Search blocks..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="h-8 pl-7 pr-7 text-xs"
            />
            {searchQuery && (
              <button
                onClick={() => setSearchQuery("")}
                className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
              >
                <X className="h-3.5 w-3.5" />
              </button>
            )}
          </div>
          <p className="text-[10px] text-muted-foreground">
            Drag to canvas or double-click to add
          </p>
        </div>

        <ScrollArea className="flex-1 min-h-0">
          <div className="p-2 space-y-1">
            {categories.map((category) => {
              const categoryBlocks = filteredBlocks.filter((b) => b.category === category);
              if (categoryBlocks.length === 0) return null;

              const isCollapsed = collapsedCategories.has(category);

              return (
                <Collapsible
                  key={category}
                  open={!isCollapsed}
                  onOpenChange={() => toggleCategory(category)}
                >
                  <CollapsibleTrigger className="flex items-center gap-1.5 w-full px-1 py-1.5 hover:bg-accent/50 rounded-md transition-colors">
                    {isCollapsed ? (
                      <ChevronRight className="h-3 w-3 text-muted-foreground" />
                    ) : (
                      <ChevronDown className="h-3 w-3 text-muted-foreground" />
                    )}
                    <span
                      className="text-xs font-medium uppercase tracking-wide flex-1 text-left"
                      style={{ color: categoryColors[category] }}
                    >
                      {categoryLabels[category]}
                    </span>
                    <span className="text-[10px] text-muted-foreground">
                      {categoryBlocks.length}
                    </span>
                  </CollapsibleTrigger>
                  <CollapsibleContent>
                    <div className="space-y-0.5 mt-1 ml-1">
                      {categoryBlocks.map((block) => (
                        <Tooltip key={block.type}>
                          <TooltipTrigger asChild>
                            <div
                              draggable
                              onDragStart={(e) => onDragStart(e, block)}
                              onDoubleClick={() => onQuickAdd?.(block)}
                              className="flex items-center gap-2 p-2 rounded-md border bg-card hover:bg-accent hover:border-primary/30 cursor-grab active:cursor-grabbing transition-all group"
                            >
                              <GripVertical className="h-3 w-3 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
                              <div
                                className="p-1 rounded shrink-0"
                                style={{
                                  backgroundColor: `${categoryColors[category]}15`,
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
                          </TooltipTrigger>
                          <TooltipContent side="right" className="max-w-[200px]">
                            <p className="font-medium text-sm">{block.display_name}</p>
                            <p className="text-xs text-muted-foreground mt-0.5">
                              {block.description || `Add ${block.display_name} to workflow`}
                            </p>
                          </TooltipContent>
                        </Tooltip>
                      ))}
                    </div>
                  </CollapsibleContent>
                </Collapsible>
              );
            })}

            {/* No results message */}
            {filteredBlocks.length === 0 && searchQuery && (
              <div className="text-center py-8 text-muted-foreground">
                <Search className="h-8 w-8 mx-auto mb-2 opacity-50" />
                <p className="text-xs">No blocks found for "{searchQuery}"</p>
              </div>
            )}
          </div>
        </ScrollArea>

        {/* Keyboard hint */}
        <div className="p-2 border-t bg-muted/50 shrink-0">
          <div className="flex items-center gap-1.5 text-[10px] text-muted-foreground">
            <Keyboard className="h-3 w-3" />
            <span>Press</span>
            <kbd className="px-1 py-0.5 bg-background border rounded text-[9px]">⌘K</kbd>
            <span>for quick add</span>
          </div>
        </div>
      </div>
    </TooltipProvider>
  );
}

// History state for undo/redo
interface HistoryState {
  nodes: WorkflowNode[];
  edges: Edge[];
}

// Main editor component
function WorkflowEditorContent() {
  const params = useParams();
  const router = useRouter();
  const queryClient = useQueryClient();
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const { screenToFlowPosition, zoomIn, zoomOut, fitView, getZoom } = useReactFlow();

  const workflowId = params.id as string;

  // State with proper types
  const [nodes, setNodes] = useState<WorkflowNode[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);
  const [parameters, setParameters] = useState<WorkflowParameter[]>([]);
  const [selectedNode, setSelectedNode] = useState<WorkflowNode | null>(null);
  const [parametersOpen, setParametersOpen] = useState(false);
  const [testPanelOpen, setTestPanelOpen] = useState(false);
  const [hasChanges, setHasChanges] = useState(false);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [showShortcuts, setShowShortcuts] = useState(false);

  // History for undo/redo
  const [history, setHistory] = useState<HistoryState[]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const isUndoRedo = useRef(false);

  // Auto-save timer
  const autoSaveTimer = useRef<NodeJS.Timeout | null>(null);

  // Track zoom level
  useEffect(() => {
    const interval = setInterval(() => {
      try {
        setZoomLevel(getZoom());
      } catch {
        // Ignore if not mounted
      }
    }, 100);
    return () => clearInterval(interval);
  }, [getZoom]);

  // Push to history (debounced)
  const pushHistory = useCallback(() => {
    if (isUndoRedo.current) {
      isUndoRedo.current = false;
      return;
    }
    setHistory((prev) => {
      const newHistory = prev.slice(0, historyIndex + 1);
      newHistory.push({ nodes: [...nodes], edges: [...edges] });
      // Limit history to 50 states
      if (newHistory.length > 50) newHistory.shift();
      return newHistory;
    });
    setHistoryIndex((prev) => Math.min(prev + 1, 49));
  }, [nodes, edges, historyIndex]);

  // Undo
  const handleUndo = useCallback(() => {
    if (historyIndex > 0) {
      isUndoRedo.current = true;
      const prevState = history[historyIndex - 1];
      setNodes(prevState.nodes);
      setEdges(prevState.edges);
      setHistoryIndex((prev) => prev - 1);
      setHasChanges(true);
    }
  }, [history, historyIndex]);

  // Redo
  const handleRedo = useCallback(() => {
    if (historyIndex < history.length - 1) {
      isUndoRedo.current = true;
      const nextState = history[historyIndex + 1];
      setNodes(nextState.nodes);
      setEdges(nextState.edges);
      setHistoryIndex((prev) => prev + 1);
      setHasChanges(true);
    }
  }, [history, historyIndex]);

  // Auto-save state
  const [isAutoSaving, setIsAutoSaving] = useState(false);
  const lastSavedNodesRef = useRef<string>("");
  const lastSavedEdgesRef = useRef<string>("");

  // Auto-save with debounce (5 seconds after last change)
  useEffect(() => {
    if (hasChanges && nodes.length > 0) {
      // Check if content actually changed from last save
      const currentNodesHash = JSON.stringify(nodes.map(n => ({ id: n.id, pos: n.position, data: n.data })));
      const currentEdgesHash = JSON.stringify(edges.map(e => ({ id: e.id, s: e.source, t: e.target })));

      if (currentNodesHash === lastSavedNodesRef.current && currentEdgesHash === lastSavedEdgesRef.current) {
        return; // No actual changes
      }

      if (autoSaveTimer.current) {
        clearTimeout(autoSaveTimer.current);
      }
      autoSaveTimer.current = setTimeout(async () => {
        setIsAutoSaving(true);
        try {
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
            edges: edges.map((edge) => {
              const edgeData = edge.data as { sourcePort?: string; targetPort?: string } | undefined;
              return {
                id: edge.id,
                source: edge.source,
                target: edge.target,
                sourceHandle: edgeData?.sourcePort || edge.sourceHandle,
                targetHandle: edgeData?.targetPort || edge.targetHandle,
              };
            }),
            parameters: parameters,
          };
          await apiClient.updateWorkflow(workflowId, { definition });
          lastSavedNodesRef.current = currentNodesHash;
          lastSavedEdgesRef.current = currentEdgesHash;
          setHasChanges(false);
          toast.success("Auto-saved", { duration: 1500 });
        } catch {
          // Silent fail for auto-save, user can manually save
        } finally {
          setIsAutoSaving(false);
        }
      }, 5000); // 5 second debounce
    }
    return () => {
      if (autoSaveTimer.current) {
        clearTimeout(autoSaveTimer.current);
      }
    };
  }, [hasChanges, nodes, edges, parameters, workflowId]);

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
        // Don't set sourceHandle/targetHandle so edges render to default handles
        // Store actual port mapping in data for backend serialization
        data: {
          sourcePort: edge.sourceHandle,
          targetPort: edge.targetHandle,
        },
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
        edges: edges.map((edge) => {
          // Port mapping can be in edge.data (new format) or edge.sourceHandle/targetHandle (old format)
          const edgeData = edge.data as { sourcePort?: string; targetPort?: string } | undefined;
          return {
            id: edge.id,
            source: edge.source,
            target: edge.target,
            sourceHandle: edgeData?.sourcePort || edge.sourceHandle,
            targetHandle: edgeData?.targetPort || edge.targetHandle,
          };
        }),
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

  // Handle edge connection from canvas drag
  const onConnect = useCallback(
    (connection: Connection) => {
      if (!connection.source || !connection.target) return;

      // Create edge with default port mapping
      // User can refine exact ports via the drawer
      const newEdge: Edge = {
        id: `e-${Date.now()}`,
        source: connection.source,
        target: connection.target,
        data: {
          sourcePort: "output",  // Default output port
          targetPort: "input",   // Default input port
        },
      };
      setEdges((eds) => [...eds, newEdge]);
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

  // Quick add block to center of viewport
  const handleQuickAdd = useCallback(
    (block: BlockDef) => {
      // Get viewport center
      const position = screenToFlowPosition({
        x: window.innerWidth / 2,
        y: window.innerHeight / 2,
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
      pushHistory();
      toast.success(`Added ${block.display_name}`);
    },
    [screenToFlowPosition, pushHistory]
  );

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Cmd/Ctrl + S: Save
      if ((e.metaKey || e.ctrlKey) && e.key === "s") {
        e.preventDefault();
        if (hasChanges && !saveMutation.isPending) {
          saveMutation.mutate();
        }
      }

      // Cmd/Ctrl + Z: Undo
      if ((e.metaKey || e.ctrlKey) && e.key === "z" && !e.shiftKey) {
        e.preventDefault();
        handleUndo();
      }

      // Cmd/Ctrl + Shift + Z or Cmd/Ctrl + Y: Redo
      if (
        ((e.metaKey || e.ctrlKey) && e.shiftKey && e.key === "z") ||
        ((e.metaKey || e.ctrlKey) && e.key === "y")
      ) {
        e.preventDefault();
        handleRedo();
      }

      // Cmd/Ctrl + D: Duplicate selected node
      if ((e.metaKey || e.ctrlKey) && e.key === "d" && selectedNode) {
        e.preventDefault();
        const newNode: WorkflowNode = {
          id: `${selectedNode.data.type}_${Date.now()}`,
          type: "workflowNode",
          position: {
            x: selectedNode.position.x + 50,
            y: selectedNode.position.y + 50,
          },
          data: {
            ...selectedNode.data,
            label: `${selectedNode.data.label} (Copy)`,
          },
        };
        setNodes((nds) => [...nds, newNode]);
        setHasChanges(true);
        pushHistory();
        toast.success("Node duplicated");
      }

      // Escape: Deselect node
      if (e.key === "Escape") {
        setSelectedNode(null);
      }

      // ?: Show shortcuts
      if (e.key === "?" && !e.metaKey && !e.ctrlKey) {
        e.preventDefault();
        setShowShortcuts((prev) => !prev);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [hasChanges, saveMutation, handleUndo, handleRedo, selectedNode, pushHistory]);

  // Push history on significant changes
  useEffect(() => {
    if (nodes.length > 0 || edges.length > 0) {
      const timer = setTimeout(() => {
        pushHistory();
      }, 500);
      return () => clearTimeout(timer);
    }
  }, [nodes.length, edges.length]);

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
    <TooltipProvider delayDuration={300}>
      <div className="h-screen flex flex-col">
        {/* Header */}
        <div className="h-14 border-b bg-background flex items-center justify-between px-4 shrink-0">
          <div className="flex items-center gap-3">
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => router.push("/workflows")}
                >
                  <ArrowLeft className="h-4 w-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Back to Workflows</TooltipContent>
            </Tooltip>
            <Separator orientation="vertical" className="h-6" />
            <div>
              <div className="flex items-center gap-2">
                <h1 className="font-semibold">{workflow.name}</h1>
                <Badge className={status.color}>{status.label}</Badge>
                {isAutoSaving ? (
                  <Badge variant="outline" className="text-blue-600 border-blue-300">
                    <Loader2 className="h-3 w-3 mr-1 animate-spin" />
                    Saving...
                  </Badge>
                ) : hasChanges ? (
                  <Badge variant="outline" className="text-orange-600 border-orange-300 animate-pulse">
                    Unsaved
                  </Badge>
                ) : null}
              </div>
            </div>
          </div>

          {/* Center toolbar */}
          <div className="flex items-center gap-1 bg-muted/50 rounded-lg p-1">
            {/* Undo/Redo */}
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8"
                  onClick={handleUndo}
                  disabled={historyIndex <= 0}
                >
                  <Undo2 className="h-4 w-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Undo (⌘Z)</TooltipContent>
            </Tooltip>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8"
                  onClick={handleRedo}
                  disabled={historyIndex >= history.length - 1}
                >
                  <Redo2 className="h-4 w-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Redo (⌘⇧Z)</TooltipContent>
            </Tooltip>

            <Separator orientation="vertical" className="h-4 mx-1" />

            {/* Zoom controls */}
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8"
                  onClick={() => zoomOut()}
                >
                  <ZoomOut className="h-4 w-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Zoom Out</TooltipContent>
            </Tooltip>
            <span className="text-xs text-muted-foreground w-12 text-center tabular-nums">
              {Math.round(zoomLevel * 100)}%
            </span>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8"
                  onClick={() => zoomIn()}
                >
                  <ZoomIn className="h-4 w-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Zoom In</TooltipContent>
            </Tooltip>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8"
                  onClick={() => fitView({ padding: 0.2 })}
                >
                  <Maximize className="h-4 w-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Fit View</TooltipContent>
            </Tooltip>

            <Separator orientation="vertical" className="h-4 mx-1" />

            {/* Shortcuts */}
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8"
                  onClick={() => setShowShortcuts(true)}
                >
                  <Keyboard className="h-4 w-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Shortcuts (?)</TooltipContent>
            </Tooltip>
          </div>

          {/* Right actions */}
          <div className="flex items-center gap-2">
            <Tooltip>
              <TooltipTrigger asChild>
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
              </TooltipTrigger>
              <TooltipContent>Parameters</TooltipContent>
            </Tooltip>
            <Separator orientation="vertical" className="h-6" />
            <Tooltip>
              <TooltipTrigger asChild>
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
              </TooltipTrigger>
              <TooltipContent>Save (⌘S)</TooltipContent>
            </Tooltip>
            <Button
              onClick={() => setTestPanelOpen(true)}
              disabled={nodes.length === 0}
            >
              <Play className="h-4 w-4 mr-2" />
              Test
            </Button>
          </div>
        </div>

        {/* Keyboard Shortcuts Dialog */}
        {showShortcuts && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50" onClick={() => setShowShortcuts(false)}>
            <div className="bg-background border rounded-lg shadow-lg p-6 max-w-md w-full mx-4" onClick={(e) => e.stopPropagation()}>
              <div className="flex items-center justify-between mb-4">
                <h2 className="font-semibold text-lg">Keyboard Shortcuts</h2>
                <Button variant="ghost" size="icon" onClick={() => setShowShortcuts(false)}>
                  <X className="h-4 w-4" />
                </Button>
              </div>
              <div className="space-y-3 text-sm">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Save workflow</span>
                  <kbd className="px-2 py-1 bg-muted rounded text-xs">⌘S</kbd>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Undo</span>
                  <kbd className="px-2 py-1 bg-muted rounded text-xs">⌘Z</kbd>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Redo</span>
                  <kbd className="px-2 py-1 bg-muted rounded text-xs">⌘⇧Z</kbd>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Duplicate node</span>
                  <kbd className="px-2 py-1 bg-muted rounded text-xs">⌘D</kbd>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Delete selected</span>
                  <kbd className="px-2 py-1 bg-muted rounded text-xs">⌫</kbd>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Deselect</span>
                  <kbd className="px-2 py-1 bg-muted rounded text-xs">Esc</kbd>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Show shortcuts</span>
                  <kbd className="px-2 py-1 bg-muted rounded text-xs">?</kbd>
                </div>
              </div>
            </div>
          </div>
        )}

      {/* Main content */}
      <div className="flex-1 flex min-h-0">
        {/* Block palette */}
        <BlockPalette blocks={(blocks as BlockDef[]) || []} onDragStart={onDragStart} onQuickAdd={handleQuickAdd} />

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
          node={selectedNode ? (() => {
            // Get fresh node data from nodes array to avoid stale state
            const currentNode = nodes.find(n => n.id === selectedNode.id);
            return currentNode
              ? { id: currentNode.id, data: currentNode.data as NodeData }
              : { id: selectedNode.id, data: selectedNode.data as NodeData };
          })() : null}
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
          edges={edges.map((e): EdgeInfo => {
            // Get port mapping from edge.data (new format) or edge handles (old format)
            const edgeData = e.data as { sourcePort?: string; targetPort?: string } | undefined;
            return {
              id: e.id,
              source: e.source,
              target: e.target,
              sourceHandle: edgeData?.sourcePort || e.sourceHandle || undefined,
              targetHandle: edgeData?.targetPort || e.targetHandle || undefined,
            };
          })}
          onEdgeChange={(sourceId, sourceHandle, targetId, targetHandle) => {
            // Remove existing edge to this target (only one connection per target input)
            setEdges((eds) => {
              const filtered = eds.filter(
                (e) => !(e.target === targetId && (e.data as { targetPort?: string })?.targetPort === targetHandle)
              );
              // Add new edge - store port mapping in data for backend,
              // use undefined handles for visual rendering (default handles)
              const newEdge: Edge = {
                id: `e_${sourceId}_${targetId}_${Date.now()}`,
                source: sourceId,
                target: targetId,
                // Don't set sourceHandle/targetHandle so edge renders to default handles
                data: {
                  sourcePort: sourceHandle,  // Actual source port name for backend
                  targetPort: targetHandle,  // Actual target port name for backend
                },
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

        {/* Workflow Test Panel */}
        <WorkflowTestPanel
          open={testPanelOpen}
          onClose={() => setTestPanelOpen(false)}
          workflowId={workflowId}
          workflowName={workflow?.name || "Workflow"}
          parameters={parameters}
        />
      </div>
    </div>
    </TooltipProvider>
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
