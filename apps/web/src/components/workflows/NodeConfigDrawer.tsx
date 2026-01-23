"use client";

import { ReactNode, useMemo } from "react";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import {
  Image,
  ScanSearch,
  Tags,
  Binary,
  Search,
  Crop,
  Filter,
  GitBranch,
  Grid3X3,
  FileJson,
  Eraser,
  PenTool,
  Settings2,
  Trash2,
  Copy,
  X,
  Box,
  ArrowRight,
  Circle,
  CheckCircle2,
  AlertCircle,
} from "lucide-react";
import { ModelSelector } from "./ModelSelector";
import { NodeConfigPanel, NodeConfig } from "./NodeConfigPanel";

// =============================================================================
// Types
// =============================================================================

export interface NodeData {
  label: string;
  type: string;
  category: string;
  config?: NodeConfig;
  model_id?: string;
  model_source?: "pretrained" | "trained";
}

/** Port definition for inputs/outputs */
export interface PortDefinition {
  name: string;
  type: string;
  required?: boolean;
  description?: string;
}

/** Edge connection info */
export interface EdgeInfo {
  id: string;
  source: string;
  target: string;
  sourceHandle?: string;
  targetHandle?: string;
}

/** Simple node info for dropdown */
export interface NodeInfo {
  id: string;
  label: string;
  type: string;
  outputPorts: PortDefinition[];
}

export interface NodeConfigDrawerProps {
  /** Whether the drawer is open */
  open: boolean;
  /** Callback when drawer should close */
  onClose: () => void;
  /** The selected node data */
  node: { id: string; data: NodeData } | null;
  /** Callback when node data changes */
  onNodeChange: (nodeId: string, updates: Partial<NodeData>) => void;
  /** Callback when node should be deleted */
  onNodeDelete: (nodeId: string) => void;
  /** Callback when node should be duplicated */
  onNodeDuplicate?: (nodeId: string) => void;
  /** All nodes in the workflow (for input mapping) */
  allNodes?: NodeInfo[];
  /** All edges in the workflow */
  edges?: EdgeInfo[];
  /** Callback when edge should be created/updated */
  onEdgeChange?: (sourceId: string, sourceHandle: string, targetId: string, targetHandle: string) => void;
}

// =============================================================================
// Constants
// =============================================================================

/** Node type icons mapping */
const NODE_ICONS: Record<string, ReactNode> = {
  image_input: <Image className="h-4 w-4" />,
  parameter_input: <Settings2 className="h-4 w-4" />,
  detection: <ScanSearch className="h-4 w-4" />,
  classification: <Tags className="h-4 w-4" />,
  embedding: <Binary className="h-4 w-4" />,
  similarity_search: <Search className="h-4 w-4" />,
  crop: <Crop className="h-4 w-4" />,
  filter: <Filter className="h-4 w-4" />,
  condition: <GitBranch className="h-4 w-4" />,
  grid_builder: <Grid3X3 className="h-4 w-4" />,
  json_output: <FileJson className="h-4 w-4" />,
  blur_region: <Eraser className="h-4 w-4" />,
  draw_boxes: <PenTool className="h-4 w-4" />,
};

/** Category colors for badges */
const CATEGORY_COLORS: Record<string, string> = {
  input: "bg-blue-500/10 text-blue-500 border-blue-500/20",
  model: "bg-purple-500/10 text-purple-500 border-purple-500/20",
  transform: "bg-green-500/10 text-green-500 border-green-500/20",
  logic: "bg-orange-500/10 text-orange-500 border-orange-500/20",
  output: "bg-red-500/10 text-red-500 border-red-500/20",
};

/** Category display names */
const CATEGORY_NAMES: Record<string, string> = {
  input: "Input",
  model: "Model",
  transform: "Transform",
  logic: "Logic",
  output: "Output",
};

/** Model block types */
const MODEL_BLOCK_TYPES = ["detection", "classification", "embedding"];

/** Block port definitions - what each block type accepts and produces */
const BLOCK_PORTS: Record<string, { inputs: PortDefinition[]; outputs: PortDefinition[] }> = {
  image_input: {
    inputs: [],
    outputs: [
      { name: "image", type: "image", description: "The processed input image" },
      { name: "image_url", type: "string", description: "URL of the image (if provided)" },
      { name: "width", type: "number", description: "Final image width in pixels" },
      { name: "height", type: "number", description: "Final image height in pixels" },
      { name: "original_width", type: "number", description: "Original image width" },
      { name: "original_height", type: "number", description: "Original image height" },
    ],
  },
  parameter_input: {
    inputs: [],
    outputs: [
      { name: "parameters", type: "object", description: "All input parameters" },
    ],
  },
  detection: {
    inputs: [
      { name: "image", type: "image", required: true, description: "Input image" },
    ],
    outputs: [
      { name: "detections", type: "array", description: "Detected objects with bbox, class, confidence" },
      { name: "annotated_image", type: "image", description: "Image with bounding boxes" },
      { name: "count", type: "number", description: "Number of detections" },
    ],
  },
  classification: {
    inputs: [
      { name: "image", type: "image", required: false, description: "Single image" },
      { name: "images", type: "array", required: false, description: "Array of images" },
    ],
    outputs: [
      { name: "predictions", type: "array", description: "Classification predictions with top-k" },
    ],
  },
  embedding: {
    inputs: [
      { name: "image", type: "image", required: false, description: "Single image" },
      { name: "images", type: "array", required: false, description: "Array of images" },
    ],
    outputs: [
      { name: "embeddings", type: "array", description: "Embedding vectors" },
    ],
  },
  similarity_search: {
    inputs: [
      { name: "embeddings", type: "array", required: true, description: "Query embeddings" },
    ],
    outputs: [
      { name: "matches", type: "array", description: "Matching products with scores" },
    ],
  },
  crop: {
    inputs: [
      { name: "image", type: "image", required: true, description: "Input image" },
      { name: "detections", type: "array", required: true, description: "Detection bboxes to crop" },
    ],
    outputs: [
      { name: "crops", type: "array", description: "Cropped image regions as base64" },
      { name: "crop_metadata", type: "array", description: "Metadata for each crop (index, detection, box, size)" },
    ],
  },
  blur_region: {
    inputs: [
      { name: "image", type: "image", required: true, description: "Input image" },
      { name: "regions", type: "array", required: true, description: "Regions to blur (detections)" },
    ],
    outputs: [
      { name: "image", type: "image", description: "Image with blurred regions" },
    ],
  },
  draw_boxes: {
    inputs: [
      { name: "image", type: "image", required: true, description: "Input image" },
      { name: "detections", type: "array", required: true, description: "Detections with bbox to draw" },
    ],
    outputs: [
      { name: "image", type: "image", description: "Image with boxes drawn" },
    ],
  },
  filter: {
    inputs: [
      { name: "items", type: "array", required: true, description: "Array to filter" },
    ],
    outputs: [
      { name: "passed", type: "array", description: "Items that passed the filter" },
      { name: "rejected", type: "array", description: "Items that failed the filter" },
    ],
  },
  condition: {
    inputs: [
      { name: "value", type: "any", required: true, description: "Value to evaluate" },
    ],
    outputs: [
      { name: "true_output", type: "any", description: "Output when condition is true" },
      { name: "false_output", type: "any", description: "Output when condition is false" },
    ],
  },
  grid_builder: {
    inputs: [
      { name: "detections", type: "array", required: true, description: "Detection data" },
    ],
    outputs: [
      { name: "grid", type: "array", description: "2D grid representation" },
      { name: "realogram", type: "object", description: "Full realogram data structure" },
    ],
  },
  json_output: {
    inputs: [
      { name: "data", type: "any", required: true, description: "Data to output" },
    ],
    outputs: [
      { name: "json", type: "object", description: "Formatted JSON output" },
    ],
  },
  segmentation: {
    inputs: [
      { name: "image", type: "image", required: true, description: "Input image" },
      { name: "detections", type: "array", required: false, description: "Optional detection boxes as prompts" },
    ],
    outputs: [
      { name: "masks", type: "array", description: "Segmentation masks" },
      { name: "masked_image", type: "image", description: "Image with masks applied" },
    ],
  },
};

/** Port type colors */
const PORT_TYPE_COLORS: Record<string, string> = {
  image: "text-blue-500",
  array: "text-orange-500",
  number: "text-green-500",
  string: "text-purple-500",
  object: "text-pink-500",
  any: "text-muted-foreground",
};

// =============================================================================
// Sub-components
// =============================================================================

/** Section wrapper with title */
function Section({
  title,
  children,
  className = "",
}: {
  title: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={`space-y-3 ${className}`}>
      <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
        {title}
      </h3>
      {children}
    </div>
  );
}

/** Form field wrapper */
function Field({
  label,
  description,
  children,
}: {
  label: string;
  description?: string;
  children: ReactNode;
}) {
  return (
    <div className="space-y-1.5">
      <Label className="text-sm font-medium">{label}</Label>
      {children}
      {description && (
        <p className="text-xs text-muted-foreground">{description}</p>
      )}
    </div>
  );
}

// =============================================================================
// Main Component
// =============================================================================

export function NodeConfigDrawer({
  open,
  onClose,
  node,
  onNodeChange,
  onNodeDelete,
  onNodeDuplicate,
  allNodes = [],
  edges = [],
  onEdgeChange,
}: NodeConfigDrawerProps) {
  // Get node icon
  const nodeIcon = useMemo(() => {
    if (!node) return <Box className="h-4 w-4" />;
    return NODE_ICONS[node.data.type] || <Box className="h-4 w-4" />;
  }, [node]);

  // Check if this is a model block
  const isModelBlock = useMemo(() => {
    if (!node) return false;
    return MODEL_BLOCK_TYPES.includes(node.data.type);
  }, [node]);

  // Get category badge color
  const categoryColor = useMemo(() => {
    if (!node) return "";
    return CATEGORY_COLORS[node.data.category] || "bg-muted text-muted-foreground";
  }, [node]);

  // Get port definitions for this node type
  const portDefs = useMemo(() => {
    if (!node) return { inputs: [], outputs: [] };
    return BLOCK_PORTS[node.data.type] || { inputs: [], outputs: [] };
  }, [node]);

  // Get current input connections for this node
  const inputConnections = useMemo(() => {
    if (!node) return {};
    const connections: Record<string, { sourceId: string; sourceHandle: string; sourceLabel: string }> = {};

    edges.forEach((edge) => {
      if (edge.target === node.id && edge.targetHandle) {
        const sourceNode = allNodes.find((n) => n.id === edge.source);
        connections[edge.targetHandle] = {
          sourceId: edge.source,
          sourceHandle: edge.sourceHandle || "output",
          sourceLabel: sourceNode?.label || edge.source,
        };
      }
    });

    return connections;
  }, [node, edges, allNodes]);

  // Get available source nodes (nodes that come before this one)
  const availableSourceNodes = useMemo(() => {
    if (!node) return [];
    // For now, return all nodes except self - in a real implementation,
    // we'd filter based on topological order
    return allNodes.filter((n) => n.id !== node.id);
  }, [node, allNodes]);

  // Handle config change
  const handleConfigChange = (key: string, value: unknown) => {
    if (!node) return;
    onNodeChange(node.id, {
      config: {
        ...(node.data.config || {}),
        [key]: value,
      },
    });
  };

  // Handle model change
  const handleModelChange = (modelId: string, model?: { source: "pretrained" | "trained" }) => {
    if (!node) return;
    onNodeChange(node.id, {
      model_id: modelId,
      model_source: model?.source,
    });
  };

  // Handle label change
  const handleLabelChange = (label: string) => {
    if (!node) return;
    onNodeChange(node.id, { label });
  };

  // Handle input mapping change
  const handleInputChange = (inputPort: string, sourceNodeId: string, sourceHandle: string) => {
    if (!node || !onEdgeChange) return;
    onEdgeChange(sourceNodeId, sourceHandle, node.id, inputPort);
  };

  if (!node) return null;

  return (
    <Sheet open={open} onOpenChange={(isOpen) => !isOpen && onClose()}>
      <SheetContent className="w-[360px] p-0 flex flex-col">
        {/* Header - Fixed */}
        <SheetHeader className="px-6 py-4 border-b bg-card/50 backdrop-blur-sm">
          <div className="flex items-start justify-between gap-3">
            <div className="flex items-center gap-3 min-w-0">
              <div className="flex-shrink-0 h-10 w-10 rounded-lg bg-muted flex items-center justify-center">
                {nodeIcon}
              </div>
              <div className="min-w-0">
                <SheetTitle className="text-base font-semibold truncate">
                  {node.data.label}
                </SheetTitle>
                <div className="flex items-center gap-2 mt-1">
                  <Badge
                    variant="outline"
                    className={`text-[10px] px-1.5 py-0 h-5 ${categoryColor}`}
                  >
                    {CATEGORY_NAMES[node.data.category] || node.data.category}
                  </Badge>
                  <span className="text-xs text-muted-foreground">
                    {node.data.type}
                  </span>
                </div>
              </div>
            </div>
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8 -mr-2"
              onClick={onClose}
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
        </SheetHeader>

        {/* Content - Scrollable (h-0 + flex-1 enables proper scrolling) */}
        <ScrollArea className="flex-1 h-0">
          <div className="px-6 py-5 space-y-6">
            {/* General Section */}
            <Section title="General">
              <Field label="Display Name" description="Custom name shown on the canvas">
                <Input
                  value={node.data.label}
                  onChange={(e) => handleLabelChange(e.target.value)}
                  placeholder="Enter node name..."
                  className="h-9"
                />
              </Field>
            </Section>

            {/* Input Mapping Section - Only show if node has inputs */}
            {portDefs.inputs.length > 0 && (
              <>
                <Separator />
                <Section title="Inputs">
                  <div className="space-y-3">
                    {portDefs.inputs.map((inputPort) => {
                      const connection = inputConnections[inputPort.name];
                      const isConnected = !!connection;

                      return (
                        <div
                          key={inputPort.name}
                          className="flex items-start gap-3 p-3 rounded-lg border bg-muted/30"
                        >
                          <div className="flex-shrink-0 mt-0.5">
                            {isConnected ? (
                              <CheckCircle2 className="h-4 w-4 text-green-500" />
                            ) : inputPort.required ? (
                              <AlertCircle className="h-4 w-4 text-destructive" />
                            ) : (
                              <Circle className="h-4 w-4 text-muted-foreground" />
                            )}
                          </div>
                          <div className="flex-1 min-w-0 space-y-2">
                            <div className="flex items-center gap-2">
                              <span className="text-sm font-medium">{inputPort.name}</span>
                              <Badge variant="outline" className={`text-[10px] px-1 py-0 ${PORT_TYPE_COLORS[inputPort.type] || ""}`}>
                                {inputPort.type}
                              </Badge>
                              {inputPort.required && (
                                <Badge variant="outline" className="text-[10px] px-1 py-0 text-destructive border-destructive/30">
                                  required
                                </Badge>
                              )}
                            </div>
                            {inputPort.description && (
                              <p className="text-xs text-muted-foreground">{inputPort.description}</p>
                            )}
                            {/* Source Selection */}
                            <div className="flex items-center gap-2">
                              <Select
                                value={connection ? `${connection.sourceId}::${connection.sourceHandle}` : ""}
                                onValueChange={(val) => {
                                  if (val && onEdgeChange) {
                                    const [sourceId, sourceHandle] = val.split("::");
                                    handleInputChange(inputPort.name, sourceId, sourceHandle);
                                  }
                                }}
                              >
                                <SelectTrigger className="h-8 text-xs">
                                  <SelectValue placeholder="Select source..." />
                                </SelectTrigger>
                                <SelectContent>
                                  {availableSourceNodes.map((sourceNode) => {
                                    const sourcePorts = BLOCK_PORTS[sourceNode.type]?.outputs || [];
                                    // Filter compatible ports
                                    const compatiblePorts = sourcePorts.filter(
                                      (p) => p.type === inputPort.type || p.type === "any" || inputPort.type === "any"
                                    );

                                    if (compatiblePorts.length === 0) return null;

                                    return compatiblePorts.map((outputPort) => (
                                      <SelectItem
                                        key={`${sourceNode.id}::${outputPort.name}`}
                                        value={`${sourceNode.id}::${outputPort.name}`}
                                      >
                                        <div className="flex items-center gap-2">
                                          <span>{sourceNode.label}</span>
                                          <ArrowRight className="h-3 w-3 text-muted-foreground" />
                                          <span className={PORT_TYPE_COLORS[outputPort.type]}>{outputPort.name}</span>
                                        </div>
                                      </SelectItem>
                                    ));
                                  })}
                                </SelectContent>
                              </Select>
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </Section>
              </>
            )}

            {/* Model Selection - Only for model blocks */}
            {isModelBlock && (
              <>
                <Separator />
                <Section title="Model">
                  <Field label="Select Model">
                    <ModelSelector
                      value={node.data.model_id}
                      category={node.data.type as "detection" | "classification" | "embedding"}
                      onValueChange={handleModelChange}
                      placeholder="Choose a model..."
                    />
                  </Field>
                </Section>
              </>
            )}

            {/* Configuration Section */}
            <Separator />
            <Section title="Configuration">
              <NodeConfigPanel
                nodeType={node.data.type}
                config={node.data.config || {}}
                onConfigChange={handleConfigChange}
              />
            </Section>

            {/* Outputs Section - Show what this node produces */}
            {portDefs.outputs.length > 0 && (
              <>
                <Separator />
                <Section title="Outputs">
                  <div className="space-y-2">
                    {portDefs.outputs.map((outputPort) => (
                      <div
                        key={outputPort.name}
                        className="flex items-center gap-3 p-2.5 rounded-lg border bg-muted/30"
                      >
                        <Circle className={`h-3 w-3 flex-shrink-0 ${PORT_TYPE_COLORS[outputPort.type] || "text-muted-foreground"}`} />
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2">
                            <span className="text-sm font-medium">{outputPort.name}</span>
                            <Badge variant="outline" className={`text-[10px] px-1 py-0 ${PORT_TYPE_COLORS[outputPort.type] || ""}`}>
                              {outputPort.type}
                            </Badge>
                          </div>
                          {outputPort.description && (
                            <p className="text-xs text-muted-foreground mt-0.5">{outputPort.description}</p>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                  <p className="text-xs text-muted-foreground mt-2">
                    Reference: <code className="bg-muted px-1 rounded">{"$nodes." + node.id + ".<output>"}</code>
                  </p>
                </Section>
              </>
            )}
          </div>
        </ScrollArea>

        {/* Footer - Fixed */}
        <div className="px-6 py-4 border-t bg-card/50 backdrop-blur-sm">
          <div className="flex items-center gap-2">
            {/* Duplicate Button */}
            {onNodeDuplicate && (
              <Button
                variant="outline"
                size="sm"
                className="flex-1"
                onClick={() => onNodeDuplicate(node.id)}
              >
                <Copy className="h-3.5 w-3.5 mr-1.5" />
                Duplicate
              </Button>
            )}

            {/* Delete Button with Confirmation */}
            <AlertDialog>
              <AlertDialogTrigger asChild>
                <Button
                  variant="outline"
                  size="sm"
                  className={`${onNodeDuplicate ? "flex-1" : "w-full"} text-destructive hover:text-destructive hover:bg-destructive/10 border-destructive/30`}
                >
                  <Trash2 className="h-3.5 w-3.5 mr-1.5" />
                  Delete
                </Button>
              </AlertDialogTrigger>
              <AlertDialogContent>
                <AlertDialogHeader>
                  <AlertDialogTitle>Delete Node</AlertDialogTitle>
                  <AlertDialogDescription>
                    Are you sure you want to delete "{node.data.label}"? This action cannot be undone and will also remove all connections to this node.
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel>Cancel</AlertDialogCancel>
                  <AlertDialogAction
                    className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                    onClick={() => onNodeDelete(node.id)}
                  >
                    Delete
                  </AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
          </div>
        </div>
      </SheetContent>
    </Sheet>
  );
}
