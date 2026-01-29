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
  Scaling,
  LayoutGrid,
  Combine,
  RotateCw,
  Palette,
  Repeat,
  ListPlus,
  Shuffle,
} from "lucide-react";
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
  /** Callback when edge should be removed */
  onEdgeRemove?: (targetId: string, targetHandle: string) => void;
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
  resize: <Scaling className="h-4 w-4" />,
  tile: <LayoutGrid className="h-4 w-4" />,
  stitch: <Combine className="h-4 w-4" />,
  rotate_flip: <RotateCw className="h-4 w-4" />,
  normalize: <Palette className="h-4 w-4" />,
  filter: <Filter className="h-4 w-4" />,
  condition: <GitBranch className="h-4 w-4" />,
  foreach: <Repeat className="h-4 w-4" />,
  collect: <ListPlus className="h-4 w-4" />,
  map: <Shuffle className="h-4 w-4" />,
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
      { name: "image", type: "image", required: true, description: "Input image to detect objects" },
      { name: "text_prompt", type: "string", required: false, description: "Text prompt for open-vocab detection (Grounding DINO, SAM3)" },
      { name: "class_filter", type: "array", required: false, description: "Filter to specific class names" },
      { name: "roi", type: "object", required: false, description: "Region of interest to detect within" },
    ],
    outputs: [
      { name: "detections", type: "array", description: "All detected objects with bbox, class, confidence" },
      { name: "boxes", type: "array", description: "Bounding boxes only [x1,y1,x2,y2]" },
      { name: "labels", type: "array", description: "Class labels for each detection" },
      { name: "scores", type: "array", description: "Confidence scores for each detection" },
      { name: "annotated_image", type: "image", description: "Image with bounding boxes drawn" },
      { name: "crops", type: "array", description: "Cropped images for each detection" },
      { name: "count", type: "number", description: "Total number of detections" },
      { name: "class_counts", type: "object", description: "Count per class {class: count}" },
    ],
  },
  classification: {
    inputs: [
      { name: "image", type: "image", required: false, description: "Single image to classify" },
      { name: "images", type: "array", required: false, description: "Array of images to classify" },
      { name: "crops", type: "array", required: false, description: "Detection crops to classify" },
    ],
    outputs: [
      { name: "predictions", type: "array", description: "Full predictions with top-k classes" },
      { name: "label", type: "string", description: "Top predicted class label" },
      { name: "labels", type: "array", description: "Top labels for each image (batch)" },
      { name: "confidence", type: "number", description: "Confidence of top prediction" },
      { name: "probabilities", type: "object", description: "All class probabilities" },
      { name: "is_uncertain", type: "boolean", description: "Whether prediction is uncertain" },
      { name: "decision", type: "string", description: "Binary decision output (if configured)" },
    ],
  },
  embedding: {
    inputs: [
      { name: "image", type: "image", required: false, description: "Single image to embed" },
      { name: "images", type: "array", required: false, description: "Array of images to embed" },
      { name: "crops", type: "array", required: false, description: "Bounding boxes to crop and embed" },
      { name: "mask", type: "image", required: false, description: "Mask to focus embedding region" },
    ],
    outputs: [
      { name: "embedding", type: "array", description: "Single embedding vector (for single image)" },
      { name: "embeddings", type: "array", description: "Array of embedding vectors" },
      { name: "dimension", type: "number", description: "Embedding dimension size" },
      { name: "norm", type: "number", description: "L2 norm of embedding (quality indicator)" },
      { name: "attention_map", type: "image", description: "Attention visualization (if enabled)" },
    ],
  },
  similarity_search: {
    inputs: [
      { name: "embedding", type: "array", required: false, description: "Single query embedding" },
      { name: "embeddings", type: "array", required: false, description: "Multiple query embeddings" },
      { name: "filter", type: "object", required: false, description: "Metadata filter criteria" },
      { name: "text_query", type: "string", required: false, description: "Text query for hybrid search" },
    ],
    outputs: [
      { name: "matches", type: "array", description: "All matching items with scores and payloads" },
      { name: "top_match", type: "object", description: "Best matching item with full data" },
      { name: "match_ids", type: "array", description: "Just the IDs of matched items" },
      { name: "match_scores", type: "array", description: "Normalized similarity scores (0-1)" },
      { name: "match_payloads", type: "array", description: "Payloads/metadata of matched items" },
      { name: "distances", type: "array", description: "Raw distance values (metric-dependent)" },
      { name: "match_count", type: "number", description: "Total number of matches found" },
      { name: "unique_count", type: "number", description: "Unique items after deduplication" },
      { name: "has_matches", type: "boolean", description: "Whether any matches were found" },
      { name: "avg_score", type: "number", description: "Average similarity score of results" },
      { name: "groups", type: "object", description: "Grouped results (if group_by enabled)" },
      { name: "reranked", type: "array", description: "Re-ranked results (if reranking enabled)" },
      { name: "search_time_ms", type: "number", description: "Search execution time in milliseconds" },
    ],
  },
  crop: {
    inputs: [
      { name: "image", type: "image", required: true, description: "Input image to crop from" },
      { name: "detections", type: "array", required: false, description: "Detection results with bboxes" },
      { name: "boxes", type: "array", required: false, description: "Raw bounding boxes [x1,y1,x2,y2]" },
      { name: "masks", type: "array", required: false, description: "Segmentation masks for tight cropping" },
    ],
    outputs: [
      { name: "crops", type: "array", description: "Cropped image regions" },
      { name: "crop_images", type: "array", description: "Cropped images as separate outputs" },
      { name: "crop_boxes", type: "array", description: "Final crop coordinates after padding" },
      { name: "crop_sizes", type: "array", description: "Width/height of each crop" },
      { name: "crop_metadata", type: "array", description: "Full metadata (source box, class, confidence)" },
      { name: "valid_crops", type: "array", description: "Crops that passed size/confidence filters" },
      { name: "rejected_crops", type: "array", description: "Crops that failed filters" },
      { name: "crop_count", type: "number", description: "Number of valid crops produced" },
    ],
  },
  resize: {
    inputs: [
      { name: "image", type: "image", required: true, description: "Input image to resize" },
      { name: "images", type: "array", required: false, description: "Array of images to resize (batch)" },
      { name: "target_width", type: "number", required: false, description: "Dynamic target width override" },
      { name: "target_height", type: "number", required: false, description: "Dynamic target height override" },
    ],
    outputs: [
      { name: "image", type: "image", description: "Resized image" },
      { name: "images", type: "array", description: "Resized images (batch mode)" },
      { name: "width", type: "number", description: "Final width in pixels" },
      { name: "height", type: "number", description: "Final height in pixels" },
      { name: "original_width", type: "number", description: "Original width before resize" },
      { name: "original_height", type: "number", description: "Original height before resize" },
      { name: "scale_x", type: "number", description: "Horizontal scale factor applied" },
      { name: "scale_y", type: "number", description: "Vertical scale factor applied" },
      { name: "was_resized", type: "boolean", description: "Whether resize was actually applied" },
      { name: "padding", type: "object", description: "Padding added {top, bottom, left, right}" },
    ],
  },
  tile: {
    inputs: [
      { name: "image", type: "image", required: true, description: "Input image to tile" },
      { name: "tile_size", type: "number", required: false, description: "Dynamic tile size override" },
      { name: "overlap", type: "number", required: false, description: "Dynamic overlap override" },
    ],
    outputs: [
      { name: "tiles", type: "array", description: "Array of tile images" },
      { name: "tile_coords", type: "array", description: "Coordinates of each tile [{x, y, width, height}]" },
      { name: "tile_indices", type: "array", description: "Grid indices [{row, col}] for each tile" },
      { name: "tile_count", type: "number", description: "Total number of tiles generated" },
      { name: "grid_size", type: "object", description: "Grid dimensions {rows, cols}" },
      { name: "original_size", type: "object", description: "Original image {width, height}" },
      { name: "tile_metadata", type: "array", description: "Full metadata for reconstruction" },
      { name: "overlap_info", type: "object", description: "Overlap pixels {x, y}" },
    ],
  },
  stitch: {
    inputs: [
      { name: "tiles", type: "array", required: true, description: "Array of tile images or results" },
      { name: "tile_coords", type: "array", required: true, description: "Coordinates from tile node" },
      { name: "detections", type: "array", required: false, description: "Detections to merge from tiles" },
      { name: "original_size", type: "object", required: false, description: "Original image size for reconstruction" },
      { name: "tile_metadata", type: "array", required: false, description: "Full tile metadata" },
    ],
    outputs: [
      { name: "image", type: "image", description: "Stitched/reconstructed image" },
      { name: "merged_detections", type: "array", description: "Merged detections with NMS" },
      { name: "all_detections", type: "array", description: "All detections before NMS" },
      { name: "detection_count", type: "number", description: "Final detection count after merge" },
      { name: "duplicate_count", type: "number", description: "Detections removed by NMS" },
      { name: "tile_detection_counts", type: "array", description: "Detection count per tile" },
    ],
  },
  rotate_flip: {
    inputs: [
      { name: "image", type: "image", required: true, description: "Input image to transform" },
      { name: "images", type: "array", required: false, description: "Array of images (batch)" },
      { name: "boxes", type: "array", required: false, description: "Bounding boxes to transform with image" },
      { name: "angle", type: "number", required: false, description: "Dynamic rotation angle override" },
    ],
    outputs: [
      { name: "image", type: "image", description: "Transformed image" },
      { name: "images", type: "array", description: "Transformed images (batch)" },
      { name: "boxes", type: "array", description: "Transformed bounding boxes" },
      { name: "transform_matrix", type: "array", description: "Affine transform matrix [2x3]" },
      { name: "was_transformed", type: "boolean", description: "Whether any transform was applied" },
      { name: "new_size", type: "object", description: "New image dimensions after transform" },
    ],
  },
  normalize: {
    inputs: [
      { name: "image", type: "image", required: true, description: "Input image to normalize" },
      { name: "images", type: "array", required: false, description: "Array of images (batch)" },
    ],
    outputs: [
      { name: "image", type: "image", description: "Normalized image" },
      { name: "images", type: "array", description: "Normalized images (batch)" },
      { name: "tensor", type: "array", description: "Normalized tensor [C,H,W]" },
      { name: "tensors", type: "array", description: "Batch of normalized tensors" },
      { name: "mean_used", type: "array", description: "Mean values used [R,G,B]" },
      { name: "std_used", type: "array", description: "Std values used [R,G,B]" },
      { name: "original_dtype", type: "string", description: "Original image dtype" },
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
      { name: "items", type: "array", required: true, description: "Array of items to filter (detections, crops, etc.)" },
      { name: "detections", type: "array", required: false, description: "Detection results to filter" },
      { name: "threshold", type: "number", required: false, description: "Dynamic threshold override" },
      { name: "class_list", type: "array", required: false, description: "Dynamic class whitelist" },
      { name: "exclude_list", type: "array", required: false, description: "Dynamic class blacklist" },
    ],
    outputs: [
      { name: "passed", type: "array", description: "Items that passed all filter conditions" },
      { name: "rejected", type: "array", description: "Items that failed filter conditions" },
      { name: "passed_count", type: "number", description: "Number of items that passed" },
      { name: "rejected_count", type: "number", description: "Number of items rejected" },
      { name: "total_count", type: "number", description: "Total items processed" },
      { name: "pass_rate", type: "number", description: "Percentage of items that passed (0-1)" },
      { name: "first_passed", type: "object", description: "First item that passed the filter" },
      { name: "top_n", type: "array", description: "Top N items by sort criteria" },
      { name: "sorted", type: "array", description: "Passed items sorted by criteria" },
      { name: "grouped", type: "object", description: "Items grouped by field value" },
      { name: "filter_stats", type: "object", description: "Statistics {min, max, avg, sum by field}" },
      { name: "unique_values", type: "array", description: "Unique values of grouped field" },
    ],
  },
  condition: {
    inputs: [
      { name: "value", type: "any", required: true, description: "Primary value to evaluate" },
      { name: "compare_to", type: "any", required: false, description: "Value to compare against (can use field reference)" },
      { name: "context", type: "object", required: false, description: "Additional context for field access" },
    ],
    outputs: [
      { name: "true_output", type: "any", description: "Output when condition passes" },
      { name: "false_output", type: "any", description: "Output when condition fails" },
      { name: "result", type: "boolean", description: "Boolean result of evaluation" },
      { name: "matched_conditions", type: "array", description: "Which conditions passed (by index)" },
      { name: "evaluation_details", type: "object", description: "Full evaluation metadata (if enabled)" },
    ],
  },
  foreach: {
    inputs: [
      { name: "items", type: "array", required: true, description: "Array of items to iterate over" },
      { name: "context", type: "any", required: false, description: "Additional context passed to each iteration" },
    ],
    outputs: [
      { name: "item", type: "any", description: "Current item in iteration" },
      { name: "index", type: "number", description: "Current index (0-based)" },
      { name: "total", type: "number", description: "Total number of items" },
      { name: "context", type: "any", description: "Passed through context" },
      { name: "is_first", type: "boolean", description: "True if first item" },
      { name: "is_last", type: "boolean", description: "True if last item" },
    ],
  },
  collect: {
    inputs: [
      { name: "item", type: "any", required: true, description: "Item from each iteration" },
      { name: "index", type: "number", required: false, description: "Index from ForEach" },
    ],
    outputs: [
      { name: "results", type: "array", description: "Collected results array" },
      { name: "count", type: "number", description: "Number of collected items" },
    ],
  },
  map: {
    inputs: [
      { name: "items", type: "array", required: true, description: "Array of items to transform" },
    ],
    outputs: [
      { name: "results", type: "array", description: "Transformed items" },
      { name: "count", type: "number", description: "Number of items" },
    ],
  },
  grid_builder: {
    inputs: [
      { name: "detections", type: "array", required: true, description: "Detection data with bboxes" },
      { name: "image", type: "image", required: false, description: "Original image (for shelf edge detection)" },
      { name: "expected_layout", type: "array", required: false, description: "Reference planogram for comparison" },
      { name: "shelf_lines", type: "array", required: false, description: "Pre-detected shelf boundaries" },
    ],
    outputs: [
      { name: "grid", type: "array", description: "2D grid [[row1], [row2], ...]" },
      { name: "realogram", type: "object", description: "Full realogram with metadata" },
      { name: "shelves", type: "array", description: "Shelf-grouped items [{shelf: 1, items: [...]}]" },
      { name: "row_count", type: "number", description: "Number of detected rows/shelves" },
      { name: "col_count", type: "number", description: "Maximum columns per row" },
      { name: "total_cells", type: "number", description: "Total occupied cells" },
      { name: "empty_cells", type: "number", description: "Detected empty positions" },
      { name: "facings", type: "object", description: "Facing counts per product (if enabled)" },
      { name: "comparison", type: "object", description: "Planogram comparison results (if enabled)" },
      { name: "shelf_edges", type: "array", description: "Detected shelf edge positions (if enabled)" },
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
      { name: "image", type: "image", required: true, description: "Input image to segment" },
      { name: "boxes", type: "array", required: false, description: "Bounding boxes as prompts" },
      { name: "points", type: "array", required: false, description: "Point coordinates as prompts" },
      { name: "point_labels", type: "array", required: false, description: "Point labels (1=foreground, 0=background)" },
      { name: "text_prompt", type: "string", required: false, description: "Text prompt for SAM3" },
    ],
    outputs: [
      { name: "masks", type: "array", description: "Binary segmentation masks" },
      { name: "polygons", type: "array", description: "Mask contours as polygons" },
      { name: "rle_masks", type: "array", description: "Run-length encoded masks (compact)" },
      { name: "masked_image", type: "image", description: "Image with masks overlaid" },
      { name: "cropped_objects", type: "array", description: "Cropped objects using masks" },
      { name: "mask_scores", type: "array", description: "Confidence scores for each mask" },
      { name: "areas", type: "array", description: "Pixel area of each mask" },
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
  onEdgeRemove,
}: NodeConfigDrawerProps) {
  // Get node icon
  const nodeIcon = useMemo(() => {
    if (!node) return <Box className="h-4 w-4" />;
    return NODE_ICONS[node.data.type] || <Box className="h-4 w-4" />;
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

  // Handle config change - supports both single key and object with multiple keys
  const handleConfigChange = (keyOrObject: string | Record<string, unknown>, value?: unknown) => {
    if (!node) return;

    // Support batch updates: onConfigChange({ model_id: "x", model_source: "y" })
    if (typeof keyOrObject === "object") {
      onNodeChange(node.id, {
        config: {
          ...(node.data.config || {}),
          ...keyOrObject,
        },
      });
    } else {
      // Single key update: onConfigChange("key", value)
      onNodeChange(node.id, {
        config: {
          ...(node.data.config || {}),
          [keyOrObject]: value,
        },
      });
    }
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
                              <div className="relative flex-1">
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
                                {isConnected && onEdgeRemove && (
                                  <Button
                                    variant="ghost"
                                    size="icon"
                                    className="absolute right-8 top-1/2 -translate-y-1/2 h-6 w-6 hover:bg-destructive/10 z-10"
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      onEdgeRemove(node.id, inputPort.name);
                                    }}
                                  >
                                    <span className="text-xs text-muted-foreground hover:text-destructive">✕</span>
                                  </Button>
                                )}
                              </div>
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </Section>
              </>
            )}

            {/* Configuration Section - includes model selection for model blocks */}
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
