"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { toast } from "sonner";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetDescription,
} from "@/components/ui/sheet";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Separator } from "@/components/ui/separator";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  Play,
  Loader2,
  Upload,
  Link2,
  Database,
  ChevronDown,
  ChevronRight,
  CheckCircle,
  XCircle,
  Clock,
  Copy,
  Maximize2,
  X,
  Search,
  Package,
  Trash2,
  FolderOpen,
} from "lucide-react";
import { apiClient } from "@/lib/api-client";

// =============================================================================
// Types
// =============================================================================

interface WorkflowParameter {
  name: string;
  type: string;
  default?: unknown;
  description?: string;
}

interface Detection {
  id: number;
  class_name: string;
  class_id: number;
  confidence: number;
  bbox: {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
  };
  area?: number;
}

interface ExecutionResult {
  id: string;
  status: string;
  output_data?: {
    detections?: Detection[];
    annotated_image?: string;
    count?: number;
    predictions?: Array<{
      class_name: string;
      confidence: number;
    }>;
    embedding?: number[];
    [key: string]: unknown;
  };
  node_metrics?: Record<string, {
    type: string;
    duration_ms: number;
    success: boolean;
    [key: string]: unknown;
  }>;
  duration_ms?: number;
  error_message?: string;
  error_node_id?: string;
}

interface WorkflowTestPanelProps {
  open: boolean;
  onClose: () => void;
  workflowId: string;
  workflowName: string;
  parameters?: WorkflowParameter[];
}

// =============================================================================
// Detection Overlay Component
// =============================================================================

function DetectionOverlay({
  detections,
  imageWidth,
  imageHeight,
}: {
  detections: Detection[];
  imageWidth: number;
  imageHeight: number;
}) {
  const getColor = (classId: number) => {
    const colors = [
      "#3B82F6", "#10B981", "#F59E0B", "#EF4444",
      "#8B5CF6", "#EC4899", "#06B6D4", "#84CC16",
    ];
    return colors[classId % colors.length];
  };

  return (
    <svg
      className="absolute inset-0 w-full h-full pointer-events-none"
      viewBox={`0 0 ${imageWidth} ${imageHeight}`}
      preserveAspectRatio="none"
    >
      {detections.map((det) => {
        const color = getColor(det.class_id);
        const x = det.bbox.x1 * imageWidth;
        const y = det.bbox.y1 * imageHeight;
        const w = (det.bbox.x2 - det.bbox.x1) * imageWidth;
        const h = (det.bbox.y2 - det.bbox.y1) * imageHeight;

        return (
          <g key={det.id}>
            <rect x={x} y={y} width={w} height={h} fill="none" stroke={color} strokeWidth={2} />
            <rect x={x} y={y - 20} width={Math.max(80, det.class_name.length * 8 + 40)} height={20} fill={color} rx={2} />
            <text x={x + 4} y={y - 6} fill="white" fontSize={12} fontFamily="monospace">
              {det.class_name} {(det.confidence * 100).toFixed(0)}%
            </text>
          </g>
        );
      })}
    </svg>
  );
}

// =============================================================================
// Image Preview with Detections
// =============================================================================

function ImagePreviewWithDetections({
  src,
  detections,
  onFullscreen,
}: {
  src: string;
  detections?: Detection[];
  onFullscreen?: () => void;
}) {
  const [imageDimensions, setImageDimensions] = useState({ width: 0, height: 0 });
  const imgRef = useRef<HTMLImageElement>(null);

  return (
    <div className="relative group">
      <img
        ref={imgRef}
        src={src}
        alt="Input"
        className="w-full h-auto rounded-lg"
        onLoad={(e) => {
          const img = e.target as HTMLImageElement;
          setImageDimensions({ width: img.naturalWidth, height: img.naturalHeight });
        }}
      />
      {detections && detections.length > 0 && imageDimensions.width > 0 && (
        <DetectionOverlay
          detections={detections}
          imageWidth={imageDimensions.width}
          imageHeight={imageDimensions.height}
        />
      )}
      {onFullscreen && (
        <Button
          variant="secondary"
          size="icon"
          className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity"
          onClick={onFullscreen}
        >
          <Maximize2 className="h-4 w-4" />
        </Button>
      )}
    </div>
  );
}

// =============================================================================
// Node Results Display
// =============================================================================

function NodeResultsSection({ metrics }: { metrics?: ExecutionResult["node_metrics"] }) {
  const [expandedNodes, setExpandedNodes] = useState<Set<string>>(new Set());

  if (!metrics) return null;

  const toggleNode = (nodeId: string) => {
    setExpandedNodes((prev) => {
      const next = new Set(prev);
      if (next.has(nodeId)) next.delete(nodeId);
      else next.add(nodeId);
      return next;
    });
  };

  return (
    <div className="space-y-2">
      <Label className="text-xs text-muted-foreground uppercase tracking-wide">Node Results</Label>
      <div className="space-y-1">
        {Object.entries(metrics).map(([nodeId, nodeMetrics]) => {
          const isExpanded = expandedNodes.has(nodeId);
          const StatusIcon = nodeMetrics.success ? CheckCircle : XCircle;
          const statusColor = nodeMetrics.success ? "text-green-500" : "text-red-500";

          return (
            <Collapsible key={nodeId} open={isExpanded} onOpenChange={() => toggleNode(nodeId)}>
              <CollapsibleTrigger className="flex items-center gap-2 w-full p-2 rounded-md hover:bg-muted/50 transition-colors">
                {isExpanded ? <ChevronDown className="h-3 w-3 text-muted-foreground" /> : <ChevronRight className="h-3 w-3 text-muted-foreground" />}
                <StatusIcon className={`h-3 w-3 ${statusColor}`} />
                <span className="text-sm font-medium flex-1 text-left truncate">{nodeId}</span>
                <Badge variant="outline" className="text-[10px]">{nodeMetrics.type}</Badge>
                <span className="text-xs text-muted-foreground">{nodeMetrics.duration_ms.toFixed(0)}ms</span>
              </CollapsibleTrigger>
              <CollapsibleContent>
                <div className="pl-8 pr-2 pb-2">
                  <pre className="text-xs bg-muted/50 p-2 rounded overflow-x-auto max-h-32">
                    {JSON.stringify(nodeMetrics, null, 2)}
                  </pre>
                </div>
              </CollapsibleContent>
            </Collapsible>
          );
        })}
      </div>
    </div>
  );
}

// =============================================================================
// Main Component
// =============================================================================

export function WorkflowTestPanel({
  open,
  onClose,
  workflowId,
  workflowName,
  parameters = [],
}: WorkflowTestPanelProps) {
  // State
  const [activeTab, setActiveTab] = useState<"upload" | "url" | "products" | "datasets">("upload");
  const [imageUrl, setImageUrl] = useState("");
  const [imageBase64, setImageBase64] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [paramValues, setParamValues] = useState<Record<string, unknown>>({});
  const [executionResult, setExecutionResult] = useState<ExecutionResult | null>(null);
  const [selectedItem, setSelectedItem] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedDatasetId, setSelectedDatasetId] = useState<string | null>(null);
  const [fullscreenImage, setFullscreenImage] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Initialize parameter defaults
  useEffect(() => {
    const defaults: Record<string, unknown> = {};
    parameters.forEach((p) => {
      if (p.default !== undefined) defaults[p.name] = p.default;
    });
    setParamValues(defaults);
  }, [parameters]);

  // Fetch products
  const { data: productsData, isLoading: productsLoading } = useQuery({
    queryKey: ["products-for-workflow", searchQuery],
    queryFn: () => apiClient.getProducts({ search: searchQuery || undefined, limit: 20 }),
    enabled: activeTab === "products",
  });

  // Fetch OD datasets
  const { data: datasetsData, isLoading: datasetsLoading } = useQuery({
    queryKey: ["od-datasets-for-workflow"],
    queryFn: () => apiClient.getODDatasets(),
    enabled: activeTab === "datasets",
  });

  // Fetch dataset images when a dataset is selected
  const { data: datasetImagesData, isLoading: datasetImagesLoading } = useQuery({
    queryKey: ["dataset-images-for-workflow", selectedDatasetId],
    queryFn: () => apiClient.getODDatasetImages(selectedDatasetId!, { limit: 50 }),
    enabled: activeTab === "datasets" && !!selectedDatasetId,
  });

  // Execute workflow mutation
  const executeMutation = useMutation({
    mutationFn: async () => {
      const inputs: Record<string, unknown> = { parameters: paramValues };

      // Use previewUrl for URL-based images, imageBase64 for uploaded images
      if (imageBase64) {
        inputs.image_base64 = imageBase64;
      } else if (previewUrl) {
        inputs.image_url = previewUrl;
      }

      console.log("=== Executing workflow ===");
      console.log("Workflow ID:", workflowId);
      console.log("Inputs:", { ...inputs, image_base64: inputs.image_base64 ? "[base64 data]" : undefined });

      const result = await apiClient.executeWorkflow(workflowId, inputs);
      console.log("Execution result:", result);
      return result;
    },
    onSuccess: async (data) => {
      const details = await apiClient.getWorkflowExecution(data.id);
      setExecutionResult(details);
      if (details.status === "completed") {
        toast.success("Workflow completed successfully");
      } else if (details.status === "failed") {
        toast.error(`Workflow failed: ${details.error_message}`);
      }
    },
    onError: (error: Error) => {
      toast.error(`Execution failed: ${error.message}`);
    },
  });

  // Handle file upload
  const handleFileUpload = useCallback((file: File) => {
    if (!file.type.startsWith("image/")) {
      toast.error("Please upload an image file");
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      toast.error("Image must be smaller than 10MB");
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      const result = e.target?.result as string;
      const base64 = result.split(",")[1];
      setImageBase64(base64);
      setPreviewUrl(result);
      setImageUrl("");
      setSelectedItem(null);
    };
    reader.readAsDataURL(file);
  }, []);

  // Handle drag & drop
  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) handleFileUpload(file);
  }, [handleFileUpload]);

  // Handle URL submit
  const handleUrlSubmit = useCallback(() => {
    if (!imageUrl.trim()) return;
    setPreviewUrl(imageUrl.trim());
    setImageBase64(null);
    setSelectedItem(null);
  }, [imageUrl]);

  // Handle image selection from products or datasets
  const handleImageSelect = useCallback((id: string, url: string) => {
    setSelectedItem(id);
    setPreviewUrl(url);
    setImageBase64(null);
    setImageUrl("");
  }, []);

  // Reset state
  const handleReset = useCallback(() => {
    setImageUrl("");
    setImageBase64(null);
    setPreviewUrl(null);
    setExecutionResult(null);
    setSelectedItem(null);
    setSelectedDatasetId(null);
  }, []);

  // Copy JSON to clipboard
  const copyToClipboard = (data: unknown) => {
    navigator.clipboard.writeText(JSON.stringify(data, null, 2));
    toast.success("Copied to clipboard");
  };

  const hasImage = !!previewUrl || !!imageBase64;
  const canExecute = hasImage && !executeMutation.isPending;

  return (
    <>
      <Sheet open={open} onOpenChange={(isOpen) => !isOpen && onClose()}>
        <SheetContent side="right" className="w-[600px] sm:max-w-[600px] p-0 flex flex-col h-full">
            {/* Header - Fixed */}
            <SheetHeader className="p-4 border-b shrink-0">
              <div className="flex items-center justify-between">
                <div>
                  <SheetTitle className="flex items-center gap-2">
                    <Play className="h-4 w-4" />
                    Test Workflow
                  </SheetTitle>
                  <SheetDescription>{workflowName}</SheetDescription>
                </div>
                <Button variant="ghost" size="icon" onClick={onClose}>
                  <X className="h-4 w-4" />
                </Button>
              </div>
            </SheetHeader>

            {/* Scrollable Content */}
            <div className="flex-1 overflow-y-auto min-h-0">
              <div className="p-4 space-y-6">
                {/* Input Section */}
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <Label className="text-sm font-medium">Input Image</Label>
                    {hasImage && (
                      <Button variant="ghost" size="sm" onClick={handleReset} className="gap-1.5 text-destructive hover:text-destructive">
                        <Trash2 className="h-3 w-3" />
                        Clear
                      </Button>
                    )}
                  </div>

                  <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as typeof activeTab)}>
                    <TabsList className="grid w-full grid-cols-4">
                      <TabsTrigger value="upload" className="gap-1">
                        <Upload className="h-3 w-3" />
                        <span className="hidden sm:inline">Upload</span>
                      </TabsTrigger>
                      <TabsTrigger value="url" className="gap-1">
                        <Link2 className="h-3 w-3" />
                        <span className="hidden sm:inline">URL</span>
                      </TabsTrigger>
                      <TabsTrigger value="products" className="gap-1">
                        <Package className="h-3 w-3" />
                        <span className="hidden sm:inline">Products</span>
                      </TabsTrigger>
                      <TabsTrigger value="datasets" className="gap-1">
                        <FolderOpen className="h-3 w-3" />
                        <span className="hidden sm:inline">Datasets</span>
                      </TabsTrigger>
                    </TabsList>

                    {/* Upload Tab */}
                    <TabsContent value="upload" className="mt-4">
                      <div
                        className="border-2 border-dashed rounded-lg p-8 text-center hover:border-primary/50 transition-colors cursor-pointer"
                        onDrop={handleDrop}
                        onDragOver={(e) => e.preventDefault()}
                        onClick={() => fileInputRef.current?.click()}
                      >
                        <input
                          ref={fileInputRef}
                          type="file"
                          accept="image/*"
                          className="hidden"
                          onChange={(e) => {
                            const file = e.target.files?.[0];
                            if (file) handleFileUpload(file);
                          }}
                        />
                        <Upload className="h-8 w-8 mx-auto mb-2 text-muted-foreground" />
                        <p className="text-sm text-muted-foreground">Drag & drop or click to upload</p>
                        <p className="text-xs text-muted-foreground mt-1">Max 10MB, JPEG/PNG/WebP</p>
                      </div>
                    </TabsContent>

                    {/* URL Tab */}
                    <TabsContent value="url" className="mt-4 space-y-2">
                      <div className="flex gap-2">
                        <Input
                          placeholder="https://example.com/image.jpg"
                          value={imageUrl}
                          onChange={(e) => setImageUrl(e.target.value)}
                          onKeyDown={(e) => e.key === "Enter" && handleUrlSubmit()}
                        />
                        <Button onClick={handleUrlSubmit} disabled={!imageUrl.trim()}>
                          Load
                        </Button>
                      </div>
                    </TabsContent>

                    {/* Products Tab */}
                    <TabsContent value="products" className="mt-4 space-y-3">
                      <div className="relative">
                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                        <Input
                          placeholder="Search products..."
                          value={searchQuery}
                          onChange={(e) => setSearchQuery(e.target.value)}
                          className="pl-9"
                        />
                      </div>
                      <ScrollArea className="h-[200px] border rounded-lg">
                        {productsLoading ? (
                          <div className="flex items-center justify-center py-8">
                            <Loader2 className="h-6 w-6 animate-spin" />
                          </div>
                        ) : (
                          <div className="p-2 space-y-1">
                            {productsData?.items?.map((product) => (
                              <button
                                key={product.id}
                                className={`w-full flex items-center gap-3 p-2 rounded-md hover:bg-muted/50 transition-colors text-left ${
                                  selectedItem === product.id ? "bg-primary/10 ring-1 ring-primary/30" : ""
                                }`}
                                onClick={() => product.primary_image_url && handleImageSelect(product.id, product.primary_image_url)}
                                disabled={!product.primary_image_url}
                              >
                                {product.primary_image_url ? (
                                  <img src={product.primary_image_url} alt="" className="h-10 w-10 rounded object-cover" />
                                ) : (
                                  <div className="h-10 w-10 rounded bg-muted flex items-center justify-center">
                                    <Package className="h-4 w-4 text-muted-foreground" />
                                  </div>
                                )}
                                <div className="flex-1 min-w-0">
                                  <p className="text-sm font-medium truncate">{product.product_name || "Unnamed"}</p>
                                  <p className="text-xs text-muted-foreground">{product.barcode}</p>
                                </div>
                              </button>
                            ))}
                            {productsData?.items?.length === 0 && (
                              <p className="text-center text-sm text-muted-foreground py-4">No products found</p>
                            )}
                          </div>
                        )}
                      </ScrollArea>
                    </TabsContent>

                    {/* Datasets Tab */}
                    <TabsContent value="datasets" className="mt-4 space-y-3">
                      {/* Dataset Selector */}
                      <Select
                        value={selectedDatasetId || ""}
                        onValueChange={(v) => {
                          setSelectedDatasetId(v);
                          setSelectedItem(null);
                        }}
                      >
                        <SelectTrigger>
                          <SelectValue placeholder="Select a dataset..." />
                        </SelectTrigger>
                        <SelectContent>
                          {datasetsLoading ? (
                            <div className="p-2 text-center">
                              <Loader2 className="h-4 w-4 animate-spin mx-auto" />
                            </div>
                          ) : (
                            datasetsData?.map((dataset) => (
                              <SelectItem key={dataset.id} value={dataset.id}>
                                {dataset.name} ({dataset.image_count || 0} images)
                              </SelectItem>
                            ))
                          )}
                        </SelectContent>
                      </Select>

                      {/* Dataset Images */}
                      {selectedDatasetId && (
                        <ScrollArea className="h-[200px] border rounded-lg">
                          {datasetImagesLoading ? (
                            <div className="flex items-center justify-center py-8">
                              <Loader2 className="h-6 w-6 animate-spin" />
                            </div>
                          ) : (
                            <div className="p-2 grid grid-cols-4 gap-2">
                              {datasetImagesData?.images?.map((item) => (
                                <button
                                  key={item.id}
                                  className={`relative aspect-square rounded-md overflow-hidden hover:ring-2 hover:ring-primary/50 transition-all ${
                                    selectedItem === item.id ? "ring-2 ring-primary" : ""
                                  }`}
                                  onClick={() => item.image?.image_url && handleImageSelect(item.id, item.image.image_url)}
                                >
                                  <img
                                    src={item.image?.image_url}
                                    alt=""
                                    className="w-full h-full object-cover"
                                  />
                                  {selectedItem === item.id && (
                                    <div className="absolute inset-0 bg-primary/20 flex items-center justify-center">
                                      <CheckCircle className="h-6 w-6 text-primary" />
                                    </div>
                                  )}
                                </button>
                              ))}
                              {datasetImagesData?.images?.length === 0 && (
                                <p className="col-span-4 text-center text-sm text-muted-foreground py-4">
                                  No images in this dataset
                                </p>
                              )}
                            </div>
                          )}
                        </ScrollArea>
                      )}

                      {!selectedDatasetId && (
                        <div className="text-center py-8 text-muted-foreground">
                          <Database className="h-8 w-8 mx-auto mb-2 opacity-50" />
                          <p className="text-sm">Select a dataset to browse images</p>
                        </div>
                      )}
                    </TabsContent>
                  </Tabs>

                  {/* Image Preview - show below tabs after selection */}
                  {previewUrl && (
                    <div className="relative mt-4">
                      <ImagePreviewWithDetections
                        src={previewUrl}
                        detections={executionResult?.output_data?.detections}
                        onFullscreen={() => setFullscreenImage(true)}
                      />
                      <Badge className="absolute bottom-2 left-2 bg-black/60">
                        {selectedItem ? "Selected" : imageBase64 ? "Uploaded" : "URL"}
                      </Badge>
                    </div>
                  )}
                </div>

                <Separator />

                {/* Parameters Section */}
                {parameters.length > 0 && (
                  <>
                    <div className="space-y-4">
                      <Label className="text-sm font-medium">Parameters</Label>
                      <div className="space-y-3">
                        {parameters.map((param) => (
                          <div key={param.name} className="space-y-1.5">
                            <Label className="text-xs">{param.name}</Label>
                            {param.type === "number" ? (
                              <Input
                                type="number"
                                value={(paramValues[param.name] as number) ?? param.default ?? ""}
                                onChange={(e) => setParamValues((prev) => ({ ...prev, [param.name]: parseFloat(e.target.value) }))}
                              />
                            ) : param.type === "boolean" ? (
                              <Select
                                value={String(paramValues[param.name] ?? param.default ?? false)}
                                onValueChange={(v) => setParamValues((prev) => ({ ...prev, [param.name]: v === "true" }))}
                              >
                                <SelectTrigger><SelectValue /></SelectTrigger>
                                <SelectContent>
                                  <SelectItem value="true">True</SelectItem>
                                  <SelectItem value="false">False</SelectItem>
                                </SelectContent>
                              </Select>
                            ) : (
                              <Input
                                value={(paramValues[param.name] as string) ?? param.default ?? ""}
                                onChange={(e) => setParamValues((prev) => ({ ...prev, [param.name]: e.target.value }))}
                              />
                            )}
                            {param.description && (
                              <p className="text-xs text-muted-foreground">{param.description}</p>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                    <Separator />
                  </>
                )}

                {/* Results Section */}
                {executionResult && (
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <Label className="text-sm font-medium">Results</Label>
                      <div className="flex items-center gap-2">
                        {executionResult.status === "completed" ? (
                          <Badge className="bg-green-100 text-green-800">
                            <CheckCircle className="h-3 w-3 mr-1" />
                            Completed
                          </Badge>
                        ) : executionResult.status === "failed" ? (
                          <Badge className="bg-red-100 text-red-800">
                            <XCircle className="h-3 w-3 mr-1" />
                            Failed
                          </Badge>
                        ) : (
                          <Badge>
                            <Clock className="h-3 w-3 mr-1" />
                            {executionResult.status}
                          </Badge>
                        )}
                        {executionResult.duration_ms && (
                          <Badge variant="outline">{executionResult.duration_ms.toFixed(0)}ms</Badge>
                        )}
                      </div>
                    </div>

                    {/* Error Message */}
                    {executionResult.error_message && (
                      <Card className="border-red-200 bg-red-50">
                        <CardContent className="p-3">
                          <p className="text-sm text-red-700">{executionResult.error_message}</p>
                          {executionResult.error_node_id && (
                            <p className="text-xs text-red-500 mt-1">Node: {executionResult.error_node_id}</p>
                          )}
                        </CardContent>
                      </Card>
                    )}

                    {/* Detection Results */}
                    {executionResult.output_data?.detections && (
                      <Card>
                        <CardHeader className="pb-2">
                          <CardTitle className="text-sm flex items-center gap-2">
                            Detections
                            <Badge variant="secondary">{executionResult.output_data.detections.length}</Badge>
                          </CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className="space-y-1 max-h-32 overflow-y-auto">
                            {executionResult.output_data.detections.map((det) => (
                              <div key={det.id} className="flex items-center justify-between text-sm py-1 border-b last:border-0">
                                <span className="font-medium">{det.class_name}</span>
                                <Badge variant="outline">{(det.confidence * 100).toFixed(1)}%</Badge>
                              </div>
                            ))}
                          </div>
                        </CardContent>
                      </Card>
                    )}

                    {/* Classification Results */}
                    {executionResult.output_data?.predictions && (
                      <Card>
                        <CardHeader className="pb-2">
                          <CardTitle className="text-sm">Classifications</CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className="space-y-1">
                            {executionResult.output_data.predictions.slice(0, 5).map((pred, i) => (
                              <div key={i} className="flex items-center justify-between text-sm py-1 border-b last:border-0">
                                <span className="font-medium">{pred.class_name}</span>
                                <Badge variant="outline">{(pred.confidence * 100).toFixed(1)}%</Badge>
                              </div>
                            ))}
                          </div>
                        </CardContent>
                      </Card>
                    )}

                    {/* Node Metrics */}
                    <NodeResultsSection metrics={executionResult.node_metrics} />

                    {/* Raw Output */}
                    <Collapsible>
                      <CollapsibleTrigger className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground">
                        <ChevronRight className="h-3 w-3" />
                        Raw JSON Output
                      </CollapsibleTrigger>
                      <CollapsibleContent>
                        <div className="mt-2 relative">
                          <pre className="text-xs bg-muted p-3 rounded-lg overflow-x-auto max-h-48">
                            {JSON.stringify(executionResult.output_data, null, 2)}
                          </pre>
                          <Button
                            variant="ghost"
                            size="icon"
                            className="absolute top-2 right-2 h-6 w-6"
                            onClick={() => copyToClipboard(executionResult.output_data)}
                          >
                            <Copy className="h-3 w-3" />
                          </Button>
                        </div>
                      </CollapsibleContent>
                    </Collapsible>
                  </div>
                )}
              </div>
            </div>

            {/* Footer Actions - Fixed */}
            <div className="p-4 border-t bg-muted/30 shrink-0">
              <div className="flex items-center justify-between">
                <p className="text-xs text-muted-foreground">
                  {hasImage ? "Ready to run" : "Select an image to test"}
                </p>
                <Button onClick={() => {
                  console.log("Run button clicked!", { canExecute, hasImage, previewUrl, imageBase64: !!imageBase64 });
                  executeMutation.mutate();
                }} disabled={!canExecute} className="gap-2">
                  {executeMutation.isPending ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin" />
                      Running...
                    </>
                  ) : (
                    <>
                      <Play className="h-4 w-4" />
                      Run Workflow
                    </>
                  )}
                </Button>
              </div>
            </div>
        </SheetContent>
      </Sheet>

      {/* Fullscreen Image Modal */}
      {fullscreenImage && previewUrl && (
        <div className="fixed inset-0 z-50 bg-black/90 flex items-center justify-center" onClick={() => setFullscreenImage(false)}>
          <Button
            variant="ghost"
            size="icon"
            className="absolute top-4 right-4 text-white hover:bg-white/20"
            onClick={() => setFullscreenImage(false)}
          >
            <X className="h-6 w-6" />
          </Button>
          <div className="max-w-[90vw] max-h-[90vh] relative">
            <ImagePreviewWithDetections
              src={previewUrl}
              detections={executionResult?.output_data?.detections}
            />
          </div>
        </div>
      )}
    </>
  );
}
