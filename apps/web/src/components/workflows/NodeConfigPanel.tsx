"use client";

import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { ChevronDown, Settings2, Variable, Sliders, Database, Sparkles } from "lucide-react";
import { useState, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { apiClient } from "@/lib/api-client";

// =============================================================================
// Types & Hooks
// =============================================================================

interface WorkflowModel {
  id: string;
  name: string;
  model_type: string;
  category: string;
  source: "pretrained" | "trained";
  provider?: string;
  is_active: boolean;
  is_default?: boolean;
  metrics?: {
    map?: number;
    accuracy?: number;
    recall_at_1?: number;
  };
  embedding_dim?: number;
  class_count?: number;
}

/**
 * Hook to fetch workflow models for a specific category
 */
function useWorkflowModels(category: "detection" | "classification" | "embedding" | "segmentation") {
  return useQuery({
    queryKey: ["workflow-models-list", category],
    queryFn: async () => {
      const result = await apiClient.getWorkflowModelsList({ model_type: category });
      return result.items as WorkflowModel[];
    },
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
}

// =============================================================================
// Parameter-aware Input Components
// =============================================================================

/**
 * Check if a value is a parameter reference
 */
function isParamRef(value: unknown): boolean {
  if (typeof value !== "string") return false;
  return value.includes("{{") && value.includes("}}");
}

/**
 * ParamSlider - A slider that can switch to parameter mode
 */
function ParamSlider({
  value,
  onChange,
  min,
  max,
  step,
  formatValue,
  label,
}: {
  value: unknown;
  onChange: (value: unknown) => void;
  min: number;
  max: number;
  step: number;
  formatValue?: (v: number) => string;
  label?: string;
}) {
  const isParam = isParamRef(value);
  const [paramMode, setParamMode] = useState(isParam);

  const numValue = useMemo(() => {
    if (typeof value === "number") return value;
    if (typeof value === "string" && !isParamRef(value)) {
      const parsed = parseFloat(value);
      return isNaN(parsed) ? min : parsed;
    }
    return min;
  }, [value, min]);

  const displayValue = formatValue ? formatValue(numValue) : String(numValue);

  if (paramMode) {
    return (
      <div className="flex gap-2">
        <Input
          value={typeof value === "string" ? value : ""}
          onChange={(e) => onChange(e.target.value)}
          placeholder="{{ params.name }}"
          className="h-8 flex-1 font-mono text-xs"
        />
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="outline"
                size="icon"
                className="h-8 w-8 shrink-0"
                onClick={() => {
                  setParamMode(false);
                  if (isParamRef(value)) {
                    onChange(min);
                  }
                }}
              >
                <Sliders className="h-3.5 w-3.5" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>Switch to slider</TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <Slider
          value={[numValue]}
          onValueChange={([v]) => onChange(v)}
          min={min}
          max={max}
          step={step}
          className="flex-1"
        />
        <span className="text-xs text-muted-foreground w-12 text-right">
          {displayValue}
        </span>
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="ghost"
                size="icon"
                className="h-6 w-6 shrink-0"
                onClick={() => setParamMode(true)}
              >
                <Variable className="h-3 w-3" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>Use parameter</TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>
    </div>
  );
}

/**
 * ParamInput - A text/number input that shows parameter hint
 */
function ParamInput({
  value,
  onChange,
  type = "text",
  placeholder,
  className,
  min,
  max,
}: {
  value: unknown;
  onChange: (value: unknown) => void;
  type?: "text" | "number";
  placeholder?: string;
  className?: string;
  min?: number;
  max?: number;
}) {
  const isParam = isParamRef(value);
  const stringValue = value !== undefined && value !== null ? String(value) : "";

  return (
    <div className="relative">
      <Input
        type={isParam ? "text" : type}
        value={stringValue}
        onChange={(e) => {
          const val = e.target.value;
          if (type === "number" && !isParamRef(val)) {
            onChange(val === "" ? undefined : parseFloat(val));
          } else {
            onChange(val);
          }
        }}
        placeholder={placeholder || (type === "number" ? "0 or {{ params.name }}" : "Value or {{ params.name }}")}
        className={`${className || "h-8"} ${isParam ? "font-mono text-xs pr-8" : ""}`}
        min={min}
        max={max}
      />
      {isParam && (
        <Variable className="absolute right-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-blue-500" />
      )}
    </div>
  );
}

export interface NodeConfig {
  [key: string]: unknown;
}

export interface NodeConfigPanelProps {
  nodeType: string;
  config: NodeConfig;
  onConfigChange: (key: string, value: unknown) => void;
}

/**
 * NodeConfigPanel - Shows SOTA configuration options based on node type
 */
export function NodeConfigPanel({
  nodeType,
  config,
  onConfigChange,
}: NodeConfigPanelProps) {
  const [advancedOpen, setAdvancedOpen] = useState(false);

  switch (nodeType) {
    case "detection":
      return (
        <DetectionConfig
          config={config}
          onConfigChange={onConfigChange}
          advancedOpen={advancedOpen}
          setAdvancedOpen={setAdvancedOpen}
        />
      );
    case "classification":
      return (
        <ClassificationConfig
          config={config}
          onConfigChange={onConfigChange}
          advancedOpen={advancedOpen}
          setAdvancedOpen={setAdvancedOpen}
        />
      );
    case "embedding":
      return (
        <EmbeddingConfig
          config={config}
          onConfigChange={onConfigChange}
          advancedOpen={advancedOpen}
          setAdvancedOpen={setAdvancedOpen}
        />
      );
    case "similarity_search":
      return (
        <SimilaritySearchConfig
          config={config}
          onConfigChange={onConfigChange}
          advancedOpen={advancedOpen}
          setAdvancedOpen={setAdvancedOpen}
        />
      );
    case "crop":
      return (
        <CropConfig config={config} onConfigChange={onConfigChange} />
      );
    case "filter":
      return (
        <FilterConfig config={config} onConfigChange={onConfigChange} />
      );
    case "image_input":
      return (
        <ImageInputConfig config={config} onConfigChange={onConfigChange} />
      );
    case "parameter_input":
      return (
        <ParameterInputConfig config={config} onConfigChange={onConfigChange} />
      );
    case "blur_region":
      return (
        <BlurRegionConfig config={config} onConfigChange={onConfigChange} />
      );
    case "draw_boxes":
      return (
        <DrawBoxesConfig
          config={config}
          onConfigChange={onConfigChange}
          advancedOpen={advancedOpen}
          setAdvancedOpen={setAdvancedOpen}
        />
      );
    case "condition":
      return (
        <ConditionConfig config={config} onConfigChange={onConfigChange} />
      );
    case "grid_builder":
      return (
        <GridBuilderConfig config={config} onConfigChange={onConfigChange} />
      );
    case "json_output":
      return (
        <JsonOutputConfig config={config} onConfigChange={onConfigChange} />
      );
    default:
      return (
        <div className="text-xs text-muted-foreground text-center py-4">
          No configuration options for this node type.
        </div>
      );
  }
}

// ============================================================================
// Detection Config
// ============================================================================
interface ConfigSectionProps {
  config: NodeConfig;
  onConfigChange: (key: string, value: unknown) => void;
  advancedOpen?: boolean;
  setAdvancedOpen?: (open: boolean) => void;
}

function DetectionConfig({
  config,
  onConfigChange,
  advancedOpen,
  setAdvancedOpen,
}: ConfigSectionProps) {
  // Fetch all detection models from API
  const { data: models, isLoading } = useWorkflowModels("detection");

  // Group models by source
  const groupedModels = useMemo(() => {
    if (!models) return { pretrained: [], trained: [] };
    return {
      pretrained: models.filter((m) => m.source === "pretrained"),
      trained: models.filter((m) => m.source === "trained"),
    };
  }, [models]);

  // Find selected model for display
  const selectedModel = useMemo(() => {
    const modelId = config.model_id as string;
    if (!modelId || !models) return undefined;
    return models.find((m) => m.id === modelId);
  }, [config.model_id, models]);

  // Handle model selection - sets both model_id and model_source
  const handleModelSelect = (modelId: string) => {
    const model = models?.find((m) => m.id === modelId);
    onConfigChange("model_id", modelId);
    onConfigChange("model_source", model?.source ?? "pretrained");
  };

  return (
    <div className="space-y-4">
      {/* Model Selection - Single grouped dropdown */}
      <div className="space-y-2">
        <Label className="text-xs">Model</Label>
        {isLoading ? (
          <Skeleton className="h-8 w-full" />
        ) : (
          <Select
            value={(config.model_id as string) ?? ""}
            onValueChange={handleModelSelect}
          >
            <SelectTrigger className="h-8">
              <SelectValue placeholder="Select a model...">
                {selectedModel && (
                  <div className="flex items-center gap-2">
                    {selectedModel.source === "pretrained" ? (
                      <Sparkles className="h-3 w-3 text-purple-500" />
                    ) : (
                      <Database className="h-3 w-3 text-blue-500" />
                    )}
                    <span className="truncate">{selectedModel.name}</span>
                    {selectedModel.is_default && (
                      <Badge variant="secondary" className="text-[10px] py-0 px-1">Default</Badge>
                    )}
                  </div>
                )}
              </SelectValue>
            </SelectTrigger>
            <SelectContent>
              {/* Pretrained Models Group */}
              {groupedModels.pretrained.length > 0 && (
                <SelectGroup>
                  <SelectLabel className="flex items-center gap-2 text-xs">
                    <Sparkles className="h-3 w-3 text-purple-500" />
                    Pretrained Models
                  </SelectLabel>
                  {groupedModels.pretrained.map((model) => (
                    <SelectItem key={model.id} value={model.id}>
                      <div className="flex items-center gap-2">
                        <span>{model.name}</span>
                        <span className="text-xs text-muted-foreground">
                          {model.model_type}
                        </span>
                        {model.is_default && (
                          <Badge variant="secondary" className="text-[10px] py-0 px-1">Default</Badge>
                        )}
                      </div>
                    </SelectItem>
                  ))}
                </SelectGroup>
              )}
              {/* Trained Models Group */}
              {groupedModels.trained.length > 0 && (
                <SelectGroup>
                  <SelectLabel className="flex items-center gap-2 text-xs">
                    <Database className="h-3 w-3 text-blue-500" />
                    Your Trained Models
                  </SelectLabel>
                  {groupedModels.trained.map((model) => (
                    <SelectItem key={model.id} value={model.id}>
                      <div className="flex flex-col">
                        <div className="flex items-center gap-2">
                          <span>{model.name}</span>
                          {model.is_default && (
                            <Badge variant="secondary" className="text-[10px] py-0 px-1">Default</Badge>
                          )}
                        </div>
                        <span className="text-xs text-muted-foreground">
                          {model.provider || model.model_type}
                          {model.metrics?.map && ` • mAP: ${(model.metrics.map * 100).toFixed(1)}%`}
                          {model.class_count && ` • ${model.class_count} classes`}
                        </span>
                      </div>
                    </SelectItem>
                  ))}
                </SelectGroup>
              )}
              {/* Empty state */}
              {groupedModels.pretrained.length === 0 && groupedModels.trained.length === 0 && (
                <div className="py-4 text-center text-xs text-muted-foreground">
                  No models available
                </div>
              )}
            </SelectContent>
          </Select>
        )}
        <p className="text-xs text-muted-foreground">
          {selectedModel
            ? `${selectedModel.source === "pretrained" ? "Pretrained" : "Trained"} ${selectedModel.provider || selectedModel.model_type} model`
            : "Select a detection model (YOLO, RT-DETR, D-FINE, RF-DETR...)"}
        </p>
      </div>

      {/* Confidence Threshold */}
      <div className="space-y-2">
        <Label className="text-xs">Confidence Threshold</Label>
        <ParamSlider
          value={config.confidence ?? 0.5}
          onChange={(v) => onConfigChange("confidence", v)}
          min={0}
          max={1}
          step={0.05}
          formatValue={(v) => v.toFixed(2)}
        />
      </div>

      {/* IoU Threshold */}
      <div className="space-y-2">
        <Label className="text-xs">IoU Threshold (NMS)</Label>
        <ParamSlider
          value={config.iou_threshold ?? 0.45}
          onChange={(v) => onConfigChange("iou_threshold", v)}
          min={0}
          max={1}
          step={0.05}
          formatValue={(v) => v.toFixed(2)}
        />
      </div>

      {/* Max Detections */}
      <div className="space-y-2">
        <Label className="text-xs">Max Detections</Label>
        <ParamInput
          type="number"
          value={config.max_detections ?? 300}
          onChange={(v) => onConfigChange("max_detections", v)}
          min={1}
          max={1000}
        />
      </div>

      {/* Class Filter */}
      <div className="space-y-2">
        <Label className="text-xs">Class Filter (optional)</Label>
        <Input
          type="text"
          value={
            Array.isArray(config.classes)
              ? (config.classes as string[]).join(", ")
              : ""
          }
          onChange={(e) => {
            const classes = e.target.value
              .split(",")
              .map((c) => c.trim())
              .filter(Boolean);
            onConfigChange("classes", classes.length ? classes : null);
          }}
          placeholder="e.g., person, car, bottle"
          className="h-8"
        />
        <p className="text-xs text-muted-foreground">
          Comma-separated class names. Leave empty to detect all.
        </p>
      </div>

      {/* Advanced Options */}
      <Collapsible open={advancedOpen} onOpenChange={setAdvancedOpen}>
        <CollapsibleTrigger className="flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground w-full py-2">
          <Settings2 className="h-3 w-3" />
          <span>Advanced Options</span>
          <ChevronDown
            className={`h-3 w-3 ml-auto transition-transform ${
              advancedOpen ? "rotate-180" : ""
            }`}
          />
        </CollapsibleTrigger>
        <CollapsibleContent className="space-y-4 pt-2">
          {/* Half Precision */}
          <div className="flex items-center justify-between">
            <div>
              <Label className="text-xs">Half Precision (FP16)</Label>
              <p className="text-xs text-muted-foreground">Faster GPU inference</p>
            </div>
            <Switch
              checked={(config.half_precision as boolean) ?? true}
              onCheckedChange={(v) => onConfigChange("half_precision", v)}
            />
          </div>

          {/* Agnostic NMS */}
          <div className="flex items-center justify-between">
            <div>
              <Label className="text-xs">Agnostic NMS</Label>
              <p className="text-xs text-muted-foreground">
                All classes compete in NMS
              </p>
            </div>
            <Switch
              checked={(config.agnostic_nms as boolean) ?? false}
              onCheckedChange={(v) => onConfigChange("agnostic_nms", v)}
            />
          </div>

          {/* Coordinate Format */}
          <div className="space-y-2">
            <Label className="text-xs">Coordinate Format</Label>
            <Select
              value={(config.coordinate_format as string) ?? "xyxy"}
              onValueChange={(v) => onConfigChange("coordinate_format", v)}
            >
              <SelectTrigger className="h-8">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="xyxy">xyxy (x1, y1, x2, y2)</SelectItem>
                <SelectItem value="xywh">xywh (x, y, width, height)</SelectItem>
                <SelectItem value="cxcywh">cxcywh (center x, center y, w, h)</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Normalize Coordinates */}
          <div className="flex items-center justify-between">
            <div>
              <Label className="text-xs">Normalize Coordinates</Label>
              <p className="text-xs text-muted-foreground">0-1 range output</p>
            </div>
            <Switch
              checked={(config.normalize_coords as boolean) ?? true}
              onCheckedChange={(v) => onConfigChange("normalize_coords", v)}
            />
          </div>
        </CollapsibleContent>
      </Collapsible>
    </div>
  );
}

// ============================================================================
// Classification Config
// ============================================================================
function ClassificationConfig({
  config,
  onConfigChange,
  advancedOpen,
  setAdvancedOpen,
}: ConfigSectionProps) {
  // Fetch all classification models from API
  const { data: models, isLoading } = useWorkflowModels("classification");

  // Group models by source
  const groupedModels = useMemo(() => {
    if (!models) return { pretrained: [], trained: [] };
    return {
      pretrained: models.filter((m) => m.source === "pretrained"),
      trained: models.filter((m) => m.source === "trained"),
    };
  }, [models]);

  // Find selected model for display
  const selectedModel = useMemo(() => {
    const modelId = config.model_id as string;
    if (!modelId || !models) return undefined;
    return models.find((m) => m.id === modelId);
  }, [config.model_id, models]);

  // Handle model selection
  const handleModelSelect = (modelId: string) => {
    const model = models?.find((m) => m.id === modelId);
    onConfigChange("model_id", modelId);
    onConfigChange("model_source", model?.source ?? "trained");
  };

  return (
    <div className="space-y-4">
      {/* Model Selection - Single grouped dropdown */}
      <div className="space-y-2">
        <Label className="text-xs">Model</Label>
        {isLoading ? (
          <Skeleton className="h-8 w-full" />
        ) : (
          <Select
            value={(config.model_id as string) ?? ""}
            onValueChange={handleModelSelect}
          >
            <SelectTrigger className="h-8">
              <SelectValue placeholder="Select a model...">
                {selectedModel && (
                  <div className="flex items-center gap-2">
                    {selectedModel.source === "pretrained" ? (
                      <Sparkles className="h-3 w-3 text-purple-500" />
                    ) : (
                      <Database className="h-3 w-3 text-blue-500" />
                    )}
                    <span className="truncate">{selectedModel.name}</span>
                    {selectedModel.is_default && (
                      <Badge variant="secondary" className="text-[10px] py-0 px-1">Default</Badge>
                    )}
                  </div>
                )}
              </SelectValue>
            </SelectTrigger>
            <SelectContent>
              {/* Trained Models First (more relevant for classification) */}
              {groupedModels.trained.length > 0 && (
                <SelectGroup>
                  <SelectLabel className="flex items-center gap-2 text-xs">
                    <Database className="h-3 w-3 text-blue-500" />
                    Your Trained Models
                  </SelectLabel>
                  {groupedModels.trained.map((model) => (
                    <SelectItem key={model.id} value={model.id}>
                      <div className="flex flex-col">
                        <div className="flex items-center gap-2">
                          <span>{model.name}</span>
                          {model.is_default && (
                            <Badge variant="secondary" className="text-[10px] py-0 px-1">Default</Badge>
                          )}
                        </div>
                        <span className="text-xs text-muted-foreground">
                          {model.provider || model.model_type}
                          {model.metrics?.accuracy && ` • Acc: ${(model.metrics.accuracy * 100).toFixed(1)}%`}
                          {model.class_count && ` • ${model.class_count} classes`}
                        </span>
                      </div>
                    </SelectItem>
                  ))}
                </SelectGroup>
              )}
              {/* Pretrained Models Group */}
              {groupedModels.pretrained.length > 0 && (
                <SelectGroup>
                  <SelectLabel className="flex items-center gap-2 text-xs">
                    <Sparkles className="h-3 w-3 text-purple-500" />
                    Pretrained Models
                  </SelectLabel>
                  {groupedModels.pretrained.map((model) => (
                    <SelectItem key={model.id} value={model.id}>
                      <div className="flex items-center gap-2">
                        <span>{model.name}</span>
                        <span className="text-xs text-muted-foreground">
                          {model.model_type}
                        </span>
                      </div>
                    </SelectItem>
                  ))}
                </SelectGroup>
              )}
              {/* Empty state */}
              {groupedModels.pretrained.length === 0 && groupedModels.trained.length === 0 && (
                <div className="py-4 text-center text-xs text-muted-foreground">
                  No models available
                </div>
              )}
            </SelectContent>
          </Select>
        )}
        <p className="text-xs text-muted-foreground">
          {selectedModel
            ? `${selectedModel.source === "pretrained" ? "Pretrained" : "Trained"} ${selectedModel.provider || selectedModel.model_type} model`
            : "Select a classification model (ViT, ConvNeXt, Swin, EfficientNet...)"}
        </p>
      </div>

      {/* Top K */}
      <div className="space-y-2">
        <Label className="text-xs">Top K Predictions</Label>
        <ParamInput
          type="number"
          value={config.top_k ?? 5}
          onChange={(v) => onConfigChange("top_k", v)}
          min={1}
          max={100}
        />
      </div>

      {/* Confidence Threshold */}
      <div className="space-y-2">
        <Label className="text-xs">Min Confidence</Label>
        <ParamSlider
          value={config.threshold ?? 0}
          onChange={(v) => onConfigChange("threshold", v)}
          min={0}
          max={1}
          step={0.05}
          formatValue={(v) => v.toFixed(2)}
        />
      </div>

      {/* Advanced Options */}
      <Collapsible open={advancedOpen} onOpenChange={setAdvancedOpen}>
        <CollapsibleTrigger className="flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground w-full py-2">
          <Settings2 className="h-3 w-3" />
          <span>Advanced Options</span>
          <ChevronDown
            className={`h-3 w-3 ml-auto transition-transform ${
              advancedOpen ? "rotate-180" : ""
            }`}
          />
        </CollapsibleTrigger>
        <CollapsibleContent className="space-y-4 pt-2">
          {/* Temperature */}
          <div className="space-y-2">
            <Label className="text-xs">Temperature</Label>
            <ParamSlider
              value={config.temperature ?? 1.0}
              onChange={(v) => onConfigChange("temperature", v)}
              min={0.1}
              max={2.0}
              step={0.1}
              formatValue={(v) => v.toFixed(2)}
            />
            <p className="text-xs text-muted-foreground">
              Lower = sharper probabilities, Higher = softer
            </p>
          </div>

          {/* Multi-crop Ensemble */}
          <div className="flex items-center justify-between">
            <div>
              <Label className="text-xs">Multi-crop Ensemble</Label>
              <p className="text-xs text-muted-foreground">
                SOTA for fine-grained classification
              </p>
            </div>
            <Switch
              checked={Boolean((config.multi_crop as Record<string, unknown>)?.enabled)}
              onCheckedChange={(v) =>
                onConfigChange("multi_crop", {
                  enabled: v,
                  scales: [1.0, 0.875, 0.75],
                  merge_mode: "mean",
                })
              }
            />
          </div>
        </CollapsibleContent>
      </Collapsible>
    </div>
  );
}

// ============================================================================
// Embedding Config
// ============================================================================
function EmbeddingConfig({
  config,
  onConfigChange,
  advancedOpen,
  setAdvancedOpen,
}: ConfigSectionProps) {
  // Fetch all embedding models from API
  const { data: models, isLoading } = useWorkflowModels("embedding");

  // Group models by source
  const groupedModels = useMemo(() => {
    if (!models) return { pretrained: [], trained: [] };
    return {
      pretrained: models.filter((m) => m.source === "pretrained"),
      trained: models.filter((m) => m.source === "trained"),
    };
  }, [models]);

  // Find selected model for display
  const selectedModel = useMemo(() => {
    const modelId = config.model_id as string;
    if (!modelId || !models) return undefined;
    return models.find((m) => m.id === modelId);
  }, [config.model_id, models]);

  // Handle model selection
  const handleModelSelect = (modelId: string) => {
    const model = models?.find((m) => m.id === modelId);
    onConfigChange("model_id", modelId);
    onConfigChange("model_source", model?.source ?? "pretrained");
  };

  return (
    <div className="space-y-4">
      {/* Model Selection - Single grouped dropdown */}
      <div className="space-y-2">
        <Label className="text-xs">Model</Label>
        {isLoading ? (
          <Skeleton className="h-8 w-full" />
        ) : (
          <Select
            value={(config.model_id as string) ?? ""}
            onValueChange={handleModelSelect}
          >
            <SelectTrigger className="h-8">
              <SelectValue placeholder="Select a model...">
                {selectedModel && (
                  <div className="flex items-center gap-2">
                    {selectedModel.source === "pretrained" ? (
                      <Sparkles className="h-3 w-3 text-purple-500" />
                    ) : (
                      <Database className="h-3 w-3 text-blue-500" />
                    )}
                    <span className="truncate">{selectedModel.name}</span>
                    {selectedModel.embedding_dim && (
                      <span className="text-xs text-muted-foreground">{selectedModel.embedding_dim}d</span>
                    )}
                    {selectedModel.is_default && (
                      <Badge variant="secondary" className="text-[10px] py-0 px-1">Default</Badge>
                    )}
                  </div>
                )}
              </SelectValue>
            </SelectTrigger>
            <SelectContent>
              {/* Pretrained Models First (foundation models) */}
              {groupedModels.pretrained.length > 0 && (
                <SelectGroup>
                  <SelectLabel className="flex items-center gap-2 text-xs">
                    <Sparkles className="h-3 w-3 text-purple-500" />
                    Pretrained Models
                  </SelectLabel>
                  {groupedModels.pretrained.map((model) => (
                    <SelectItem key={model.id} value={model.id}>
                      <div className="flex items-center gap-2">
                        <span>{model.name}</span>
                        <span className="text-xs text-muted-foreground">
                          {model.embedding_dim && `${model.embedding_dim}d • `}{model.model_type}
                        </span>
                      </div>
                    </SelectItem>
                  ))}
                </SelectGroup>
              )}
              {/* Trained Models Group */}
              {groupedModels.trained.length > 0 && (
                <SelectGroup>
                  <SelectLabel className="flex items-center gap-2 text-xs">
                    <Database className="h-3 w-3 text-blue-500" />
                    Your Trained Models
                  </SelectLabel>
                  {groupedModels.trained.map((model) => (
                    <SelectItem key={model.id} value={model.id}>
                      <div className="flex flex-col">
                        <div className="flex items-center gap-2">
                          <span>{model.name}</span>
                          {model.is_default && (
                            <Badge variant="secondary" className="text-[10px] py-0 px-1">Default</Badge>
                          )}
                        </div>
                        <span className="text-xs text-muted-foreground">
                          {model.embedding_dim && `${model.embedding_dim}d • `}
                          {model.provider || model.model_type}
                          {model.metrics?.recall_at_1 && ` • R@1: ${(model.metrics.recall_at_1 * 100).toFixed(1)}%`}
                        </span>
                      </div>
                    </SelectItem>
                  ))}
                </SelectGroup>
              )}
              {/* Empty state */}
              {groupedModels.pretrained.length === 0 && groupedModels.trained.length === 0 && (
                <div className="py-4 text-center text-xs text-muted-foreground">
                  No models available
                </div>
              )}
            </SelectContent>
          </Select>
        )}
        <p className="text-xs text-muted-foreground">
          {selectedModel
            ? `${selectedModel.source === "pretrained" ? "Pretrained" : "Fine-tuned"} ${selectedModel.provider || selectedModel.model_type}`
            : "Select an embedding model (DINOv2, CLIP, SigLIP...)"}
        </p>
      </div>

      {/* Normalize */}
      <div className="flex items-center justify-between">
        <div>
          <Label className="text-xs">L2 Normalize</Label>
          <p className="text-xs text-muted-foreground">
            Required for cosine similarity
          </p>
        </div>
        <Switch
          checked={(config.normalize as boolean) ?? true}
          onCheckedChange={(v) => onConfigChange("normalize", v)}
        />
      </div>

      {/* Pooling Strategy */}
      <div className="space-y-2">
        <Label className="text-xs">Pooling Strategy</Label>
        <Select
          value={(config.pooling as string) ?? "cls"}
          onValueChange={(v) => onConfigChange("pooling", v)}
        >
          <SelectTrigger className="h-8">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="cls">CLS Token (Default)</SelectItem>
            <SelectItem value="mean">Mean Pooling</SelectItem>
            <SelectItem value="gem">GeM Pooling (SOTA)</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* GeM Power - only show when pooling is gem */}
      {config.pooling === "gem" && (
        <div className="space-y-2">
          <Label className="text-xs">GeM Power (p)</Label>
          <ParamSlider
            value={config.gem_p ?? 3.0}
            onChange={(v) => onConfigChange("gem_p", v)}
            min={1}
            max={10}
            step={0.5}
            formatValue={(v) => v.toFixed(1)}
          />
        </div>
      )}

      {/* Advanced Options */}
      <Collapsible open={advancedOpen} onOpenChange={setAdvancedOpen}>
        <CollapsibleTrigger className="flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground w-full py-2">
          <Settings2 className="h-3 w-3" />
          <span>Advanced Options</span>
          <ChevronDown
            className={`h-3 w-3 ml-auto transition-transform ${
              advancedOpen ? "rotate-180" : ""
            }`}
          />
        </CollapsibleTrigger>
        <CollapsibleContent className="space-y-4 pt-2">
          {/* Layer Selection */}
          <div className="space-y-2">
            <Label className="text-xs">Transformer Layer</Label>
            <Input
              type="number"
              value={(config.layer as number) ?? -1}
              onChange={(e) =>
                onConfigChange("layer", parseInt(e.target.value) || -1)
              }
              min={-12}
              max={-1}
              className="h-8"
            />
            <p className="text-xs text-muted-foreground">
              -1 = last layer, -2 = second to last, etc.
            </p>
          </div>
        </CollapsibleContent>
      </Collapsible>
    </div>
  );
}

// ============================================================================
// Similarity Search Config
// ============================================================================
function SimilaritySearchConfig({
  config,
  onConfigChange,
  advancedOpen,
  setAdvancedOpen,
}: ConfigSectionProps) {
  return (
    <div className="space-y-4">
      {/* Collection */}
      <div className="space-y-2">
        <Label className="text-xs">Collection Name</Label>
        <ParamInput
          value={config.collection ?? ""}
          onChange={(v) => onConfigChange("collection", v)}
          placeholder="e.g., products or {{ params.collection }}"
        />
      </div>

      {/* Top K */}
      <div className="space-y-2">
        <Label className="text-xs">Top K Results</Label>
        <ParamInput
          type="number"
          value={config.top_k ?? 5}
          onChange={(v) => onConfigChange("top_k", v)}
          min={1}
          max={100}
        />
      </div>

      {/* Similarity Threshold */}
      <div className="space-y-2">
        <Label className="text-xs">Similarity Threshold</Label>
        <ParamSlider
          value={config.threshold ?? 0.7}
          onChange={(v) => onConfigChange("threshold", v)}
          min={0}
          max={1}
          step={0.05}
          formatValue={(v) => v.toFixed(2)}
        />
      </div>

      {/* Advanced Options */}
      <Collapsible open={advancedOpen} onOpenChange={setAdvancedOpen}>
        <CollapsibleTrigger className="flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground w-full py-2">
          <Settings2 className="h-3 w-3" />
          <span>Advanced Options</span>
          <ChevronDown
            className={`h-3 w-3 ml-auto transition-transform ${
              advancedOpen ? "rotate-180" : ""
            }`}
          />
        </CollapsibleTrigger>
        <CollapsibleContent className="space-y-4 pt-2">
          {/* Group By */}
          <div className="space-y-2">
            <Label className="text-xs">Group By Field</Label>
            <Input
              type="text"
              value={(config.group_by as string) ?? ""}
              onChange={(e) => onConfigChange("group_by", e.target.value || null)}
              placeholder="e.g., product_id"
              className="h-8"
            />
          </div>

          {/* Group Size */}
          {Boolean(config.group_by) && (
            <div className="space-y-2">
              <Label className="text-xs">Results per Group</Label>
              <Input
                type="number"
                value={(config.group_size as number) ?? 3}
                onChange={(e) =>
                  onConfigChange("group_size", parseInt(e.target.value) || 3)
                }
                min={1}
                max={10}
                className="h-8"
              />
            </div>
          )}

          {/* Include Payload */}
          <div className="flex items-center justify-between">
            <div>
              <Label className="text-xs">Include Payload</Label>
              <p className="text-xs text-muted-foreground">
                Return stored metadata
              </p>
            </div>
            <Switch
              checked={(config.with_payload as boolean) ?? true}
              onCheckedChange={(v) => onConfigChange("with_payload", v)}
            />
          </div>
        </CollapsibleContent>
      </Collapsible>
    </div>
  );
}

// ============================================================================
// Crop Config
// ============================================================================
function CropConfig({ config, onConfigChange }: Omit<ConfigSectionProps, 'advancedOpen' | 'setAdvancedOpen'>) {
  return (
    <div className="space-y-4">
      {/* Padding */}
      <div className="space-y-2">
        <Label className="text-xs">Padding (px)</Label>
        <ParamSlider
          value={config.padding ?? 0}
          onChange={(v) => onConfigChange("padding", v)}
          min={0}
          max={100}
          step={5}
          formatValue={(v) => `${v}px`}
        />
      </div>

      {/* Min Size */}
      <div className="space-y-2">
        <Label className="text-xs">Minimum Size (px)</Label>
        <ParamInput
          type="number"
          value={config.min_size ?? 32}
          onChange={(v) => onConfigChange("min_size", v)}
          min={1}
          max={512}
        />
      </div>
    </div>
  );
}

// ============================================================================
// Filter Config
// ============================================================================
function FilterConfig({ config, onConfigChange }: Omit<ConfigSectionProps, 'advancedOpen' | 'setAdvancedOpen'>) {
  return (
    <div className="space-y-4">
      {/* Field */}
      <div className="space-y-2">
        <Label className="text-xs">Filter Field</Label>
        <Select
          value={(config.field as string) ?? "confidence"}
          onValueChange={(v) => onConfigChange("field", v)}
        >
          <SelectTrigger className="h-8">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="confidence">Confidence</SelectItem>
            <SelectItem value="class_name">Class Name</SelectItem>
            <SelectItem value="area">Area</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* Operator */}
      <div className="space-y-2">
        <Label className="text-xs">Operator</Label>
        <Select
          value={(config.operator as string) ?? "greater_than"}
          onValueChange={(v) => onConfigChange("operator", v)}
        >
          <SelectTrigger className="h-8">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="greater_than">Greater Than (&gt;)</SelectItem>
            <SelectItem value="less_than">Less Than (&lt;)</SelectItem>
            <SelectItem value="equals">Equals (=)</SelectItem>
            <SelectItem value="not_equals">Not Equals (!=)</SelectItem>
            <SelectItem value="in">In List</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* Value */}
      <div className="space-y-2">
        <Label className="text-xs">Value</Label>
        <ParamInput
          type={config.field === "class_name" ? "text" : "number"}
          value={config.value ?? ""}
          onChange={(v) => onConfigChange("value", v)}
          placeholder={config.field === "class_name" ? "Class name or {{ params.x }}" : "0 or {{ params.x }}"}
        />
      </div>
    </div>
  );
}

// ============================================================================
// Image Input Config - SOTA
// ============================================================================
function ImageInputConfig({ config, onConfigChange }: Omit<ConfigSectionProps, 'advancedOpen' | 'setAdvancedOpen'>) {
  const [showAdvanced, setShowAdvanced] = useState(false);

  return (
    <div className="space-y-4">
      {/* Input Source Info */}
      <div className="text-xs text-muted-foreground bg-muted/50 rounded-md p-3">
        <p className="font-medium text-foreground mb-1">Image Input Node</p>
        <p>This node accepts images via:</p>
        <ul className="list-disc list-inside mt-1 space-y-1">
          <li><code className="bg-muted px-1 rounded">image_url</code> - URL to image</li>
          <li><code className="bg-muted px-1 rounded">image_base64</code> - Base64 encoded</li>
        </ul>
      </div>

      {/* EXIF Rotation */}
      <div className="flex items-center justify-between">
        <div>
          <Label className="text-xs">Fix Photo Rotation</Label>
          <p className="text-xs text-muted-foreground">Auto-correct phone photos</p>
        </div>
        <Switch
          checked={(config.apply_exif_rotation as boolean) ?? true}
          onCheckedChange={(v) => onConfigChange("apply_exif_rotation", v)}
        />
      </div>

      {/* Auto Resize */}
      <div className="flex items-center justify-between">
        <div>
          <Label className="text-xs">Auto Resize</Label>
          <p className="text-xs text-muted-foreground">Downscale large images</p>
        </div>
        <Switch
          checked={(config.auto_resize as boolean) ?? false}
          onCheckedChange={(v) => onConfigChange("auto_resize", v)}
        />
      </div>

      {Boolean(config.auto_resize) && (
        <div className="space-y-2 pl-4 border-l-2 border-muted">
          <div className="flex items-center justify-between">
            <Label className="text-xs">Max Dimension</Label>
            <span className="text-xs text-muted-foreground">
              {(config.max_dimension as number) ?? 1920}px
            </span>
          </div>
          <Slider
            value={[(config.max_dimension as number) ?? 1920]}
            onValueChange={([v]) => onConfigChange("max_dimension", v)}
            min={512}
            max={4096}
            step={128}
            className="w-full"
          />
        </div>
      )}

      {/* Convert to RGB */}
      <div className="flex items-center justify-between">
        <div>
          <Label className="text-xs">Convert to RGB</Label>
          <p className="text-xs text-muted-foreground">Normalize all formats</p>
        </div>
        <Switch
          checked={(config.convert_to_rgb as boolean) ?? true}
          onCheckedChange={(v) => onConfigChange("convert_to_rgb", v)}
        />
      </div>

      {/* Advanced Options */}
      <Collapsible open={showAdvanced} onOpenChange={setShowAdvanced}>
        <CollapsibleTrigger className="flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground w-full py-2">
          <Settings2 className="h-3 w-3" />
          <span>Validation & Limits</span>
          <ChevronDown
            className={`h-3 w-3 ml-auto transition-transform ${
              showAdvanced ? "rotate-180" : ""
            }`}
          />
        </CollapsibleTrigger>
        <CollapsibleContent className="space-y-4 pt-2">
          {/* Format Validation */}
          <div className="flex items-center justify-between">
            <div>
              <Label className="text-xs">Validate Format</Label>
              <p className="text-xs text-muted-foreground">Restrict file types</p>
            </div>
            <Switch
              checked={(config.validate_format as boolean) ?? false}
              onCheckedChange={(v) => onConfigChange("validate_format", v)}
            />
          </div>

          {Boolean(config.validate_format) && (
            <div className="space-y-2 pl-4 border-l-2 border-muted">
              <Label className="text-xs">Allowed Formats</Label>
              <div className="flex flex-wrap gap-1">
                {["JPEG", "PNG", "WEBP", "GIF", "BMP", "HEIC"].map((format) => {
                  const allowed = (config.allowed_formats as string[]) ?? ["JPEG", "PNG", "WEBP"];
                  const isSelected = allowed.includes(format);
                  return (
                    <button
                      key={format}
                      type="button"
                      onClick={() => {
                        const newFormats = isSelected
                          ? allowed.filter((f) => f !== format)
                          : [...allowed, format];
                        onConfigChange("allowed_formats", newFormats);
                      }}
                      className={`px-2 py-0.5 text-xs rounded border transition-colors ${
                        isSelected
                          ? "bg-primary text-primary-foreground border-primary"
                          : "bg-muted/50 text-muted-foreground border-muted hover:border-primary/50"
                      }`}
                    >
                      {format}
                    </button>
                  );
                })}
              </div>
            </div>
          )}

          {/* Min Dimension */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label className="text-xs">Min Dimension</Label>
              <span className="text-xs text-muted-foreground">
                {(config.min_dimension as number) || "No limit"}
                {(config.min_dimension as number) > 0 && "px"}
              </span>
            </div>
            <Slider
              value={[(config.min_dimension as number) ?? 0]}
              onValueChange={([v]) => onConfigChange("min_dimension", v)}
              min={0}
              max={1024}
              step={32}
              className="w-full"
            />
          </div>

          {/* Max File Size */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label className="text-xs">Max File Size</Label>
              <span className="text-xs text-muted-foreground">
                {(config.max_file_size_mb as number) || "No limit"}
                {(config.max_file_size_mb as number) > 0 && " MB"}
              </span>
            </div>
            <Slider
              value={[(config.max_file_size_mb as number) ?? 0]}
              onValueChange={([v]) => onConfigChange("max_file_size_mb", v)}
              min={0}
              max={50}
              step={1}
              className="w-full"
            />
          </div>
        </CollapsibleContent>
      </Collapsible>
    </div>
  );
}

// ============================================================================
// Parameter Input Config
// ============================================================================
function ParameterInputConfig({ config, onConfigChange }: Omit<ConfigSectionProps, 'advancedOpen' | 'setAdvancedOpen'>) {
  return (
    <div className="space-y-4">
      {/* Parameter Name */}
      <div className="space-y-2">
        <Label className="text-xs">Parameter Name</Label>
        <Input
          type="text"
          value={(config.parameter_name as string) ?? ""}
          onChange={(e) => onConfigChange("parameter_name", e.target.value)}
          placeholder="e.g., threshold, category_id"
          className="h-8"
        />
        <p className="text-xs text-muted-foreground">
          Name used to reference this parameter at runtime
        </p>
      </div>

      {/* Default Value */}
      <div className="space-y-2">
        <Label className="text-xs">Default Value</Label>
        <Input
          type="text"
          value={(config.default_value as string) ?? ""}
          onChange={(e) => onConfigChange("default_value", e.target.value)}
          placeholder="Optional default value"
          className="h-8"
        />
      </div>

      {/* Type Hint */}
      <div className="space-y-2">
        <Label className="text-xs">Type Hint</Label>
        <Select
          value={(config.type_hint as string) ?? "string"}
          onValueChange={(v) => onConfigChange("type_hint", v)}
        >
          <SelectTrigger className="h-8">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="string">String</SelectItem>
            <SelectItem value="number">Number</SelectItem>
            <SelectItem value="boolean">Boolean</SelectItem>
            <SelectItem value="array">Array</SelectItem>
            <SelectItem value="object">Object</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* Required */}
      <div className="flex items-center justify-between">
        <div>
          <Label className="text-xs">Required</Label>
          <p className="text-xs text-muted-foreground">
            Workflow fails if not provided
          </p>
        </div>
        <Switch
          checked={(config.required as boolean) ?? true}
          onCheckedChange={(v) => onConfigChange("required", v)}
        />
      </div>
    </div>
  );
}

// ============================================================================
// Blur Region Config
// ============================================================================
function BlurRegionConfig({ config, onConfigChange }: Omit<ConfigSectionProps, 'advancedOpen' | 'setAdvancedOpen'>) {
  return (
    <div className="space-y-4">
      {/* Blur Type */}
      <div className="space-y-2">
        <Label className="text-xs">Blur Type</Label>
        <Select
          value={(config.blur_type as string) ?? "gaussian"}
          onValueChange={(v) => onConfigChange("blur_type", v)}
        >
          <SelectTrigger className="h-8">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="gaussian">Gaussian Blur</SelectItem>
            <SelectItem value="pixelate">Pixelate</SelectItem>
            <SelectItem value="black">Black Fill</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* Intensity */}
      {config.blur_type !== "black" && (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label className="text-xs">
              {config.blur_type === "pixelate" ? "Pixel Size" : "Blur Intensity"}
            </Label>
            <span className="text-xs text-muted-foreground">
              {(config.intensity as number) ?? 21}
            </span>
          </div>
          <Slider
            value={[(config.intensity as number) ?? 21]}
            onValueChange={([v]) => onConfigChange("intensity", v)}
            min={config.blur_type === "pixelate" ? 5 : 3}
            max={config.blur_type === "pixelate" ? 50 : 99}
            step={config.blur_type === "pixelate" ? 5 : 2}
            className="w-full"
          />
        </div>
      )}

      {/* Filter Classes */}
      <div className="space-y-2">
        <Label className="text-xs">Filter Classes (optional)</Label>
        <Input
          type="text"
          value={
            Array.isArray(config.filter_classes)
              ? (config.filter_classes as string[]).join(", ")
              : ""
          }
          onChange={(e) => {
            const classes = e.target.value
              .split(",")
              .map((c) => c.trim())
              .filter(Boolean);
            onConfigChange("filter_classes", classes.length ? classes : null);
          }}
          placeholder="e.g., face, license_plate"
          className="h-8"
        />
        <p className="text-xs text-muted-foreground">
          Comma-separated list. Leave empty to blur all detections.
        </p>
      </div>
    </div>
  );
}

// ============================================================================
// Draw Boxes Config
// ============================================================================
function DrawBoxesConfig({
  config,
  onConfigChange,
  advancedOpen,
  setAdvancedOpen,
}: ConfigSectionProps) {
  return (
    <div className="space-y-4">
      {/* Line Width */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label className="text-xs">Line Width (px)</Label>
          <span className="text-xs text-muted-foreground">
            {(config.line_width as number) ?? 2}
          </span>
        </div>
        <Slider
          value={[(config.line_width as number) ?? 2]}
          onValueChange={([v]) => onConfigChange("line_width", v)}
          min={1}
          max={10}
          step={1}
          className="w-full"
        />
      </div>

      {/* Show Labels */}
      <div className="flex items-center justify-between">
        <div>
          <Label className="text-xs">Show Labels</Label>
          <p className="text-xs text-muted-foreground">Display class names</p>
        </div>
        <Switch
          checked={(config.show_labels as boolean) ?? true}
          onCheckedChange={(v) => onConfigChange("show_labels", v)}
        />
      </div>

      {/* Show Confidence */}
      <div className="flex items-center justify-between">
        <div>
          <Label className="text-xs">Show Confidence</Label>
          <p className="text-xs text-muted-foreground">Display confidence %</p>
        </div>
        <Switch
          checked={(config.show_confidence as boolean) ?? true}
          onCheckedChange={(v) => onConfigChange("show_confidence", v)}
        />
      </div>

      {/* Advanced Options */}
      <Collapsible open={advancedOpen} onOpenChange={setAdvancedOpen}>
        <CollapsibleTrigger className="flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground w-full py-2">
          <Settings2 className="h-3 w-3" />
          <span>Advanced Options</span>
          <ChevronDown
            className={`h-3 w-3 ml-auto transition-transform ${
              advancedOpen ? "rotate-180" : ""
            }`}
          />
        </CollapsibleTrigger>
        <CollapsibleContent className="space-y-4 pt-2">
          {/* Color By Class */}
          <div className="flex items-center justify-between">
            <div>
              <Label className="text-xs">Color By Class</Label>
              <p className="text-xs text-muted-foreground">
                Different color per class
              </p>
            </div>
            <Switch
              checked={(config.color_by_class as boolean) ?? true}
              onCheckedChange={(v) => onConfigChange("color_by_class", v)}
            />
          </div>

          {/* Default Color */}
          {!config.color_by_class && (
            <div className="space-y-2">
              <Label className="text-xs">Default Color</Label>
              <div className="flex gap-2">
                <Input
                  type="color"
                  value={(config.default_color as string) ?? "#00FF00"}
                  onChange={(e) => onConfigChange("default_color", e.target.value)}
                  className="h-8 w-12 p-1"
                />
                <Input
                  type="text"
                  value={(config.default_color as string) ?? "#00FF00"}
                  onChange={(e) => onConfigChange("default_color", e.target.value)}
                  className="h-8 flex-1"
                />
              </div>
            </div>
          )}

          {/* Font Size */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label className="text-xs">Font Size</Label>
              <span className="text-xs text-muted-foreground">
                {(config.font_size as number) ?? 12}px
              </span>
            </div>
            <Slider
              value={[(config.font_size as number) ?? 12]}
              onValueChange={([v]) => onConfigChange("font_size", v)}
              min={8}
              max={32}
              step={2}
              className="w-full"
            />
          </div>
        </CollapsibleContent>
      </Collapsible>
    </div>
  );
}

// ============================================================================
// Condition Config
// ============================================================================
function ConditionConfig({ config, onConfigChange }: Omit<ConfigSectionProps, 'advancedOpen' | 'setAdvancedOpen'>) {
  return (
    <div className="space-y-4">
      {/* Expression */}
      <div className="space-y-2">
        <Label className="text-xs">Expression Field</Label>
        <Input
          type="text"
          value={(config.expression as string) ?? ""}
          onChange={(e) => onConfigChange("expression", e.target.value)}
          placeholder="e.g., detections.length, confidence"
          className="h-8"
        />
        <p className="text-xs text-muted-foreground">
          Field to evaluate from previous step output
        </p>
      </div>

      {/* Operator */}
      <div className="space-y-2">
        <Label className="text-xs">Operator</Label>
        <Select
          value={(config.operator as string) ?? "greater_than"}
          onValueChange={(v) => onConfigChange("operator", v)}
        >
          <SelectTrigger className="h-8">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="greater_than">Greater Than (&gt;)</SelectItem>
            <SelectItem value="less_than">Less Than (&lt;)</SelectItem>
            <SelectItem value="equals">Equals (==)</SelectItem>
            <SelectItem value="not_equals">Not Equals (!=)</SelectItem>
            <SelectItem value="contains">Contains</SelectItem>
            <SelectItem value="exists">Exists (not null)</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* Compare Value */}
      {config.operator !== "exists" && (
        <div className="space-y-2">
          <Label className="text-xs">Compare Value</Label>
          <Input
            type="text"
            value={(config.compare_value as string) ?? ""}
            onChange={(e) => onConfigChange("compare_value", e.target.value)}
            placeholder="Value to compare against"
            className="h-8"
          />
        </div>
      )}

      {/* Info Box */}
      <div className="text-xs text-muted-foreground bg-muted/50 rounded-md p-3">
        <p className="font-medium text-foreground mb-1">Branch Logic</p>
        <p>
          <strong>True branch:</strong> Condition passes<br />
          <strong>False branch:</strong> Condition fails
        </p>
      </div>
    </div>
  );
}

// ============================================================================
// Grid Builder Config
// ============================================================================
function GridBuilderConfig({ config, onConfigChange }: Omit<ConfigSectionProps, 'advancedOpen' | 'setAdvancedOpen'>) {
  return (
    <div className="space-y-4">
      {/* Sort By */}
      <div className="space-y-2">
        <Label className="text-xs">Sort By</Label>
        <Select
          value={(config.sort_by as string) ?? "position"}
          onValueChange={(v) => onConfigChange("sort_by", v)}
        >
          <SelectTrigger className="h-8">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="position">Position (Top-Left)</SelectItem>
            <SelectItem value="confidence">Confidence</SelectItem>
            <SelectItem value="area">Area (Size)</SelectItem>
            <SelectItem value="none">No Sorting</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* Sort Order */}
      <div className="space-y-2">
        <Label className="text-xs">Sort Order</Label>
        <Select
          value={(config.sort_order as string) ?? "asc"}
          onValueChange={(v) => onConfigChange("sort_order", v)}
        >
          <SelectTrigger className="h-8">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="asc">Ascending</SelectItem>
            <SelectItem value="desc">Descending</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* Group By Shelf */}
      <div className="flex items-center justify-between">
        <div>
          <Label className="text-xs">Group by Shelf</Label>
          <p className="text-xs text-muted-foreground">
            Group items by row position
          </p>
        </div>
        <Switch
          checked={(config.group_by_shelf as boolean) ?? false}
          onCheckedChange={(v) => onConfigChange("group_by_shelf", v)}
        />
      </div>

      {/* Shelf Tolerance */}
      {Boolean(config.group_by_shelf) && (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label className="text-xs">Shelf Tolerance (px)</Label>
            <span className="text-xs text-muted-foreground">
              {(config.shelf_tolerance as number) ?? 50}
            </span>
          </div>
          <Slider
            value={[(config.shelf_tolerance as number) ?? 50]}
            onValueChange={([v]) => onConfigChange("shelf_tolerance", v)}
            min={10}
            max={200}
            step={10}
            className="w-full"
          />
          <p className="text-xs text-muted-foreground">
            Y-coordinate tolerance for grouping items into same shelf
          </p>
        </div>
      )}
    </div>
  );
}

// ============================================================================
// JSON Output Config
// ============================================================================
function JsonOutputConfig({ config, onConfigChange }: Omit<ConfigSectionProps, 'advancedOpen' | 'setAdvancedOpen'>) {
  return (
    <div className="space-y-4">
      {/* Include Metadata */}
      <div className="flex items-center justify-between">
        <div>
          <Label className="text-xs">Include Metadata</Label>
          <p className="text-xs text-muted-foreground">
            Add timing, version info
          </p>
        </div>
        <Switch
          checked={(config.include_metadata as boolean) ?? true}
          onCheckedChange={(v) => onConfigChange("include_metadata", v)}
        />
      </div>

      {/* Flatten Output */}
      <div className="flex items-center justify-between">
        <div>
          <Label className="text-xs">Flatten Output</Label>
          <p className="text-xs text-muted-foreground">
            Simplify nested structures
          </p>
        </div>
        <Switch
          checked={(config.flatten as boolean) ?? false}
          onCheckedChange={(v) => onConfigChange("flatten", v)}
        />
      </div>

      {/* Include Fields */}
      <div className="space-y-2">
        <Label className="text-xs">Include Fields (optional)</Label>
        <Input
          type="text"
          value={
            Array.isArray(config.include_fields)
              ? (config.include_fields as string[]).join(", ")
              : ""
          }
          onChange={(e) => {
            const fields = e.target.value
              .split(",")
              .map((f) => f.trim())
              .filter(Boolean);
            onConfigChange("include_fields", fields.length ? fields : null);
          }}
          placeholder="e.g., detections, classifications"
          className="h-8"
        />
        <p className="text-xs text-muted-foreground">
          Comma-separated. Leave empty to include all.
        </p>
      </div>

      {/* Exclude Fields */}
      <div className="space-y-2">
        <Label className="text-xs">Exclude Fields (optional)</Label>
        <Input
          type="text"
          value={
            Array.isArray(config.exclude_fields)
              ? (config.exclude_fields as string[]).join(", ")
              : ""
          }
          onChange={(e) => {
            const fields = e.target.value
              .split(",")
              .map((f) => f.trim())
              .filter(Boolean);
            onConfigChange("exclude_fields", fields.length ? fields : null);
          }}
          placeholder="e.g., embeddings, raw_data"
          className="h-8"
        />
        <p className="text-xs text-muted-foreground">
          Comma-separated fields to exclude from output.
        </p>
      </div>
    </div>
  );
}
