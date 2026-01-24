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
import { ChevronDown, Settings2, Variable, Sliders, Database, Sparkles, Plus, X } from "lucide-react";
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
  input_size?: number;
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
        <CropConfig
          config={config}
          onConfigChange={onConfigChange}
          advancedOpen={advancedOpen}
          setAdvancedOpen={setAdvancedOpen}
        />
      );
    case "resize":
      return (
        <ResizeConfig
          config={config}
          onConfigChange={onConfigChange}
          advancedOpen={advancedOpen}
          setAdvancedOpen={setAdvancedOpen}
        />
      );
    case "tile":
      return (
        <TileConfig
          config={config}
          onConfigChange={onConfigChange}
          advancedOpen={advancedOpen}
          setAdvancedOpen={setAdvancedOpen}
        />
      );
    case "stitch":
      return (
        <StitchConfig
          config={config}
          onConfigChange={onConfigChange}
          advancedOpen={advancedOpen}
          setAdvancedOpen={setAdvancedOpen}
        />
      );
    case "rotate_flip":
      return (
        <RotateFlipConfig
          config={config}
          onConfigChange={onConfigChange}
          advancedOpen={advancedOpen}
          setAdvancedOpen={setAdvancedOpen}
        />
      );
    case "normalize":
      return (
        <NormalizeConfig
          config={config}
          onConfigChange={onConfigChange}
          advancedOpen={advancedOpen}
          setAdvancedOpen={setAdvancedOpen}
        />
      );
    case "filter":
      return (
        <FilterConfig
          config={config}
          onConfigChange={onConfigChange}
          advancedOpen={advancedOpen}
          setAdvancedOpen={setAdvancedOpen}
        />
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
        <ConditionConfig
          config={config}
          onConfigChange={onConfigChange}
          advancedOpen={advancedOpen}
          setAdvancedOpen={setAdvancedOpen}
        />
      );
    case "grid_builder":
      return (
        <GridBuilderConfig
          config={config}
          onConfigChange={onConfigChange}
          advancedOpen={advancedOpen}
          setAdvancedOpen={setAdvancedOpen}
        />
      );
    case "json_output":
      return (
        <JsonOutputConfig config={config} onConfigChange={onConfigChange} />
      );
    case "segmentation":
      return (
        <SegmentationConfig
          config={config}
          onConfigChange={onConfigChange}
          advancedOpen={advancedOpen}
          setAdvancedOpen={setAdvancedOpen}
        />
      );
    case "foreach":
      return (
        <ForEachConfig
          config={config}
          onConfigChange={onConfigChange}
          advancedOpen={advancedOpen}
          setAdvancedOpen={setAdvancedOpen}
        />
      );
    case "collect":
      return (
        <CollectConfig
          config={config}
          onConfigChange={onConfigChange}
          advancedOpen={advancedOpen}
          setAdvancedOpen={setAdvancedOpen}
        />
      );
    case "map":
      return (
        <MapConfig
          config={config}
          onConfigChange={onConfigChange}
          advancedOpen={advancedOpen}
          setAdvancedOpen={setAdvancedOpen}
        />
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

  // Check if the selected model is an open-vocabulary model (supports text prompts)
  const isOpenVocabulary = useMemo(() => {
    const modelId = (config.model_id as string)?.toLowerCase() ?? "";
    return (
      modelId.includes("grounding") ||
      modelId.includes("dino") ||
      modelId.includes("owl") ||
      modelId.includes("glip") ||
      modelId.includes("yolo-world") ||
      modelId.includes("florence")
    );
  }, [config.model_id]);

  // Handle model selection - sets both model_id and model_source in single call
  const handleModelSelect = (modelId: string) => {
    const model = models?.find((m) => m.id === modelId);
    // Use batch update to avoid stale state issues
    (onConfigChange as (updates: Record<string, unknown>) => void)({
      model_id: modelId,
      model_source: model?.source ?? "pretrained",
    });
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

      {/* Text Prompt - for open-vocabulary models like Grounding DINO, OWL-ViT */}
      {isOpenVocabulary && (
        <div className="space-y-2">
          <Label className="text-xs">Text Prompt (Open-Vocabulary)</Label>
          <ParamInput
            value={config.text_prompt ?? ""}
            onChange={(v) => onConfigChange("text_prompt", v)}
            placeholder="e.g., person. car. bottle. or {{ params.prompt }}"
          />
          <p className="text-xs text-muted-foreground">
            Describe what to detect. Use periods or commas to separate classes.
            {(config.model_id as string)?.toLowerCase().includes("grounding") && (
              <span className="block mt-1 text-blue-500">
                Grounding DINO: Use &quot;.&quot; to separate classes (e.g., &quot;person. car.&quot;)
              </span>
            )}
          </p>
        </div>
      )}

      {/* Box/Text Thresholds - for open-vocabulary models */}
      {isOpenVocabulary && (
        <div className="space-y-3 pl-4 border-l-2 border-blue-200 dark:border-blue-900">
          <div className="space-y-2">
            <Label className="text-xs">Box Threshold</Label>
            <ParamSlider
              value={config.box_threshold ?? 0.35}
              onChange={(v) => onConfigChange("box_threshold", v)}
              min={0}
              max={1}
              step={0.05}
              formatValue={(v) => v.toFixed(2)}
            />
            <p className="text-xs text-muted-foreground">
              Minimum confidence for detected boxes
            </p>
          </div>
          <div className="space-y-2">
            <Label className="text-xs">Text Threshold</Label>
            <ParamSlider
              value={config.text_threshold ?? 0.25}
              onChange={(v) => onConfigChange("text_threshold", v)}
              min={0}
              max={1}
              step={0.05}
              formatValue={(v) => v.toFixed(2)}
            />
            <p className="text-xs text-muted-foreground">
              Minimum text-box matching score
            </p>
          </div>
        </div>
      )}

      {/* Input Resolution */}
      <div className="space-y-2">
        <Label className="text-xs">Input Resolution</Label>
        <Select
          value={String(config.input_size ?? 640)}
          onValueChange={(v) => onConfigChange("input_size", parseInt(v))}
        >
          <SelectTrigger className="h-8">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="320">320px (Fastest)</SelectItem>
            <SelectItem value="480">480px (Fast)</SelectItem>
            <SelectItem value="640">640px (Balanced)</SelectItem>
            <SelectItem value="800">800px (Quality)</SelectItem>
            <SelectItem value="1024">1024px (High Quality)</SelectItem>
            <SelectItem value="1280">1280px (Best Quality)</SelectItem>
          </SelectContent>
        </Select>
        <p className="text-xs text-muted-foreground">
          Higher resolution = better small object detection
        </p>
      </div>

      {/* Tiled Inference (SAHI) */}
      <div className="flex items-center justify-between">
        <div>
          <Label className="text-xs">Tiled Inference (SAHI)</Label>
          <p className="text-xs text-muted-foreground">
            Better small object detection
          </p>
        </div>
        <Switch
          checked={(config.tiled_inference as boolean) ?? false}
          onCheckedChange={(v) => onConfigChange("tiled_inference", v)}
        />
      </div>

      {/* Tile Settings - only show when tiled inference is enabled */}
      {Boolean(config.tiled_inference) && (
        <div className="space-y-3 pl-4 border-l-2 border-muted">
          <div className="space-y-2">
            <Label className="text-xs">Tile Size</Label>
            <Select
              value={String(config.tile_size ?? 640)}
              onValueChange={(v) => onConfigChange("tile_size", parseInt(v))}
            >
              <SelectTrigger className="h-8">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="256">256px</SelectItem>
                <SelectItem value="320">320px</SelectItem>
                <SelectItem value="512">512px</SelectItem>
                <SelectItem value="640">640px</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-2">
            <Label className="text-xs">Tile Overlap</Label>
            <ParamSlider
              value={config.tile_overlap ?? 0.2}
              onChange={(v) => onConfigChange("tile_overlap", v)}
              min={0.1}
              max={0.5}
              step={0.05}
              formatValue={(v) => `${(v * 100).toFixed(0)}%`}
            />
          </div>
        </div>
      )}

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

          {/* Draw Boxes on Output */}
          <div className="flex items-center justify-between">
            <div>
              <Label className="text-xs">Draw Boxes on Output</Label>
              <p className="text-xs text-muted-foreground">
                Visualize detections on image
              </p>
            </div>
            <Switch
              checked={(config.draw_boxes as boolean) ?? false}
              onCheckedChange={(v) => onConfigChange("draw_boxes", v)}
            />
          </div>

          {/* Draw settings - only show when draw_boxes is enabled */}
          {Boolean(config.draw_boxes) && (
            <div className="space-y-3 pl-4 border-l-2 border-muted">
              <div className="flex items-center justify-between">
                <Label className="text-xs">Show Labels</Label>
                <Switch
                  checked={(config.draw_labels as boolean) ?? true}
                  onCheckedChange={(v) => onConfigChange("draw_labels", v)}
                />
              </div>
              <div className="flex items-center justify-between">
                <Label className="text-xs">Show Confidence</Label>
                <Switch
                  checked={(config.draw_confidence as boolean) ?? true}
                  onCheckedChange={(v) => onConfigChange("draw_confidence", v)}
                />
              </div>
              <div className="space-y-2">
                <Label className="text-xs">Box Thickness</Label>
                <ParamSlider
                  value={config.box_thickness ?? 2}
                  onChange={(v) => onConfigChange("box_thickness", v)}
                  min={1}
                  max={5}
                  step={1}
                  formatValue={(v) => `${v}px`}
                />
              </div>
            </div>
          )}

          {/* Class Name Mapping */}
          <div className="space-y-2">
            <Label className="text-xs">Class Name Mapping</Label>
            <Input
              type="text"
              value={(config.class_mapping as string) ?? ""}
              onChange={(e) => onConfigChange("class_mapping", e.target.value || null)}
              placeholder="person:müşteri, car:araç"
              className="h-8"
            />
            <p className="text-xs text-muted-foreground">
              Rename classes: original:new, separated by comma
            </p>
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

  // Handle model selection - batch update to avoid stale state
  const handleModelSelect = (modelId: string) => {
    const model = models?.find((m) => m.id === modelId);
    (onConfigChange as (updates: Record<string, unknown>) => void)({
      model_id: modelId,
      model_source: model?.source ?? "trained",
    });
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

      {/* Input Resolution */}
      <div className="space-y-2">
        <Label className="text-xs">Input Resolution</Label>
        <Select
          value={String(config.input_size ?? 224)}
          onValueChange={(v) => onConfigChange("input_size", parseInt(v))}
        >
          <SelectTrigger className="h-8">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="224">224px (Standard)</SelectItem>
            <SelectItem value="256">256px (Better)</SelectItem>
            <SelectItem value="384">384px (High Quality)</SelectItem>
            <SelectItem value="448">448px (Best Quality)</SelectItem>
            <SelectItem value="518">518px (DINOv2 Native)</SelectItem>
          </SelectContent>
        </Select>
        <p className="text-xs text-muted-foreground">
          Higher resolution improves fine-grained accuracy
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

      {/* Classification Mode */}
      <div className="space-y-2">
        <Label className="text-xs">Classification Mode</Label>
        <Select
          value={(config.mode as string) ?? "single_label"}
          onValueChange={(v) => onConfigChange("mode", v)}
        >
          <SelectTrigger className="h-8">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="single_label">Single Label (Softmax)</SelectItem>
            <SelectItem value="multi_label">Multi Label (Sigmoid)</SelectItem>
          </SelectContent>
        </Select>
        <p className="text-xs text-muted-foreground">
          Single: one class per image. Multi: multiple classes possible
        </p>
      </div>

      {/* Decision Mode */}
      <div className="space-y-2">
        <Label className="text-xs">Decision Mode</Label>
        <Select
          value={(config.decision_mode as string) ?? "label_only"}
          onValueChange={(v) => onConfigChange("decision_mode", v)}
        >
          <SelectTrigger className="h-8">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="label_only">Label Only</SelectItem>
            <SelectItem value="with_confidence">Label + Confidence</SelectItem>
            <SelectItem value="binary_decision">Binary Decision (pass/fail)</SelectItem>
            <SelectItem value="uncertain_aware">Uncertain Aware (3-way)</SelectItem>
          </SelectContent>
        </Select>
        <p className="text-xs text-muted-foreground">
          {(config.decision_mode as string) === "binary_decision"
            ? "Output: true/false based on threshold"
            : (config.decision_mode as string) === "uncertain_aware"
            ? "Output: label, uncertain, or rejected"
            : "Standard label output"}
        </p>
      </div>

      {/* Uncertainty Settings - only show when uncertain_aware */}
      {(config.decision_mode as string) === "uncertain_aware" && (
        <div className="space-y-3 pl-4 border-l-2 border-muted">
          <div className="space-y-2">
            <Label className="text-xs">Uncertainty Threshold</Label>
            <ParamSlider
              value={config.uncertainty_threshold ?? 0.7}
              onChange={(v) => onConfigChange("uncertainty_threshold", v)}
              min={0.5}
              max={0.95}
              step={0.05}
              formatValue={(v) => v.toFixed(2)}
            />
            <p className="text-xs text-muted-foreground">
              Below this → "uncertain"
            </p>
          </div>
          <div className="space-y-2">
            <Label className="text-xs">Reject Threshold</Label>
            <ParamSlider
              value={config.reject_threshold ?? 0.3}
              onChange={(v) => onConfigChange("reject_threshold", v)}
              min={0.1}
              max={0.5}
              step={0.05}
              formatValue={(v) => v.toFixed(2)}
            />
            <p className="text-xs text-muted-foreground">
              Below this → "rejected"
            </p>
          </div>
        </div>
      )}

      {/* Binary Decision Settings - only show when binary_decision */}
      {(config.decision_mode as string) === "binary_decision" && (
        <div className="space-y-3 pl-4 border-l-2 border-muted">
          <div className="space-y-2">
            <Label className="text-xs">Target Class</Label>
            <Input
              type="text"
              value={(config.target_class as string) ?? ""}
              onChange={(e) => onConfigChange("target_class", e.target.value || null)}
              placeholder="e.g., full, positive, defective"
              className="h-8"
            />
            <p className="text-xs text-muted-foreground">
              Class that triggers "true" output
            </p>
          </div>
          <div className="space-y-2">
            <Label className="text-xs">Decision Threshold</Label>
            <ParamSlider
              value={config.decision_threshold ?? 0.5}
              onChange={(v) => onConfigChange("decision_threshold", v)}
              min={0.1}
              max={0.9}
              step={0.05}
              formatValue={(v) => v.toFixed(2)}
            />
          </div>
        </div>
      )}

      {/* Test Time Augmentation (TTA) */}
      <div className="flex items-center justify-between">
        <div>
          <Label className="text-xs">Test Time Augmentation</Label>
          <p className="text-xs text-muted-foreground">
            Better accuracy, slower inference
          </p>
        </div>
        <Switch
          checked={(config.tta_enabled as boolean) ?? false}
          onCheckedChange={(v) => onConfigChange("tta_enabled", v)}
        />
      </div>

      {/* TTA Settings - only show when enabled */}
      {Boolean(config.tta_enabled) && (
        <div className="space-y-3 pl-4 border-l-2 border-muted">
          <div className="flex items-center justify-between">
            <Label className="text-xs">Horizontal Flip</Label>
            <Switch
              checked={(config.tta_hflip as boolean) ?? true}
              onCheckedChange={(v) => onConfigChange("tta_hflip", v)}
            />
          </div>
          <div className="flex items-center justify-between">
            <Label className="text-xs">5-Crop</Label>
            <Switch
              checked={(config.tta_five_crop as boolean) ?? false}
              onCheckedChange={(v) => onConfigChange("tta_five_crop", v)}
            />
          </div>
          <div className="space-y-2">
            <Label className="text-xs">Merge Strategy</Label>
            <Select
              value={(config.tta_merge as string) ?? "mean"}
              onValueChange={(v) => onConfigChange("tta_merge", v)}
            >
              <SelectTrigger className="h-8">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="mean">Mean</SelectItem>
                <SelectItem value="max">Max</SelectItem>
                <SelectItem value="vote">Majority Vote</SelectItem>
              </SelectContent>
            </Select>
          </div>
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

          {/* Multi-crop Settings - only show when enabled */}
          {Boolean((config.multi_crop as Record<string, unknown>)?.enabled) && (
            <div className="space-y-3 pl-4 border-l-2 border-muted">
              <div className="space-y-2">
                <Label className="text-xs">Scales</Label>
                <Input
                  type="text"
                  value={
                    Array.isArray((config.multi_crop as Record<string, unknown>)?.scales)
                      ? ((config.multi_crop as Record<string, unknown>)?.scales as number[]).join(", ")
                      : "1.0, 0.875, 0.75"
                  }
                  onChange={(e) => {
                    const scales = e.target.value
                      .split(",")
                      .map((s) => parseFloat(s.trim()))
                      .filter((n) => !isNaN(n));
                    onConfigChange("multi_crop", {
                      ...(config.multi_crop as Record<string, unknown>),
                      scales: scales.length ? scales : [1.0, 0.875, 0.75],
                    });
                  }}
                  placeholder="1.0, 0.875, 0.75"
                  className="h-8"
                />
              </div>
              <div className="space-y-2">
                <Label className="text-xs">Merge Mode</Label>
                <Select
                  value={((config.multi_crop as Record<string, unknown>)?.merge_mode as string) ?? "mean"}
                  onValueChange={(v) =>
                    onConfigChange("multi_crop", {
                      ...(config.multi_crop as Record<string, unknown>),
                      merge_mode: v,
                    })
                  }
                >
                  <SelectTrigger className="h-8">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="mean">Mean</SelectItem>
                    <SelectItem value="max">Max</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          )}

          {/* Class Name Mapping */}
          <div className="space-y-2">
            <Label className="text-xs">Class Name Mapping</Label>
            <Input
              type="text"
              value={(config.class_mapping as string) ?? ""}
              onChange={(e) => onConfigChange("class_mapping", e.target.value || null)}
              placeholder="0:empty, 1:full"
              className="h-8"
            />
            <p className="text-xs text-muted-foreground">
              Map class indices or names: original:new
            </p>
          </div>

          {/* Include Probabilities */}
          <div className="flex items-center justify-between">
            <div>
              <Label className="text-xs">Include All Probabilities</Label>
              <p className="text-xs text-muted-foreground">
                Output full probability distribution
              </p>
            </div>
            <Switch
              checked={(config.include_probs as boolean) ?? false}
              onCheckedChange={(v) => onConfigChange("include_probs", v)}
            />
          </div>

          {/* Include Second Best */}
          <div className="flex items-center justify-between">
            <div>
              <Label className="text-xs">Include Second Best</Label>
              <p className="text-xs text-muted-foreground">
                Output runner-up prediction
              </p>
            </div>
            <Switch
              checked={(config.include_second_best as boolean) ?? false}
              onCheckedChange={(v) => onConfigChange("include_second_best", v)}
            />
          </div>

          {/* Include Entropy */}
          <div className="flex items-center justify-between">
            <div>
              <Label className="text-xs">Include Entropy</Label>
              <p className="text-xs text-muted-foreground">
                Output prediction uncertainty score
              </p>
            </div>
            <Switch
              checked={(config.include_entropy as boolean) ?? false}
              onCheckedChange={(v) => onConfigChange("include_entropy", v)}
            />
          </div>

          {/* Output Format */}
          <div className="space-y-2">
            <Label className="text-xs">Output Format</Label>
            <Select
              value={(config.output_format as string) ?? "standard"}
              onValueChange={(v) => onConfigChange("output_format", v)}
            >
              <SelectTrigger className="h-8">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="standard">Standard (label, confidence)</SelectItem>
                <SelectItem value="detailed">Detailed (all metadata)</SelectItem>
                <SelectItem value="minimal">Minimal (label only)</SelectItem>
                <SelectItem value="decision">Decision (boolean + reason)</SelectItem>
              </SelectContent>
            </Select>
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

  // Handle model selection - batch update to avoid stale state
  const handleModelSelect = (modelId: string) => {
    const model = models?.find((m) => m.id === modelId);
    (onConfigChange as (updates: Record<string, unknown>) => void)({
      model_id: modelId,
      model_source: model?.source ?? "pretrained",
    });
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

      {/* Input Resolution */}
      <div className="space-y-2">
        <Label className="text-xs">Input Resolution</Label>
        <Select
          value={String(config.input_size ?? 224)}
          onValueChange={(v) => onConfigChange("input_size", parseInt(v))}
        >
          <SelectTrigger className="h-8">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="224">224px (Standard)</SelectItem>
            <SelectItem value="336">336px (CLIP Large)</SelectItem>
            <SelectItem value="384">384px (ViT-L)</SelectItem>
            <SelectItem value="518">518px (DINOv2)</SelectItem>
          </SelectContent>
        </Select>
        <p className="text-xs text-muted-foreground">
          Higher resolution captures more detail but uses more memory
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

      {/* Similarity Metric */}
      <div className="space-y-2">
        <Label className="text-xs">Similarity Metric</Label>
        <Select
          value={(config.similarity_metric as string) ?? "cosine"}
          onValueChange={(v) => onConfigChange("similarity_metric", v)}
        >
          <SelectTrigger className="h-8">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="cosine">Cosine Similarity (Recommended)</SelectItem>
            <SelectItem value="euclidean">Euclidean Distance</SelectItem>
            <SelectItem value="dot">Dot Product</SelectItem>
          </SelectContent>
        </Select>
        <p className="text-xs text-muted-foreground">
          Cosine works best with normalized embeddings
        </p>
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
          {/* Multi-scale Embedding */}
          <div className="flex items-center justify-between">
            <div>
              <Label className="text-xs">Multi-scale Embedding</Label>
              <p className="text-xs text-muted-foreground">
                Extract at multiple scales and concatenate
              </p>
            </div>
            <Switch
              checked={(config.multi_scale as boolean) ?? false}
              onCheckedChange={(v) => onConfigChange("multi_scale", v)}
            />
          </div>

          {/* Multi-scale Settings */}
          {Boolean(config.multi_scale) && (
            <div className="space-y-3 pl-4 border-l-2 border-muted">
              <div className="space-y-2">
                <Label className="text-xs">Scales</Label>
                <Select
                  value={(config.multi_scale_factors as string) ?? "1.0,0.75,0.5"}
                  onValueChange={(v) => onConfigChange("multi_scale_factors", v)}
                >
                  <SelectTrigger className="h-8">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="1.0,0.75">2 scales (1.0x, 0.75x)</SelectItem>
                    <SelectItem value="1.0,0.75,0.5">3 scales (1.0x, 0.75x, 0.5x)</SelectItem>
                    <SelectItem value="1.25,1.0,0.75,0.5">4 scales (1.25x - 0.5x)</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label className="text-xs">Aggregation Method</Label>
                <Select
                  value={(config.multi_scale_agg as string) ?? "concat"}
                  onValueChange={(v) => onConfigChange("multi_scale_agg", v)}
                >
                  <SelectTrigger className="h-8">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="concat">Concatenate (larger dim)</SelectItem>
                    <SelectItem value="mean">Average (same dim)</SelectItem>
                    <SelectItem value="max">Max Pooling (same dim)</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          )}

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

          {/* PCA Dimension Reduction */}
          <div className="flex items-center justify-between">
            <div>
              <Label className="text-xs">PCA Reduction</Label>
              <p className="text-xs text-muted-foreground">
                Reduce embedding dimension with PCA
              </p>
            </div>
            <Switch
              checked={(config.pca_enabled as boolean) ?? false}
              onCheckedChange={(v) => onConfigChange("pca_enabled", v)}
            />
          </div>

          {/* PCA Settings */}
          {Boolean(config.pca_enabled) && (
            <div className="space-y-3 pl-4 border-l-2 border-muted">
              <div className="space-y-2">
                <Label className="text-xs">Target Dimension</Label>
                <Select
                  value={String(config.pca_dim ?? 256)}
                  onValueChange={(v) => onConfigChange("pca_dim", parseInt(v))}
                >
                  <SelectTrigger className="h-8">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="64">64 dims (fast search)</SelectItem>
                    <SelectItem value="128">128 dims</SelectItem>
                    <SelectItem value="256">256 dims (balanced)</SelectItem>
                    <SelectItem value="512">512 dims (high quality)</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="flex items-center justify-between">
                <div>
                  <Label className="text-xs">Whiten</Label>
                  <p className="text-xs text-muted-foreground">
                    Apply whitening transformation
                  </p>
                </div>
                <Switch
                  checked={(config.pca_whiten as boolean) ?? false}
                  onCheckedChange={(v) => onConfigChange("pca_whiten", v)}
                />
              </div>
            </div>
          )}

          {/* Output Format */}
          <div className="space-y-2">
            <Label className="text-xs">Output Format</Label>
            <Select
              value={(config.output_format as string) ?? "vector"}
              onValueChange={(v) => onConfigChange("output_format", v)}
            >
              <SelectTrigger className="h-8">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="vector">Vector Only</SelectItem>
                <SelectItem value="vector_with_meta">Vector + Metadata</SelectItem>
                <SelectItem value="base64">Base64 Encoded</SelectItem>
              </SelectContent>
            </Select>
            <p className="text-xs text-muted-foreground">
              How to format the embedding output
            </p>
          </div>

          {/* Include Metadata */}
          {(config.output_format as string) === "vector_with_meta" && (
            <div className="space-y-3 pl-4 border-l-2 border-muted">
              <div className="flex items-center justify-between">
                <Label className="text-xs">Include Model Info</Label>
                <Switch
                  checked={(config.include_model_info as boolean) ?? true}
                  onCheckedChange={(v) => onConfigChange("include_model_info", v)}
                />
              </div>
              <div className="flex items-center justify-between">
                <Label className="text-xs">Include Processing Time</Label>
                <Switch
                  checked={(config.include_timing as boolean) ?? false}
                  onCheckedChange={(v) => onConfigChange("include_timing", v)}
                />
              </div>
            </div>
          )}
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
          value={config.top_k ?? 10}
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
        <p className="text-xs text-muted-foreground">
          Minimum score to include in results
        </p>
      </div>

      {/* Distance Metric */}
      <div className="space-y-2">
        <Label className="text-xs">Distance Metric</Label>
        <Select
          value={(config.distance_metric as string) ?? "cosine"}
          onValueChange={(v) => onConfigChange("distance_metric", v)}
        >
          <SelectTrigger className="h-8">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="cosine">Cosine Similarity (Recommended)</SelectItem>
            <SelectItem value="euclidean">Euclidean Distance (L2)</SelectItem>
            <SelectItem value="dot">Dot Product (Inner Product)</SelectItem>
            <SelectItem value="manhattan">Manhattan Distance (L1)</SelectItem>
          </SelectContent>
        </Select>
        <p className="text-xs text-muted-foreground">
          Must match the metric used when indexing
        </p>
      </div>

      {/* Search Mode */}
      <div className="space-y-2">
        <Label className="text-xs">Search Mode</Label>
        <Select
          value={(config.search_mode as string) ?? "approximate"}
          onValueChange={(v) => onConfigChange("search_mode", v)}
        >
          <SelectTrigger className="h-8">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="approximate">Approximate (ANN - Fast)</SelectItem>
            <SelectItem value="exact">Exact (Brute Force - Accurate)</SelectItem>
            <SelectItem value="hybrid">Hybrid (Dense + Sparse/BM25)</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* Hybrid Search Settings */}
      {(config.search_mode as string) === "hybrid" && (
        <div className="space-y-3 pl-4 border-l-2 border-muted">
          <div className="space-y-2">
            <Label className="text-xs">Dense Weight</Label>
            <ParamSlider
              value={config.dense_weight ?? 0.7}
              onChange={(v) => onConfigChange("dense_weight", v)}
              min={0}
              max={1}
              step={0.1}
              formatValue={(v) => `${Math.round(v * 100)}%`}
            />
            <p className="text-xs text-muted-foreground">
              Sparse weight: {Math.round((1 - (config.dense_weight as number ?? 0.7)) * 100)}%
            </p>
          </div>
        </div>
      )}

      {/* Score Normalization */}
      <div className="space-y-2">
        <Label className="text-xs">Score Normalization</Label>
        <Select
          value={(config.score_normalization as string) ?? "none"}
          onValueChange={(v) => onConfigChange("score_normalization", v)}
        >
          <SelectTrigger className="h-8">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="none">None (Raw Scores)</SelectItem>
            <SelectItem value="minmax">Min-Max (0-1 range)</SelectItem>
            <SelectItem value="softmax">Softmax (Probabilities)</SelectItem>
            <SelectItem value="zscore">Z-Score (Standard deviation)</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* Multi-Query Fusion */}
      <div className="space-y-2">
        <Label className="text-xs">Multi-Query Fusion</Label>
        <Select
          value={(config.fusion_method as string) ?? "none"}
          onValueChange={(v) => onConfigChange("fusion_method", v)}
        >
          <SelectTrigger className="h-8">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="none">None (Single Query)</SelectItem>
            <SelectItem value="rrf">RRF (Reciprocal Rank Fusion)</SelectItem>
            <SelectItem value="avg">Average Scores</SelectItem>
            <SelectItem value="max">Max Score</SelectItem>
            <SelectItem value="weighted">Weighted Average</SelectItem>
          </SelectContent>
        </Select>
        <p className="text-xs text-muted-foreground">
          How to combine results from multiple embeddings
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
          {/* Re-ranking */}
          <div className="flex items-center justify-between">
            <div>
              <Label className="text-xs">Enable Re-ranking</Label>
              <p className="text-xs text-muted-foreground">
                Re-rank top results with cross-encoder
              </p>
            </div>
            <Switch
              checked={(config.rerank_enabled as boolean) ?? false}
              onCheckedChange={(v) => onConfigChange("rerank_enabled", v)}
            />
          </div>

          {/* Re-ranking Settings */}
          {Boolean(config.rerank_enabled) && (
            <div className="space-y-3 pl-4 border-l-2 border-muted">
              <div className="space-y-2">
                <Label className="text-xs">Re-rank Top N</Label>
                <Select
                  value={String(config.rerank_top_n ?? 50)}
                  onValueChange={(v) => onConfigChange("rerank_top_n", parseInt(v))}
                >
                  <SelectTrigger className="h-8">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="20">Top 20</SelectItem>
                    <SelectItem value="50">Top 50</SelectItem>
                    <SelectItem value="100">Top 100</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label className="text-xs">Re-ranker Model</Label>
                <Select
                  value={(config.reranker_model as string) ?? "cross-encoder"}
                  onValueChange={(v) => onConfigChange("reranker_model", v)}
                >
                  <SelectTrigger className="h-8">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="cross-encoder">Cross-Encoder (Accurate)</SelectItem>
                    <SelectItem value="colbert">ColBERT (Fast)</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          )}

          {/* Deduplication */}
          <div className="flex items-center justify-between">
            <div>
              <Label className="text-xs">Deduplication</Label>
              <p className="text-xs text-muted-foreground">
                Remove duplicate items from results
              </p>
            </div>
            <Switch
              checked={(config.dedupe_enabled as boolean) ?? false}
              onCheckedChange={(v) => onConfigChange("dedupe_enabled", v)}
            />
          </div>

          {/* Dedup Settings */}
          {Boolean(config.dedupe_enabled) && (
            <div className="space-y-3 pl-4 border-l-2 border-muted">
              <div className="space-y-2">
                <Label className="text-xs">Dedup Field</Label>
                <Input
                  type="text"
                  value={(config.dedupe_field as string) ?? ""}
                  onChange={(e) => onConfigChange("dedupe_field", e.target.value)}
                  placeholder="e.g., product_id, sku"
                  className="h-8"
                />
              </div>
              <div className="space-y-2">
                <Label className="text-xs">Dedup Strategy</Label>
                <Select
                  value={(config.dedupe_strategy as string) ?? "best"}
                  onValueChange={(v) => onConfigChange("dedupe_strategy", v)}
                >
                  <SelectTrigger className="h-8">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="best">Keep Best Score</SelectItem>
                    <SelectItem value="first">Keep First Found</SelectItem>
                    <SelectItem value="avg">Average Scores</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          )}

          {/* Group By */}
          <div className="space-y-2">
            <Label className="text-xs">Group By Field</Label>
            <Input
              type="text"
              value={(config.group_by as string) ?? ""}
              onChange={(e) => onConfigChange("group_by", e.target.value || null)}
              placeholder="e.g., category, brand"
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

          {/* Metadata Filter */}
          <div className="space-y-2">
            <Label className="text-xs">Metadata Filter (JSON)</Label>
            <Input
              type="text"
              value={(config.metadata_filter as string) ?? ""}
              onChange={(e) => onConfigChange("metadata_filter", e.target.value)}
              placeholder='e.g., {"category": "electronics"}'
              className="h-8 font-mono text-xs"
            />
            <p className="text-xs text-muted-foreground">
              Filter by payload fields before search
            </p>
          </div>

          {/* Fallback Strategy */}
          <div className="space-y-2">
            <Label className="text-xs">No Results Fallback</Label>
            <Select
              value={(config.fallback_strategy as string) ?? "none"}
              onValueChange={(v) => onConfigChange("fallback_strategy", v)}
            >
              <SelectTrigger className="h-8">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="none">None (Return Empty)</SelectItem>
                <SelectItem value="lower_threshold">Lower Threshold</SelectItem>
                <SelectItem value="expand_k">Expand K</SelectItem>
                <SelectItem value="remove_filter">Remove Filter</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Return Format */}
          <div className="space-y-2">
            <Label className="text-xs">Return Format</Label>
            <Select
              value={(config.return_format as string) ?? "full"}
              onValueChange={(v) => onConfigChange("return_format", v)}
            >
              <SelectTrigger className="h-8">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="full">Full (scores + payloads + vectors)</SelectItem>
                <SelectItem value="standard">Standard (scores + payloads)</SelectItem>
                <SelectItem value="ids_only">IDs Only (minimal)</SelectItem>
                <SelectItem value="scores_only">Scores Only</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Include Options */}
          <div className="space-y-3 pt-2 border-t">
            <Label className="text-xs font-medium">Include in Response</Label>

            <div className="flex items-center justify-between">
              <Label className="text-xs">Include Payload</Label>
              <Switch
                checked={(config.with_payload as boolean) ?? true}
                onCheckedChange={(v) => onConfigChange("with_payload", v)}
              />
            </div>

            <div className="flex items-center justify-between">
              <Label className="text-xs">Include Vectors</Label>
              <Switch
                checked={(config.with_vectors as boolean) ?? false}
                onCheckedChange={(v) => onConfigChange("with_vectors", v)}
              />
            </div>

            <div className="flex items-center justify-between">
              <Label className="text-xs">Include Search Timing</Label>
              <Switch
                checked={(config.include_timing as boolean) ?? false}
                onCheckedChange={(v) => onConfigChange("include_timing", v)}
              />
            </div>
          </div>

          {/* Batch Settings */}
          <div className="space-y-3 pt-2 border-t">
            <Label className="text-xs font-medium">Batch Processing</Label>

            <div className="flex items-center justify-between">
              <div>
                <Label className="text-xs">Parallel Queries</Label>
                <p className="text-xs text-muted-foreground">
                  Process multiple queries in parallel
                </p>
              </div>
              <Switch
                checked={(config.parallel_queries as boolean) ?? true}
                onCheckedChange={(v) => onConfigChange("parallel_queries", v)}
              />
            </div>

            <div className="space-y-2">
              <Label className="text-xs">Max Concurrent</Label>
              <Select
                value={String(config.max_concurrent ?? 10)}
                onValueChange={(v) => onConfigChange("max_concurrent", parseInt(v))}
              >
                <SelectTrigger className="h-8">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="5">5 queries</SelectItem>
                  <SelectItem value="10">10 queries</SelectItem>
                  <SelectItem value="20">20 queries</SelectItem>
                  <SelectItem value="50">50 queries</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        </CollapsibleContent>
      </Collapsible>
    </div>
  );
}

// ============================================================================
// Segmentation Config (SAM2, SAM2.1, SAM3)
// ============================================================================
function SegmentationConfig({
  config,
  onConfigChange,
  advancedOpen,
  setAdvancedOpen,
}: ConfigSectionProps) {
  // Fetch all segmentation models from API
  const { data: models, isLoading } = useWorkflowModels("segmentation");

  // Group models by SAM version
  const groupedModels = useMemo(() => {
    if (!models) return { sam3: [], sam21: [], sam2: [] };
    return {
      sam3: models.filter((m) => m.id.startsWith("sam3")),
      sam21: models.filter((m) => m.id.startsWith("sam2.1-")),
      sam2: models.filter((m) => m.id.startsWith("sam2-")),
    };
  }, [models]);

  // Find selected model for display
  const selectedModel = useMemo(() => {
    const modelId = config.model_id as string;
    if (!modelId || !models) return undefined;
    return models.find((m) => m.id === modelId);
  }, [config.model_id, models]);

  // Handle model selection - batch update to avoid stale state
  const handleModelSelect = (modelId: string) => {
    const model = models?.find((m) => m.id === modelId);
    (onConfigChange as (updates: Record<string, unknown>) => void)({
      model_id: modelId,
      model_source: model?.source ?? "pretrained",
    });
  };

  // Check if SAM3 model is selected (supports text prompts)
  const isSAM3 = (config.model_id as string)?.startsWith("sam3");

  return (
    <div className="space-y-4">
      {/* Model Selection */}
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
                    <Sparkles className="h-3 w-3 text-purple-500" />
                    <span className="truncate">{selectedModel.name}</span>
                    {selectedModel.input_size && (
                      <span className="text-xs text-muted-foreground">{selectedModel.input_size}px</span>
                    )}
                  </div>
                )}
              </SelectValue>
            </SelectTrigger>
            <SelectContent>
              {/* SAM3 Models (Text-prompted) */}
              {groupedModels.sam3.length > 0 && (
                <SelectGroup>
                  <SelectLabel className="flex items-center gap-2 text-xs">
                    <Sparkles className="h-3 w-3 text-green-500" />
                    SAM3 (Text Prompts)
                  </SelectLabel>
                  {groupedModels.sam3.map((model) => (
                    <SelectItem key={model.id} value={model.id}>
                      <div className="flex items-center gap-2">
                        <span>{model.name}</span>
                        {model.input_size && (
                          <span className="text-xs text-muted-foreground">{model.input_size}px</span>
                        )}
                      </div>
                    </SelectItem>
                  ))}
                </SelectGroup>
              )}
              {/* SAM 2.1 Models */}
              {groupedModels.sam21.length > 0 && (
                <SelectGroup>
                  <SelectLabel className="flex items-center gap-2 text-xs">
                    <Sparkles className="h-3 w-3 text-blue-500" />
                    SAM 2.1 (Point/Box)
                  </SelectLabel>
                  {groupedModels.sam21.map((model) => (
                    <SelectItem key={model.id} value={model.id}>
                      <div className="flex items-center gap-2">
                        <span>{model.name}</span>
                        {model.input_size && (
                          <span className="text-xs text-muted-foreground">{model.input_size}px</span>
                        )}
                      </div>
                    </SelectItem>
                  ))}
                </SelectGroup>
              )}
              {/* SAM2 Models */}
              {groupedModels.sam2.length > 0 && (
                <SelectGroup>
                  <SelectLabel className="flex items-center gap-2 text-xs">
                    <Sparkles className="h-3 w-3 text-purple-500" />
                    SAM2 (Legacy)
                  </SelectLabel>
                  {groupedModels.sam2.map((model) => (
                    <SelectItem key={model.id} value={model.id}>
                      <div className="flex items-center gap-2">
                        <span>{model.name}</span>
                        {model.input_size && (
                          <span className="text-xs text-muted-foreground">{model.input_size}px</span>
                        )}
                      </div>
                    </SelectItem>
                  ))}
                </SelectGroup>
              )}
            </SelectContent>
          </Select>
        )}
        <p className="text-xs text-muted-foreground">
          {isSAM3
            ? "SAM3 supports text prompts for zero-shot segmentation"
            : "SAM2 uses point/box prompts for interactive segmentation"}
        </p>
      </div>

      {/* Prompt Mode */}
      <div className="space-y-2">
        <Label className="text-xs">Prompt Mode</Label>
        <Select
          value={(config.prompt_mode as string) ?? "auto"}
          onValueChange={(v) => onConfigChange("prompt_mode", v)}
        >
          <SelectTrigger className="h-8">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="auto">Auto (from input connections)</SelectItem>
            <SelectItem value="boxes">Box Prompts (from detections)</SelectItem>
            <SelectItem value="points">Point Prompts (click coordinates)</SelectItem>
            {isSAM3 && <SelectItem value="text">Text Prompt (SAM3 only)</SelectItem>}
            <SelectItem value="grid">Automatic Grid (no prompts)</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* Text Prompt - only for SAM3 */}
      {isSAM3 && (config.prompt_mode as string) === "text" && (
        <div className="space-y-2">
          <Label className="text-xs">Text Prompt</Label>
          <ParamInput
            value={config.text_prompt ?? ""}
            onChange={(v) => onConfigChange("text_prompt", v)}
            placeholder="e.g., the red car, all people, {{ params.target }}"
          />
          <p className="text-xs text-muted-foreground">
            Describe what to segment in natural language
          </p>
        </div>
      )}

      {/* Points Per Side - for grid mode */}
      {(config.prompt_mode as string) === "grid" && (
        <div className="space-y-2">
          <Label className="text-xs">Points Per Side</Label>
          <Select
            value={String(config.points_per_side ?? 32)}
            onValueChange={(v) => onConfigChange("points_per_side", parseInt(v))}
          >
            <SelectTrigger className="h-8">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="16">16 (Faster, fewer masks)</SelectItem>
              <SelectItem value="32">32 (Balanced)</SelectItem>
              <SelectItem value="64">64 (More masks)</SelectItem>
              <SelectItem value="128">128 (Dense, slow)</SelectItem>
            </SelectContent>
          </Select>
        </div>
      )}

      {/* Mask Output Format */}
      <div className="space-y-2">
        <Label className="text-xs">Mask Output Format</Label>
        <Select
          value={(config.mask_format as string) ?? "binary"}
          onValueChange={(v) => onConfigChange("mask_format", v)}
        >
          <SelectTrigger className="h-8">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="binary">Binary Mask (H×W boolean)</SelectItem>
            <SelectItem value="polygon">Polygon (contour points)</SelectItem>
            <SelectItem value="rle">RLE (run-length encoded)</SelectItem>
            <SelectItem value="coco">COCO Format (polygon + RLE)</SelectItem>
          </SelectContent>
        </Select>
        <p className="text-xs text-muted-foreground">
          {(config.mask_format as string) === "rle" || (config.mask_format as string) === "coco"
            ? "Compact format, good for storage"
            : (config.mask_format as string) === "polygon"
            ? "Vector format, good for editing"
            : "Full resolution mask array"}
        </p>
      </div>

      {/* Multi-mask Output */}
      <div className="flex items-center justify-between">
        <div>
          <Label className="text-xs">Multi-mask Output</Label>
          <p className="text-xs text-muted-foreground">
            Return multiple mask candidates per prompt
          </p>
        </div>
        <Switch
          checked={(config.multi_mask as boolean) ?? false}
          onCheckedChange={(v) => onConfigChange("multi_mask", v)}
        />
      </div>

      {/* Multi-mask Settings */}
      {Boolean(config.multi_mask) && (
        <div className="space-y-3 pl-4 border-l-2 border-muted">
          <div className="space-y-2">
            <Label className="text-xs">Masks Per Prompt</Label>
            <Select
              value={String(config.masks_per_prompt ?? 3)}
              onValueChange={(v) => onConfigChange("masks_per_prompt", parseInt(v))}
            >
              <SelectTrigger className="h-8">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="1">1 (Best only)</SelectItem>
                <SelectItem value="3">3 (Default SAM)</SelectItem>
                <SelectItem value="5">5 (More options)</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="flex items-center justify-between">
            <Label className="text-xs">Return Best Only</Label>
            <Switch
              checked={(config.return_best as boolean) ?? true}
              onCheckedChange={(v) => onConfigChange("return_best", v)}
            />
          </div>
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
          {/* Mask Threshold */}
          <div className="space-y-2">
            <Label className="text-xs">Mask Threshold</Label>
            <ParamSlider
              value={config.mask_threshold ?? 0.0}
              onChange={(v) => onConfigChange("mask_threshold", v)}
              min={-1}
              max={1}
              step={0.1}
              formatValue={(v) => v.toFixed(1)}
            />
            <p className="text-xs text-muted-foreground">
              Higher = stricter mask boundaries (default 0.0)
            </p>
          </div>

          {/* Stability Score Threshold */}
          <div className="space-y-2">
            <Label className="text-xs">Stability Score Threshold</Label>
            <ParamSlider
              value={config.stability_score_thresh ?? 0.95}
              onChange={(v) => onConfigChange("stability_score_thresh", v)}
              min={0.5}
              max={1}
              step={0.05}
              formatValue={(v) => v.toFixed(2)}
            />
            <p className="text-xs text-muted-foreground">
              Filter low-quality masks (higher = stricter)
            </p>
          </div>

          {/* Box NMS Threshold */}
          <div className="space-y-2">
            <Label className="text-xs">Box NMS Threshold</Label>
            <ParamSlider
              value={config.box_nms_thresh ?? 0.7}
              onChange={(v) => onConfigChange("box_nms_thresh", v)}
              min={0.1}
              max={1}
              step={0.05}
              formatValue={(v) => v.toFixed(2)}
            />
            <p className="text-xs text-muted-foreground">
              Remove overlapping masks (lower = more filtering)
            </p>
          </div>

          {/* Min Mask Region Area */}
          <div className="space-y-2">
            <Label className="text-xs">Min Mask Area (px²)</Label>
            <ParamInput
              type="number"
              value={config.min_mask_area ?? 0}
              onChange={(v) => onConfigChange("min_mask_area", v)}
              min={0}
              max={10000}
            />
            <p className="text-xs text-muted-foreground">
              Filter tiny masks below this area
            </p>
          </div>

          {/* Mask Refinement */}
          <div className="flex items-center justify-between">
            <div>
              <Label className="text-xs">Mask Refinement</Label>
              <p className="text-xs text-muted-foreground">
                Post-process to smooth edges
              </p>
            </div>
            <Switch
              checked={(config.refine_masks as boolean) ?? false}
              onCheckedChange={(v) => onConfigChange("refine_masks", v)}
            />
          </div>

          {/* Refinement Iterations */}
          {Boolean(config.refine_masks) && (
            <div className="space-y-2 pl-4 border-l-2 border-muted">
              <Label className="text-xs">Refinement Iterations</Label>
              <Select
                value={String(config.refinement_iterations ?? 4)}
                onValueChange={(v) => onConfigChange("refinement_iterations", parseInt(v))}
              >
                <SelectTrigger className="h-8">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="2">2 (Fast)</SelectItem>
                  <SelectItem value="4">4 (Default)</SelectItem>
                  <SelectItem value="8">8 (High quality)</SelectItem>
                </SelectContent>
              </Select>
            </div>
          )}

          {/* Output Options */}
          <div className="space-y-3 pt-2 border-t">
            <Label className="text-xs font-medium">Output Options</Label>

            <div className="flex items-center justify-between">
              <Label className="text-xs">Include Masked Image</Label>
              <Switch
                checked={(config.output_masked_image as boolean) ?? true}
                onCheckedChange={(v) => onConfigChange("output_masked_image", v)}
              />
            </div>

            <div className="flex items-center justify-between">
              <Label className="text-xs">Include Cropped Objects</Label>
              <Switch
                checked={(config.output_crops as boolean) ?? false}
                onCheckedChange={(v) => onConfigChange("output_crops", v)}
              />
            </div>

            <div className="flex items-center justify-between">
              <Label className="text-xs">Include Mask Areas</Label>
              <Switch
                checked={(config.output_areas as boolean) ?? true}
                onCheckedChange={(v) => onConfigChange("output_areas", v)}
              />
            </div>

            <div className="flex items-center justify-between">
              <Label className="text-xs">Include Polygons</Label>
              <Switch
                checked={(config.output_polygons as boolean) ?? false}
                onCheckedChange={(v) => onConfigChange("output_polygons", v)}
              />
            </div>
          </div>

          {/* Visualization */}
          <div className="space-y-3 pt-2 border-t">
            <Label className="text-xs font-medium">Visualization</Label>

            <div className="flex items-center justify-between">
              <div>
                <Label className="text-xs">Draw Masks on Output</Label>
                <p className="text-xs text-muted-foreground">
                  Overlay colored masks on image
                </p>
              </div>
              <Switch
                checked={(config.draw_masks as boolean) ?? true}
                onCheckedChange={(v) => onConfigChange("draw_masks", v)}
              />
            </div>

            {Boolean(config.draw_masks) && (
              <div className="space-y-3 pl-4 border-l-2 border-muted">
                <div className="space-y-2">
                  <Label className="text-xs">Mask Opacity</Label>
                  <ParamSlider
                    value={config.mask_opacity ?? 0.5}
                    onChange={(v) => onConfigChange("mask_opacity", v)}
                    min={0.1}
                    max={1}
                    step={0.1}
                    formatValue={(v) => `${Math.round(v * 100)}%`}
                  />
                </div>
                <div className="flex items-center justify-between">
                  <Label className="text-xs">Draw Contours</Label>
                  <Switch
                    checked={(config.draw_contours as boolean) ?? true}
                    onCheckedChange={(v) => onConfigChange("draw_contours", v)}
                  />
                </div>
                <div className="space-y-2">
                  <Label className="text-xs">Color Mode</Label>
                  <Select
                    value={(config.color_mode as string) ?? "random"}
                    onValueChange={(v) => onConfigChange("color_mode", v)}
                  >
                    <SelectTrigger className="h-8">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="random">Random Colors</SelectItem>
                      <SelectItem value="rainbow">Rainbow Palette</SelectItem>
                      <SelectItem value="category">Category Colors</SelectItem>
                      <SelectItem value="single">Single Color</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            )}
          </div>
        </CollapsibleContent>
      </Collapsible>
    </div>
  );
}

// ============================================================================
// Crop Config
// ============================================================================
function CropConfig({
  config,
  onConfigChange,
  advancedOpen,
  setAdvancedOpen,
}: ConfigSectionProps) {
  return (
    <div className="space-y-4">
      {/* Padding Mode */}
      <div className="space-y-2">
        <Label className="text-xs">Padding Mode</Label>
        <Select
          value={(config.padding_mode as string) ?? "pixel"}
          onValueChange={(v) => onConfigChange("padding_mode", v)}
        >
          <SelectTrigger className="h-8">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="pixel">Fixed Pixels</SelectItem>
            <SelectItem value="percent">Percentage of Box</SelectItem>
            <SelectItem value="none">No Padding</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* Padding Value */}
      {(config.padding_mode as string) !== "none" && (
        <div className="space-y-2">
          <Label className="text-xs">
            Padding {(config.padding_mode as string) === "percent" ? "(%)" : "(px)"}
          </Label>
          <ParamSlider
            value={config.padding ?? ((config.padding_mode as string) === "percent" ? 10 : 0)}
            onChange={(v) => onConfigChange("padding", v)}
            min={0}
            max={(config.padding_mode as string) === "percent" ? 50 : 100}
            step={(config.padding_mode as string) === "percent" ? 5 : 5}
            formatValue={(v) => (config.padding_mode as string) === "percent" ? `${v}%` : `${v}px`}
          />
        </div>
      )}

      {/* Aspect Ratio */}
      <div className="space-y-2">
        <Label className="text-xs">Aspect Ratio</Label>
        <Select
          value={(config.aspect_ratio as string) ?? "original"}
          onValueChange={(v) => onConfigChange("aspect_ratio", v)}
        >
          <SelectTrigger className="h-8">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="original">Original (Keep as-is)</SelectItem>
            <SelectItem value="square">Square (1:1)</SelectItem>
            <SelectItem value="4:3">4:3 Landscape</SelectItem>
            <SelectItem value="3:4">3:4 Portrait</SelectItem>
            <SelectItem value="16:9">16:9 Wide</SelectItem>
            <SelectItem value="custom">Custom Ratio</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* Custom Aspect Ratio */}
      {(config.aspect_ratio as string) === "custom" && (
        <div className="space-y-2 pl-4 border-l-2 border-muted">
          <Label className="text-xs">Custom Ratio (W:H)</Label>
          <div className="flex items-center gap-2">
            <Input
              type="number"
              value={(config.aspect_w as number) ?? 1}
              onChange={(e) => onConfigChange("aspect_w", parseInt(e.target.value) || 1)}
              min={1}
              max={10}
              className="h-8 w-16"
            />
            <span className="text-xs text-muted-foreground">:</span>
            <Input
              type="number"
              value={(config.aspect_h as number) ?? 1}
              onChange={(e) => onConfigChange("aspect_h", parseInt(e.target.value) || 1)}
              min={1}
              max={10}
              className="h-8 w-16"
            />
          </div>
        </div>
      )}

      {/* Output Size */}
      <div className="space-y-2">
        <Label className="text-xs">Output Size</Label>
        <Select
          value={(config.output_size as string) ?? "original"}
          onValueChange={(v) => onConfigChange("output_size", v)}
        >
          <SelectTrigger className="h-8">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="original">Original Size</SelectItem>
            <SelectItem value="64">64px (Thumbnail)</SelectItem>
            <SelectItem value="128">128px</SelectItem>
            <SelectItem value="224">224px (ViT/CLIP)</SelectItem>
            <SelectItem value="256">256px</SelectItem>
            <SelectItem value="384">384px</SelectItem>
            <SelectItem value="512">512px</SelectItem>
            <SelectItem value="custom">Custom Size</SelectItem>
          </SelectContent>
        </Select>
        <p className="text-xs text-muted-foreground">
          Resize all crops to consistent size
        </p>
      </div>

      {/* Custom Output Size */}
      {(config.output_size as string) === "custom" && (
        <div className="space-y-2 pl-4 border-l-2 border-muted">
          <Label className="text-xs">Custom Size (px)</Label>
          <ParamInput
            type="number"
            value={config.custom_size ?? 256}
            onChange={(v) => onConfigChange("custom_size", v)}
            min={32}
            max={1024}
          />
        </div>
      )}

      {/* Min Size Filter */}
      <div className="space-y-2">
        <Label className="text-xs">Minimum Size (px)</Label>
        <ParamInput
          type="number"
          value={config.min_size ?? 32}
          onChange={(v) => onConfigChange("min_size", v)}
          min={1}
          max={512}
        />
        <p className="text-xs text-muted-foreground">
          Skip crops smaller than this
        </p>
      </div>

      {/* Confidence Filter */}
      <div className="flex items-center justify-between">
        <div>
          <Label className="text-xs">Filter by Confidence</Label>
          <p className="text-xs text-muted-foreground">
            Only crop high confidence detections
          </p>
        </div>
        <Switch
          checked={(config.filter_by_confidence as boolean) ?? false}
          onCheckedChange={(v) => onConfigChange("filter_by_confidence", v)}
        />
      </div>

      {/* Confidence Threshold */}
      {Boolean(config.filter_by_confidence) && (
        <div className="space-y-2 pl-4 border-l-2 border-muted">
          <Label className="text-xs">Min Confidence</Label>
          <ParamSlider
            value={config.min_confidence ?? 0.5}
            onChange={(v) => onConfigChange("min_confidence", v)}
            min={0}
            max={1}
            step={0.05}
            formatValue={(v) => v.toFixed(2)}
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
          {/* Padding Fill Mode */}
          <div className="space-y-2">
            <Label className="text-xs">Padding Fill</Label>
            <Select
              value={(config.padding_fill as string) ?? "constant"}
              onValueChange={(v) => onConfigChange("padding_fill", v)}
            >
              <SelectTrigger className="h-8">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="constant">Solid Color</SelectItem>
                <SelectItem value="reflect">Reflect (Mirror)</SelectItem>
                <SelectItem value="replicate">Replicate (Edge)</SelectItem>
                <SelectItem value="wrap">Wrap (Tile)</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Background Color */}
          {(config.padding_fill as string) === "constant" && (
            <div className="space-y-2">
              <Label className="text-xs">Background Color</Label>
              <Select
                value={(config.bg_color as string) ?? "black"}
                onValueChange={(v) => onConfigChange("bg_color", v)}
              >
                <SelectTrigger className="h-8">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="black">Black (#000000)</SelectItem>
                  <SelectItem value="white">White (#FFFFFF)</SelectItem>
                  <SelectItem value="gray">Gray (#808080)</SelectItem>
                  <SelectItem value="transparent">Transparent</SelectItem>
                  <SelectItem value="custom">Custom Color</SelectItem>
                </SelectContent>
              </Select>
            </div>
          )}

          {/* Custom Color */}
          {(config.padding_fill as string) === "constant" && (config.bg_color as string) === "custom" && (
            <div className="space-y-2 pl-4 border-l-2 border-muted">
              <Label className="text-xs">Custom Color (Hex)</Label>
              <Input
                type="text"
                value={(config.custom_color as string) ?? "#000000"}
                onChange={(e) => onConfigChange("custom_color", e.target.value)}
                placeholder="#000000"
                className="h-8 font-mono"
              />
            </div>
          )}

          {/* Interpolation */}
          <div className="space-y-2">
            <Label className="text-xs">Interpolation Method</Label>
            <Select
              value={(config.interpolation as string) ?? "bilinear"}
              onValueChange={(v) => onConfigChange("interpolation", v)}
            >
              <SelectTrigger className="h-8">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="nearest">Nearest (Fast, pixelated)</SelectItem>
                <SelectItem value="bilinear">Bilinear (Balanced)</SelectItem>
                <SelectItem value="bicubic">Bicubic (Smooth)</SelectItem>
                <SelectItem value="lanczos">Lanczos (Best quality)</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Max Crops */}
          <div className="space-y-2">
            <Label className="text-xs">Max Crops</Label>
            <ParamInput
              type="number"
              value={config.max_crops ?? 100}
              onChange={(v) => onConfigChange("max_crops", v)}
              min={1}
              max={1000}
            />
            <p className="text-xs text-muted-foreground">
              Limit number of crops (by confidence order)
            </p>
          </div>

          {/* Class Filter */}
          <div className="space-y-2">
            <Label className="text-xs">Class Filter</Label>
            <Input
              type="text"
              value={(config.class_filter as string) ?? ""}
              onChange={(e) => onConfigChange("class_filter", e.target.value)}
              placeholder="person, car, dog (comma-separated)"
              className="h-8"
            />
            <p className="text-xs text-muted-foreground">
              Only crop specific classes (empty = all)
            </p>
          </div>

          {/* Output Options */}
          <div className="space-y-3 pt-2 border-t">
            <Label className="text-xs font-medium">Output Options</Label>

            <div className="flex items-center justify-between">
              <Label className="text-xs">Include Source Box</Label>
              <Switch
                checked={(config.include_source_box as boolean) ?? true}
                onCheckedChange={(v) => onConfigChange("include_source_box", v)}
              />
            </div>

            <div className="flex items-center justify-between">
              <Label className="text-xs">Include Class Info</Label>
              <Switch
                checked={(config.include_class_info as boolean) ?? true}
                onCheckedChange={(v) => onConfigChange("include_class_info", v)}
              />
            </div>

            <div className="flex items-center justify-between">
              <Label className="text-xs">Include Crop Index</Label>
              <Switch
                checked={(config.include_index as boolean) ?? true}
                onCheckedChange={(v) => onConfigChange("include_index", v)}
              />
            </div>
          </div>

          {/* Output Format */}
          <div className="space-y-2">
            <Label className="text-xs">Output Format</Label>
            <Select
              value={(config.output_format as string) ?? "array"}
              onValueChange={(v) => onConfigChange("output_format", v)}
            >
              <SelectTrigger className="h-8">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="array">Array of Images</SelectItem>
                <SelectItem value="base64">Base64 Encoded</SelectItem>
                <SelectItem value="tensor">Tensor (for ML)</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Sort Order */}
          <div className="space-y-2">
            <Label className="text-xs">Sort Crops By</Label>
            <Select
              value={(config.sort_by as string) ?? "none"}
              onValueChange={(v) => onConfigChange("sort_by", v)}
            >
              <SelectTrigger className="h-8">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="none">Original Order</SelectItem>
                <SelectItem value="confidence_desc">Confidence (High to Low)</SelectItem>
                <SelectItem value="confidence_asc">Confidence (Low to High)</SelectItem>
                <SelectItem value="area_desc">Area (Large to Small)</SelectItem>
                <SelectItem value="area_asc">Area (Small to Large)</SelectItem>
                <SelectItem value="left_right">Left to Right</SelectItem>
                <SelectItem value="top_bottom">Top to Bottom</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CollapsibleContent>
      </Collapsible>
    </div>
  );
}

// ============================================================================
// Resize Config - SOTA
// ============================================================================
function ResizeConfig({ config, onConfigChange, advancedOpen, setAdvancedOpen }: ConfigSectionProps) {
  return (
    <div className="space-y-4">
      {/* Resize Mode */}
      <div className="space-y-2">
        <Label className="text-xs">Resize Mode</Label>
        <Select
          value={(config.mode as string) ?? "fit_within"}
          onValueChange={(v) => onConfigChange("mode", v)}
        >
          <SelectTrigger className="h-8">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="exact">Exact Size (stretch)</SelectItem>
            <SelectItem value="fit_within">Fit Within (preserve ratio, shrink)</SelectItem>
            <SelectItem value="fit_outside">Fit Outside (preserve ratio, crop)</SelectItem>
            <SelectItem value="scale_factor">Scale Factor (multiply)</SelectItem>
            <SelectItem value="width_only">Width Only (auto height)</SelectItem>
            <SelectItem value="height_only">Height Only (auto width)</SelectItem>
            <SelectItem value="longest_edge">Longest Edge (common for models)</SelectItem>
            <SelectItem value="shortest_edge">Shortest Edge</SelectItem>
          </SelectContent>
        </Select>
        <p className="text-xs text-muted-foreground">
          {config.mode === "exact" && "Stretch to exact dimensions"}
          {config.mode === "fit_within" && "Fit inside bounds, may be smaller"}
          {config.mode === "fit_outside" && "Fill bounds, excess cropped"}
          {config.mode === "scale_factor" && "Multiply dimensions by factor"}
          {config.mode === "width_only" && "Set width, height auto-calculated"}
          {config.mode === "height_only" && "Set height, width auto-calculated"}
          {config.mode === "longest_edge" && "Set longest edge, scale proportionally"}
          {config.mode === "shortest_edge" && "Set shortest edge, scale proportionally"}
          {!config.mode && "Fit inside bounds, may be smaller"}
        </p>
      </div>

      {/* Target Size based on mode */}
      {config.mode === "scale_factor" ? (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label className="text-xs">Scale Factor</Label>
            <span className="text-xs text-muted-foreground">
              {(config.scale_factor as number) ?? 1.0}x
            </span>
          </div>
          <ParamSlider
            value={(config.scale_factor as number) ?? 1.0}
            onChange={(v) => onConfigChange("scale_factor", v)}
            min={0.1}
            max={4.0}
            step={0.1}
          />
        </div>
      ) : config.mode === "width_only" ? (
        <div className="space-y-2">
          <Label className="text-xs">Target Width</Label>
          <ParamInput
            type="number"
            value={(config.width as number) ?? 640}
            onChange={(v) => onConfigChange("width", Number(v))}
            placeholder="640 or {{ params.width }}"
          />
        </div>
      ) : config.mode === "height_only" ? (
        <div className="space-y-2">
          <Label className="text-xs">Target Height</Label>
          <ParamInput
            type="number"
            value={(config.height as number) ?? 640}
            onChange={(v) => onConfigChange("height", Number(v))}
            placeholder="640 or {{ params.height }}"
          />
        </div>
      ) : config.mode === "longest_edge" || config.mode === "shortest_edge" ? (
        <div className="space-y-2">
          <Label className="text-xs">Edge Size</Label>
          <ParamInput
            type="number"
            value={(config.edge_size as number) ?? 640}
            onChange={(v) => onConfigChange("edge_size", Number(v))}
            placeholder="640 or {{ params.size }}"
          />
          <p className="text-xs text-muted-foreground">
            Common: 224 (ViT), 384 (DINOv2), 640 (YOLO), 1024 (SAM)
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-2 gap-3">
          <div className="space-y-2">
            <Label className="text-xs">Width</Label>
            <ParamInput
              type="number"
              value={(config.width as number) ?? 640}
              onChange={(v) => onConfigChange("width", Number(v))}
              placeholder="640"
            />
          </div>
          <div className="space-y-2">
            <Label className="text-xs">Height</Label>
            <ParamInput
              type="number"
              value={(config.height as number) ?? 640}
              onChange={(v) => onConfigChange("height", Number(v))}
              placeholder="640"
            />
          </div>
        </div>
      )}

      {/* Interpolation Method */}
      <div className="space-y-2">
        <Label className="text-xs">Interpolation</Label>
        <Select
          value={(config.interpolation as string) ?? "bilinear"}
          onValueChange={(v) => onConfigChange("interpolation", v)}
        >
          <SelectTrigger className="h-8">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="nearest">Nearest (fastest, pixelated)</SelectItem>
            <SelectItem value="bilinear">Bilinear (good balance)</SelectItem>
            <SelectItem value="bicubic">Bicubic (sharper)</SelectItem>
            <SelectItem value="lanczos">Lanczos (best quality)</SelectItem>
            <SelectItem value="area">Area (best for downscale)</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* Upscale/Downscale Policy */}
      <div className="grid grid-cols-2 gap-3">
        <div className="space-y-2">
          <Label className="text-xs">Upscale</Label>
          <Select
            value={(config.upscale_policy as string) ?? "allow"}
            onValueChange={(v) => onConfigChange("upscale_policy", v)}
          >
            <SelectTrigger className="h-8">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="allow">Allow</SelectItem>
              <SelectItem value="forbid">Forbid</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div className="space-y-2">
          <Label className="text-xs">Downscale</Label>
          <Select
            value={(config.downscale_policy as string) ?? "allow"}
            onValueChange={(v) => onConfigChange("downscale_policy", v)}
          >
            <SelectTrigger className="h-8">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="allow">Allow</SelectItem>
              <SelectItem value="forbid">Forbid</SelectItem>
            </SelectContent>
          </Select>
        </div>
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
          {/* Padding Mode (for fit_within) */}
          {(config.mode === "fit_within" || !config.mode) && (
            <>
              <div className="flex items-center justify-between">
                <div>
                  <Label className="text-xs">Add Padding</Label>
                  <p className="text-xs text-muted-foreground">Pad to exact size (letterbox)</p>
                </div>
                <Switch
                  checked={(config.add_padding as boolean) ?? false}
                  onCheckedChange={(v) => onConfigChange("add_padding", v)}
                />
              </div>

              {Boolean(config.add_padding) && (
                <div className="space-y-3 pl-4 border-l-2 border-muted">
                  <div className="space-y-2">
                    <Label className="text-xs">Padding Position</Label>
                    <Select
                      value={(config.padding_position as string) ?? "center"}
                      onValueChange={(v) => onConfigChange("padding_position", v)}
                    >
                      <SelectTrigger className="h-8">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="center">Center</SelectItem>
                        <SelectItem value="top_left">Top Left</SelectItem>
                        <SelectItem value="top_right">Top Right</SelectItem>
                        <SelectItem value="bottom_left">Bottom Left</SelectItem>
                        <SelectItem value="bottom_right">Bottom Right</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label className="text-xs">Padding Color</Label>
                    <div className="flex gap-2">
                      <Input
                        type="color"
                        value={(config.padding_color as string) ?? "#000000"}
                        onChange={(e) => onConfigChange("padding_color", e.target.value)}
                        className="h-8 w-12 p-1"
                      />
                      <Input
                        type="text"
                        value={(config.padding_color as string) ?? "#000000"}
                        onChange={(e) => onConfigChange("padding_color", e.target.value)}
                        placeholder="#000000"
                        className="h-8 flex-1 font-mono text-xs"
                      />
                    </div>
                  </div>
                </div>
              )}
            </>
          )}

          {/* Crop Position (for fit_outside) */}
          {config.mode === "fit_outside" && (
            <div className="space-y-2">
              <Label className="text-xs">Crop Position</Label>
              <Select
                value={(config.crop_position as string) ?? "center"}
                onValueChange={(v) => onConfigChange("crop_position", v)}
              >
                <SelectTrigger className="h-8">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="center">Center</SelectItem>
                  <SelectItem value="top">Top</SelectItem>
                  <SelectItem value="bottom">Bottom</SelectItem>
                  <SelectItem value="left">Left</SelectItem>
                  <SelectItem value="right">Right</SelectItem>
                </SelectContent>
              </Select>
            </div>
          )}

          {/* Anti-aliasing */}
          <div className="flex items-center justify-between">
            <div>
              <Label className="text-xs">Anti-aliasing</Label>
              <p className="text-xs text-muted-foreground">Smooth edges on downscale</p>
            </div>
            <Switch
              checked={(config.antialias as boolean) ?? true}
              onCheckedChange={(v) => onConfigChange("antialias", v)}
            />
          </div>

          {/* Preserve Aspect Ratio Info */}
          <div className="flex items-center justify-between">
            <div>
              <Label className="text-xs">Output Scale Info</Label>
              <p className="text-xs text-muted-foreground">Include scale factors in output</p>
            </div>
            <Switch
              checked={(config.output_scale_info as boolean) ?? true}
              onCheckedChange={(v) => onConfigChange("output_scale_info", v)}
            />
          </div>

          {/* Multiple of (for model requirements) */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label className="text-xs">Multiple Of</Label>
              <span className="text-xs text-muted-foreground">
                {(config.multiple_of as number) || "None"}
              </span>
            </div>
            <ParamSlider
              value={(config.multiple_of as number) ?? 0}
              onChange={(v) => onConfigChange("multiple_of", v)}
              min={0}
              max={64}
              step={8}
            />
            <p className="text-xs text-muted-foreground">
              Force dimensions to be multiple of N (0 = off). Common: 32 for YOLO, 14 for ViT
            </p>
          </div>

          {/* Min/Max Size Constraints */}
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-2">
              <Label className="text-xs">Min Size</Label>
              <ParamInput
                type="number"
                value={(config.min_size as number) ?? ""}
                onChange={(v) => onConfigChange("min_size", v ? Number(v) : null)}
                placeholder="No min"
              />
            </div>
            <div className="space-y-2">
              <Label className="text-xs">Max Size</Label>
              <ParamInput
                type="number"
                value={(config.max_size as number) ?? ""}
                onChange={(v) => onConfigChange("max_size", v ? Number(v) : null)}
                placeholder="No max"
              />
            </div>
          </div>
        </CollapsibleContent>
      </Collapsible>
    </div>
  );
}

// ============================================================================
// Tile Config - SOTA (SAHI-style slicing)
// ============================================================================
function TileConfig({ config, onConfigChange, advancedOpen, setAdvancedOpen }: ConfigSectionProps) {
  return (
    <div className="space-y-4">
      {/* Info about SAHI */}
      <div className="text-xs text-muted-foreground bg-muted/50 rounded-md p-3">
        <p className="font-medium text-foreground mb-1">Sliced Inference (SAHI)</p>
        <p>Split large images into overlapping tiles for better small object detection. Run detection on each tile, then merge with NMS.</p>
      </div>

      {/* Tile Size */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label className="text-xs">Tile Size</Label>
          <span className="text-xs text-muted-foreground">
            {(config.tile_size as number) ?? 640}px
          </span>
        </div>
        <ParamSlider
          value={(config.tile_size as number) ?? 640}
          onChange={(v) => onConfigChange("tile_size", v)}
          min={256}
          max={1280}
          step={32}
        />
        <p className="text-xs text-muted-foreground">
          Match your model input size. Common: 640 (YOLO), 800 (DINO), 1024 (SAM)
        </p>
      </div>

      {/* Overlap Ratio */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label className="text-xs">Overlap Ratio</Label>
          <span className="text-xs text-muted-foreground">
            {((config.overlap_ratio as number) ?? 0.2) * 100}%
          </span>
        </div>
        <ParamSlider
          value={(config.overlap_ratio as number) ?? 0.2}
          onChange={(v) => onConfigChange("overlap_ratio", v)}
          min={0}
          max={0.5}
          step={0.05}
        />
        <p className="text-xs text-muted-foreground">
          Higher overlap = better edge detection, more compute
        </p>
      </div>

      {/* Tiling Mode */}
      <div className="space-y-2">
        <Label className="text-xs">Tiling Mode</Label>
        <Select
          value={(config.tiling_mode as string) ?? "full"}
          onValueChange={(v) => onConfigChange("tiling_mode", v)}
        >
          <SelectTrigger className="h-8">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="full">Full Coverage (all tiles)</SelectItem>
            <SelectItem value="auto">Auto (skip if image smaller than tile)</SelectItem>
            <SelectItem value="grid">Fixed Grid (rows × cols)</SelectItem>
            <SelectItem value="adaptive">Adaptive (based on image size)</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* Fixed Grid Settings */}
      {config.tiling_mode === "grid" && (
        <div className="grid grid-cols-2 gap-3 pl-4 border-l-2 border-muted">
          <div className="space-y-2">
            <Label className="text-xs">Rows</Label>
            <ParamInput
              type="number"
              value={(config.grid_rows as number) ?? 2}
              onChange={(v) => onConfigChange("grid_rows", Number(v))}
              placeholder="2"
            />
          </div>
          <div className="space-y-2">
            <Label className="text-xs">Columns</Label>
            <ParamInput
              type="number"
              value={(config.grid_cols as number) ?? 2}
              onChange={(v) => onConfigChange("grid_cols", Number(v))}
              placeholder="2"
            />
          </div>
        </div>
      )}

      {/* Min Area Ratio */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label className="text-xs">Min Tile Area Ratio</Label>
          <span className="text-xs text-muted-foreground">
            {((config.min_area_ratio as number) ?? 0.1) * 100}%
          </span>
        </div>
        <ParamSlider
          value={(config.min_area_ratio as number) ?? 0.1}
          onChange={(v) => onConfigChange("min_area_ratio", v)}
          min={0}
          max={0.5}
          step={0.05}
        />
        <p className="text-xs text-muted-foreground">
          Skip edge tiles smaller than this % of full tile
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
          {/* Padding for Edge Tiles */}
          <div className="flex items-center justify-between">
            <div>
              <Label className="text-xs">Pad Edge Tiles</Label>
              <p className="text-xs text-muted-foreground">Pad to full tile size</p>
            </div>
            <Switch
              checked={(config.pad_edges as boolean) ?? true}
              onCheckedChange={(v) => onConfigChange("pad_edges", v)}
            />
          </div>

          {Boolean(config.pad_edges) && (
            <div className="space-y-2 pl-4 border-l-2 border-muted">
              <Label className="text-xs">Padding Color</Label>
              <div className="flex gap-2">
                <Input
                  type="color"
                  value={(config.padding_color as string) ?? "#000000"}
                  onChange={(e) => onConfigChange("padding_color", e.target.value)}
                  className="h-8 w-12 p-1"
                />
                <Input
                  type="text"
                  value={(config.padding_color as string) ?? "#000000"}
                  onChange={(e) => onConfigChange("padding_color", e.target.value)}
                  placeholder="#000000"
                  className="h-8 flex-1 font-mono text-xs"
                />
              </div>
            </div>
          )}

          {/* Include Full Image */}
          <div className="flex items-center justify-between">
            <div>
              <Label className="text-xs">Include Full Image</Label>
              <p className="text-xs text-muted-foreground">Also run on downscaled full image</p>
            </div>
            <Switch
              checked={(config.include_full_image as boolean) ?? false}
              onCheckedChange={(v) => onConfigChange("include_full_image", v)}
            />
          </div>

          {Boolean(config.include_full_image) && (
            <div className="space-y-2 pl-4 border-l-2 border-muted">
              <div className="flex items-center justify-between">
                <Label className="text-xs">Full Image Scale</Label>
                <span className="text-xs text-muted-foreground">
                  {(config.full_image_scale as number) ?? 1.0}x
                </span>
              </div>
              <ParamSlider
                value={(config.full_image_scale as number) ?? 1.0}
                onChange={(v) => onConfigChange("full_image_scale", v)}
                min={0.25}
                max={1.0}
                step={0.05}
              />
            </div>
          )}

          {/* Output Tile Images */}
          <div className="flex items-center justify-between">
            <div>
              <Label className="text-xs">Output Tile Images</Label>
              <p className="text-xs text-muted-foreground">Include cropped tile images</p>
            </div>
            <Switch
              checked={(config.output_images as boolean) ?? true}
              onCheckedChange={(v) => onConfigChange("output_images", v)}
            />
          </div>

          {/* Max Tiles Limit */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label className="text-xs">Max Tiles</Label>
              <span className="text-xs text-muted-foreground">
                {(config.max_tiles as number) || "No limit"}
              </span>
            </div>
            <ParamSlider
              value={(config.max_tiles as number) ?? 0}
              onChange={(v) => onConfigChange("max_tiles", v)}
              min={0}
              max={100}
              step={1}
            />
            <p className="text-xs text-muted-foreground">
              Limit total tiles (0 = no limit). Useful for very large images.
            </p>
          </div>
        </CollapsibleContent>
      </Collapsible>
    </div>
  );
}

// ============================================================================
// Stitch Config - SOTA (Merge tiled results)
// ============================================================================
function StitchConfig({ config, onConfigChange, advancedOpen, setAdvancedOpen }: ConfigSectionProps) {
  return (
    <div className="space-y-4">
      {/* Info */}
      <div className="text-xs text-muted-foreground bg-muted/50 rounded-md p-3">
        <p className="font-medium text-foreground mb-1">Merge Tiled Results</p>
        <p>Combine detections from multiple tiles, removing duplicates at tile boundaries using NMS.</p>
      </div>

      {/* Merge Mode */}
      <div className="space-y-2">
        <Label className="text-xs">Merge Mode</Label>
        <Select
          value={(config.merge_mode as string) ?? "nms"}
          onValueChange={(v) => onConfigChange("merge_mode", v)}
        >
          <SelectTrigger className="h-8">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="nms">NMS (Non-Maximum Suppression)</SelectItem>
            <SelectItem value="soft_nms">Soft NMS (decay scores)</SelectItem>
            <SelectItem value="nms_class">Class-wise NMS</SelectItem>
            <SelectItem value="union">Union (keep all)</SelectItem>
            <SelectItem value="greedy">Greedy NMS</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* IoU Threshold */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label className="text-xs">IoU Threshold</Label>
          <span className="text-xs text-muted-foreground">
            {(config.iou_threshold as number) ?? 0.5}
          </span>
        </div>
        <ParamSlider
          value={(config.iou_threshold as number) ?? 0.5}
          onChange={(v) => onConfigChange("iou_threshold", v)}
          min={0.1}
          max={0.9}
          step={0.05}
        />
        <p className="text-xs text-muted-foreground">
          Boxes with IoU above this are considered duplicates
        </p>
      </div>

      {/* Score Threshold */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label className="text-xs">Score Threshold</Label>
          <span className="text-xs text-muted-foreground">
            {(config.score_threshold as number) ?? 0.3}
          </span>
        </div>
        <ParamSlider
          value={(config.score_threshold as number) ?? 0.3}
          onChange={(v) => onConfigChange("score_threshold", v)}
          min={0.0}
          max={0.9}
          step={0.05}
        />
        <p className="text-xs text-muted-foreground">
          Filter low-confidence detections before merge
        </p>
      </div>

      {/* Coordinate Transform */}
      <div className="flex items-center justify-between">
        <div>
          <Label className="text-xs">Transform Coordinates</Label>
          <p className="text-xs text-muted-foreground">Convert to original image coords</p>
        </div>
        <Switch
          checked={(config.transform_coords as boolean) ?? true}
          onCheckedChange={(v) => onConfigChange("transform_coords", v)}
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
          {/* Edge Detection Handling */}
          <div className="space-y-2">
            <Label className="text-xs">Edge Box Handling</Label>
            <Select
              value={(config.edge_handling as string) ?? "keep"}
              onValueChange={(v) => onConfigChange("edge_handling", v)}
            >
              <SelectTrigger className="h-8">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="keep">Keep All</SelectItem>
                <SelectItem value="clip">Clip to Image Bounds</SelectItem>
                <SelectItem value="remove">Remove Edge-Touching</SelectItem>
                <SelectItem value="merge">Merge Cross-Tile Boxes</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Max Detections */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label className="text-xs">Max Detections</Label>
              <span className="text-xs text-muted-foreground">
                {(config.max_detections as number) || "No limit"}
              </span>
            </div>
            <ParamSlider
              value={(config.max_detections as number) ?? 0}
              onChange={(v) => onConfigChange("max_detections", v)}
              min={0}
              max={1000}
              step={10}
            />
          </div>

          {/* Soft NMS Sigma */}
          {config.merge_mode === "soft_nms" && (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label className="text-xs">Soft NMS Sigma</Label>
                <span className="text-xs text-muted-foreground">
                  {(config.soft_nms_sigma as number) ?? 0.5}
                </span>
              </div>
              <ParamSlider
                value={(config.soft_nms_sigma as number) ?? 0.5}
                onChange={(v) => onConfigChange("soft_nms_sigma", v)}
                min={0.1}
                max={2.0}
                step={0.1}
              />
            </div>
          )}

          {/* Reconstruct Image */}
          <div className="flex items-center justify-between">
            <div>
              <Label className="text-xs">Reconstruct Image</Label>
              <p className="text-xs text-muted-foreground">Stitch tile images together</p>
            </div>
            <Switch
              checked={(config.reconstruct_image as boolean) ?? false}
              onCheckedChange={(v) => onConfigChange("reconstruct_image", v)}
            />
          </div>

          {/* Include Stats */}
          <div className="flex items-center justify-between">
            <div>
              <Label className="text-xs">Include Statistics</Label>
              <p className="text-xs text-muted-foreground">Output merge stats</p>
            </div>
            <Switch
              checked={(config.include_stats as boolean) ?? true}
              onCheckedChange={(v) => onConfigChange("include_stats", v)}
            />
          </div>

          {/* Preserve Tile Info */}
          <div className="flex items-center justify-between">
            <div>
              <Label className="text-xs">Preserve Tile Info</Label>
              <p className="text-xs text-muted-foreground">Add source tile to each detection</p>
            </div>
            <Switch
              checked={(config.preserve_tile_info as boolean) ?? false}
              onCheckedChange={(v) => onConfigChange("preserve_tile_info", v)}
            />
          </div>
        </CollapsibleContent>
      </Collapsible>
    </div>
  );
}

// ============================================================================
// Rotate/Flip Config - SOTA
// ============================================================================
function RotateFlipConfig({ config, onConfigChange, advancedOpen, setAdvancedOpen }: ConfigSectionProps) {
  return (
    <div className="space-y-4">
      {/* Rotation */}
      <div className="space-y-2">
        <Label className="text-xs">Rotation</Label>
        <Select
          value={(config.rotation as string) ?? "none"}
          onValueChange={(v) => onConfigChange("rotation", v)}
        >
          <SelectTrigger className="h-8">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="none">No Rotation</SelectItem>
            <SelectItem value="90">90° Clockwise</SelectItem>
            <SelectItem value="180">180°</SelectItem>
            <SelectItem value="270">270° (90° Counter-clockwise)</SelectItem>
            <SelectItem value="auto_exif">Auto (from EXIF)</SelectItem>
            <SelectItem value="custom">Custom Angle</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* Custom Angle */}
      {config.rotation === "custom" && (
        <div className="space-y-2 pl-4 border-l-2 border-muted">
          <div className="flex items-center justify-between">
            <Label className="text-xs">Angle (degrees)</Label>
            <span className="text-xs text-muted-foreground">
              {(config.custom_angle as number) ?? 0}°
            </span>
          </div>
          <ParamSlider
            value={(config.custom_angle as number) ?? 0}
            onChange={(v) => onConfigChange("custom_angle", v)}
            min={-180}
            max={180}
            step={1}
          />
        </div>
      )}

      {/* Flip Horizontal */}
      <div className="flex items-center justify-between">
        <div>
          <Label className="text-xs">Flip Horizontal</Label>
          <p className="text-xs text-muted-foreground">Mirror left-right</p>
        </div>
        <Switch
          checked={(config.flip_horizontal as boolean) ?? false}
          onCheckedChange={(v) => onConfigChange("flip_horizontal", v)}
        />
      </div>

      {/* Flip Vertical */}
      <div className="flex items-center justify-between">
        <div>
          <Label className="text-xs">Flip Vertical</Label>
          <p className="text-xs text-muted-foreground">Mirror top-bottom</p>
        </div>
        <Switch
          checked={(config.flip_vertical as boolean) ?? false}
          onCheckedChange={(v) => onConfigChange("flip_vertical", v)}
        />
      </div>

      {/* Transform Boxes */}
      <div className="flex items-center justify-between">
        <div>
          <Label className="text-xs">Transform Boxes</Label>
          <p className="text-xs text-muted-foreground">Apply same transform to bboxes</p>
        </div>
        <Switch
          checked={(config.transform_boxes as boolean) ?? true}
          onCheckedChange={(v) => onConfigChange("transform_boxes", v)}
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
          {/* Expand Canvas (for arbitrary rotation) */}
          {config.rotation === "custom" && (
            <div className="flex items-center justify-between">
              <div>
                <Label className="text-xs">Expand Canvas</Label>
                <p className="text-xs text-muted-foreground">Fit rotated image without crop</p>
              </div>
              <Switch
                checked={(config.expand_canvas as boolean) ?? true}
                onCheckedChange={(v) => onConfigChange("expand_canvas", v)}
              />
            </div>
          )}

          {/* Background Color */}
          {config.rotation === "custom" && (
            <div className="space-y-2">
              <Label className="text-xs">Background Color</Label>
              <div className="flex gap-2">
                <Input
                  type="color"
                  value={(config.background_color as string) ?? "#000000"}
                  onChange={(e) => onConfigChange("background_color", e.target.value)}
                  className="h-8 w-12 p-1"
                />
                <Input
                  type="text"
                  value={(config.background_color as string) ?? "#000000"}
                  onChange={(e) => onConfigChange("background_color", e.target.value)}
                  placeholder="#000000"
                  className="h-8 flex-1 font-mono text-xs"
                />
              </div>
            </div>
          )}

          {/* Interpolation */}
          <div className="space-y-2">
            <Label className="text-xs">Interpolation</Label>
            <Select
              value={(config.interpolation as string) ?? "bilinear"}
              onValueChange={(v) => onConfigChange("interpolation", v)}
            >
              <SelectTrigger className="h-8">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="nearest">Nearest</SelectItem>
                <SelectItem value="bilinear">Bilinear</SelectItem>
                <SelectItem value="bicubic">Bicubic</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Center Point */}
          <div className="space-y-2">
            <Label className="text-xs">Rotation Center</Label>
            <Select
              value={(config.center as string) ?? "center"}
              onValueChange={(v) => onConfigChange("center", v)}
            >
              <SelectTrigger className="h-8">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="center">Image Center</SelectItem>
                <SelectItem value="top_left">Top Left</SelectItem>
                <SelectItem value="custom">Custom Point</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Output Transform Matrix */}
          <div className="flex items-center justify-between">
            <div>
              <Label className="text-xs">Output Transform Matrix</Label>
              <p className="text-xs text-muted-foreground">For coordinate mapping</p>
            </div>
            <Switch
              checked={(config.output_matrix as boolean) ?? false}
              onCheckedChange={(v) => onConfigChange("output_matrix", v)}
            />
          </div>
        </CollapsibleContent>
      </Collapsible>
    </div>
  );
}

// ============================================================================
// Normalize Config - SOTA
// ============================================================================
function NormalizeConfig({ config, onConfigChange, advancedOpen, setAdvancedOpen }: ConfigSectionProps) {
  return (
    <div className="space-y-4">
      {/* Normalization Preset */}
      <div className="space-y-2">
        <Label className="text-xs">Normalization Preset</Label>
        <Select
          value={(config.preset as string) ?? "imagenet"}
          onValueChange={(v) => onConfigChange("preset", v)}
        >
          <SelectTrigger className="h-8">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="imagenet">ImageNet (most models)</SelectItem>
            <SelectItem value="clip">CLIP / SigLIP</SelectItem>
            <SelectItem value="dinov2">DINOv2</SelectItem>
            <SelectItem value="0_1">0-1 Range (divide by 255)</SelectItem>
            <SelectItem value="-1_1">-1 to 1 Range</SelectItem>
            <SelectItem value="none">None (keep 0-255)</SelectItem>
            <SelectItem value="custom">Custom Mean/Std</SelectItem>
          </SelectContent>
        </Select>
        <p className="text-xs text-muted-foreground">
          {config.preset === "imagenet" && "mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]"}
          {config.preset === "clip" && "mean=[0.481,0.458,0.408], std=[0.269,0.261,0.276]"}
          {config.preset === "dinov2" && "mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]"}
          {config.preset === "0_1" && "Divide by 255, range [0, 1]"}
          {config.preset === "-1_1" && "Scale to range [-1, 1]"}
          {config.preset === "none" && "Keep original pixel values"}
          {config.preset === "custom" && "Custom mean and std values"}
          {!config.preset && "mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]"}
        </p>
      </div>

      {/* Custom Mean/Std */}
      {config.preset === "custom" && (
        <div className="space-y-3 pl-4 border-l-2 border-muted">
          <div className="space-y-2">
            <Label className="text-xs">Mean (R, G, B)</Label>
            <div className="grid grid-cols-3 gap-2">
              <Input
                type="number"
                value={(config.mean_r as number) ?? 0.485}
                onChange={(e) => onConfigChange("mean_r", parseFloat(e.target.value))}
                step={0.001}
                className="h-8 text-xs"
                placeholder="R"
              />
              <Input
                type="number"
                value={(config.mean_g as number) ?? 0.456}
                onChange={(e) => onConfigChange("mean_g", parseFloat(e.target.value))}
                step={0.001}
                className="h-8 text-xs"
                placeholder="G"
              />
              <Input
                type="number"
                value={(config.mean_b as number) ?? 0.406}
                onChange={(e) => onConfigChange("mean_b", parseFloat(e.target.value))}
                step={0.001}
                className="h-8 text-xs"
                placeholder="B"
              />
            </div>
          </div>
          <div className="space-y-2">
            <Label className="text-xs">Std (R, G, B)</Label>
            <div className="grid grid-cols-3 gap-2">
              <Input
                type="number"
                value={(config.std_r as number) ?? 0.229}
                onChange={(e) => onConfigChange("std_r", parseFloat(e.target.value))}
                step={0.001}
                className="h-8 text-xs"
                placeholder="R"
              />
              <Input
                type="number"
                value={(config.std_g as number) ?? 0.224}
                onChange={(e) => onConfigChange("std_g", parseFloat(e.target.value))}
                step={0.001}
                className="h-8 text-xs"
                placeholder="G"
              />
              <Input
                type="number"
                value={(config.std_b as number) ?? 0.225}
                onChange={(e) => onConfigChange("std_b", parseFloat(e.target.value))}
                step={0.001}
                className="h-8 text-xs"
                placeholder="B"
              />
            </div>
          </div>
        </div>
      )}

      {/* Output Format */}
      <div className="space-y-2">
        <Label className="text-xs">Output Format</Label>
        <Select
          value={(config.output_format as string) ?? "image"}
          onValueChange={(v) => onConfigChange("output_format", v)}
        >
          <SelectTrigger className="h-8">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="image">Image (for display)</SelectItem>
            <SelectItem value="tensor">Tensor [C,H,W] (for models)</SelectItem>
            <SelectItem value="both">Both</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* Channel Order */}
      <div className="space-y-2">
        <Label className="text-xs">Channel Order</Label>
        <Select
          value={(config.channel_order as string) ?? "rgb"}
          onValueChange={(v) => onConfigChange("channel_order", v)}
        >
          <SelectTrigger className="h-8">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="rgb">RGB (most models)</SelectItem>
            <SelectItem value="bgr">BGR (OpenCV style)</SelectItem>
          </SelectContent>
        </Select>
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
          {/* Data Type */}
          <div className="space-y-2">
            <Label className="text-xs">Output Data Type</Label>
            <Select
              value={(config.dtype as string) ?? "float32"}
              onValueChange={(v) => onConfigChange("dtype", v)}
            >
              <SelectTrigger className="h-8">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="float32">float32 (default)</SelectItem>
                <SelectItem value="float16">float16 (half precision)</SelectItem>
                <SelectItem value="uint8">uint8 (image only)</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Denormalize Option */}
          <div className="flex items-center justify-between">
            <div>
              <Label className="text-xs">Also Output Denormalized</Label>
              <p className="text-xs text-muted-foreground">Include original scale image</p>
            </div>
            <Switch
              checked={(config.output_denormalized as boolean) ?? false}
              onCheckedChange={(v) => onConfigChange("output_denormalized", v)}
            />
          </div>

          {/* Clip Values */}
          <div className="flex items-center justify-between">
            <div>
              <Label className="text-xs">Clip Values</Label>
              <p className="text-xs text-muted-foreground">Clamp to valid range</p>
            </div>
            <Switch
              checked={(config.clip_values as boolean) ?? true}
              onCheckedChange={(v) => onConfigChange("clip_values", v)}
            />
          </div>

          {/* Per-channel stats */}
          <div className="flex items-center justify-between">
            <div>
              <Label className="text-xs">Output Channel Stats</Label>
              <p className="text-xs text-muted-foreground">Include mean/std info</p>
            </div>
            <Switch
              checked={(config.output_stats as boolean) ?? false}
              onCheckedChange={(v) => onConfigChange("output_stats", v)}
            />
          </div>
        </CollapsibleContent>
      </Collapsible>
    </div>
  );
}

// ============================================================================
// Filter Config - SOTA
// ============================================================================
function FilterConfig({ config, onConfigChange, advancedOpen, setAdvancedOpen }: ConfigSectionProps) {
  // Filter conditions (support multiple conditions)
  const conditions = (config.conditions as Array<{field: string; operator: string; value: unknown}>) ?? [
    { field: "confidence", operator: "greater_than", value: 0.5 }
  ];

  const updateCondition = (index: number, key: string, value: unknown) => {
    const newConditions = [...conditions];
    newConditions[index] = { ...newConditions[index], [key]: value };
    onConfigChange("conditions", newConditions);
  };

  const addCondition = () => {
    onConfigChange("conditions", [
      ...conditions,
      { field: "confidence", operator: "greater_than", value: 0.5 }
    ]);
  };

  const removeCondition = (index: number) => {
    if (conditions.length > 1) {
      onConfigChange("conditions", conditions.filter((_, i) => i !== index));
    }
  };

  // Field type determination for value input
  const getFieldType = (field: string): "text" | "number" => {
    const textFields = ["class_name", "label", "id", "category"];
    return textFields.includes(field) ? "text" : "number";
  };

  // Show range input for "between" operator
  const showRangeInput = (operator: string) => operator === "between" || operator === "in_range";

  return (
    <div className="space-y-4">
      {/* Logic Mode */}
      <div className="space-y-2">
        <Label className="text-xs">Condition Logic</Label>
        <Select
          value={(config.logic_mode as string) ?? "all"}
          onValueChange={(v) => onConfigChange("logic_mode", v)}
        >
          <SelectTrigger className="h-8">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All conditions (AND)</SelectItem>
            <SelectItem value="any">Any condition (OR)</SelectItem>
            <SelectItem value="none">None match (NOR)</SelectItem>
          </SelectContent>
        </Select>
        <p className="text-xs text-muted-foreground">
          {config.logic_mode === "any" ? "Pass if ANY condition matches" :
           config.logic_mode === "none" ? "Pass if NO condition matches" :
           "Pass if ALL conditions match"}
        </p>
      </div>

      {/* Filter Conditions */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <Label className="text-xs">Filter Conditions</Label>
          <Button
            variant="ghost"
            size="sm"
            className="h-6 text-xs"
            onClick={addCondition}
          >
            <Plus className="h-3 w-3 mr-1" />
            Add
          </Button>
        </div>

        {conditions.map((condition, index) => (
          <div key={index} className="space-y-2 p-3 bg-muted/30 rounded-lg border border-muted">
            <div className="flex items-center justify-between">
              <span className="text-xs text-muted-foreground">Condition {index + 1}</span>
              {conditions.length > 1 && (
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-5 w-5 p-0 text-muted-foreground hover:text-destructive"
                  onClick={() => removeCondition(index)}
                >
                  <X className="h-3 w-3" />
                </Button>
              )}
            </div>

            {/* Field Selection */}
            <Select
              value={condition.field ?? "confidence"}
              onValueChange={(v) => updateCondition(index, "field", v)}
            >
              <SelectTrigger className="h-8">
                <SelectValue placeholder="Select field" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="confidence">Confidence Score</SelectItem>
                <SelectItem value="class_name">Class Name</SelectItem>
                <SelectItem value="class_id">Class ID</SelectItem>
                <SelectItem value="label">Label</SelectItem>
                <SelectItem value="area">Area (pixels²)</SelectItem>
                <SelectItem value="area_ratio">Area Ratio (% of image)</SelectItem>
                <SelectItem value="width">Width (pixels)</SelectItem>
                <SelectItem value="height">Height (pixels)</SelectItem>
                <SelectItem value="aspect_ratio">Aspect Ratio (w/h)</SelectItem>
                <SelectItem value="x">X Position (left)</SelectItem>
                <SelectItem value="y">Y Position (top)</SelectItem>
                <SelectItem value="center_x">Center X</SelectItem>
                <SelectItem value="center_y">Center Y</SelectItem>
                <SelectItem value="x_ratio">X Ratio (0-1)</SelectItem>
                <SelectItem value="y_ratio">Y Ratio (0-1)</SelectItem>
                <SelectItem value="index">Detection Index</SelectItem>
                <SelectItem value="score">Similarity Score</SelectItem>
                <SelectItem value="distance">Distance</SelectItem>
              </SelectContent>
            </Select>

            {/* Operator Selection */}
            <Select
              value={condition.operator ?? "greater_than"}
              onValueChange={(v) => updateCondition(index, "operator", v)}
            >
              <SelectTrigger className="h-8">
                <SelectValue placeholder="Select operator" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="greater_than">Greater than (&gt;)</SelectItem>
                <SelectItem value="greater_equal">Greater or equal (≥)</SelectItem>
                <SelectItem value="less_than">Less than (&lt;)</SelectItem>
                <SelectItem value="less_equal">Less or equal (≤)</SelectItem>
                <SelectItem value="equals">Equals (=)</SelectItem>
                <SelectItem value="not_equals">Not equals (≠)</SelectItem>
                <SelectItem value="between">Between (range)</SelectItem>
                <SelectItem value="in_range">In range (inclusive)</SelectItem>
                <SelectItem value="in">In list</SelectItem>
                <SelectItem value="not_in">Not in list</SelectItem>
                <SelectItem value="contains">Contains</SelectItem>
                <SelectItem value="not_contains">Not contains</SelectItem>
                <SelectItem value="starts_with">Starts with</SelectItem>
                <SelectItem value="ends_with">Ends with</SelectItem>
                <SelectItem value="regex">Regex match</SelectItem>
                <SelectItem value="is_null">Is null/empty</SelectItem>
                <SelectItem value="is_not_null">Is not null</SelectItem>
              </SelectContent>
            </Select>

            {/* Value Input(s) */}
            {condition.operator !== "is_null" && condition.operator !== "is_not_null" && (
              showRangeInput(condition.operator) ? (
                <div className="flex items-center gap-2">
                  <ParamInput
                    type="number"
                    value={(condition.value as { min?: number })?.min ?? ""}
                    onChange={(v) => updateCondition(index, "value", {
                      ...(condition.value as object || {}),
                      min: v
                    })}
                    placeholder="Min"
                    className="flex-1"
                  />
                  <span className="text-xs text-muted-foreground">to</span>
                  <ParamInput
                    type="number"
                    value={(condition.value as { max?: number })?.max ?? ""}
                    onChange={(v) => updateCondition(index, "value", {
                      ...(condition.value as object || {}),
                      max: v
                    })}
                    placeholder="Max"
                    className="flex-1"
                  />
                </div>
              ) : condition.operator === "in" || condition.operator === "not_in" ? (
                <ParamInput
                  type="text"
                  value={condition.value as string ?? ""}
                  onChange={(v) => updateCondition(index, "value", v)}
                  placeholder="item1, item2, item3 or {{ params.list }}"
                />
              ) : (
                <ParamInput
                  type={getFieldType(condition.field)}
                  value={condition.value as string ?? ""}
                  onChange={(v) => updateCondition(index, "value", v)}
                  placeholder={getFieldType(condition.field) === "text"
                    ? "Value or {{ params.x }}"
                    : "0.5 or {{ params.x }}"}
                />
              )
            )}
          </div>
        ))}
      </div>

      {/* Invert Filter */}
      <div className="flex items-center justify-between">
        <div>
          <Label className="text-xs">Invert Filter</Label>
          <p className="text-xs text-muted-foreground">Swap passed/rejected</p>
        </div>
        <Switch
          checked={(config.invert as boolean) ?? false}
          onCheckedChange={(v) => onConfigChange("invert", v)}
        />
      </div>

      {/* Sort Results */}
      <div className="flex items-center justify-between">
        <div>
          <Label className="text-xs">Sort Results</Label>
          <p className="text-xs text-muted-foreground">Order passed items</p>
        </div>
        <Switch
          checked={(config.sort_enabled as boolean) ?? false}
          onCheckedChange={(v) => onConfigChange("sort_enabled", v)}
        />
      </div>

      {Boolean(config.sort_enabled) && (
        <div className="space-y-3 pl-4 border-l-2 border-muted">
          <div className="space-y-2">
            <Label className="text-xs">Sort By</Label>
            <Select
              value={(config.sort_by as string) ?? "confidence"}
              onValueChange={(v) => onConfigChange("sort_by", v)}
            >
              <SelectTrigger className="h-8">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="confidence">Confidence</SelectItem>
                <SelectItem value="area">Area</SelectItem>
                <SelectItem value="width">Width</SelectItem>
                <SelectItem value="height">Height</SelectItem>
                <SelectItem value="x">X Position</SelectItem>
                <SelectItem value="y">Y Position</SelectItem>
                <SelectItem value="aspect_ratio">Aspect Ratio</SelectItem>
                <SelectItem value="score">Similarity Score</SelectItem>
                <SelectItem value="distance">Distance</SelectItem>
                <SelectItem value="index">Original Index</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label className="text-xs">Sort Order</Label>
            <Select
              value={(config.sort_order as string) ?? "desc"}
              onValueChange={(v) => onConfigChange("sort_order", v)}
            >
              <SelectTrigger className="h-8">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="desc">Descending (high to low)</SelectItem>
                <SelectItem value="asc">Ascending (low to high)</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
      )}

      {/* Limit Results */}
      <div className="flex items-center justify-between">
        <div>
          <Label className="text-xs">Limit Results</Label>
          <p className="text-xs text-muted-foreground">Cap number of outputs</p>
        </div>
        <Switch
          checked={(config.limit_enabled as boolean) ?? false}
          onCheckedChange={(v) => onConfigChange("limit_enabled", v)}
        />
      </div>

      {Boolean(config.limit_enabled) && (
        <div className="space-y-3 pl-4 border-l-2 border-muted">
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label className="text-xs">Max Results</Label>
              <span className="text-xs text-muted-foreground">
                {(config.max_results as number) ?? 10}
              </span>
            </div>
            <ParamSlider
              value={(config.max_results as number) ?? 10}
              onChange={(v) => onConfigChange("max_results", v)}
              min={1}
              max={100}
              step={1}
            />
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label className="text-xs">Skip First</Label>
              <span className="text-xs text-muted-foreground">
                {(config.skip as number) ?? 0}
              </span>
            </div>
            <ParamSlider
              value={(config.skip as number) ?? 0}
              onChange={(v) => onConfigChange("skip", v)}
              min={0}
              max={50}
              step={1}
            />
          </div>
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
          {/* Group By */}
          <div className="flex items-center justify-between">
            <div>
              <Label className="text-xs">Group Results</Label>
              <p className="text-xs text-muted-foreground">Group by field value</p>
            </div>
            <Switch
              checked={(config.group_enabled as boolean) ?? false}
              onCheckedChange={(v) => onConfigChange("group_enabled", v)}
            />
          </div>

          {Boolean(config.group_enabled) && (
            <div className="space-y-2 pl-4 border-l-2 border-muted">
              <Label className="text-xs">Group By Field</Label>
              <Select
                value={(config.group_by as string) ?? "class_name"}
                onValueChange={(v) => onConfigChange("group_by", v)}
              >
                <SelectTrigger className="h-8">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="class_name">Class Name</SelectItem>
                  <SelectItem value="class_id">Class ID</SelectItem>
                  <SelectItem value="label">Label</SelectItem>
                  <SelectItem value="category">Category</SelectItem>
                </SelectContent>
              </Select>
            </div>
          )}

          {/* NMS (Non-Maximum Suppression) */}
          <div className="flex items-center justify-between">
            <div>
              <Label className="text-xs">Apply NMS</Label>
              <p className="text-xs text-muted-foreground">Remove overlapping boxes</p>
            </div>
            <Switch
              checked={(config.apply_nms as boolean) ?? false}
              onCheckedChange={(v) => onConfigChange("apply_nms", v)}
            />
          </div>

          {Boolean(config.apply_nms) && (
            <div className="space-y-2 pl-4 border-l-2 border-muted">
              <div className="flex items-center justify-between">
                <Label className="text-xs">IoU Threshold</Label>
                <span className="text-xs text-muted-foreground">
                  {(config.nms_iou_threshold as number) ?? 0.5}
                </span>
              </div>
              <ParamSlider
                value={(config.nms_iou_threshold as number) ?? 0.5}
                onChange={(v) => onConfigChange("nms_iou_threshold", v)}
                min={0.1}
                max={1.0}
                step={0.05}
              />
              <p className="text-xs text-muted-foreground">
                Remove boxes with IoU above this threshold
              </p>
            </div>
          )}

          {/* Deduplicate */}
          <div className="flex items-center justify-between">
            <div>
              <Label className="text-xs">Deduplicate</Label>
              <p className="text-xs text-muted-foreground">Remove duplicate values</p>
            </div>
            <Switch
              checked={(config.deduplicate as boolean) ?? false}
              onCheckedChange={(v) => onConfigChange("deduplicate", v)}
            />
          </div>

          {Boolean(config.deduplicate) && (
            <div className="space-y-2 pl-4 border-l-2 border-muted">
              <Label className="text-xs">Dedupe By</Label>
              <Select
                value={(config.dedupe_by as string) ?? "id"}
                onValueChange={(v) => onConfigChange("dedupe_by", v)}
              >
                <SelectTrigger className="h-8">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="id">ID</SelectItem>
                  <SelectItem value="class_name">Class Name</SelectItem>
                  <SelectItem value="position">Position (bbox)</SelectItem>
                </SelectContent>
              </Select>
            </div>
          )}

          {/* Include Metadata */}
          <div className="flex items-center justify-between">
            <div>
              <Label className="text-xs">Include Filter Stats</Label>
              <p className="text-xs text-muted-foreground">Output min/max/avg statistics</p>
            </div>
            <Switch
              checked={(config.include_stats as boolean) ?? false}
              onCheckedChange={(v) => onConfigChange("include_stats", v)}
            />
          </div>

          {/* Preserve Original Index */}
          <div className="flex items-center justify-between">
            <div>
              <Label className="text-xs">Preserve Original Index</Label>
              <p className="text-xs text-muted-foreground">Add original_index to items</p>
            </div>
            <Switch
              checked={(config.preserve_index as boolean) ?? true}
              onCheckedChange={(v) => onConfigChange("preserve_index", v)}
            />
          </div>

          {/* Custom Expression */}
          <div className="flex items-center justify-between">
            <div>
              <Label className="text-xs">Custom Expression</Label>
              <p className="text-xs text-muted-foreground">JavaScript-like filter</p>
            </div>
            <Switch
              checked={(config.use_expression as boolean) ?? false}
              onCheckedChange={(v) => onConfigChange("use_expression", v)}
            />
          </div>

          {Boolean(config.use_expression) && (
            <div className="space-y-2 pl-4 border-l-2 border-muted">
              <Label className="text-xs">Filter Expression</Label>
              <Input
                type="text"
                value={(config.expression as string) ?? ""}
                onChange={(e) => onConfigChange("expression", e.target.value)}
                placeholder="item.confidence > 0.5 && item.area > 1000"
                className="h-8 font-mono text-xs"
              />
              <p className="text-xs text-muted-foreground">
                Use <code className="bg-muted px-1 rounded">item</code> to access each element
              </p>
            </div>
          )}

          {/* Empty Result Handling */}
          <div className="space-y-2">
            <Label className="text-xs">On Empty Result</Label>
            <Select
              value={(config.on_empty as string) ?? "empty_array"}
              onValueChange={(v) => onConfigChange("on_empty", v)}
            >
              <SelectTrigger className="h-8">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="empty_array">Return empty array</SelectItem>
                <SelectItem value="null">Return null</SelectItem>
                <SelectItem value="pass_all">Pass all items (fallback)</SelectItem>
                <SelectItem value="error">Raise error</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CollapsibleContent>
      </Collapsible>
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
// Condition Config - SOTA with multiple conditions, operators, and branches
// ============================================================================
function ConditionConfig({ config, onConfigChange, advancedOpen, setAdvancedOpen }: ConfigSectionProps) {
  // Conditions array management
  const conditions = Array.isArray(config.conditions)
    ? (config.conditions as Array<{ field: string; operator: string; value: string; type: string }>)
    : [{ field: "", operator: "greater_than", value: "", type: "auto" }];

  const updateCondition = (index: number, key: string, value: string) => {
    const newConditions = [...conditions];
    newConditions[index] = { ...newConditions[index], [key]: value };
    onConfigChange("conditions", newConditions);
  };

  const addCondition = () => {
    onConfigChange("conditions", [...conditions, { field: "", operator: "greater_than", value: "", type: "auto" }]);
  };

  const removeCondition = (index: number) => {
    if (conditions.length > 1) {
      onConfigChange("conditions", conditions.filter((_, i) => i !== index));
    }
  };

  // Operators grouped by category
  const operatorGroups = {
    comparison: [
      { value: "greater_than", label: "> Greater Than" },
      { value: "greater_equal", label: "≥ Greater or Equal" },
      { value: "less_than", label: "< Less Than" },
      { value: "less_equal", label: "≤ Less or Equal" },
      { value: "equals", label: "= Equals" },
      { value: "not_equals", label: "≠ Not Equals" },
    ],
    range: [
      { value: "between", label: "Between (inclusive)" },
      { value: "not_between", label: "Not Between" },
    ],
    string: [
      { value: "contains", label: "Contains" },
      { value: "not_contains", label: "Not Contains" },
      { value: "starts_with", label: "Starts With" },
      { value: "ends_with", label: "Ends With" },
      { value: "regex_match", label: "Regex Match" },
    ],
    array: [
      { value: "in_array", label: "In Array" },
      { value: "not_in_array", label: "Not In Array" },
      { value: "array_contains", label: "Array Contains" },
      { value: "array_empty", label: "Array Empty" },
      { value: "array_not_empty", label: "Array Not Empty" },
      { value: "array_length_equals", label: "Array Length =" },
      { value: "array_length_greater", label: "Array Length >" },
    ],
    type: [
      { value: "exists", label: "Exists (not null)" },
      { value: "not_exists", label: "Not Exists (null)" },
      { value: "is_true", label: "Is True" },
      { value: "is_false", label: "Is False" },
      { value: "is_type", label: "Is Type" },
    ],
  };

  const noValueOperators = ["exists", "not_exists", "is_true", "is_false", "array_empty", "array_not_empty"];
  const rangeOperators = ["between", "not_between"];

  return (
    <div className="space-y-4">
      {/* Logic Mode */}
      <div className="space-y-2">
        <Label className="text-xs">Combine Conditions</Label>
        <Select
          value={(config.logic_mode as string) ?? "and"}
          onValueChange={(v) => onConfigChange("logic_mode", v)}
        >
          <SelectTrigger className="h-8">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="and">AND - All must pass</SelectItem>
            <SelectItem value="or">OR - Any can pass</SelectItem>
            <SelectItem value="xor">XOR - Exactly one passes</SelectItem>
            <SelectItem value="nand">NAND - Not all pass</SelectItem>
            <SelectItem value="custom">Custom Expression</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* Custom Expression (when logic_mode is custom) */}
      {config.logic_mode === "custom" && (
        <div className="space-y-2">
          <Label className="text-xs">Custom Logic Expression</Label>
          <Input
            type="text"
            value={(config.custom_expression as string) ?? ""}
            onChange={(e) => onConfigChange("custom_expression", e.target.value)}
            placeholder="e.g., (c1 AND c2) OR c3"
            className="h-8 font-mono text-xs"
          />
          <p className="text-xs text-muted-foreground">
            Use c1, c2, c3... for conditions. Supports AND, OR, NOT, ()
          </p>
        </div>
      )}

      {/* Conditions */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <Label className="text-xs">Conditions</Label>
          <Button
            variant="outline"
            size="sm"
            onClick={addCondition}
            className="h-6 text-xs"
          >
            <Plus className="h-3 w-3 mr-1" />
            Add
          </Button>
        </div>

        {conditions.map((condition, index) => (
          <div key={index} className="space-y-2 p-3 border rounded-md bg-muted/30">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-muted-foreground">
                Condition {index + 1}
              </span>
              {conditions.length > 1 && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => removeCondition(index)}
                  className="h-5 w-5 p-0"
                >
                  <X className="h-3 w-3" />
                </Button>
              )}
            </div>

            {/* Field Path */}
            <div className="space-y-1">
              <Label className="text-xs text-muted-foreground">Field</Label>
              <Input
                type="text"
                value={condition.field}
                onChange={(e) => updateCondition(index, "field", e.target.value)}
                placeholder="e.g., detections.length, items[0].confidence"
                className="h-7 text-xs font-mono"
              />
            </div>

            {/* Operator */}
            <div className="space-y-1">
              <Label className="text-xs text-muted-foreground">Operator</Label>
              <Select
                value={condition.operator}
                onValueChange={(v) => updateCondition(index, "operator", v)}
              >
                <SelectTrigger className="h-7 text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="" disabled className="text-xs font-medium text-muted-foreground">
                    — Comparison —
                  </SelectItem>
                  {operatorGroups.comparison.map(op => (
                    <SelectItem key={op.value} value={op.value}>{op.label}</SelectItem>
                  ))}
                  <SelectItem value="" disabled className="text-xs font-medium text-muted-foreground">
                    — Range —
                  </SelectItem>
                  {operatorGroups.range.map(op => (
                    <SelectItem key={op.value} value={op.value}>{op.label}</SelectItem>
                  ))}
                  <SelectItem value="" disabled className="text-xs font-medium text-muted-foreground">
                    — String —
                  </SelectItem>
                  {operatorGroups.string.map(op => (
                    <SelectItem key={op.value} value={op.value}>{op.label}</SelectItem>
                  ))}
                  <SelectItem value="" disabled className="text-xs font-medium text-muted-foreground">
                    — Array —
                  </SelectItem>
                  {operatorGroups.array.map(op => (
                    <SelectItem key={op.value} value={op.value}>{op.label}</SelectItem>
                  ))}
                  <SelectItem value="" disabled className="text-xs font-medium text-muted-foreground">
                    — Type Check —
                  </SelectItem>
                  {operatorGroups.type.map(op => (
                    <SelectItem key={op.value} value={op.value}>{op.label}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Value (unless no-value operator) */}
            {!noValueOperators.includes(condition.operator) && (
              <div className="space-y-1">
                <Label className="text-xs text-muted-foreground">
                  {rangeOperators.includes(condition.operator) ? "Min, Max (comma-separated)" : "Value"}
                </Label>
                <Input
                  type="text"
                  value={condition.value}
                  onChange={(e) => updateCondition(index, "value", e.target.value)}
                  placeholder={
                    rangeOperators.includes(condition.operator) ? "e.g., 0.5, 0.9" :
                    condition.operator === "in_array" ? "e.g., apple, banana, orange" :
                    condition.operator === "is_type" ? "e.g., string, number, array" :
                    condition.operator === "regex_match" ? "e.g., ^[A-Z].*" :
                    "Value to compare"
                  }
                  className="h-7 text-xs"
                />
              </div>
            )}

            {/* Value Type */}
            <div className="space-y-1">
              <Label className="text-xs text-muted-foreground">Parse Value As</Label>
              <Select
                value={condition.type || "auto"}
                onValueChange={(v) => updateCondition(index, "type", v)}
              >
                <SelectTrigger className="h-7 text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="auto">Auto-detect</SelectItem>
                  <SelectItem value="string">String</SelectItem>
                  <SelectItem value="number">Number</SelectItem>
                  <SelectItem value="boolean">Boolean</SelectItem>
                  <SelectItem value="field">Field Reference</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        ))}
      </div>

      {/* Invert Result */}
      <div className="flex items-center justify-between">
        <div>
          <Label className="text-xs">Invert Result</Label>
          <p className="text-xs text-muted-foreground">
            Swap true/false outputs
          </p>
        </div>
        <Switch
          checked={(config.invert as boolean) ?? false}
          onCheckedChange={(v) => onConfigChange("invert", v)}
        />
      </div>

      {/* Branch Info */}
      <div className="text-xs text-muted-foreground bg-muted/50 rounded-md p-3">
        <p className="font-medium text-foreground mb-1">Output Branches</p>
        <p>
          <strong>true_output:</strong> When condition {config.invert ? "fails" : "passes"}<br />
          <strong>false_output:</strong> When condition {config.invert ? "passes" : "fails"}
        </p>
      </div>

      {/* Advanced Section */}
      <Collapsible open={advancedOpen} onOpenChange={setAdvancedOpen}>
        <CollapsibleTrigger asChild>
          <Button variant="ghost" size="sm" className="w-full justify-between h-8">
            <span className="text-xs">Advanced Options</span>
            <ChevronDown className={`h-4 w-4 transition-transform ${advancedOpen ? "rotate-180" : ""}`} />
          </Button>
        </CollapsibleTrigger>
        <CollapsibleContent className="space-y-4 pt-2">
          {/* Default Branch */}
          <div className="space-y-2">
            <Label className="text-xs">Default Branch (on error)</Label>
            <Select
              value={(config.default_branch as string) ?? "false"}
              onValueChange={(v) => onConfigChange("default_branch", v)}
            >
              <SelectTrigger className="h-8">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="true">True Branch</SelectItem>
                <SelectItem value="false">False Branch</SelectItem>
                <SelectItem value="error">Throw Error</SelectItem>
              </SelectContent>
            </Select>
            <p className="text-xs text-muted-foreground">
              Which branch to take if evaluation fails
            </p>
          </div>

          {/* Fallback Value */}
          <div className="space-y-2">
            <Label className="text-xs">Fallback Value</Label>
            <Input
              type="text"
              value={(config.fallback_value as string) ?? ""}
              onChange={(e) => onConfigChange("fallback_value", e.target.value)}
              placeholder="Value if field is missing"
              className="h-8"
            />
            <p className="text-xs text-muted-foreground">
              Use this value when the field path doesn&apos;t exist
            </p>
          </div>

          {/* Strict Type Comparison */}
          <div className="flex items-center justify-between">
            <div>
              <Label className="text-xs">Strict Comparison</Label>
              <p className="text-xs text-muted-foreground">
                Type-sensitive equality (=== vs ==)
              </p>
            </div>
            <Switch
              checked={(config.strict as boolean) ?? true}
              onCheckedChange={(v) => onConfigChange("strict", v)}
            />
          </div>

          {/* Case Sensitive */}
          <div className="flex items-center justify-between">
            <div>
              <Label className="text-xs">Case Sensitive</Label>
              <p className="text-xs text-muted-foreground">
                For string comparisons
              </p>
            </div>
            <Switch
              checked={(config.case_sensitive as boolean) ?? true}
              onCheckedChange={(v) => onConfigChange("case_sensitive", v)}
            />
          </div>

          {/* Pass Through Data */}
          <div className="flex items-center justify-between">
            <div>
              <Label className="text-xs">Pass Through Input</Label>
              <p className="text-xs text-muted-foreground">
                Include original input in output
              </p>
            </div>
            <Switch
              checked={(config.pass_through as boolean) ?? true}
              onCheckedChange={(v) => onConfigChange("pass_through", v)}
            />
          </div>

          {/* Add Metadata */}
          <div className="flex items-center justify-between">
            <div>
              <Label className="text-xs">Add Evaluation Metadata</Label>
              <p className="text-xs text-muted-foreground">
                Include which conditions passed/failed
              </p>
            </div>
            <Switch
              checked={(config.add_metadata as boolean) ?? false}
              onCheckedChange={(v) => onConfigChange("add_metadata", v)}
            />
          </div>
        </CollapsibleContent>
      </Collapsible>
    </div>
  );
}

// ============================================================================
// Grid Builder Config - SOTA Realogram/Planogram Builder
// ============================================================================
function GridBuilderConfig({ config, onConfigChange, advancedOpen, setAdvancedOpen }: ConfigSectionProps) {
  return (
    <div className="space-y-4">
      {/* Grid Detection Mode */}
      <div className="space-y-2">
        <Label className="text-xs">Grid Detection Mode</Label>
        <Select
          value={(config.detection_mode as string) ?? "clustering"}
          onValueChange={(v) => onConfigChange("detection_mode", v)}
        >
          <SelectTrigger className="h-8">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="clustering">Clustering (DBSCAN)</SelectItem>
            <SelectItem value="threshold">Fixed Threshold</SelectItem>
            <SelectItem value="adaptive">Adaptive (Auto-detect)</SelectItem>
            <SelectItem value="shelf_lines">Shelf Line Detection</SelectItem>
            <SelectItem value="uniform">Uniform Grid</SelectItem>
          </SelectContent>
        </Select>
        <p className="text-xs text-muted-foreground">
          Algorithm for detecting rows and columns
        </p>
      </div>

      {/* Row Detection */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label className="text-xs">Row Tolerance (px)</Label>
          <span className="text-xs text-muted-foreground">
            {(config.row_tolerance as number) ?? 50}
          </span>
        </div>
        <Slider
          value={[(config.row_tolerance as number) ?? 50]}
          onValueChange={([v]) => onConfigChange("row_tolerance", v)}
          min={10}
          max={200}
          step={5}
          className="w-full"
        />
        <p className="text-xs text-muted-foreground">
          Y-coordinate tolerance for grouping into same row/shelf
        </p>
      </div>

      {/* Column Detection */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label className="text-xs">Column Tolerance (px)</Label>
          <span className="text-xs text-muted-foreground">
            {(config.col_tolerance as number) ?? 30}
          </span>
        </div>
        <Slider
          value={[(config.col_tolerance as number) ?? 30]}
          onValueChange={([v]) => onConfigChange("col_tolerance", v)}
          min={5}
          max={150}
          step={5}
          className="w-full"
        />
        <p className="text-xs text-muted-foreground">
          X-coordinate tolerance for column alignment
        </p>
      </div>

      {/* Sort Direction */}
      <div className="grid grid-cols-2 gap-3">
        <div className="space-y-2">
          <Label className="text-xs">Row Order</Label>
          <Select
            value={(config.row_order as string) ?? "top_to_bottom"}
            onValueChange={(v) => onConfigChange("row_order", v)}
          >
            <SelectTrigger className="h-8">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="top_to_bottom">Top → Bottom</SelectItem>
              <SelectItem value="bottom_to_top">Bottom → Top</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div className="space-y-2">
          <Label className="text-xs">Column Order</Label>
          <Select
            value={(config.col_order as string) ?? "left_to_right"}
            onValueChange={(v) => onConfigChange("col_order", v)}
          >
            <SelectTrigger className="h-8">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="left_to_right">Left → Right</SelectItem>
              <SelectItem value="right_to_left">Right → Left</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      {/* Gap Handling */}
      <div className="space-y-2">
        <Label className="text-xs">Gap Handling</Label>
        <Select
          value={(config.gap_handling as string) ?? "detect"}
          onValueChange={(v) => onConfigChange("gap_handling", v)}
        >
          <SelectTrigger className="h-8">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="detect">Detect Empty Cells</SelectItem>
            <SelectItem value="ignore">Ignore Gaps (Compact)</SelectItem>
            <SelectItem value="fill_null">Fill with Null</SelectItem>
            <SelectItem value="interpolate">Interpolate Position</SelectItem>
          </SelectContent>
        </Select>
        <p className="text-xs text-muted-foreground">
          How to handle empty spaces in the grid
        </p>
      </div>

      {/* Min Items per Row/Col */}
      <div className="grid grid-cols-2 gap-3">
        <div className="space-y-2">
          <Label className="text-xs">Min Items/Row</Label>
          <Input
            type="number"
            min={1}
            value={(config.min_items_per_row as number) ?? 1}
            onChange={(e) => onConfigChange("min_items_per_row", parseInt(e.target.value) || 1)}
            className="h-8"
          />
        </div>
        <div className="space-y-2">
          <Label className="text-xs">Min Rows</Label>
          <Input
            type="number"
            min={1}
            value={(config.min_rows as number) ?? 1}
            onChange={(e) => onConfigChange("min_rows", parseInt(e.target.value) || 1)}
            className="h-8"
          />
        </div>
      </div>

      {/* Output Format */}
      <div className="space-y-2">
        <Label className="text-xs">Output Format</Label>
        <Select
          value={(config.output_format as string) ?? "nested"}
          onValueChange={(v) => onConfigChange("output_format", v)}
        >
          <SelectTrigger className="h-8">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="nested">Nested Array [[row1], [row2]]</SelectItem>
            <SelectItem value="flat">Flat with Coordinates</SelectItem>
            <SelectItem value="realogram">Realogram Object</SelectItem>
            <SelectItem value="adjacency">Adjacency Map</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* Include Product Info */}
      <div className="flex items-center justify-between">
        <div>
          <Label className="text-xs">Include Product Metadata</Label>
          <p className="text-xs text-muted-foreground">
            Add class, confidence, bbox to each cell
          </p>
        </div>
        <Switch
          checked={(config.include_metadata as boolean) ?? true}
          onCheckedChange={(v) => onConfigChange("include_metadata", v)}
        />
      </div>

      {/* Advanced Section */}
      <Collapsible open={advancedOpen} onOpenChange={setAdvancedOpen}>
        <CollapsibleTrigger asChild>
          <Button variant="ghost" size="sm" className="w-full justify-between h-8">
            <span className="text-xs">Advanced Options</span>
            <ChevronDown className={`h-4 w-4 transition-transform ${advancedOpen ? "rotate-180" : ""}`} />
          </Button>
        </CollapsibleTrigger>
        <CollapsibleContent className="space-y-4 pt-2">
          {/* Coordinate System */}
          <div className="space-y-2">
            <Label className="text-xs">Position Reference</Label>
            <Select
              value={(config.position_ref as string) ?? "center"}
              onValueChange={(v) => onConfigChange("position_ref", v)}
            >
              <SelectTrigger className="h-8">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="center">Box Center</SelectItem>
                <SelectItem value="top_left">Top-Left Corner</SelectItem>
                <SelectItem value="bottom_center">Bottom-Center</SelectItem>
                <SelectItem value="centroid">Centroid (Mask)</SelectItem>
              </SelectContent>
            </Select>
            <p className="text-xs text-muted-foreground">
              Which point to use for grid positioning
            </p>
          </div>

          {/* Facing Detection */}
          <div className="flex items-center justify-between">
            <div>
              <Label className="text-xs">Detect Facings</Label>
              <p className="text-xs text-muted-foreground">
                Count product facings per cell
              </p>
            </div>
            <Switch
              checked={(config.detect_facings as boolean) ?? false}
              onCheckedChange={(v) => onConfigChange("detect_facings", v)}
            />
          </div>

          {/* Facing Threshold */}
          {Boolean(config.detect_facings) && (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label className="text-xs">Facing IoU Threshold</Label>
                <span className="text-xs text-muted-foreground">
                  {(config.facing_iou as number) ?? 0.3}
                </span>
              </div>
              <Slider
                value={[(config.facing_iou as number) ?? 0.3]}
                onValueChange={([v]) => onConfigChange("facing_iou", v)}
                min={0.1}
                max={0.8}
                step={0.05}
                className="w-full"
              />
              <p className="text-xs text-muted-foreground">
                Max overlap to count as separate facing
              </p>
            </div>
          )}

          {/* Shelf Edge Detection */}
          <div className="flex items-center justify-between">
            <div>
              <Label className="text-xs">Detect Shelf Edges</Label>
              <p className="text-xs text-muted-foreground">
                Find physical shelf boundaries
              </p>
            </div>
            <Switch
              checked={(config.detect_shelf_edges as boolean) ?? false}
              onCheckedChange={(v) => onConfigChange("detect_shelf_edges", v)}
            />
          </div>

          {/* Merge Overlapping */}
          <div className="flex items-center justify-between">
            <div>
              <Label className="text-xs">Merge Overlapping</Label>
              <p className="text-xs text-muted-foreground">
                Combine highly overlapping detections
              </p>
            </div>
            <Switch
              checked={(config.merge_overlapping as boolean) ?? false}
              onCheckedChange={(v) => onConfigChange("merge_overlapping", v)}
            />
          </div>

          {/* Merge Threshold */}
          {Boolean(config.merge_overlapping) && (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label className="text-xs">Merge IoU Threshold</Label>
                <span className="text-xs text-muted-foreground">
                  {(config.merge_iou as number) ?? 0.7}
                </span>
              </div>
              <Slider
                value={[(config.merge_iou as number) ?? 0.7]}
                onValueChange={([v]) => onConfigChange("merge_iou", v)}
                min={0.3}
                max={0.95}
                step={0.05}
                className="w-full"
              />
            </div>
          )}

          {/* Planogram Comparison */}
          <div className="flex items-center justify-between">
            <div>
              <Label className="text-xs">Enable Planogram Mode</Label>
              <p className="text-xs text-muted-foreground">
                Compare against reference layout
              </p>
            </div>
            <Switch
              checked={(config.planogram_mode as boolean) ?? false}
              onCheckedChange={(v) => onConfigChange("planogram_mode", v)}
            />
          </div>

          {/* Planogram Reference Field */}
          {Boolean(config.planogram_mode) && (
            <div className="space-y-2">
              <Label className="text-xs">Reference Planogram Field</Label>
              <Input
                type="text"
                value={(config.planogram_ref as string) ?? ""}
                onChange={(e) => onConfigChange("planogram_ref", e.target.value)}
                placeholder="e.g., parameters.expected_layout"
                className="h-8 font-mono text-xs"
              />
              <p className="text-xs text-muted-foreground">
                Field path to expected grid layout
              </p>
            </div>
          )}

          {/* Include Shelf Labels */}
          <div className="flex items-center justify-between">
            <div>
              <Label className="text-xs">Include Shelf Labels</Label>
              <p className="text-xs text-muted-foreground">
                Add shelf_1, shelf_2... labels
              </p>
            </div>
            <Switch
              checked={(config.shelf_labels as boolean) ?? true}
              onCheckedChange={(v) => onConfigChange("shelf_labels", v)}
            />
          </div>

          {/* Cell Numbering */}
          <div className="space-y-2">
            <Label className="text-xs">Cell Numbering</Label>
            <Select
              value={(config.cell_numbering as string) ?? "row_col"}
              onValueChange={(v) => onConfigChange("cell_numbering", v)}
            >
              <SelectTrigger className="h-8">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="row_col">Row-Column (1-1, 1-2...)</SelectItem>
                <SelectItem value="sequential">Sequential (1, 2, 3...)</SelectItem>
                <SelectItem value="excel">Excel Style (A1, A2, B1...)</SelectItem>
                <SelectItem value="none">No Numbering</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CollapsibleContent>
      </Collapsible>
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

// ============================================================================
// ForEach Config - Iteration Control
// ============================================================================
function ForEachConfig({ config, onConfigChange, advancedOpen, setAdvancedOpen }: ConfigSectionProps) {
  return (
    <div className="space-y-4">
      {/* Iteration Mode */}
      <div className="space-y-2">
        <Label className="text-xs">Iteration Mode</Label>
        <Select
          value={(config.iteration_mode as string) ?? "sequential"}
          onValueChange={(v) => onConfigChange("iteration_mode", v)}
        >
          <SelectTrigger className="h-8">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="sequential">Sequential (one by one)</SelectItem>
            <SelectItem value="parallel">Parallel (concurrent)</SelectItem>
            <SelectItem value="batch">Batch Processing</SelectItem>
          </SelectContent>
        </Select>
        <p className="text-xs text-muted-foreground">
          {config.iteration_mode === "parallel"
            ? "Process all items concurrently (faster, more resources)"
            : config.iteration_mode === "batch"
            ? "Process items in fixed-size batches"
            : "Process items one at a time (slower, predictable order)"}
        </p>
      </div>

      {/* Batch Size (only for batch mode) */}
      {config.iteration_mode === "batch" && (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label className="text-xs">Batch Size</Label>
            <span className="text-xs text-muted-foreground">
              {(config.batch_size as number) ?? 10}
            </span>
          </div>
          <Slider
            value={[(config.batch_size as number) ?? 10]}
            onValueChange={([v]) => onConfigChange("batch_size", v)}
            min={1}
            max={100}
            step={1}
            className="w-full"
          />
        </div>
      )}

      {/* Parallel Concurrency (only for parallel mode) */}
      {config.iteration_mode === "parallel" && (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label className="text-xs">Max Concurrency</Label>
            <span className="text-xs text-muted-foreground">
              {(config.max_concurrency as number) ?? 5}
            </span>
          </div>
          <Slider
            value={[(config.max_concurrency as number) ?? 5]}
            onValueChange={([v]) => onConfigChange("max_concurrency", v)}
            min={1}
            max={50}
            step={1}
            className="w-full"
          />
          <p className="text-xs text-muted-foreground">
            Maximum parallel iterations running at once
          </p>
        </div>
      )}

      {/* Limit Items */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label className="text-xs">Max Items to Process</Label>
          <span className="text-xs text-muted-foreground">
            {(config.max_items as number) === 0 ? "Unlimited" : (config.max_items as number) ?? "Unlimited"}
          </span>
        </div>
        <Slider
          value={[(config.max_items as number) ?? 0]}
          onValueChange={([v]) => onConfigChange("max_items", v)}
          min={0}
          max={1000}
          step={1}
          className="w-full"
        />
        <p className="text-xs text-muted-foreground">
          0 = process all items, or set a limit
        </p>
      </div>

      {/* Start Index */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label className="text-xs">Start from Index</Label>
          <span className="text-xs text-muted-foreground">
            {(config.start_index as number) ?? 0}
          </span>
        </div>
        <ParamSlider
          value={(config.start_index as number) ?? 0}
          onChange={(v) => onConfigChange("start_index", v)}
          min={0}
          max={100}
          step={1}
        />
        <p className="text-xs text-muted-foreground">
          Skip first N items (useful for resuming)
        </p>
      </div>

      {/* Advanced Options */}
      <Collapsible open={advancedOpen} onOpenChange={setAdvancedOpen}>
        <CollapsibleTrigger asChild>
          <Button variant="ghost" size="sm" className="w-full justify-between h-8">
            <span className="text-xs">Advanced Options</span>
            <ChevronDown className={`h-4 w-4 transition-transform ${advancedOpen ? "rotate-180" : ""}`} />
          </Button>
        </CollapsibleTrigger>
        <CollapsibleContent className="space-y-4 pt-2">
          {/* Error Handling */}
          <div className="space-y-2">
            <Label className="text-xs">On Error</Label>
            <Select
              value={(config.on_error as string) ?? "continue"}
              onValueChange={(v) => onConfigChange("on_error", v)}
            >
              <SelectTrigger className="h-8">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="continue">Continue (skip failed item)</SelectItem>
                <SelectItem value="stop">Stop (halt iteration)</SelectItem>
                <SelectItem value="collect_errors">Collect Errors (output errors separately)</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Include Metadata */}
          <div className="flex items-center justify-between">
            <div>
              <Label className="text-xs">Include Iteration Metadata</Label>
              <p className="text-xs text-muted-foreground">
                Add index, is_first, is_last to output
              </p>
            </div>
            <Switch
              checked={(config.include_metadata as boolean) ?? true}
              onCheckedChange={(v) => onConfigChange("include_metadata", v)}
            />
          </div>

          {/* Timeout per Item */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label className="text-xs">Timeout per Item (ms)</Label>
              <span className="text-xs text-muted-foreground">
                {(config.item_timeout as number) ?? 30000}ms
              </span>
            </div>
            <Slider
              value={[(config.item_timeout as number) ?? 30000]}
              onValueChange={([v]) => onConfigChange("item_timeout", v)}
              min={1000}
              max={300000}
              step={1000}
              className="w-full"
            />
          </div>
        </CollapsibleContent>
      </Collapsible>

      {/* Info Box */}
      <div className="text-xs text-muted-foreground bg-muted/50 rounded-md p-3">
        <p className="font-medium text-foreground mb-1">Outputs per Iteration</p>
        <p>
          <strong>item:</strong> Current array element<br />
          <strong>index:</strong> Current position (0-based)<br />
          <strong>total:</strong> Total number of items<br />
          <strong>is_first / is_last:</strong> Position flags
        </p>
      </div>
    </div>
  );
}

// ============================================================================
// Collect Config - Aggregation Control
// ============================================================================
function CollectConfig({ config, onConfigChange, advancedOpen, setAdvancedOpen }: ConfigSectionProps) {
  return (
    <div className="space-y-4">
      {/* Collection Mode */}
      <div className="space-y-2">
        <Label className="text-xs">Collection Mode</Label>
        <Select
          value={(config.collection_mode as string) ?? "array"}
          onValueChange={(v) => onConfigChange("collection_mode", v)}
        >
          <SelectTrigger className="h-8">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="array">Array (ordered list)</SelectItem>
            <SelectItem value="object">Object (keyed by field)</SelectItem>
            <SelectItem value="first">First Only</SelectItem>
            <SelectItem value="last">Last Only</SelectItem>
            <SelectItem value="unique">Unique Values</SelectItem>
          </SelectContent>
        </Select>
        <p className="text-xs text-muted-foreground">
          {config.collection_mode === "object"
            ? "Group items by a key field into an object"
            : config.collection_mode === "first"
            ? "Return only the first collected item"
            : config.collection_mode === "last"
            ? "Return only the last collected item"
            : config.collection_mode === "unique"
            ? "Remove duplicate values"
            : "Collect all items into an ordered array"}
        </p>
      </div>

      {/* Key Field (for object mode) */}
      {config.collection_mode === "object" && (
        <div className="space-y-2">
          <Label className="text-xs">Key Field</Label>
          <Input
            type="text"
            value={(config.key_field as string) ?? ""}
            onChange={(e) => onConfigChange("key_field", e.target.value)}
            placeholder="e.g., id, class_name, index"
            className="h-8 font-mono text-xs"
          />
          <p className="text-xs text-muted-foreground">
            Field to use as object key for each item
          </p>
        </div>
      )}

      {/* Filter Nulls */}
      <div className="flex items-center justify-between">
        <div>
          <Label className="text-xs">Filter Null/Empty Values</Label>
          <p className="text-xs text-muted-foreground">
            Exclude null, undefined, empty from results
          </p>
        </div>
        <Switch
          checked={(config.filter_nulls as boolean) ?? true}
          onCheckedChange={(v) => onConfigChange("filter_nulls", v)}
        />
      </div>

      {/* Flatten */}
      <div className="flex items-center justify-between">
        <div>
          <Label className="text-xs">Flatten Nested Arrays</Label>
          <p className="text-xs text-muted-foreground">
            If item is an array, spread its elements
          </p>
        </div>
        <Switch
          checked={(config.flatten as boolean) ?? false}
          onCheckedChange={(v) => onConfigChange("flatten", v)}
        />
      </div>

      {/* Flatten Depth */}
      {config.flatten && (
        <div className="space-y-2 pl-4 border-l-2 border-muted">
          <div className="flex items-center justify-between">
            <Label className="text-xs">Flatten Depth</Label>
            <span className="text-xs text-muted-foreground">
              {(config.flatten_depth as number) ?? 1}
            </span>
          </div>
          <Slider
            value={[(config.flatten_depth as number) ?? 1]}
            onValueChange={([v]) => onConfigChange("flatten_depth", v)}
            min={1}
            max={10}
            step={1}
            className="w-full"
          />
        </div>
      )}

      {/* Advanced Options */}
      <Collapsible open={advancedOpen} onOpenChange={setAdvancedOpen}>
        <CollapsibleTrigger asChild>
          <Button variant="ghost" size="sm" className="w-full justify-between h-8">
            <span className="text-xs">Advanced Options</span>
            <ChevronDown className={`h-4 w-4 transition-transform ${advancedOpen ? "rotate-180" : ""}`} />
          </Button>
        </CollapsibleTrigger>
        <CollapsibleContent className="space-y-4 pt-2">
          {/* Sort Results */}
          <div className="flex items-center justify-between">
            <div>
              <Label className="text-xs">Sort Results</Label>
              <p className="text-xs text-muted-foreground">Order collected items</p>
            </div>
            <Switch
              checked={(config.sort_results as boolean) ?? false}
              onCheckedChange={(v) => onConfigChange("sort_results", v)}
            />
          </div>

          {config.sort_results && (
            <div className="space-y-3 pl-4 border-l-2 border-muted">
              <div className="space-y-2">
                <Label className="text-xs">Sort By</Label>
                <Input
                  type="text"
                  value={(config.sort_field as string) ?? ""}
                  onChange={(e) => onConfigChange("sort_field", e.target.value)}
                  placeholder="e.g., confidence, index, name"
                  className="h-8 font-mono text-xs"
                />
              </div>
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
                    <SelectItem value="asc">Ascending (A→Z, 0→9)</SelectItem>
                    <SelectItem value="desc">Descending (Z→A, 9→0)</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          )}

          {/* Limit Results */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label className="text-xs">Limit Results</Label>
              <span className="text-xs text-muted-foreground">
                {(config.limit as number) === 0 ? "No limit" : (config.limit as number) ?? "No limit"}
              </span>
            </div>
            <Slider
              value={[(config.limit as number) ?? 0]}
              onValueChange={([v]) => onConfigChange("limit", v)}
              min={0}
              max={1000}
              step={1}
              className="w-full"
            />
          </div>

          {/* Preserve Order */}
          <div className="flex items-center justify-between">
            <div>
              <Label className="text-xs">Preserve Original Order</Label>
              <p className="text-xs text-muted-foreground">
                Maintain iteration order (for parallel)
              </p>
            </div>
            <Switch
              checked={(config.preserve_order as boolean) ?? true}
              onCheckedChange={(v) => onConfigChange("preserve_order", v)}
            />
          </div>
        </CollapsibleContent>
      </Collapsible>

      {/* Info Box */}
      <div className="text-xs text-muted-foreground bg-muted/50 rounded-md p-3">
        <p className="font-medium text-foreground mb-1">Outputs</p>
        <p>
          <strong>results:</strong> Collected items (array or object)<br />
          <strong>count:</strong> Number of collected items
        </p>
      </div>
    </div>
  );
}

// ============================================================================
// Map Config - Array Transformation
// ============================================================================
function MapConfig({ config, onConfigChange, advancedOpen, setAdvancedOpen }: ConfigSectionProps) {
  return (
    <div className="space-y-4">
      {/* Transform Type */}
      <div className="space-y-2">
        <Label className="text-xs">Transform Type</Label>
        <Select
          value={(config.transform_type as string) ?? "extract"}
          onValueChange={(v) => onConfigChange("transform_type", v)}
        >
          <SelectTrigger className="h-8">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="extract">Extract Field</SelectItem>
            <SelectItem value="rename">Rename Fields</SelectItem>
            <SelectItem value="pick">Pick Fields (keep only)</SelectItem>
            <SelectItem value="omit">Omit Fields (remove)</SelectItem>
            <SelectItem value="compute">Compute Expression</SelectItem>
            <SelectItem value="convert">Type Conversion</SelectItem>
            <SelectItem value="template">String Template</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* Extract Field */}
      {config.transform_type === "extract" && (
        <div className="space-y-2">
          <Label className="text-xs">Field to Extract</Label>
          <Input
            type="text"
            value={(config.extract_field as string) ?? ""}
            onChange={(e) => onConfigChange("extract_field", e.target.value)}
            placeholder="e.g., bbox, confidence, class_name"
            className="h-8 font-mono text-xs"
          />
          <p className="text-xs text-muted-foreground">
            Extract a single field from each object
          </p>
        </div>
      )}

      {/* Rename Fields */}
      {config.transform_type === "rename" && (
        <div className="space-y-2">
          <Label className="text-xs">Field Mappings</Label>
          <Input
            type="text"
            value={(config.rename_mappings as string) ?? ""}
            onChange={(e) => onConfigChange("rename_mappings", e.target.value)}
            placeholder="old1:new1, old2:new2"
            className="h-8 font-mono text-xs"
          />
          <p className="text-xs text-muted-foreground">
            Rename fields: oldName:newName, separated by commas
          </p>
        </div>
      )}

      {/* Pick Fields */}
      {config.transform_type === "pick" && (
        <div className="space-y-2">
          <Label className="text-xs">Fields to Keep</Label>
          <Input
            type="text"
            value={(config.pick_fields as string) ?? ""}
            onChange={(e) => onConfigChange("pick_fields", e.target.value)}
            placeholder="e.g., id, confidence, bbox"
            className="h-8 font-mono text-xs"
          />
          <p className="text-xs text-muted-foreground">
            Only keep these fields (comma-separated)
          </p>
        </div>
      )}

      {/* Omit Fields */}
      {config.transform_type === "omit" && (
        <div className="space-y-2">
          <Label className="text-xs">Fields to Remove</Label>
          <Input
            type="text"
            value={(config.omit_fields as string) ?? ""}
            onChange={(e) => onConfigChange("omit_fields", e.target.value)}
            placeholder="e.g., _internal, metadata, temp"
            className="h-8 font-mono text-xs"
          />
          <p className="text-xs text-muted-foreground">
            Remove these fields from each object
          </p>
        </div>
      )}

      {/* Compute Expression */}
      {config.transform_type === "compute" && (
        <div className="space-y-2">
          <Label className="text-xs">Expression</Label>
          <Input
            type="text"
            value={(config.compute_expression as string) ?? ""}
            onChange={(e) => onConfigChange("compute_expression", e.target.value)}
            placeholder="e.g., item.width * item.height"
            className="h-8 font-mono text-xs"
          />
          <p className="text-xs text-muted-foreground">
            JavaScript expression. Use "item" for current element.
          </p>
        </div>
      )}

      {/* Type Conversion */}
      {config.transform_type === "convert" && (
        <div className="space-y-3">
          <div className="space-y-2">
            <Label className="text-xs">Source Field</Label>
            <Input
              type="text"
              value={(config.convert_field as string) ?? ""}
              onChange={(e) => onConfigChange("convert_field", e.target.value)}
              placeholder="e.g., confidence"
              className="h-8 font-mono text-xs"
            />
          </div>
          <div className="space-y-2">
            <Label className="text-xs">Target Type</Label>
            <Select
              value={(config.target_type as string) ?? "string"}
              onValueChange={(v) => onConfigChange("target_type", v)}
            >
              <SelectTrigger className="h-8">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="string">String</SelectItem>
                <SelectItem value="number">Number</SelectItem>
                <SelectItem value="boolean">Boolean</SelectItem>
                <SelectItem value="array">Array (split string)</SelectItem>
                <SelectItem value="bbox_array">BBox Array [x1,y1,x2,y2]</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
      )}

      {/* String Template */}
      {config.transform_type === "template" && (
        <div className="space-y-2">
          <Label className="text-xs">Template String</Label>
          <Input
            type="text"
            value={(config.template_string as string) ?? ""}
            onChange={(e) => onConfigChange("template_string", e.target.value)}
            placeholder="e.g., {class_name}: {confidence}%"
            className="h-8 font-mono text-xs"
          />
          <p className="text-xs text-muted-foreground">
            Use {"{field}"} to insert values from each item
          </p>
        </div>
      )}

      {/* Add/Set Fields */}
      <div className="space-y-2">
        <Label className="text-xs">Add/Set Field (Optional)</Label>
        <div className="flex gap-2">
          <Input
            type="text"
            value={(config.add_field as string) ?? ""}
            onChange={(e) => onConfigChange("add_field", e.target.value)}
            placeholder="field name"
            className="h-8 flex-1 font-mono text-xs"
          />
          <Input
            type="text"
            value={(config.add_value as string) ?? ""}
            onChange={(e) => onConfigChange("add_value", e.target.value)}
            placeholder="value"
            className="h-8 flex-1 font-mono text-xs"
          />
        </div>
        <p className="text-xs text-muted-foreground">
          Add a constant field to each item
        </p>
      </div>

      {/* Advanced Options */}
      <Collapsible open={advancedOpen} onOpenChange={setAdvancedOpen}>
        <CollapsibleTrigger asChild>
          <Button variant="ghost" size="sm" className="w-full justify-between h-8">
            <span className="text-xs">Advanced Options</span>
            <ChevronDown className={`h-4 w-4 transition-transform ${advancedOpen ? "rotate-180" : ""}`} />
          </Button>
        </CollapsibleTrigger>
        <CollapsibleContent className="space-y-4 pt-2">
          {/* Filter Nulls */}
          <div className="flex items-center justify-between">
            <div>
              <Label className="text-xs">Remove Null Results</Label>
              <p className="text-xs text-muted-foreground">
                Filter out null/undefined after transform
              </p>
            </div>
            <Switch
              checked={(config.filter_nulls as boolean) ?? false}
              onCheckedChange={(v) => onConfigChange("filter_nulls", v)}
            />
          </div>

          {/* Preserve Original */}
          <div className="flex items-center justify-between">
            <div>
              <Label className="text-xs">Merge with Original</Label>
              <p className="text-xs text-muted-foreground">
                Keep original fields, add transformed
              </p>
            </div>
            <Switch
              checked={(config.merge_original as boolean) ?? false}
              onCheckedChange={(v) => onConfigChange("merge_original", v)}
            />
          </div>

          {/* Deep Clone */}
          <div className="flex items-center justify-between">
            <div>
              <Label className="text-xs">Deep Clone Objects</Label>
              <p className="text-xs text-muted-foreground">
                Create new objects (don't mutate input)
              </p>
            </div>
            <Switch
              checked={(config.deep_clone as boolean) ?? true}
              onCheckedChange={(v) => onConfigChange("deep_clone", v)}
            />
          </div>
        </CollapsibleContent>
      </Collapsible>

      {/* Info Box */}
      <div className="text-xs text-muted-foreground bg-muted/50 rounded-md p-3">
        <p className="font-medium text-foreground mb-1">Outputs</p>
        <p>
          <strong>results:</strong> Transformed array<br />
          <strong>count:</strong> Number of items
        </p>
      </div>
    </div>
  );
}
