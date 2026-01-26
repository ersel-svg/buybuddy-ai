"use client";

import { Label } from "@/components/ui/label";
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
import { Database, Sparkles } from "lucide-react";
import { useMemo } from "react";
import { useWorkflowModels, WorkflowModel } from "./utils";

// =============================================================================
// Types
// =============================================================================

export type ModelCategory = "detection" | "classification" | "embedding" | "segmentation";

export interface ModelItem {
  id: string;
  name: string;
  model_type: string;
  source: "pretrained" | "trained";
  provider?: string;
  is_default?: boolean;
  metrics?: {
    map?: number;
    accuracy?: number;
    recall_at_1?: number;
  };
  embedding_dim?: number;
  class_count?: number;
}

export interface ModelSelectorProps {
  value: string | undefined;
  onChange: (modelId: string, model?: WorkflowModel) => void;
  category: ModelCategory;
  label?: string;
  description?: string;
  placeholder?: string;
}

// =============================================================================
// Helper Hook
// =============================================================================

/**
 * Hook to get model info and metadata
 */
export function useModelInfo(
  modelId: string | undefined,
  category: ModelCategory
): {
  model: WorkflowModel | undefined;
  isOpenVocabulary: boolean;
  displayText: string;
} {
  const { data: models } = useWorkflowModels(category);

  const model = useMemo(() => {
    if (!modelId || !models) return undefined;
    return models.find((m) => m.id === modelId);
  }, [modelId, models]);

  // Check if the model supports open-vocabulary detection (text prompts)
  const isOpenVocabulary = useMemo(() => {
    if (!modelId || category !== "detection") return false;
    const modelIdLower = modelId.toLowerCase();
    return (
      modelIdLower.includes("grounding") ||
      modelIdLower.includes("dino") ||
      modelIdLower.includes("owl") ||
      modelIdLower.includes("glip") ||
      modelIdLower.includes("yolo-world") ||
      modelIdLower.includes("florence")
    );
  }, [modelId, category]);

  const displayText = useMemo(() => {
    if (!model) return "";
    const sourceText = model.source === "pretrained" ? "Pretrained" : "Trained";
    return `${sourceText} ${model.provider || model.model_type} model`;
  }, [model]);

  return { model, isOpenVocabulary, displayText };
}

// =============================================================================
// Component
// =============================================================================

/**
 * ModelSelector - Dropdown for selecting ML models by category
 *
 * Features:
 * - Groups models by source (pretrained/trained)
 * - Shows metrics for trained models
 * - Marks default models
 * - Loading state
 */
export function ModelSelector({
  value,
  onChange,
  category,
  label = "Model",
  description,
  placeholder = "Select a model...",
}: ModelSelectorProps) {
  const { data: models, isLoading } = useWorkflowModels(category);

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
    if (!value || !models) return undefined;
    return models.find((m) => m.id === value);
  }, [value, models]);

  // Handle model selection
  const handleModelSelect = (modelId: string) => {
    const model = models?.find((m) => m.id === modelId);
    onChange(modelId, model);
  };

  // Get category-specific default description
  const defaultDescription = useMemo(() => {
    switch (category) {
      case "detection":
        return "Select a detection model (YOLO, RT-DETR, D-FINE, RF-DETR...)";
      case "classification":
        return "Select a classification model (ResNet, EfficientNet, ViT...)";
      case "embedding":
        return "Select an embedding model (CLIP, DINOv2, ResNet...)";
      case "segmentation":
        return "Select a segmentation model (SAM, YOLO-Seg...)";
      default:
        return `Select a ${category} model`;
    }
  }, [category]);

  // Format metrics for trained models
  const formatMetrics = (model: WorkflowModel): string => {
    const parts: string[] = [];
    if (model.provider || model.model_type) {
      parts.push(model.provider || model.model_type);
    }
    if (model.metrics?.map) {
      parts.push(`mAP: ${(model.metrics.map * 100).toFixed(1)}%`);
    }
    if (model.metrics?.accuracy) {
      parts.push(`Acc: ${(model.metrics.accuracy * 100).toFixed(1)}%`);
    }
    if (model.metrics?.recall_at_1) {
      parts.push(`R@1: ${(model.metrics.recall_at_1 * 100).toFixed(1)}%`);
    }
    if (model.class_count) {
      parts.push(`${model.class_count} classes`);
    }
    if (model.embedding_dim) {
      parts.push(`${model.embedding_dim}d`);
    }
    return parts.join(" • ");
  };

  return (
    <div className="space-y-2">
      <Label className="text-xs">{label}</Label>
      {isLoading ? (
        <Skeleton className="h-8 w-full" />
      ) : (
        <Select value={value ?? ""} onValueChange={handleModelSelect}>
          <SelectTrigger className="h-8">
            <SelectValue placeholder={placeholder}>
              {selectedModel && (
                <div className="flex items-center gap-2">
                  {selectedModel.source === "pretrained" ? (
                    <Sparkles className="h-3 w-3 text-purple-500" />
                  ) : (
                    <Database className="h-3 w-3 text-blue-500" />
                  )}
                  <span className="truncate">{selectedModel.name}</span>
                  {selectedModel.is_default && (
                    <Badge variant="secondary" className="text-[10px] py-0 px-1">
                      Default
                    </Badge>
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
                        <Badge variant="secondary" className="text-[10px] py-0 px-1">
                          Default
                        </Badge>
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
                          <Badge variant="secondary" className="text-[10px] py-0 px-1">
                            Default
                          </Badge>
                        )}
                      </div>
                      <span className="text-xs text-muted-foreground">
                        {formatMetrics(model)}
                      </span>
                    </div>
                  </SelectItem>
                ))}
              </SelectGroup>
            )}

            {/* Empty state */}
            {groupedModels.pretrained.length === 0 &&
              groupedModels.trained.length === 0 && (
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
          : description || defaultDescription}
      </p>
    </div>
  );
}
