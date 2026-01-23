"use client";

import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { apiClient } from "@/lib/api-client";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Database, Sparkles } from "lucide-react";

export interface ModelItem {
  id: string;
  name: string;
  model_type: string;
  category: string;
  source: "pretrained" | "trained";
  is_active: boolean;
  is_default?: boolean;
  checkpoint_url?: string;
  class_mapping?: Record<string, string>;
  created_at?: string;
}

export interface ModelSelectorProps {
  /** The currently selected model ID */
  value?: string;
  /** Callback when model selection changes */
  onValueChange: (modelId: string, model: ModelItem | undefined) => void;
  /** Filter models by category (detection, classification, embedding) */
  category?: "detection" | "classification" | "embedding" | "segmentation";
  /** Filter models by source */
  source?: "pretrained" | "trained";
  /** Placeholder text */
  placeholder?: string;
  /** Whether the selector is disabled */
  disabled?: boolean;
  /** Include inactive models */
  includeInactive?: boolean;
  /** Custom class name for the trigger */
  className?: string;
}

/**
 * ModelSelector - A reusable model picker for workflow blocks
 *
 * Features:
 * - Groups models by source (pretrained/trained)
 * - Shows model type, name, and metadata
 * - Supports filtering by category
 * - Highlights default models
 */
export function ModelSelector({
  value,
  onValueChange,
  category,
  source,
  placeholder = "Select model...",
  disabled = false,
  includeInactive = false,
  className,
}: ModelSelectorProps) {
  // Fetch models list
  const { data, isLoading } = useQuery({
    queryKey: ["workflow-models-list", category, source, includeInactive],
    queryFn: () =>
      apiClient.getWorkflowModelsList({
        model_type: category,
        source,
        include_inactive: includeInactive,
      }),
  });

  // Extract items for memoization
  const items = data?.items;

  // Group models by source
  const groupedModels = useMemo(() => {
    if (!items) return { pretrained: [], trained: [] };

    return {
      pretrained: items.filter((m) => m.source === "pretrained"),
      trained: items.filter((m) => m.source === "trained"),
    };
  }, [items]);

  // Find selected model
  const selectedModel = useMemo(() => {
    if (!value || !items) return undefined;
    return items.find((m) => m.id === value);
  }, [value, items]);

  // Handle value change
  const handleValueChange = (modelId: string) => {
    const model = data?.items?.find((m) => m.id === modelId);
    onValueChange(modelId, model);
  };

  if (isLoading) {
    return <Skeleton className="h-10 w-full" />;
  }

  const hasPretrainedModels = groupedModels.pretrained.length > 0;
  const hasTrainedModels = groupedModels.trained.length > 0;

  return (
    <Select
      value={value}
      onValueChange={handleValueChange}
      disabled={disabled}
    >
      <SelectTrigger className={className}>
        <SelectValue placeholder={placeholder}>
          {selectedModel && (
            <div className="flex items-center gap-2">
              {selectedModel.source === "pretrained" ? (
                <Sparkles className="h-3.5 w-3.5 text-purple-500" />
              ) : (
                <Database className="h-3.5 w-3.5 text-blue-500" />
              )}
              <span className="truncate">{selectedModel.name}</span>
              {selectedModel.is_default && (
                <Badge variant="secondary" className="text-xs py-0 px-1">
                  Default
                </Badge>
              )}
            </div>
          )}
        </SelectValue>
      </SelectTrigger>
      <SelectContent>
        {/* Pretrained Models Group */}
        {hasPretrainedModels && (
          <SelectGroup>
            <SelectLabel className="flex items-center gap-2">
              <Sparkles className="h-3.5 w-3.5 text-purple-500" />
              Pretrained Models
            </SelectLabel>
            {groupedModels.pretrained.map((model) => (
              <SelectItem key={model.id} value={model.id}>
                <div className="flex items-center gap-2 w-full">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="truncate">{model.name}</span>
                      {model.is_default && (
                        <Badge variant="secondary" className="text-xs py-0 px-1">
                          Default
                        </Badge>
                      )}
                    </div>
                    <div className="text-xs text-muted-foreground">
                      {model.model_type}
                    </div>
                  </div>
                </div>
              </SelectItem>
            ))}
          </SelectGroup>
        )}

        {/* Trained Models Group */}
        {hasTrainedModels && (
          <SelectGroup>
            <SelectLabel className="flex items-center gap-2">
              <Database className="h-3.5 w-3.5 text-blue-500" />
              Trained Models
            </SelectLabel>
            {groupedModels.trained.map((model) => (
              <SelectItem key={model.id} value={model.id}>
                <div className="flex items-center gap-2 w-full">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="truncate">{model.name}</span>
                      {model.is_default && (
                        <Badge variant="secondary" className="text-xs py-0 px-1">
                          Default
                        </Badge>
                      )}
                    </div>
                    <div className="text-xs text-muted-foreground flex items-center gap-1">
                      <span>{model.model_type}</span>
                      {model.class_mapping && (
                        <span className="text-muted-foreground/60">
                          ({Object.keys(model.class_mapping).length} classes)
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              </SelectItem>
            ))}
          </SelectGroup>
        )}

        {/* Empty state */}
        {!hasPretrainedModels && !hasTrainedModels && (
          <div className="py-6 text-center text-sm text-muted-foreground">
            No models available
          </div>
        )}
      </SelectContent>
    </Select>
  );
}

/**
 * Hook to get model info by ID
 */
export function useModelInfo(modelId: string | undefined) {
  const { data } = useQuery({
    queryKey: ["workflow-models-list"],
    queryFn: () => apiClient.getWorkflowModelsList(),
    enabled: !!modelId,
  });

  const items = data?.items;

  return useMemo(() => {
    if (!modelId || !items) return undefined;
    return items.find((m) => m.id === modelId);
  }, [modelId, items]);
}
