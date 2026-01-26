"use client";

import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Variable, Sliders } from "lucide-react";
import { useState, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { apiClient } from "@/lib/api-client";

// =============================================================================
// Types
// =============================================================================

export interface WorkflowModel {
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

// =============================================================================
// Hooks
// =============================================================================

/**
 * Hook to fetch workflow models for a specific category
 */
export function useWorkflowModels(category: "detection" | "classification" | "embedding" | "segmentation") {
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
// Parameter Utilities
// =============================================================================

/**
 * Check if a value is a parameter reference
 */
export function isParamRef(value: unknown): boolean {
  if (typeof value !== "string") return false;
  return value.includes("{{") && value.includes("}}");
}

// =============================================================================
// Parameter-aware Input Components
// =============================================================================

export interface ParamSliderProps {
  value: unknown;
  onChange: (value: unknown) => void;
  min: number;
  max: number;
  step: number;
  formatValue?: (v: number) => string;
  label?: string;
}

/**
 * ParamSlider - A slider that can switch to parameter mode
 */
export function ParamSlider({
  value,
  onChange,
  min,
  max,
  step,
  formatValue,
}: ParamSliderProps) {
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

export interface ParamInputProps {
  value: unknown;
  onChange: (value: unknown) => void;
  type?: "text" | "number";
  placeholder?: string;
  className?: string;
  min?: number;
  max?: number;
}

/**
 * ParamInput - A text/number input that shows parameter hint
 */
export function ParamInput({
  value,
  onChange,
  type = "text",
  placeholder,
  className,
  min,
  max,
}: ParamInputProps) {
  const isParam = isParamRef(value);
  const stringValue = value !== undefined && value !== null ? String(value) : "";
  const hasValue = stringValue !== "";

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
        className={`${className || "h-8"} ${isParam ? "font-mono text-xs pr-16" : hasValue ? "pr-8" : ""}`}
        min={min}
        max={max}
      />
      {hasValue && (
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="ghost"
                size="icon"
                className="absolute right-1 top-1/2 -translate-y-1/2 h-6 w-6 hover:bg-destructive/10"
                onClick={() => onChange(undefined)}
              >
                <span className="text-xs text-muted-foreground hover:text-destructive">✕</span>
              </Button>
            </TooltipTrigger>
            <TooltipContent>Clear value</TooltipContent>
          </Tooltip>
        </TooltipProvider>
      )}
      {isParam && (
        <Variable className="absolute right-9 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-blue-500" />
      )}
    </div>
  );
}
