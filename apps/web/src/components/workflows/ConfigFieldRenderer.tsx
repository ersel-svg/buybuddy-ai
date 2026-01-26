"use client";

/**
 * ConfigFieldRenderer
 *
 * Renders config fields from block registry definitions.
 * Supports all field types: string, number, boolean, select, model, slider, array, object, code.
 */

import { useState, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Textarea } from "@/components/ui/textarea";
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
import { ChevronDown, Variable, Sliders, Plus, X, Database, Sparkles } from "lucide-react";
import { apiClient } from "@/lib/api-client";
import type {
  ConfigField,
  StringConfigField,
  NumberConfigField,
  BooleanConfigField,
  SelectConfigField,
  ModelConfigField,
  SliderConfigField,
  ArrayConfigField,
  CodeConfigField,
} from "@/lib/workflow/types";
import { shouldShowField } from "@/lib/workflow";

// =============================================================================
// Types
// =============================================================================

interface ConfigFieldRendererProps {
  fields: ConfigField[];
  config: Record<string, unknown>;
  onConfigChange: (keyOrObject: string | Record<string, unknown>, value?: unknown) => void;
  showAdvanced?: boolean;
}

interface FieldProps<T extends ConfigField> {
  field: T;
  value: unknown;
  onChange: (value: unknown) => void;
}

// =============================================================================
// Helper: Check if value is a parameter reference
// =============================================================================

function isParamRef(value: unknown): boolean {
  if (typeof value !== "string") return false;
  return value.includes("{{") && value.includes("}}");
}

// =============================================================================
// Field Components
// =============================================================================

/** String Input Field */
function StringField({ field, value, onChange }: FieldProps<StringConfigField>) {
  const strValue = typeof value === "string" ? value : "";

  return (
    <div className="space-y-1.5">
      <Label className="text-xs">{field.label}</Label>
      <Input
        value={strValue}
        onChange={(e) => onChange(e.target.value)}
        placeholder={field.placeholder}
        className="h-8"
      />
      {field.description && (
        <p className="text-xs text-muted-foreground">{field.description}</p>
      )}
    </div>
  );
}

/** Number Input Field */
function NumberField({ field, value, onChange }: FieldProps<NumberConfigField>) {
  const numValue = typeof value === "number" ? value : (field.default ?? 0);

  return (
    <div className="space-y-1.5">
      <Label className="text-xs">{field.label}</Label>
      <Input
        type="number"
        value={numValue}
        onChange={(e) => onChange(parseFloat(e.target.value) || 0)}
        min={field.min}
        max={field.max}
        step={field.step}
        className="h-8"
      />
      {field.description && (
        <p className="text-xs text-muted-foreground">{field.description}</p>
      )}
    </div>
  );
}

/** Boolean Switch Field */
function BooleanField({ field, value, onChange }: FieldProps<BooleanConfigField>) {
  const boolValue = typeof value === "boolean" ? value : (field.default ?? false);

  return (
    <div className="flex items-center justify-between py-1">
      <div className="space-y-0.5">
        <Label className="text-xs">{field.label}</Label>
        {field.description && (
          <p className="text-xs text-muted-foreground">{field.description}</p>
        )}
      </div>
      <Switch checked={boolValue} onCheckedChange={onChange} />
    </div>
  );
}

/** Select Dropdown Field */
function SelectField({ field, value, onChange }: FieldProps<SelectConfigField>) {
  const strValue = typeof value === "string" ? value : (field.default ?? "");

  return (
    <div className="space-y-1.5">
      <Label className="text-xs">{field.label}</Label>
      <Select value={strValue} onValueChange={onChange}>
        <SelectTrigger className="h-8">
          <SelectValue placeholder="Select..." />
        </SelectTrigger>
        <SelectContent>
          {field.options.map((opt) => (
            <SelectItem key={opt.value} value={opt.value}>
              <div className="flex flex-col">
                <span>{opt.label}</span>
                {opt.description && (
                  <span className="text-xs text-muted-foreground">{opt.description}</span>
                )}
              </div>
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      {field.description && (
        <p className="text-xs text-muted-foreground">{field.description}</p>
      )}
    </div>
  );
}

/** Model Selector Field */
function ModelField({ field, value, onChange }: FieldProps<ModelConfigField>) {
  const modelId = typeof value === "string" ? value : "";

  // Fetch models from API
  const { data: models, isLoading } = useQuery({
    queryKey: ["workflow-models-list", field.category],
    queryFn: async () => {
      const result = await apiClient.getWorkflowModelsList({ model_type: field.category });
      return result.items as Array<{
        id: string;
        name: string;
        model_type: string;
        source: "pretrained" | "trained";
        is_default?: boolean;
        metrics?: { map?: number; accuracy?: number };
        class_count?: number;
      }>;
    },
    staleTime: 5 * 60 * 1000,
  });

  const groupedModels = useMemo(() => {
    if (!models) return { pretrained: [], trained: [] };
    return {
      pretrained: models.filter((m) => m.source === "pretrained"),
      trained: models.filter((m) => m.source === "trained"),
    };
  }, [models]);

  const selectedModel = useMemo(() => {
    if (!modelId || !models) return undefined;
    return models.find((m) => m.id === modelId);
  }, [modelId, models]);

  if (isLoading) {
    return (
      <div className="space-y-1.5">
        <Label className="text-xs">{field.label}</Label>
        <Skeleton className="h-8 w-full" />
      </div>
    );
  }

  return (
    <div className="space-y-1.5">
      <Label className="text-xs">{field.label}</Label>
      <Select value={modelId} onValueChange={onChange}>
        <SelectTrigger className="h-8">
          <SelectValue placeholder="Select model...">
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
                    <span className="text-xs text-muted-foreground">{model.model_type}</span>
                    {model.is_default && (
                      <Badge variant="secondary" className="text-[10px] py-0 px-1">Default</Badge>
                    )}
                  </div>
                </SelectItem>
              ))}
            </SelectGroup>
          )}
          {groupedModels.trained.length > 0 && (
            <SelectGroup>
              <SelectLabel className="flex items-center gap-2 text-xs">
                <Database className="h-3 w-3 text-blue-500" />
                Trained Models
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
                      {model.model_type}
                      {model.metrics?.map && ` • mAP: ${(model.metrics.map * 100).toFixed(1)}%`}
                      {model.class_count && ` • ${model.class_count} classes`}
                    </span>
                  </div>
                </SelectItem>
              ))}
            </SelectGroup>
          )}
          {groupedModels.pretrained.length === 0 && groupedModels.trained.length === 0 && (
            <div className="py-4 text-center text-xs text-muted-foreground">
              No models available
            </div>
          )}
        </SelectContent>
      </Select>
      {field.description && (
        <p className="text-xs text-muted-foreground">{field.description}</p>
      )}
    </div>
  );
}

/** Slider Field with optional parameter mode */
function SliderField({ field, value, onChange }: FieldProps<SliderConfigField>) {
  const isParam = isParamRef(value);
  const [paramMode, setParamMode] = useState(isParam);

  const numValue = useMemo(() => {
    if (typeof value === "number") return value;
    if (typeof value === "string" && !isParamRef(value)) {
      const parsed = parseFloat(value);
      return isNaN(parsed) ? field.min : parsed;
    }
    return field.default ?? field.min;
  }, [value, field.min, field.default]);

  const displayValue = field.formatValue ? field.formatValue(numValue) : String(numValue);

  if (paramMode && field.supportsParams) {
    return (
      <div className="space-y-1.5">
        <Label className="text-xs">{field.label}</Label>
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
                    if (isParamRef(value)) onChange(field.min);
                  }}
                >
                  <Sliders className="h-3.5 w-3.5" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Switch to slider</TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
        {field.description && (
          <p className="text-xs text-muted-foreground">{field.description}</p>
        )}
      </div>
    );
  }

  return (
    <div className="space-y-1.5">
      <Label className="text-xs">{field.label}</Label>
      <div className="flex items-center gap-2">
        <Slider
          value={[numValue]}
          onValueChange={([v]) => onChange(v)}
          min={field.min}
          max={field.max}
          step={field.step}
          className="flex-1"
        />
        <span className="text-xs text-muted-foreground w-12 text-right">{displayValue}</span>
        {field.supportsParams && (
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
        )}
      </div>
      {field.description && (
        <p className="text-xs text-muted-foreground">{field.description}</p>
      )}
    </div>
  );
}

/** Array Input Field */
function ArrayField({ field, value, onChange }: FieldProps<ArrayConfigField>) {
  const arrayValue = Array.isArray(value) ? value : (field.default ?? []);
  const [inputValue, setInputValue] = useState("");

  const addItem = () => {
    if (inputValue.trim()) {
      onChange([...arrayValue, inputValue.trim()]);
      setInputValue("");
    }
  };

  const removeItem = (index: number) => {
    onChange(arrayValue.filter((_, i) => i !== index));
  };

  return (
    <div className="space-y-1.5">
      <Label className="text-xs">{field.label}</Label>
      <div className="flex gap-2">
        <Input
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && (e.preventDefault(), addItem())}
          placeholder="Add item..."
          className="h-8 flex-1"
        />
        <Button variant="outline" size="icon" className="h-8 w-8" onClick={addItem}>
          <Plus className="h-3 w-3" />
        </Button>
      </div>
      {arrayValue.length > 0 && (
        <div className="flex flex-wrap gap-1 mt-2">
          {arrayValue.map((item, i) => (
            <Badge key={i} variant="secondary" className="gap-1 pr-1">
              {String(item)}
              <button onClick={() => removeItem(i)} className="hover:text-destructive">
                <X className="h-3 w-3" />
              </button>
            </Badge>
          ))}
        </div>
      )}
      {field.description && (
        <p className="text-xs text-muted-foreground">{field.description}</p>
      )}
    </div>
  );
}

/** Code/Expression Field */
function CodeField({ field, value, onChange }: FieldProps<CodeConfigField>) {
  const strValue = typeof value === "string" ? value : (field.default ?? "");

  return (
    <div className="space-y-1.5">
      <Label className="text-xs">{field.label}</Label>
      <Textarea
        value={strValue}
        onChange={(e) => onChange(e.target.value)}
        className="font-mono text-xs min-h-[80px]"
        placeholder={field.language === "json" ? '{ "key": "value" }' : "expression"}
      />
      {field.description && (
        <p className="text-xs text-muted-foreground">{field.description}</p>
      )}
    </div>
  );
}

// =============================================================================
// Main Component
// =============================================================================

export function ConfigFieldRenderer({
  fields,
  config,
  onConfigChange,
  showAdvanced = false,
}: ConfigFieldRendererProps) {
  const [advancedOpen, setAdvancedOpen] = useState(false);

  // Split fields into basic and advanced
  const { basicFields, advancedFields } = useMemo(() => {
    const basic: ConfigField[] = [];
    const advanced: ConfigField[] = [];

    for (const field of fields) {
      // Check showWhen condition
      if (!shouldShowField(field, config)) continue;

      if (field.advanced) {
        advanced.push(field);
      } else {
        basic.push(field);
      }
    }

    return { basicFields: basic, advancedFields: advanced };
  }, [fields, config]);

  // Render a single field based on type
  const renderField = (field: ConfigField) => {
    const value = config[field.key];
    const handleChange = (newValue: unknown) => {
      onConfigChange(field.key, newValue);
    };

    switch (field.type) {
      case "string":
        return <StringField key={field.key} field={field} value={value} onChange={handleChange} />;
      case "number":
        return <NumberField key={field.key} field={field} value={value} onChange={handleChange} />;
      case "boolean":
        return <BooleanField key={field.key} field={field} value={value} onChange={handleChange} />;
      case "select":
        return <SelectField key={field.key} field={field} value={value} onChange={handleChange} />;
      case "model":
        return <ModelField key={field.key} field={field} value={value} onChange={handleChange} />;
      case "slider":
        return <SliderField key={field.key} field={field} value={value} onChange={handleChange} />;
      case "array":
        return <ArrayField key={field.key} field={field} value={value} onChange={handleChange} />;
      case "code":
        return <CodeField key={field.key} field={field} value={value} onChange={handleChange} />;
      default:
        return null;
    }
  };

  return (
    <div className="space-y-4">
      {/* Basic Fields */}
      {basicFields.map(renderField)}

      {/* Advanced Fields (Collapsible) */}
      {advancedFields.length > 0 && showAdvanced && (
        <Collapsible open={advancedOpen} onOpenChange={setAdvancedOpen}>
          <CollapsibleTrigger asChild>
            <Button variant="ghost" size="sm" className="w-full justify-between h-8 text-xs">
              Advanced Options
              <ChevronDown className={`h-3 w-3 transition-transform ${advancedOpen ? "rotate-180" : ""}`} />
            </Button>
          </CollapsibleTrigger>
          <CollapsibleContent className="space-y-4 pt-2">
            {advancedFields.map(renderField)}
          </CollapsibleContent>
        </Collapsible>
      )}
    </div>
  );
}
