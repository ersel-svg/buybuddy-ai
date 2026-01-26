"use client";

import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useMemo } from "react";
import type {
  ConfigField,
  StringConfigField,
  NumberConfigField,
  BooleanConfigField,
  SelectConfigField,
  ModelConfigField,
  SliderConfigField,
  ArrayConfigField,
  ObjectConfigField,
  CodeConfigField,
} from "@/lib/workflow/types";
import { ParamSlider, ParamInput, isParamRef } from "./utils";
import { ModelSelector } from "./ModelSelector";
import type { WorkflowModel } from "./utils";

// =============================================================================
// Types
// =============================================================================

export interface ConfigFieldRendererProps {
  field: ConfigField;
  value: unknown;
  onChange: (value: unknown) => void;
  /** All config values, needed for showWhen conditions */
  allValues: Record<string, unknown>;
  /** Callback for batch updates (e.g., model_id + model_source) */
  onBatchChange?: (updates: Record<string, unknown>) => void;
}

// =============================================================================
// Condition Checker
// =============================================================================

/**
 * Check if a field should be shown based on showWhen condition
 */
function shouldShowField(
  field: ConfigField,
  allValues: Record<string, unknown>
): boolean {
  if (!field.showWhen) return true;

  const { field: targetField, value: targetValue, operator = "eq" } = field.showWhen;
  const currentValue = allValues[targetField];

  switch (operator) {
    case "eq":
      return currentValue === targetValue;
    case "neq":
      return currentValue !== targetValue;
    case "includes":
      if (Array.isArray(currentValue)) {
        return currentValue.includes(targetValue);
      }
      if (typeof currentValue === "string" && typeof targetValue === "string") {
        return currentValue.includes(targetValue);
      }
      return false;
    case "gt":
      return typeof currentValue === "number" && typeof targetValue === "number"
        ? currentValue > targetValue
        : false;
    case "lt":
      return typeof currentValue === "number" && typeof targetValue === "number"
        ? currentValue < targetValue
        : false;
    default:
      return true;
  }
}

// =============================================================================
// Field Renderers
// =============================================================================

function StringFieldRenderer({
  field,
  value,
  onChange,
}: {
  field: StringConfigField;
  value: unknown;
  onChange: (value: unknown) => void;
}) {
  if (field.supportsParams) {
    return (
      <ParamInput
        value={value}
        onChange={onChange}
        placeholder={field.placeholder}
      />
    );
  }

  return (
    <Input
      type="text"
      value={(value as string) ?? field.default ?? ""}
      onChange={(e) => onChange(e.target.value)}
      placeholder={field.placeholder}
      className="h-8"
    />
  );
}

function NumberFieldRenderer({
  field,
  value,
  onChange,
}: {
  field: NumberConfigField;
  value: unknown;
  onChange: (value: unknown) => void;
}) {
  if (field.supportsParams) {
    return (
      <ParamInput
        value={value}
        onChange={onChange}
        type="number"
        min={field.min}
        max={field.max}
      />
    );
  }

  return (
    <Input
      type="number"
      value={(value as number) ?? field.default ?? ""}
      onChange={(e) => {
        const val = e.target.value;
        onChange(val === "" ? undefined : parseFloat(val));
      }}
      min={field.min}
      max={field.max}
      step={field.step}
      className="h-8"
    />
  );
}

function BooleanFieldRenderer({
  field,
  value,
  onChange,
}: {
  field: BooleanConfigField;
  value: unknown;
  onChange: (value: unknown) => void;
}) {
  return (
    <div className="flex items-center justify-between">
      <div>
        <Label className="text-xs">{field.label}</Label>
        {field.description && (
          <p className="text-xs text-muted-foreground">{field.description}</p>
        )}
      </div>
      <Switch
        checked={(value as boolean) ?? field.default ?? false}
        onCheckedChange={onChange}
      />
    </div>
  );
}

function SelectFieldRenderer({
  field,
  value,
  onChange,
}: {
  field: SelectConfigField;
  value: unknown;
  onChange: (value: unknown) => void;
}) {
  return (
    <Select
      value={(value as string) ?? field.default ?? ""}
      onValueChange={onChange}
    >
      <SelectTrigger className="h-8">
        <SelectValue placeholder="Select..." />
      </SelectTrigger>
      <SelectContent>
        {field.options.map((option) => (
          <SelectItem key={option.value} value={option.value}>
            <div className="flex flex-col">
              <span>{option.label}</span>
              {option.description && (
                <span className="text-xs text-muted-foreground">
                  {option.description}
                </span>
              )}
            </div>
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

function ModelFieldRenderer({
  field,
  value,
  onChange,
  onBatchChange,
}: {
  field: ModelConfigField;
  value: unknown;
  onChange: (value: unknown) => void;
  onBatchChange?: (updates: Record<string, unknown>) => void;
}) {
  const handleModelChange = (modelId: string, model?: WorkflowModel) => {
    // If batch change handler is available, set both model_id and model_source
    if (onBatchChange && model) {
      onBatchChange({
        [field.key]: modelId,
        model_source: model.source,
      });
    } else {
      onChange(modelId);
    }
  };

  return (
    <ModelSelector
      value={value as string | undefined}
      onChange={handleModelChange}
      category={field.category}
      label={field.label}
      description={field.description}
    />
  );
}

function SliderFieldRenderer({
  field,
  value,
  onChange,
}: {
  field: SliderConfigField;
  value: unknown;
  onChange: (value: unknown) => void;
}) {
  return (
    <ParamSlider
      value={value ?? field.default ?? field.min}
      onChange={onChange}
      min={field.min}
      max={field.max}
      step={field.step}
      formatValue={field.formatValue}
    />
  );
}

function ArrayFieldRenderer({
  field,
  value,
  onChange,
}: {
  field: ArrayConfigField;
  value: unknown;
  onChange: (value: unknown) => void;
}) {
  // For array fields with parameter support
  if (field.supportsParams && isParamRef(value)) {
    return (
      <ParamInput
        value={value}
        onChange={onChange}
        placeholder="e.g., {{ params.items }}"
      />
    );
  }

  // Simple array as comma-separated values
  const arrayValue = Array.isArray(value) ? value : field.default ?? [];
  const stringValue = arrayValue.join(", ");

  return (
    <Input
      type="text"
      value={stringValue}
      onChange={(e) => {
        const val = e.target.value;
        if (val === "") {
          onChange([]);
        } else {
          // Split by comma, trim whitespace
          const items = val.split(",").map((s) => s.trim()).filter(Boolean);
          onChange(items);
        }
      }}
      placeholder="Enter comma-separated values..."
      className="h-8"
    />
  );
}

function ObjectFieldRenderer({
  field,
  value,
  onChange,
}: {
  field: ObjectConfigField;
  value: unknown;
  onChange: (value: unknown) => void;
}) {
  // Render as JSON textarea
  const jsonString = useMemo(() => {
    try {
      return JSON.stringify(value ?? field.default ?? {}, null, 2);
    } catch {
      return "{}";
    }
  }, [value, field.default]);

  return (
    <Textarea
      value={jsonString}
      onChange={(e) => {
        try {
          const parsed = JSON.parse(e.target.value);
          onChange(parsed);
        } catch {
          // Keep invalid JSON in textarea but don't update state
        }
      }}
      placeholder="{}"
      className="font-mono text-xs min-h-[80px]"
    />
  );
}

function CodeFieldRenderer({
  field,
  value,
  onChange,
}: {
  field: CodeConfigField;
  value: unknown;
  onChange: (value: unknown) => void;
}) {
  return (
    <Textarea
      value={(value as string) ?? field.default ?? ""}
      onChange={(e) => onChange(e.target.value)}
      placeholder={
        field.language === "jmespath"
          ? "e.g., detections[?confidence > `0.5`].class_name"
          : field.language === "json"
            ? "{}"
            : "// code here"
      }
      className="font-mono text-xs min-h-[80px]"
    />
  );
}

// =============================================================================
// Main Component
// =============================================================================

/**
 * ConfigFieldRenderer - Renders a config field based on its type
 *
 * Features:
 * - Automatic field type detection
 * - showWhen condition checking
 * - Parameter support for applicable fields
 * - Label and description rendering (except for boolean which handles its own)
 */
export function ConfigFieldRenderer({
  field,
  value,
  onChange,
  allValues,
  onBatchChange,
}: ConfigFieldRendererProps) {
  // Check showWhen condition
  if (!shouldShowField(field, allValues)) {
    return null;
  }

  // Boolean fields handle their own label
  if (field.type === "boolean") {
    return (
      <BooleanFieldRenderer
        field={field}
        value={value}
        onChange={onChange}
      />
    );
  }

  // Model fields use ModelSelector which has its own label
  if (field.type === "model") {
    return (
      <ModelFieldRenderer
        field={field}
        value={value}
        onChange={onChange}
        onBatchChange={onBatchChange}
      />
    );
  }

  // All other fields need label wrapper
  return (
    <div className="space-y-2">
      <Label className="text-xs">{field.label}</Label>
      {renderFieldByType(field, value, onChange)}
      {field.description && (
        <p className="text-xs text-muted-foreground">{field.description}</p>
      )}
    </div>
  );
}

/**
 * Render field content by type
 */
function renderFieldByType(
  field: ConfigField,
  value: unknown,
  onChange: (value: unknown) => void
): React.ReactNode {
  switch (field.type) {
    case "string":
      return (
        <StringFieldRenderer field={field} value={value} onChange={onChange} />
      );
    case "number":
      return (
        <NumberFieldRenderer field={field} value={value} onChange={onChange} />
      );
    case "select":
      return (
        <SelectFieldRenderer field={field} value={value} onChange={onChange} />
      );
    case "slider":
      return (
        <SliderFieldRenderer field={field} value={value} onChange={onChange} />
      );
    case "array":
      return (
        <ArrayFieldRenderer field={field} value={value} onChange={onChange} />
      );
    case "object":
      return (
        <ObjectFieldRenderer field={field} value={value} onChange={onChange} />
      );
    case "code":
      return (
        <CodeFieldRenderer field={field} value={value} onChange={onChange} />
      );
    default:
      // Fallback to string input
      return (
        <Input
          type="text"
          value={String(value ?? "")}
          onChange={(e) => onChange(e.target.value)}
          className="h-8"
        />
      );
  }
}

// =============================================================================
// Exports
// =============================================================================

export { shouldShowField };
