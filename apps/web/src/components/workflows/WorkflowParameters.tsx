"use client";

import { useState } from "react";
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
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Plus,
  Trash2,
  X,
  Variable,
  Info,
  Copy,
  Check,
} from "lucide-react";

// =============================================================================
// Types
// =============================================================================

export interface WorkflowParameter {
  name: string;
  type: "string" | "number" | "boolean" | "array" | "object";
  default?: unknown;
  description?: string;
  required?: boolean;
}

export interface WorkflowParametersProps {
  open: boolean;
  onClose: () => void;
  parameters: WorkflowParameter[];
  onParametersChange: (parameters: WorkflowParameter[]) => void;
}

// =============================================================================
// Constants
// =============================================================================

const PARAM_TYPES = [
  { value: "string", label: "String", description: "Text value" },
  { value: "number", label: "Number", description: "Numeric value (float)" },
  { value: "boolean", label: "Boolean", description: "True or false" },
  { value: "array", label: "Array", description: "List of values" },
  { value: "object", label: "Object", description: "Key-value pairs" },
] as const;

const TYPE_COLORS: Record<string, string> = {
  string: "bg-blue-500/10 text-blue-500 border-blue-500/20",
  number: "bg-green-500/10 text-green-500 border-green-500/20",
  boolean: "bg-purple-500/10 text-purple-500 border-purple-500/20",
  array: "bg-orange-500/10 text-orange-500 border-orange-500/20",
  object: "bg-pink-500/10 text-pink-500 border-pink-500/20",
};

// =============================================================================
// Sub-components
// =============================================================================

function ParameterCard({
  param,
  index,
  onUpdate,
  onDelete,
}: {
  param: WorkflowParameter;
  index: number;
  onUpdate: (index: number, updates: Partial<WorkflowParameter>) => void;
  onDelete: (index: number) => void;
}) {
  const [copied, setCopied] = useState(false);

  const copyReference = () => {
    navigator.clipboard.writeText(`{{ params.${param.name} }}`);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="border rounded-lg p-4 space-y-4 bg-card">
      {/* Header */}
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-2 flex-1 min-w-0">
          <Variable className="h-4 w-4 text-muted-foreground flex-shrink-0" />
          <Input
            value={param.name}
            onChange={(e) => {
              const newName = e.target.value
                .toLowerCase()
                .replace(/[^a-z0-9_]/g, "_")
                .replace(/^[0-9]/, "_");
              onUpdate(index, { name: newName });
            }}
            placeholder="parameter_name"
            className="h-8 font-mono text-sm"
          />
        </div>
        <div className="flex items-center gap-1">
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8"
                  onClick={copyReference}
                >
                  {copied ? (
                    <Check className="h-3.5 w-3.5 text-green-500" />
                  ) : (
                    <Copy className="h-3.5 w-3.5" />
                  )}
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                <p>Copy reference</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
          <AlertDialog>
            <AlertDialogTrigger asChild>
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8 text-destructive hover:text-destructive"
              >
                <Trash2 className="h-3.5 w-3.5" />
              </Button>
            </AlertDialogTrigger>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle>Delete Parameter</AlertDialogTitle>
                <AlertDialogDescription>
                  Are you sure you want to delete "{param.name}"? Nodes referencing this parameter will show undefined values.
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>Cancel</AlertDialogCancel>
                <AlertDialogAction
                  className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                  onClick={() => onDelete(index)}
                >
                  Delete
                </AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
        </div>
      </div>

      {/* Type and Required */}
      <div className="grid grid-cols-2 gap-3">
        <div className="space-y-1.5">
          <Label className="text-xs text-muted-foreground">Type</Label>
          <Select
            value={param.type}
            onValueChange={(value) =>
              onUpdate(index, { type: value as WorkflowParameter["type"] })
            }
          >
            <SelectTrigger className="h-8">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {PARAM_TYPES.map((type) => (
                <SelectItem key={type.value} value={type.value}>
                  <div className="flex items-center gap-2">
                    <Badge
                      variant="outline"
                      className={`text-[10px] px-1 py-0 ${TYPE_COLORS[type.value]}`}
                    >
                      {type.label}
                    </Badge>
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="space-y-1.5">
          <Label className="text-xs text-muted-foreground">Default Value</Label>
          <DefaultValueInput
            type={param.type}
            value={param.default}
            onChange={(value) => onUpdate(index, { default: value })}
          />
        </div>
      </div>

      {/* Description */}
      <div className="space-y-1.5">
        <Label className="text-xs text-muted-foreground">Description</Label>
        <Input
          value={param.description || ""}
          onChange={(e) => onUpdate(index, { description: e.target.value })}
          placeholder="What this parameter is used for..."
          className="h-8 text-sm"
        />
      </div>

      {/* Reference hint */}
      <div className="flex items-center gap-2 text-xs text-muted-foreground bg-muted/50 rounded px-2 py-1.5">
        <Info className="h-3 w-3 flex-shrink-0" />
        <span>
          Use <code className="bg-muted px-1 rounded font-mono">{"{{ params." + param.name + " }}"}</code> in node configs
        </span>
      </div>
    </div>
  );
}

function DefaultValueInput({
  type,
  value,
  onChange,
}: {
  type: WorkflowParameter["type"];
  value: unknown;
  onChange: (value: unknown) => void;
}) {
  switch (type) {
    case "boolean":
      return (
        <Select
          value={value === true ? "true" : value === false ? "false" : ""}
          onValueChange={(v) => onChange(v === "true" ? true : v === "false" ? false : undefined)}
        >
          <SelectTrigger className="h-8">
            <SelectValue placeholder="Select..." />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="true">true</SelectItem>
            <SelectItem value="false">false</SelectItem>
          </SelectContent>
        </Select>
      );
    case "number":
      return (
        <Input
          type="number"
          value={value !== undefined ? String(value) : ""}
          onChange={(e) => {
            const v = e.target.value;
            onChange(v === "" ? undefined : parseFloat(v));
          }}
          placeholder="0"
          className="h-8 text-sm"
          step="any"
        />
      );
    case "array":
    case "object":
      return (
        <Input
          value={value !== undefined ? JSON.stringify(value) : ""}
          onChange={(e) => {
            try {
              onChange(e.target.value ? JSON.parse(e.target.value) : undefined);
            } catch {
              // Invalid JSON, keep as string for now
            }
          }}
          placeholder={type === "array" ? "[]" : "{}"}
          className="h-8 text-sm font-mono"
        />
      );
    default:
      return (
        <Input
          value={value !== undefined ? String(value) : ""}
          onChange={(e) => onChange(e.target.value || undefined)}
          placeholder="Default value..."
          className="h-8 text-sm"
        />
      );
  }
}

// =============================================================================
// Main Component
// =============================================================================

export function WorkflowParameters({
  open,
  onClose,
  parameters,
  onParametersChange,
}: WorkflowParametersProps) {
  const addParameter = () => {
    const baseName = "param";
    let counter = 1;
    let name = baseName;
    while (parameters.some((p) => p.name === name)) {
      name = `${baseName}_${counter}`;
      counter++;
    }

    onParametersChange([
      ...parameters,
      {
        name,
        type: "string",
        default: undefined,
        description: "",
      },
    ]);
  };

  const updateParameter = (index: number, updates: Partial<WorkflowParameter>) => {
    onParametersChange(
      parameters.map((p, i) => (i === index ? { ...p, ...updates } : p))
    );
  };

  const deleteParameter = (index: number) => {
    onParametersChange(parameters.filter((_, i) => i !== index));
  };

  return (
    <Sheet open={open} onOpenChange={(isOpen) => !isOpen && onClose()}>
      <SheetContent className="w-[420px] p-0 flex flex-col">
        {/* Header */}
        <SheetHeader className="px-6 py-4 border-b bg-card/50 backdrop-blur-sm">
          <div className="flex items-start justify-between gap-3">
            <div className="flex items-center gap-3">
              <div className="h-10 w-10 rounded-lg bg-gradient-to-br from-blue-500/20 to-purple-500/20 flex items-center justify-center">
                <Variable className="h-5 w-5 text-blue-500" />
              </div>
              <div>
                <SheetTitle className="text-base font-semibold">
                  Workflow Parameters
                </SheetTitle>
                <SheetDescription className="text-xs">
                  Define reusable parameters for your workflow
                </SheetDescription>
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

        {/* Content */}
        <ScrollArea className="flex-1">
          <div className="px-6 py-5 space-y-4">
            {parameters.length === 0 ? (
              <div className="text-center py-8">
                <Variable className="h-10 w-10 mx-auto text-muted-foreground/50 mb-3" />
                <p className="text-sm text-muted-foreground">
                  No parameters defined
                </p>
                <p className="text-xs text-muted-foreground mt-1">
                  Add parameters to make your workflow configurable
                </p>
              </div>
            ) : (
              parameters.map((param, index) => (
                <ParameterCard
                  key={index}
                  param={param}
                  index={index}
                  onUpdate={updateParameter}
                  onDelete={deleteParameter}
                />
              ))
            )}
          </div>
        </ScrollArea>

        {/* Footer */}
        <div className="px-6 py-4 border-t bg-card/50 backdrop-blur-sm">
          <Button
            variant="outline"
            className="w-full"
            onClick={addParameter}
          >
            <Plus className="h-4 w-4 mr-2" />
            Add Parameter
          </Button>
        </div>
      </SheetContent>
    </Sheet>
  );
}
