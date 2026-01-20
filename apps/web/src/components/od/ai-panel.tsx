"use client";

import { useState, useMemo, useEffect } from "react";
import { useMutation } from "@tanstack/react-query";
import { toast } from "sonner";
import { apiClient } from "@/lib/api-client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Checkbox } from "@/components/ui/checkbox";
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
  Loader2,
  Sparkles,
  ChevronDown,
  Check,
  X,
  Wand2,
  Info,
  SquareCheck,
  Square,
} from "lucide-react";
import type { AIPrediction } from "@/types/od";

// Minimal class type - only fields actually used by this component
type MinimalClass = {
  id: string;
  name: string;
  display_name?: string;
  color: string;
};

interface AIPanelProps {
  imageId: string;
  datasetId: string;
  classes: MinimalClass[];
  onPredictionsReceived: (predictions: AIPrediction[]) => void;
  onAcceptPrediction: (prediction: AIPrediction, targetClassId?: string) => void;
  onAcceptSelectedPredictions: (indices: number[], targetClassId: string) => void;
  onRejectPrediction: (index: number) => void;
  onAcceptAll: () => void;
  onRejectAll: () => void;
  predictions: AIPrediction[];
  selectedPredictionIndices?: Set<number>;
  onPredictionSelectionChange?: (indices: Set<number>) => void;
  isLoading?: boolean;
}

const AI_MODELS = [
  {
    id: "grounding_dino",
    name: "Grounding DINO",
    description: "SOTA text→bbox detection",
    icon: "🎯",
  },
  {
    id: "sam3",
    name: "SAM 3",
    description: "Text→mask→bbox with grounding",
    icon: "✨",
  },
  {
    id: "florence2",
    name: "Florence-2",
    description: "Microsoft's versatile vision model",
    icon: "🔮",
  },
];

export function AIPanel({
  imageId,
  datasetId,
  classes,
  onPredictionsReceived,
  onAcceptPrediction,
  onAcceptSelectedPredictions,
  onRejectPrediction,
  onAcceptAll,
  onRejectAll,
  predictions,
  selectedPredictionIndices: externalSelectedIndices,
  onPredictionSelectionChange,
  isLoading: externalLoading,
}: AIPanelProps) {
  const [isOpen, setIsOpen] = useState(true);
  const [selectedModel, setSelectedModel] = useState("grounding_dino");
  const [textPrompt, setTextPrompt] = useState("");
  const [boxThreshold, setBoxThreshold] = useState(0.3);
  const [textThreshold, setTextThreshold] = useState(0.25);
  const [useNms, setUseNms] = useState(true); // NMS enabled by default
  const [nmsThreshold, setNmsThreshold] = useState(0.5);

  // Selection state (internal if not controlled externally)
  const [internalSelectedIndices, setInternalSelectedIndices] = useState<Set<number>>(new Set());
  const selectedIndices = externalSelectedIndices ?? internalSelectedIndices;
  const setSelectedIndices = onPredictionSelectionChange ?? setInternalSelectedIndices;

  // Target class for assignment
  const [targetClassId, setTargetClassId] = useState<string>("");

  // Reset selection when predictions change
  useEffect(() => {
    if (predictions.length > 0) {
      // Auto-select all predictions initially
      setSelectedIndices(new Set(predictions.map((_, i) => i)));
    } else {
      setSelectedIndices(new Set());
    }
  }, [predictions.length]);

  // Auto-select first class if available and no class selected
  useEffect(() => {
    if (classes.length > 0 && !targetClassId) {
      setTargetClassId(classes[0].id);
    }
  }, [classes, targetClassId]);

  const predictMutation = useMutation({
    mutationFn: async () => {
      if (!textPrompt.trim()) {
        throw new Error("Please enter a text prompt");
      }

      return apiClient.predictODAI({
        image_id: imageId,
        model: selectedModel,
        text_prompt: textPrompt.trim(),
        box_threshold: boxThreshold,
        text_threshold: textThreshold,
        use_nms: useNms,
        nms_threshold: nmsThreshold,
      });
    },
    onSuccess: (data) => {
      // Debug: Log raw predictions from API
      console.log("[AI Panel] Raw predictions from API:", JSON.stringify(data.predictions, null, 2));
      data.predictions.forEach((pred, i) => {
        console.log(`[AI Panel] Prediction ${i}: bbox=${JSON.stringify(pred.bbox)}, label=${pred.label}, confidence=${pred.confidence}`);
      });

      if (data.predictions.length === 0) {
        toast.info("No objects found for the given prompt");
      } else {
        toast.success(`Found ${data.predictions.length} prediction(s)`);
        onPredictionsReceived(data.predictions);
      }
    },
    onError: (error: Error) => {
      toast.error(error.message || "AI prediction failed");
    },
  });

  const isLoading = externalLoading || predictMutation.isPending;

  const handleDetect = () => {
    if (!textPrompt.trim()) {
      toast.error("Please enter a text prompt");
      return;
    }
    predictMutation.mutate();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleDetect();
    }
  };

  const handleToggleSelection = (index: number) => {
    const newSelected = new Set(selectedIndices);
    if (newSelected.has(index)) {
      newSelected.delete(index);
    } else {
      newSelected.add(index);
    }
    setSelectedIndices(newSelected);
  };

  const handleSelectAll = () => {
    if (selectedIndices.size === predictions.length) {
      // Deselect all
      setSelectedIndices(new Set());
    } else {
      // Select all
      setSelectedIndices(new Set(predictions.map((_, i) => i)));
    }
  };

  const handleAcceptSelected = () => {
    if (selectedIndices.size === 0) {
      toast.error("No predictions selected");
      return;
    }
    if (!targetClassId) {
      toast.error("Please select a class");
      return;
    }

    const indices = Array.from(selectedIndices);
    onAcceptSelectedPredictions(indices, targetClassId);

    // Clear selection after accepting
    setSelectedIndices(new Set());
  };

  const selectedModelInfo = AI_MODELS.find((m) => m.id === selectedModel);
  const selectedCount = selectedIndices.size;
  const allSelected = predictions.length > 0 && selectedCount === predictions.length;
  const targetClass = classes.find(c => c.id === targetClassId);

  return (
    <Collapsible open={isOpen} onOpenChange={setIsOpen}>
      <CollapsibleTrigger asChild>
        <div className="flex items-center justify-between p-3 border-t cursor-pointer hover:bg-muted/50 transition-colors">
          <h3 className="font-medium text-sm flex items-center gap-2">
            <Sparkles className="h-4 w-4 text-primary" />
            AI Assist
            {predictions.length > 0 && (
              <Badge variant="secondary" className="ml-1">
                {predictions.length}
              </Badge>
            )}
          </h3>
          <ChevronDown
            className={`h-4 w-4 text-muted-foreground transition-transform ${
              isOpen ? "rotate-180" : ""
            }`}
          />
        </div>
      </CollapsibleTrigger>

      <CollapsibleContent>
        <div className="p-3 pt-0 space-y-3">
          {/* Model selector */}
          <div className="space-y-1.5">
            <Label className="text-xs text-muted-foreground">Model</Label>
            <Select value={selectedModel} onValueChange={setSelectedModel}>
              <SelectTrigger className="h-8 text-sm">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {AI_MODELS.map((model) => (
                  <SelectItem key={model.id} value={model.id}>
                    <div className="flex items-center gap-2">
                      <span>{model.icon}</span>
                      <span>{model.name}</span>
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {selectedModelInfo && (
              <p className="text-xs text-muted-foreground flex items-center gap-1">
                <Info className="h-3 w-3" />
                {selectedModelInfo.description}
              </p>
            )}
          </div>

          {/* Text prompt */}
          <div className="space-y-1.5">
            <Label className="text-xs text-muted-foreground">
              Detection prompt
            </Label>
            <Input
              value={textPrompt}
              onChange={(e) => setTextPrompt(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="shelf . product . price tag"
              className="h-8 text-sm"
              disabled={isLoading}
            />
            <p className="text-xs text-muted-foreground">
              Separate classes with &quot; . &quot; (space dot space)
            </p>
          </div>

          {/* Confidence threshold */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <Label className="text-xs text-muted-foreground">
                Confidence
              </Label>
              <span className="text-xs font-mono text-muted-foreground">
                {boxThreshold.toFixed(2)}
              </span>
            </div>
            <Slider
              value={[boxThreshold]}
              onValueChange={([v]) => setBoxThreshold(v)}
              min={0.1}
              max={0.9}
              step={0.05}
              disabled={isLoading}
            />
          </div>

          {/* NMS Settings */}
          <div className="space-y-2 pt-1 border-t border-border/50">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Switch
                  id="nms-toggle"
                  checked={useNms}
                  onCheckedChange={setUseNms}
                  disabled={isLoading}
                />
                <Label htmlFor="nms-toggle" className="text-xs cursor-pointer">
                  NMS Filter
                </Label>
              </div>
              {useNms && (
                <span className="text-xs font-mono text-muted-foreground">
                  IoU: {nmsThreshold.toFixed(2)}
                </span>
              )}
            </div>
            {useNms && (
              <div className="space-y-1">
                <Slider
                  value={[nmsThreshold]}
                  onValueChange={([v]) => setNmsThreshold(v)}
                  min={0.1}
                  max={0.9}
                  step={0.05}
                  disabled={isLoading}
                />
                <p className="text-[10px] text-muted-foreground">
                  Lower = more aggressive filtering
                </p>
              </div>
            )}
          </div>

          {/* Detect button */}
          <Button
            onClick={handleDetect}
            disabled={isLoading || !textPrompt.trim()}
            className="w-full h-8"
            size="sm"
          >
            {isLoading ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Detecting...
              </>
            ) : (
              <>
                <Wand2 className="h-4 w-4 mr-2" />
                Detect
              </>
            )}
          </Button>

          {/* Predictions list */}
          {predictions.length > 0 && (
            <div className="space-y-2 pt-2 border-t">
              {/* Header with select all */}
              <div className="flex items-center justify-between">
                <button
                  onClick={handleSelectAll}
                  className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
                >
                  {allSelected ? (
                    <SquareCheck className="h-4 w-4 text-primary" />
                  ) : (
                    <Square className="h-4 w-4" />
                  )}
                  <span>
                    {selectedCount > 0
                      ? `${selectedCount}/${predictions.length} selected`
                      : `Select All (${predictions.length})`
                    }
                  </span>
                </button>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-6 text-xs px-2 text-red-600 hover:text-red-700 hover:bg-red-100"
                  onClick={onRejectAll}
                >
                  <X className="h-3 w-3 mr-1" />
                  Clear
                </Button>
              </div>

              {/* Class assignment dropdown */}
              <div className="space-y-1.5">
                <Label className="text-xs text-muted-foreground">
                  Assign to class
                </Label>
                <Select value={targetClassId} onValueChange={setTargetClassId}>
                  <SelectTrigger className="h-8 text-sm">
                    <SelectValue placeholder="Select a class..." />
                  </SelectTrigger>
                  <SelectContent>
                    {classes.map((cls) => (
                      <SelectItem key={cls.id} value={cls.id}>
                        <div className="flex items-center gap-2">
                          <div
                            className="w-3 h-3 rounded-sm border"
                            style={{ backgroundColor: cls.color }}
                          />
                          <span>{cls.display_name || cls.name}</span>
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* Predictions list with checkboxes */}
              <div className="space-y-1 max-h-40 overflow-y-auto">
                {predictions.map((pred, index) => {
                  const isSelected = selectedIndices.has(index);
                  return (
                    <div
                      key={index}
                      className={`flex items-center gap-2 px-2 py-1.5 rounded-md border border-dashed transition-colors cursor-pointer ${
                        isSelected
                          ? "bg-primary/10 border-primary/50"
                          : "bg-muted/50 border-muted-foreground/30 hover:bg-muted"
                      }`}
                      onClick={() => handleToggleSelection(index)}
                    >
                      <Checkbox
                        checked={isSelected}
                        onCheckedChange={() => handleToggleSelection(index)}
                        onClick={(e) => e.stopPropagation()}
                        className="h-4 w-4"
                      />
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium truncate">
                          {pred.label}
                        </p>
                        <p className="text-xs text-muted-foreground">
                          {(pred.confidence * 100).toFixed(0)}% confidence
                        </p>
                      </div>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-6 w-6 text-red-600 hover:text-red-700 hover:bg-red-100"
                        onClick={(e) => {
                          e.stopPropagation();
                          onRejectPrediction(index);
                        }}
                      >
                        <X className="h-3.5 w-3.5" />
                      </Button>
                    </div>
                  );
                })}
              </div>

              {/* Accept selected button */}
              <Button
                onClick={handleAcceptSelected}
                disabled={selectedCount === 0 || !targetClassId}
                className="w-full h-8"
                size="sm"
                variant="default"
              >
                <Check className="h-4 w-4 mr-2" />
                Accept {selectedCount} as {targetClass?.display_name || targetClass?.name || "..."}
              </Button>
            </div>
          )}

          {/* Empty state */}
          {predictions.length === 0 && !isLoading && (
            <div className="text-center py-4 text-muted-foreground">
              <Sparkles className="h-8 w-8 mx-auto mb-2 opacity-30" />
              <p className="text-xs">
                Enter a prompt and click Detect to find objects
              </p>
            </div>
          )}
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
}
