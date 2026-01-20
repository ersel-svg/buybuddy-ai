"use client";

import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { toast } from "sonner";
import { apiClient } from "@/lib/api-client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";
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
} from "lucide-react";
import type { AIPrediction } from "@/types/od";

interface AIPanelProps {
  imageId: string;
  datasetId: string;
  onPredictionsReceived: (predictions: AIPrediction[]) => void;
  onAcceptPrediction: (prediction: AIPrediction) => void;
  onRejectPrediction: (index: number) => void;
  onAcceptAll: () => void;
  onRejectAll: () => void;
  predictions: AIPrediction[];
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
  onPredictionsReceived,
  onAcceptPrediction,
  onRejectPrediction,
  onAcceptAll,
  onRejectAll,
  predictions,
  isLoading: externalLoading,
}: AIPanelProps) {
  const [isOpen, setIsOpen] = useState(true);
  const [selectedModel, setSelectedModel] = useState("grounding_dino");
  const [textPrompt, setTextPrompt] = useState("");
  const [boxThreshold, setBoxThreshold] = useState(0.3);
  const [textThreshold, setTextThreshold] = useState(0.25);

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
      });
    },
    onSuccess: (data) => {
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

  const selectedModelInfo = AI_MODELS.find((m) => m.id === selectedModel);

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
              <div className="flex items-center justify-between">
                <p className="text-xs text-muted-foreground">
                  AI Predictions ({predictions.length})
                </p>
                <div className="flex gap-1">
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-6 text-xs px-2"
                    onClick={onAcceptAll}
                  >
                    <Check className="h-3 w-3 mr-1" />
                    All
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-6 text-xs px-2"
                    onClick={onRejectAll}
                  >
                    <X className="h-3 w-3 mr-1" />
                    Clear
                  </Button>
                </div>
              </div>

              <div className="space-y-1 max-h-40 overflow-y-auto">
                {predictions.map((pred, index) => (
                  <div
                    key={index}
                    className="flex items-center gap-2 px-2 py-1.5 rounded-md bg-muted/50 border border-dashed border-primary/30"
                  >
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium truncate">
                        {pred.label}
                      </p>
                      <p className="text-xs text-muted-foreground">
                        {(pred.confidence * 100).toFixed(0)}% confidence
                      </p>
                    </div>
                    <div className="flex gap-1">
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-6 w-6 text-green-600 hover:text-green-700 hover:bg-green-100"
                        onClick={() => onAcceptPrediction(pred)}
                      >
                        <Check className="h-3.5 w-3.5" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-6 w-6 text-red-600 hover:text-red-700 hover:bg-red-100"
                        onClick={() => onRejectPrediction(index)}
                      >
                        <X className="h-3.5 w-3.5" />
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
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
