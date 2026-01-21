/**
 * ModelStep Component
 *
 * Step 5: Model type and size selection.
 */

"use client";

import { Box, Cpu, Zap, Gauge, HardDrive } from "lucide-react";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { ModelStepData, ModelType, ModelSize } from "../../types/wizard";

interface ModelStepProps {
  data: ModelStepData;
  onChange: (data: Partial<ModelStepData>) => void;
  errors?: string[];
}

const MODEL_INFO: Record<ModelType, {
  name: string;
  description: string;
  features: string[];
}> = {
  "rt-detr": {
    name: "RT-DETR",
    description: "Real-Time Detection Transformer by Baidu",
    features: [
      "Fast inference (50+ FPS)",
      "No NMS post-processing",
      "COCO pretrained",
      "Good for general detection",
    ],
  },
  "d-fine": {
    name: "D-FINE",
    description: "Detection by Fine-grained feature refinement",
    features: [
      "State-of-the-art accuracy",
      "Better small object detection",
      "Advanced feature pyramid",
      "Ideal for complex scenes",
    ],
  },
};

const SIZE_INFO: Record<ModelSize, {
  name: string;
  params: { "rt-detr": string; "d-fine": string };
  vram: { "rt-detr": string; "d-fine": string };
  speed: { "rt-detr": string; "d-fine": string };
}> = {
  s: {
    name: "Small",
    params: { "rt-detr": "20M", "d-fine": "25M" },
    vram: { "rt-detr": "~4GB", "d-fine": "~5GB" },
    speed: { "rt-detr": "80+ FPS", "d-fine": "65+ FPS" },
  },
  m: {
    name: "Medium",
    params: { "rt-detr": "32M", "d-fine": "52M" },
    vram: { "rt-detr": "~6GB", "d-fine": "~8GB" },
    speed: { "rt-detr": "55+ FPS", "d-fine": "45+ FPS" },
  },
  l: {
    name: "Large",
    params: { "rt-detr": "42M", "d-fine": "62M" },
    vram: { "rt-detr": "~8GB", "d-fine": "~10GB" },
    speed: { "rt-detr": "40+ FPS", "d-fine": "35+ FPS" },
  },
  x: {
    name: "XLarge",
    params: { "rt-detr": "-", "d-fine": "82M" },
    vram: { "rt-detr": "-", "d-fine": "~12GB" },
    speed: { "rt-detr": "-", "d-fine": "25+ FPS" },
  },
};

export function ModelStep({ data, onChange, errors }: ModelStepProps) {
  const availableSizes: ModelSize[] = data.modelType === "d-fine"
    ? ["s", "m", "l", "x"]
    : ["s", "m", "l"];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-lg font-semibold flex items-center gap-2">
          <Box className="h-5 w-5" />
          Select Model
        </h2>
        <p className="text-sm text-muted-foreground mt-1">
          Choose the model architecture and size for your training.
        </p>
      </div>

      {/* Model Type Selection */}
      <div className="space-y-3">
        <Label className="text-base">Model Architecture</Label>
        <RadioGroup
          value={data.modelType}
          onValueChange={(value: ModelType) => {
            // Reset size if switching to RT-DETR and size is 'x'
            const newSize = value === "rt-detr" && data.modelSize === "x" ? "l" : data.modelSize;
            onChange({ modelType: value, modelSize: newSize });
          }}
          className="grid grid-cols-1 md:grid-cols-2 gap-4"
        >
          {(Object.keys(MODEL_INFO) as ModelType[]).map((type) => {
            const info = MODEL_INFO[type];
            const isSelected = data.modelType === type;

            return (
              <Card
                key={type}
                className={cn(
                  "cursor-pointer transition-all",
                  isSelected && "border-primary ring-1 ring-primary"
                )}
                onClick={() => onChange({ modelType: type })}
              >
                <CardContent className="p-4">
                  <div className="flex items-start gap-3">
                    <RadioGroupItem value={type} id={type} className="mt-1" />
                    <div className="flex-1">
                      <Label htmlFor={type} className="text-base font-semibold cursor-pointer">
                        {info.name}
                      </Label>
                      <p className="text-sm text-muted-foreground mt-1">
                        {info.description}
                      </p>
                      <ul className="mt-2 space-y-1">
                        {info.features.map((feature, i) => (
                          <li key={i} className="text-xs text-muted-foreground flex items-center gap-1">
                            <span className="w-1 h-1 rounded-full bg-primary" />
                            {feature}
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </RadioGroup>
      </div>

      {/* Model Size Selection */}
      <div className="space-y-3">
        <Label className="text-base">Model Size</Label>
        <RadioGroup
          value={data.modelSize}
          onValueChange={(value: ModelSize) => onChange({ modelSize: value })}
          className="grid grid-cols-2 md:grid-cols-4 gap-3"
        >
          {availableSizes.map((size) => {
            const info = SIZE_INFO[size];
            const isSelected = data.modelSize === size;

            return (
              <Card
                key={size}
                className={cn(
                  "cursor-pointer transition-all",
                  isSelected && "border-primary ring-1 ring-primary"
                )}
                onClick={() => onChange({ modelSize: size })}
              >
                <CardContent className="p-3">
                  <div className="flex items-center gap-2 mb-2">
                    <RadioGroupItem value={size} id={`size-${size}`} />
                    <Label htmlFor={`size-${size}`} className="font-semibold cursor-pointer">
                      {info.name}
                    </Label>
                  </div>
                  <div className="space-y-1 text-xs text-muted-foreground">
                    <div className="flex items-center gap-1">
                      <Cpu className="h-3 w-3" />
                      {info.params[data.modelType]} params
                    </div>
                    <div className="flex items-center gap-1">
                      <HardDrive className="h-3 w-3" />
                      {info.vram[data.modelType]} VRAM
                    </div>
                    <div className="flex items-center gap-1">
                      <Zap className="h-3 w-3" />
                      {info.speed[data.modelType]}
                    </div>
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </RadioGroup>
      </div>

      {/* Pretrained & Freeze Options */}
      <div className="space-y-4 pt-4 border-t">
        <div className="flex items-center justify-between">
          <div>
            <Label className="text-base">Use Pretrained Weights</Label>
            <p className="text-sm text-muted-foreground">
              Start from COCO pretrained weights (recommended)
            </p>
          </div>
          <Switch
            checked={data.pretrained}
            onCheckedChange={(checked) => onChange({ pretrained: checked })}
          />
        </div>

        <div className="flex items-center justify-between">
          <div>
            <Label className="text-base">Freeze Backbone</Label>
            <p className="text-sm text-muted-foreground">
              Freeze backbone layers for initial epochs
            </p>
          </div>
          <Switch
            checked={data.freezeBackbone}
            onCheckedChange={(checked) => onChange({ freezeBackbone: checked })}
          />
        </div>

        {data.freezeBackbone && (
          <div className="space-y-2 pl-4 border-l-2 border-muted">
            <div className="flex justify-between">
              <Label>Freeze Epochs</Label>
              <span className="text-sm font-medium">{data.freezeEpochs}</span>
            </div>
            <Slider
              value={[data.freezeEpochs]}
              onValueChange={(value) => onChange({ freezeEpochs: value[0] })}
              min={1}
              max={50}
              step={1}
            />
            <p className="text-xs text-muted-foreground">
              Backbone will be frozen for the first {data.freezeEpochs} epochs
            </p>
          </div>
        )}
      </div>

      {/* Selected Model Summary */}
      <Card className="bg-muted/50">
        <CardContent className="p-4">
          <div className="flex items-center gap-2 mb-2">
            <Gauge className="h-4 w-4" />
            <span className="font-medium">Selected Configuration</span>
          </div>
          <div className="flex flex-wrap gap-2">
            <Badge>{MODEL_INFO[data.modelType].name}</Badge>
            <Badge variant="secondary">{SIZE_INFO[data.modelSize].name}</Badge>
            <Badge variant="outline">
              {SIZE_INFO[data.modelSize].params[data.modelType]} parameters
            </Badge>
            {data.pretrained && <Badge variant="outline">Pretrained</Badge>}
            {data.freezeBackbone && (
              <Badge variant="outline">Freeze {data.freezeEpochs} epochs</Badge>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
