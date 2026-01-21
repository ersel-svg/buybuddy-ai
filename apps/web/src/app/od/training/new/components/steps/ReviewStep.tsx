/**
 * ReviewStep Component
 *
 * Step 7: Review configuration and start training.
 */

"use client";

import { FileCheck, Database, Image, Layers, Box, Settings, Zap, Clock } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { cn } from "@/lib/utils";
import type { WizardState, ReviewStepData } from "../../types/wizard";
import { estimateTrainingTime, getPresetDisplayInfo, getModelDisplayInfo } from "../../utils/apiConverter";

interface ReviewStepProps {
  data: ReviewStepData;
  onChange: (data: Partial<ReviewStepData>) => void;
  wizardState: WizardState;
  errors?: string[];
}

export function ReviewStep({ data, onChange, wizardState, errors }: ReviewStepProps) {
  const { dataset, preprocess, offlineAug, onlineAug, model, hyperparams, datasetInfo } = wizardState;

  const presetInfo = getPresetDisplayInfo(onlineAug.preset);
  const modelInfo = getModelDisplayInfo(model.modelType, model.modelSize);

  const estimatedTime = datasetInfo
    ? estimateTrainingTime(
        datasetInfo.annotatedImageCount,
        hyperparams.epochs,
        hyperparams.batchSize,
        model.modelType,
        model.modelSize
      )
    : "Unknown";

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-lg font-semibold flex items-center gap-2">
          <FileCheck className="h-5 w-5" />
          Review & Start Training
        </h2>
        <p className="text-sm text-muted-foreground mt-1">
          Review your configuration and give your training run a name.
        </p>
      </div>

      {/* Training Name & Description */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Training Details</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="name">Training Name *</Label>
            <Input
              id="name"
              value={data.name}
              onChange={(e) => onChange({ name: e.target.value })}
              placeholder="e.g., product-detector-v1"
              className={cn(errors?.includes("Training name is required") && "border-destructive")}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="description">Description (Optional)</Label>
            <Textarea
              id="description"
              value={data.description}
              onChange={(e) => onChange({ description: e.target.value })}
              placeholder="Add notes about this training run..."
              rows={2}
            />
          </div>
        </CardContent>
      </Card>

      {/* Configuration Summary */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Dataset */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm flex items-center gap-2">
              <Database className="h-4 w-4" />
              Dataset
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-muted-foreground">Name</span>
              <span className="font-medium">{datasetInfo?.name || "Unknown"}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Images</span>
              <span className="font-medium">{datasetInfo?.annotatedImageCount.toLocaleString() || 0}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Split</span>
              <span className="font-medium">
                {Math.round(dataset.trainSplit * 100)}% / {Math.round(dataset.valSplit * 100)}% / {Math.round(dataset.testSplit * 100)}%
              </span>
            </div>
          </CardContent>
        </Card>

        {/* Preprocessing */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm flex items-center gap-2">
              <Image className="h-4 w-4" />
              Preprocessing
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-muted-foreground">Image Size</span>
              <span className="font-medium">{preprocess.targetSize}px</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Resize</span>
              <span className="font-medium capitalize">{preprocess.resizeStrategy}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Tiling</span>
              <span className="font-medium">{preprocess.tiling.enabled ? "Enabled" : "Disabled"}</span>
            </div>
          </CardContent>
        </Card>

        {/* Augmentation */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm flex items-center gap-2">
              <Layers className="h-4 w-4" />
              Augmentation
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-muted-foreground">Offline</span>
              <span className="font-medium">
                {offlineAug.enabled ? `${offlineAug.multiplier}x` : "Disabled"}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Online Preset</span>
              <span className="font-medium">{presetInfo.name}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Expected Boost</span>
              <Badge variant="secondary" className="text-xs">{presetInfo.boost}</Badge>
            </div>
          </CardContent>
        </Card>

        {/* Model */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm flex items-center gap-2">
              <Box className="h-4 w-4" />
              Model
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-muted-foreground">Architecture</span>
              <span className="font-medium">{modelInfo.name}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Parameters</span>
              <span className="font-medium">{modelInfo.params}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">VRAM Required</span>
              <span className="font-medium">{modelInfo.vram}</span>
            </div>
          </CardContent>
        </Card>

        {/* Training */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm flex items-center gap-2">
              <Settings className="h-4 w-4" />
              Training
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-muted-foreground">Epochs</span>
              <span className="font-medium">{hyperparams.epochs}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Batch Size</span>
              <span className="font-medium">{hyperparams.batchSize}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Learning Rate</span>
              <span className="font-medium">{hyperparams.learningRate}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Optimizer</span>
              <span className="font-medium uppercase">{hyperparams.optimizer}</span>
            </div>
          </CardContent>
        </Card>

        {/* SOTA Features */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm flex items-center gap-2">
              <Zap className="h-4 w-4" />
              SOTA Features
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-sm">
            <div className="flex flex-wrap gap-1">
              {hyperparams.useEma && <Badge variant="outline" className="text-xs">EMA</Badge>}
              {hyperparams.useMixedPrecision && <Badge variant="outline" className="text-xs">Mixed Precision</Badge>}
              {hyperparams.useLlrd && <Badge variant="outline" className="text-xs">LLRD</Badge>}
              {hyperparams.gradientClip > 0 && <Badge variant="outline" className="text-xs">Grad Clip</Badge>}
              {model.freezeBackbone && <Badge variant="outline" className="text-xs">Freeze Backbone</Badge>}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Estimated Time */}
      <Card className="bg-muted/50">
        <CardContent className="p-4">
          <div className="flex items-center gap-3">
            <Clock className="h-5 w-5 text-muted-foreground" />
            <div>
              <p className="font-medium">Estimated Training Time</p>
              <p className="text-2xl font-bold text-primary">{estimatedTime}</p>
              <p className="text-xs text-muted-foreground">
                Based on GPU performance estimates. Actual time may vary.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
