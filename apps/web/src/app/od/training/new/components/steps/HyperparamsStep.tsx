/**
 * HyperparamsStep Component
 *
 * Step 6: Training hyperparameters configuration.
 */

"use client";

import { SlidersHorizontal, Sparkles } from "lucide-react";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { DataLoadingConfigPanel } from "@/components/training/DataLoadingConfig";
import type { HyperparamsStepData, OptimizerType, SchedulerType } from "../../types/wizard";

interface HyperparamsStepProps {
  data: HyperparamsStepData;
  onChange: (data: Partial<HyperparamsStepData>) => void;
  errors?: string[];
}

export function HyperparamsStep({ data, onChange, errors }: HyperparamsStepProps) {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-lg font-semibold flex items-center gap-2">
          <SlidersHorizontal className="h-5 w-5" />
          Hyperparameters
        </h2>
        <p className="text-sm text-muted-foreground mt-1">
          Fine-tune training parameters. Default values are optimized for most use cases.
        </p>
      </div>

      {/* Basic Settings */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base">Basic Settings</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Epochs */}
            <div className="space-y-2">
              <Label>Epochs</Label>
              <Input
                type="number"
                value={data.epochs}
                onChange={(e) => onChange({ epochs: parseInt(e.target.value) || 100 })}
                min={1}
                max={500}
              />
            </div>

            {/* Batch Size */}
            <div className="space-y-2">
              <Label>Batch Size</Label>
              <Select
                value={data.batchSize.toString()}
                onValueChange={(value) => onChange({ batchSize: parseInt(value) })}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {[4, 8, 12, 16, 24, 32, 48, 64].map((size) => (
                    <SelectItem key={size} value={size.toString()}>
                      {size}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Learning Rate */}
            <div className="space-y-2">
              <Label>Learning Rate</Label>
              <Select
                value={data.learningRate.toString()}
                onValueChange={(value) => onChange({ learningRate: parseFloat(value) })}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {[0.00001, 0.00005, 0.0001, 0.0002, 0.0005, 0.001].map((lr) => (
                    <SelectItem key={lr} value={lr.toString()}>
                      {lr}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Optimizer & Scheduler */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base">Optimizer & Scheduler</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Optimizer */}
            <div className="space-y-2">
              <Label>Optimizer</Label>
              <Select
                value={data.optimizer}
                onValueChange={(value: OptimizerType) => onChange({ optimizer: value })}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="adamw">AdamW (Recommended)</SelectItem>
                  <SelectItem value="adam">Adam</SelectItem>
                  <SelectItem value="sgd">SGD</SelectItem>
                  <SelectItem value="lion">Lion</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Scheduler */}
            <div className="space-y-2">
              <Label>LR Scheduler</Label>
              <Select
                value={data.scheduler}
                onValueChange={(value: SchedulerType) => onChange({ scheduler: value })}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="cosine">Cosine (Recommended)</SelectItem>
                  <SelectItem value="step">Step</SelectItem>
                  <SelectItem value="linear">Linear</SelectItem>
                  <SelectItem value="onecycle">OneCycle</SelectItem>
                  <SelectItem value="plateau">Plateau</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Warmup Epochs */}
            <div className="space-y-2">
              <div className="flex justify-between">
                <Label>Warmup Epochs</Label>
                <span className="text-sm">{data.warmupEpochs}</span>
              </div>
              <Slider
                value={[data.warmupEpochs]}
                onValueChange={(value) => onChange({ warmupEpochs: value[0] })}
                min={0}
                max={10}
                step={1}
              />
            </div>

            {/* Weight Decay */}
            <div className="space-y-2">
              <Label>Weight Decay</Label>
              <Input
                type="number"
                value={data.weightDecay}
                onChange={(e) => onChange({ weightDecay: parseFloat(e.target.value) || 0.0001 })}
                step={0.0001}
                min={0}
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* SOTA Features */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base flex items-center gap-2">
            <Sparkles className="h-4 w-4" />
            SOTA Features
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* EMA */}
          <div className="flex items-center justify-between">
            <div>
              <Label>Exponential Moving Average (EMA)</Label>
              <p className="text-xs text-muted-foreground">
                Maintains moving average of model weights for smoother training
              </p>
            </div>
            <Switch
              checked={data.useEma}
              onCheckedChange={(useEma) => onChange({ useEma })}
            />
          </div>

          {/* Mixed Precision */}
          <div className="flex items-center justify-between">
            <div>
              <Label>Mixed Precision (FP16)</Label>
              <p className="text-xs text-muted-foreground">
                Faster training with reduced memory usage
              </p>
            </div>
            <Switch
              checked={data.useMixedPrecision}
              onCheckedChange={(useMixedPrecision) => onChange({ useMixedPrecision })}
            />
          </div>

          {/* LLRD */}
          <div className="flex items-center justify-between">
            <div>
              <Label>Layer-wise LR Decay (LLRD)</Label>
              <p className="text-xs text-muted-foreground">
                Lower learning rates for earlier layers
              </p>
            </div>
            <Switch
              checked={data.useLlrd}
              onCheckedChange={(useLlrd) => onChange({ useLlrd })}
            />
          </div>

          {/* Gradient Clipping */}
          <div className="space-y-2">
            <div className="flex justify-between">
              <Label>Gradient Clipping</Label>
              <span className="text-sm">{data.gradientClip}</span>
            </div>
            <Slider
              value={[data.gradientClip]}
              onValueChange={(value) => onChange({ gradientClip: value[0] })}
              min={0}
              max={5}
              step={0.1}
            />
          </div>
        </CardContent>
      </Card>

      {/* Early Stopping */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base">Early Stopping</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <div className="flex justify-between">
              <Label>Patience (epochs)</Label>
              <span className="text-sm">{data.patience}</span>
            </div>
            <Slider
              value={[data.patience]}
              onValueChange={(value) => onChange({ patience: value[0] })}
              min={5}
              max={50}
              step={5}
            />
            <p className="text-xs text-muted-foreground">
              Stop training if no improvement for {data.patience} epochs
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Data Loading (Advanced) */}
      <DataLoadingConfigPanel
        value={data.dataLoading}
        onChange={(dataLoading) => onChange({ dataLoading })}
        showDataLoader={true}
        defaultOpen={true}
      />
    </div>
  );
}
