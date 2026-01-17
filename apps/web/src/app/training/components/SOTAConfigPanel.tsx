"use client";

import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Button } from "@/components/ui/button";
import {
  ChevronDown,
  Zap,
  Target,
  Layers,
  GraduationCap,
  Globe,
  Info,
  Hand,
} from "lucide-react";
import type { SOTAConfig } from "@/types";

interface SOTAConfigPanelProps {
  config: SOTAConfig;
  onChange: (config: SOTAConfig) => void;
  disabled?: boolean;
}

export function SOTAConfigPanel({ config, onChange, disabled }: SOTAConfigPanelProps) {
  const [lossOpen, setLossOpen] = useState(false);
  const [samplingOpen, setSamplingOpen] = useState(false);
  const [curriculumOpen, setCurriculumOpen] = useState(false);

  const updateConfig = (updates: Partial<SOTAConfig>) => {
    onChange({ ...config, ...updates });
  };

  const updateLoss = (updates: Partial<SOTAConfig["loss"]>) => {
    onChange({
      ...config,
      loss: { ...config.loss, ...updates },
    });
  };

  const updateSampling = (updates: Partial<SOTAConfig["sampling"]>) => {
    onChange({
      ...config,
      sampling: { ...config.sampling, ...updates },
    });
  };

  const updateCurriculum = (updates: Partial<SOTAConfig["curriculum"]>) => {
    onChange({
      ...config,
      curriculum: { ...config.curriculum, ...updates },
    });
  };

  return (
    <Card className={disabled ? "opacity-50" : ""}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Zap className="h-5 w-5 text-amber-500" />
            <CardTitle className="text-base">SOTA Training</CardTitle>
            <Badge variant="outline" className="text-xs">Advanced</Badge>
          </div>
          <Switch
            checked={config.enabled}
            onCheckedChange={(enabled) => updateConfig({ enabled })}
            disabled={disabled}
          />
        </div>
        <CardDescription className="text-xs">
          State-of-the-art training techniques: multi-loss, P-K sampling, curriculum learning
        </CardDescription>
      </CardHeader>

      {config.enabled && (
        <CardContent className="space-y-4">
          {/* Feature Toggles */}
          <div className="grid grid-cols-2 gap-3">
            {/* Combined Loss */}
            <FeatureToggle
              icon={<Target className="h-4 w-4" />}
              label="Combined Loss"
              description="ArcFace + Triplet + Domain"
              checked={config.use_combined_loss}
              onChange={(v) => updateConfig({ use_combined_loss: v })}
              disabled={disabled}
            />

            {/* P-K Sampling */}
            <FeatureToggle
              icon={<Layers className="h-4 w-4" />}
              label="P-K Sampling"
              description="P products x K samples"
              checked={config.use_pk_sampling}
              onChange={(v) => updateConfig({ use_pk_sampling: v })}
              disabled={disabled}
            />

            {/* Curriculum Learning */}
            <FeatureToggle
              icon={<GraduationCap className="h-4 w-4" />}
              label="Curriculum"
              description="Progressive difficulty"
              checked={config.use_curriculum}
              onChange={(v) => updateConfig({ use_curriculum: v })}
              disabled={disabled}
            />

            {/* Domain Adaptation */}
            <FeatureToggle
              icon={<Globe className="h-4 w-4" />}
              label="Domain Adapt"
              description="Synth-Real bridging"
              checked={config.use_domain_adaptation}
              onChange={(v) => updateConfig({ use_domain_adaptation: v })}
              disabled={disabled}
            />

            {/* Early Stopping */}
            <FeatureToggle
              icon={<Hand className="h-4 w-4" />}
              label="Early Stop"
              description="Stop on no improvement"
              checked={config.use_early_stopping}
              onChange={(v) => updateConfig({ use_early_stopping: v })}
              disabled={disabled}
            />
          </div>

          {/* Early Stopping Patience */}
          {config.use_early_stopping && (
            <div className="p-3 border rounded-lg bg-muted/30">
              <SliderWithLabel
                label="Early Stopping Patience"
                value={config.early_stopping_patience}
                onChange={(v) => updateConfig({ early_stopping_patience: v })}
                min={2}
                max={15}
                step={1}
                disabled={disabled}
                tooltip="Number of epochs without improvement before stopping training"
                formatValue={(v) => `${v} epochs`}
              />
            </div>
          )}

          <Separator />

          {/* Loss Configuration */}
          {config.use_combined_loss && (
            <Collapsible open={lossOpen} onOpenChange={setLossOpen}>
              <CollapsibleTrigger asChild>
                <Button variant="ghost" size="sm" className="w-full justify-between p-2 h-auto">
                  <span className="flex items-center gap-2 text-sm">
                    <Target className="h-4 w-4 text-blue-500" />
                    Loss Weights
                  </span>
                  <ChevronDown className={`h-4 w-4 transition-transform ${lossOpen ? "rotate-180" : ""}`} />
                </Button>
              </CollapsibleTrigger>
              <CollapsibleContent className="space-y-3 pt-2">
                {/* ArcFace Weight */}
                <SliderWithLabel
                  label="ArcFace Weight"
                  value={config.loss.arcface_weight}
                  onChange={(v) => updateLoss({ arcface_weight: v })}
                  min={0}
                  max={2}
                  step={0.1}
                  disabled={disabled}
                  tooltip="Weight for ArcFace classification loss (angular margin)"
                />

                {/* Triplet Weight */}
                <SliderWithLabel
                  label="Triplet Weight"
                  value={config.loss.triplet_weight}
                  onChange={(v) => updateLoss({ triplet_weight: v })}
                  min={0}
                  max={2}
                  step={0.1}
                  disabled={disabled}
                  tooltip="Weight for triplet loss (hard negative mining)"
                />

                {/* Domain Weight */}
                <SliderWithLabel
                  label="Domain Weight"
                  value={config.loss.domain_weight}
                  onChange={(v) => updateLoss({ domain_weight: v })}
                  min={0}
                  max={1}
                  step={0.05}
                  disabled={disabled}
                  tooltip="Weight for domain adversarial loss (synth-real alignment)"
                />

                <Separator className="my-2" />

                {/* ArcFace Margin */}
                <SliderWithLabel
                  label="ArcFace Margin"
                  value={config.loss.arcface_margin}
                  onChange={(v) => updateLoss({ arcface_margin: v })}
                  min={0.1}
                  max={1.0}
                  step={0.05}
                  disabled={disabled}
                  tooltip="Angular margin for ArcFace (higher = stricter separation)"
                />

                {/* ArcFace Scale */}
                <SliderWithLabel
                  label="ArcFace Scale"
                  value={config.loss.arcface_scale}
                  onChange={(v) => updateLoss({ arcface_scale: v })}
                  min={16}
                  max={128}
                  step={8}
                  disabled={disabled}
                  tooltip="Scale factor for ArcFace logits"
                />

                {/* Triplet Margin */}
                <SliderWithLabel
                  label="Triplet Margin"
                  value={config.loss.triplet_margin}
                  onChange={(v) => updateLoss({ triplet_margin: v })}
                  min={0.1}
                  max={1.0}
                  step={0.05}
                  disabled={disabled}
                  tooltip="Margin for triplet loss (distance between positive and negative)"
                />
              </CollapsibleContent>
            </Collapsible>
          )}

          {/* Sampling Configuration */}
          {config.use_pk_sampling && (
            <Collapsible open={samplingOpen} onOpenChange={setSamplingOpen}>
              <CollapsibleTrigger asChild>
                <Button variant="ghost" size="sm" className="w-full justify-between p-2 h-auto">
                  <span className="flex items-center gap-2 text-sm">
                    <Layers className="h-4 w-4 text-green-500" />
                    P-K Sampling
                  </span>
                  <ChevronDown className={`h-4 w-4 transition-transform ${samplingOpen ? "rotate-180" : ""}`} />
                </Button>
              </CollapsibleTrigger>
              <CollapsibleContent className="space-y-3 pt-2">
                {/* Products per Batch (P) */}
                <SliderWithLabel
                  label="Products per Batch (P)"
                  value={config.sampling.products_per_batch}
                  onChange={(v) => updateSampling({ products_per_batch: v })}
                  min={4}
                  max={32}
                  step={2}
                  disabled={disabled}
                  tooltip="Number of unique products in each batch"
                />

                {/* Samples per Product (K) */}
                <SliderWithLabel
                  label="Samples per Product (K)"
                  value={config.sampling.samples_per_product}
                  onChange={(v) => updateSampling({ samples_per_product: v })}
                  min={2}
                  max={8}
                  step={1}
                  disabled={disabled}
                  tooltip="Number of images per product in each batch"
                />

                {/* Effective Batch Size Display */}
                <div className="bg-muted/50 p-2 rounded text-xs">
                  <span className="text-muted-foreground">Effective batch size: </span>
                  <span className="font-medium">
                    {config.sampling.products_per_batch * config.sampling.samples_per_product}
                  </span>
                  <span className="text-muted-foreground"> (P x K)</span>
                </div>

                {/* Synthetic Ratio */}
                <SliderWithLabel
                  label="Synthetic Ratio"
                  value={config.sampling.synthetic_ratio}
                  onChange={(v) => updateSampling({ synthetic_ratio: v })}
                  min={0}
                  max={1}
                  step={0.1}
                  disabled={disabled}
                  tooltip="Target ratio of synthetic vs real images in each batch"
                  formatValue={(v) => `${(v * 100).toFixed(0)}%`}
                />
              </CollapsibleContent>
            </Collapsible>
          )}

          {/* Curriculum Configuration */}
          {config.use_curriculum && (
            <Collapsible open={curriculumOpen} onOpenChange={setCurriculumOpen}>
              <CollapsibleTrigger asChild>
                <Button variant="ghost" size="sm" className="w-full justify-between p-2 h-auto">
                  <span className="flex items-center gap-2 text-sm">
                    <GraduationCap className="h-4 w-4 text-purple-500" />
                    Curriculum Phases
                  </span>
                  <ChevronDown className={`h-4 w-4 transition-transform ${curriculumOpen ? "rotate-180" : ""}`} />
                </Button>
              </CollapsibleTrigger>
              <CollapsibleContent className="space-y-3 pt-2">
                {/* Phase visualization */}
                <div className="flex items-center gap-1 text-xs">
                  <Badge variant="outline" className="bg-blue-50 border-blue-200">
                    Warmup: {config.curriculum.warmup_epochs}
                  </Badge>
                  <span className="text-muted-foreground">→</span>
                  <Badge variant="outline" className="bg-green-50 border-green-200">
                    Easy: {config.curriculum.easy_epochs}
                  </Badge>
                  <span className="text-muted-foreground">→</span>
                  <Badge variant="outline" className="bg-orange-50 border-orange-200">
                    Hard: {config.curriculum.hard_epochs}
                  </Badge>
                  <span className="text-muted-foreground">→</span>
                  <Badge variant="outline" className="bg-purple-50 border-purple-200">
                    Finetune: {config.curriculum.finetune_epochs}
                  </Badge>
                </div>

                {/* Total epochs from curriculum */}
                <div className="bg-muted/50 p-2 rounded text-xs">
                  <span className="text-muted-foreground">Total curriculum epochs: </span>
                  <span className="font-medium">
                    {config.curriculum.warmup_epochs +
                      config.curriculum.easy_epochs +
                      config.curriculum.hard_epochs +
                      config.curriculum.finetune_epochs}
                  </span>
                </div>

                {/* Warmup Epochs */}
                <SliderWithLabel
                  label="Warmup Epochs"
                  value={config.curriculum.warmup_epochs}
                  onChange={(v) => updateCurriculum({ warmup_epochs: v })}
                  min={0}
                  max={5}
                  step={1}
                  disabled={disabled}
                  tooltip="Initial epochs with lower LR, no hard mining"
                />

                {/* Easy Epochs */}
                <SliderWithLabel
                  label="Easy Epochs"
                  value={config.curriculum.easy_epochs}
                  onChange={(v) => updateCurriculum({ easy_epochs: v })}
                  min={1}
                  max={15}
                  step={1}
                  disabled={disabled}
                  tooltip="Train on easy examples (high-confidence)"
                />

                {/* Hard Epochs */}
                <SliderWithLabel
                  label="Hard Epochs"
                  value={config.curriculum.hard_epochs}
                  onChange={(v) => updateCurriculum({ hard_epochs: v })}
                  min={1}
                  max={20}
                  step={1}
                  disabled={disabled}
                  tooltip="Train on hard examples (low-confidence, confused pairs)"
                />

                {/* Finetune Epochs */}
                <SliderWithLabel
                  label="Finetune Epochs"
                  value={config.curriculum.finetune_epochs}
                  onChange={(v) => updateCurriculum({ finetune_epochs: v })}
                  min={0}
                  max={10}
                  step={1}
                  disabled={disabled}
                  tooltip="Final epochs with full dataset and low LR"
                />
              </CollapsibleContent>
            </Collapsible>
          )}

          {/* Triplet Mining Run ID */}
          {config.use_combined_loss && (
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Label className="text-xs">Triplet Mining Run ID</Label>
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger>
                      <Info className="h-3 w-3 text-muted-foreground" />
                    </TooltipTrigger>
                    <TooltipContent>
                      <p className="text-xs max-w-xs">
                        Optional: ID of a triplet mining run to use pre-computed hard negatives.
                        Create one from the Triplets page first.
                      </p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </div>
              <Input
                placeholder="e.g., abc123-def456..."
                value={config.triplet_mining_run_id || ""}
                onChange={(e) => updateConfig({ triplet_mining_run_id: e.target.value || undefined })}
                disabled={disabled}
                className="text-xs font-mono h-8"
              />
            </div>
          )}
        </CardContent>
      )}
    </Card>
  );
}

// Helper component for feature toggles
function FeatureToggle({
  icon,
  label,
  description,
  checked,
  onChange,
  disabled,
}: {
  icon: React.ReactNode;
  label: string;
  description: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
}) {
  return (
    <div
      className={`flex items-center justify-between p-2 rounded-md border cursor-pointer transition-colors ${
        checked
          ? "border-primary/50 bg-primary/5"
          : "border-border hover:border-primary/30"
      } ${disabled ? "opacity-50 cursor-not-allowed" : ""}`}
      onClick={() => !disabled && onChange(!checked)}
    >
      <div className="flex items-center gap-2">
        <div className={checked ? "text-primary" : "text-muted-foreground"}>{icon}</div>
        <div>
          <p className="text-xs font-medium">{label}</p>
          <p className="text-[10px] text-muted-foreground">{description}</p>
        </div>
      </div>
      <Switch checked={checked} onCheckedChange={onChange} disabled={disabled} />
    </div>
  );
}

// Helper component for slider with label
function SliderWithLabel({
  label,
  value,
  onChange,
  min,
  max,
  step,
  disabled,
  tooltip,
  formatValue = (v) => v.toFixed(2),
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  step: number;
  disabled?: boolean;
  tooltip?: string;
  formatValue?: (value: number) => string;
}) {
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1">
          <Label className="text-xs">{label}</Label>
          {tooltip && (
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger>
                  <Info className="h-3 w-3 text-muted-foreground" />
                </TooltipTrigger>
                <TooltipContent>
                  <p className="text-xs max-w-xs">{tooltip}</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          )}
        </div>
        <span className="text-xs font-mono text-muted-foreground">{formatValue(value)}</span>
      </div>
      <Slider
        value={[value]}
        onValueChange={([v]) => onChange(v)}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        className="w-full"
      />
    </div>
  );
}

export default SOTAConfigPanel;
