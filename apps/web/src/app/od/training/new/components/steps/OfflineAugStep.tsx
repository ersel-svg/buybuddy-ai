/**
 * OfflineAugStep Component
 *
 * Step 3: Offline augmentation configuration.
 */

"use client";

import { Copy, Info, AlertTriangle } from "lucide-react";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { cn } from "@/lib/utils";
import type { OfflineAugStepData, OfflineMultiplier, DatasetInfo } from "../../types/wizard";

interface OfflineAugStepProps {
  data: OfflineAugStepData;
  onChange: (data: Partial<OfflineAugStepData>) => void;
  datasetInfo: DatasetInfo | null;
  errors?: string[];
}

const MULTIPLIERS: { value: OfflineMultiplier; label: string; description: string }[] = [
  { value: 1, label: "1x", description: "No augmentation" },
  { value: 2, label: "2x", description: "Double dataset size" },
  { value: 3, label: "3x", description: "Triple dataset size" },
  { value: 5, label: "5x", description: "5x dataset size" },
  { value: 10, label: "10x", description: "10x dataset size (large storage)" },
];

export function OfflineAugStep({ data, onChange, datasetInfo, errors }: OfflineAugStepProps) {
  const resultingSize = datasetInfo
    ? datasetInfo.annotatedImageCount * (data.enabled ? data.multiplier : 1)
    : 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-lg font-semibold flex items-center gap-2">
          <Copy className="h-5 w-5" />
          Offline Augmentation
        </h2>
        <p className="text-sm text-muted-foreground mt-1">
          Generate augmented copies before training to increase dataset size.
        </p>
      </div>

      {/* Enable toggle */}
      <Card>
        <CardContent className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <Label className="text-base">Enable Offline Augmentation</Label>
              <p className="text-sm text-muted-foreground">
                Pre-generate augmented images (increases storage but improves training)
              </p>
            </div>
            <Switch
              checked={data.enabled}
              onCheckedChange={(enabled) => onChange({ enabled })}
            />
          </div>
        </CardContent>
      </Card>

      {data.enabled && (
        <>
          {/* Multiplier selection */}
          <div className="space-y-3">
            <Label className="text-base">Dataset Multiplier</Label>
            <div className="flex flex-wrap gap-2">
              {MULTIPLIERS.map((mult) => (
                <button
                  key={mult.value}
                  type="button"
                  onClick={() => onChange({ multiplier: mult.value })}
                  className={cn(
                    "px-4 py-2 rounded-md border text-sm font-medium transition-colors",
                    data.multiplier === mult.value
                      ? "border-primary bg-primary/10 text-primary"
                      : "border-input hover:border-primary/50"
                  )}
                >
                  {mult.label}
                </button>
              ))}
            </div>
            <p className="text-xs text-muted-foreground">
              {MULTIPLIERS.find((m) => m.value === data.multiplier)?.description}
            </p>
          </div>

          {/* Resulting size */}
          <Card className="bg-muted/50">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Resulting Dataset Size</span>
                <div className="text-right">
                  <span className="text-2xl font-bold">{resultingSize.toLocaleString()}</span>
                  <span className="text-sm text-muted-foreground ml-2">images</span>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Warning for large datasets */}
          {resultingSize > 50000 && (
            <Alert variant="default">
              <AlertTriangle className="h-4 w-4" />
              <AlertDescription>
                Large resulting dataset ({resultingSize.toLocaleString()} images) will require significant storage and preprocessing time.
              </AlertDescription>
            </Alert>
          )}

          {/* Augmentation options */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Augmentation Types</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Horizontal Flip */}
              <AugOption
                label="Horizontal Flip"
                enabled={data.augmentations.horizontalFlip.enabled}
                probability={data.augmentations.horizontalFlip.probability}
                onToggle={(enabled) =>
                  onChange({
                    augmentations: {
                      ...data.augmentations,
                      horizontalFlip: { ...data.augmentations.horizontalFlip, enabled },
                    },
                  })
                }
                onProbChange={(probability) =>
                  onChange({
                    augmentations: {
                      ...data.augmentations,
                      horizontalFlip: { ...data.augmentations.horizontalFlip, probability },
                    },
                  })
                }
              />

              {/* Rotate 90 */}
              <AugOption
                label="Rotate 90"
                enabled={data.augmentations.rotate90.enabled}
                probability={data.augmentations.rotate90.probability}
                onToggle={(enabled) =>
                  onChange({
                    augmentations: {
                      ...data.augmentations,
                      rotate90: { ...data.augmentations.rotate90, enabled },
                    },
                  })
                }
                onProbChange={(probability) =>
                  onChange({
                    augmentations: {
                      ...data.augmentations,
                      rotate90: { ...data.augmentations.rotate90, probability },
                    },
                  })
                }
              />

              {/* Brightness/Contrast */}
              <AugOption
                label="Brightness/Contrast"
                enabled={data.augmentations.brightnessContrast.enabled}
                probability={data.augmentations.brightnessContrast.probability}
                onToggle={(enabled) =>
                  onChange({
                    augmentations: {
                      ...data.augmentations,
                      brightnessContrast: { ...data.augmentations.brightnessContrast, enabled },
                    },
                  })
                }
                onProbChange={(probability) =>
                  onChange({
                    augmentations: {
                      ...data.augmentations,
                      brightnessContrast: { ...data.augmentations.brightnessContrast, probability },
                    },
                  })
                }
              />

              {/* Hue/Saturation */}
              <AugOption
                label="Hue/Saturation"
                enabled={data.augmentations.hueSaturation.enabled}
                probability={data.augmentations.hueSaturation.probability}
                onToggle={(enabled) =>
                  onChange({
                    augmentations: {
                      ...data.augmentations,
                      hueSaturation: { ...data.augmentations.hueSaturation, enabled },
                    },
                  })
                }
                onProbChange={(probability) =>
                  onChange({
                    augmentations: {
                      ...data.augmentations,
                      hueSaturation: { ...data.augmentations.hueSaturation, probability },
                    },
                  })
                }
              />
            </CardContent>
          </Card>
        </>
      )}

      {/* Info */}
      <Card className="bg-muted/50">
        <CardContent className="p-4">
          <div className="flex items-start gap-2">
            <Info className="h-4 w-4 mt-0.5 text-muted-foreground" />
            <div className="text-sm text-muted-foreground">
              <p>
                <strong>Offline augmentation</strong> creates new image files before training.
                This is useful for small datasets but increases storage requirements.
              </p>
              <p className="mt-2">
                For most cases, <strong>online augmentation</strong> (next step) is sufficient
                and more storage-efficient.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

interface AugOptionProps {
  label: string;
  enabled: boolean;
  probability: number;
  onToggle: (enabled: boolean) => void;
  onProbChange: (probability: number) => void;
}

function AugOption({ label, enabled, probability, onToggle, onProbChange }: AugOptionProps) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <Label>{label}</Label>
        <Switch checked={enabled} onCheckedChange={onToggle} />
      </div>
      {enabled && (
        <div className="pl-4 space-y-1">
          <div className="flex justify-between text-xs">
            <span className="text-muted-foreground">Probability</span>
            <span>{Math.round(probability * 100)}%</span>
          </div>
          <Slider
            value={[probability * 100]}
            onValueChange={(value) => onProbChange(value[0] / 100)}
            min={10}
            max={100}
            step={10}
          />
        </div>
      )}
    </div>
  );
}
