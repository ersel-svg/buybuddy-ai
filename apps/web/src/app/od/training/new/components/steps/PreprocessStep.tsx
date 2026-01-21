/**
 * PreprocessStep Component
 *
 * Step 2: Image preprocessing configuration.
 */

"use client";

import { Image, Maximize, Grid3X3, RotateCw } from "lucide-react";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import type { PreprocessStepData, ResizeStrategy } from "../../types/wizard";

interface PreprocessStepProps {
  data: PreprocessStepData;
  onChange: (data: Partial<PreprocessStepData>) => void;
  errors?: string[];
}

const TARGET_SIZES = [416, 512, 640, 800, 1024];

const RESIZE_STRATEGIES: { value: ResizeStrategy; label: string; description: string }[] = [
  { value: "letterbox", label: "Letterbox", description: "Maintain aspect ratio with padding (recommended)" },
  { value: "stretch", label: "Stretch", description: "Stretch to target size (may distort)" },
  { value: "crop", label: "Center Crop", description: "Crop to target size from center" },
  { value: "fit", label: "Fit", description: "Fit within target, no padding" },
];

export function PreprocessStep({ data, onChange, errors }: PreprocessStepProps) {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-lg font-semibold flex items-center gap-2">
          <Image className="h-5 w-5" />
          Preprocessing
        </h2>
        <p className="text-sm text-muted-foreground mt-1">
          Configure how images are processed before training.
        </p>
      </div>

      {/* Target Size */}
      <div className="space-y-3">
        <Label className="text-base">Target Image Size</Label>
        <div className="flex flex-wrap gap-2">
          {TARGET_SIZES.map((size) => (
            <button
              key={size}
              type="button"
              onClick={() => onChange({ targetSize: size })}
              className={cn(
                "px-4 py-2 rounded-md border text-sm font-medium transition-colors",
                data.targetSize === size
                  ? "border-primary bg-primary/10 text-primary"
                  : "border-input hover:border-primary/50"
              )}
            >
              {size}px
            </button>
          ))}
        </div>
        <p className="text-xs text-muted-foreground">
          Larger sizes improve detection of small objects but require more GPU memory.
        </p>
      </div>

      {/* Resize Strategy */}
      <div className="space-y-3">
        <Label className="text-base">Resize Strategy</Label>
        <RadioGroup
          value={data.resizeStrategy}
          onValueChange={(value: ResizeStrategy) => onChange({ resizeStrategy: value })}
          className="grid grid-cols-1 md:grid-cols-2 gap-3"
        >
          {RESIZE_STRATEGIES.map((strategy) => (
            <Card
              key={strategy.value}
              className={cn(
                "cursor-pointer transition-all",
                data.resizeStrategy === strategy.value && "border-primary ring-1 ring-primary"
              )}
              onClick={() => onChange({ resizeStrategy: strategy.value })}
            >
              <CardContent className="p-3">
                <div className="flex items-center gap-2">
                  <RadioGroupItem value={strategy.value} id={strategy.value} />
                  <div>
                    <Label htmlFor={strategy.value} className="font-medium cursor-pointer">
                      {strategy.label}
                    </Label>
                    <p className="text-xs text-muted-foreground">{strategy.description}</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </RadioGroup>
      </div>

      {/* Tiling */}
      <div className="space-y-4 pt-4 border-t">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Grid3X3 className="h-4 w-4" />
            <div>
              <Label className="text-base">Image Tiling</Label>
              <p className="text-sm text-muted-foreground">
                Split large images into tiles for better small object detection
              </p>
            </div>
          </div>
          <Switch
            checked={data.tiling.enabled}
            onCheckedChange={(enabled) =>
              onChange({ tiling: { ...data.tiling, enabled } })
            }
          />
        </div>

        {data.tiling.enabled && (
          <div className="space-y-4 pl-6 border-l-2 border-muted">
            <div className="space-y-2">
              <div className="flex justify-between">
                <Label>Tile Size</Label>
                <span className="text-sm font-medium">{data.tiling.tileSize}px</span>
              </div>
              <Slider
                value={[data.tiling.tileSize]}
                onValueChange={(value) =>
                  onChange({ tiling: { ...data.tiling, tileSize: value[0] } })
                }
                min={320}
                max={1024}
                step={64}
              />
            </div>

            <div className="space-y-2">
              <div className="flex justify-between">
                <Label>Overlap</Label>
                <span className="text-sm font-medium">{Math.round(data.tiling.overlap * 100)}%</span>
              </div>
              <Slider
                value={[data.tiling.overlap * 100]}
                onValueChange={(value) =>
                  onChange({ tiling: { ...data.tiling, overlap: value[0] / 100 } })
                }
                min={0}
                max={50}
                step={5}
              />
            </div>
          </div>
        )}
      </div>

      {/* Auto Orient */}
      <div className="flex items-center justify-between pt-4 border-t">
        <div className="flex items-center gap-2">
          <RotateCw className="h-4 w-4" />
          <div>
            <Label className="text-base">Auto-Orient Images</Label>
            <p className="text-sm text-muted-foreground">
              Automatically fix EXIF orientation
            </p>
          </div>
        </div>
        <Switch
          checked={data.autoOrient}
          onCheckedChange={(autoOrient) => onChange({ autoOrient })}
        />
      </div>
    </div>
  );
}
