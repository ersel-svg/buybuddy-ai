/**
 * OnlineAugStep Component
 *
 * Step 4: Online augmentation preset selection.
 */

"use client";

import {
  Rocket,
  Star,
  Flame,
  Zap,
  Feather,
  X,
  Settings,
  Info,
  Check,
} from "lucide-react";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import type { OnlineAugStepData, AugmentationPreset } from "../../types/wizard";

interface OnlineAugStepProps {
  data: OnlineAugStepData;
  onChange: (data: Partial<OnlineAugStepData>) => void;
  recommendedPreset?: AugmentationPreset;
  errors?: string[];
}

const PRESET_INFO: Record<AugmentationPreset, {
  name: string;
  description: string;
  icon: React.ReactNode;
  boost: string;
  features: string[];
  recommended?: boolean;
}> = {
  "sota-v2": {
    name: "SOTA-v2",
    description: "Next-gen augmentation combining YOLOv8, RT-DETR, and D-FINE best practices",
    icon: <Rocket className="h-5 w-5 text-primary" />,
    boost: "+4-6% mAP",
    features: [
      "Optimized Mosaic (p=0.5)",
      "MixUp with high alpha",
      "CopyPaste augmentation",
      "ShiftScaleRotate",
      "ColorJitter & RandomGamma",
      "JPEG compression & downscale",
      "Coarse dropout",
    ],
    recommended: true,
  },
  sota: {
    name: "SOTA (Legacy)",
    description: "Classic state-of-the-art augmentation pipeline",
    icon: <Star className="h-5 w-5 text-yellow-500" />,
    boost: "+3-5% mAP",
    features: [
      "Mosaic 4-image",
      "MixUp blending",
      "CopyPaste",
      "Brightness/Contrast",
      "Hue/Saturation",
      "Gaussian blur",
    ],
  },
  heavy: {
    name: "Heavy",
    description: "Maximum augmentation for small datasets (<1000 images)",
    icon: <Flame className="h-5 w-5 text-orange-500" />,
    boost: "+5-8% mAP",
    features: [
      "All multi-image augs",
      "Full geometric suite",
      "Comprehensive color augs",
      "Weather effects",
      "Noise & blur",
      "Grid dropout",
    ],
  },
  medium: {
    name: "Medium",
    description: "Balanced augmentation for general use",
    icon: <Zap className="h-5 w-5 text-blue-500" />,
    boost: "+2-3% mAP",
    features: [
      "Mosaic & MixUp (lower prob)",
      "Essential geometric",
      "Basic color augs",
      "Light JPEG compression",
    ],
  },
  light: {
    name: "Light",
    description: "Basic augmentations only, fast training",
    icon: <Feather className="h-5 w-5 text-green-500" />,
    boost: "+1% mAP",
    features: [
      "Horizontal flip",
      "Light scale",
      "Light brightness",
    ],
  },
  none: {
    name: "None",
    description: "No augmentation, for baseline comparison",
    icon: <X className="h-5 w-5 text-muted-foreground" />,
    boost: "Baseline",
    features: [],
  },
  custom: {
    name: "Custom",
    description: "Configure augmentations manually",
    icon: <Settings className="h-5 w-5 text-purple-500" />,
    boost: "Variable",
    features: [
      "Full control over each augmentation",
      "Set individual probabilities",
      "Configure parameters",
    ],
  },
};

export function OnlineAugStep({
  data,
  onChange,
  recommendedPreset,
  errors,
}: OnlineAugStepProps) {
  const presets: AugmentationPreset[] = [
    "sota-v2",
    "sota",
    "heavy",
    "medium",
    "light",
    "none",
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-lg font-semibold flex items-center gap-2">
          <Zap className="h-5 w-5" />
          Online Augmentation
        </h2>
        <p className="text-sm text-muted-foreground mt-1">
          Select an augmentation preset. Augmentations are applied during training to improve model generalization.
        </p>
      </div>

      {/* Recommendation */}
      {recommendedPreset && (
        <Card className="border-primary/30 bg-primary/5">
          <CardContent className="p-4">
            <div className="flex items-start gap-3">
              <div className="p-2 rounded-full bg-primary/10">
                <Rocket className="h-4 w-4 text-primary" />
              </div>
              <div>
                <p className="font-medium">
                  Recommended: {PRESET_INFO[recommendedPreset].name}
                </p>
                <p className="text-sm text-muted-foreground">
                  Based on your dataset size and characteristics
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Preset Selection */}
      <RadioGroup
        value={data.preset}
        onValueChange={(value: AugmentationPreset) => onChange({ preset: value })}
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4"
      >
        {presets.map((preset) => {
          const info = PRESET_INFO[preset];
          const isSelected = data.preset === preset;
          const isRecommended = preset === recommendedPreset || info.recommended;

          return (
            <Card
              key={preset}
              className={cn(
                "cursor-pointer transition-all relative",
                isSelected && "border-primary ring-1 ring-primary",
                !isSelected && "hover:border-muted-foreground/30"
              )}
              onClick={() => onChange({ preset })}
            >
              {isRecommended && (
                <Badge
                  className="absolute -top-2 -right-2 z-10"
                  variant="default"
                >
                  Recommended
                </Badge>
              )}
              <CardContent className="p-4">
                <div className="flex items-start gap-3">
                  <RadioGroupItem value={preset} id={preset} className="mt-1" />
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      {info.icon}
                      <Label htmlFor={preset} className="font-semibold cursor-pointer">
                        {info.name}
                      </Label>
                    </div>
                    <p className="text-sm text-muted-foreground mt-1">
                      {info.description}
                    </p>
                    <Badge variant="secondary" className="mt-2">
                      {info.boost}
                    </Badge>

                    {/* Features list */}
                    {info.features.length > 0 && (
                      <ul className="mt-3 space-y-1">
                        {info.features.slice(0, 4).map((feature, i) => (
                          <li
                            key={i}
                            className="text-xs text-muted-foreground flex items-center gap-1"
                          >
                            <Check className="h-3 w-3 text-green-500 flex-shrink-0" />
                            {feature}
                          </li>
                        ))}
                        {info.features.length > 4 && (
                          <li className="text-xs text-muted-foreground">
                            +{info.features.length - 4} more...
                          </li>
                        )}
                      </ul>
                    )}
                  </div>
                </div>
              </CardContent>
            </Card>
          );
        })}

        {/* Custom option */}
        <Card
          className={cn(
            "cursor-pointer transition-all opacity-60",
            data.preset === "custom" && "border-primary ring-1 ring-primary opacity-100"
          )}
          onClick={() => onChange({ preset: "custom" })}
        >
          <CardContent className="p-4">
            <div className="flex items-start gap-3">
              <RadioGroupItem
                value="custom"
                id="custom"
                className="mt-1"
              />
              <div className="flex-1">
                <div className="flex items-center gap-2">
                  {PRESET_INFO.custom.icon}
                  <Label htmlFor="custom" className="font-semibold cursor-pointer">
                    {PRESET_INFO.custom.name}
                  </Label>
                  <Badge variant="outline">Advanced</Badge>
                </div>
                <p className="text-sm text-muted-foreground mt-1">
                  {PRESET_INFO.custom.description}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </RadioGroup>

      {/* Info box */}
      <Card className="bg-muted/50">
        <CardContent className="p-4">
          <div className="flex items-start gap-2">
            <Info className="h-4 w-4 text-muted-foreground mt-0.5" />
            <div className="text-sm text-muted-foreground">
              <p>
                <strong>Online augmentation</strong> is applied during training in real-time.
                It doesn't increase storage but adds computational overhead.
              </p>
              <p className="mt-2">
                For small datasets (&lt;1000 images), consider using{" "}
                <strong>Heavy</strong> augmentation or enabling{" "}
                <strong>Offline Augmentation</strong> in the previous step.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
