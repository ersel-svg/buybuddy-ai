/**
 * DatasetStatsCard Component
 *
 * Displays dataset statistics in a card format.
 */

"use client";

import {
  Image,
  Tag,
  Layers,
  BarChart3,
  AlertTriangle,
  CheckCircle,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";
import type { DatasetInfo } from "../types/wizard";
import {
  formatDatasetSize,
  getClassBalanceStatus,
  formatImageSizeRange,
} from "../hooks/useDatasetStats";

interface DatasetStatsCardProps {
  datasetInfo: DatasetInfo;
  className?: string;
}

export function DatasetStatsCard({
  datasetInfo,
  className,
}: DatasetStatsCardProps) {
  const sizeLabel = formatDatasetSize(datasetInfo.annotatedImageCount);
  const balanceStatus = getClassBalanceStatus(datasetInfo.classDistribution);
  const imageSizeRange = formatImageSizeRange(
    datasetInfo.minImageSize,
    datasetInfo.maxImageSize
  );

  const annotationPercentage =
    datasetInfo.imageCount > 0
      ? (datasetInfo.annotatedImageCount / datasetInfo.imageCount) * 100
      : 0;

  return (
    <Card className={className}>
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex items-center gap-2">
          <BarChart3 className="h-4 w-4" />
          Dataset Statistics
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Image counts */}
        <div className="grid grid-cols-2 gap-4">
          <StatItem
            icon={<Image className="h-4 w-4" />}
            label="Total Images"
            value={datasetInfo.imageCount.toLocaleString()}
          />
          <StatItem
            icon={<Tag className="h-4 w-4" />}
            label="Annotations"
            value={datasetInfo.annotationCount.toLocaleString()}
          />
        </div>

        {/* Annotation coverage */}
        <div className="space-y-1">
          <div className="flex justify-between text-sm">
            <span className="text-muted-foreground">Annotation Coverage</span>
            <span className="font-medium">
              {datasetInfo.annotatedImageCount.toLocaleString()} / {datasetInfo.imageCount.toLocaleString()}
            </span>
          </div>
          <Progress value={annotationPercentage} className="h-2" />
          {annotationPercentage < 100 && (
            <p className="text-xs text-muted-foreground">
              {(datasetInfo.imageCount - datasetInfo.annotatedImageCount).toLocaleString()} images not annotated
            </p>
          )}
        </div>

        {/* Size and classes */}
        <div className="grid grid-cols-2 gap-4">
          <StatItem
            icon={<Layers className="h-4 w-4" />}
            label="Classes"
            value={datasetInfo.classNames.length.toString()}
          />
          <div>
            <p className="text-xs text-muted-foreground mb-1">Dataset Size</p>
            <Badge variant="secondary">{sizeLabel}</Badge>
          </div>
        </div>

        {/* Image size range */}
        <div>
          <p className="text-xs text-muted-foreground mb-1">Image Size Range</p>
          <p className="text-sm font-medium">{imageSizeRange}</p>
        </div>

        {/* Avg annotations per image */}
        <StatItem
          icon={<Tag className="h-4 w-4" />}
          label="Avg Annotations/Image"
          value={datasetInfo.avgAnnotationsPerImage.toFixed(1)}
        />

        {/* Class balance status */}
        <div className="pt-2 border-t">
          <ClassBalanceIndicator status={balanceStatus} />
        </div>
      </CardContent>
    </Card>
  );
}

interface StatItemProps {
  icon: React.ReactNode;
  label: string;
  value: string;
}

function StatItem({ icon, label, value }: StatItemProps) {
  return (
    <div className="flex items-start gap-2">
      <div className="text-muted-foreground mt-0.5">{icon}</div>
      <div>
        <p className="text-xs text-muted-foreground">{label}</p>
        <p className="text-sm font-medium">{value}</p>
      </div>
    </div>
  );
}

interface ClassBalanceIndicatorProps {
  status: "balanced" | "slightly-imbalanced" | "imbalanced" | "severely-imbalanced";
}

function ClassBalanceIndicator({ status }: ClassBalanceIndicatorProps) {
  const configs = {
    balanced: {
      icon: <CheckCircle className="h-4 w-4 text-green-500" />,
      label: "Balanced",
      description: "Classes are well distributed",
      color: "text-green-600",
    },
    "slightly-imbalanced": {
      icon: <CheckCircle className="h-4 w-4 text-yellow-500" />,
      label: "Slightly Imbalanced",
      description: "Minor class imbalance detected",
      color: "text-yellow-600",
    },
    imbalanced: {
      icon: <AlertTriangle className="h-4 w-4 text-orange-500" />,
      label: "Imbalanced",
      description: "Consider class weighting or oversampling",
      color: "text-orange-600",
    },
    "severely-imbalanced": {
      icon: <AlertTriangle className="h-4 w-4 text-red-500" />,
      label: "Severely Imbalanced",
      description: "Highly recommend data augmentation for minority classes",
      color: "text-red-600",
    },
  };

  const config = configs[status];

  return (
    <div className="flex items-start gap-2">
      {config.icon}
      <div>
        <p className={cn("text-sm font-medium", config.color)}>
          {config.label}
        </p>
        <p className="text-xs text-muted-foreground">{config.description}</p>
      </div>
    </div>
  );
}

/**
 * Compact stats display for inline use
 */
export function DatasetStatsCompact({
  datasetInfo,
  className,
}: DatasetStatsCardProps) {
  return (
    <div className={cn("flex flex-wrap gap-4 text-sm", className)}>
      <span>
        <strong>{datasetInfo.annotatedImageCount.toLocaleString()}</strong> images
      </span>
      <span>
        <strong>{datasetInfo.annotationCount.toLocaleString()}</strong> annotations
      </span>
      <span>
        <strong>{datasetInfo.classNames.length}</strong> classes
      </span>
      <span>
        <strong>{datasetInfo.avgAnnotationsPerImage.toFixed(1)}</strong> avg/image
      </span>
    </div>
  );
}
