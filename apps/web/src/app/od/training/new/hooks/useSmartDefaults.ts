/**
 * Smart Defaults Hook
 *
 * Provides smart default recommendations based on dataset characteristics.
 */

"use client";

import { useMemo, useCallback } from "react";
import type { DatasetInfo, SmartDefaults, AugmentationPreset } from "../types/wizard";
import {
  generateSmartDefaults,
  analyzeDatasetForDefaults,
  getPresetRecommendation,
  recommendBatchSize,
} from "../utils/smartDefaults";

interface UseSmartDefaultsProps {
  datasetInfo: DatasetInfo | null;
}

interface UseSmartDefaultsReturn {
  smartDefaults: SmartDefaults | null;
  presetRecommendation: { preset: AugmentationPreset; reason: string } | null;
  recommendedBatchSize: number;
  isAnalyzing: boolean;
  analyzeDataset: (info: DatasetInfo) => SmartDefaults;
}

/**
 * Hook for generating and accessing smart defaults
 */
export function useSmartDefaults({
  datasetInfo,
}: UseSmartDefaultsProps): UseSmartDefaultsReturn {
  // Generate smart defaults when dataset info is available
  const smartDefaults = useMemo(() => {
    if (!datasetInfo) return null;

    const input = analyzeDatasetForDefaults({
      imageCount: datasetInfo.imageCount,
      annotatedImageCount: datasetInfo.annotatedImageCount,
      annotationCount: datasetInfo.annotationCount,
      classCount: datasetInfo.classNames.length,
      avgImageSize: datasetInfo.avgImageSize,
      avgAnnotationsPerImage: datasetInfo.avgAnnotationsPerImage,
    });

    return generateSmartDefaults(input);
  }, [datasetInfo]);

  // Preset recommendation
  const presetRecommendation = useMemo(() => {
    if (!datasetInfo) return null;
    return getPresetRecommendation(datasetInfo.annotatedImageCount);
  }, [datasetInfo]);

  // Recommended batch size based on model defaults
  const recommendedBatchSize = useMemo(() => {
    if (!smartDefaults?.model) return 16;

    return recommendBatchSize(
      smartDefaults.model.modelType || "rt-detr",
      smartDefaults.model.modelSize || "l",
      smartDefaults.preprocess?.targetSize || 640,
      16 // Default GPU memory assumption
    );
  }, [smartDefaults]);

  // Manual analysis function
  const analyzeDataset = useCallback((info: DatasetInfo): SmartDefaults => {
    const input = analyzeDatasetForDefaults({
      imageCount: info.imageCount,
      annotatedImageCount: info.annotatedImageCount,
      annotationCount: info.annotationCount,
      classCount: info.classNames.length,
      avgImageSize: info.avgImageSize,
      avgAnnotationsPerImage: info.avgAnnotationsPerImage,
    });

    return generateSmartDefaults(input);
  }, []);

  return {
    smartDefaults,
    presetRecommendation,
    recommendedBatchSize,
    isAnalyzing: false, // Could be used with async analysis
    analyzeDataset,
  };
}

/**
 * Get recommendation summary text
 */
export function getRecommendationSummary(
  smartDefaults: SmartDefaults | null
): string[] {
  if (!smartDefaults) return [];

  const summary: string[] = [];

  // Preprocessing
  if (smartDefaults.preprocess?.targetSize) {
    summary.push(
      `Image size: ${smartDefaults.preprocess.targetSize}px`
    );
  }

  if (smartDefaults.preprocess?.tiling?.enabled) {
    summary.push("Tiling enabled for large images");
  }

  // Offline augmentation
  if (smartDefaults.offlineAug?.enabled) {
    summary.push(
      `Offline augmentation: ${smartDefaults.offlineAug.multiplier}x multiplier`
    );
  }

  // Online augmentation
  if (smartDefaults.onlineAug?.preset) {
    summary.push(`Augmentation preset: ${smartDefaults.onlineAug.preset}`);
  }

  // Model
  if (smartDefaults.model?.modelType && smartDefaults.model?.modelSize) {
    summary.push(
      `Model: ${smartDefaults.model.modelType.toUpperCase()}-${smartDefaults.model.modelSize.toUpperCase()}`
    );
  }

  // Training
  if (smartDefaults.hyperparams?.epochs) {
    summary.push(`Epochs: ${smartDefaults.hyperparams.epochs}`);
  }

  if (smartDefaults.hyperparams?.batchSize) {
    summary.push(`Batch size: ${smartDefaults.hyperparams.batchSize}`);
  }

  return summary;
}

/**
 * Check if defaults differ significantly from user config
 */
export function hasSignificantDifference(
  current: Record<string, unknown>,
  recommended: Record<string, unknown>
): boolean {
  const significantKeys = [
    "epochs",
    "batchSize",
    "learningRate",
    "preset",
    "modelType",
    "modelSize",
  ];

  for (const key of significantKeys) {
    if (current[key] !== undefined && recommended[key] !== undefined) {
      if (current[key] !== recommended[key]) {
        return true;
      }
    }
  }

  return false;
}
