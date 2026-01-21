/**
 * Dataset Stats Hook
 *
 * Fetches and manages dataset statistics for the wizard.
 */

"use client";

import { useState, useEffect, useCallback } from "react";
import type { DatasetInfo, DatasetVersion } from "../types/wizard";

interface UseDatasetStatsProps {
  datasetId: string | null;
  versionId?: string | null;
}

interface UseDatasetStatsReturn {
  datasetInfo: DatasetInfo | null;
  versions: DatasetVersion[];
  isLoading: boolean;
  error: string | null;
  refetch: () => Promise<void>;
}

/**
 * Hook for fetching dataset statistics
 */
export function useDatasetStats({
  datasetId,
  versionId,
}: UseDatasetStatsProps): UseDatasetStatsReturn {
  const [datasetInfo, setDatasetInfo] = useState<DatasetInfo | null>(null);
  const [versions, setVersions] = useState<DatasetVersion[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchStats = useCallback(async () => {
    if (!datasetId) {
      setDatasetInfo(null);
      setVersions([]);
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      // Fetch dataset info
      const infoUrl = versionId
        ? `/api/v1/od/datasets/${datasetId}/versions/${versionId}/stats`
        : `/api/v1/od/datasets/${datasetId}/stats`;

      const infoResponse = await fetch(infoUrl);

      if (!infoResponse.ok) {
        throw new Error("Failed to fetch dataset stats");
      }

      const infoData = await infoResponse.json();

      // Transform API response to DatasetInfo
      const transformedInfo: DatasetInfo = {
        id: datasetId,
        name: infoData.name || "Unknown",
        imageCount: infoData.image_count || 0,
        annotatedImageCount: infoData.annotated_image_count || infoData.image_count || 0,
        annotationCount: infoData.annotation_count || 0,
        classNames: infoData.class_names || [],
        classDistribution: infoData.class_distribution || {},
        avgAnnotationsPerImage: infoData.avg_annotations_per_image || 0,
        minImageSize: infoData.min_image_size || { width: 0, height: 0 },
        maxImageSize: infoData.max_image_size || { width: 0, height: 0 },
        avgImageSize: infoData.avg_image_size || { width: 800, height: 600 },
      };

      setDatasetInfo(transformedInfo);

      // Fetch versions
      const versionsResponse = await fetch(
        `/api/v1/od/datasets/${datasetId}/versions`
      );

      if (versionsResponse.ok) {
        const versionsData = await versionsResponse.json();
        const transformedVersions: DatasetVersion[] = (versionsData.versions || []).map(
          (v: Record<string, unknown>) => ({
            id: v.id as string,
            versionNumber: v.version_number as number,
            name: v.name as string | null,
            imageCount: v.image_count as number,
            createdAt: v.created_at as string,
          })
        );
        setVersions(transformedVersions);
      }
    } catch (err) {
      console.error("Error fetching dataset stats:", err);
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setIsLoading(false);
    }
  }, [datasetId, versionId]);

  // Fetch on mount and when dataset changes
  useEffect(() => {
    fetchStats();
  }, [fetchStats]);

  return {
    datasetInfo,
    versions,
    isLoading,
    error,
    refetch: fetchStats,
  };
}

/**
 * Helper to format dataset size for display
 */
export function formatDatasetSize(count: number): string {
  if (count < 100) return "Very Small";
  if (count < 500) return "Small";
  if (count < 2000) return "Medium";
  if (count < 10000) return "Large";
  return "Very Large";
}

/**
 * Helper to get class balance status
 */
export function getClassBalanceStatus(
  distribution: Record<string, number>
): "balanced" | "slightly-imbalanced" | "imbalanced" | "severely-imbalanced" {
  const counts = Object.values(distribution);
  if (counts.length === 0) return "balanced";

  const min = Math.min(...counts);
  const max = Math.max(...counts);

  if (max <= min * 2) return "balanced";
  if (max <= min * 5) return "slightly-imbalanced";
  if (max <= min * 10) return "imbalanced";
  return "severely-imbalanced";
}

/**
 * Helper to format image size range
 */
export function formatImageSizeRange(
  min: { width: number; height: number },
  max: { width: number; height: number }
): string {
  const minDim = Math.min(min.width, min.height);
  const maxDim = Math.max(max.width, max.height);

  if (minDim === maxDim) {
    return `${minDim}px`;
  }

  return `${minDim}px - ${maxDim}px`;
}

/**
 * Check if dataset has small objects (based on avg annotation size)
 */
export function hasSmallObjects(
  avgImageSize: { width: number; height: number },
  avgAnnotationsPerImage: number,
  annotationCount: number,
  imageCount: number
): boolean {
  // Rough heuristic: if there are many annotations per image and images are large,
  // there's likely small objects
  if (avgAnnotationsPerImage > 20) return true;

  // If images are large but annotations are many
  const avgDim = (avgImageSize.width + avgImageSize.height) / 2;
  if (avgDim > 1000 && avgAnnotationsPerImage > 10) return true;

  return false;
}

/**
 * Check if dataset has dense annotations
 */
export function hasDenseAnnotations(avgAnnotationsPerImage: number): boolean {
  return avgAnnotationsPerImage > 20;
}
