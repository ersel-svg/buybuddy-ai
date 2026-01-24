/**
 * DatasetStep Component
 *
 * Step 1: Dataset selection and train/val split configuration.
 */

"use client";

import { useEffect, useState } from "react";
import { Database, ChevronDown, AlertCircle, CheckCircle2 } from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";
import { apiClient } from "@/lib/api-client";
import type { DatasetStepData, DatasetInfo, DatasetVersion } from "../../types/wizard";
import { DatasetStatsCard } from "../DatasetStatsCard";

interface DatasetStepProps {
  data: DatasetStepData;
  onChange: (data: Partial<DatasetStepData>) => void;
  datasetInfo: DatasetInfo | null;
  onDatasetInfoChange: (info: DatasetInfo | null) => void;
  errors?: string[];
}

interface Dataset {
  id: string;
  name: string;
  imageCount: number;
  annotatedImageCount: number;
}

export function DatasetStep({
  data,
  onChange,
  datasetInfo,
  onDatasetInfoChange,
  errors,
}: DatasetStepProps) {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [versions, setVersions] = useState<DatasetVersion[]>([]);
  const [isLoadingDatasets, setIsLoadingDatasets] = useState(true);
  const [isLoadingInfo, setIsLoadingInfo] = useState(false);
  const [fetchError, setFetchError] = useState<string | null>(null);

  // Fetch datasets on mount
  useEffect(() => {
    async function fetchDatasets() {
      setIsLoadingDatasets(true);
      try {
        const result = await apiClient.getODDatasets();
        setDatasets(
          result.map((d) => ({
            id: d.id,
            name: d.name,
            imageCount: d.image_count,
            annotatedImageCount: d.annotated_image_count,
          }))
        );
      } catch (err) {
        setFetchError("Failed to load datasets");
      } finally {
        setIsLoadingDatasets(false);
      }
    }
    fetchDatasets();
  }, []);

  // Fetch dataset info when selection changes
  useEffect(() => {
    async function fetchDatasetInfo() {
      if (!data.datasetId) {
        onDatasetInfoChange(null);
        setVersions([]);
        return;
      }

      setIsLoadingInfo(true);
      try {
        // Fetch detailed stats for training
        const statsData = await apiClient.getODDatasetTrainingStats(
          data.datasetId,
          data.versionId
        );

        onDatasetInfoChange({
          id: data.datasetId,
          name: statsData.name || "Unknown",
          imageCount: statsData.image_count || 0,
          annotatedImageCount: statsData.annotated_image_count || statsData.image_count || 0,
          annotationCount: statsData.annotation_count || 0,
          classNames: statsData.class_names || [],
          classDistribution: statsData.class_distribution || {},
          avgAnnotationsPerImage: statsData.avg_annotations_per_image || 0,
          minImageSize: statsData.min_image_size || { width: 0, height: 0 },
          maxImageSize: statsData.max_image_size || { width: 0, height: 0 },
          avgImageSize: statsData.avg_image_size || { width: 800, height: 600 },
        });

        // Fetch versions
        try {
          const versionsData = await apiClient.getODDatasetVersions(data.datasetId);
          setVersions(
            versionsData.map((v) => ({
              id: v.id,
              versionNumber: v.version_number,
              name: v.name || null,
              imageCount: v.image_count,
              createdAt: v.created_at,
            }))
          );
        } catch {
          // Versions fetch failed, not critical
          setVersions([]);
        }
      } catch (err) {
        console.error("Error fetching dataset info:", err);
      } finally {
        setIsLoadingInfo(false);
      }
    }

    fetchDatasetInfo();
  }, [data.datasetId, data.versionId, onDatasetInfoChange]);

  // Calculate test split from train and val
  const testSplit = Math.max(0, 1 - data.trainSplit - data.valSplit);

  // Handle split changes
  const handleTrainSplitChange = (value: number[]) => {
    const newTrain = value[0] / 100;
    // Adjust val if needed to not exceed 100%
    const maxVal = 1 - newTrain;
    const newVal = Math.min(data.valSplit, maxVal);
    onChange({ trainSplit: newTrain, valSplit: newVal, testSplit: 1 - newTrain - newVal });
  };

  const handleValSplitChange = (value: number[]) => {
    const newVal = value[0] / 100;
    const maxVal = 1 - data.trainSplit;
    onChange({ valSplit: Math.min(newVal, maxVal), testSplit: 1 - data.trainSplit - Math.min(newVal, maxVal) });
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-lg font-semibold flex items-center gap-2">
          <Database className="h-5 w-5" />
          Select Dataset
        </h2>
        <p className="text-sm text-muted-foreground mt-1">
          Choose a dataset and configure the train/validation split.
        </p>
      </div>

      {/* Errors */}
      {errors && errors.length > 0 && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            <ul className="list-disc list-inside">
              {errors.map((error, i) => (
                <li key={i}>{error}</li>
              ))}
            </ul>
          </AlertDescription>
        </Alert>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left column: Selection */}
        <div className="space-y-4">
          {/* Dataset selector */}
          <div className="space-y-2">
            <Label>Dataset *</Label>
            {isLoadingDatasets ? (
              <Skeleton className="h-10 w-full" />
            ) : fetchError ? (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>{fetchError}</AlertDescription>
              </Alert>
            ) : (
              <Select
                value={data.datasetId}
                onValueChange={(value) => onChange({ datasetId: value, versionId: undefined })}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select a dataset" />
                </SelectTrigger>
                <SelectContent>
                  {datasets.map((dataset) => (
                    <SelectItem key={dataset.id} value={dataset.id}>
                      <div className="flex items-center justify-between gap-4">
                        <span>{dataset.name}</span>
                        <span className="text-xs text-muted-foreground">
                          {dataset.annotatedImageCount} images
                        </span>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}
          </div>

          {/* Version selector (optional) */}
          {versions.length > 0 && (
            <div className="space-y-2">
              <Label>Version (Optional)</Label>
              <Select
                value={data.versionId || "latest"}
                onValueChange={(value) =>
                  onChange({ versionId: value === "latest" ? undefined : value })
                }
              >
                <SelectTrigger>
                  <SelectValue placeholder="Latest" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="latest">Latest (All images)</SelectItem>
                  {versions.map((version) => (
                    <SelectItem key={version.id} value={version.id}>
                      v{version.versionNumber}
                      {version.name && ` - ${version.name}`}
                      <span className="text-xs text-muted-foreground ml-2">
                        ({version.imageCount} images)
                      </span>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}

          {/* Train/Val/Test Split */}
          <div className="space-y-4 pt-4 border-t">
            <div className="space-y-3">
              <div className="flex justify-between">
                <Label>Training Split</Label>
                <span className="text-sm font-medium">{Math.round(data.trainSplit * 100)}%</span>
              </div>
              <Slider
                value={[data.trainSplit * 100]}
                onValueChange={handleTrainSplitChange}
                min={50}
                max={95}
                step={5}
              />
            </div>

            <div className="space-y-3">
              <div className="flex justify-between">
                <Label>Validation Split</Label>
                <span className="text-sm font-medium">{Math.round(data.valSplit * 100)}%</span>
              </div>
              <Slider
                value={[data.valSplit * 100]}
                onValueChange={handleValSplitChange}
                min={5}
                max={40}
                step={5}
              />
            </div>

            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Test Split</span>
              <span className="font-medium">{Math.round(testSplit * 100)}%</span>
            </div>

            {/* Split visualization */}
            <div className="h-4 rounded-full overflow-hidden flex">
              <div
                className="bg-primary h-full"
                style={{ width: `${data.trainSplit * 100}%` }}
                title="Training"
              />
              <div
                className="bg-blue-400 h-full"
                style={{ width: `${data.valSplit * 100}%` }}
                title="Validation"
              />
              <div
                className="bg-muted h-full"
                style={{ width: `${testSplit * 100}%` }}
                title="Test"
              />
            </div>
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>Train</span>
              <span>Val</span>
              <span>Test</span>
            </div>
          </div>

          {/* Random seed */}
          <div className="space-y-2">
            <Label>Random Seed</Label>
            <Input
              type="number"
              value={data.seed}
              onChange={(e) => onChange({ seed: parseInt(e.target.value) || 42 })}
              className="w-32"
            />
            <p className="text-xs text-muted-foreground">
              Same seed ensures reproducible splits
            </p>
          </div>
        </div>

        {/* Right column: Stats */}
        <div>
          {isLoadingInfo ? (
            <div className="space-y-4">
              <Skeleton className="h-8 w-1/2" />
              <Skeleton className="h-32 w-full" />
              <Skeleton className="h-20 w-full" />
            </div>
          ) : datasetInfo ? (
            <DatasetStatsCard datasetInfo={datasetInfo} />
          ) : (
            <div className="flex flex-col items-center justify-center h-full text-center p-8 border-2 border-dashed rounded-lg">
              <Database className="h-12 w-12 text-muted-foreground mb-4" />
              <p className="text-muted-foreground">
                Select a dataset to view statistics
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
