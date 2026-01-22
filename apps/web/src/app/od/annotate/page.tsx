"use client";

import { Suspense, useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useRouter, useSearchParams } from "next/navigation";
import { apiClient } from "@/lib/api-client";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Loader2,
  PenTool,
  FolderOpen,
  ImageIcon,
  ArrowRight,
} from "lucide-react";
import Link from "next/link";

function ODAnnotateContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const preselectedDataset = searchParams.get("dataset");

  const [selectedDatasetId, setSelectedDatasetId] = useState<string>(preselectedDataset || "");

  // Fetch datasets
  const { data: datasets, isLoading: datasetsLoading } = useQuery({
    queryKey: ["od-datasets"],
    queryFn: () => apiClient.getODDatasets(),
    staleTime: 60000, // 1 minute - datasets don't change often
    gcTime: 5 * 60 * 1000, // 5 minutes
  });

  // Fetch images for selected dataset
  const { data: imagesData, isLoading: imagesLoading } = useQuery({
    queryKey: ["od-dataset-images", selectedDatasetId, "pending"],
    queryFn: () =>
      apiClient.getODDatasetImages(selectedDatasetId, {
        page: 1,
        limit: 1,
        status: "pending",
      }),
    enabled: !!selectedDatasetId,
    staleTime: 10000, // 10 seconds - images can change during annotation
    gcTime: 60000, // 1 minute
  });

  // If dataset is preselected and has images, redirect to first image
  useEffect(() => {
    if (preselectedDataset && imagesData?.images && imagesData.images.length > 0) {
      const firstImage = imagesData.images[0];
      router.push(`/od/annotate/${preselectedDataset}/${firstImage.image_id}`);
    }
  }, [preselectedDataset, imagesData, router]);

  const selectedDataset = datasets?.find((d) => d.id === selectedDatasetId);

  const handleStartAnnotating = () => {
    if (!selectedDatasetId) return;

    if (imagesData?.images && imagesData.images.length > 0) {
      const firstImage = imagesData.images[0];
      router.push(`/od/annotate/${selectedDatasetId}/${firstImage.image_id}`);
    } else {
      // No pending images, go to dataset to add images
      router.push(`/od/datasets/${selectedDatasetId}`);
    }
  };

  // Show loading while checking preselected dataset
  if (preselectedDataset && (imagesLoading || datasetsLoading)) {
    return (
      <div className="flex items-center justify-center py-24">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Annotate</h1>
        <p className="text-muted-foreground">
          Draw bounding boxes and annotate images
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Dataset Selection */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FolderOpen className="h-5 w-5" />
              Select Dataset
            </CardTitle>
            <CardDescription>
              Choose a dataset to start annotating images
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {datasetsLoading ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
              </div>
            ) : datasets?.length === 0 ? (
              <div className="text-center py-8">
                <FolderOpen className="h-12 w-12 mx-auto text-muted-foreground mb-2" />
                <p className="text-muted-foreground">No datasets found</p>
                <Button asChild className="mt-4">
                  <Link href="/od/datasets">
                    Create Dataset
                  </Link>
                </Button>
              </div>
            ) : (
              <>
                <Select value={selectedDatasetId} onValueChange={setSelectedDatasetId}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select a dataset..." />
                  </SelectTrigger>
                  <SelectContent>
                    {datasets?.map((dataset) => (
                      <SelectItem key={dataset.id} value={dataset.id}>
                        <div className="flex items-center justify-between w-full">
                          <span>{dataset.name}</span>
                          <span className="text-muted-foreground ml-2">
                            ({dataset.image_count} images)
                          </span>
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>

                {selectedDataset && (
                  <div className="bg-muted rounded-lg p-4 space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm text-muted-foreground">Images:</span>
                      <span className="font-medium">{selectedDataset.image_count}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-muted-foreground">Annotated:</span>
                      <span className="font-medium">{selectedDataset.annotated_image_count}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-muted-foreground">Annotations:</span>
                      <span className="font-medium">{selectedDataset.annotation_count}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-muted-foreground">Progress:</span>
                      <span className="font-medium">
                        {selectedDataset.image_count > 0
                          ? Math.round((selectedDataset.annotated_image_count / selectedDataset.image_count) * 100)
                          : 0}%
                      </span>
                    </div>
                  </div>
                )}

                <Button
                  className="w-full"
                  onClick={handleStartAnnotating}
                  disabled={!selectedDatasetId || imagesLoading}
                >
                  {imagesLoading ? (
                    <>
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      Loading...
                    </>
                  ) : (
                    <>
                      <PenTool className="h-4 w-4 mr-2" />
                      Start Annotating
                      <ArrowRight className="h-4 w-4 ml-2" />
                    </>
                  )}
                </Button>
              </>
            )}
          </CardContent>
        </Card>

        {/* Quick Tips */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <PenTool className="h-5 w-5" />
              Annotation Guide
            </CardTitle>
            <CardDescription>
              Tips for efficient annotation
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-3">
              <div className="flex gap-3">
                <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0">
                  <span className="text-sm font-bold text-primary">1</span>
                </div>
                <div>
                  <p className="font-medium">Select a class</p>
                  <p className="text-sm text-muted-foreground">
                    Choose the object class from the sidebar before drawing
                  </p>
                </div>
              </div>

              <div className="flex gap-3">
                <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0">
                  <span className="text-sm font-bold text-primary">2</span>
                </div>
                <div>
                  <p className="font-medium">Draw bounding box</p>
                  <p className="text-sm text-muted-foreground">
                    Click and drag to draw a box around the object
                  </p>
                </div>
              </div>

              <div className="flex gap-3">
                <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0">
                  <span className="text-sm font-bold text-primary">3</span>
                </div>
                <div>
                  <p className="font-medium">Adjust if needed</p>
                  <p className="text-sm text-muted-foreground">
                    Click on boxes to select and resize or delete them
                  </p>
                </div>
              </div>

              <div className="flex gap-3">
                <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0">
                  <span className="text-sm font-bold text-primary">4</span>
                </div>
                <div>
                  <p className="font-medium">Save and continue</p>
                  <p className="text-sm text-muted-foreground">
                    Annotations auto-save. Use arrow keys to navigate images.
                  </p>
                </div>
              </div>
            </div>

            <div className="border-t pt-4">
              <p className="text-sm font-medium mb-2">Keyboard Shortcuts</p>
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Next image</span>
                  <kbd className="px-2 py-0.5 bg-muted rounded">→</kbd>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Previous</span>
                  <kbd className="px-2 py-0.5 bg-muted rounded">←</kbd>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Delete box</span>
                  <kbd className="px-2 py-0.5 bg-muted rounded">Del</kbd>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Undo</span>
                  <kbd className="px-2 py-0.5 bg-muted rounded">⌘Z</kbd>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Recent Datasets */}
      {datasets && datasets.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Recent Datasets</CardTitle>
            <CardDescription>
              Quick access to your datasets
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {datasets.slice(0, 6).map((dataset) => (
                <div
                  key={dataset.id}
                  className="border rounded-lg p-4 hover:border-primary cursor-pointer transition-colors"
                  onClick={() => {
                    setSelectedDatasetId(dataset.id);
                  }}
                >
                  <div className="flex items-start justify-between">
                    <div>
                      <p className="font-medium">{dataset.name}</p>
                      <p className="text-sm text-muted-foreground">
                        {dataset.image_count} images &bull; {dataset.annotation_count} annotations
                      </p>
                    </div>
                    <div className="text-right">
                      <span className="text-lg font-bold">
                        {dataset.image_count > 0
                          ? Math.round((dataset.annotated_image_count / dataset.image_count) * 100)
                          : 0}%
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

export default function ODAnnotatePage() {
  return (
    <Suspense
      fallback={
        <div className="flex items-center justify-center py-24">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        </div>
      }
    >
      <ODAnnotateContent />
    </Suspense>
  );
}
