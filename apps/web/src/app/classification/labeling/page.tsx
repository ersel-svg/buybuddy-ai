"use client";

import { useState, useEffect, useCallback, Suspense } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { apiClient } from "@/lib/api-client";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Loader2,
  FolderOpen,
  ChevronLeft,
  ChevronRight,
  SkipForward,
  CheckCircle,
  AlertTriangle,
  Keyboard,
  X,
  Sparkles,
  Wand2,
} from "lucide-react";
import Image from "next/image";

export default function CLSLabelingPage() {
  return (
    <Suspense fallback={
      <div className="flex items-center justify-center min-h-[60vh]">
        <Loader2 className="h-8 w-8 animate-spin" />
      </div>
    }>
      <CLSLabelingPageContent />
    </Suspense>
  );
}

function CLSLabelingPageContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const queryClient = useQueryClient();

  const datasetId = searchParams.get("dataset");
  const imageIdParam = searchParams.get("image");

  // State
  const [currentImageId, setCurrentImageId] = useState<string | null>(imageIdParam);
  const [selectedClassId, setSelectedClassId] = useState<string | null>(null);
  const [queueMode, setQueueMode] = useState<"unlabeled" | "all" | "review">("unlabeled");
  const [aiSuggestions, setAiSuggestions] = useState<Array<{ class_id: string; class_name: string; confidence: number }>>([]);

  // Fetch datasets for selector
  const { data: datasets } = useQuery({
    queryKey: ["cls-datasets"],
    queryFn: () => apiClient.getCLSDatasets(),
    staleTime: 60000,
  });

  // Fetch classes for the selected dataset
  const { data: classes } = useQuery({
    queryKey: ["cls-dataset-classes", datasetId],
    queryFn: () => apiClient.getCLSDatasetClasses(datasetId!),
    enabled: !!datasetId,
    staleTime: 30000,
  });

  // Fetch labeling queue
  const { data: queue, isLoading: queueLoading } = useQuery({
    queryKey: ["cls-labeling-queue", datasetId, queueMode],
    queryFn: () => apiClient.getCLSLabelingQueue(datasetId!, { mode: queueMode, limit: 100 }),
    enabled: !!datasetId,
    staleTime: 10000,
  });

  // Fetch progress
  const { data: progress } = useQuery({
    queryKey: ["cls-labeling-progress", datasetId],
    queryFn: () => apiClient.getCLSLabelingProgress(datasetId!),
    enabled: !!datasetId,
    staleTime: 10000,
  });

  // Fetch current image details
  const { data: imageData, isLoading: imageLoading } = useQuery({
    queryKey: ["cls-labeling-image", datasetId, currentImageId],
    queryFn: () => apiClient.getCLSLabelingImage(datasetId!, currentImageId!),
    enabled: !!datasetId && !!currentImageId,
  });

  // Set initial image when queue loads
  useEffect(() => {
    if (!currentImageId && queue?.image_ids && queue.image_ids.length > 0) {
      setCurrentImageId(queue.image_ids[0]);
    }
  }, [queue, currentImageId]);

  // Set selected class from current labels
  useEffect(() => {
    if (imageData?.current_labels && imageData.current_labels.length > 0) {
      setSelectedClassId(imageData.current_labels[0].class_id);
    } else {
      setSelectedClassId(null);
    }
  }, [imageData]);

  // Submit label mutation
  const submitMutation = useMutation({
    mutationFn: async ({ action, classId }: { action: "label" | "skip" | "review"; classId?: string }) => {
      return apiClient.submitCLSLabeling(datasetId!, currentImageId!, {
        action,
        class_id: classId,
      });
    },
    onSuccess: (result, { action }) => {
      if (action === "label") {
        toast.success("Label saved");
      } else if (action === "skip") {
        toast.info("Image skipped");
      } else {
        toast.info("Marked for review");
      }

      queryClient.invalidateQueries({ queryKey: ["cls-labeling-progress", datasetId] });
      queryClient.invalidateQueries({ queryKey: ["cls-labeling-queue", datasetId] });

      // Go to next image
      if (result.next_image_id) {
        setCurrentImageId(result.next_image_id);
        router.push(`/classification/labeling?dataset=${datasetId}&image=${result.next_image_id}`);
      } else {
        // No more images
        toast.success("All images labeled!");
        setCurrentImageId(null);
      }
    },
    onError: (error) => {
      toast.error(`Failed to save: ${error.message}`);
    },
  });

  // AI suggestion mutation
  const aiSuggestMutation = useMutation({
    mutationFn: async () => {
      return apiClient.predictCLSAI({
        image_id: currentImageId!,
        dataset_id: datasetId!,
        model: "clip",
        top_k: 3,
        threshold: 0.1,
      });
    },
    onSuccess: (result) => {
      setAiSuggestions(result.predictions);
      if (result.predictions.length > 0) {
        // Auto-select top suggestion
        setSelectedClassId(result.predictions[0].class_id);
        toast.success(`AI suggests: ${result.predictions[0].class_name} (${(result.predictions[0].confidence * 100).toFixed(0)}%)`);
      } else {
        toast.info("No confident AI suggestions");
      }
    },
    onError: (error) => {
      toast.error(`AI suggestion failed: ${error.message}`);
    },
  });

  // Clear AI suggestions when image changes
  useEffect(() => {
    setAiSuggestions([]);
  }, [currentImageId]);

  // Keyboard shortcuts
  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if (!datasetId || !currentImageId) return;

    // Number keys 1-9 for class selection
    if (e.key >= "1" && e.key <= "9" && classes) {
      const index = parseInt(e.key) - 1;
      if (index < classes.length) {
        const classToSelect = classes[index];
        setSelectedClassId(classToSelect.id);
        // Auto-submit on number key press
        submitMutation.mutate({ action: "label", classId: classToSelect.id });
      }
      return;
    }

    switch (e.key) {
      case "Enter":
        if (selectedClassId) {
          submitMutation.mutate({ action: "label", classId: selectedClassId });
        }
        break;
      case "s":
        submitMutation.mutate({ action: "skip" });
        break;
      case "r":
        submitMutation.mutate({ action: "review" });
        break;
      case "ArrowLeft":
        if (imageData?.prev_image_id) {
          setCurrentImageId(imageData.prev_image_id);
          router.push(`/classification/labeling?dataset=${datasetId}&image=${imageData.prev_image_id}`);
        }
        break;
      case "ArrowRight":
        if (imageData?.next_image_id) {
          setCurrentImageId(imageData.next_image_id);
          router.push(`/classification/labeling?dataset=${datasetId}&image=${imageData.next_image_id}`);
        }
        break;
      case "Escape":
        setSelectedClassId(null);
        break;
      case "a":
        // AI suggest
        if (!aiSuggestMutation.isPending) {
          aiSuggestMutation.mutate();
        }
        break;
    }
  }, [datasetId, currentImageId, selectedClassId, classes, imageData, submitMutation, aiSuggestMutation, router]);

  useEffect(() => {
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleKeyDown]);

  // No dataset selected
  if (!datasetId) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-2xl font-bold">Labeling</h1>
          <p className="text-muted-foreground">
            Select a dataset to start labeling images
          </p>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Select Dataset</CardTitle>
            <CardDescription>Choose a dataset to label</CardDescription>
          </CardHeader>
          <CardContent>
            {datasets?.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                <FolderOpen className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>No datasets found</p>
                <p className="text-sm">Create a dataset first</p>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {datasets?.map((dataset) => {
                  const labelProgress = dataset.image_count > 0
                    ? Math.round((dataset.labeled_image_count / dataset.image_count) * 100)
                    : 0;

                  return (
                    <Card
                      key={dataset.id}
                      className="cursor-pointer hover:border-primary transition-colors"
                      onClick={() => router.push(`/classification/labeling?dataset=${dataset.id}`)}
                    >
                      <CardHeader className="pb-2">
                        <CardTitle className="text-base">{dataset.name}</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <Progress value={labelProgress} className="h-2 mb-2" />
                        <div className="flex justify-between text-sm text-muted-foreground">
                          <span>{dataset.labeled_image_count}/{dataset.image_count} labeled</span>
                          <span>{labelProgress}%</span>
                        </div>
                      </CardContent>
                    </Card>
                  );
                })}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    );
  }

  // Loading state
  if (queueLoading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <Loader2 className="h-8 w-8 animate-spin" />
      </div>
    );
  }

  // No images in queue
  if (!queue?.image_ids?.length && !currentImageId) {
    return (
      <div className="space-y-6">
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-2xl font-bold">Labeling</h1>
            <p className="text-muted-foreground">
              {datasets?.find(d => d.id === datasetId)?.name}
            </p>
          </div>
          <Select value={queueMode} onValueChange={(v) => setQueueMode(v as typeof queueMode)}>
            <SelectTrigger className="w-40">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="unlabeled">Unlabeled</SelectItem>
              <SelectItem value="all">All Images</SelectItem>
              <SelectItem value="review">Review</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <Card>
          <CardContent className="py-16 text-center">
            <CheckCircle className="h-16 w-16 mx-auto text-green-500 mb-4" />
            <h3 className="text-xl font-medium">All Done!</h3>
            <p className="text-muted-foreground mt-2">
              {queueMode === "unlabeled"
                ? "All images in this dataset have been labeled"
                : "No images match the current filter"}
            </p>
            <Button
              className="mt-4"
              variant="outline"
              onClick={() => router.push("/classification/datasets")}
            >
              Back to Datasets
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <TooltipProvider>
      <div className="h-[calc(100vh-8rem)] flex flex-col">
        {/* Header */}
        <div className="flex justify-between items-center mb-4">
          <div className="flex items-center gap-4">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => router.push("/classification/datasets")}
            >
              <ChevronLeft className="h-4 w-4 mr-1" />
              Back
            </Button>
            <div>
              <h1 className="text-lg font-bold">
                {datasets?.find(d => d.id === datasetId)?.name}
              </h1>
              <p className="text-sm text-muted-foreground">
                {imageData?.position ?? 0} of {imageData?.total ?? 0}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            {/* Progress */}
            {progress && (
              <div className="flex items-center gap-2 text-sm">
                <span className="text-muted-foreground">Progress:</span>
                <Progress value={progress.progress_pct} className="w-32 h-2" />
                <span className="font-medium">{progress.progress_pct.toFixed(0)}%</span>
              </div>
            )}

            {/* Queue Mode */}
            <Select value={queueMode} onValueChange={(v) => setQueueMode(v as typeof queueMode)}>
              <SelectTrigger className="w-32">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="unlabeled">Unlabeled</SelectItem>
                <SelectItem value="all">All</SelectItem>
                <SelectItem value="review">Review</SelectItem>
              </SelectContent>
            </Select>

            {/* Keyboard shortcuts */}
            <Tooltip>
              <TooltipTrigger asChild>
                <Button variant="ghost" size="icon">
                  <Keyboard className="h-4 w-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent side="bottom" className="max-w-xs">
                <div className="space-y-1 text-xs">
                  <p><kbd className="px-1 bg-muted rounded">1-9</kbd> Select & save class</p>
                  <p><kbd className="px-1 bg-muted rounded">Enter</kbd> Save current label</p>
                  <p><kbd className="px-1 bg-muted rounded">S</kbd> Skip image</p>
                  <p><kbd className="px-1 bg-muted rounded">R</kbd> Mark for review</p>
                  <p><kbd className="px-1 bg-muted rounded">A</kbd> AI suggest</p>
                  <p><kbd className="px-1 bg-muted rounded">←/→</kbd> Navigate</p>
                  <p><kbd className="px-1 bg-muted rounded">Esc</kbd> Clear selection</p>
                </div>
              </TooltipContent>
            </Tooltip>
          </div>
        </div>

        {/* Main Content */}
        <div className="flex-1 grid grid-cols-[1fr_300px] gap-4 min-h-0">
          {/* Image Panel */}
          <Card className="flex flex-col">
            <CardContent className="flex-1 p-4 flex items-center justify-center bg-muted/50">
              {imageLoading ? (
                <Loader2 className="h-12 w-12 animate-spin text-muted-foreground" />
              ) : imageData?.image ? (
                <div className="relative w-full h-full flex items-center justify-center">
                  <Image
                    src={imageData.image.image_url}
                    alt={imageData.image.filename}
                    fill
                    className="object-contain"
                    sizes="(max-width: 1200px) 70vw, 800px"
                    priority
                  />
                </div>
              ) : (
                <div className="text-muted-foreground">No image</div>
              )}
            </CardContent>

            {/* Navigation */}
            <div className="p-4 border-t flex items-center justify-between">
              <Button
                variant="outline"
                size="sm"
                disabled={!imageData?.prev_image_id}
                onClick={() => {
                  if (imageData?.prev_image_id) {
                    setCurrentImageId(imageData.prev_image_id);
                    router.push(`/classification/labeling?dataset=${datasetId}&image=${imageData.prev_image_id}`);
                  }
                }}
              >
                <ChevronLeft className="h-4 w-4 mr-1" />
                Previous
              </Button>

              <span className="text-sm text-muted-foreground">
                {imageData?.image?.filename}
              </span>

              <Button
                variant="outline"
                size="sm"
                disabled={!imageData?.next_image_id}
                onClick={() => {
                  if (imageData?.next_image_id) {
                    setCurrentImageId(imageData.next_image_id);
                    router.push(`/classification/labeling?dataset=${datasetId}&image=${imageData.next_image_id}`);
                  }
                }}
              >
                Next
                <ChevronRight className="h-4 w-4 ml-1" />
              </Button>
            </div>
          </Card>

          {/* Classes Panel */}
          <div className="flex flex-col gap-4">
            {/* Current Label */}
            {imageData?.current_labels && imageData.current_labels.length > 0 && (
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm">Current Label</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center gap-2">
                    <div
                      className="w-4 h-4 rounded-full"
                      style={{ backgroundColor: imageData.current_labels[0]?.class_color }}
                    />
                    <span className="font-medium">{imageData.current_labels[0]?.class_name}</span>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-6 w-6 ml-auto"
                      onClick={() => setSelectedClassId(null)}
                    >
                      <X className="h-3 w-3" />
                    </Button>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Class List */}
            <Card className="flex-1 flex flex-col min-h-0">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">Classes</CardTitle>
                <CardDescription>Click or press 1-9 to select</CardDescription>
              </CardHeader>
              <CardContent className="flex-1 overflow-y-auto">
                <div className="space-y-1">
                  {classes?.map((cls, index) => (
                    <button
                      key={cls.id}
                      className={`w-full flex items-center gap-3 p-2 rounded-lg text-left transition-colors ${
                        selectedClassId === cls.id
                          ? "bg-primary text-primary-foreground"
                          : "hover:bg-muted"
                      }`}
                      onClick={() => {
                        setSelectedClassId(cls.id);
                        submitMutation.mutate({ action: "label", classId: cls.id });
                      }}
                    >
                      <div
                        className="w-4 h-4 rounded-full flex-shrink-0"
                        style={{ backgroundColor: cls.color }}
                      />
                      <span className="flex-1 truncate">{cls.display_name || cls.name}</span>
                      {index < 9 && (
                        <kbd className={`px-1.5 py-0.5 text-xs rounded ${
                          selectedClassId === cls.id ? "bg-primary-foreground/20" : "bg-muted"
                        }`}>
                          {index + 1}
                        </kbd>
                      )}
                      <span className={`text-xs ${selectedClassId === cls.id ? "text-primary-foreground/70" : "text-muted-foreground"}`}>
                        {cls.image_count}
                      </span>
                    </button>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Actions */}
            <Card>
              <CardContent className="p-4">
                <div className="space-y-2">
                  <Button
                    className="w-full"
                    disabled={!selectedClassId || submitMutation.isPending}
                    onClick={() => {
                      if (selectedClassId) {
                        submitMutation.mutate({ action: "label", classId: selectedClassId });
                      }
                    }}
                  >
                    {submitMutation.isPending ? (
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    ) : (
                      <CheckCircle className="h-4 w-4 mr-2" />
                    )}
                    Save Label
                    <kbd className="ml-2 px-1.5 py-0.5 text-xs bg-primary-foreground/20 rounded">Enter</kbd>
                  </Button>

                  <Button
                    variant="secondary"
                    className="w-full"
                    disabled={aiSuggestMutation.isPending}
                    onClick={() => aiSuggestMutation.mutate()}
                  >
                    {aiSuggestMutation.isPending ? (
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    ) : (
                      <Sparkles className="h-4 w-4 mr-2" />
                    )}
                    AI Suggest
                    <kbd className="ml-2 px-1.5 py-0.5 text-xs bg-secondary-foreground/20 rounded">A</kbd>
                  </Button>

                  {/* AI Suggestions */}
                  {aiSuggestions.length > 0 && (
                    <div className="p-2 rounded-lg bg-purple-50 dark:bg-purple-950/30 border border-purple-200 dark:border-purple-900">
                      <p className="text-xs font-medium text-purple-600 dark:text-purple-400 mb-2 flex items-center gap-1">
                        <Wand2 className="h-3 w-3" />
                        AI Suggestions
                      </p>
                      <div className="space-y-1">
                        {aiSuggestions.map((suggestion) => (
                          <button
                            key={suggestion.class_id}
                            className={`w-full flex items-center justify-between p-1.5 rounded text-sm transition-colors ${
                              selectedClassId === suggestion.class_id
                                ? "bg-purple-200 dark:bg-purple-900/50"
                                : "hover:bg-purple-100 dark:hover:bg-purple-900/30"
                            }`}
                            onClick={() => {
                              setSelectedClassId(suggestion.class_id);
                              submitMutation.mutate({ action: "label", classId: suggestion.class_id });
                            }}
                          >
                            <span>{suggestion.class_name}</span>
                            <Badge variant="outline" className="text-xs">
                              {(suggestion.confidence * 100).toFixed(0)}%
                            </Badge>
                          </button>
                        ))}
                      </div>
                    </div>
                  )}

                  <div className="grid grid-cols-2 gap-2">
                    <Button
                      variant="outline"
                      disabled={submitMutation.isPending}
                      onClick={() => submitMutation.mutate({ action: "skip" })}
                    >
                      <SkipForward className="h-4 w-4 mr-2" />
                      Skip
                      <kbd className="ml-1 px-1 py-0.5 text-xs bg-muted rounded">S</kbd>
                    </Button>

                    <Button
                      variant="outline"
                      disabled={submitMutation.isPending}
                      onClick={() => submitMutation.mutate({ action: "review" })}
                    >
                      <AlertTriangle className="h-4 w-4 mr-2" />
                      Review
                      <kbd className="ml-1 px-1 py-0.5 text-xs bg-muted rounded">R</kbd>
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Stats */}
            {progress && (
              <Card>
                <CardContent className="p-4">
                  <div className="grid grid-cols-2 gap-2 text-center text-sm">
                    <div className="p-2 rounded bg-muted">
                      <p className="text-lg font-bold text-green-600">{progress.completed}</p>
                      <p className="text-xs text-muted-foreground">Completed</p>
                    </div>
                    <div className="p-2 rounded bg-muted">
                      <p className="text-lg font-bold text-orange-600">{progress.pending}</p>
                      <p className="text-xs text-muted-foreground">Pending</p>
                    </div>
                    <div className="p-2 rounded bg-muted">
                      <p className="text-lg font-bold text-yellow-600">{progress.review}</p>
                      <p className="text-xs text-muted-foreground">Review</p>
                    </div>
                    <div className="p-2 rounded bg-muted">
                      <p className="text-lg font-bold text-gray-600">{progress.skipped}</p>
                      <p className="text-xs text-muted-foreground">Skipped</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </div>
    </TooltipProvider>
  );
}
