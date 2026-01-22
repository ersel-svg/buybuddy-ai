"use client";

import { useState, useEffect, useCallback, Suspense, useMemo, useRef } from "react";
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
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandSeparator,
} from "@/components/ui/command";
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
  Keyboard,
  Plus,
  Star,
  Tag,
} from "lucide-react";
import Image from "next/image";

// Color palette for auto-assigning colors to new classes
const CLASS_COLORS = [
  "#3B82F6", "#EF4444", "#10B981", "#F59E0B", "#8B5CF6",
  "#EC4899", "#06B6D4", "#84CC16", "#F97316", "#6366F1",
  "#14B8A6", "#A855F7", "#F43F5E", "#0EA5E9", "#22C55E",
];

export default function CLSLabelingPage() {
  return (
    <Suspense fallback={
      <div className="flex items-center justify-center h-screen">
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
  const inputRef = useRef<HTMLInputElement>(null);

  const datasetId = searchParams.get("dataset");
  const imageIdParam = searchParams.get("image");

  // State
  const [currentImageId, setCurrentImageId] = useState<string | null>(imageIdParam);
  const [searchValue, setSearchValue] = useState("");
  const [selectedClassId, setSelectedClassId] = useState<string | null>(null);
  const [queueMode, setQueueMode] = useState<"unlabeled" | "all" | "review">("all");
  const [recentClassIds, setRecentClassIds] = useState<string[]>([]);

  // Load recent classes from localStorage
  useEffect(() => {
    if (datasetId) {
      const stored = localStorage.getItem(`cls-recent-${datasetId}`);
      if (stored) {
        try {
          setRecentClassIds(JSON.parse(stored));
        } catch {}
      }
    }
  }, [datasetId]);

  // Save recent class to localStorage
  const addToRecent = useCallback((classId: string) => {
    if (!datasetId) return;
    setRecentClassIds(prev => {
      const updated = [classId, ...prev.filter(id => id !== classId)].slice(0, 9);
      localStorage.setItem(`cls-recent-${datasetId}`, JSON.stringify(updated));
      return updated;
    });
  }, [datasetId]);

  // Fetch datasets for selector
  const { data: datasets } = useQuery({
    queryKey: ["cls-datasets"],
    queryFn: () => apiClient.getCLSDatasets(),
    staleTime: 60000,
  });

  // Fetch all classes
  const { data: allClasses, refetch: refetchClasses } = useQuery({
    queryKey: ["cls-classes"],
    queryFn: () => apiClient.getCLSClasses(),
    staleTime: 30000,
  });

  // Fetch dataset-specific class counts
  const { data: datasetClasses } = useQuery({
    queryKey: ["cls-dataset-classes", datasetId],
    queryFn: () => apiClient.getCLSDatasetClasses(datasetId!),
    enabled: !!datasetId,
    staleTime: 30000,
  });

  // Merge class data - use allClasses but with dataset-specific counts
  const classes = useMemo(() => {
    if (!allClasses) return [];
    const countMap = new Map(datasetClasses?.map(c => [c.id, c.image_count]) || []);
    return allClasses.map(c => ({
      ...c,
      image_count: countMap.get(c.id) || 0,
    }));
  }, [allClasses, datasetClasses]);

  // Recent classes with full data
  const recentClasses = useMemo(() => {
    return recentClassIds
      .map(id => classes.find(c => c.id === id))
      .filter(Boolean) as typeof classes;
  }, [recentClassIds, classes]);

  // Filtered classes based on search
  const filteredClasses = useMemo(() => {
    if (!searchValue.trim()) return classes;
    const search = searchValue.toLowerCase();
    return classes.filter(c => 
      c.name.toLowerCase().includes(search) ||
      c.display_name?.toLowerCase().includes(search)
    );
  }, [classes, searchValue]);

  // Check if search value matches any existing class
  const exactMatch = useMemo(() => {
    if (!searchValue.trim()) return null;
    const search = searchValue.toLowerCase().trim();
    return classes.find(c => 
      c.name.toLowerCase() === search ||
      c.display_name?.toLowerCase() === search
    );
  }, [classes, searchValue]);

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

  // Create class mutation
  const createClassMutation = useMutation({
    mutationFn: (name: string) => apiClient.createCLSClass({
      name: name.toLowerCase().replace(/\s+/g, '_'),
      display_name: name,
      color: CLASS_COLORS[classes.length % CLASS_COLORS.length],
    }),
    onSuccess: (newClass) => {
      toast.success(`Class "${newClass.name}" created`);
      refetchClasses();
      // Auto-select and submit with new class
      handleSelectClass(newClass.id);
    },
    onError: (error: Error) => {
      toast.error(`Failed to create class: ${error.message}`);
    },
  });

  // Submit label mutation
  const submitMutation = useMutation({
    mutationFn: async ({ action, classId }: { action: "label" | "skip" | "review"; classId?: string }) => {
      return apiClient.submitCLSLabeling(datasetId!, currentImageId!, {
        action,
        class_id: classId,
      });
    },
    onSuccess: (result, { action, classId }) => {
      if (action === "label" && classId) {
        addToRecent(classId);
        toast.success("Label saved");
      } else if (action === "skip") {
        toast.info("Image skipped");
      } else {
        toast.info("Marked for review");
      }

      queryClient.invalidateQueries({ queryKey: ["cls-labeling-progress", datasetId] });
      queryClient.invalidateQueries({ queryKey: ["cls-labeling-queue", datasetId] });
      queryClient.invalidateQueries({ queryKey: ["cls-dataset-classes", datasetId] });

      // Reset search
      setSearchValue("");
      setSelectedClassId(null);

      // Go to next image
      if (result.next_image_id) {
        setCurrentImageId(result.next_image_id);
        router.push(`/classification/labeling?dataset=${datasetId}&image=${result.next_image_id}`, { scroll: false });
      } else {
        toast.success("All images labeled!");
        setCurrentImageId(null);
      }

      // Focus input for next label
      setTimeout(() => inputRef.current?.focus(), 100);
    },
    onError: (error: Error) => {
      toast.error(`Failed to save: ${error.message}`);
    },
  });

  // Handle class selection
  const handleSelectClass = useCallback((classId: string) => {
    setSelectedClassId(classId);
    submitMutation.mutate({ action: "label", classId });
  }, [submitMutation]);

  // Handle create new class
  const handleCreateClass = useCallback(() => {
    if (!searchValue.trim()) return;
    createClassMutation.mutate(searchValue.trim());
  }, [searchValue, createClassMutation]);

  // Keyboard shortcuts
  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if (!datasetId || !currentImageId) return;

    // Don't handle if typing in input (except special keys)
    const isTyping = document.activeElement?.tagName === "INPUT";

    // Number keys 1-9 for recent class selection (works even when typing)
    if (e.key >= "1" && e.key <= "9" && !e.ctrlKey && !e.metaKey) {
      const index = parseInt(e.key) - 1;
      if (index < recentClasses.length) {
        e.preventDefault();
        handleSelectClass(recentClasses[index].id);
      }
      return;
    }

    if (isTyping) return;

    switch (e.key) {
      case "s":
        e.preventDefault();
        submitMutation.mutate({ action: "skip" });
        break;
      case "ArrowLeft":
        e.preventDefault();
        if (imageData?.prev_image_id) {
          setCurrentImageId(imageData.prev_image_id);
          router.push(`/classification/labeling?dataset=${datasetId}&image=${imageData.prev_image_id}`, { scroll: false });
        }
        break;
      case "ArrowRight":
        e.preventDefault();
        if (imageData?.next_image_id) {
          setCurrentImageId(imageData.next_image_id);
          router.push(`/classification/labeling?dataset=${datasetId}&image=${imageData.next_image_id}`, { scroll: false });
        }
        break;
      case "/":
        e.preventDefault();
        inputRef.current?.focus();
        break;
    }
  }, [datasetId, currentImageId, recentClasses, imageData, submitMutation, handleSelectClass, router]);

  useEffect(() => {
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleKeyDown]);

  // No dataset selected - show dataset selector
  if (!datasetId) {
    return (
      <div className="p-6 space-y-6">
        <div>
          <h1 className="text-2xl font-bold">Labeling</h1>
          <p className="text-muted-foreground">Select a dataset to start labeling images</p>
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
      <div className="flex items-center justify-center h-screen">
        <Loader2 className="h-8 w-8 animate-spin" />
      </div>
    );
  }

  // No images in queue
  if (!queue?.image_ids?.length && !currentImageId) {
    return (
      <div className="p-6 space-y-6">
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
      <div className="h-screen flex flex-col overflow-hidden">
        {/* Header */}
        <div className="flex-none flex justify-between items-center px-4 py-3 border-b bg-background">
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
                  <p><kbd className="px-1 bg-muted rounded">1-9</kbd> Quick select recent class</p>
                  <p><kbd className="px-1 bg-muted rounded">/</kbd> Focus search</p>
                  <p><kbd className="px-1 bg-muted rounded">S</kbd> Skip image</p>
                  <p><kbd className="px-1 bg-muted rounded">←/→</kbd> Navigate</p>
                </div>
              </TooltipContent>
            </Tooltip>
          </div>
        </div>

        {/* Main Content */}
        <div className="flex-1 flex min-h-0">
          {/* Image Panel */}
          <div className="flex-1 flex flex-col min-w-0">
            <div className="flex-1 p-4 flex items-center justify-center bg-muted/30 min-h-0">
              {imageLoading ? (
                <Loader2 className="h-12 w-12 animate-spin text-muted-foreground" />
              ) : imageData?.image ? (
                <div className="relative w-full h-full flex items-center justify-center">
                  <Image
                    src={imageData.image.image_url}
                    alt={imageData.image.filename}
                    fill
                    className="object-contain"
                    sizes="(max-width: 1200px) 60vw, 800px"
                    priority
                  />
                </div>
              ) : (
                <div className="text-muted-foreground">No image</div>
              )}
            </div>

            {/* Navigation Bar */}
            <div className="flex-none p-3 border-t flex items-center justify-between bg-background">
              <Button
                variant="outline"
                size="sm"
                disabled={!imageData?.prev_image_id}
                onClick={() => {
                  if (imageData?.prev_image_id) {
                    setCurrentImageId(imageData.prev_image_id);
                    router.push(`/classification/labeling?dataset=${datasetId}&image=${imageData.prev_image_id}`, { scroll: false });
                  }
                }}
              >
                <ChevronLeft className="h-4 w-4 mr-1" />
                Previous
              </Button>

              <span className="text-sm text-muted-foreground truncate max-w-[200px]">
                {imageData?.image?.filename}
              </span>

              <Button
                variant="outline"
                size="sm"
                disabled={!imageData?.next_image_id}
                onClick={() => {
                  if (imageData?.next_image_id) {
                    setCurrentImageId(imageData.next_image_id);
                    router.push(`/classification/labeling?dataset=${datasetId}&image=${imageData.next_image_id}`, { scroll: false });
                  }
                }}
              >
                Next
                <ChevronRight className="h-4 w-4 ml-1" />
              </Button>
            </div>
          </div>

          {/* Right Panel - Class Selector */}
          <div className="w-80 flex-none border-l flex flex-col bg-background">
            {/* Current Label */}
            {imageData?.current_labels && imageData.current_labels.length > 0 && (
              <div className="flex-none p-3 border-b bg-green-50 dark:bg-green-950/30">
                <p className="text-xs text-muted-foreground mb-1">Current Label</p>
                <div className="flex items-center gap-2">
                  <div
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: imageData.current_labels[0]?.class_color }}
                  />
                  <span className="font-medium">{imageData.current_labels[0]?.class_name}</span>
                </div>
              </div>
            )}

            {/* Search/Create Input */}
            <div className="flex-none p-3 border-b">
              <Command className="rounded-lg border" shouldFilter={false}>
                <CommandInput
                  ref={inputRef}
                  placeholder="Search or create class..."
                  value={searchValue}
                  onValueChange={setSearchValue}
                />
                <CommandList className="max-h-none">
                  {/* Recent Classes */}
                  {recentClasses.length > 0 && !searchValue && (
                    <CommandGroup heading="Recent (1-9)">
                      {recentClasses.map((cls, index) => (
                        <CommandItem
                          key={cls.id}
                          value={cls.id}
                          onSelect={() => handleSelectClass(cls.id)}
                          className="cursor-pointer"
                        >
                          <div
                            className="w-3 h-3 rounded-full mr-2"
                            style={{ backgroundColor: cls.color }}
                          />
                          <span className="flex-1">{cls.display_name || cls.name}</span>
                          <kbd className="px-1.5 py-0.5 text-xs bg-muted rounded">
                            {index + 1}
                          </kbd>
                          <Badge variant="secondary" className="ml-2 text-xs">
                            {cls.image_count}
                          </Badge>
                        </CommandItem>
                      ))}
                    </CommandGroup>
                  )}

                  {recentClasses.length > 0 && !searchValue && <CommandSeparator />}

                  {/* Create New Class Option */}
                  {searchValue && !exactMatch && (
                    <CommandGroup heading="Create New">
                      <CommandItem
                        onSelect={handleCreateClass}
                        className="cursor-pointer text-primary"
                      >
                        <Plus className="h-4 w-4 mr-2" />
                        Create &quot;{searchValue}&quot;
                        {createClassMutation.isPending && (
                          <Loader2 className="h-3 w-3 ml-2 animate-spin" />
                        )}
                      </CommandItem>
                    </CommandGroup>
                  )}

                  {searchValue && !exactMatch && filteredClasses.length > 0 && <CommandSeparator />}

                  {/* All/Filtered Classes */}
                  <CommandGroup heading={searchValue ? "Matching Classes" : "All Classes"}>
                    {filteredClasses.length === 0 ? (
                      <CommandEmpty>No classes found</CommandEmpty>
                    ) : (
                      filteredClasses.map((cls) => (
                        <CommandItem
                          key={cls.id}
                          value={cls.id}
                          onSelect={() => handleSelectClass(cls.id)}
                          className="cursor-pointer"
                        >
                          <div
                            className="w-3 h-3 rounded-full mr-2"
                            style={{ backgroundColor: cls.color }}
                          />
                          <span className="flex-1">{cls.display_name || cls.name}</span>
                          <Badge variant="secondary" className="text-xs">
                            {cls.image_count}
                          </Badge>
                        </CommandItem>
                      ))
                    )}
                  </CommandGroup>
                </CommandList>
              </Command>
            </div>

            {/* Stats */}
            {progress && (
              <div className="flex-none p-3 border-b">
                <div className="grid grid-cols-2 gap-2 text-center text-xs">
                  <div className="p-2 rounded bg-muted">
                    <p className="text-base font-bold text-green-600">{progress.completed + progress.labeled}</p>
                    <p className="text-muted-foreground">Labeled</p>
                  </div>
                  <div className="p-2 rounded bg-muted">
                    <p className="text-base font-bold text-orange-600">{progress.pending}</p>
                    <p className="text-muted-foreground">Pending</p>
                  </div>
                </div>
              </div>
            )}

            {/* Quick Actions */}
            <div className="flex-none p-3 mt-auto border-t">
              <Button
                variant="outline"
                className="w-full"
                disabled={submitMutation.isPending}
                onClick={() => submitMutation.mutate({ action: "skip" })}
              >
                <SkipForward className="h-4 w-4 mr-2" />
                Skip
                <kbd className="ml-auto px-1.5 py-0.5 text-xs bg-muted rounded">S</kbd>
              </Button>
            </div>
          </div>
        </div>
      </div>
    </TooltipProvider>
  );
}
