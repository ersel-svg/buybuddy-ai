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
// Command removed - using custom class selector
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
  Sparkles,
  Wand2,
  Zap,
  Undo2,
  ZoomIn,
  ZoomOut,
  Maximize,
  RotateCcw,
  Trash2,
  Settings2,
} from "lucide-react";
import Image from "next/image";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";

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

  // AI State
  const [selectedAIModel, setSelectedAIModel] = useState<string>("clip");
  const [aiPredictions, setAiPredictions] = useState<Array<{ class_id: string; class_name: string; confidence: number }>>([]);
  const [isAIPredicting, setIsAIPredicting] = useState(false);

  // Zoom & Pan State
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const imageContainerRef = useRef<HTMLDivElement>(null);

  // Undo State
  const [undoStack, setUndoStack] = useState<Array<{ imageId: string; classId: string | null; action: string }>>([]);

  // Auto-advance AI State
  const [autoAdvanceAI, setAutoAdvanceAI] = useState(false);
  const [autoAdvanceThreshold, setAutoAdvanceThreshold] = useState(0.85);

  // Class delete state
  const [classToDelete, setClassToDelete] = useState<string | null>(null);

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

  // Fetch dataset classes (classes are now dataset-specific)
  const { data: classes, refetch: refetchClasses } = useQuery({
    queryKey: ["cls-dataset-classes", datasetId],
    queryFn: () => apiClient.getCLSDatasetClasses(datasetId!),
    enabled: !!datasetId,
    staleTime: 30000,
  });

  // Recent classes with full data
  const recentClasses = useMemo(() => {
    if (!classes) return [];
    return recentClassIds
      .map(id => classes.find(c => c.id === id))
      .filter(Boolean) as NonNullable<typeof classes>;
  }, [recentClassIds, classes]);

  // Filtered classes based on search
  const filteredClasses = useMemo(() => {
    if (!classes) return [];
    if (!searchValue.trim()) return classes;
    const search = searchValue.toLowerCase();
    return classes.filter(c => 
      c.name.toLowerCase().includes(search) ||
      c.display_name?.toLowerCase().includes(search)
    );
  }, [classes, searchValue]);

  // Check if search value matches any existing class
  const exactMatch = useMemo(() => {
    if (!classes || !searchValue.trim()) return null;
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
    staleTime: 0, // Always consider stale to ensure fresh label data
    refetchOnMount: "always", // Refetch when navigating back to this image
  });

  // Fetch available AI models
  const { data: aiModels } = useQuery({
    queryKey: ["cls-ai-models"],
    queryFn: () => apiClient.getCLSAIModels(),
    staleTime: 300000, // 5 minutes
  });

  // Clear AI predictions when image changes
  useEffect(() => {
    setAiPredictions([]);
  }, [currentImageId]);

  // AI Predict function
  const handleAIPredict = useCallback(async () => {
    if (!datasetId || !currentImageId || isAIPredicting) return;

    setIsAIPredicting(true);
    setAiPredictions([]);

    try {
      const result = await apiClient.predictCLSAI({
        image_id: currentImageId,
        dataset_id: datasetId,
        model: selectedAIModel,
        top_k: 5,
        threshold: 0.1,
      });

      setAiPredictions(result.predictions);

      if (result.predictions.length === 0) {
        toast.info("No confident predictions found");
      } else {
        toast.success(`Found ${result.predictions.length} predictions (${result.processing_time_ms}ms)`);
      }
    } catch (error) {
      toast.error(`AI prediction failed: ${error instanceof Error ? error.message : "Unknown error"}`);
    } finally {
      setIsAIPredicting(false);
    }
  }, [datasetId, currentImageId, selectedAIModel, isAIPredicting]);

  // Reset zoom when image changes
  useEffect(() => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }, [currentImageId]);

  // Zoom handlers
  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? -0.1 : 0.1;
    setZoom(prev => Math.min(Math.max(prev + delta, 0.5), 3));
  }, []);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (zoom > 1) {
      setIsDragging(true);
      setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
    }
  }, [zoom, pan]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (isDragging && zoom > 1) {
      setPan({ x: e.clientX - dragStart.x, y: e.clientY - dragStart.y });
    }
  }, [isDragging, dragStart, zoom]);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  const resetZoom = useCallback(() => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }, []);

  // Prefetch next images for instant navigation
  const nextImageIds = useMemo(() => {
    if (!queue?.image_ids || !currentImageId) return [];
    const currentIndex = queue.image_ids.indexOf(currentImageId);
    if (currentIndex === -1) return [];
    return queue.image_ids.slice(currentIndex + 1, currentIndex + 4);
  }, [queue?.image_ids, currentImageId]);

  // Prefetch queries for next images
  useEffect(() => {
    nextImageIds.forEach(imageId => {
      queryClient.prefetchQuery({
        queryKey: ["cls-labeling-image", datasetId, imageId],
        queryFn: () => apiClient.getCLSLabelingImage(datasetId!, imageId),
        staleTime: 60000,
      });
    });
  }, [nextImageIds, datasetId, queryClient]);

  // Undo function
  const handleUndo = useCallback(async () => {
    if (undoStack.length === 0) {
      toast.info("Nothing to undo");
      return;
    }
    const lastAction = undoStack[undoStack.length - 1];
    try {
      // Clear the label for the last labeled image
      await apiClient.submitCLSLabeling(datasetId!, lastAction.imageId, {
        action: "skip", // This effectively removes the label
      });
      setUndoStack(prev => prev.slice(0, -1));
      setCurrentImageId(lastAction.imageId);
      router.push(`/classification/labeling?dataset=${datasetId}&image=${lastAction.imageId}`, { scroll: false });
      toast.success("Undone last label");
      queryClient.invalidateQueries({ queryKey: ["cls-labeling-progress", datasetId] });
      queryClient.invalidateQueries({ queryKey: ["cls-labeling-queue", datasetId] });
    } catch (error) {
      toast.error("Failed to undo");
    }
  }, [undoStack, datasetId, router, queryClient]);

  // Set initial image when queue loads
  useEffect(() => {
    if (!currentImageId && queue?.image_ids && queue.image_ids.length > 0) {
      setCurrentImageId(queue.image_ids[0]);
    }
  }, [queue, currentImageId]);

  // Delete class mutation
  const deleteClassMutation = useMutation({
    mutationFn: (classId: string) => apiClient.deleteCLSClass(classId),
    onSuccess: () => {
      toast.success("Class deleted");
      refetchClasses();
      setClassToDelete(null);
      queryClient.invalidateQueries({ queryKey: ["cls-labeling-progress", datasetId] });
    },
    onError: (error: Error) => {
      toast.error(`Failed to delete class: ${error.message}`);
    },
  });

  // Create class mutation
  const createClassMutation = useMutation({
    mutationFn: (name: string) => apiClient.createCLSClass({
      dataset_id: datasetId!,
      name: name.toLowerCase().replace(/\s+/g, '_'),
      display_name: name,
      color: CLASS_COLORS[(classes?.length || 0) % CLASS_COLORS.length],
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
        // Add to undo stack
        setUndoStack(prev => [...prev.slice(-9), { imageId: currentImageId!, classId, action: "label" }]);
        toast.success("Label saved");
      } else if (action === "skip") {
        toast.info("Image skipped");
      } else {
        toast.info("Marked for review");
      }

      queryClient.invalidateQueries({ queryKey: ["cls-labeling-progress", datasetId] });
      queryClient.invalidateQueries({ queryKey: ["cls-labeling-queue", datasetId] });
      queryClient.invalidateQueries({ queryKey: ["cls-dataset-classes", datasetId] });
      // Invalidate ALL image queries for this dataset to ensure fresh data when navigating
      queryClient.invalidateQueries({
        queryKey: ["cls-labeling-image", datasetId],
        exact: false // This will invalidate all queries that START with this key
      });

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
    // Ctrl+Z for undo (works anywhere)
    if ((e.ctrlKey || e.metaKey) && e.key === "z") {
      e.preventDefault();
      handleUndo();
      return;
    }

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
      case "a":
        e.preventDefault();
        handleAIPredict();
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
      case "r":
        // Reset zoom
        e.preventDefault();
        resetZoom();
        break;
      case "=":
      case "+":
        // Zoom in
        e.preventDefault();
        setZoom(prev => Math.min(prev + 0.25, 3));
        break;
      case "-":
        // Zoom out
        e.preventDefault();
        setZoom(prev => Math.max(prev - 0.25, 0.5));
        break;
    }
  }, [datasetId, currentImageId, recentClasses, imageData, submitMutation, handleSelectClass, handleAIPredict, handleUndo, resetZoom, router]);

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

            {/* AI Model Selector */}
            <Select value={selectedAIModel} onValueChange={setSelectedAIModel}>
              <SelectTrigger className="w-36">
                <Sparkles className="h-4 w-4 mr-2" />
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {aiModels?.zero_shot_models?.map((model) => (
                  <SelectItem key={model.id} value={model.id}>
                    {model.name}
                  </SelectItem>
                ))}
                {aiModels?.trained_models && aiModels.trained_models.length > 0 && (
                  <>
                    <SelectItem disabled value="divider">──────────</SelectItem>
                    {aiModels.trained_models.map((model) => (
                      <SelectItem key={model.id} value={model.id}>
                        {model.name}
                      </SelectItem>
                    ))}
                  </>
                )}
              </SelectContent>
            </Select>

            {/* AI Suggest Button */}
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleAIPredict}
                  disabled={isAIPredicting || !currentImageId}
                >
                  {isAIPredicting ? (
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <Wand2 className="h-4 w-4 mr-2" />
                  )}
                  AI Suggest
                  <kbd className="ml-2 px-1 py-0.5 text-xs bg-muted rounded">A</kbd>
                </Button>
              </TooltipTrigger>
              <TooltipContent>Run AI classification on current image</TooltipContent>
            </Tooltip>

            {/* Undo Button */}
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={handleUndo}
                  disabled={undoStack.length === 0}
                >
                  <Undo2 className="h-4 w-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Undo last label (Ctrl+Z)</TooltipContent>
            </Tooltip>

            {/* Auto-advance AI Settings */}
            <Popover>
              <PopoverTrigger asChild>
                <Button variant="ghost" size="icon">
                  <Settings2 className="h-4 w-4" />
                </Button>
              </PopoverTrigger>
              <PopoverContent className="w-72" align="end">
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <Label htmlFor="auto-advance" className="text-sm">Auto-advance AI</Label>
                    <Switch
                      id="auto-advance"
                      checked={autoAdvanceAI}
                      onCheckedChange={setAutoAdvanceAI}
                    />
                  </div>
                  {autoAdvanceAI && (
                    <div className="space-y-2">
                      <Label className="text-xs text-muted-foreground">
                        Threshold: {(autoAdvanceThreshold * 100).toFixed(0)}%
                      </Label>
                      <Slider
                        value={[autoAdvanceThreshold]}
                        onValueChange={([v]) => setAutoAdvanceThreshold(v)}
                        min={0.5}
                        max={0.99}
                        step={0.01}
                      />
                      <p className="text-xs text-muted-foreground">
                        Auto-label when AI confidence exceeds threshold
                      </p>
                    </div>
                  )}
                </div>
              </PopoverContent>
            </Popover>

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
                  <p><kbd className="px-1 bg-muted rounded">A</kbd> AI Suggest</p>
                  <p><kbd className="px-1 bg-muted rounded">←/→</kbd> Navigate</p>
                  <p><kbd className="px-1 bg-muted rounded">Ctrl+Z</kbd> Undo</p>
                  <p><kbd className="px-1 bg-muted rounded">+/-</kbd> Zoom</p>
                  <p><kbd className="px-1 bg-muted rounded">R</kbd> Reset zoom</p>
                </div>
              </TooltipContent>
            </Tooltip>
          </div>
        </div>

        {/* Main Content */}
        <div className="flex-1 flex min-h-0">
          {/* Image Panel with Zoom */}
          <div className="flex-1 flex flex-col min-w-0">
            <div
              ref={imageContainerRef}
              className="flex-1 p-4 flex items-center justify-center bg-muted/30 min-h-0 overflow-hidden relative"
              onWheel={handleWheel}
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseUp}
              style={{ cursor: zoom > 1 ? (isDragging ? "grabbing" : "grab") : "default" }}
            >
              {imageLoading ? (
                <Loader2 className="h-12 w-12 animate-spin text-muted-foreground" />
              ) : imageData?.image ? (
                <div
                  className="relative w-full h-full flex items-center justify-center transition-transform duration-100"
                  style={{
                    transform: `scale(${zoom}) translate(${pan.x / zoom}px, ${pan.y / zoom}px)`,
                  }}
                >
                  <Image
                    src={imageData.image.image_url}
                    alt={imageData.image.filename}
                    fill
                    className="object-contain pointer-events-none"
                    sizes="(max-width: 1200px) 60vw, 800px"
                    priority
                    draggable={false}
                  />
                </div>
              ) : (
                <div className="text-muted-foreground">No image</div>
              )}

              {/* Zoom Controls Overlay */}
              {zoom !== 1 && (
                <div className="absolute bottom-4 left-4 flex items-center gap-2 bg-background/80 backdrop-blur-sm rounded-lg px-3 py-2 shadow-lg">
                  <Button variant="ghost" size="icon" className="h-7 w-7" onClick={() => setZoom(prev => Math.max(prev - 0.25, 0.5))}>
                    <ZoomOut className="h-4 w-4" />
                  </Button>
                  <span className="text-sm font-medium w-12 text-center">{Math.round(zoom * 100)}%</span>
                  <Button variant="ghost" size="icon" className="h-7 w-7" onClick={() => setZoom(prev => Math.min(prev + 0.25, 3))}>
                    <ZoomIn className="h-4 w-4" />
                  </Button>
                  <Button variant="ghost" size="icon" className="h-7 w-7" onClick={resetZoom}>
                    <RotateCcw className="h-4 w-4" />
                  </Button>
                </div>
              )}
            </div>

            {/* Thumbnail Strip */}
            {queue?.image_ids && queue.image_ids.length > 0 && (
              <div className="flex-none border-t bg-muted/30">
                <div className="flex items-center gap-1 p-2 overflow-x-auto">
                  {queue.image_ids.slice(0, 20).map((imgId, index) => {
                    const isActive = imgId === currentImageId;
                    return (
                      <button
                        key={imgId}
                        onClick={() => {
                          setCurrentImageId(imgId);
                          router.push(`/classification/labeling?dataset=${datasetId}&image=${imgId}`, { scroll: false });
                        }}
                        className={`relative flex-shrink-0 w-12 h-12 rounded border-2 overflow-hidden transition-all ${
                          isActive
                            ? "border-primary ring-2 ring-primary/30"
                            : "border-transparent hover:border-muted-foreground/30"
                        }`}
                      >
                        <div className="absolute inset-0 bg-muted flex items-center justify-center text-xs text-muted-foreground">
                          {index + 1}
                        </div>
                      </button>
                    );
                  })}
                  {queue.image_ids.length > 20 && (
                    <span className="text-xs text-muted-foreground px-2">
                      +{queue.image_ids.length - 20} more
                    </span>
                  )}
                </div>
              </div>
            )}

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
          <div className="w-80 flex-none border-l flex flex-col bg-background min-h-0 overflow-hidden">
            {/* Current Label - shows selected class or existing label */}
            {(() => {
              const selectedClass = selectedClassId ? classes?.find(c => c.id === selectedClassId) : null;
              const existingLabel = imageData?.current_labels?.[0];
              const displayClass = selectedClass || existingLabel;

              if (!displayClass) return null;

              const isSelected = !!selectedClass;
              const isPending = isSelected && submitMutation.isPending;

              return (
                <div className={`flex-none p-3 border-b ${isSelected ? "bg-blue-50 dark:bg-blue-950/30" : "bg-green-50 dark:bg-green-950/30"}`}>
                  <p className="text-xs text-muted-foreground mb-1 flex items-center gap-2">
                    {isSelected ? "Selected" : "Current Label"}
                    {isPending && <Loader2 className="h-3 w-3 animate-spin" />}
                  </p>
                  <div className="flex items-center gap-2">
                    <div
                      className="w-4 h-4 rounded-full border-2 border-white shadow-sm"
                      style={{ backgroundColor: selectedClass?.color || existingLabel?.class_color }}
                    />
                    <span className="font-medium text-base">
                      {selectedClass?.display_name || selectedClass?.name || existingLabel?.class_name}
                    </span>
                    {isSelected && (
                      <Badge variant="outline" className="ml-auto text-blue-600 border-blue-300">
                        Saving...
                      </Badge>
                    )}
                  </div>
                </div>
              );
            })()}

            {/* AI Predictions */}
            {aiPredictions.length > 0 && (
              <div className="flex-none p-3 border-b bg-purple-50 dark:bg-purple-950/30">
                <div className="flex items-center gap-2 mb-2">
                  <Sparkles className="h-4 w-4 text-purple-600" />
                  <p className="text-xs font-medium text-purple-600">AI Suggestions</p>
                </div>
                <div className="space-y-1">
                  {aiPredictions.map((pred, index) => {
                    const classData = classes?.find(c => c.id === pred.class_id);
                    return (
                      <button
                        key={pred.class_id}
                        onClick={() => handleSelectClass(pred.class_id)}
                        className="w-full flex items-center gap-2 p-2 rounded-md hover:bg-purple-100 dark:hover:bg-purple-900/30 transition-colors text-left"
                      >
                        <div
                          className="w-3 h-3 rounded-full flex-shrink-0"
                          style={{ backgroundColor: classData?.color || "#8b5cf6" }}
                        />
                        <span className="flex-1 text-sm truncate">
                          {pred.class_name}
                        </span>
                        <Badge variant="secondary" className="text-xs">
                          {(pred.confidence * 100).toFixed(0)}%
                        </Badge>
                        {index === 0 && (
                          <Zap className="h-3 w-3 text-yellow-500" />
                        )}
                      </button>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Recent Classes - Quick Select */}
            {recentClasses.length > 0 && !searchValue && (
              <div className="flex-none p-3 border-b">
                <p className="text-xs text-muted-foreground mb-2 font-medium">Recent (1-9)</p>
                <div className="space-y-1">
                  {recentClasses.map((cls, index) => (
                    <button
                      key={cls.id}
                      onClick={() => handleSelectClass(cls.id)}
                      className={`w-full flex items-center gap-2 p-2 rounded-md transition-all text-left
                        hover:bg-primary/10 hover:border-primary
                        border border-transparent
                        ${selectedClassId === cls.id ? "bg-primary/20 border-primary" : ""}
                      `}
                    >
                      <div
                        className="w-4 h-4 rounded-full flex-shrink-0 border-2 border-white shadow-sm"
                        style={{ backgroundColor: cls.color }}
                      />
                      <span className="flex-1 text-sm font-medium truncate">
                        {cls.display_name || cls.name}
                      </span>
                      <kbd className="px-1.5 py-0.5 text-xs bg-muted rounded font-mono">
                        {index + 1}
                      </kbd>
                      <Badge variant="secondary" className="text-xs">
                        {cls.image_count}
                      </Badge>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Search Input */}
            <div className="flex-none p-3 border-b">
              <div className="relative">
                <input
                  ref={inputRef}
                  type="text"
                  placeholder="Search or create class..."
                  value={searchValue}
                  onChange={(e) => setSearchValue(e.target.value)}
                  className="w-full px-3 py-2 text-sm border rounded-md bg-background focus:outline-none focus:ring-2 focus:ring-primary"
                />
                {searchValue && (
                  <button
                    onClick={() => setSearchValue("")}
                    className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                  >
                    ×
                  </button>
                )}
              </div>

              {/* Create New Class Option */}
              {searchValue && !exactMatch && (
                <button
                  onClick={handleCreateClass}
                  disabled={createClassMutation.isPending}
                  className="w-full mt-2 flex items-center gap-2 p-2 rounded-md bg-primary/10 text-primary hover:bg-primary/20 transition-colors text-left"
                >
                  <Plus className="h-4 w-4" />
                  <span className="text-sm font-medium">Create &quot;{searchValue}&quot;</span>
                  {createClassMutation.isPending && (
                    <Loader2 className="h-3 w-3 ml-auto animate-spin" />
                  )}
                </button>
              )}
            </div>

            {/* All Classes - Scrollable */}
            <div className="flex-1 overflow-y-auto min-h-0">
              <div className="p-3">
                <p className="text-xs text-muted-foreground mb-2 font-medium">
                  {searchValue ? `Matching (${filteredClasses.length})` : `All Classes (${classes?.length || 0})`}
                </p>
                {filteredClasses.length === 0 ? (
                  <p className="text-sm text-muted-foreground text-center py-4">No classes found</p>
                ) : (
                  <div className="space-y-1">
                    {filteredClasses.map((cls) => (
                      <div
                        key={cls.id}
                        role="button"
                        tabIndex={0}
                        onClick={() => handleSelectClass(cls.id)}
                        onKeyDown={(e) => e.key === "Enter" && handleSelectClass(cls.id)}
                        className={`w-full flex items-center gap-2 p-2 rounded-md transition-all text-left group cursor-pointer
                          hover:bg-accent
                          ${selectedClassId === cls.id ? "bg-accent ring-2 ring-primary" : ""}
                        `}
                      >
                        <div
                          className="w-4 h-4 rounded-full flex-shrink-0"
                          style={{ backgroundColor: cls.color }}
                        />
                        <span className="flex-1 text-sm truncate">{cls.display_name || cls.name}</span>
                        <Badge variant="secondary" className="text-xs">
                          {cls.image_count}
                        </Badge>
                        <button
                          className="h-6 w-6 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity rounded hover:bg-destructive/10"
                          onClick={(e) => {
                            e.stopPropagation();
                            setClassToDelete(cls.id);
                          }}
                        >
                          <Trash2 className="h-3 w-3 text-destructive" />
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* Stats */}
            {progress && (
              <div className="flex-none p-3 border-t">
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
            <div className="flex-none p-3 border-t">
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

      {/* Delete Class Confirmation Dialog */}
      <AlertDialog open={!!classToDelete} onOpenChange={(open) => !open && setClassToDelete(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Class</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete this class? This will remove all labels associated with this class.
              This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
              onClick={() => classToDelete && deleteClassMutation.mutate(classToDelete)}
              disabled={deleteClassMutation.isPending}
            >
              {deleteClassMutation.isPending ? (
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              ) : (
                <Trash2 className="h-4 w-4 mr-2" />
              )}
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </TooltipProvider>
  );
}
