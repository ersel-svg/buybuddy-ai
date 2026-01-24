"use client";

import { useState } from "react";
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
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { Checkbox } from "@/components/ui/checkbox";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Play,
  Loader2,
  ChevronDown,
  Database,
  FlaskConical,
  AlertCircle,
  Info,
  GraduationCap,
  Layers,
} from "lucide-react";
import type {
  EmbeddingModel,
  Dataset,
  EvaluationExtractionRequest,
  ImageType,
  FrameSelection,
  CollectionMode,
  TrainingRun,
  TrainingCheckpoint,
} from "@/types";

interface EvaluationExtractionTabProps {
  models: EmbeddingModel[] | undefined;
}

export function EvaluationExtractionTab({ models }: EvaluationExtractionTabProps) {
  const queryClient = useQueryClient();

  // Model source: base model or trained model
  const [modelSource, setModelSource] = useState<"base" | "trained">("base");

  // Base model selection
  const [selectedModelId, setSelectedModelId] = useState<string>("");

  // Trained model selection
  const [selectedTrainingRunId, setSelectedTrainingRunId] = useState<string>("");
  const [selectedCheckpointId, setSelectedCheckpointId] = useState<string>("");

  // Dataset selection
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>("");

  // Image types selection
  const [imageTypes, setImageTypes] = useState<ImageType[]>(["synthetic"]);

  // Frame selection
  const [frameSelection, setFrameSelection] = useState<FrameSelection | "all">("first");
  const [frameInterval, setFrameInterval] = useState([5]);
  const [maxFrames, setMaxFrames] = useState([4]);

  // Collection config
  const [collectionMode, setCollectionMode] = useState<CollectionMode>("create");
  const [collectionName, setCollectionName] = useState("");

  // Collapsible states
  const [frameConfigOpen, setFrameConfigOpen] = useState(false);
  const [collectionConfigOpen, setCollectionConfigOpen] = useState(false);

  // Fetch datasets
  const { data: datasets, isLoading: datasetsLoading } = useQuery({
    queryKey: ["datasets"],
    queryFn: () => apiClient.getDatasets(),
  });

  // Fetch existing Qdrant collections for append mode
  const { data: collections } = useQuery({
    queryKey: ["qdrant-collections"],
    queryFn: () => apiClient.getQdrantCollections(),
  });

  // Fetch completed training runs
  const { data: trainingRunsData, isLoading: trainingRunsLoading } = useQuery({
    queryKey: ["training-runs", "completed"],
    queryFn: () => apiClient.getTrainingRuns({ status: "completed", limit: 100 }),
  });
  const trainingRuns = trainingRunsData?.items || [];

  // Fetch checkpoints for selected training run
  const { data: checkpoints, isLoading: checkpointsLoading } = useQuery({
    queryKey: ["training-checkpoints", selectedTrainingRunId],
    queryFn: () => apiClient.getTrainingCheckpoints(selectedTrainingRunId),
    enabled: !!selectedTrainingRunId,
  });

  // Selected collection for append mode
  const [selectedCollection, setSelectedCollection] = useState<string>("");

  // Get selected dataset
  const selectedDataset = datasets?.find((d) => d.id === selectedDatasetId);

  // Get selected base model
  const selectedModel = models?.find((m) => m.id === selectedModelId);

  // Get selected training run
  const selectedTrainingRun = trainingRuns.find((r: TrainingRun) => r.id === selectedTrainingRunId);

  // Get selected checkpoint
  const selectedCheckpoint = checkpoints?.find((c: TrainingCheckpoint) => c.id === selectedCheckpointId);

  // Get effective embedding dimension based on model source
  const effectiveEmbeddingDim = modelSource === "base"
    ? selectedModel?.embedding_dim
    : selectedTrainingRun?.training_config?.embedding_dim;

  // Start evaluation extraction
  const startExtractionMutation = useMutation({
    mutationFn: (params: EvaluationExtractionRequest) =>
      apiClient.startEvaluationExtraction(params),
    onSuccess: (data) => {
      toast.success(
        `Evaluation extraction completed. ${data.total_embeddings} embeddings created in "${data.collection_name}".`
      );
      queryClient.invalidateQueries({ queryKey: ["embedding-jobs"] });
      queryClient.invalidateQueries({ queryKey: ["qdrant-collections"] });
    },
    onError: (error) => {
      toast.error(`Failed to start extraction: ${error.message}`);
    },
  });

  const handleImageTypeToggle = (type: ImageType) => {
    setImageTypes((prev) =>
      prev.includes(type) ? prev.filter((t) => t !== type) : [...prev, type]
    );
  };

  const handleStartExtraction = () => {
    // Validate model selection based on source
    if (modelSource === "base") {
      if (!selectedModelId) {
        toast.error("Select a base model to evaluate");
        return;
      }
    } else {
      if (!selectedTrainingRunId) {
        toast.error("Select a training run");
        return;
      }
      if (!selectedCheckpointId) {
        toast.error("Select a checkpoint");
        return;
      }
    }

    if (!selectedDatasetId) {
      toast.error("Select a dataset");
      return;
    }

    if (imageTypes.length === 0) {
      toast.error("Select at least one image type");
      return;
    }

    const request: EvaluationExtractionRequest = {
      model_id: modelSource === "base" ? selectedModelId : undefined,
      training_run_id: modelSource === "trained" ? selectedTrainingRunId : undefined,
      checkpoint_id: modelSource === "trained" ? selectedCheckpointId : undefined,
      dataset_id: selectedDatasetId,
      image_types: imageTypes,
      frame_selection: frameSelection,
      frame_interval: frameInterval[0],
      max_frames: maxFrames[0],
      collection_mode: collectionMode,
      collection_name: collectionName || undefined,
    };

    startExtractionMutation.mutate(request);
  };

  const datasetName = selectedDataset?.name.toLowerCase().replace(/\s+/g, "_") || "dataset";
  const modelName = modelSource === "base"
    ? (selectedModel?.name.toLowerCase().replace(/\s+/g, "_") || "model")
    : (selectedTrainingRun?.name.toLowerCase().replace(/\s+/g, "_") || "trained");
  const checkpointSuffix = selectedCheckpoint ? `_ep${selectedCheckpoint.epoch}` : "";
  const defaultCollectionName = `eval_${datasetName}_${modelName}${checkpointSuffix}`;

  const canStartExtraction = modelSource === "base"
    ? (selectedModelId && selectedDatasetId && imageTypes.length > 0)
    : (selectedTrainingRunId && selectedCheckpointId && selectedDatasetId && imageTypes.length > 0);

  return (
    <div className="space-y-6">
      {/* Description Card */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <FlaskConical className="h-5 w-5" />
            Evaluation Extraction
          </CardTitle>
          <CardDescription>
            Extract embeddings from a dataset using a specific trained model for evaluation.
            Use this to measure model performance on held-out test sets.
          </CardDescription>
        </CardHeader>
      </Card>

      {/* Configuration */}
      <div className="grid grid-cols-2 gap-6">
        {/* Left Column: Model & Dataset Selection */}
        <div className="space-y-6">
          {/* Model Selection */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                Model Selection
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger>
                      <Info className="h-4 w-4 text-muted-foreground" />
                    </TooltipTrigger>
                    <TooltipContent side="right" className="max-w-xs">
                      <p className="text-xs">Choose between evaluating a base pre-trained model (e.g., DINOv2, CLIP) or a fine-tuned model from a completed training run. Comparing both helps measure training improvement.</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </CardTitle>
              <CardDescription>
                Select a base model or a trained model with checkpoint
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <Tabs value={modelSource} onValueChange={(v) => setModelSource(v as "base" | "trained")}>
                <TabsList className="grid w-full grid-cols-2">
                  <TabsTrigger value="base" className="flex items-center gap-2">
                    <Layers className="h-4 w-4" />
                    Base Model
                  </TabsTrigger>
                  <TabsTrigger value="trained" className="flex items-center gap-2">
                    <GraduationCap className="h-4 w-4" />
                    Trained Model
                  </TabsTrigger>
                </TabsList>

                {/* Base Model Selection */}
                <TabsContent value="base" className="space-y-4 mt-4">
                  <div className="space-y-2">
                    <Label>Embedding Model</Label>
                    <Select value={selectedModelId} onValueChange={setSelectedModelId}>
                      <SelectTrigger>
                        <SelectValue placeholder="Select a model" />
                      </SelectTrigger>
                      <SelectContent>
                        {models?.map((model) => (
                          <SelectItem key={model.id} value={model.id}>
                            <div className="flex items-center gap-2">
                              {model.name}
                              {model.is_matching_active && (
                                <Badge variant="secondary" className="text-xs">
                                  Active
                                </Badge>
                              )}
                            </div>
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  {selectedModel && (
                    <div className="p-3 bg-muted rounded-lg text-sm">
                      <div className="grid grid-cols-2 gap-2">
                        <div>
                          <p className="text-muted-foreground">Type</p>
                          <p className="font-medium">{selectedModel.model_type}</p>
                        </div>
                        <div>
                          <p className="text-muted-foreground">Dimension</p>
                          <p className="font-medium">{selectedModel.embedding_dim}d</p>
                        </div>
                      </div>
                    </div>
                  )}
                </TabsContent>

                {/* Trained Model Selection */}
                <TabsContent value="trained" className="space-y-4 mt-4">
                  {/* Training Run Selection */}
                  <div className="space-y-2">
                    <Label>Training Run</Label>
                    <Select
                      value={selectedTrainingRunId}
                      onValueChange={(v) => {
                        setSelectedTrainingRunId(v);
                        setSelectedCheckpointId(""); // Reset checkpoint when run changes
                      }}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Select a completed training run" />
                      </SelectTrigger>
                      <SelectContent>
                        {trainingRunsLoading ? (
                          <SelectItem value="_loading" disabled>
                            <Loader2 className="h-4 w-4 animate-spin mr-2" />
                            Loading...
                          </SelectItem>
                        ) : trainingRuns.length === 0 ? (
                          <SelectItem value="_none" disabled>
                            No completed training runs
                          </SelectItem>
                        ) : (
                          trainingRuns.map((run: TrainingRun) => (
                            <SelectItem key={run.id} value={run.id}>
                              <div className="flex items-center gap-2">
                                <span>{run.name}</span>
                                <Badge variant="outline" className="text-xs">
                                  {run.base_model_type}
                                </Badge>
                                {run.best_val_recall_at_1 && (
                                  <Badge variant="secondary" className="text-xs">
                                    R@1: {(run.best_val_recall_at_1 * 100).toFixed(1)}%
                                  </Badge>
                                )}
                              </div>
                            </SelectItem>
                          ))
                        )}
                      </SelectContent>
                    </Select>
                  </div>

                  {/* Checkpoint Selection */}
                  {selectedTrainingRunId && (
                    <div className="space-y-2">
                      <Label>Checkpoint</Label>
                      <Select value={selectedCheckpointId} onValueChange={setSelectedCheckpointId}>
                        <SelectTrigger>
                          <SelectValue placeholder="Select a checkpoint" />
                        </SelectTrigger>
                        <SelectContent>
                          {checkpointsLoading ? (
                            <SelectItem value="_loading" disabled>
                              <Loader2 className="h-4 w-4 animate-spin mr-2" />
                              Loading...
                            </SelectItem>
                          ) : !checkpoints || checkpoints.length === 0 ? (
                            <SelectItem value="_none" disabled>
                              No checkpoints available
                            </SelectItem>
                          ) : (
                            checkpoints.map((cp: TrainingCheckpoint) => (
                              <SelectItem key={cp.id} value={cp.id}>
                                <div className="flex items-center gap-2">
                                  <span>Epoch {cp.epoch}</span>
                                  {cp.is_best && (
                                    <Badge variant="default" className="text-xs">
                                      Best
                                    </Badge>
                                  )}
                                  {cp.is_final && (
                                    <Badge variant="secondary" className="text-xs">
                                      Final
                                    </Badge>
                                  )}
                                  {cp.val_recall_at_1 && (
                                    <span className="text-xs text-muted-foreground">
                                      R@1: {(cp.val_recall_at_1 * 100).toFixed(1)}%
                                    </span>
                                  )}
                                </div>
                              </SelectItem>
                            ))
                          )}
                        </SelectContent>
                      </Select>
                    </div>
                  )}

                  {/* Training Run Info */}
                  {selectedTrainingRun && (
                    <div className="p-3 bg-muted rounded-lg text-sm">
                      <div className="grid grid-cols-2 gap-2">
                        <div>
                          <p className="text-muted-foreground">Base Model</p>
                          <p className="font-medium">{selectedTrainingRun.base_model_type}</p>
                        </div>
                        <div>
                          <p className="text-muted-foreground">Dimension</p>
                          <p className="font-medium">{selectedTrainingRun.training_config?.embedding_dim || "N/A"}d</p>
                        </div>
                        <div>
                          <p className="text-muted-foreground">Train Products</p>
                          <p className="font-medium">{selectedTrainingRun.train_product_count}</p>
                        </div>
                        <div>
                          <p className="text-muted-foreground">Best Epoch</p>
                          <p className="font-medium">{selectedTrainingRun.best_epoch || "N/A"}</p>
                        </div>
                      </div>
                    </div>
                  )}
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>

          {/* Dataset Selection */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                Dataset Selection
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger>
                      <Info className="h-4 w-4 text-muted-foreground" />
                    </TooltipTrigger>
                    <TooltipContent side="right" className="max-w-xs">
                      <p className="text-xs">Select a held-out test dataset for evaluation. Use a dataset not seen during training to get unbiased performance metrics (Recall@1, Recall@5, mAP).</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </CardTitle>
              <CardDescription>
                Select the evaluation dataset
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {datasetsLoading ? (
                <div className="flex items-center justify-center py-4">
                  <Loader2 className="h-6 w-6 animate-spin" />
                </div>
              ) : !datasets || datasets.length === 0 ? (
                <div className="text-center py-4">
                  <AlertCircle className="h-8 w-8 mx-auto text-muted-foreground mb-2" />
                  <p className="text-muted-foreground">No datasets available</p>
                  <p className="text-xs text-muted-foreground mt-1">
                    Create a dataset on the Datasets page first
                  </p>
                </div>
              ) : (
                <>
                  <div className="space-y-2">
                    <Label>Dataset</Label>
                    <Select value={selectedDatasetId} onValueChange={setSelectedDatasetId}>
                      <SelectTrigger>
                        <SelectValue placeholder="Select a dataset" />
                      </SelectTrigger>
                      <SelectContent>
                        {datasets.map((dataset: Dataset) => (
                          <SelectItem key={dataset.id} value={dataset.id}>
                            {dataset.name} ({dataset.product_count} products)
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  {selectedDataset && (
                    <div className="p-3 bg-muted rounded-lg text-sm">
                      <div className="grid grid-cols-2 gap-2">
                        <div>
                          <p className="text-muted-foreground">Products</p>
                          <p className="font-medium">{selectedDataset.product_count}</p>
                        </div>
                        <div>
                          <p className="text-muted-foreground">Created</p>
                          <p className="font-medium">
                            {new Date(selectedDataset.created_at).toLocaleDateString()}
                          </p>
                        </div>
                      </div>
                      {selectedDataset.description && (
                        <p className="mt-2 text-muted-foreground">
                          {selectedDataset.description}
                        </p>
                      )}
                    </div>
                  )}
                </>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Right Column: Image Types & Output */}
        <div className="space-y-6">
          {/* Image Types */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                Image Configuration
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger>
                      <Info className="h-4 w-4 text-muted-foreground" />
                    </TooltipTrigger>
                    <TooltipContent side="right" className="max-w-xs">
                      <p className="text-xs">Select which image types to include in evaluation. For realistic performance measurement, evaluate on the same image types you expect in production (typically synthetic for products, real for cutouts).</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </CardTitle>
              <CardDescription>Select image types to evaluate</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-3">
                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="eval-synthetic"
                    checked={imageTypes.includes("synthetic")}
                    onCheckedChange={() => handleImageTypeToggle("synthetic")}
                  />
                  <div className="grid gap-1.5 leading-none">
                    <div className="flex items-center gap-1">
                      <Label htmlFor="eval-synthetic" className="font-medium">
                        Synthetic Frames
                      </Label>
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger>
                            <Info className="h-3 w-3 text-muted-foreground" />
                          </TooltipTrigger>
                          <TooltipContent side="right" className="max-w-xs">
                            <p className="text-xs">360° rotation video frames. Best for evaluating product catalog embeddings as gallery set.</p>
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    </div>
                    <p className="text-xs text-muted-foreground">
                      Product video frames
                    </p>
                  </div>
                </div>

                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="eval-real"
                    checked={imageTypes.includes("real")}
                    onCheckedChange={() => handleImageTypeToggle("real")}
                  />
                  <div className="grid gap-1.5 leading-none">
                    <div className="flex items-center gap-1">
                      <Label htmlFor="eval-real" className="font-medium">
                        Real Images
                      </Label>
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger>
                            <Info className="h-3 w-3 text-muted-foreground" />
                          </TooltipTrigger>
                          <TooltipContent side="right" className="max-w-xs">
                            <p className="text-xs">Real store shelf photos. Best for evaluating query set performance (simulating production queries).</p>
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    </div>
                    <p className="text-xs text-muted-foreground">
                      Real product photos
                    </p>
                  </div>
                </div>

                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="eval-augmented"
                    checked={imageTypes.includes("augmented")}
                    onCheckedChange={() => handleImageTypeToggle("augmented")}
                  />
                  <div className="grid gap-1.5 leading-none">
                    <div className="flex items-center gap-1">
                      <Label htmlFor="eval-augmented" className="font-medium">
                        Augmented Images
                      </Label>
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger>
                            <Info className="h-3 w-3 text-muted-foreground" />
                          </TooltipTrigger>
                          <TooltipContent side="right" className="max-w-xs">
                            <p className="text-xs">Augmented versions with transforms. Useful for testing model robustness to variations.</p>
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    </div>
                    <p className="text-xs text-muted-foreground">
                      Augmented versions
                    </p>
                  </div>
                </div>
              </div>

              {/* Frame Selection */}
              <Collapsible open={frameConfigOpen} onOpenChange={setFrameConfigOpen}>
                <CollapsibleTrigger asChild>
                  <Button variant="outline" className="w-full justify-between mt-4">
                    <div className="flex items-center gap-2">
                      <span>Frame Selection</span>
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger asChild onClick={(e) => e.stopPropagation()}>
                            <span className="inline-flex"><Info className="h-3 w-3 text-muted-foreground" /></span>
                          </TooltipTrigger>
                          <TooltipContent side="right" className="max-w-xs">
                            <p className="text-xs">For evaluation, first frame only is usually sufficient - it represents the canonical product view. Use key frames if you want to evaluate multi-view matching.</p>
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    </div>
                    <ChevronDown
                      className={`h-4 w-4 transition-transform ${frameConfigOpen ? "rotate-180" : ""}`}
                    />
                  </Button>
                </CollapsibleTrigger>
                <CollapsibleContent className="mt-4">
                  <div className="space-y-4 p-4 border rounded-lg bg-muted/50">
                    <RadioGroup
                      value={frameSelection}
                      onValueChange={(v: FrameSelection | "all") => setFrameSelection(v)}
                    >
                      <div className="flex items-center space-x-2">
                        <RadioGroupItem value="first" id="eval-first" />
                        <Label htmlFor="eval-first" className="font-normal">
                          First frame only (recommended)
                        </Label>
                      </div>
                      <div className="flex items-center space-x-2">
                        <RadioGroupItem value="key_frames" id="eval-key" />
                        <Label htmlFor="eval-key" className="font-normal">
                          Key frames (4 angles)
                        </Label>
                      </div>
                      <div className="flex items-center space-x-2">
                        <RadioGroupItem value="interval" id="eval-interval" />
                        <Label htmlFor="eval-interval" className="font-normal">
                          Every N frames
                        </Label>
                      </div>
                      <div className="flex items-center space-x-2">
                        <RadioGroupItem value="all" id="eval-all" />
                        <Label htmlFor="eval-all" className="font-normal">
                          All frames
                        </Label>
                      </div>
                    </RadioGroup>

                    {frameSelection === "interval" && (
                      <div className="space-y-3">
                        <div className="flex justify-between">
                          <Label>Frame Interval</Label>
                          <span className="text-sm font-medium">{frameInterval[0]}</span>
                        </div>
                        <Slider
                          value={frameInterval}
                          onValueChange={setFrameInterval}
                          min={2}
                          max={10}
                          step={1}
                        />
                      </div>
                    )}

                    {frameSelection !== "first" && (
                      <div className="space-y-3">
                        <div className="flex justify-between">
                          <Label>Max Frames per Product</Label>
                          <span className="text-sm font-medium">{maxFrames[0]}</span>
                        </div>
                        <Slider
                          value={maxFrames}
                          onValueChange={setMaxFrames}
                          min={2}
                          max={10}
                          step={1}
                        />
                      </div>
                    )}
                  </div>
                </CollapsibleContent>
              </Collapsible>
            </CardContent>
          </Card>

          {/* Output Configuration */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Database className="h-5 w-5" />
                Output Collection
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger>
                      <Info className="h-4 w-4 text-muted-foreground" />
                    </TooltipTrigger>
                    <TooltipContent side="right" className="max-w-xs">
                      <p className="text-xs">Evaluation embeddings are stored in a separate collection for computing retrieval metrics. Name typically includes model and dataset for easy comparison across experiments.</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label>Collection Mode</Label>
                <Select
                  value={collectionMode}
                  onValueChange={(v: CollectionMode) => {
                    setCollectionMode(v);
                    if (v !== "append") {
                      setSelectedCollection("");
                    }
                  }}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="create">
                      <div className="flex flex-col items-start">
                        <span>Create New</span>
                        <span className="text-xs text-muted-foreground">Create new collection (fails if exists)</span>
                      </div>
                    </SelectItem>
                    <SelectItem value="replace">
                      <div className="flex flex-col items-start">
                        <span>Replace Existing</span>
                        <span className="text-xs text-muted-foreground">Delete and recreate collection</span>
                      </div>
                    </SelectItem>
                    <SelectItem value="append">
                      <div className="flex flex-col items-start">
                        <span>Append to Existing</span>
                        <span className="text-xs text-muted-foreground">Add vectors to existing collection</span>
                      </div>
                    </SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Existing collection dropdown for append mode */}
              {collectionMode === "append" && (
                <div className="space-y-2">
                  <Label>Select Existing Collection</Label>
                  <Select value={selectedCollection} onValueChange={setSelectedCollection}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select a collection" />
                    </SelectTrigger>
                    <SelectContent>
                      {collections?.map((col) => (
                        <SelectItem key={col.name} value={col.name}>
                          <div className="flex items-center gap-2">
                            <span>{col.name}</span>
                            <Badge variant="outline" className="text-xs">
                              {col.vectors_count} vectors
                            </Badge>
                            <Badge variant="secondary" className="text-xs">
                              {col.vector_size}d
                            </Badge>
                          </div>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  {/* Dimension validation warning */}
                  {selectedCollection && effectiveEmbeddingDim && (() => {
                    const col = collections?.find(c => c.name === selectedCollection);
                    if (col && col.vector_size !== effectiveEmbeddingDim) {
                      return (
                        <div className="flex items-center gap-2 p-2 bg-destructive/10 text-destructive rounded text-sm">
                          <AlertCircle className="h-4 w-4" />
                          <span>
                            Dimension mismatch: Collection is {col.vector_size}d but model outputs {effectiveEmbeddingDim}d
                          </span>
                        </div>
                      );
                    }
                    return null;
                  })()}
                </div>
              )}

              {/* Custom collection name for create/replace modes */}
              {collectionMode !== "append" && (
                <Collapsible open={collectionConfigOpen} onOpenChange={setCollectionConfigOpen}>
                  <CollapsibleTrigger asChild>
                    <Button variant="outline" className="w-full justify-between">
                      <span>Custom Collection Name</span>
                      <ChevronDown
                        className={`h-4 w-4 transition-transform ${collectionConfigOpen ? "rotate-180" : ""}`}
                      />
                    </Button>
                  </CollapsibleTrigger>
                  <CollapsibleContent className="mt-4">
                    <div className="space-y-2">
                      <Label>Collection Name</Label>
                      <Input
                        placeholder={defaultCollectionName}
                        value={collectionName}
                        onChange={(e) => setCollectionName(e.target.value)}
                      />
                    </div>
                  </CollapsibleContent>
                </Collapsible>
              )}

              {/* Preview */}
              {((modelSource === "base" && selectedModelId) || (modelSource === "trained" && selectedTrainingRunId)) && selectedDatasetId && (
                <div className="p-3 bg-muted rounded-lg text-sm">
                  <p className="flex items-center gap-1 text-muted-foreground mb-2">
                    <Info className="h-4 w-4" />
                    {collectionMode === "append" ? "Appending to:" : "Collection to be created:"}
                  </p>
                  <Badge variant="outline" className="font-mono text-xs">
                    {collectionMode === "append"
                      ? (selectedCollection || "Select a collection")
                      : (collectionName || defaultCollectionName)}
                  </Badge>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Start Button */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium">Ready to extract evaluation embeddings</p>
              <p className="text-sm text-muted-foreground">
                {modelSource === "base"
                  ? (selectedModel?.name || "No model selected")
                  : (selectedTrainingRun?.name
                      ? `${selectedTrainingRun.name}${selectedCheckpoint ? ` (Epoch ${selectedCheckpoint.epoch})` : ""}`
                      : "No training run selected")
                }{" "}
                + {selectedDataset?.name || "No dataset selected"} ({imageTypes.length}{" "}
                image type{imageTypes.length !== 1 ? "s" : ""})
              </p>
            </div>
            <Button
              size="lg"
              onClick={handleStartExtraction}
              disabled={startExtractionMutation.isPending || !canStartExtraction}
            >
              {startExtractionMutation.isPending ? (
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              ) : (
                <Play className="h-4 w-4 mr-2" />
              )}
              Start Evaluation Extraction
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
