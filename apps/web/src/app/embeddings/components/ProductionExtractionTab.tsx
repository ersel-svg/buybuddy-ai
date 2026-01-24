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
  Rocket,
  AlertCircle,
  Info,
  GraduationCap,
  Layers,
  CheckCircle2,
} from "lucide-react";
import type {
  EmbeddingModel,
  Dataset,
  ProductionExtractionRequest,
  ImageType,
  FrameSelection,
  TrainingRun,
  TrainingCheckpoint,
} from "@/types";

interface ProductionExtractionTabProps {
  models: EmbeddingModel[] | undefined;
}

export function ProductionExtractionTab({ models }: ProductionExtractionTabProps) {
  const queryClient = useQueryClient();

  // Model source: base model or trained model
  const [modelSource, setModelSource] = useState<"base" | "trained">("trained");

  // Base model selection
  const [selectedModelId, setSelectedModelId] = useState<string>("");

  // Trained model selection
  const [selectedTrainingRunId, setSelectedTrainingRunId] = useState<string>("");
  const [selectedCheckpointId, setSelectedCheckpointId] = useState<string>("");

  // Product source
  const [productSource, setProductSource] = useState<"all" | "dataset" | "selected">("all");
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>("");

  // Image types selection - ALL by default for production
  const [imageTypes, setImageTypes] = useState<ImageType[]>(["synthetic", "real", "augmented"]);

  // Frame selection - key_frames by default for better coverage
  const [frameSelection, setFrameSelection] = useState<FrameSelection | "all">("key_frames");
  const [frameInterval, setFrameInterval] = useState([5]);
  const [maxFrames, setMaxFrames] = useState([10]);

  // Collection config
  const [collectionMode, setCollectionMode] = useState<"create" | "replace">("create");
  const [collectionName, setCollectionName] = useState("");

  // Collapsible states
  const [frameConfigOpen, setFrameConfigOpen] = useState(false);
  const [collectionConfigOpen, setCollectionConfigOpen] = useState(false);

  // Fetch datasets
  const { data: datasets, isLoading: datasetsLoading } = useQuery({
    queryKey: ["datasets"],
    queryFn: () => apiClient.getDatasets(),
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

  // Get selected dataset
  const selectedDataset = datasets?.find((d) => d.id === selectedDatasetId);

  // Get selected base model
  const selectedModel = models?.find((m) => m.id === selectedModelId);

  // Get selected training run
  const selectedTrainingRun = trainingRuns.find((r: TrainingRun) => r.id === selectedTrainingRunId);

  // Get selected checkpoint
  const selectedCheckpoint = checkpoints?.find((c: TrainingCheckpoint) => c.id === selectedCheckpointId);

  // Start production extraction
  const startExtractionMutation = useMutation({
    mutationFn: (params: ProductionExtractionRequest) =>
      apiClient.startProductionExtraction(params),
    onSuccess: (data) => {
      toast.success(
        `Production extraction completed! ${data.total_embeddings} embeddings in "${data.collection_name}".`
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
        toast.error("Select a base model");
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

    if (productSource === "dataset" && !selectedDatasetId) {
      toast.error("Select a dataset");
      return;
    }

    if (imageTypes.length === 0) {
      toast.error("Select at least one image type");
      return;
    }

    const request: ProductionExtractionRequest = {
      model_id: modelSource === "base" ? selectedModelId : undefined,
      training_run_id: modelSource === "trained" ? selectedTrainingRunId : undefined,
      checkpoint_id: modelSource === "trained" ? selectedCheckpointId : undefined,
      product_source: productSource,
      product_dataset_id: productSource === "dataset" ? selectedDatasetId : undefined,
      image_types: imageTypes,
      frame_selection: frameSelection,
      frame_interval: frameInterval[0],
      max_frames: maxFrames[0],
      collection_mode: collectionMode,
      collection_name: collectionName || undefined,
    };

    startExtractionMutation.mutate(request);
  };

  const modelName = modelSource === "base"
    ? (selectedModel?.name.toLowerCase().replace(/\s+/g, "_") || "model")
    : (selectedTrainingRun?.name.toLowerCase().replace(/\s+/g, "_") || "trained");
  const checkpointSuffix = selectedCheckpoint ? `_ep${selectedCheckpoint.epoch}` : "";
  const defaultCollectionName = `production_${modelName}${checkpointSuffix}`;

  const canStartExtraction = modelSource === "base"
    ? (selectedModelId && imageTypes.length > 0)
    : (selectedTrainingRunId && selectedCheckpointId && imageTypes.length > 0);

  return (
    <div className="space-y-6">
      {/* Description Card */}
      <Card className="border-green-500/30 bg-green-500/5">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Rocket className="h-5 w-5 text-green-500" />
            Production Extraction
            <Badge variant="secondary" className="ml-2">SOTA</Badge>
          </CardTitle>
          <CardDescription>
            Create a production-ready embedding collection with <strong>ALL image types</strong> (synthetic, real, augmented)
            for similarity search / inference. This multi-view + augmentation approach maximizes recall.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-start gap-2 p-3 bg-green-500/10 rounded-lg text-sm">
            <CheckCircle2 className="h-5 w-5 text-green-500 mt-0.5 shrink-0" />
            <div>
              <p className="font-medium text-green-700 dark:text-green-300">Why include all image types?</p>
              <ul className="mt-1 text-muted-foreground list-disc list-inside space-y-1">
                <li>Query images can match any angle or variation</li>
                <li>Augmented versions improve robustness to lighting/background changes</li>
                <li>Real + Synthetic coverage closes the domain gap</li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Configuration */}
      <div className="grid grid-cols-2 gap-6">
        {/* Left Column: Model Selection */}
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
                      <p className="text-xs">For production, use a fine-tuned model trained on your data for best results. Base models work but trained models have higher recall.</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </CardTitle>
              <CardDescription>
                Select a base model or fine-tuned model
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
                              <Badge variant="outline" className="text-xs">
                                {model.embedding_dim}d
                              </Badge>
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
                                    <Badge variant="default" className="text-xs bg-green-600">
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
                          <p className="font-medium">{selectedTrainingRun.training_config?.embedding_dim || 512}d</p>
                        </div>
                        <div>
                          <p className="text-muted-foreground">Best R@1</p>
                          <p className="font-medium">
                            {selectedTrainingRun.best_val_recall_at_1
                              ? `${(selectedTrainingRun.best_val_recall_at_1 * 100).toFixed(1)}%`
                              : "N/A"}
                          </p>
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

          {/* Product Source */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                Product Source
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger>
                      <Info className="h-4 w-4 text-muted-foreground" />
                    </TooltipTrigger>
                    <TooltipContent side="right" className="max-w-xs">
                      <p className="text-xs">For production, typically use "All Products" to index your entire catalog. Use "Dataset" to index a specific subset.</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </CardTitle>
              <CardDescription>Select which products to index</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label>Source</Label>
                <Select
                  value={productSource}
                  onValueChange={(v: "all" | "dataset" | "selected") => setProductSource(v)}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">
                      <div className="flex flex-col items-start">
                        <span>All Products</span>
                        <span className="text-xs text-muted-foreground">Index entire product catalog</span>
                      </div>
                    </SelectItem>
                    <SelectItem value="dataset">
                      <div className="flex flex-col items-start">
                        <span>From Dataset</span>
                        <span className="text-xs text-muted-foreground">Index products in a specific dataset</span>
                      </div>
                    </SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {productSource === "dataset" && (
                <div className="space-y-2">
                  <Label>Dataset</Label>
                  {datasetsLoading ? (
                    <div className="flex items-center gap-2 p-2">
                      <Loader2 className="h-4 w-4 animate-spin" />
                      <span className="text-sm text-muted-foreground">Loading datasets...</span>
                    </div>
                  ) : (
                    <Select value={selectedDatasetId} onValueChange={setSelectedDatasetId}>
                      <SelectTrigger>
                        <SelectValue placeholder="Select a dataset" />
                      </SelectTrigger>
                      <SelectContent>
                        {datasets?.map((dataset: Dataset) => (
                          <SelectItem key={dataset.id} value={dataset.id}>
                            {dataset.name} ({dataset.product_count} products)
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  )}
                </div>
              )}

              {productSource === "dataset" && selectedDataset && (
                <div className="p-3 bg-muted rounded-lg text-sm">
                  <p className="text-muted-foreground">Products: <span className="font-medium text-foreground">{selectedDataset.product_count}</span></p>
                </div>
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
                Image Types
                <Badge variant="default" className="bg-green-600">All Recommended</Badge>
              </CardTitle>
              <CardDescription>Select image types to include in production index</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-3">
                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="prod-synthetic"
                    checked={imageTypes.includes("synthetic")}
                    onCheckedChange={() => handleImageTypeToggle("synthetic")}
                  />
                  <div className="grid gap-1.5 leading-none">
                    <Label htmlFor="prod-synthetic" className="font-medium">
                      Synthetic Frames
                    </Label>
                    <p className="text-xs text-muted-foreground">
                      360° video frames - multiple angles
                    </p>
                  </div>
                </div>

                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="prod-real"
                    checked={imageTypes.includes("real")}
                    onCheckedChange={() => handleImageTypeToggle("real")}
                  />
                  <div className="grid gap-1.5 leading-none">
                    <Label htmlFor="prod-real" className="font-medium">
                      Real Images (Cutouts)
                    </Label>
                    <p className="text-xs text-muted-foreground">
                      Matched shelf photos - real-world conditions
                    </p>
                  </div>
                </div>

                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="prod-augmented"
                    checked={imageTypes.includes("augmented")}
                    onCheckedChange={() => handleImageTypeToggle("augmented")}
                  />
                  <div className="grid gap-1.5 leading-none">
                    <Label htmlFor="prod-augmented" className="font-medium">
                      Augmented Images
                    </Label>
                    <p className="text-xs text-muted-foreground">
                      Light/background variations - robustness
                    </p>
                  </div>
                </div>
              </div>

              {imageTypes.length < 3 && (
                <div className="flex items-center gap-2 p-2 bg-yellow-500/10 border border-yellow-500/30 rounded text-sm text-yellow-600 dark:text-yellow-400">
                  <AlertCircle className="h-4 w-4" />
                  <span>Selecting all image types maximizes recall</span>
                </div>
              )}

              {/* Frame Selection */}
              <Collapsible open={frameConfigOpen} onOpenChange={setFrameConfigOpen}>
                <CollapsibleTrigger asChild>
                  <Button variant="outline" className="w-full justify-between mt-4">
                    <div className="flex items-center gap-2">
                      <span>Frame Selection</span>
                      <Badge variant="secondary" className="text-xs">{frameSelection}</Badge>
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
                        <RadioGroupItem value="first" id="prod-first" />
                        <Label htmlFor="prod-first" className="font-normal">
                          First frame only
                        </Label>
                      </div>
                      <div className="flex items-center space-x-2">
                        <RadioGroupItem value="key_frames" id="prod-key" />
                        <Label htmlFor="prod-key" className="font-normal">
                          Key frames (4 angles) - Recommended
                        </Label>
                      </div>
                      <div className="flex items-center space-x-2">
                        <RadioGroupItem value="interval" id="prod-interval" />
                        <Label htmlFor="prod-interval" className="font-normal">
                          Every N frames
                        </Label>
                      </div>
                      <div className="flex items-center space-x-2">
                        <RadioGroupItem value="all" id="prod-all" />
                        <Label htmlFor="prod-all" className="font-normal">
                          All frames (max coverage)
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
                          <Label>Max Frames per Type</Label>
                          <span className="text-sm font-medium">{maxFrames[0]}</span>
                        </div>
                        <Slider
                          value={maxFrames}
                          onValueChange={setMaxFrames}
                          min={2}
                          max={20}
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
                Production Collection
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label>Collection Mode</Label>
                <Select
                  value={collectionMode}
                  onValueChange={(v: "create" | "replace") => setCollectionMode(v)}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="create">
                      <div className="flex flex-col items-start">
                        <span>Create New</span>
                        <span className="text-xs text-muted-foreground">Fails if collection exists</span>
                      </div>
                    </SelectItem>
                    <SelectItem value="replace">
                      <div className="flex flex-col items-start">
                        <span>Replace Existing</span>
                        <span className="text-xs text-muted-foreground">Delete and recreate</span>
                      </div>
                    </SelectItem>
                  </SelectContent>
                </Select>
              </div>

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

              {/* Preview */}
              <div className="p-3 bg-muted rounded-lg text-sm">
                <p className="flex items-center gap-1 text-muted-foreground mb-2">
                  <Info className="h-4 w-4" />
                  Collection to be {collectionMode === "replace" ? "replaced" : "created"}:
                </p>
                <Badge variant="outline" className="font-mono text-xs">
                  {collectionName || defaultCollectionName}
                </Badge>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Start Button */}
      <Card className="border-green-500/30">
        <CardContent className="pt-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium">Ready to create production collection</p>
              <p className="text-sm text-muted-foreground">
                {modelSource === "base"
                  ? (selectedModel?.name || "No model selected")
                  : (selectedTrainingRun?.name
                      ? `${selectedTrainingRun.name}${selectedCheckpoint ? ` (Epoch ${selectedCheckpoint.epoch})` : ""}`
                      : "No training run selected")
                }
                {" • "}
                {productSource === "all" ? "All products" : (selectedDataset?.name || "No dataset")}
                {" • "}
                {imageTypes.length} image type{imageTypes.length !== 1 ? "s" : ""}
              </p>
            </div>
            <Button
              size="lg"
              onClick={handleStartExtraction}
              disabled={startExtractionMutation.isPending || !canStartExtraction}
              className="bg-green-600 hover:bg-green-700"
            >
              {startExtractionMutation.isPending ? (
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              ) : (
                <Rocket className="h-4 w-4 mr-2" />
              )}
              Start Production Extraction
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
