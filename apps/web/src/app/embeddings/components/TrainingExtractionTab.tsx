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
import { Switch } from "@/components/ui/switch";
import { Checkbox } from "@/components/ui/checkbox";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
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
  FileDown,
  GraduationCap,
  AlertCircle,
  Info,
  CheckCircle2,
  AlertTriangle,
  X,
} from "lucide-react";
import type {
  EmbeddingModel,
  CollectionInfo,
  TrainingExtractionRequest,
  ImageType,
  FrameSelection,
  CollectionMode,
  OutputTarget,
  EmbeddingJob,
} from "@/types";

interface TrainingExtractionTabProps {
  activeModel: EmbeddingModel | null;
  models?: EmbeddingModel[];
}

export function TrainingExtractionTab({ activeModel, models }: TrainingExtractionTabProps) {
  const queryClient = useQueryClient();

  // Selected model (defaults to active model)
  const [selectedModelId, setSelectedModelId] = useState<string>(activeModel?.id || "");

  // Get the selected model object
  const selectedModel = models?.find(m => m.id === selectedModelId) || activeModel;

  // Image types selection
  const [imageTypes, setImageTypes] = useState<ImageType[]>(["synthetic", "real"]);

  // Frame selection
  const [frameSelection, setFrameSelection] = useState<FrameSelection | "all">("key_frames");
  const [frameInterval, setFrameInterval] = useState([5]);
  const [maxFrames, setMaxFrames] = useState([10]);

  // Include matched cutouts
  const [includeMatchedCutouts, setIncludeMatchedCutouts] = useState(true);

  // Output configuration
  const [outputTarget, setOutputTarget] = useState<OutputTarget>("qdrant");
  const [collectionMode, setCollectionMode] = useState<CollectionMode>("create");
  const [collectionName, setCollectionName] = useState("");
  const [selectedCollection, setSelectedCollection] = useState("");

  // Collapsible states
  const [frameConfigOpen, setFrameConfigOpen] = useState(false);
  const [outputConfigOpen, setOutputConfigOpen] = useState(false);

  // Fetch active jobs with polling
  const { data: activeJobs } = useQuery({
    queryKey: ["embedding-jobs-active"],
    queryFn: () => apiClient.getEmbeddingJobs("running"),
    refetchInterval: 3000,
  });

  // Fetch matched products stats
  const { data: matchedStats, isLoading: statsLoading } = useQuery({
    queryKey: ["matched-products-stats"],
    queryFn: () => apiClient.getMatchedProductsStats(),
  });

  // Fetch existing collections for append mode
  const { data: collections } = useQuery({
    queryKey: ["qdrant-collections"],
    queryFn: () => apiClient.getQdrantCollections(),
  });

  // Filter collections by dimension when in append mode
  const compatibleCollections = collections?.filter(
    (c: CollectionInfo) => !selectedModel || c.vector_size === selectedModel.embedding_dim
  ) || [];

  // Get selected collection info for dimension validation
  const selectedCollectionInfo = collections?.find(
    (c: CollectionInfo) => c.name === selectedCollection
  );

  // Dimension mismatch warning
  const hasDimensionMismatch = collectionMode === "append" && selectedModel &&
    selectedCollection && selectedCollectionInfo &&
    selectedCollectionInfo.vector_size !== selectedModel.embedding_dim;

  // Cancel job mutation
  const cancelJobMutation = useMutation({
    mutationFn: (jobId: string) => apiClient.cancelEmbeddingJob(jobId),
    onSuccess: () => {
      toast.success("Job cancelled successfully");
      queryClient.invalidateQueries({ queryKey: ["embedding-jobs-active"] });
      queryClient.invalidateQueries({ queryKey: ["embedding-jobs"] });
    },
    onError: (error) => {
      toast.error(`Failed to cancel job: ${error.message}`);
    },
  });

  // Start training extraction
  const startExtractionMutation = useMutation({
    mutationFn: (params: TrainingExtractionRequest) =>
      apiClient.startTrainingExtraction(params),
    onSuccess: (data) => {
      toast.success(
        `Training extraction started. ${data.total_embeddings} embeddings created.`
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
    if (!selectedModel) {
      toast.error("No model selected");
      return;
    }

    if (imageTypes.length === 0) {
      toast.error("Select at least one image type");
      return;
    }

    if (collectionMode === "append" && !selectedCollection) {
      toast.error("Please select a collection to append to");
      return;
    }

    if (hasDimensionMismatch) {
      toast.error("Selected collection dimension doesn't match model embedding dimension");
      return;
    }

    // For append mode, use selected collection name
    const collection = collectionMode === "append"
      ? selectedCollection
      : collectionName || undefined;

    const request: TrainingExtractionRequest = {
      model_id: selectedModel.id,
      image_types: imageTypes,
      frame_selection: frameSelection,
      frame_interval: frameInterval[0],
      max_frames: maxFrames[0],
      include_matched_cutouts: includeMatchedCutouts,
      output_target: outputTarget,
      collection_mode: collectionMode,
      collection_name: collection,
    };

    startExtractionMutation.mutate(request);
  };

  const modelName = selectedModel?.name.toLowerCase().replace(/\s+/g, "_") || "default";
  const defaultCollectionName = `training_${modelName}`;

  const hasMatchedProducts = (matchedStats?.total_matched_products || 0) > 0;

  return (
    <div className="space-y-6">
      {/* Description Card */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <GraduationCap className="h-5 w-5" />
            Training Extraction
          </CardTitle>
          <CardDescription>
            Extract embeddings from matched products for triplet mining and model training.
            Only products with verified matches (real cutout images) are included.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            <Label>Embedding Model</Label>
            <Select
              value={selectedModelId}
              onValueChange={setSelectedModelId}
            >
              <SelectTrigger className="w-[350px]">
                <SelectValue placeholder="Select a model" />
              </SelectTrigger>
              <SelectContent>
                {models?.map((model) => (
                  <SelectItem key={model.id} value={model.id}>
                    <div className="flex items-center gap-2">
                      <span>{model.name}</span>
                      <Badge variant="outline" className="text-xs">
                        {model.embedding_dim}d
                      </Badge>
                      {model.is_matching_active && (
                        <Badge variant="default" className="text-xs bg-green-600">
                          Active
                        </Badge>
                      )}
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {selectedModel && (
              <p className="text-xs text-muted-foreground">
                Model family: {selectedModel.model_family || selectedModel.model_type?.split("-")[0]}
                {" • "}Dimension: {selectedModel.embedding_dim}d
              </p>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Stats Card */}
      <Card className={hasMatchedProducts ? "border-green-500/50" : "border-orange-500/50"}>
        <CardContent className="pt-6">
          <div className="flex items-center gap-4">
            {hasMatchedProducts ? (
              <CheckCircle2 className="h-8 w-8 text-green-500" />
            ) : (
              <AlertCircle className="h-8 w-8 text-orange-500" />
            )}
            <div>
              <p className="text-2xl font-bold">
                {statsLoading ? (
                  <Loader2 className="h-6 w-6 animate-spin" />
                ) : (
                  matchedStats?.total_matched_products || 0
                )}
              </p>
              <p className="text-muted-foreground">
                {hasMatchedProducts
                  ? "Products with matched cutouts available for training"
                  : "No matched products yet. Match some cutouts on the Matching page first."}
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Active Jobs Warning */}
      {activeJobs && activeJobs.length > 0 && (
        <Card className="border-orange-500/50 bg-orange-500/5">
          <CardHeader className="pb-3">
            <CardTitle className="text-lg flex items-center gap-2">
              <Loader2 className="h-5 w-5 animate-spin text-orange-500" />
              Active Extraction Job
            </CardTitle>
          </CardHeader>
          <CardContent>
            {activeJobs.map((job: EmbeddingJob) => (
              <div key={job.id} className="flex items-center justify-between p-3 bg-background rounded-lg">
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <p className="text-sm font-medium">
                      Processing {job.processed_images.toLocaleString()} / {job.total_images.toLocaleString()} images
                    </p>
                    <Badge variant="secondary" className="text-xs">
                      {Math.round((job.processed_images / job.total_images) * 100)}%
                    </Badge>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Job ID: {job.id}
                  </p>
                </div>
                <Button
                  variant="destructive"
                  size="sm"
                  onClick={() => cancelJobMutation.mutate(job.id)}
                  disabled={cancelJobMutation.isPending}
                >
                  <X className="h-4 w-4 mr-2" />
                  Cancel
                </Button>
              </div>
            ))}
          </CardContent>
        </Card>
      )}

      {hasMatchedProducts && (
        <>
          {/* Configuration */}
          <div className="grid grid-cols-2 gap-6">
            {/* Left Column: Image Types */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                  Image Types
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger>
                        <Info className="h-4 w-4 text-muted-foreground" />
                      </TooltipTrigger>
                      <TooltipContent side="right" className="max-w-xs">
                        <p className="text-xs">Select which types of product images to include in training. Using multiple types helps the model learn domain-invariant features and improves generalization.</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </CardTitle>
                <CardDescription>
                  Select which image types to include in training data
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-3">
                  <div className="flex items-center space-x-2">
                    <Checkbox
                      id="synthetic"
                      checked={imageTypes.includes("synthetic")}
                      onCheckedChange={() => handleImageTypeToggle("synthetic")}
                    />
                    <div className="grid gap-1.5 leading-none">
                      <div className="flex items-center gap-1">
                        <Label htmlFor="synthetic" className="font-medium">
                          Synthetic Frames
                        </Label>
                        <TooltipProvider>
                          <Tooltip>
                            <TooltipTrigger>
                              <Info className="h-3 w-3 text-muted-foreground" />
                            </TooltipTrigger>
                            <TooltipContent side="right" className="max-w-xs">
                              <p className="text-xs">360° rotation video frames rendered from 3D product models. High quality, consistent lighting, but may differ from real-world appearance.</p>
                            </TooltipContent>
                          </Tooltip>
                        </TooltipProvider>
                      </div>
                      <p className="text-xs text-muted-foreground">
                        Product video frames (360° rotation)
                      </p>
                    </div>
                  </div>

                  <div className="flex items-center space-x-2">
                    <Checkbox
                      id="real"
                      checked={imageTypes.includes("real")}
                      onCheckedChange={() => handleImageTypeToggle("real")}
                    />
                    <div className="grid gap-1.5 leading-none">
                      <div className="flex items-center gap-1">
                        <Label htmlFor="real" className="font-medium">
                          Real Images
                        </Label>
                        <TooltipProvider>
                          <Tooltip>
                            <TooltipTrigger>
                              <Info className="h-3 w-3 text-muted-foreground" />
                            </TooltipTrigger>
                            <TooltipContent side="right" className="max-w-xs">
                              <p className="text-xs">Real photographs of products taken in store environments. Variable quality and lighting, but represents actual deployment conditions.</p>
                            </TooltipContent>
                          </Tooltip>
                        </TooltipProvider>
                      </div>
                      <p className="text-xs text-muted-foreground">
                        Manually added real product photos
                      </p>
                    </div>
                  </div>

                  <div className="flex items-center space-x-2">
                    <Checkbox
                      id="augmented"
                      checked={imageTypes.includes("augmented")}
                      onCheckedChange={() => handleImageTypeToggle("augmented")}
                    />
                    <div className="grid gap-1.5 leading-none">
                      <div className="flex items-center gap-1">
                        <Label htmlFor="augmented" className="font-medium">
                          Augmented Images
                        </Label>
                        <TooltipProvider>
                          <Tooltip>
                            <TooltipTrigger>
                              <Info className="h-3 w-3 text-muted-foreground" />
                            </TooltipTrigger>
                            <TooltipContent side="right" className="max-w-xs">
                              <p className="text-xs">Synthetically augmented versions with random transforms (rotation, crop, color jitter, blur). Increases training variety and improves robustness.</p>
                            </TooltipContent>
                          </Tooltip>
                        </TooltipProvider>
                      </div>
                      <p className="text-xs text-muted-foreground">
                        Synthetically augmented images
                      </p>
                    </div>
                  </div>
                </div>

                {/* Matched Cutouts Toggle */}
                <div className="pt-4 border-t">
                  <div className="flex items-center justify-between">
                    <div className="space-y-1 flex items-center gap-2">
                      <div>
                        <Label>Include Matched Cutouts</Label>
                        <p className="text-xs text-muted-foreground">
                          Add cutout images as positive pairs
                        </p>
                      </div>
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger>
                            <Info className="h-3 w-3 text-muted-foreground" />
                          </TooltipTrigger>
                          <TooltipContent side="right" className="max-w-xs">
                            <p className="text-xs">Include verified cutout matches as additional positive samples. Critical for triplet mining - provides real-world anchor-positive pairs that bridge the synthetic-real domain gap.</p>
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    </div>
                    <Switch
                      checked={includeMatchedCutouts}
                      onCheckedChange={setIncludeMatchedCutouts}
                    />
                  </div>
                </div>

                {/* Frame Selection */}
                <Collapsible open={frameConfigOpen} onOpenChange={setFrameConfigOpen}>
                  <CollapsibleTrigger asChild>
                    <Button variant="outline" className="w-full justify-between mt-4">
                      <div className="flex items-center gap-2">
                        <span>Frame Selection Settings</span>
                        <TooltipProvider>
                          <Tooltip>
                            <TooltipTrigger onClick={(e) => e.stopPropagation()}>
                              <Info className="h-3 w-3 text-muted-foreground" />
                            </TooltipTrigger>
                            <TooltipContent side="right" className="max-w-xs">
                              <p className="text-xs">Configure how many frames to extract from 360° rotation videos. Key frames (4 angles) recommended for training - provides view diversity without excessive redundancy.</p>
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
                          <RadioGroupItem value="first" id="train-first" />
                          <Label htmlFor="train-first" className="font-normal">
                            First frame only
                          </Label>
                        </div>
                        <div className="flex items-center space-x-2">
                          <RadioGroupItem value="key_frames" id="train-key" />
                          <Label htmlFor="train-key" className="font-normal">
                            Key frames (recommended for training)
                          </Label>
                        </div>
                        <div className="flex items-center space-x-2">
                          <RadioGroupItem value="interval" id="train-interval" />
                          <Label htmlFor="train-interval" className="font-normal">
                            Every N frames
                          </Label>
                        </div>
                        <div className="flex items-center space-x-2">
                          <RadioGroupItem value="all" id="train-all" />
                          <Label htmlFor="train-all" className="font-normal">
                            All frames
                          </Label>
                        </div>
                      </RadioGroup>

                      {frameSelection === "interval" && (
                        <div className="space-y-3">
                          <div className="flex justify-between">
                            <Label>Frame Interval</Label>
                            <span className="text-sm font-medium">{frameInterval[0]} frames</span>
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

            {/* Right Column: Output Configuration */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                  Output Configuration
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger>
                        <Info className="h-4 w-4 text-muted-foreground" />
                      </TooltipTrigger>
                      <TooltipContent side="right" className="max-w-xs">
                        <p className="text-xs">Configure where to store extracted training embeddings. Qdrant is recommended for fast similarity search during triplet mining and training.</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </CardTitle>
                <CardDescription>Choose where to store extracted embeddings</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label>Output Target</Label>
                  <RadioGroup
                    value={outputTarget}
                    onValueChange={(v: OutputTarget) => setOutputTarget(v)}
                    className="grid grid-cols-2 gap-4"
                  >
                    <div>
                      <RadioGroupItem
                        value="qdrant"
                        id="qdrant"
                        className="peer sr-only"
                      />
                      <Label
                        htmlFor="qdrant"
                        className="flex flex-col items-center justify-between rounded-md border-2 border-muted bg-popover p-4 hover:bg-accent hover:text-accent-foreground peer-data-[state=checked]:border-primary [&:has([data-state=checked])]:border-primary cursor-pointer"
                      >
                        <Database className="mb-3 h-6 w-6" />
                        <span className="text-sm font-medium">Qdrant</span>
                        <span className="text-xs text-muted-foreground">
                          Vector database
                        </span>
                      </Label>
                    </div>
                    <div>
                      <RadioGroupItem
                        value="file"
                        id="file"
                        className="peer sr-only"
                        disabled
                      />
                      <Label
                        htmlFor="file"
                        className="flex flex-col items-center justify-between rounded-md border-2 border-muted bg-popover p-4 hover:bg-accent hover:text-accent-foreground peer-data-[state=checked]:border-primary [&:has([data-state=checked])]:border-primary cursor-pointer opacity-50"
                      >
                        <FileDown className="mb-3 h-6 w-6" />
                        <span className="text-sm font-medium">File Export</span>
                        <span className="text-xs text-muted-foreground">
                          Coming soon
                        </span>
                      </Label>
                    </div>
                  </RadioGroup>
                </div>

                {outputTarget === "qdrant" && (
                  <>
                    <div className="space-y-2">
                      <Label>Collection Mode</Label>
                      <Select
                        value={collectionMode}
                        onValueChange={(v: CollectionMode) => setCollectionMode(v)}
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="create">
                            <div className="flex flex-col">
                              <span>Create New</span>
                              <span className="text-xs text-muted-foreground">Create new collection (fails if exists)</span>
                            </div>
                          </SelectItem>
                          <SelectItem value="replace">
                            <div className="flex flex-col">
                              <span>Replace Existing</span>
                              <span className="text-xs text-muted-foreground">Delete and recreate collection</span>
                            </div>
                          </SelectItem>
                          <SelectItem value="append">
                            <div className="flex flex-col">
                              <span>Append to Existing</span>
                              <span className="text-xs text-muted-foreground">Add vectors to existing collection</span>
                            </div>
                          </SelectItem>
                        </SelectContent>
                      </Select>
                    </div>

                    {/* Append Mode: Select existing collection */}
                    {collectionMode === "append" && (
                      <div className="space-y-3 p-3 border rounded-lg bg-muted/30">
                        <div className="space-y-2">
                          <Label>Select Collection</Label>
                          <Select
                            value={selectedCollection}
                            onValueChange={setSelectedCollection}
                          >
                            <SelectTrigger>
                              <SelectValue placeholder="Select collection to append to" />
                            </SelectTrigger>
                            <SelectContent>
                              {compatibleCollections.length === 0 ? (
                                <SelectItem value="_none" disabled>
                                  No compatible collections found
                                </SelectItem>
                              ) : (
                                compatibleCollections.map((c: CollectionInfo) => (
                                  <SelectItem key={c.name} value={c.name}>
                                    <div className="flex items-center gap-2">
                                      <span>{c.name}</span>
                                      <Badge variant="outline" className="text-xs">
                                        {c.vector_size}d
                                      </Badge>
                                      <span className="text-xs text-muted-foreground">
                                        ({c.vectors_count} vectors)
                                      </span>
                                    </div>
                                  </SelectItem>
                                ))
                              )}
                            </SelectContent>
                          </Select>
                        </div>

                        {/* Dimension mismatch warning */}
                        {hasDimensionMismatch && (
                          <div className="flex items-center gap-2 p-2 bg-orange-500/10 border border-orange-500/30 rounded text-sm text-orange-600">
                            <AlertTriangle className="h-4 w-4" />
                            <span>Vector dimension mismatch! Model: {selectedModel?.embedding_dim}d</span>
                          </div>
                        )}
                      </div>
                    )}

                    {/* Create/Replace Mode: Custom collection name */}
                    {collectionMode !== "append" && (
                      <Collapsible open={outputConfigOpen} onOpenChange={setOutputConfigOpen}>
                        <CollapsibleTrigger asChild>
                          <Button variant="outline" className="w-full justify-between">
                            <span>Custom Collection Name</span>
                            <ChevronDown
                              className={`h-4 w-4 transition-transform ${outputConfigOpen ? "rotate-180" : ""}`}
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
                    <div className="p-3 bg-muted rounded-lg text-sm">
                      <p className="flex items-center gap-1 text-muted-foreground mb-2">
                        <Info className="h-4 w-4" />
                        {collectionMode === "create" && "Collection to be created:"}
                        {collectionMode === "replace" && "Collection to be replaced:"}
                        {collectionMode === "append" && "Collection to append to:"}
                      </p>
                      <Badge variant="outline" className="font-mono text-xs">
                        {collectionMode === "append"
                          ? (selectedCollection || "Select a collection")
                          : (collectionName || defaultCollectionName)}
                      </Badge>
                    </div>
                  </>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Start Button */}
          <Card>
            <CardContent className="pt-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium">Ready to extract training embeddings</p>
                  <p className="text-sm text-muted-foreground">
                    {matchedStats?.total_matched_products || 0} matched products,{" "}
                    {imageTypes.length} image type(s) selected
                  </p>
                </div>
                <Button
                  size="lg"
                  onClick={handleStartExtraction}
                  disabled={
                    startExtractionMutation.isPending ||
                    !selectedModel ||
                    imageTypes.length === 0
                  }
                >
                  {startExtractionMutation.isPending ? (
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <Play className="h-4 w-4 mr-2" />
                  )}
                  Start Training Extraction
                </Button>
              </div>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}
