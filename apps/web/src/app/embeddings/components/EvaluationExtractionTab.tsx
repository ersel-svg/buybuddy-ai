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
import {
  Play,
  Loader2,
  ChevronDown,
  Database,
  FlaskConical,
  AlertCircle,
  Info,
} from "lucide-react";
import type {
  EmbeddingModel,
  Dataset,
  EvaluationExtractionRequest,
  ImageType,
  FrameSelection,
  CollectionMode,
} from "@/types";

interface EvaluationExtractionTabProps {
  models: EmbeddingModel[] | undefined;
}

export function EvaluationExtractionTab({ models }: EvaluationExtractionTabProps) {
  const queryClient = useQueryClient();

  // Model selection (specific trained model for evaluation)
  const [selectedModelId, setSelectedModelId] = useState<string>("");

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

  // Selected collection for append mode
  const [selectedCollection, setSelectedCollection] = useState<string>("");

  // Get selected dataset
  const selectedDataset = datasets?.find((d) => d.id === selectedDatasetId);

  // Get selected model
  const selectedModel = models?.find((m) => m.id === selectedModelId);

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
    if (!selectedModelId) {
      toast.error("Select a model to evaluate");
      return;
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
      model_id: selectedModelId,
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
  const modelName = selectedModel?.name.toLowerCase().replace(/\s+/g, "_") || "model";
  const defaultCollectionName = `eval_${datasetName}_${modelName}`;

  const canStartExtraction =
    selectedModelId && selectedDatasetId && imageTypes.length > 0;

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
              <CardTitle className="text-lg">Model Selection</CardTitle>
              <CardDescription>
                Select the trained model to evaluate
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
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
            </CardContent>
          </Card>

          {/* Dataset Selection */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Dataset Selection</CardTitle>
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
              <CardTitle className="text-lg">Image Configuration</CardTitle>
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
                    <Label htmlFor="eval-synthetic" className="font-medium">
                      Synthetic Frames
                    </Label>
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
                    <Label htmlFor="eval-real" className="font-medium">
                      Real Images
                    </Label>
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
                    <Label htmlFor="eval-augmented" className="font-medium">
                      Augmented Images
                    </Label>
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
                    <span>Frame Selection</span>
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
                  {selectedCollection && selectedModel && (() => {
                    const col = collections?.find(c => c.name === selectedCollection);
                    if (col && col.vector_size !== selectedModel.embedding_dim) {
                      return (
                        <div className="flex items-center gap-2 p-2 bg-destructive/10 text-destructive rounded text-sm">
                          <AlertCircle className="h-4 w-4" />
                          <span>
                            Dimension mismatch: Collection is {col.vector_size}d but model outputs {selectedModel.embedding_dim}d
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
              {selectedModelId && selectedDatasetId && (
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
                {selectedModel?.name || "No model selected"} +{" "}
                {selectedDataset?.name || "No dataset selected"} ({imageTypes.length}{" "}
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
