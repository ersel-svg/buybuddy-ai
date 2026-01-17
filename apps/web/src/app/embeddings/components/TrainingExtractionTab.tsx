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
  Play,
  Loader2,
  ChevronDown,
  Database,
  FileDown,
  GraduationCap,
  AlertCircle,
  Info,
  CheckCircle2,
} from "lucide-react";
import type {
  EmbeddingModel,
  TrainingExtractionRequest,
  ImageType,
  FrameSelection,
  CollectionMode,
  OutputTarget,
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

  // Collapsible states
  const [frameConfigOpen, setFrameConfigOpen] = useState(false);
  const [outputConfigOpen, setOutputConfigOpen] = useState(false);

  // Fetch matched products stats
  const { data: matchedStats, isLoading: statsLoading } = useQuery({
    queryKey: ["matched-products-stats"],
    queryFn: () => apiClient.getMatchedProductsStats(),
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

    const request: TrainingExtractionRequest = {
      model_id: selectedModel.id,
      image_types: imageTypes,
      frame_selection: frameSelection,
      frame_interval: frameInterval[0],
      max_frames: maxFrames[0],
      include_matched_cutouts: includeMatchedCutouts,
      output_target: outputTarget,
      collection_mode: collectionMode,
      collection_name: collectionName || undefined,
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

      {hasMatchedProducts && (
        <>
          {/* Configuration */}
          <div className="grid grid-cols-2 gap-6">
            {/* Left Column: Image Types */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Image Types</CardTitle>
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
                      <Label htmlFor="synthetic" className="font-medium">
                        Synthetic Frames
                      </Label>
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
                      <Label htmlFor="real" className="font-medium">
                        Real Images
                      </Label>
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
                      <Label htmlFor="augmented" className="font-medium">
                        Augmented Images
                      </Label>
                      <p className="text-xs text-muted-foreground">
                        Synthetically augmented images
                      </p>
                    </div>
                  </div>
                </div>

                {/* Matched Cutouts Toggle */}
                <div className="pt-4 border-t">
                  <div className="flex items-center justify-between">
                    <div className="space-y-1">
                      <Label>Include Matched Cutouts</Label>
                      <p className="text-xs text-muted-foreground">
                        Add cutout images as positive pairs
                      </p>
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
                      <span>Frame Selection Settings</span>
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
                <CardTitle className="text-lg">Output Configuration</CardTitle>
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
                          <SelectItem value="create">Create New (replace existing)</SelectItem>
                          <SelectItem value="append">Append to Existing</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>

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

                    {/* Preview */}
                    <div className="p-3 bg-muted rounded-lg text-sm">
                      <p className="flex items-center gap-1 text-muted-foreground mb-2">
                        <Info className="h-4 w-4" />
                        Collection to be {collectionMode === "create" ? "created" : "updated"}:
                      </p>
                      <Badge variant="outline" className="font-mono text-xs">
                        {collectionName || defaultCollectionName}
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
