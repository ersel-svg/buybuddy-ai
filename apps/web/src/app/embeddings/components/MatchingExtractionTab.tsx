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
  Settings2,
  Database,
  Package,
  ImageIcon,
  Info,
  AlertTriangle,
} from "lucide-react";
import type {
  EmbeddingModel,
  Dataset,
  CollectionInfo,
  MatchingExtractionRequest,
  ProductSource,
  FrameSelection,
  CollectionMode,
} from "@/types";

interface MatchingExtractionTabProps {
  activeModel: EmbeddingModel | null;
  models?: EmbeddingModel[];
}

export function MatchingExtractionTab({ activeModel, models }: MatchingExtractionTabProps) {
  const queryClient = useQueryClient();

  // Selected model (defaults to active model)
  const [selectedModelId, setSelectedModelId] = useState<string>(activeModel?.id || "");

  // Get the selected model object
  const selectedModel = models?.find(m => m.id === selectedModelId) || activeModel;

  // Product source config
  const [productSource, setProductSource] = useState<ProductSource>("all");
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>("");

  // Include cutouts
  const [includeCutouts, setIncludeCutouts] = useState(true);
  const [cutoutFilterHasUpc, setCutoutFilterHasUpc] = useState(false);

  // Collection config
  const [collectionMode, setCollectionMode] = useState<CollectionMode>("create");
  const [productCollectionName, setProductCollectionName] = useState("");
  const [cutoutCollectionName, setCutoutCollectionName] = useState("");
  const [selectedProductCollection, setSelectedProductCollection] = useState("");
  const [selectedCutoutCollection, setSelectedCutoutCollection] = useState("");

  // Frame selection
  const [frameSelection, setFrameSelection] = useState<FrameSelection | "all">("first");
  const [frameInterval, setFrameInterval] = useState([5]);
  const [maxFrames, setMaxFrames] = useState([10]);

  // Collapsible states
  const [productConfigOpen, setProductConfigOpen] = useState(false);
  const [collectionConfigOpen, setCollectionConfigOpen] = useState(false);

  // Fetch datasets for source selection
  const { data: datasets } = useQuery({
    queryKey: ["datasets"],
    queryFn: () => apiClient.getDatasets(),
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
  const selectedProductCollectionInfo = collections?.find(
    (c: CollectionInfo) => c.name === selectedProductCollection
  );
  const selectedCutoutCollectionInfo = collections?.find(
    (c: CollectionInfo) => c.name === selectedCutoutCollection
  );

  // Dimension mismatch warning
  const hasDimensionMismatch = collectionMode === "append" && selectedModel && (
    (selectedProductCollection && selectedProductCollectionInfo &&
     selectedProductCollectionInfo.vector_size !== selectedModel.embedding_dim) ||
    (selectedCutoutCollection && selectedCutoutCollectionInfo &&
     selectedCutoutCollectionInfo.vector_size !== selectedModel.embedding_dim)
  );

  // Start matching extraction
  const startExtractionMutation = useMutation({
    mutationFn: (params: MatchingExtractionRequest) =>
      apiClient.startMatchingExtraction(params),
    onSuccess: (data) => {
      toast.success(
        `Matching extraction started. ${data.total_embeddings} embeddings created.`
      );
      queryClient.invalidateQueries({ queryKey: ["embedding-jobs"] });
      queryClient.invalidateQueries({ queryKey: ["qdrant-collections"] });
    },
    onError: (error) => {
      toast.error(`Failed to start extraction: ${error.message}`);
    },
  });

  const handleStartExtraction = () => {
    if (!selectedModel) {
      toast.error("No model selected");
      return;
    }

    if (collectionMode === "append" && !selectedProductCollection) {
      toast.error("Please select a collection to append to");
      return;
    }

    if (hasDimensionMismatch) {
      toast.error("Selected collection dimension doesn't match model embedding dimension");
      return;
    }

    // For append mode, use selected collection names
    const productCollection = collectionMode === "append"
      ? selectedProductCollection
      : productCollectionName || undefined;
    const cutoutCollection = collectionMode === "append"
      ? selectedCutoutCollection
      : cutoutCollectionName || undefined;

    const request: MatchingExtractionRequest = {
      model_id: selectedModel.id,
      product_source: productSource,
      product_dataset_id: productSource === "dataset" ? selectedDatasetId : undefined,
      include_cutouts: includeCutouts,
      cutout_filter_has_upc: cutoutFilterHasUpc,
      collection_mode: collectionMode,
      product_collection_name: productCollection,
      cutout_collection_name: cutoutCollection,
      frame_selection: frameSelection,
      frame_interval: frameInterval[0],
      max_frames: maxFrames[0],
    };

    startExtractionMutation.mutate(request);
  };

  const modelName = selectedModel?.name.toLowerCase().replace(/\s+/g, "_") || "default";
  const defaultProductCollection = `products_${modelName}`;
  const defaultCutoutCollection = `cutouts_${modelName}`;

  return (
    <div className="space-y-6">
      {/* Description Card */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Package className="h-5 w-5" />
            Matching Extraction
          </CardTitle>
          <CardDescription>
            Extract embeddings for products and cutouts to enable similarity-based matching.
            Extracted embeddings are stored in Qdrant collections and used by the Matching page.
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

      {/* Configuration */}
      <div className="grid grid-cols-2 gap-6">
        {/* Left Column: Source Configuration */}
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
                    <p className="text-xs">Choose which products to extract embeddings for. This determines the product catalog that will be searchable in the Matching page.</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label>Select Products From</Label>
              <Select
                value={productSource}
                onValueChange={(v: ProductSource) => setProductSource(v)}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">
                    <div className="flex flex-col items-start">
                      <span>All Products</span>
                      <span className="text-xs text-muted-foreground">Extract embeddings for entire product catalog</span>
                    </div>
                  </SelectItem>
                  <SelectItem value="dataset">
                    <div className="flex flex-col items-start">
                      <span>From Dataset</span>
                      <span className="text-xs text-muted-foreground">Only products in a specific dataset</span>
                    </div>
                  </SelectItem>
                  <SelectItem value="matched">
                    <div className="flex flex-col items-start">
                      <span>Matched Products Only</span>
                      <span className="text-xs text-muted-foreground">Products with verified cutout matches</span>
                    </div>
                  </SelectItem>
                  <SelectItem value="new">
                    <div className="flex flex-col items-start">
                      <span>New Products</span>
                      <span className="text-xs text-muted-foreground">Only products without existing embeddings</span>
                    </div>
                  </SelectItem>
                </SelectContent>
              </Select>
            </div>

            {productSource === "dataset" && (
              <div className="space-y-2">
                <Label>Dataset</Label>
                <Select
                  value={selectedDatasetId}
                  onValueChange={setSelectedDatasetId}
                >
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
              </div>
            )}

            {/* Product Frame Selection */}
            <Collapsible open={productConfigOpen} onOpenChange={setProductConfigOpen}>
              <CollapsibleTrigger asChild>
                <Button variant="outline" className="w-full justify-between">
                  <div className="flex items-center gap-2">
                    <Settings2 className="h-4 w-4" />
                    Frame Selection
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger onClick={(e) => e.stopPropagation()}>
                          <Info className="h-3 w-3 text-muted-foreground" />
                        </TooltipTrigger>
                        <TooltipContent side="right" className="max-w-xs">
                          <p className="text-xs">Products have 360° rotation videos with multiple frames. Choose how many frames to extract embeddings from. More frames = better multi-angle matching but higher storage/computation cost.</p>
                        </TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                  </div>
                  <ChevronDown
                    className={`h-4 w-4 transition-transform ${productConfigOpen ? "rotate-180" : ""}`}
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
                      <RadioGroupItem value="first" id="first" />
                      <div className="grid gap-0.5">
                        <Label htmlFor="first" className="font-normal">
                          First frame only
                        </Label>
                        <p className="text-[10px] text-muted-foreground">Fastest - single front-facing view</p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="key_frames" id="key_frames" />
                      <div className="grid gap-0.5">
                        <Label htmlFor="key_frames" className="font-normal">
                          Key frames (0°, 90°, 180°, 270°)
                        </Label>
                        <p className="text-[10px] text-muted-foreground">Recommended - 4 cardinal angles for good coverage</p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="interval" id="interval" />
                      <div className="grid gap-0.5">
                        <Label htmlFor="interval" className="font-normal">
                          Every N frames
                        </Label>
                        <p className="text-[10px] text-muted-foreground">Custom interval sampling</p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="all" id="all" />
                      <div className="grid gap-0.5">
                        <Label htmlFor="all" className="font-normal">
                          All frames
                        </Label>
                        <p className="text-[10px] text-muted-foreground">Maximum coverage - high storage cost</p>
                      </div>
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

        {/* Right Column: Cutout & Collection Configuration */}
        <div className="space-y-6">
          {/* Cutout Configuration */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <ImageIcon className="h-5 w-5" />
                Cutout Configuration
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger>
                      <Info className="h-4 w-4 text-muted-foreground" />
                    </TooltipTrigger>
                    <TooltipContent side="right" className="max-w-xs">
                      <p className="text-xs">Cutouts are product images cropped from store shelf photos. They represent real-world query images that need to be matched to product catalog embeddings.</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="space-y-1 flex items-center gap-2">
                  <div>
                    <Label>Include Cutouts</Label>
                    <p className="text-xs text-muted-foreground">
                      Extract embeddings for cutout images
                    </p>
                  </div>
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger>
                        <Info className="h-3 w-3 text-muted-foreground" />
                      </TooltipTrigger>
                      <TooltipContent side="right" className="max-w-xs">
                        <p className="text-xs">Enable to also extract embeddings for cutout images stored in a separate collection. Required for matching cutouts against products on the Matching page.</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>
                <Switch checked={includeCutouts} onCheckedChange={setIncludeCutouts} />
              </div>

              {includeCutouts && (
                <div className="flex items-center justify-between">
                  <div className="space-y-1 flex items-center gap-2">
                    <div>
                      <Label>Filter: Has UPC</Label>
                      <p className="text-xs text-muted-foreground">
                        Only cutouts with predicted UPC
                      </p>
                    </div>
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger>
                          <Info className="h-3 w-3 text-muted-foreground" />
                        </TooltipTrigger>
                        <TooltipContent side="right" className="max-w-xs">
                          <p className="text-xs">Filter to only include cutouts where OCR has detected a UPC barcode. These cutouts can be validated against known product barcodes for higher confidence matching.</p>
                        </TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                  </div>
                  <Switch
                    checked={cutoutFilterHasUpc}
                    onCheckedChange={setCutoutFilterHasUpc}
                  />
                </div>
              )}
            </CardContent>
          </Card>

          {/* Collection Configuration */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Database className="h-5 w-5" />
                Collection Configuration
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger>
                      <Info className="h-4 w-4 text-muted-foreground" />
                    </TooltipTrigger>
                    <TooltipContent side="right" className="max-w-xs">
                      <p className="text-xs">Embeddings are stored in Qdrant vector collections. Products and cutouts are stored in separate collections for efficient cross-domain similarity search.</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <Label>Collection Mode</Label>
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger>
                        <Info className="h-3 w-3 text-muted-foreground" />
                      </TooltipTrigger>
                      <TooltipContent side="right" className="max-w-xs">
                        <p className="text-xs">Choose how to handle existing collections: Create (fails if exists), Replace (delete and recreate), or Append (add to existing). Use Append for incremental updates.</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>
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

              {/* Append Mode: Select existing collections */}
              {collectionMode === "append" && (
                <div className="space-y-4 p-3 border rounded-lg bg-muted/30">
                  <div className="space-y-2">
                    <Label>Product Collection</Label>
                    <Select
                      value={selectedProductCollection}
                      onValueChange={setSelectedProductCollection}
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

                  {includeCutouts && (
                    <div className="space-y-2">
                      <Label>Cutout Collection</Label>
                      <Select
                        value={selectedCutoutCollection}
                        onValueChange={setSelectedCutoutCollection}
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
                  )}

                  {/* Dimension mismatch warning */}
                  {hasDimensionMismatch && (
                    <div className="flex items-center gap-2 p-2 bg-orange-500/10 border border-orange-500/30 rounded text-sm text-orange-600">
                      <AlertTriangle className="h-4 w-4" />
                      <span>Vector dimension mismatch! Model: {selectedModel?.embedding_dim}d</span>
                    </div>
                  )}
                </div>
              )}

              {/* Create/Replace Mode: Custom collection names */}
              {collectionMode !== "append" && (
                <Collapsible
                  open={collectionConfigOpen}
                  onOpenChange={setCollectionConfigOpen}
                >
                  <CollapsibleTrigger asChild>
                    <Button variant="outline" className="w-full justify-between">
                      <span>Custom Collection Names</span>
                      <ChevronDown
                        className={`h-4 w-4 transition-transform ${collectionConfigOpen ? "rotate-180" : ""}`}
                      />
                    </Button>
                  </CollapsibleTrigger>
                  <CollapsibleContent className="mt-4 space-y-4">
                    <div className="space-y-2">
                      <Label>Product Collection</Label>
                      <Input
                        placeholder={defaultProductCollection}
                        value={productCollectionName}
                        onChange={(e) => setProductCollectionName(e.target.value)}
                      />
                    </div>
                    {includeCutouts && (
                      <div className="space-y-2">
                        <Label>Cutout Collection</Label>
                        <Input
                          placeholder={defaultCutoutCollection}
                          value={cutoutCollectionName}
                          onChange={(e) => setCutoutCollectionName(e.target.value)}
                        />
                      </div>
                    )}
                  </CollapsibleContent>
                </Collapsible>
              )}

              {/* Preview */}
              <div className="p-3 bg-muted rounded-lg text-sm">
                <p className="flex items-center gap-1 text-muted-foreground mb-2">
                  <Info className="h-4 w-4" />
                  {collectionMode === "create" && "Collections to be created:"}
                  {collectionMode === "replace" && "Collections to be replaced:"}
                  {collectionMode === "append" && "Collections to append to:"}
                </p>
                <ul className="space-y-1">
                  <li>
                    <Badge variant="outline" className="font-mono text-xs">
                      {collectionMode === "append"
                        ? (selectedProductCollection || "Select a collection")
                        : (productCollectionName || defaultProductCollection)}
                    </Badge>
                  </li>
                  {includeCutouts && (
                    <li>
                      <Badge variant="outline" className="font-mono text-xs">
                        {collectionMode === "append"
                          ? (selectedCutoutCollection || "Select a collection")
                          : (cutoutCollectionName || defaultCutoutCollection)}
                      </Badge>
                    </li>
                  )}
                </ul>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Start Button */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium">Ready to extract embeddings</p>
              <p className="text-sm text-muted-foreground">
                Using model: {selectedModel?.name || "No model selected"}
              </p>
            </div>
            <Button
              size="lg"
              onClick={handleStartExtraction}
              disabled={startExtractionMutation.isPending || !selectedModel}
            >
              {startExtractionMutation.isPending ? (
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              ) : (
                <Play className="h-4 w-4 mr-2" />
              )}
              Start Matching Extraction
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
