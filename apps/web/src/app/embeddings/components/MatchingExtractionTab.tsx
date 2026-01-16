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
  Play,
  Loader2,
  ChevronDown,
  Settings2,
  Database,
  Package,
  ImageIcon,
  Info,
} from "lucide-react";
import type {
  EmbeddingModel,
  Dataset,
  MatchingExtractionRequest,
  ProductSource,
  FrameSelection,
  CollectionMode,
} from "@/types";

interface MatchingExtractionTabProps {
  activeModel: EmbeddingModel | null;
}

export function MatchingExtractionTab({ activeModel }: MatchingExtractionTabProps) {
  const queryClient = useQueryClient();

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

  // Frame selection
  const [frameSelection, setFrameSelection] = useState<FrameSelection>("first");
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
    if (!activeModel) {
      toast.error("No active model selected");
      return;
    }

    const request: MatchingExtractionRequest = {
      model_id: activeModel.id,
      product_source: productSource,
      product_dataset_id: productSource === "dataset" ? selectedDatasetId : undefined,
      include_cutouts: includeCutouts,
      cutout_filter_has_upc: cutoutFilterHasUpc,
      collection_mode: collectionMode,
      product_collection_name: productCollectionName || undefined,
      cutout_collection_name: cutoutCollectionName || undefined,
      frame_selection: frameSelection,
      frame_interval: frameInterval[0],
      max_frames: maxFrames[0],
    };

    startExtractionMutation.mutate(request);
  };

  const modelName = activeModel?.name.toLowerCase().replace(/\s+/g, "_") || "default";
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
      </Card>

      {/* Configuration */}
      <div className="grid grid-cols-2 gap-6">
        {/* Left Column: Source Configuration */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Product Source</CardTitle>
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
                  <SelectItem value="all">All Products</SelectItem>
                  <SelectItem value="dataset">From Dataset</SelectItem>
                  <SelectItem value="matched">Matched Products Only</SelectItem>
                  <SelectItem value="new">New Products (no embeddings)</SelectItem>
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
                    onValueChange={(v: FrameSelection) => setFrameSelection(v)}
                  >
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="first" id="first" />
                      <Label htmlFor="first" className="font-normal">
                        First frame only (fastest)
                      </Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="key_frames" id="key_frames" />
                      <Label htmlFor="key_frames" className="font-normal">
                        Key frames (0°, 90°, 180°, 270°)
                      </Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="interval" id="interval" />
                      <Label htmlFor="interval" className="font-normal">
                        Every N frames
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

        {/* Right Column: Cutout & Collection Configuration */}
        <div className="space-y-6">
          {/* Cutout Configuration */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <ImageIcon className="h-5 w-5" />
                Cutout Configuration
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="space-y-1">
                  <Label>Include Cutouts</Label>
                  <p className="text-xs text-muted-foreground">
                    Extract embeddings for cutout images
                  </p>
                </div>
                <Switch checked={includeCutouts} onCheckedChange={setIncludeCutouts} />
              </div>

              {includeCutouts && (
                <div className="flex items-center justify-between">
                  <div className="space-y-1">
                    <Label>Filter: Has UPC</Label>
                    <p className="text-xs text-muted-foreground">
                      Only cutouts with predicted UPC
                    </p>
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
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
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

              {/* Preview */}
              <div className="p-3 bg-muted rounded-lg text-sm">
                <p className="flex items-center gap-1 text-muted-foreground mb-2">
                  <Info className="h-4 w-4" />
                  Collections to be {collectionMode === "create" ? "created" : "updated"}:
                </p>
                <ul className="space-y-1">
                  <li>
                    <Badge variant="outline" className="font-mono text-xs">
                      {productCollectionName || defaultProductCollection}
                    </Badge>
                  </li>
                  {includeCutouts && (
                    <li>
                      <Badge variant="outline" className="font-mono text-xs">
                        {cutoutCollectionName || defaultCutoutCollection}
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
                Using model: {activeModel?.name || "No model selected"}
              </p>
            </div>
            <Button
              size="lg"
              onClick={handleStartExtraction}
              disabled={startExtractionMutation.isPending || !activeModel}
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
