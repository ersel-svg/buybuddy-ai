"use client";

import { useState, useMemo } from "react";
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
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Separator } from "@/components/ui/separator";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import {
  Loader2,
  CheckCircle,
  XCircle,
  Clock,
  Sparkles,
  RefreshCw,
  Wand2,
  Package,
  Info,
  Sun,
  Zap,
  Target,
  Eye,
  SlidersHorizontal,
} from "lucide-react";
import type { Job, Dataset, AugmentationPreset, AugmentationRequest, ProductWithFrameCounts } from "@/types";

// ===========================================
// PRESET DEFINITIONS
// ===========================================
const PRESETS: Record<AugmentationPreset, {
  label: string;
  description: string;
  icon: React.ReactNode;
  color: string;
  defaults: Partial<AugmentationRequest["augmentation_config"]>;
}> = {
  clean: {
    label: "Clean",
    description: "Minimal effects, clean backgrounds. Best for simple recognition tasks.",
    icon: <Sun className="h-4 w-4" />,
    color: "bg-green-100 text-green-700 border-green-200",
    defaults: {
      PROB_HEAVY_AUGMENTATION: 0.1,
      PROB_NEIGHBORING_PRODUCTS: 0.0,
      PROB_SHADOW: 0.2,
      PROB_CAMERA_NOISE: 0.2,
      PROB_LENS_DISTORTION: 0.0,
    },
  },
  normal: {
    label: "Normal",
    description: "Balanced augmentation with moderate shelf scene composition. Recommended for most cases.",
    icon: <Target className="h-4 w-4" />,
    color: "bg-blue-100 text-blue-700 border-blue-200",
    defaults: {
      PROB_HEAVY_AUGMENTATION: 0.5,
      PROB_NEIGHBORING_PRODUCTS: 0.5,
      PROB_SHADOW: 0.7,
      PROB_CAMERA_NOISE: 0.6,
      PROB_LENS_DISTORTION: 0.4,
    },
  },
  realistic: {
    label: "Realistic",
    description: "High realism with neighboring products, shadows, and camera effects.",
    icon: <Eye className="h-4 w-4" />,
    color: "bg-purple-100 text-purple-700 border-purple-200",
    defaults: {
      PROB_HEAVY_AUGMENTATION: 0.8,
      PROB_NEIGHBORING_PRODUCTS: 0.6,
      PROB_SHADOW: 0.8,
      PROB_CAMERA_NOISE: 0.7,
      PROB_LENS_DISTORTION: 0.5,
    },
  },
  extreme: {
    label: "Extreme",
    description: "Maximum diversity with all effects at high probability. For challenging environments.",
    icon: <Zap className="h-4 w-4" />,
    color: "bg-orange-100 text-orange-700 border-orange-200",
    defaults: {
      PROB_HEAVY_AUGMENTATION: 0.9,
      PROB_NEIGHBORING_PRODUCTS: 0.8,
      PROB_SHADOW: 0.9,
      PROB_CAMERA_NOISE: 0.8,
      PROB_LENS_DISTORTION: 0.6,
    },
  },
  custom: {
    label: "Custom",
    description: "Fine-tune all probabilities manually for complete control.",
    icon: <SlidersHorizontal className="h-4 w-4" />,
    color: "bg-gray-100 text-gray-700 border-gray-200",
    defaults: {},
  },
};

// ===========================================
// SLIDER COMPONENT
// ===========================================
function ProbabilitySlider({
  label,
  description,
  value,
  onChange,
  disabled,
}: {
  label: string;
  description: string;
  value: number;
  onChange: (value: number) => void;
  disabled?: boolean;
}) {
  return (
    <div className="space-y-2">
      <div className="flex justify-between items-center">
        <div>
          <Label className="text-sm font-medium">{label}</Label>
          <p className="text-xs text-gray-500">{description}</p>
        </div>
        <span className="text-sm font-mono bg-gray-100 px-2 py-1 rounded">
          {Math.round(value * 100)}%
        </span>
      </div>
      <Slider
        value={[value]}
        onValueChange={([v]) => onChange(v)}
        max={1}
        step={0.05}
        disabled={disabled}
        className={disabled ? "opacity-50" : ""}
      />
    </div>
  );
}

// ===========================================
// MAIN PAGE
// ===========================================
export default function AugmentationPage() {
  const queryClient = useQueryClient();
  const [activeTab, setActiveTab] = useState("new");

  // Main config state
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>("");
  const [synTarget, setSynTarget] = useState(600);
  const [realTarget, setRealTarget] = useState(400);
  const [selectedPreset, setSelectedPreset] = useState<AugmentationPreset>("normal");
  const [useDiversityPyramid, setUseDiversityPyramid] = useState(true);
  const [includeNeighbors, setIncludeNeighbors] = useState(true);
  const [frameInterval, setFrameInterval] = useState(1);

  // Custom config state
  const [customConfig, setCustomConfig] = useState({
    PROB_HEAVY_AUGMENTATION: 0.5,
    PROB_NEIGHBORING_PRODUCTS: 0.5,
    PROB_SHADOW: 0.7,
    PROB_PRICE_TAG: 0.3,
    PROB_SHELF_RAIL: 0.4,
    PROB_CAMPAIGN_STICKER: 0.15,
    PROB_FLUORESCENT_BANDING: 0.4,
    PROB_COLOR_TRANSFER: 0.3,
    PROB_SHELF_REFLECTION: 0.4,
    PROB_PERSPECTIVE_CHANGE: 0.3,
    PROB_LENS_DISTORTION: 0.4,
    PROB_CAMERA_NOISE: 0.6,
    PROB_HSV_SHIFT: 0.5,
    PROB_RGB_SHIFT: 0.4,
    PROB_CONDENSATION: 0.0,
    PROB_FROST_CRYSTALS: 0.15,
    // Geometric limits (degrees)
    ROTATION_LIMIT: 15,
    SHEAR_LIMIT: 10,
  });

  // Fetch datasets
  const { data: datasets } = useQuery({
    queryKey: ["datasets"],
    queryFn: () => apiClient.getDatasets(),
  });

  // Fetch selected dataset with products for frame counts
  const { data: selectedDataset } = useQuery({
    queryKey: ["dataset", selectedDatasetId],
    queryFn: () => apiClient.getDataset(selectedDatasetId),
    enabled: !!selectedDatasetId,
  });

  // Fetch augmentation jobs
  const { data: jobs, isLoading: isLoadingJobs } = useQuery({
    queryKey: ["jobs", "augmentation"],
    queryFn: () => apiClient.getJobs("augmentation"),
    refetchInterval: 5000,
  });

  // Calculate augmentation plan
  const augmentationPlan = useMemo(() => {
    if (!selectedDataset?.products) return { totalNew: 0, products: [] };

    const products = selectedDataset.products.map((product: ProductWithFrameCounts) => {
      const synFrames = product.frame_counts?.synthetic || product.frame_count || 0;
      const augFrames = product.frame_counts?.augmented || 0;
      const currentTotal = synFrames + augFrames;
      const needed = Math.max(0, synTarget - currentTotal);
      const augsPerFrame = synFrames > 0 ? Math.ceil(needed / synFrames) : 0;
      const willCreate = augsPerFrame * synFrames;

      return {
        id: product.id,
        barcode: product.barcode,
        synFrames,
        augFrames,
        currentTotal,
        needed,
        augsPerFrame,
        willCreate,
      };
    });

    const totalNew = products.reduce((sum, p) => sum + p.willCreate, 0);

    return { totalNew, products };
  }, [selectedDataset?.products, synTarget]);

  // Build request
  const buildRequest = (): AugmentationRequest => {
    const config = selectedPreset === "custom" ? customConfig : PRESETS[selectedPreset].defaults;

    return {
      syn_target: synTarget,
      real_target: realTarget,
      use_diversity_pyramid: useDiversityPyramid,
      include_neighbors: includeNeighbors,
      frame_interval: frameInterval,
      augmentation_config: {
        preset: selectedPreset,
        ...config,
      },
    };
  };

  // Start augmentation mutation
  const startMutation = useMutation({
    mutationFn: () => apiClient.startAugmentation(selectedDatasetId, buildRequest()),
    onSuccess: () => {
      toast.success("Augmentation job started successfully!");
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
      setActiveTab("jobs");
    },
    onError: (error) => {
      toast.error("Failed to start augmentation: " + (error as Error).message);
    },
  });

  // Status config for jobs
  const statusConfig: Record<string, { icon: React.ReactNode; color: string; label: string }> = {
    pending: { icon: <Clock className="h-4 w-4" />, color: "bg-yellow-100 text-yellow-800", label: "Pending" },
    queued: { icon: <Clock className="h-4 w-4" />, color: "bg-yellow-100 text-yellow-800", label: "Queued" },
    running: { icon: <Loader2 className="h-4 w-4 animate-spin" />, color: "bg-blue-100 text-blue-800", label: "Running" },
    completed: { icon: <CheckCircle className="h-4 w-4" />, color: "bg-green-100 text-green-800", label: "Completed" },
    failed: { icon: <XCircle className="h-4 w-4" />, color: "bg-red-100 text-red-800", label: "Failed" },
  };

  const runningJobs = jobs?.filter((j: Job) => j.status === "running").length || 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold">Augmentation</h1>
        <p className="text-gray-500">
          Full shelf scene composition with BiRefNet segmentation + DiversityPyramid
        </p>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList>
          <TabsTrigger value="new">New Augmentation</TabsTrigger>
          <TabsTrigger value="jobs">
            Jobs {runningJobs > 0 && <Badge variant="secondary" className="ml-2">{runningJobs} running</Badge>}
          </TabsTrigger>
        </TabsList>

        {/* New Augmentation Tab */}
        <TabsContent value="new" className="mt-6">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

            {/* Left Column: Dataset & Targets */}
            <div className="space-y-6">
              {/* Dataset Selection */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Package className="h-5 w-5" />
                    Dataset
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <Select value={selectedDatasetId} onValueChange={setSelectedDatasetId}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select a dataset..." />
                    </SelectTrigger>
                    <SelectContent>
                      {datasets?.map((dataset: Dataset) => (
                        <SelectItem key={dataset.id} value={dataset.id}>
                          {dataset.name} ({dataset.product_count} products)
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>

                  {selectedDataset && (
                    <div className="grid grid-cols-3 gap-2 pt-2">
                      <div className="text-center p-2 bg-blue-50 rounded">
                        <p className="text-lg font-bold text-blue-600">
                          {selectedDataset.total_synthetic || 0}
                        </p>
                        <p className="text-xs text-gray-500">Synthetic</p>
                      </div>
                      <div className="text-center p-2 bg-green-50 rounded">
                        <p className="text-lg font-bold text-green-600">
                          {selectedDataset.total_real || 0}
                        </p>
                        <p className="text-xs text-gray-500">Real</p>
                      </div>
                      <div className="text-center p-2 bg-purple-50 rounded">
                        <p className="text-lg font-bold text-purple-600">
                          {selectedDataset.total_augmented || 0}
                        </p>
                        <p className="text-xs text-gray-500">Augmented</p>
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Target Settings */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Target className="h-5 w-5" />
                    Targets (per product)
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <Label>Synthetic Target</Label>
                      <Input
                        type="number"
                        value={synTarget}
                        onChange={(e) => setSynTarget(parseInt(e.target.value) || 0)}
                        className="w-24 text-right"
                      />
                    </div>
                    <Slider
                      value={[synTarget]}
                      onValueChange={([v]) => setSynTarget(v)}
                      min={100}
                      max={1000}
                      step={50}
                    />
                  </div>

                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <Label>Real Target</Label>
                      <Input
                        type="number"
                        value={realTarget}
                        onChange={(e) => setRealTarget(parseInt(e.target.value) || 0)}
                        className="w-24 text-right"
                      />
                    </div>
                    <Slider
                      value={[realTarget]}
                      onValueChange={([v]) => setRealTarget(v)}
                      min={50}
                      max={500}
                      step={25}
                    />
                  </div>
                </CardContent>
              </Card>

              {/* Equalization Preview */}
              {selectedDataset && augmentationPlan.totalNew > 0 && (
                <Card className="border-purple-200 bg-purple-50/50">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm flex items-center gap-2">
                      <Info className="h-4 w-4 text-purple-600" />
                      Equalization Preview
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span>New augmentations:</span>
                        <span className="font-bold text-purple-700">
                          {augmentationPlan.totalNew.toLocaleString()}
                        </span>
                      </div>
                      <p className="text-xs text-gray-500">
                        Formula: (target - current) / source_frames = augs_per_frame
                      </p>

                      {augmentationPlan.products.slice(0, 3).map((p) => (
                        <div key={p.id} className="flex justify-between text-xs py-1 border-t">
                          <span className="font-mono">{p.barcode}</span>
                          <span>
                            {p.synFrames} × {p.augsPerFrame} = {p.willCreate}
                          </span>
                        </div>
                      ))}
                      {augmentationPlan.products.length > 3 && (
                        <p className="text-xs text-gray-400">
                          +{augmentationPlan.products.length - 3} more products...
                        </p>
                      )}
                    </div>
                  </CardContent>
                </Card>
              )}
            </div>

            {/* Middle Column: Preset Selection */}
            <div className="space-y-6">
              {/* Preset Cards */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Sparkles className="h-5 w-5" />
                    Augmentation Preset
                  </CardTitle>
                  <CardDescription>
                    Choose a preset or customize all settings
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  {(Object.entries(PRESETS) as [AugmentationPreset, typeof PRESETS[AugmentationPreset]][]).map(
                    ([key, preset]) => (
                      <div
                        key={key}
                        onClick={() => setSelectedPreset(key)}
                        className={`p-3 rounded-lg border-2 cursor-pointer transition-all ${
                          selectedPreset === key
                            ? `${preset.color} border-current`
                            : "border-gray-100 hover:border-gray-200"
                        }`}
                      >
                        <div className="flex items-center gap-2">
                          {preset.icon}
                          <span className="font-medium">{preset.label}</span>
                          {key === "normal" && (
                            <Badge variant="outline" className="text-xs">Recommended</Badge>
                          )}
                        </div>
                        <p className="text-xs text-gray-500 mt-1">{preset.description}</p>
                      </div>
                    )
                  )}
                </CardContent>
              </Card>

              {/* Advanced Options */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Advanced Options</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <Label>Diversity Pyramid</Label>
                      <p className="text-xs text-gray-500">
                        Randomly vary intensity per image (clean/normal/realistic/extreme)
                      </p>
                    </div>
                    <Switch
                      checked={useDiversityPyramid}
                      onCheckedChange={setUseDiversityPyramid}
                    />
                  </div>

                  <div className="flex items-center justify-between">
                    <div>
                      <Label>Include Neighbors</Label>
                      <p className="text-xs text-gray-500">
                        Add neighboring products to shelf scenes
                      </p>
                    </div>
                    <Switch
                      checked={includeNeighbors}
                      onCheckedChange={setIncludeNeighbors}
                    />
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Right Column: Custom Settings & Start */}
            <div className="space-y-6">
              {/* Custom Settings (only for custom preset) */}
              {selectedPreset === "custom" && (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <SlidersHorizontal className="h-5 w-5" />
                      Custom Probabilities
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <Accordion type="single" collapsible className="w-full">
                      <AccordionItem value="geometric">
                        <AccordionTrigger className="text-sm">Geometric Transforms</AccordionTrigger>
                        <AccordionContent className="space-y-4 pt-2">
                          <div className="space-y-2">
                            <div className="flex justify-between items-center">
                              <div>
                                <Label className="text-sm font-medium">Rotation Limit</Label>
                                <p className="text-xs text-gray-500">Max rotation angle (±degrees)</p>
                              </div>
                              <span className="text-sm font-mono bg-gray-100 px-2 py-1 rounded">
                                ±{customConfig.ROTATION_LIMIT}°
                              </span>
                            </div>
                            <Slider
                              value={[customConfig.ROTATION_LIMIT]}
                              onValueChange={([v]) => setCustomConfig({ ...customConfig, ROTATION_LIMIT: v })}
                              min={5}
                              max={180}
                              step={5}
                            />
                            <p className="text-xs text-gray-400">
                              15° = normal, 45° = tilted, 90° = sideways, 180° = upside down
                            </p>
                          </div>
                          <div className="space-y-2">
                            <div className="flex justify-between items-center">
                              <div>
                                <Label className="text-sm font-medium">Shear Limit</Label>
                                <p className="text-xs text-gray-500">Max shear angle (±degrees)</p>
                              </div>
                              <span className="text-sm font-mono bg-gray-100 px-2 py-1 rounded">
                                ±{customConfig.SHEAR_LIMIT}°
                              </span>
                            </div>
                            <Slider
                              value={[customConfig.SHEAR_LIMIT]}
                              onValueChange={([v]) => setCustomConfig({ ...customConfig, SHEAR_LIMIT: v })}
                              min={5}
                              max={30}
                              step={5}
                            />
                          </div>
                        </AccordionContent>
                      </AccordionItem>

                      <AccordionItem value="scene">
                        <AccordionTrigger className="text-sm">Scene Composition</AccordionTrigger>
                        <AccordionContent className="space-y-4 pt-2">
                          <ProbabilitySlider
                            label="Heavy Augmentation"
                            description="Strong distortions vs light transforms"
                            value={customConfig.PROB_HEAVY_AUGMENTATION}
                            onChange={(v) => setCustomConfig({ ...customConfig, PROB_HEAVY_AUGMENTATION: v })}
                          />
                          <ProbabilitySlider
                            label="Neighboring Products"
                            description="Add other products to shelf"
                            value={customConfig.PROB_NEIGHBORING_PRODUCTS}
                            onChange={(v) => setCustomConfig({ ...customConfig, PROB_NEIGHBORING_PRODUCTS: v })}
                          />
                          <ProbabilitySlider
                            label="Shadow"
                            description="Product shadow on shelf"
                            value={customConfig.PROB_SHADOW}
                            onChange={(v) => setCustomConfig({ ...customConfig, PROB_SHADOW: v })}
                          />
                        </AccordionContent>
                      </AccordionItem>

                      <AccordionItem value="shelf">
                        <AccordionTrigger className="text-sm">Shelf Elements</AccordionTrigger>
                        <AccordionContent className="space-y-4 pt-2">
                          <ProbabilitySlider
                            label="Price Tag"
                            description="Add realistic price tags"
                            value={customConfig.PROB_PRICE_TAG}
                            onChange={(v) => setCustomConfig({ ...customConfig, PROB_PRICE_TAG: v })}
                          />
                          <ProbabilitySlider
                            label="Shelf Rail"
                            description="Bottom price rail"
                            value={customConfig.PROB_SHELF_RAIL}
                            onChange={(v) => setCustomConfig({ ...customConfig, PROB_SHELF_RAIL: v })}
                          />
                          <ProbabilitySlider
                            label="Campaign Sticker"
                            description="Discount/promo stickers"
                            value={customConfig.PROB_CAMPAIGN_STICKER}
                            onChange={(v) => setCustomConfig({ ...customConfig, PROB_CAMPAIGN_STICKER: v })}
                          />
                        </AccordionContent>
                      </AccordionItem>

                      <AccordionItem value="camera">
                        <AccordionTrigger className="text-sm">Camera Effects</AccordionTrigger>
                        <AccordionContent className="space-y-4 pt-2">
                          <ProbabilitySlider
                            label="Camera Noise"
                            description="ISO noise simulation"
                            value={customConfig.PROB_CAMERA_NOISE}
                            onChange={(v) => setCustomConfig({ ...customConfig, PROB_CAMERA_NOISE: v })}
                          />
                          <ProbabilitySlider
                            label="Lens Distortion"
                            description="Barrel/pincushion distortion"
                            value={customConfig.PROB_LENS_DISTORTION}
                            onChange={(v) => setCustomConfig({ ...customConfig, PROB_LENS_DISTORTION: v })}
                          />
                          <ProbabilitySlider
                            label="Perspective Change"
                            description="Looking up/down angle"
                            value={customConfig.PROB_PERSPECTIVE_CHANGE}
                            onChange={(v) => setCustomConfig({ ...customConfig, PROB_PERSPECTIVE_CHANGE: v })}
                          />
                        </AccordionContent>
                      </AccordionItem>

                      <AccordionItem value="lighting">
                        <AccordionTrigger className="text-sm">Lighting Effects</AccordionTrigger>
                        <AccordionContent className="space-y-4 pt-2">
                          <ProbabilitySlider
                            label="Fluorescent Banding"
                            description="Store lighting artifacts"
                            value={customConfig.PROB_FLUORESCENT_BANDING}
                            onChange={(v) => setCustomConfig({ ...customConfig, PROB_FLUORESCENT_BANDING: v })}
                          />
                          <ProbabilitySlider
                            label="Color Transfer"
                            description="Match background colors"
                            value={customConfig.PROB_COLOR_TRANSFER}
                            onChange={(v) => setCustomConfig({ ...customConfig, PROB_COLOR_TRANSFER: v })}
                          />
                          <ProbabilitySlider
                            label="Shelf Reflection"
                            description="Product reflection below"
                            value={customConfig.PROB_SHELF_REFLECTION}
                            onChange={(v) => setCustomConfig({ ...customConfig, PROB_SHELF_REFLECTION: v })}
                          />
                        </AccordionContent>
                      </AccordionItem>

                      <AccordionItem value="fridge">
                        <AccordionTrigger className="text-sm">Refrigerator Effects</AccordionTrigger>
                        <AccordionContent className="space-y-4 pt-2">
                          <ProbabilitySlider
                            label="Condensation"
                            description="Fog/moisture effect"
                            value={customConfig.PROB_CONDENSATION}
                            onChange={(v) => setCustomConfig({ ...customConfig, PROB_CONDENSATION: v })}
                          />
                          <ProbabilitySlider
                            label="Frost Crystals"
                            description="Ice crystal overlay"
                            value={customConfig.PROB_FROST_CRYSTALS}
                            onChange={(v) => setCustomConfig({ ...customConfig, PROB_FROST_CRYSTALS: v })}
                          />
                        </AccordionContent>
                      </AccordionItem>
                    </Accordion>
                  </CardContent>
                </Card>
              )}

              {/* Start Button Card */}
              <Card className="border-2 border-dashed">
                <CardContent className="pt-6">
                  <div className="space-y-4">
                    <div className="text-center space-y-1">
                      <p className="text-2xl font-bold">
                        {augmentationPlan.totalNew.toLocaleString()}
                      </p>
                      <p className="text-sm text-gray-500">new images will be created</p>
                    </div>

                    <Separator />

                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-500">Preset:</span>
                        <span className="font-medium">{PRESETS[selectedPreset].label}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-500">Diversity Pyramid:</span>
                        <span className="font-medium">{useDiversityPyramid ? "On" : "Off"}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-500">Neighbors:</span>
                        <span className="font-medium">{includeNeighbors ? "On" : "Off"}</span>
                      </div>
                    </div>

                    <Button
                      className="w-full"
                      size="lg"
                      onClick={() => startMutation.mutate()}
                      disabled={!selectedDatasetId || startMutation.isPending || augmentationPlan.totalNew === 0}
                    >
                      {startMutation.isPending ? (
                        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      ) : (
                        <Sparkles className="h-4 w-4 mr-2" />
                      )}
                      Start Augmentation
                    </Button>

                    {!selectedDatasetId && (
                      <p className="text-xs text-center text-yellow-600">
                        Please select a dataset first
                      </p>
                    )}
                  </div>
                </CardContent>
              </Card>

              {/* Features Info */}
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm flex items-center gap-2">
                    <Wand2 className="h-4 w-4" />
                    Shelf Scene Features
                  </CardTitle>
                </CardHeader>
                <CardContent className="text-xs text-gray-500 space-y-1">
                  <p>• BiRefNet GPU segmentation (half-precision)</p>
                  <p>• 822 real shelf/fridge backgrounds</p>
                  <p>• Neighboring products composition</p>
                  <p>• Realistic shadows & reflections</p>
                  <p>• Price tags & campaign stickers</p>
                  <p>• Camera noise & lens effects</p>
                  <p>• Direct resize to 384×384 (no borders)</p>
                </CardContent>
              </Card>
            </div>
          </div>
        </TabsContent>

        {/* Jobs Tab */}
        <TabsContent value="jobs" className="mt-6">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <div>
                <CardTitle>Augmentation Jobs</CardTitle>
                <CardDescription>Monitor your augmentation runs</CardDescription>
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={() => queryClient.invalidateQueries({ queryKey: ["jobs"] })}
              >
                <RefreshCw className="h-4 w-4" />
              </Button>
            </CardHeader>
            <CardContent>
              {isLoadingJobs ? (
                <div className="flex items-center justify-center py-8">
                  <Loader2 className="h-6 w-6 animate-spin" />
                </div>
              ) : jobs?.length === 0 ? (
                <div className="text-center py-8 text-gray-500">
                  <Sparkles className="h-12 w-12 mx-auto mb-2 opacity-50" />
                  <p>No augmentation jobs yet</p>
                  <p className="text-sm mt-1">Start an augmentation to generate training images</p>
                </div>
              ) : (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>ID</TableHead>
                      <TableHead>Progress</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead>Started</TableHead>
                      <TableHead>Output</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {jobs?.map((job: Job) => (
                      <TableRow key={job.id}>
                        <TableCell className="font-mono text-sm">
                          {job.id.slice(0, 8)}...
                        </TableCell>
                        <TableCell>
                          <div className="w-32">
                            <Progress value={job.progress} className="h-2" />
                            <p className="text-xs text-gray-500 mt-1">{job.progress}%</p>
                          </div>
                        </TableCell>
                        <TableCell>
                          <Badge className={statusConfig[job.status]?.color}>
                            <span className="mr-1">{statusConfig[job.status]?.icon}</span>
                            {statusConfig[job.status]?.label}
                          </Badge>
                        </TableCell>
                        <TableCell className="text-gray-500 text-sm" suppressHydrationWarning>
                          {new Date(job.created_at).toLocaleString()}
                        </TableCell>
                        <TableCell>
                          {job.result && (
                            <span className="text-sm">
                              {(job.result as { syn_produced?: number; real_produced?: number })?.syn_produced || 0} syn,{" "}
                              {(job.result as { syn_produced?: number; real_produced?: number })?.real_produced || 0} real
                            </span>
                          )}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
