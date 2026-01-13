"use client";

import { useState, useMemo } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useParams } from "next/navigation";
import { toast } from "sonner";
import Link from "next/link";
import { apiClient } from "@/lib/api-client";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  ArrowLeft,
  Plus,
  Trash2,
  Sparkles,
  Brain,
  Layers,
  Loader2,
  Package,
  FolderOpen,
  Info,
  ImageIcon,
  Camera,
  Wand2,
} from "lucide-react";
import type { ProductWithFrameCounts, AugmentationPreset, AugmentationRequest } from "@/types";

// Preset descriptions
const PRESET_INFO: Record<AugmentationPreset, { label: string; description: string }> = {
  clean: {
    label: "Clean",
    description: "Minimal effects, clean backgrounds, ideal for simple recognition tasks",
  },
  normal: {
    label: "Normal (Recommended)",
    description: "Balanced augmentation with moderate shelf scene composition",
  },
  realistic: {
    label: "Realistic",
    description: "High realism with neighboring products, shadows, and camera effects",
  },
  extreme: {
    label: "Extreme",
    description: "Maximum diversity with all effects enabled at high probability",
  },
  custom: {
    label: "Custom",
    description: "Configure all probabilities manually",
  },
};

// Calculate augmentation plan
function calculateAugmentationPlan(
  products: ProductWithFrameCounts[],
  synTarget: number,
  frameInterval: number = 1
) {
  return products.map((product) => {
    const synFrames = product.frame_counts?.synthetic || 0;
    const augFrames = product.frame_counts?.augmented || 0;
    const currentTotal = synFrames + augFrames;
    const needed = Math.max(0, synTarget - currentTotal);
    // Calculate selected frames based on interval (e.g., 200 frames / 20 interval = 10 selected)
    const selectedFrames = Math.max(1, Math.floor(synFrames / frameInterval));
    const augsPerFrame = selectedFrames > 0 ? Math.ceil(needed / selectedFrames) : 0;

    return {
      id: product.id,
      barcode: product.barcode,
      synFrames,
      selectedFrames,
      augFrames,
      currentTotal,
      needed,
      augsPerFrame,
      projected: currentTotal + (augsPerFrame * selectedFrames),
    };
  });
}

export default function DatasetDetailPage() {
  const { id } = useParams<{ id: string }>();
  const queryClient = useQueryClient();

  // Augmentation dialog state
  const [augDialogOpen, setAugDialogOpen] = useState(false);
  const [augConfig, setAugConfig] = useState<AugmentationRequest>({
    syn_target: 600,
    real_target: 400,
    use_diversity_pyramid: true,
    include_neighbors: true,
    frame_interval: 1,
    augmentation_config: {
      preset: "normal",
    },
  });

  // Fetch dataset with products
  const {
    data: dataset,
    isLoading,
    error,
  } = useQuery({
    queryKey: ["dataset", id],
    queryFn: () => apiClient.getDataset(id),
    enabled: !!id,
  });

  // Calculate augmentation plan
  const augmentationPlan = useMemo(() => {
    if (!dataset?.products) return [];
    return calculateAugmentationPlan(dataset.products, augConfig.syn_target, augConfig.frame_interval);
  }, [dataset?.products, augConfig.syn_target, augConfig.frame_interval]);

  // Total augmentations that will be created
  const totalAugmentations = useMemo(() => {
    return augmentationPlan.reduce((sum, p) => sum + (p.augsPerFrame * p.selectedFrames), 0);
  }, [augmentationPlan]);

  // Remove product mutation
  const removeProductMutation = useMutation({
    mutationFn: (productId: string) =>
      apiClient.removeProductFromDataset(id, productId),
    onSuccess: () => {
      toast.success("Product removed");
      queryClient.invalidateQueries({ queryKey: ["dataset", id] });
    },
    onError: () => {
      toast.error("Failed to remove product");
    },
  });

  // Augmentation mutation with full config
  const augmentMutation = useMutation({
    mutationFn: (config: AugmentationRequest) =>
      apiClient.startAugmentation(id, config),
    onSuccess: () => {
      toast.success("Augmentation job started");
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
      setAugDialogOpen(false);
    },
    onError: () => {
      toast.error("Failed to start augmentation");
    },
  });

  const trainMutation = useMutation({
    mutationFn: () =>
      apiClient.startTrainingFromDataset(id, {
        dataset_id: id,
        model_name: "facebook/dinov2-large",
        proj_dim: 512,
        epochs: 30,
        batch_size: 32,
        learning_rate: 1e-4,
        weight_decay: 0.01,
        label_smoothing: 0.1,
        warmup_epochs: 3,
        grad_clip: 1.0,
        llrd_decay: 0.95,
        domain_aware_ratio: 0.3,
        hard_negative_pool_size: 10,
        use_hardest_negatives: true,
        use_mixed_precision: true,
        train_ratio: 0.8,
        valid_ratio: 0.1,
        test_ratio: 0.1,
        save_every: 5,
        seed: 42,
      }),
    onSuccess: () => {
      toast.success("Training job started");
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
    },
    onError: () => {
      toast.error("Failed to start training");
    },
  });

  const extractMutation = useMutation({
    mutationFn: () => apiClient.startEmbeddingExtraction(id, "active"),
    onSuccess: () => {
      toast.success("Embedding extraction started");
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
    },
    onError: () => {
      toast.error("Failed to start embedding extraction");
    },
  });

  // Handle augmentation config changes
  const handlePresetChange = (preset: AugmentationPreset) => {
    setAugConfig((prev) => ({
      ...prev,
      augmentation_config: {
        ...prev.augmentation_config,
        preset,
      },
    }));
  };

  const handleStartAugmentation = () => {
    augmentMutation.mutate(augConfig);
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <Loader2 className="h-8 w-8 animate-spin" />
      </div>
    );
  }

  if (error || !dataset) {
    return (
      <div className="flex flex-col items-center justify-center h-96">
        <FolderOpen className="h-16 w-16 text-gray-300 mb-4" />
        <h2 className="text-xl font-semibold">Dataset not found</h2>
        <p className="text-gray-500 mb-4">
          The dataset you&apos;re looking for doesn&apos;t exist.
        </p>
        <Link href="/datasets">
          <Button>Back to Datasets</Button>
        </Link>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Link href="/datasets">
            <Button variant="ghost" size="icon">
              <ArrowLeft className="h-4 w-4" />
            </Button>
          </Link>
          <div>
            <h1 className="text-2xl font-bold">{dataset.name}</h1>
            <p className="text-gray-500">
              {dataset.description || "No description"}
            </p>
          </div>
        </div>
        <div className="text-sm text-gray-500" suppressHydrationWarning>
          Created {new Date(dataset.created_at).toLocaleDateString()}
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-4 gap-4">
        <Card>
          <CardContent className="pt-4">
            <div className="flex items-center gap-2">
              <Package className="h-4 w-4 text-gray-500" />
              <span className="text-sm text-gray-500">Products</span>
            </div>
            <p className="text-2xl font-bold">{dataset.product_count}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4">
            <div className="flex items-center gap-2">
              <ImageIcon className="h-4 w-4 text-blue-500" />
              <span className="text-sm text-gray-500">Synthetic</span>
            </div>
            <p className="text-2xl font-bold text-blue-600">{dataset.total_synthetic || 0}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4">
            <div className="flex items-center gap-2">
              <Camera className="h-4 w-4 text-green-500" />
              <span className="text-sm text-gray-500">Real</span>
            </div>
            <p className="text-2xl font-bold text-green-600">{dataset.total_real || 0}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4">
            <div className="flex items-center gap-2">
              <Wand2 className="h-4 w-4 text-purple-500" />
              <span className="text-sm text-gray-500">Augmented</span>
            </div>
            <p className="text-2xl font-bold text-purple-600">{dataset.total_augmented || 0}</p>
          </CardContent>
        </Card>
      </div>

      {/* Actions Card */}
      <Card>
        <CardHeader>
          <CardTitle>Dataset Actions</CardTitle>
          <CardDescription>
            Run GPU-intensive operations on this dataset
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-4">
            {/* Augmentation Dialog */}
            <Dialog open={augDialogOpen} onOpenChange={setAugDialogOpen}>
              <DialogTrigger asChild>
                <Button disabled={dataset.product_count === 0}>
                  <Sparkles className="h-4 w-4 mr-2" />
                  Run Augmentation
                </Button>
              </DialogTrigger>
              <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
                <DialogHeader>
                  <DialogTitle>Augmentation Settings</DialogTitle>
                  <DialogDescription>
                    Configure augmentation parameters for shelf scene composition
                  </DialogDescription>
                </DialogHeader>

                <div className="space-y-6 py-4">
                  {/* Target Settings */}
                  <div className="grid grid-cols-3 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="syn_target">Synthetic Target (per product)</Label>
                      <Input
                        id="syn_target"
                        type="number"
                        value={augConfig.syn_target}
                        onChange={(e) =>
                          setAugConfig((prev) => ({
                            ...prev,
                            syn_target: parseInt(e.target.value) || 0,
                          }))
                        }
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="real_target">Real Target (per product)</Label>
                      <Input
                        id="real_target"
                        type="number"
                        value={augConfig.real_target}
                        onChange={(e) =>
                          setAugConfig((prev) => ({
                            ...prev,
                            real_target: parseInt(e.target.value) || 0,
                          }))
                        }
                      />
                    </div>
                    <div className="space-y-2">
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <Label htmlFor="frame_interval" className="flex items-center gap-1 cursor-help">
                              Frame Interval
                              <Info className="h-3 w-3 text-gray-400" />
                            </Label>
                          </TooltipTrigger>
                          <TooltipContent className="max-w-xs">
                            <p>Select 1 frame every N frames for angle diversity from 360 degree rotating videos.</p>
                            <p className="mt-1 text-xs">E.g., interval=20 with 200 frames = 10 unique angles</p>
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                      <Input
                        id="frame_interval"
                        type="number"
                        min={1}
                        value={augConfig.frame_interval}
                        onChange={(e) =>
                          setAugConfig((prev) => ({
                            ...prev,
                            frame_interval: Math.max(1, parseInt(e.target.value) || 1),
                          }))
                        }
                      />
                    </div>
                  </div>

                  {/* Equalization Preview */}
                  <div className="bg-gray-50 rounded-lg p-4 space-y-2">
                    <div className="flex items-center gap-2">
                      <Info className="h-4 w-4 text-blue-500" />
                      <span className="font-medium">Augmentation Preview</span>
                    </div>
                    <div className="text-sm text-gray-600">
                      <p>Total new augmentations: <span className="font-bold text-purple-600">{totalAugmentations.toLocaleString()}</span></p>
                      <p className="text-xs mt-1">
                        Formula: (target - current) / selected_frames = augs per frame
                        {augConfig.frame_interval > 1 && (
                          <span className="text-blue-600 ml-1">
                            (using 1 of every {augConfig.frame_interval} frames for angle diversity)
                          </span>
                        )}
                      </p>
                    </div>
                    {augmentationPlan.length > 0 && augmentationPlan.length <= 5 && (
                      <div className="mt-2 text-xs">
                        {augmentationPlan.map((p) => (
                          <div key={p.id} className="flex justify-between py-1 border-b border-gray-200 last:border-0">
                            <span className="font-mono">{p.barcode}</span>
                            <span>
                              {p.selectedFrames} frames {augConfig.frame_interval > 1 && <span className="text-gray-400">(of {p.synFrames})</span>} x {p.augsPerFrame} = {p.augsPerFrame * p.selectedFrames} new
                              {p.needed === 0 && <span className="text-green-600 ml-1">(target met)</span>}
                            </span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>

                  {/* Preset Selection */}
                  <div className="space-y-2">
                    <Label>Augmentation Preset</Label>
                    <Select
                      value={augConfig.augmentation_config?.preset || "normal"}
                      onValueChange={(value) => handlePresetChange(value as AugmentationPreset)}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Select preset" />
                      </SelectTrigger>
                      <SelectContent>
                        {Object.entries(PRESET_INFO).map(([key, info]) => (
                          <SelectItem key={key} value={key}>
                            <div className="flex flex-col">
                              <span>{info.label}</span>
                            </div>
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    <p className="text-sm text-gray-500">
                      {PRESET_INFO[augConfig.augmentation_config?.preset || "normal"].description}
                    </p>
                  </div>

                  {/* Advanced Options */}
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <div className="space-y-0.5">
                        <Label>Diversity Pyramid</Label>
                        <p className="text-sm text-gray-500">
                          Randomly vary augmentation intensity for each image
                        </p>
                      </div>
                      <Switch
                        checked={augConfig.use_diversity_pyramid}
                        onCheckedChange={(checked) =>
                          setAugConfig((prev) => ({
                            ...prev,
                            use_diversity_pyramid: checked,
                          }))
                        }
                      />
                    </div>

                    <div className="flex items-center justify-between">
                      <div className="space-y-0.5">
                        <Label>Include Neighbors</Label>
                        <p className="text-sm text-gray-500">
                          Add neighboring products to shelf scenes
                        </p>
                      </div>
                      <Switch
                        checked={augConfig.include_neighbors}
                        onCheckedChange={(checked) =>
                          setAugConfig((prev) => ({
                            ...prev,
                            include_neighbors: checked,
                          }))
                        }
                      />
                    </div>
                  </div>

                  {/* Custom Settings (only shown for custom preset) */}
                  {augConfig.augmentation_config?.preset === "custom" && (
                    <div className="space-y-4 border-t pt-4">
                      <h4 className="font-medium">Custom Probabilities</h4>

                      <div className="grid grid-cols-2 gap-4">
                        <div className="space-y-2">
                          <Label className="text-sm">Heavy Augmentation</Label>
                          <Slider
                            value={[augConfig.augmentation_config?.PROB_HEAVY_AUGMENTATION ?? 0.5]}
                            max={1}
                            step={0.1}
                            onValueChange={([value]) =>
                              setAugConfig((prev) => ({
                                ...prev,
                                augmentation_config: {
                                  ...prev.augmentation_config!,
                                  PROB_HEAVY_AUGMENTATION: value,
                                },
                              }))
                            }
                          />
                        </div>

                        <div className="space-y-2">
                          <Label className="text-sm">Neighboring Products</Label>
                          <Slider
                            value={[augConfig.augmentation_config?.PROB_NEIGHBORING_PRODUCTS ?? 0.5]}
                            max={1}
                            step={0.1}
                            onValueChange={([value]) =>
                              setAugConfig((prev) => ({
                                ...prev,
                                augmentation_config: {
                                  ...prev.augmentation_config!,
                                  PROB_NEIGHBORING_PRODUCTS: value,
                                },
                              }))
                            }
                          />
                        </div>

                        <div className="space-y-2">
                          <Label className="text-sm">Shadow</Label>
                          <Slider
                            value={[augConfig.augmentation_config?.PROB_SHADOW ?? 0.7]}
                            max={1}
                            step={0.1}
                            onValueChange={([value]) =>
                              setAugConfig((prev) => ({
                                ...prev,
                                augmentation_config: {
                                  ...prev.augmentation_config!,
                                  PROB_SHADOW: value,
                                },
                              }))
                            }
                          />
                        </div>

                        <div className="space-y-2">
                          <Label className="text-sm">Camera Noise</Label>
                          <Slider
                            value={[augConfig.augmentation_config?.PROB_CAMERA_NOISE ?? 0.6]}
                            max={1}
                            step={0.1}
                            onValueChange={([value]) =>
                              setAugConfig((prev) => ({
                                ...prev,
                                augmentation_config: {
                                  ...prev.augmentation_config!,
                                  PROB_CAMERA_NOISE: value,
                                },
                              }))
                            }
                          />
                        </div>

                        <div className="space-y-2">
                          <Label className="text-sm">Lens Distortion</Label>
                          <Slider
                            value={[augConfig.augmentation_config?.PROB_LENS_DISTORTION ?? 0.4]}
                            max={1}
                            step={0.1}
                            onValueChange={([value]) =>
                              setAugConfig((prev) => ({
                                ...prev,
                                augmentation_config: {
                                  ...prev.augmentation_config!,
                                  PROB_LENS_DISTORTION: value,
                                },
                              }))
                            }
                          />
                        </div>

                        <div className="space-y-2">
                          <Label className="text-sm">Price Tag</Label>
                          <Slider
                            value={[augConfig.augmentation_config?.PROB_PRICE_TAG ?? 0.3]}
                            max={1}
                            step={0.1}
                            onValueChange={([value]) =>
                              setAugConfig((prev) => ({
                                ...prev,
                                augmentation_config: {
                                  ...prev.augmentation_config!,
                                  PROB_PRICE_TAG: value,
                                },
                              }))
                            }
                          />
                        </div>
                      </div>
                    </div>
                  )}
                </div>

                <DialogFooter>
                  <Button variant="outline" onClick={() => setAugDialogOpen(false)}>
                    Cancel
                  </Button>
                  <Button
                    onClick={handleStartAugmentation}
                    disabled={augmentMutation.isPending}
                  >
                    {augmentMutation.isPending ? (
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    ) : (
                      <Sparkles className="h-4 w-4 mr-2" />
                    )}
                    Start Augmentation
                  </Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>

            <Button
              variant="outline"
              onClick={() => trainMutation.mutate()}
              disabled={trainMutation.isPending || dataset.product_count === 0}
            >
              {trainMutation.isPending ? (
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              ) : (
                <Brain className="h-4 w-4 mr-2" />
              )}
              Start Training
            </Button>
            <Button
              variant="outline"
              onClick={() => extractMutation.mutate()}
              disabled={extractMutation.isPending || dataset.product_count === 0}
            >
              {extractMutation.isPending ? (
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              ) : (
                <Layers className="h-4 w-4 mr-2" />
              )}
              Extract Embeddings
            </Button>
          </div>
          {dataset.product_count === 0 && (
            <p className="text-sm text-yellow-600 mt-3">
              Add products to this dataset before running actions.
            </p>
          )}
        </CardContent>
      </Card>

      {/* Products Card */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <div>
            <CardTitle>Products ({dataset.product_count})</CardTitle>
            <CardDescription>
              Products included in this dataset with frame counts
            </CardDescription>
          </div>
          <Link href={`/datasets/${id}/add-products`}>
            <Button>
              <Plus className="h-4 w-4 mr-2" />
              Add Products
            </Button>
          </Link>
        </CardHeader>
        <CardContent>
          {dataset.products?.length === 0 ? (
            <div className="text-center py-8">
              <Package className="h-12 w-12 mx-auto text-gray-300 mb-2" />
              <p className="text-gray-500">No products in this dataset</p>
              <p className="text-gray-400 text-sm">
                Click &quot;Add Products&quot; to get started
              </p>
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Image</TableHead>
                  <TableHead>Barcode</TableHead>
                  <TableHead>Brand</TableHead>
                  <TableHead>Product</TableHead>
                  <TableHead className="text-center">
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger className="flex items-center gap-1">
                          <ImageIcon className="h-3 w-3 text-blue-500" />
                          Syn
                        </TooltipTrigger>
                        <TooltipContent>Synthetic frames from video</TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                  </TableHead>
                  <TableHead className="text-center">
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger className="flex items-center gap-1">
                          <Camera className="h-3 w-3 text-green-500" />
                          Real
                        </TooltipTrigger>
                        <TooltipContent>Real matched images</TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                  </TableHead>
                  <TableHead className="text-center">
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger className="flex items-center gap-1">
                          <Wand2 className="h-3 w-3 text-purple-500" />
                          Aug
                        </TooltipTrigger>
                        <TooltipContent>Augmented images</TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                  </TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead className="w-12"></TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {dataset.products?.map((product: ProductWithFrameCounts) => (
                  <TableRow key={product.id}>
                    <TableCell>
                      <div className="w-10 h-10 bg-gray-100 rounded flex items-center justify-center">
                        {product.primary_image_url ? (
                          <img
                            src={product.primary_image_url}
                            alt=""
                            className="w-full h-full object-cover rounded"
                          />
                        ) : (
                          <Package className="h-5 w-5 text-gray-400" />
                        )}
                      </div>
                    </TableCell>
                    <TableCell className="font-mono text-sm">
                      {product.barcode}
                    </TableCell>
                    <TableCell>{product.brand_name || "-"}</TableCell>
                    <TableCell className="max-w-[200px] truncate">
                      {product.product_name || "-"}
                    </TableCell>
                    <TableCell className="text-center">
                      <span className="text-blue-600 font-medium">
                        {product.frame_counts?.synthetic || 0}
                      </span>
                    </TableCell>
                    <TableCell className="text-center">
                      <span className="text-green-600 font-medium">
                        {product.frame_counts?.real || 0}
                      </span>
                    </TableCell>
                    <TableCell className="text-center">
                      <span className="text-purple-600 font-medium">
                        {product.frame_counts?.augmented || 0}
                      </span>
                    </TableCell>
                    <TableCell>
                      <Badge
                        variant="outline"
                        className={
                          product.status === "ready"
                            ? "bg-green-50 text-green-700"
                            : ""
                        }
                      >
                        {product.status}
                      </Badge>
                    </TableCell>
                    <TableCell>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="text-red-500 hover:text-red-600 hover:bg-red-50"
                        onClick={() => removeProductMutation.mutate(product.id)}
                        disabled={removeProductMutation.isPending}
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
