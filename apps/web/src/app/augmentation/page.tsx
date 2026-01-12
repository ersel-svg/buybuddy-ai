"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
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
  Play,
  Loader2,
  CheckCircle,
  XCircle,
  Clock,
  Sparkles,
  RefreshCw,
  Image as ImageIcon,
  Layers,
} from "lucide-react";
import type { Job, Dataset } from "@/types";

interface AugmentationConfig {
  dataset_id: string;
  syn_per_class: number;
  real_per_class: number;
  use_light_aug: boolean;
  use_heavy_aug: boolean;
  use_real_aug: boolean;
  background_removal: boolean;
  color_jitter: boolean;
  geometric_transforms: boolean;
}

const DEFAULT_CONFIG: AugmentationConfig = {
  dataset_id: "",
  syn_per_class: 100,
  real_per_class: 50,
  use_light_aug: true,
  use_heavy_aug: true,
  use_real_aug: true,
  background_removal: true,
  color_jitter: true,
  geometric_transforms: true,
};

export default function AugmentationPage() {
  const queryClient = useQueryClient();
  const [activeTab, setActiveTab] = useState("new");
  const [config, setConfig] = useState<AugmentationConfig>(DEFAULT_CONFIG);

  // Fetch datasets
  const { data: datasets } = useQuery({
    queryKey: ["datasets"],
    queryFn: () => apiClient.getDatasets(),
  });

  // Fetch augmentation jobs
  const { data: jobs, isLoading: isLoadingJobs } = useQuery({
    queryKey: ["jobs", "augmentation"],
    queryFn: () => apiClient.getJobs("augmentation"),
    refetchInterval: 5000,
  });

  // Start augmentation mutation
  const startMutation = useMutation({
    mutationFn: (config: AugmentationConfig) =>
      apiClient.startAugmentation(config.dataset_id, {
        syn_per_class: config.syn_per_class,
        real_per_class: config.real_per_class,
        use_light_aug: config.use_light_aug,
        use_heavy_aug: config.use_heavy_aug,
        use_real_aug: config.use_real_aug,
        background_removal: config.background_removal,
        color_jitter: config.color_jitter,
        geometric_transforms: config.geometric_transforms,
      }),
    onSuccess: () => {
      toast.success("Augmentation job started");
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
      setActiveTab("jobs");
    },
    onError: () => {
      toast.error("Failed to start augmentation");
    },
  });

  // Status config
  const statusConfig: Record<
    string,
    { icon: React.ReactNode; color: string; label: string }
  > = {
    pending: {
      icon: <Clock className="h-4 w-4" />,
      color: "bg-yellow-100 text-yellow-800",
      label: "Pending",
    },
    queued: {
      icon: <Clock className="h-4 w-4" />,
      color: "bg-yellow-100 text-yellow-800",
      label: "Queued",
    },
    running: {
      icon: <Loader2 className="h-4 w-4 animate-spin" />,
      color: "bg-blue-100 text-blue-800",
      label: "Running",
    },
    completed: {
      icon: <CheckCircle className="h-4 w-4" />,
      color: "bg-green-100 text-green-800",
      label: "Completed",
    },
    failed: {
      icon: <XCircle className="h-4 w-4" />,
      color: "bg-red-100 text-red-800",
      label: "Failed",
    },
  };

  const runningJobs = jobs?.filter((j: Job) => j.status === "running").length || 0;
  const selectedDataset = datasets?.find((d) => d.id === config.dataset_id);

  // Calculate estimated output
  const estimatedSyntheticImages = selectedDataset
    ? selectedDataset.product_count * config.syn_per_class
    : 0;
  const estimatedRealImages = selectedDataset
    ? selectedDataset.product_count * config.real_per_class
    : 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold">Augmentation</h1>
        <p className="text-gray-500">
          Generate augmented images for training using BiRefNet + Albumentations
        </p>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList>
          <TabsTrigger value="new">New Augmentation</TabsTrigger>
          <TabsTrigger value="jobs">
            Jobs {runningJobs > 0 && `(${runningJobs} running)`}
          </TabsTrigger>
        </TabsList>

        {/* New Augmentation Tab */}
        <TabsContent value="new" className="mt-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Left: Configuration */}
            <div className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Sparkles className="h-5 w-5" />
                    Augmentation Configuration
                  </CardTitle>
                  <CardDescription>
                    Configure how images will be augmented
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  {/* Dataset Selection */}
                  <div className="space-y-2">
                    <Label>Dataset</Label>
                    <Select
                      value={config.dataset_id}
                      onValueChange={(value) =>
                        setConfig({ ...config, dataset_id: value })
                      }
                    >
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
                  </div>

                  <Separator />

                  {/* Image Count Settings */}
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <Label>
                        Synthetic images per product: {config.syn_per_class}
                      </Label>
                      <Slider
                        value={[config.syn_per_class]}
                        onValueChange={([value]) =>
                          setConfig({ ...config, syn_per_class: value })
                        }
                        min={10}
                        max={200}
                        step={10}
                      />
                      <p className="text-xs text-gray-500">
                        Total: ~{estimatedSyntheticImages.toLocaleString()}{" "}
                        images
                      </p>
                    </div>

                    <div className="space-y-2">
                      <Label>
                        Real images per product: {config.real_per_class}
                      </Label>
                      <Slider
                        value={[config.real_per_class]}
                        onValueChange={([value]) =>
                          setConfig({ ...config, real_per_class: value })
                        }
                        min={10}
                        max={100}
                        step={10}
                      />
                      <p className="text-xs text-gray-500">
                        Total: ~{estimatedRealImages.toLocaleString()} images
                      </p>
                    </div>
                  </div>

                  <Separator />

                  {/* Augmentation Pipelines */}
                  <div className="space-y-4">
                    <Label>Augmentation Pipelines</Label>
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="font-medium">Light Augmentation</p>
                          <p className="text-xs text-gray-500">
                            Subtle color and geometric changes
                          </p>
                        </div>
                        <Switch
                          checked={config.use_light_aug}
                          onCheckedChange={(checked) =>
                            setConfig({ ...config, use_light_aug: checked })
                          }
                        />
                      </div>
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="font-medium">Heavy Augmentation</p>
                          <p className="text-xs text-gray-500">
                            Strong distortions and noise
                          </p>
                        </div>
                        <Switch
                          checked={config.use_heavy_aug}
                          onCheckedChange={(checked) =>
                            setConfig({ ...config, use_heavy_aug: checked })
                          }
                        />
                      </div>
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="font-medium">Real Image Augmentation</p>
                          <p className="text-xs text-gray-500">
                            Augment matched real images
                          </p>
                        </div>
                        <Switch
                          checked={config.use_real_aug}
                          onCheckedChange={(checked) =>
                            setConfig({ ...config, use_real_aug: checked })
                          }
                        />
                      </div>
                    </div>
                  </div>

                  <Separator />

                  {/* Transform Options */}
                  <div className="space-y-4">
                    <Label>Transform Options</Label>
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="font-medium">Background Removal</p>
                          <p className="text-xs text-gray-500">
                            Use BiRefNet for segmentation
                          </p>
                        </div>
                        <Switch
                          checked={config.background_removal}
                          onCheckedChange={(checked) =>
                            setConfig({ ...config, background_removal: checked })
                          }
                        />
                      </div>
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="font-medium">Color Jitter</p>
                          <p className="text-xs text-gray-500">
                            Brightness, contrast, saturation
                          </p>
                        </div>
                        <Switch
                          checked={config.color_jitter}
                          onCheckedChange={(checked) =>
                            setConfig({ ...config, color_jitter: checked })
                          }
                        />
                      </div>
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="font-medium">Geometric Transforms</p>
                          <p className="text-xs text-gray-500">
                            Rotation, flip, perspective
                          </p>
                        </div>
                        <Switch
                          checked={config.geometric_transforms}
                          onCheckedChange={(checked) =>
                            setConfig({
                              ...config,
                              geometric_transforms: checked,
                            })
                          }
                        />
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Right: Summary & Start */}
            <div className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle>Augmentation Summary</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-500">Dataset</span>
                    <span className="font-medium">
                      {selectedDataset?.name || "Not selected"}
                    </span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-500">Products</span>
                    <span className="font-medium">
                      {selectedDataset?.product_count || 0}
                    </span>
                  </div>

                  <Separator className="my-4" />

                  <div className="flex justify-between text-sm">
                    <span className="text-gray-500">Est. Synthetic Images</span>
                    <span className="font-medium">
                      {estimatedSyntheticImages.toLocaleString()}
                    </span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-500">Est. Real Images</span>
                    <span className="font-medium">
                      {estimatedRealImages.toLocaleString()}
                    </span>
                  </div>
                  <div className="flex justify-between text-sm font-medium">
                    <span className="text-gray-500">Total Output</span>
                    <span className="text-blue-600">
                      ~
                      {(
                        estimatedSyntheticImages + estimatedRealImages
                      ).toLocaleString()}{" "}
                      images
                    </span>
                  </div>

                  <Separator className="my-4" />

                  <div className="flex justify-between text-sm">
                    <span className="text-gray-500">Pipelines Active</span>
                    <span className="font-medium">
                      {[
                        config.use_light_aug && "Light",
                        config.use_heavy_aug && "Heavy",
                        config.use_real_aug && "Real",
                      ]
                        .filter(Boolean)
                        .join(", ") || "None"}
                    </span>
                  </div>

                  <Separator className="my-4" />

                  <Button
                    className="w-full"
                    size="lg"
                    onClick={() => startMutation.mutate(config)}
                    disabled={!config.dataset_id || startMutation.isPending}
                  >
                    {startMutation.isPending ? (
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    ) : (
                      <Sparkles className="h-4 w-4 mr-2" />
                    )}
                    Start Augmentation
                  </Button>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <ImageIcon className="h-5 w-5" />
                    Pipeline Info
                  </CardTitle>
                </CardHeader>
                <CardContent className="text-sm text-gray-600 space-y-2">
                  <p>
                    <strong>Light:</strong> Subtle changes that preserve product
                    identity - small rotations, brightness adjustments
                  </p>
                  <p>
                    <strong>Heavy:</strong> Strong augmentations for
                    robustness - blur, noise, cutout, color distortion
                  </p>
                  <p>
                    <strong>Real:</strong> Apply augmentations to matched real
                    images to increase variety
                  </p>
                  <p className="pt-2 text-gray-500">
                    Based on final_augmentor_v3.py with BiRefNet segmentation
                  </p>
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
                onClick={() =>
                  queryClient.invalidateQueries({ queryKey: ["jobs"] })
                }
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
                  <p className="text-sm mt-1">
                    Start an augmentation to generate training images
                  </p>
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
                            <p className="text-xs text-gray-500 mt-1">
                              {job.progress}%
                            </p>
                          </div>
                        </TableCell>
                        <TableCell>
                          <Badge className={statusConfig[job.status]?.color}>
                            <span className="mr-1">
                              {statusConfig[job.status]?.icon}
                            </span>
                            {statusConfig[job.status]?.label}
                          </Badge>
                        </TableCell>
                        <TableCell className="text-gray-500 text-sm">
                          {new Date(job.created_at).toLocaleString()}
                        </TableCell>
                        <TableCell>
                          {job.result && (
                            <span className="text-sm">
                              {(job.result as { total_images?: number })
                                .total_images || 0}{" "}
                              images
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
