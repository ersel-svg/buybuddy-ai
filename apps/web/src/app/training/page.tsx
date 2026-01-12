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
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
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
  Download,
  Brain,
  Clock,
  RefreshCw,
  Star,
  Zap,
} from "lucide-react";
import type { TrainingConfig, TrainingJob, ModelArtifact } from "@/types";

// Default config matching train_optimized_v14.sh
const DEFAULT_CONFIG: TrainingConfig = {
  dataset_id: "",
  model_name: "facebook/dinov2-large",
  proj_dim: 512,
  epochs: 30,
  batch_size: 32,
  learning_rate: 0.0001,
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
};

export default function TrainingPage() {
  const queryClient = useQueryClient();
  const [activeTab, setActiveTab] = useState("new");
  const [config, setConfig] = useState<TrainingConfig>(DEFAULT_CONFIG);

  // Fetch datasets
  const { data: datasets } = useQuery({
    queryKey: ["datasets"],
    queryFn: () => apiClient.getDatasets(),
  });

  // Fetch training jobs
  const { data: trainingJobs, isLoading: isLoadingJobs } = useQuery({
    queryKey: ["training-jobs"],
    queryFn: () => apiClient.getTrainingJobs(),
    refetchInterval: 5000,
  });

  // Fetch models
  const { data: models, isLoading: isLoadingModels } = useQuery({
    queryKey: ["training-models"],
    queryFn: () => apiClient.getTrainingModels(),
  });

  // Start training mutation
  const startTrainingMutation = useMutation({
    mutationFn: (config: TrainingConfig) => apiClient.startTrainingJob(config),
    onSuccess: () => {
      toast.success("Training job started");
      queryClient.invalidateQueries({ queryKey: ["training-jobs"] });
      setActiveTab("jobs");
    },
    onError: () => {
      toast.error("Failed to start training");
    },
  });

  // Activate model mutation
  const activateModelMutation = useMutation({
    mutationFn: (modelId: string) => apiClient.activateModel(modelId),
    onSuccess: () => {
      toast.success("Model activated");
      queryClient.invalidateQueries({ queryKey: ["training-models"] });
    },
    onError: () => {
      toast.error("Failed to activate model");
    },
  });

  // Handle model selection auto-adjust
  const handleModelChange = (modelName: TrainingConfig["model_name"]) => {
    if (modelName === "facebook/dinov2-large") {
      setConfig({
        ...config,
        model_name: modelName,
        proj_dim: 512,
        batch_size: 16,
        learning_rate: 0.00002,
        warmup_epochs: 5,
        grad_clip: 0.5,
      });
    } else {
      setConfig({
        ...config,
        model_name: modelName,
        proj_dim: 512,
        batch_size: 32,
        learning_rate: 0.0001,
        warmup_epochs: 3,
        grad_clip: 1.0,
      });
    }
  };

  // Status badges
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

  const runningJobs =
    trainingJobs?.filter((j) => j.status === "running").length || 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold">Training</h1>
        <p className="text-gray-500">
          Train DINOv2 + ArcFace models on your datasets
        </p>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList>
          <TabsTrigger value="new">New Training</TabsTrigger>
          <TabsTrigger value="jobs">
            Jobs {runningJobs > 0 && `(${runningJobs} running)`}
          </TabsTrigger>
          <TabsTrigger value="models">Models ({models?.length || 0})</TabsTrigger>
        </TabsList>

        {/* New Training Tab */}
        <TabsContent value="new" className="mt-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Left: Configuration */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Brain className="h-5 w-5" />
                  Model Configuration
                </CardTitle>
                <CardDescription>
                  Configure training parameters
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
                      {datasets?.map((dataset) => (
                        <SelectItem key={dataset.id} value={dataset.id}>
                          {dataset.name} ({dataset.product_count} products)
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                {/* Model Selection */}
                <div className="space-y-2">
                  <Label>Model</Label>
                  <Select
                    value={config.model_name}
                    onValueChange={(value) =>
                      handleModelChange(value as TrainingConfig["model_name"])
                    }
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="facebook/dinov2-large">
                        DINOv2-LARGE (304M) - Higher accuracy
                      </SelectItem>
                      <SelectItem value="facebook/dinov2-base">
                        DINOv2-BASE (86M) - Faster training
                      </SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <Separator />

                {/* Training Parameters */}
                <div className="space-y-4">
                  <div className="space-y-2">
                    <Label>Epochs: {config.epochs}</Label>
                    <Slider
                      value={[config.epochs]}
                      onValueChange={([value]) =>
                        setConfig({ ...config, epochs: value })
                      }
                      min={5}
                      max={100}
                      step={5}
                    />
                  </div>

                  <div className="space-y-2">
                    <Label>Batch Size: {config.batch_size}</Label>
                    <Slider
                      value={[config.batch_size]}
                      onValueChange={([value]) =>
                        setConfig({ ...config, batch_size: value })
                      }
                      min={8}
                      max={64}
                      step={8}
                    />
                  </div>

                  <div className="space-y-2">
                    <Label>
                      Learning Rate: {config.learning_rate.toExponential(1)}
                    </Label>
                    <Slider
                      value={[Math.log10(config.learning_rate)]}
                      onValueChange={([value]) =>
                        setConfig({ ...config, learning_rate: Math.pow(10, value) })
                      }
                      min={-5}
                      max={-3}
                      step={0.1}
                    />
                  </div>
                </div>

                <Separator />

                {/* Switches */}
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <Label>Mixed Precision</Label>
                      <p className="text-xs text-gray-500">
                        Faster training with FP16
                      </p>
                    </div>
                    <Switch
                      checked={config.use_mixed_precision}
                      onCheckedChange={(checked) =>
                        setConfig({ ...config, use_mixed_precision: checked })
                      }
                    />
                  </div>
                  <div className="flex items-center justify-between">
                    <div>
                      <Label>Hard Negative Mining</Label>
                      <p className="text-xs text-gray-500">
                        Better embeddings
                      </p>
                    </div>
                    <Switch
                      checked={config.use_hardest_negatives}
                      onCheckedChange={(checked) =>
                        setConfig({ ...config, use_hardest_negatives: checked })
                      }
                    />
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Right: Summary & Start */}
            <div className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle>Training Summary</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-500">Dataset</span>
                    <span className="font-medium">
                      {datasets?.find((d) => d.id === config.dataset_id)?.name ||
                        "Not selected"}
                    </span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-500">Model</span>
                    <span className="font-medium">
                      {config.model_name.split("/")[1]}
                    </span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-500">Epochs</span>
                    <span className="font-medium">{config.epochs}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-500">Batch Size</span>
                    <span className="font-medium">{config.batch_size}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-500">Mixed Precision</span>
                    <span className="font-medium">
                      {config.use_mixed_precision ? "Yes" : "No"}
                    </span>
                  </div>

                  <Separator className="my-4" />

                  <Button
                    className="w-full"
                    size="lg"
                    onClick={() => startTrainingMutation.mutate(config)}
                    disabled={
                      !config.dataset_id || startTrainingMutation.isPending
                    }
                  >
                    {startTrainingMutation.isPending ? (
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    ) : (
                      <Play className="h-4 w-4 mr-2" />
                    )}
                    Start Training
                  </Button>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Zap className="h-5 w-5" />
                    Quick Tips
                  </CardTitle>
                </CardHeader>
                <CardContent className="text-sm text-gray-600 space-y-2">
                  <p>
                    <strong>DINOv2-LARGE:</strong> Best for production, 2x
                    slower but +3-5% accuracy
                  </p>
                  <p>
                    <strong>Mixed Precision:</strong> Enable for faster training
                    on RTX cards
                  </p>
                  <p>
                    <strong>Epochs:</strong> 30 epochs usually sufficient,
                    monitor loss curve
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
                <CardTitle>Training Jobs</CardTitle>
                <CardDescription>
                  Monitor your training runs
                </CardDescription>
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={() =>
                  queryClient.invalidateQueries({ queryKey: ["training-jobs"] })
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
              ) : trainingJobs?.length === 0 ? (
                <div className="text-center py-8 text-gray-500">
                  <Brain className="h-12 w-12 mx-auto mb-2 opacity-50" />
                  <p>No training jobs yet</p>
                </div>
              ) : (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Dataset</TableHead>
                      <TableHead>Model</TableHead>
                      <TableHead>Progress</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead>Started</TableHead>
                      <TableHead></TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {trainingJobs?.map((job: TrainingJob) => (
                      <TableRow key={job.id}>
                        <TableCell className="font-medium">
                          {job.dataset_name || job.dataset_id}
                        </TableCell>
                        <TableCell>
                          {(job.config?.model_name as string)?.split("/")[1] || "DINOv2"}
                        </TableCell>
                        <TableCell>
                          <div className="w-32">
                            <Progress value={job.progress} className="h-2" />
                            <p className="text-xs text-gray-500 mt-1">
                              {job.epochs_completed || 0}/{job.epochs} epochs
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
                          {new Date(job.created_at).toLocaleDateString()}
                        </TableCell>
                        <TableCell>
                          {job.checkpoint_url && (
                            <Button variant="ghost" size="sm">
                              <Download className="h-4 w-4" />
                            </Button>
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

        {/* Models Tab */}
        <TabsContent value="models" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle>Trained Models</CardTitle>
              <CardDescription>
                Manage your trained model checkpoints
              </CardDescription>
            </CardHeader>
            <CardContent>
              {isLoadingModels ? (
                <div className="flex items-center justify-center py-8">
                  <Loader2 className="h-6 w-6 animate-spin" />
                </div>
              ) : models?.length === 0 ? (
                <div className="text-center py-8 text-gray-500">
                  <Brain className="h-12 w-12 mx-auto mb-2 opacity-50" />
                  <p>No models trained yet</p>
                  <p className="text-sm mt-1">
                    Start a training job to create your first model
                  </p>
                </div>
              ) : (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Name</TableHead>
                      <TableHead>Version</TableHead>
                      <TableHead>Embedding Dim</TableHead>
                      <TableHead>Classes</TableHead>
                      <TableHead>Final Loss</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead></TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {models?.map((model: ModelArtifact) => (
                      <TableRow key={model.id}>
                        <TableCell className="font-medium">
                          {model.name}
                        </TableCell>
                        <TableCell>{model.version}</TableCell>
                        <TableCell>{model.embedding_dim}</TableCell>
                        <TableCell>{model.num_classes}</TableCell>
                        <TableCell>{model.final_loss?.toFixed(4)}</TableCell>
                        <TableCell>
                          {model.is_active ? (
                            <Badge className="bg-green-100 text-green-800">
                              <Star className="h-3 w-3 mr-1" />
                              Active
                            </Badge>
                          ) : (
                            <Badge variant="outline">Inactive</Badge>
                          )}
                        </TableCell>
                        <TableCell>
                          <div className="flex gap-2">
                            {!model.is_active && (
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() =>
                                  activateModelMutation.mutate(model.id)
                                }
                              >
                                Activate
                              </Button>
                            )}
                            <Button variant="ghost" size="sm">
                              <Download className="h-4 w-4" />
                            </Button>
                          </div>
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
