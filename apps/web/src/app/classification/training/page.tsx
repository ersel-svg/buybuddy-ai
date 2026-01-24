"use client";

import { useState } from "react";
import Link from "next/link";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { useRouter } from "next/navigation";
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
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  RefreshCw,
  Loader2,
  Brain,
  Plus,
  Trash2,
  Play,
  Square,
  BarChart3,
  Clock,
  CheckCircle,
  XCircle,
  AlertCircle,
  MoreHorizontal,
  Star,
  Download,
  Power,
  PowerOff,
  Eye,
  Search,
  Cpu,
} from "lucide-react";
import { DataLoadingConfigPanel } from "@/components/training/DataLoadingConfig";
import type { DataLoadingConfig } from "@/types";

interface TrainingRun {
  id: string;
  name: string;
  description?: string;
  status: string;
  dataset_id: string;
  model_type: string;
  model_size: string;
  current_epoch: number;
  total_epochs: number;
  best_accuracy?: number;
  best_f1?: number;
  created_at: string;
  started_at?: string;
  completed_at?: string;
}

interface TrainedModel {
  id: string;
  name: string;
  model_type: string;
  model_size?: string;
  task_type: string;
  accuracy?: number;
  f1_score?: number;
  num_classes: number;
  is_active: boolean;
  is_default: boolean;
  created_at: string;
}

const MODEL_TYPES = [
  { value: "vit", label: "Vision Transformer (ViT)", sizes: ["tiny", "small", "base", "large"] },
  { value: "convnext", label: "ConvNeXt", sizes: ["tiny", "small", "base", "large"] },
  { value: "efficientnet", label: "EfficientNet", sizes: ["s", "m", "l"] },
  { value: "swin", label: "Swin Transformer", sizes: ["tiny", "small", "base"] },
  { value: "dinov2", label: "DINOv2", sizes: ["small", "base", "large"] },
  { value: "clip", label: "CLIP", sizes: ["vit-b-16", "vit-l-14"] },
];

const AUGMENTATION_PRESETS = [
  { value: "sota", label: "SOTA (Recommended)", description: "State-of-the-art augmentations" },
  { value: "heavy", label: "Heavy", description: "Strong augmentations for difficult data" },
  { value: "medium", label: "Medium", description: "Balanced augmentation strength" },
  { value: "light", label: "Light", description: "Minimal augmentations" },
  { value: "none", label: "None", description: "No augmentations" },
];

export default function CLSTrainingPage() {
  const router = useRouter();
  const queryClient = useQueryClient();

  // Tab state
  const [activeTab, setActiveTab] = useState<"runs" | "models">("runs");

  // Create dialog state
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false);
  const [newRunName, setNewRunName] = useState("");
  const [newRunDescription, setNewRunDescription] = useState("");
  const [selectedDatasetId, setSelectedDatasetId] = useState("");
  const [selectedModelType, setSelectedModelType] = useState("vit");
  const [selectedModelSize, setSelectedModelSize] = useState("base");
  const [epochs, setEpochs] = useState(100);
  const [batchSize, setBatchSize] = useState(32);
  const [learningRate, setLearningRate] = useState(0.0001);
  const [augmentationPreset, setAugmentationPreset] = useState("sota");
  const [dataLoadingConfig, setDataLoadingConfig] = useState<DataLoadingConfig | undefined>(undefined);

  // Models tab state
  const [modelsSearchQuery, setModelsSearchQuery] = useState("");
  const [modelsFilterType, setModelsFilterType] = useState<string>("all");

  // Fetch training runs
  const { data: trainingRuns, isLoading, isFetching, refetch } = useQuery({
    queryKey: ["cls-training-runs"],
    queryFn: () => apiClient.getCLSTrainingRuns(),
    staleTime: 10000,
    refetchInterval: 15000,
  });

  // Fetch datasets
  const { data: datasets } = useQuery({
    queryKey: ["cls-datasets"],
    queryFn: () => apiClient.getCLSDatasets(),
    staleTime: 60000,
  });

  // Fetch stats
  const { data: stats } = useQuery({
    queryKey: ["cls-stats"],
    queryFn: () => apiClient.getCLSStats(),
    staleTime: 30000,
  });

  // Fetch trained models
  const { data: trainedModels, isLoading: isLoadingModels, isFetching: isFetchingModels } = useQuery({
    queryKey: ["cls-models"],
    queryFn: () => apiClient.getCLSModels(),
    staleTime: 30000,
    enabled: activeTab === "models",
  });

  // Create training run mutation
  const createMutation = useMutation({
    mutationFn: async () => {
      return apiClient.createCLSTrainingRun({
        name: newRunName,
        description: newRunDescription || undefined,
        dataset_id: selectedDatasetId,
        config: {
          model_type: selectedModelType as "vit" | "convnext" | "efficientnet" | "swin" | "dinov2" | "clip",
          model_size: selectedModelSize,
          epochs,
          batch_size: batchSize,
          learning_rate: learningRate,
          augmentation_preset: augmentationPreset as "sota" | "heavy" | "medium" | "light" | "none",
          use_ema: true,
          mixed_precision: true,
          early_stopping: true,
          data_loading: dataLoadingConfig,
        },
      });
    },
    onSuccess: () => {
      toast.success("Training run created");
      queryClient.invalidateQueries({ queryKey: ["cls-training-runs"] });
      queryClient.invalidateQueries({ queryKey: ["cls-stats"] });
      setIsCreateDialogOpen(false);
      resetCreateForm();
    },
    onError: (error) => {
      toast.error(`Failed to create training run: ${error.message}`);
    },
  });

  // Cancel training mutation
  const cancelMutation = useMutation({
    mutationFn: async (id: string) => {
      return apiClient.cancelCLSTrainingRun(id);
    },
    onSuccess: () => {
      toast.success("Training cancelled");
      queryClient.invalidateQueries({ queryKey: ["cls-training-runs"] });
    },
    onError: (error) => {
      toast.error(`Failed to cancel: ${error.message}`);
    },
  });

  // Delete training mutation
  const deleteMutation = useMutation({
    mutationFn: async (id: string) => {
      return apiClient.deleteCLSTrainingRun(id);
    },
    onSuccess: () => {
      toast.success("Training run deleted");
      queryClient.invalidateQueries({ queryKey: ["cls-training-runs"] });
      queryClient.invalidateQueries({ queryKey: ["cls-stats"] });
    },
    onError: (error) => {
      toast.error(`Failed to delete: ${error.message}`);
    },
  });

  // Model mutations
  const activateModelMutation = useMutation({
    mutationFn: (id: string) => apiClient.activateCLSModel(id),
    onSuccess: () => {
      toast.success("Model activated");
      queryClient.invalidateQueries({ queryKey: ["cls-models"] });
    },
    onError: (error) => toast.error(`Failed to activate model: ${error.message}`),
  });

  const deactivateModelMutation = useMutation({
    mutationFn: (id: string) => apiClient.deactivateCLSModel(id),
    onSuccess: () => {
      toast.success("Model deactivated");
      queryClient.invalidateQueries({ queryKey: ["cls-models"] });
    },
    onError: (error) => toast.error(`Failed to deactivate model: ${error.message}`),
  });

  const setDefaultModelMutation = useMutation({
    mutationFn: (id: string) => apiClient.setDefaultCLSModel(id),
    onSuccess: () => {
      toast.success("Default model set");
      queryClient.invalidateQueries({ queryKey: ["cls-models"] });
    },
    onError: (error) => toast.error(`Failed to set default: ${error.message}`),
  });

  const deleteModelMutation = useMutation({
    mutationFn: (id: string) => apiClient.deleteCLSModel(id),
    onSuccess: () => {
      toast.success("Model deleted");
      queryClient.invalidateQueries({ queryKey: ["cls-models"] });
      queryClient.invalidateQueries({ queryKey: ["cls-stats"] });
    },
    onError: (error) => toast.error(`Failed to delete model: ${error.message}`),
  });

  const resetCreateForm = () => {
    setNewRunName("");
    setNewRunDescription("");
    setSelectedDatasetId("");
    setSelectedModelType("vit");
    setSelectedModelSize("base");
    setEpochs(100);
    setBatchSize(32);
    setLearningRate(0.0001);
    setAugmentationPreset("sota");
    setDataLoadingConfig(undefined);
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "completed":
        return (
          <Badge className="bg-green-600">
            <CheckCircle className="h-3 w-3 mr-1" />
            Completed
          </Badge>
        );
      case "training":
        return (
          <Badge className="bg-blue-600">
            <Loader2 className="h-3 w-3 mr-1 animate-spin" />
            Training
          </Badge>
        );
      case "preparing":
      case "queued":
        return (
          <Badge className="bg-yellow-600">
            <Clock className="h-3 w-3 mr-1" />
            {status === "preparing" ? "Preparing" : "Queued"}
          </Badge>
        );
      case "failed":
        return (
          <Badge variant="destructive">
            <XCircle className="h-3 w-3 mr-1" />
            Failed
          </Badge>
        );
      case "cancelled":
        return (
          <Badge variant="secondary">
            <Square className="h-3 w-3 mr-1" />
            Cancelled
          </Badge>
        );
      default:
        return (
          <Badge variant="outline">
            <AlertCircle className="h-3 w-3 mr-1" />
            {status}
          </Badge>
        );
    }
  };

  const availableSizes = MODEL_TYPES.find((m) => m.value === selectedModelType)?.sizes || [];

  // Filter models
  const filteredModels = trainedModels?.filter((model: TrainedModel) => {
    const matchesSearch = model.name.toLowerCase().includes(modelsSearchQuery.toLowerCase());
    const matchesType = modelsFilterType === "all" || model.model_type === modelsFilterType;
    return matchesSearch && matchesType;
  }) ?? [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold">Training</h1>
          <p className="text-muted-foreground">
            Train classification models with SOTA architectures
          </p>
        </div>
        <div className="flex gap-2">
          <Button
            variant="outline"
            onClick={() => {
              refetch();
              queryClient.invalidateQueries({ queryKey: ["cls-models"] });
            }}
            disabled={isFetching || isFetchingModels}
          >
            <RefreshCw className={`h-4 w-4 mr-2 ${(isFetching || isFetchingModels) ? "animate-spin" : ""}`} />
            Refresh
          </Button>
          <Button onClick={() => setIsCreateDialogOpen(true)}>
            <Plus className="h-4 w-4 mr-2" />
            New Training
          </Button>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">Total Runs</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">{trainingRuns?.length ?? 0}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">Active Training</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-blue-600">
              {stats?.active_training_runs ?? 0}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">Trained Models</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-green-600">
              {stats?.total_models ?? 0}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">Best Accuracy</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-purple-600">
              {trainingRuns && trainingRuns.length > 0
                ? `${trainingRuns.reduce((best: number, run: TrainingRun) => Math.max(best, run.best_accuracy ?? 0), 0).toFixed(1)}%`
                : "-"}
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Tabs */}
      <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as typeof activeTab)}>
        <TabsList>
          <TabsTrigger value="runs">
            Training Runs ({trainingRuns?.length ?? 0})
          </TabsTrigger>
          <TabsTrigger value="models">
            Trained Models ({trainedModels?.length ?? 0})
          </TabsTrigger>
        </TabsList>

        {/* Training Runs Tab */}
        <TabsContent value="runs" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle>Training Runs</CardTitle>
              <CardDescription>
                {trainingRuns?.length ?? 0} training runs
              </CardDescription>
            </CardHeader>
            <CardContent>
              {isLoading ? (
                <div className="flex items-center justify-center py-12">
                  <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
                </div>
              ) : trainingRuns?.length === 0 ? (
                <div className="text-center py-12">
                  <Brain className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                  <h3 className="text-lg font-medium">No training runs</h3>
                  <p className="text-muted-foreground mt-1">
                    Start a new training run to train classification models
                  </p>
                  <Button className="mt-4" onClick={() => setIsCreateDialogOpen(true)}>
                    <Plus className="h-4 w-4 mr-2" />
                    New Training
                  </Button>
                </div>
              ) : (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Name</TableHead>
                      <TableHead>Model</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead>Progress</TableHead>
                      <TableHead className="text-right">Accuracy</TableHead>
                      <TableHead className="text-right">F1</TableHead>
                      <TableHead>Created</TableHead>
                      <TableHead className="w-24"></TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {trainingRuns?.map((run: TrainingRun) => (
                      <TableRow
                        key={run.id}
                        className="cursor-pointer hover:bg-muted/50"
                        onClick={() => router.push(`/classification/training/${run.id}`)}
                      >
                        <TableCell className="font-medium">{run.name}</TableCell>
                        <TableCell>
                          <div className="text-sm">
                            <span className="font-medium">{run.model_type.toUpperCase()}</span>
                            <span className="text-muted-foreground ml-1">({run.model_size})</span>
                          </div>
                        </TableCell>
                        <TableCell>{getStatusBadge(run.status)}</TableCell>
                        <TableCell>
                          <div className="flex items-center gap-2">
                            <Progress
                              value={(run.current_epoch / run.total_epochs) * 100}
                              className="h-2 w-20"
                            />
                            <span className="text-sm text-muted-foreground">
                              {run.current_epoch}/{run.total_epochs}
                            </span>
                          </div>
                        </TableCell>
                        <TableCell className="text-right">
                          {run.best_accuracy ? `${run.best_accuracy.toFixed(1)}%` : "-"}
                        </TableCell>
                        <TableCell className="text-right">
                          {run.best_f1 ? run.best_f1.toFixed(3) : "-"}
                        </TableCell>
                        <TableCell className="text-muted-foreground">
                          {new Date(run.created_at).toLocaleDateString()}
                        </TableCell>
                        <TableCell>
                          <div className="flex gap-1" onClick={(e) => e.stopPropagation()}>
                            <Button
                              variant="ghost"
                              size="icon"
                              onClick={() => router.push(`/classification/training/${run.id}`)}
                            >
                              <Eye className="h-4 w-4" />
                            </Button>
                            {(run.status === "training" || run.status === "preparing" || run.status === "queued") && (
                              <Button
                                variant="ghost"
                                size="icon"
                                onClick={() => {
                                  if (confirm("Cancel this training run?")) {
                                    cancelMutation.mutate(run.id);
                                  }
                                }}
                              >
                                <Square className="h-4 w-4" />
                              </Button>
                            )}
                            <Button
                              variant="ghost"
                              size="icon"
                              onClick={() => {
                                if (confirm("Delete this training run?")) {
                                  deleteMutation.mutate(run.id);
                                }
                              }}
                            >
                              <Trash2 className="h-4 w-4" />
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

        {/* Trained Models Tab */}
        <TabsContent value="models" className="mt-6">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Trained Models</CardTitle>
                  <CardDescription>
                    {trainedModels?.length ?? 0} trained models available
                  </CardDescription>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => queryClient.invalidateQueries({ queryKey: ["cls-models"] })}
                  disabled={isFetchingModels}
                >
                  <RefreshCw className={`h-4 w-4 ${isFetchingModels ? "animate-spin" : ""}`} />
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              {/* Filters */}
              <div className="flex gap-4 mb-6">
                <div className="relative flex-1 max-w-sm">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <Input
                    placeholder="Search models..."
                    value={modelsSearchQuery}
                    onChange={(e) => setModelsSearchQuery(e.target.value)}
                    className="pl-10"
                  />
                </div>
                <Select value={modelsFilterType} onValueChange={setModelsFilterType}>
                  <SelectTrigger className="w-[180px]">
                    <SelectValue placeholder="Model type" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Types</SelectItem>
                    {MODEL_TYPES.map((type) => (
                      <SelectItem key={type.value} value={type.value}>
                        {type.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {isLoadingModels ? (
                <div className="flex items-center justify-center py-12">
                  <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
                </div>
              ) : filteredModels.length === 0 ? (
                <div className="text-center py-12">
                  <Cpu className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                  <h3 className="text-lg font-medium">No trained models</h3>
                  <p className="text-muted-foreground mt-1">
                    Models will appear here after successful training runs
                  </p>
                </div>
              ) : (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Name</TableHead>
                      <TableHead>Model</TableHead>
                      <TableHead className="text-right">Accuracy</TableHead>
                      <TableHead className="text-right">F1 Score</TableHead>
                      <TableHead className="text-right">Classes</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead>Created</TableHead>
                      <TableHead className="w-12"></TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {filteredModels.map((model: TrainedModel) => (
                      <TableRow key={model.id}>
                        <TableCell>
                          <div className="flex items-center gap-2">
                            <span className="font-medium">{model.name}</span>
                            {model.is_default && (
                              <Badge variant="outline" className="text-yellow-600 border-yellow-600">
                                <Star className="h-3 w-3 mr-1" />
                                Default
                              </Badge>
                            )}
                          </div>
                        </TableCell>
                        <TableCell>
                          <div className="text-sm">
                            <span className="font-medium">{model.model_type.toUpperCase()}</span>
                            {model.model_size && <span className="text-muted-foreground ml-1">({model.model_size})</span>}
                          </div>
                        </TableCell>
                        <TableCell className="text-right font-mono">
                          {model.accuracy != null ? `${model.accuracy.toFixed(2)}%` : "-"}
                        </TableCell>
                        <TableCell className="text-right font-mono">
                          {model.f1_score != null ? model.f1_score.toFixed(4) : "-"}
                        </TableCell>
                        <TableCell className="text-right">
                          {model.num_classes}
                        </TableCell>
                        <TableCell>
                          {model.is_active ? (
                            <Badge className="bg-green-600">
                              <Power className="h-3 w-3 mr-1" />
                              Active
                            </Badge>
                          ) : (
                            <Badge variant="secondary">
                              <PowerOff className="h-3 w-3 mr-1" />
                              Inactive
                            </Badge>
                          )}
                        </TableCell>
                        <TableCell className="text-muted-foreground">
                          {new Date(model.created_at).toLocaleDateString()}
                        </TableCell>
                        <TableCell>
                          <DropdownMenu>
                            <DropdownMenuTrigger asChild>
                              <Button variant="ghost" size="icon">
                                <MoreHorizontal className="h-4 w-4" />
                              </Button>
                            </DropdownMenuTrigger>
                            <DropdownMenuContent align="end">
                              {model.is_active ? (
                                <DropdownMenuItem onClick={() => deactivateModelMutation.mutate(model.id)}>
                                  <PowerOff className="h-4 w-4 mr-2" />
                                  Deactivate
                                </DropdownMenuItem>
                              ) : (
                                <DropdownMenuItem onClick={() => activateModelMutation.mutate(model.id)}>
                                  <Power className="h-4 w-4 mr-2" />
                                  Activate
                                </DropdownMenuItem>
                              )}
                              {!model.is_default && (
                                <DropdownMenuItem onClick={() => setDefaultModelMutation.mutate(model.id)}>
                                  <Star className="h-4 w-4 mr-2" />
                                  Set as Default
                                </DropdownMenuItem>
                              )}
                              <DropdownMenuSeparator />
                              <DropdownMenuItem
                                onClick={async () => {
                                  try {
                                    const urls = await apiClient.getCLSModelDownloadUrls(model.id);
                                    if (urls.checkpoint_url) {
                                      window.open(urls.checkpoint_url, "_blank");
                                    } else {
                                      toast.error("No checkpoint available for download");
                                    }
                                  } catch (error) {
                                    toast.error("Failed to get download URL");
                                  }
                                }}
                              >
                                <Download className="h-4 w-4 mr-2" />
                                Download Checkpoint
                              </DropdownMenuItem>
                              <DropdownMenuItem
                                className="text-destructive"
                                onClick={() => {
                                  if (confirm("Delete this model?")) {
                                    deleteModelMutation.mutate(model.id);
                                  }
                                }}
                              >
                                <Trash2 className="h-4 w-4 mr-2" />
                                Delete
                              </DropdownMenuItem>
                            </DropdownMenuContent>
                          </DropdownMenu>
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

      {/* Create Training Dialog */}
      <Dialog open={isCreateDialogOpen} onOpenChange={setIsCreateDialogOpen}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>New Training Run</DialogTitle>
            <DialogDescription>
              Configure and start a new classification model training
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-6 py-4">
            {/* Basic Info */}
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <label className="text-sm font-medium">Name *</label>
                <Input
                  placeholder="e.g., ViT Base Experiment 1"
                  value={newRunName}
                  onChange={(e) => setNewRunName(e.target.value)}
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Dataset *</label>
                <Select value={selectedDatasetId} onValueChange={setSelectedDatasetId}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select dataset" />
                  </SelectTrigger>
                  <SelectContent>
                    {datasets?.map((dataset) => (
                      <SelectItem key={dataset.id} value={dataset.id}>
                        {dataset.name} ({dataset.labeled_image_count} labeled)
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Description</label>
              <Textarea
                placeholder="Optional description..."
                value={newRunDescription}
                onChange={(e) => setNewRunDescription(e.target.value)}
              />
            </div>

            {/* Model Selection */}
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <label className="text-sm font-medium">Model Type</label>
                <Select value={selectedModelType} onValueChange={(v) => {
                  setSelectedModelType(v);
                  const sizes = MODEL_TYPES.find((m) => m.value === v)?.sizes || [];
                  setSelectedModelSize(sizes.includes("base") ? "base" : sizes[0]);
                }}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {MODEL_TYPES.map((model) => (
                      <SelectItem key={model.value} value={model.value}>
                        {model.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Model Size</label>
                <Select value={selectedModelSize} onValueChange={setSelectedModelSize}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {availableSizes.map((size) => (
                      <SelectItem key={size} value={size}>
                        {size}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>

            {/* Training Parameters */}
            <div className="grid grid-cols-3 gap-4">
              <div className="space-y-2">
                <label className="text-sm font-medium">Epochs</label>
                <Input
                  type="number"
                  value={epochs}
                  onChange={(e) => setEpochs(parseInt(e.target.value) || 100)}
                  min={1}
                  max={500}
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Batch Size</label>
                <Input
                  type="number"
                  value={batchSize}
                  onChange={(e) => setBatchSize(parseInt(e.target.value) || 32)}
                  min={1}
                  max={256}
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Learning Rate</label>
                <Input
                  type="number"
                  value={learningRate}
                  onChange={(e) => setLearningRate(parseFloat(e.target.value) || 0.0001)}
                  step="0.0001"
                  min={0.000001}
                />
              </div>
            </div>

            {/* Augmentation */}
            <div className="space-y-2">
              <label className="text-sm font-medium">Augmentation Preset</label>
              <Select value={augmentationPreset} onValueChange={setAugmentationPreset}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {AUGMENTATION_PRESETS.map((preset) => (
                    <SelectItem key={preset.value} value={preset.value}>
                      <div>
                        <span className="font-medium">{preset.label}</span>
                        <span className="text-muted-foreground ml-2">{preset.description}</span>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Data Loading Config */}
            <DataLoadingConfigPanel
              value={dataLoadingConfig}
              onChange={setDataLoadingConfig}
              showDataLoader={true}
              defaultOpen={true}
            />

            {/* Info */}
            <div className="p-4 bg-muted rounded-lg text-sm space-y-1">
              <p><strong>SOTA Features Enabled:</strong></p>
              <ul className="list-disc list-inside text-muted-foreground">
                <li>Exponential Moving Average (EMA)</li>
                <li>Mixed Precision Training (FP16)</li>
                <li>Cosine Learning Rate Schedule with Warmup</li>
                <li>Early Stopping (patience: 15 epochs)</li>
                <li>Label Smoothing (0.1)</li>
              </ul>
            </div>
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setIsCreateDialogOpen(false)}>
              Cancel
            </Button>
            <Button
              onClick={() => createMutation.mutate()}
              disabled={!newRunName.trim() || !selectedDatasetId || createMutation.isPending}
            >
              {createMutation.isPending ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Starting...
                </>
              ) : (
                <>
                  <Play className="h-4 w-4 mr-2" />
                  Start Training
                </>
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
