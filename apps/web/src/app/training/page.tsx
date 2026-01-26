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
import { Textarea } from "@/components/ui/textarea";
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
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
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
  Plus,
  ChevronDown,
  ChevronUp,
  ChevronLeft,
  ChevronRight,
  ChevronsLeft,
  ChevronsRight,
  Trash2,
  Eye,
  StopCircle,
  BarChart2,
  Settings,
  Scale,
  X,
  ExternalLink,
  Search,
  ArrowUpDown,
  MoreHorizontal,
  AlertTriangle,
} from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Checkbox } from "@/components/ui/checkbox";
import type {
  TrainingRun,
  TrainingRunCreate,
  TrainingRunConfig,
  TrainedModel,
  TrainingCheckpoint,
  ModelPreset,
  TrainingDataSource,
  LabelFieldType,
  LabelStatsResponse,
  ModelComparisonResult,
  SOTAConfig,
  ImageConfig,
  ImageType,
  FrameSelection,
  DataLoadingConfig,
} from "@/types";
import { DEFAULT_SOTA_CONFIG, DEFAULT_IMAGE_CONFIG } from "@/types";
import { SOTAConfigPanel } from "./components/SOTAConfigPanel";
import { DataLoadingConfigPanel } from "@/components/training/DataLoadingConfig";

// Default training config
const DEFAULT_CONFIG: TrainingRunConfig = {
  epochs: 30,
  batch_size: 32,
  learning_rate: 0.0001,
  weight_decay: 0.01,
  warmup_epochs: 3,
  early_stopping_patience: 5,
  embedding_dim: 512,
  use_arcface: true,
  arcface_margin: 0.5,
  arcface_scale: 64.0,
  use_llrd: true,
  llrd_factor: 0.9,
  gradient_accumulation_steps: 1,
  mixed_precision: true,
  label_smoothing: 0.1,
  save_every_n_epochs: 5,
};

// Status badge configuration
const statusConfig: Record<string, { icon: React.ReactNode; color: string; label: string }> = {
  pending: {
    icon: <Clock className="h-4 w-4" />,
    color: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200",
    label: "Pending",
  },
  preparing: {
    icon: <Loader2 className="h-4 w-4 animate-spin" />,
    color: "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200",
    label: "Preparing",
  },
  running: {
    icon: <Loader2 className="h-4 w-4 animate-spin" />,
    color: "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200",
    label: "Running",
  },
  completed: {
    icon: <CheckCircle className="h-4 w-4" />,
    color: "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200",
    label: "Completed",
  },
  failed: {
    icon: <XCircle className="h-4 w-4" />,
    color: "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200",
    label: "Failed",
  },
  cancelled: {
    icon: <StopCircle className="h-4 w-4" />,
    color: "bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200",
    label: "Cancelled",
  },
};

export default function TrainingPage() {
  const queryClient = useQueryClient();
  const [activeTab, setActiveTab] = useState("runs");
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false);
  const [selectedRun, setSelectedRun] = useState<TrainingRun | null>(null);
  const [advancedOpen, setAdvancedOpen] = useState(false);

  // Model comparison state
  const [selectedModelsForCompare, setSelectedModelsForCompare] = useState<Set<string>>(new Set());
  const [isCompareDialogOpen, setIsCompareDialogOpen] = useState(false);
  const [comparisonData, setComparisonData] = useState<ModelComparisonResult[] | null>(null);

  // Selected trained model for details view
  const [selectedModel, setSelectedModel] = useState<TrainedModel | null>(null);

  // Force delete state
  const [forceDeleteDialog, setForceDeleteDialog] = useState<{
    open: boolean;
    runId: string;
    runName: string;
    linkedModels: Array<{ id: string; name: string }>;
  }>({ open: false, runId: "", runName: "", linkedModels: [] });

  // Smart table state for training runs
  const [selectedRunIds, setSelectedRunIds] = useState<Set<string>>(new Set());
  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState<string>("all");
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const [sortConfig, setSortConfig] = useState<{ key: string; direction: "asc" | "desc" }>({
    key: "created_at",
    direction: "desc",
  });
  const [bulkDeleteDialogOpen, setBulkDeleteDialogOpen] = useState(false);

  // Form state for new training run
  const [formData, setFormData] = useState({
    name: "",
    description: "",
    base_model_type: "dinov3-base",
    data_source: "all_products" as TrainingDataSource,
    dataset_id: "",
    // Image configuration - which image types to include
    image_config: { ...DEFAULT_IMAGE_CONFIG },
    label_config: {
      label_field: "product_id" as LabelFieldType,
      min_samples_per_class: 2,
    },
    split_config: {
      train_ratio: 0.70,
      val_ratio: 0.15,
      test_ratio: 0.15,
      seed: 42,
    },
    training_config: { ...DEFAULT_CONFIG },
    // SOTA training configuration
    sota_config: { ...DEFAULT_SOTA_CONFIG },
    // Hard negatives: pairs of product IDs that look similar but are different
    hard_negative_pairs: [] as Array<[string, string]>,
    // Data loading configuration
    data_loading: undefined as DataLoadingConfig | undefined,
  });

  // Fetch training runs
  const { data: runsData, isLoading: isLoadingRuns } = useQuery({
    queryKey: ["training-runs"],
    queryFn: () => apiClient.getTrainingRuns(),
    refetchInterval: 5000,
  });

  // Fetch trained models
  const { data: trainedModels, isLoading: isLoadingModels } = useQuery({
    queryKey: ["trained-models"],
    queryFn: () => apiClient.getTrainedModels(),
  });

  // Fetch model presets
  const { data: presetsData } = useQuery({
    queryKey: ["model-presets"],
    queryFn: () => apiClient.getModelPresets(),
  });

  // Fetch datasets
  const { data: datasets } = useQuery({
    queryKey: ["datasets"],
    queryFn: () => apiClient.getDatasets(),
  });

  // Fetch products count for training
  const { data: productsData } = useQuery({
    queryKey: ["training-products", formData.data_source, formData.dataset_id],
    queryFn: () => apiClient.getProductsForTraining({
      data_source: formData.data_source,
      dataset_id: formData.dataset_id || undefined,
      limit: 10000,
    }),
    enabled: isCreateDialogOpen && !(formData.data_source === "dataset" && !formData.dataset_id),
  });

  // Fetch label field statistics
  const { data: labelStats, isLoading: isLoadingLabelStats, error: labelStatsError } = useQuery({
    queryKey: ["label-stats", formData.data_source, formData.dataset_id],
    queryFn: () => apiClient.getLabelFieldStats({
      data_source: formData.data_source,
      dataset_id: formData.dataset_id || undefined,
    }),
    enabled: isCreateDialogOpen && !(formData.data_source === "dataset" && !formData.dataset_id),
  });

  // Debug: log label stats errors
  if (labelStatsError) {
    console.error("Label stats error:", labelStatsError);
  }

  // Create training run mutation
  const createRunMutation = useMutation({
    mutationFn: (data: TrainingRunCreate) => apiClient.createTrainingRun(data),
    onSuccess: () => {
      toast.success("Training run created and started");
      queryClient.invalidateQueries({ queryKey: ["training-runs"] });
      setIsCreateDialogOpen(false);
      resetForm();
    },
    onError: (error: Error) => {
      toast.error(`Failed to create training run: ${error.message}`);
    },
  });

  // Cancel training run mutation
  const cancelRunMutation = useMutation({
    mutationFn: (id: string) => apiClient.cancelTrainingRun(id),
    onSuccess: () => {
      toast.success("Training run cancelled");
      queryClient.invalidateQueries({ queryKey: ["training-runs"] });
    },
    onError: () => {
      toast.error("Failed to cancel training run");
    },
  });

  // Delete training run mutation
  const deleteRunMutation = useMutation({
    mutationFn: ({ id, force }: { id: string; force?: boolean }) =>
      apiClient.deleteTrainingRun(id, force),
    onSuccess: () => {
      toast.success("Training run deleted");
      setForceDeleteDialog({ open: false, runId: "", runName: "", linkedModels: [] });
      queryClient.invalidateQueries({ queryKey: ["training-runs"] });
      queryClient.invalidateQueries({ queryKey: ["trained-models"] });
    },
    onError: () => {
      toast.error("Failed to delete training run");
    },
  });

  // Resume training run mutation
  const resumeRunMutation = useMutation({
    mutationFn: (id: string) => apiClient.resumeTrainingRun(id),
    onSuccess: (data) => {
      toast.success(`Training resumed from epoch ${data.resumed_from_epoch}. ${data.remaining_epochs} epochs remaining.`);
      queryClient.invalidateQueries({ queryKey: ["training-runs"] });
    },
    onError: (error: Error) => {
      toast.error(`Failed to resume training: ${error.message}`);
    },
  });

  // Bulk delete training runs mutation
  const bulkDeleteMutation = useMutation({
    mutationFn: async (ids: string[]) => {
      const results = await Promise.allSettled(
        ids.map(id => apiClient.deleteTrainingRun(id, true))
      );
      const failed = results.filter(r => r.status === "rejected").length;
      if (failed > 0) {
        throw new Error(`${failed} of ${ids.length} deletions failed`);
      }
      return results;
    },
    onSuccess: (_, ids) => {
      toast.success(`${ids.length} training run(s) deleted`);
      setSelectedRunIds(new Set());
      setBulkDeleteDialogOpen(false);
      queryClient.invalidateQueries({ queryKey: ["training-runs"] });
      queryClient.invalidateQueries({ queryKey: ["trained-models"] });
    },
    onError: (error: Error) => {
      toast.error(error.message);
    },
  });

  // Compare models mutation
  const compareModelsMutation = useMutation({
    mutationFn: (modelIds: string[]) => apiClient.compareModels(modelIds),
    onSuccess: (data) => {
      setComparisonData(data);
      setIsCompareDialogOpen(true);
    },
    onError: (error: Error) => {
      toast.error(`Failed to compare models: ${error.message}`);
    },
  });

  const handleToggleModelSelection = (modelId: string) => {
    setSelectedModelsForCompare(prev => {
      const newSet = new Set(prev);
      if (newSet.has(modelId)) {
        newSet.delete(modelId);
      } else {
        if (newSet.size >= 5) {
          toast.error("Maximum 5 models can be compared");
          return prev;
        }
        newSet.add(modelId);
      }
      return newSet;
    });
  };

  const handleCompareModels = () => {
    if (selectedModelsForCompare.size < 2) {
      toast.error("Select at least 2 models to compare");
      return;
    }
    compareModelsMutation.mutate(Array.from(selectedModelsForCompare));
  };

  const resetForm = () => {
    setFormData({
      name: "",
      description: "",
      base_model_type: "dinov3-base",
      data_source: "all_products",
      dataset_id: "",
      image_config: { ...DEFAULT_IMAGE_CONFIG },
      label_config: {
        label_field: "product_id" as LabelFieldType,
        min_samples_per_class: 2,
      },
      split_config: {
        train_ratio: 0.70,
        val_ratio: 0.15,
        test_ratio: 0.15,
        seed: 42,
      },
      training_config: { ...DEFAULT_CONFIG },
      sota_config: { ...DEFAULT_SOTA_CONFIG },
      hard_negative_pairs: [],
      data_loading: undefined,
    });
    setAdvancedOpen(false);
  };

  const handleCreateRun = () => {
    if (!formData.name) {
      toast.error("Please provide a name for the training run");
      return;
    }
    if (formData.data_source === "dataset" && !formData.dataset_id) {
      toast.error("Please select a dataset");
      return;
    }
    // Clean up empty dataset_id to avoid backend validation issues
    const payload = {
      ...formData,
      dataset_id: formData.dataset_id || undefined,
    };
    createRunMutation.mutate(payload);
  };

  // Update config based on model selection
  const handleModelChange = (modelType: string) => {
    const preset = presetsData?.presets.find(p => p.model_type === modelType);
    setFormData(prev => ({
      ...prev,
      base_model_type: modelType,
      training_config: {
        ...prev.training_config,
        embedding_dim: preset?.embedding_dim || 512,
        // Adjust batch size based on model size
        batch_size: modelType.includes("large") ? 16 : 32,
      },
    }));
  };

  const runs = runsData?.items || [];
  const runningCount = runs.filter(r => r.status === "running" || r.status === "preparing").length;

  // Filter runs based on search and status
  const filteredRuns = runs.filter((run: TrainingRun) => {
    const matchesSearch = searchQuery === "" ||
      run.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      run.base_model_type.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesStatus = statusFilter === "all" || run.status === statusFilter;
    return matchesSearch && matchesStatus;
  });

  // Sort runs
  const sortedRuns = [...filteredRuns].sort((a: TrainingRun, b: TrainingRun) => {
    let aValue: string | number | Date = "";
    let bValue: string | number | Date = "";

    switch (sortConfig.key) {
      case "name":
        aValue = a.name.toLowerCase();
        bValue = b.name.toLowerCase();
        break;
      case "status":
        aValue = a.status;
        bValue = b.status;
        break;
      case "created_at":
        aValue = new Date(a.created_at);
        bValue = new Date(b.created_at);
        break;
      case "progress":
        aValue = a.current_epoch / a.total_epochs;
        bValue = b.current_epoch / b.total_epochs;
        break;
      case "recall":
        aValue = a.best_val_recall_at_1 || 0;
        bValue = b.best_val_recall_at_1 || 0;
        break;
      default:
        return 0;
    }

    if (aValue < bValue) return sortConfig.direction === "asc" ? -1 : 1;
    if (aValue > bValue) return sortConfig.direction === "asc" ? 1 : -1;
    return 0;
  });

  // Paginate runs
  const totalPages = Math.ceil(sortedRuns.length / pageSize);
  const paginatedRuns = sortedRuns.slice(
    (currentPage - 1) * pageSize,
    currentPage * pageSize
  );

  // Handle sort toggle
  const handleSort = (key: string) => {
    setSortConfig(prev => ({
      key,
      direction: prev.key === key && prev.direction === "asc" ? "desc" : "asc",
    }));
  };

  // Handle select all on current page
  const handleSelectAll = (checked: boolean) => {
    if (checked) {
      const selectableIds = paginatedRuns
        .filter((run: TrainingRun) => run.status !== "running" && run.status !== "preparing")
        .map((run: TrainingRun) => run.id);
      setSelectedRunIds(new Set([...selectedRunIds, ...selectableIds]));
    } else {
      const currentPageIds = new Set(paginatedRuns.map((run: TrainingRun) => run.id));
      setSelectedRunIds(new Set([...selectedRunIds].filter(id => !currentPageIds.has(id))));
    }
  };

  // Check if all selectable items on current page are selected
  const selectableOnPage = paginatedRuns.filter((run: TrainingRun) => run.status !== "running" && run.status !== "preparing");
  const allSelectedOnPage = selectableOnPage.length > 0 &&
    selectableOnPage.every((run: TrainingRun) => selectedRunIds.has(run.id));
  const someSelectedOnPage = selectableOnPage.some((run: TrainingRun) => selectedRunIds.has(run.id));

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Training</h1>
          <p className="text-muted-foreground">
            Fine-tune embedding models using product_id-based training
          </p>
        </div>
        <Dialog open={isCreateDialogOpen} onOpenChange={setIsCreateDialogOpen}>
          <DialogTrigger asChild>
            <Button>
              <Plus className="h-4 w-4 mr-2" />
              New Training Run
            </Button>
          </DialogTrigger>
          <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
            <DialogHeader>
              <DialogTitle>Create Training Run</DialogTitle>
              <DialogDescription>
                Configure and start a new model training run
              </DialogDescription>
            </DialogHeader>

            <div className="space-y-6 py-4">
              {/* Basic Info */}
              <div className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="name">Run Name *</Label>
                  <Input
                    id="name"
                    placeholder="e.g., dinov3-base-v1"
                    value={formData.name}
                    onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="description">Description</Label>
                  <Textarea
                    id="description"
                    placeholder="Optional description..."
                    value={formData.description}
                    onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                  />
                </div>
              </div>

              <Separator />

              {/* Model Selection */}
              <div className="space-y-4">
                <Label>Base Model</Label>
                <div className="grid grid-cols-3 gap-4">
                  {presetsData?.families.map(family => (
                    <div key={family.id} className="space-y-2">
                      <p className="text-sm font-medium">{family.name}</p>
                      {presetsData.presets
                        .filter(p => p.model_family === family.id)
                        .map(preset => (
                          <Button
                            key={preset.model_type}
                            variant={formData.base_model_type === preset.model_type ? "default" : "outline"}
                            size="sm"
                            className="w-full justify-start text-xs"
                            onClick={() => handleModelChange(preset.model_type)}
                          >
                            {preset.name.split(" ").pop()} ({preset.embedding_dim}d)
                          </Button>
                        ))}
                    </div>
                  ))}
                </div>
                {formData.base_model_type && (
                  <p className="text-sm text-muted-foreground">
                    {presetsData?.presets.find(p => p.model_type === formData.base_model_type)?.description}
                  </p>
                )}
              </div>

              <Separator />

              {/* Data Source */}
              <div className="space-y-4">
                <Label>Data Source</Label>
                <Select
                  value={formData.data_source}
                  onValueChange={(value: TrainingDataSource) => setFormData({ ...formData, data_source: value })}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all_products">All Products with Frames</SelectItem>
                    <SelectItem value="matched_products">Matched Products Only</SelectItem>
                    <SelectItem value="dataset">From Dataset</SelectItem>
                  </SelectContent>
                </Select>

                {formData.data_source === "dataset" && (
                  <Select
                    value={formData.dataset_id}
                    onValueChange={(value) => setFormData({ ...formData, dataset_id: value })}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select dataset..." />
                    </SelectTrigger>
                    <SelectContent>
                      {datasets?.map(ds => (
                        <SelectItem key={ds.id} value={ds.id}>
                          {ds.name} ({ds.product_count} products)
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                )}

                {productsData && (
                  <div className="bg-muted p-3 rounded-md text-sm">
                    <p><strong>{productsData.total}</strong> products available</p>
                    <p className="text-muted-foreground">
                      Train: ~{Math.floor(productsData.total * formData.split_config.train_ratio)} |
                      Val: ~{Math.floor(productsData.total * formData.split_config.val_ratio)} |
                      Test: ~{Math.floor(productsData.total * formData.split_config.test_ratio)}
                    </p>
                  </div>
                )}
              </div>

              <Separator />

              {/* Image Configuration */}
              <div className="space-y-4">
                <Label>Image Types</Label>
                <p className="text-sm text-muted-foreground">
                  Select which image types to include in training
                </p>
                <div className="flex flex-wrap gap-4">
                  {(["synthetic", "real", "augmented"] as const).map((type) => (
                    <label key={type} className="flex items-center gap-2 cursor-pointer">
                      <Checkbox
                        checked={formData.image_config.image_types.includes(type)}
                        onCheckedChange={(checked) => {
                          const newTypes = checked
                            ? [...formData.image_config.image_types, type]
                            : formData.image_config.image_types.filter((t) => t !== type);
                          setFormData({
                            ...formData,
                            image_config: { ...formData.image_config, image_types: newTypes as ImageType[] },
                          });
                        }}
                      />
                      <span className="capitalize">{type}</span>
                      <span className="text-xs text-muted-foreground">
                        {type === "synthetic" && "(3D rendered frames)"}
                        {type === "real" && "(from matching)"}
                        {type === "augmented" && "(AI augmented)"}
                      </span>
                    </label>
                  ))}
                </div>

                <div className="flex items-center gap-2 mt-2">
                  <Switch
                    checked={formData.image_config.include_matched_cutouts}
                    onCheckedChange={(checked) =>
                      setFormData({
                        ...formData,
                        image_config: { ...formData.image_config, include_matched_cutouts: checked },
                      })
                    }
                  />
                  <Label className="text-sm font-normal cursor-pointer">
                    Include matched cutouts as real-domain training data
                  </Label>
                </div>

                <div className="grid grid-cols-2 gap-4 mt-4">
                  <div className="space-y-2">
                    <Label className="text-xs">Frame Selection</Label>
                    <Select
                      value={formData.image_config.frame_selection}
                      onValueChange={(v) =>
                        setFormData({
                          ...formData,
                          image_config: { ...formData.image_config, frame_selection: v as FrameSelection },
                        })
                      }
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="first">First frame only</SelectItem>
                        <SelectItem value="key_frames">Key frames (4 angles)</SelectItem>
                        <SelectItem value="interval">Every N frames</SelectItem>
                        <SelectItem value="all">All frames</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  {formData.image_config.frame_selection === "interval" && (
                    <div className="space-y-2">
                      <Label className="text-xs">Frame Interval: {formData.image_config.frame_interval}</Label>
                      <Slider
                        value={[formData.image_config.frame_interval]}
                        onValueChange={([v]) =>
                          setFormData({
                            ...formData,
                            image_config: { ...formData.image_config, frame_interval: v },
                          })
                        }
                        min={1}
                        max={10}
                        step={1}
                      />
                    </div>
                  )}

                  {formData.image_config.frame_selection !== "all" && (
                    <div className="space-y-2">
                      <Label className="text-xs">Max Frames per Type: {formData.image_config.max_frames_per_type}</Label>
                      <Slider
                        value={[formData.image_config.max_frames_per_type]}
                        onValueChange={([v]) =>
                          setFormData({
                            ...formData,
                            image_config: { ...formData.image_config, max_frames_per_type: v },
                          })
                        }
                        min={1}
                        max={50}
                        step={1}
                      />
                    </div>
                  )}
                </div>
              </div>

              <Separator />

              {/* Label Field Configuration */}
              <div className="space-y-4">
                <Label>Training Target (Label Field)</Label>
                <p className="text-sm text-muted-foreground">
                  Choose what the model should learn to classify
                </p>
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
                  {labelStats && Object.entries(labelStats.label_fields)
                    .filter(([_, stats]) => stats.total_classes > 0)
                    .sort((a, b) => {
                      // product_id first, then by coverage
                      if (a[0] === "product_id") return -1;
                      if (b[0] === "product_id") return 1;
                      return (b[1].coverage_percent || 100) - (a[1].coverage_percent || 100);
                    })
                    .map(([field, stats]) => {
                      const isSelected = formData.label_config.label_field === field;
                      const coverage = stats.coverage_percent ?? 100;
                      return (
                        <button
                          key={field}
                          type="button"
                          onClick={() => setFormData({
                            ...formData,
                            label_config: { ...formData.label_config, label_field: field }
                          })}
                          className={`p-3 rounded-lg border text-left transition-colors ${
                            isSelected
                              ? "border-primary bg-primary/10"
                              : coverage < 50
                                ? "border-border/50 opacity-60 hover:opacity-100 hover:border-primary/50"
                                : "border-border hover:border-primary/50"
                          }`}
                        >
                          <p className="font-medium text-sm">
                            {stats.label}
                            {stats.is_custom && <span className="ml-1 text-xs text-muted-foreground">(custom)</span>}
                          </p>
                          <p className="text-xs text-muted-foreground mt-1">
                            {stats.total_classes} classes
                          </p>
                          <p className="text-xs text-muted-foreground">
                            {stats.avg_samples_per_class.toFixed(1)} avg samples
                          </p>
                          {field !== "product_id" && (
                            <p className={`text-xs mt-1 ${coverage >= 80 ? "text-green-600" : coverage >= 50 ? "text-yellow-600" : "text-red-500"}`}>
                              {coverage}% coverage
                            </p>
                          )}
                        </button>
                      );
                    })}
                  {isLoadingLabelStats && (
                    <p className="text-sm text-muted-foreground col-span-full flex items-center gap-2">
                      <Loader2 className="h-4 w-4 animate-spin" />
                      Loading label field options...
                    </p>
                  )}
                  {labelStatsError && (
                    <p className="text-sm text-red-500 col-span-full">
                      Error loading label fields: {(labelStatsError as Error).message}
                    </p>
                  )}
                  {!labelStats && !isLoadingLabelStats && !labelStatsError && (
                    <p className="text-sm text-muted-foreground col-span-full">
                      No label field data available
                    </p>
                  )}
                </div>
                {labelStats && formData.label_config.label_field !== "product_id" && (
                  <div className="bg-muted p-3 rounded-md text-sm">
                    <p className="font-medium mb-1">Top Classes:</p>
                    <div className="flex flex-wrap gap-1">
                      {labelStats.label_fields[formData.label_config.label_field]?.top_classes?.slice(0, 8).map(([name, count]) => (
                        <Badge key={name} variant="secondary" className="text-xs">
                          {name}: {count}
                        </Badge>
                      ))}
                    </div>
                    {(labelStats.label_fields[formData.label_config.label_field]?.unknown_count ?? 0) > 0 && (
                      <p className="text-xs text-muted-foreground mt-2">
                        {labelStats.label_fields[formData.label_config.label_field]?.unknown_count} products have no value for this field
                      </p>
                    )}
                  </div>
                )}
                <div className="space-y-2">
                  <Label className="text-xs">Min Samples per Class: {formData.label_config.min_samples_per_class}</Label>
                  <Slider
                    value={[formData.label_config.min_samples_per_class]}
                    onValueChange={([v]) => setFormData({
                      ...formData,
                      label_config: { ...formData.label_config, min_samples_per_class: v }
                    })}
                    min={1}
                    max={10}
                    step={1}
                  />
                  <p className="text-xs text-muted-foreground">
                    Classes with fewer samples will be excluded
                  </p>
                </div>
              </div>

              <Separator />

              {/* Split Configuration */}
              <div className="space-y-4">
                <Label>Data Split</Label>
                <div className="grid grid-cols-3 gap-4">
                  <div className="space-y-2">
                    <Label className="text-xs">Train: {(formData.split_config.train_ratio * 100).toFixed(0)}%</Label>
                    <Slider
                      value={[formData.split_config.train_ratio * 100]}
                      onValueChange={([v]) => setFormData({
                        ...formData,
                        split_config: { ...formData.split_config, train_ratio: v / 100 }
                      })}
                      min={50}
                      max={90}
                      step={5}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label className="text-xs">Val: {(formData.split_config.val_ratio * 100).toFixed(0)}%</Label>
                    <Slider
                      value={[formData.split_config.val_ratio * 100]}
                      onValueChange={([v]) => setFormData({
                        ...formData,
                        split_config: { ...formData.split_config, val_ratio: v / 100 }
                      })}
                      min={5}
                      max={25}
                      step={5}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label className="text-xs">Test: {(formData.split_config.test_ratio * 100).toFixed(0)}%</Label>
                    <Slider
                      value={[formData.split_config.test_ratio * 100]}
                      onValueChange={([v]) => setFormData({
                        ...formData,
                        split_config: { ...formData.split_config, test_ratio: v / 100 }
                      })}
                      min={5}
                      max={25}
                      step={5}
                    />
                  </div>
                </div>
                <p className="text-xs text-muted-foreground">
                  Split is by product_id - no product appears in multiple splits (no data leakage)
                </p>
              </div>

              {/* Training Config */}
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label>Epochs: {formData.training_config.epochs}</Label>
                    <Slider
                      value={[formData.training_config.epochs]}
                      onValueChange={([v]) => setFormData({
                        ...formData,
                        training_config: { ...formData.training_config, epochs: v }
                      })}
                      min={5}
                      max={100}
                      step={5}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>Batch Size: {formData.training_config.batch_size}</Label>
                    <Slider
                      value={[formData.training_config.batch_size]}
                      onValueChange={([v]) => setFormData({
                        ...formData,
                        training_config: { ...formData.training_config, batch_size: v }
                      })}
                      min={8}
                      max={64}
                      step={8}
                    />
                  </div>
                </div>
              </div>

              {/* SOTA Training Configuration */}
              <SOTAConfigPanel
                config={formData.sota_config}
                onChange={(sota_config) => setFormData({ ...formData, sota_config })}
              />

              {/* Data Loading Configuration */}
              <Separator />
              <DataLoadingConfigPanel
                value={formData.data_loading}
                onChange={(data_loading) => setFormData({ ...formData, data_loading })}
                showDataLoader={true}
                defaultOpen={true}
              />

              {/* Advanced Options */}
              <Collapsible open={advancedOpen} onOpenChange={setAdvancedOpen}>
                <CollapsibleTrigger asChild>
                  <Button variant="ghost" className="w-full justify-between">
                    <span className="flex items-center">
                      <Settings className="h-4 w-4 mr-2" />
                      Advanced Options
                    </span>
                    <ChevronDown className={`h-4 w-4 transition-transform ${advancedOpen ? "rotate-180" : ""}`} />
                  </Button>
                </CollapsibleTrigger>
                <CollapsibleContent className="space-y-4 pt-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label>Learning Rate: {formData.training_config.learning_rate.toExponential(1)}</Label>
                      <Slider
                        value={[Math.log10(formData.training_config.learning_rate)]}
                        onValueChange={([v]) => setFormData({
                          ...formData,
                          training_config: { ...formData.training_config, learning_rate: Math.pow(10, v) }
                        })}
                        min={-5}
                        max={-3}
                        step={0.1}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>ArcFace Margin: {formData.training_config.arcface_margin}</Label>
                      <Slider
                        value={[formData.training_config.arcface_margin]}
                        onValueChange={([v]) => setFormData({
                          ...formData,
                          training_config: { ...formData.training_config, arcface_margin: v }
                        })}
                        min={0.1}
                        max={1.0}
                        step={0.1}
                      />
                    </div>
                  </div>
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <Label>Mixed Precision (FP16)</Label>
                        <p className="text-xs text-muted-foreground">Faster training on modern GPUs</p>
                      </div>
                      <Switch
                        checked={formData.training_config.mixed_precision}
                        onCheckedChange={(v) => setFormData({
                          ...formData,
                          training_config: { ...formData.training_config, mixed_precision: v }
                        })}
                      />
                    </div>
                    <div className="flex items-center justify-between">
                      <div>
                        <Label>Layer-wise LR Decay (LLRD)</Label>
                        <p className="text-xs text-muted-foreground">Better fine-tuning</p>
                      </div>
                      <Switch
                        checked={formData.training_config.use_llrd}
                        onCheckedChange={(v) => setFormData({
                          ...formData,
                          training_config: { ...formData.training_config, use_llrd: v }
                        })}
                      />
                    </div>
                    <div className="flex items-center justify-between">
                      <div>
                        <Label>ArcFace Loss</Label>
                        <p className="text-xs text-muted-foreground">Metric learning head</p>
                      </div>
                      <Switch
                        checked={formData.training_config.use_arcface}
                        onCheckedChange={(v) => setFormData({
                          ...formData,
                          training_config: { ...formData.training_config, use_arcface: v }
                        })}
                      />
                    </div>
                  </div>

                  <Separator />

                  {/* Hard Negative Pairs */}
                  <div className="space-y-3">
                    <div>
                      <Label>Hard Negative Pairs (Confused Products)</Label>
                      <p className="text-xs text-muted-foreground">
                        Specify pairs of product IDs that look similar but are different.
                        The model will focus on distinguishing these during training.
                      </p>
                    </div>

                    {formData.hard_negative_pairs.length > 0 && (
                      <div className="space-y-2">
                        {formData.hard_negative_pairs.map((pair, index) => (
                          <div key={index} className="flex items-center gap-2 text-sm">
                            <Badge variant="outline" className="font-mono">{pair[0]}</Badge>
                            <span className="text-muted-foreground">↔</span>
                            <Badge variant="outline" className="font-mono">{pair[1]}</Badge>
                            <Button
                              type="button"
                              variant="ghost"
                              size="sm"
                              className="h-6 w-6 p-0 ml-auto"
                              onClick={() => {
                                const newPairs = [...formData.hard_negative_pairs];
                                newPairs.splice(index, 1);
                                setFormData({ ...formData, hard_negative_pairs: newPairs });
                              }}
                            >
                              <X className="h-3 w-3" />
                            </Button>
                          </div>
                        ))}
                      </div>
                    )}

                    <div className="flex gap-2">
                      <Input
                        placeholder="Product ID 1"
                        id="hard-neg-1"
                        className="font-mono text-xs"
                      />
                      <Input
                        placeholder="Product ID 2"
                        id="hard-neg-2"
                        className="font-mono text-xs"
                      />
                      <Button
                        type="button"
                        variant="outline"
                        size="sm"
                        onClick={() => {
                          const input1 = document.getElementById("hard-neg-1") as HTMLInputElement;
                          const input2 = document.getElementById("hard-neg-2") as HTMLInputElement;
                          if (input1.value && input2.value) {
                            setFormData({
                              ...formData,
                              hard_negative_pairs: [
                                ...formData.hard_negative_pairs,
                                [input1.value.trim(), input2.value.trim()]
                              ]
                            });
                            input1.value = "";
                            input2.value = "";
                          }
                        }}
                      >
                        <Plus className="h-4 w-4" />
                      </Button>
                    </div>
                    <p className="text-xs text-muted-foreground">
                      Tip: You can find confused pairs from previous model evaluations in the &quot;Hard Cases&quot; section
                    </p>
                  </div>
                </CollapsibleContent>
              </Collapsible>
            </div>

            <DialogFooter>
              <Button variant="outline" onClick={() => setIsCreateDialogOpen(false)}>
                Cancel
              </Button>
              <Button onClick={handleCreateRun} disabled={createRunMutation.isPending}>
                {createRunMutation.isPending && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
                Start Training
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList>
          <TabsTrigger value="runs">
            Training Runs {runningCount > 0 && `(${runningCount} active)`}
          </TabsTrigger>
          <TabsTrigger value="models">
            Trained Models ({trainedModels?.length || 0})
          </TabsTrigger>
        </TabsList>

        {/* Training Runs Tab */}
        <TabsContent value="runs" className="mt-6">
          <Card>
            <CardHeader className="pb-4">
              <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                <div>
                  <CardTitle>Training Runs</CardTitle>
                  <CardDescription>Monitor and manage training jobs</CardDescription>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => queryClient.invalidateQueries({ queryKey: ["training-runs"] })}
                >
                  <RefreshCw className="h-4 w-4" />
                </Button>
              </div>

              {/* Filters and Search */}
              <div className="flex flex-col gap-3 sm:flex-row sm:items-center mt-4">
                <div className="relative flex-1 max-w-sm">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <Input
                    placeholder="Search runs..."
                    value={searchQuery}
                    onChange={(e) => {
                      setSearchQuery(e.target.value);
                      setCurrentPage(1);
                    }}
                    className="pl-9"
                  />
                </div>
                <Select
                  value={statusFilter}
                  onValueChange={(v) => {
                    setStatusFilter(v);
                    setCurrentPage(1);
                  }}
                >
                  <SelectTrigger className="w-[150px]">
                    <SelectValue placeholder="All Statuses" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Statuses</SelectItem>
                    <SelectItem value="running">Running</SelectItem>
                    <SelectItem value="completed">Completed</SelectItem>
                    <SelectItem value="failed">Failed</SelectItem>
                    <SelectItem value="cancelled">Cancelled</SelectItem>
                    <SelectItem value="pending">Pending</SelectItem>
                  </SelectContent>
                </Select>
                <Select
                  value={pageSize.toString()}
                  onValueChange={(v) => {
                    setPageSize(Number(v));
                    setCurrentPage(1);
                  }}
                >
                  <SelectTrigger className="w-[100px]">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="5">5 / page</SelectItem>
                    <SelectItem value="10">10 / page</SelectItem>
                    <SelectItem value="20">20 / page</SelectItem>
                    <SelectItem value="50">50 / page</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Bulk Actions Bar */}
              {selectedRunIds.size > 0 && (
                <div className="flex items-center gap-3 p-3 bg-muted rounded-lg mt-4">
                  <span className="text-sm font-medium">
                    {selectedRunIds.size} selected
                  </span>
                  <Separator orientation="vertical" className="h-4" />
                  <Button
                    variant="destructive"
                    size="sm"
                    onClick={() => setBulkDeleteDialogOpen(true)}
                  >
                    <Trash2 className="h-4 w-4 mr-2" />
                    Delete Selected
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setSelectedRunIds(new Set())}
                  >
                    Clear Selection
                  </Button>
                </div>
              )}
            </CardHeader>

            <CardContent>
              {isLoadingRuns ? (
                <div className="flex items-center justify-center py-8">
                  <Loader2 className="h-6 w-6 animate-spin" />
                </div>
              ) : runs.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground">
                  <Brain className="h-12 w-12 mx-auto mb-2 opacity-50" />
                  <p>No training runs yet</p>
                  <p className="text-sm mt-1">Click &quot;New Training Run&quot; to get started</p>
                </div>
              ) : filteredRuns.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground">
                  <Search className="h-12 w-12 mx-auto mb-2 opacity-50" />
                  <p>No matching runs found</p>
                  <p className="text-sm mt-1">Try adjusting your search or filters</p>
                </div>
              ) : (
                <>
                  <div className="rounded-md border">
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead className="w-12">
                            <Checkbox
                              checked={allSelectedOnPage}
                              onCheckedChange={handleSelectAll}
                              aria-label="Select all"
                              className={someSelectedOnPage && !allSelectedOnPage ? "opacity-50" : ""}
                            />
                          </TableHead>
                          <TableHead>
                            <button
                              className="flex items-center gap-1 hover:text-foreground transition-colors"
                              onClick={() => handleSort("name")}
                            >
                              Name
                              <ArrowUpDown className="h-4 w-4" />
                              {sortConfig.key === "name" && (
                                sortConfig.direction === "asc" ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />
                              )}
                            </button>
                          </TableHead>
                          <TableHead>Model</TableHead>
                          <TableHead>Target</TableHead>
                          <TableHead>
                            <button
                              className="flex items-center gap-1 hover:text-foreground transition-colors"
                              onClick={() => handleSort("progress")}
                            >
                              Progress
                              <ArrowUpDown className="h-4 w-4" />
                            </button>
                          </TableHead>
                          <TableHead>
                            <button
                              className="flex items-center gap-1 hover:text-foreground transition-colors"
                              onClick={() => handleSort("recall")}
                            >
                              Best R@1
                              <ArrowUpDown className="h-4 w-4" />
                            </button>
                          </TableHead>
                          <TableHead>
                            <button
                              className="flex items-center gap-1 hover:text-foreground transition-colors"
                              onClick={() => handleSort("status")}
                            >
                              Status
                              <ArrowUpDown className="h-4 w-4" />
                            </button>
                          </TableHead>
                          <TableHead className="w-12"></TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {paginatedRuns.map((run: TrainingRun) => {
                          const isSelectable = run.status !== "running" && run.status !== "preparing";
                          const isSelected = selectedRunIds.has(run.id);
                          return (
                            <TableRow
                              key={run.id}
                              className={isSelected ? "bg-muted/50" : ""}
                            >
                              <TableCell>
                                <Checkbox
                                  checked={isSelected}
                                  onCheckedChange={(checked) => {
                                    const newSet = new Set(selectedRunIds);
                                    if (checked) {
                                      newSet.add(run.id);
                                    } else {
                                      newSet.delete(run.id);
                                    }
                                    setSelectedRunIds(newSet);
                                  }}
                                  disabled={!isSelectable}
                                  aria-label={`Select ${run.name}`}
                                />
                              </TableCell>
                              <TableCell>
                                <div>
                                  <p className="font-medium">{run.name}</p>
                                  <p className="text-xs text-muted-foreground">
                                    {new Date(run.created_at).toLocaleDateString()}
                                  </p>
                                </div>
                              </TableCell>
                              <TableCell>
                                <Badge variant="outline">{run.base_model_type}</Badge>
                              </TableCell>
                              <TableCell>
                                <Badge variant="secondary" className="text-xs">
                                  {run.label_config?.label_field === "product_id" ? "Product" :
                                   run.label_config?.label_field === "category" ? "Category" :
                                   run.label_config?.label_field === "brand_name" ? "Brand" :
                                   "Product"}
                                </Badge>
                              </TableCell>
                              <TableCell>
                                <div className="w-24">
                                  <Progress
                                    value={(run.current_epoch / run.total_epochs) * 100}
                                    className="h-2"
                                  />
                                  <p className="text-xs text-muted-foreground mt-1">
                                    {run.current_epoch}/{run.total_epochs}
                                  </p>
                                </div>
                              </TableCell>
                              <TableCell>
                                {run.best_val_recall_at_1 ? (
                                  <div className="text-sm">
                                    <p className="font-medium">{(run.best_val_recall_at_1 * 100).toFixed(1)}%</p>
                                    <p className="text-muted-foreground text-xs">
                                      epoch {run.best_epoch}
                                    </p>
                                  </div>
                                ) : (
                                  <span className="text-muted-foreground">-</span>
                                )}
                              </TableCell>
                              <TableCell>
                                <Badge className={statusConfig[run.status]?.color}>
                                  <span className="mr-1">{statusConfig[run.status]?.icon}</span>
                                  {statusConfig[run.status]?.label}
                                </Badge>
                              </TableCell>
                              <TableCell>
                                <DropdownMenu>
                                  <DropdownMenuTrigger asChild>
                                    <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
                                      <MoreHorizontal className="h-4 w-4" />
                                    </Button>
                                  </DropdownMenuTrigger>
                                  <DropdownMenuContent align="end">
                                    <DropdownMenuItem onClick={() => setSelectedRun(run)}>
                                      <Eye className="h-4 w-4 mr-2" />
                                      Quick View
                                    </DropdownMenuItem>
                                    <DropdownMenuItem asChild>
                                      <Link href={`/training/${run.id}`}>
                                        <ExternalLink className="h-4 w-4 mr-2" />
                                        Open Details
                                      </Link>
                                    </DropdownMenuItem>
                                    {(run.status === "running" || run.status === "preparing") && (
                                      <>
                                        <DropdownMenuSeparator />
                                        <DropdownMenuItem
                                          onClick={() => cancelRunMutation.mutate(run.id)}
                                          className="text-orange-600"
                                        >
                                          <StopCircle className="h-4 w-4 mr-2" />
                                          Cancel Training
                                        </DropdownMenuItem>
                                      </>
                                    )}
                                    {(run.status === "failed" || run.status === "cancelled") && (
                                      <>
                                        <DropdownMenuSeparator />
                                        <DropdownMenuItem
                                          onClick={() => resumeRunMutation.mutate(run.id)}
                                          disabled={resumeRunMutation.isPending}
                                        >
                                          <Play className="h-4 w-4 mr-2" />
                                          Resume Training
                                        </DropdownMenuItem>
                                      </>
                                    )}
                                    {isSelectable && (
                                      <>
                                        <DropdownMenuSeparator />
                                        <DropdownMenuItem
                                          onClick={async () => {
                                            try {
                                              await apiClient.deleteTrainingRun(run.id, false);
                                              toast.success("Training run deleted");
                                              queryClient.invalidateQueries({ queryKey: ["training-runs"] });
                                            } catch (error: unknown) {
                                              const err = error as { status?: number; body?: { detail?: { linked_models?: Array<{ id: string; name: string }> } } };
                                              if (err.status === 409 && err.body?.detail?.linked_models) {
                                                setForceDeleteDialog({
                                                  open: true,
                                                  runId: run.id,
                                                  runName: run.name,
                                                  linkedModels: err.body.detail.linked_models,
                                                });
                                              } else {
                                                toast.error("Failed to delete training run");
                                              }
                                            }
                                          }}
                                          className="text-destructive"
                                        >
                                          <Trash2 className="h-4 w-4 mr-2" />
                                          Delete
                                        </DropdownMenuItem>
                                      </>
                                    )}
                                  </DropdownMenuContent>
                                </DropdownMenu>
                              </TableCell>
                            </TableRow>
                          );
                        })}
                      </TableBody>
                    </Table>
                  </div>

                  {/* Pagination */}
                  <div className="flex items-center justify-between mt-4">
                    <p className="text-sm text-muted-foreground">
                      Showing {((currentPage - 1) * pageSize) + 1} to {Math.min(currentPage * pageSize, sortedRuns.length)} of {sortedRuns.length} runs
                      {filteredRuns.length !== runs.length && ` (filtered from ${runs.length})`}
                    </p>
                    <div className="flex items-center gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setCurrentPage(1)}
                        disabled={currentPage === 1}
                      >
                        <ChevronsLeft className="h-4 w-4" />
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                        disabled={currentPage === 1}
                      >
                        <ChevronLeft className="h-4 w-4" />
                      </Button>
                      <span className="text-sm px-2">
                        Page {currentPage} of {totalPages || 1}
                      </span>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                        disabled={currentPage >= totalPages}
                      >
                        <ChevronRight className="h-4 w-4" />
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setCurrentPage(totalPages)}
                        disabled={currentPage >= totalPages}
                      >
                        <ChevronsRight className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                </>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Bulk Delete Confirmation Dialog */}
        <Dialog open={bulkDeleteDialogOpen} onOpenChange={setBulkDeleteDialogOpen}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle className="flex items-center gap-2">
                <AlertTriangle className="h-5 w-5 text-destructive" />
                Delete {selectedRunIds.size} Training Run{selectedRunIds.size > 1 ? "s" : ""}?
              </DialogTitle>
              <DialogDescription>
                This will permanently delete the selected training runs and their associated checkpoints.
                This action cannot be undone.
              </DialogDescription>
            </DialogHeader>
            <DialogFooter className="gap-2">
              <Button
                variant="outline"
                onClick={() => setBulkDeleteDialogOpen(false)}
              >
                Cancel
              </Button>
              <Button
                variant="destructive"
                onClick={() => bulkDeleteMutation.mutate(Array.from(selectedRunIds))}
                disabled={bulkDeleteMutation.isPending}
              >
                {bulkDeleteMutation.isPending ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Deleting...
                  </>
                ) : (
                  <>
                    <Trash2 className="h-4 w-4 mr-2" />
                    Delete {selectedRunIds.size} Run{selectedRunIds.size > 1 ? "s" : ""}
                  </>
                )}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        {/* Trained Models Tab */}
        <TabsContent value="models" className="mt-6">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <div>
                <CardTitle>Trained Models</CardTitle>
                <CardDescription>
                  Registered models ready for deployment
                </CardDescription>
              </div>
              {selectedModelsForCompare.size >= 2 && (
                <Button
                  onClick={handleCompareModels}
                  disabled={compareModelsMutation.isPending}
                >
                  {compareModelsMutation.isPending ? (
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <Scale className="h-4 w-4 mr-2" />
                  )}
                  Compare ({selectedModelsForCompare.size})
                </Button>
              )}
            </CardHeader>
            <CardContent>
              {isLoadingModels ? (
                <div className="flex items-center justify-center py-8">
                  <Loader2 className="h-6 w-6 animate-spin" />
                </div>
              ) : trainedModels?.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground">
                  <Brain className="h-12 w-12 mx-auto mb-2 opacity-50" />
                  <p>No trained models yet</p>
                  <p className="text-sm mt-1">Complete a training run and register a checkpoint</p>
                </div>
              ) : (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="w-10"></TableHead>
                      <TableHead>Name</TableHead>
                      <TableHead>Training Run</TableHead>
                      <TableHead>Test Metrics</TableHead>
                      <TableHead>Cross-Domain</TableHead>
                      <TableHead></TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {trainedModels?.map((model: TrainedModel) => (
                      <TableRow key={model.id}>
                        <TableCell>
                          <Checkbox
                            checked={selectedModelsForCompare.has(model.id)}
                            onCheckedChange={() => handleToggleModelSelection(model.id)}
                            disabled={!model.test_evaluated}
                            title={model.test_evaluated ? "Select for comparison" : "Model must be evaluated first"}
                          />
                        </TableCell>
                        <TableCell>
                          <div>
                            <p className="font-medium">{model.name}</p>
                            <p className="text-xs text-muted-foreground">
                              {model.description || "No description"}
                            </p>
                          </div>
                        </TableCell>
                        <TableCell>
                          {model.training_run?.name || "-"}
                        </TableCell>
                        <TableCell>
                          {model.test_evaluated && model.test_metrics ? (
                            <div className="text-sm">
                              <p>R@1: {(model.test_metrics.recall_at_1 * 100).toFixed(1)}%</p>
                              <p className="text-muted-foreground text-xs">
                                mAP: {(model.test_metrics.map * 100).toFixed(1)}%
                              </p>
                            </div>
                          ) : (
                            <Badge variant="outline">Not evaluated</Badge>
                          )}
                        </TableCell>
                        <TableCell>
                          {model.cross_domain_metrics ? (
                            <div className="text-sm">
                              {model.cross_domain_metrics.real_to_synth && (
                                <p className="text-xs">
                                  R→S: {(model.cross_domain_metrics.real_to_synth.recall_at_1 * 100).toFixed(0)}%
                                </p>
                              )}
                              {model.cross_domain_metrics.synth_to_real && (
                                <p className="text-xs text-muted-foreground">
                                  S→R: {(model.cross_domain_metrics.synth_to_real.recall_at_1 * 100).toFixed(0)}%
                                </p>
                              )}
                            </div>
                          ) : (
                            <span className="text-muted-foreground text-xs">-</span>
                          )}
                        </TableCell>
                        <TableCell>
                          <div className="flex gap-1">
                            {!model.test_evaluated && (
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => apiClient.evaluateTrainedModel(model.id)}
                                title="Evaluate model"
                              >
                                <BarChart2 className="h-4 w-4" />
                              </Button>
                            )}
                            {model.test_evaluated && (
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => setSelectedModel(model)}
                                title="View evaluation details"
                              >
                                <Eye className="h-4 w-4" />
                              </Button>
                            )}
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

      {/* Run Details Dialog */}
      {selectedRun && (
        <RunDetailsDialog
          run={selectedRun}
          onClose={() => setSelectedRun(null)}
        />
      )}

      {/* Model Comparison Dialog */}
      <Dialog open={isCompareDialogOpen} onOpenChange={setIsCompareDialogOpen}>
        <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Model Comparison</DialogTitle>
            <DialogDescription>
              Side-by-side comparison of selected models
            </DialogDescription>
          </DialogHeader>

          {comparisonData && comparisonData.length > 0 && (
            <div className="space-y-6 py-4">
              {/* Comparison Table */}
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Metric</TableHead>
                    {comparisonData.map((model) => (
                      <TableHead key={model.id} className="text-center">
                        {model.name}
                      </TableHead>
                    ))}
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {/* Recall@1 */}
                  <TableRow>
                    <TableCell className="font-medium">Recall@1</TableCell>
                    {comparisonData.map((model) => (
                      <TableCell key={model.id} className="text-center">
                        {model.test_metrics?.recall_at_1
                          ? `${(model.test_metrics.recall_at_1 * 100).toFixed(1)}%`
                          : "-"}
                      </TableCell>
                    ))}
                  </TableRow>
                  {/* Recall@5 */}
                  <TableRow>
                    <TableCell className="font-medium">Recall@5</TableCell>
                    {comparisonData.map((model) => (
                      <TableCell key={model.id} className="text-center">
                        {model.test_metrics?.recall_at_5
                          ? `${(model.test_metrics.recall_at_5 * 100).toFixed(1)}%`
                          : "-"}
                      </TableCell>
                    ))}
                  </TableRow>
                  {/* Recall@10 */}
                  <TableRow>
                    <TableCell className="font-medium">Recall@10</TableCell>
                    {comparisonData.map((model) => (
                      <TableCell key={model.id} className="text-center">
                        {model.test_metrics?.recall_at_10
                          ? `${(model.test_metrics.recall_at_10 * 100).toFixed(1)}%`
                          : "-"}
                      </TableCell>
                    ))}
                  </TableRow>
                  {/* mAP */}
                  <TableRow>
                    <TableCell className="font-medium">mAP</TableCell>
                    {comparisonData.map((model) => (
                      <TableCell key={model.id} className="text-center">
                        {model.test_metrics?.map
                          ? `${(model.test_metrics.map * 100).toFixed(1)}%`
                          : "-"}
                      </TableCell>
                    ))}
                  </TableRow>
                  {/* Cross-Domain: Real → Synth */}
                  {comparisonData.some((m) => m.cross_domain_metrics?.real_to_synth) && (
                    <>
                      <TableRow className="bg-muted/50">
                        <TableCell colSpan={comparisonData.length + 1} className="font-medium text-muted-foreground">
                          Cross-Domain Metrics
                        </TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell className="font-medium pl-4">Real → Synth R@1</TableCell>
                        {comparisonData.map((model) => (
                          <TableCell key={model.id} className="text-center">
                            {model.cross_domain_metrics?.real_to_synth?.recall_at_1
                              ? `${(model.cross_domain_metrics.real_to_synth.recall_at_1 * 100).toFixed(1)}%`
                              : "-"}
                          </TableCell>
                        ))}
                      </TableRow>
                      <TableRow>
                        <TableCell className="font-medium pl-4">Synth → Real R@1</TableCell>
                        {comparisonData.map((model) => (
                          <TableCell key={model.id} className="text-center">
                            {model.cross_domain_metrics?.synth_to_real?.recall_at_1
                              ? `${(model.cross_domain_metrics.synth_to_real.recall_at_1 * 100).toFixed(1)}%`
                              : "-"}
                          </TableCell>
                        ))}
                      </TableRow>
                    </>
                  )}
                </TableBody>
              </Table>
            </div>
          )}

          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => {
                setIsCompareDialogOpen(false);
                setComparisonData(null);
                setSelectedModelsForCompare(new Set());
              }}
            >
              Close
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Trained Model Details Dialog */}
      {selectedModel && (
        <TrainedModelDetailsDialog
          model={selectedModel}
          onClose={() => setSelectedModel(null)}
        />
      )}

      {/* Force Delete Confirmation Dialog */}
      <Dialog
        open={forceDeleteDialog.open}
        onOpenChange={(open) => {
          if (!open) setForceDeleteDialog({ open: false, runId: "", runName: "", linkedModels: [] });
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete Training Run?</DialogTitle>
            <DialogDescription>
              <span className="font-semibold">{forceDeleteDialog.runName}</span> has registered models that will also be deleted:
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-2 py-4">
            {forceDeleteDialog.linkedModels.map((model) => (
              <div key={model.id} className="flex items-center gap-2 p-2 bg-muted rounded-md">
                <Brain className="h-4 w-4 text-primary" />
                <span>{model.name}</span>
              </div>
            ))}
          </div>
          <DialogFooter className="gap-2">
            <Button
              variant="outline"
              onClick={() => setForceDeleteDialog({ open: false, runId: "", runName: "", linkedModels: [] })}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={() => deleteRunMutation.mutate({ id: forceDeleteDialog.runId, force: true })}
              disabled={deleteRunMutation.isPending}
            >
              {deleteRunMutation.isPending ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Deleting...
                </>
              ) : (
                "Delete All"
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

// Trained Model Details Dialog Component
function TrainedModelDetailsDialog({
  model,
  onClose,
}: {
  model: TrainedModel;
  onClose: () => void;
}) {
  // Fetch evaluations for this model
  const { data: evaluations } = useQuery({
    queryKey: ["model-evaluations", model.id],
    queryFn: () => apiClient.getModelEvaluations(model.id),
  });

  const latestEvaluation = evaluations?.[0];

  return (
    <Dialog open={true} onOpenChange={onClose}>
      <DialogContent className="max-w-3xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>{model.name}</DialogTitle>
          <DialogDescription>
            {model.description || "Model evaluation details and hard cases"}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-6 py-4">
          {/* Overall Metrics */}
          {model.test_metrics && (
            <div>
              <h4 className="font-medium mb-3">Test Metrics</h4>
              <div className="grid grid-cols-4 gap-4 text-sm">
                <div className="bg-muted p-3 rounded-md text-center">
                  <p className="text-2xl font-bold text-primary">
                    {(model.test_metrics.recall_at_1 * 100).toFixed(1)}%
                  </p>
                  <p className="text-muted-foreground">Recall@1</p>
                </div>
                <div className="bg-muted p-3 rounded-md text-center">
                  <p className="text-2xl font-bold">
                    {(model.test_metrics.recall_at_5 * 100).toFixed(1)}%
                  </p>
                  <p className="text-muted-foreground">Recall@5</p>
                </div>
                <div className="bg-muted p-3 rounded-md text-center">
                  <p className="text-2xl font-bold">
                    {model.test_metrics.recall_at_10
                      ? `${(model.test_metrics.recall_at_10 * 100).toFixed(1)}%`
                      : "-"}
                  </p>
                  <p className="text-muted-foreground">Recall@10</p>
                </div>
                <div className="bg-muted p-3 rounded-md text-center">
                  <p className="text-2xl font-bold">
                    {(model.test_metrics.map * 100).toFixed(1)}%
                  </p>
                  <p className="text-muted-foreground">mAP</p>
                </div>
              </div>
            </div>
          )}

          {/* Cross-Domain Metrics */}
          {model.cross_domain_metrics && (
            <>
              <Separator />
              <div>
                <h4 className="font-medium mb-3">Cross-Domain Metrics</h4>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  {model.cross_domain_metrics.real_to_synth && (
                    <div className="bg-muted p-3 rounded-md">
                      <p className="text-muted-foreground mb-1">Real → Synthetic</p>
                      <p>R@1: {(model.cross_domain_metrics.real_to_synth.recall_at_1 * 100).toFixed(1)}%</p>
                      <p className="text-xs text-muted-foreground">
                        R@5: {(model.cross_domain_metrics.real_to_synth.recall_at_5 * 100).toFixed(1)}%
                      </p>
                    </div>
                  )}
                  {model.cross_domain_metrics.synth_to_real && (
                    <div className="bg-muted p-3 rounded-md">
                      <p className="text-muted-foreground mb-1">Synthetic → Real</p>
                      <p>R@1: {(model.cross_domain_metrics.synth_to_real.recall_at_1 * 100).toFixed(1)}%</p>
                      <p className="text-xs text-muted-foreground">
                        R@5: {(model.cross_domain_metrics.synth_to_real.recall_at_5 * 100).toFixed(1)}%
                      </p>
                    </div>
                  )}
                </div>
              </div>
            </>
          )}

          {/* Hard Cases - Worst Products */}
          {latestEvaluation?.worst_product_ids && latestEvaluation.worst_product_ids.length > 0 && (
            <>
              <Separator />
              <div>
                <h4 className="font-medium mb-3 text-orange-600">Worst Performing Products</h4>
                <p className="text-sm text-muted-foreground mb-2">
                  Products with lowest recall - may need more training data
                </p>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Product ID</TableHead>
                      <TableHead className="text-right">Recall@1</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {latestEvaluation.worst_product_ids.slice(0, 10).map((item, idx) => (
                      <TableRow key={idx}>
                        <TableCell className="font-mono text-xs">{item.product_id}</TableCell>
                        <TableCell className="text-right text-red-600">
                          {(item.recall_at_1 * 100).toFixed(1)}%
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </>
          )}

          {/* Hard Cases - Confused Pairs */}
          {latestEvaluation?.most_confused_pairs && latestEvaluation.most_confused_pairs.length > 0 && (
            <>
              <Separator />
              <div>
                <h4 className="font-medium mb-3 text-orange-600">Most Confused Product Pairs</h4>
                <p className="text-sm text-muted-foreground mb-2">
                  Product pairs frequently misidentified as each other
                </p>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Product 1</TableHead>
                      <TableHead>Product 2</TableHead>
                      <TableHead className="text-right">Similarity</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {latestEvaluation.most_confused_pairs.slice(0, 10).map((pair, idx) => (
                      <TableRow key={idx}>
                        <TableCell className="font-mono text-xs">{pair.product_id_1}</TableCell>
                        <TableCell className="font-mono text-xs">{pair.product_id_2}</TableCell>
                        <TableCell className="text-right text-orange-600">
                          {(pair.similarity * 100).toFixed(1)}%
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </>
          )}

          {/* No hard cases message */}
          {!latestEvaluation?.worst_product_ids?.length && !latestEvaluation?.most_confused_pairs?.length && (
            <>
              <Separator />
              <div className="text-center py-4 text-muted-foreground">
                <p>No hard cases data available</p>
                <p className="text-sm">Run a detailed evaluation to see problematic products</p>
              </div>
            </>
          )}
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={onClose}>
            Close
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

// Run Details Dialog Component
function RunDetailsDialog({
  run,
  onClose,
}: {
  run: TrainingRun;
  onClose: () => void;
}) {
  const queryClient = useQueryClient();

  // Fetch checkpoints
  const { data: checkpoints } = useQuery({
    queryKey: ["training-checkpoints", run.id],
    queryFn: () => apiClient.getTrainingCheckpoints(run.id),
  });

  // Register model mutation
  const registerModelMutation = useMutation({
    mutationFn: (data: { checkpoint_id: string; name: string }) =>
      apiClient.registerTrainedModel(data),
    onSuccess: () => {
      toast.success("Model registered successfully");
      queryClient.invalidateQueries({ queryKey: ["trained-models"] });
      onClose();
    },
    onError: () => {
      toast.error("Failed to register model");
    },
  });

  const bestCheckpoint = checkpoints?.find(c => c.is_best);

  return (
    <Dialog open={true} onOpenChange={onClose}>
      <DialogContent className="max-w-3xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>{run.name}</DialogTitle>
          <DialogDescription>{run.description || "No description"}</DialogDescription>
        </DialogHeader>

        <div className="space-y-6 py-4">
          {/* Run Info */}
          <div className="grid grid-cols-3 gap-4 text-sm">
            <div>
              <p className="text-muted-foreground">Model</p>
              <p className="font-medium">{run.base_model_type}</p>
            </div>
            <div>
              <p className="text-muted-foreground">Status</p>
              <Badge className={statusConfig[run.status]?.color}>
                {statusConfig[run.status]?.label}
              </Badge>
            </div>
            <div>
              <p className="text-muted-foreground">Progress</p>
              <p className="font-medium">{run.current_epoch}/{run.total_epochs} epochs</p>
            </div>
          </div>

          <Separator />

          {/* Data Split */}
          <div>
            <h4 className="font-medium mb-2">Data Split (product_id based)</h4>
            <div className="grid grid-cols-3 gap-4 text-sm">
              <div className="bg-muted p-3 rounded-md">
                <p className="text-muted-foreground">Train</p>
                <p className="font-medium">{run.train_product_count} products</p>
                <p className="text-xs text-muted-foreground">{run.train_image_count} images</p>
              </div>
              <div className="bg-muted p-3 rounded-md">
                <p className="text-muted-foreground">Validation</p>
                <p className="font-medium">{run.val_product_count} products</p>
                <p className="text-xs text-muted-foreground">{run.val_image_count} images</p>
              </div>
              <div className="bg-muted p-3 rounded-md">
                <p className="text-muted-foreground">Test (held-out)</p>
                <p className="font-medium">{run.test_product_count} products</p>
                <p className="text-xs text-muted-foreground">{run.test_image_count} images</p>
              </div>
            </div>
          </div>

          <Separator />

          {/* Checkpoints */}
          <div>
            <h4 className="font-medium mb-2">Checkpoints</h4>
            {checkpoints?.length === 0 ? (
              <p className="text-muted-foreground text-sm">No checkpoints yet</p>
            ) : (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Epoch</TableHead>
                    <TableHead>Val Loss</TableHead>
                    <TableHead>R@1</TableHead>
                    <TableHead>R@5</TableHead>
                    <TableHead>mAP</TableHead>
                    <TableHead></TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {checkpoints?.map((checkpoint: TrainingCheckpoint) => (
                    <TableRow key={checkpoint.id}>
                      <TableCell>
                        <div className="flex items-center gap-2">
                          {checkpoint.epoch}
                          {checkpoint.is_best && (
                            <Badge variant="outline" className="text-xs">Best</Badge>
                          )}
                          {checkpoint.is_final && (
                            <Badge variant="outline" className="text-xs">Final</Badge>
                          )}
                        </div>
                      </TableCell>
                      <TableCell>{checkpoint.val_loss?.toFixed(4) || "-"}</TableCell>
                      <TableCell>
                        {checkpoint.val_recall_at_1
                          ? `${(checkpoint.val_recall_at_1 * 100).toFixed(1)}%`
                          : "-"}
                      </TableCell>
                      <TableCell>
                        {checkpoint.val_recall_at_5
                          ? `${(checkpoint.val_recall_at_5 * 100).toFixed(1)}%`
                          : "-"}
                      </TableCell>
                      <TableCell>
                        {checkpoint.val_map
                          ? `${(checkpoint.val_map * 100).toFixed(1)}%`
                          : "-"}
                      </TableCell>
                      <TableCell>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => {
                            const name = prompt("Model name:", `${run.name}-epoch${checkpoint.epoch}`);
                            if (name) {
                              registerModelMutation.mutate({
                                checkpoint_id: checkpoint.id,
                                name,
                              });
                            }
                          }}
                        >
                          Register
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}
          </div>

          {/* Error Message */}
          {run.error_message && (
            <>
              <Separator />
              <div>
                <h4 className="font-medium mb-2 text-red-600">Error</h4>
                <pre className="bg-red-50 dark:bg-red-950 p-3 rounded-md text-sm overflow-x-auto">
                  {run.error_message}
                </pre>
              </div>
            </>
          )}
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={onClose}>Close</Button>
          {bestCheckpoint && run.status === "completed" && (
            <Button
              onClick={() => {
                const name = prompt("Model name:", `${run.name}-best`);
                if (name) {
                  registerModelMutation.mutate({
                    checkpoint_id: bestCheckpoint.id,
                    name,
                  });
                }
              }}
            >
              Register Best Checkpoint
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
