"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { useRouter } from "next/navigation";
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
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Textarea } from "@/components/ui/textarea";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Checkbox } from "@/components/ui/checkbox";
import { Separator } from "@/components/ui/separator";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Slider } from "@/components/ui/slider";
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
  Play,
  Square,
  CheckCircle,
  XCircle,
  Clock,
  AlertTriangle,
  Cpu,
  Settings,
  Rocket,
  MoreHorizontal,
  Eye,
  StopCircle,
  Zap,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  ChevronsLeft,
  ChevronsRight,
  Sparkles,
  Search,
  ArrowUpDown,
  Trash2,
  Star,
  Download,
  Power,
  PowerOff,
  Plus,
} from "lucide-react";
import { Switch } from "@/components/ui/switch";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";

// Apache 2.0 Licensed Models Only
const MODEL_TYPES = [
  { id: "rt-detr", name: "RT-DETR", description: "Real-Time DETR - Fast & accurate transformer detector" },
  { id: "d-fine", name: "D-FINE", description: "Detection-aware FINe-grained matching - SOTA accuracy" },
];

const MODEL_SIZES: Record<string, Array<{ id: string; name: string; description: string }>> = {
  "rt-detr": [
    { id: "s", name: "Small", description: "ResNet-18 backbone, fastest" },
    { id: "m", name: "Medium", description: "ResNet-50 backbone, balanced" },
    { id: "l", name: "Large", description: "ResNet-101 backbone, best accuracy" },
  ],
  "d-fine": [
    { id: "s", name: "Small", description: "Lightweight, fast inference" },
    { id: "m", name: "Medium", description: "Balanced speed and accuracy" },
    { id: "l", name: "Large", description: "High accuracy" },
    { id: "x", name: "XLarge", description: "Best accuracy, slower" },
  ],
};

const AUGMENTATION_PRESETS = [
  { id: "sota", name: "SOTA (Recommended)", description: "Mosaic, MixUp, CopyPaste + standard", icon: "⭐", boost: "+3-5% mAP" },
  { id: "heavy", name: "Heavy", description: "All augmentations, ideal for small datasets", icon: "🔥", boost: "+5-8% mAP" },
  { id: "medium", name: "Medium", description: "Balanced augmentations", icon: "⚖️", boost: "+2-3% mAP" },
  { id: "light", name: "Light", description: "Basic augmentations only", icon: "🌱", boost: "+1% mAP" },
  { id: "none", name: "None", description: "No augmentation (baseline)", icon: "📊", boost: "Baseline" },
];

// Status badge configuration
const statusConfig: Record<string, { icon: React.ReactNode; color: string; label: string }> = {
  pending: {
    icon: <Clock className="h-3 w-3" />,
    color: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200",
    label: "Pending",
  },
  preparing: {
    icon: <Loader2 className="h-3 w-3 animate-spin" />,
    color: "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200",
    label: "Preparing",
  },
  queued: {
    icon: <Clock className="h-3 w-3" />,
    color: "bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200",
    label: "Queued",
  },
  training: {
    icon: <Loader2 className="h-3 w-3 animate-spin" />,
    color: "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200",
    label: "Training",
  },
  completed: {
    icon: <CheckCircle className="h-3 w-3" />,
    color: "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200",
    label: "Completed",
  },
  failed: {
    icon: <XCircle className="h-3 w-3" />,
    color: "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200",
    label: "Failed",
  },
  cancelled: {
    icon: <Square className="h-3 w-3" />,
    color: "bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200",
    label: "Cancelled",
  },
};

export default function ODTrainingPage() {
  const router = useRouter();
  const queryClient = useQueryClient();

  // Tab state
  const [activeTab, setActiveTab] = useState("runs");

  // Dialog state
  const [isNewTrainingOpen, setIsNewTrainingOpen] = useState(false);

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

  // Models tab state
  const [modelsSearchQuery, setModelsSearchQuery] = useState("");
  const [modelsFilterType, setModelsFilterType] = useState<string>("all");
  const [modelsCurrentPage, setModelsCurrentPage] = useState(1);
  const [modelsPageSize, setModelsPageSize] = useState(10);
  const [deleteModelId, setDeleteModelId] = useState<string | null>(null);

  // Form state - Basic
  const [formName, setFormName] = useState("");
  const [formDescription, setFormDescription] = useState("");
  const [formDatasetId, setFormDatasetId] = useState("");
  const [formVersionId, setFormVersionId] = useState("");
  const [formModelType, setFormModelType] = useState("rt-detr");
  const [formModelSize, setFormModelSize] = useState("l");
  const [formEpochs, setFormEpochs] = useState(100);
  const [formBatchSize, setFormBatchSize] = useState(16);
  const [formLearningRate, setFormLearningRate] = useState(0.0001);
  const [formImageSize, setFormImageSize] = useState(640);

  // Form state - SOTA features
  const [formAugPreset, setFormAugPreset] = useState("sota");
  const [formUseEma, setFormUseEma] = useState(true);
  const [formLlrdDecay, setFormLlrdDecay] = useState(0.9);
  const [formWarmupEpochs, setFormWarmupEpochs] = useState(3);
  const [formMixedPrecision, setFormMixedPrecision] = useState(true);
  const [formPatience, setFormPatience] = useState(20);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Individual augmentation controls
  const [useCustomAug, setUseCustomAug] = useState(false);
  const [formUseMosaic, setFormUseMosaic] = useState(true);
  const [formMosaicProb, setFormMosaicProb] = useState(0.5);
  const [formUseMixUp, setFormUseMixUp] = useState(true);
  const [formMixUpProb, setFormMixUpProb] = useState(0.3);
  const [formUseCopyPaste, setFormUseCopyPaste] = useState(true);
  const [formCopyPasteProb, setFormCopyPasteProb] = useState(0.3);
  const [formUseHorizontalFlip, setFormUseHorizontalFlip] = useState(true);
  const [formHorizontalFlipProb, setFormHorizontalFlipProb] = useState(0.5);
  const [formUseVerticalFlip, setFormUseVerticalFlip] = useState(false);
  const [formVerticalFlipProb, setFormVerticalFlipProb] = useState(0.5);
  const [formUseColorJitter, setFormUseColorJitter] = useState(true);
  const [formColorJitterStrength, setFormColorJitterStrength] = useState(0.3);
  const [formUseRandomCrop, setFormUseRandomCrop] = useState(true);
  const [formRandomCropScale, setFormRandomCropScale] = useState(0.5);
  const [showAugDetails, setShowAugDetails] = useState(false);

  // Fetch training runs
  const { data: trainingRuns, isLoading: isLoadingRuns, isFetching: isFetchingRuns } = useQuery({
    queryKey: ["od-training-runs"],
    queryFn: () => apiClient.getODTrainingRuns(),
    refetchInterval: 10000,
  });

  // Fetch trained models
  const { data: trainedModels, isLoading: isLoadingModels, isFetching: isFetchingModels } = useQuery({
    queryKey: ["od-models", modelsFilterType],
    queryFn: () => apiClient.getODTrainedModels({
      model_type: modelsFilterType !== "all" ? modelsFilterType : undefined,
    }),
  });

  // Fetch datasets for selection
  const { data: datasets } = useQuery({
    queryKey: ["od-datasets"],
    queryFn: () => apiClient.getODDatasets(),
  });

  // Fetch versions for selected dataset
  const { data: versions } = useQuery({
    queryKey: ["od-dataset-versions", formDatasetId],
    queryFn: () => apiClient.getODDatasetVersions(formDatasetId),
    enabled: !!formDatasetId,
  });

  // Start training mutation
  const startTrainingMutation = useMutation({
    mutationFn: (params: Parameters<typeof apiClient.createODTrainingRun>[0]) =>
      apiClient.createODTrainingRun(params),
    onSuccess: (data) => {
      toast.success("Training job started");
      queryClient.invalidateQueries({ queryKey: ["od-training-runs"] });
      setIsNewTrainingOpen(false);
      resetForm();
      router.push(`/od/training/${data.id}`);
    },
    onError: (error: Error) => {
      toast.error(`Failed to start training: ${error.message}`);
    },
  });

  // Cancel training mutation
  const cancelTrainingMutation = useMutation({
    mutationFn: (trainingId: string) => apiClient.cancelODTrainingRun(trainingId),
    onSuccess: () => {
      toast.success("Training cancelled");
      queryClient.invalidateQueries({ queryKey: ["od-training-runs"] });
    },
    onError: (error: Error) => {
      toast.error(`Failed to cancel training: ${error.message}`);
    },
  });

  // Delete training run mutation
  const deleteRunMutation = useMutation({
    mutationFn: (id: string) => apiClient.deleteODTrainingRun(id),
    onSuccess: () => {
      toast.success("Training run deleted");
      queryClient.invalidateQueries({ queryKey: ["od-training-runs"] });
    },
    onError: (error: Error) => {
      toast.error(`Failed to delete: ${error.message}`);
    },
  });

  // Bulk delete training runs mutation
  const bulkDeleteMutation = useMutation({
    mutationFn: async (ids: string[]) => {
      const results = await Promise.allSettled(
        ids.map(id => apiClient.deleteODTrainingRun(id))
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
      queryClient.invalidateQueries({ queryKey: ["od-training-runs"] });
    },
    onError: (error: Error) => {
      toast.error(error.message);
    },
  });

  // Model mutations
  const setDefaultMutation = useMutation({
    mutationFn: (modelId: string) => apiClient.setDefaultODModel(modelId),
    onSuccess: () => {
      toast.success("Model set as default");
      queryClient.invalidateQueries({ queryKey: ["od-models"] });
    },
    onError: (error: Error) => {
      toast.error(`Failed to set default: ${error.message}`);
    },
  });

  const activateMutation = useMutation({
    mutationFn: (modelId: string) => apiClient.updateODTrainedModel(modelId, { is_active: true }),
    onSuccess: () => {
      toast.success("Model activated");
      queryClient.invalidateQueries({ queryKey: ["od-models"] });
    },
    onError: (error: Error) => {
      toast.error(`Failed to activate: ${error.message}`);
    },
  });

  const deactivateMutation = useMutation({
    mutationFn: (modelId: string) => apiClient.updateODTrainedModel(modelId, { is_active: false }),
    onSuccess: () => {
      toast.success("Model deactivated");
      queryClient.invalidateQueries({ queryKey: ["od-models"] });
    },
    onError: (error: Error) => {
      toast.error(`Failed to deactivate: ${error.message}`);
    },
  });

  const deleteModelMutation = useMutation({
    mutationFn: (modelId: string) => apiClient.deleteODTrainedModel(modelId),
    onSuccess: () => {
      toast.success("Model deleted");
      queryClient.invalidateQueries({ queryKey: ["od-models"] });
      setDeleteModelId(null);
    },
    onError: (error: Error) => {
      toast.error(`Failed to delete: ${error.message}`);
    },
  });

  const handleDownload = async (modelId: string, modelName: string) => {
    try {
      toast.info("Preparing download...");
      const blob = await apiClient.downloadODModel(modelId);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${modelName}.pt`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      toast.success("Download started");
    } catch {
      toast.error("Failed to download model");
    }
  };

  const resetForm = () => {
    setFormName("");
    setFormDescription("");
    setFormDatasetId("");
    setFormVersionId("");
    setFormModelType("rt-detr");
    setFormModelSize("l");
    setFormEpochs(100);
    setFormBatchSize(16);
    setFormLearningRate(0.0001);
    setFormImageSize(640);
    setFormAugPreset("sota");
    setFormUseEma(true);
    setFormLlrdDecay(0.9);
    setFormWarmupEpochs(3);
    setFormMixedPrecision(true);
    setFormPatience(20);
    setShowAdvanced(false);
    setUseCustomAug(false);
    setFormUseMosaic(true);
    setFormMosaicProb(0.5);
    setFormUseMixUp(true);
    setFormMixUpProb(0.3);
    setFormUseCopyPaste(true);
    setFormCopyPasteProb(0.3);
    setFormUseHorizontalFlip(true);
    setFormHorizontalFlipProb(0.5);
    setFormUseVerticalFlip(false);
    setFormVerticalFlipProb(0.5);
    setFormUseColorJitter(true);
    setFormColorJitterStrength(0.3);
    setFormUseRandomCrop(true);
    setFormRandomCropScale(0.5);
    setShowAugDetails(false);
  };

  const handleStartTraining = () => {
    if (!formName || !formDatasetId) {
      toast.error("Please fill in all required fields");
      return;
    }
    startTrainingMutation.mutate({
      name: formName,
      description: formDescription || undefined,
      dataset_id: formDatasetId,
      dataset_version_id: formVersionId || undefined,
      model_type: formModelType,
      model_size: formModelSize,
      config: {
        epochs: formEpochs,
        batch_size: formBatchSize,
        learning_rate: formLearningRate,
        image_size: formImageSize,
        augmentation_preset: useCustomAug ? "custom" : formAugPreset,
        use_ema: formUseEma,
        llrd_decay: formLlrdDecay,
        warmup_epochs: formWarmupEpochs,
        mixed_precision: formMixedPrecision,
        patience: formPatience,
        ...(useCustomAug && {
          augmentation_config: {
            mosaic: { enabled: formUseMosaic, probability: formMosaicProb },
            mixup: { enabled: formUseMixUp, probability: formMixUpProb },
            copy_paste: { enabled: formUseCopyPaste, probability: formCopyPasteProb },
            horizontal_flip: { enabled: formUseHorizontalFlip, probability: formHorizontalFlipProb },
            vertical_flip: { enabled: formUseVerticalFlip, probability: formVerticalFlipProb },
            color_jitter: { enabled: formUseColorJitter, strength: formColorJitterStrength },
            random_crop: { enabled: formUseRandomCrop, scale: formRandomCropScale },
          },
        }),
      },
    });
  };

  // Filter and sort training runs
  const runs = trainingRuns || [];
  const filteredRuns = runs.filter((run) => {
    const matchesSearch = searchQuery === "" ||
      run.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      run.model_type.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesStatus = statusFilter === "all" || run.status === statusFilter;
    return matchesSearch && matchesStatus;
  });

  const sortedRuns = [...filteredRuns].sort((a, b) => {
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
      case "map":
        aValue = a.best_map || 0;
        bValue = b.best_map || 0;
        break;
      default:
        return 0;
    }

    if (aValue < bValue) return sortConfig.direction === "asc" ? -1 : 1;
    if (aValue > bValue) return sortConfig.direction === "asc" ? 1 : -1;
    return 0;
  });

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
        .filter((run) => !["training", "preparing", "queued"].includes(run.status))
        .map((run) => run.id);
      setSelectedRunIds(new Set([...selectedRunIds, ...selectableIds]));
    } else {
      const currentPageIds = new Set(paginatedRuns.map((run) => run.id));
      setSelectedRunIds(new Set([...selectedRunIds].filter(id => !currentPageIds.has(id))));
    }
  };

  const selectableOnPage = paginatedRuns.filter((run) => !["training", "preparing", "queued"].includes(run.status));
  const allSelectedOnPage = selectableOnPage.length > 0 &&
    selectableOnPage.every((run) => selectedRunIds.has(run.id));
  const someSelectedOnPage = selectableOnPage.some((run) => selectedRunIds.has(run.id));

  // Filter models
  const filteredModels = trainedModels?.filter(
    (m) =>
      m.name.toLowerCase().includes(modelsSearchQuery.toLowerCase()) ||
      m.model_type.toLowerCase().includes(modelsSearchQuery.toLowerCase())
  ) || [];

  const modelsTotalPages = Math.ceil(filteredModels.length / modelsPageSize);
  const paginatedModels = filteredModels.slice(
    (modelsCurrentPage - 1) * modelsPageSize,
    modelsCurrentPage * modelsPageSize
  );

  const selectedDataset = datasets?.find((d) => d.id === formDatasetId);
  const runningCount = runs.filter((r) => ["training", "preparing", "queued"].includes(r.status)).length;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Training</h1>
          <p className="text-muted-foreground">
            Train object detection models with RT-DETR or D-FINE
          </p>
        </div>
        <Button onClick={() => {
          resetForm();
          setIsNewTrainingOpen(true);
        }}>
          <Plus className="h-4 w-4 mr-2" />
          New Training Run
        </Button>
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
                  onClick={() => queryClient.invalidateQueries({ queryKey: ["od-training-runs"] })}
                  disabled={isFetchingRuns}
                >
                  <RefreshCw className={`h-4 w-4 ${isFetchingRuns ? "animate-spin" : ""}`} />
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
                    <SelectItem value="training">Training</SelectItem>
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
                            </button>
                          </TableHead>
                          <TableHead>Model</TableHead>
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
                              onClick={() => handleSort("map")}
                            >
                              Best mAP
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
                        {paginatedRuns.map((run) => {
                          const isSelectable = !["training", "preparing", "queued"].includes(run.status);
                          const isSelected = selectedRunIds.has(run.id);
                          const status = statusConfig[run.status] || statusConfig.pending;
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
                                <div className="flex flex-col">
                                  <Badge variant="outline" className="w-fit">
                                    {run.model_type.toUpperCase()}
                                  </Badge>
                                  <span className="text-xs text-muted-foreground mt-1">
                                    {run.model_size}
                                  </span>
                                </div>
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
                                {run.best_map !== null && run.best_map !== undefined ? (
                                  <div className="text-sm">
                                    <p className="font-medium">{(run.best_map * 100).toFixed(1)}%</p>
                                    <p className="text-muted-foreground text-xs">
                                      epoch {run.best_epoch}
                                    </p>
                                  </div>
                                ) : (
                                  <span className="text-muted-foreground">-</span>
                                )}
                              </TableCell>
                              <TableCell>
                                <Badge className={status.color}>
                                  <span className="mr-1">{status.icon}</span>
                                  {status.label}
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
                                    <DropdownMenuItem asChild>
                                      <Link href={`/od/training/${run.id}`}>
                                        <Eye className="h-4 w-4 mr-2" />
                                        View Details
                                      </Link>
                                    </DropdownMenuItem>
                                    {["training", "preparing", "queued", "pending"].includes(run.status) && (
                                      <>
                                        <DropdownMenuSeparator />
                                        <DropdownMenuItem
                                          onClick={() => cancelTrainingMutation.mutate(run.id)}
                                          className="text-orange-600"
                                        >
                                          <StopCircle className="h-4 w-4 mr-2" />
                                          Cancel Training
                                        </DropdownMenuItem>
                                      </>
                                    )}
                                    {isSelectable && (
                                      <>
                                        <DropdownMenuSeparator />
                                        <DropdownMenuItem
                                          onClick={() => deleteRunMutation.mutate(run.id)}
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

        {/* Trained Models Tab */}
        <TabsContent value="models" className="mt-6">
          <Card>
            <CardHeader className="pb-4">
              <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                <div>
                  <CardTitle>Trained Models</CardTitle>
                  <CardDescription>Registered models ready for deployment</CardDescription>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => queryClient.invalidateQueries({ queryKey: ["od-models"] })}
                  disabled={isFetchingModels}
                >
                  <RefreshCw className={`h-4 w-4 ${isFetchingModels ? "animate-spin" : ""}`} />
                </Button>
              </div>

              {/* Filters */}
              <div className="flex flex-col gap-3 sm:flex-row sm:items-center mt-4">
                <div className="relative flex-1 max-w-sm">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <Input
                    placeholder="Search models..."
                    value={modelsSearchQuery}
                    onChange={(e) => {
                      setModelsSearchQuery(e.target.value);
                      setModelsCurrentPage(1);
                    }}
                    className="pl-9"
                  />
                </div>
                <Select value={modelsFilterType} onValueChange={setModelsFilterType}>
                  <SelectTrigger className="w-[150px]">
                    <SelectValue placeholder="Model Type" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Types</SelectItem>
                    <SelectItem value="rt-detr">RT-DETR</SelectItem>
                    <SelectItem value="d-fine">D-FINE</SelectItem>
                  </SelectContent>
                </Select>
                <Select
                  value={modelsPageSize.toString()}
                  onValueChange={(v) => {
                    setModelsPageSize(Number(v));
                    setModelsCurrentPage(1);
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
            </CardHeader>

            <CardContent>
              {isLoadingModels ? (
                <div className="flex items-center justify-center py-8">
                  <Loader2 className="h-6 w-6 animate-spin" />
                </div>
              ) : filteredModels.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground">
                  <Cpu className="h-12 w-12 mx-auto mb-2 opacity-50" />
                  <p>No models found</p>
                  <p className="text-sm mt-1">
                    {modelsSearchQuery || modelsFilterType !== "all"
                      ? "Try adjusting your filters"
                      : "Train your first model to see it here"}
                  </p>
                </div>
              ) : (
                <>
                  <div className="rounded-md border">
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>Name</TableHead>
                          <TableHead>Type</TableHead>
                          <TableHead>mAP</TableHead>
                          <TableHead>mAP@50</TableHead>
                          <TableHead>Classes</TableHead>
                          <TableHead>Status</TableHead>
                          <TableHead>Created</TableHead>
                          <TableHead className="w-12"></TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {paginatedModels.map((model) => (
                          <TableRow
                            key={model.id}
                            className="cursor-pointer hover:bg-muted/50"
                            onClick={() => router.push(`/od/models/${model.id}`)}
                          >
                            <TableCell>
                              <div className="flex items-center gap-2">
                                <span className="font-medium">{model.name}</span>
                                {model.is_default && (
                                  <Star className="h-4 w-4 fill-yellow-500 text-yellow-500" />
                                )}
                              </div>
                            </TableCell>
                            <TableCell>
                              <Badge variant="outline">{model.model_type.toUpperCase()}</Badge>
                            </TableCell>
                            <TableCell className="font-mono">
                              {model.map !== null && model.map !== undefined
                                ? `${(model.map * 100).toFixed(1)}%`
                                : "-"}
                            </TableCell>
                            <TableCell className="font-mono">
                              {model.map_50 !== null && model.map_50 !== undefined
                                ? `${(model.map_50 * 100).toFixed(1)}%`
                                : "-"}
                            </TableCell>
                            <TableCell>{model.class_count}</TableCell>
                            <TableCell>
                              {model.is_active ? (
                                <Badge variant="default" className="bg-green-600">
                                  <CheckCircle className="h-3 w-3 mr-1" />
                                  Active
                                </Badge>
                              ) : (
                                <Badge variant="secondary">Inactive</Badge>
                              )}
                            </TableCell>
                            <TableCell className="text-muted-foreground">
                              {new Date(model.created_at).toLocaleDateString()}
                            </TableCell>
                            <TableCell onClick={(e) => e.stopPropagation()}>
                              <DropdownMenu>
                                <DropdownMenuTrigger asChild>
                                  <Button variant="ghost" size="icon" className="h-8 w-8">
                                    <MoreHorizontal className="h-4 w-4" />
                                  </Button>
                                </DropdownMenuTrigger>
                                <DropdownMenuContent align="end">
                                  <DropdownMenuItem
                                    onClick={() => router.push(`/od/models/${model.id}`)}
                                  >
                                    <Eye className="h-4 w-4 mr-2" />
                                    View Details
                                  </DropdownMenuItem>
                                  <DropdownMenuItem
                                    onClick={() => handleDownload(model.id, model.name)}
                                  >
                                    <Download className="h-4 w-4 mr-2" />
                                    Download
                                  </DropdownMenuItem>
                                  <DropdownMenuSeparator />
                                  {model.is_active ? (
                                    <DropdownMenuItem
                                      onClick={() => deactivateMutation.mutate(model.id)}
                                      disabled={model.is_default}
                                    >
                                      <PowerOff className="h-4 w-4 mr-2" />
                                      Deactivate
                                    </DropdownMenuItem>
                                  ) : (
                                    <DropdownMenuItem
                                      onClick={() => activateMutation.mutate(model.id)}
                                    >
                                      <Power className="h-4 w-4 mr-2" />
                                      Activate
                                    </DropdownMenuItem>
                                  )}
                                  {!model.is_default && model.is_active && (
                                    <DropdownMenuItem
                                      onClick={() => setDefaultMutation.mutate(model.id)}
                                    >
                                      <Star className="h-4 w-4 mr-2" />
                                      Set as Default
                                    </DropdownMenuItem>
                                  )}
                                  <DropdownMenuSeparator />
                                  <DropdownMenuItem
                                    className="text-destructive"
                                    onClick={() => setDeleteModelId(model.id)}
                                    disabled={model.is_default}
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
                  </div>

                  {/* Pagination */}
                  <div className="flex items-center justify-between mt-4">
                    <p className="text-sm text-muted-foreground">
                      Showing {((modelsCurrentPage - 1) * modelsPageSize) + 1} to {Math.min(modelsCurrentPage * modelsPageSize, filteredModels.length)} of {filteredModels.length} models
                    </p>
                    <div className="flex items-center gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setModelsCurrentPage(1)}
                        disabled={modelsCurrentPage === 1}
                      >
                        <ChevronsLeft className="h-4 w-4" />
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setModelsCurrentPage(p => Math.max(1, p - 1))}
                        disabled={modelsCurrentPage === 1}
                      >
                        <ChevronLeft className="h-4 w-4" />
                      </Button>
                      <span className="text-sm px-2">
                        Page {modelsCurrentPage} of {modelsTotalPages || 1}
                      </span>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setModelsCurrentPage(p => Math.min(modelsTotalPages, p + 1))}
                        disabled={modelsCurrentPage >= modelsTotalPages}
                      >
                        <ChevronRight className="h-4 w-4" />
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setModelsCurrentPage(modelsTotalPages)}
                        disabled={modelsCurrentPage >= modelsTotalPages}
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
      </Tabs>

      {/* Bulk Delete Confirmation Dialog */}
      <Dialog open={bulkDeleteDialogOpen} onOpenChange={setBulkDeleteDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <AlertTriangle className="h-5 w-5 text-destructive" />
              Delete {selectedRunIds.size} Training Run{selectedRunIds.size > 1 ? "s" : ""}?
            </DialogTitle>
            <DialogDescription>
              This will permanently delete the selected training runs.
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

      {/* Delete Model Confirmation Dialog */}
      <AlertDialog open={!!deleteModelId} onOpenChange={() => setDeleteModelId(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Model</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete this model? This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
              onClick={() => deleteModelId && deleteModelMutation.mutate(deleteModelId)}
            >
              {deleteModelMutation.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                "Delete"
              )}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* New Training Dialog */}
      <Dialog open={isNewTrainingOpen} onOpenChange={setIsNewTrainingOpen}>
        <DialogContent className="max-w-3xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Rocket className="h-5 w-5" />
              New Training Job
            </DialogTitle>
            <DialogDescription>
              Configure and start a new model training run
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-6 py-4">
            {/* Basic Info */}
            <div className="space-y-4">
              <Label className="text-base font-semibold">Basic Information</Label>
              <div className="space-y-2">
                <Label className="text-sm">Training Name *</Label>
                <Input
                  placeholder="e.g., Shelf Detection v1"
                  value={formName}
                  onChange={(e) => setFormName(e.target.value)}
                />
              </div>
              <div className="space-y-2">
                <Label className="text-sm">Description</Label>
                <Textarea
                  placeholder="Optional description for this training run..."
                  value={formDescription}
                  onChange={(e) => setFormDescription(e.target.value)}
                  rows={2}
                />
              </div>
            </div>

            <Separator />

            {/* Dataset Selection */}
            <div className="space-y-4">
              <Label className="text-base font-semibold">Dataset</Label>
              <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label>Dataset *</Label>
                <Select value={formDatasetId} onValueChange={(v) => {
                  setFormDatasetId(v);
                  setFormVersionId("");
                }}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select dataset..." />
                  </SelectTrigger>
                  <SelectContent>
                    {datasets?.map((ds) => (
                      <SelectItem key={ds.id} value={ds.id}>
                        {ds.name} ({ds.annotated_image_count} annotated)
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label>Version (Optional)</Label>
                <Select
                  value={formVersionId || "__latest__"}
                  onValueChange={(v) => setFormVersionId(v === "__latest__" ? "" : v)}
                  disabled={!formDatasetId || !versions?.length}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Latest (all images)" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="__latest__">Latest (all images)</SelectItem>
                    {versions?.map((v) => (
                      <SelectItem key={v.id} value={v.id}>
                        v{v.version_number} - {v.name || "Unnamed"} ({v.image_count} images)
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>

              {selectedDataset && (
                <div className="bg-muted rounded-lg p-4">
                  <p className="text-sm font-medium">Dataset Info</p>
                  <div className="grid grid-cols-3 gap-4 mt-2 text-sm">
                    <div>
                      <span className="text-muted-foreground">Images:</span>{" "}
                      <span className="font-medium">{selectedDataset.image_count}</span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Annotated:</span>{" "}
                      <span className="font-medium">{selectedDataset.annotated_image_count}</span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Annotations:</span>{" "}
                      <span className="font-medium">{selectedDataset.annotation_count}</span>
                    </div>
                  </div>
                  {selectedDataset.annotation_count === 0 && (
                    <div className="flex items-center gap-2 mt-3 text-red-600 text-sm">
                      <AlertTriangle className="h-4 w-4" />
                      Dataset has no annotations. Please annotate images before training.
                    </div>
                  )}
                  {selectedDataset.annotated_image_count > 0 &&
                    selectedDataset.annotated_image_count < selectedDataset.image_count && (
                      <div className="flex items-center gap-2 mt-3 text-orange-600 text-sm">
                        <AlertTriangle className="h-4 w-4" />
                        Not all images are annotated. Consider completing annotations first.
                      </div>
                    )}
                </div>
              )}
            </div>

            <Separator />

            {/* Model Selection */}
            <div className="space-y-4">
              <Label className="text-base font-semibold">Model</Label>
              <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label>Model Type</Label>
                <Select value={formModelType} onValueChange={(v) => {
                  setFormModelType(v);
                  setFormModelSize(MODEL_SIZES[v]?.[0]?.id || "l");
                }}>
                  <SelectTrigger>
                    <SelectValue>
                      {MODEL_TYPES.find(m => m.id === formModelType)?.name || "Select..."}
                    </SelectValue>
                  </SelectTrigger>
                  <SelectContent>
                    {MODEL_TYPES.map((model) => (
                      <SelectItem key={model.id} value={model.id}>
                        <div className="flex flex-col">
                          <span className="font-medium">{model.name}</span>
                          <span className="text-xs text-muted-foreground">
                            {model.description}
                          </span>
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <p className="text-xs text-muted-foreground">
                  {MODEL_TYPES.find(m => m.id === formModelType)?.description}
                </p>
              </div>
              <div className="space-y-2">
                <Label>Model Size</Label>
                <Select value={formModelSize} onValueChange={setFormModelSize}>
                  <SelectTrigger>
                    <SelectValue>
                      {MODEL_SIZES[formModelType]?.find(s => s.id === formModelSize)?.name || "Select..."}
                    </SelectValue>
                  </SelectTrigger>
                  <SelectContent>
                    {(MODEL_SIZES[formModelType] || []).map((size) => (
                      <SelectItem key={size.id} value={size.id}>
                        <div className="flex flex-col">
                          <span className="font-medium capitalize">{size.name}</span>
                          <span className="text-xs text-muted-foreground">
                            {size.description}
                          </span>
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <p className="text-xs text-muted-foreground">
                  {MODEL_SIZES[formModelType]?.find(s => s.id === formModelSize)?.description}
                </p>
              </div>
              </div>
            </div>

            <Separator />

            {/* Training Parameters */}
            <div className="space-y-4">
              <div className="flex items-center gap-2">
                <Settings className="h-4 w-4" />
                <Label className="text-base font-semibold">Training Parameters</Label>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>Epochs: {formEpochs}</Label>
                  <Slider
                    value={[formEpochs]}
                    onValueChange={(v) => setFormEpochs(v[0])}
                    min={10}
                    max={300}
                    step={10}
                  />
                </div>
                <div className="space-y-2">
                  <Label>Batch Size: {formBatchSize}</Label>
                  <Slider
                    value={[formBatchSize]}
                    onValueChange={(v) => setFormBatchSize(v[0])}
                    min={4}
                    max={64}
                    step={4}
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>Learning Rate</Label>
                  <Select
                    value={formLearningRate.toString()}
                    onValueChange={(v) => setFormLearningRate(parseFloat(v))}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="0.001">0.001 (High)</SelectItem>
                      <SelectItem value="0.0001">0.0001 (Default)</SelectItem>
                      <SelectItem value="0.00001">0.00001 (Low)</SelectItem>
                      <SelectItem value="0.000001">0.000001 (Very Low)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label>Image Size</Label>
                  <Select
                    value={formImageSize.toString()}
                    onValueChange={(v) => setFormImageSize(parseInt(v))}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="416">416px</SelectItem>
                      <SelectItem value="512">512px</SelectItem>
                      <SelectItem value="640">640px (Default)</SelectItem>
                      <SelectItem value="800">800px</SelectItem>
                      <SelectItem value="1024">1024px</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </div>

            <Separator />

            {/* SOTA Features */}
            <div className="space-y-4">
              <div className="flex items-center gap-2">
                <Zap className="h-4 w-4 text-amber-500" />
                <Label className="text-base font-semibold">SOTA Features</Label>
                <Badge variant="outline" className="text-xs">Advanced</Badge>
              </div>

              {/* Augmentation Preset or Custom */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <Label>Augmentation Settings</Label>
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-muted-foreground">Preset</span>
                    <Switch checked={useCustomAug} onCheckedChange={setUseCustomAug} />
                    <span className="text-xs text-muted-foreground">Custom</span>
                  </div>
                </div>

                {!useCustomAug ? (
                  <Select value={formAugPreset} onValueChange={setFormAugPreset}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {AUGMENTATION_PRESETS.map((preset) => (
                        <SelectItem key={preset.id} value={preset.id}>
                          <div className="flex items-center gap-2">
                            <span>{preset.icon}</span>
                            <div className="flex flex-col">
                              <span className="font-medium">{preset.name}</span>
                              <span className="text-xs text-muted-foreground">
                                {preset.description}
                              </span>
                            </div>
                            <Badge variant="secondary" className="ml-auto text-xs">
                              {preset.boost}
                            </Badge>
                          </div>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                ) : (
                  <div className="space-y-3 border rounded-lg p-4 bg-muted/30">
                    <p className="text-sm font-medium text-muted-foreground">
                      Configure augmentations individually
                    </p>

                    {/* Mosaic */}
                    <div className="flex items-center justify-between py-2 border-b">
                      <div className="flex items-center gap-3">
                        <Switch checked={formUseMosaic} onCheckedChange={setFormUseMosaic} />
                        <div>
                          <p className="text-sm font-medium">Mosaic</p>
                          <p className="text-xs text-muted-foreground">Combines 4 images, ideal for small objects</p>
                        </div>
                      </div>
                      {formUseMosaic && (
                        <div className="flex items-center gap-2 w-32">
                          <Slider
                            value={[formMosaicProb]}
                            onValueChange={(v) => setFormMosaicProb(v[0])}
                            min={0.1}
                            max={1.0}
                            step={0.1}
                            className="flex-1"
                          />
                          <span className="text-xs w-8">{(formMosaicProb * 100).toFixed(0)}%</span>
                        </div>
                      )}
                    </div>

                    {/* MixUp */}
                    <div className="flex items-center justify-between py-2 border-b">
                      <div className="flex items-center gap-3">
                        <Switch checked={formUseMixUp} onCheckedChange={setFormUseMixUp} />
                        <div>
                          <p className="text-sm font-medium">MixUp</p>
                          <p className="text-xs text-muted-foreground">Blends images together, provides regularization</p>
                        </div>
                      </div>
                      {formUseMixUp && (
                        <div className="flex items-center gap-2 w-32">
                          <Slider
                            value={[formMixUpProb]}
                            onValueChange={(v) => setFormMixUpProb(v[0])}
                            min={0.1}
                            max={1.0}
                            step={0.1}
                            className="flex-1"
                          />
                          <span className="text-xs w-8">{(formMixUpProb * 100).toFixed(0)}%</span>
                        </div>
                      )}
                    </div>

                    {/* CopyPaste */}
                    <div className="flex items-center justify-between py-2 border-b">
                      <div className="flex items-center gap-3">
                        <Switch checked={formUseCopyPaste} onCheckedChange={setFormUseCopyPaste} />
                        <div>
                          <p className="text-sm font-medium">CopyPaste</p>
                          <p className="text-xs text-muted-foreground">Copies and pastes objects between images</p>
                        </div>
                      </div>
                      {formUseCopyPaste && (
                        <div className="flex items-center gap-2 w-32">
                          <Slider
                            value={[formCopyPasteProb]}
                            onValueChange={(v) => setFormCopyPasteProb(v[0])}
                            min={0.1}
                            max={1.0}
                            step={0.1}
                            className="flex-1"
                          />
                          <span className="text-xs w-8">{(formCopyPasteProb * 100).toFixed(0)}%</span>
                        </div>
                      )}
                    </div>

                    {/* More Augmentations */}
                    <Collapsible open={showAugDetails} onOpenChange={setShowAugDetails}>
                      <CollapsibleTrigger asChild>
                        <Button variant="ghost" size="sm" className="w-full justify-between mt-2">
                          <span>More Augmentations</span>
                          <ChevronDown className={`h-4 w-4 transition-transform ${showAugDetails ? "rotate-180" : ""}`} />
                        </Button>
                      </CollapsibleTrigger>
                      <CollapsibleContent className="space-y-3 pt-2">
                        {/* Horizontal Flip */}
                        <div className="flex items-center justify-between py-2 border-b">
                          <div className="flex items-center gap-3">
                            <Switch checked={formUseHorizontalFlip} onCheckedChange={setFormUseHorizontalFlip} />
                            <div>
                              <p className="text-sm font-medium">Horizontal Flip</p>
                              <p className="text-xs text-muted-foreground">Flips image horizontally</p>
                            </div>
                          </div>
                          {formUseHorizontalFlip && (
                            <div className="flex items-center gap-2 w-32">
                              <Slider
                                value={[formHorizontalFlipProb]}
                                onValueChange={(v) => setFormHorizontalFlipProb(v[0])}
                                min={0.1}
                                max={1.0}
                                step={0.1}
                                className="flex-1"
                              />
                              <span className="text-xs w-8">{(formHorizontalFlipProb * 100).toFixed(0)}%</span>
                            </div>
                          )}
                        </div>

                        {/* Vertical Flip */}
                        <div className="flex items-center justify-between py-2 border-b">
                          <div className="flex items-center gap-3">
                            <Switch checked={formUseVerticalFlip} onCheckedChange={setFormUseVerticalFlip} />
                            <div>
                              <p className="text-sm font-medium">Vertical Flip</p>
                              <p className="text-xs text-muted-foreground">Flips image vertically (caution: not suitable for some objects)</p>
                            </div>
                          </div>
                          {formUseVerticalFlip && (
                            <div className="flex items-center gap-2 w-32">
                              <Slider
                                value={[formVerticalFlipProb]}
                                onValueChange={(v) => setFormVerticalFlipProb(v[0])}
                                min={0.1}
                                max={1.0}
                                step={0.1}
                                className="flex-1"
                              />
                              <span className="text-xs w-8">{(formVerticalFlipProb * 100).toFixed(0)}%</span>
                            </div>
                          )}
                        </div>

                        {/* Color Jitter */}
                        <div className="flex items-center justify-between py-2 border-b">
                          <div className="flex items-center gap-3">
                            <Switch checked={formUseColorJitter} onCheckedChange={setFormUseColorJitter} />
                            <div>
                              <p className="text-sm font-medium">Color Jitter</p>
                              <p className="text-xs text-muted-foreground">Color variations (brightness, contrast, saturation)</p>
                            </div>
                          </div>
                          {formUseColorJitter && (
                            <div className="flex items-center gap-2 w-32">
                              <Slider
                                value={[formColorJitterStrength]}
                                onValueChange={(v) => setFormColorJitterStrength(v[0])}
                                min={0.1}
                                max={0.5}
                                step={0.05}
                                className="flex-1"
                              />
                              <span className="text-xs w-8">{(formColorJitterStrength * 100).toFixed(0)}%</span>
                            </div>
                          )}
                        </div>

                        {/* Random Crop */}
                        <div className="flex items-center justify-between py-2">
                          <div className="flex items-center gap-3">
                            <Switch checked={formUseRandomCrop} onCheckedChange={setFormUseRandomCrop} />
                            <div>
                              <p className="text-sm font-medium">Random Crop</p>
                              <p className="text-xs text-muted-foreground">Random cropping and zoom</p>
                            </div>
                          </div>
                          {formUseRandomCrop && (
                            <div className="flex items-center gap-2 w-32">
                              <Slider
                                value={[formRandomCropScale]}
                                onValueChange={(v) => setFormRandomCropScale(v[0])}
                                min={0.3}
                                max={0.9}
                                step={0.1}
                                className="flex-1"
                              />
                              <span className="text-xs w-8">{(formRandomCropScale * 100).toFixed(0)}%</span>
                            </div>
                          )}
                        </div>
                      </CollapsibleContent>
                    </Collapsible>
                  </div>
                )}
              </div>

              {/* Quick Toggles */}
              <div className="grid grid-cols-2 gap-3">
                <div className="flex items-center justify-between p-3 border rounded-lg">
                  <div className="flex items-center gap-2">
                    <Sparkles className="h-4 w-4 text-blue-500" />
                    <div>
                      <p className="text-sm font-medium">EMA</p>
                      <p className="text-xs text-muted-foreground">Stable training</p>
                    </div>
                  </div>
                  <Switch checked={formUseEma} onCheckedChange={setFormUseEma} />
                </div>
                <div className="flex items-center justify-between p-3 border rounded-lg">
                  <div className="flex items-center gap-2">
                    <Zap className="h-4 w-4 text-green-500" />
                    <div>
                      <p className="text-sm font-medium">Mixed Precision</p>
                      <p className="text-xs text-muted-foreground">FP16 acceleration</p>
                    </div>
                  </div>
                  <Switch checked={formMixedPrecision} onCheckedChange={setFormMixedPrecision} />
                </div>
              </div>

              {/* Advanced Settings */}
              <Collapsible open={showAdvanced} onOpenChange={setShowAdvanced}>
                <CollapsibleTrigger asChild>
                  <Button variant="ghost" size="sm" className="w-full justify-between">
                    <span>Advanced Settings</span>
                    <ChevronDown className={`h-4 w-4 transition-transform ${showAdvanced ? "rotate-180" : ""}`} />
                  </Button>
                </CollapsibleTrigger>
                <CollapsibleContent className="space-y-4 pt-2">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label>LLRD Decay: {formLlrdDecay}</Label>
                      <Slider
                        value={[formLlrdDecay]}
                        onValueChange={(v) => setFormLlrdDecay(v[0])}
                        min={0.7}
                        max={1.0}
                        step={0.05}
                      />
                      <p className="text-xs text-muted-foreground">Layer-wise Learning Rate Decay</p>
                    </div>
                    <div className="space-y-2">
                      <Label>Warmup Epochs: {formWarmupEpochs}</Label>
                      <Slider
                        value={[formWarmupEpochs]}
                        onValueChange={(v) => setFormWarmupEpochs(v[0])}
                        min={0}
                        max={10}
                        step={1}
                      />
                      <p className="text-xs text-muted-foreground">Number of warmup epochs</p>
                    </div>
                  </div>
                  <div className="space-y-2">
                    <Label>Early Stopping Patience: {formPatience}</Label>
                    <Slider
                      value={[formPatience]}
                      onValueChange={(v) => setFormPatience(v[0])}
                      min={5}
                      max={50}
                      step={5}
                    />
                    <p className="text-xs text-muted-foreground">Epochs to wait without improvement</p>
                  </div>
                </CollapsibleContent>
              </Collapsible>
            </div>
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setIsNewTrainingOpen(false)}>
              Cancel
            </Button>
            <Button
              onClick={handleStartTraining}
              disabled={
                !formName ||
                !formDatasetId ||
                selectedDataset?.annotation_count === 0 ||
                startTrainingMutation.isPending
              }
            >
              {startTrainingMutation.isPending ? (
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
