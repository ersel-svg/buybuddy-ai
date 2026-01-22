"use client";

import { useState, use, useMemo, useEffect } from "react";
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
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";
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
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetDescription,
} from "@/components/ui/sheet";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  RefreshCw,
  Loader2,
  ImageIcon,
  Search,
  Plus,
  MoreHorizontal,
  Trash2,
  PenTool,
  ArrowLeft,
  CheckCircle,
  Clock,
  AlertCircle,
  XCircle,
  BarChart3,
  Layers,
  ChevronRight,
  Upload,
  FolderOpen,
  Tags,
  Edit,
  Palette,
  ArrowRight,
  History,
  SplitSquareVertical,
  Brain,
} from "lucide-react";
import Image from "next/image";
import Link from "next/link";

// Debounce hook
function useDebounce<T>(value: T, delay: number): T {
  const [debouncedValue, setDebouncedValue] = useState<T>(value);
  useEffect(() => {
    const timer = setTimeout(() => setDebouncedValue(value), delay);
    return () => clearTimeout(timer);
  }, [value, delay]);
  return debouncedValue;
}

// Predefined colors for class labels
const PRESET_COLORS = [
  "#ef4444", "#f97316", "#eab308", "#22c55e", "#14b8a6",
  "#3b82f6", "#8b5cf6", "#ec4899", "#6366f1", "#06b6d4",
];

interface CLSClass {
  id: string;
  name: string;
  display_name?: string;
  color: string;
  image_count: number;
  is_active: boolean;
}

interface DatasetImage {
  id: string;
  dataset_id: string;
  image_id: string;
  status: string;
  split?: string;
  added_at: string;
  image: {
    id: string;
    filename: string;
    image_url: string;
    thumbnail_url?: string;
  };
  labels?: Array<{
    id: string;
    class_id: string;
    class_name?: string;
    class_color?: string;
    confidence?: number;
  }>;
}

const PAGE_SIZE = 48;

export default function CLSDatasetDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id: datasetId } = use(params);
  const router = useRouter();
  const queryClient = useQueryClient();

  // Tab state
  const [activeTab, setActiveTab] = useState<"images" | "classes" | "versions">("images");

  // Images tab state
  const [page, setPage] = useState(1);
  const [searchInput, setSearchInput] = useState("");
  const debouncedSearch = useDebounce(searchInput, 300);
  const [filterStatus, setFilterStatus] = useState<string>("all");
  const [filterSplit, setFilterSplit] = useState<string>("all");
  const [selectedImages, setSelectedImages] = useState<Set<string>>(new Set());

  // Add images sheet state
  const [isAddImagesSheetOpen, setIsAddImagesSheetOpen] = useState(false);
  const [availableImagesPage, setAvailableImagesPage] = useState(1);
  const [availableSearch, setAvailableSearch] = useState("");
  const [selectedToAdd, setSelectedToAdd] = useState<Set<string>>(new Set());

  // Classes tab state
  const [classSearch, setClassSearch] = useState("");
  const [isCreateClassDialogOpen, setIsCreateClassDialogOpen] = useState(false);
  const [formName, setFormName] = useState("");
  const [formDisplayName, setFormDisplayName] = useState("");
  const [formColor, setFormColor] = useState(PRESET_COLORS[0]);

  // Versions tab state
  const [isCreateVersionDialogOpen, setIsCreateVersionDialogOpen] = useState(false);
  const [versionName, setVersionName] = useState("");
  const [versionDescription, setVersionDescription] = useState("");

  // Auto-split dialog state
  const [isAutoSplitDialogOpen, setIsAutoSplitDialogOpen] = useState(false);
  const [trainRatio, setTrainRatio] = useState(0.8);
  const [valRatio, setValRatio] = useState(0.1);
  const [testRatio, setTestRatio] = useState(0.1);

  // Fetch dataset info
  const { data: dataset, isLoading: datasetLoading } = useQuery({
    queryKey: ["cls-dataset", datasetId],
    queryFn: () => apiClient.getCLSDataset(datasetId),
    staleTime: 30000,
  });

  // Fetch dataset images
  const {
    data: imagesData,
    isLoading: imagesLoading,
    isFetching,
  } = useQuery({
    queryKey: ["cls-dataset-images", datasetId, page, filterStatus, filterSplit],
    queryFn: () =>
      apiClient.getCLSDatasetImages(datasetId, {
        page,
        limit: PAGE_SIZE,
        has_label: filterStatus === "labeled" ? true : filterStatus === "unlabeled" ? false : undefined,
        split: filterSplit !== "all" ? filterSplit : undefined,
      }),
    staleTime: 30000,
  });

  // Fetch dataset classes
  const { data: classes } = useQuery({
    queryKey: ["cls-dataset-classes", datasetId],
    queryFn: () => apiClient.getCLSDatasetClasses(datasetId),
    staleTime: 60000,
  });

  // Fetch dataset versions
  const { data: versions } = useQuery({
    queryKey: ["cls-dataset-versions", datasetId],
    queryFn: () => apiClient.getCLSDatasetVersions(datasetId),
    staleTime: 60000,
    enabled: activeTab === "versions",
  });

  // Fetch split stats
  const { data: splitStats } = useQuery({
    queryKey: ["cls-dataset-split-stats", datasetId],
    queryFn: () => apiClient.getCLSDatasetSplitStats(datasetId),
    staleTime: 60000,
  });

  // Fetch available images (for add sheet)
  const { data: availableImagesData, isLoading: availableLoading } = useQuery({
    queryKey: ["cls-images-available", availableImagesPage, availableSearch],
    queryFn: () =>
      apiClient.getCLSImages({
        page: availableImagesPage,
        limit: 50,
        search: availableSearch || undefined,
      }),
    enabled: isAddImagesSheetOpen,
    staleTime: 30000,
  });

  // Add images mutation
  const addImagesMutation = useMutation({
    mutationFn: async (imageIds: string[]) => {
      return apiClient.addImagesToCLSDataset(datasetId, imageIds);
    },
    onSuccess: (data) => {
      toast.success(`Added ${data.added} images to dataset`);
      if (data.skipped > 0) {
        toast.info(`${data.skipped} images were already in the dataset`);
      }
      queryClient.invalidateQueries({ queryKey: ["cls-dataset", datasetId] });
      queryClient.invalidateQueries({ queryKey: ["cls-dataset-images", datasetId] });
      setIsAddImagesSheetOpen(false);
      setSelectedToAdd(new Set());
    },
    onError: (error) => {
      toast.error(`Failed to add images: ${error.message}`);
    },
  });

  // Remove images mutation
  const removeImagesMutation = useMutation({
    mutationFn: async (imageIds: string[]) => {
      return apiClient.removeImagesFromCLSDataset(datasetId, imageIds);
    },
    onSuccess: (data) => {
      toast.success(`Removed ${data.removed} images from dataset`);
      queryClient.invalidateQueries({ queryKey: ["cls-dataset", datasetId] });
      queryClient.invalidateQueries({ queryKey: ["cls-dataset-images", datasetId] });
      setSelectedImages(new Set());
    },
    onError: (error) => {
      toast.error(`Failed to remove images: ${error.message}`);
    },
  });

  // Create class mutation
  const createClassMutation = useMutation({
    mutationFn: async () => {
      return apiClient.createCLSClass({
        name: formName,
        display_name: formDisplayName || undefined,
        color: formColor,
      });
    },
    onSuccess: () => {
      toast.success("Class created successfully");
      queryClient.invalidateQueries({ queryKey: ["cls-dataset-classes", datasetId] });
      queryClient.invalidateQueries({ queryKey: ["cls-classes"] });
      setIsCreateClassDialogOpen(false);
      resetClassForm();
    },
    onError: (error) => {
      toast.error(`Failed to create class: ${error.message}`);
    },
  });

  // Auto-split mutation
  const autoSplitMutation = useMutation({
    mutationFn: async () => {
      return apiClient.autoSplitCLSDataset(datasetId, {
        train_ratio: trainRatio,
        val_ratio: valRatio,
        test_ratio: testRatio,
        stratified: true,
      });
    },
    onSuccess: (data) => {
      toast.success(`Split complete: ${data.train_count} train, ${data.val_count} val, ${data.test_count} test`);
      queryClient.invalidateQueries({ queryKey: ["cls-dataset", datasetId] });
      queryClient.invalidateQueries({ queryKey: ["cls-dataset-images", datasetId] });
      queryClient.invalidateQueries({ queryKey: ["cls-dataset-split-stats", datasetId] });
      setIsAutoSplitDialogOpen(false);
    },
    onError: (error) => {
      toast.error(`Failed to split dataset: ${error.message}`);
    },
  });

  // Create version mutation
  const createVersionMutation = useMutation({
    mutationFn: async () => {
      return apiClient.createCLSDatasetVersion(datasetId, {
        name: versionName || undefined,
        description: versionDescription || undefined,
      });
    },
    onSuccess: () => {
      toast.success("Version created successfully");
      queryClient.invalidateQueries({ queryKey: ["cls-dataset-versions", datasetId] });
      queryClient.invalidateQueries({ queryKey: ["cls-dataset", datasetId] });
      setIsCreateVersionDialogOpen(false);
      setVersionName("");
      setVersionDescription("");
    },
    onError: (error) => {
      toast.error(`Failed to create version: ${error.message}`);
    },
  });

  const resetClassForm = () => {
    setFormName("");
    setFormDisplayName("");
    setFormColor(PRESET_COLORS[0]);
  };

  // Filter classes
  const filteredClasses = useMemo(() => {
    if (!classes) return [];
    return classes.filter((c) =>
      c.name.toLowerCase().includes(classSearch.toLowerCase()) ||
      c.display_name?.toLowerCase().includes(classSearch.toLowerCase())
    );
  }, [classes, classSearch]);

  // Calculate stats
  const labeledCount = dataset?.labeled_image_count ?? 0;
  const totalCount = dataset?.image_count ?? 0;
  const labelingProgress = totalCount > 0 ? Math.round((labeledCount / totalCount) * 100) : 0;

  if (datasetLoading) {
    return (
      <div className="flex items-center justify-center py-24">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (!dataset) {
    return (
      <div className="text-center py-24">
        <FolderOpen className="h-16 w-16 mx-auto text-muted-foreground mb-4" />
        <h3 className="text-lg font-medium">Dataset not found</h3>
        <Button className="mt-4" onClick={() => router.push("/classification/datasets")}>
          <ArrowLeft className="h-4 w-4 mr-2" />
          Back to Datasets
        </Button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-start">
        <div className="flex items-center gap-4">
          <Button variant="ghost" size="icon" onClick={() => router.push("/classification/datasets")}>
            <ArrowLeft className="h-5 w-5" />
          </Button>
          <div>
            <div className="flex items-center gap-3">
              <h1 className="text-2xl font-bold">{dataset.name}</h1>
              <Badge variant="outline">
                {dataset.task_type === "single_label" ? "Single Label" : "Multi Label"}
              </Badge>
              <Badge variant="secondary">v{dataset.version}</Badge>
            </div>
            {dataset.description && (
              <p className="text-muted-foreground mt-1">{dataset.description}</p>
            )}
          </div>
        </div>
        <div className="flex gap-2">
          <Button
            variant="outline"
            onClick={() => queryClient.invalidateQueries({ queryKey: ["cls-dataset", datasetId] })}
            disabled={isFetching}
          >
            <RefreshCw className={`h-4 w-4 mr-2 ${isFetching ? "animate-spin" : ""}`} />
            Refresh
          </Button>
          <Button asChild>
            <Link href={`/classification/labeling?dataset=${datasetId}`}>
              <PenTool className="h-4 w-4 mr-2" />
              Start Labeling
            </Link>
          </Button>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-5 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground flex items-center gap-2">
              <ImageIcon className="h-4 w-4" />
              Images
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">{totalCount.toLocaleString()}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground flex items-center gap-2">
              <CheckCircle className="h-4 w-4" />
              Labeled
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-green-600">{labeledCount.toLocaleString()}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground flex items-center gap-2">
              <Clock className="h-4 w-4" />
              Unlabeled
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-orange-600">{(totalCount - labeledCount).toLocaleString()}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground flex items-center gap-2">
              <Tags className="h-4 w-4" />
              Classes
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">{dataset.class_count}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">Progress</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <p className="text-2xl font-bold">{labelingProgress}%</p>
              <Progress value={labelingProgress} className="h-2" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Split Stats */}
      {splitStats && (
        <Card>
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm flex items-center gap-2">
                <SplitSquareVertical className="h-4 w-4" />
                Data Splits
              </CardTitle>
              <Button variant="outline" size="sm" onClick={() => setIsAutoSplitDialogOpen(true)}>
                <SplitSquareVertical className="h-4 w-4 mr-2" />
                Auto-Split
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-4 gap-4">
              <div className="text-center p-3 bg-muted rounded-lg">
                <p className="text-sm text-muted-foreground">Train</p>
                <p className="text-xl font-bold text-blue-600">{splitStats.train_count ?? 0}</p>
              </div>
              <div className="text-center p-3 bg-muted rounded-lg">
                <p className="text-sm text-muted-foreground">Validation</p>
                <p className="text-xl font-bold text-green-600">{splitStats.val_count ?? 0}</p>
              </div>
              <div className="text-center p-3 bg-muted rounded-lg">
                <p className="text-sm text-muted-foreground">Test</p>
                <p className="text-xl font-bold text-orange-600">{splitStats.test_count ?? 0}</p>
              </div>
              <div className="text-center p-3 bg-muted rounded-lg">
                <p className="text-sm text-muted-foreground">Unassigned</p>
                <p className="text-xl font-bold text-gray-600">{splitStats.unassigned_count ?? 0}</p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Tabs */}
      <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as typeof activeTab)}>
        <TabsList>
          <TabsTrigger value="images" className="flex items-center gap-2">
            <ImageIcon className="h-4 w-4" />
            Images ({totalCount})
          </TabsTrigger>
          <TabsTrigger value="classes" className="flex items-center gap-2">
            <Tags className="h-4 w-4" />
            Classes ({dataset.class_count})
          </TabsTrigger>
          <TabsTrigger value="versions" className="flex items-center gap-2">
            <History className="h-4 w-4" />
            Versions
          </TabsTrigger>
        </TabsList>

        {/* Images Tab */}
        <TabsContent value="images" className="mt-6 space-y-4">
          {/* Toolbar */}
          <div className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-4 flex-1">
              <div className="relative max-w-sm flex-1">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search images..."
                  value={searchInput}
                  onChange={(e) => setSearchInput(e.target.value)}
                  className="pl-10"
                />
              </div>
              <Select value={filterStatus} onValueChange={setFilterStatus}>
                <SelectTrigger className="w-[140px]">
                  <SelectValue placeholder="Status" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Status</SelectItem>
                  <SelectItem value="labeled">Labeled</SelectItem>
                  <SelectItem value="unlabeled">Unlabeled</SelectItem>
                </SelectContent>
              </Select>
              <Select value={filterSplit} onValueChange={setFilterSplit}>
                <SelectTrigger className="w-[140px]">
                  <SelectValue placeholder="Split" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Splits</SelectItem>
                  <SelectItem value="train">Train</SelectItem>
                  <SelectItem value="val">Validation</SelectItem>
                  <SelectItem value="test">Test</SelectItem>
                  <SelectItem value="unassigned">Unassigned</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="flex gap-2">
              {selectedImages.size > 0 && (
                <Button
                  variant="destructive"
                  size="sm"
                  onClick={() => {
                    if (confirm(`Remove ${selectedImages.size} images from dataset?`)) {
                      removeImagesMutation.mutate(Array.from(selectedImages));
                    }
                  }}
                  disabled={removeImagesMutation.isPending}
                >
                  <Trash2 className="h-4 w-4 mr-2" />
                  Remove ({selectedImages.size})
                </Button>
              )}
              <Button onClick={() => setIsAddImagesSheetOpen(true)}>
                <Plus className="h-4 w-4 mr-2" />
                Add Images
              </Button>
            </div>
          </div>

          {/* Images Grid */}
          {imagesLoading ? (
            <div className="flex items-center justify-center py-24">
              <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
            </div>
          ) : !imagesData?.images?.length ? (
            <div className="text-center py-24">
              <ImageIcon className="h-16 w-16 mx-auto text-muted-foreground mb-4" />
              <h3 className="text-lg font-medium">No images in this dataset</h3>
              <p className="text-muted-foreground mt-1">Add images to start labeling</p>
              <Button className="mt-4" onClick={() => setIsAddImagesSheetOpen(true)}>
                <Plus className="h-4 w-4 mr-2" />
                Add Images
              </Button>
            </div>
          ) : (
            <>
              <div className="grid grid-cols-6 gap-3">
                {imagesData.images.map((img: DatasetImage) => (
                  <Card
                    key={img.id}
                    className={`group relative cursor-pointer transition-all ${
                      selectedImages.has(img.image_id) ? "ring-2 ring-primary" : ""
                    }`}
                    onClick={() => {
                      const newSelected = new Set(selectedImages);
                      if (newSelected.has(img.image_id)) {
                        newSelected.delete(img.image_id);
                      } else {
                        newSelected.add(img.image_id);
                      }
                      setSelectedImages(newSelected);
                    }}
                  >
                    <div className="aspect-square relative overflow-hidden rounded-t-lg bg-muted">
                      <Image
                        src={img.image.image_url}
                        alt={img.image.filename}
                        fill
                        className="object-cover"
                        sizes="(max-width: 768px) 50vw, 16vw"
                      />
                      {/* Selection checkbox */}
                      <div className="absolute top-2 left-2">
                        <Checkbox
                          checked={selectedImages.has(img.image_id)}
                          className="bg-background/80"
                          onClick={(e) => e.stopPropagation()}
                        />
                      </div>
                      {/* Label badge */}
                      {img.labels && img.labels.length > 0 && (
                        <div className="absolute bottom-2 left-2 right-2">
                          <Badge
                            className="truncate max-w-full"
                            style={{ backgroundColor: img.labels[0].class_color || "#6b7280" }}
                          >
                            {img.labels[0].class_name || "Labeled"}
                          </Badge>
                        </div>
                      )}
                      {/* Split badge */}
                      {img.split && (
                        <div className="absolute top-2 right-2">
                          <Badge variant="secondary" className="text-xs">
                            {img.split}
                          </Badge>
                        </div>
                      )}
                    </div>
                    <CardContent className="p-2">
                      <p className="text-xs text-muted-foreground truncate">
                        {img.image.filename}
                      </p>
                    </CardContent>
                  </Card>
                ))}
              </div>

              {/* Pagination */}
              {imagesData.total > PAGE_SIZE && (
                <div className="flex items-center justify-center gap-2 mt-6">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setPage((p) => Math.max(1, p - 1))}
                    disabled={page === 1}
                  >
                    Previous
                  </Button>
                  <span className="text-sm text-muted-foreground">
                    Page {page} of {Math.ceil(imagesData.total / PAGE_SIZE)}
                  </span>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setPage((p) => p + 1)}
                    disabled={page >= Math.ceil(imagesData.total / PAGE_SIZE)}
                  >
                    Next
                  </Button>
                </div>
              )}
            </>
          )}
        </TabsContent>

        {/* Classes Tab */}
        <TabsContent value="classes" className="mt-6 space-y-4">
          <div className="flex items-center justify-between">
            <div className="relative max-w-sm">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search classes..."
                value={classSearch}
                onChange={(e) => setClassSearch(e.target.value)}
                className="pl-10"
              />
            </div>
            <Button onClick={() => setIsCreateClassDialogOpen(true)}>
              <Plus className="h-4 w-4 mr-2" />
              Add Class
            </Button>
          </div>

          {filteredClasses.length === 0 ? (
            <div className="text-center py-24">
              <Tags className="h-16 w-16 mx-auto text-muted-foreground mb-4" />
              <h3 className="text-lg font-medium">No classes found</h3>
              <p className="text-muted-foreground mt-1">Create classes to label your images</p>
              <Button className="mt-4" onClick={() => setIsCreateClassDialogOpen(true)}>
                <Plus className="h-4 w-4 mr-2" />
                Create Class
              </Button>
            </div>
          ) : (
            <div className="grid grid-cols-4 gap-4">
              {filteredClasses.map((cls) => (
                <Card key={cls.id}>
                  <CardHeader className="pb-2">
                    <div className="flex items-center gap-3">
                      <div
                        className="w-4 h-4 rounded-full"
                        style={{ backgroundColor: cls.color }}
                      />
                      <div className="flex-1 min-w-0">
                        <CardTitle className="text-sm truncate">
                          {cls.display_name || cls.name}
                        </CardTitle>
                        {cls.display_name && (
                          <CardDescription className="text-xs truncate">
                            {cls.name}
                          </CardDescription>
                        )}
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <p className="text-2xl font-bold">{cls.image_count}</p>
                    <p className="text-xs text-muted-foreground">images labeled</p>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </TabsContent>

        {/* Versions Tab */}
        <TabsContent value="versions" className="mt-6 space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-medium">Dataset Versions</h3>
              <p className="text-sm text-muted-foreground">
                Create snapshots of your dataset for reproducible training
              </p>
            </div>
            <Button onClick={() => setIsCreateVersionDialogOpen(true)}>
              <Plus className="h-4 w-4 mr-2" />
              Create Version
            </Button>
          </div>

          {!versions?.length ? (
            <div className="text-center py-24">
              <History className="h-16 w-16 mx-auto text-muted-foreground mb-4" />
              <h3 className="text-lg font-medium">No versions yet</h3>
              <p className="text-muted-foreground mt-1">
                Create a version to snapshot your dataset state
              </p>
              <Button className="mt-4" onClick={() => setIsCreateVersionDialogOpen(true)}>
                <Plus className="h-4 w-4 mr-2" />
                Create Version
              </Button>
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Version</TableHead>
                  <TableHead>Name</TableHead>
                  <TableHead>Images</TableHead>
                  <TableHead>Classes</TableHead>
                  <TableHead>Created</TableHead>
                  <TableHead>Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {versions.map((version) => (
                  <TableRow key={version.id}>
                    <TableCell>
                      <Badge variant="outline">v{version.version_number}</Badge>
                    </TableCell>
                    <TableCell>{version.name || "-"}</TableCell>
                    <TableCell>{version.image_count}</TableCell>
                    <TableCell>{version.class_count}</TableCell>
                    <TableCell>
                      {new Date(version.created_at).toLocaleDateString()}
                    </TableCell>
                    <TableCell>
                      <Button variant="ghost" size="sm" asChild>
                        <Link href={`/classification/training/new?dataset=${datasetId}&version=${version.id}`}>
                          <Brain className="h-4 w-4 mr-2" />
                          Train
                        </Link>
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </TabsContent>
      </Tabs>

      {/* Add Images Sheet */}
      <Sheet open={isAddImagesSheetOpen} onOpenChange={setIsAddImagesSheetOpen}>
        <SheetContent className="w-[800px] sm:max-w-[800px]">
          <SheetHeader>
            <SheetTitle>Add Images to Dataset</SheetTitle>
            <SheetDescription>
              Select images to add to this dataset
            </SheetDescription>
          </SheetHeader>

          <div className="mt-6 space-y-4">
            <div className="flex items-center gap-4">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search images..."
                  value={availableSearch}
                  onChange={(e) => setAvailableSearch(e.target.value)}
                  className="pl-10"
                />
              </div>
              <Button
                onClick={() => addImagesMutation.mutate(Array.from(selectedToAdd))}
                disabled={selectedToAdd.size === 0 || addImagesMutation.isPending}
              >
                {addImagesMutation.isPending ? (
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                ) : (
                  <Plus className="h-4 w-4 mr-2" />
                )}
                Add {selectedToAdd.size} Images
              </Button>
            </div>

            {availableLoading ? (
              <div className="flex items-center justify-center py-12">
                <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
              </div>
            ) : !availableImagesData?.images?.length ? (
              <div className="text-center py-12">
                <ImageIcon className="h-12 w-12 mx-auto text-muted-foreground mb-2" />
                <p className="text-muted-foreground">No images available</p>
              </div>
            ) : (
              <div className="grid grid-cols-5 gap-2 max-h-[60vh] overflow-y-auto">
                {availableImagesData.images.map((img) => (
                  <Card
                    key={img.id}
                    className={`cursor-pointer transition-all ${
                      selectedToAdd.has(img.id) ? "ring-2 ring-primary" : ""
                    }`}
                    onClick={() => {
                      const newSelected = new Set(selectedToAdd);
                      if (newSelected.has(img.id)) {
                        newSelected.delete(img.id);
                      } else {
                        newSelected.add(img.id);
                      }
                      setSelectedToAdd(newSelected);
                    }}
                  >
                    <div className="aspect-square relative overflow-hidden rounded-t-lg bg-muted">
                      <Image
                        src={img.image_url}
                        alt={img.filename}
                        fill
                        className="object-cover"
                        sizes="150px"
                      />
                      <div className="absolute top-2 left-2">
                        <Checkbox
                          checked={selectedToAdd.has(img.id)}
                          className="bg-background/80"
                        />
                      </div>
                    </div>
                  </Card>
                ))}
              </div>
            )}

            {availableImagesData && availableImagesData.total > 50 && (
              <div className="flex items-center justify-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setAvailableImagesPage((p) => Math.max(1, p - 1))}
                  disabled={availableImagesPage === 1}
                >
                  Previous
                </Button>
                <span className="text-sm text-muted-foreground">
                  Page {availableImagesPage}
                </span>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setAvailableImagesPage((p) => p + 1)}
                  disabled={availableImagesPage >= Math.ceil(availableImagesData.total / 50)}
                >
                  Next
                </Button>
              </div>
            )}
          </div>
        </SheetContent>
      </Sheet>

      {/* Create Class Dialog */}
      <Dialog open={isCreateClassDialogOpen} onOpenChange={setIsCreateClassDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Create New Class</DialogTitle>
            <DialogDescription>
              Add a new classification label
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label>Name *</Label>
              <Input
                placeholder="e.g., laptop"
                value={formName}
                onChange={(e) => setFormName(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label>Display Name</Label>
              <Input
                placeholder="e.g., Laptop Computer"
                value={formDisplayName}
                onChange={(e) => setFormDisplayName(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label>Color</Label>
              <div className="flex gap-2 flex-wrap">
                {PRESET_COLORS.map((color) => (
                  <button
                    key={color}
                    type="button"
                    className={`w-8 h-8 rounded-full border-2 ${
                      formColor === color ? "border-foreground" : "border-transparent"
                    }`}
                    style={{ backgroundColor: color }}
                    onClick={() => setFormColor(color)}
                  />
                ))}
              </div>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setIsCreateClassDialogOpen(false)}>
              Cancel
            </Button>
            <Button
              onClick={() => createClassMutation.mutate()}
              disabled={!formName.trim() || createClassMutation.isPending}
            >
              {createClassMutation.isPending ? (
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              ) : null}
              Create
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Create Version Dialog */}
      <Dialog open={isCreateVersionDialogOpen} onOpenChange={setIsCreateVersionDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Create Dataset Version</DialogTitle>
            <DialogDescription>
              Create a snapshot of the current dataset state
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label>Version Name (optional)</Label>
              <Input
                placeholder="e.g., Initial Release"
                value={versionName}
                onChange={(e) => setVersionName(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label>Description (optional)</Label>
              <Input
                placeholder="e.g., First labeled batch"
                value={versionDescription}
                onChange={(e) => setVersionDescription(e.target.value)}
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setIsCreateVersionDialogOpen(false)}>
              Cancel
            </Button>
            <Button
              onClick={() => createVersionMutation.mutate()}
              disabled={createVersionMutation.isPending}
            >
              {createVersionMutation.isPending ? (
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              ) : null}
              Create Version
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Auto-Split Dialog */}
      <Dialog open={isAutoSplitDialogOpen} onOpenChange={setIsAutoSplitDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Auto-Split Dataset</DialogTitle>
            <DialogDescription>
              Automatically split images into train/validation/test sets
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label>Train Ratio: {(trainRatio * 100).toFixed(0)}%</Label>
              <Input
                type="range"
                min="0.5"
                max="0.9"
                step="0.05"
                value={trainRatio}
                onChange={(e) => {
                  const newTrain = parseFloat(e.target.value);
                  setTrainRatio(newTrain);
                  const remaining = 1 - newTrain;
                  setValRatio(remaining * 0.5);
                  setTestRatio(remaining * 0.5);
                }}
              />
            </div>
            <div className="space-y-2">
              <Label>Validation Ratio: {(valRatio * 100).toFixed(0)}%</Label>
              <Input
                type="range"
                min="0.05"
                max="0.3"
                step="0.05"
                value={valRatio}
                onChange={(e) => {
                  const newVal = parseFloat(e.target.value);
                  setValRatio(newVal);
                  setTestRatio(1 - trainRatio - newVal);
                }}
              />
            </div>
            <div className="space-y-2">
              <Label>Test Ratio: {(testRatio * 100).toFixed(0)}%</Label>
              <Input
                type="range"
                min="0.05"
                max="0.2"
                step="0.05"
                value={testRatio}
                onChange={(e) => {
                  const newTest = parseFloat(e.target.value);
                  setTestRatio(newTest);
                  setValRatio(1 - trainRatio - newTest);
                }}
              />
            </div>
            <div className="p-3 bg-muted rounded-lg">
              <p className="text-sm">
                This will stratify the split by class labels to maintain class distribution.
              </p>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setIsAutoSplitDialogOpen(false)}>
              Cancel
            </Button>
            <Button
              onClick={() => autoSplitMutation.mutate()}
              disabled={autoSplitMutation.isPending}
            >
              {autoSplitMutation.isPending ? (
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              ) : null}
              Apply Split
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
