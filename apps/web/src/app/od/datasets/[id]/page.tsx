"use client";

import { useState, use, useMemo } from "react";
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
  FilterDrawer,
  FilterTrigger,
  useFilterState,
  type FilterSection,
} from "@/components/filters/filter-drawer";
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
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
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
  Package,
  FolderOpen,
  Tags,
  Edit,
  Merge,
  Lock,
  Palette,
  AlertTriangle,
  ArrowRight,
  Copy,
  ArrowUpDown,
  ArrowUp,
  ArrowDown,
  X,
  CheckSquare,
  Sparkles,
} from "lucide-react";
import Image from "next/image";
import Link from "next/link";
import { ImportModal } from "@/components/od/import-modal";
import { BulkAnnotateModal } from "@/components/od/bulk-annotate-modal";

// Predefined colors for class labels
const PRESET_COLORS = [
  "#ef4444", // red
  "#f97316", // orange
  "#eab308", // yellow
  "#22c55e", // green
  "#14b8a6", // teal
  "#3b82f6", // blue
  "#8b5cf6", // violet
  "#ec4899", // pink
  "#6366f1", // indigo
  "#06b6d4", // cyan
];

interface ODClass {
  id: string;
  name: string;
  display_name?: string;
  color: string;
  category?: string;
  annotation_count: number;
  is_system: boolean;
  dataset_id?: string;
}

const PAGE_SIZE = 48;

interface DatasetImage {
  id: string;
  image_id: string;
  status: string;
  annotation_count: number;
  image: {
    id: string;
    filename: string;
    image_url: string;
    width: number;
    height: number;
  };
}

export default function ODDatasetDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id: datasetId } = use(params);
  const router = useRouter();
  const queryClient = useQueryClient();

  // Tab state
  const [activeTab, setActiveTab] = useState<"images" | "classes">("images");

  // Images tab state
  const [page, setPage] = useState(1);
  const [search, setSearch] = useState("");
  const [filterStatus, setFilterStatus] = useState<string>("all");
  const [selectedImages, setSelectedImages] = useState<Set<string>>(new Set());

  // Add images sheet state
  const [isAddImagesSheetOpen, setIsAddImagesSheetOpen] = useState(false);
  const [availableImagesPage, setAvailableImagesPage] = useState(1);
  const [availableSearch, setAvailableSearch] = useState("");
  const [addImagesFilterDrawerOpen, setAddImagesFilterDrawerOpen] = useState(false);
  const [selectAllFilteredAddMode, setSelectAllFilteredAddMode] = useState(false);

  // Filter state for Add Images sheet
  const {
    filterState: addFilterState,
    setFilter: setAddFilter,
    clearSection: clearAddSection,
    clearAll: clearAddFilters,
    getTotalCount: getAddFilterCount,
  } = useFilterState();
  const [selectedToAdd, setSelectedToAdd] = useState<Set<string>>(new Set());

  // Advanced import modal (with dataset context)
  const [isImportModalOpen, setIsImportModalOpen] = useState(false);

  // Bulk AI annotation modal
  const [isBulkAnnotateOpen, setIsBulkAnnotateOpen] = useState(false);

  // Classes tab state
  const [classSearch, setClassSearch] = useState("");
  const [filterCategory, setFilterCategory] = useState<string>("all");
  const [isCreateClassDialogOpen, setIsCreateClassDialogOpen] = useState(false);
  const [isEditClassDialogOpen, setIsEditClassDialogOpen] = useState(false);
  const [isMergeDialogOpen, setIsMergeDialogOpen] = useState(false);
  const [editingClass, setEditingClass] = useState<ODClass | null>(null);
  const [formName, setFormName] = useState("");
  const [formDisplayName, setFormDisplayName] = useState("");
  const [formColor, setFormColor] = useState(PRESET_COLORS[0]);
  const [formCategory, setFormCategory] = useState("");
  const [mergeSourceIds, setMergeSourceIds] = useState<string[]>([]);
  const [mergeTargetId, setMergeTargetId] = useState("");
  const [selectedClassIds, setSelectedClassIds] = useState<Set<string>>(new Set());
  const [classSortBy, setClassSortBy] = useState<"name" | "annotation_count" | "category">("name");
  const [classSortOrder, setClassSortOrder] = useState<"asc" | "desc">("asc");

  // Fetch dataset info
  const { data: dataset, isLoading: datasetLoading } = useQuery({
    queryKey: ["od-dataset", datasetId],
    queryFn: () => apiClient.getODDataset(datasetId),
  });

  // Fetch dataset images
  const {
    data: imagesData,
    isLoading: imagesLoading,
    isFetching,
  } = useQuery({
    queryKey: ["od-dataset-images", datasetId, page, filterStatus],
    queryFn: () =>
      apiClient.getODDatasetImages(datasetId, {
        page,
        limit: PAGE_SIZE,
        status: filterStatus !== "all" ? filterStatus : undefined,
      }),
  });

  // Fetch filter options for Add Images sheet
  const { data: addImagesFilterOptions } = useQuery({
    queryKey: ["od-image-filter-options"],
    queryFn: () => apiClient.getODImageFilterOptions(),
    enabled: isAddImagesSheetOpen,
  });

  // Build filter sections for Add Images sheet
  const addImagesFilterSections: FilterSection[] = useMemo(() => {
    if (!addImagesFilterOptions) return [];

    const sections: FilterSection[] = [];

    if (addImagesFilterOptions.status?.length > 0) {
      sections.push({
        id: "status",
        label: "Status",
        type: "checkbox",
        options: addImagesFilterOptions.status,
        defaultExpanded: true,
      });
    }

    if (addImagesFilterOptions.source?.length > 0) {
      sections.push({
        id: "source",
        label: "Source",
        type: "checkbox",
        options: addImagesFilterOptions.source,
        defaultExpanded: true,
      });
    }

    if (addImagesFilterOptions.merchant?.length > 0) {
      sections.push({
        id: "merchant",
        label: "Merchant",
        type: "checkbox",
        options: addImagesFilterOptions.merchant,
        searchable: true,
        defaultExpanded: false,
      });
    }

    if (addImagesFilterOptions.store?.length > 0) {
      sections.push({
        id: "store",
        label: "Store",
        type: "checkbox",
        options: addImagesFilterOptions.store,
        searchable: true,
        defaultExpanded: false,
      });
    }

    return sections;
  }, [addImagesFilterOptions]);

  // Convert filter state to API params for Add Images
  const addApiFilters = useMemo(() => {
    const params: Record<string, string | undefined> = {};

    const setToString = (key: string): string | undefined => {
      const filter = addFilterState[key] as Set<string> | undefined;
      if (filter?.size) {
        return Array.from(filter).join(",");
      }
      return undefined;
    };

    params.statuses = setToString("status");
    params.sources = setToString("source");
    params.merchant_ids = setToString("merchant");
    params.store_ids = setToString("store");

    return params;
  }, [addFilterState]);

  const addActiveFilterCount = getAddFilterCount();

  // Fetch available images (for add sheet)
  const { data: availableImagesData, isLoading: availableLoading } = useQuery({
    queryKey: ["od-images-available", availableImagesPage, availableSearch, addApiFilters],
    queryFn: () =>
      apiClient.getODImages({
        page: availableImagesPage,
        limit: 50,
        search: availableSearch || undefined,
        ...addApiFilters,
      }),
    enabled: isAddImagesSheetOpen,
  });

  // Add images mutation (by IDs)
  const addImagesMutation = useMutation({
    mutationFn: async (imageIds: string[]) => {
      return apiClient.addImagesToODDataset(datasetId, imageIds);
    },
    onSuccess: (data) => {
      toast.success(`Added ${data.added} images to dataset`);
      if (data.skipped > 0) {
        toast.info(`${data.skipped} images were already in the dataset`);
      }
      queryClient.invalidateQueries({ queryKey: ["od-dataset", datasetId] });
      queryClient.invalidateQueries({ queryKey: ["od-dataset-images", datasetId] });
      setIsAddImagesSheetOpen(false);
      setSelectedToAdd(new Set());
      setSelectAllFilteredAddMode(false);
    },
    onError: (error) => {
      toast.error(`Failed to add images: ${error.message}`);
    },
  });

  // Add images by filters mutation (for selectAllFilteredAddMode)
  const addImagesByFiltersMutation = useMutation({
    mutationFn: async () => {
      return apiClient.addFilteredImagesToODDataset(datasetId, {
        search: availableSearch || undefined,
        ...addApiFilters,
      });
    },
    onSuccess: (data) => {
      toast.success(`Added ${data.added} images to dataset`);
      if (data.skipped > 0) {
        toast.info(`${data.skipped} images were already in the dataset`);
      }
      queryClient.invalidateQueries({ queryKey: ["od-dataset", datasetId] });
      queryClient.invalidateQueries({ queryKey: ["od-dataset-images", datasetId] });
      setIsAddImagesSheetOpen(false);
      setSelectedToAdd(new Set());
      setSelectAllFilteredAddMode(false);
    },
    onError: (error) => {
      toast.error(`Failed to add images: ${error.message}`);
    },
  });

  // Remove image mutation
  const removeImageMutation = useMutation({
    mutationFn: async (imageId: string) => {
      return apiClient.removeImageFromODDataset(datasetId, imageId);
    },
    onSuccess: () => {
      toast.success("Image removed from dataset");
      queryClient.invalidateQueries({ queryKey: ["od-dataset", datasetId] });
      queryClient.invalidateQueries({ queryKey: ["od-dataset-images", datasetId] });
      setSelectedImages(new Set());
    },
    onError: (error) => {
      toast.error(`Failed to remove image: ${error.message}`);
    },
  });

  // Bulk remove mutation
  const bulkRemoveMutation = useMutation({
    mutationFn: async (imageIds: string[]) => {
      return apiClient.removeImagesFromODDatasetBulk(datasetId, imageIds);
    },
    onSuccess: (data) => {
      toast.success(`Removed ${data.removed} images from dataset`);
      queryClient.invalidateQueries({ queryKey: ["od-dataset", datasetId] });
      queryClient.invalidateQueries({ queryKey: ["od-dataset-images", datasetId] });
      setSelectedImages(new Set());
    },
    onError: (error) => {
      toast.error(`Failed to remove images: ${error.message}`);
    },
  });

  // Update status mutation
  const updateStatusMutation = useMutation({
    mutationFn: async ({ imageId, status }: { imageId: string; status: string }) => {
      return apiClient.updateODDatasetImageStatus(datasetId, imageId, status);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["od-dataset-images", datasetId] });
    },
    onError: (error) => {
      toast.error(`Failed to update status: ${error.message}`);
    },
  });

  // ============ Classes Tab Queries & Mutations ============

  // Fetch classes for this dataset
  const { data: classes, isLoading: classesLoading, isFetching: classesFetching } = useQuery({
    queryKey: ["od-classes", datasetId, filterCategory],
    queryFn: () =>
      apiClient.getODClasses({
        dataset_id: datasetId,
        category: filterCategory !== "all" ? filterCategory : undefined,
        is_active: true,
      }),
    enabled: activeTab === "classes",
  });

  // Fetch potential duplicates
  const { data: duplicates } = useQuery({
    queryKey: ["od-class-duplicates", datasetId],
    queryFn: () => apiClient.getODClassDuplicates(datasetId),
    enabled: activeTab === "classes" && (classes?.length ?? 0) > 1,
    staleTime: 30000,
  });

  // Create class mutation
  const createClassMutation = useMutation({
    mutationFn: async () => {
      return apiClient.createODDatasetClass(datasetId, {
        name: formName,
        display_name: formDisplayName || undefined,
        color: formColor,
        category: formCategory || undefined,
      });
    },
    onSuccess: () => {
      toast.success(`Class "${formName}" created`);
      queryClient.invalidateQueries({ queryKey: ["od-classes", datasetId] });
      setIsCreateClassDialogOpen(false);
      resetClassForm();
    },
    onError: (error) => {
      toast.error(`Failed to create class: ${error.message}`);
    },
  });

  // Update class mutation
  const updateClassMutation = useMutation({
    mutationFn: async () => {
      if (!editingClass) return;
      return apiClient.updateODClass(editingClass.id, {
        name: formName,
        display_name: formDisplayName || undefined,
        color: formColor,
        category: formCategory || undefined,
      });
    },
    onSuccess: () => {
      toast.success("Class updated");
      queryClient.invalidateQueries({ queryKey: ["od-classes", datasetId] });
      setIsEditClassDialogOpen(false);
      setEditingClass(null);
      resetClassForm();
    },
    onError: (error) => {
      toast.error(`Failed to update class: ${error.message}`);
    },
  });

  // Delete class mutation
  const deleteClassMutation = useMutation({
    mutationFn: async (id: string) => {
      return apiClient.deleteODClass(id, true);
    },
    onSuccess: () => {
      toast.success("Class deleted");
      queryClient.invalidateQueries({ queryKey: ["od-classes", datasetId] });
    },
    onError: (error) => {
      toast.error(`Failed to delete class: ${error.message}`);
    },
  });

  // Merge classes mutation
  const mergeClassesMutation = useMutation({
    mutationFn: async () => {
      return apiClient.mergeODClasses(mergeSourceIds, mergeTargetId);
    },
    onSuccess: (data) => {
      toast.success(`Merged ${data.merged_count} classes, moved ${data.annotations_moved} annotations`);
      queryClient.invalidateQueries({ queryKey: ["od-classes", datasetId] });
      setIsMergeDialogOpen(false);
      setMergeSourceIds([]);
      setMergeTargetId("");
      setSelectedClassIds(new Set());
    },
    onError: (error) => {
      toast.error(`Failed to merge classes: ${error.message}`);
    },
  });

  // Class helper functions
  const resetClassForm = () => {
    setFormName("");
    setFormDisplayName("");
    setFormColor(PRESET_COLORS[Math.floor(Math.random() * PRESET_COLORS.length)]);
    setFormCategory("");
  };

  const openEditClassDialog = (cls: ODClass) => {
    setEditingClass(cls);
    setFormName(cls.name);
    setFormDisplayName(cls.display_name || "");
    setFormColor(cls.color);
    setFormCategory(cls.category || "");
    setIsEditClassDialogOpen(true);
  };

  const handleDeleteClass = (cls: ODClass) => {
    const warning = cls.is_system ? " (System Class)" : "";
    if (cls.annotation_count > 0) {
      if (!confirm(`This class${warning} has ${cls.annotation_count} annotations. Delete anyway?`)) {
        return;
      }
    } else {
      if (!confirm(`Delete class "${cls.name}"${warning}?`)) {
        return;
      }
    }
    deleteClassMutation.mutate(cls.id);
  };

  // Filter and sort classes
  const filteredClasses = useMemo(() => {
    let result = classes?.filter((cls) =>
      cls.name.toLowerCase().includes(classSearch.toLowerCase()) ||
      cls.display_name?.toLowerCase().includes(classSearch.toLowerCase())
    ) || [];

    result = [...result].sort((a, b) => {
      let comparison = 0;
      switch (classSortBy) {
        case "name":
          comparison = a.name.localeCompare(b.name);
          break;
        case "annotation_count":
          comparison = a.annotation_count - b.annotation_count;
          break;
        case "category":
          comparison = (a.category || "").localeCompare(b.category || "");
          break;
      }
      return classSortOrder === "asc" ? comparison : -comparison;
    });

    return result;
  }, [classes, classSearch, classSortBy, classSortOrder]);

  // Class selection helpers
  const selectableClasses = filteredClasses || [];
  const allClassesSelected = selectableClasses.length > 0 &&
    selectableClasses.every(c => selectedClassIds.has(c.id));
  const someClassesSelected = selectedClassIds.size > 0;

  const toggleSelectAllClasses = () => {
    if (allClassesSelected) {
      setSelectedClassIds(new Set());
    } else {
      setSelectedClassIds(new Set(selectableClasses.map(c => c.id)));
    }
  };

  const toggleSelectClass = (id: string) => {
    const newSet = new Set(selectedClassIds);
    if (newSet.has(id)) {
      newSet.delete(id);
    } else {
      newSet.add(id);
    }
    setSelectedClassIds(newSet);
  };

  const handleClassSort = (column: "name" | "annotation_count" | "category") => {
    if (classSortBy === column) {
      setClassSortOrder(classSortOrder === "asc" ? "desc" : "asc");
    } else {
      setClassSortBy(column);
      setClassSortOrder("asc");
    }
  };

  const handleBulkDeleteClasses = () => {
    const count = selectedClassIds.size;
    const totalAnnotations = Array.from(selectedClassIds).reduce((sum, id) => {
      const cls = classes?.find(c => c.id === id);
      return sum + (cls?.annotation_count || 0);
    }, 0);

    const message = totalAnnotations > 0
      ? `Delete ${count} class(es) with ${totalAnnotations} total annotations?`
      : `Delete ${count} class(es)?`;

    if (confirm(message)) {
      Array.from(selectedClassIds).forEach(id => {
        deleteClassMutation.mutate(id);
      });
      setSelectedClassIds(new Set());
    }
  };

  const handleBulkMergeClasses = () => {
    setMergeSourceIds(Array.from(selectedClassIds));
    setMergeTargetId("");
    setIsMergeDialogOpen(true);
  };

  // Merge preview calculation
  const mergePreview = useMemo(() => {
    if (!classes || mergeSourceIds.length === 0) return null;

    const sourceAnnotations = mergeSourceIds.reduce((sum, id) => {
      const cls = classes.find(c => c.id === id);
      return sum + (cls?.annotation_count || 0);
    }, 0);

    const targetClass = classes.find(c => c.id === mergeTargetId);
    const targetAnnotations = targetClass?.annotation_count || 0;

    return {
      sourceAnnotations,
      targetAnnotations,
      totalAfterMerge: sourceAnnotations + targetAnnotations,
      targetName: targetClass?.name || "target",
    };
  }, [classes, mergeSourceIds, mergeTargetId]);

  const handleDuplicateQuickMerge = (group: NonNullable<typeof duplicates>["groups"][0]) => {
    setMergeSourceIds(group.suggested_sources);
    setMergeTargetId(group.suggested_target);
    setIsMergeDialogOpen(true);
  };

  // Get unique categories
  const categories = [...new Set(classes?.map((c) => c.category).filter(Boolean) || [])];

  // Calculate class stats
  const totalAnnotations = classes?.reduce((sum, c) => sum + c.annotation_count, 0) || 0;
  const systemClasses = classes?.filter((c) => c.is_system).length || 0;
  const customClasses = classes?.filter((c) => !c.is_system).length || 0;

  // ============ Images Tab Functions ============

  const toggleImageSelection = (imageId: string) => {
    setSelectedImages((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(imageId)) {
        newSet.delete(imageId);
      } else {
        newSet.add(imageId);
      }
      return newSet;
    });
  };

  const toggleSelectAll = () => {
    if (!imagesData?.images) return;
    if (selectedImages.size === imagesData.images.length) {
      setSelectedImages(new Set());
    } else {
      setSelectedImages(new Set(imagesData.images.map((img) => img.image_id)));
    }
  };

  const handleBulkRemove = () => {
    if (selectedImages.size === 0) return;
    if (!confirm(`Remove ${selectedImages.size} images from this dataset?`)) return;
    bulkRemoveMutation.mutate(Array.from(selectedImages));
  };

  const toggleAddSelection = (imageId: string) => {
    setSelectedToAdd((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(imageId)) {
        newSet.delete(imageId);
      } else {
        newSet.add(imageId);
      }
      return newSet;
    });
  };

  const totalPages = imagesData ? Math.ceil(imagesData.total / PAGE_SIZE) : 0;
  const availableTotalPages = availableImagesData ? Math.ceil(availableImagesData.total / 50) : 0;

  // Get IDs of images already in dataset (to show "already added" state)
  const imagesInDataset = new Set(imagesData?.images.map((img) => img.image_id) ?? []);

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "completed":
        return (
          <Badge variant="default" className="bg-green-600">
            <CheckCircle className="h-3 w-3 mr-1" />
            Completed
          </Badge>
        );
      case "annotating":
        return (
          <Badge variant="default" className="bg-blue-600">
            <Clock className="h-3 w-3 mr-1" />
            Annotating
          </Badge>
        );
      case "skipped":
        return (
          <Badge variant="secondary">
            <XCircle className="h-3 w-3 mr-1" />
            Skipped
          </Badge>
        );
      default:
        return (
          <Badge variant="outline">
            <AlertCircle className="h-3 w-3 mr-1" />
            Pending
          </Badge>
        );
    }
  };

  const getAnnotationProgress = () => {
    if (!dataset || dataset.image_count === 0) return 0;
    return Math.round(((dataset as any).annotated_image_count / dataset.image_count) * 100);
  };

  // Filter images by search (client-side)
  const filteredImages = imagesData?.images.filter((img) =>
    img.image.filename.toLowerCase().includes(search.toLowerCase())
  );

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
        <h2 className="text-xl font-bold">Dataset not found</h2>
        <Button className="mt-4" onClick={() => router.push("/od/datasets")}>
          <ArrowLeft className="h-4 w-4 mr-2" />
          Back to Datasets
        </Button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Breadcrumb & Header */}
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <Link href="/od/datasets" className="hover:text-foreground">
          Datasets
        </Link>
        <ChevronRight className="h-4 w-4" />
        <span className="text-foreground">{dataset.name}</span>
      </div>

      <div className="flex justify-between items-start">
        <div>
          <h1 className="text-2xl font-bold">{dataset.name}</h1>
          {(dataset as any).description && (
            <p className="text-muted-foreground mt-1">{(dataset as any).description}</p>
          )}
        </div>
        <div className="flex gap-2">
          <Button
            variant="outline"
            onClick={() => {
              queryClient.invalidateQueries({ queryKey: ["od-dataset", datasetId] });
              queryClient.invalidateQueries({ queryKey: ["od-dataset-images", datasetId] });
            }}
            disabled={isFetching}
          >
            <RefreshCw className={`h-4 w-4 mr-2 ${isFetching ? "animate-spin" : ""}`} />
            Refresh
          </Button>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline">
                <Plus className="h-4 w-4 mr-2" />
                Add Images
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onClick={() => setIsAddImagesSheetOpen(true)}>
                <FolderOpen className="h-4 w-4 mr-2" />
                Select from Library
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => setIsImportModalOpen(true)}>
                <Package className="h-4 w-4 mr-2" />
                Import New Images
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
          <Button
            variant="outline"
            onClick={() => {
              if (dataset.image_count === 0) {
                toast.error("Add images to the dataset first");
                return;
              }
              setIsBulkAnnotateOpen(true);
            }}
            disabled={dataset.image_count === 0}
          >
            <Sparkles className="h-4 w-4 mr-2" />
            AI Annotate
          </Button>
          <Button
            onClick={() => {
              if (dataset.image_count === 0) {
                toast.error("Add images to the dataset first");
                return;
              }
              router.push(`/od/annotate?dataset=${datasetId}`);
            }}
            disabled={dataset.image_count === 0}
          >
            <PenTool className="h-4 w-4 mr-2" />
            Start Annotating
          </Button>
        </div>
      </div>

      {/* Tabs */}
      <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as "images" | "classes")}>
        <TabsList>
          <TabsTrigger value="images" className="gap-2">
            <ImageIcon className="h-4 w-4" />
            Images
            <Badge variant="secondary" className="ml-1">{dataset.image_count}</Badge>
          </TabsTrigger>
          <TabsTrigger value="classes" className="gap-2">
            <Tags className="h-4 w-4" />
            Classes
            {classes && <Badge variant="secondary" className="ml-1">{classes.length}</Badge>}
          </TabsTrigger>
        </TabsList>

        {/* Images Tab */}
        <TabsContent value="images" className="space-y-6 mt-6">
          {/* Stats Cards */}
          <div className="grid grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">Total Images</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">{dataset.image_count}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">Annotated</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-green-600">
              {(dataset as any).annotated_image_count ?? 0}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">Annotations</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-blue-600">{dataset.annotation_count}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">Progress</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">{getAnnotationProgress()}%</p>
            <Progress value={getAnnotationProgress()} className="h-2 mt-2" />
          </CardContent>
        </Card>
      </div>

      {/* Images Section */}
      <Card>
        <CardHeader>
          <div className="flex justify-between items-center">
            <div>
              <CardTitle>Dataset Images</CardTitle>
              <CardDescription>
                {imagesData?.total ?? 0} images in this dataset
              </CardDescription>
            </div>
            {selectedImages.size > 0 && (
              <div className="flex items-center gap-2">
                <span className="text-sm text-muted-foreground">
                  {selectedImages.size} selected
                </span>
                <Button
                  variant="destructive"
                  size="sm"
                  onClick={handleBulkRemove}
                  disabled={bulkRemoveMutation.isPending}
                >
                  <Trash2 className="h-4 w-4 mr-1" />
                  Remove
                </Button>
              </div>
            )}
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Filters */}
          <div className="flex gap-4">
            <div className="flex-1">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search by filename..."
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  className="pl-10"
                />
              </div>
            </div>
            <Select
              value={filterStatus}
              onValueChange={(v) => {
                setFilterStatus(v);
                setPage(1);
              }}
            >
              <SelectTrigger className="w-[150px]">
                <SelectValue placeholder="Status" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Status</SelectItem>
                <SelectItem value="pending">Pending</SelectItem>
                <SelectItem value="annotating">Annotating</SelectItem>
                <SelectItem value="completed">Completed</SelectItem>
                <SelectItem value="skipped">Skipped</SelectItem>
              </SelectContent>
            </Select>
            {imagesData?.images && imagesData.images.length > 0 && (
              <div className="flex items-center gap-2">
                <Checkbox
                  checked={selectedImages.size === imagesData.images.length && imagesData.images.length > 0}
                  onCheckedChange={toggleSelectAll}
                />
                <span className="text-sm text-muted-foreground">Select all</span>
              </div>
            )}
          </div>

          {/* Image Grid */}
          {imagesLoading ? (
            <div className="flex items-center justify-center py-24">
              <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
            </div>
          ) : filteredImages?.length === 0 ? (
            <div className="text-center py-24">
              <ImageIcon className="h-16 w-16 mx-auto text-muted-foreground mb-4" />
              <h3 className="text-lg font-medium">No images in this dataset</h3>
              <p className="text-muted-foreground mt-1">
                Add images from your library or import new ones with annotations
              </p>
              <div className="flex gap-2 justify-center mt-4">
                <Button variant="outline" onClick={() => setIsAddImagesSheetOpen(true)}>
                  <FolderOpen className="h-4 w-4 mr-2" />
                  Select from Library
                </Button>
                <Button onClick={() => setIsImportModalOpen(true)}>
                  <Package className="h-4 w-4 mr-2" />
                  Import Images
                </Button>
              </div>
            </div>
          ) : (
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-4">
              {filteredImages?.map((item: DatasetImage) => (
                <div
                  key={item.id}
                  className={`group relative aspect-square rounded-lg overflow-hidden bg-muted border-2 transition-colors cursor-pointer ${
                    selectedImages.has(item.image_id) ? "border-primary" : "border-transparent hover:border-muted-foreground/30"
                  }`}
                  onClick={() => toggleImageSelection(item.image_id)}
                >
                  <Image
                    src={item.image.image_url}
                    alt={item.image.filename}
                    fill
                    className="object-cover"
                    unoptimized
                  />
                  {/* Selection checkbox */}
                  <div className={`absolute top-2 left-2 transition-opacity ${selectedImages.has(item.image_id) ? "opacity-100" : "opacity-0 group-hover:opacity-100"}`}>
                    <Checkbox
                      checked={selectedImages.has(item.image_id)}
                      onCheckedChange={() => toggleImageSelection(item.image_id)}
                      onClick={(e) => e.stopPropagation()}
                      className="bg-white/80"
                    />
                  </div>
                  {/* Status badge */}
                  <div className="absolute top-2 right-2">
                    {getStatusBadge(item.status)}
                  </div>
                  {/* Annotation count badge */}
                  {item.annotation_count > 0 && (
                    <div className="absolute bottom-2 left-2">
                      <Badge variant="secondary" className="bg-black/60 text-white">
                        <Layers className="h-3 w-3 mr-1" />
                        {item.annotation_count}
                      </Badge>
                    </div>
                  )}
                  {/* Action menu */}
                  <div className="absolute bottom-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild onClick={(e) => e.stopPropagation()}>
                        <Button variant="secondary" size="icon" className="h-7 w-7">
                          <MoreHorizontal className="h-4 w-4" />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end">
                        <DropdownMenuItem
                          onClick={(e) => {
                            e.stopPropagation();
                            router.push(`/od/annotate/${datasetId}/${item.image_id}`);
                          }}
                        >
                          <PenTool className="h-4 w-4 mr-2" />
                          Annotate
                        </DropdownMenuItem>
                        <DropdownMenuSeparator />
                        <DropdownMenuItem
                          onClick={(e) => {
                            e.stopPropagation();
                            updateStatusMutation.mutate({ imageId: item.image_id, status: "completed" });
                          }}
                        >
                          <CheckCircle className="h-4 w-4 mr-2" />
                          Mark Completed
                        </DropdownMenuItem>
                        <DropdownMenuItem
                          onClick={(e) => {
                            e.stopPropagation();
                            updateStatusMutation.mutate({ imageId: item.image_id, status: "skipped" });
                          }}
                        >
                          <XCircle className="h-4 w-4 mr-2" />
                          Skip
                        </DropdownMenuItem>
                        <DropdownMenuSeparator />
                        <DropdownMenuItem
                          className="text-destructive"
                          onClick={(e) => {
                            e.stopPropagation();
                            if (confirm("Remove this image from the dataset?")) {
                              removeImageMutation.mutate(item.image_id);
                            }
                          }}
                        >
                          <Trash2 className="h-4 w-4 mr-2" />
                          Remove from Dataset
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="flex items-center justify-between pt-4">
              <p className="text-sm text-muted-foreground">
                Page {page} of {totalPages} ({imagesData?.total} total)
              </p>
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setPage((p) => Math.max(1, p - 1))}
                  disabled={page === 1}
                >
                  Previous
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                  disabled={page === totalPages}
                >
                  Next
                </Button>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
        </TabsContent>

        {/* Classes Tab */}
        <TabsContent value="classes" className="space-y-6 mt-6">
          {/* Duplicate Detection Banner */}
          {duplicates && duplicates.total_groups > 0 && (
            <Alert className="border-amber-200 bg-amber-50">
              <AlertTriangle className="h-4 w-4 text-amber-600" />
              <AlertTitle className="text-amber-800">
                Potential Duplicate Classes Detected
              </AlertTitle>
              <AlertDescription className="text-amber-700">
                <p className="mb-3">
                  Found {duplicates.total_groups} group(s) of similar class names.
                </p>
                <div className="space-y-2">
                  {duplicates.groups.slice(0, 3).map((group, idx) => (
                    <div
                      key={idx}
                      className="flex items-center justify-between bg-white/50 rounded-md p-2"
                    >
                      <div className="flex items-center gap-2 flex-wrap">
                        {group.classes.map((cls, i) => (
                          <span key={cls.id} className="flex items-center gap-1">
                            <div
                              className="w-3 h-3 rounded"
                              style={{ backgroundColor: cls.color }}
                            />
                            <span className="font-medium">{cls.name}</span>
                            <span className="text-xs text-amber-600">
                              ({cls.annotation_count})
                            </span>
                            {i < group.classes.length - 1 && (
                              <Copy className="h-3 w-3 mx-1 text-amber-400" />
                            )}
                          </span>
                        ))}
                        <span className="text-xs text-amber-500 ml-2">
                          {Math.round(group.max_similarity * 100)}% similar
                        </span>
                      </div>
                      <Button
                        size="sm"
                        variant="outline"
                        className="border-amber-300 text-amber-700 hover:bg-amber-100"
                        onClick={() => handleDuplicateQuickMerge(group)}
                      >
                        <Merge className="h-3 w-3 mr-1" />
                        Quick Merge
                      </Button>
                    </div>
                  ))}
                  {duplicates.total_groups > 3 && (
                    <p className="text-xs text-amber-600">
                      ...and {duplicates.total_groups - 3} more group(s)
                    </p>
                  )}
                </div>
              </AlertDescription>
            </Alert>
          )}

          {/* Class Stats */}
          <div className="grid grid-cols-4 gap-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm text-muted-foreground">Total Classes</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-2xl font-bold">{classes?.length ?? "-"}</p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm text-muted-foreground">System Classes</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-2xl font-bold text-blue-600">{systemClasses}</p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm text-muted-foreground">Custom Classes</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-2xl font-bold text-green-600">{customClasses}</p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm text-muted-foreground">Total Annotations</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-2xl font-bold">{totalAnnotations.toLocaleString()}</p>
              </CardContent>
            </Card>
          </div>

          {/* Classes Table */}
          <Card>
            <CardHeader>
              <div className="flex justify-between items-center">
                <div>
                  <CardTitle>Classes</CardTitle>
                  <CardDescription>
                    {classes?.length ?? 0} detection classes in this dataset
                  </CardDescription>
                </div>
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => queryClient.invalidateQueries({ queryKey: ["od-classes", datasetId] })}
                    disabled={classesFetching}
                  >
                    <RefreshCw className={`h-4 w-4 mr-2 ${classesFetching ? "animate-spin" : ""}`} />
                    Refresh
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setIsMergeDialogOpen(true)}
                    disabled={!classes || classes.length < 2}
                  >
                    <Merge className="h-4 w-4 mr-2" />
                    Merge
                  </Button>
                  <Button size="sm" onClick={() => { resetClassForm(); setIsCreateClassDialogOpen(true); }}>
                    <Plus className="h-4 w-4 mr-2" />
                    New Class
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Filters */}
              <div className="flex gap-4">
                <div className="flex-1">
                  <div className="relative">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                    <Input
                      placeholder="Search classes..."
                      value={classSearch}
                      onChange={(e) => setClassSearch(e.target.value)}
                      className="pl-10"
                    />
                  </div>
                </div>
                <Select value={filterCategory} onValueChange={setFilterCategory}>
                  <SelectTrigger className="w-[180px]">
                    <SelectValue placeholder="Category" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Categories</SelectItem>
                    {categories.map((cat) => (
                      <SelectItem key={cat} value={cat as string}>
                        {cat}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* Bulk Actions Bar */}
              {someClassesSelected && (
                <div className="flex items-center justify-between bg-muted/50 rounded-lg p-3 border">
                  <div className="flex items-center gap-3">
                    <CheckSquare className="h-5 w-5 text-primary" />
                    <span className="font-medium">
                      {selectedClassIds.size} class(es) selected
                    </span>
                    <Button variant="ghost" size="sm" onClick={() => setSelectedClassIds(new Set())}>
                      <X className="h-4 w-4 mr-1" />
                      Clear
                    </Button>
                  </div>
                  <div className="flex gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={handleBulkMergeClasses}
                      disabled={selectedClassIds.size < 2}
                    >
                      <Merge className="h-4 w-4 mr-2" />
                      Merge Selected
                    </Button>
                    <Button
                      variant="destructive"
                      size="sm"
                      onClick={handleBulkDeleteClasses}
                    >
                      <Trash2 className="h-4 w-4 mr-2" />
                      Delete Selected
                    </Button>
                  </div>
                </div>
              )}

              {/* Table */}
              {classesLoading ? (
                <div className="flex items-center justify-center py-12">
                  <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
                </div>
              ) : filteredClasses?.length === 0 ? (
                <div className="text-center py-12">
                  <Tags className="h-12 w-12 mx-auto text-muted-foreground mb-2" />
                  <p className="text-muted-foreground">No classes found</p>
                  <Button className="mt-4" onClick={() => { resetClassForm(); setIsCreateClassDialogOpen(true); }}>
                    <Plus className="h-4 w-4 mr-2" />
                    Create Class
                  </Button>
                </div>
              ) : (
                <div className="border rounded-lg">
                  <Table>
                    <TableHeader>
                      <TableRow className="bg-muted/30">
                        <TableHead className="w-[50px]">
                          <Checkbox
                            checked={allClassesSelected}
                            onCheckedChange={toggleSelectAllClasses}
                            aria-label="Select all"
                          />
                        </TableHead>
                        <TableHead className="w-[60px]">Color</TableHead>
                        <TableHead>
                          <Button
                            variant="ghost"
                            size="sm"
                            className="-ml-3 h-8"
                            onClick={() => handleClassSort("name")}
                          >
                            Name
                            {classSortBy === "name" ? (
                              classSortOrder === "asc" ? <ArrowUp className="ml-2 h-4 w-4" /> : <ArrowDown className="ml-2 h-4 w-4" />
                            ) : (
                              <ArrowUpDown className="ml-2 h-4 w-4 opacity-50" />
                            )}
                          </Button>
                        </TableHead>
                        <TableHead>Display Name</TableHead>
                        <TableHead>
                          <Button
                            variant="ghost"
                            size="sm"
                            className="-ml-3 h-8"
                            onClick={() => handleClassSort("category")}
                          >
                            Category
                            {classSortBy === "category" ? (
                              classSortOrder === "asc" ? <ArrowUp className="ml-2 h-4 w-4" /> : <ArrowDown className="ml-2 h-4 w-4" />
                            ) : (
                              <ArrowUpDown className="ml-2 h-4 w-4 opacity-50" />
                            )}
                          </Button>
                        </TableHead>
                        <TableHead>
                          <Button
                            variant="ghost"
                            size="sm"
                            className="-ml-3 h-8"
                            onClick={() => handleClassSort("annotation_count")}
                          >
                            Annotations
                            {classSortBy === "annotation_count" ? (
                              classSortOrder === "asc" ? <ArrowUp className="ml-2 h-4 w-4" /> : <ArrowDown className="ml-2 h-4 w-4" />
                            ) : (
                              <ArrowUpDown className="ml-2 h-4 w-4 opacity-50" />
                            )}
                          </Button>
                        </TableHead>
                        <TableHead>Type</TableHead>
                        <TableHead className="w-[50px]"></TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {filteredClasses?.map((cls) => (
                        <TableRow
                          key={cls.id}
                          className={selectedClassIds.has(cls.id) ? "bg-primary/5" : ""}
                        >
                          <TableCell>
                            <Checkbox
                              checked={selectedClassIds.has(cls.id)}
                              onCheckedChange={() => toggleSelectClass(cls.id)}
                              aria-label={`Select ${cls.name}`}
                            />
                          </TableCell>
                          <TableCell>
                            <div
                              className="w-8 h-8 rounded-md border shadow-sm"
                              style={{ backgroundColor: cls.color }}
                            />
                          </TableCell>
                          <TableCell className="font-medium">{cls.name}</TableCell>
                          <TableCell className="text-muted-foreground">
                            {cls.display_name || "-"}
                          </TableCell>
                          <TableCell>
                            {cls.category ? (
                              <Badge variant="secondary">{cls.category}</Badge>
                            ) : (
                              <span className="text-muted-foreground">-</span>
                            )}
                          </TableCell>
                          <TableCell className="font-mono text-right tabular-nums">
                            {cls.annotation_count.toLocaleString()}
                          </TableCell>
                          <TableCell>
                            {cls.is_system ? (
                              <Badge variant="outline" className="gap-1">
                                <Lock className="h-3 w-3" />
                                System
                              </Badge>
                            ) : (
                              <Badge variant="secondary">Custom</Badge>
                            )}
                          </TableCell>
                          <TableCell>
                            <DropdownMenu>
                              <DropdownMenuTrigger asChild>
                                <Button variant="ghost" size="icon">
                                  <MoreHorizontal className="h-4 w-4" />
                                </Button>
                              </DropdownMenuTrigger>
                              <DropdownMenuContent align="end">
                                <DropdownMenuItem onClick={() => openEditClassDialog(cls)}>
                                  <Edit className="h-4 w-4 mr-2" />
                                  Edit
                                </DropdownMenuItem>
                                <DropdownMenuSeparator />
                                <DropdownMenuItem
                                  className="text-destructive"
                                  onClick={() => handleDeleteClass(cls)}
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
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Add Images Sheet */}
      <Sheet open={isAddImagesSheetOpen} onOpenChange={setIsAddImagesSheetOpen}>
        <SheetContent side="right" className="w-full sm:max-w-3xl lg:max-w-5xl p-0 flex flex-col">
          <SheetHeader className="px-6 py-4 border-b">
            <SheetTitle className="flex items-center gap-2">
              <Plus className="h-5 w-5" />
              Add Images to Dataset
            </SheetTitle>
            <SheetDescription>
              Select images from your library to add to this dataset.
            </SheetDescription>
          </SheetHeader>

          <div className="flex-1 overflow-hidden flex flex-col p-6 space-y-4">
            {/* Search & Filters Row */}
            <div className="flex gap-3 items-center">
              <div className="relative flex-1 max-w-md">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search images..."
                  value={availableSearch}
                  onChange={(e) => {
                    setAvailableSearch(e.target.value);
                    setAvailableImagesPage(1);
                    setSelectAllFilteredAddMode(false);
                  }}
                  className="pl-10"
                />
              </div>

              {/* Filter Button */}
              <FilterTrigger
                onClick={() => setAddImagesFilterDrawerOpen(true)}
                activeCount={addActiveFilterCount}
              />
              {addActiveFilterCount > 0 && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => {
                    clearAddFilters();
                    setAvailableImagesPage(1);
                    setSelectAllFilteredAddMode(false);
                  }}
                  className="text-muted-foreground hover:text-foreground"
                >
                  Clear filters
                </Button>
              )}

              {/* Selection Controls */}
              <div className="flex items-center gap-2 ml-auto">
                {availableImagesData?.images && availableImagesData.images.length > 0 && (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      if (selectAllFilteredAddMode) {
                        setSelectAllFilteredAddMode(false);
                        setSelectedToAdd(new Set());
                      } else {
                        const currentPageIds = availableImagesData.images
                          .filter((img) => !imagesInDataset.has(img.id))
                          .map((img) => img.id);
                        const allSelected = currentPageIds.every((id) => selectedToAdd.has(id));
                        if (allSelected) {
                          setSelectedToAdd((prev) => {
                            const newSet = new Set(prev);
                            currentPageIds.forEach((id) => newSet.delete(id));
                            return newSet;
                          });
                        } else {
                          setSelectedToAdd((prev) => {
                            const newSet = new Set(prev);
                            currentPageIds.forEach((id) => newSet.add(id));
                            return newSet;
                          });
                        }
                      }
                    }}
                  >
                    {selectAllFilteredAddMode
                      ? "Clear Selection"
                      : availableImagesData?.images
                          .filter((img) => !imagesInDataset.has(img.id))
                          .every((img) => selectedToAdd.has(img.id))
                      ? "Deselect Page"
                      : "Select Page"}
                  </Button>
                )}
              </div>
            </div>

            {/* Selection Summary Bar */}
            {(selectedToAdd.size > 0 || selectAllFilteredAddMode) && (
              <div className="flex items-center gap-3 px-4 py-2 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
                <span className="text-sm font-medium text-blue-700 dark:text-blue-300">
                  {selectAllFilteredAddMode
                    ? `All ${availableImagesData?.total || 0} filtered images selected`
                    : `${selectedToAdd.size} image(s) selected`}
                </span>
                {!selectAllFilteredAddMode && selectedToAdd.size > 0 && (availableImagesData?.total || 0) > (availableImagesData?.images?.length || 0) && (
                  <Button
                    variant="link"
                    size="sm"
                    className="text-blue-600 dark:text-blue-400 p-0 h-auto"
                    onClick={() => {
                      setSelectAllFilteredAddMode(true);
                      setSelectedToAdd(new Set());
                    }}
                  >
                    Select all {availableImagesData?.total} filtered images
                  </Button>
                )}
                {selectAllFilteredAddMode && (
                  <Button
                    variant="link"
                    size="sm"
                    className="text-muted-foreground p-0 h-auto"
                    onClick={() => {
                      setSelectAllFilteredAddMode(false);
                      setSelectedToAdd(new Set());
                    }}
                  >
                    Clear selection
                  </Button>
                )}
              </div>
            )}

            {/* Image Grid */}
            <div className="flex-1 overflow-y-auto">
              {availableLoading ? (
                <div className="flex items-center justify-center py-12">
                  <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
                </div>
              ) : availableImagesData?.images.length === 0 ? (
                <div className="text-center py-12">
                  <ImageIcon className="h-12 w-12 mx-auto text-muted-foreground mb-2" />
                  <p className="text-muted-foreground">
                    {availableSearch || addActiveFilterCount > 0
                      ? "No images found matching filters"
                      : "No images found in library"}
                  </p>
                </div>
              ) : (
                <div className="grid grid-cols-4 sm:grid-cols-5 md:grid-cols-6 lg:grid-cols-7 gap-3">
                  {availableImagesData?.images.map((image) => {
                    const isInDataset = imagesInDataset.has(image.id);
                    const isSelected = selectAllFilteredAddMode ? !isInDataset : selectedToAdd.has(image.id);

                    return (
                      <div
                        key={image.id}
                        className={`relative aspect-square rounded-lg overflow-hidden bg-muted border-2 transition-all ${
                          isInDataset
                            ? "border-green-500/50 opacity-60 cursor-not-allowed"
                            : isSelected
                            ? "border-primary ring-2 ring-primary/20 cursor-pointer"
                            : "border-transparent hover:border-muted-foreground/30 cursor-pointer"
                        }`}
                        onClick={() => {
                          if (isInDataset) return;
                          if (selectAllFilteredAddMode) {
                            setSelectAllFilteredAddMode(false);
                            // Select all except this one
                            const allIds = availableImagesData.images
                              .filter((img) => !imagesInDataset.has(img.id) && img.id !== image.id)
                              .map((img) => img.id);
                            setSelectedToAdd(new Set(allIds));
                          } else {
                            toggleAddSelection(image.id);
                          }
                        }}
                      >
                        <Image
                          src={image.image_url}
                          alt={image.filename}
                          fill
                          className="object-cover"
                          unoptimized
                        />
                        {isInDataset ? (
                          <div className="absolute inset-0 bg-green-600/30 flex items-center justify-center">
                            <Badge variant="secondary" className="bg-green-600 text-white text-xs">
                              <CheckCircle className="h-3 w-3 mr-1" />
                              In Dataset
                            </Badge>
                          </div>
                        ) : isSelected && (
                          <div className="absolute inset-0 bg-primary/20 flex items-center justify-center">
                            <CheckCircle className="h-8 w-8 text-primary" />
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>

            {/* Pagination */}
            {availableTotalPages > 1 && (
              <div className="flex items-center justify-between pt-2 border-t">
                <p className="text-sm text-muted-foreground">
                  Page {availableImagesPage} of {availableTotalPages} ({availableImagesData?.total} total)
                </p>
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setAvailableImagesPage((p) => Math.max(1, p - 1))}
                    disabled={availableImagesPage === 1}
                  >
                    Previous
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setAvailableImagesPage((p) => Math.min(availableTotalPages, p + 1))}
                    disabled={availableImagesPage === availableTotalPages}
                  >
                    Next
                  </Button>
                </div>
              </div>
            )}
          </div>

          {/* Footer */}
          <div className="px-6 py-4 border-t flex items-center justify-between">
            <span className="text-sm text-muted-foreground">
              {selectAllFilteredAddMode
                ? `Will add up to ${availableImagesData?.total || 0} images`
                : `${selectedToAdd.size} image(s) selected`}
            </span>
            <div className="flex gap-2">
              <Button variant="outline" onClick={() => setIsAddImagesSheetOpen(false)}>
                Cancel
              </Button>
              <Button
                onClick={() => {
                  if (selectAllFilteredAddMode) {
                    addImagesByFiltersMutation.mutate();
                  } else {
                    addImagesMutation.mutate(Array.from(selectedToAdd));
                  }
                }}
                disabled={
                  (!selectAllFilteredAddMode && selectedToAdd.size === 0) ||
                  addImagesMutation.isPending ||
                  addImagesByFiltersMutation.isPending
                }
              >
                {addImagesMutation.isPending || addImagesByFiltersMutation.isPending ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Adding...
                  </>
                ) : (
                  <>
                    <Plus className="h-4 w-4 mr-2" />
                    {selectAllFilteredAddMode
                      ? `Add All (${availableImagesData?.total || 0})`
                      : `Add (${selectedToAdd.size})`}
                  </>
                )}
              </Button>
            </div>
          </div>
        </SheetContent>
      </Sheet>

      {/* Filter Drawer for Add Images */}
      <FilterDrawer
        open={addImagesFilterDrawerOpen}
        onOpenChange={setAddImagesFilterDrawerOpen}
        sections={addImagesFilterSections}
        filterState={addFilterState}
        onFilterChange={(sectionId, value) => {
          setAddFilter(sectionId, value);
          setAvailableImagesPage(1);
          setSelectAllFilteredAddMode(false);
        }}
        onClearAll={() => {
          clearAddFilters();
          setAvailableImagesPage(1);
          setSelectAllFilteredAddMode(false);
        }}
        onClearSection={(sectionId) => {
          clearAddSection(sectionId);
          setAvailableImagesPage(1);
          setSelectAllFilteredAddMode(false);
        }}
        title="Image Filters"
        description="Filter images by status, source, merchant, and store"
      />

      {/* Create Class Dialog */}
      <Dialog open={isCreateClassDialogOpen} onOpenChange={setIsCreateClassDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Plus className="h-5 w-5" />
              New Class
            </DialogTitle>
            <DialogDescription>
              Create a new detection class for this dataset.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label>Name *</Label>
              <Input
                placeholder="e.g., product, price_tag, shelf"
                value={formName}
                onChange={(e) => setFormName(e.target.value.toLowerCase().replace(/\s+/g, "_"))}
              />
              <p className="text-xs text-muted-foreground">
                Use lowercase with underscores (e.g., price_tag)
              </p>
            </div>

            <div className="space-y-2">
              <Label>Display Name</Label>
              <Input
                placeholder="e.g., Product, Price Tag, Shelf"
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
                    className={`w-8 h-8 rounded-md border-2 transition-all ${
                      formColor === color ? "border-foreground scale-110" : "border-transparent"
                    }`}
                    style={{ backgroundColor: color }}
                    onClick={() => setFormColor(color)}
                  />
                ))}
              </div>
              <div className="flex items-center gap-2 mt-2">
                <Palette className="h-4 w-4 text-muted-foreground" />
                <Input
                  type="color"
                  value={formColor}
                  onChange={(e) => setFormColor(e.target.value)}
                  className="w-20 h-8 p-1"
                />
                <span className="text-sm text-muted-foreground font-mono">{formColor}</span>
              </div>
            </div>

            <div className="space-y-2">
              <Label>Category</Label>
              <Input
                placeholder="e.g., retail, shelf_elements"
                value={formCategory}
                onChange={(e) => setFormCategory(e.target.value)}
              />
            </div>
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setIsCreateClassDialogOpen(false)}>
              Cancel
            </Button>
            <Button
              onClick={() => createClassMutation.mutate()}
              disabled={!formName || createClassMutation.isPending}
            >
              {createClassMutation.isPending ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Creating...
                </>
              ) : (
                "Create Class"
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Edit Class Dialog */}
      <Dialog open={isEditClassDialogOpen} onOpenChange={setIsEditClassDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Edit className="h-5 w-5" />
              Edit Class
            </DialogTitle>
          </DialogHeader>

          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label>Name *</Label>
              <Input
                placeholder="e.g., product, price_tag, shelf"
                value={formName}
                onChange={(e) => setFormName(e.target.value.toLowerCase().replace(/\s+/g, "_"))}
              />
            </div>

            <div className="space-y-2">
              <Label>Display Name</Label>
              <Input
                placeholder="e.g., Product, Price Tag, Shelf"
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
                    className={`w-8 h-8 rounded-md border-2 transition-all ${
                      formColor === color ? "border-foreground scale-110" : "border-transparent"
                    }`}
                    style={{ backgroundColor: color }}
                    onClick={() => setFormColor(color)}
                  />
                ))}
              </div>
              <div className="flex items-center gap-2 mt-2">
                <Palette className="h-4 w-4 text-muted-foreground" />
                <Input
                  type="color"
                  value={formColor}
                  onChange={(e) => setFormColor(e.target.value)}
                  className="w-20 h-8 p-1"
                />
              </div>
            </div>

            <div className="space-y-2">
              <Label>Category</Label>
              <Input
                placeholder="e.g., retail, shelf_elements"
                value={formCategory}
                onChange={(e) => setFormCategory(e.target.value)}
              />
            </div>
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setIsEditClassDialogOpen(false)}>
              Cancel
            </Button>
            <Button
              onClick={() => updateClassMutation.mutate()}
              disabled={!formName || updateClassMutation.isPending}
            >
              {updateClassMutation.isPending ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Saving...
                </>
              ) : (
                "Save Changes"
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Merge Classes Dialog */}
      <Dialog open={isMergeDialogOpen} onOpenChange={setIsMergeDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Merge className="h-5 w-5" />
              Merge Classes
            </DialogTitle>
            <DialogDescription>
              Combine multiple classes into one. All annotations from source classes will be moved to the target.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label>Source Classes (to merge from)</Label>
              <div className="border rounded-md p-3 max-h-40 overflow-y-auto space-y-2">
                {classes?.filter((c) => c.id !== mergeTargetId).map((cls) => (
                  <label key={cls.id} className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={mergeSourceIds.includes(cls.id)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setMergeSourceIds([...mergeSourceIds, cls.id]);
                        } else {
                          setMergeSourceIds(mergeSourceIds.filter((id) => id !== cls.id));
                        }
                      }}
                      className="rounded"
                    />
                    <div
                      className="w-4 h-4 rounded"
                      style={{ backgroundColor: cls.color }}
                    />
                    <span>{cls.name}</span>
                    <span className="text-muted-foreground text-sm">
                      ({cls.annotation_count} annotations)
                    </span>
                  </label>
                ))}
              </div>
              <p className="text-xs text-muted-foreground">
                {mergeSourceIds.length} class(es) selected
              </p>
            </div>

            <div className="space-y-2">
              <Label>Target Class (merge into)</Label>
              <Select value={mergeTargetId} onValueChange={setMergeTargetId}>
                <SelectTrigger>
                  <SelectValue placeholder="Select target class" />
                </SelectTrigger>
                <SelectContent>
                  {classes?.filter((c) => !mergeSourceIds.includes(c.id)).map((cls) => (
                    <SelectItem key={cls.id} value={cls.id}>
                      <div className="flex items-center gap-2">
                        <div
                          className="w-4 h-4 rounded"
                          style={{ backgroundColor: cls.color }}
                        />
                        {cls.name} ({cls.annotation_count})
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Merge Preview Summary */}
            {mergePreview && mergeTargetId && (
              <div className="bg-muted/50 rounded-lg p-4 space-y-3">
                <h4 className="font-medium text-sm">Merge Summary</h4>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <p className="text-muted-foreground">Annotations to move</p>
                    <p className="text-xl font-bold text-blue-600">
                      {mergePreview.sourceAnnotations.toLocaleString()}
                    </p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Target current</p>
                    <p className="text-xl font-bold">
                      {mergePreview.targetAnnotations.toLocaleString()}
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-2 pt-2 border-t">
                  <span className="text-sm text-muted-foreground">After merge:</span>
                  <span className="font-mono font-bold">{mergePreview.targetName}</span>
                  <ArrowRight className="h-4 w-4 text-muted-foreground" />
                  <span className="text-lg font-bold text-green-600">
                    {mergePreview.totalAfterMerge.toLocaleString()} annotations
                  </span>
                </div>
              </div>
            )}

            {mergeSourceIds.length > 0 && mergeTargetId && (
              <div className="bg-orange-50 text-orange-800 p-3 rounded-md text-sm">
                <strong>Warning:</strong> This will permanently delete {mergeSourceIds.length} source class(es) and move all their annotations to the target class.
              </div>
            )}
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setIsMergeDialogOpen(false)}>
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={() => mergeClassesMutation.mutate()}
              disabled={mergeSourceIds.length === 0 || !mergeTargetId || mergeClassesMutation.isPending}
            >
              {mergeClassesMutation.isPending ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Merging...
                </>
              ) : (
                <>
                  <Merge className="h-4 w-4 mr-2" />
                  Merge {mergeSourceIds.length} Class(es)
                </>
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Advanced Import Modal (with dataset context) */}
      <ImportModal
        open={isImportModalOpen}
        onOpenChange={setIsImportModalOpen}
        datasetId={datasetId}
        onSuccess={() => {
          queryClient.invalidateQueries({ queryKey: ["od-dataset", datasetId] });
          queryClient.invalidateQueries({ queryKey: ["od-dataset-images", datasetId] });
          queryClient.invalidateQueries({ queryKey: ["od-classes", datasetId] });
        }}
      />

      {/* Bulk AI Annotation Modal */}
      <BulkAnnotateModal
        open={isBulkAnnotateOpen}
        onOpenChange={setIsBulkAnnotateOpen}
        datasetId={datasetId}
        totalImages={dataset.image_count}
        unannotatedCount={dataset.image_count - ((dataset as any).annotated_image_count ?? 0)}
        selectedImageIds={Array.from(selectedImages)}
        existingClasses={classes || []}
        onSuccess={() => {
          queryClient.invalidateQueries({ queryKey: ["od-dataset", datasetId] });
          queryClient.invalidateQueries({ queryKey: ["od-dataset-images", datasetId] });
        }}
      />
    </div>
  );
}
