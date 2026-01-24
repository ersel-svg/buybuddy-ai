"use client";

import { useState, useCallback, useMemo, useEffect } from "react";
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
import { Input } from "@/components/ui/input";
import { Checkbox } from "@/components/ui/checkbox";
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
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  FilterDrawer,
  FilterTrigger,
  useFilterState,
  type FilterSection,
} from "@/components/filters/filter-drawer";
import {
  RefreshCw,
  Loader2,
  ImageIcon,
  Search,
  Upload,
  Trash2,
  MoreHorizontal,
  FolderOpen,
  Clock,
  CheckCircle,
  AlertCircle,
  XCircle,
  Grid3X3,
  List,
  Plus,
  FolderPlus,
  Package,
} from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import Image from "next/image";
import { useDropzone } from "react-dropzone";
import { ImportModal } from "@/components/od/import-modal";
import { JobProgressModal } from "@/components/common/job-progress-modal";

const PAGE_SIZE = 48;
const ASYNC_THRESHOLD = 500; // Use async for more than 500 images

// Custom hook for debounced value
function useDebounce<T>(value: T, delay: number): T {
  const [debouncedValue, setDebouncedValue] = useState<T>(value);

  useEffect(() => {
    const timer = setTimeout(() => setDebouncedValue(value), delay);
    return () => clearTimeout(timer);
  }, [value, delay]);

  return debouncedValue;
}

interface ODImage {
  id: string;
  filename: string;
  image_url: string;
  thumbnail_url?: string;
  width: number;
  height: number;
  status: string;
  source: string;
  folder?: string;
  created_at: string;
}

export default function ODImagesPage() {
  const queryClient = useQueryClient();
  const [page, setPage] = useState(1);
  const [searchInput, setSearchInput] = useState("");
  const debouncedSearch = useDebounce(searchInput, 300); // 300ms debounce
  const [viewMode, setViewMode] = useState<"grid" | "list">("grid");
  const [filterDrawerOpen, setFilterDrawerOpen] = useState(false);

  // Filter state using the hook
  const {
    filterState,
    setFilter,
    clearSection,
    clearAll,
    getTotalCount,
  } = useFilterState();

  // Selection state
  const [selectedImages, setSelectedImages] = useState<Set<string>>(new Set());
  const [selectAllFilteredMode, setSelectAllFilteredMode] = useState(false);

  // Upload dialog state
  const [isUploadDialogOpen, setIsUploadDialogOpen] = useState(false);
  const [uploadFiles, setUploadFiles] = useState<File[]>([]);
  const [uploadFolder, setUploadFolder] = useState("");

  // Advanced import modal state
  const [isImportModalOpen, setIsImportModalOpen] = useState(false);

  // Add to dataset dialog state
  const [isAddToDatasetOpen, setIsAddToDatasetOpen] = useState(false);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>("");

  // Background job state
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [activeDeleteJobId, setActiveDeleteJobId] = useState<string | null>(null);

  // Fetch OD stats
  const { data: stats } = useQuery({
    queryKey: ["od-stats"],
    queryFn: () => apiClient.getODStats(),
    staleTime: 30000, // 30 seconds
    gcTime: 5 * 60 * 1000, // 5 minutes
  });

  // Fetch datasets for "Add to Dataset" dialog
  const { data: datasets } = useQuery({
    queryKey: ["od-datasets"],
    queryFn: () => apiClient.getODDatasets(),
    staleTime: 60000, // 1 minute
    gcTime: 5 * 60 * 1000, // 5 minutes
  });

  // Fetch filter options with counts
  const { data: filterOptions } = useQuery({
    queryKey: ["od-image-filter-options"],
    queryFn: () => apiClient.getODImageFilterOptions(),
    staleTime: 5 * 60 * 1000, // 5 minutes - filter options change rarely
    gcTime: 10 * 60 * 1000, // 10 minutes
  });

  // Build filter sections from API data
  const filterSections: FilterSection[] = useMemo(() => {
    if (!filterOptions) return [];

    const sections: FilterSection[] = [];

    // Status filter
    if (filterOptions.status?.length > 0) {
      sections.push({
        id: "status",
        label: "Status",
        type: "checkbox",
        options: filterOptions.status,
        defaultExpanded: true,
      });
    }

    // Source filter
    if (filterOptions.source?.length > 0) {
      sections.push({
        id: "source",
        label: "Source",
        type: "checkbox",
        options: filterOptions.source,
        defaultExpanded: true,
      });
    }

    // Merchant filter
    if (filterOptions.merchant?.length > 0) {
      sections.push({
        id: "merchant",
        label: "Merchant",
        type: "checkbox",
        options: filterOptions.merchant,
        searchable: true,
        defaultExpanded: true,
      });
    }

    // Store filter
    if (filterOptions.store?.length > 0) {
      sections.push({
        id: "store",
        label: "Store",
        type: "checkbox",
        options: filterOptions.store,
        searchable: true,
        defaultExpanded: false,
      });
    }

    // Folder filter
    if (filterOptions.folder?.length > 0) {
      sections.push({
        id: "folder",
        label: "Folder",
        type: "checkbox",
        options: filterOptions.folder,
        searchable: true,
        defaultExpanded: false,
      });
    }

    return sections;
  }, [filterOptions]);

  // Convert filter state to API params
  const apiFilters = useMemo(() => {
    const params: Record<string, string | undefined> = {};

    // Convert Set to comma-separated string
    const setToString = (key: string): string | undefined => {
      const filter = filterState[key] as Set<string> | undefined;
      if (filter?.size) {
        return Array.from(filter).join(",");
      }
      return undefined;
    };

    params.statuses = setToString("status");
    params.sources = setToString("source");
    params.merchant_ids = setToString("merchant");
    params.store_ids = setToString("store");
    params.folders = setToString("folder");

    return params;
  }, [filterState]);

  const activeFilterCount = getTotalCount();

  // Fetch images with multi-select filters
  const {
    data: imagesData,
    isLoading,
    isFetching,
  } = useQuery({
    queryKey: ["od-images", page, debouncedSearch, apiFilters],
    queryFn: () =>
      apiClient.getODImages({
        page,
        limit: PAGE_SIZE,
        search: debouncedSearch || undefined,
        ...apiFilters,
      }),
    staleTime: 30000, // 30 seconds
    gcTime: 5 * 60 * 1000, // 5 minutes
  });

  // Upload mutation
  const uploadMutation = useMutation({
    mutationFn: async () => {
      if (uploadFiles.length === 1) {
        return apiClient.uploadODImage(uploadFiles[0], uploadFolder || undefined);
      } else {
        return apiClient.uploadODImagesBulk(uploadFiles);
      }
    },
    onSuccess: (data) => {
      const count = Array.isArray(data) ? data.length : 1;
      toast.success(`Uploaded ${count} image${count > 1 ? "s" : ""}`);
      queryClient.invalidateQueries({ queryKey: ["od-images"] });
      queryClient.invalidateQueries({ queryKey: ["od-stats"] });
      setIsUploadDialogOpen(false);
      setUploadFiles([]);
      setUploadFolder("");
    },
    onError: (error) => {
      toast.error(`Upload failed: ${error.message}`);
    },
  });

  // Delete mutation
  const deleteMutation = useMutation({
    mutationFn: async (imageIds: string[]) => {
      if (imageIds.length === 1) {
        return apiClient.deleteODImage(imageIds[0]);
      } else {
        return apiClient.deleteODImagesBulk(imageIds);
      }
    },
    onSuccess: (_, imageIds) => {
      toast.success(`Deleted ${imageIds.length} image${imageIds.length > 1 ? "s" : ""}`);
      queryClient.invalidateQueries({ queryKey: ["od-images"] });
      queryClient.invalidateQueries({ queryKey: ["od-stats"] });
      setSelectedImages(new Set());
    },
    onError: (error) => {
      toast.error(`Delete failed: ${error.message}`);
    },
  });

  // Add to dataset mutation (by IDs)
  const addToDatasetMutation = useMutation({
    mutationFn: async ({ datasetId, imageIds }: { datasetId: string; imageIds: string[] }) => {
      return apiClient.addImagesToODDataset(datasetId, imageIds);
    },
    onSuccess: (result) => {
      toast.success(`Added ${result.added} image${result.added !== 1 ? "s" : ""} to dataset${result.skipped > 0 ? ` (${result.skipped} already in dataset)` : ""}`);
      queryClient.invalidateQueries({ queryKey: ["od-datasets"] });
      queryClient.invalidateQueries({ queryKey: ["od-stats"] });
      setIsAddToDatasetOpen(false);
      setSelectedDatasetId("");
      clearSelection();
    },
    onError: (error) => {
      toast.error(`Failed to add images: ${error.message}`);
    },
  });

  // Delete by filters mutation (for selectAllFilteredMode - sync)
  const deleteByFiltersMutation = useMutation({
    mutationFn: async () => {
      return apiClient.deleteODImagesByFilters({
        search: debouncedSearch || undefined,
        ...apiFilters,
      });
    },
    onSuccess: (result) => {
      toast.success(`Deleted ${result.deleted} image${result.deleted !== 1 ? "s" : ""}`);
      queryClient.invalidateQueries({ queryKey: ["od-images"] });
      queryClient.invalidateQueries({ queryKey: ["od-stats"] });
      queryClient.invalidateQueries({ queryKey: ["od-image-filter-options"] });
      clearSelection();
    },
    onError: (error) => {
      toast.error(`Delete failed: ${error.message}`);
    },
  });

  // Delete by filters mutation (async - for large batches)
  const deleteByFiltersAsyncMutation = useMutation({
    mutationFn: async () => {
      return apiClient.deleteODImagesByFiltersAsync({
        search: debouncedSearch || undefined,
        ...apiFilters,
      });
    },
    onSuccess: (result) => {
      setActiveDeleteJobId(result.job_id);
      toast.info("Delete started in background");
    },
    onError: (error) => {
      toast.error(`Failed to start delete: ${error.message}`);
    },
  });

  // Add to dataset by filters mutation (for selectAllFilteredMode)
  const addToDatasetByFiltersMutation = useMutation({
    mutationFn: async ({ datasetId }: { datasetId: string }) => {
      return apiClient.addFilteredImagesToODDataset(datasetId, {
        search: debouncedSearch || undefined,
        ...apiFilters,
      });
    },
    onSuccess: (result) => {
      toast.success(`Added ${result.added} image${result.added !== 1 ? "s" : ""} to dataset${result.skipped > 0 ? ` (${result.skipped} already in dataset)` : ""}`);
      queryClient.invalidateQueries({ queryKey: ["od-datasets"] });
      queryClient.invalidateQueries({ queryKey: ["od-stats"] });
      setIsAddToDatasetOpen(false);
      setSelectedDatasetId("");
      clearSelection();
    },
    onError: (error) => {
      toast.error(`Failed to add images: ${error.message}`);
    },
  });

  const handleAddToDataset = async () => {
    if (!selectedDatasetId) return;

    if (selectAllFilteredMode) {
      const total = imagesData?.total || 0;

      // Use async endpoint for large batches
      if (total > ASYNC_THRESHOLD) {
        try {
          const result = await apiClient.addFilteredImagesToODDatasetAsync(
            selectedDatasetId,
            {
              search: debouncedSearch || undefined,
              ...apiFilters,
            }
          );
          setActiveJobId(result.job_id);
          setIsAddToDatasetOpen(false);
          setSelectedDatasetId("");
        } catch (error) {
          toast.error(`Failed to start job: ${error instanceof Error ? error.message : "Unknown error"}`);
        }
      } else {
        // Use sync endpoint for small batches
        addToDatasetByFiltersMutation.mutate({ datasetId: selectedDatasetId });
      }
    } else {
      if (selectedImages.size === 0) return;
      addToDatasetMutation.mutate({
        datasetId: selectedDatasetId,
        imageIds: Array.from(selectedImages),
      });
    }
  };

  // Dropzone for upload
  const onDrop = useCallback((acceptedFiles: File[]) => {
    setUploadFiles((prev) => [...prev, ...acceptedFiles]);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "image/*": [".png", ".jpg", ".jpeg", ".webp"],
    },
  });

  // Image selection handlers
  const toggleImageSelection = (imageId: string) => {
    // Exit selectAllFilteredMode when individual selection is made
    if (selectAllFilteredMode) {
      setSelectAllFilteredMode(false);
    }
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
    // If selectAllFilteredMode is on or all page items selected, clear everything
    if (selectAllFilteredMode || selectedImages.size === imagesData.images.length) {
      setSelectAllFilteredMode(false);
      setSelectedImages(new Set());
    } else {
      // Select current page items
      setSelectedImages(new Set(imagesData.images.map((img) => img.id)));
    }
  };

  const handleSelectAllFiltered = () => {
    setSelectAllFilteredMode(true);
    setSelectedImages(new Set());
  };

  const clearSelection = () => {
    setSelectAllFilteredMode(false);
    setSelectedImages(new Set());
  };

  const handleDeleteSelected = () => {
    if (selectAllFilteredMode) {
      const total = imagesData?.total || 0;
      if (!confirm(`Delete ALL ${total} filtered images? This cannot be undone.`)) return;

      // Use async for large batches
      if (total > ASYNC_THRESHOLD) {
        deleteByFiltersAsyncMutation.mutate();
      } else {
        deleteByFiltersMutation.mutate();
      }
    } else {
      if (selectedImages.size === 0) return;
      if (!confirm(`Delete ${selectedImages.size} selected images?`)) return;
      deleteMutation.mutate(Array.from(selectedImages));
    }
  };

  const totalPages = imagesData ? Math.ceil(imagesData.total / PAGE_SIZE) : 0;

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

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold">Images</h1>
          <p className="text-muted-foreground">
            Upload and manage images for object detection
          </p>
        </div>
        <div className="flex gap-2">
          <Button
            variant="outline"
            onClick={() => {
              queryClient.invalidateQueries({ queryKey: ["od-images"] });
              queryClient.invalidateQueries({ queryKey: ["od-stats"] });
            }}
            disabled={isFetching}
          >
            <RefreshCw className={`h-4 w-4 mr-2 ${isFetching ? "animate-spin" : ""}`} />
            Refresh
          </Button>
          <Button variant="outline" onClick={() => setIsUploadDialogOpen(true)}>
            <Upload className="h-4 w-4 mr-2" />
            Quick Upload
          </Button>
          <Button onClick={() => setIsImportModalOpen(true)}>
            <Package className="h-4 w-4 mr-2" />
            Import
          </Button>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-5 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">Total Images</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">{stats?.total_images?.toLocaleString() ?? "-"}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">Pending</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-orange-600">
              {stats?.images_by_status?.pending?.toLocaleString() ?? "-"}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">Annotating</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-blue-600">
              {stats?.images_by_status?.annotating?.toLocaleString() ?? "-"}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">Completed</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-green-600">
              {stats?.images_by_status?.completed?.toLocaleString() ?? "-"}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">Skipped</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-gray-500">
              {stats?.images_by_status?.skipped?.toLocaleString() ?? "-"}
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Filters & Grid */}
      <Card>
        <CardHeader>
          <div className="flex justify-between items-center">
            <div>
              <CardTitle>Image Library</CardTitle>
              <CardDescription>
                {imagesData?.total ?? 0} images in library
              </CardDescription>
            </div>
            <div className="flex items-center gap-2">
              {(selectedImages.size > 0 || selectAllFilteredMode) && (
                <>
                  <span className="text-sm text-muted-foreground">
                    {selectAllFilteredMode
                      ? `All ${imagesData?.total || 0} filtered images selected`
                      : `${selectedImages.size} selected`}
                  </span>
                  {/* Show "Select all filtered" link when page items are selected */}
                  {!selectAllFilteredMode && selectedImages.size > 0 && (imagesData?.total || 0) > (imagesData?.images?.length || 0) && (
                    <Button
                      variant="link"
                      size="sm"
                      className="text-blue-600 p-0 h-auto"
                      onClick={handleSelectAllFiltered}
                    >
                      Select all {imagesData?.total} filtered images
                    </Button>
                  )}
                  {selectAllFilteredMode && (
                    <Button
                      variant="link"
                      size="sm"
                      className="text-muted-foreground p-0 h-auto"
                      onClick={clearSelection}
                    >
                      Clear selection
                    </Button>
                  )}
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setIsAddToDatasetOpen(true)}
                  >
                    <FolderPlus className="h-4 w-4 mr-1" />
                    Add to Dataset
                  </Button>
                  <Button
                    variant="destructive"
                    size="sm"
                    onClick={handleDeleteSelected}
                    disabled={deleteMutation.isPending || deleteByFiltersMutation.isPending || deleteByFiltersAsyncMutation.isPending}
                  >
                    {(deleteMutation.isPending || deleteByFiltersMutation.isPending || deleteByFiltersAsyncMutation.isPending) ? (
                      <>
                        <Loader2 className="h-4 w-4 mr-1 animate-spin" />
                        Deleting...
                      </>
                    ) : (
                      <>
                        <Trash2 className="h-4 w-4 mr-1" />
                        Delete
                      </>
                    )}
                  </Button>
                </>
              )}
              <div className="flex border rounded-md">
                <Button
                  variant={viewMode === "grid" ? "secondary" : "ghost"}
                  size="sm"
                  className="rounded-r-none"
                  onClick={() => setViewMode("grid")}
                >
                  <Grid3X3 className="h-4 w-4" />
                </Button>
                <Button
                  variant={viewMode === "list" ? "secondary" : "ghost"}
                  size="sm"
                  className="rounded-l-none"
                  onClick={() => setViewMode("list")}
                >
                  <List className="h-4 w-4" />
                </Button>
              </div>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Filter Row */}
          <div className="flex gap-4 items-center">
            <div className="flex-1 max-w-md">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search by filename..."
                  value={searchInput}
                  onChange={(e) => {
                    setSearchInput(e.target.value);
                    setPage(1);
                  }}
                  className="pl-10"
                />
              </div>
            </div>

            {/* Filter Button */}
            <div className="flex items-center gap-2">
              <FilterTrigger
                onClick={() => setFilterDrawerOpen(true)}
                activeCount={activeFilterCount}
              />
              {activeFilterCount > 0 && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => {
                    clearAll();
                    setPage(1);
                  }}
                  className="text-muted-foreground hover:text-foreground"
                >
                  Clear filters
                </Button>
              )}
            </div>

            {imagesData?.images && imagesData.images.length > 0 && (
              <div className="flex items-center gap-2 ml-auto">
                <Checkbox
                  checked={selectAllFilteredMode || (selectedImages.size === imagesData.images.length && imagesData.images.length > 0)}
                  onCheckedChange={toggleSelectAll}
                />
                <span className="text-sm text-muted-foreground">Select all</span>
              </div>
            )}
          </div>

          {/* Filter Drawer */}
          <FilterDrawer
            open={filterDrawerOpen}
            onOpenChange={setFilterDrawerOpen}
            sections={filterSections}
            filterState={filterState}
            onFilterChange={(sectionId, value) => {
              setFilter(sectionId, value);
              setPage(1);
            }}
            onClearAll={() => {
              clearAll();
              setPage(1);
            }}
            onClearSection={(sectionId) => {
              clearSection(sectionId);
              setPage(1);
            }}
            title="Image Filters"
            description="Filter images by status, source, merchant, store, and folder"
          />

          {/* Image Grid/List */}
          {isLoading ? (
            <div className="flex items-center justify-center py-24">
              <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
            </div>
          ) : imagesData?.images.length === 0 ? (
            <div className="text-center py-24">
              <ImageIcon className="h-16 w-16 mx-auto text-muted-foreground mb-4" />
              <h3 className="text-lg font-medium">No images found</h3>
              <p className="text-muted-foreground mt-1">
                {debouncedSearch || activeFilterCount > 0
                  ? "Try adjusting your filters"
                  : "Upload images or import from URL, COCO, YOLO formats"}
              </p>
              <div className="flex gap-2 justify-center mt-4">
                <Button variant="outline" onClick={() => setIsUploadDialogOpen(true)}>
                  <Upload className="h-4 w-4 mr-2" />
                  Quick Upload
                </Button>
                <Button onClick={() => setIsImportModalOpen(true)}>
                  <Package className="h-4 w-4 mr-2" />
                  Import
                </Button>
              </div>
            </div>
          ) : viewMode === "grid" ? (
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-4">
              {imagesData?.images.map((image: ODImage) => (
                <div
                  key={image.id}
                  className={`group relative aspect-square rounded-lg overflow-hidden bg-muted border-2 transition-colors cursor-pointer ${
                    (selectAllFilteredMode || selectedImages.has(image.id)) ? "border-primary" : "border-transparent hover:border-muted-foreground/30"
                  }`}
                  onClick={() => toggleImageSelection(image.id)}
                >
                  <Image
                    src={image.thumbnail_url || image.image_url}
                    alt={image.filename}
                    fill
                    className="object-cover"
                    sizes="(max-width: 640px) 50vw, (max-width: 768px) 33vw, (max-width: 1024px) 25vw, 16vw"
                    loading="lazy"
                  />
                  {/* Selection checkbox */}
                  <div className={`absolute top-2 left-2 transition-opacity ${(selectAllFilteredMode || selectedImages.has(image.id)) ? "opacity-100" : "opacity-0 group-hover:opacity-100"}`}>
                    <Checkbox
                      checked={selectAllFilteredMode || selectedImages.has(image.id)}
                      onCheckedChange={() => toggleImageSelection(image.id)}
                      onClick={(e) => e.stopPropagation()}
                      className="bg-white/80"
                    />
                  </div>
                  {/* Status badge */}
                  <div className="absolute top-2 right-2">
                    {getStatusBadge(image.status)}
                  </div>
                  {/* Info overlay */}
                  <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/70 to-transparent p-2 opacity-0 group-hover:opacity-100 transition-opacity">
                    <p className="text-white text-xs truncate">{image.filename}</p>
                    <p className="text-white/70 text-xs">
                      {image.width}x{image.height}
                    </p>
                  </div>
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
                            window.open(image.image_url, "_blank");
                          }}
                        >
                          View full size
                        </DropdownMenuItem>
                        <DropdownMenuSeparator />
                        <DropdownMenuItem
                          className="text-destructive"
                          onClick={(e) => {
                            e.stopPropagation();
                            if (confirm("Delete this image?")) {
                              deleteMutation.mutate([image.id]);
                            }
                          }}
                        >
                          <Trash2 className="h-4 w-4 mr-2" />
                          Delete
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            // List view
            <div className="space-y-2">
              {imagesData?.images.map((image: ODImage) => (
                <div
                  key={image.id}
                  className={`flex items-center gap-4 p-3 rounded-lg border cursor-pointer transition-colors ${
                    (selectAllFilteredMode || selectedImages.has(image.id)) ? "border-primary bg-primary/5" : "hover:bg-muted/50"
                  }`}
                  onClick={() => toggleImageSelection(image.id)}
                >
                  <Checkbox
                    checked={selectAllFilteredMode || selectedImages.has(image.id)}
                    onCheckedChange={() => toggleImageSelection(image.id)}
                    onClick={(e) => e.stopPropagation()}
                  />
                  <div className="relative w-16 h-16 rounded overflow-hidden bg-muted flex-shrink-0">
                    <Image
                      src={image.thumbnail_url || image.image_url}
                      alt={image.filename}
                      fill
                      className="object-cover"
                      sizes="64px"
                      loading="lazy"
                    />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="font-medium truncate">{image.filename}</p>
                    <p className="text-sm text-muted-foreground">
                      {image.width}x{image.height} &bull; {image.source}
                      {image.folder && <> &bull; <FolderOpen className="h-3 w-3 inline mx-1" />{image.folder}</>}
                    </p>
                  </div>
                  {getStatusBadge(image.status)}
                  <p className="text-sm text-muted-foreground">
                    {new Date(image.created_at).toLocaleDateString()}
                  </p>
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild onClick={(e) => e.stopPropagation()}>
                      <Button variant="ghost" size="icon">
                        <MoreHorizontal className="h-4 w-4" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                      <DropdownMenuItem
                        onClick={(e) => {
                          e.stopPropagation();
                          window.open(image.image_url, "_blank");
                        }}
                      >
                        View full size
                      </DropdownMenuItem>
                      <DropdownMenuSeparator />
                      <DropdownMenuItem
                        className="text-destructive"
                        onClick={(e) => {
                          e.stopPropagation();
                          if (confirm("Delete this image?")) {
                            deleteMutation.mutate([image.id]);
                          }
                        }}
                      >
                        <Trash2 className="h-4 w-4 mr-2" />
                        Delete
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
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

      {/* Upload Dialog */}
      <Dialog open={isUploadDialogOpen} onOpenChange={setIsUploadDialogOpen}>
        <DialogContent className="max-w-xl">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Upload className="h-5 w-5" />
              Upload Images
            </DialogTitle>
            <DialogDescription>
              Drag and drop images or click to browse. Supports PNG, JPG, JPEG, WebP.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4 py-4">
            {/* Dropzone */}
            <div
              {...getRootProps()}
              className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
                isDragActive ? "border-primary bg-primary/5" : "border-muted-foreground/30 hover:border-primary/50"
              }`}
            >
              <input {...getInputProps()} />
              <ImageIcon className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
              {isDragActive ? (
                <p className="text-primary">Drop images here...</p>
              ) : (
                <>
                  <p>Drag &amp; drop images here</p>
                  <p className="text-sm text-muted-foreground mt-1">or click to browse</p>
                </>
              )}
            </div>

            {/* File list */}
            {uploadFiles.length > 0 && (
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <p className="text-sm font-medium">{uploadFiles.length} file(s) selected</p>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setUploadFiles([])}
                  >
                    Clear all
                  </Button>
                </div>
                <div className="max-h-40 overflow-y-auto space-y-1">
                  {uploadFiles.map((file, i) => (
                    <div key={i} className="flex items-center justify-between text-sm bg-muted rounded px-3 py-2">
                      <span className="truncate flex-1">{file.name}</span>
                      <span className="text-muted-foreground ml-2">
                        {(file.size / 1024).toFixed(0)} KB
                      </span>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-6 w-6 ml-2"
                        onClick={() => setUploadFiles((prev) => prev.filter((_, j) => j !== i))}
                      >
                        <XCircle className="h-4 w-4" />
                      </Button>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Optional folder */}
            <div className="space-y-2">
              <label className="text-sm font-medium">Folder (optional)</label>
              <Input
                placeholder="e.g., shelf-images, store-001"
                value={uploadFolder}
                onChange={(e) => setUploadFolder(e.target.value)}
              />
              <p className="text-xs text-muted-foreground">
                Organize images into folders for easier management
              </p>
            </div>
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setIsUploadDialogOpen(false)}>
              Cancel
            </Button>
            <Button
              onClick={() => uploadMutation.mutate()}
              disabled={uploadFiles.length === 0 || uploadMutation.isPending}
            >
              {uploadMutation.isPending ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Uploading...
                </>
              ) : (
                <>
                  <Upload className="h-4 w-4 mr-2" />
                  Upload {uploadFiles.length > 0 && `(${uploadFiles.length})`}
                </>
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Add to Dataset Dialog */}
      <Dialog open={isAddToDatasetOpen} onOpenChange={setIsAddToDatasetOpen}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <FolderPlus className="h-5 w-5" />
              Add to Dataset
            </DialogTitle>
            <DialogDescription>
              Add {selectAllFilteredMode ? `all ${imagesData?.total || 0} filtered` : selectedImages.size} image{(selectAllFilteredMode ? (imagesData?.total || 0) : selectedImages.size) !== 1 ? "s" : ""} to a dataset.
            </DialogDescription>
          </DialogHeader>

          <div className="py-4">
            <label className="text-sm font-medium mb-2 block">Select Dataset</label>
            {datasets && datasets.length > 0 ? (
              <ScrollArea className="h-[200px] border rounded-md">
                <div className="p-2 space-y-1">
                  {datasets.map((dataset) => (
                    <div
                      key={dataset.id}
                      onClick={() => setSelectedDatasetId(dataset.id)}
                      className={`flex items-center justify-between p-3 rounded-md cursor-pointer transition-colors ${
                        selectedDatasetId === dataset.id
                          ? "bg-primary text-primary-foreground"
                          : "hover:bg-muted"
                      }`}
                    >
                      <div>
                        <p className="font-medium">{dataset.name}</p>
                        <p className={`text-xs ${selectedDatasetId === dataset.id ? "text-primary-foreground/70" : "text-muted-foreground"}`}>
                          {dataset.image_count} images · {dataset.annotation_count} annotations
                        </p>
                      </div>
                      {selectedDatasetId === dataset.id && (
                        <CheckCircle className="h-4 w-4" />
                      )}
                    </div>
                  ))}
                </div>
              </ScrollArea>
            ) : (
              <div className="text-center py-8 text-muted-foreground">
                <FolderOpen className="h-8 w-8 mx-auto mb-2 opacity-50" />
                <p>No datasets found</p>
                <p className="text-sm">Create a dataset first</p>
              </div>
            )}
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setIsAddToDatasetOpen(false)}>
              Cancel
            </Button>
            <Button
              onClick={handleAddToDataset}
              disabled={!selectedDatasetId || addToDatasetMutation.isPending || addToDatasetByFiltersMutation.isPending}
            >
              {(addToDatasetMutation.isPending || addToDatasetByFiltersMutation.isPending) ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Adding...
                </>
              ) : (
                <>
                  <Plus className="h-4 w-4 mr-2" />
                  Add to Dataset
                </>
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Advanced Import Modal */}
      <ImportModal
        open={isImportModalOpen}
        onOpenChange={setIsImportModalOpen}
        onSuccess={() => {
          queryClient.invalidateQueries({ queryKey: ["od-images"] });
          queryClient.invalidateQueries({ queryKey: ["od-stats"] });
        }}
      />

      {/* Background Job Progress Modal - Add to Dataset */}
      <JobProgressModal
        jobId={activeJobId}
        title="Adding images to dataset"
        onClose={() => {
          setActiveJobId(null);
          clearSelection();
        }}
        onComplete={(result) => {
          toast.success(result?.message || `Added ${result?.added || 0} images to dataset`);
        }}
        invalidateOnComplete={[["od-datasets"], ["od-stats"]]}
      />

      {/* Background Job Progress Modal - Delete Images */}
      <JobProgressModal
        jobId={activeDeleteJobId}
        title="Deleting images"
        onClose={() => {
          setActiveDeleteJobId(null);
          clearSelection();
        }}
        onComplete={(result) => {
          toast.success(result?.message || `Deleted ${result?.deleted || 0} images`);
        }}
        invalidateOnComplete={[["od-images"], ["od-stats"], ["od-image-filter-options"]]}
      />
    </div>
  );
}
