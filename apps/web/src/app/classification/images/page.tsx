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
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";
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
  Link as LinkIcon,
  Tags,
} from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import Image from "next/image";
import { useDropzone } from "react-dropzone";

const PAGE_SIZE = 48;

// Custom hook for debounced value
function useDebounce<T>(value: T, delay: number): T {
  const [debouncedValue, setDebouncedValue] = useState<T>(value);

  useEffect(() => {
    const timer = setTimeout(() => setDebouncedValue(value), delay);
    return () => clearTimeout(timer);
  }, [value, delay]);

  return debouncedValue;
}

interface CLSImage {
  id: string;
  filename: string;
  image_url: string;
  thumbnail_url?: string;
  width?: number;
  height?: number;
  status: string;
  source: string;
  folder?: string;
  tags?: string[];
  created_at: string;
}

export default function CLSImagesPage() {
  const queryClient = useQueryClient();
  const [page, setPage] = useState(1);
  const [searchInput, setSearchInput] = useState("");
  const debouncedSearch = useDebounce(searchInput, 300);
  const [viewMode, setViewMode] = useState<"grid" | "list">("grid");
  const [statusFilter, setStatusFilter] = useState<string>("all");
  const [sourceFilter, setSourceFilter] = useState<string>("all");

  // Selection state
  const [selectedImages, setSelectedImages] = useState<Set<string>>(new Set());

  // Upload dialog state
  const [isUploadDialogOpen, setIsUploadDialogOpen] = useState(false);
  const [uploadFiles, setUploadFiles] = useState<File[]>([]);
  const [uploadFolder, setUploadFolder] = useState("");

  // Import modal state
  const [isImportModalOpen, setIsImportModalOpen] = useState(false);
  const [importTab, setImportTab] = useState<"url" | "products" | "cutouts" | "od">("url");
  const [importUrls, setImportUrls] = useState("");

  // Add to dataset dialog state
  const [isAddToDatasetOpen, setIsAddToDatasetOpen] = useState(false);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>("");

  // Fetch CLS stats
  const { data: stats } = useQuery({
    queryKey: ["cls-stats"],
    queryFn: () => apiClient.getCLSStats(),
    staleTime: 30000,
  });

  // Fetch datasets for "Add to Dataset" dialog
  const { data: datasets } = useQuery({
    queryKey: ["cls-datasets"],
    queryFn: () => apiClient.getCLSDatasets(),
    staleTime: 60000,
  });

  // Fetch images
  const {
    data: imagesData,
    isLoading,
    isFetching,
  } = useQuery({
    queryKey: ["cls-images", page, debouncedSearch, statusFilter, sourceFilter],
    queryFn: () =>
      apiClient.getCLSImages({
        page,
        limit: PAGE_SIZE,
        search: debouncedSearch || undefined,
        status: statusFilter !== "all" ? statusFilter : undefined,
        source: sourceFilter !== "all" ? sourceFilter : undefined,
      }),
    staleTime: 30000,
  });

  // Upload mutation
  const uploadMutation = useMutation({
    mutationFn: async () => {
      if (uploadFiles.length === 1) {
        return apiClient.uploadCLSImage(uploadFiles[0], uploadFolder || undefined);
      } else {
        return apiClient.uploadCLSImagesBulk(uploadFiles);
      }
    },
    onSuccess: (data) => {
      const count = Array.isArray(data) ? data.length : 1;
      toast.success(`Uploaded ${count} image${count > 1 ? "s" : ""}`);
      queryClient.invalidateQueries({ queryKey: ["cls-images"] });
      queryClient.invalidateQueries({ queryKey: ["cls-stats"] });
      setIsUploadDialogOpen(false);
      setUploadFiles([]);
      setUploadFolder("");
    },
    onError: (error) => {
      toast.error(`Upload failed: ${error.message}`);
    },
  });

  // Import from URLs mutation
  const importUrlsMutation = useMutation({
    mutationFn: async () => {
      const urls = importUrls.split("\n").map(u => u.trim()).filter(Boolean);
      return apiClient.importCLSImagesFromURLs({ urls, skip_duplicates: true });
    },
    onSuccess: (result) => {
      toast.success(`Imported ${result.images_imported} images${result.images_skipped > 0 ? ` (${result.images_skipped} skipped)` : ""}`);
      queryClient.invalidateQueries({ queryKey: ["cls-images"] });
      queryClient.invalidateQueries({ queryKey: ["cls-stats"] });
      setIsImportModalOpen(false);
      setImportUrls("");
    },
    onError: (error) => {
      toast.error(`Import failed: ${error.message}`);
    },
  });

  // Import from Products mutation
  const importProductsMutation = useMutation({
    mutationFn: async () => {
      return apiClient.importCLSImagesFromProducts({
        label_source: "category",
        skip_duplicates: true,
      });
    },
    onSuccess: (result) => {
      toast.success(`Imported ${result.images_imported} images, ${result.labels_created} labels, ${result.classes_created} classes`);
      queryClient.invalidateQueries({ queryKey: ["cls-images"] });
      queryClient.invalidateQueries({ queryKey: ["cls-stats"] });
      queryClient.invalidateQueries({ queryKey: ["cls-classes"] });
      setIsImportModalOpen(false);
    },
    onError: (error) => {
      toast.error(`Import failed: ${error.message}`);
    },
  });

  // Import from Cutouts mutation
  const importCutoutsMutation = useMutation({
    mutationFn: async () => {
      return apiClient.importCLSImagesFromCutouts({
        label_source: "matched_product_category",
        only_matched: true,
        skip_duplicates: true,
      });
    },
    onSuccess: (result) => {
      toast.success(`Imported ${result.images_imported} images, ${result.labels_created} labels`);
      queryClient.invalidateQueries({ queryKey: ["cls-images"] });
      queryClient.invalidateQueries({ queryKey: ["cls-stats"] });
      setIsImportModalOpen(false);
    },
    onError: (error) => {
      toast.error(`Import failed: ${error.message}`);
    },
  });

  // Import from OD mutation
  const importODMutation = useMutation({
    mutationFn: async () => {
      return apiClient.importCLSImagesFromOD({ skip_duplicates: true });
    },
    onSuccess: (result) => {
      toast.success(`Imported ${result.images_imported} images${result.images_skipped > 0 ? ` (${result.images_skipped} skipped)` : ""}`);
      queryClient.invalidateQueries({ queryKey: ["cls-images"] });
      queryClient.invalidateQueries({ queryKey: ["cls-stats"] });
      setIsImportModalOpen(false);
    },
    onError: (error) => {
      toast.error(`Import failed: ${error.message}`);
    },
  });

  // Delete mutation
  const deleteMutation = useMutation({
    mutationFn: async (imageIds: string[]) => {
      if (imageIds.length === 1) {
        return apiClient.deleteCLSImage(imageIds[0]);
      } else {
        return apiClient.deleteCLSImagesBulk(imageIds);
      }
    },
    onSuccess: (_, imageIds) => {
      toast.success(`Deleted ${imageIds.length} image${imageIds.length > 1 ? "s" : ""}`);
      queryClient.invalidateQueries({ queryKey: ["cls-images"] });
      queryClient.invalidateQueries({ queryKey: ["cls-stats"] });
      setSelectedImages(new Set());
    },
    onError: (error) => {
      toast.error(`Delete failed: ${error.message}`);
    },
  });

  // Add to dataset mutation
  const addToDatasetMutation = useMutation({
    mutationFn: async ({ datasetId, imageIds }: { datasetId: string; imageIds: string[] }) => {
      return apiClient.addImagesToCLSDataset(datasetId, imageIds);
    },
    onSuccess: (result) => {
      toast.success(`Added ${result.added} image${result.added !== 1 ? "s" : ""} to dataset${result.skipped > 0 ? ` (${result.skipped} already in dataset)` : ""}`);
      queryClient.invalidateQueries({ queryKey: ["cls-datasets"] });
      queryClient.invalidateQueries({ queryKey: ["cls-stats"] });
      setIsAddToDatasetOpen(false);
      setSelectedDatasetId("");
      setSelectedImages(new Set());
    },
    onError: (error) => {
      toast.error(`Failed to add images: ${error.message}`);
    },
  });

  const handleAddToDataset = () => {
    if (!selectedDatasetId || selectedImages.size === 0) return;
    addToDatasetMutation.mutate({
      datasetId: selectedDatasetId,
      imageIds: Array.from(selectedImages),
    });
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
      setSelectedImages(new Set(imagesData.images.map((img) => img.id)));
    }
  };

  const handleDeleteSelected = () => {
    if (selectedImages.size === 0) return;
    if (!confirm(`Delete ${selectedImages.size} selected images?`)) return;
    deleteMutation.mutate(Array.from(selectedImages));
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
      case "labeled":
        return (
          <Badge variant="default" className="bg-blue-600">
            <Tags className="h-3 w-3 mr-1" />
            Labeled
          </Badge>
        );
      case "review":
        return (
          <Badge variant="default" className="bg-yellow-600">
            <Clock className="h-3 w-3 mr-1" />
            Review
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
            Upload and manage images for classification
          </p>
        </div>
        <div className="flex gap-2">
          <Button
            variant="outline"
            onClick={() => {
              queryClient.invalidateQueries({ queryKey: ["cls-images"] });
              queryClient.invalidateQueries({ queryKey: ["cls-stats"] });
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
            <CardTitle className="text-sm text-muted-foreground">Labeled</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-blue-600">
              {stats?.images_by_status?.labeled?.toLocaleString() ?? "-"}
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
              {selectedImages.size > 0 && (
                <>
                  <span className="text-sm text-muted-foreground">
                    {selectedImages.size} selected
                  </span>
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
                    disabled={deleteMutation.isPending}
                  >
                    {deleteMutation.isPending ? (
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

            <Select value={statusFilter} onValueChange={(v) => { setStatusFilter(v); setPage(1); }}>
              <SelectTrigger className="w-[150px]">
                <SelectValue placeholder="Status" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Status</SelectItem>
                <SelectItem value="pending">Pending</SelectItem>
                <SelectItem value="labeled">Labeled</SelectItem>
                <SelectItem value="review">Review</SelectItem>
                <SelectItem value="completed">Completed</SelectItem>
                <SelectItem value="skipped">Skipped</SelectItem>
              </SelectContent>
            </Select>

            <Select value={sourceFilter} onValueChange={(v) => { setSourceFilter(v); setPage(1); }}>
              <SelectTrigger className="w-[150px]">
                <SelectValue placeholder="Source" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Sources</SelectItem>
                <SelectItem value="upload">Upload</SelectItem>
                <SelectItem value="url">URL Import</SelectItem>
                <SelectItem value="products">Products</SelectItem>
                <SelectItem value="cutouts">Cutouts</SelectItem>
                <SelectItem value="od">Object Detection</SelectItem>
              </SelectContent>
            </Select>

            {imagesData?.images && imagesData.images.length > 0 && (
              <div className="flex items-center gap-2 ml-auto">
                <Checkbox
                  checked={selectedImages.size === imagesData.images.length && imagesData.images.length > 0}
                  onCheckedChange={toggleSelectAll}
                />
                <span className="text-sm text-muted-foreground">Select all</span>
              </div>
            )}
          </div>

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
                {debouncedSearch || statusFilter !== "all" || sourceFilter !== "all"
                  ? "Try adjusting your filters"
                  : "Upload images or import from URL, Products, Cutouts, or OD"}
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
              {imagesData?.images.map((image: CLSImage) => (
                <div
                  key={image.id}
                  className={`group relative aspect-square rounded-lg overflow-hidden bg-muted border-2 transition-colors cursor-pointer ${
                    selectedImages.has(image.id) ? "border-primary" : "border-transparent hover:border-muted-foreground/30"
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
                  <div className={`absolute top-2 left-2 transition-opacity ${selectedImages.has(image.id) ? "opacity-100" : "opacity-0 group-hover:opacity-100"}`}>
                    <Checkbox
                      checked={selectedImages.has(image.id)}
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
                    {image.width && image.height && (
                      <p className="text-white/70 text-xs">
                        {image.width}x{image.height}
                      </p>
                    )}
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
              {imagesData?.images.map((image: CLSImage) => (
                <div
                  key={image.id}
                  className={`flex items-center gap-4 p-3 rounded-lg border cursor-pointer transition-colors ${
                    selectedImages.has(image.id) ? "border-primary bg-primary/5" : "hover:bg-muted/50"
                  }`}
                  onClick={() => toggleImageSelection(image.id)}
                >
                  <Checkbox
                    checked={selectedImages.has(image.id)}
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
                      {image.width && image.height && `${image.width}x${image.height} - `}{image.source}
                      {image.folder && <> - <FolderOpen className="h-3 w-3 inline mx-1" />{image.folder}</>}
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
                placeholder="e.g., products, brands"
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

      {/* Import Modal */}
      <Dialog open={isImportModalOpen} onOpenChange={setIsImportModalOpen}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Package className="h-5 w-5" />
              Import Images
            </DialogTitle>
            <DialogDescription>
              Import images from URLs or existing data sources
            </DialogDescription>
          </DialogHeader>

          <Tabs value={importTab} onValueChange={(v) => setImportTab(v as typeof importTab)}>
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="url">
                <LinkIcon className="h-4 w-4 mr-2" />
                URLs
              </TabsTrigger>
              <TabsTrigger value="products">
                <Package className="h-4 w-4 mr-2" />
                Products
              </TabsTrigger>
              <TabsTrigger value="cutouts">
                <ImageIcon className="h-4 w-4 mr-2" />
                Cutouts
              </TabsTrigger>
              <TabsTrigger value="od">
                <ImageIcon className="h-4 w-4 mr-2" />
                OD Images
              </TabsTrigger>
            </TabsList>

            <TabsContent value="url" className="space-y-4 pt-4">
              <div className="space-y-2">
                <label className="text-sm font-medium">Image URLs (one per line)</label>
                <textarea
                  className="w-full h-40 p-3 border rounded-md text-sm"
                  placeholder="https://example.com/image1.jpg&#10;https://example.com/image2.jpg"
                  value={importUrls}
                  onChange={(e) => setImportUrls(e.target.value)}
                />
              </div>
              <Button
                onClick={() => importUrlsMutation.mutate()}
                disabled={!importUrls.trim() || importUrlsMutation.isPending}
                className="w-full"
              >
                {importUrlsMutation.isPending ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Importing...
                  </>
                ) : (
                  <>
                    <Upload className="h-4 w-4 mr-2" />
                    Import from URLs
                  </>
                )}
              </Button>
            </TabsContent>

            <TabsContent value="products" className="space-y-4 pt-4">
              <div className="text-sm text-muted-foreground">
                <p>Import images from the Products module.</p>
                <ul className="list-disc list-inside mt-2 space-y-1">
                  <li>Synthetic and real product images will be imported</li>
                  <li>Classes will be auto-created based on product categories</li>
                  <li>Labels will be auto-assigned to imported images</li>
                </ul>
              </div>
              <Button
                onClick={() => importProductsMutation.mutate()}
                disabled={importProductsMutation.isPending}
                className="w-full"
              >
                {importProductsMutation.isPending ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Importing...
                  </>
                ) : (
                  <>
                    <Package className="h-4 w-4 mr-2" />
                    Import from Products
                  </>
                )}
              </Button>
            </TabsContent>

            <TabsContent value="cutouts" className="space-y-4 pt-4">
              <div className="text-sm text-muted-foreground">
                <p>Import images from the Cutouts module.</p>
                <ul className="list-disc list-inside mt-2 space-y-1">
                  <li>Only matched cutouts will be imported</li>
                  <li>Classes will be auto-created based on matched product categories</li>
                  <li>Labels will be auto-assigned to imported images</li>
                </ul>
              </div>
              <Button
                onClick={() => importCutoutsMutation.mutate()}
                disabled={importCutoutsMutation.isPending}
                className="w-full"
              >
                {importCutoutsMutation.isPending ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Importing...
                  </>
                ) : (
                  <>
                    <ImageIcon className="h-4 w-4 mr-2" />
                    Import from Cutouts
                  </>
                )}
              </Button>
            </TabsContent>

            <TabsContent value="od" className="space-y-4 pt-4">
              <div className="text-sm text-muted-foreground">
                <p>Import images from the Object Detection module.</p>
                <ul className="list-disc list-inside mt-2 space-y-1">
                  <li>Images will be copied to the classification library</li>
                  <li>Duplicates will be skipped automatically</li>
                  <li>No labels will be assigned (manual labeling required)</li>
                </ul>
              </div>
              <Button
                onClick={() => importODMutation.mutate()}
                disabled={importODMutation.isPending}
                className="w-full"
              >
                {importODMutation.isPending ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Importing...
                  </>
                ) : (
                  <>
                    <ImageIcon className="h-4 w-4 mr-2" />
                    Import from OD Images
                  </>
                )}
              </Button>
            </TabsContent>
          </Tabs>
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
              Add {selectedImages.size} selected image{selectedImages.size !== 1 ? "s" : ""} to a dataset.
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
                          {dataset.image_count} images - {dataset.labeled_image_count} labeled
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
              disabled={!selectedDatasetId || addToDatasetMutation.isPending}
            >
              {addToDatasetMutation.isPending ? (
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
    </div>
  );
}
