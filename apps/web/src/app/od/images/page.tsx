"use client";

import { useState, useCallback } from "react";
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
} from "lucide-react";
import Image from "next/image";
import { useDropzone } from "react-dropzone";

const PAGE_SIZE = 48;

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
  const [search, setSearch] = useState("");
  const [filterStatus, setFilterStatus] = useState<string>("all");
  const [filterSource, setFilterSource] = useState<string>("all");
  const [filterFolder, setFilterFolder] = useState<string>("all");
  const [viewMode, setViewMode] = useState<"grid" | "list">("grid");

  // Selection state
  const [selectedImages, setSelectedImages] = useState<Set<string>>(new Set());

  // Upload dialog state
  const [isUploadDialogOpen, setIsUploadDialogOpen] = useState(false);
  const [uploadFiles, setUploadFiles] = useState<File[]>([]);
  const [uploadFolder, setUploadFolder] = useState("");

  // Fetch OD stats
  const { data: stats } = useQuery({
    queryKey: ["od-stats"],
    queryFn: () => apiClient.getODStats(),
  });

  // Fetch images
  const {
    data: imagesData,
    isLoading,
    isFetching,
  } = useQuery({
    queryKey: ["od-images", page, search, filterStatus, filterSource, filterFolder],
    queryFn: () =>
      apiClient.getODImages({
        page,
        limit: PAGE_SIZE,
        search: search || undefined,
        status: filterStatus !== "all" ? filterStatus : undefined,
        source: filterSource !== "all" ? filterSource : undefined,
        folder: filterFolder !== "all" ? filterFolder : undefined,
      }),
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
          <Button onClick={() => setIsUploadDialogOpen(true)}>
            <Upload className="h-4 w-4 mr-2" />
            Upload Images
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
              {selectedImages.size > 0 && (
                <>
                  <span className="text-sm text-muted-foreground">
                    {selectedImages.size} selected
                  </span>
                  <Button
                    variant="destructive"
                    size="sm"
                    onClick={handleDeleteSelected}
                    disabled={deleteMutation.isPending}
                  >
                    <Trash2 className="h-4 w-4 mr-1" />
                    Delete
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
          <div className="flex gap-4">
            <div className="flex-1">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search by filename..."
                  value={search}
                  onChange={(e) => {
                    setSearch(e.target.value);
                    setPage(1);
                  }}
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
            <Select
              value={filterSource}
              onValueChange={(v) => {
                setFilterSource(v);
                setPage(1);
              }}
            >
              <SelectTrigger className="w-[150px]">
                <SelectValue placeholder="Source" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Sources</SelectItem>
                <SelectItem value="upload">Upload</SelectItem>
                <SelectItem value="buybuddy_sync">BuyBuddy Sync</SelectItem>
                <SelectItem value="import">Import</SelectItem>
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
                {search || filterStatus !== "all" || filterSource !== "all"
                  ? "Try adjusting your filters"
                  : "Upload images to get started"}
              </p>
              <Button className="mt-4" onClick={() => setIsUploadDialogOpen(true)}>
                <Upload className="h-4 w-4 mr-2" />
                Upload Images
              </Button>
            </div>
          ) : viewMode === "grid" ? (
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-4">
              {imagesData?.images.map((image: ODImage) => (
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
                    unoptimized
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
                      unoptimized
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
    </div>
  );
}
