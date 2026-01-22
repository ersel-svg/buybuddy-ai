"use client";

import { useState } from "react";
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
import { Textarea } from "@/components/ui/textarea";
import { Progress } from "@/components/ui/progress";
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
  RefreshCw,
  Loader2,
  FolderOpen,
  Search,
  Plus,
  MoreHorizontal,
  Edit,
  Trash2,
  ImageIcon,
  PenTool,
  ArrowRight,
  BarChart3,
  Box,
  Layers,
} from "lucide-react";
import Link from "next/link";

interface ODDataset {
  id: string;
  name: string;
  description?: string;
  annotation_type: string;
  image_count: number;
  annotated_image_count: number;
  annotation_count: number;
  created_at: string;
}

export default function ODDatasetsPage() {
  const router = useRouter();
  const queryClient = useQueryClient();
  const [search, setSearch] = useState("");

  // Dialog states
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false);
  const [isEditDialogOpen, setIsEditDialogOpen] = useState(false);
  const [editingDataset, setEditingDataset] = useState<ODDataset | null>(null);

  // Form states
  const [formName, setFormName] = useState("");
  const [formDescription, setFormDescription] = useState("");
  const [formAnnotationType, setFormAnnotationType] = useState<string>("bbox");

  // Fetch stats
  const { data: stats } = useQuery({
    queryKey: ["od-stats"],
    queryFn: () => apiClient.getODStats(),
    staleTime: 30000, // 30 seconds
    gcTime: 5 * 60 * 1000, // 5 minutes
  });

  // Fetch datasets
  const { data: datasets, isLoading, isFetching } = useQuery({
    queryKey: ["od-datasets"],
    queryFn: () => apiClient.getODDatasets(),
    staleTime: 60000, // 1 minute - datasets don't change often
    gcTime: 5 * 60 * 1000, // 5 minutes
  });

  // Create mutation
  const createMutation = useMutation({
    mutationFn: async () => {
      return apiClient.createODDataset({
        name: formName,
        description: formDescription || undefined,
        annotation_type: formAnnotationType,
      });
    },
    onSuccess: (data) => {
      toast.success(`Dataset "${formName}" created`);
      queryClient.invalidateQueries({ queryKey: ["od-datasets"] });
      queryClient.invalidateQueries({ queryKey: ["od-stats"] });
      setIsCreateDialogOpen(false);
      resetForm();
      // Navigate to new dataset
      router.push(`/od/datasets/${data.id}`);
    },
    onError: (error) => {
      toast.error(`Failed to create dataset: ${error.message}`);
    },
  });

  // Update mutation
  const updateMutation = useMutation({
    mutationFn: async () => {
      if (!editingDataset) return;
      return apiClient.updateODDataset(editingDataset.id, {
        name: formName,
        description: formDescription || undefined,
      });
    },
    onSuccess: () => {
      toast.success("Dataset updated");
      queryClient.invalidateQueries({ queryKey: ["od-datasets"] });
      setIsEditDialogOpen(false);
      setEditingDataset(null);
      resetForm();
    },
    onError: (error) => {
      toast.error(`Failed to update dataset: ${error.message}`);
    },
  });

  // Delete mutation
  const deleteMutation = useMutation({
    mutationFn: async (id: string) => {
      return apiClient.deleteODDataset(id);
    },
    onSuccess: () => {
      toast.success("Dataset deleted");
      queryClient.invalidateQueries({ queryKey: ["od-datasets"] });
      queryClient.invalidateQueries({ queryKey: ["od-stats"] });
    },
    onError: (error) => {
      toast.error(`Failed to delete dataset: ${error.message}`);
    },
  });

  const resetForm = () => {
    setFormName("");
    setFormDescription("");
    setFormAnnotationType("bbox");
  };

  const openEditDialog = (dataset: ODDataset) => {
    setEditingDataset(dataset);
    setFormName(dataset.name);
    setFormDescription(dataset.description || "");
    setFormAnnotationType(dataset.annotation_type);
    setIsEditDialogOpen(true);
  };

  const handleDelete = (dataset: ODDataset) => {
    const message = dataset.annotation_count > 0
      ? `This will delete the dataset "${dataset.name}" and all ${dataset.annotation_count} annotations. Continue?`
      : `Delete dataset "${dataset.name}"?`;
    if (!confirm(message)) return;
    deleteMutation.mutate(dataset.id);
  };

  // Filter datasets by search
  const filteredDatasets = datasets?.filter((d) =>
    d.name.toLowerCase().includes(search.toLowerCase()) ||
    d.description?.toLowerCase().includes(search.toLowerCase())
  );

  const getAnnotationTypeLabel = (type: string) => {
    switch (type) {
      case "bbox":
        return "Bounding Box";
      case "polygon":
        return "Polygon";
      case "segmentation":
        return "Segmentation";
      default:
        return type;
    }
  };

  const getAnnotationProgress = (dataset: ODDataset) => {
    if (dataset.image_count === 0) return 0;
    return Math.round((dataset.annotated_image_count / dataset.image_count) * 100);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold">Datasets</h1>
          <p className="text-muted-foreground">
            Organize images into datasets for annotation and training
          </p>
        </div>
        <div className="flex gap-2">
          <Button
            variant="outline"
            onClick={() => queryClient.invalidateQueries({ queryKey: ["od-datasets"] })}
            disabled={isFetching}
          >
            <RefreshCw className={`h-4 w-4 mr-2 ${isFetching ? "animate-spin" : ""}`} />
            Refresh
          </Button>
          <Button onClick={() => {
            resetForm();
            setIsCreateDialogOpen(true);
          }}>
            <Plus className="h-4 w-4 mr-2" />
            New Dataset
          </Button>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">Total Datasets</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">{stats?.total_datasets ?? "-"}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">Available Images</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-blue-600">{stats?.total_images ?? "-"}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">Total Annotations</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-green-600">{stats?.total_annotations ?? "-"}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">Classes</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">{stats?.total_classes ?? "-"}</p>
          </CardContent>
        </Card>
      </div>

      {/* Search */}
      <div className="flex gap-4">
        <div className="flex-1">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search datasets..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="pl-10"
            />
          </div>
        </div>
      </div>

      {/* Datasets Grid */}
      {isLoading ? (
        <div className="flex items-center justify-center py-24">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        </div>
      ) : filteredDatasets?.length === 0 ? (
        <Card className="py-12">
          <CardContent className="text-center">
            <FolderOpen className="h-16 w-16 mx-auto text-muted-foreground mb-4" />
            <h3 className="text-lg font-medium">No datasets found</h3>
            <p className="text-muted-foreground mt-1">
              {search ? "Try a different search term" : "Create your first dataset to get started"}
            </p>
            <Button className="mt-4" onClick={() => {
              resetForm();
              setIsCreateDialogOpen(true);
            }}>
              <Plus className="h-4 w-4 mr-2" />
              Create Dataset
            </Button>
          </CardContent>
        </Card>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredDatasets?.map((dataset) => (
            <Card key={dataset.id} className="hover:shadow-md transition-shadow">
              <CardHeader className="pb-3">
                <div className="flex justify-between items-start">
                  <div className="flex-1 min-w-0">
                    <CardTitle className="truncate">{dataset.name}</CardTitle>
                    {dataset.description && (
                      <CardDescription className="line-clamp-2 mt-1">
                        {dataset.description}
                      </CardDescription>
                    )}
                  </div>
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button variant="ghost" size="icon" className="flex-shrink-0">
                        <MoreHorizontal className="h-4 w-4" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                      <DropdownMenuItem onClick={() => router.push(`/od/datasets/${dataset.id}`)}>
                        <FolderOpen className="h-4 w-4 mr-2" />
                        Open Dataset
                      </DropdownMenuItem>
                      <DropdownMenuItem onClick={() => openEditDialog(dataset)}>
                        <Edit className="h-4 w-4 mr-2" />
                        Edit Details
                      </DropdownMenuItem>
                      <DropdownMenuSeparator />
                      <DropdownMenuItem
                        className="text-destructive"
                        onClick={() => handleDelete(dataset)}
                      >
                        <Trash2 className="h-4 w-4 mr-2" />
                        Delete
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>
                <Badge variant="secondary" className="w-fit mt-2">
                  <Box className="h-3 w-3 mr-1" />
                  {getAnnotationTypeLabel(dataset.annotation_type)}
                </Badge>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* Stats row */}
                <div className="grid grid-cols-3 gap-4 text-center">
                  <div>
                    <div className="flex items-center justify-center gap-1 text-muted-foreground text-xs mb-1">
                      <ImageIcon className="h-3 w-3" />
                      Images
                    </div>
                    <p className="font-bold">{dataset.image_count}</p>
                  </div>
                  <div>
                    <div className="flex items-center justify-center gap-1 text-muted-foreground text-xs mb-1">
                      <Layers className="h-3 w-3" />
                      Annotations
                    </div>
                    <p className="font-bold">{dataset.annotation_count}</p>
                  </div>
                  <div>
                    <div className="flex items-center justify-center gap-1 text-muted-foreground text-xs mb-1">
                      <BarChart3 className="h-3 w-3" />
                      Progress
                    </div>
                    <p className="font-bold">{getAnnotationProgress(dataset)}%</p>
                  </div>
                </div>

                {/* Progress bar */}
                <div className="space-y-1">
                  <Progress value={getAnnotationProgress(dataset)} className="h-2" />
                  <p className="text-xs text-muted-foreground text-center">
                    {dataset.annotated_image_count} of {dataset.image_count} images annotated
                  </p>
                </div>

                {/* Actions */}
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    className="flex-1"
                    onClick={() => router.push(`/od/datasets/${dataset.id}`)}
                  >
                    <ImageIcon className="h-4 w-4 mr-1" />
                    Manage
                  </Button>
                  <Button
                    size="sm"
                    className="flex-1"
                    onClick={() => {
                      if (dataset.image_count === 0) {
                        toast.error("Add images to the dataset first");
                        return;
                      }
                      router.push(`/od/annotate?dataset=${dataset.id}`);
                    }}
                    disabled={dataset.image_count === 0}
                  >
                    <PenTool className="h-4 w-4 mr-1" />
                    Annotate
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Create Dialog */}
      <Dialog open={isCreateDialogOpen} onOpenChange={setIsCreateDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Plus className="h-5 w-5" />
              New Dataset
            </DialogTitle>
            <DialogDescription>
              Create a new dataset to organize images for annotation.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label>Name *</Label>
              <Input
                placeholder="e.g., Shelf Detection v1"
                value={formName}
                onChange={(e) => setFormName(e.target.value)}
              />
            </div>

            <div className="space-y-2">
              <Label>Description</Label>
              <Textarea
                placeholder="Describe what this dataset is for..."
                value={formDescription}
                onChange={(e) => setFormDescription(e.target.value)}
                rows={3}
              />
            </div>

            <div className="space-y-2">
              <Label>Annotation Type</Label>
              <Select value={formAnnotationType} onValueChange={setFormAnnotationType}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="bbox">
                    <div className="flex items-center gap-2">
                      <Box className="h-4 w-4" />
                      Bounding Box
                    </div>
                  </SelectItem>
                  <SelectItem value="polygon">
                    <div className="flex items-center gap-2">
                      <Layers className="h-4 w-4" />
                      Polygon
                    </div>
                  </SelectItem>
                  <SelectItem value="segmentation">
                    <div className="flex items-center gap-2">
                      <PenTool className="h-4 w-4" />
                      Segmentation
                    </div>
                  </SelectItem>
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground">
                Choose the annotation format for this dataset. Bounding boxes are recommended for object detection.
              </p>
            </div>
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setIsCreateDialogOpen(false)}>
              Cancel
            </Button>
            <Button
              onClick={() => createMutation.mutate()}
              disabled={!formName || createMutation.isPending}
            >
              {createMutation.isPending ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Creating...
                </>
              ) : (
                <>
                  Create Dataset
                  <ArrowRight className="h-4 w-4 ml-2" />
                </>
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Edit Dialog */}
      <Dialog open={isEditDialogOpen} onOpenChange={setIsEditDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Edit className="h-5 w-5" />
              Edit Dataset
            </DialogTitle>
            <DialogDescription>
              Update the dataset details.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label>Name *</Label>
              <Input
                placeholder="e.g., Shelf Detection v1"
                value={formName}
                onChange={(e) => setFormName(e.target.value)}
              />
            </div>

            <div className="space-y-2">
              <Label>Description</Label>
              <Textarea
                placeholder="Describe what this dataset is for..."
                value={formDescription}
                onChange={(e) => setFormDescription(e.target.value)}
                rows={3}
              />
            </div>

            <div className="space-y-2">
              <Label>Annotation Type</Label>
              <Select value={formAnnotationType} disabled>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="bbox">Bounding Box</SelectItem>
                  <SelectItem value="polygon">Polygon</SelectItem>
                  <SelectItem value="segmentation">Segmentation</SelectItem>
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground">
                Annotation type cannot be changed after creation.
              </p>
            </div>
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setIsEditDialogOpen(false)}>
              Cancel
            </Button>
            <Button
              onClick={() => updateMutation.mutate()}
              disabled={!formName || updateMutation.isPending}
            >
              {updateMutation.isPending ? (
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
    </div>
  );
}
