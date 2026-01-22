"use client";

import { useState } from "react";
import Link from "next/link";
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
import { Textarea } from "@/components/ui/textarea";
import { Progress } from "@/components/ui/progress";
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
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  RefreshCw,
  Loader2,
  FolderOpen,
  Search,
  Plus,
  Trash2,
  MoreHorizontal,
  Edit,
  ImageIcon,
  Tags,
  PenTool,
  ArrowRight,
  BarChart3,
} from "lucide-react";

interface CLSDataset {
  id: string;
  name: string;
  description?: string;
  task_type: string;
  image_count: number;
  labeled_image_count: number;
  class_count: number;
  version: number;
  created_at: string;
}

export default function CLSDatasetsPage() {
  const queryClient = useQueryClient();
  const [searchQuery, setSearchQuery] = useState("");

  // Create dialog state
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false);
  const [newDatasetName, setNewDatasetName] = useState("");
  const [newDatasetDescription, setNewDatasetDescription] = useState("");
  const [newDatasetTaskType, setNewDatasetTaskType] = useState<"single_label" | "multi_label">("single_label");

  // Fetch datasets
  const { data: datasets, isLoading, isFetching } = useQuery({
    queryKey: ["cls-datasets"],
    queryFn: () => apiClient.getCLSDatasets(),
    staleTime: 30000,
  });

  // Fetch stats
  const { data: stats } = useQuery({
    queryKey: ["cls-stats"],
    queryFn: () => apiClient.getCLSStats(),
    staleTime: 30000,
  });

  // Create mutation
  const createMutation = useMutation({
    mutationFn: async () => {
      return apiClient.createCLSDataset({
        name: newDatasetName,
        description: newDatasetDescription || undefined,
        task_type: newDatasetTaskType,
      });
    },
    onSuccess: () => {
      toast.success(`Dataset "${newDatasetName}" created`);
      queryClient.invalidateQueries({ queryKey: ["cls-datasets"] });
      queryClient.invalidateQueries({ queryKey: ["cls-stats"] });
      setIsCreateDialogOpen(false);
      resetCreateForm();
    },
    onError: (error) => {
      toast.error(`Failed to create dataset: ${error.message}`);
    },
  });

  // Delete mutation
  const deleteMutation = useMutation({
    mutationFn: async (id: string) => {
      return apiClient.deleteCLSDataset(id);
    },
    onSuccess: () => {
      toast.success("Dataset deleted");
      queryClient.invalidateQueries({ queryKey: ["cls-datasets"] });
      queryClient.invalidateQueries({ queryKey: ["cls-stats"] });
    },
    onError: (error) => {
      toast.error(`Failed to delete dataset: ${error.message}`);
    },
  });

  const resetCreateForm = () => {
    setNewDatasetName("");
    setNewDatasetDescription("");
    setNewDatasetTaskType("single_label");
  };

  const handleDelete = (dataset: CLSDataset) => {
    if (dataset.image_count > 0) {
      if (!confirm(`This dataset has ${dataset.image_count} images. Are you sure you want to delete it?`)) {
        return;
      }
    } else {
      if (!confirm(`Delete dataset "${dataset.name}"?`)) {
        return;
      }
    }
    deleteMutation.mutate(dataset.id);
  };

  const filteredDatasets = datasets?.filter(
    (ds) => ds.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold">Datasets</h1>
          <p className="text-muted-foreground">
            Organize images into datasets for labeling and training
          </p>
        </div>
        <div className="flex gap-2">
          <Button
            variant="outline"
            onClick={() => queryClient.invalidateQueries({ queryKey: ["cls-datasets"] })}
            disabled={isFetching}
          >
            <RefreshCw className={`h-4 w-4 mr-2 ${isFetching ? "animate-spin" : ""}`} />
            Refresh
          </Button>
          <Button onClick={() => setIsCreateDialogOpen(true)}>
            <Plus className="h-4 w-4 mr-2" />
            New Dataset
          </Button>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">Total Datasets</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">{stats?.total_datasets ?? 0}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">Total Images</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-blue-600">
              {stats?.total_images?.toLocaleString() ?? 0}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">Total Labels</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-green-600">
              {stats?.total_labels?.toLocaleString() ?? 0}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">Active Training</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-orange-600">
              {stats?.active_training_runs ?? 0}
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Search */}
      <div className="flex gap-4 items-center">
        <div className="flex-1 max-w-md">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search datasets..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
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
        <div className="text-center py-24">
          <FolderOpen className="h-16 w-16 mx-auto text-muted-foreground mb-4" />
          <h3 className="text-lg font-medium">No datasets found</h3>
          <p className="text-muted-foreground mt-1">
            {searchQuery ? "Try a different search term" : "Create your first dataset to get started"}
          </p>
          {!searchQuery && (
            <Button className="mt-4" onClick={() => setIsCreateDialogOpen(true)}>
              <Plus className="h-4 w-4 mr-2" />
              Create Dataset
            </Button>
          )}
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredDatasets?.map((dataset) => {
            const progress = dataset.image_count > 0
              ? Math.round((dataset.labeled_image_count / dataset.image_count) * 100)
              : 0;

            return (
              <Card key={dataset.id} className="hover:border-primary/50 transition-colors">
                <CardHeader className="pb-2">
                  <div className="flex items-start justify-between">
                    <div>
                      <CardTitle className="text-lg">{dataset.name}</CardTitle>
                      <CardDescription className="line-clamp-2">
                        {dataset.description || "No description"}
                      </CardDescription>
                    </div>
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button variant="ghost" size="icon" className="h-8 w-8">
                          <MoreHorizontal className="h-4 w-4" />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end">
                        <DropdownMenuItem asChild>
                          <Link href={`/classification/datasets/${dataset.id}`}>
                            <FolderOpen className="h-4 w-4 mr-2" />
                            View Details
                          </Link>
                        </DropdownMenuItem>
                        <DropdownMenuItem asChild>
                          <Link href={`/classification/labeling?dataset=${dataset.id}`}>
                            <PenTool className="h-4 w-4 mr-2" />
                            Start Labeling
                          </Link>
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
                </CardHeader>
                <CardContent className="space-y-4">
                  {/* Task Type Badge */}
                  <Badge variant="outline">
                    {dataset.task_type === "single_label" ? "Single Label" : "Multi Label"}
                  </Badge>

                  {/* Progress */}
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Labeling Progress</span>
                      <span className="font-medium">{progress}%</span>
                    </div>
                    <Progress value={progress} className="h-2" />
                  </div>

                  {/* Stats */}
                  <div className="grid grid-cols-3 gap-2 text-center">
                    <div className="p-2 rounded-lg bg-muted">
                      <div className="flex items-center justify-center gap-1 text-muted-foreground mb-1">
                        <ImageIcon className="h-3 w-3" />
                      </div>
                      <p className="text-lg font-bold">{dataset.image_count}</p>
                      <p className="text-xs text-muted-foreground">Images</p>
                    </div>
                    <div className="p-2 rounded-lg bg-muted">
                      <div className="flex items-center justify-center gap-1 text-muted-foreground mb-1">
                        <PenTool className="h-3 w-3" />
                      </div>
                      <p className="text-lg font-bold">{dataset.labeled_image_count}</p>
                      <p className="text-xs text-muted-foreground">Labeled</p>
                    </div>
                    <div className="p-2 rounded-lg bg-muted">
                      <div className="flex items-center justify-center gap-1 text-muted-foreground mb-1">
                        <Tags className="h-3 w-3" />
                      </div>
                      <p className="text-lg font-bold">{dataset.class_count}</p>
                      <p className="text-xs text-muted-foreground">Classes</p>
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="flex gap-2">
                    <Button variant="outline" size="sm" className="flex-1" asChild>
                      <Link href={`/classification/datasets/${dataset.id}`}>
                        <BarChart3 className="h-4 w-4 mr-2" />
                        Details
                      </Link>
                    </Button>
                    <Button size="sm" className="flex-1" asChild>
                      <Link href={`/classification/labeling?dataset=${dataset.id}`}>
                        <PenTool className="h-4 w-4 mr-2" />
                        Label
                      </Link>
                    </Button>
                  </div>

                  {/* Version & Date */}
                  <div className="flex justify-between text-xs text-muted-foreground pt-2 border-t">
                    <span>v{dataset.version}</span>
                    <span>{new Date(dataset.created_at).toLocaleDateString()}</span>
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </div>
      )}

      {/* Create Dialog */}
      <Dialog open={isCreateDialogOpen} onOpenChange={setIsCreateDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Create New Dataset</DialogTitle>
            <DialogDescription>
              Create a new dataset to organize images for labeling.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Name *</label>
              <Input
                placeholder="e.g., Product Categories v1"
                value={newDatasetName}
                onChange={(e) => setNewDatasetName(e.target.value)}
              />
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Description</label>
              <Textarea
                placeholder="Optional description..."
                value={newDatasetDescription}
                onChange={(e) => setNewDatasetDescription(e.target.value)}
              />
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Task Type</label>
              <Select value={newDatasetTaskType} onValueChange={(v) => setNewDatasetTaskType(v as typeof newDatasetTaskType)}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="single_label">
                    <div>
                      <p className="font-medium">Single Label</p>
                      <p className="text-xs text-muted-foreground">Each image has exactly one class</p>
                    </div>
                  </SelectItem>
                  <SelectItem value="multi_label">
                    <div>
                      <p className="font-medium">Multi Label</p>
                      <p className="text-xs text-muted-foreground">Each image can have multiple classes</p>
                    </div>
                  </SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setIsCreateDialogOpen(false)}>
              Cancel
            </Button>
            <Button
              onClick={() => createMutation.mutate()}
              disabled={!newDatasetName.trim() || createMutation.isPending}
            >
              {createMutation.isPending ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Creating...
                </>
              ) : (
                "Create Dataset"
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
