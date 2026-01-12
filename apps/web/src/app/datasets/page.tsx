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
import { Textarea } from "@/components/ui/textarea";
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
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
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
import {
  Plus,
  FolderOpen,
  MoreHorizontal,
  Trash2,
  Loader2,
  Database,
  RefreshCw,
  Sparkles,
  Brain,
  Layers,
} from "lucide-react";
import type { Dataset } from "@/types";

export default function DatasetsPage() {
  const queryClient = useQueryClient();
  const [isCreateOpen, setIsCreateOpen] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [datasetToDelete, setDatasetToDelete] = useState<Dataset | null>(null);
  const [newDataset, setNewDataset] = useState({ name: "", description: "" });

  // Fetch datasets
  const {
    data: datasets,
    isLoading,
    refetch,
    isFetching,
  } = useQuery({
    queryKey: ["datasets"],
    queryFn: () => apiClient.getDatasets(),
  });

  // Create mutation
  const createMutation = useMutation({
    mutationFn: (data: { name: string; description?: string }) =>
      apiClient.createDataset(data),
    onSuccess: () => {
      toast.success("Dataset created");
      queryClient.invalidateQueries({ queryKey: ["datasets"] });
      setIsCreateOpen(false);
      setNewDataset({ name: "", description: "" });
    },
    onError: () => {
      toast.error("Failed to create dataset");
    },
  });

  // Delete mutation
  const deleteMutation = useMutation({
    mutationFn: (id: string) => apiClient.deleteDataset(id),
    onSuccess: () => {
      toast.success("Dataset deleted");
      queryClient.invalidateQueries({ queryKey: ["datasets"] });
      setDeleteDialogOpen(false);
      setDatasetToDelete(null);
    },
    onError: () => {
      toast.error("Failed to delete dataset");
    },
  });

  const handleDelete = (dataset: Dataset) => {
    setDatasetToDelete(dataset);
    setDeleteDialogOpen(true);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold">Datasets</h1>
          <p className="text-gray-500">
            Manage product datasets for training and augmentation
          </p>
        </div>
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="icon"
            onClick={() => refetch()}
            disabled={isFetching}
          >
            <RefreshCw
              className={`h-4 w-4 ${isFetching ? "animate-spin" : ""}`}
            />
          </Button>
          <Dialog open={isCreateOpen} onOpenChange={setIsCreateOpen}>
            <DialogTrigger asChild>
              <Button>
                <Plus className="h-4 w-4 mr-2" />
                New Dataset
              </Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Create New Dataset</DialogTitle>
                <DialogDescription>
                  Create a dataset to organize products for training and
                  augmentation.
                </DialogDescription>
              </DialogHeader>
              <div className="space-y-4 py-4">
                <div className="space-y-2">
                  <Label htmlFor="name">Name</Label>
                  <Input
                    id="name"
                    value={newDataset.name}
                    onChange={(e) =>
                      setNewDataset({ ...newDataset, name: e.target.value })
                    }
                    placeholder="e.g., Beverages v1"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="description">Description (optional)</Label>
                  <Textarea
                    id="description"
                    value={newDataset.description}
                    onChange={(e) =>
                      setNewDataset({
                        ...newDataset,
                        description: e.target.value,
                      })
                    }
                    placeholder="What products are in this dataset?"
                    rows={3}
                  />
                </div>
              </div>
              <DialogFooter>
                <Button
                  variant="outline"
                  onClick={() => setIsCreateOpen(false)}
                >
                  Cancel
                </Button>
                <Button
                  onClick={() => createMutation.mutate(newDataset)}
                  disabled={!newDataset.name.trim() || createMutation.isPending}
                >
                  {createMutation.isPending ? (
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  ) : null}
                  Create
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
        </div>
      </div>

      {/* Content */}
      {isLoading ? (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="h-8 w-8 animate-spin" />
        </div>
      ) : datasets?.length === 0 ? (
        <Card className="py-12">
          <CardContent className="text-center">
            <FolderOpen className="h-12 w-12 mx-auto text-gray-300" />
            <h3 className="mt-4 text-lg font-medium">No datasets yet</h3>
            <p className="text-gray-500 mt-1">
              Create your first dataset to organize products
            </p>
            <Button className="mt-4" onClick={() => setIsCreateOpen(true)}>
              <Plus className="h-4 w-4 mr-2" />
              Create Dataset
            </Button>
          </CardContent>
        </Card>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {datasets?.map((dataset) => (
            <Card
              key={dataset.id}
              className="hover:border-slate-400 transition-colors group"
            >
              <CardHeader className="flex flex-row items-start justify-between">
                <div className="flex-1 min-w-0">
                  <CardTitle className="truncate">
                    <Link
                      href={`/datasets/${dataset.id}`}
                      className="hover:underline"
                    >
                      {dataset.name}
                    </Link>
                  </CardTitle>
                  <CardDescription className="mt-1 line-clamp-2">
                    {dataset.description || "No description"}
                  </CardDescription>
                </div>
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="opacity-0 group-hover:opacity-100 transition-opacity"
                    >
                      <MoreHorizontal className="h-4 w-4" />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end">
                    <DropdownMenuItem asChild>
                      <Link href={`/datasets/${dataset.id}`}>
                        <Database className="h-4 w-4 mr-2" />
                        View Details
                      </Link>
                    </DropdownMenuItem>
                    <DropdownMenuSeparator />
                    <DropdownMenuItem asChild>
                      <Link href={`/datasets/${dataset.id}?action=augment`}>
                        <Sparkles className="h-4 w-4 mr-2" />
                        Augment
                      </Link>
                    </DropdownMenuItem>
                    <DropdownMenuItem asChild>
                      <Link href={`/datasets/${dataset.id}?action=train`}>
                        <Brain className="h-4 w-4 mr-2" />
                        Train Model
                      </Link>
                    </DropdownMenuItem>
                    <DropdownMenuItem asChild>
                      <Link href={`/datasets/${dataset.id}?action=extract`}>
                        <Layers className="h-4 w-4 mr-2" />
                        Extract Embeddings
                      </Link>
                    </DropdownMenuItem>
                    <DropdownMenuSeparator />
                    <DropdownMenuItem
                      className="text-red-600"
                      onClick={() => handleDelete(dataset)}
                    >
                      <Trash2 className="h-4 w-4 mr-2" />
                      Delete
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </CardHeader>
              <CardContent>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-500">Products</span>
                  <span className="font-medium">{dataset.product_count}</span>
                </div>
                <div className="flex justify-between text-sm mt-1">
                  <span className="text-gray-500">Created</span>
                  <span>
                    {new Date(dataset.created_at).toLocaleDateString()}
                  </span>
                </div>
                <div className="flex justify-between text-sm mt-1">
                  <span className="text-gray-500">Updated</span>
                  <span>
                    {new Date(dataset.updated_at).toLocaleDateString()}
                  </span>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Delete Confirmation */}
      <AlertDialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Dataset</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete &quot;{datasetToDelete?.name}
              &quot;? This will remove the dataset but not the products in it.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              className="bg-red-600 hover:bg-red-700"
              onClick={() =>
                datasetToDelete && deleteMutation.mutate(datasetToDelete.id)
              }
              disabled={deleteMutation.isPending}
            >
              {deleteMutation.isPending ? (
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              ) : null}
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
