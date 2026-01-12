"use client";

import { useState } from "react";
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
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
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
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
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
  Layers,
  RefreshCw,
  Loader2,
  MoreHorizontal,
  Trash2,
  Database,
} from "lucide-react";
import type { EmbeddingIndex } from "@/types";

export default function EmbeddingsPage() {
  const queryClient = useQueryClient();
  const [isCreateOpen, setIsCreateOpen] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [indexToDelete, setIndexToDelete] = useState<EmbeddingIndex | null>(
    null
  );
  const [newIndex, setNewIndex] = useState({ name: "", model_id: "" });

  // Fetch embedding indexes
  const {
    data: indexes,
    isLoading,
    refetch,
    isFetching,
  } = useQuery({
    queryKey: ["embedding-indexes"],
    queryFn: () => apiClient.getEmbeddingIndexes(),
  });

  // Fetch available models
  const { data: models } = useQuery({
    queryKey: ["training-models"],
    queryFn: () => apiClient.getTrainingModels(),
  });

  // Create index mutation
  const createMutation = useMutation({
    mutationFn: (data: { name: string; model_id: string }) =>
      apiClient.createEmbeddingIndex(data.name, data.model_id),
    onSuccess: () => {
      toast.success("Index created");
      queryClient.invalidateQueries({ queryKey: ["embedding-indexes"] });
      setIsCreateOpen(false);
      setNewIndex({ name: "", model_id: "" });
    },
    onError: () => {
      toast.error("Failed to create index");
    },
  });

  // Delete index mutation (placeholder - would need API method)
  const deleteMutation = useMutation({
    mutationFn: async (id: string) => {
      // API method would go here
      throw new Error("Not implemented");
    },
    onSuccess: () => {
      toast.success("Index deleted");
      queryClient.invalidateQueries({ queryKey: ["embedding-indexes"] });
      setDeleteDialogOpen(false);
      setIndexToDelete(null);
    },
    onError: () => {
      toast.error("Failed to delete index");
    },
  });

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold">Embeddings</h1>
          <p className="text-gray-500">
            Manage FAISS indexes for product matching
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
                New Index
              </Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Create Embedding Index</DialogTitle>
                <DialogDescription>
                  Create a new FAISS index using a trained model
                </DialogDescription>
              </DialogHeader>
              <div className="space-y-4 py-4">
                <div className="space-y-2">
                  <Label htmlFor="name">Index Name</Label>
                  <Input
                    id="name"
                    value={newIndex.name}
                    onChange={(e) =>
                      setNewIndex({ ...newIndex, name: e.target.value })
                    }
                    placeholder="e.g., Beverages Index v1"
                  />
                </div>
                <div className="space-y-2">
                  <Label>Model</Label>
                  <Select
                    value={newIndex.model_id}
                    onValueChange={(value) =>
                      setNewIndex({ ...newIndex, model_id: value })
                    }
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select a trained model..." />
                    </SelectTrigger>
                    <SelectContent>
                      {models?.map((model) => (
                        <SelectItem key={model.id} value={model.id}>
                          {model.name} (v{model.version})
                          {model.is_active && (
                            <Badge className="ml-2" variant="secondary">
                              Active
                            </Badge>
                          )}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
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
                  onClick={() => createMutation.mutate(newIndex)}
                  disabled={
                    !newIndex.name.trim() ||
                    !newIndex.model_id ||
                    createMutation.isPending
                  }
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

      {/* Stats */}
      <div className="grid grid-cols-3 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-gray-500">
              Total Indexes
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">{indexes?.length || 0}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-gray-500">
              Total Vectors
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">
              {indexes?.reduce((sum, idx) => sum + idx.vector_count, 0) || 0}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-gray-500">
              Available Models
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">{models?.length || 0}</p>
          </CardContent>
        </Card>
      </div>

      {/* Indexes Table */}
      <Card>
        <CardHeader>
          <CardTitle>Embedding Indexes</CardTitle>
          <CardDescription>
            FAISS indexes for similarity search
          </CardDescription>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-6 w-6 animate-spin" />
            </div>
          ) : indexes?.length === 0 ? (
            <div className="text-center py-8">
              <Layers className="h-12 w-12 mx-auto text-gray-300 mb-2" />
              <p className="text-gray-500">No embedding indexes yet</p>
              <p className="text-gray-400 text-sm mt-1">
                Train a model first, then create an index
              </p>
              <Button
                className="mt-4"
                variant="outline"
                onClick={() => setIsCreateOpen(true)}
              >
                <Plus className="h-4 w-4 mr-2" />
                Create Index
              </Button>
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Name</TableHead>
                  <TableHead>Model</TableHead>
                  <TableHead>Vectors</TableHead>
                  <TableHead>Index Path</TableHead>
                  <TableHead>Created</TableHead>
                  <TableHead className="w-12"></TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {indexes?.map((index: EmbeddingIndex) => (
                  <TableRow key={index.id}>
                    <TableCell className="font-medium">{index.name}</TableCell>
                    <TableCell>
                      {index.model_name || index.model_artifact_id}
                    </TableCell>
                    <TableCell>
                      <Badge variant="secondary">
                        <Database className="h-3 w-3 mr-1" />
                        {index.vector_count.toLocaleString()}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-gray-500 font-mono text-sm">
                      {index.index_path}
                    </TableCell>
                    <TableCell className="text-gray-500 text-sm">
                      {new Date(index.created_at).toLocaleDateString()}
                    </TableCell>
                    <TableCell>
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button variant="ghost" size="icon">
                            <MoreHorizontal className="h-4 w-4" />
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end">
                          <DropdownMenuItem
                            className="text-red-600"
                            onClick={() => {
                              setIndexToDelete(index);
                              setDeleteDialogOpen(true);
                            }}
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
          )}
        </CardContent>
      </Card>

      {/* Delete Confirmation */}
      <AlertDialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Index</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete &quot;{indexToDelete?.name}&quot;?
              This will remove the index and all its vectors.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              className="bg-red-600 hover:bg-red-700"
              onClick={() =>
                indexToDelete && deleteMutation.mutate(indexToDelete.id)
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
