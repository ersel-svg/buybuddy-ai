"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useParams } from "next/navigation";
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
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import {
  ArrowLeft,
  Plus,
  Trash2,
  Sparkles,
  Brain,
  Layers,
  Loader2,
  Package,
  FolderOpen,
} from "lucide-react";
import type { Product } from "@/types";

export default function DatasetDetailPage() {
  const { id } = useParams<{ id: string }>();
  const queryClient = useQueryClient();

  // Fetch dataset with products
  const {
    data: dataset,
    isLoading,
    error,
  } = useQuery({
    queryKey: ["dataset", id],
    queryFn: () => apiClient.getDataset(id),
    enabled: !!id,
  });

  // Remove product mutation
  const removeProductMutation = useMutation({
    mutationFn: (productId: string) =>
      apiClient.removeProductFromDataset(id, productId),
    onSuccess: () => {
      toast.success("Product removed");
      queryClient.invalidateQueries({ queryKey: ["dataset", id] });
    },
    onError: () => {
      toast.error("Failed to remove product");
    },
  });

  // Action mutations
  const augmentMutation = useMutation({
    mutationFn: () =>
      apiClient.startAugmentation(id, {
        syn_per_class: 100,
        real_per_class: 50,
      }),
    onSuccess: () => {
      toast.success("Augmentation job started");
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
    },
    onError: () => {
      toast.error("Failed to start augmentation");
    },
  });

  const trainMutation = useMutation({
    mutationFn: () =>
      apiClient.startTrainingFromDataset(id, {
        dataset_id: id,
        model_name: "facebook/dinov2-large",
        proj_dim: 512,
        epochs: 30,
        batch_size: 32,
        learning_rate: 1e-4,
        weight_decay: 0.01,
        label_smoothing: 0.1,
        warmup_epochs: 3,
        grad_clip: 1.0,
        llrd_decay: 0.95,
        domain_aware_ratio: 0.3,
        hard_negative_pool_size: 10,
        use_hardest_negatives: true,
        use_mixed_precision: true,
        train_ratio: 0.8,
        valid_ratio: 0.1,
        test_ratio: 0.1,
        save_every: 5,
        seed: 42,
      }),
    onSuccess: () => {
      toast.success("Training job started");
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
    },
    onError: () => {
      toast.error("Failed to start training");
    },
  });

  const extractMutation = useMutation({
    mutationFn: () => apiClient.startEmbeddingExtraction(id, "active"),
    onSuccess: () => {
      toast.success("Embedding extraction started");
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
    },
    onError: () => {
      toast.error("Failed to start embedding extraction");
    },
  });

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <Loader2 className="h-8 w-8 animate-spin" />
      </div>
    );
  }

  if (error || !dataset) {
    return (
      <div className="flex flex-col items-center justify-center h-96">
        <FolderOpen className="h-16 w-16 text-gray-300 mb-4" />
        <h2 className="text-xl font-semibold">Dataset not found</h2>
        <p className="text-gray-500 mb-4">
          The dataset you&apos;re looking for doesn&apos;t exist.
        </p>
        <Link href="/datasets">
          <Button>Back to Datasets</Button>
        </Link>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Link href="/datasets">
            <Button variant="ghost" size="icon">
              <ArrowLeft className="h-4 w-4" />
            </Button>
          </Link>
          <div>
            <h1 className="text-2xl font-bold">{dataset.name}</h1>
            <p className="text-gray-500">
              {dataset.description || "No description"}
            </p>
          </div>
        </div>
        <div className="text-sm text-gray-500">
          Created {new Date(dataset.created_at).toLocaleDateString()}
        </div>
      </div>

      {/* Actions Card */}
      <Card>
        <CardHeader>
          <CardTitle>Dataset Actions</CardTitle>
          <CardDescription>
            Run GPU-intensive operations on this dataset
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-4">
            <Button
              onClick={() => augmentMutation.mutate()}
              disabled={augmentMutation.isPending || dataset.product_count === 0}
            >
              {augmentMutation.isPending ? (
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              ) : (
                <Sparkles className="h-4 w-4 mr-2" />
              )}
              Run Augmentation
            </Button>
            <Button
              variant="outline"
              onClick={() => trainMutation.mutate()}
              disabled={trainMutation.isPending || dataset.product_count === 0}
            >
              {trainMutation.isPending ? (
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              ) : (
                <Brain className="h-4 w-4 mr-2" />
              )}
              Start Training
            </Button>
            <Button
              variant="outline"
              onClick={() => extractMutation.mutate()}
              disabled={extractMutation.isPending || dataset.product_count === 0}
            >
              {extractMutation.isPending ? (
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              ) : (
                <Layers className="h-4 w-4 mr-2" />
              )}
              Extract Embeddings
            </Button>
          </div>
          {dataset.product_count === 0 && (
            <p className="text-sm text-yellow-600 mt-3">
              Add products to this dataset before running actions.
            </p>
          )}
        </CardContent>
      </Card>

      {/* Products Card */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <div>
            <CardTitle>Products ({dataset.product_count})</CardTitle>
            <CardDescription>
              Products included in this dataset
            </CardDescription>
          </div>
          <Link href={`/datasets/${id}/add-products`}>
            <Button>
              <Plus className="h-4 w-4 mr-2" />
              Add Products
            </Button>
          </Link>
        </CardHeader>
        <CardContent>
          {dataset.products?.length === 0 ? (
            <div className="text-center py-8">
              <Package className="h-12 w-12 mx-auto text-gray-300 mb-2" />
              <p className="text-gray-500">No products in this dataset</p>
              <p className="text-gray-400 text-sm">
                Click &quot;Add Products&quot; to get started
              </p>
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Image</TableHead>
                  <TableHead>Barcode</TableHead>
                  <TableHead>Brand</TableHead>
                  <TableHead>Product</TableHead>
                  <TableHead>Frames</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead className="w-12"></TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {dataset.products?.map((product: Product) => (
                  <TableRow key={product.id}>
                    <TableCell>
                      <div className="w-10 h-10 bg-gray-100 rounded flex items-center justify-center">
                        {product.primary_image_url ? (
                          <img
                            src={product.primary_image_url}
                            alt=""
                            className="w-full h-full object-cover rounded"
                          />
                        ) : (
                          <Package className="h-5 w-5 text-gray-400" />
                        )}
                      </div>
                    </TableCell>
                    <TableCell className="font-mono text-sm">
                      {product.barcode}
                    </TableCell>
                    <TableCell>{product.brand_name || "-"}</TableCell>
                    <TableCell className="max-w-[200px] truncate">
                      {product.product_name || "-"}
                    </TableCell>
                    <TableCell>{product.frame_count}</TableCell>
                    <TableCell>
                      <Badge
                        variant="outline"
                        className={
                          product.status === "ready"
                            ? "bg-green-50 text-green-700"
                            : ""
                        }
                      >
                        {product.status}
                      </Badge>
                    </TableCell>
                    <TableCell>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="text-red-500 hover:text-red-600 hover:bg-red-50"
                        onClick={() => removeProductMutation.mutate(product.id)}
                        disabled={removeProductMutation.isPending}
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
