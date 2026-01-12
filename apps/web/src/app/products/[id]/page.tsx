"use client";

import { useState, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useParams, useSearchParams } from "next/navigation";
import { toast } from "sonner";
import Link from "next/link";
import { apiClient } from "@/lib/api-client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  ArrowLeft,
  Save,
  Play,
  Image as ImageIcon,
  Loader2,
  Download,
  X,
  Pencil,
  Package,
} from "lucide-react";
import type { Product, ProductStatus } from "@/types";

export default function ProductDetailPage() {
  const { id } = useParams<{ id: string }>();
  const searchParams = useSearchParams();
  const queryClient = useQueryClient();

  const [isEditing, setIsEditing] = useState(
    searchParams.get("edit") === "true"
  );
  const [editData, setEditData] = useState<Partial<Product>>({});
  const [selectedFrame, setSelectedFrame] = useState<string | null>(null);

  // Fetch product
  const {
    data: product,
    isLoading,
    error,
  } = useQuery({
    queryKey: ["product", id],
    queryFn: () => apiClient.getProduct(id),
    enabled: !!id,
  });

  // Fetch frames
  const { data: frames } = useQuery({
    queryKey: ["product-frames", id],
    queryFn: () => apiClient.getProductFrames(id),
    enabled: !!product,
  });

  // Update mutation
  const updateMutation = useMutation({
    mutationFn: (data: Partial<Product>) =>
      apiClient.updateProduct(id, data, product?.version),
    onSuccess: () => {
      toast.success("Product updated");
      queryClient.invalidateQueries({ queryKey: ["product", id] });
      setIsEditing(false);
    },
    onError: (error) => {
      if (error instanceof Error && error.message.includes("409")) {
        toast.error("Product was modified by another user. Please refresh.");
      } else {
        toast.error("Failed to update product");
      }
    },
  });

  // Initialize edit data when product loads
  useEffect(() => {
    if (product && isEditing) {
      setEditData(product);
    }
  }, [product, isEditing]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <Loader2 className="h-8 w-8 animate-spin" />
      </div>
    );
  }

  if (error || !product) {
    return (
      <div className="flex flex-col items-center justify-center h-96">
        <Package className="h-16 w-16 text-gray-300 mb-4" />
        <h2 className="text-xl font-semibold">Product not found</h2>
        <p className="text-gray-500 mb-4">
          The product you&apos;re looking for doesn&apos;t exist.
        </p>
        <Link href="/products">
          <Button>Back to Products</Button>
        </Link>
      </div>
    );
  }

  const handleSave = () => {
    updateMutation.mutate(editData);
  };

  const handleDownloadFrames = async () => {
    try {
      const blob = await apiClient.downloadProduct(id);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `product_${product.barcode}.zip`;
      a.click();
      URL.revokeObjectURL(url);
      toast.success("Download started");
    } catch {
      toast.error("Download failed");
    }
  };

  const statusColors: Record<ProductStatus, string> = {
    pending: "bg-yellow-100 text-yellow-800",
    processing: "bg-blue-100 text-blue-800",
    needs_matching: "bg-purple-100 text-purple-800",
    ready: "bg-green-100 text-green-800",
    rejected: "bg-red-100 text-red-800",
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Link href="/products">
            <Button variant="ghost" size="icon">
              <ArrowLeft className="h-4 w-4" />
            </Button>
          </Link>
          <div>
            <h1 className="text-2xl font-bold">
              {product.brand_name || "Unknown Brand"}{" "}
              {product.product_name || "Unknown Product"}
            </h1>
            <p className="text-gray-500 font-mono">{product.barcode}</p>
          </div>
          <Badge className={statusColors[product.status]}>
            {product.status.replace("_", " ")}
          </Badge>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" onClick={handleDownloadFrames}>
            <Download className="h-4 w-4 mr-2" />
            Download Frames
          </Button>
          {isEditing ? (
            <>
              <Button
                variant="outline"
                onClick={() => {
                  setIsEditing(false);
                  setEditData({});
                }}
              >
                <X className="h-4 w-4 mr-2" />
                Cancel
              </Button>
              <Button
                onClick={handleSave}
                disabled={updateMutation.isPending}
              >
                {updateMutation.isPending ? (
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                ) : (
                  <Save className="h-4 w-4 mr-2" />
                )}
                Save
              </Button>
            </>
          ) : (
            <Button
              onClick={() => {
                setEditData(product);
                setIsEditing(true);
              }}
            >
              <Pencil className="h-4 w-4 mr-2" />
              Edit
            </Button>
          )}
        </div>
      </div>

      {/* Content */}
      <div className="grid grid-cols-3 gap-6">
        {/* Left: Video & Info */}
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Play className="h-4 w-4" />
                Source Video
              </CardTitle>
            </CardHeader>
            <CardContent>
              {product.video_url ? (
                <video
                  src={product.video_url}
                  controls
                  className="w-full rounded-lg"
                  poster={product.primary_image_url}
                />
              ) : (
                <div className="aspect-video bg-gray-100 rounded-lg flex items-center justify-center">
                  <div className="text-center text-gray-500">
                    <Play className="h-12 w-12 mx-auto mb-2 opacity-50" />
                    <p>No video available</p>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Info</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex justify-between text-sm">
                <span className="text-gray-500">Status</span>
                <Badge className={statusColors[product.status]}>
                  {product.status.replace("_", " ")}
                </Badge>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-gray-500">Frames</span>
                <span className="font-medium">{product.frame_count}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-gray-500">Category</span>
                <span className="font-medium">{product.category || "-"}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-gray-500">Created</span>
                <span className="font-medium">
                  {new Date(product.created_at).toLocaleDateString()}
                </span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-gray-500">Updated</span>
                <span className="font-medium">
                  {new Date(product.updated_at).toLocaleDateString()}
                </span>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Right: Tabs */}
        <div className="col-span-2">
          <Tabs defaultValue="metadata">
            <TabsList>
              <TabsTrigger value="metadata">Metadata</TabsTrigger>
              <TabsTrigger value="frames">
                Frames ({product.frame_count})
              </TabsTrigger>
              <TabsTrigger value="nutrition">Nutrition</TabsTrigger>
            </TabsList>

            {/* Metadata Tab */}
            <TabsContent value="metadata" className="mt-4">
              <Card>
                <CardHeader>
                  <CardTitle>Product Information</CardTitle>
                  <CardDescription>
                    Metadata extracted from video using Gemini
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label>Brand Name</Label>
                      {isEditing ? (
                        <Input
                          value={editData.brand_name || ""}
                          onChange={(e) =>
                            setEditData({
                              ...editData,
                              brand_name: e.target.value,
                            })
                          }
                        />
                      ) : (
                        <p className="p-2 bg-gray-50 rounded">
                          {product.brand_name || "-"}
                        </p>
                      )}
                    </div>
                    <div className="space-y-2">
                      <Label>Sub Brand</Label>
                      {isEditing ? (
                        <Input
                          value={editData.sub_brand || ""}
                          onChange={(e) =>
                            setEditData({
                              ...editData,
                              sub_brand: e.target.value,
                            })
                          }
                        />
                      ) : (
                        <p className="p-2 bg-gray-50 rounded">
                          {product.sub_brand || "-"}
                        </p>
                      )}
                    </div>
                    <div className="space-y-2">
                      <Label>Product Name</Label>
                      {isEditing ? (
                        <Input
                          value={editData.product_name || ""}
                          onChange={(e) =>
                            setEditData({
                              ...editData,
                              product_name: e.target.value,
                            })
                          }
                        />
                      ) : (
                        <p className="p-2 bg-gray-50 rounded">
                          {product.product_name || "-"}
                        </p>
                      )}
                    </div>
                    <div className="space-y-2">
                      <Label>Variant/Flavor</Label>
                      {isEditing ? (
                        <Input
                          value={editData.variant_flavor || ""}
                          onChange={(e) =>
                            setEditData({
                              ...editData,
                              variant_flavor: e.target.value,
                            })
                          }
                        />
                      ) : (
                        <p className="p-2 bg-gray-50 rounded">
                          {product.variant_flavor || "-"}
                        </p>
                      )}
                    </div>
                    <div className="space-y-2">
                      <Label>Category</Label>
                      {isEditing ? (
                        <Input
                          value={editData.category || ""}
                          onChange={(e) =>
                            setEditData({
                              ...editData,
                              category: e.target.value,
                            })
                          }
                        />
                      ) : (
                        <p className="p-2 bg-gray-50 rounded">
                          {product.category || "-"}
                        </p>
                      )}
                    </div>
                    <div className="space-y-2">
                      <Label>Container Type</Label>
                      {isEditing ? (
                        <Input
                          value={editData.container_type || ""}
                          onChange={(e) =>
                            setEditData({
                              ...editData,
                              container_type: e.target.value,
                            })
                          }
                        />
                      ) : (
                        <p className="p-2 bg-gray-50 rounded">
                          {product.container_type || "-"}
                        </p>
                      )}
                    </div>
                    <div className="space-y-2">
                      <Label>Net Quantity</Label>
                      {isEditing ? (
                        <Input
                          value={editData.net_quantity || ""}
                          onChange={(e) =>
                            setEditData({
                              ...editData,
                              net_quantity: e.target.value,
                            })
                          }
                        />
                      ) : (
                        <p className="p-2 bg-gray-50 rounded">
                          {product.net_quantity || "-"}
                        </p>
                      )}
                    </div>
                    <div className="space-y-2">
                      <Label>Status</Label>
                      {isEditing ? (
                        <Select
                          value={editData.status}
                          onValueChange={(value) =>
                            setEditData({
                              ...editData,
                              status: value as ProductStatus,
                            })
                          }
                        >
                          <SelectTrigger>
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="pending">Pending</SelectItem>
                            <SelectItem value="processing">
                              Processing
                            </SelectItem>
                            <SelectItem value="needs_matching">
                              Needs Matching
                            </SelectItem>
                            <SelectItem value="ready">Ready</SelectItem>
                            <SelectItem value="rejected">Rejected</SelectItem>
                          </SelectContent>
                        </Select>
                      ) : (
                        <Badge className={statusColors[product.status]}>
                          {product.status.replace("_", " ")}
                        </Badge>
                      )}
                    </div>
                  </div>

                  {/* Grounding Prompt */}
                  <div className="space-y-2">
                    <Label>Grounding Prompt (for SAM3 segmentation)</Label>
                    {isEditing ? (
                      <Textarea
                        value={editData.grounding_prompt || ""}
                        onChange={(e) =>
                          setEditData({
                            ...editData,
                            grounding_prompt: e.target.value,
                          })
                        }
                        rows={3}
                      />
                    ) : (
                      <p className="p-2 bg-gray-50 rounded font-mono text-sm">
                        {product.grounding_prompt || "-"}
                      </p>
                    )}
                  </div>

                  {/* Claims */}
                  {product.claims && product.claims.length > 0 && (
                    <div className="space-y-2">
                      <Label>Claims</Label>
                      <div className="flex flex-wrap gap-2">
                        {product.claims.map((claim, i) => (
                          <Badge key={i} variant="secondary">
                            {claim}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            {/* Frames Tab */}
            <TabsContent value="frames" className="mt-4">
              <Card>
                <CardHeader>
                  <CardTitle>Extracted Frames</CardTitle>
                  <CardDescription>
                    Synthetic frames segmented using SAM3
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {frames?.frames && frames.frames.length > 0 ? (
                    <div className="grid grid-cols-6 gap-2">
                      {frames.frames.map((frame, i) => (
                        <div
                          key={i}
                          className="aspect-square bg-gray-100 rounded overflow-hidden cursor-pointer hover:ring-2 hover:ring-blue-500 transition-all"
                          onClick={() => setSelectedFrame(frame.url)}
                        >
                          <img
                            src={frame.url}
                            alt={`Frame ${frame.index}`}
                            className="w-full h-full object-cover"
                          />
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-center py-8">
                      <ImageIcon className="h-12 w-12 mx-auto text-gray-300" />
                      <p className="text-gray-500 mt-2">No frames extracted</p>
                      <p className="text-gray-400 text-sm">
                        Process the video to extract frames
                      </p>
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            {/* Nutrition Tab */}
            <TabsContent value="nutrition" className="mt-4">
              <Card>
                <CardHeader>
                  <CardTitle>Nutrition Facts</CardTitle>
                  <CardDescription>
                    Extracted from product packaging
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {product.nutrition_facts ? (
                    <div className="grid grid-cols-2 gap-4">
                      {Object.entries(product.nutrition_facts).map(
                        ([key, value]) => (
                          <div
                            key={key}
                            className="flex justify-between p-2 bg-gray-50 rounded"
                          >
                            <span className="text-gray-600 capitalize">
                              {key.replace("_", " ")}
                            </span>
                            <span className="font-medium">{value}</span>
                          </div>
                        )
                      )}
                    </div>
                  ) : (
                    <div className="text-center py-8">
                      <p className="text-gray-500">
                        No nutrition facts available
                      </p>
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      </div>

      {/* Frame Preview Modal */}
      {selectedFrame && (
        <div
          className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center p-8"
          onClick={() => setSelectedFrame(null)}
        >
          <div className="relative max-w-4xl max-h-full">
            <img
              src={selectedFrame}
              alt="Frame preview"
              className="max-w-full max-h-[80vh] object-contain rounded-lg"
            />
            <Button
              variant="secondary"
              size="icon"
              className="absolute top-4 right-4"
              onClick={() => setSelectedFrame(null)}
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
