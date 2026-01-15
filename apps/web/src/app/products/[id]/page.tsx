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
  Trash2,
  CheckSquare,
  Square,
  Star,
} from "lucide-react";
import type { Product, ProductStatus, ProductIdentifierCreate } from "@/types";
import { IdentifiersEditor } from "@/components/products/identifiers-editor";
import { CustomFieldsEditor } from "@/components/products/custom-fields-editor";

export default function ProductDetailPage() {
  const { id } = useParams<{ id: string }>();
  const searchParams = useSearchParams();
  const queryClient = useQueryClient();

  const [isEditing, setIsEditing] = useState(
    searchParams.get("edit") === "true"
  );
  const [editData, setEditData] = useState<Partial<Product>>({});
  const [selectedFrame, setSelectedFrame] = useState<string | null>(null);
  const [frameTab, setFrameTab] = useState<"synthetic" | "real" | "augmented">("synthetic");
  const [selectedFrameIds, setSelectedFrameIds] = useState<Set<string>>(new Set());
  const [isDeleting, setIsDeleting] = useState(false);
  const [editIdentifiers, setEditIdentifiers] = useState<ProductIdentifierCreate[]>([]);
  const [editCustomFields, setEditCustomFields] = useState<Record<string, string>>({});

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
    onError: (error) => {
      if (error instanceof Error && error.message.includes("409")) {
        toast.error("Product was modified by another user. Please refresh.");
      } else {
        toast.error("Failed to update product");
      }
    },
  });

  // Set primary image mutation
  const setPrimaryMutation = useMutation({
    mutationFn: (imageUrl: string) =>
      apiClient.setPrimaryImage(id, imageUrl, product?.version),
    onSuccess: () => {
      toast.success("Primary image updated");
      queryClient.invalidateQueries({ queryKey: ["product", id] });
      queryClient.invalidateQueries({ queryKey: ["products"] });
    },
    onError: (error) => {
      if (error instanceof Error && error.message.includes("409")) {
        toast.error("Product was modified. Please refresh and try again.");
      } else {
        toast.error("Failed to set primary image");
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

  const handleSave = async () => {
    try {
      // Save product data
      await updateMutation.mutateAsync(editData);

      // Save identifiers
      await apiClient.updateProductIdentifiers(id, editIdentifiers);

      // Save custom fields
      await apiClient.updateCustomFields(id, editCustomFields);

      // Refetch to get updated data
      queryClient.invalidateQueries({ queryKey: ["product", id] });

      // Reset edit state
      setIsEditing(false);
      setEditData({});
      setEditIdentifiers([]);
      setEditCustomFields({});

      toast.success("Product saved successfully");
    } catch (error) {
      // Error handling is done in mutation onError
      console.error("Save error:", error);
    }
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
            <p className="text-gray-500 font-mono">
              {product.identifiers_list?.find((i) => i.is_primary)?.identifier_value ||
                product.barcode ||
                "No identifier"}
            </p>
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
                  setEditIdentifiers([]);
                  setEditCustomFields({});
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
                // Initialize identifier and custom field state from product
                setEditIdentifiers(
                  (product.identifiers_list || []).map((i) => ({
                    identifier_type: i.identifier_type,
                    identifier_value: i.identifier_value,
                    custom_label: i.custom_label,
                    is_primary: i.is_primary,
                  }))
                );
                setEditCustomFields(product.custom_fields || {});
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
                        <p className="p-2 bg-muted rounded">
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
                        <p className="p-2 bg-muted rounded">
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
                        <p className="p-2 bg-muted rounded">
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
                        <p className="p-2 bg-muted rounded">
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
                        <p className="p-2 bg-muted rounded">
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
                        <p className="p-2 bg-muted rounded">
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
                        <p className="p-2 bg-muted rounded">
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

                  {/* Second Row of Fields */}
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label>Manufacturer Country</Label>
                      {isEditing ? (
                        <Input
                          value={editData.manufacturer_country || ""}
                          onChange={(e) =>
                            setEditData({
                              ...editData,
                              manufacturer_country: e.target.value,
                            })
                          }
                        />
                      ) : (
                        <p className="p-2 bg-muted rounded">
                          {product.manufacturer_country || "-"}
                        </p>
                      )}
                    </div>
                    <div className="space-y-2">
                      <Label>Visibility Score</Label>
                      {isEditing ? (
                        <Input
                          type="number"
                          min={0}
                          max={100}
                          value={editData.visibility_score ?? ""}
                          onChange={(e) =>
                            setEditData({
                              ...editData,
                              visibility_score: e.target.value ? parseInt(e.target.value) : undefined,
                            })
                          }
                        />
                      ) : (
                        <p className="p-2 bg-muted rounded">
                          {product.visibility_score !== undefined ? `${product.visibility_score}%` : "-"}
                        </p>
                      )}
                    </div>
                  </div>

                  {/* Pack Configuration */}
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label>Pack Type</Label>
                      {isEditing ? (
                        <Select
                          value={editData.pack_configuration?.type || ""}
                          onValueChange={(value) =>
                            setEditData({
                              ...editData,
                              pack_configuration: {
                                ...editData.pack_configuration,
                                type: value as "single_unit" | "multipack",
                                item_count: editData.pack_configuration?.item_count || 1,
                              },
                            })
                          }
                        >
                          <SelectTrigger>
                            <SelectValue placeholder="Select type" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="single_unit">Single Unit</SelectItem>
                            <SelectItem value="multipack">Multipack</SelectItem>
                          </SelectContent>
                        </Select>
                      ) : (
                        <p className="p-2 bg-muted rounded">
                          {product.pack_configuration?.type?.replace("_", " ") || "-"}
                        </p>
                      )}
                    </div>
                    <div className="space-y-2">
                      <Label>Item Count</Label>
                      {isEditing ? (
                        <Input
                          type="number"
                          min={1}
                          value={editData.pack_configuration?.item_count ?? ""}
                          onChange={(e) =>
                            setEditData({
                              ...editData,
                              pack_configuration: {
                                ...editData.pack_configuration,
                                type: editData.pack_configuration?.type || "single_unit",
                                item_count: e.target.value ? parseInt(e.target.value) : 1,
                              },
                            })
                          }
                        />
                      ) : (
                        <p className="p-2 bg-muted rounded">
                          {product.pack_configuration?.item_count || "-"}
                        </p>
                      )}
                    </div>
                  </div>

                  {/* Marketing Description */}
                  <div className="space-y-2">
                    <Label>Marketing Description</Label>
                    {isEditing ? (
                      <Textarea
                        value={editData.marketing_description || ""}
                        onChange={(e) =>
                          setEditData({
                            ...editData,
                            marketing_description: e.target.value,
                          })
                        }
                        rows={3}
                        placeholder="Marketing copy or product description..."
                      />
                    ) : (
                      <p className="p-2 bg-muted rounded text-sm">
                        {product.marketing_description || "-"}
                      </p>
                    )}
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
                      <p className="p-2 bg-muted rounded font-mono text-sm">
                        {product.grounding_prompt || "-"}
                      </p>
                    )}
                  </div>

                  {/* Claims */}
                  <div className="space-y-2">
                    <Label>Claims</Label>
                    {isEditing ? (
                      <Textarea
                        value={(editData.claims || []).join(", ")}
                        onChange={(e) =>
                          setEditData({
                            ...editData,
                            claims: e.target.value.split(",").map((c) => c.trim()).filter(Boolean),
                          })
                        }
                        rows={2}
                        placeholder="Enter claims separated by commas..."
                      />
                    ) : product.claims && product.claims.length > 0 ? (
                      <div className="flex flex-wrap gap-2">
                        {product.claims.map((claim, i) => (
                          <Badge key={i} variant="secondary">
                            {claim}
                          </Badge>
                        ))}
                      </div>
                    ) : (
                      <p className="p-2 bg-muted rounded text-sm text-gray-500">No claims</p>
                    )}
                  </div>

                  {/* Issues Detected */}
                  {(product.issues_detected && product.issues_detected.length > 0) || isEditing ? (
                    <div className="space-y-2">
                      <Label>Issues Detected</Label>
                      {isEditing ? (
                        <Textarea
                          value={(editData.issues_detected || []).join(", ")}
                          onChange={(e) =>
                            setEditData({
                              ...editData,
                              issues_detected: e.target.value.split(",").map((c) => c.trim()).filter(Boolean),
                            })
                          }
                          rows={2}
                          placeholder="Enter issues separated by commas..."
                        />
                      ) : (
                        <div className="flex flex-wrap gap-2">
                          {product.issues_detected?.map((issue, i) => (
                            <Badge key={i} variant="destructive">
                              {issue}
                            </Badge>
                          ))}
                        </div>
                      )}
                    </div>
                  ) : null}
                </CardContent>
              </Card>

              {/* Product Identifiers */}
              <div className="mt-4">
                <IdentifiersEditor
                  identifiers={product.identifiers_list || []}
                  isEditing={isEditing}
                  onUpdate={setEditIdentifiers}
                />
              </div>

              {/* Custom Fields */}
              <div className="mt-4">
                <CustomFieldsEditor
                  fields={product.custom_fields || {}}
                  isEditing={isEditing}
                  onUpdate={setEditCustomFields}
                />
              </div>
            </TabsContent>

            {/* Frames Tab */}
            <TabsContent value="frames" className="mt-4">
              <Card>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div>
                      <CardTitle>Product Frames</CardTitle>
                      <CardDescription>
                        All image types: synthetic, real, and augmented
                      </CardDescription>
                    </div>
                    {selectedFrameIds.size > 0 && (
                      <Button
                        variant="destructive"
                        size="sm"
                        disabled={isDeleting}
                        onClick={async () => {
                          const ids = Array.from(selectedFrameIds);
                          if (ids.some(id => id === null)) {
                            toast.error("Legacy frames without DB records cannot be deleted");
                            return;
                          }
                          setIsDeleting(true);
                          try {
                            await apiClient.deleteProductFrames(id, ids.filter(Boolean) as string[]);
                            toast.success(`Deleted ${ids.length} frames`);
                            setSelectedFrameIds(new Set());
                            queryClient.invalidateQueries({ queryKey: ["product-frames", id] });
                          } catch {
                            toast.error("Failed to delete frames");
                          } finally {
                            setIsDeleting(false);
                          }
                        }}
                      >
                        {isDeleting ? (
                          <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                        ) : (
                          <Trash2 className="h-4 w-4 mr-2" />
                        )}
                        Delete {selectedFrameIds.size} selected
                      </Button>
                    )}
                  </div>
                </CardHeader>
                <CardContent>
                  {/* Frame Type Tabs */}
                  <Tabs value={frameTab} onValueChange={(v) => {
                    setFrameTab(v as typeof frameTab);
                    setSelectedFrameIds(new Set());
                  }}>
                    <TabsList className="mb-4">
                      <TabsTrigger value="synthetic" className="gap-2">
                        Synthetic
                        <Badge variant="secondary" className="ml-1">
                          {frames?.counts?.synthetic || 0}
                        </Badge>
                      </TabsTrigger>
                      <TabsTrigger value="real" className="gap-2">
                        Real
                        <Badge variant="secondary" className="ml-1">
                          {frames?.counts?.real || 0}
                        </Badge>
                      </TabsTrigger>
                      <TabsTrigger value="augmented" className="gap-2">
                        Augmented
                        <Badge variant="secondary" className="ml-1">
                          {frames?.counts?.augmented || 0}
                        </Badge>
                      </TabsTrigger>
                    </TabsList>

                    {/* Frame Grid */}
                    {(() => {
                      const filteredFrames = frames?.frames?.filter(f => f.image_type === frameTab) || [];
                      const selectableFrames = filteredFrames.filter(f => f.id !== null);
                      const allSelected = selectableFrames.length > 0 &&
                        selectableFrames.every(f => selectedFrameIds.has(f.id!));

                      return (
                        <>
                          {filteredFrames.length > 0 && selectableFrames.length > 0 && (
                            <div className="flex items-center gap-2 mb-3">
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() => {
                                  if (allSelected) {
                                    setSelectedFrameIds(new Set());
                                  } else {
                                    setSelectedFrameIds(new Set(selectableFrames.map(f => f.id!)));
                                  }
                                }}
                              >
                                {allSelected ? (
                                  <CheckSquare className="h-4 w-4 mr-2" />
                                ) : (
                                  <Square className="h-4 w-4 mr-2" />
                                )}
                                {allSelected ? "Deselect All" : "Select All"}
                              </Button>
                              <span className="text-sm text-muted-foreground">
                                {selectedFrameIds.size} of {selectableFrames.length} selected
                              </span>
                            </div>
                          )}

                          {filteredFrames.length > 0 ? (
                            <div className="grid grid-cols-6 gap-2">
                              {filteredFrames.map((frame, i) => {
                                const isSelected = frame.id ? selectedFrameIds.has(frame.id) : false;
                                const canSelect = frame.id !== null;

                                return (
                                  <div
                                    key={frame.id || i}
                                    className={`relative aspect-square bg-gray-100 rounded overflow-hidden cursor-pointer transition-all ${
                                      isSelected ? "ring-2 ring-blue-500" : "hover:ring-2 hover:ring-gray-300"
                                    }`}
                                  >
                                    {/* Selection checkbox */}
                                    {canSelect && (
                                      <div
                                        className="absolute top-1 left-1 z-10"
                                        onClick={(e) => {
                                          e.stopPropagation();
                                          const newSet = new Set(selectedFrameIds);
                                          if (isSelected) {
                                            newSet.delete(frame.id!);
                                          } else {
                                            newSet.add(frame.id!);
                                          }
                                          setSelectedFrameIds(newSet);
                                        }}
                                      >
                                        <div className={`w-5 h-5 rounded border-2 flex items-center justify-center ${
                                          isSelected ? "bg-blue-500 border-blue-500" : "bg-white/80 border-gray-400"
                                        }`}>
                                          {isSelected && <X className="h-3 w-3 text-white" />}
                                        </div>
                                      </div>
                                    )}

                                    {/* Legacy badge */}
                                    {!canSelect && (
                                      <div className="absolute top-1 left-1 z-10">
                                        <Badge variant="outline" className="text-xs bg-white/80">
                                          Legacy
                                        </Badge>
                                      </div>
                                    )}

                                    {/* Set as Primary button */}
                                    <div
                                      className="absolute top-1 right-1 z-10"
                                      onClick={(e) => {
                                        e.stopPropagation();
                                        if (!setPrimaryMutation.isPending) {
                                          setPrimaryMutation.mutate(frame.url);
                                        }
                                      }}
                                    >
                                      <div
                                        className={`w-6 h-6 rounded-full flex items-center justify-center cursor-pointer transition-all ${
                                          product.primary_image_url === frame.url
                                            ? "bg-yellow-400 text-yellow-900"
                                            : "bg-white/80 text-gray-400 hover:bg-yellow-100 hover:text-yellow-600"
                                        }`}
                                        title={product.primary_image_url === frame.url ? "Current thumbnail" : "Set as thumbnail"}
                                      >
                                        <Star
                                          className={`h-4 w-4 ${product.primary_image_url === frame.url ? "fill-current" : ""}`}
                                        />
                                      </div>
                                    </div>

                                    {/* Image */}
                                    <img
                                      src={frame.url}
                                      alt={`${frameTab} frame ${frame.index}`}
                                      className="w-full h-full object-cover"
                                      onClick={() => setSelectedFrame(frame.url)}
                                    />
                                  </div>
                                );
                              })}
                            </div>
                          ) : (
                            <div className="text-center py-8">
                              <ImageIcon className="h-12 w-12 mx-auto text-gray-300" />
                              <p className="text-gray-500 mt-2">
                                No {frameTab} frames
                              </p>
                              <p className="text-gray-400 text-sm">
                                {frameTab === "synthetic" && "Process the video to extract frames"}
                                {frameTab === "real" && "Add real images via matching"}
                                {frameTab === "augmented" && "Run augmentation to generate variants"}
                              </p>
                            </div>
                          )}
                        </>
                      );
                    })()}
                  </Tabs>
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
                            className="flex justify-between p-2 bg-muted rounded"
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
