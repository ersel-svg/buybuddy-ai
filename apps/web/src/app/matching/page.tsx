"use client";

import { useState, useMemo, useCallback, useRef } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useVirtualizer } from "@tanstack/react-virtual";
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
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Search,
  Check,
  X,
  RefreshCw,
  Package,
  Loader2,
  GitCompare,
  ZoomIn,
  ImageIcon,
  Film,
  ChevronRight,
  CheckSquare,
  Square,
} from "lucide-react";
import { cn } from "@/lib/utils";
import type { ProductSummary, MatchCandidate } from "@/types";

// Extended candidate type with barcode for UPC matching
interface CandidateWithBarcode extends MatchCandidate {
  barcode?: string;
  is_upc_match?: boolean;
}

export default function MatchingPage() {
  const queryClient = useQueryClient();
  const parentRef = useRef<HTMLDivElement>(null);

  // State
  const [selectedProductId, setSelectedProductId] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [similarityThreshold, setSimilarityThreshold] = useState(0.7);
  const [selectedCandidates, setSelectedCandidates] = useState<Set<string>>(
    new Set()
  );
  const [previewImage, setPreviewImage] = useState<string | null>(null);

  // Fetch all products for matching
  const { data: products, isLoading: isLoadingProducts } = useQuery({
    queryKey: ["matching-products"],
    queryFn: () => apiClient.getMatchingProducts({ limit: 500 }),
  });

  // Filter products by search query
  const filteredProducts = useMemo(() => {
    if (!products) return [];
    if (!searchQuery.trim()) return products;

    const query = searchQuery.toLowerCase();
    return products.filter(
      (p) =>
        p.barcode?.toLowerCase().includes(query) ||
        p.brand_name?.toLowerCase().includes(query) ||
        p.product_name?.toLowerCase().includes(query)
    );
  }, [products, searchQuery]);

  // Virtual list for products
  const rowVirtualizer = useVirtualizer({
    count: filteredProducts.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 88,
    overscan: 5,
  });

  // Fetch selected product details
  const { data: productDetails, isLoading: isLoadingDetails } = useQuery({
    queryKey: ["matching-product", selectedProductId],
    queryFn: () => apiClient.getMatchingProduct(selectedProductId!),
    enabled: !!selectedProductId,
  });

  // Search candidates for selected product
  const {
    data: candidatesResult,
    isLoading: isLoadingCandidates,
    refetch: refetchCandidates,
  } = useQuery({
    queryKey: ["matching-candidates", selectedProductId, similarityThreshold],
    queryFn: () =>
      selectedProductId
        ? apiClient.searchMatches(
            productDetails?.product?.barcode || "",
            100
          )
        : Promise.resolve({ candidates: [] }),
    enabled: !!selectedProductId && !!productDetails?.product?.barcode,
  });

  // Process candidates with UPC matching info
  const candidates = useMemo(() => {
    if (!candidatesResult?.candidates) return [];

    const productBarcode = productDetails?.product?.barcode;

    return candidatesResult.candidates
      .map((c): CandidateWithBarcode => ({
        ...c,
        // Check if candidate barcode matches product barcode
        is_upc_match: c.metadata?.barcode === productBarcode ||
                      (c.metadata?.folder_upc_hint as string) === productBarcode,
        barcode: (c.metadata?.barcode as string) || (c.metadata?.folder_upc_hint as string),
      }))
      .filter((c) => c.similarity >= similarityThreshold || c.is_upc_match)
      .sort((a, b) => {
        // UPC matches first, then by similarity
        if (a.is_upc_match && !b.is_upc_match) return -1;
        if (!a.is_upc_match && b.is_upc_match) return 1;
        return b.similarity - a.similarity;
      });
  }, [candidatesResult, similarityThreshold, productDetails?.product?.barcode]);

  // Count UPC matches
  const upcMatchCount = useMemo(() => {
    return candidates.filter((c) => c.is_upc_match).length;
  }, [candidates]);

  // Add real images mutation
  const addRealImagesMutation = useMutation({
    mutationFn: async (imageUrls: string[]) => {
      if (!selectedProductId) throw new Error("No product selected");
      return apiClient.addRealImages(selectedProductId, imageUrls);
    },
    onSuccess: (data) => {
      toast.success(`${data.added} real images added`);
      setSelectedCandidates(new Set());
      queryClient.invalidateQueries({
        queryKey: ["matching-product", selectedProductId],
      });
      queryClient.invalidateQueries({ queryKey: ["matching-products"] });
    },
    onError: () => {
      toast.error("Failed to add real images");
    },
  });

  // Handle approve selected
  const handleApproveSelected = useCallback(async () => {
    const imageUrls = candidates
      .filter((c) => selectedCandidates.has(c.id))
      .map((c) => c.image_url);

    if (imageUrls.length > 0) {
      await addRealImagesMutation.mutateAsync(imageUrls);
    }
  }, [candidates, selectedCandidates, addRealImagesMutation]);

  // Toggle candidate selection
  const toggleCandidate = useCallback((candidateId: string) => {
    setSelectedCandidates((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(candidateId)) {
        newSet.delete(candidateId);
      } else {
        newSet.add(candidateId);
      }
      return newSet;
    });
  }, []);

  // Select All
  const handleSelectAll = useCallback(() => {
    const allIds = new Set(candidates.map((c) => c.id));
    setSelectedCandidates(allIds);
  }, [candidates]);

  // Deselect All
  const handleDeselectAll = useCallback(() => {
    setSelectedCandidates(new Set());
  }, []);

  // Select product
  const handleSelectProduct = useCallback((productId: string) => {
    setSelectedProductId(productId);
    setSelectedCandidates(new Set());
  }, []);

  return (
    <div className="h-[calc(100vh-8rem)] flex gap-4">
      {/* Left Panel: Product List */}
      <div className="w-80 flex flex-col gap-3">
        <Card className="flex-1 flex flex-col overflow-hidden">
          <CardHeader className="pb-3">
            <CardTitle className="text-lg">Products</CardTitle>
            <CardDescription>
              Select a product to add real images
            </CardDescription>
          </CardHeader>
          <CardContent className="flex-1 flex flex-col gap-3 overflow-hidden p-3 pt-0">
            {/* Search */}
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
              <Input
                placeholder="Search products..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10"
              />
            </div>

            {/* Product Count */}
            <div className="text-xs text-muted-foreground px-1">
              {filteredProducts.length} products
              {searchQuery && ` (filtered from ${products?.length || 0})`}
            </div>

            {/* Virtualized Product List */}
            <div
              ref={parentRef}
              className="flex-1 overflow-auto"
              style={{ contain: "strict" }}
            >
              {isLoadingProducts ? (
                <div className="space-y-2 p-2">
                  {[...Array(5)].map((_, i) => (
                    <Skeleton key={i} className="h-20 w-full" />
                  ))}
                </div>
              ) : filteredProducts.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-full text-muted-foreground">
                  <Package className="h-8 w-8 mb-2 opacity-50" />
                  <p className="text-sm">No products found</p>
                </div>
              ) : (
                <div
                  style={{
                    height: `${rowVirtualizer.getTotalSize()}px`,
                    width: "100%",
                    position: "relative",
                  }}
                >
                  {rowVirtualizer.getVirtualItems().map((virtualRow) => {
                    const product = filteredProducts[virtualRow.index];
                    const isSelected = product.id === selectedProductId;

                    return (
                      <div
                        key={virtualRow.key}
                        style={{
                          position: "absolute",
                          top: 0,
                          left: 0,
                          width: "100%",
                          height: `${virtualRow.size}px`,
                          transform: `translateY(${virtualRow.start}px)`,
                        }}
                        className="p-1"
                      >
                        <div
                          className={cn(
                            "flex gap-3 p-2 rounded-lg cursor-pointer transition-colors border",
                            isSelected
                              ? "bg-primary/10 border-primary"
                              : "hover:bg-muted border-transparent"
                          )}
                          onClick={() => handleSelectProduct(product.id)}
                        >
                          {/* Product Image */}
                          <div className="w-14 h-14 rounded-md overflow-hidden bg-muted flex-shrink-0">
                            {product.primary_image_url ? (
                              <img
                                src={product.primary_image_url}
                                alt="Product"
                                className="w-full h-full object-cover"
                              />
                            ) : (
                              <div className="w-full h-full flex items-center justify-center">
                                <Package className="h-6 w-6 text-muted-foreground/50" />
                              </div>
                            )}
                          </div>

                          {/* Product Info */}
                          <div className="flex-1 min-w-0">
                            <p className="font-mono text-xs text-muted-foreground truncate">
                              {product.barcode || "No barcode"}
                            </p>
                            <p className="text-sm font-medium truncate">
                              {product.brand_name || "Unknown"}
                            </p>
                            <p className="text-xs text-muted-foreground truncate">
                              {product.product_name || "No name"}
                            </p>

                            {/* Frame Counts */}
                            <div className="flex gap-2 mt-1">
                              <span className="inline-flex items-center gap-1 text-xs">
                                <Film className="h-3 w-3 text-blue-500" />
                                <span className="text-muted-foreground">
                                  {product.frame_count}
                                </span>
                              </span>
                              <span className="inline-flex items-center gap-1 text-xs">
                                <ImageIcon className="h-3 w-3 text-green-500" />
                                <span className="text-muted-foreground">
                                  {product.real_image_count}
                                </span>
                              </span>
                            </div>
                          </div>

                          {isSelected && (
                            <ChevronRight className="h-4 w-4 text-primary flex-shrink-0 self-center" />
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Middle Panel: Selected Product & Candidates */}
      <div className="flex-1 flex flex-col gap-4">
        {/* Controls */}
        <Card>
          <CardContent className="py-4">
            <div className="flex items-center gap-4 flex-wrap">
              {/* Similarity Threshold */}
              <div className="flex-1 min-w-[200px] max-w-xs">
                <Label className="text-xs text-muted-foreground">
                  Similarity Threshold: {(similarityThreshold * 100).toFixed(0)}%
                </Label>
                <Slider
                  value={[similarityThreshold]}
                  onValueChange={([value]) => setSimilarityThreshold(value)}
                  min={0.5}
                  max={0.99}
                  step={0.01}
                  className="mt-2"
                />
              </div>

              {/* Refresh */}
              <Button
                variant="outline"
                size="sm"
                onClick={() => refetchCandidates()}
                disabled={isLoadingCandidates || !selectedProductId}
              >
                <RefreshCw
                  className={cn(
                    "h-4 w-4",
                    isLoadingCandidates && "animate-spin"
                  )}
                />
              </Button>

              {/* Select All / Deselect All */}
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleSelectAll}
                  disabled={candidates.length === 0}
                >
                  <CheckSquare className="h-4 w-4 mr-1" />
                  Select All
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleDeselectAll}
                  disabled={selectedCandidates.size === 0}
                >
                  <Square className="h-4 w-4 mr-1" />
                  Deselect All
                </Button>
              </div>

              {/* Actions */}
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleDeselectAll}
                  disabled={selectedCandidates.size === 0}
                  className="text-red-600 hover:text-red-700 hover:bg-red-50"
                >
                  <X className="h-4 w-4 mr-1" />
                  Clear ({selectedCandidates.size})
                </Button>
                <Button
                  size="sm"
                  onClick={handleApproveSelected}
                  disabled={selectedCandidates.size === 0}
                  className="bg-green-600 hover:bg-green-700"
                >
                  <Check className="h-4 w-4 mr-1" />
                  Add Real Images ({selectedCandidates.size})
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Content Area */}
        <div className="flex-1 flex gap-4 overflow-hidden">
          {/* Selected Product Preview - Larger */}
          {selectedProductId && productDetails && (
            <Card className="w-80 flex-shrink-0 overflow-auto">
              <CardHeader className="py-3">
                <CardTitle className="text-sm">Reference Product</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* Large Primary Image */}
                <div
                  className="aspect-square bg-muted rounded-lg overflow-hidden cursor-pointer border-2 border-dashed border-primary/30"
                  onClick={() => productDetails.product.primary_image_url &&
                    setPreviewImage(productDetails.product.primary_image_url)}
                >
                  {productDetails.product.primary_image_url ? (
                    <img
                      src={productDetails.product.primary_image_url}
                      alt="Product"
                      className="w-full h-full object-contain"
                    />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center">
                      <Package className="h-16 w-16 text-muted-foreground/30" />
                    </div>
                  )}
                </div>

                {/* Product Info */}
                <div className="space-y-2 p-3 bg-muted/50 rounded-lg">
                  <p className="font-mono text-sm font-bold text-primary">
                    {productDetails.product.barcode}
                  </p>
                  <p className="font-medium">
                    {productDetails.product.brand_name}
                  </p>
                  <p className="text-sm text-muted-foreground">
                    {productDetails.product.product_name}
                  </p>
                  <p className="text-xs text-muted-foreground">
                    {productDetails.product.category || "No category"}
                  </p>
                </div>

                {/* Stats */}
                <div className="flex gap-4 p-3 bg-muted/50 rounded-lg">
                  <div className="flex items-center gap-2">
                    <Film className="h-4 w-4 text-blue-500" />
                    <span className="text-sm font-medium">
                      {productDetails.synthetic_frames.length}
                    </span>
                    <span className="text-xs text-muted-foreground">frames</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <ImageIcon className="h-4 w-4 text-green-500" />
                    <span className="text-sm font-medium">
                      {productDetails.real_images.length}
                    </span>
                    <span className="text-xs text-muted-foreground">real</span>
                  </div>
                </div>

                {/* Existing Real Images */}
                {productDetails.real_images.length > 0 && (
                  <div>
                    <p className="text-xs text-muted-foreground mb-2 font-medium">
                      Existing Real Images ({productDetails.real_images.length})
                    </p>
                    <div className="grid grid-cols-3 gap-2">
                      {productDetails.real_images.slice(0, 9).map((img) => (
                        <div
                          key={img.id}
                          className="aspect-square rounded overflow-hidden bg-muted cursor-pointer hover:ring-2 hover:ring-primary transition-all"
                          onClick={() => setPreviewImage(img.image_url)}
                        >
                          <img
                            src={img.image_url}
                            alt="Real"
                            className="w-full h-full object-cover"
                          />
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          )}

          {/* Candidate Grid */}
          <Card className="flex-1 overflow-hidden flex flex-col">
            <CardHeader className="py-3 flex-shrink-0">
              <div className="flex items-center justify-between">
                <CardTitle className="text-sm flex items-center gap-2">
                  <GitCompare className="h-4 w-4" />
                  Candidates ({candidates.length})
                </CardTitle>
                {upcMatchCount > 0 && (
                  <div className="flex gap-2">
                    <Badge className="bg-green-600">
                      {upcMatchCount} UPC Match
                    </Badge>
                    <Badge variant="secondary" className="bg-orange-100 text-orange-700">
                      {candidates.length - upcMatchCount} Visual Match
                    </Badge>
                  </div>
                )}
              </div>
            </CardHeader>
            <CardContent className="overflow-auto flex-1 p-3">
              {!selectedProductId ? (
                <div className="text-center py-12 text-muted-foreground">
                  <GitCompare className="h-12 w-12 mx-auto mb-2 opacity-50" />
                  <p>Select a product to see matching candidates</p>
                </div>
              ) : isLoadingCandidates || isLoadingDetails ? (
                <div className="flex items-center justify-center py-12">
                  <Loader2 className="h-8 w-8 animate-spin" />
                </div>
              ) : candidates.length === 0 ? (
                <div className="text-center py-12 text-muted-foreground">
                  <Package className="h-12 w-12 mx-auto mb-2 opacity-50" />
                  <p>No candidates found above threshold</p>
                  <p className="text-sm mt-1">Try lowering the threshold</p>
                </div>
              ) : (
                <div className="grid grid-cols-4 xl:grid-cols-5 2xl:grid-cols-6 gap-3">
                  {candidates.map((candidate: CandidateWithBarcode) => (
                    <div
                      key={candidate.id}
                      className={cn(
                        "relative aspect-square rounded-lg overflow-hidden border-2 cursor-pointer transition-all group",
                        selectedCandidates.has(candidate.id)
                          ? "border-blue-500 ring-2 ring-blue-200"
                          : "border-transparent hover:border-gray-300"
                      )}
                      onClick={() => toggleCandidate(candidate.id)}
                    >
                      <img
                        src={candidate.image_url}
                        alt="Candidate"
                        className="w-full h-full object-cover"
                      />

                      {/* UPC Match Badge - Top Left */}
                      <div className="absolute top-1 left-1 flex flex-col gap-1">
                        {candidate.is_upc_match ? (
                          <Badge className="bg-green-600 hover:bg-green-600 text-[10px] px-1.5">
                            UPC MATCH
                          </Badge>
                        ) : (
                          <Badge className="bg-orange-500 hover:bg-orange-500 text-[10px] px-1.5">
                            VISUAL
                          </Badge>
                        )}
                        {/* Similarity Score */}
                        <Badge
                          variant="secondary"
                          className={cn(
                            "text-[10px] px-1.5",
                            candidate.similarity >= 0.9 && "bg-green-100 text-green-800",
                            candidate.similarity >= 0.8 && candidate.similarity < 0.9 && "bg-yellow-100 text-yellow-800",
                            candidate.similarity < 0.8 && "bg-gray-100 text-gray-800"
                          )}
                        >
                          {(candidate.similarity * 100).toFixed(0)}%
                        </Badge>
                      </div>

                      {/* Selection Indicator */}
                      {selectedCandidates.has(candidate.id) && (
                        <div className="absolute inset-0 bg-blue-500/20 flex items-center justify-center">
                          <Check className="h-10 w-10 text-blue-600 drop-shadow-lg" />
                        </div>
                      )}

                      {/* Zoom Button */}
                      <Button
                        size="icon"
                        variant="secondary"
                        className="absolute top-1 right-1 h-6 w-6 opacity-0 group-hover:opacity-100 transition-opacity"
                        onClick={(e) => {
                          e.stopPropagation();
                          setPreviewImage(candidate.image_url);
                        }}
                      >
                        <ZoomIn className="h-3 w-3" />
                      </Button>

                      {/* Barcode at bottom */}
                      {candidate.barcode && (
                        <div className="absolute bottom-0 left-0 right-0 bg-black/60 px-1 py-0.5">
                          <p className="text-[9px] text-white font-mono truncate text-center">
                            {candidate.barcode}
                          </p>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Image Preview Modal */}
      {previewImage && (
        <div
          className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center p-8"
          onClick={() => setPreviewImage(null)}
        >
          <div className="relative max-w-4xl max-h-full">
            <img
              src={previewImage}
              alt="Preview"
              className="max-w-full max-h-[80vh] object-contain rounded-lg"
            />
            <Button
              variant="secondary"
              size="icon"
              className="absolute top-4 right-4"
              onClick={() => setPreviewImage(null)}
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
