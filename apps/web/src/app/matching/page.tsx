"use client";

import { useState, useMemo, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { apiClient } from "@/lib/api-client";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Search,
  Check,
  X,
  Package,
  Loader2,
  Image as ImageIcon,
  Link,
  Filter,
  AlertCircle,
} from "lucide-react";
import { cn } from "@/lib/utils";

const CANDIDATES_PAGE_SIZE = 50;

export default function MatchingPage() {
  const queryClient = useQueryClient();
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedProductId, setSelectedProductId] = useState<string | null>(null);
  const [selectedCandidates, setSelectedCandidates] = useState<Set<string>>(new Set());
  const [minSimilarity, setMinSimilarity] = useState(0.5);
  const [debouncedMinSimilarity, setDebouncedMinSimilarity] = useState(0.5);
  const [candidatesPage, setCandidatesPage] = useState(1);

  // Debounce similarity slider (500ms delay)
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedMinSimilarity(minSimilarity);
    }, 500);
    return () => clearTimeout(timer);
  }, [minSimilarity]);

  // Reset page when selecting a new product or changing similarity
  useEffect(() => {
    setCandidatesPage(1);
  }, [selectedProductId, debouncedMinSimilarity]);

  // Fetch products
  const { data: productsData, isLoading: isLoadingProducts } = useQuery({
    queryKey: ["products-for-matching", searchQuery],
    queryFn: () =>
      apiClient.getProducts({
        limit: 100,
        search: searchQuery || undefined,
      }),
  });

  // Fetch candidates for selected product
  const { data: candidatesData, isLoading: isLoadingCandidates } = useQuery({
    queryKey: ["product-candidates", selectedProductId, debouncedMinSimilarity],
    queryFn: () =>
      selectedProductId
        ? apiClient.getProductCandidates(selectedProductId, {
            min_similarity: debouncedMinSimilarity,
            include_matched: false,
            limit: 500,
          })
        : null,
    enabled: !!selectedProductId,
  });

  // Auto-select barcode matches when candidates load
  useEffect(() => {
    if (candidatesData?.candidates) {
      const barcodeMatches = candidatesData.candidates
        .filter((c) => c.match_type === "barcode" || c.match_type === "both")
        .map((c) => c.id);
      if (barcodeMatches.length > 0) {
        setSelectedCandidates(new Set(barcodeMatches));
      }
    }
  }, [candidatesData?.candidates]);

  // Client-side pagination
  const paginatedCandidates = useMemo(() => {
    if (!candidatesData?.candidates) return [];
    const start = (candidatesPage - 1) * CANDIDATES_PAGE_SIZE;
    return candidatesData.candidates.slice(start, start + CANDIDATES_PAGE_SIZE);
  }, [candidatesData?.candidates, candidatesPage]);

  const totalCandidatesPages = Math.ceil(
    (candidatesData?.candidates.length || 0) / CANDIDATES_PAGE_SIZE
  );

  // Bulk match mutation
  const matchMutation = useMutation({
    mutationFn: () => {
      if (!selectedProductId || selectedCandidates.size === 0) {
        throw new Error("No product or candidates selected");
      }
      const cutoutIds = Array.from(selectedCandidates);
      const similarities = cutoutIds.map((id) => {
        const candidate = candidatesData?.candidates.find((c) => c.id === id);
        return candidate?.similarity || 0;
      });
      return apiClient.bulkMatchCutouts(selectedProductId, cutoutIds, similarities);
    },
    onSuccess: (data) => {
      toast.success(`Matched ${data.matched_count} cutouts to product`);
      setSelectedCandidates(new Set());
      queryClient.invalidateQueries({ queryKey: ["product-candidates"] });
      queryClient.invalidateQueries({ queryKey: ["cutouts"] });
      queryClient.invalidateQueries({ queryKey: ["cutout-stats"] });
    },
    onError: (error) => {
      toast.error(`Match failed: ${error.message}`);
    },
  });

  const handleSelectAll = () => {
    if (!candidatesData?.candidates) return;
    const allIds = new Set(candidatesData.candidates.map((c) => c.id));
    setSelectedCandidates(allIds);
  };

  const handleSelectBarcodeMatches = () => {
    if (!candidatesData?.candidates) return;
    const barcodeIds = candidatesData.candidates
      .filter((c) => c.match_type === "barcode" || c.match_type === "both")
      .map((c) => c.id);
    setSelectedCandidates(new Set(barcodeIds));
  };

  const handleDeselectAll = () => {
    setSelectedCandidates(new Set());
  };

  const toggleCandidate = (id: string) => {
    setSelectedCandidates((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const selectedProduct = productsData?.items.find((p) => p.id === selectedProductId);
  const hasProductEmbedding = candidatesData?.has_product_embedding ?? false;
  const isSliderDebouncing = minSimilarity !== debouncedMinSimilarity;

  return (
    <div className="h-[calc(100vh-8rem)] flex flex-col gap-4">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold">Match</h1>
        <p className="text-muted-foreground">
          Match cutout images to products using barcode and similarity
        </p>
      </div>

      {/* Main Content */}
      <div className="flex gap-4 flex-1 min-h-0">
        {/* Left: Product List */}
        <Card className="w-80 flex flex-col">
          <div className="p-4 border-b">
            <CardTitle className="text-sm mb-2">Select Product</CardTitle>
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search by barcode or name..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10"
              />
            </div>
          </div>
          <CardContent className="flex-1 overflow-auto p-2">
            {isLoadingProducts ? (
              <div className="space-y-2">
                {[...Array(5)].map((_, i) => (
                  <Skeleton key={i} className="h-16" />
                ))}
              </div>
            ) : (
              <div className="space-y-1">
                {productsData?.items.map((product) => (
                  <div
                    key={product.id}
                    className={cn(
                      "flex gap-2 p-2 rounded-lg cursor-pointer border-2 transition-all",
                      selectedProductId === product.id
                        ? "border-primary bg-primary/5"
                        : "border-transparent hover:bg-muted"
                    )}
                    onClick={() => {
                      setSelectedProductId(product.id);
                      setSelectedCandidates(new Set());
                    }}
                  >
                    <div className="w-10 h-10 rounded overflow-hidden bg-muted flex-shrink-0">
                      {product.primary_image_url ? (
                        <img
                          src={product.primary_image_url}
                          alt="Product"
                          className="w-full h-full object-cover"
                        />
                      ) : (
                        <div className="w-full h-full flex items-center justify-center">
                          <Package className="h-4 w-4 text-muted-foreground/50" />
                        </div>
                      )}
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="font-mono text-[10px] text-muted-foreground">{product.barcode}</p>
                      <p className="text-xs font-medium truncate">{product.brand_name}</p>
                      <p className="text-[10px] text-muted-foreground truncate">{product.product_name}</p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Right: Candidates */}
        <Card className="flex-1 flex flex-col min-w-0">
          <div className="p-4 border-b">
            {/* Product Info Header */}
            <div className="flex items-start gap-6">
              {/* Large Product Image */}
              {selectedProduct && (
                <div className="flex-shrink-0">
                  <div className="w-32 h-32 rounded-xl overflow-hidden border-2 shadow-lg">
                    {selectedProduct.primary_image_url ? (
                      <img
                        src={selectedProduct.primary_image_url}
                        alt="Product"
                        className="w-full h-full object-cover"
                      />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center bg-muted">
                        <Package className="h-10 w-10 text-muted-foreground/50" />
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Product Details & Controls */}
              <div className="flex-1 min-w-0">
                <CardTitle className="text-lg mb-1">
                  {selectedProduct ? (
                    <>{selectedProduct.brand_name} - {selectedProduct.product_name}</>
                  ) : (
                    "Select a product to see candidates"
                  )}
                </CardTitle>

                {selectedProduct && (
                  <p className="text-sm text-muted-foreground font-mono mb-3">
                    {selectedProduct.barcode}
                  </p>
                )}

                {candidatesData && (
                  <div className="flex items-center gap-4 mb-3">
                    <Badge variant="outline" className="gap-1">
                      <span className="font-semibold">{candidatesData.barcode_match_count}</span> UPC matches
                    </Badge>
                    <Badge variant="outline" className="gap-1">
                      <span className="font-semibold">{candidatesData.similarity_match_count}</span> similarity matches
                    </Badge>
                    {hasProductEmbedding ? (
                      <Badge className="bg-green-100 text-green-700">Embedding OK</Badge>
                    ) : (
                      <Badge variant="destructive">No Embedding</Badge>
                    )}
                  </div>
                )}

                {/* Similarity Threshold Slider */}
                {selectedProductId && (
                  <div className="flex items-center gap-4 p-3 bg-muted/50 rounded-lg">
                    <div className="flex items-center gap-2 min-w-[140px]">
                      <Filter className="h-4 w-4 text-muted-foreground" />
                      <span className="text-sm font-medium">Min Similarity:</span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="100"
                      value={minSimilarity * 100}
                      onChange={(e) => setMinSimilarity(Number(e.target.value) / 100)}
                      className="flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-primary"
                    />
                    <span className="text-sm font-bold min-w-[50px] text-right flex items-center gap-1">
                      {(minSimilarity * 100).toFixed(0)}%
                      {isSliderDebouncing && <Loader2 className="h-3 w-3 animate-spin" />}
                    </span>
                  </div>
                )}
              </div>
            </div>

            {/* Action Buttons */}
            {selectedProductId && (
              <div className="flex items-center gap-3 mt-4 pt-4 border-t">
                <Button size="sm" variant="outline" onClick={handleSelectBarcodeMatches} disabled={!candidatesData?.barcode_match_count}>
                  <Check className="h-4 w-4 mr-1" />
                  Select UPC
                </Button>
                <Button size="sm" variant="outline" onClick={handleSelectAll} disabled={!candidatesData?.candidates.length}>
                  <Check className="h-4 w-4 mr-1" />
                  Select All
                </Button>
                <Button size="sm" variant="outline" onClick={handleDeselectAll} disabled={selectedCandidates.size === 0}>
                  <X className="h-4 w-4 mr-1" />
                  Clear
                </Button>
                <div className="flex-1" />
                <span className="text-sm text-muted-foreground">
                  <span className="font-semibold">{selectedCandidates.size}</span> selected
                </span>
                <Button
                  onClick={() => matchMutation.mutate()}
                  disabled={selectedCandidates.size === 0 || matchMutation.isPending}
                >
                  {matchMutation.isPending ? (
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <Link className="h-4 w-4 mr-2" />
                  )}
                  Match Selected
                </Button>
              </div>
            )}
          </div>

          <CardContent className="flex-1 overflow-auto p-4 flex flex-col">
            {!selectedProductId ? (
              <div className="flex flex-col items-center justify-center h-full text-muted-foreground">
                <Package className="h-16 w-16 mb-4 opacity-30" />
                <p className="text-lg font-medium">Select a product from the left panel</p>
                <p className="text-sm">to see matching cutout candidates</p>
              </div>
            ) : isLoadingCandidates || isSliderDebouncing ? (
              <div className="grid grid-cols-5 xl:grid-cols-6 2xl:grid-cols-8 gap-3">
                {[...Array(16)].map((_, i) => (
                  <Skeleton key={i} className="aspect-square rounded-lg" />
                ))}
              </div>
            ) : candidatesData?.candidates.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full text-muted-foreground">
                <ImageIcon className="h-16 w-16 mb-4 opacity-30" />
                <p className="text-lg font-medium">No matching candidates found</p>
                {!hasProductEmbedding ? (
                  <div className="mt-2 p-4 bg-orange-50 rounded-lg text-center max-w-md">
                    <AlertCircle className="h-5 w-5 text-orange-500 mx-auto mb-2" />
                    <p className="text-sm text-orange-700">
                      This product has no embedding. Go to the Embeddings page to extract embeddings first.
                    </p>
                  </div>
                ) : (
                  <p className="text-sm">
                    Try lowering the similarity threshold ({(minSimilarity * 100).toFixed(0)}%)
                  </p>
                )}
              </div>
            ) : (
              <>
                {/* Candidates Grid */}
                <div className="grid grid-cols-5 xl:grid-cols-6 2xl:grid-cols-8 gap-3 flex-1">
                  {paginatedCandidates.map((candidate) => {
                    const isBarcodeMatch = candidate.match_type === "barcode" || candidate.match_type === "both";
                    const isSelected = selectedCandidates.has(candidate.id);

                    return (
                      <div
                        key={candidate.id}
                        className={cn(
                          "relative aspect-square rounded-lg overflow-hidden cursor-pointer transition-all hover:scale-105",
                          isBarcodeMatch && !isSelected && "border-4 border-green-500 ring-2 ring-green-500/30 shadow-lg",
                          isSelected && "border-4 border-blue-500 ring-2 ring-blue-500/30 shadow-lg",
                          !isBarcodeMatch && !isSelected && "border-2 border-transparent hover:border-primary/50 hover:shadow-md"
                        )}
                        onClick={() => toggleCandidate(candidate.id)}
                      >
                        <img
                          src={candidate.image_url}
                          alt="Candidate"
                          className="w-full h-full object-cover"
                        />

                        {/* Match type badge */}
                        <div className="absolute top-1 left-1">
                          {isBarcodeMatch ? (
                            <Badge className={cn(
                              "text-[10px] px-1.5 font-bold",
                              (candidate.similarity || 0) >= 0.7 ? "bg-green-600" : "bg-orange-500"
                            )}>
                              UPC
                            </Badge>
                          ) : (
                            <Badge className="bg-blue-600 text-[10px] px-1.5">SIM</Badge>
                          )}
                        </div>

                        {/* Similarity score */}
                        {candidate.similarity !== undefined && (
                          <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent px-2 py-1">
                            <p className="text-xs text-white text-center font-medium">
                              {(candidate.similarity * 100).toFixed(0)}%
                            </p>
                          </div>
                        )}

                        {/* Selection indicator */}
                        {isSelected && (
                          <div className="absolute top-1 right-1">
                            <div className="w-6 h-6 bg-blue-500 rounded-full flex items-center justify-center shadow-md">
                              <Check className="h-4 w-4 text-white" />
                            </div>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>

                {/* Pagination */}
                {totalCandidatesPages > 1 && (
                  <div className="flex items-center justify-center gap-4 pt-4 mt-4 border-t">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setCandidatesPage((p) => Math.max(1, p - 1))}
                      disabled={candidatesPage === 1}
                    >
                      Previous
                    </Button>
                    <span className="text-sm text-muted-foreground">
                      Page {candidatesPage} of {totalCandidatesPages} ({candidatesData?.candidates.length || 0} total)
                    </span>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setCandidatesPage((p) => Math.min(totalCandidatesPages, p + 1))}
                      disabled={candidatesPage === totalCandidatesPages}
                    >
                      Next
                    </Button>
                  </div>
                )}
              </>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
