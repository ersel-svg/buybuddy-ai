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
  FilterDrawer,
  FilterTrigger,
  useFilterState,
  FilterSection,
} from "@/components/filters/filter-drawer";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Label } from "@/components/ui/label";
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
  ChevronLeft,
  ChevronRight,
  Database,
  Info,
} from "lucide-react";
import { cn } from "@/lib/utils";

const PRODUCTS_PAGE_SIZE = 50;

export default function MatchingPage() {
  const queryClient = useQueryClient();
  const [searchQuery, setSearchQuery] = useState("");
  const [debouncedSearch, setDebouncedSearch] = useState("");
  const [selectedProductId, setSelectedProductId] = useState<string | null>(null);
  const [selectedCandidates, setSelectedCandidates] = useState<Set<string>>(new Set());
  const [minSimilarity, setMinSimilarity] = useState(0.5);
  const [debouncedMinSimilarity, setDebouncedMinSimilarity] = useState(0.5);
  const [candidatesPage, setCandidatesPage] = useState(1);
  const [productsPage, setProductsPage] = useState(1);
  const [filterDrawerOpen, setFilterDrawerOpen] = useState(false);
  const [productCollection, setProductCollection] = useState<string>("");
  const [cutoutCollection, setCutoutCollection] = useState<string>("");
  const [matchTypeFilter, setMatchTypeFilter] = useState<string | null>(null); // null = all, "both", "barcode", "similarity"

  // Filter state hook
  const { filterState, setFilter, clearSection, clearAll, getTotalCount } =
    useFilterState();

  // Debounce search
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedSearch(searchQuery);
      setProductsPage(1);
    }, 300);
    return () => clearTimeout(timer);
  }, [searchQuery]);

  // Reset products page when filters change
  useEffect(() => {
    setProductsPage(1);
  }, [filterState]);

  // Debounce similarity slider (500ms delay)
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedMinSimilarity(minSimilarity);
    }, 500);
    return () => clearTimeout(timer);
  }, [minSimilarity]);

  // Reset candidates page when selecting a new product or changing similarity
  useEffect(() => {
    setCandidatesPage(1);
  }, [selectedProductId, debouncedMinSimilarity]);

  // Convert FilterState to API parameters
  const apiFilters = useMemo(() => {
    const params: Record<string, string | number | boolean | undefined> = {};

    const setToString = (key: string): string | undefined => {
      const filter = filterState[key] as Set<string> | undefined;
      if (filter?.size) {
        return Array.from(filter).join(",");
      }
      return undefined;
    };

    params.status = setToString("status");
    params.category = setToString("category");
    params.brand = setToString("brand");
    params.sub_brand = setToString("subBrand");
    params.product_name = setToString("productName");
    params.variant_flavor = setToString("flavor");
    params.container_type = setToString("container");
    params.net_quantity = setToString("netQuantity");
    params.pack_type = setToString("packType");
    params.manufacturer_country = setToString("country");
    params.claims = setToString("claims");

    const booleanKeys = ["hasVideo", "hasImage", "hasNutrition", "hasDescription", "hasPrompt", "hasIssues"];
    const booleanApiKeys = ["has_video", "has_image", "has_nutrition", "has_description", "has_prompt", "has_issues"];
    booleanKeys.forEach((key, i) => {
      const value = filterState[key] as boolean | undefined;
      if (value !== undefined) {
        params[booleanApiKeys[i]] = value;
      }
    });

    const frameRange = filterState["frameCount"] as { min: number; max: number } | undefined;
    if (frameRange) {
      params.frame_count_min = frameRange.min;
      params.frame_count_max = frameRange.max;
    }

    const visibilityRange = filterState["visibilityScore"] as { min: number; max: number } | undefined;
    if (visibilityRange) {
      params.visibility_score_min = visibilityRange.min;
      params.visibility_score_max = visibilityRange.max;
    }

    return params;
  }, [filterState]);

  // Fetch filter options - cascading based on current filters
  const { data: filterOptions } = useQuery({
    queryKey: ["filter-options", apiFilters],
    queryFn: () => apiClient.getFilterOptions(apiFilters),
    staleTime: 30000,
    placeholderData: (previousData) => previousData,
  });

  // Fetch Qdrant collections for manual selection
  const { data: collections, isLoading: isLoadingCollections, isError: isCollectionsError, error: collectionsError } = useQuery({
    queryKey: ["qdrant-collections"],
    queryFn: () => apiClient.getQdrantCollections(),
    staleTime: 60000,
    retry: 1,
  });

  // Filter collections by type
  // Support multiple naming patterns:
  // - products_* = dedicated product collection
  // - cutouts_* = dedicated cutout collection
  // - embeddings_* = mixed collection (legacy, contains both)
  // - *_Products_* or *_Cutouts_* = custom naming
  // - All_Cutouts, etc. = show in relevant category
  const productCollections = useMemo(() => {
    if (!collections) return [];
    return collections.filter(c => {
      const name = c.name.toLowerCase();
      // Show in product list if: contains 'product', starts with 'embeddings_', or starts with 'products_'
      return name.startsWith("products_") ||
             name.startsWith("embeddings_") ||
             name.includes("product");
    });
  }, [collections]);

  const cutoutCollections = useMemo(() => {
    if (!collections) return [];
    return collections.filter(c => {
      const name = c.name.toLowerCase();
      // Show in cutout list if: contains 'cutout', starts with 'embeddings_', or starts with 'cutouts_'
      return name.startsWith("cutouts_") ||
             name.startsWith("embeddings_") ||
             name.includes("cutout");
    });
  }, [collections]);

  // Auto-select first available collection if none selected
  useEffect(() => {
    if (!productCollection && productCollections.length > 0) {
      setProductCollection(productCollections[0].name);
    }
    if (!cutoutCollection && cutoutCollections.length > 0) {
      setCutoutCollection(cutoutCollections[0].name);
    }
  }, [productCollections, cutoutCollections, productCollection, cutoutCollection]);

  // Reset products page when collection changes
  useEffect(() => {
    setProductsPage(1);
    setSelectedProductId(null);
    setSelectedCandidates(new Set());
    setCandidatesPage(1);
  }, [productCollection]);

  // Reset candidates page when product or filter changes
  useEffect(() => {
    setCandidatesPage(1);
  }, [selectedProductId, debouncedMinSimilarity, matchTypeFilter]);

  // Reset filter when product changes
  useEffect(() => {
    setMatchTypeFilter(null);
  }, [selectedProductId]);

  // Fetch products from selected collection (NEW: collection-based)
  const { data: collectionProductsData, isLoading: isLoadingCollectionProducts } = useQuery({
    queryKey: ["collection-products", productCollection, { page: productsPage, search: debouncedSearch, ...apiFilters }],
    queryFn: () =>
      apiClient.getCollectionProducts(productCollection, {
        page: productsPage,
        limit: PRODUCTS_PAGE_SIZE,
        search: debouncedSearch || undefined,
        ...apiFilters,
      }),
    enabled: !!productCollection,
  });

  // Products data from collection
  const productsData = collectionProductsData ? {
    items: collectionProductsData.products,
    total: collectionProductsData.total_count,
    page: collectionProductsData.page,
    limit: collectionProductsData.limit,
  } : undefined;

  const isLoadingProducts = isLoadingCollectionProducts;

  // Candidates pagination
  const CANDIDATES_PAGE_SIZE = 100;

  // Fetch candidates for selected product
  const { data: candidatesData, isLoading: isLoadingCandidates, isError: isCandidatesError, error: candidatesError } = useQuery({
    queryKey: ["product-candidates", selectedProductId, debouncedMinSimilarity, productCollection, cutoutCollection, matchTypeFilter],
    queryFn: async () => {
      if (!selectedProductId) return null;
      const result = await apiClient.getProductCandidates(selectedProductId, {
        min_similarity: debouncedMinSimilarity,
        include_matched: false,
        limit: 500,
        match_type_filter: matchTypeFilter || undefined,
        product_collection: productCollection || undefined,
        cutout_collection: cutoutCollection || undefined,
      });
      return result;
    },
    enabled: !!selectedProductId,
    retry: 1,
  });

  // Paginated candidates for display
  const totalCandidates = candidatesData?.candidates?.length || 0;
  const totalCandidatesPages = Math.ceil(totalCandidates / CANDIDATES_PAGE_SIZE);
  const paginatedCandidates = useMemo(() => {
    if (!candidatesData?.candidates) return [];
    const start = (candidatesPage - 1) * CANDIDATES_PAGE_SIZE;
    const end = start + CANDIDATES_PAGE_SIZE;
    return candidatesData.candidates.slice(start, end);
  }, [candidatesData?.candidates, candidatesPage]);

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

  const totalProductsPages = Math.ceil(
    (productsData?.total || 0) / PRODUCTS_PAGE_SIZE
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
  const activeFilterCount = getTotalCount();

  // Build filter sections
  const filterSections: FilterSection[] = useMemo(() => {
    if (!filterOptions) return [];

    return [
      {
        id: "status",
        label: "Status",
        type: "checkbox",
        options: filterOptions.status,
        defaultExpanded: true,
      },
      {
        id: "category",
        label: "Category",
        type: "checkbox",
        options: filterOptions.category,
        searchable: true,
        defaultExpanded: true,
      },
      {
        id: "brand",
        label: "Brand",
        type: "checkbox",
        options: filterOptions.brand,
        searchable: true,
        defaultExpanded: true,
      },
      {
        id: "subBrand",
        label: "Sub Brand",
        type: "checkbox",
        options: filterOptions.subBrand,
        searchable: true,
        defaultExpanded: false,
      },
      {
        id: "productName",
        label: "Product Name",
        type: "checkbox",
        options: filterOptions.productName,
        searchable: true,
        defaultExpanded: false,
      },
      {
        id: "flavor",
        label: "Variant / Flavor",
        type: "checkbox",
        options: filterOptions.flavor,
        searchable: true,
        defaultExpanded: false,
      },
      {
        id: "container",
        label: "Container Type",
        type: "checkbox",
        options: filterOptions.container,
        defaultExpanded: false,
      },
      {
        id: "netQuantity",
        label: "Net Quantity",
        type: "checkbox",
        options: filterOptions.netQuantity,
        searchable: true,
        defaultExpanded: false,
      },
      {
        id: "packType",
        label: "Pack Type",
        type: "checkbox",
        options: filterOptions.packType,
        defaultExpanded: false,
      },
      {
        id: "country",
        label: "Manufacturer Country",
        type: "checkbox",
        options: filterOptions.country,
        searchable: true,
        defaultExpanded: false,
      },
      {
        id: "claims",
        label: "Product Claims",
        type: "checkbox",
        options: filterOptions.claims,
        searchable: true,
        defaultExpanded: false,
      },
      {
        id: "frameCount",
        label: "Frame Count",
        type: "range",
        min: filterOptions.frameCount.min,
        max: filterOptions.frameCount.max || 100,
        step: 1,
        defaultExpanded: false,
      },
      {
        id: "visibilityScore",
        label: "Visibility Score",
        type: "range",
        min: filterOptions.visibilityScore.min,
        max: filterOptions.visibilityScore.max || 100,
        step: 1,
        unit: "%",
        defaultExpanded: false,
      },
      {
        id: "hasVideo",
        label: "Video",
        type: "boolean",
        trueLabel: "Has Video",
        falseLabel: "No Video",
        trueCount: filterOptions.hasVideo.trueCount,
        falseCount: filterOptions.hasVideo.falseCount,
        defaultExpanded: false,
      },
      {
        id: "hasImage",
        label: "Primary Image",
        type: "boolean",
        trueLabel: "Has Image",
        falseLabel: "No Image",
        trueCount: filterOptions.hasImage.trueCount,
        falseCount: filterOptions.hasImage.falseCount,
        defaultExpanded: false,
      },
      {
        id: "hasNutrition",
        label: "Nutrition Facts",
        type: "boolean",
        trueLabel: "Has Nutrition Info",
        falseLabel: "No Nutrition Info",
        trueCount: filterOptions.hasNutrition.trueCount,
        falseCount: filterOptions.hasNutrition.falseCount,
        defaultExpanded: false,
      },
      {
        id: "hasDescription",
        label: "Marketing Description",
        type: "boolean",
        trueLabel: "Has Description",
        falseLabel: "No Description",
        trueCount: filterOptions.hasDescription.trueCount,
        falseCount: filterOptions.hasDescription.falseCount,
        defaultExpanded: false,
      },
      {
        id: "hasPrompt",
        label: "Grounding Prompt",
        type: "boolean",
        trueLabel: "Has Prompt",
        falseLabel: "No Prompt",
        trueCount: filterOptions.hasPrompt.trueCount,
        falseCount: filterOptions.hasPrompt.falseCount,
        defaultExpanded: false,
      },
      {
        id: "hasIssues",
        label: "Has Issues",
        type: "boolean",
        trueLabel: "Has Issues",
        falseLabel: "No Issues",
        trueCount: filterOptions.hasIssues.trueCount,
        falseCount: filterOptions.hasIssues.falseCount,
        defaultExpanded: false,
      },
    ];
  }, [filterOptions]);

  const hasCollections = productCollections.length > 0 || cutoutCollections.length > 0;

  return (
    <div className="h-[calc(100vh-8rem)] flex flex-col gap-4">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold">Match</h1>
          <p className="text-muted-foreground">
            Match cutout images to products using barcode and similarity
          </p>
        </div>
      </div>

      {/* Collection Selector - Always visible at top */}
      <Card className="p-4">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Database className="h-5 w-5 text-muted-foreground" />
            <span className="font-medium">Embedding Collections</span>
          </div>
          {isLoadingCollections && <Loader2 className="h-4 w-4 animate-spin" />}
          {!isLoadingCollections && !hasCollections && (
            <Badge variant="outline" className="text-xs text-orange-600 border-orange-300 bg-orange-50">
              <AlertCircle className="h-3 w-3 mr-1" />
              No collections
            </Badge>
          )}
        </div>

        {isCollectionsError ? (
          <div className="flex items-start gap-2 p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">
            <AlertCircle className="h-5 w-5 flex-shrink-0 mt-0.5" />
            <div>
              <p className="font-medium">Failed to load collections</p>
              <p className="text-red-600 mt-1">
                {(collectionsError as Error)?.message || "Check that you're logged in and the API is running."}
              </p>
            </div>
          </div>
        ) : !isLoadingCollections && !hasCollections ? (
          <div className="flex items-start gap-2 p-3 bg-orange-50 border border-orange-200 rounded-lg text-sm text-orange-700">
            <Info className="h-5 w-5 flex-shrink-0 mt-0.5" />
            <div>
              <p className="font-medium">Embedding collections not found</p>
              <p className="text-orange-600 mt-1">
                Go to <span className="font-semibold">Embeddings → Matching tab</span> and run an extraction
                to create <code className="bg-orange-100 px-1 rounded">products_*</code> and <code className="bg-orange-100 px-1 rounded">cutouts_*</code> collections.
                Similarity matching requires embeddings.
              </p>
            </div>
          </div>
        ) : (
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-2">
              <Label className="text-sm text-muted-foreground whitespace-nowrap">Products</Label>
              <Select
                value={productCollection || ""}
                onValueChange={setProductCollection}
                disabled={productCollections.length === 0}
              >
                <SelectTrigger className="w-[220px]">
                  <SelectValue placeholder={productCollections.length > 0 ? "Select collection" : "No collections"} />
                </SelectTrigger>
                <SelectContent>
                  {productCollections.map((c) => (
                    <SelectItem key={c.name} value={c.name}>
                      {c.name} ({c.vectors_count?.toLocaleString()} vectors)
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="flex items-center gap-2">
              <Label className="text-sm text-muted-foreground whitespace-nowrap">Cutouts</Label>
              <Select
                value={cutoutCollection || ""}
                onValueChange={setCutoutCollection}
                disabled={cutoutCollections.length === 0}
              >
                <SelectTrigger className="w-[220px]">
                  <SelectValue placeholder={cutoutCollections.length > 0 ? "Select collection" : "No collections"} />
                </SelectTrigger>
                <SelectContent>
                  {cutoutCollections.map((c) => (
                    <SelectItem key={c.name} value={c.name}>
                      {c.name} ({c.vectors_count?.toLocaleString()} vectors)
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <p className="text-xs text-muted-foreground ml-auto">
              <Info className="h-3 w-3 inline mr-1" />
              Products from the selected collection will be shown below
            </p>
          </div>
        )}
      </Card>

      {/* Main Content */}
      <div className="flex gap-4 flex-1 min-h-0">
        {/* Left: Product List */}
        <Card className="w-80 flex flex-col">
          <div className="p-4 border-b space-y-2">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm">Select Product</CardTitle>
              <FilterTrigger
                onClick={() => setFilterDrawerOpen(true)}
                activeCount={activeFilterCount}
              />
            </div>
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search by barcode or name..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10"
              />
            </div>
            <p className="text-xs text-muted-foreground">
              {productsData?.total || 0} products
            </p>
          </div>
          <CardContent className="flex-1 overflow-auto p-2">
            {!productCollection ? (
              <div className="flex flex-col items-center justify-center h-full text-center p-4">
                <Database className="h-8 w-8 text-muted-foreground/50 mb-2" />
                <p className="text-sm text-muted-foreground">No collection selected</p>
                <p className="text-xs text-muted-foreground mt-1">
                  Select a products collection above
                </p>
              </div>
            ) : isLoadingProducts ? (
              <div className="space-y-2">
                {[...Array(5)].map((_, i) => (
                  <Skeleton key={i} className="h-16" />
                ))}
              </div>
            ) : !productsData?.items?.length ? (
              <div className="flex flex-col items-center justify-center h-full text-center p-4">
                <Package className="h-8 w-8 text-muted-foreground/50 mb-2" />
                <p className="text-sm text-muted-foreground">No products found</p>
                <p className="text-xs text-muted-foreground mt-1">
                  {debouncedSearch ? "Try a different search" : "This collection is empty"}
                </p>
              </div>
            ) : (
              <div className="space-y-1">
                {productsData.items.map((product) => (
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
                      {product.frame_counts && (
                        <div className="flex gap-1.5 mt-0.5">
                          <span className="text-[9px] text-blue-600" title="Synthetic">
                            S:{product.frame_counts.synthetic || 0}
                          </span>
                          <span className="text-[9px] text-green-600" title="Real">
                            R:{product.frame_counts.real || 0}
                          </span>
                          <span className="text-[9px] text-purple-600" title="Augmented">
                            A:{product.frame_counts.augmented || 0}
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
          {/* Products Pagination */}
          {totalProductsPages > 1 && (
            <div className="p-2 border-t flex items-center justify-between text-xs">
              <Button
                variant="ghost"
                size="icon"
                className="h-7 w-7"
                onClick={() => setProductsPage((p) => Math.max(1, p - 1))}
                disabled={productsPage === 1}
              >
                <ChevronLeft className="h-4 w-4" />
              </Button>
              <span className="text-muted-foreground">
                {productsPage} / {totalProductsPages}
              </span>
              <Button
                variant="ghost"
                size="icon"
                className="h-7 w-7"
                onClick={() => setProductsPage((p) => Math.min(totalProductsPages, p + 1))}
                disabled={productsPage === totalProductsPages}
              >
                <ChevronRight className="h-4 w-4" />
              </Button>
            </div>
          )}
        </Card>

        {/* Right: Candidates */}
        <Card className="flex-1 flex flex-col min-w-0 overflow-hidden">
          <div className="flex-1 overflow-auto p-4">
            {/* Product Info Header */}
            <div className="flex items-start gap-6">
              {/* Large Product Image - Extra large for better visibility */}
              {selectedProduct && (
                <div className="flex-shrink-0">
                  <div className="w-64 h-64 rounded-xl overflow-hidden border-2 shadow-lg bg-white">
                    {selectedProduct.primary_image_url ? (
                      <img
                        src={selectedProduct.primary_image_url}
                        alt="Product"
                        className="w-full h-full object-contain"
                      />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center bg-muted">
                        <Package className="h-20 w-20 text-muted-foreground/50" />
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

            {/* Filter Tabs & Action Buttons */}
            {selectedProductId && (
              <div className="mt-4 pt-4 border-t space-y-3">
                {/* Match Type Filter Tabs */}
                <div className="flex items-center gap-2">
                  <span className="text-sm text-muted-foreground mr-2">Show:</span>
                  <div className="flex gap-1 bg-muted p-1 rounded-lg">
                    <Button
                      size="sm"
                      variant={matchTypeFilter === null ? "default" : "ghost"}
                      className="h-7 text-xs"
                      onClick={() => setMatchTypeFilter(null)}
                    >
                      All
                    </Button>
                    <Button
                      size="sm"
                      variant={matchTypeFilter === "both" ? "default" : "ghost"}
                      className="h-7 text-xs"
                      onClick={() => setMatchTypeFilter("both")}
                    >
                      UPC + Similar
                    </Button>
                    <Button
                      size="sm"
                      variant={matchTypeFilter === "barcode" ? "default" : "ghost"}
                      className="h-7 text-xs"
                      onClick={() => setMatchTypeFilter("barcode")}
                    >
                      UPC Only
                    </Button>
                    <Button
                      size="sm"
                      variant={matchTypeFilter === "similarity" ? "default" : "ghost"}
                      className="h-7 text-xs"
                      onClick={() => setMatchTypeFilter("similarity")}
                    >
                      Similar Only
                    </Button>
                  </div>
                </div>

                {/* Selection Buttons */}
                <div className="flex items-center gap-3">
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
                </div>
              </div>
            )}

            {/* Candidates Section */}
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
            ) : isCandidatesError ? (
              <div className="flex flex-col items-center justify-center h-full text-red-600">
                <AlertCircle className="h-16 w-16 mb-4 opacity-50" />
                <p className="text-lg font-medium">Failed to load candidates</p>
                <pre className="text-xs text-red-500 mt-2 max-w-lg text-left bg-red-50 p-2 rounded overflow-auto">
                  {JSON.stringify(candidatesError, null, 2)}
                </pre>
              </div>
            ) : !candidatesData || candidatesData?.candidates?.length === 0 ? (
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
          </div>

          {/* Sticky Footer - Always visible */}
          {selectedProductId && (
            <div className="border-t bg-background p-3 flex items-center justify-between">
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
        </Card>
      </div>

      {/* Filter Drawer */}
      <FilterDrawer
        open={filterDrawerOpen}
        onOpenChange={setFilterDrawerOpen}
        sections={filterSections}
        filterState={filterState}
        onFilterChange={setFilter}
        onClearAll={clearAll}
        onClearSection={clearSection}
        title="Filter Products"
        description="Filter products to match with cutouts"
      />
    </div>
  );
}
