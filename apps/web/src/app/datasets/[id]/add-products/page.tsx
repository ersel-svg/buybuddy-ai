"use client";

import { useState, useMemo, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useParams, useRouter } from "next/navigation";
import { toast } from "sonner";
import Link from "next/link";
import { apiClient } from "@/lib/api-client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import {
  FilterDrawer,
  FilterTrigger,
  useFilterState,
  FilterSection,
} from "@/components/filters/filter-drawer";
import { JobProgressModal } from "@/components/common/job-progress-modal";
import {
  ArrowLeft,
  Plus,
  Search,
  Loader2,
  Package,
  CheckSquare,
  Square,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";

export default function AddProductsPage() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();
  const queryClient = useQueryClient();

  // States
  const [searchQuery, setSearchQuery] = useState("");
  const [debouncedSearch, setDebouncedSearch] = useState("");
  const [selectedProducts, setSelectedProducts] = useState<Set<string>>(
    new Set()
  );
  const [filterDrawerOpen, setFilterDrawerOpen] = useState(false);
  const [page, setPage] = useState(1);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const limit = 100;

  // Filter state hook
  const { filterState, setFilter, clearSection, clearAll, getTotalCount } =
    useFilterState();

  // Debounce search
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedSearch(searchQuery);
      setPage(1);
    }, 300);
    return () => clearTimeout(timer);
  }, [searchQuery]);

  // Reset page when filters change
  useEffect(() => {
    setPage(1);
  }, [filterState]);

  // Fetch dataset info
  const { data: dataset } = useQuery({
    queryKey: ["dataset", id],
    queryFn: () => apiClient.getDataset(id),
    enabled: !!id,
  });

  // Convert FilterState to API parameters (same as products page)
  const apiFilters = useMemo(() => {
    const params: Record<string, string | number | boolean | undefined> = {};

    // Helper to convert Set to comma-separated string
    const setToString = (key: string): string | undefined => {
      const filter = filterState[key] as Set<string> | undefined;
      if (filter?.size) {
        return Array.from(filter).join(",");
      }
      return undefined;
    };

    // Checkbox filters -> comma-separated strings
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

    // Boolean filters
    const booleanKeys = [
      "hasVideo",
      "hasImage",
      "hasNutrition",
      "hasDescription",
      "hasPrompt",
      "hasIssues",
    ];
    const booleanApiKeys = [
      "has_video",
      "has_image",
      "has_nutrition",
      "has_description",
      "has_prompt",
      "has_issues",
    ];
    booleanKeys.forEach((key, i) => {
      const value = filterState[key] as boolean | undefined;
      if (value !== undefined) {
        params[booleanApiKeys[i]] = value;
      }
    });

    // Range filters
    const frameRange = filterState["frameCount"] as
      | { min: number; max: number }
      | undefined;
    if (frameRange) {
      params.frame_count_min = frameRange.min;
      params.frame_count_max = frameRange.max;
    }

    const visibilityRange = filterState["visibilityScore"] as
      | { min: number; max: number }
      | undefined;
    if (visibilityRange) {
      params.visibility_score_min = visibilityRange.min;
      params.visibility_score_max = visibilityRange.max;
    }

    return params;
  }, [filterState]);

  // Fetch filter options - cascading based on current filters, excluding dataset products
  const { data: filterOptions } = useQuery({
    queryKey: ["filter-options", { ...apiFilters, exclude_dataset_id: id }],
    queryFn: () =>
      apiClient.getFilterOptions({
        ...apiFilters,
        exclude_dataset_id: id,
      }),
    staleTime: 30000,
    placeholderData: (previousData) => previousData,
  });

  // Fetch products with pagination and filters (excluding products already in dataset)
  const {
    data: productsData,
    isLoading: isLoadingProducts,
    error: productsError,
  } = useQuery({
    queryKey: [
      "products-for-dataset",
      id,
      { page, search: debouncedSearch, ...apiFilters },
    ],
    queryFn: () =>
      apiClient.getProducts({
        page,
        limit,
        search: debouncedSearch || undefined,
        exclude_dataset_id: id,
        ...apiFilters,
      }),
  });

  // Add products mutation
  const addProductsMutation = useMutation({
    mutationFn: (productIds: string[]) =>
      apiClient.addProductsToDataset(id, productIds),
    onSuccess: (result) => {
      // If backend returns job_id, track it with modal
      if (result.job_id) {
        setActiveJobId(result.job_id);
        toast.success("Background job started for adding products");
      } else {
        // Small batch completed synchronously
        toast.success(`Added ${result.added_count} products to dataset`);
        queryClient.invalidateQueries({ queryKey: ["dataset", id] });
        router.push(`/datasets/${id}`);
      }
    },
    onError: () => {
      toast.error("Failed to add products");
    },
  });

  // Toggle product selection
  const toggleProduct = (productId: string) => {
    const newSet = new Set(selectedProducts);
    if (newSet.has(productId)) {
      newSet.delete(productId);
    } else {
      newSet.add(productId);
    }
    setSelectedProducts(newSet);
  };

  // Products list
  const products = productsData?.items || [];
  const totalProducts = productsData?.total || 0;
  const totalPages = Math.ceil(totalProducts / limit);

  // Select/Clear helpers
  const selectAllVisible = () => {
    const newSet = new Set(selectedProducts);
    products.forEach((p) => newSet.add(p.id));
    setSelectedProducts(newSet);
  };

  const clearAllSelected = () => {
    setSelectedProducts(new Set());
  };

  // Build filter sections from options (same structure as products page)
  const filterSections: FilterSection[] = useMemo(() => {
    if (!filterOptions) return [];

    return [
      // Status
      {
        id: "status",
        label: "Status",
        type: "checkbox",
        options: filterOptions.status,
        defaultExpanded: true,
      },
      // Category
      {
        id: "category",
        label: "Category",
        type: "checkbox",
        options: filterOptions.category,
        searchable: true,
        defaultExpanded: true,
      },
      // Brand
      {
        id: "brand",
        label: "Brand",
        type: "checkbox",
        options: filterOptions.brand,
        searchable: true,
        defaultExpanded: true,
      },
      // Sub Brand
      {
        id: "subBrand",
        label: "Sub Brand",
        type: "checkbox",
        options: filterOptions.subBrand,
        searchable: true,
        defaultExpanded: false,
      },
      // Product Name
      {
        id: "productName",
        label: "Product Name",
        type: "checkbox",
        options: filterOptions.productName,
        searchable: true,
        defaultExpanded: false,
      },
      // Variant / Flavor
      {
        id: "flavor",
        label: "Variant / Flavor",
        type: "checkbox",
        options: filterOptions.flavor,
        searchable: true,
        defaultExpanded: false,
      },
      // Container Type
      {
        id: "container",
        label: "Container Type",
        type: "checkbox",
        options: filterOptions.container,
        defaultExpanded: false,
      },
      // Net Quantity
      {
        id: "netQuantity",
        label: "Net Quantity",
        type: "checkbox",
        options: filterOptions.netQuantity,
        searchable: true,
        defaultExpanded: false,
      },
      // Pack Type
      {
        id: "packType",
        label: "Pack Type",
        type: "checkbox",
        options: filterOptions.packType,
        defaultExpanded: false,
      },
      // Manufacturer Country
      {
        id: "country",
        label: "Manufacturer Country",
        type: "checkbox",
        options: filterOptions.country,
        searchable: true,
        defaultExpanded: false,
      },
      // Claims
      {
        id: "claims",
        label: "Product Claims",
        type: "checkbox",
        options: filterOptions.claims,
        searchable: true,
        defaultExpanded: false,
      },
      // Issue Types
      {
        id: "issueTypes",
        label: "Issue Types",
        type: "checkbox",
        options: filterOptions.issueTypes,
        searchable: true,
        defaultExpanded: false,
      },
      // Frame Count Range
      {
        id: "frameCount",
        label: "Frame Count",
        type: "range",
        min: filterOptions.frameCount.min,
        max: filterOptions.frameCount.max || 100,
        step: 1,
        defaultExpanded: false,
      },
      // Visibility Score Range
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
      // Has Video
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
      // Has Image
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
      // Has Nutrition Facts
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
      // Has Marketing Description
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
      // Has Grounding Prompt
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
      // Has Issues
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

  const activeFilterCount = getTotalCount();
  const hasActiveFilters = activeFilterCount > 0 || debouncedSearch;

  return (
    <div className="h-[calc(100vh-4rem)] flex flex-col">
      {/* Header */}
      <div className="border-b bg-card px-6 py-4 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-4">
          <Link href={`/datasets/${id}`}>
            <Button variant="ghost" size="icon">
              <ArrowLeft className="h-4 w-4" />
            </Button>
          </Link>
          <div>
            <h1 className="text-xl font-bold">Add Products to Dataset</h1>
            <p className="text-sm text-gray-500">
              {dataset?.name || "Loading..."} - Select products to add
            </p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          {selectedProducts.size > 0 && (
            <Badge variant="secondary" className="text-sm px-3 py-1">
              {selectedProducts.size} selected
            </Badge>
          )}
          <Button
            variant="outline"
            onClick={() => router.push(`/datasets/${id}`)}
          >
            Cancel
          </Button>
          <Button
            onClick={() =>
              addProductsMutation.mutate(Array.from(selectedProducts))
            }
            disabled={
              selectedProducts.size === 0 || addProductsMutation.isPending
            }
          >
            {addProductsMutation.isPending ? (
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            ) : (
              <Plus className="h-4 w-4 mr-2" />
            )}
            Add {selectedProducts.size || ""} Product
            {selectedProducts.size !== 1 ? "s" : ""}
          </Button>
        </div>
      </div>

      {/* Toolbar */}
      <div className="px-6 py-3 border-b bg-gray-50/50 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-3">
          {/* Search */}
          <div className="relative w-80">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
            <Input
              placeholder="Search barcode, name, brand..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10 h-9"
            />
          </div>

          {/* Filter Trigger */}
          <FilterTrigger
            onClick={() => setFilterDrawerOpen(true)}
            activeCount={activeFilterCount}
          />

          {/* Selection actions */}
          <div className="flex items-center gap-2 ml-2">
            <Button
              variant="outline"
              size="sm"
              onClick={selectAllVisible}
              className="h-8"
              disabled={products.length === 0}
            >
              <CheckSquare className="h-3.5 w-3.5 mr-1.5" />
              Select Page
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={clearAllSelected}
              disabled={selectedProducts.size === 0}
              className="h-8"
            >
              <Square className="h-3.5 w-3.5 mr-1.5" />
              Clear
            </Button>
          </div>
        </div>

        {/* Stats */}
        <div className="flex items-center gap-4 text-sm text-gray-500">
          <span>
            Showing {products.length} of {totalProducts} products
          </span>
          {selectedProducts.size > 0 && (
            <Badge className="bg-blue-600 text-white">
              {selectedProducts.size} selected
            </Badge>
          )}
        </div>
      </div>

      {/* Products Grid */}
      <div className="flex-1 overflow-y-auto p-6">
        {productsError ? (
          <div className="flex flex-col items-center justify-center h-64 text-red-500">
            <p className="font-medium text-lg">Error loading products</p>
            <p className="text-sm mt-1">
              {productsError instanceof Error
                ? productsError.message
                : JSON.stringify(productsError)}
            </p>
          </div>
        ) : isLoadingProducts ? (
          <div className="flex items-center justify-center h-64">
            <Loader2 className="h-8 w-8 animate-spin text-gray-400" />
          </div>
        ) : products.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-64 text-gray-500">
            <Package className="h-16 w-16 mb-4 opacity-30" />
            <p className="font-medium text-lg">No products found</p>
            <p className="text-sm mt-1">
              {hasActiveFilters
                ? "Try adjusting your filters"
                : "All products are already in this dataset"}
            </p>
            {hasActiveFilters && (
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  clearAll();
                  setSearchQuery("");
                }}
                className="mt-4"
              >
                Clear filters
              </Button>
            )}
          </div>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 2xl:grid-cols-8 gap-4">
            {products.map((product) => (
              <div
                key={product.id}
                className={`relative p-3 border rounded-xl cursor-pointer transition-all hover:shadow-lg ${
                  selectedProducts.has(product.id)
                    ? "border-blue-500 bg-blue-50 ring-2 ring-blue-500"
                    : "bg-card border-gray-200 hover:border-gray-300"
                }`}
                onClick={() => toggleProduct(product.id)}
              >
                {/* Checkbox */}
                <div className="absolute top-2 right-2 z-10">
                  <Checkbox
                    checked={selectedProducts.has(product.id)}
                    onCheckedChange={() => toggleProduct(product.id)}
                  />
                </div>

                {/* Product Image */}
                <div className="w-full aspect-square bg-gray-50 rounded-lg mb-2 flex items-center justify-center overflow-hidden border">
                  {product.primary_image_url ? (
                    <img
                      src={product.primary_image_url}
                      alt=""
                      className="w-full h-full object-contain"
                    />
                  ) : (
                    <Package className="h-10 w-10 text-gray-300" />
                  )}
                </div>

                {/* Product Info */}
                <div className="space-y-1">
                  <p className="font-semibold text-xs line-clamp-1 leading-tight">
                    {product.brand_name || "Unknown"}
                  </p>
                  <p className="text-xs text-gray-600 line-clamp-2 leading-tight min-h-[2rem]">
                    {product.product_name || "-"}
                  </p>
                  <p className="text-[10px] text-gray-400 font-mono truncate">
                    {product.barcode}
                  </p>
                  <div className="flex flex-wrap gap-1 pt-1">
                    <Badge variant="outline" className="text-[10px] px-1.5 py-0">
                      {product.frame_count}f
                    </Badge>
                    {product.category && (
                      <Badge
                        variant="secondary"
                        className="text-[10px] px-1.5 py-0 truncate max-w-[80px]"
                      >
                        {product.category}
                      </Badge>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="px-6 py-4 border-t-2 bg-muted/50 flex items-center justify-between shrink-0 shadow-[0_-2px_10px_rgba(0,0,0,0.05)]">
          <p className="text-base font-medium text-foreground">
            Page {page} of {totalPages}
          </p>
          <div className="flex items-center gap-3">
            <Button
              variant="outline"
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={page === 1}
            >
              <ChevronLeft className="h-4 w-4 mr-1" />
              Previous
            </Button>
            <Button
              variant="outline"
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              disabled={page === totalPages}
            >
              Next
              <ChevronRight className="h-4 w-4 ml-1" />
            </Button>
          </div>
        </div>
      )}

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
        description="Filter products to add to dataset"
      />

      {/* Job Progress Modal */}
      <JobProgressModal
        jobId={activeJobId}
        title="Adding Products to Dataset"
        onClose={() => {
          setActiveJobId(null);
          queryClient.invalidateQueries({ queryKey: ["dataset", id] });
          router.push(`/datasets/${id}`);
        }}
        onComplete={(result) => {
          toast.success(
            `Added ${result?.added || 0} products (${result?.skipped || 0} skipped)`
          );
        }}
        invalidateOnComplete={[["dataset", id]]}
      />
    </div>
  );
}
