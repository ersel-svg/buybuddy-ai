"use client";

import { useState, useMemo } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useParams, useRouter } from "next/navigation";
import { toast } from "sonner";
import Link from "next/link";
import { apiClient } from "@/lib/api-client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import {
  ArrowLeft,
  Plus,
  Search,
  Loader2,
  Package,
  Filter,
  CheckSquare,
  Square,
  X,
} from "lucide-react";

export default function AddProductsPage() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();
  const queryClient = useQueryClient();

  const [searchQuery, setSearchQuery] = useState("");
  const [selectedProducts, setSelectedProducts] = useState<Set<string>>(
    new Set()
  );

  // Filter states
  const [filterCategory, setFilterCategory] = useState<string>("all");
  const [filterBrand, setFilterBrand] = useState<string>("all");
  const [filterStatus, setFilterStatus] = useState<string>("all");
  const [filterContainerType, setFilterContainerType] = useState<string>("all");

  // Fetch dataset info
  const { data: dataset } = useQuery({
    queryKey: ["dataset", id],
    queryFn: () => apiClient.getDataset(id),
    enabled: !!id,
  });

  // Fetch all products
  const { data: allProducts, isLoading: isLoadingProducts, error: productsError } = useQuery({
    queryKey: ["all-products"],
    queryFn: () => apiClient.getProducts({ limit: 1000 }),
  });

  // Add products mutation
  const addProductsMutation = useMutation({
    mutationFn: (productIds: string[]) =>
      apiClient.addProductsToDataset(id, productIds),
    onSuccess: (result) => {
      toast.success(`Added ${result.added_count} products to dataset`);
      queryClient.invalidateQueries({ queryKey: ["dataset", id] });
      router.push(`/datasets/${id}`);
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

  // Filter out products already in dataset
  const existingProductIds = useMemo(
    () => new Set(dataset?.products?.map((p) => p.id) || []),
    [dataset?.products]
  );

  const productsNotInDataset = useMemo(
    () => (allProducts?.items || []).filter((p) => !existingProductIds.has(p.id)),
    [allProducts?.items, existingProductIds]
  );

  // Extract unique values for filters
  const filterOptions = useMemo(() => {
    const categories = new Set<string>();
    const brands = new Set<string>();
    const statuses = new Set<string>();
    const containerTypes = new Set<string>();

    productsNotInDataset.forEach((p) => {
      if (p.category) categories.add(p.category);
      if (p.brand_name) brands.add(p.brand_name);
      if (p.status) statuses.add(p.status);
      if (p.container_type) containerTypes.add(p.container_type);
    });

    return {
      categories: Array.from(categories).sort(),
      brands: Array.from(brands).sort(),
      statuses: Array.from(statuses).sort(),
      containerTypes: Array.from(containerTypes).sort(),
    };
  }, [productsNotInDataset]);

  // Apply all filters
  const availableProducts = useMemo(() => {
    return productsNotInDataset.filter((p) => {
      // Search filter
      if (searchQuery) {
        const query = searchQuery.toLowerCase();
        const matchesSearch =
          p.barcode?.toLowerCase().includes(query) ||
          p.brand_name?.toLowerCase().includes(query) ||
          p.product_name?.toLowerCase().includes(query);
        if (!matchesSearch) return false;
      }

      // Category filter
      if (filterCategory !== "all" && p.category !== filterCategory) return false;

      // Brand filter
      if (filterBrand !== "all" && p.brand_name !== filterBrand) return false;

      // Status filter
      if (filterStatus !== "all" && p.status !== filterStatus) return false;

      // Container type filter
      if (filterContainerType !== "all" && p.container_type !== filterContainerType) return false;

      return true;
    });
  }, [productsNotInDataset, searchQuery, filterCategory, filterBrand, filterStatus, filterContainerType]);

  // Select/Clear helpers
  const selectAllVisible = () => {
    const newSet = new Set(selectedProducts);
    availableProducts.forEach((p) => newSet.add(p.id));
    setSelectedProducts(newSet);
  };

  const clearAllSelected = () => {
    setSelectedProducts(new Set());
  };

  const clearFilters = () => {
    setSearchQuery("");
    setFilterCategory("all");
    setFilterBrand("all");
    setFilterStatus("all");
    setFilterContainerType("all");
  };

  const hasActiveFilters =
    searchQuery ||
    filterCategory !== "all" ||
    filterBrand !== "all" ||
    filterStatus !== "all" ||
    filterContainerType !== "all";

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
            onClick={() => addProductsMutation.mutate(Array.from(selectedProducts))}
            disabled={selectedProducts.size === 0 || addProductsMutation.isPending}
          >
            {addProductsMutation.isPending ? (
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            ) : (
              <Plus className="h-4 w-4 mr-2" />
            )}
            Add {selectedProducts.size || ""} Product{selectedProducts.size !== 1 ? "s" : ""}
          </Button>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex flex-1 min-h-0">
        {/* Filters Sidebar */}
        <div className="w-72 border-r bg-gray-50/50 p-6 flex flex-col gap-4 overflow-y-auto shrink-0">
          <div className="flex items-center justify-between">
            <h3 className="font-semibold flex items-center gap-2">
              <Filter className="h-4 w-4" />
              Filters
            </h3>
            {hasActiveFilters && (
              <Button
                variant="ghost"
                size="sm"
                onClick={clearFilters}
                className="h-7 text-xs"
              >
                <X className="h-3 w-3 mr-1" />
                Clear
              </Button>
            )}
          </div>

          <Separator />

          {/* Search */}
          <div className="space-y-2">
            <Label className="text-xs font-medium text-gray-600">Search</Label>
            <div className="relative">
              <Search className="absolute left-2.5 top-1/2 transform -translate-y-1/2 h-3.5 w-3.5 text-gray-400" />
              <Input
                placeholder="Barcode, name, brand..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-8 h-9 text-sm"
              />
            </div>
          </div>

          {/* Category Filter */}
          <div className="space-y-2">
            <Label className="text-xs font-medium text-gray-600">Category</Label>
            <Select value={filterCategory} onValueChange={setFilterCategory}>
              <SelectTrigger className="h-9 text-sm">
                <SelectValue placeholder="All categories" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All categories ({filterOptions.categories.length})</SelectItem>
                {filterOptions.categories.map((cat) => (
                  <SelectItem key={cat} value={cat}>
                    {cat}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Brand Filter */}
          <div className="space-y-2">
            <Label className="text-xs font-medium text-gray-600">Brand</Label>
            <Select value={filterBrand} onValueChange={setFilterBrand}>
              <SelectTrigger className="h-9 text-sm">
                <SelectValue placeholder="All brands" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All brands ({filterOptions.brands.length})</SelectItem>
                {filterOptions.brands.map((brand) => (
                  <SelectItem key={brand} value={brand}>
                    {brand}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Status Filter */}
          <div className="space-y-2">
            <Label className="text-xs font-medium text-gray-600">Status</Label>
            <Select value={filterStatus} onValueChange={setFilterStatus}>
              <SelectTrigger className="h-9 text-sm">
                <SelectValue placeholder="All statuses" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All statuses</SelectItem>
                {filterOptions.statuses.map((status) => (
                  <SelectItem key={status} value={status}>
                    {status.replace("_", " ")}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Container Type Filter */}
          <div className="space-y-2">
            <Label className="text-xs font-medium text-gray-600">Container Type</Label>
            <Select value={filterContainerType} onValueChange={setFilterContainerType}>
              <SelectTrigger className="h-9 text-sm">
                <SelectValue placeholder="All types" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All types</SelectItem>
                {filterOptions.containerTypes.map((type) => (
                  <SelectItem key={type} value={type}>
                    {type}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <Separator />

          {/* Stats */}
          <div className="bg-card rounded-lg p-4 border space-y-2">
            <p className="text-xs text-gray-500">
              Total available: <span className="font-semibold text-gray-900">{productsNotInDataset.length}</span>
            </p>
            <p className="text-xs text-gray-500">
              Filtered: <span className="font-semibold text-gray-900">{availableProducts.length}</span>
            </p>
            <p className="text-xs text-blue-600">
              Selected: <span className="font-bold">{selectedProducts.size}</span>
            </p>
          </div>
        </div>

        {/* Products Grid */}
        <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
          {/* Action Bar */}
          <div className="px-6 py-3 border-b bg-card flex items-center justify-between shrink-0">
            <div className="flex items-center gap-3">
              <Button
                variant="outline"
                size="sm"
                onClick={selectAllVisible}
                className="h-8"
              >
                <CheckSquare className="h-3.5 w-3.5 mr-1.5" />
                Select All Visible ({availableProducts.length})
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={clearAllSelected}
                disabled={selectedProducts.size === 0}
                className="h-8"
              >
                <Square className="h-3.5 w-3.5 mr-1.5" />
                Clear Selection
              </Button>
            </div>
            <p className="text-sm text-gray-500">
              Showing {availableProducts.length} of {productsNotInDataset.length} products
            </p>
          </div>

          {/* Scrollable Grid */}
          <div className="flex-1 overflow-y-auto p-6">
            {productsError ? (
              <div className="flex flex-col items-center justify-center h-64 text-red-500">
                <p className="font-medium text-lg">Error loading products</p>
                <p className="text-sm mt-1">
                  {productsError instanceof Error ? productsError.message : JSON.stringify(productsError)}
                </p>
                <pre className="text-xs mt-2 bg-red-50 p-2 rounded max-w-md overflow-auto">
                  {JSON.stringify(productsError, null, 2)}
                </pre>
              </div>
            ) : isLoadingProducts ? (
              <div className="flex items-center justify-center h-64">
                <Loader2 className="h-8 w-8 animate-spin text-gray-400" />
              </div>
            ) : availableProducts.length === 0 ? (
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
                    onClick={clearFilters}
                    className="mt-4"
                  >
                    Clear filters
                  </Button>
                )}
              </div>
            ) : (
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 2xl:grid-cols-7 gap-4">
                {availableProducts.map((product) => (
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
                          <Badge variant="secondary" className="text-[10px] px-1.5 py-0 truncate max-w-[80px]">
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
        </div>
      </div>
    </div>
  );
}
