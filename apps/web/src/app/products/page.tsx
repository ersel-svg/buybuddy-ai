"use client";

import { useState, useMemo, useEffect, useCallback, Suspense } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useSearchParams, useRouter } from "next/navigation";
import { toast } from "sonner";
import Link from "next/link";
import { apiClient, type ExportFilters } from "@/lib/api-client";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
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
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  FilterDrawer,
  FilterTrigger,
  ActiveFilterChips,
  useFilterStateWithURL,
  type FilterSection,
} from "@/components/filters/filter-drawer";
import {
  Search,
  MoreHorizontal,
  Download,
  FileJson,
  FileSpreadsheet,
  Trash2,
  FolderPlus,
  Loader2,
  Package,
  RefreshCw,
  ArrowUpDown,
  ArrowUp,
  ArrowDown,
  RotateCcw,
} from "lucide-react";
import type { ProductStatus, Product, Dataset } from "@/types";

// Sortable columns
type SortColumn =
  | "barcode"
  | "brand_name"
  | "sub_brand"
  | "product_name"
  | "variant_flavor"
  | "category"
  | "container_type"
  | "net_quantity"
  | "pack_type"
  | "manufacturer_country"
  | "status"
  | "synthetic_count"
  | "real_count"
  | "augmented_count"
  | "created_at"
  | "updated_at";

type SortDirection = "asc" | "desc";

function ProductsPageContent() {
  const queryClient = useQueryClient();
  const searchParams = useSearchParams();
  const router = useRouter();
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(1);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [selectAllFilteredMode, setSelectAllFilteredMode] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [filterDrawerOpen, setFilterDrawerOpen] = useState(false);
  const [datasetDialogOpen, setDatasetDialogOpen] = useState(false);
  const [datasetMode, setDatasetMode] = useState<"existing" | "new">("existing");
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>("");
  const [newDatasetName, setNewDatasetName] = useState("");
  const [newDatasetDescription, setNewDatasetDescription] = useState("");

  // Read sorting state from URL query parameters
  const sortColumn = useMemo(() => {
    const col = searchParams.get("sortBy");
    if (col && ["barcode", "brand_name", "sub_brand", "product_name", "variant_flavor", "category", "container_type", "net_quantity", "pack_type", "manufacturer_country", "status", "synthetic_count", "real_count", "augmented_count", "created_at", "updated_at"].includes(col)) {
      return col as SortColumn;
    }
    return null;
  }, [searchParams]);

  const sortDirection = useMemo(() => {
    const dir = searchParams.get("sortDir");
    return dir === "desc" ? "desc" : "asc";
  }, [searchParams]);

  // Helper function to update URL with new sort parameters
  const updateSortParams = useCallback((column: SortColumn | null, direction: SortDirection) => {
    const params = new URLSearchParams(searchParams.toString());
    if (column) {
      params.set("sortBy", column);
      params.set("sortDir", direction);
    } else {
      params.delete("sortBy");
      params.delete("sortDir");
    }
    router.push(`/products?${params.toString()}`, { scroll: false });
  }, [searchParams, router]);

  // Use URL-synced filter state hook
  const {
    filterState,
    setFilter,
    clearSection,
    clearAll,
    getTotalCount,
  } = useFilterStateWithURL();

  // Convert FilterState to API parameters
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
    const booleanKeys = ["hasVideo", "hasImage", "hasNutrition", "hasDescription", "hasPrompt", "hasIssues"];
    const booleanApiKeys = ["has_video", "has_image", "has_nutrition", "has_description", "has_prompt", "has_issues"];
    booleanKeys.forEach((key, i) => {
      const value = filterState[key] as boolean | undefined;
      if (value !== undefined) {
        params[booleanApiKeys[i]] = value;
      }
    });

    // Range filters
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

  // Map frontend sort columns to backend column names
  const serverSortColumn = useMemo(() => {
    // These columns can be sorted server-side
    const serverSortableColumns: Record<string, string> = {
      barcode: "barcode",
      brand_name: "brand_name",
      sub_brand: "sub_brand",
      product_name: "product_name",
      variant_flavor: "variant_flavor",
      category: "category",
      container_type: "container_type",
      net_quantity: "net_quantity",
      manufacturer_country: "manufacturer_country",
      status: "status",
      created_at: "created_at",
      updated_at: "updated_at",
    };
    return sortColumn ? serverSortableColumns[sortColumn] : undefined;
  }, [sortColumn]);

  // Fetch products with server-side filtering and sorting
  const { data, isLoading, refetch, isFetching } = useQuery({
    queryKey: ["products", { page, search, ...apiFilters, sort_by: serverSortColumn, sort_order: sortDirection, include_frame_counts: true }],
    queryFn: () =>
      apiClient.getProducts({
        page,
        limit: 100,
        search: search || undefined,
        sort_by: serverSortColumn,
        sort_order: sortDirection,
        include_frame_counts: true,
        ...apiFilters,
      }),
  });

  // Reset page and selection mode when filters or sorting changes
  useEffect(() => {
    setPage(1);
    setSelectAllFilteredMode(false);
  }, [apiFilters, serverSortColumn, sortDirection]);

  // Fetch filter options - cascading based on current filters
  const { data: filterOptions, isLoading: isLoadingFilters } = useQuery({
    queryKey: ["filter-options", apiFilters],
    queryFn: () => apiClient.getFilterOptions(apiFilters),
    staleTime: 30000, // Cache for 30 seconds
    placeholderData: (previousData) => previousData, // Keep previous data while loading
  });

  // Fetch datasets for "Add to Dataset" dialog
  const { data: datasets } = useQuery({
    queryKey: ["datasets"],
    queryFn: () => apiClient.getDatasets(),
    enabled: datasetDialogOpen,
  });

  // Build filter sections from filter options API (covers ALL products in database)
  const filterSections: FilterSection[] = useMemo(() => {
    if (!filterOptions) return [];

    const sections: FilterSection[] = [
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
      // Issues Detected
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
        label: "Issues",
        type: "boolean",
        trueLabel: "Has Issues",
        falseLabel: "No Issues",
        trueCount: filterOptions.hasIssues.trueCount,
        falseCount: filterOptions.hasIssues.falseCount,
        defaultExpanded: false,
      },
    ];

    return sections;
  }, [filterOptions]);

  // Handler to remove a single filter value (for chips)
  const handleRemoveFilter = useCallback((sectionId: string, value: string) => {
    const currentValue = filterState[sectionId];

    if (currentValue instanceof Set) {
      // For checkbox filters, find the actual value (not label) and remove it
      const section = filterSections.find(s => s.id === sectionId);
      if (section && section.type === "checkbox") {
        const option = section.options.find(o => o.label === value || o.value === value);
        if (option) {
          const newSet = new Set(currentValue);
          newSet.delete(option.value);
          if (newSet.size === 0) {
            clearSection(sectionId);
          } else {
            setFilter(sectionId, newSet);
          }
        }
      }
    } else {
      // For boolean and range filters, clear the entire section
      clearSection(sectionId);
    }
  }, [filterState, filterSections, setFilter, clearSection]);

  // Convert FilterState to ExportFilters for API calls
  const buildExportFilters = (selectedProductIds?: string[]): ExportFilters => {
    const filters: ExportFilters = {};

    // If specific product IDs are selected
    if (selectedProductIds && selectedProductIds.length > 0) {
      filters.product_ids = selectedProductIds;
    }

    // Search term
    if (search) {
      filters.search = search;
    }

    // Checkbox filters (Sets)
    const setToArray = (key: string): string[] | undefined => {
      const filter = filterState[key] as Set<string> | undefined;
      if (filter?.size) {
        return Array.from(filter);
      }
      return undefined;
    };

    filters.status = setToArray("status");
    filters.category = setToArray("category");
    filters.brand = setToArray("brand");
    filters.sub_brand = setToArray("subBrand");
    filters.product_name = setToArray("productName");
    filters.variant_flavor = setToArray("flavor");
    filters.container_type = setToArray("container");
    filters.net_quantity = setToArray("netQuantity");
    filters.pack_type = setToArray("packType");
    filters.manufacturer_country = setToArray("country");
    filters.claims = setToArray("claims");

    // Boolean filters
    const getBooleanFilter = (key: string): boolean | undefined => {
      const filter = filterState[key] as boolean | undefined;
      return filter;
    };

    filters.has_video = getBooleanFilter("hasVideo");
    filters.has_image = getBooleanFilter("hasImage");
    filters.has_nutrition = getBooleanFilter("hasNutrition");
    filters.has_description = getBooleanFilter("hasDescription");
    filters.has_prompt = getBooleanFilter("hasPrompt");
    filters.has_issues = getBooleanFilter("hasIssues");

    // Range filters
    const frameCountRange = filterState["frameCount"] as { min: number; max: number } | undefined;
    if (frameCountRange) {
      filters.frame_count_min = frameCountRange.min;
      filters.frame_count_max = frameCountRange.max;
    }

    const visibilityRange = filterState["visibilityScore"] as { min: number; max: number } | undefined;
    if (visibilityRange) {
      filters.visibility_score_min = visibilityRange.min;
      filters.visibility_score_max = visibilityRange.max;
    }

    // Clean up undefined values
    Object.keys(filters).forEach((key) => {
      if (filters[key as keyof ExportFilters] === undefined) {
        delete filters[key as keyof ExportFilters];
      }
    });

    return filters;
  };

  // Client-side filtering with ALL filters
  // Filtering is now done server-side, just apply local filters for fields not supported by API
  const filteredItems = useMemo(() => {
    if (!data?.items) return [];

    // Most filtering is done server-side, only apply client-side filters for unsupported fields
    let items = data.items;

    // SKU Code filter (not supported server-side)
    const skuFilter = filterState["skuCode"] as Set<string> | undefined;
    if (skuFilter?.size) {
      items = items.filter((p) => {
        const sku = p.identifiers?.sku_model_code;
        return sku && skuFilter.has(sku);
      });
    }

    // Issue Types filter (not supported server-side)
    const issueTypesFilter = filterState["issueTypes"] as Set<string> | undefined;
    if (issueTypesFilter?.size) {
      items = items.filter((p) => {
        const issues = p.issues_detected || [];
        return issues.some((issue) => issueTypesFilter.has(issue));
      });
    }

    return items;
  }, [data?.items, filterState]);

  // Sorting logic - server sorts most columns, client sorts frame counts and pack_type
  const sortedItems = useMemo(() => {
    if (!sortColumn) return filteredItems;

    // If sorting by a server-sortable column, data is already sorted from API
    if (serverSortColumn) {
      return filteredItems;
    }

    // Client-side sorting only for columns not supported by server
    return [...filteredItems].sort((a, b) => {
      let aVal: string | number | undefined;
      let bVal: string | number | undefined;

      switch (sortColumn) {
        case "pack_type":
          aVal = a.pack_configuration?.type;
          bVal = b.pack_configuration?.type;
          break;
        case "synthetic_count":
          aVal = a.frame_counts?.synthetic ?? 0;
          bVal = b.frame_counts?.synthetic ?? 0;
          break;
        case "real_count":
          aVal = a.frame_counts?.real ?? 0;
          bVal = b.frame_counts?.real ?? 0;
          break;
        case "augmented_count":
          aVal = a.frame_counts?.augmented ?? 0;
          bVal = b.frame_counts?.augmented ?? 0;
          break;
        default:
          return 0;
      }

      // Handle undefined values - push to end
      if (aVal === undefined && bVal === undefined) return 0;
      if (aVal === undefined) return 1;
      if (bVal === undefined) return -1;

      // Compare
      let result = 0;
      if (typeof aVal === "number" && typeof bVal === "number") {
        result = aVal - bVal;
      } else {
        result = String(aVal).localeCompare(String(bVal));
      }

      return sortDirection === "desc" ? -result : result;
    });
  }, [filteredItems, sortColumn, sortDirection, serverSortColumn]);

  // Handle sort column click
  const handleSort = (column: SortColumn) => {
    if (sortColumn === column) {
      // Toggle direction or clear
      if (sortDirection === "asc") {
        updateSortParams(column, "desc");
      } else {
        updateSortParams(null, "asc");
      }
    } else {
      updateSortParams(column, "asc");
    }
  };

  // Sortable header component
  const SortableHeader = ({
    column,
    children,
    className = "",
  }: {
    column: SortColumn;
    children: React.ReactNode;
    className?: string;
  }) => (
    <TableHead
      className={`cursor-pointer select-none hover:bg-muted/80 transition-colors ${className}`}
      onClick={() => handleSort(column)}
    >
      <div className="flex items-center gap-1">
        {children}
        {sortColumn === column ? (
          sortDirection === "asc" ? (
            <ArrowUp className="h-3 w-3" />
          ) : (
            <ArrowDown className="h-3 w-3" />
          )
        ) : (
          <ArrowUpDown className="h-3 w-3 opacity-30" />
        )}
      </div>
    </TableHead>
  );

  // Bulk selection helpers
  const allSelected = useMemo(() => {
    if (!filteredItems.length) return false;
    return filteredItems.every((p) => selectedIds.has(p.id));
  }, [filteredItems, selectedIds]);

  const someSelected = useMemo(() => {
    if (!filteredItems.length) return false;
    return filteredItems.some((p) => selectedIds.has(p.id)) && !allSelected;
  }, [filteredItems, selectedIds, allSelected]);

  const toggleAll = () => {
    if (allSelected) {
      setSelectedIds(new Set());
    } else {
      setSelectedIds(new Set(filteredItems.map((p) => p.id)));
    }
  };

  const toggleOne = (id: string) => {
    // If in selectAllFilteredMode, exit it when individual selection is made
    if (selectAllFilteredMode) {
      setSelectAllFilteredMode(false);
    }
    const newSet = new Set(selectedIds);
    if (newSet.has(id)) {
      newSet.delete(id);
    } else {
      newSet.add(id);
    }
    setSelectedIds(newSet);
  };

  // Download selected products (with all filters applied)
  const handleDownloadSelected = async () => {
    if (selectedIds.size === 0) return;

    setIsDownloading(true);
    try {
      const filters = buildExportFilters(Array.from(selectedIds));
      const blob = await apiClient.downloadProducts(filters);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `products_${Date.now()}.zip`;
      a.click();
      URL.revokeObjectURL(url);

      toast.success(`${selectedIds.size} products downloading...`);
    } catch (error) {
      toast.error("Download failed");
    } finally {
      setIsDownloading(false);
    }
  };

  // Download ALL filtered products (all pages)
  const handleDownloadAll = async () => {
    setIsDownloading(true);
    try {
      const filters = buildExportFilters();
      const blob = await apiClient.downloadAllProducts(filters);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      const filterCount = activeFilterCount;
      a.download = filterCount > 0
        ? `filtered_products_${Date.now()}.zip`
        : `all_products_${Date.now()}.zip`;
      a.click();
      URL.revokeObjectURL(url);

      toast.success(filterCount > 0
        ? "Filtered products downloading..."
        : "All products downloading...");
    } catch (error) {
      toast.error("Download failed");
    } finally {
      setIsDownloading(false);
    }
  };

  // Export to CSV (with all filters applied - all pages)
  const handleExportCSV = async () => {
    setIsDownloading(true);
    try {
      const filters = selectedIds.size > 0
        ? buildExportFilters(Array.from(selectedIds))
        : buildExportFilters();
      const blob = await apiClient.exportProductsCSV(filters);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `products_${Date.now()}.csv`;
      a.click();
      URL.revokeObjectURL(url);
      toast.success("CSV exported with all columns");
    } catch (error) {
      toast.error("Export failed");
    } finally {
      setIsDownloading(false);
    }
  };

  // Export to JSON (with all filters applied - all pages)
  const handleExportJSON = async () => {
    setIsDownloading(true);
    try {
      const filters = selectedIds.size > 0
        ? buildExportFilters(Array.from(selectedIds))
        : buildExportFilters();
      const blob = await apiClient.exportProductsJSON(filters);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `products_${Date.now()}.json`;
      a.click();
      URL.revokeObjectURL(url);
      toast.success("JSON exported with all fields");
    } catch (error) {
      toast.error("Export failed");
    } finally {
      setIsDownloading(false);
    }
  };

  // Bulk delete mutation
  const deleteMutation = useMutation({
    mutationFn: (ids: string[]) => apiClient.deleteProducts(ids),
    onSuccess: (result) => {
      toast.success(`${result.deleted_count} products deleted`);
      setSelectedIds(new Set());
      setDeleteDialogOpen(false);
      queryClient.invalidateQueries({ queryKey: ["products"] });
    },
    onError: () => {
      toast.error("Delete failed");
    },
  });

  // Reprocess mutation (for single product)
  const [reprocessingProductId, setReprocessingProductId] = useState<string | null>(null);
  const reprocessMutation = useMutation({
    mutationFn: (productId: string) => apiClient.reprocessProduct(productId),
    onSuccess: (result) => {
      toast.success(result.message || "Reprocessing started");
      setReprocessingProductId(null);
      queryClient.invalidateQueries({ queryKey: ["products"] });
    },
    onError: (error: Error) => {
      toast.error(error.message || "Reprocess failed");
      setReprocessingProductId(null);
    },
  });

  // Add to existing dataset mutation (by product IDs)
  const addToDatasetMutation = useMutation({
    mutationFn: ({ datasetId, productIds }: { datasetId: string; productIds: string[] }) =>
      apiClient.addProductsToDataset(datasetId, productIds),
    onSuccess: () => {
      toast.success(`${selectedIds.size} product(s) added to dataset`);
      setSelectedIds(new Set());
      setSelectAllFilteredMode(false);
      setDatasetDialogOpen(false);
      resetDatasetDialog();
      queryClient.invalidateQueries({ queryKey: ["datasets"] });
    },
    onError: () => {
      toast.error("Failed to add products to dataset");
    },
  });

  // Add to existing dataset mutation (by filters - for "Select All Filtered")
  const addFilteredToDatasetMutation = useMutation({
    mutationFn: ({ datasetId, filters }: { datasetId: string; filters: ExportFilters }) =>
      apiClient.addFilteredProductsToDataset(datasetId, filters),
    onSuccess: (result) => {
      toast.success(`${result.added_count} product(s) added to dataset`);
      setSelectedIds(new Set());
      setSelectAllFilteredMode(false);
      setDatasetDialogOpen(false);
      resetDatasetDialog();
      queryClient.invalidateQueries({ queryKey: ["datasets"] });
    },
    onError: () => {
      toast.error("Failed to add products to dataset");
    },
  });

  // Create new dataset mutation
  const createDatasetMutation = useMutation({
    mutationFn: (data: { name: string; description?: string; product_ids: string[] }) =>
      apiClient.createDataset(data),
    onSuccess: (dataset) => {
      toast.success(`Dataset "${dataset.name}" created with ${selectedIds.size} product(s)`);
      setSelectedIds(new Set());
      setDatasetDialogOpen(false);
      resetDatasetDialog();
      queryClient.invalidateQueries({ queryKey: ["datasets"] });
    },
    onError: () => {
      toast.error("Failed to create dataset");
    },
  });

  // Reset dataset dialog state
  const resetDatasetDialog = () => {
    setDatasetMode("existing");
    setSelectedDatasetId("");
    setNewDatasetName("");
    setNewDatasetDescription("");
  };

  // Handle add to dataset submission
  const handleAddToDataset = () => {
    if (datasetMode === "existing") {
      if (!selectedDatasetId) {
        toast.error("Please select a dataset");
        return;
      }

      // Use filters if "Select All Filtered" mode is active
      if (selectAllFilteredMode) {
        const filters = buildExportFilters();
        addFilteredToDatasetMutation.mutate({ datasetId: selectedDatasetId, filters });
      } else {
        const productIds = Array.from(selectedIds);
        addToDatasetMutation.mutate({ datasetId: selectedDatasetId, productIds });
      }
    } else {
      if (!newDatasetName.trim()) {
        toast.error("Please enter a dataset name");
        return;
      }

      // For new dataset, we need to use product_ids
      // If selectAllFilteredMode, we'd need a different API (not implemented yet)
      if (selectAllFilteredMode) {
        toast.error("Creating new dataset with all filtered products is not supported yet. Please select an existing dataset.");
        return;
      }

      const productIds = Array.from(selectedIds);
      createDatasetMutation.mutate({
        name: newDatasetName.trim(),
        description: newDatasetDescription.trim() || undefined,
        product_ids: productIds,
      });
    }
  };

  // Status badge colors
  const statusColors: Record<ProductStatus, string> = {
    pending: "bg-yellow-100 text-yellow-800 hover:bg-yellow-100",
    processing: "bg-blue-100 text-blue-800 hover:bg-blue-100",
    needs_matching: "bg-purple-100 text-purple-800 hover:bg-purple-100",
    ready: "bg-green-100 text-green-800 hover:bg-green-100",
    rejected: "bg-red-100 text-red-800 hover:bg-red-100",
  };

  const activeFilterCount = getTotalCount();

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold">Products</h1>
          <p className="text-gray-500">
            {sortedItems.length} of {data?.total || 0} products
            {activeFilterCount > 0 && (
              <span className="text-blue-600 ml-1">
                ({activeFilterCount} filters)
              </span>
            )}
          </p>
        </div>

        {/* Action Buttons */}
        <div className="flex gap-2">
          <Button variant="outline" asChild>
            <Link href="/products/matcher">
              <FileSpreadsheet className="h-4 w-4 mr-2" />
              Product Matcher
            </Link>
          </Button>

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

          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" disabled={isDownloading}>
                {isDownloading ? (
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                ) : (
                  <Download className="h-4 w-4 mr-2" />
                )}
                Download
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-64">
              <DropdownMenuItem
                onClick={handleDownloadSelected}
                disabled={selectedIds.size === 0}
              >
                <Package className="h-4 w-4 mr-2" />
                Download Selected ({selectedIds.size})
              </DropdownMenuItem>
              <DropdownMenuItem onClick={handleDownloadAll}>
                <Download className="h-4 w-4 mr-2" />
                {activeFilterCount > 0
                  ? `Download Filtered (${activeFilterCount} filters)`
                  : "Download All Products"}
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={handleExportCSV}>
                <FileSpreadsheet className="h-4 w-4 mr-2" />
                <div className="flex flex-col">
                  <span>Export to CSV</span>
                  <span className="text-xs text-muted-foreground">
                    {selectedIds.size > 0
                      ? `${selectedIds.size} selected`
                      : activeFilterCount > 0
                      ? `Filtered (all pages)`
                      : "All products (all pages)"}
                  </span>
                </div>
              </DropdownMenuItem>
              <DropdownMenuItem onClick={handleExportJSON}>
                <FileJson className="h-4 w-4 mr-2" />
                <div className="flex flex-col">
                  <span>Export to JSON</span>
                  <span className="text-xs text-muted-foreground">
                    {selectedIds.size > 0
                      ? `${selectedIds.size} selected`
                      : activeFilterCount > 0
                      ? `Filtered (all pages)`
                      : "All products (all pages)"}
                  </span>
                </div>
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>

      {/* Search & Filters */}
      <div className="flex gap-4 flex-wrap">
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
          <Input
            placeholder="Search by barcode, name, or brand..."
            value={search}
            onChange={(e) => {
              setSearch(e.target.value);
              setPage(1);
            }}
            className="pl-10"
          />
        </div>

        <div className="flex items-center gap-2">
          <FilterTrigger
            onClick={() => setFilterDrawerOpen(true)}
            activeCount={activeFilterCount}
          />
          {activeFilterCount > 0 && (
            <Button
              variant="ghost"
              size="sm"
              onClick={clearAll}
              className="text-muted-foreground hover:text-foreground"
            >
              Clear filters
            </Button>
          )}
        </div>
      </div>

      {/* Active Filter Chips */}
      <ActiveFilterChips
        filterState={filterState}
        sections={filterSections}
        onRemoveFilter={handleRemoveFilter}
        onClearAll={clearAll}
      />

      {/* Filter Drawer */}
      <FilterDrawer
        open={filterDrawerOpen}
        onOpenChange={setFilterDrawerOpen}
        sections={filterSections}
        filterState={filterState}
        onFilterChange={setFilter}
        onClearAll={clearAll}
        onClearSection={clearSection}
        title="Product Filters"
        description="Filter by all product attributes"
      />

      {/* Bulk Actions Bar */}
      {(selectedIds.size > 0 || selectAllFilteredMode) && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-sm text-blue-700 font-medium">
              {selectAllFilteredMode
                ? `All ${data?.total || 0} filtered products selected`
                : `${selectedIds.size} product${selectedIds.size > 1 ? "s" : ""} selected`}
            </span>
            {/* Show "Select All Filtered" option when page items are selected but not all filtered */}
            {!selectAllFilteredMode && selectedIds.size > 0 && (data?.total || 0) > sortedItems.length && (
              <Button
                variant="link"
                size="sm"
                className="text-blue-600 p-0 h-auto"
                onClick={() => {
                  setSelectAllFilteredMode(true);
                  setSelectedIds(new Set());
                }}
              >
                Select all {data?.total} filtered products
              </Button>
            )}
            {selectAllFilteredMode && (
              <Button
                variant="link"
                size="sm"
                className="text-blue-600 p-0 h-auto"
                onClick={() => {
                  setSelectAllFilteredMode(false);
                  setSelectedIds(new Set());
                }}
              >
                Clear selection
              </Button>
            )}
          </div>
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={selectAllFilteredMode ? handleDownloadAll : handleDownloadSelected}
              disabled={isDownloading}
            >
              <Download className="h-4 w-4 mr-1" />
              Download
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setDatasetDialogOpen(true)}
            >
              <FolderPlus className="h-4 w-4 mr-1" />
              Add to Dataset
            </Button>
            <Button
              variant="outline"
              size="sm"
              className="text-red-600 hover:text-red-700 hover:bg-red-50"
              onClick={() => setDeleteDialogOpen(true)}
            >
              <Trash2 className="h-4 w-4 mr-1" />
              Delete
            </Button>
          </div>
        </div>
      )}

      {/* Table */}
      <div className="border rounded-lg bg-card overflow-hidden">
        <div className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow className="bg-muted/50">
                <TableHead className="w-12">
                  <Checkbox
                    checked={allSelected ? true : someSelected ? "indeterminate" : false}
                    onCheckedChange={toggleAll}
                  />
                </TableHead>
                <TableHead className="w-16">Image</TableHead>
                <SortableHeader column="barcode">Barcode</SortableHeader>
                <SortableHeader column="brand_name">Brand</SortableHeader>
                <SortableHeader column="sub_brand">Sub Brand</SortableHeader>
                <SortableHeader column="product_name">Product Name</SortableHeader>
                <SortableHeader column="variant_flavor">Variant/Flavor</SortableHeader>
                <SortableHeader column="category">Category</SortableHeader>
                <SortableHeader column="container_type">Container</SortableHeader>
                <SortableHeader column="net_quantity">Net Qty</SortableHeader>
                <SortableHeader column="pack_type">Pack</SortableHeader>
                <SortableHeader column="manufacturer_country">Country</SortableHeader>
                <SortableHeader column="status">Status</SortableHeader>
                <SortableHeader column="updated_at">Processed</SortableHeader>
                <SortableHeader column="synthetic_count" className="text-center text-xs">Syn</SortableHeader>
                <SortableHeader column="real_count" className="text-center text-xs">Real</SortableHeader>
                <SortableHeader column="augmented_count" className="text-center text-xs">Aug</SortableHeader>
                <TableHead className="w-12"></TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {isLoading ? (
                <TableRow>
                  <TableCell colSpan={18} className="text-center py-8">
                    <Loader2 className="h-6 w-6 animate-spin mx-auto" />
                  </TableCell>
                </TableRow>
              ) : sortedItems.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={18} className="text-center py-8">
                    <div className="text-gray-500">
                      <Package className="h-12 w-12 mx-auto mb-2 opacity-50" />
                      <p>No products found</p>
                      {activeFilterCount > 0 && (
                        <Button
                          variant="link"
                          size="sm"
                          onClick={clearAll}
                          className="mt-2"
                        >
                          Clear all filters
                        </Button>
                      )}
                      {search && (
                        <div className="mt-4 pt-4 border-t">
                          <p className="text-sm mb-2">Can&apos;t find what you&apos;re looking for?</p>
                          <Link href={`/scan-requests/new?barcode=${encodeURIComponent(search)}`}>
                            <Button variant="outline" size="sm">
                              Request a scan for this product
                            </Button>
                          </Link>
                        </div>
                      )}
                    </div>
                  </TableCell>
                </TableRow>
              ) : (
                sortedItems.map((product) => (
                  <TableRow
                    key={product.id}
                    className={selectedIds.has(product.id) ? "bg-blue-50" : ""}
                  >
                    <TableCell>
                      <Checkbox
                        checked={selectedIds.has(product.id)}
                        onCheckedChange={() => toggleOne(product.id)}
                      />
                    </TableCell>
                    <TableCell>
                      {product.primary_image_url ? (
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <img
                              src={product.primary_image_url}
                              alt=""
                              className="w-10 h-10 object-cover rounded cursor-pointer"
                            />
                          </TooltipTrigger>
                          <TooltipContent
                            side="right"
                            className="p-0 bg-transparent border-0"
                            sideOffset={8}
                          >
                            <img
                              src={product.primary_image_url}
                              alt={product.product_name || product.barcode}
                              className="w-48 h-48 object-contain rounded-lg shadow-lg bg-white"
                            />
                          </TooltipContent>
                        </Tooltip>
                      ) : (
                        <div className="w-10 h-10 bg-gray-100 rounded flex items-center justify-center">
                          <Package className="h-5 w-5 text-gray-400" />
                        </div>
                      )}
                    </TableCell>
                    <TableCell className="font-mono text-xs">
                      {product.barcode}
                    </TableCell>
                    <TableCell className="text-sm font-medium">
                      {product.brand_name || "-"}
                    </TableCell>
                    <TableCell className="text-sm text-gray-600">
                      {product.sub_brand || "-"}
                    </TableCell>
                    <TableCell className="max-w-[150px] truncate text-sm">
                      {product.product_name || "-"}
                    </TableCell>
                    <TableCell className="text-sm text-gray-600">
                      {product.variant_flavor || "-"}
                    </TableCell>
                    <TableCell className="text-sm">
                      {product.category || "-"}
                    </TableCell>
                    <TableCell className="text-sm text-gray-600">
                      {product.container_type || "-"}
                    </TableCell>
                    <TableCell className="text-sm text-gray-600">
                      {product.net_quantity || "-"}
                    </TableCell>
                    <TableCell className="text-sm">
                      {product.pack_configuration?.type ? (
                        <Badge variant="outline" className="text-xs">
                          {product.pack_configuration.type === "multipack"
                            ? `${product.pack_configuration.item_count}x`
                            : "1x"}
                        </Badge>
                      ) : (
                        "-"
                      )}
                    </TableCell>
                    <TableCell className="text-sm text-gray-600">
                      {product.manufacturer_country || "-"}
                    </TableCell>
                    <TableCell>
                      <Badge className={statusColors[product.status]}>
                        {product.status.replace("_", " ")}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-xs text-gray-500 whitespace-nowrap">
                      {product.updated_at
                        ? new Date(product.updated_at).toLocaleDateString("en-US", {
                            month: "short",
                            day: "numeric",
                            hour: "2-digit",
                            minute: "2-digit",
                          })
                        : "-"}
                    </TableCell>
                    <TableCell className="text-center text-sm text-blue-600">
                      {product.frame_counts?.synthetic ?? product.frame_count ?? 0}
                    </TableCell>
                    <TableCell className="text-center text-sm text-green-600">
                      {product.frame_counts?.real ?? 0}
                    </TableCell>
                    <TableCell className="text-center text-sm text-purple-600">
                      {product.frame_counts?.augmented ?? 0}
                    </TableCell>
                    <TableCell>
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button variant="ghost" size="icon">
                            <MoreHorizontal className="h-4 w-4" />
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end">
                          <DropdownMenuItem asChild>
                            <Link href={`/products/${product.id}`}>
                              View Details
                            </Link>
                          </DropdownMenuItem>
                          <DropdownMenuItem asChild>
                            <Link href={`/products/${product.id}?edit=true`}>
                              Edit
                            </Link>
                          </DropdownMenuItem>
                          <DropdownMenuItem
                            onClick={async () => {
                              try {
                                const blob = await apiClient.downloadProduct(
                                  product.id
                                );
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
                            }}
                          >
                            <Download className="h-4 w-4 mr-2" />
                            Download Frames
                          </DropdownMenuItem>
                          {product.video_url && (
                            <DropdownMenuItem
                              onClick={() => {
                                setReprocessingProductId(product.id);
                                reprocessMutation.mutate(product.id);
                              }}
                              disabled={reprocessingProductId === product.id}
                            >
                              {reprocessingProductId === product.id ? (
                                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                              ) : (
                                <RotateCcw className="h-4 w-4 mr-2" />
                              )}
                              Reprocess
                            </DropdownMenuItem>
                          )}
                          <DropdownMenuSeparator />
                          <DropdownMenuItem
                            className="text-red-600"
                            onClick={() => {
                              setSelectedIds(new Set([product.id]));
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
                ))
              )}
            </TableBody>
          </Table>
        </div>
      </div>

      {/* Pagination */}
      <div className="flex justify-between items-center">
        <p className="text-sm text-gray-500">
          Showing {sortedItems.length} of {data?.total || 0} products
        </p>
        <div className="flex gap-2">
          <Button
            variant="outline"
            disabled={page === 1}
            onClick={() => setPage(page - 1)}
          >
            Previous
          </Button>
          <Button
            variant="outline"
            disabled={!data || page * 100 >= data.total}
            onClick={() => setPage(page + 1)}
          >
            Next
          </Button>
        </div>
      </div>

      {/* Add to Dataset Dialog */}
      <Dialog
        open={datasetDialogOpen}
        onOpenChange={(open) => {
          setDatasetDialogOpen(open);
          if (!open) resetDatasetDialog();
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Add to Dataset</DialogTitle>
            <DialogDescription>
              Add {selectedIds.size} selected product{selectedIds.size > 1 ? "s" : ""} to a dataset.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4 py-4">
            {/* Mode Selection */}
            <div className="flex gap-2">
              <Button
                variant={datasetMode === "existing" ? "default" : "outline"}
                size="sm"
                onClick={() => setDatasetMode("existing")}
                className="flex-1"
              >
                Existing Dataset
              </Button>
              <Button
                variant={datasetMode === "new" ? "default" : "outline"}
                size="sm"
                onClick={() => setDatasetMode("new")}
                className="flex-1"
              >
                New Dataset
              </Button>
            </div>

            {datasetMode === "existing" ? (
              <div className="space-y-2">
                <Label htmlFor="dataset-select">Select Dataset</Label>
                <Select value={selectedDatasetId} onValueChange={setSelectedDatasetId}>
                  <SelectTrigger id="dataset-select">
                    <SelectValue placeholder="Choose a dataset..." />
                  </SelectTrigger>
                  <SelectContent>
                    {datasets && datasets.length > 0 ? (
                      datasets.map((dataset) => (
                        <SelectItem key={dataset.id} value={dataset.id}>
                          {dataset.name} ({dataset.product_count} products)
                        </SelectItem>
                      ))
                    ) : (
                      <div className="px-2 py-4 text-sm text-gray-500 text-center">
                        No datasets found. Create a new one.
                      </div>
                    )}
                  </SelectContent>
                </Select>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="dataset-name">Dataset Name *</Label>
                  <Input
                    id="dataset-name"
                    placeholder="Enter dataset name..."
                    value={newDatasetName}
                    onChange={(e) => setNewDatasetName(e.target.value)}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="dataset-description">Description (optional)</Label>
                  <Input
                    id="dataset-description"
                    placeholder="Enter description..."
                    value={newDatasetDescription}
                    onChange={(e) => setNewDatasetDescription(e.target.value)}
                  />
                </div>
              </div>
            )}
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setDatasetDialogOpen(false)}>
              Cancel
            </Button>
            <Button
              onClick={handleAddToDataset}
              disabled={
                addToDatasetMutation.isPending ||
                createDatasetMutation.isPending ||
                (datasetMode === "existing" && !selectedDatasetId) ||
                (datasetMode === "new" && !newDatasetName.trim())
              }
            >
              {(addToDatasetMutation.isPending || createDatasetMutation.isPending) && (
                <Loader2 className="h-4 w-4 animate-spin mr-2" />
              )}
              {datasetMode === "existing" ? "Add to Dataset" : "Create & Add"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <AlertDialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Products</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete {selectedIds.size} product
              {selectedIds.size > 1 ? "s" : ""}? This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              className="bg-red-600 hover:bg-red-700"
              onClick={() => deleteMutation.mutate(Array.from(selectedIds))}
              disabled={deleteMutation.isPending}
            >
              {deleteMutation.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin mr-2" />
              ) : null}
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}

export default function ProductsPage() {
  return (
    <Suspense fallback={<div className="flex items-center justify-center h-64"><Loader2 className="h-8 w-8 animate-spin text-gray-400" /></div>}>
      <ProductsPageContent />
    </Suspense>
  );
}
