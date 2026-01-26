"use client";

import { useState, useMemo, useEffect, useDeferredValue, memo } from "react";
import Image from "next/image";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useParams } from "next/navigation";
import { toast } from "sonner";
import Link from "next/link";
import { apiClient } from "@/lib/api-client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Checkbox } from "@/components/ui/checkbox";
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
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  FilterDrawer,
  FilterTrigger,
  useFilterState,
  type FilterSection,
} from "@/components/filters/filter-drawer";
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
  Info,
  ImageIcon,
  Camera,
  Wand2,
  Search,
  MoreHorizontal,
  Download,
  FileJson,
  FileSpreadsheet,
  RefreshCw,
  ArrowUpDown,
  ArrowUp,
  ArrowDown,
} from "lucide-react";
import type { ProductWithFrameCounts, AugmentationPreset, AugmentationRequest, ProductStatus } from "@/types";

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

// Preset descriptions
const PRESET_INFO: Record<AugmentationPreset, { label: string; description: string }> = {
  clean: {
    label: "Clean",
    description: "Minimal effects, clean backgrounds, ideal for simple recognition tasks",
  },
  normal: {
    label: "Normal (Recommended)",
    description: "Balanced augmentation with moderate shelf scene composition",
  },
  realistic: {
    label: "Realistic",
    description: "High realism with neighboring products, shadows, and camera effects",
  },
  extreme: {
    label: "Extreme",
    description: "Maximum diversity with all effects enabled at high probability",
  },
  custom: {
    label: "Custom",
    description: "Configure all probabilities manually",
  },
};

// Calculate augmentation plan
function calculateAugmentationPlan(
  products: ProductWithFrameCounts[],
  synTarget: number,
  frameInterval: number = 1
) {
  return products.map((product) => {
    const synFrames = product.frame_counts?.synthetic || 0;
    const augFrames = product.frame_counts?.augmented || 0;
    const currentTotal = synFrames + augFrames;
    const needed = Math.max(0, synTarget - currentTotal);
    const selectedFrames = Math.max(1, Math.floor(synFrames / frameInterval));
    const augsPerFrame = selectedFrames > 0 ? Math.ceil(needed / selectedFrames) : 0;

    return {
      id: product.id,
      barcode: product.barcode,
      synFrames,
      selectedFrames,
      augFrames,
      currentTotal,
      needed,
      augsPerFrame,
      projected: currentTotal + (augsPerFrame * selectedFrames),
    };
  });
}

// Status badge colors
const statusColors: Record<ProductStatus, string> = {
  pending: "bg-yellow-100 text-yellow-800 hover:bg-yellow-100",
  processing: "bg-blue-100 text-blue-800 hover:bg-blue-100",
  needs_matching: "bg-purple-100 text-purple-800 hover:bg-purple-100",
  ready: "bg-green-100 text-green-800 hover:bg-green-100",
  rejected: "bg-red-100 text-red-800 hover:bg-red-100",
};

// ===========================================
// Memoized Dataset Product Table Row Component
// ===========================================
interface DatasetProductRowProps {
  product: ProductWithFrameCounts;
  isSelected: boolean;
  onToggleSelect: (id: string) => void;
  onRemove: (id: string) => void;
  isRemoving: boolean;
}

const DatasetProductRow = memo(function DatasetProductRow({
  product,
  isSelected,
  onToggleSelect,
  onRemove,
  isRemoving,
}: DatasetProductRowProps) {
  return (
    <TableRow className={isSelected ? "bg-blue-50" : ""}>
      <TableCell>
        <Checkbox
          checked={isSelected}
          onCheckedChange={() => onToggleSelect(product.id)}
        />
      </TableCell>
      <TableCell>
        {product.primary_image_url ? (
          <Tooltip>
            <TooltipTrigger asChild>
              <div className="relative w-10 h-10 rounded cursor-pointer overflow-hidden">
                <Image
                  src={product.primary_image_url}
                  alt=""
                  fill
                  className="object-cover"
                  loading="lazy"
                  quality={60}
                  sizes="40px"
                />
              </div>
            </TooltipTrigger>
            <TooltipContent
              side="right"
              className="p-0 bg-transparent border-0"
              sideOffset={8}
            >
              <div className="relative w-48 h-48 rounded-lg shadow-lg bg-white overflow-hidden">
                <Image
                  src={product.primary_image_url}
                  alt={product.product_name || product.barcode}
                  fill
                  className="object-contain"
                  quality={75}
                  sizes="192px"
                />
              </div>
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
            })
          : "-"}
      </TableCell>
      <TableCell className="text-center text-sm text-blue-600">
        {product.frame_counts?.synthetic ?? 0}
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
                  const blob = await apiClient.downloadProduct(product.id);
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
            <DropdownMenuSeparator />
            <DropdownMenuItem
              className="text-red-600"
              onClick={() => onRemove(product.id)}
              disabled={isRemoving}
            >
              <Trash2 className="h-4 w-4 mr-2" />
              Remove from Dataset
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </TableCell>
    </TableRow>
  );
});

export default function DatasetDetailPage() {
  const { id } = useParams<{ id: string }>();
  const queryClient = useQueryClient();

  // Search, pagination, sorting state
  const [search, setSearch] = useState("");
  const deferredSearch = useDeferredValue(search);
  const [page, setPage] = useState(1);
  const [sortColumn, setSortColumn] = useState<SortColumn | null>(null);
  const [sortDirection, setSortDirection] = useState<SortDirection>("asc");

  // Selection state
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [isDownloading, setIsDownloading] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [filterDrawerOpen, setFilterDrawerOpen] = useState(false);

  // Augmentation dialog state
  const [augDialogOpen, setAugDialogOpen] = useState(false);
  const [augConfig, setAugConfig] = useState<AugmentationRequest>({
    syn_target: 600,
    real_target: 400,
    use_diversity_pyramid: true,
    include_neighbors: true,
    frame_interval: 1,
    augmentation_config: {
      preset: "normal",
    },
  });

  // Use filter state hook
  const {
    filterState,
    setFilter,
    clearSection,
    clearAll,
    getTotalCount,
  } = useFilterState();

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

  // Map frontend sort columns to backend column names
  const serverSortColumn = useMemo(() => {
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

  // Fetch dataset with products (with filters)
  const {
    data: dataset,
    isLoading,
    error,
    refetch,
    isFetching,
  } = useQuery({
    queryKey: ["dataset", id, { page, search: deferredSearch, ...apiFilters, sort_by: serverSortColumn, sort_order: sortDirection }],
    queryFn: () =>
      apiClient.getDataset(id, {
        page,
        limit: 25,
        search: deferredSearch || undefined,
        sort_by: serverSortColumn,
        sort_order: sortDirection,
        include_frame_counts: true,
        ...apiFilters,
      }),
    enabled: !!id,
  });

  // Fetch filter options for this dataset
  const { data: filterOptions } = useQuery({
    queryKey: ["dataset-filter-options", id],
    queryFn: () => apiClient.getDatasetFilterOptions(id),
    enabled: !!id,
    staleTime: 60000,
  });

  // Reset page when filters change
  useEffect(() => {
    setPage(1);
  }, [apiFilters, serverSortColumn, sortDirection, deferredSearch]);

  // Build filter sections from filter options
  const filterSections: FilterSection[] = useMemo(() => {
    if (!filterOptions) return [];

    const sections: FilterSection[] = [
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
        id: "issueTypes",
        label: "Issue Types",
        type: "checkbox",
        options: filterOptions.issueTypes,
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

  // Products from dataset
  const products = dataset?.products || [];
  const productsTotal = dataset?.products_total || 0;

  // Client-side sorting for frame count columns (not supported by server)
  const sortedProducts = useMemo(() => {
    if (!sortColumn) return products;
    if (serverSortColumn) return products; // Server already sorted

    return [...products].sort((a, b) => {
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

      if (aVal === undefined && bVal === undefined) return 0;
      if (aVal === undefined) return 1;
      if (bVal === undefined) return -1;

      let result = 0;
      if (typeof aVal === "number" && typeof bVal === "number") {
        result = aVal - bVal;
      } else {
        result = String(aVal).localeCompare(String(bVal));
      }

      return sortDirection === "desc" ? -result : result;
    });
  }, [products, sortColumn, sortDirection, serverSortColumn]);

  // Handle sort column click
  const handleSort = (column: SortColumn) => {
    if (sortColumn === column) {
      if (sortDirection === "asc") {
        setSortDirection("desc");
      } else {
        setSortColumn(null);
        setSortDirection("asc");
      }
    } else {
      setSortColumn(column);
      setSortDirection("asc");
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
    if (!sortedProducts.length) return false;
    return sortedProducts.every((p) => selectedIds.has(p.id));
  }, [sortedProducts, selectedIds]);

  const someSelected = useMemo(() => {
    if (!sortedProducts.length) return false;
    return sortedProducts.some((p) => selectedIds.has(p.id)) && !allSelected;
  }, [sortedProducts, selectedIds, allSelected]);

  const toggleAll = () => {
    if (allSelected) {
      setSelectedIds(new Set());
    } else {
      setSelectedIds(new Set(sortedProducts.map((p) => p.id)));
    }
  };

  const toggleOne = (productId: string) => {
    const newSet = new Set(selectedIds);
    if (newSet.has(productId)) {
      newSet.delete(productId);
    } else {
      newSet.add(productId);
    }
    setSelectedIds(newSet);
  };

  // Calculate augmentation plan
  const augmentationPlan = useMemo(() => {
    if (!products.length) return [];
    return calculateAugmentationPlan(products, augConfig.syn_target, augConfig.frame_interval);
  }, [products, augConfig.syn_target, augConfig.frame_interval]);

  const totalAugmentations = useMemo(() => {
    return augmentationPlan.reduce((sum, p) => sum + (p.augsPerFrame * p.selectedFrames), 0);
  }, [augmentationPlan]);

  // Remove product mutation
  const removeProductMutation = useMutation({
    mutationFn: (productId: string) =>
      apiClient.removeProductFromDataset(id, productId),
    onSuccess: () => {
      toast.success("Product removed from dataset");
      setSelectedIds(new Set());
      queryClient.invalidateQueries({ queryKey: ["dataset", id] });
      queryClient.invalidateQueries({ queryKey: ["dataset-filter-options", id] });
    },
    onError: () => {
      toast.error("Failed to remove product");
    },
  });

  // Bulk remove mutation
  const bulkRemoveMutation = useMutation({
    mutationFn: async (productIds: string[]) => {
      for (const productId of productIds) {
        await apiClient.removeProductFromDataset(id, productId);
      }
    },
    onSuccess: () => {
      toast.success(`${selectedIds.size} product(s) removed from dataset`);
      setSelectedIds(new Set());
      setDeleteDialogOpen(false);
      queryClient.invalidateQueries({ queryKey: ["dataset", id] });
      queryClient.invalidateQueries({ queryKey: ["dataset-filter-options", id] });
    },
    onError: () => {
      toast.error("Failed to remove products");
    },
  });

  // Augmentation mutation
  const augmentMutation = useMutation({
    mutationFn: (config: AugmentationRequest) =>
      apiClient.startAugmentation(id, config),
    onSuccess: () => {
      toast.success("Augmentation job started");
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
      setAugDialogOpen(false);
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

  // Download handlers
  const handleDownloadSelected = async () => {
    if (selectedIds.size === 0) return;
    setIsDownloading(true);
    try {
      const blob = await apiClient.downloadProducts({ product_ids: Array.from(selectedIds) });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `dataset_products_${Date.now()}.zip`;
      a.click();
      URL.revokeObjectURL(url);
      toast.success(`${selectedIds.size} products downloading...`);
    } catch (error) {
      toast.error("Download failed");
    } finally {
      setIsDownloading(false);
    }
  };

  const handleExportCSV = async () => {
    setIsDownloading(true);
    try {
      const productIds = selectedIds.size > 0 ? Array.from(selectedIds) : products.map(p => p.id);
      const blob = await apiClient.exportProductsCSV({ product_ids: productIds });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `dataset_${id}_products.csv`;
      a.click();
      URL.revokeObjectURL(url);
      toast.success("CSV exported");
    } catch (error) {
      toast.error("Export failed");
    } finally {
      setIsDownloading(false);
    }
  };

  const handleExportJSON = async () => {
    setIsDownloading(true);
    try {
      const productIds = selectedIds.size > 0 ? Array.from(selectedIds) : products.map(p => p.id);
      const blob = await apiClient.exportProductsJSON({ product_ids: productIds });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `dataset_${id}_products.json`;
      a.click();
      URL.revokeObjectURL(url);
      toast.success("JSON exported");
    } catch (error) {
      toast.error("Export failed");
    } finally {
      setIsDownloading(false);
    }
  };

  // Handle augmentation config changes
  const handlePresetChange = (preset: AugmentationPreset) => {
    setAugConfig((prev) => ({
      ...prev,
      augmentation_config: {
        ...prev.augmentation_config,
        preset,
      },
    }));
  };

  const handleStartAugmentation = () => {
    augmentMutation.mutate(augConfig);
  };

  const activeFilterCount = getTotalCount();

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
        <div className="text-sm text-gray-500" suppressHydrationWarning>
          Created {new Date(dataset.created_at).toLocaleDateString()}
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-4 gap-4">
        <Card>
          <CardContent className="pt-4">
            <div className="flex items-center gap-2">
              <Package className="h-4 w-4 text-gray-500" />
              <span className="text-sm text-gray-500">Products</span>
            </div>
            <p className="text-2xl font-bold">{dataset.product_count}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4">
            <div className="flex items-center gap-2">
              <ImageIcon className="h-4 w-4 text-blue-500" />
              <span className="text-sm text-gray-500">Synthetic</span>
            </div>
            <p className="text-2xl font-bold text-blue-600">{dataset.total_synthetic || 0}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4">
            <div className="flex items-center gap-2">
              <Camera className="h-4 w-4 text-green-500" />
              <span className="text-sm text-gray-500">Real</span>
            </div>
            <p className="text-2xl font-bold text-green-600">{dataset.total_real || 0}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4">
            <div className="flex items-center gap-2">
              <Wand2 className="h-4 w-4 text-purple-500" />
              <span className="text-sm text-gray-500">Augmented</span>
            </div>
            <p className="text-2xl font-bold text-purple-600">{dataset.total_augmented || 0}</p>
          </CardContent>
        </Card>
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
            {/* Augmentation Dialog */}
            <Dialog open={augDialogOpen} onOpenChange={setAugDialogOpen}>
              <DialogTrigger asChild>
                <Button disabled={dataset.product_count === 0}>
                  <Sparkles className="h-4 w-4 mr-2" />
                  Run Augmentation
                </Button>
              </DialogTrigger>
              <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
                <DialogHeader>
                  <DialogTitle>Augmentation Settings</DialogTitle>
                  <DialogDescription>
                    Configure augmentation parameters for shelf scene composition
                  </DialogDescription>
                </DialogHeader>

                <div className="space-y-6 py-4">
                  {/* Target Settings */}
                  <div className="grid grid-cols-3 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="syn_target">Synthetic Target (per product)</Label>
                      <Input
                        id="syn_target"
                        type="number"
                        value={augConfig.syn_target}
                        onChange={(e) =>
                          setAugConfig((prev) => ({
                            ...prev,
                            syn_target: parseInt(e.target.value) || 0,
                          }))
                        }
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="real_target">Real Target (per product)</Label>
                      <Input
                        id="real_target"
                        type="number"
                        value={augConfig.real_target}
                        onChange={(e) =>
                          setAugConfig((prev) => ({
                            ...prev,
                            real_target: parseInt(e.target.value) || 0,
                          }))
                        }
                      />
                    </div>
                    <div className="space-y-2">
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <Label htmlFor="frame_interval" className="flex items-center gap-1 cursor-help">
                              Frame Interval
                              <Info className="h-3 w-3 text-gray-400" />
                            </Label>
                          </TooltipTrigger>
                          <TooltipContent className="max-w-xs">
                            <p>Select 1 frame every N frames for angle diversity from 360 degree rotating videos.</p>
                            <p className="mt-1 text-xs">E.g., interval=20 with 200 frames = 10 unique angles</p>
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                      <Input
                        id="frame_interval"
                        type="number"
                        min={1}
                        value={augConfig.frame_interval}
                        onChange={(e) =>
                          setAugConfig((prev) => ({
                            ...prev,
                            frame_interval: Math.max(1, parseInt(e.target.value) || 1),
                          }))
                        }
                      />
                    </div>
                  </div>

                  {/* Equalization Preview */}
                  <div className="bg-gray-50 rounded-lg p-4 space-y-2">
                    <div className="flex items-center gap-2">
                      <Info className="h-4 w-4 text-blue-500" />
                      <span className="font-medium">Augmentation Preview</span>
                    </div>
                    <div className="text-sm text-gray-600">
                      <p>Total new augmentations: <span className="font-bold text-purple-600">{totalAugmentations.toLocaleString()}</span></p>
                      <p className="text-xs mt-1">
                        Formula: (target - current) / selected_frames = augs per frame
                        {augConfig.frame_interval > 1 && (
                          <span className="text-blue-600 ml-1">
                            (using 1 of every {augConfig.frame_interval} frames for angle diversity)
                          </span>
                        )}
                      </p>
                    </div>
                  </div>

                  {/* Preset Selection */}
                  <div className="space-y-2">
                    <Label>Augmentation Preset</Label>
                    <Select
                      value={augConfig.augmentation_config?.preset || "normal"}
                      onValueChange={(value) => handlePresetChange(value as AugmentationPreset)}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Select preset" />
                      </SelectTrigger>
                      <SelectContent>
                        {Object.entries(PRESET_INFO).map(([key, info]) => (
                          <SelectItem key={key} value={key}>
                            <div className="flex flex-col">
                              <span>{info.label}</span>
                            </div>
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    <p className="text-sm text-gray-500">
                      {PRESET_INFO[augConfig.augmentation_config?.preset || "normal"].description}
                    </p>
                  </div>

                  {/* Advanced Options */}
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <div className="space-y-0.5">
                        <Label>Diversity Pyramid</Label>
                        <p className="text-sm text-gray-500">
                          Randomly vary augmentation intensity for each image
                        </p>
                      </div>
                      <Switch
                        checked={augConfig.use_diversity_pyramid}
                        onCheckedChange={(checked) =>
                          setAugConfig((prev) => ({
                            ...prev,
                            use_diversity_pyramid: checked,
                          }))
                        }
                      />
                    </div>

                    <div className="flex items-center justify-between">
                      <div className="space-y-0.5">
                        <Label>Include Neighbors</Label>
                        <p className="text-sm text-gray-500">
                          Add neighboring products to shelf scenes
                        </p>
                      </div>
                      <Switch
                        checked={augConfig.include_neighbors}
                        onCheckedChange={(checked) =>
                          setAugConfig((prev) => ({
                            ...prev,
                            include_neighbors: checked,
                          }))
                        }
                      />
                    </div>
                  </div>

                  {/* Custom Settings */}
                  {augConfig.augmentation_config?.preset === "custom" && (
                    <div className="space-y-4 border-t pt-4">
                      <h4 className="font-medium">Custom Probabilities</h4>
                      <div className="grid grid-cols-2 gap-4">
                        <div className="space-y-2">
                          <Label className="text-sm">Heavy Augmentation</Label>
                          <Slider
                            value={[augConfig.augmentation_config?.PROB_HEAVY_AUGMENTATION ?? 0.5]}
                            max={1}
                            step={0.1}
                            onValueChange={([value]) =>
                              setAugConfig((prev) => ({
                                ...prev,
                                augmentation_config: {
                                  ...prev.augmentation_config!,
                                  PROB_HEAVY_AUGMENTATION: value,
                                },
                              }))
                            }
                          />
                        </div>
                        <div className="space-y-2">
                          <Label className="text-sm">Shadow</Label>
                          <Slider
                            value={[augConfig.augmentation_config?.PROB_SHADOW ?? 0.7]}
                            max={1}
                            step={0.1}
                            onValueChange={([value]) =>
                              setAugConfig((prev) => ({
                                ...prev,
                                augmentation_config: {
                                  ...prev.augmentation_config!,
                                  PROB_SHADOW: value,
                                },
                              }))
                            }
                          />
                        </div>
                      </div>
                    </div>
                  )}
                </div>

                <DialogFooter>
                  <Button variant="outline" onClick={() => setAugDialogOpen(false)}>
                    Cancel
                  </Button>
                  <Button
                    onClick={handleStartAugmentation}
                    disabled={augmentMutation.isPending}
                  >
                    {augmentMutation.isPending ? (
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    ) : (
                      <Sparkles className="h-4 w-4 mr-2" />
                    )}
                    Start Augmentation
                  </Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>

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
            <CardTitle>Products ({productsTotal})</CardTitle>
            <CardDescription>
              Products included in this dataset with frame counts
              {activeFilterCount > 0 && (
                <span className="text-blue-600 ml-1">
                  ({activeFilterCount} filters active)
                </span>
              )}
            </CardDescription>
          </div>
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="icon"
              onClick={() => refetch()}
              disabled={isFetching}
            >
              <RefreshCw className={`h-4 w-4 ${isFetching ? "animate-spin" : ""}`} />
            </Button>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" disabled={isDownloading}>
                  {isDownloading ? (
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <Download className="h-4 w-4 mr-2" />
                  )}
                  Export
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-56">
                <DropdownMenuItem
                  onClick={handleDownloadSelected}
                  disabled={selectedIds.size === 0}
                >
                  <Package className="h-4 w-4 mr-2" />
                  Download Selected ({selectedIds.size})
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem onClick={handleExportCSV}>
                  <FileSpreadsheet className="h-4 w-4 mr-2" />
                  Export to CSV
                </DropdownMenuItem>
                <DropdownMenuItem onClick={handleExportJSON}>
                  <FileJson className="h-4 w-4 mr-2" />
                  Export to JSON
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
            <Link href={`/datasets/${id}/add-products`}>
              <Button>
                <Plus className="h-4 w-4 mr-2" />
                Add Products
              </Button>
            </Link>
          </div>
        </CardHeader>
        <CardContent>
          {/* Search & Filters */}
          <div className="flex gap-4 flex-wrap mb-4">
            <div className="relative flex-1 max-w-md">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
              <Input
                placeholder="Search by barcode, name, or brand..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="pl-10"
              />
            </div>
            <FilterTrigger
              onClick={() => setFilterDrawerOpen(true)}
              activeCount={activeFilterCount}
            />
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
            title="Dataset Product Filters"
            description="Filter products in this dataset"
          />

          {/* Bulk Actions Bar */}
          {selectedIds.size > 0 && (
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 flex items-center justify-between mb-4">
              <span className="text-sm text-blue-700">
                {selectedIds.size} product{selectedIds.size > 1 ? "s" : ""} selected
              </span>
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleDownloadSelected}
                  disabled={isDownloading}
                >
                  <Download className="h-4 w-4 mr-1" />
                  Download
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  className="text-red-600 hover:text-red-700 hover:bg-red-50"
                  onClick={() => setDeleteDialogOpen(true)}
                >
                  <Trash2 className="h-4 w-4 mr-1" />
                  Remove from Dataset
                </Button>
              </div>
            </div>
          )}

          {/* Products Table */}
          {sortedProducts.length === 0 ? (
            <div className="text-center py-8">
              <Package className="h-12 w-12 mx-auto text-gray-300 mb-2" />
              <p className="text-gray-500">No products found</p>
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
            </div>
          ) : (
            <>
              <div className="border rounded-lg overflow-hidden">
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
                        <SortableHeader column="variant_flavor">Variant</SortableHeader>
                        <SortableHeader column="category">Category</SortableHeader>
                        <SortableHeader column="container_type">Container</SortableHeader>
                        <SortableHeader column="net_quantity">Net Qty</SortableHeader>
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
                      {sortedProducts.map((product: ProductWithFrameCounts) => (
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
                                  <div className="relative w-10 h-10 rounded cursor-pointer overflow-hidden">
                                    <Image
                                      src={product.primary_image_url}
                                      alt=""
                                      fill
                                      className="object-cover"
                                      loading="lazy"
                                      quality={60}
                                      sizes="40px"
                                    />
                                  </div>
                                </TooltipTrigger>
                                <TooltipContent
                                  side="right"
                                  className="p-0 bg-transparent border-0"
                                  sideOffset={8}
                                >
                                  <div className="relative w-48 h-48 rounded-lg shadow-lg bg-white overflow-hidden">
                                    <Image
                                      src={product.primary_image_url}
                                      alt={product.product_name || product.barcode}
                                      fill
                                      className="object-contain"
                                      quality={75}
                                      sizes="192px"
                                    />
                                  </div>
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
                                })
                              : "-"}
                          </TableCell>
                          <TableCell className="text-center text-sm text-blue-600">
                            {product.frame_counts?.synthetic ?? 0}
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
                                      const blob = await apiClient.downloadProduct(product.id);
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
                                <DropdownMenuSeparator />
                                <DropdownMenuItem
                                  className="text-red-600"
                                  onClick={() => removeProductMutation.mutate(product.id)}
                                  disabled={removeProductMutation.isPending}
                                >
                                  <Trash2 className="h-4 w-4 mr-2" />
                                  Remove from Dataset
                                </DropdownMenuItem>
                              </DropdownMenuContent>
                            </DropdownMenu>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              </div>

              {/* Pagination */}
              <div className="flex justify-between items-center mt-4">
                <p className="text-sm text-gray-500">
                  Showing {sortedProducts.length} of {productsTotal} products
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
                    disabled={page * 100 >= productsTotal}
                    onClick={() => setPage(page + 1)}
                  >
                    Next
                  </Button>
                </div>
              </div>
            </>
          )}
        </CardContent>
      </Card>

      {/* Delete Confirmation Dialog */}
      <AlertDialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Remove Products from Dataset</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to remove {selectedIds.size} product
              {selectedIds.size > 1 ? "s" : ""} from this dataset? The products will not be deleted from the system.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              className="bg-red-600 hover:bg-red-700"
              onClick={() => bulkRemoveMutation.mutate(Array.from(selectedIds))}
              disabled={bulkRemoveMutation.isPending}
            >
              {bulkRemoveMutation.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin mr-2" />
              ) : null}
              Remove
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
