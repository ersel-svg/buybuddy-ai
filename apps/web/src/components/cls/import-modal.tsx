"use client";

import { useState, useCallback, useMemo } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { apiClient } from "@/lib/api-client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import { ImportProgressUI } from "@/components/ui/import-progress";
import * as DialogPrimitive from "@radix-ui/react-dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Package,
  Scissors,
  ImageIcon,
  Upload,
  Link as LinkIcon,
  Loader2,
  Search,
  X,
  CheckCircle,
  CheckSquare,
  Square,
  ListChecks,
} from "lucide-react";
import { useDropzone } from "react-dropzone";
import Image from "next/image";
import { cn } from "@/lib/utils";
import { useChunkedImport, type ChunkResult } from "@/hooks/use-chunked-import";

// ===========================================
// Pagination Component with Page Numbers
// ===========================================
function PaginationControl({
  currentPage,
  totalPages,
  onPageChange,
}: {
  currentPage: number;
  totalPages: number;
  onPageChange: (page: number) => void;
}) {
  const getPageNumbers = () => {
    const pages: (number | string)[] = [];
    const showPages = 5; // Show max 5 page numbers
    
    if (totalPages <= showPages + 2) {
      // Show all pages if total is small
      for (let i = 1; i <= totalPages; i++) pages.push(i);
    } else {
      // Always show first page
      pages.push(1);
      
      // Calculate range around current page
      let start = Math.max(2, currentPage - 2);
      let end = Math.min(totalPages - 1, currentPage + 2);
      
      // Adjust if at edges
      if (currentPage <= 3) {
        end = Math.min(totalPages - 1, showPages);
      } else if (currentPage >= totalPages - 2) {
        start = Math.max(2, totalPages - showPages + 1);
      }
      
      // Add ellipsis before if needed
      if (start > 2) pages.push("...");
      
      // Add middle pages
      for (let i = start; i <= end; i++) pages.push(i);
      
      // Add ellipsis after if needed
      if (end < totalPages - 1) pages.push("...");
      
      // Always show last page
      pages.push(totalPages);
    }
    
    return pages;
  };

  return (
    <div className="flex items-center justify-center gap-1 mt-4">
      <Button
        variant="outline"
        size="sm"
        disabled={currentPage === 1}
        onClick={() => onPageChange(currentPage - 1)}
        className="px-2"
      >
        &lt;
      </Button>
      
      {getPageNumbers().map((page, idx) => (
        typeof page === "number" ? (
          <Button
            key={idx}
            variant={page === currentPage ? "default" : "outline"}
            size="sm"
            onClick={() => onPageChange(page)}
            className="min-w-[36px]"
          >
            {page}
          </Button>
        ) : (
          <span key={idx} className="px-2 text-muted-foreground">...</span>
        )
      ))}
      
      <Button
        variant="outline"
        size="sm"
        disabled={currentPage >= totalPages}
        onClick={() => onPageChange(currentPage + 1)}
        className="px-2"
      >
        &gt;
      </Button>
      
      {/* Direct page input */}
      <div className="flex items-center gap-1 ml-4">
        <span className="text-sm text-muted-foreground">Go to:</span>
        <Input
          type="number"
          min={1}
          max={totalPages}
          className="w-16 h-8 text-center"
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              const val = parseInt((e.target as HTMLInputElement).value);
              if (val >= 1 && val <= totalPages) {
                onPageChange(val);
                (e.target as HTMLInputElement).value = "";
              }
            }
          }}
          placeholder={String(currentPage)}
        />
      </div>
    </div>
  );
}

interface CLSImportModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  datasetId: string;
  onSuccess?: () => void;
}

type SourceTab = "products" | "cutouts" | "od" | "upload" | "url";

// Selected product with image type preferences
interface SelectedProduct {
  id: string;
  name: string;
  category?: string;
  brand?: string;
  primary_image_url?: string;
  includeSynthetic: boolean;
  includeAugmented: boolean;
  includeReal: boolean;
}

// Selected cutout
interface SelectedCutout {
  id: string;
  image_url: string;
  predicted_upc?: string;
  matched_product_name?: string;
}

// Selected OD image
interface SelectedODImage {
  id: string;
  image_url: string;
  filename?: string;
}

export function CLSImportModal({
  open,
  onOpenChange,
  datasetId,
  onSuccess,
}: CLSImportModalProps) {
  const queryClient = useQueryClient();
  const [activeTab, setActiveTab] = useState<SourceTab>("products");

  // ===========================================
  // Products Tab State
  // ===========================================
  const [selectedProducts, setSelectedProducts] = useState<Map<string, SelectedProduct>>(new Map());
  const [productsSearch, setProductsSearch] = useState("");
  const [productsPage, setProductsPage] = useState(1);
  const [productsCategoryFilter, setProductsCategoryFilter] = useState<string>("all");
  const [productsBrandFilter, setProductsBrandFilter] = useState<string>("all");
  // Default image types to include
  const [defaultIncludeSynthetic, setDefaultIncludeSynthetic] = useState(true);
  const [defaultIncludeAugmented, setDefaultIncludeAugmented] = useState(false);
  const [defaultIncludeReal, setDefaultIncludeReal] = useState(false);

  // ===========================================
  // Cutouts Tab State
  // ===========================================
  const [selectedCutouts, setSelectedCutouts] = useState<Map<string, SelectedCutout>>(new Map());
  const [cutoutsSearch, setCutoutsSearch] = useState("");
  const [cutoutsPage, setCutoutsPage] = useState(1);
  const [cutoutsMatchedOnly, setCutoutsMatchedOnly] = useState(false);

  // ===========================================
  // OD Images Tab State
  // ===========================================
  const [selectedODImages, setSelectedODImages] = useState<Map<string, SelectedODImage>>(new Map());
  const [odSearch, setOdSearch] = useState("");
  const [odPage, setOdPage] = useState(1);

  // ===========================================
  // Upload Tab State
  // ===========================================
  const [uploadFiles, setUploadFiles] = useState<File[]>([]);
  const [isUploading, setIsUploading] = useState(false);

  // ===========================================
  // URL Tab State
  // ===========================================
  const [urlInput, setUrlInput] = useState("");

  // ===========================================
  // Select All Filtered State
  // ===========================================
  const [isSelectingAll, setIsSelectingAll] = useState(false);

  // ===========================================
  // Chunked Import Hook
  // ===========================================
  const {
    progress: importProgress,
    importChunked,
    abort: abortImport,
    retryFailed,
    reset: resetImportProgress,
    isImporting,
  } = useChunkedImport({
    chunkSize: 500,
    maxRetries: 2,
    onComplete: (result) => {
      if (result.status === "completed") {
        toast.success(`Successfully imported ${result.imported.toLocaleString()} images`);
        queryClient.invalidateQueries({ queryKey: ["cls-dataset", datasetId] });
        queryClient.invalidateQueries({ queryKey: ["cls-dataset-images", datasetId] });
        // Reset selections
        setSelectedProducts(new Map());
        setSelectedCutouts(new Map());
        setSelectedODImages(new Map());
        setUploadFiles([]);
        setUrlInput("");
        onSuccess?.();
        onOpenChange(false);
      } else if (result.status === "error") {
        toast.warning(
          `Import completed with ${result.failedChunks.length} failed chunks. You can retry them.`
        );
        queryClient.invalidateQueries({ queryKey: ["cls-dataset", datasetId] });
        queryClient.invalidateQueries({ queryKey: ["cls-dataset-images", datasetId] });
      }
    },
  });

  // ===========================================
  // Label Strategy - Per Source Type
  // ===========================================
  // Products: "custom" | "product_id" | "category" | "brand" | "product_name" | "unlabeled"
  const [productsLabelStrategy, setProductsLabelStrategy] = useState<string>("product_id");
  const [productsCustomLabel, setProductsCustomLabel] = useState<string>("");
  
  // Cutouts: "custom" | "matched_product_id" | "unlabeled"
  const [cutoutsLabelStrategy, setCutoutsLabelStrategy] = useState<string>("matched_product_id");
  const [cutoutsCustomLabel, setCutoutsCustomLabel] = useState<string>("");
  
  // OD Images: "custom" | "unlabeled"
  const [odLabelStrategy, setOdLabelStrategy] = useState<string>("unlabeled");
  const [odCustomLabel, setOdCustomLabel] = useState<string>("");
  
  // Upload: "custom" | "unlabeled"
  const [uploadLabelStrategy, setUploadLabelStrategy] = useState<string>("unlabeled");
  const [uploadCustomLabel, setUploadCustomLabel] = useState<string>("");
  
  // URL: "custom" | "unlabeled"
  const [urlLabelStrategy, setUrlLabelStrategy] = useState<string>("unlabeled");
  const [urlCustomLabel, setUrlCustomLabel] = useState<string>("");

  // Legacy - keep for backward compatibility
  const [labelStrategy, setLabelStrategy] = useState<"category" | "brand" | "product_name">("category");

  // ===========================================
  // Products Data - Fetch PRODUCTS not individual frames
  // ===========================================
  const {
    data: productsData,
    isLoading: productsLoading,
  } = useQuery({
    queryKey: ["cls-import-products", productsPage, productsSearch, productsCategoryFilter, productsBrandFilter],
    queryFn: () =>
      apiClient.getProducts({
        page: productsPage,
        limit: 30,
        search: productsSearch || undefined,
        category: productsCategoryFilter !== "all" ? productsCategoryFilter : undefined,
        brand: productsBrandFilter !== "all" ? productsBrandFilter : undefined,
        has_image: true, // Only products with images
      }),
    enabled: open && activeTab === "products",
    staleTime: 60000,
    refetchOnWindowFocus: false,
    refetchOnMount: false,
  });

  // Products filter options
  const { data: productsFilterOptions } = useQuery({
    queryKey: ["products-filter-options"],
    queryFn: () => apiClient.getFilterOptions(),
    enabled: open && activeTab === "products",
    staleTime: 300000,
    refetchOnWindowFocus: false,
  });

  // ===========================================
  // Cutouts Data
  // ===========================================
  const {
    data: cutoutsData,
    isLoading: cutoutsLoading,
  } = useQuery({
    queryKey: ["cls-import-cutouts", cutoutsPage, cutoutsSearch, cutoutsMatchedOnly],
    queryFn: () =>
      apiClient.getCutouts({
        page: cutoutsPage,
        limit: 30,
        search: cutoutsSearch || undefined,
        is_matched: cutoutsMatchedOnly ? true : undefined,
      }),
    enabled: open && activeTab === "cutouts",
    staleTime: 60000,
    refetchOnWindowFocus: false,
    refetchOnMount: false,
  });

  // ===========================================
  // OD Images Data
  // ===========================================
  const {
    data: odData,
    isLoading: odLoading,
  } = useQuery({
    queryKey: ["cls-import-od", odPage, odSearch],
    queryFn: () =>
      apiClient.getODImages({
        page: odPage,
        limit: 30,
        search: odSearch || undefined,
      }),
    enabled: open && activeTab === "od",
    staleTime: 60000,
    refetchOnWindowFocus: false,
    refetchOnMount: false,
  });

  // ===========================================
  // Product Selection Handlers
  // ===========================================
  const toggleProductSelection = useCallback((product: any) => {
    setSelectedProducts(prev => {
      const newMap = new Map(prev);
      if (newMap.has(product.id)) {
        newMap.delete(product.id);
      } else {
        newMap.set(product.id, {
          id: product.id,
          name: product.product_name || product.barcode,
          category: product.category,
          brand: product.brand_name,
          primary_image_url: product.primary_image_url,
          includeSynthetic: defaultIncludeSynthetic,
          includeAugmented: defaultIncludeAugmented,
          includeReal: defaultIncludeReal,
        });
      }
      return newMap;
    });
  }, [defaultIncludeSynthetic, defaultIncludeAugmented, defaultIncludeReal]);

  const selectAllProducts = useCallback(() => {
    if (!productsData?.items) return;
    setSelectedProducts(prev => {
      const newMap = new Map(prev);
      productsData.items.forEach((product: any) => {
        if (!newMap.has(product.id)) {
          newMap.set(product.id, {
            id: product.id,
            name: product.product_name || product.barcode,
            category: product.category,
            brand: product.brand_name,
            primary_image_url: product.primary_image_url,
            includeSynthetic: defaultIncludeSynthetic,
            includeAugmented: defaultIncludeAugmented,
            includeReal: defaultIncludeReal,
          });
        }
      });
      return newMap;
    });
  }, [productsData, defaultIncludeSynthetic, defaultIncludeAugmented, defaultIncludeReal]);

  const clearProductSelection = useCallback(() => {
    setSelectedProducts(new Map());
  }, []);

  // ===========================================
  // Cutout Selection Handlers
  // ===========================================
  const toggleCutoutSelection = useCallback((cutout: any) => {
    setSelectedCutouts(prev => {
      const newMap = new Map(prev);
      if (newMap.has(cutout.id)) {
        newMap.delete(cutout.id);
      } else {
        newMap.set(cutout.id, {
          id: cutout.id,
          image_url: cutout.image_url,
          predicted_upc: cutout.predicted_upc,
          matched_product_name: cutout.matched_product?.product_name,
        });
      }
      return newMap;
    });
  }, []);

  const selectAllCutoutsInPage = useCallback(() => {
    if (!cutoutsData?.items) return;
    setSelectedCutouts(prev => {
      const newMap = new Map(prev);
      cutoutsData.items.forEach((cutout: any) => {
        if (!newMap.has(cutout.id)) {
          newMap.set(cutout.id, {
            id: cutout.id,
            image_url: cutout.image_url,
            predicted_upc: cutout.predicted_upc,
            matched_product_name: cutout.matched_product?.product_name,
          });
        }
      });
      return newMap;
    });
  }, [cutoutsData]);

  const clearCutoutSelection = useCallback(() => {
    setSelectedCutouts(new Map());
  }, []);

  // ===========================================
  // OD Image Selection Handlers
  // ===========================================
  const toggleODSelection = useCallback((odImage: any) => {
    setSelectedODImages(prev => {
      const newMap = new Map(prev);
      if (newMap.has(odImage.id)) {
        newMap.delete(odImage.id);
      } else {
        newMap.set(odImage.id, {
          id: odImage.id,
          image_url: odImage.image_url || odImage.thumbnail_url,
          filename: odImage.filename,
        });
      }
      return newMap;
    });
  }, []);

  const selectAllODInPage = useCallback(() => {
    if (!odData?.images) return;
    setSelectedODImages(prev => {
      const newMap = new Map(prev);
      odData.images.forEach((odImage: any) => {
        if (!newMap.has(odImage.id)) {
          newMap.set(odImage.id, {
            id: odImage.id,
            image_url: odImage.image_url || odImage.thumbnail_url,
            filename: odImage.filename,
          });
        }
      });
      return newMap;
    });
  }, [odData]);

  const clearODSelection = useCallback(() => {
    setSelectedODImages(new Map());
  }, []);

  // ===========================================
  // Select All Filtered Handlers
  // ===========================================
  const handleSelectAllFilteredProducts = useCallback(async () => {
    setIsSelectingAll(true);
    try {
      const result = await apiClient.getProductBulkIds({
        search: productsSearch || undefined,
        category: productsCategoryFilter !== "all" ? productsCategoryFilter : undefined,
        brand: productsBrandFilter !== "all" ? productsBrandFilter : undefined,
        has_image: true,
      });

      const productSelections = new Map<string, SelectedProduct>();
      result.ids.forEach((id) => {
        productSelections.set(id, {
          id,
          name: `Product ${id.slice(0, 8)}...`,
          includeSynthetic: defaultIncludeSynthetic,
          includeAugmented: defaultIncludeAugmented,
          includeReal: defaultIncludeReal,
        });
      });
      setSelectedProducts(productSelections);
      toast.success(`Selected ${result.total.toLocaleString()} products`);
    } catch (error) {
      toast.error(`Failed to fetch products: ${error}`);
    } finally {
      setIsSelectingAll(false);
    }
  }, [productsSearch, productsCategoryFilter, productsBrandFilter, defaultIncludeSynthetic, defaultIncludeAugmented, defaultIncludeReal]);

  const handleSelectAllFilteredCutouts = useCallback(async () => {
    setIsSelectingAll(true);
    try {
      const result = await apiClient.getCutoutBulkIds({
        search: cutoutsSearch || undefined,
        is_matched: cutoutsMatchedOnly ? true : undefined,
      });

      const cutoutSelections = new Map<string, SelectedCutout>();
      result.ids.forEach((id) => {
        cutoutSelections.set(id, {
          id,
          image_url: "",
        });
      });
      setSelectedCutouts(cutoutSelections);
      toast.success(`Selected ${result.total.toLocaleString()} cutouts`);
    } catch (error) {
      toast.error(`Failed to fetch cutouts: ${error}`);
    } finally {
      setIsSelectingAll(false);
    }
  }, [cutoutsSearch, cutoutsMatchedOnly]);

  const handleSelectAllFilteredOD = useCallback(async () => {
    setIsSelectingAll(true);
    try {
      const result = await apiClient.getODImageBulkIds({
        search: odSearch || undefined,
      });

      const odSelections = new Map<string, SelectedODImage>();
      result.ids.forEach((id) => {
        odSelections.set(id, {
          id,
          image_url: "",
        });
      });
      setSelectedODImages(odSelections);
      toast.success(`Selected ${result.total.toLocaleString()} OD images`);
    } catch (error) {
      toast.error(`Failed to fetch OD images: ${error}`);
    } finally {
      setIsSelectingAll(false);
    }
  }, [odSearch]);

  // ===========================================
  // Upload Handling
  // ===========================================
  const onDrop = useCallback((acceptedFiles: File[]) => {
    setUploadFiles(prev => [...prev, ...acceptedFiles]);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [".jpg", ".jpeg", ".png", ".webp"] },
    multiple: true,
  });

  const removeUploadFile = useCallback((index: number) => {
    setUploadFiles(prev => prev.filter((_, i) => i !== index));
  }, []);

  // ===========================================
  // Chunked Import Handler
  // ===========================================
  const handleChunkedImport = useCallback(async () => {
    resetImportProgress();

    // Determine which source to import from (prioritize based on active tab)
    if (activeTab === "products" && selectedProducts.size > 0) {
      const productIds = Array.from(selectedProducts.keys());
      const imageTypes: string[] = [];

      if (defaultIncludeSynthetic) imageTypes.push("synthetic");
      if (defaultIncludeAugmented) imageTypes.push("augmented");
      if (defaultIncludeReal) imageTypes.push("real");
      if (imageTypes.length === 0) {
        imageTypes.push("synthetic", "augmented", "real");
      }

      let productLabelSource = productsLabelStrategy;
      let productLabel: string | null = null;
      if (productsLabelStrategy === "custom") {
        productLabel = productsCustomLabel.trim() || null;
      } else if (productsLabelStrategy === "unlabeled") {
        productLabelSource = "none";
      }

      await importChunked(productIds, async (chunkIds): Promise<ChunkResult> => {
        const result = await apiClient.importCLSImagesFromProducts({
          product_ids: chunkIds,
          dataset_id: datasetId,
          label_source: productLabelSource,
          custom_label: productLabel,
          image_types: imageTypes,
          skip_duplicates: true,
        });
        return {
          success: result.success ?? true,
          images_imported: result.images_imported || 0,
          labels_created: result.labels_created || 0,
          classes_created: result.classes_created || 0,
          errors: result.errors || [],
        };
      });
    } else if (activeTab === "cutouts" && selectedCutouts.size > 0) {
      const cutoutIds = Array.from(selectedCutouts.keys());

      let cutoutLabelSource = cutoutsLabelStrategy;
      let cutoutLabel: string | null = null;
      if (cutoutsLabelStrategy === "custom") {
        cutoutLabel = cutoutsCustomLabel.trim() || null;
      } else if (cutoutsLabelStrategy === "unlabeled") {
        cutoutLabelSource = "none";
      }

      await importChunked(cutoutIds, async (chunkIds): Promise<ChunkResult> => {
        const result = await apiClient.importCLSImagesFromCutouts({
          cutout_ids: chunkIds,
          dataset_id: datasetId,
          label_source: cutoutLabelSource,
          custom_label: cutoutLabel,
          skip_duplicates: true,
        });
        return {
          success: result.success ?? true,
          images_imported: result.images_imported || 0,
          labels_created: result.labels_created || 0,
          classes_created: result.classes_created || 0,
          errors: result.errors || [],
        };
      });
    } else if (activeTab === "od" && selectedODImages.size > 0) {
      const odIds = Array.from(selectedODImages.keys());

      let odLabel: string | null = null;
      if (odLabelStrategy === "custom") {
        odLabel = odCustomLabel.trim() || null;
      }

      await importChunked(odIds, async (chunkIds): Promise<ChunkResult> => {
        const result = await apiClient.importCLSImagesFromOD({
          od_image_ids: chunkIds,
          dataset_id: datasetId,
          label: odLabel,
          skip_duplicates: true,
        });
        return {
          success: result.success ?? true,
          images_imported: result.images_imported || 0,
          images_skipped: result.images_skipped || 0,
          errors: result.errors || [],
        };
      });
    } else if (activeTab === "upload" && uploadFiles.length > 0) {
      // Upload files sequentially (not chunked, but show progress)
      setIsUploading(true);
      let uploadLabel: string | null = null;
      if (uploadLabelStrategy === "custom") {
        uploadLabel = uploadCustomLabel.trim() || null;
      }

      let imported = 0;
      let errors = 0;
      for (const file of uploadFiles) {
        try {
          await apiClient.uploadCLSImage(file, datasetId, uploadLabel);
          imported++;
        } catch {
          errors++;
        }
      }
      setIsUploading(false);

      if (errors === 0) {
        toast.success(`Uploaded ${imported} images`);
        queryClient.invalidateQueries({ queryKey: ["cls-dataset", datasetId] });
        queryClient.invalidateQueries({ queryKey: ["cls-dataset-images", datasetId] });
        setUploadFiles([]);
        onSuccess?.();
        onOpenChange(false);
      } else {
        toast.warning(`Uploaded ${imported} images with ${errors} errors`);
        queryClient.invalidateQueries({ queryKey: ["cls-dataset", datasetId] });
      }
    } else if (activeTab === "url" && urlInput.trim()) {
      const urls = urlInput.split("\n").map((u) => u.trim()).filter(Boolean);

      let urlLabel: string | null = null;
      if (urlLabelStrategy === "custom") {
        urlLabel = urlCustomLabel.trim() || null;
      }

      // Chunk URLs (100 per chunk as backend limit)
      await importChunked(urls, async (chunkUrls): Promise<ChunkResult> => {
        const result = await apiClient.importCLSImagesFromURLs({
          urls: chunkUrls,
          dataset_id: datasetId,
          label: urlLabel,
          skip_duplicates: true,
        });
        return {
          success: result.success ?? true,
          images_imported: result.images_imported || 0,
          images_skipped: result.images_skipped || 0,
          duplicates_found: result.duplicates_found || 0,
          errors: result.errors || [],
        };
      });
    }
  }, [
    activeTab,
    selectedProducts,
    selectedCutouts,
    selectedODImages,
    uploadFiles,
    urlInput,
    datasetId,
    defaultIncludeSynthetic,
    defaultIncludeAugmented,
    defaultIncludeReal,
    productsLabelStrategy,
    productsCustomLabel,
    cutoutsLabelStrategy,
    cutoutsCustomLabel,
    odLabelStrategy,
    odCustomLabel,
    uploadLabelStrategy,
    uploadCustomLabel,
    urlLabelStrategy,
    urlCustomLabel,
    importChunked,
    resetImportProgress,
    queryClient,
    onSuccess,
    onOpenChange,
  ]);

  // Retry handler for failed chunks
  const handleRetryFailed = useCallback(async () => {
    if (activeTab === "products") {
      const imageTypes: string[] = [];
      if (defaultIncludeSynthetic) imageTypes.push("synthetic");
      if (defaultIncludeAugmented) imageTypes.push("augmented");
      if (defaultIncludeReal) imageTypes.push("real");
      if (imageTypes.length === 0) imageTypes.push("synthetic", "augmented", "real");

      let productLabelSource = productsLabelStrategy;
      let productLabel: string | null = null;
      if (productsLabelStrategy === "custom") productLabel = productsCustomLabel.trim() || null;
      else if (productsLabelStrategy === "unlabeled") productLabelSource = "none";

      await retryFailed(async (chunkIds): Promise<ChunkResult> => {
        const result = await apiClient.importCLSImagesFromProducts({
          product_ids: chunkIds,
          dataset_id: datasetId,
          label_source: productLabelSource,
          custom_label: productLabel,
          image_types: imageTypes,
          skip_duplicates: true,
        });
        return {
          success: result.success ?? true,
          images_imported: result.images_imported || 0,
          labels_created: result.labels_created || 0,
          classes_created: result.classes_created || 0,
          errors: result.errors || [],
        };
      });
    } else if (activeTab === "cutouts") {
      let cutoutLabelSource = cutoutsLabelStrategy;
      let cutoutLabel: string | null = null;
      if (cutoutsLabelStrategy === "custom") cutoutLabel = cutoutsCustomLabel.trim() || null;
      else if (cutoutsLabelStrategy === "unlabeled") cutoutLabelSource = "none";

      await retryFailed(async (chunkIds): Promise<ChunkResult> => {
        const result = await apiClient.importCLSImagesFromCutouts({
          cutout_ids: chunkIds,
          dataset_id: datasetId,
          label_source: cutoutLabelSource,
          custom_label: cutoutLabel,
          skip_duplicates: true,
        });
        return {
          success: result.success ?? true,
          images_imported: result.images_imported || 0,
          labels_created: result.labels_created || 0,
          classes_created: result.classes_created || 0,
          errors: result.errors || [],
        };
      });
    } else if (activeTab === "od") {
      let odLabel: string | null = null;
      if (odLabelStrategy === "custom") odLabel = odCustomLabel.trim() || null;

      await retryFailed(async (chunkIds): Promise<ChunkResult> => {
        const result = await apiClient.importCLSImagesFromOD({
          od_image_ids: chunkIds,
          dataset_id: datasetId,
          label: odLabel,
          skip_duplicates: true,
        });
        return {
          success: result.success ?? true,
          images_imported: result.images_imported || 0,
          images_skipped: result.images_skipped || 0,
          errors: result.errors || [],
        };
      });
    } else if (activeTab === "url") {
      let urlLabel: string | null = null;
      if (urlLabelStrategy === "custom") urlLabel = urlCustomLabel.trim() || null;

      await retryFailed(async (chunkUrls): Promise<ChunkResult> => {
        const result = await apiClient.importCLSImagesFromURLs({
          urls: chunkUrls,
          dataset_id: datasetId,
          label: urlLabel,
          skip_duplicates: true,
        });
        return {
          success: result.success ?? true,
          images_imported: result.images_imported || 0,
          images_skipped: result.images_skipped || 0,
          duplicates_found: result.duplicates_found || 0,
          errors: result.errors || [],
        };
      });
    }
  }, [
    activeTab,
    datasetId,
    defaultIncludeSynthetic,
    defaultIncludeAugmented,
    defaultIncludeReal,
    productsLabelStrategy,
    productsCustomLabel,
    cutoutsLabelStrategy,
    cutoutsCustomLabel,
    odLabelStrategy,
    odCustomLabel,
    urlLabelStrategy,
    urlCustomLabel,
    retryFailed,
  ]);

  // ===========================================
  // Selection Counts
  // ===========================================
  const totalSelected = useMemo(() => {
    return selectedProducts.size + selectedCutouts.size + selectedODImages.size + uploadFiles.length + 
      (urlInput.trim() ? urlInput.split("\n").filter(Boolean).length : 0);
  }, [selectedProducts, selectedCutouts, selectedODImages, uploadFiles, urlInput]);

  // ===========================================
  // Reset on close
  // ===========================================
  const handleClose = useCallback(() => {
    setSelectedProducts(new Map());
    setSelectedCutouts(new Map());
    setSelectedODImages(new Map());
    setUploadFiles([]);
    setUrlInput("");
    setProductsPage(1);
    setCutoutsPage(1);
    setOdPage(1);
    onOpenChange(false);
  }, [onOpenChange]);

  // ===========================================
  // Render
  // ===========================================
  return (
    <DialogPrimitive.Root open={open} onOpenChange={handleClose}>
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay className="fixed inset-0 z-50 bg-black/50" />
        <DialogPrimitive.Content className="fixed left-[50%] top-[50%] z-50 translate-x-[-50%] translate-y-[-50%] w-[95vw] max-w-[1400px] h-[90vh] bg-background rounded-lg border shadow-lg flex flex-col overflow-hidden">
          
          {/* Header */}
          <div className="flex-shrink-0 px-6 py-4 border-b bg-background flex items-start justify-between">
            <div>
              <DialogPrimitive.Title className="text-lg font-semibold flex items-center gap-2">
                <Package className="h-5 w-5" />
                Import Images to Dataset
              </DialogPrimitive.Title>
              <DialogPrimitive.Description className="text-sm text-muted-foreground mt-1">
                Select products, cutouts, or OD images to add to your classification dataset
              </DialogPrimitive.Description>
            </div>
            <DialogPrimitive.Close asChild>
              <Button variant="ghost" size="icon" className="h-8 w-8 rounded-full">
                <X className="h-4 w-4" />
              </Button>
            </DialogPrimitive.Close>
          </div>

          {/* Main Content */}
          <div className="flex-1 flex min-h-0 overflow-hidden">
            {/* Sidebar - Source Tabs */}
            <div className="w-48 flex-shrink-0 border-r bg-muted/30 p-2 flex flex-col gap-1">
              {[
                { id: "products" as const, label: "Products", icon: Package, count: selectedProducts.size },
                { id: "cutouts" as const, label: "Cutouts", icon: Scissors, count: selectedCutouts.size },
                { id: "od" as const, label: "OD Images", icon: ImageIcon, count: selectedODImages.size },
                { id: "upload" as const, label: "Upload", icon: Upload, count: uploadFiles.length },
                { id: "url" as const, label: "URL Import", icon: LinkIcon, count: urlInput.trim() ? urlInput.split("\n").filter(Boolean).length : 0 },
              ].map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={cn(
                    "w-full flex items-center gap-2 px-3 py-2 rounded-md text-sm transition-colors text-left",
                    activeTab === tab.id
                      ? "bg-primary text-primary-foreground"
                      : "hover:bg-muted"
                  )}
                >
                  <tab.icon className="h-4 w-4 flex-shrink-0" />
                  <span className="flex-1">{tab.label}</span>
                  {tab.count > 0 && (
                    <Badge variant={activeTab === tab.id ? "secondary" : "default"} className="text-xs">
                      {tab.count}
                    </Badge>
                  )}
                </button>
              ))}
            </div>

            {/* Content Area */}
            <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
              
              {/* ==================== PRODUCTS TAB ==================== */}
              {activeTab === "products" && (
                <>
                  {/* Filters */}
                  <div className="flex-shrink-0 p-3 border-b bg-muted/20 flex flex-wrap gap-2 items-center">
                    <div className="relative flex-1 min-w-[200px] max-w-[300px]">
                      <Search className="absolute left-2 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                      <Input
                        placeholder="Search products..."
                        value={productsSearch}
                        onChange={(e) => {
                          setProductsSearch(e.target.value);
                          setProductsPage(1);
                        }}
                        className="pl-8 h-9"
                      />
                    </div>
                    <Select value={productsCategoryFilter} onValueChange={(v) => { setProductsCategoryFilter(v); setProductsPage(1); }}>
                      <SelectTrigger className="w-[150px] h-9">
                        <SelectValue placeholder="Category" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="all">All Categories</SelectItem>
                        {productsFilterOptions?.category?.map((cat: any, idx: number) => {
                          const value = typeof cat === 'string' ? cat : cat.value;
                          const label = typeof cat === 'string' ? cat : (cat.label || cat.value);
                          return <SelectItem key={`cat-${idx}-${value}`} value={value}>{label}</SelectItem>;
                        })}
                      </SelectContent>
                    </Select>
                    <Select value={productsBrandFilter} onValueChange={(v) => { setProductsBrandFilter(v); setProductsPage(1); }}>
                      <SelectTrigger className="w-[150px] h-9">
                        <SelectValue placeholder="Brand" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="all">All Brands</SelectItem>
                        {productsFilterOptions?.brand?.map((brand: any, idx: number) => {
                          const value = typeof brand === 'string' ? brand : brand.value;
                          const label = typeof brand === 'string' ? brand : (brand.label || brand.value);
                          return <SelectItem key={`brand-${idx}-${value}`} value={value}>{label}</SelectItem>;
                        })}
                      </SelectContent>
                    </Select>
                    <div className="flex items-center gap-2 ml-auto">
                      <Button variant="outline" size="sm" onClick={selectAllProducts} disabled={isSelectingAll || isImporting}>
                        <CheckSquare className="h-4 w-4 mr-1" />
                        Select Page
                      </Button>
                      <Button
                        variant="default"
                        size="sm"
                        onClick={handleSelectAllFilteredProducts}
                        disabled={isSelectingAll || isImporting}
                      >
                        {isSelectingAll ? (
                          <Loader2 className="h-4 w-4 mr-1 animate-spin" />
                        ) : (
                          <ListChecks className="h-4 w-4 mr-1" />
                        )}
                        Select All ({productsData?.total?.toLocaleString() || 0})
                      </Button>
                      {selectedProducts.size > 0 && (
                        <Button variant="ghost" size="sm" onClick={clearProductSelection} disabled={isImporting}>
                          <X className="h-4 w-4 mr-1" />
                          Clear ({selectedProducts.size.toLocaleString()})
                        </Button>
                      )}
                    </div>
                  </div>

                  {/* Image Type Selection */}
                  <div className="flex-shrink-0 px-3 py-2 border-b bg-muted/10 flex items-center gap-4">
                    <span className="text-sm text-muted-foreground">Include frame types:</span>
                    <label className="flex items-center gap-2 cursor-pointer">
                      <Checkbox
                        checked={defaultIncludeSynthetic}
                        onCheckedChange={(c) => setDefaultIncludeSynthetic(!!c)}
                      />
                      <span className="text-sm">Synthetic</span>
                    </label>
                    <label className="flex items-center gap-2 cursor-pointer">
                      <Checkbox
                        checked={defaultIncludeAugmented}
                        onCheckedChange={(c) => setDefaultIncludeAugmented(!!c)}
                      />
                      <span className="text-sm">Augmented</span>
                    </label>
                    <label className="flex items-center gap-2 cursor-pointer">
                      <Checkbox
                        checked={defaultIncludeReal}
                        onCheckedChange={(c) => setDefaultIncludeReal(!!c)}
                      />
                      <span className="text-sm">Real</span>
                    </label>
                  </div>

                  {/* Product Grid */}
                  <ScrollArea className="flex-1">
                    <div className="p-3">
                      {productsLoading ? (
                        <div className="flex items-center justify-center py-12">
                          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
                        </div>
                      ) : !productsData?.items?.length ? (
                        <div className="text-center py-12 text-muted-foreground">
                          No products found
                        </div>
                      ) : (
                        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-3">
                          {productsData.items.map((product: any) => {
                            const isSelected = selectedProducts.has(product.id);
                            return (
                              <div
                                key={product.id}
                                onClick={() => toggleProductSelection(product)}
                                className={cn(
                                  "relative rounded-lg border-2 overflow-hidden cursor-pointer transition-all hover:shadow-md",
                                  isSelected ? "border-primary bg-primary/5" : "border-transparent hover:border-muted-foreground/30"
                                )}
                              >
                                <div className="aspect-square relative bg-muted">
                                  {product.primary_image_url ? (
                                    <Image
                                      src={product.primary_image_url}
                                      alt={product.product_name || "Product"}
                                      fill
                                      className="object-cover"
                                      sizes="(max-width: 640px) 50vw, (max-width: 1024px) 33vw, 16vw"
                                    />
                                  ) : (
                                    <div className="flex items-center justify-center h-full">
                                      <Package className="h-8 w-8 text-muted-foreground" />
                                    </div>
                                  )}
                                  {/* Selection indicator */}
                                  <div className={cn(
                                    "absolute top-2 right-2 h-6 w-6 rounded-full flex items-center justify-center transition-colors",
                                    isSelected ? "bg-primary text-primary-foreground" : "bg-background/80"
                                  )}>
                                    {isSelected ? (
                                      <CheckCircle className="h-4 w-4" />
                                    ) : (
                                      <div className="h-4 w-4 rounded-full border-2" />
                                    )}
                                  </div>
                                </div>
                                <div className="p-2">
                                  <p className="text-xs font-medium truncate">{product.product_name || product.barcode}</p>
                                  <p className="text-xs text-muted-foreground truncate">{product.category || product.brand_name}</p>
                                  <div className="flex gap-1 mt-1">
                                    {product.frame_count > 0 && (
                                      <Badge variant="secondary" className="text-[10px] px-1">
                                        {product.frame_count} frames
                                      </Badge>
                                    )}
                                  </div>
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      )}
                      
                      {/* Pagination - Always show if there's data */}
                      {productsData && productsData.items?.length > 0 && (
                        <div className="mt-4">
                          <div className="text-center text-sm text-muted-foreground mb-2">
                            Showing {productsData.items.length} of {productsData.total} products
                          </div>
                          {productsData.total > 30 && (
                            <PaginationControl
                              currentPage={productsPage}
                              totalPages={Math.ceil(productsData.total / 30)}
                              onPageChange={setProductsPage}
                            />
                          )}
                        </div>
                      )}
                    </div>
                  </ScrollArea>
                </>
              )}

              {/* ==================== CUTOUTS TAB ==================== */}
              {activeTab === "cutouts" && (
                <>
                  <div className="flex-shrink-0 p-3 border-b bg-muted/20 flex flex-wrap gap-2 items-center">
                    <div className="relative flex-1 min-w-[200px] max-w-[300px]">
                      <Search className="absolute left-2 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                      <Input
                        placeholder="Search cutouts..."
                        value={cutoutsSearch}
                        onChange={(e) => { setCutoutsSearch(e.target.value); setCutoutsPage(1); }}
                        className="pl-8 h-9"
                      />
                    </div>
                    <label className="flex items-center gap-2 cursor-pointer">
                      <Checkbox
                        checked={cutoutsMatchedOnly}
                        onCheckedChange={(c) => { setCutoutsMatchedOnly(!!c); setCutoutsPage(1); }}
                      />
                      <span className="text-sm">Matched only</span>
                    </label>
                    <div className="flex items-center gap-2 ml-auto">
                      <Button variant="outline" size="sm" onClick={selectAllCutoutsInPage} disabled={isSelectingAll || isImporting}>
                        <CheckSquare className="h-4 w-4 mr-1" />
                        Select Page
                      </Button>
                      <Button
                        variant="default"
                        size="sm"
                        onClick={handleSelectAllFilteredCutouts}
                        disabled={isSelectingAll || isImporting}
                      >
                        {isSelectingAll ? (
                          <Loader2 className="h-4 w-4 mr-1 animate-spin" />
                        ) : (
                          <ListChecks className="h-4 w-4 mr-1" />
                        )}
                        Select All ({cutoutsData?.total?.toLocaleString() || 0})
                      </Button>
                      {selectedCutouts.size > 0 && (
                        <Button variant="ghost" size="sm" onClick={clearCutoutSelection} disabled={isImporting}>
                          <X className="h-4 w-4 mr-1" />
                          Clear ({selectedCutouts.size.toLocaleString()})
                        </Button>
                      )}
                    </div>
                  </div>

                  <ScrollArea className="flex-1">
                    <div className="p-3">
                      {cutoutsLoading ? (
                        <div className="flex items-center justify-center py-12">
                          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
                        </div>
                      ) : !cutoutsData?.items?.length ? (
                        <div className="text-center py-12 text-muted-foreground">
                          No cutouts found
                        </div>
                      ) : (
                        <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-5 lg:grid-cols-6 xl:grid-cols-8 gap-2">
                          {cutoutsData.items.map((cutout: any) => {
                            const isSelected = selectedCutouts.has(cutout.id);
                            return (
                              <div
                                key={cutout.id}
                                onClick={() => toggleCutoutSelection(cutout)}
                                className={cn(
                                  "relative rounded-lg border-2 overflow-hidden cursor-pointer transition-all hover:shadow-md",
                                  isSelected ? "border-primary bg-primary/5" : "border-transparent hover:border-muted-foreground/30"
                                )}
                              >
                                <div className="aspect-square relative bg-muted">
                                  <Image
                                    src={cutout.image_url}
                                    alt={cutout.predicted_upc || "Cutout"}
                                    fill
                                    className="object-cover"
                                    sizes="120px"
                                  />
                                  <div className={cn(
                                    "absolute top-1 right-1 h-5 w-5 rounded-full flex items-center justify-center transition-colors",
                                    isSelected ? "bg-primary text-primary-foreground" : "bg-background/80"
                                  )}>
                                    {isSelected ? <CheckCircle className="h-3 w-3" /> : <div className="h-3 w-3 rounded-full border-2" />}
                                  </div>
                                  {cutout.matched_product_id && (
                                    <Badge className="absolute bottom-1 left-1 text-[10px] px-1" variant="secondary">
                                      Matched
                                    </Badge>
                                  )}
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      )}

                      {cutoutsData && cutoutsData.items?.length > 0 && (
                        <div className="mt-4">
                          <div className="text-center text-sm text-muted-foreground mb-2">
                            Showing {cutoutsData.items.length} of {cutoutsData.total} cutouts
                          </div>
                          {cutoutsData.total > 30 && (
                            <PaginationControl
                              currentPage={cutoutsPage}
                              totalPages={Math.ceil(cutoutsData.total / 30)}
                              onPageChange={setCutoutsPage}
                            />
                          )}
                        </div>
                      )}
                    </div>
                  </ScrollArea>
                </>
              )}

              {/* ==================== OD IMAGES TAB ==================== */}
              {activeTab === "od" && (
                <>
                  <div className="flex-shrink-0 p-3 border-b bg-muted/20 flex flex-wrap gap-2 items-center">
                    <div className="relative flex-1 min-w-[200px] max-w-[300px]">
                      <Search className="absolute left-2 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                      <Input
                        placeholder="Search OD images..."
                        value={odSearch}
                        onChange={(e) => { setOdSearch(e.target.value); setOdPage(1); }}
                        className="pl-8 h-9"
                      />
                    </div>
                    <div className="flex items-center gap-2 ml-auto">
                      <Button variant="outline" size="sm" onClick={selectAllODInPage} disabled={isSelectingAll || isImporting}>
                        <CheckSquare className="h-4 w-4 mr-1" />
                        Select Page
                      </Button>
                      <Button
                        variant="default"
                        size="sm"
                        onClick={handleSelectAllFilteredOD}
                        disabled={isSelectingAll || isImporting}
                      >
                        {isSelectingAll ? (
                          <Loader2 className="h-4 w-4 mr-1 animate-spin" />
                        ) : (
                          <ListChecks className="h-4 w-4 mr-1" />
                        )}
                        Select All ({odData?.total?.toLocaleString() || 0})
                      </Button>
                      {selectedODImages.size > 0 && (
                        <Button variant="ghost" size="sm" onClick={clearODSelection} disabled={isImporting}>
                          <X className="h-4 w-4 mr-1" />
                          Clear ({selectedODImages.size.toLocaleString()})
                        </Button>
                      )}
                    </div>
                  </div>

                  <ScrollArea className="flex-1">
                    <div className="p-3">
                      {odLoading ? (
                        <div className="flex items-center justify-center py-12">
                          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
                        </div>
                      ) : !odData?.images?.length ? (
                        <div className="text-center py-12 text-muted-foreground">
                          No OD images found
                        </div>
                      ) : (
                        <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-5 lg:grid-cols-6 xl:grid-cols-8 gap-2">
                          {odData.images.map((odImage: any) => {
                            const isSelected = selectedODImages.has(odImage.id);
                            return (
                              <div
                                key={odImage.id}
                                onClick={() => toggleODSelection(odImage)}
                                className={cn(
                                  "relative rounded-lg border-2 overflow-hidden cursor-pointer transition-all hover:shadow-md",
                                  isSelected ? "border-primary bg-primary/5" : "border-transparent hover:border-muted-foreground/30"
                                )}
                              >
                                <div className="aspect-square relative bg-muted">
                                  <Image
                                    src={odImage.thumbnail_url || odImage.image_url}
                                    alt={odImage.filename || "OD Image"}
                                    fill
                                    className="object-cover"
                                    sizes="120px"
                                  />
                                  <div className={cn(
                                    "absolute top-1 right-1 h-5 w-5 rounded-full flex items-center justify-center transition-colors",
                                    isSelected ? "bg-primary text-primary-foreground" : "bg-background/80"
                                  )}>
                                    {isSelected ? <CheckCircle className="h-3 w-3" /> : <div className="h-3 w-3 rounded-full border-2" />}
                                  </div>
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      )}

                      {odData && odData.images?.length > 0 && (
                        <div className="mt-4">
                          <div className="text-center text-sm text-muted-foreground mb-2">
                            Showing {odData.images.length} of {odData.total} OD images
                          </div>
                          {odData.total > 30 && (
                            <PaginationControl
                              currentPage={odPage}
                              totalPages={Math.ceil(odData.total / 30)}
                              onPageChange={setOdPage}
                            />
                          )}
                        </div>
                      )}
                    </div>
                  </ScrollArea>
                </>
              )}

              {/* ==================== UPLOAD TAB ==================== */}
              {activeTab === "upload" && (
                <div className="flex-1 p-4 overflow-auto">
                  <div
                    {...getRootProps()}
                    className={cn(
                      "border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors",
                      isDragActive ? "border-primary bg-primary/5" : "border-muted-foreground/30 hover:border-primary"
                    )}
                  >
                    <input {...getInputProps()} />
                    <Upload className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                    <p className="text-lg font-medium">Drop images here or click to browse</p>
                    <p className="text-sm text-muted-foreground mt-1">Supports JPG, PNG, WebP</p>
                  </div>

                  {uploadFiles.length > 0 && (
                    <div className="mt-4">
                      <div className="flex items-center justify-between mb-2">
                        <span className="font-medium">{uploadFiles.length} files selected</span>
                        <Button variant="ghost" size="sm" onClick={() => setUploadFiles([])}>
                          Clear all
                        </Button>
                      </div>
                      <div className="grid grid-cols-4 sm:grid-cols-6 md:grid-cols-8 gap-2">
                        {uploadFiles.map((file, index) => (
                          <div key={index} className="relative group">
                            <div className="aspect-square rounded-lg overflow-hidden bg-muted">
                              <Image
                                src={URL.createObjectURL(file)}
                                alt={file.name}
                                fill
                                className="object-cover"
                              />
                            </div>
                            <button
                              onClick={() => removeUploadFile(index)}
                              className="absolute -top-1 -right-1 h-5 w-5 rounded-full bg-destructive text-destructive-foreground flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
                            >
                              <X className="h-3 w-3" />
                            </button>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* ==================== URL TAB ==================== */}
              {activeTab === "url" && (
                <div className="flex-1 p-4 overflow-auto">
                  <Label htmlFor="url-input">Image URLs (one per line)</Label>
                  <textarea
                    id="url-input"
                    value={urlInput}
                    onChange={(e) => setUrlInput(e.target.value)}
                    placeholder="https://example.com/image1.jpg&#10;https://example.com/image2.jpg"
                    className="mt-2 w-full h-[300px] p-3 rounded-md border bg-background resize-none font-mono text-sm"
                  />
                  {urlInput.trim() && (
                    <p className="mt-2 text-sm text-muted-foreground">
                      {urlInput.split("\n").filter(Boolean).length} URLs entered
                    </p>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Footer - Label Strategy Selection */}
          <div className="flex-shrink-0 px-6 py-4 border-t bg-muted/30 flex flex-col gap-3">
            
            {/* Label Strategy Selection per Tab */}
            <div className="flex flex-wrap items-center gap-3">
              <span className="text-sm font-medium">Label Strategy:</span>
              
              {/* Products Label Strategy */}
              {activeTab === "products" && selectedProducts.size > 0 && (
                <>
                  <Select value={productsLabelStrategy} onValueChange={setProductsLabelStrategy}>
                    <SelectTrigger className="w-[200px] h-9">
                      <SelectValue placeholder="How to label?" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="product_id">Product ID</SelectItem>
                      <SelectItem value="category">Category</SelectItem>
                      <SelectItem value="brand">Brand</SelectItem>
                      <SelectItem value="product_name">Product Name</SelectItem>
                      <SelectItem value="custom">Custom Class</SelectItem>
                      <SelectItem value="unlabeled">No Label (Unlabeled)</SelectItem>
                    </SelectContent>
                  </Select>
                  {productsLabelStrategy === "custom" && (
                    <Input
                      placeholder="Enter class name..."
                      value={productsCustomLabel}
                      onChange={(e) => setProductsCustomLabel(e.target.value)}
                      className="w-[200px] h-9"
                    />
                  )}
                </>
              )}
              
              {/* Cutouts Label Strategy */}
              {activeTab === "cutouts" && selectedCutouts.size > 0 && (
                <>
                  <Select value={cutoutsLabelStrategy} onValueChange={setCutoutsLabelStrategy}>
                    <SelectTrigger className="w-[200px] h-9">
                      <SelectValue placeholder="How to label?" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="matched_product_id">Matched Product ID</SelectItem>
                      <SelectItem value="custom">Custom Class</SelectItem>
                      <SelectItem value="unlabeled">No Label (Unlabeled)</SelectItem>
                    </SelectContent>
                  </Select>
                  {cutoutsLabelStrategy === "custom" && (
                    <Input
                      placeholder="Enter class name..."
                      value={cutoutsCustomLabel}
                      onChange={(e) => setCutoutsCustomLabel(e.target.value)}
                      className="w-[200px] h-9"
                    />
                  )}
                  {cutoutsLabelStrategy === "matched_product_id" && (
                    <span className="text-xs text-muted-foreground">
                      (Unmatched cutouts will be unlabeled)
                    </span>
                  )}
                </>
              )}
              
              {/* OD Images Label Strategy */}
              {activeTab === "od" && selectedODImages.size > 0 && (
                <>
                  <Select value={odLabelStrategy} onValueChange={setOdLabelStrategy}>
                    <SelectTrigger className="w-[200px] h-9">
                      <SelectValue placeholder="How to label?" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="custom">Custom Class</SelectItem>
                      <SelectItem value="unlabeled">No Label (Unlabeled)</SelectItem>
                    </SelectContent>
                  </Select>
                  {odLabelStrategy === "custom" && (
                    <Input
                      placeholder="Enter class name..."
                      value={odCustomLabel}
                      onChange={(e) => setOdCustomLabel(e.target.value)}
                      className="w-[200px] h-9"
                    />
                  )}
                </>
              )}
              
              {/* Upload Label Strategy */}
              {activeTab === "upload" && uploadFiles.length > 0 && (
                <>
                  <Select value={uploadLabelStrategy} onValueChange={setUploadLabelStrategy}>
                    <SelectTrigger className="w-[200px] h-9">
                      <SelectValue placeholder="How to label?" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="custom">Custom Class</SelectItem>
                      <SelectItem value="unlabeled">No Label (Unlabeled)</SelectItem>
                    </SelectContent>
                  </Select>
                  {uploadLabelStrategy === "custom" && (
                    <Input
                      placeholder="Enter class name..."
                      value={uploadCustomLabel}
                      onChange={(e) => setUploadCustomLabel(e.target.value)}
                      className="w-[200px] h-9"
                    />
                  )}
                </>
              )}
              
              {/* URL Label Strategy */}
              {activeTab === "url" && urlInput.trim() && (
                <>
                  <Select value={urlLabelStrategy} onValueChange={setUrlLabelStrategy}>
                    <SelectTrigger className="w-[200px] h-9">
                      <SelectValue placeholder="How to label?" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="custom">Custom Class</SelectItem>
                      <SelectItem value="unlabeled">No Label (Unlabeled)</SelectItem>
                    </SelectContent>
                  </Select>
                  {urlLabelStrategy === "custom" && (
                    <Input
                      placeholder="Enter class name..."
                      value={urlCustomLabel}
                      onChange={(e) => setUrlCustomLabel(e.target.value)}
                      className="w-[200px] h-9"
                    />
                  )}
                </>
              )}
              
              {totalSelected === 0 && (
                <span className="text-sm text-muted-foreground">Select items to configure labeling</span>
              )}
            </div>

            {/* Import Progress UI */}
            {importProgress.status !== "idle" && (
              <ImportProgressUI
                progress={importProgress}
                onCancel={abortImport}
                onRetry={handleRetryFailed}
                className="mb-3"
              />
            )}

            {/* Action Buttons */}
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">
                {totalSelected.toLocaleString()} item{totalSelected !== 1 ? "s" : ""} selected
              </span>
              <div className="flex gap-2">
                <Button variant="outline" onClick={handleClose} disabled={isImporting}>
                  Cancel
                </Button>
                <Button
                  onClick={handleChunkedImport}
                  disabled={totalSelected === 0 || isImporting || isSelectingAll}
                >
                  {isImporting ? (
                    <>
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      Importing...
                    </>
                  ) : (
                    <>
                      <CheckCircle className="h-4 w-4 mr-2" />
                      Import {totalSelected.toLocaleString()} Items
                    </>
                  )}
                </Button>
              </div>
            </div>
          </div>
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  );
}
