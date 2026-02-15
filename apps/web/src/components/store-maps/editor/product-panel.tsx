"use client";

import { useState, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import {
  Search,
  Package,
  GripVertical,
  Plus,
  X,
  Loader2,
  ChevronRight,
} from "lucide-react";
import { apiClient } from "@/lib/api-client";
import { useCanvasStore } from "@/hooks/store-maps/use-canvas-store";
import { cn } from "@/lib/utils";
import type { ProductRef } from "@/types/store-maps";

export function ProductPanel() {
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(1);
  const {
    selectedAreaId,
    areas,
    updateArea,
    showProductPanel,
    setShowProductPanel,
  } = useCanvasStore();

  const selectedArea = areas.find((a) => a.fabricId === selectedAreaId);

  // Fetch products for search
  const { data: productsData, isLoading } = useQuery({
    queryKey: ["products-search", search, page],
    queryFn: () =>
      apiClient.getProducts({
        page,
        page_size: 30,
        search: search || undefined,
      }),
    enabled: showProductPanel,
  });

  const products = productsData?.items ?? productsData ?? [];

  const isProductAssigned = useCallback(
    (productId: string): boolean => {
      if (!selectedArea?.products) return false;
      return selectedArea.products.some(
        (p) => p.products.some((pr) => pr.product_id === productId)
      );
    },
    [selectedArea]
  );

  const handleAssignProduct = useCallback(
    (product: { id: string; barcode?: string; product_name?: string; image_url?: string }) => {
      if (!selectedAreaId || !selectedArea) return;

      const productRef: ProductRef = {
        product_id: product.id,
        name: product.product_name ?? product.barcode ?? product.id,
        barcode: product.barcode,
        image_url: product.image_url,
      };

      const existingMapping = selectedArea.products ?? [];
      const areaMapping = existingMapping.find(
        (m) => m.area_id === (selectedArea.areaId ?? 0)
      );

      if (areaMapping) {
        // Check if already assigned
        if (areaMapping.products.some((p) => p.product_id === product.id))
          return;
        const updatedProducts = [
          ...existingMapping.filter(
            (m) => m.area_id !== (selectedArea.areaId ?? 0)
          ),
          {
            ...areaMapping,
            products: [...areaMapping.products, productRef],
          },
        ];
        updateArea(selectedAreaId, { products: updatedProducts });
      } else {
        updateArea(selectedAreaId, {
          products: [
            ...existingMapping,
            {
              area_id: selectedArea.areaId ?? 0,
              products: [productRef],
            },
          ],
        });
      }
    },
    [selectedAreaId, selectedArea, updateArea]
  );

  const handleRemoveProduct = useCallback(
    (productId: string) => {
      if (!selectedAreaId || !selectedArea?.products) return;
      const updated = selectedArea.products.map((m) => ({
        ...m,
        products: m.products.filter((p) => p.product_id !== productId),
      }));
      updateArea(selectedAreaId, { products: updated });
    },
    [selectedAreaId, selectedArea, updateArea]
  );

  const assignedProducts =
    selectedArea?.products?.flatMap((m) => m.products) ?? [];

  if (!showProductPanel) return null;

  return (
    <div className="w-72 border-l bg-card flex flex-col overflow-hidden min-h-0">
      {/* Header */}
      <div className="p-3 border-b flex items-center justify-between shrink-0">
        <div className="flex items-center gap-1.5">
          <Package className="h-4 w-4 text-muted-foreground" />
          <h3 className="font-medium text-sm">Products</h3>
        </div>
        <Button
          variant="ghost"
          size="icon"
          className="h-7 w-7"
          onClick={() => setShowProductPanel(false)}
        >
          <X className="h-4 w-4" />
        </Button>
      </div>

      {/* Assigned Products */}
      {selectedArea && assignedProducts.length > 0 && (
        <div className="p-3 border-b shrink-0">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
              Assigned to {selectedArea.name}
            </span>
            <Badge variant="secondary" className="text-xs">
              {assignedProducts.length}
            </Badge>
          </div>
          <div className="space-y-1 max-h-40 overflow-y-auto">
            {assignedProducts.map((product) => (
              <div
                key={product.product_id}
                className="flex items-center gap-2 p-1.5 rounded-md bg-accent/50 text-sm"
              >
                {product.image_url ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={product.image_url}
                    alt=""
                    className="w-6 h-6 rounded object-cover"
                  />
                ) : (
                  <div className="w-6 h-6 rounded bg-muted flex items-center justify-center">
                    <Package className="h-3 w-3 text-muted-foreground" />
                  </div>
                )}
                <span className="text-xs flex-1 truncate">{product.name}</span>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-5 w-5 shrink-0"
                  onClick={() => handleRemoveProduct(product.product_id)}
                >
                  <X className="h-3 w-3" />
                </Button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Search */}
      <div className="p-3 border-b shrink-0">
        <div className="relative">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
          <Input
            placeholder="Search products..."
            className="pl-8 h-8 text-sm"
            value={search}
            onChange={(e) => {
              setSearch(e.target.value);
              setPage(1);
            }}
          />
        </div>
      </div>

      {/* No area selected warning */}
      {!selectedArea && (
        <div className="p-4 text-center">
          <p className="text-xs text-muted-foreground">
            Select an area on the map first, then assign products to it.
          </p>
        </div>
      )}

      {/* Product List */}
      <ScrollArea className="flex-1 min-h-0">
        <div className="p-2 space-y-0.5">
          {isLoading ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
            </div>
          ) : !Array.isArray(products) || products.length === 0 ? (
            <div className="text-center py-8">
              <p className="text-xs text-muted-foreground">
                {search ? "No products found" : "No products available"}
              </p>
            </div>
          ) : (
            products.map((product: {
              id: string;
              barcode?: string;
              product_name?: string;
              image_url?: string;
            }) => {
              const assigned = isProductAssigned(product.id);
              return (
                <div
                  key={product.id}
                  className={cn(
                    "flex items-center gap-2 p-2 rounded-md text-sm cursor-pointer transition-colors",
                    assigned
                      ? "bg-primary/10 border border-primary/20"
                      : "hover:bg-accent"
                  )}
                  onClick={() => {
                    if (selectedArea && !assigned) {
                      handleAssignProduct(product);
                    }
                  }}
                >
                  <GripVertical className="h-3 w-3 text-muted-foreground shrink-0" />
                  {product.image_url ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img
                      src={product.image_url}
                      alt=""
                      className="w-8 h-8 rounded object-cover shrink-0"
                    />
                  ) : (
                    <div className="w-8 h-8 rounded bg-muted flex items-center justify-center shrink-0">
                      <Package className="h-4 w-4 text-muted-foreground" />
                    </div>
                  )}
                  <div className="flex-1 min-w-0">
                    <p className="text-xs font-medium truncate">
                      {product.product_name ?? product.barcode ?? product.id}
                    </p>
                    {product.barcode && (
                      <p className="text-[10px] text-muted-foreground font-mono truncate">
                        {product.barcode}
                      </p>
                    )}
                  </div>
                  {assigned ? (
                    <Badge className="text-[10px] h-5 shrink-0">
                      Assigned
                    </Badge>
                  ) : selectedArea ? (
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-6 w-6 shrink-0"
                    >
                      <Plus className="h-3 w-3" />
                    </Button>
                  ) : null}
                </div>
              );
            })
          )}
        </div>
      </ScrollArea>
    </div>
  );
}
