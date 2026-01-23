"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { Checkbox } from "@/components/ui/checkbox";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import Image from "next/image";

export interface PickerImage {
  id: string;
  image_url: string;
  label?: string;        // Primary label (product name, etc)
  sublabel?: string;     // Secondary label (category, brand, etc)
  badge?: string;        // Badge text (image_type, status, etc)
  badgeVariant?: "default" | "secondary" | "outline" | "destructive";
  // Additional data for import
  product_id?: string;   // For product images - the actual product ID
}

interface ImagePickerGridProps {
  images: PickerImage[];
  selectedIds: Set<string>;
  onSelectionChange: (ids: Set<string>) => void;
  isLoading?: boolean;
  hasMore?: boolean;
  onLoadMore?: () => void;
  thumbnailSize?: "sm" | "md" | "lg";
  emptyMessage?: string;
  className?: string;
}

const SIZE_CONFIG = {
  sm: { grid: "grid-cols-6 sm:grid-cols-8 lg:grid-cols-10", size: 80 },
  md: { grid: "grid-cols-4 sm:grid-cols-6 lg:grid-cols-8", size: 120 },
  lg: { grid: "grid-cols-3 sm:grid-cols-4 lg:grid-cols-6", size: 160 },
};

export function ImagePickerGrid({
  images,
  selectedIds,
  onSelectionChange,
  isLoading = false,
  hasMore = false,
  onLoadMore,
  thumbnailSize = "md",
  emptyMessage = "No images found",
  className,
}: ImagePickerGridProps) {
  const [lastSelectedIndex, setLastSelectedIndex] = useState<number | null>(null);
  const observerRef = useRef<IntersectionObserver | null>(null);
  const loadMoreRef = useRef<HTMLDivElement>(null);

  const config = SIZE_CONFIG[thumbnailSize];

  // Intersection Observer for infinite scroll
  useEffect(() => {
    if (!hasMore || !onLoadMore) return;

    observerRef.current = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting && hasMore && !isLoading) {
          onLoadMore();
        }
      },
      { threshold: 0.1 }
    );

    if (loadMoreRef.current) {
      observerRef.current.observe(loadMoreRef.current);
    }

    return () => observerRef.current?.disconnect();
  }, [hasMore, isLoading, onLoadMore]);

  const handleSelect = useCallback(
    (id: string, index: number, event: React.MouseEvent) => {
      const newSelected = new Set(selectedIds);

      if (event.shiftKey && lastSelectedIndex !== null) {
        // Range selection
        const start = Math.min(lastSelectedIndex, index);
        const end = Math.max(lastSelectedIndex, index);
        for (let i = start; i <= end; i++) {
          newSelected.add(images[i].id);
        }
      } else if (event.ctrlKey || event.metaKey) {
        // Toggle single
        if (newSelected.has(id)) {
          newSelected.delete(id);
        } else {
          newSelected.add(id);
        }
      } else {
        // Normal click - toggle
        if (newSelected.has(id)) {
          newSelected.delete(id);
        } else {
          newSelected.add(id);
        }
      }

      setLastSelectedIndex(index);
      onSelectionChange(newSelected);
    },
    [selectedIds, lastSelectedIndex, images, onSelectionChange]
  );

  const handleCheckboxChange = useCallback(
    (id: string, checked: boolean) => {
      const newSelected = new Set(selectedIds);
      if (checked) {
        newSelected.add(id);
      } else {
        newSelected.delete(id);
      }
      onSelectionChange(newSelected);
    },
    [selectedIds, onSelectionChange]
  );

  // Loading skeleton
  if (isLoading && images.length === 0) {
    return (
      <div className={cn("grid gap-2", config.grid, className)}>
        {Array.from({ length: 12 }).map((_, i) => (
          <Skeleton
            key={i}
            className="aspect-square rounded-lg"
            style={{ width: config.size, height: config.size }}
          />
        ))}
      </div>
    );
  }

  // Empty state
  if (!isLoading && images.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
        <p>{emptyMessage}</p>
      </div>
    );
  }

  return (
    <TooltipProvider delayDuration={300}>
      <div className={cn("grid gap-2", config.grid, className)}>
        {images.map((image, index) => {
          const isSelected = selectedIds.has(image.id);

          return (
            <Tooltip key={image.id}>
              <TooltipTrigger asChild>
                <div
                  className={cn(
                    "relative aspect-square rounded-lg overflow-hidden cursor-pointer group",
                    "border-2 transition-all duration-150",
                    isSelected
                      ? "border-primary ring-2 ring-primary/30"
                      : "border-transparent hover:border-muted-foreground/30"
                  )}
                  onClick={(e) => handleSelect(image.id, index, e)}
                >
                  {/* Image */}
                  <Image
                    src={image.image_url}
                    alt={image.label || "Image"}
                    fill
                    className="object-cover"
                    sizes={`${config.size}px`}
                    loading="lazy"
                  />

                  {/* Selection overlay */}
                  <div
                    className={cn(
                      "absolute inset-0 transition-opacity",
                      isSelected ? "bg-primary/20" : "bg-transparent"
                    )}
                  />

                  {/* Checkbox */}
                  <div
                    className={cn(
                      "absolute top-1 left-1 transition-opacity",
                      isSelected || "opacity-0 group-hover:opacity-100"
                    )}
                    onClick={(e) => e.stopPropagation()}
                  >
                    <Checkbox
                      checked={isSelected}
                      onCheckedChange={(checked) =>
                        handleCheckboxChange(image.id, !!checked)
                      }
                      className="bg-white/90 border-gray-400"
                    />
                  </div>

                  {/* Badge */}
                  {image.badge && (
                    <div className="absolute top-1 right-1">
                      <Badge
                        variant={image.badgeVariant || "secondary"}
                        className="text-[10px] px-1 py-0"
                      >
                        {image.badge}
                      </Badge>
                    </div>
                  )}

                  {/* Label on hover */}
                  {image.label && (
                    <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/70 to-transparent p-1.5 opacity-0 group-hover:opacity-100 transition-opacity">
                      <p className="text-white text-[10px] truncate font-medium">
                        {image.label}
                      </p>
                      {image.sublabel && (
                        <p className="text-white/70 text-[9px] truncate">
                          {image.sublabel}
                        </p>
                      )}
                    </div>
                  )}
                </div>
              </TooltipTrigger>
              <TooltipContent side="top" className="max-w-xs">
                <div className="space-y-1">
                  {image.label && <p className="font-medium">{image.label}</p>}
                  {image.sublabel && (
                    <p className="text-muted-foreground text-sm">{image.sublabel}</p>
                  )}
                  {image.badge && (
                    <Badge variant={image.badgeVariant || "secondary"} className="text-xs">
                      {image.badge}
                    </Badge>
                  )}
                </div>
              </TooltipContent>
            </Tooltip>
          );
        })}

        {/* Load more trigger */}
        {hasMore && (
          <div ref={loadMoreRef} className="col-span-full flex justify-center py-4">
            {isLoading ? (
              <div className="flex items-center gap-2 text-muted-foreground">
                <div className="h-4 w-4 animate-spin rounded-full border-2 border-primary border-t-transparent" />
                Loading more...
              </div>
            ) : (
              <button
                onClick={onLoadMore}
                className="text-sm text-muted-foreground hover:text-foreground"
              >
                Load more
              </button>
            )}
          </div>
        )}
      </div>
    </TooltipProvider>
  );
}
