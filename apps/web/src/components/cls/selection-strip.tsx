"use client";

import { useRef, useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea, ScrollBar } from "@/components/ui/scroll-area";
import { X, Trash2, ChevronLeft, ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";
import Image from "next/image";

export interface SelectedImage {
  id: string;
  image_url: string;
  label?: string;
}

interface SelectionStripProps {
  selectedImages: SelectedImage[];
  onRemove: (id: string) => void;
  onClearAll: () => void;
  maxVisible?: number;
  className?: string;
}

export function SelectionStrip({
  selectedImages,
  onRemove,
  onClearAll,
  maxVisible = 20,
  className,
}: SelectionStripProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [canScrollLeft, setCanScrollLeft] = useState(false);
  const [canScrollRight, setCanScrollRight] = useState(false);

  // Check scroll state
  useEffect(() => {
    const checkScroll = () => {
      if (!scrollRef.current) return;
      const { scrollLeft, scrollWidth, clientWidth } = scrollRef.current;
      setCanScrollLeft(scrollLeft > 0);
      setCanScrollRight(scrollLeft + clientWidth < scrollWidth - 10);
    };

    checkScroll();
    const el = scrollRef.current;
    el?.addEventListener("scroll", checkScroll);
    window.addEventListener("resize", checkScroll);

    return () => {
      el?.removeEventListener("scroll", checkScroll);
      window.removeEventListener("resize", checkScroll);
    };
  }, [selectedImages.length]);

  const scroll = (direction: "left" | "right") => {
    if (!scrollRef.current) return;
    const scrollAmount = 200;
    scrollRef.current.scrollBy({
      left: direction === "left" ? -scrollAmount : scrollAmount,
      behavior: "smooth",
    });
  };

  if (selectedImages.length === 0) {
    return null;
  }

  const visibleImages = selectedImages.slice(0, maxVisible);
  const hiddenCount = selectedImages.length - maxVisible;

  return (
    <div className={cn("border-t bg-muted/30 p-3", className)}>
      <div className="flex items-center gap-3">
        {/* Count and clear */}
        <div className="flex items-center gap-2 flex-shrink-0">
          <Badge variant="secondary" className="text-sm font-medium">
            {selectedImages.length} selected
          </Badge>
          <Button
            variant="ghost"
            size="sm"
            onClick={onClearAll}
            className="h-7 px-2 text-muted-foreground hover:text-destructive"
          >
            <Trash2 className="h-3.5 w-3.5 mr-1" />
            Clear
          </Button>
        </div>

        {/* Scroll left button */}
        {canScrollLeft && (
          <Button
            variant="ghost"
            size="sm"
            onClick={() => scroll("left")}
            className="h-8 w-8 p-0 flex-shrink-0"
          >
            <ChevronLeft className="h-4 w-4" />
          </Button>
        )}

        {/* Selected images strip */}
        <div
          ref={scrollRef}
          className="flex-1 overflow-x-auto scrollbar-hide"
          style={{ scrollbarWidth: "none", msOverflowStyle: "none" }}
        >
          <div className="flex gap-2">
            {visibleImages.map((image) => (
              <div
                key={image.id}
                className="relative flex-shrink-0 group"
                style={{ width: 56, height: 56 }}
              >
                <div className="w-full h-full rounded-md overflow-hidden border bg-muted">
                  <Image
                    src={image.image_url}
                    alt={image.label || "Selected"}
                    width={56}
                    height={56}
                    className="object-cover w-full h-full"
                  />
                </div>
                {/* Remove button */}
                <button
                  onClick={() => onRemove(image.id)}
                  className="absolute -top-1.5 -right-1.5 h-5 w-5 rounded-full bg-destructive text-white flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity hover:bg-destructive/90"
                >
                  <X className="h-3 w-3" />
                </button>
              </div>
            ))}

            {/* Hidden count indicator */}
            {hiddenCount > 0 && (
              <div
                className="flex-shrink-0 w-14 h-14 rounded-md border bg-muted flex items-center justify-center"
              >
                <span className="text-sm text-muted-foreground font-medium">
                  +{hiddenCount}
                </span>
              </div>
            )}
          </div>
        </div>

        {/* Scroll right button */}
        {canScrollRight && (
          <Button
            variant="ghost"
            size="sm"
            onClick={() => scroll("right")}
            className="h-8 w-8 p-0 flex-shrink-0"
          >
            <ChevronRight className="h-4 w-4" />
          </Button>
        )}
      </div>
    </div>
  );
}
