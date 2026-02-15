"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Check, Plus, Search, Tag, X } from "lucide-react";
import { useCanvasStore } from "@/hooks/store-maps/use-canvas-store";
import { cn } from "@/lib/utils";

// Common retail categories
const CATEGORY_TREE = [
  {
    name: "Gida",
    children: [
      "Sut Urunleri",
      "Et & Tavuk",
      "Meyve & Sebze",
      "Ekmek & Firincilik",
      "Konserve & Hazir Gida",
      "Baharat & Sos",
      "Kahvaltilik",
      "Atistirmalik",
      "Seker & Cikolata",
      "Dondurulmus Gida",
    ],
  },
  {
    name: "Icecek",
    children: [
      "Su",
      "Meyve Suyu",
      "Gazli Icecek",
      "Cay & Kahve",
      "Sut Icecekleri",
      "Enerji Icecegi",
    ],
  },
  {
    name: "Temizlik",
    children: [
      "Camasir Deterjan",
      "Bulasik Deterjan",
      "Yuzey Temizleyici",
      "Tuvalet Kagidi",
      "Kagit Havlu",
      "Cope Torbasi",
    ],
  },
  {
    name: "Kisisel Bakim",
    children: [
      "Sampuan & Sac Bakimi",
      "Sabun & Dus Jeli",
      "Dis Bakimi",
      "Deodorant",
      "Cilt Bakimi",
      "Bebek Bakimi",
    ],
  },
  {
    name: "Ev & Yasam",
    children: [
      "Mutfak Gerecleri",
      "Elektrikli Aletler",
      "Aydinlatma",
      "Dekorasyon",
    ],
  },
];

interface CategoryPickerProps {
  selectedCategories: string[];
  onCategoriesChange: (categories: string[]) => void;
}

export function CategoryPicker({
  selectedCategories,
  onCategoriesChange,
}: CategoryPickerProps) {
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState("");

  const allCategories = CATEGORY_TREE.flatMap((group) => [
    group.name,
    ...group.children,
  ]);

  const filteredCategories = search
    ? allCategories.filter((c) =>
        c.toLowerCase().includes(search.toLowerCase())
      )
    : [];

  const toggleCategory = (category: string) => {
    if (selectedCategories.includes(category)) {
      onCategoriesChange(
        selectedCategories.filter((c) => c !== category)
      );
    } else {
      onCategoriesChange([...selectedCategories, category]);
    }
  };

  const removeCategory = (category: string) => {
    onCategoriesChange(selectedCategories.filter((c) => c !== category));
  };

  return (
    <div className="space-y-2">
      {/* Selected Categories */}
      {selectedCategories.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {selectedCategories.map((cat) => (
            <Badge
              key={cat}
              variant="secondary"
              className="text-xs gap-1 pr-1"
            >
              {cat}
              <button
                className="ml-0.5 hover:text-red-500 transition-colors"
                onClick={() => removeCategory(cat)}
              >
                <X className="h-3 w-3" />
              </button>
            </Badge>
          ))}
        </div>
      )}

      {/* Add Category Popover */}
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <Button variant="outline" size="sm" className="w-full text-xs">
            <Tag className="h-3 w-3 mr-1" />
            Add Category
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-64 p-0" align="start">
          <div className="p-2 border-b">
            <div className="relative">
              <Search className="absolute left-2 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
              <Input
                placeholder="Search categories..."
                className="pl-7 h-7 text-xs"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                autoFocus
              />
            </div>
          </div>

          <ScrollArea className="max-h-60">
            <div className="p-1">
              {search ? (
                // Search results
                filteredCategories.length === 0 ? (
                  <p className="text-xs text-muted-foreground text-center py-4">
                    No categories found
                  </p>
                ) : (
                  filteredCategories.map((cat) => (
                    <button
                      key={cat}
                      className={cn(
                        "w-full flex items-center gap-2 px-2 py-1.5 rounded-md text-xs transition-colors text-left",
                        selectedCategories.includes(cat)
                          ? "bg-primary/10 text-primary"
                          : "hover:bg-accent"
                      )}
                      onClick={() => toggleCategory(cat)}
                    >
                      <Check
                        className={cn(
                          "h-3 w-3",
                          selectedCategories.includes(cat)
                            ? "opacity-100"
                            : "opacity-0"
                        )}
                      />
                      {cat}
                    </button>
                  ))
                )
              ) : (
                // Category tree
                CATEGORY_TREE.map((group) => (
                  <div key={group.name} className="mb-1">
                    <button
                      className={cn(
                        "w-full flex items-center gap-2 px-2 py-1.5 rounded-md text-xs font-medium transition-colors text-left",
                        selectedCategories.includes(group.name)
                          ? "bg-primary/10 text-primary"
                          : "hover:bg-accent"
                      )}
                      onClick={() => toggleCategory(group.name)}
                    >
                      <Check
                        className={cn(
                          "h-3 w-3",
                          selectedCategories.includes(group.name)
                            ? "opacity-100"
                            : "opacity-0"
                        )}
                      />
                      {group.name}
                    </button>
                    <div className="ml-4">
                      {group.children.map((child) => (
                        <button
                          key={child}
                          className={cn(
                            "w-full flex items-center gap-2 px-2 py-1 rounded-md text-xs transition-colors text-left",
                            selectedCategories.includes(child)
                              ? "bg-primary/10 text-primary"
                              : "hover:bg-accent text-muted-foreground"
                          )}
                          onClick={() => toggleCategory(child)}
                        >
                          <Check
                            className={cn(
                              "h-3 w-3",
                              selectedCategories.includes(child)
                                ? "opacity-100"
                                : "opacity-0"
                            )}
                          />
                          {child}
                        </button>
                      ))}
                    </div>
                  </div>
                ))
              )}
            </div>
          </ScrollArea>
        </PopoverContent>
      </Popover>
    </div>
  );
}
