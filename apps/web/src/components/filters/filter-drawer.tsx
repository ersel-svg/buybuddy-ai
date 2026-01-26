"use client";

import { useState, useMemo, useCallback, useEffect, memo, useRef } from "react";
import { useSearchParams, useRouter, usePathname } from "next/navigation";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Checkbox } from "@/components/ui/checkbox";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  Filter,
  ChevronDown,
  ChevronRight,
  Search,
  RotateCcw,
} from "lucide-react";
import { cn } from "@/lib/utils";

// ===========================================
// Types
// ===========================================

export interface FilterOption {
  value: string;
  label: string;
  count?: number;
}

export interface CheckboxFilterSection {
  id: string;
  label: string;
  type: "checkbox";
  options: FilterOption[];
  searchable?: boolean;
  defaultExpanded?: boolean;
}

export interface BooleanFilterSection {
  id: string;
  label: string;
  type: "boolean";
  trueLabel: string;
  falseLabel: string;
  trueCount?: number;
  falseCount?: number;
  defaultExpanded?: boolean;
}

export interface RangeFilterSection {
  id: string;
  label: string;
  type: "range";
  min: number;
  max: number;
  step?: number;
  unit?: string;
  defaultExpanded?: boolean;
}

export type FilterSection = CheckboxFilterSection | BooleanFilterSection | RangeFilterSection;

export interface FilterState {
  [sectionId: string]: Set<string> | { min: number; max: number } | boolean | undefined;
}

export interface FilterDrawerProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  sections: FilterSection[];
  filterState: FilterState;
  onFilterChange: (sectionId: string, value: Set<string> | { min: number; max: number } | boolean | undefined) => void;
  onClearAll: () => void;
  onClearSection: (sectionId: string) => void;
  title?: string;
  description?: string;
}

// ===========================================
// Checkbox Filter Section Component
// ===========================================

interface CheckboxSectionProps {
  section: CheckboxFilterSection;
  selectedValues: Set<string>;
  onToggle: (value: string) => void;
  onClear: () => void;
  onSelectAll: (values: string[]) => void;
}

const CheckboxSection = memo(function CheckboxSection({
  section,
  selectedValues,
  onToggle,
  onClear,
  onSelectAll,
}: CheckboxSectionProps) {
  const [isExpanded, setIsExpanded] = useState(section.defaultExpanded ?? true);
  const [searchQuery, setSearchQuery] = useState("");

  const filteredOptions = useMemo(() => {
    if (!searchQuery.trim()) return section.options;
    const query = searchQuery.toLowerCase();
    return section.options.filter(
      (opt) =>
        opt.label.toLowerCase().includes(query) ||
        opt.value.toLowerCase().includes(query)
    );
  }, [section.options, searchQuery]);

  const selectedCount = selectedValues.size;
  const showSearch = section.searchable !== false && section.options.length > 5;
  const isEmpty = section.options.length === 0;

  // Check if all filtered options are selected
  const allFilteredSelected = filteredOptions.length > 0 &&
    filteredOptions.every(opt => selectedValues.has(opt.value));

  // Check if we're showing filtered results (search is active)
  const isFiltered = searchQuery.trim().length > 0;

  return (
    <Collapsible open={isExpanded} onOpenChange={setIsExpanded}>
      <div className={cn(
        "border rounded-lg overflow-hidden",
        isEmpty
          ? "border-gray-50 bg-gray-50/50 dark:border-gray-800 dark:bg-gray-800/50"
          : "border-gray-100 bg-white dark:border-gray-700 dark:bg-gray-800"
      )}>
        <CollapsibleTrigger asChild>
          <button className="w-full flex items-center justify-between p-3 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors">
            <div className="flex items-center gap-2">
              {isExpanded ? (
                <ChevronDown className="h-4 w-4 text-gray-500 dark:text-gray-400" />
              ) : (
                <ChevronRight className="h-4 w-4 text-gray-500 dark:text-gray-400" />
              )}
              <span className="font-medium text-sm text-gray-900 dark:text-gray-100">
                {section.label}
              </span>
              <Badge variant="secondary" className="text-xs px-1.5 py-0 h-5">
                {section.options.length}
              </Badge>
            </div>
            {selectedCount > 0 && (
              <Badge className="bg-blue-600 hover:bg-blue-600 text-white text-xs px-1.5 py-0 h-5">
                {selectedCount}
              </Badge>
            )}
          </button>
        </CollapsibleTrigger>

        <CollapsibleContent>
          <div className="px-3 pb-3 space-y-2">
            {isEmpty ? (
              <p className="text-xs text-gray-400 dark:text-gray-500 py-2 text-center italic">
                No data in current results
              </p>
            ) : (
              <>
                {showSearch && (
                  <div className="relative">
                    <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-gray-400 dark:text-gray-500" />
                    <Input
                      placeholder={`Search...`}
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className="h-8 pl-8 text-sm bg-gray-50 border-gray-200 dark:bg-gray-700 dark:border-gray-600 dark:text-gray-100 dark:placeholder:text-gray-400"
                    />
                  </div>
                )}

                {/* Action buttons: Select All / Clear */}
                <div className="flex items-center gap-3">
                  {!allFilteredSelected && filteredOptions.length > 0 && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onSelectAll(filteredOptions.map(opt => opt.value));
                      }}
                      className="text-xs text-blue-600 hover:text-blue-700 hover:underline dark:text-blue-400 dark:hover:text-blue-300"
                    >
                      {isFiltered ? `Select filtered (${filteredOptions.length})` : "Select all"}
                    </button>
                  )}
                  {selectedCount > 0 && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onClear();
                      }}
                      className="text-xs text-gray-500 hover:text-gray-700 hover:underline dark:text-gray-400 dark:hover:text-gray-300"
                    >
                      Clear selection
                    </button>
                  )}
                </div>

                <div
                  className={cn(
                    "space-y-0.5",
                    section.options.length > 8 && "max-h-[200px] overflow-y-auto pr-1"
                  )}
                >
                  {filteredOptions.length === 0 ? (
                    <p className="text-xs text-gray-500 dark:text-gray-400 py-2 text-center">
                      No options found
                    </p>
                  ) : (
                    filteredOptions.map((option) => (
                      <label
                        key={option.value}
                        className={cn(
                          "flex items-center gap-2.5 py-1.5 px-2 rounded-md cursor-pointer transition-colors",
                          selectedValues.has(option.value)
                            ? "bg-blue-50 dark:bg-blue-900/30"
                            : "hover:bg-gray-50 dark:hover:bg-gray-700"
                        )}
                      >
                        <Checkbox
                          checked={selectedValues.has(option.value)}
                          onCheckedChange={() => onToggle(option.value)}
                          className="h-4 w-4"
                        />
                        <span
                          className={cn(
                            "text-sm flex-1",
                            selectedValues.has(option.value)
                              ? "text-blue-900 font-medium dark:text-blue-300"
                              : "text-gray-700 dark:text-gray-300"
                          )}
                        >
                          {option.label}
                        </span>
                        {option.count !== undefined && (
                          <span className="text-xs text-gray-400 dark:text-gray-500">
                            ({option.count})
                          </span>
                        )}
                      </label>
                    ))
                  )}
                </div>
              </>
            )}
          </div>
        </CollapsibleContent>
      </div>
    </Collapsible>
  );
});

// ===========================================
// Boolean Filter Section Component
// ===========================================

interface BooleanSectionProps {
  section: BooleanFilterSection;
  value: boolean | undefined;
  onChange: (value: boolean | undefined) => void;
}

const BooleanSection = memo(function BooleanSection({ section, value, onChange }: BooleanSectionProps) {
  const [isExpanded, setIsExpanded] = useState(section.defaultExpanded ?? true);

  return (
    <Collapsible open={isExpanded} onOpenChange={setIsExpanded}>
      <div className="border border-gray-100 rounded-lg bg-white dark:border-gray-700 dark:bg-gray-800 overflow-hidden">
        <CollapsibleTrigger asChild>
          <button className="w-full flex items-center justify-between p-3 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors">
            <div className="flex items-center gap-2">
              {isExpanded ? (
                <ChevronDown className="h-4 w-4 text-gray-500 dark:text-gray-400" />
              ) : (
                <ChevronRight className="h-4 w-4 text-gray-500 dark:text-gray-400" />
              )}
              <span className="font-medium text-sm text-gray-900 dark:text-gray-100">
                {section.label}
              </span>
            </div>
            {value !== undefined && (
              <Badge className="bg-blue-600 hover:bg-blue-600 text-white text-xs px-1.5 py-0 h-5">
                1
              </Badge>
            )}
          </button>
        </CollapsibleTrigger>

        <CollapsibleContent>
          <div className="px-3 pb-3 space-y-1">
            <label
              className={cn(
                "flex items-center gap-2.5 py-1.5 px-2 rounded-md cursor-pointer transition-colors",
                value === true
                  ? "bg-blue-50 dark:bg-blue-900/30"
                  : "hover:bg-gray-50 dark:hover:bg-gray-700"
              )}
            >
              <Checkbox
                checked={value === true}
                onCheckedChange={(checked) => onChange(checked ? true : undefined)}
                className="h-4 w-4"
              />
              <span className={cn(
                "text-sm flex-1",
                value === true
                  ? "text-blue-900 font-medium dark:text-blue-300"
                  : "text-gray-700 dark:text-gray-300"
              )}>
                {section.trueLabel}
              </span>
              {section.trueCount !== undefined && (
                <span className="text-xs text-gray-400 dark:text-gray-500">({section.trueCount})</span>
              )}
            </label>
            <label
              className={cn(
                "flex items-center gap-2.5 py-1.5 px-2 rounded-md cursor-pointer transition-colors",
                value === false
                  ? "bg-blue-50 dark:bg-blue-900/30"
                  : "hover:bg-gray-50 dark:hover:bg-gray-700"
              )}
            >
              <Checkbox
                checked={value === false}
                onCheckedChange={(checked) => onChange(checked ? false : undefined)}
                className="h-4 w-4"
              />
              <span className={cn(
                "text-sm flex-1",
                value === false
                  ? "text-blue-900 font-medium dark:text-blue-300"
                  : "text-gray-700 dark:text-gray-300"
              )}>
                {section.falseLabel}
              </span>
              {section.falseCount !== undefined && (
                <span className="text-xs text-gray-400 dark:text-gray-500">({section.falseCount})</span>
              )}
            </label>
          </div>
        </CollapsibleContent>
      </div>
    </Collapsible>
  );
});

// ===========================================
// Range Filter Section Component
// ===========================================

interface RangeSectionProps {
  section: RangeFilterSection;
  value: { min: number; max: number } | undefined;
  onChange: (value: { min: number; max: number } | undefined) => void;
  onClear: () => void;
}

const RangeSection = memo(function RangeSection({ section, value, onChange, onClear }: RangeSectionProps) {
  const [isExpanded, setIsExpanded] = useState(section.defaultExpanded ?? true);
  const currentMin = value?.min ?? section.min;
  const currentMax = value?.max ?? section.max;
  const isActive = value !== undefined;

  return (
    <Collapsible open={isExpanded} onOpenChange={setIsExpanded}>
      <div className="border border-gray-100 rounded-lg bg-white dark:border-gray-700 dark:bg-gray-800 overflow-hidden">
        <CollapsibleTrigger asChild>
          <button className="w-full flex items-center justify-between p-3 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors">
            <div className="flex items-center gap-2">
              {isExpanded ? (
                <ChevronDown className="h-4 w-4 text-gray-500 dark:text-gray-400" />
              ) : (
                <ChevronRight className="h-4 w-4 text-gray-500 dark:text-gray-400" />
              )}
              <span className="font-medium text-sm text-gray-900 dark:text-gray-100">
                {section.label}
              </span>
            </div>
            {isActive && (
              <Badge className="bg-blue-600 hover:bg-blue-600 text-white text-xs px-1.5 py-0 h-5">
                {currentMin}-{currentMax}{section.unit || ""}
              </Badge>
            )}
          </button>
        </CollapsibleTrigger>

        <CollapsibleContent>
          <div className="px-3 pb-3 space-y-3">
            {isActive && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onClear();
                }}
                className="text-xs text-blue-600 hover:text-blue-700 hover:underline dark:text-blue-400 dark:hover:text-blue-300"
              >
                Clear range
              </button>
            )}

            <div className="px-2">
              <Slider
                value={[currentMin, currentMax]}
                min={section.min}
                max={section.max}
                step={section.step || 1}
                onValueChange={([min, max]) => onChange({ min, max })}
                className="w-full"
              />
            </div>

            <div className="flex items-center justify-between text-xs text-gray-500 dark:text-gray-400">
              <span>{currentMin}{section.unit || ""}</span>
              <span>{currentMax}{section.unit || ""}</span>
            </div>
          </div>
        </CollapsibleContent>
      </div>
    </Collapsible>
  );
});

// ===========================================
// Main Filter Drawer Component
// ===========================================

export function FilterDrawer({
  open,
  onOpenChange,
  sections,
  filterState,
  onFilterChange,
  onClearAll,
  onClearSection,
  title = "Filters",
  description,
}: FilterDrawerProps) {
  // Use ref to access filterState in callbacks without re-creating them
  const filterStateRef = useRef(filterState);
  useEffect(() => {
    filterStateRef.current = filterState;
  }, [filterState]);

  const totalActiveFilters = useMemo(() => {
    let count = 0;
    Object.entries(filterState).forEach(([, value]) => {
      if (value instanceof Set) {
        count += value.size;
      } else if (typeof value === "boolean") {
        count += 1;
      } else if (value && typeof value === "object" && "min" in value) {
        count += 1;
      }
    });
    return count;
  }, [filterState]);

  const handleCheckboxToggle = useCallback(
    (sectionId: string, value: string) => {
      const currentSet = (filterStateRef.current[sectionId] as Set<string>) || new Set();
      const newSet = new Set(currentSet);
      if (newSet.has(value)) {
        newSet.delete(value);
      } else {
        newSet.add(value);
      }
      onFilterChange(sectionId, newSet);
    },
    [onFilterChange]
  );

  const handleSelectAll = useCallback(
    (sectionId: string, values: string[]) => {
      const currentSet = (filterStateRef.current[sectionId] as Set<string>) || new Set();
      const newSet = new Set(currentSet);
      values.forEach(value => newSet.add(value));
      onFilterChange(sectionId, newSet);
    },
    [onFilterChange]
  );

  // Show all sections, even empty ones (with "No options" message)
  const visibleSections = sections;

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent className="w-[400px] sm:w-[480px] p-0 h-full flex flex-col overflow-hidden bg-white dark:bg-gray-900">
        {/* Header - Fixed */}
        <div className="flex-shrink-0">
          <SheetHeader className="px-5 py-4 border-b bg-gray-50/80 dark:bg-gray-800/80 dark:border-gray-700">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2.5">
                <div className="p-1.5 bg-blue-100 dark:bg-blue-900/50 rounded-lg">
                  <Filter className="h-4 w-4 text-blue-600 dark:text-blue-400" />
                </div>
                <div>
                  <SheetTitle className="text-base font-semibold dark:text-gray-100">
                    {title}
                  </SheetTitle>
                  {description && (
                    <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">{description}</p>
                  )}
                </div>
              </div>
              {totalActiveFilters > 0 && (
                <Badge className="bg-blue-600 text-white px-2 py-0.5">
                  {totalActiveFilters} active
                </Badge>
              )}
            </div>
          </SheetHeader>

          {/* Clear All Bar - Fixed */}
          {totalActiveFilters > 0 && (
            <div className="px-5 py-2.5 bg-blue-50 border-b border-blue-100 dark:bg-blue-900/20 dark:border-blue-800">
              <Button
                variant="ghost"
                size="sm"
                onClick={onClearAll}
                className="h-7 text-blue-700 hover:text-blue-800 hover:bg-blue-100 dark:text-blue-400 dark:hover:text-blue-300 dark:hover:bg-blue-900/30 px-2 -ml-2"
              >
                <RotateCcw className="h-3.5 w-3.5 mr-1.5" />
                Clear all filters ({totalActiveFilters})
              </Button>
            </div>
          )}
        </div>

        {/* Filter Sections - Scrollable */}
        <div className="flex-1 overflow-y-auto min-h-0">
          <div className="p-4 space-y-3">
            {visibleSections.length === 0 ? (
              <div className="text-center py-8">
                <Filter className="h-10 w-10 text-gray-300 dark:text-gray-600 mx-auto mb-2" />
                <p className="text-sm text-gray-500 dark:text-gray-400">No filters available</p>
              </div>
            ) : (
              visibleSections.map((section) => {
                if (section.type === "checkbox") {
                  return (
                    <CheckboxSection
                      key={section.id}
                      section={section}
                      selectedValues={(filterState[section.id] as Set<string>) || new Set()}
                      onToggle={(value) => handleCheckboxToggle(section.id, value)}
                      onClear={() => onClearSection(section.id)}
                      onSelectAll={(values) => handleSelectAll(section.id, values)}
                    />
                  );
                }
                if (section.type === "boolean") {
                  return (
                    <BooleanSection
                      key={section.id}
                      section={section}
                      value={filterState[section.id] as boolean | undefined}
                      onChange={(value) => onFilterChange(section.id, value)}
                    />
                  );
                }
                if (section.type === "range") {
                  return (
                    <RangeSection
                      key={section.id}
                      section={section}
                      value={filterState[section.id] as { min: number; max: number } | undefined}
                      onChange={(value) => onFilterChange(section.id, value)}
                      onClear={() => onClearSection(section.id)}
                    />
                  );
                }
                return null;
              })
            )}
          </div>
        </div>

        {/* Footer - Fixed */}
        <div className="flex-shrink-0 px-5 py-3 border-t bg-gray-50/80 dark:bg-gray-800/80 dark:border-gray-700">
          <div className="flex items-center justify-between">
            <p className="text-xs text-gray-500 dark:text-gray-400">
              {totalActiveFilters === 0
                ? "No filters applied"
                : `${totalActiveFilters} filter${totalActiveFilters > 1 ? "s" : ""} applied`}
            </p>
            <Button
              onClick={() => onOpenChange(false)}
              size="sm"
              className="bg-blue-600 hover:bg-blue-700"
            >
              Apply Filters
            </Button>
          </div>
        </div>
      </SheetContent>
    </Sheet>
  );
}

// ===========================================
// Filter Trigger Button Component
// ===========================================

export interface FilterTriggerProps {
  onClick: () => void;
  activeCount: number;
  className?: string;
}

export function FilterTrigger({
  onClick,
  activeCount,
  className,
}: FilterTriggerProps) {
  return (
    <Button
      variant="outline"
      onClick={onClick}
      className={cn(
        "relative",
        activeCount > 0 && "border-blue-300 bg-blue-50 hover:bg-blue-100 dark:border-blue-700 dark:bg-blue-900/30 dark:hover:bg-blue-900/50",
        className
      )}
    >
      <Filter
        className={cn("h-4 w-4 mr-2", activeCount > 0 && "text-blue-600 dark:text-blue-400")}
      />
      <span className={cn(activeCount > 0 && "text-blue-700 dark:text-blue-300")}>Filters</span>
      {activeCount > 0 && (
        <Badge className="ml-2 bg-blue-600 text-white hover:bg-blue-600 px-1.5 py-0 h-5 min-w-[20px] flex items-center justify-center">
          {activeCount}
        </Badge>
      )}
    </Button>
  );
}

// ===========================================
// Active Filter Chips Component
// ===========================================

export interface ActiveFilterChip {
  id: string;
  sectionId: string;
  label: string;
  value: string;
}

export interface ActiveFilterChipsProps {
  filterState: FilterState;
  sections: FilterSection[];
  onRemoveFilter: (sectionId: string, value: string) => void;
  onClearAll: () => void;
  className?: string;
}

export function ActiveFilterChips({
  filterState,
  sections,
  onRemoveFilter,
  onClearAll,
  className,
}: ActiveFilterChipsProps) {
  // Build list of active filter chips
  const chips: ActiveFilterChip[] = useMemo(() => {
    const result: ActiveFilterChip[] = [];

    for (const [sectionId, value] of Object.entries(filterState)) {
      const section = sections.find((s) => s.id === sectionId);
      if (!section) continue;

      if (value instanceof Set) {
        // Checkbox filter - create chip for each selected value
        const checkboxSection = section as CheckboxFilterSection;
        for (const v of value) {
          const option = checkboxSection.options.find((o) => o.value === v);
          result.push({
            id: `${sectionId}-${v}`,
            sectionId,
            label: section.label,
            value: option?.label || v,
          });
        }
      } else if (typeof value === "boolean") {
        // Boolean filter
        const boolSection = section as BooleanFilterSection;
        result.push({
          id: sectionId,
          sectionId,
          label: section.label,
          value: value ? boolSection.trueLabel : boolSection.falseLabel,
        });
      } else if (value && typeof value === "object" && "min" in value) {
        // Range filter
        const rangeSection = section as RangeFilterSection;
        result.push({
          id: sectionId,
          sectionId,
          label: section.label,
          value: `${value.min}-${value.max}${rangeSection.unit || ""}`,
        });
      }
    }

    return result;
  }, [filterState, sections]);

  if (chips.length === 0) {
    return null;
  }

  return (
    <div className={cn("flex flex-wrap items-center gap-2", className)}>
      {chips.map((chip) => (
        <Badge
          key={chip.id}
          variant="secondary"
          className="flex items-center gap-1.5 px-2 py-1 h-7 bg-blue-50 text-blue-700 hover:bg-blue-100 dark:bg-blue-900/30 dark:text-blue-300 dark:hover:bg-blue-900/50 cursor-pointer group"
          onClick={() => onRemoveFilter(chip.sectionId, chip.value)}
        >
          <span className="text-xs font-medium text-blue-500 dark:text-blue-400">
            {chip.label}:
          </span>
          <span className="text-xs">{chip.value}</span>
          <button
            onClick={(e) => {
              e.stopPropagation();
              onRemoveFilter(chip.sectionId, chip.value);
            }}
            className="ml-0.5 text-blue-400 hover:text-blue-600 dark:text-blue-500 dark:hover:text-blue-300"
          >
            <svg
              className="h-3 w-3"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={2}
            >
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </Badge>
      ))}
      {chips.length > 1 && (
        <button
          onClick={onClearAll}
          className="text-xs text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 underline"
        >
          Clear all
        </button>
      )}
    </div>
  );
}

// ===========================================
// Hook for managing filter state
// ===========================================

export function useFilterState(initialState?: FilterState) {
  const [filterState, setFilterState] = useState<FilterState>(
    initialState || {}
  );

  const setFilter = useCallback((sectionId: string, value: Set<string> | { min: number; max: number } | boolean | undefined) => {
    setFilterState((prev) => ({
      ...prev,
      [sectionId]: value,
    }));
  }, []);

  const clearSection = useCallback((sectionId: string) => {
    setFilterState((prev) => {
      const newState = { ...prev };
      delete newState[sectionId];
      return newState;
    });
  }, []);

  const clearAll = useCallback(() => {
    setFilterState({});
  }, []);

  const getTotalCount = useCallback(() => {
    let count = 0;
    Object.entries(filterState).forEach(([, value]) => {
      if (value instanceof Set) {
        count += value.size;
      } else if (typeof value === "boolean") {
        count += 1;
      } else if (value && typeof value === "object" && "min" in value) {
        count += 1;
      }
    });
    return count;
  }, [filterState]);

  const hasActiveFilters = useCallback(() => {
    return getTotalCount() > 0;
  }, [getTotalCount]);

  return {
    filterState,
    setFilter,
    clearSection,
    clearAll,
    getTotalCount,
    hasActiveFilters,
  };
}

// ===========================================
// Hook for managing filter state with URL sync
// ===========================================

// URL parameter names mapping (frontend filter ID -> URL param name)
const FILTER_URL_PARAMS: Record<string, string> = {
  status: "status",
  category: "category",
  brand: "brand",
  subBrand: "sub_brand",
  productName: "product_name",
  flavor: "flavor",
  container: "container",
  netQuantity: "net_quantity",
  packType: "pack_type",
  country: "country",
  claims: "claims",
  issueTypes: "issue_types",
  hasVideo: "has_video",
  hasImage: "has_image",
  hasNutrition: "has_nutrition",
  hasDescription: "has_description",
  hasPrompt: "has_prompt",
  hasIssues: "has_issues",
  frameCount: "frame_count",
  visibilityScore: "visibility_score",
};

// Reverse mapping (URL param name -> frontend filter ID)
const URL_PARAM_TO_FILTER: Record<string, string> = Object.fromEntries(
  Object.entries(FILTER_URL_PARAMS).map(([k, v]) => [v, k])
);

// Parse filter state from URL search params
function parseFiltersFromURL(searchParams: URLSearchParams): FilterState {
  const state: FilterState = {};

  for (const [urlParam, filterId] of Object.entries(URL_PARAM_TO_FILTER)) {
    const value = searchParams.get(urlParam);
    if (!value) continue;

    // Boolean filters
    if (filterId.startsWith("has")) {
      if (value === "true") {
        state[filterId] = true;
      } else if (value === "false") {
        state[filterId] = false;
      }
    }
    // Range filters (format: "min-max")
    else if (filterId === "frameCount" || filterId === "visibilityScore") {
      const match = value.match(/^(\d+)-(\d+)$/);
      if (match) {
        state[filterId] = {
          min: parseInt(match[1], 10),
          max: parseInt(match[2], 10),
        };
      }
    }
    // Checkbox filters (comma-separated values)
    else {
      const values = value.split(",").filter(Boolean);
      if (values.length > 0) {
        state[filterId] = new Set(values);
      }
    }
  }

  return state;
}

// Serialize filter state to URL search params
function serializeFiltersToURL(
  filterState: FilterState,
  currentParams: URLSearchParams
): URLSearchParams {
  const params = new URLSearchParams(currentParams.toString());

  // Remove all filter params first
  for (const urlParam of Object.values(FILTER_URL_PARAMS)) {
    params.delete(urlParam);
  }

  // Add active filters
  for (const [filterId, value] of Object.entries(filterState)) {
    const urlParam = FILTER_URL_PARAMS[filterId];
    if (!urlParam) continue;

    if (value instanceof Set && value.size > 0) {
      params.set(urlParam, Array.from(value).join(","));
    } else if (typeof value === "boolean") {
      params.set(urlParam, String(value));
    } else if (value && typeof value === "object" && "min" in value) {
      params.set(urlParam, `${value.min}-${value.max}`);
    }
  }

  return params;
}

export function useFilterStateWithURL() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();

  // Initialize state from URL on first render
  const [filterState, setFilterState] = useState<FilterState>(() =>
    parseFiltersFromURL(searchParams)
  );

  // Track if we're initialized and if URL update is needed
  const [isInitialized, setIsInitialized] = useState(false);
  const [shouldUpdateURL, setShouldUpdateURL] = useState(false);

  // Sync URL -> State when URL changes externally (e.g., browser back/forward)
  useEffect(() => {
    const urlState = parseFiltersFromURL(searchParams);

    // Only update state if URL params actually differ
    const currentKeys = Object.keys(filterState);
    const urlKeys = Object.keys(urlState);

    const statesMatch =
      currentKeys.length === urlKeys.length &&
      currentKeys.every(key => {
        const current = filterState[key];
        const fromUrl = urlState[key];

        if (current instanceof Set && fromUrl instanceof Set) {
          return current.size === fromUrl.size &&
            Array.from(current).every(v => fromUrl.has(v));
        }
        if (typeof current === "object" && typeof fromUrl === "object" &&
            current !== null && fromUrl !== null &&
            "min" in current && "min" in fromUrl) {
          return current.min === fromUrl.min && current.max === fromUrl.max;
        }
        return current === fromUrl;
      });

    if (!statesMatch && !shouldUpdateURL) {
      setFilterState(urlState);
    }

    setIsInitialized(true);
  }, [searchParams]); // eslint-disable-line react-hooks/exhaustive-deps

  // Update URL when filter state changes (via useEffect to avoid render-time setState)
  useEffect(() => {
    if (isInitialized && shouldUpdateURL) {
      const params = serializeFiltersToURL(filterState, searchParams);
      // Reset page when filters change
      params.delete("page");
      router.push(`${pathname}?${params.toString()}`, { scroll: false });
      setShouldUpdateURL(false);
    }
  }, [filterState, shouldUpdateURL, isInitialized, searchParams, router, pathname]);

  const setFilter = useCallback(
    (sectionId: string, value: Set<string> | { min: number; max: number } | boolean | undefined) => {
      setFilterState((prev) => {
        const newState = { ...prev };
        if (value === undefined || (value instanceof Set && value.size === 0)) {
          delete newState[sectionId];
        } else {
          newState[sectionId] = value;
        }
        return newState;
      });
      setShouldUpdateURL(true);
    },
    []
  );

  const clearSection = useCallback(
    (sectionId: string) => {
      setFilterState((prev) => {
        const newState = { ...prev };
        delete newState[sectionId];
        return newState;
      });
      setShouldUpdateURL(true);
    },
    []
  );

  const clearAll = useCallback(() => {
    setFilterState({});
    setShouldUpdateURL(true);
  }, []);

  const getTotalCount = useCallback(() => {
    let count = 0;
    Object.entries(filterState).forEach(([, value]) => {
      if (value instanceof Set) {
        count += value.size;
      } else if (typeof value === "boolean") {
        count += 1;
      } else if (value && typeof value === "object" && "min" in value) {
        count += 1;
      }
    });
    return count;
  }, [filterState]);

  const hasActiveFilters = useCallback(() => {
    return getTotalCount() > 0;
  }, [getTotalCount]);

  return {
    filterState,
    setFilter,
    clearSection,
    clearAll,
    getTotalCount,
    hasActiveFilters,
  };
}
