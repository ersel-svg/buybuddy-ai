"use client";

import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Trash2, Layers, Package } from "lucide-react";
import { useCanvasStore } from "@/hooks/store-maps/use-canvas-store";
import { AREA_TYPE_CONFIGS } from "@/types/store-maps";
import type { AreaType, CanvasAreaObject } from "@/types/store-maps";
import { CategoryPicker } from "./category-picker";
import {
  pixelsToCm,
  formatMeasurement,
  formatArea,
  polygonArea,
  polygonPerimeter,
  circleArea,
} from "@/lib/store-maps/measurement";

export function PropertiesPanel() {
  const {
    selectedAreaId,
    areas,
    updateArea,
    removeArea,
    measurement,
    baseRatio,
    showProductPanel,
    setShowProductPanel,
  } = useCanvasStore();

  const selectedArea = areas.find((a) => a.fabricId === selectedAreaId);
  const [name, setName] = useState("");

  useEffect(() => {
    if (selectedArea) {
      setName(selectedArea.name);
    }
  }, [selectedArea]);

  const handleNameChange = (newName: string) => {
    setName(newName);
    if (selectedAreaId) {
      updateArea(selectedAreaId, { name: newName });
    }
  };

  const handleTypeChange = (type: AreaType) => {
    if (selectedAreaId) {
      updateArea(selectedAreaId, { areaType: type });
    }
  };

  const handleDelete = () => {
    if (selectedAreaId) {
      removeArea(selectedAreaId);
    }
  };

  // Calculate measurements
  const getMeasurements = (area: CanvasAreaObject) => {
    const unit = measurement.unitSystem;

    if (area.isCircle && area.radius) {
      const radiusCm = pixelsToCm(area.radius, baseRatio);
      const areaPxSq = circleArea(area.radius);
      const areaCmSq = areaPxSq * baseRatio * baseRatio;
      const perimeterCm = pixelsToCm(2 * Math.PI * area.radius, baseRatio);
      return {
        dimensions: `r: ${formatMeasurement(radiusCm, unit)}`,
        area: formatArea(areaCmSq, unit),
        perimeter: formatMeasurement(perimeterCm, unit),
      };
    }

    if (area.coordinates.length >= 3) {
      const areaPx = polygonArea(area.coordinates);
      const areaCmSq = areaPx * baseRatio * baseRatio;
      const perimPx = polygonPerimeter(area.coordinates);
      const perimCm = pixelsToCm(perimPx, baseRatio);

      // Bounding box
      const xs = area.coordinates.map((c) => c.x);
      const ys = area.coordinates.map((c) => c.y);
      const widthCm = pixelsToCm(Math.max(...xs) - Math.min(...xs), baseRatio);
      const heightCm = pixelsToCm(Math.max(...ys) - Math.min(...ys), baseRatio);

      return {
        dimensions: `${formatMeasurement(widthCm, unit)} × ${formatMeasurement(heightCm, unit)}`,
        area: formatArea(areaCmSq, unit),
        perimeter: formatMeasurement(perimCm, unit),
      };
    }

    return { dimensions: "-", area: "-", perimeter: "-" };
  };

  if (!selectedArea) {
    return (
      <div className="w-72 border-l bg-card flex flex-col">
        <div className="p-4 border-b">
          <h3 className="font-medium text-sm">Properties</h3>
        </div>
        <div className="flex-1 flex items-center justify-center">
          <p className="text-sm text-muted-foreground text-center px-4">
            Select an area on the map to view its properties.
          </p>
        </div>
      </div>
    );
  }

  const config = AREA_TYPE_CONFIGS[selectedArea.areaType];
  const measurements = getMeasurements(selectedArea);

  return (
    <div className="w-72 border-l bg-card flex flex-col overflow-hidden min-h-0">
      <div className="p-4 border-b flex items-center justify-between">
        <h3 className="font-medium text-sm">Properties</h3>
        <div
          className="w-3 h-3 rounded-full"
          style={{ backgroundColor: config.color }}
        />
      </div>

      <ScrollArea className="flex-1 min-h-0">
        <div className="p-4 space-y-4">
          {/* Name */}
          <div className="space-y-2">
            <Label className="text-xs">Name</Label>
            <Input
              value={name}
              onChange={(e) => handleNameChange(e.target.value)}
              placeholder="Area name..."
              className="h-8 text-sm"
            />
          </div>

          {/* Type */}
          <div className="space-y-2">
            <Label className="text-xs">Type</Label>
            <Select
              value={selectedArea.areaType}
              onValueChange={(v) => handleTypeChange(v as AreaType)}
            >
              <SelectTrigger className="h-8 text-sm">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {Object.values(AREA_TYPE_CONFIGS).map((cfg) => (
                  <SelectItem key={cfg.type} value={cfg.type}>
                    <div className="flex items-center gap-2">
                      <div
                        className="w-3 h-3 rounded-sm"
                        style={{ backgroundColor: cfg.color }}
                      />
                      {cfg.label}
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <Separator />

          {/* Measurements */}
          <div className="space-y-3">
            <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
              Measurements
            </h4>
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div className="space-y-1">
                <p className="text-xs text-muted-foreground">Dimensions</p>
                <p className="font-mono text-xs">{measurements.dimensions}</p>
              </div>
              <div className="space-y-1">
                <p className="text-xs text-muted-foreground">Area</p>
                <p className="font-mono text-xs">{measurements.area}</p>
              </div>
              <div className="space-y-1">
                <p className="text-xs text-muted-foreground">Perimeter</p>
                <p className="font-mono text-xs">{measurements.perimeter}</p>
              </div>
              <div className="space-y-1">
                <p className="text-xs text-muted-foreground">Vertices</p>
                <p className="font-mono text-xs">
                  {selectedArea.coordinates.length}
                </p>
              </div>
            </div>
          </div>

          <Separator />

          {/* Products */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
                Products
              </h4>
              <Badge variant="secondary" className="text-xs">
                {selectedArea.products?.length ?? 0}
              </Badge>
            </div>
            <Button
              variant="outline"
              size="sm"
              className="w-full text-xs"
              onClick={() => setShowProductPanel(!showProductPanel)}
            >
              <Package className="h-3 w-3 mr-1" />
              {showProductPanel ? "Hide Product Panel" : "Assign Products"}
            </Button>
          </div>

          {/* Categories */}
          <div className="space-y-2">
            <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
              Categories
            </h4>
            <CategoryPicker
              selectedCategories={selectedArea.categories ?? []}
              onCategoriesChange={(categories) => {
                if (selectedAreaId) {
                  updateArea(selectedAreaId, { categories });
                }
              }}
            />
          </div>

          <Separator />

          {/* Coordinates (collapsed by default) */}
          <details>
            <summary className="text-xs font-medium text-muted-foreground uppercase tracking-wider cursor-pointer hover:text-foreground">
              Coordinates ({selectedArea.coordinates.length} points)
            </summary>
            <div className="mt-2 space-y-1">
              {selectedArea.coordinates.map((coord, idx) => (
                <div
                  key={idx}
                  className="flex gap-2 text-xs font-mono text-muted-foreground"
                >
                  <span className="w-5 text-right">{idx + 1}.</span>
                  <span>
                    ({coord.x.toFixed(1)}, {coord.y.toFixed(1)})
                  </span>
                </div>
              ))}
            </div>
          </details>

          <Separator />

          {/* Actions */}
          <div className="space-y-2">
            <Button
              variant="outline"
              size="sm"
              className="w-full text-red-600 hover:text-red-700 hover:bg-red-50 text-xs"
              onClick={handleDelete}
            >
              <Trash2 className="h-3 w-3 mr-1" />
              Delete Area
            </Button>
          </div>
        </div>
      </ScrollArea>
    </div>
  );
}
