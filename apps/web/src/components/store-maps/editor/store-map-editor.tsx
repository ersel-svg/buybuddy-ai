"use client";

import { useRef, useCallback, useEffect } from "react";
import { useRouter } from "next/navigation";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ChevronLeft } from "lucide-react";
import Link from "next/link";

import { EditorToolbar } from "./toolbar";
import { StoreMapCanvas } from "./canvas";
import { PropertiesPanel } from "./properties-panel";
import { FloorPanel } from "./floor-panel";
import { ProductPanel } from "./product-panel";
import { useCanvasStore } from "@/hooks/store-maps/use-canvas-store";
import { useStoreMap } from "@/hooks/store-maps/use-map-queries";
import { useAutoSave } from "@/hooks/store-maps/use-auto-save";
import { AREA_TYPE_CONFIGS } from "@/types/store-maps";
import type { AreaType } from "@/types/store-maps";
import type { Canvas as FabricCanvas } from "fabric";

interface StoreMapEditorProps {
  mapId: number;
}

export function StoreMapEditor({ mapId }: StoreMapEditorProps) {
  const router = useRouter();
  const fabricCanvasRef = useRef<FabricCanvas | null>(null);
  const { data: map } = useStoreMap(mapId);

  const {
    setMapId,
    setBaseRatio,
    setGridSizeCm,
    showPropertiesPanel,
    pendingAreaType,
    setPendingAreaType,
    setZoom,
    activeFloorId,
  } = useCanvasStore();

  const { save } = useAutoSave();

  // Initialize store with map data
  useEffect(() => {
    setMapId(mapId);
    if (map) {
      if (map.base_ratio) setBaseRatio(map.base_ratio);
      if (map.grid) setGridSizeCm(map.grid);
    }
  }, [mapId, map, setMapId, setBaseRatio, setGridSizeCm]);

  // Floor plan change handler
  const handleFloorChange = useCallback(
    (_floorId: number, _floorPlanUrl: string) => {
      // Canvas will reload with new floor plan
      // This is handled by re-mounting or updating the canvas component
    },
    []
  );

  // Zoom controls
  const handleZoomIn = useCallback(() => {
    const canvas = fabricCanvasRef.current;
    if (!canvas) return;
    const newZoom = Math.min(canvas.getZoom() * 1.2, 10);
    canvas.setZoom(newZoom);
    setZoom(newZoom);
    canvas.renderAll();
  }, [setZoom]);

  const handleZoomOut = useCallback(() => {
    const canvas = fabricCanvasRef.current;
    if (!canvas) return;
    const newZoom = Math.max(canvas.getZoom() / 1.2, 0.1);
    canvas.setZoom(newZoom);
    setZoom(newZoom);
    canvas.renderAll();
  }, [setZoom]);

  const handleFitToScreen = useCallback(() => {
    const canvas = fabricCanvasRef.current;
    if (!canvas) return;
    canvas.setViewportTransform([1, 0, 0, 1, 0, 0]);
    setZoom(1);
    canvas.renderAll();
  }, [setZoom]);

  const handleSave = useCallback(() => {
    save();
  }, [save]);

  // Ctrl+S handler
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "s") {
        e.preventDefault();
        handleSave();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleSave]);

  return (
    <TooltipProvider delayDuration={0}>
      <div className="h-full flex flex-col bg-background">
        {/* Header Bar */}
        <div className="flex items-center justify-between px-3 py-1.5 border-b bg-card shrink-0">
          {/* Left: Back + Title */}
          <div className="flex items-center gap-2">
            <Link href="/store-maps">
              <Button variant="ghost" size="icon" className="h-8 w-8">
                <ChevronLeft className="h-4 w-4" />
              </Button>
            </Link>
            <div className="flex items-center gap-2">
              <h2 className="text-sm font-medium">
                {map?.name ?? "Loading..."}
              </h2>
              {map?.store_name && (
                <Badge variant="secondary" className="text-xs">
                  {map.store_name}
                </Badge>
              )}
            </div>
          </div>

          {/* Center: Toolbar */}
          <EditorToolbar
            onZoomIn={handleZoomIn}
            onZoomOut={handleZoomOut}
            onFitToScreen={handleFitToScreen}
            onSave={handleSave}
          />

          {/* Right: Area type picker */}
          <div className="flex items-center gap-2">
            <span className="text-xs text-muted-foreground">Draw as:</span>
            <Select
              value={pendingAreaType}
              onValueChange={(v) => setPendingAreaType(v as AreaType)}
            >
              <SelectTrigger className="h-8 w-[130px] text-xs">
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
                      <span>{cfg.label}</span>
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>

        {/* Main Content */}
        <div className="flex-1 flex overflow-hidden min-h-0">
          {/* Left: Floor Panel */}
          <FloorPanel onFloorChange={handleFloorChange} />

          {/* Center: Canvas */}
          <StoreMapCanvas
            canvasRef={fabricCanvasRef}
            floorPlanUrl={undefined} // Will be set when floor is selected
          />

          {/* Right: Properties Panel */}
          {showPropertiesPanel && <PropertiesPanel />}

          {/* Far Right: Product Panel (when open) */}
          <ProductPanel />
        </div>
      </div>
    </TooltipProvider>
  );
}
