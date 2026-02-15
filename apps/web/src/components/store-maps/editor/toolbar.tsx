"use client";

import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Separator } from "@/components/ui/separator";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  MousePointer2,
  Square,
  Pentagon,
  Circle,
  Ruler,
  Scaling,
  Hand,
  Undo2,
  Redo2,
  ZoomIn,
  ZoomOut,
  Maximize2,
  Grid3X3,
  Eye,
  EyeOff,
  Save,
  Loader2,
} from "lucide-react";
import { useCanvasStore } from "@/hooks/store-maps/use-canvas-store";
import type { ToolMode, UnitSystem } from "@/types/store-maps";
import { cn } from "@/lib/utils";

interface ToolButtonProps {
  tool: ToolMode;
  icon: React.ReactNode;
  label: string;
}

function ToolButton({ tool, icon, label }: ToolButtonProps) {
  const { activeTool, setActiveTool } = useCanvasStore();
  const isActive = activeTool === tool;

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          className={cn(
            "h-9 w-9",
            isActive && "bg-primary text-primary-foreground hover:bg-primary/90 hover:text-primary-foreground"
          )}
          onClick={() => setActiveTool(tool)}
        >
          {icon}
        </Button>
      </TooltipTrigger>
      <TooltipContent side="bottom">{label}</TooltipContent>
    </Tooltip>
  );
}

interface EditorToolbarProps {
  onZoomIn: () => void;
  onZoomOut: () => void;
  onFitToScreen: () => void;
  onSave: () => void;
}

export function EditorToolbar({
  onZoomIn,
  onZoomOut,
  onFitToScreen,
  onSave,
}: EditorToolbarProps) {
  const {
    zoom,
    measurement,
    updateMeasurement,
    canUndo,
    canRedo,
    undo,
    redo,
    isDirty,
    isSaving,
    gridSizeCm,
    setGridSizeCm,
  } = useCanvasStore();

  const unitSystem = measurement.unitSystem;

  return (
    <div className="flex items-center gap-1 px-2">
      {/* Drawing Tools */}
      <ToolButton
        tool="select"
        icon={<MousePointer2 className="h-4 w-4" />}
        label="Select (V)"
      />
      <ToolButton
        tool="pan"
        icon={<Hand className="h-4 w-4" />}
        label="Pan (Space)"
      />
      <div className="h-6 w-px bg-border mx-1" />
      <ToolButton
        tool="rectangle"
        icon={<Square className="h-4 w-4" />}
        label="Rectangle (R)"
      />
      <ToolButton
        tool="polygon"
        icon={<Pentagon className="h-4 w-4" />}
        label="Polygon (P)"
      />
      <ToolButton
        tool="circle"
        icon={<Circle className="h-4 w-4" />}
        label="Circle (C)"
      />
      <div className="h-6 w-px bg-border mx-1" />
      <ToolButton
        tool="measure"
        icon={<Ruler className="h-4 w-4" />}
        label="Measure (M)"
      />
      <ToolButton
        tool="calibrate"
        icon={<Scaling className="h-4 w-4" />}
        label="Calibrate Scale"
      />

      <div className="h-6 w-px bg-border mx-1" />

      {/* Undo / Redo */}
      <Tooltip>
        <TooltipTrigger asChild>
          <Button
            variant="ghost"
            size="icon"
            className="h-9 w-9"
            disabled={!canUndo()}
            onClick={undo}
          >
            <Undo2 className="h-4 w-4" />
          </Button>
        </TooltipTrigger>
        <TooltipContent side="bottom">Undo (Ctrl+Z)</TooltipContent>
      </Tooltip>
      <Tooltip>
        <TooltipTrigger asChild>
          <Button
            variant="ghost"
            size="icon"
            className="h-9 w-9"
            disabled={!canRedo()}
            onClick={redo}
          >
            <Redo2 className="h-4 w-4" />
          </Button>
        </TooltipTrigger>
        <TooltipContent side="bottom">Redo (Ctrl+Shift+Z)</TooltipContent>
      </Tooltip>

      {/* Spacer */}
      <div className="flex-1" />

      {/* Grid & Measurement Controls */}
      <div className="flex items-center gap-2">
        {/* Unit Toggle */}
        <Select
          value={unitSystem}
          onValueChange={(v) =>
            updateMeasurement({ unitSystem: v as UnitSystem })
          }
        >
          <SelectTrigger className="h-8 w-[90px] text-xs">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="metric">m / cm</SelectItem>
            <SelectItem value="imperial">ft / in</SelectItem>
          </SelectContent>
        </Select>

        {/* Grid Size */}
        <Select
          value={String(gridSizeCm)}
          onValueChange={(v) => setGridSizeCm(parseFloat(v))}
        >
          <SelectTrigger className="h-8 w-[80px] text-xs">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {unitSystem === "metric" ? (
              <>
                <SelectItem value="25">25cm</SelectItem>
                <SelectItem value="50">50cm</SelectItem>
                <SelectItem value="100">1m</SelectItem>
                <SelectItem value="200">2m</SelectItem>
              </>
            ) : (
              <>
                <SelectItem value="15.24">6in</SelectItem>
                <SelectItem value="30.48">1ft</SelectItem>
                <SelectItem value="60.96">2ft</SelectItem>
                <SelectItem value="152.4">5ft</SelectItem>
              </>
            )}
          </SelectContent>
        </Select>

        <Separator orientation="vertical" className="h-6" />

        {/* Grid Toggle */}
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="ghost"
              size="icon"
              className={cn(
                "h-8 w-8",
                measurement.showGrid && "text-primary"
              )}
              onClick={() =>
                updateMeasurement({ showGrid: !measurement.showGrid })
              }
            >
              <Grid3X3 className="h-4 w-4" />
            </Button>
          </TooltipTrigger>
          <TooltipContent side="bottom">Toggle Grid</TooltipContent>
        </Tooltip>

        {/* Dimensions Toggle */}
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8"
              onClick={() =>
                updateMeasurement({
                  showDimensions:
                    measurement.showDimensions === "all"
                      ? "selected"
                      : measurement.showDimensions === "selected"
                        ? "none"
                        : "all",
                })
              }
            >
              {measurement.showDimensions === "none" ? (
                <EyeOff className="h-4 w-4" />
              ) : (
                <Eye className="h-4 w-4" />
              )}
            </Button>
          </TooltipTrigger>
          <TooltipContent side="bottom">
            Dimensions:{" "}
            {measurement.showDimensions === "all"
              ? "All"
              : measurement.showDimensions === "selected"
                ? "Selected Only"
                : "Hidden"}
          </TooltipContent>
        </Tooltip>

        <Separator orientation="vertical" className="h-6" />

        {/* Zoom Controls */}
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8"
              onClick={onZoomOut}
            >
              <ZoomOut className="h-4 w-4" />
            </Button>
          </TooltipTrigger>
          <TooltipContent side="bottom">Zoom Out</TooltipContent>
        </Tooltip>

        <span className="text-xs text-muted-foreground w-12 text-center font-mono">
          {Math.round(zoom * 100)}%
        </span>

        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8"
              onClick={onZoomIn}
            >
              <ZoomIn className="h-4 w-4" />
            </Button>
          </TooltipTrigger>
          <TooltipContent side="bottom">Zoom In</TooltipContent>
        </Tooltip>

        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8"
              onClick={onFitToScreen}
            >
              <Maximize2 className="h-4 w-4" />
            </Button>
          </TooltipTrigger>
          <TooltipContent side="bottom">Fit to Screen</TooltipContent>
        </Tooltip>

        <Separator orientation="vertical" className="h-6" />

        {/* Save */}
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant={isDirty ? "default" : "ghost"}
              size="sm"
              className="h-8"
              onClick={onSave}
              disabled={isSaving || !isDirty}
            >
              {isSaving ? (
                <Loader2 className="h-4 w-4 mr-1 animate-spin" />
              ) : (
                <Save className="h-4 w-4 mr-1" />
              )}
              {isSaving ? "Saving..." : isDirty ? "Save" : "Saved"}
            </Button>
          </TooltipTrigger>
          <TooltipContent side="bottom">Save (Ctrl+S)</TooltipContent>
        </Tooltip>
      </div>
    </div>
  );
}
