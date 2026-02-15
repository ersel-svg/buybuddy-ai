"use client";

import { useState, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Plus, Layers, Upload, Loader2 } from "lucide-react";
import { useCanvasStore } from "@/hooks/store-maps/use-canvas-store";
import {
  useMapFloors,
  useCreateMapFloor,
} from "@/hooks/store-maps/use-map-queries";
import { cn } from "@/lib/utils";
import { useDropzone } from "react-dropzone";

interface FloorPanelProps {
  onFloorChange?: (floorId: number, floorPlanUrl: string) => void;
}

export function FloorPanel({ onFloorChange }: FloorPanelProps) {
  const { mapId, activeFloorId, setActiveFloorId } = useCanvasStore();
  const { data: floors, isLoading } = useMapFloors(mapId ?? 0);
  const createFloor = useCreateMapFloor();

  const [addFloorOpen, setAddFloorOpen] = useState(false);
  const [floorNumber, setFloorNumber] = useState("0");
  const [floorFile, setFloorFile] = useState<File | null>(null);

  const onDrop = useCallback((files: File[]) => {
    if (files.length > 0) {
      setFloorFile(files[0]);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [".png", ".jpg", ".jpeg", ".webp"] },
    multiple: false,
  });

  const handleAddFloor = async () => {
    if (!mapId || !floorFile) return;
    await createFloor.mutateAsync({
      mapId,
      floor: parseInt(floorNumber),
      file: floorFile,
    });
    setAddFloorOpen(false);
    setFloorFile(null);
    setFloorNumber("0");
  };

  const handleFloorClick = (floorId: number, floorPlan: string) => {
    setActiveFloorId(floorId);
    onFloorChange?.(floorId, floorPlan);
  };

  return (
    <>
      <div className="w-48 border-r bg-card flex flex-col overflow-hidden min-h-0">
        <div className="p-3 border-b flex items-center justify-between">
          <div className="flex items-center gap-1.5">
            <Layers className="h-4 w-4 text-muted-foreground" />
            <h3 className="font-medium text-sm">Floors</h3>
          </div>
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            onClick={() => setAddFloorOpen(true)}
          >
            <Plus className="h-4 w-4" />
          </Button>
        </div>

        <ScrollArea className="flex-1 min-h-0">
          <div className="p-2 space-y-1">
            {isLoading ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
              </div>
            ) : !floors || floors.length === 0 ? (
              <div className="text-center py-6 px-2">
                <p className="text-xs text-muted-foreground">
                  No floors yet. Add a floor with a plan image.
                </p>
                <Button
                  variant="outline"
                  size="sm"
                  className="mt-2 text-xs"
                  onClick={() => setAddFloorOpen(true)}
                >
                  <Plus className="h-3 w-3 mr-1" />
                  Add Floor
                </Button>
              </div>
            ) : (
              floors.map((floor) => (
                <button
                  key={floor.id}
                  className={cn(
                    "w-full text-left rounded-lg p-2 transition-colors",
                    activeFloorId === floor.id
                      ? "bg-primary text-primary-foreground"
                      : "hover:bg-accent text-muted-foreground hover:text-foreground"
                  )}
                  onClick={() => handleFloorClick(floor.id, floor.floor_plan)}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">
                      Floor {floor.floor}
                    </span>
                    <Badge
                      variant={
                        activeFloorId === floor.id ? "outline" : "secondary"
                      }
                      className="text-[10px] h-5"
                    >
                      {floor.area_count ?? 0}
                    </Badge>
                  </div>
                  {floor.floor_plan && (
                    <div className="mt-1.5 rounded overflow-hidden bg-neutral-800 h-16">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={floor.floor_plan}
                        alt={`Floor ${floor.floor}`}
                        className="w-full h-full object-cover opacity-60"
                      />
                    </div>
                  )}
                </button>
              ))
            )}
          </div>
        </ScrollArea>
      </div>

      {/* Add Floor Dialog */}
      <Dialog open={addFloorOpen} onOpenChange={setAddFloorOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Add Floor</DialogTitle>
            <DialogDescription>
              Add a new floor to this map. Upload a floor plan image to use as
              the canvas background.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label htmlFor="floor-number">Floor Number</Label>
              <Input
                id="floor-number"
                type="number"
                value={floorNumber}
                onChange={(e) => setFloorNumber(e.target.value)}
                placeholder="0"
              />
            </div>

            <div className="space-y-2">
              <Label>Floor Plan Image *</Label>
              <div
                {...getRootProps()}
                className={cn(
                  "border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors",
                  isDragActive
                    ? "border-primary bg-primary/5"
                    : "border-border hover:border-primary/50"
                )}
              >
                <input {...getInputProps()} />
                {floorFile ? (
                  <div className="space-y-2">
                    <p className="text-sm font-medium">{floorFile.name}</p>
                    <p className="text-xs text-muted-foreground">
                      {(floorFile.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                  </div>
                ) : (
                  <div className="space-y-2">
                    <Upload className="h-8 w-8 mx-auto text-muted-foreground" />
                    <p className="text-sm text-muted-foreground">
                      Drag & drop or click to upload
                    </p>
                    <p className="text-xs text-muted-foreground">
                      PNG, JPG, WEBP
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setAddFloorOpen(false)}>
              Cancel
            </Button>
            <Button
              onClick={handleAddFloor}
              disabled={!floorFile || createFloor.isPending}
            >
              {createFloor.isPending ? (
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              ) : null}
              Add Floor
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
