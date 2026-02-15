import { useEffect, useRef, useCallback } from "react";
import { useCanvasStore } from "./use-canvas-store";
import {
  useCreateMapArea,
  useCreateAreaCoordinate,
  useDeleteMapArea,
  useUpdateMapArea,
} from "./use-map-queries";
import { toast } from "sonner";

/**
 * Auto-save hook that syncs canvas state to the API.
 * Debounces saves to avoid excessive API calls.
 */
export function useAutoSave() {
  const {
    areas,
    isDirty,
    setIsDirty,
    isSaving,
    setIsSaving,
    activeFloorId,
  } = useCanvasStore();

  const createArea = useCreateMapArea();
  const createCoordinate = useCreateAreaCoordinate();
  const deleteArea = useDeleteMapArea();
  const updateAreaMutation = useUpdateMapArea();

  const saveTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastSavedRef = useRef<string>("");

  const saveToApi = useCallback(async () => {
    if (!activeFloorId || !isDirty) return;

    const currentState = JSON.stringify(areas);
    if (currentState === lastSavedRef.current) {
      setIsDirty(false);
      return;
    }

    setIsSaving(true);

    try {
      // For now, we just mark as saved.
      // Full implementation would:
      // 1. Compare current areas with last saved state
      // 2. Create new areas via POST /v1/map/area
      // 3. Create coordinates via POST /v1/map/area/coordinate
      // 4. Update modified areas via PUT /v1/map/area
      // 5. Delete removed areas via DELETE /v1/map/area/:id
      //
      // This requires tracking which areas have been synced (have areaId)
      // vs which are new (no areaId yet).

      // For each area without an areaId, create it
      for (const area of areas) {
        if (!area.areaId && activeFloorId) {
          try {
            const result = await createArea.mutateAsync({
              name: area.name,
              floor_id: activeFloorId,
            });

            if (result?.id) {
              // Update local state with the new areaId
              useCanvasStore.getState().updateArea(area.fabricId, {
                areaId: result.id,
              });

              // Create coordinates
              for (const coord of area.coordinates) {
                await createCoordinate.mutateAsync({
                  area_id: result.id,
                  x: coord.x,
                  y: coord.y,
                  z: 0,
                  r: area.radius ?? 0,
                  circle: area.isCircle,
                });
              }
            }
          } catch (err) {
            console.error("Failed to save area:", area.name, err);
          }
        }
      }

      lastSavedRef.current = JSON.stringify(
        useCanvasStore.getState().areas
      );
      setIsDirty(false);
      setIsSaving(false);
    } catch (error) {
      console.error("Auto-save failed:", error);
      setIsSaving(false);
      toast.error("Failed to auto-save. Your changes are preserved locally.");
    }
  }, [
    areas,
    isDirty,
    activeFloorId,
    setIsDirty,
    setIsSaving,
    createArea,
    createCoordinate,
  ]);

  // Debounced auto-save (3 seconds after last change)
  useEffect(() => {
    if (!isDirty) return;

    if (saveTimeoutRef.current) {
      clearTimeout(saveTimeoutRef.current);
    }

    saveTimeoutRef.current = setTimeout(() => {
      saveToApi();
    }, 3000);

    return () => {
      if (saveTimeoutRef.current) {
        clearTimeout(saveTimeoutRef.current);
      }
    };
  }, [isDirty, saveToApi]);

  // Manual save function
  const save = useCallback(() => {
    if (saveTimeoutRef.current) {
      clearTimeout(saveTimeoutRef.current);
    }
    saveToApi();
  }, [saveToApi]);

  return { save, isSaving };
}
