"use client";

import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Loader2 } from "lucide-react";
import { useCreateStoreMap } from "@/hooks/store-maps/use-map-queries";
import type { UnitSystem } from "@/types/store-maps";
import { apiClient } from "@/lib/api-client";
import { toast } from "sonner";

interface CreateMapDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSuccess?: (mapId: number) => void;
}

export function CreateMapDialog({
  open,
  onOpenChange,
  onSuccess,
}: CreateMapDialogProps) {
  const [name, setName] = useState("");
  const [storeId, setStoreId] = useState("");
  const [unitSystem, setUnitSystem] = useState<UnitSystem>("metric");
  const [gridSize, setGridSize] = useState("50"); // cm
  const [stores, setStores] = useState<Array<{ id: number; name: string }>>([]);
  const [loadingStores, setLoadingStores] = useState(false);
  const createMap = useCreateStoreMap();

  // Fetch user stores on mount
  useEffect(() => {
    if (open) {
      console.log("🔍 CreateMapDialog opened, fetching stores...");
      setLoadingStores(true);
      apiClient
        .getUserStores()
        .then((response) => {
          console.log("✅ Stores loaded:", response);
          // API returns {data: [...]} format
          const storeList = Array.isArray(response) 
            ? response 
            : (response as any)?.data || [];
          
          // Map to the format we need: {id, name}
          const mapped = storeList.map((s: any) => ({
            id: s.merchant_store_id || s.id,
            name: s.store_name || s.name || `Store ${s.id}`,
          }));
          console.log("📦 Mapped stores:", mapped);
          setStores(mapped);
        })
        .catch((error) => {
          console.error("❌ Failed to load stores:", error);
          toast.error("Failed to load stores");
        })
        .finally(() => {
          setLoadingStores(false);
          console.log("🏁 Loading complete");
        });
    }
  }, [open]);

  const handleSubmit = async () => {
    if (!name.trim() || !storeId.trim()) return;

    const gridCm =
      unitSystem === "metric"
        ? parseFloat(gridSize)
        : parseFloat(gridSize) * 30.48;

    const result = await createMap.mutateAsync({
      store_id: parseInt(storeId),
      name: name.trim(),
      base_ratio: 1, // default, user calibrates later
      grid: gridCm,
    });

    setName("");
    setStoreId("");
    setGridSize("50");
    onOpenChange(false);
    if (onSuccess && result?.id) {
      onSuccess(result.id);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Create New Store Map</DialogTitle>
          <DialogDescription>
            Create a new map for a store location. You can upload floor plans and
            draw areas after creation.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-4">
          <div className="space-y-2">
            <Label htmlFor="map-name">Map Name *</Label>
            <Input
              id="map-name"
              placeholder="e.g. Istanbul Kadikoy Store"
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="store-id">Store *</Label>
            {/* Debug: loading={loadingStores.toString()} stores={stores.length} */}
            {loadingStores ? (
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Loader2 className="h-4 w-4 animate-spin" />
                Loading stores...
              </div>
            ) : stores.length > 0 ? (
              <Select value={storeId} onValueChange={setStoreId}>
                <SelectTrigger id="store-id">
                  <SelectValue placeholder="Select a store" />
                </SelectTrigger>
                <SelectContent>
                  {stores.map((store) => (
                    <SelectItem key={store.id} value={String(store.id)}>
                      {store.name} (ID: {store.id})
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            ) : (
              <Input
                id="store-id"
                type="number"
                placeholder="e.g. 1001"
                value={storeId}
                onChange={(e) => setStoreId(e.target.value)}
              />
            )}
          </div>

          <div className="space-y-2">
            <Label>Unit System</Label>
            <Select
              value={unitSystem}
              onValueChange={(v) => setUnitSystem(v as UnitSystem)}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="metric">Metric (m / cm)</SelectItem>
                <SelectItem value="imperial">Imperial (ft / in)</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="grid-size">
              Grid Size{" "}
              <span className="text-muted-foreground font-normal">
                ({unitSystem === "metric" ? "cm" : "ft"})
              </span>
            </Label>
            <Select value={gridSize} onValueChange={setGridSize}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {unitSystem === "metric" ? (
                  <>
                    <SelectItem value="25">25 cm</SelectItem>
                    <SelectItem value="50">50 cm</SelectItem>
                    <SelectItem value="100">1 m</SelectItem>
                    <SelectItem value="200">2 m</SelectItem>
                  </>
                ) : (
                  <>
                    <SelectItem value="0.5">6 in</SelectItem>
                    <SelectItem value="1">1 ft</SelectItem>
                    <SelectItem value="2">2 ft</SelectItem>
                    <SelectItem value="5">5 ft</SelectItem>
                  </>
                )}
              </SelectContent>
            </Select>
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button
            onClick={handleSubmit}
            disabled={
              !name.trim() || !storeId.trim() || createMap.isPending
            }
          >
            {createMap.isPending ? (
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            ) : null}
            Create Map
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
