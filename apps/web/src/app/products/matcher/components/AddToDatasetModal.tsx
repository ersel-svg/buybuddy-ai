"use client";

import { useState, useEffect, useCallback } from "react";
import { Loader2 } from "lucide-react";
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
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { JobProgressModal } from "@/components/common/job-progress-modal";
import { apiClient } from "@/lib/api-client";
import { toast } from "sonner";
import type { Dataset } from "@/types";

interface AddToDatasetModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  productIds: string[];
}

export function AddToDatasetModal({
  open,
  onOpenChange,
  productIds,
}: AddToDatasetModalProps) {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>("");
  const [isLoading, setIsLoading] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);

  const loadDatasets = useCallback(async () => {
    setIsLoading(true);
    try {
      const result = await apiClient.getDatasets();
      setDatasets(result);
      if (result.length > 0) {
        setSelectedDatasetId((prev) => prev || result[0].id);
      }
    } catch {
      toast.error("Failed to load datasets");
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    if (open) {
      loadDatasets();
    }
  }, [open, loadDatasets]);

  const handleSubmit = async () => {
    if (!selectedDatasetId || productIds.length === 0) return;

    setIsSubmitting(true);
    try {
      const result = await apiClient.addProductsToDataset(selectedDatasetId, productIds);

      // If backend returns job_id, track it with modal
      if (result.job_id) {
        setActiveJobId(result.job_id);
        toast.success("Background job started for adding products");
        onOpenChange(false);
      } else {
        // Small batch completed synchronously
        toast.success(`Added ${result.added_count || productIds.length} products to dataset`);
        onOpenChange(false);
      }
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : "Failed to add products"
      );
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Add to Dataset</DialogTitle>
          <DialogDescription>
            Add {productIds.length} matched product{productIds.length !== 1 ? "s" : ""} to a
            dataset.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-4">
          <div className="space-y-2">
            <Label>Select Dataset</Label>
            {isLoading ? (
              <div className="flex items-center gap-2 text-muted-foreground">
                <Loader2 className="h-4 w-4 animate-spin" />
                Loading datasets...
              </div>
            ) : datasets.length === 0 ? (
              <p className="text-sm text-muted-foreground">
                No datasets available. Create a dataset first.
              </p>
            ) : (
              <Select
                value={selectedDatasetId}
                onValueChange={setSelectedDatasetId}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select a dataset" />
                </SelectTrigger>
                <SelectContent>
                  {datasets.map((dataset) => (
                    <SelectItem key={dataset.id} value={dataset.id}>
                      <div className="flex items-center gap-2">
                        <span>{dataset.name}</span>
                        <span className="text-muted-foreground text-xs">
                          ({dataset.product_count} products)
                        </span>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button
            onClick={handleSubmit}
            disabled={
              !selectedDatasetId || productIds.length === 0 || isSubmitting
            }
          >
            {isSubmitting && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
            Add {productIds.length} Product{productIds.length !== 1 ? "s" : ""}
          </Button>
        </DialogFooter>
      </DialogContent>

      {/* Job Progress Modal - appears after main dialog closes */}
      <JobProgressModal
        jobId={activeJobId}
        title="Adding Products to Dataset"
        onClose={() => setActiveJobId(null)}
        onComplete={(result) => {
          toast.success(
            `Added ${result?.added || 0} products (${result?.skipped || 0} skipped)`
          );
        }}
      />
    </Dialog>
  );
}
