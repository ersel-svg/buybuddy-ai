"use client";

import { useState, useMemo } from "react";
import { Loader2, AlertCircle } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { apiClient } from "@/lib/api-client";
import { toast } from "sonner";

interface UnmatchedItem {
  source_row: Record<string, unknown>;
}

interface CreateScanRequestsModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  items: UnmatchedItem[];
  columns: string[];
  fileName: string;
}

export function CreateScanRequestsModal({
  open,
  onOpenChange,
  items,
  columns,
  fileName,
}: CreateScanRequestsModalProps) {
  const [requesterName, setRequesterName] = useState("");
  const [requesterEmail, setRequesterEmail] = useState("");
  const [notes, setNotes] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [result, setResult] = useState<{
    created_count: number;
    skipped_count: number;
    skipped_barcodes: string[];
  } | null>(null);

  // Find barcode and product name columns
  const barcodeColumn = useMemo(() => {
    const candidates = ["barcode", "barkod", "product_code", "sku", "code", "ean", "upc"];
    for (const candidate of candidates) {
      const found = columns.find((c) => c.toLowerCase().includes(candidate));
      if (found) return found;
    }
    return columns[0];
  }, [columns]);

  const productNameColumn = useMemo(() => {
    const candidates = ["product_name", "name", "title", "urun_adi", "ürün_adı"];
    for (const candidate of candidates) {
      const found = columns.find((c) => c.toLowerCase().includes(candidate));
      if (found) return found;
    }
    return null;
  }, [columns]);

  const brandColumn = useMemo(() => {
    const candidates = ["brand", "marka", "brand_name"];
    for (const candidate of candidates) {
      const found = columns.find((c) => c.toLowerCase().includes(candidate));
      if (found) return found;
    }
    return null;
  }, [columns]);

  const handleSubmit = async () => {
    if (!requesterName || !requesterEmail || items.length === 0) return;

    setIsSubmitting(true);
    setResult(null);

    try {
      const scanRequestItems = items.map((item) => ({
        barcode: String(item.source_row[barcodeColumn] || ""),
        product_name: productNameColumn
          ? String(item.source_row[productNameColumn] || "")
          : undefined,
        brand_name: brandColumn
          ? String(item.source_row[brandColumn] || "")
          : undefined,
      })).filter((item) => item.barcode);

      const response = await apiClient.createBulkScanRequests({
        items: scanRequestItems,
        requester_name: requesterName,
        requester_email: requesterEmail,
        source_file: fileName,
        notes: notes || undefined,
      });

      setResult(response);

      if (response.created_count > 0) {
        toast.success(`Created ${response.created_count} scan requests`);
      }
      if (response.skipped_count > 0) {
        toast.warning(`${response.skipped_count} duplicates skipped`);
      }
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : "Failed to create scan requests"
      );
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleClose = () => {
    setResult(null);
    onOpenChange(false);
  };

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>Create Scan Requests</DialogTitle>
          <DialogDescription>
            Create {items.length} scan request{items.length !== 1 ? "s" : ""} for
            unmatched products.
          </DialogDescription>
        </DialogHeader>

        {result ? (
          <div className="space-y-4 py-4">
            <Alert
              variant={result.created_count > 0 ? "default" : "destructive"}
            >
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>
                {result.created_count > 0 ? (
                  <>
                    Successfully created {result.created_count} scan requests.
                    {result.skipped_count > 0 && (
                      <> {result.skipped_count} duplicates were skipped.</>
                    )}
                  </>
                ) : (
                  <>All items were skipped as duplicates.</>
                )}
              </AlertDescription>
            </Alert>

            {result.skipped_barcodes.length > 0 && (
              <div className="space-y-2">
                <Label>Skipped Barcodes (already have pending requests)</Label>
                <div className="max-h-[150px] overflow-auto rounded-md border p-2">
                  <code className="text-xs">
                    {result.skipped_barcodes.join(", ")}
                  </code>
                </div>
              </div>
            )}

            <DialogFooter>
              <Button onClick={handleClose}>Close</Button>
            </DialogFooter>
          </div>
        ) : (
          <>
            <div className="space-y-4 py-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="requester-name">Your Name *</Label>
                  <Input
                    id="requester-name"
                    value={requesterName}
                    onChange={(e) => setRequesterName(e.target.value)}
                    placeholder="John Doe"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="requester-email">Your Email *</Label>
                  <Input
                    id="requester-email"
                    type="email"
                    value={requesterEmail}
                    onChange={(e) => setRequesterEmail(e.target.value)}
                    placeholder="john@example.com"
                  />
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="notes">Notes (optional)</Label>
                <Textarea
                  id="notes"
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                  placeholder="Additional context for these scan requests..."
                  rows={2}
                />
              </div>

              <div className="space-y-2">
                <Label>Preview (showing 5 of {items.length})</Label>
                <div className="rounded-md border overflow-auto max-h-[200px]">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>{barcodeColumn}</TableHead>
                        {brandColumn && <TableHead>{brandColumn}</TableHead>}
                        {productNameColumn && (
                          <TableHead>{productNameColumn}</TableHead>
                        )}
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {items.slice(0, 5).map((item, index) => (
                        <TableRow key={index}>
                          <TableCell className="font-mono text-sm">
                            {String(item.source_row[barcodeColumn] || "-")}
                          </TableCell>
                          {brandColumn && (
                            <TableCell>
                              {String(item.source_row[brandColumn] || "-")}
                            </TableCell>
                          )}
                          {productNameColumn && (
                            <TableCell className="max-w-[200px] truncate">
                              {String(item.source_row[productNameColumn] || "-")}
                            </TableCell>
                          )}
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              </div>

              <Alert>
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>
                  Existing pending/in-progress requests with the same barcode will
                  be skipped to avoid duplicates.
                </AlertDescription>
              </Alert>
            </div>

            <DialogFooter>
              <Button variant="outline" onClick={handleClose}>
                Cancel
              </Button>
              <Button
                onClick={handleSubmit}
                disabled={
                  !requesterName || !requesterEmail || items.length === 0 || isSubmitting
                }
              >
                {isSubmitting && (
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                )}
                Create {items.length} Request{items.length !== 1 ? "s" : ""}
              </Button>
            </DialogFooter>
          </>
        )}
      </DialogContent>
    </Dialog>
  );
}
