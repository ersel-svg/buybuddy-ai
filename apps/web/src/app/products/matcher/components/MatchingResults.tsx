"use client";

import { useState, useMemo } from "react";
import {
  Check,
  X,
  Download,
  Plus,
  ClipboardList,
  ExternalLink,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Checkbox } from "@/components/ui/checkbox";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { AddToDatasetModal } from "./AddToDatasetModal";
import { CreateScanRequestsModal } from "./CreateScanRequestsModal";
import { apiClient } from "@/lib/api-client";
import { toast } from "sonner";
import Link from "next/link";

interface MatchedItem {
  source_row: Record<string, unknown>;
  product: {
    id: string;
    barcode: string;
    product_name?: string;
    brand_name?: string;
    category?: string;
    status: string;
  };
  matched_by: string;
}

interface UnmatchedItem {
  source_row: Record<string, unknown>;
}

interface MatchResponse {
  matched: MatchedItem[];
  unmatched: UnmatchedItem[];
  summary: {
    total: number;
    matched_count: number;
    unmatched_count: number;
    match_rate: number;
  };
}

interface MatchingResultsProps {
  results: MatchResponse;
  fileName: string;
  columns: string[];
  onReset: () => void;
}

export function MatchingResults({
  results,
  fileName,
  columns,
  onReset,
}: MatchingResultsProps) {
  const [selectedMatched, setSelectedMatched] = useState<Set<string>>(
    new Set(results.matched.map((_, i) => String(i)))
  );
  const [selectedUnmatched, setSelectedUnmatched] = useState<Set<string>>(
    new Set(results.unmatched.map((_, i) => String(i)))
  );
  const [showDatasetModal, setShowDatasetModal] = useState(false);
  const [showScanRequestModal, setShowScanRequestModal] = useState(false);
  const [isExporting, setIsExporting] = useState(false);

  const toggleMatchedItem = (index: string) => {
    setSelectedMatched((prev) => {
      const next = new Set(prev);
      if (next.has(index)) {
        next.delete(index);
      } else {
        next.add(index);
      }
      return next;
    });
  };

  const toggleUnmatchedItem = (index: string) => {
    setSelectedUnmatched((prev) => {
      const next = new Set(prev);
      if (next.has(index)) {
        next.delete(index);
      } else {
        next.add(index);
      }
      return next;
    });
  };

  const toggleAllMatched = () => {
    if (selectedMatched.size === results.matched.length) {
      setSelectedMatched(new Set());
    } else {
      setSelectedMatched(new Set(results.matched.map((_, i) => String(i))));
    }
  };

  const toggleAllUnmatched = () => {
    if (selectedUnmatched.size === results.unmatched.length) {
      setSelectedUnmatched(new Set());
    } else {
      setSelectedUnmatched(new Set(results.unmatched.map((_, i) => String(i))));
    }
  };

  const selectedMatchedItems = useMemo(
    () =>
      results.matched.filter((_, i) => selectedMatched.has(String(i))),
    [results.matched, selectedMatched]
  );

  const selectedUnmatchedItems = useMemo(
    () =>
      results.unmatched.filter((_, i) => selectedUnmatched.has(String(i))),
    [results.unmatched, selectedUnmatched]
  );

  const handleExportMatched = async () => {
    if (selectedMatchedItems.length === 0) return;
    setIsExporting(true);
    try {
      const blob = await apiClient.exportMatchedProducts(
        selectedMatchedItems,
        columns
      );
      downloadBlob(blob, `matched_products_${selectedMatchedItems.length}.csv`);
      toast.success("Export completed");
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : "Export failed"
      );
    } finally {
      setIsExporting(false);
    }
  };

  const handleExportUnmatched = async () => {
    if (selectedUnmatchedItems.length === 0) return;
    setIsExporting(true);
    try {
      const blob = await apiClient.exportUnmatchedProducts(
        selectedUnmatchedItems,
        columns
      );
      downloadBlob(
        blob,
        `unmatched_products_${selectedUnmatchedItems.length}.csv`
      );
      toast.success("Export completed");
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : "Export failed"
      );
    } finally {
      setIsExporting(false);
    }
  };

  // Find the best column to display for source identification
  const identifierColumn = useMemo(() => {
    const preferredColumns = [
      "barcode",
      "barkod",
      "product_code",
      "sku",
      "code",
      "id",
    ];
    for (const col of preferredColumns) {
      const found = columns.find((c) => c.toLowerCase().includes(col));
      if (found) return found;
    }
    return columns[0];
  }, [columns]);

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-xl font-semibold">Matching Results</h2>
        <p className="text-muted-foreground mt-1">{fileName}</p>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-2 gap-4">
        <Card className="border-green-200 dark:border-green-900">
          <CardContent className="pt-6">
            <div className="flex items-center gap-4">
              <div className="h-12 w-12 rounded-full bg-green-100 dark:bg-green-900/30 flex items-center justify-center">
                <Check className="h-6 w-6 text-green-600 dark:text-green-400" />
              </div>
              <div>
                <p className="text-2xl font-bold text-green-600 dark:text-green-400">
                  {results.summary.matched_count}
                </p>
                <p className="text-sm text-muted-foreground">
                  Matched ({results.summary.match_rate}%)
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="border-red-200 dark:border-red-900">
          <CardContent className="pt-6">
            <div className="flex items-center gap-4">
              <div className="h-12 w-12 rounded-full bg-red-100 dark:bg-red-900/30 flex items-center justify-center">
                <X className="h-6 w-6 text-red-600 dark:text-red-400" />
              </div>
              <div>
                <p className="text-2xl font-bold text-red-600 dark:text-red-400">
                  {results.summary.unmatched_count}
                </p>
                <p className="text-sm text-muted-foreground">
                  Unmatched ({(100 - results.summary.match_rate).toFixed(1)}%)
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Results Tabs */}
      <Tabs defaultValue="matched" className="space-y-4">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="matched" className="gap-2">
            <Check className="h-4 w-4" />
            Matched ({results.summary.matched_count})
          </TabsTrigger>
          <TabsTrigger value="unmatched" className="gap-2">
            <X className="h-4 w-4" />
            Unmatched ({results.summary.unmatched_count})
          </TabsTrigger>
        </TabsList>

        {/* Matched Tab */}
        <TabsContent value="matched" className="space-y-4">
          {results.matched.length > 0 ? (
            <>
              <Card>
                <CardContent className="pt-4">
                  <div className="rounded-md border overflow-auto max-h-[400px]">
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead className="w-12">
                            <Checkbox
                              checked={
                                selectedMatched.size === results.matched.length
                              }
                              onCheckedChange={toggleAllMatched}
                            />
                          </TableHead>
                          <TableHead>Source Value</TableHead>
                          <TableHead>Matched By</TableHead>
                          <TableHead>Product</TableHead>
                          <TableHead>Brand</TableHead>
                          <TableHead className="w-12"></TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {results.matched.map((item, index) => (
                          <TableRow key={index}>
                            <TableCell>
                              <Checkbox
                                checked={selectedMatched.has(String(index))}
                                onCheckedChange={() =>
                                  toggleMatchedItem(String(index))
                                }
                              />
                            </TableCell>
                            <TableCell className="font-mono text-sm">
                              {String(item.source_row[identifierColumn] || "-")}
                            </TableCell>
                            <TableCell>
                              <Badge variant="secondary">{item.matched_by}</Badge>
                            </TableCell>
                            <TableCell className="max-w-[200px] truncate">
                              {item.product.product_name || "-"}
                            </TableCell>
                            <TableCell>
                              {item.product.brand_name || "-"}
                            </TableCell>
                            <TableCell>
                              <Link
                                href={`/products/${item.product.id}`}
                                target="_blank"
                              >
                                <Button variant="ghost" size="icon">
                                  <ExternalLink className="h-4 w-4" />
                                </Button>
                              </Link>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                </CardContent>
              </Card>

              {/* Matched Actions */}
              <div className="flex items-center gap-3">
                <span className="text-sm text-muted-foreground">
                  {selectedMatched.size} selected
                </span>
                <div className="flex-1" />
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleExportMatched}
                  disabled={selectedMatched.size === 0 || isExporting}
                >
                  <Download className="h-4 w-4 mr-2" />
                  Download CSV
                </Button>
                <Button
                  size="sm"
                  onClick={() => setShowDatasetModal(true)}
                  disabled={selectedMatched.size === 0}
                >
                  <Plus className="h-4 w-4 mr-2" />
                  Add to Dataset
                </Button>
              </div>
            </>
          ) : (
            <Card>
              <CardContent className="pt-6 text-center text-muted-foreground">
                No matched products found.
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* Unmatched Tab */}
        <TabsContent value="unmatched" className="space-y-4">
          {results.unmatched.length > 0 ? (
            <>
              <Card>
                <CardContent className="pt-4">
                  <div className="rounded-md border overflow-auto max-h-[400px]">
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead className="w-12">
                            <Checkbox
                              checked={
                                selectedUnmatched.size ===
                                results.unmatched.length
                              }
                              onCheckedChange={toggleAllUnmatched}
                            />
                          </TableHead>
                          {columns.slice(0, 4).map((col) => (
                            <TableHead key={col}>{col}</TableHead>
                          ))}
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {results.unmatched.map((item, index) => (
                          <TableRow key={index}>
                            <TableCell>
                              <Checkbox
                                checked={selectedUnmatched.has(String(index))}
                                onCheckedChange={() =>
                                  toggleUnmatchedItem(String(index))
                                }
                              />
                            </TableCell>
                            {columns.slice(0, 4).map((col) => (
                              <TableCell
                                key={col}
                                className="max-w-[150px] truncate"
                              >
                                {String(item.source_row[col] || "-")}
                              </TableCell>
                            ))}
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                </CardContent>
              </Card>

              {/* Unmatched Actions */}
              <div className="flex items-center gap-3">
                <span className="text-sm text-muted-foreground">
                  {selectedUnmatched.size} selected
                </span>
                <div className="flex-1" />
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleExportUnmatched}
                  disabled={selectedUnmatched.size === 0 || isExporting}
                >
                  <Download className="h-4 w-4 mr-2" />
                  Download CSV
                </Button>
                <Button
                  size="sm"
                  onClick={() => setShowScanRequestModal(true)}
                  disabled={selectedUnmatched.size === 0}
                >
                  <ClipboardList className="h-4 w-4 mr-2" />
                  Create Scan Requests
                </Button>
              </div>
            </>
          ) : (
            <Card>
              <CardContent className="pt-6 text-center text-muted-foreground">
                All products were matched!
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>

      {/* Bottom Actions */}
      <div className="flex justify-between">
        <Button variant="outline" onClick={onReset}>
          New Match
        </Button>
        <Button variant="secondary" asChild>
          <Link href="/products">Back to Products</Link>
        </Button>
      </div>

      {/* Modals */}
      <AddToDatasetModal
        open={showDatasetModal}
        onOpenChange={setShowDatasetModal}
        productIds={selectedMatchedItems.map((item) => item.product.id)}
      />

      <CreateScanRequestsModal
        open={showScanRequestModal}
        onOpenChange={setShowScanRequestModal}
        items={selectedUnmatchedItems}
        columns={columns}
        fileName={fileName}
      />
    </div>
  );
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
