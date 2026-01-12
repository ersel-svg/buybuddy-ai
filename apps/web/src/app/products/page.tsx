"use client";

import { useState, useMemo } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import Link from "next/link";
import { apiClient } from "@/lib/api-client";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import {
  Search,
  MoreHorizontal,
  Download,
  FileJson,
  FileSpreadsheet,
  Trash2,
  FolderPlus,
  Loader2,
  Package,
  RefreshCw,
} from "lucide-react";
import type { ProductStatus } from "@/types";

export default function ProductsPage() {
  const queryClient = useQueryClient();
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(1);
  const [statusFilter, setStatusFilter] = useState<string>("all");
  const [categoryFilter, setCategoryFilter] = useState<string>("all");
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [isDownloading, setIsDownloading] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);

  // Fetch products
  const { data, isLoading, refetch, isFetching } = useQuery({
    queryKey: ["products", { page, search, statusFilter, categoryFilter }],
    queryFn: () =>
      apiClient.getProducts({
        page,
        limit: 20,
        search: search || undefined,
        status: statusFilter !== "all" ? statusFilter : undefined,
        category: categoryFilter !== "all" ? categoryFilter : undefined,
      }),
  });

  // Fetch categories
  const { data: categories } = useQuery({
    queryKey: ["product-categories"],
    queryFn: () => apiClient.getProductCategories(),
  });

  // Bulk selection helpers
  const allSelected = useMemo(() => {
    if (!data?.items.length) return false;
    return data.items.every((p) => selectedIds.has(p.id));
  }, [data?.items, selectedIds]);

  const someSelected = useMemo(() => {
    if (!data?.items.length) return false;
    return data.items.some((p) => selectedIds.has(p.id)) && !allSelected;
  }, [data?.items, selectedIds, allSelected]);

  const toggleAll = () => {
    if (allSelected) {
      setSelectedIds(new Set());
    } else {
      setSelectedIds(new Set(data?.items.map((p) => p.id) || []));
    }
  };

  const toggleOne = (id: string) => {
    const newSet = new Set(selectedIds);
    if (newSet.has(id)) {
      newSet.delete(id);
    } else {
      newSet.add(id);
    }
    setSelectedIds(newSet);
  };

  // Download selected products
  const handleDownloadSelected = async () => {
    if (selectedIds.size === 0) return;

    setIsDownloading(true);
    try {
      const blob = await apiClient.downloadProducts(Array.from(selectedIds));
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `products_${Date.now()}.zip`;
      a.click();
      URL.revokeObjectURL(url);

      toast.success(`${selectedIds.size} products downloading...`);
    } catch (error) {
      toast.error("Download failed");
    } finally {
      setIsDownloading(false);
    }
  };

  // Download ALL products
  const handleDownloadAll = async () => {
    setIsDownloading(true);
    try {
      const blob = await apiClient.downloadAllProducts({
        status: statusFilter !== "all" ? statusFilter : undefined,
        category: categoryFilter !== "all" ? categoryFilter : undefined,
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `all_products_${Date.now()}.zip`;
      a.click();
      URL.revokeObjectURL(url);

      toast.success("All products downloading...");
    } catch (error) {
      toast.error("Download failed");
    } finally {
      setIsDownloading(false);
    }
  };

  // Export to CSV
  const handleExportCSV = async () => {
    try {
      const ids = selectedIds.size > 0 ? Array.from(selectedIds) : undefined;
      const blob = await apiClient.exportProductsCSV(ids);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `products_${Date.now()}.csv`;
      a.click();
      URL.revokeObjectURL(url);
      toast.success("CSV exported");
    } catch (error) {
      toast.error("Export failed");
    }
  };

  // Export to JSON
  const handleExportJSON = async () => {
    try {
      const ids = selectedIds.size > 0 ? Array.from(selectedIds) : undefined;
      const blob = await apiClient.exportProductsJSON(ids);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `products_${Date.now()}.json`;
      a.click();
      URL.revokeObjectURL(url);
      toast.success("JSON exported");
    } catch (error) {
      toast.error("Export failed");
    }
  };

  // Bulk delete mutation
  const deleteMutation = useMutation({
    mutationFn: (ids: string[]) => apiClient.deleteProducts(ids),
    onSuccess: (result) => {
      toast.success(`${result.deleted_count} products deleted`);
      setSelectedIds(new Set());
      setDeleteDialogOpen(false);
      queryClient.invalidateQueries({ queryKey: ["products"] });
    },
    onError: () => {
      toast.error("Delete failed");
    },
  });

  // Status badge colors
  const statusColors: Record<ProductStatus, string> = {
    pending: "bg-yellow-100 text-yellow-800 hover:bg-yellow-100",
    processing: "bg-blue-100 text-blue-800 hover:bg-blue-100",
    needs_matching: "bg-purple-100 text-purple-800 hover:bg-purple-100",
    ready: "bg-green-100 text-green-800 hover:bg-green-100",
    rejected: "bg-red-100 text-red-800 hover:bg-red-100",
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold">Products</h1>
          <p className="text-gray-500">
            {data?.total || 0} products in directory
          </p>
        </div>

        {/* Action Buttons */}
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="icon"
            onClick={() => refetch()}
            disabled={isFetching}
          >
            <RefreshCw
              className={`h-4 w-4 ${isFetching ? "animate-spin" : ""}`}
            />
          </Button>

          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" disabled={isDownloading}>
                {isDownloading ? (
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                ) : (
                  <Download className="h-4 w-4 mr-2" />
                )}
                Download
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem
                onClick={handleDownloadSelected}
                disabled={selectedIds.size === 0}
              >
                <Package className="h-4 w-4 mr-2" />
                Download Selected ({selectedIds.size})
              </DropdownMenuItem>
              <DropdownMenuItem onClick={handleDownloadAll}>
                <Download className="h-4 w-4 mr-2" />
                Download All Products (ZIP)
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={handleExportCSV}>
                <FileSpreadsheet className="h-4 w-4 mr-2" />
                Export to CSV
              </DropdownMenuItem>
              <DropdownMenuItem onClick={handleExportJSON}>
                <FileJson className="h-4 w-4 mr-2" />
                Export to JSON
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>

      {/* Search & Filters */}
      <div className="flex gap-4 flex-wrap">
        <div className="relative flex-1 max-w-sm">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
          <Input
            placeholder="Search by barcode or name..."
            value={search}
            onChange={(e) => {
              setSearch(e.target.value);
              setPage(1);
            }}
            className="pl-10"
          />
        </div>

        <Select
          value={statusFilter}
          onValueChange={(value) => {
            setStatusFilter(value);
            setPage(1);
          }}
        >
          <SelectTrigger className="w-40">
            <SelectValue placeholder="Status" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Status</SelectItem>
            <SelectItem value="pending">Pending</SelectItem>
            <SelectItem value="processing">Processing</SelectItem>
            <SelectItem value="needs_matching">Needs Matching</SelectItem>
            <SelectItem value="ready">Ready</SelectItem>
            <SelectItem value="rejected">Rejected</SelectItem>
          </SelectContent>
        </Select>

        <Select
          value={categoryFilter}
          onValueChange={(value) => {
            setCategoryFilter(value);
            setPage(1);
          }}
        >
          <SelectTrigger className="w-40">
            <SelectValue placeholder="Category" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Categories</SelectItem>
            {categories?.map((cat: string) => (
              <SelectItem key={cat} value={cat}>
                {cat}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {/* Bulk Actions Bar */}
      {selectedIds.size > 0 && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 flex items-center justify-between">
          <span className="text-sm text-blue-700">
            {selectedIds.size} product{selectedIds.size > 1 ? "s" : ""} selected
          </span>
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={handleDownloadSelected}
              disabled={isDownloading}
            >
              <Download className="h-4 w-4 mr-1" />
              Download
            </Button>
            <Button variant="outline" size="sm">
              <FolderPlus className="h-4 w-4 mr-1" />
              Add to Dataset
            </Button>
            <Button
              variant="outline"
              size="sm"
              className="text-red-600 hover:text-red-700 hover:bg-red-50"
              onClick={() => setDeleteDialogOpen(true)}
            >
              <Trash2 className="h-4 w-4 mr-1" />
              Delete
            </Button>
          </div>
        </div>
      )}

      {/* Table */}
      <div className="border rounded-lg bg-white">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-12">
                <Checkbox
                  checked={allSelected ? true : someSelected ? "indeterminate" : false}
                  onCheckedChange={toggleAll}
                />
              </TableHead>
              <TableHead>Image</TableHead>
              <TableHead>Barcode</TableHead>
              <TableHead>Brand</TableHead>
              <TableHead>Product</TableHead>
              <TableHead>Category</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>Frames</TableHead>
              <TableHead className="w-12"></TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {isLoading ? (
              <TableRow>
                <TableCell colSpan={9} className="text-center py-8">
                  <Loader2 className="h-6 w-6 animate-spin mx-auto" />
                </TableCell>
              </TableRow>
            ) : data?.items.length === 0 ? (
              <TableRow>
                <TableCell colSpan={9} className="text-center py-8">
                  <div className="text-gray-500">
                    <Package className="h-12 w-12 mx-auto mb-2 opacity-50" />
                    <p>No products found</p>
                  </div>
                </TableCell>
              </TableRow>
            ) : (
              data?.items.map((product) => (
                <TableRow
                  key={product.id}
                  className={selectedIds.has(product.id) ? "bg-blue-50" : ""}
                >
                  <TableCell>
                    <Checkbox
                      checked={selectedIds.has(product.id)}
                      onCheckedChange={() => toggleOne(product.id)}
                    />
                  </TableCell>
                  <TableCell>
                    {product.primary_image_url ? (
                      <img
                        src={product.primary_image_url}
                        alt=""
                        className="w-10 h-10 object-cover rounded"
                      />
                    ) : (
                      <div className="w-10 h-10 bg-gray-100 rounded flex items-center justify-center">
                        <Package className="h-5 w-5 text-gray-400" />
                      </div>
                    )}
                  </TableCell>
                  <TableCell className="font-mono text-sm">
                    {product.barcode}
                  </TableCell>
                  <TableCell>{product.brand_name || "-"}</TableCell>
                  <TableCell className="max-w-[200px] truncate">
                    {product.product_name || "-"}
                  </TableCell>
                  <TableCell>{product.category || "-"}</TableCell>
                  <TableCell>
                    <Badge className={statusColors[product.status]}>
                      {product.status.replace("_", " ")}
                    </Badge>
                  </TableCell>
                  <TableCell>{product.frame_count}</TableCell>
                  <TableCell>
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button variant="ghost" size="icon">
                          <MoreHorizontal className="h-4 w-4" />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end">
                        <DropdownMenuItem asChild>
                          <Link href={`/products/${product.id}`}>
                            View Details
                          </Link>
                        </DropdownMenuItem>
                        <DropdownMenuItem asChild>
                          <Link href={`/products/${product.id}?edit=true`}>
                            Edit
                          </Link>
                        </DropdownMenuItem>
                        <DropdownMenuItem
                          onClick={async () => {
                            try {
                              const blob = await apiClient.downloadProduct(
                                product.id
                              );
                              const url = URL.createObjectURL(blob);
                              const a = document.createElement("a");
                              a.href = url;
                              a.download = `product_${product.barcode}.zip`;
                              a.click();
                              URL.revokeObjectURL(url);
                              toast.success("Download started");
                            } catch {
                              toast.error("Download failed");
                            }
                          }}
                        >
                          <Download className="h-4 w-4 mr-2" />
                          Download Frames
                        </DropdownMenuItem>
                        <DropdownMenuSeparator />
                        <DropdownMenuItem
                          className="text-red-600"
                          onClick={() => {
                            setSelectedIds(new Set([product.id]));
                            setDeleteDialogOpen(true);
                          }}
                        >
                          <Trash2 className="h-4 w-4 mr-2" />
                          Delete
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </div>

      {/* Pagination */}
      <div className="flex justify-between items-center">
        <p className="text-sm text-gray-500">
          Showing {data?.items.length ? (page - 1) * 20 + 1 : 0} to{" "}
          {Math.min(page * 20, data?.total || 0)} of {data?.total || 0}
        </p>
        <div className="flex gap-2">
          <Button
            variant="outline"
            disabled={page === 1}
            onClick={() => setPage(page - 1)}
          >
            Previous
          </Button>
          <Button
            variant="outline"
            disabled={!data || page * 20 >= data.total}
            onClick={() => setPage(page + 1)}
          >
            Next
          </Button>
        </div>
      </div>

      {/* Delete Confirmation Dialog */}
      <AlertDialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Products</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete {selectedIds.size} product
              {selectedIds.size > 1 ? "s" : ""}? This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              className="bg-red-600 hover:bg-red-700"
              onClick={() => deleteMutation.mutate(Array.from(selectedIds))}
              disabled={deleteMutation.isPending}
            >
              {deleteMutation.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin mr-2" />
              ) : null}
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
