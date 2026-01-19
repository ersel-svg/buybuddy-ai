"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import Link from "next/link";
import { apiClient } from "@/lib/api-client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
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
  Plus,
  MoreHorizontal,
  Loader2,
  Package,
  CheckCircle2,
  Clock,
  XCircle,
  PlayCircle,
} from "lucide-react";
import type { ScanRequest, ScanRequestStatus } from "@/types";

const statusConfig: Record<
  ScanRequestStatus,
  { label: string; color: string; icon: React.ReactNode }
> = {
  pending: {
    label: "Pending",
    color: "bg-yellow-100 text-yellow-800",
    icon: <Clock className="h-3 w-3" />,
  },
  in_progress: {
    label: "In Progress",
    color: "bg-blue-100 text-blue-800",
    icon: <PlayCircle className="h-3 w-3" />,
  },
  completed: {
    label: "Completed",
    color: "bg-green-100 text-green-800",
    icon: <CheckCircle2 className="h-3 w-3" />,
  },
  cancelled: {
    label: "Cancelled",
    color: "bg-gray-100 text-gray-800",
    icon: <XCircle className="h-3 w-3" />,
  },
};

export default function ScanRequestsPage() {
  const queryClient = useQueryClient();
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState<string>("all");
  const [page, setPage] = useState(1);
  const [cancelDialogOpen, setCancelDialogOpen] = useState(false);
  const [selectedRequest, setSelectedRequest] = useState<ScanRequest | null>(null);

  const limit = 20;

  // Fetch scan requests
  const { data, isLoading, isFetching } = useQuery({
    queryKey: ["scan-requests", { page, search, status: statusFilter }],
    queryFn: () =>
      apiClient.getScanRequests({
        page,
        limit,
        search: search || undefined,
        status: statusFilter !== "all" ? statusFilter : undefined,
      }),
  });

  // Cancel mutation
  const cancelMutation = useMutation({
    mutationFn: (id: string) => apiClient.deleteScanRequest(id),
    onSuccess: () => {
      toast.success("Request cancelled");
      queryClient.invalidateQueries({ queryKey: ["scan-requests"] });
      setCancelDialogOpen(false);
    },
    onError: () => {
      toast.error("Failed to cancel request");
    },
  });

  // Update status mutation
  const updateStatusMutation = useMutation({
    mutationFn: ({ id, status }: { id: string; status: string }) =>
      apiClient.updateScanRequest(id, { status }),
    onSuccess: () => {
      toast.success("Status updated");
      queryClient.invalidateQueries({ queryKey: ["scan-requests"] });
    },
    onError: () => {
      toast.error("Failed to update status");
    },
  });

  const handleCancel = (request: ScanRequest) => {
    setSelectedRequest(request);
    setCancelDialogOpen(true);
  };

  const handleStatusChange = (request: ScanRequest, newStatus: string) => {
    updateStatusMutation.mutate({ id: request.id, status: newStatus });
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Scan Requests</h1>
          <p className="text-gray-500">
            Manage product scan requests
          </p>
        </div>
        <Link href="/scan-requests/new">
          <Button>
            <Plus className="mr-2 h-4 w-4" />
            New Request
          </Button>
        </Link>
      </div>

      {/* Filters */}
      <div className="flex gap-4">
        <div className="relative flex-1 max-w-sm">
          <Search className="absolute left-3 top-2.5 h-4 w-4 text-gray-400" />
          <Input
            placeholder="Search by barcode or product name..."
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
            <SelectValue placeholder="Filter by status" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Status</SelectItem>
            <SelectItem value="pending">Pending</SelectItem>
            <SelectItem value="in_progress">In Progress</SelectItem>
            <SelectItem value="completed">Completed</SelectItem>
            <SelectItem value="cancelled">Cancelled</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* Table */}
      <div className="border rounded-lg">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Barcode</TableHead>
              <TableHead>Product</TableHead>
              <TableHead>Brand</TableHead>
              <TableHead>Requester</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>Created</TableHead>
              <TableHead className="w-10"></TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {isLoading ? (
              <TableRow>
                <TableCell colSpan={7} className="text-center py-8">
                  <Loader2 className="h-6 w-6 animate-spin mx-auto text-gray-400" />
                </TableCell>
              </TableRow>
            ) : !data?.items?.length ? (
              <TableRow>
                <TableCell colSpan={7} className="text-center py-8">
                  <Package className="h-12 w-12 mx-auto text-gray-300 mb-2" />
                  <p className="text-gray-500">No scan requests found</p>
                  <Link href="/scan-requests/new">
                    <Button variant="link" className="mt-2">
                      Create your first request
                    </Button>
                  </Link>
                </TableCell>
              </TableRow>
            ) : (
              data.items.map((request) => (
                <TableRow
                  key={request.id}
                  className="cursor-pointer hover:bg-gray-50"
                  onClick={() => window.location.href = `/scan-requests/${request.id}`}
                >
                  <TableCell className="font-mono">{request.barcode}</TableCell>
                  <TableCell>{request.product_name || "-"}</TableCell>
                  <TableCell>{request.brand_name || "-"}</TableCell>
                  <TableCell>
                    <div>
                      <div className="font-medium">{request.requester_name}</div>
                      <div className="text-sm text-gray-500">
                        {request.requester_email}
                      </div>
                    </div>
                  </TableCell>
                  <TableCell>
                    <Badge
                      className={`${statusConfig[request.status].color} flex items-center gap-1 w-fit`}
                    >
                      {statusConfig[request.status].icon}
                      {statusConfig[request.status].label}
                    </Badge>
                  </TableCell>
                  <TableCell className="text-sm text-gray-500">
                    {formatDate(request.created_at)}
                  </TableCell>
                  <TableCell onClick={(e) => e.stopPropagation()}>
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button variant="ghost" size="icon">
                          <MoreHorizontal className="h-4 w-4" />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end">
                        {request.status === "pending" && (
                          <DropdownMenuItem
                            onClick={() => handleStatusChange(request, "in_progress")}
                          >
                            Mark In Progress
                          </DropdownMenuItem>
                        )}
                        {request.status === "in_progress" && (
                          <DropdownMenuItem
                            onClick={() => handleStatusChange(request, "completed")}
                          >
                            Mark Completed
                          </DropdownMenuItem>
                        )}
                        {request.completed_by_product_id && (
                          <DropdownMenuItem asChild>
                            <Link href={`/products/${request.completed_by_product_id}`}>
                              View Product
                            </Link>
                          </DropdownMenuItem>
                        )}
                        {(request.status === "pending" || request.status === "in_progress") && (
                          <DropdownMenuItem
                            className="text-red-600"
                            onClick={() => handleCancel(request)}
                          >
                            Cancel Request
                          </DropdownMenuItem>
                        )}
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
      {data && data.total > limit && (
        <div className="flex items-center justify-between">
          <p className="text-sm text-gray-500">
            Showing {(page - 1) * limit + 1} to{" "}
            {Math.min(page * limit, data.total)} of {data.total} requests
          </p>
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={page === 1 || isFetching}
            >
              Previous
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setPage((p) => p + 1)}
              disabled={!data.has_more || isFetching}
            >
              Next
            </Button>
          </div>
        </div>
      )}

      {/* Cancel Dialog */}
      <AlertDialog open={cancelDialogOpen} onOpenChange={setCancelDialogOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Cancel Scan Request</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to cancel this scan request for barcode{" "}
              <span className="font-mono font-medium">
                {selectedRequest?.barcode}
              </span>
              ? This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Keep Request</AlertDialogCancel>
            <AlertDialogAction
              className="bg-red-600 hover:bg-red-700"
              onClick={() => selectedRequest && cancelMutation.mutate(selectedRequest.id)}
            >
              {cancelMutation.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                "Cancel Request"
              )}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
