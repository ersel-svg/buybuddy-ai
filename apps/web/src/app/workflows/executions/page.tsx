"use client";

import { useState, Suspense } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useSearchParams } from "next/navigation";
import { toast } from "sonner";
import Link from "next/link";
import { apiClient } from "@/lib/api-client";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  RefreshCw,
  Loader2,
  Search,
  Play,
  Clock,
  CheckCircle,
  XCircle,
  StopCircle,
  Eye,
  MoreHorizontal,
  ChevronLeft,
  ChevronRight,
  ChevronsLeft,
  ChevronsRight,
} from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

const statusConfig: Record<
  string,
  { icon: React.ReactNode; label: string; color: string }
> = {
  pending: {
    icon: <Clock className="h-3 w-3" />,
    label: "Pending",
    color: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200",
  },
  running: {
    icon: <Loader2 className="h-3 w-3 animate-spin" />,
    label: "Running",
    color: "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200",
  },
  completed: {
    icon: <CheckCircle className="h-3 w-3" />,
    label: "Completed",
    color: "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200",
  },
  failed: {
    icon: <XCircle className="h-3 w-3" />,
    label: "Failed",
    color: "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200",
  },
  cancelled: {
    icon: <StopCircle className="h-3 w-3" />,
    label: "Cancelled",
    color: "bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200",
  },
};

function ExecutionsPageContent() {
  const searchParams = useSearchParams();
  const queryClient = useQueryClient();

  // State
  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState<string>("all");
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize] = useState(20);
  const [selectedExecution, setSelectedExecution] = useState<string | null>(
    searchParams.get("execution_id")
  );

  // Fetch executions
  const { data, isLoading, isFetching } = useQuery({
    queryKey: ["workflow-executions", statusFilter, currentPage],
    queryFn: () =>
      apiClient.getWorkflowExecutions({
        status: statusFilter !== "all" ? statusFilter : undefined,
        page: currentPage,
        limit: pageSize,
      }),
    refetchInterval: 5000, // Refresh every 5 seconds to update running status
  });

  // Fetch selected execution details
  const { data: executionDetails, isLoading: detailsLoading } = useQuery({
    queryKey: ["workflow-execution", selectedExecution],
    queryFn: () => apiClient.getWorkflowExecution(selectedExecution!),
    enabled: !!selectedExecution,
  });

  // Cancel execution mutation
  const cancelMutation = useMutation({
    mutationFn: (executionId: string) => apiClient.cancelExecution(executionId),
    onSuccess: () => {
      toast.success("Execution cancelled");
      queryClient.invalidateQueries({ queryKey: ["workflow-executions"] });
    },
    onError: (error: Error) => {
      toast.error(`Failed to cancel: ${error.message}`);
    },
  });

  const executions = data?.items || [];
  const totalPages = Math.ceil((data?.total || 0) / pageSize);

  // Filter by search
  const filteredExecutions = executions.filter(
    (e) =>
      !searchQuery ||
      e.workflow_name?.toLowerCase().includes(searchQuery.toLowerCase()) ||
      e.id.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const formatDuration = (ms?: number) => {
    if (!ms) return "-";
    if (ms < 1000) return `${ms}ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
    return `${(ms / 60000).toFixed(1)}m`;
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Executions</h1>
          <p className="text-muted-foreground">
            Monitor and manage workflow executions
          </p>
        </div>
      </div>

      {/* Main content */}
      <Card>
        <CardHeader className="pb-4">
          <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <CardTitle>Execution History</CardTitle>
              <CardDescription>
                {data?.total || 0} execution{(data?.total || 0) !== 1 ? "s" : ""} total
              </CardDescription>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() =>
                queryClient.invalidateQueries({ queryKey: ["workflow-executions"] })
              }
              disabled={isFetching}
            >
              <RefreshCw className={`h-4 w-4 ${isFetching ? "animate-spin" : ""}`} />
            </Button>
          </div>

          <div className="flex flex-col gap-3 sm:flex-row sm:items-center mt-4">
            <div className="relative flex-1 max-w-sm">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search executions..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-9"
              />
            </div>
            <Select value={statusFilter} onValueChange={setStatusFilter}>
              <SelectTrigger className="w-[150px]">
                <SelectValue placeholder="All Statuses" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Statuses</SelectItem>
                <SelectItem value="pending">Pending</SelectItem>
                <SelectItem value="running">Running</SelectItem>
                <SelectItem value="completed">Completed</SelectItem>
                <SelectItem value="failed">Failed</SelectItem>
                <SelectItem value="cancelled">Cancelled</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardHeader>

        <CardContent>
          {isLoading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="h-6 w-6 animate-spin" />
            </div>
          ) : filteredExecutions.length === 0 ? (
            <div className="text-center py-12 text-muted-foreground">
              <Play className="h-12 w-12 mx-auto mb-3 opacity-50" />
              <p className="font-medium">No executions found</p>
              <p className="text-sm mt-1">
                {searchQuery || statusFilter !== "all"
                  ? "Try adjusting your filters"
                  : "Run a workflow to see executions here"}
              </p>
            </div>
          ) : (
            <>
              <div className="rounded-md border">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Workflow</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead>Duration</TableHead>
                      <TableHead>Started</TableHead>
                      <TableHead className="w-12"></TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {filteredExecutions.map((execution) => {
                      const status =
                        statusConfig[execution.status] || statusConfig.pending;

                      return (
                        <TableRow
                          key={execution.id}
                          className="cursor-pointer hover:bg-muted/50"
                          onClick={() => setSelectedExecution(execution.id)}
                        >
                          <TableCell>
                            <div>
                              <p className="font-medium">
                                {execution.workflow_name || "Unknown Workflow"}
                              </p>
                              <p className="text-xs text-muted-foreground font-mono">
                                {execution.id.slice(0, 8)}...
                              </p>
                            </div>
                          </TableCell>
                          <TableCell>
                            <Badge className={status.color}>
                              <span className="mr-1">{status.icon}</span>
                              {status.label}
                            </Badge>
                          </TableCell>
                          <TableCell>
                            {formatDuration(execution.total_duration_ms)}
                          </TableCell>
                          <TableCell className="text-muted-foreground">
                            {execution.started_at
                              ? new Date(execution.started_at).toLocaleString()
                              : new Date(execution.created_at).toLocaleString()}
                          </TableCell>
                          <TableCell onClick={(e) => e.stopPropagation()}>
                            <DropdownMenu>
                              <DropdownMenuTrigger asChild>
                                <Button variant="ghost" size="icon" className="h-8 w-8">
                                  <MoreHorizontal className="h-4 w-4" />
                                </Button>
                              </DropdownMenuTrigger>
                              <DropdownMenuContent align="end">
                                <DropdownMenuItem
                                  onClick={() => setSelectedExecution(execution.id)}
                                >
                                  <Eye className="h-4 w-4 mr-2" />
                                  View Details
                                </DropdownMenuItem>
                                {["pending", "running"].includes(execution.status) && (
                                  <DropdownMenuItem
                                    onClick={() => cancelMutation.mutate(execution.id)}
                                    className="text-orange-600"
                                  >
                                    <StopCircle className="h-4 w-4 mr-2" />
                                    Cancel
                                  </DropdownMenuItem>
                                )}
                              </DropdownMenuContent>
                            </DropdownMenu>
                          </TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              </div>

              {/* Pagination */}
              {totalPages > 1 && (
                <div className="flex items-center justify-between mt-4">
                  <p className="text-sm text-muted-foreground">
                    Page {currentPage} of {totalPages}
                  </p>
                  <div className="flex items-center gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setCurrentPage(1)}
                      disabled={currentPage === 1}
                    >
                      <ChevronsLeft className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                      disabled={currentPage === 1}
                    >
                      <ChevronLeft className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                      disabled={currentPage >= totalPages}
                    >
                      <ChevronRight className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setCurrentPage(totalPages)}
                      disabled={currentPage >= totalPages}
                    >
                      <ChevronsRight className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              )}
            </>
          )}
        </CardContent>
      </Card>

      {/* Execution Details Dialog */}
      <Dialog
        open={!!selectedExecution}
        onOpenChange={() => setSelectedExecution(null)}
      >
        <DialogContent className="max-w-3xl max-h-[80vh]">
          <DialogHeader>
            <DialogTitle>Execution Details</DialogTitle>
            <DialogDescription>
              {selectedExecution?.slice(0, 8)}...
            </DialogDescription>
          </DialogHeader>

          {detailsLoading ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-6 w-6 animate-spin" />
            </div>
          ) : executionDetails ? (
            <ScrollArea className="max-h-[60vh]">
              <div className="space-y-4">
                {/* Status */}
                <div className="flex items-center gap-4">
                  <div>
                    <p className="text-sm text-muted-foreground">Status</p>
                    <Badge
                      className={statusConfig[executionDetails.status]?.color}
                    >
                      <span className="mr-1">
                        {statusConfig[executionDetails.status]?.icon}
                      </span>
                      {statusConfig[executionDetails.status]?.label}
                    </Badge>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Duration</p>
                    <p className="font-medium">
                      {formatDuration(executionDetails.total_duration_ms)}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Workflow</p>
                    <Link
                      href={`/workflows/${executionDetails.workflow_id}`}
                      className="font-medium text-primary hover:underline"
                    >
                      {executionDetails.workflow_name || "View Workflow"}
                    </Link>
                  </div>
                </div>

                {/* Error message */}
                {executionDetails.error_message && (
                  <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded-lg border border-red-200 dark:border-red-800">
                    <p className="text-sm font-medium text-red-800 dark:text-red-200">
                      Error
                    </p>
                    <p className="text-sm text-red-700 dark:text-red-300 mt-1">
                      {executionDetails.error_message}
                    </p>
                  </div>
                )}

                {/* Inputs */}
                <div>
                  <p className="text-sm font-medium mb-2">Inputs</p>
                  <pre className="p-3 bg-muted rounded-lg text-xs overflow-x-auto">
                    {JSON.stringify(executionDetails.inputs, null, 2)}
                  </pre>
                </div>

                {/* Outputs */}
                {executionDetails.outputs && (
                  <div>
                    <p className="text-sm font-medium mb-2">Outputs</p>
                    <pre className="p-3 bg-muted rounded-lg text-xs overflow-x-auto">
                      {JSON.stringify(executionDetails.outputs, null, 2)}
                    </pre>
                  </div>
                )}

                {/* Node outputs */}
                {executionDetails.node_outputs && (
                  <div>
                    <p className="text-sm font-medium mb-2">Node Outputs</p>
                    <pre className="p-3 bg-muted rounded-lg text-xs overflow-x-auto max-h-48">
                      {JSON.stringify(executionDetails.node_outputs, null, 2)}
                    </pre>
                  </div>
                )}

                {/* Node errors */}
                {executionDetails.node_errors &&
                  Object.keys(executionDetails.node_errors).length > 0 && (
                    <div>
                      <p className="text-sm font-medium mb-2">Node Errors</p>
                      <div className="space-y-2">
                        {Object.entries(executionDetails.node_errors).map(
                          ([nodeId, error]) => (
                            <div
                              key={nodeId}
                              className="p-2 bg-red-50 dark:bg-red-900/20 rounded text-sm"
                            >
                              <span className="font-mono text-red-600">
                                {nodeId}:
                              </span>{" "}
                              {error}
                            </div>
                          )
                        )}
                      </div>
                    </div>
                  )}
              </div>
            </ScrollArea>
          ) : null}
        </DialogContent>
      </Dialog>
    </div>
  );
}

// Wrap with Suspense for useSearchParams
export default function ExecutionsPage() {
  return (
    <Suspense fallback={<div className="flex items-center justify-center py-12"><Loader2 className="h-6 w-6 animate-spin" /></div>}>
      <ExecutionsPageContent />
    </Suspense>
  );
}
