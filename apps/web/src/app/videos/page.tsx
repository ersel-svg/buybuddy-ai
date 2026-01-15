"use client";

import { useState, useMemo } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
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
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { Progress } from "@/components/ui/progress";
import {
  RefreshCw,
  Play,
  CheckCircle,
  XCircle,
  Clock,
  Loader2,
  Video,
  ExternalLink,
  RotateCcw,
  ChevronLeft,
  ChevronRight,
  ChevronsLeft,
  ChevronsRight,
  Trash2,
} from "lucide-react";
import type { Video as VideoType, Job } from "@/types";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";

const ITEMS_PER_PAGE = 50;
const PROCESSING_ITEMS_PER_PAGE = 20;

export default function VideosPage() {
  const queryClient = useQueryClient();
  const [selectedVideoIds, setSelectedVideoIds] = useState<Set<number>>(
    new Set()
  );
  const [syncModalOpen, setSyncModalOpen] = useState(false);
  const [syncLimit, setSyncLimit] = useState<string>("");
  const [syncAll, setSyncAll] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);
  const [processingPage, setProcessingPage] = useState(1);
  const [sampleRate, setSampleRate] = useState<number>(1);
  const [maxFrames, setMaxFrames] = useState<number | null>(null);
  const [geminiModel, setGeminiModel] = useState<string>("gemini-2.0-flash");
  const [processModalOpen, setProcessModalOpen] = useState(false);

  // Fetch videos with smart polling
  const { data: videos, isLoading: isLoadingVideos } = useQuery({
    queryKey: ["videos"],
    queryFn: () => apiClient.getVideos(),
    refetchInterval: (query) => {
      const data = query.state.data;
      const hasProcessing = data?.some((v) => v.status === "processing");
      return hasProcessing ? 5000 : 30000; // 5s if processing, 30s if idle
    },
  });

  // Fetch running jobs with smart polling
  const { data: jobs } = useQuery({
    queryKey: ["jobs", "video_processing"],
    queryFn: () => apiClient.getJobs("video_processing"),
    refetchInterval: (query) => {
      const data = query.state.data;
      const hasActive = data?.some((j) =>
        ["running", "queued", "pending"].includes(j.status)
      );
      return hasActive ? 3000 : 30000; // 3s if active, 30s if idle
    },
  });

  // Sync mutation
  const syncMutation = useMutation({
    mutationFn: (limit?: number) => apiClient.syncVideos(limit),
    onSuccess: (result) => {
      toast.success(`Synced ${result.synced_count} new videos`);
      queryClient.invalidateQueries({ queryKey: ["videos"] });
      setSyncModalOpen(false);
      setSyncLimit("");
      setSyncAll(false);
    },
    onError: () => {
      toast.error("Failed to sync videos");
    },
  });

  // Handle sync submit
  const handleSyncSubmit = () => {
    if (syncAll) {
      syncMutation.mutate(undefined); // No limit = all
    } else {
      const limit = parseInt(syncLimit, 10);
      if (isNaN(limit) || limit <= 0) {
        toast.error("Please enter a valid number");
        return;
      }
      syncMutation.mutate(limit);
    }
  };

  // Process mutation
  const processMutation = useMutation({
    mutationFn: ({
      videoIds,
      sampleRate,
      maxFrames,
      geminiModel,
    }: {
      videoIds: number[];
      sampleRate?: number;
      maxFrames?: number | null;
      geminiModel?: string;
    }) =>
      apiClient.processVideos(videoIds, {
        sampleRate,
        maxFrames: maxFrames ?? undefined,
        geminiModel,
      }),
    onSuccess: (result) => {
      toast.success(`Started processing ${result.length} videos`);
      setSelectedVideoIds(new Set());
      setProcessModalOpen(false);
      queryClient.invalidateQueries({ queryKey: ["videos"] });
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
    },
    onError: (error) => {
      const message = error instanceof Error ? error.message : "Unknown error";
      toast.error(`Failed to start processing: ${message}`);
      console.error("Process error:", error);
    },
  });

  // Handle process submit
  const handleProcessSubmit = () => {
    processMutation.mutate({
      videoIds: Array.from(selectedVideoIds),
      sampleRate,
      maxFrames,
      geminiModel,
    });
  };

  // Reprocess mutation
  const reprocessMutation = useMutation({
    mutationFn: (videoId: number) => apiClient.reprocessVideo(videoId),
    onSuccess: (result) => {
      toast.success(`Reprocessing started. Cleaned up ${result.cleanup.frames_deleted} frames.`);
      queryClient.invalidateQueries({ queryKey: ["videos"] });
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
      queryClient.invalidateQueries({ queryKey: ["products"] });
    },
    onError: () => {
      toast.error("Failed to start reprocessing");
    },
  });

  // Cancel single job mutation
  const cancelJobMutation = useMutation({
    mutationFn: (jobId: string) => apiClient.cancelJob(jobId),
    onSuccess: () => {
      toast.success("Job cancelled");
      queryClient.invalidateQueries({ queryKey: ["videos"] });
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
    },
    onError: () => {
      toast.error("Failed to cancel job");
    },
  });

  // Cancel all processing jobs mutation
  const cancelAllMutation = useMutation({
    mutationFn: () => apiClient.cancelJobsBatch({ job_type: "video_processing" }),
    onSuccess: (result) => {
      toast.success(`Cancelled ${result.cancelled_count} jobs`);
      queryClient.invalidateQueries({ queryKey: ["videos"] });
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
    },
    onError: () => {
      toast.error("Failed to cancel jobs");
    },
  });

  // Sync Runpod status mutation
  const syncStatusMutation = useMutation({
    mutationFn: () => apiClient.syncRunpodStatus(),
    onSuccess: (result) => {
      const message = result.updated > 0
        ? `Synced ${result.checked} jobs: ${result.updated} updated (${result.completed} completed, ${result.failed} failed, ${result.still_running} running)`
        : `Checked ${result.checked} jobs - all are up to date`;
      toast.success(message);
      queryClient.invalidateQueries({ queryKey: ["videos"] });
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
      queryClient.invalidateQueries({ queryKey: ["products"] });
    },
    onError: () => {
      toast.error("Failed to sync Runpod status");
    },
  });

  // Clear stuck videos mutation
  const clearStuckMutation = useMutation({
    mutationFn: () => apiClient.clearStuckVideos(),
    onSuccess: (result) => {
      if (result.cleared > 0) {
        toast.success(`Cleared ${result.cleared} stuck videos (${result.no_job} without jobs, ${result.job_finished} with finished jobs)`);
      } else {
        toast.info(`No stuck videos found (checked ${result.checked})`);
      }
      queryClient.invalidateQueries({ queryKey: ["videos"] });
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
      queryClient.invalidateQueries({ queryKey: ["products"] });
    },
    onError: () => {
      toast.error("Failed to clear stuck videos");
    },
  });

  // Calculate stats
  const stats = {
    pending: videos?.filter((v) => v.status === "pending").length || 0,
    processing: videos?.filter((v) => v.status === "processing").length || 0,
    completed: videos?.filter((v) => v.status === "completed").length || 0,
    failed: videos?.filter((v) => v.status === "failed").length || 0,
  };

  // Get processing videos with their jobs
  const processingVideos = videos?.filter((v) => v.status === "processing") || [];
  const pendingVideos = videos?.filter((v) => v.status === "pending") || [];

  // Pagination for processing videos
  const processingTotalPages = Math.ceil(processingVideos.length / PROCESSING_ITEMS_PER_PAGE);
  const paginatedProcessingVideos = useMemo(() => {
    const start = (processingPage - 1) * PROCESSING_ITEMS_PER_PAGE;
    const end = start + PROCESSING_ITEMS_PER_PAGE;
    return processingVideos.slice(start, end);
  }, [processingVideos, processingPage]);

  // Pagination for pending videos
  const totalPages = Math.ceil(pendingVideos.length / ITEMS_PER_PAGE);
  const paginatedPendingVideos = useMemo(() => {
    const start = (currentPage - 1) * ITEMS_PER_PAGE;
    const end = start + ITEMS_PER_PAGE;
    return pendingVideos.slice(start, end);
  }, [pendingVideos, currentPage]);

  // Reset page when videos change significantly
  useMemo(() => {
    if (currentPage > totalPages && totalPages > 0) {
      setCurrentPage(1);
    }
    if (processingPage > processingTotalPages && processingTotalPages > 0) {
      setProcessingPage(1);
    }
  }, [totalPages, currentPage, processingTotalPages, processingPage]);

  // Status icons
  const statusConfig: Record<
    string,
    { icon: React.ReactNode; color: string }
  > = {
    pending: {
      icon: <Clock className="h-4 w-4" />,
      color: "bg-yellow-100 text-yellow-800",
    },
    processing: {
      icon: <Loader2 className="h-4 w-4 animate-spin" />,
      color: "bg-blue-100 text-blue-800",
    },
    completed: {
      icon: <CheckCircle className="h-4 w-4" />,
      color: "bg-green-100 text-green-800",
    },
    failed: {
      icon: <XCircle className="h-4 w-4" />,
      color: "bg-red-100 text-red-800",
    },
  };

  // Selection helpers
  const toggleSelectPage = () => {
    const pageIds = paginatedPendingVideos.map((v) => v.id);
    const allPageSelected = pageIds.every((id) => selectedVideoIds.has(id));

    if (allPageSelected) {
      // Deselect all on current page
      const newSet = new Set(selectedVideoIds);
      pageIds.forEach((id) => newSet.delete(id));
      setSelectedVideoIds(newSet);
    } else {
      // Select all on current page
      const newSet = new Set(selectedVideoIds);
      pageIds.forEach((id) => newSet.add(id));
      setSelectedVideoIds(newSet);
    }
  };

  const selectAllVideos = () => {
    setSelectedVideoIds(new Set(pendingVideos.map((v) => v.id)));
  };

  const deselectAll = () => {
    setSelectedVideoIds(new Set());
  };

  const toggleSelect = (id: number) => {
    const newSet = new Set(selectedVideoIds);
    if (newSet.has(id)) {
      newSet.delete(id);
    } else {
      newSet.add(id);
    }
    setSelectedVideoIds(newSet);
  };

  // Check if all on current page are selected
  const allPageSelected = paginatedPendingVideos.length > 0 &&
    paginatedPendingVideos.every((v) => selectedVideoIds.has(v.id));

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold">Videos</h1>
          <p className="text-gray-500">
            Sync and process videos from Buybuddy API
          </p>
        </div>
        <div className="flex gap-2">
          <Button
            variant="outline"
            onClick={() => setSyncModalOpen(true)}
            disabled={syncMutation.isPending}
          >
            <RefreshCw
              className={`h-4 w-4 mr-2 ${
                syncMutation.isPending ? "animate-spin" : ""
              }`}
            />
            Sync from Buybuddy
          </Button>
          <Button
            onClick={() => setProcessModalOpen(true)}
            disabled={selectedVideoIds.size === 0 || processMutation.isPending}
          >
            {processMutation.isPending ? (
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            ) : (
              <Play className="h-4 w-4 mr-2" />
            )}
            Process Selected ({selectedVideoIds.size})
          </Button>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-gray-500">Pending</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">{stats.pending}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-gray-500">Processing</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-blue-600">
              {stats.processing}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-gray-500">Completed</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-green-600">
              {stats.completed}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-gray-500">Failed</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-red-600">{stats.failed}</p>
          </CardContent>
        </Card>
      </div>

      {/* Currently Processing */}
      {processingVideos.length > 0 && (
        <Card>
          <CardHeader>
            <div className="flex justify-between items-start">
              <div>
                <CardTitle className="flex items-center gap-2">
                  <Loader2 className="h-5 w-5 animate-spin text-blue-500" />
                  Currently Processing ({processingVideos.length})
                </CardTitle>
                <CardDescription>
                  Videos being processed by Runpod workers
                </CardDescription>
              </div>
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => syncStatusMutation.mutate()}
                  disabled={syncStatusMutation.isPending}
                  title="Check actual job status from Runpod"
                >
                  {syncStatusMutation.isPending ? (
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <RefreshCw className="h-4 w-4 mr-2" />
                  )}
                  Sync Status
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => clearStuckMutation.mutate()}
                  disabled={clearStuckMutation.isPending}
                  title="Mark stuck videos as failed"
                >
                  {clearStuckMutation.isPending ? (
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <Trash2 className="h-4 w-4 mr-2" />
                  )}
                  Clear Stuck
                </Button>
                <AlertDialog>
                  <AlertDialogTrigger asChild>
                    <Button
                      variant="destructive"
                      size="sm"
                      disabled={cancelAllMutation.isPending}
                    >
                      {cancelAllMutation.isPending ? (
                        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      ) : (
                        <XCircle className="h-4 w-4 mr-2" />
                      )}
                      Cancel All
                    </Button>
                  </AlertDialogTrigger>
                  <AlertDialogContent>
                    <AlertDialogHeader>
                      <AlertDialogTitle>Cancel All Processing Jobs?</AlertDialogTitle>
                      <AlertDialogDescription>
                        This will cancel all {processingVideos.length} processing videos
                        and reset them to pending status. This action cannot be undone.
                      </AlertDialogDescription>
                    </AlertDialogHeader>
                    <AlertDialogFooter>
                      <AlertDialogCancel>Keep Running</AlertDialogCancel>
                      <AlertDialogAction
                        onClick={() => cancelAllMutation.mutate()}
                        className="bg-destructive text-destructive-foreground"
                      >
                        Cancel All Jobs
                      </AlertDialogAction>
                    </AlertDialogFooter>
                  </AlertDialogContent>
                </AlertDialog>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {paginatedProcessingVideos.map((video) => {
                const job = jobs?.find(
                  (j) =>
                    j.type === "video_processing" &&
                    j.config?.video_id === video.id
                );
                return (
                  <div key={video.id} className="flex items-center gap-4">
                    <Loader2 className="h-5 w-5 text-blue-500 animate-spin flex-shrink-0" />
                    <div className="flex-1 min-w-0">
                      <p className="font-medium font-mono truncate">{video.barcode}</p>
                      <Progress
                        value={job?.progress || 0}
                        className="h-2 mt-1"
                      />
                    </div>
                    <span className="text-sm text-gray-500 w-12 text-right">
                      {job?.progress || 0}%
                    </span>
                    {job && (
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => cancelJobMutation.mutate(job.id)}
                        disabled={cancelJobMutation.isPending}
                        className="flex-shrink-0"
                      >
                        <XCircle className="h-4 w-4 text-destructive" />
                      </Button>
                    )}
                  </div>
                );
              })}
            </div>

            {/* Pagination for processing videos */}
            {processingTotalPages > 1 && (
              <div className="flex items-center justify-between mt-4 pt-4 border-t">
                <div className="text-sm text-gray-500">
                  Showing {(processingPage - 1) * PROCESSING_ITEMS_PER_PAGE + 1} to{" "}
                  {Math.min(processingPage * PROCESSING_ITEMS_PER_PAGE, processingVideos.length)} of{" "}
                  {processingVideos.length} videos
                </div>
                <div className="flex items-center gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setProcessingPage(1)}
                    disabled={processingPage === 1}
                  >
                    <ChevronsLeft className="h-4 w-4" />
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setProcessingPage((p) => Math.max(1, p - 1))}
                    disabled={processingPage === 1}
                  >
                    <ChevronLeft className="h-4 w-4" />
                  </Button>
                  <span className="text-sm px-2">
                    Page {processingPage} of {processingTotalPages}
                  </span>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setProcessingPage((p) => Math.min(processingTotalPages, p + 1))}
                    disabled={processingPage === processingTotalPages}
                  >
                    <ChevronRight className="h-4 w-4" />
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setProcessingPage(processingTotalPages)}
                    disabled={processingPage === processingTotalPages}
                  >
                    <ChevronsRight className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Pending Videos */}
      <Card>
        <CardHeader>
          <div className="flex justify-between items-start">
            <div>
              <CardTitle>Pending Videos</CardTitle>
              <CardDescription>
                Videos ready to be processed. Select and click &quot;Process
                Selected&quot; to start.
              </CardDescription>
            </div>
            {pendingVideos.length > 0 && (
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={selectAllVideos}
                  disabled={selectedVideoIds.size === pendingVideos.length}
                >
                  Select All ({pendingVideos.length})
                </Button>
                {selectedVideoIds.size > 0 && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={deselectAll}
                  >
                    Clear Selection
                  </Button>
                )}
              </div>
            )}
          </div>
        </CardHeader>
        <CardContent>
          {isLoadingVideos ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-6 w-6 animate-spin" />
            </div>
          ) : pendingVideos.length === 0 ? (
            <div className="text-center py-8">
              <Video className="h-12 w-12 mx-auto text-gray-300 mb-2" />
              <p className="text-gray-500">No pending videos</p>
              <p className="text-gray-400 text-sm">
                Click &quot;Sync from Buybuddy&quot; to fetch new videos
              </p>
            </div>
          ) : (
            <>
              {/* Selection info */}
              {selectedVideoIds.size > 0 && (
                <div className="mb-4 p-3 bg-blue-50 rounded-lg flex items-center justify-between">
                  <span className="text-sm text-blue-700">
                    {selectedVideoIds.size} of {pendingVideos.length} videos selected
                  </span>
                </div>
              )}

              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-12">
                      <Checkbox
                        checked={allPageSelected}
                        onCheckedChange={toggleSelectPage}
                      />
                    </TableHead>
                    <TableHead>Video ID</TableHead>
                    <TableHead>Barcode</TableHead>
                    <TableHead>Video URL</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Added</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {paginatedPendingVideos.map((video) => (
                    <TableRow
                      key={video.id}
                      className={
                        selectedVideoIds.has(video.id) ? "bg-blue-50" : ""
                      }
                    >
                      <TableCell>
                        <Checkbox
                          checked={selectedVideoIds.has(video.id)}
                          onCheckedChange={() => toggleSelect(video.id)}
                        />
                      </TableCell>
                      <TableCell>{video.id}</TableCell>
                      <TableCell className="font-mono">{video.barcode}</TableCell>
                      <TableCell>
                        <a
                          href={video.video_url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-blue-600 hover:underline flex items-center gap-1"
                        >
                          View Video
                          <ExternalLink className="h-3 w-3" />
                        </a>
                      </TableCell>
                      <TableCell>
                        <Badge className={statusConfig[video.status].color}>
                          <span className="mr-1">
                            {statusConfig[video.status].icon}
                          </span>
                          {video.status}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-gray-500 text-sm">
                        {new Date(video.created_at).toLocaleDateString()}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>

              {/* Pagination */}
              {totalPages > 1 && (
                <div className="flex items-center justify-between mt-4 pt-4 border-t">
                  <div className="text-sm text-gray-500">
                    Showing {(currentPage - 1) * ITEMS_PER_PAGE + 1} to{" "}
                    {Math.min(currentPage * ITEMS_PER_PAGE, pendingVideos.length)} of{" "}
                    {pendingVideos.length} videos
                  </div>
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
                    <span className="text-sm px-2">
                      Page {currentPage} of {totalPages}
                    </span>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                      disabled={currentPage === totalPages}
                    >
                      <ChevronRight className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setCurrentPage(totalPages)}
                      disabled={currentPage === totalPages}
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

      {/* Completed Videos */}
      {stats.completed > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Recently Completed</CardTitle>
            <CardDescription>
              Successfully processed videos with extracted products
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Video ID</TableHead>
                  <TableHead>Barcode</TableHead>
                  <TableHead>Product</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Completed</TableHead>
                  <TableHead>Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {videos
                  ?.filter((v) => v.status === "completed")
                  .slice(0, 10)
                  .map((video) => (
                    <TableRow key={video.id}>
                      <TableCell>{video.id}</TableCell>
                      <TableCell className="font-mono">
                        {video.barcode}
                      </TableCell>
                      <TableCell>
                        {video.product_id ? (
                          <Link
                            href={`/products/${video.product_id}`}
                            className="text-blue-600 hover:underline"
                          >
                            View Product
                          </Link>
                        ) : (
                          <span className="text-gray-400">-</span>
                        )}
                      </TableCell>
                      <TableCell>
                        <Badge className={statusConfig[video.status].color}>
                          <span className="mr-1">
                            {statusConfig[video.status].icon}
                          </span>
                          {video.status}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-gray-500 text-sm">
                        {new Date(video.created_at).toLocaleDateString()}
                      </TableCell>
                      <TableCell>
                        <AlertDialog>
                          <AlertDialogTrigger asChild>
                            <Button
                              variant="outline"
                              size="sm"
                              disabled={!video.product_id || reprocessMutation.isPending}
                            >
                              {reprocessMutation.isPending ? (
                                <Loader2 className="h-4 w-4 animate-spin" />
                              ) : (
                                <RotateCcw className="h-4 w-4" />
                              )}
                            </Button>
                          </AlertDialogTrigger>
                          <AlertDialogContent>
                            <AlertDialogHeader>
                              <AlertDialogTitle>Reprocess Video?</AlertDialogTitle>
                              <AlertDialogDescription>
                                This will delete all existing frames for this product and re-run the entire video processing pipeline.
                                <br /><br />
                                <strong>Barcode:</strong> {video.barcode}
                                <br />
                                <strong>Video ID:</strong> {video.id}
                              </AlertDialogDescription>
                            </AlertDialogHeader>
                            <AlertDialogFooter>
                              <AlertDialogCancel>Cancel</AlertDialogCancel>
                              <AlertDialogAction
                                onClick={() => reprocessMutation.mutate(video.id)}
                              >
                                Reprocess
                              </AlertDialogAction>
                            </AlertDialogFooter>
                          </AlertDialogContent>
                        </AlertDialog>
                      </TableCell>
                    </TableRow>
                  ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}

      {/* Sync Modal */}
      <Dialog open={syncModalOpen} onOpenChange={setSyncModalOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Sync Videos from Buybuddy</DialogTitle>
            <DialogDescription>
              Choose how many videos to sync. Only new videos (by video ID) will be added.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="flex items-center space-x-2">
              <Checkbox
                id="sync-all"
                checked={syncAll}
                onCheckedChange={(checked) => {
                  setSyncAll(checked === true);
                  if (checked) setSyncLimit("");
                }}
              />
              <Label htmlFor="sync-all">Sync all available videos</Label>
            </div>
            {!syncAll && (
              <div className="space-y-2">
                <Label htmlFor="sync-limit">Number of videos to sync</Label>
                <Input
                  id="sync-limit"
                  type="number"
                  placeholder="e.g., 10"
                  value={syncLimit}
                  onChange={(e) => setSyncLimit(e.target.value)}
                  min={1}
                />
              </div>
            )}
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setSyncModalOpen(false)}>
              Cancel
            </Button>
            <Button
              onClick={handleSyncSubmit}
              disabled={syncMutation.isPending || (!syncAll && !syncLimit)}
            >
              {syncMutation.isPending ? (
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              ) : (
                <RefreshCw className="h-4 w-4 mr-2" />
              )}
              {syncAll ? "Sync All" : `Sync ${syncLimit || "..."}`}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Process Modal */}
      <Dialog open={processModalOpen} onOpenChange={setProcessModalOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Process Videos</DialogTitle>
            <DialogDescription>
              Configure processing settings for {selectedVideoIds.size} selected videos.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-6 py-4">
            {/* Sample Rate */}
            <div className="space-y-2">
              <Label htmlFor="sample-rate">
                Sample Rate: <span className="font-bold">{sampleRate}</span>
              </Label>
              <input
                id="sample-rate"
                type="range"
                min={1}
                max={10}
                step={1}
                value={sampleRate}
                onChange={(e) => setSampleRate(Number(e.target.value))}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
              />
              <p className="text-sm text-gray-500">
                Extract every Nth frame. 1 = every frame, 2 = every 2nd frame, etc.
                Higher values = faster processing, fewer frames.
              </p>
            </div>

            {/* Max Frames */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="max-frames">
                  Max Frames: <span className="font-bold">{maxFrames ?? "All"}</span>
                </Label>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setMaxFrames(maxFrames ? null : 100)}
                >
                  {maxFrames ? "Use All Frames" : "Set Limit"}
                </Button>
              </div>
              {maxFrames !== null && (
                <input
                  id="max-frames"
                  type="range"
                  min={50}
                  max={500}
                  step={10}
                  value={maxFrames}
                  onChange={(e) => setMaxFrames(Number(e.target.value))}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                />
              )}
              <p className="text-sm text-gray-500">
                Maximum number of frames to extract per video.
                &quot;All&quot; = no limit, extract all sampled frames.
              </p>
            </div>

            {/* Gemini Model */}
            <div className="space-y-2">
              <Label htmlFor="gemini-model">Gemini Model</Label>
              <Select value={geminiModel} onValueChange={setGeminiModel}>
                <SelectTrigger id="gemini-model">
                  <SelectValue placeholder="Select model" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="gemini-2.0-flash">
                    gemini-2.0-flash (Unlimited)
                  </SelectItem>
                  <SelectItem value="gemini-2.5-flash">
                    gemini-2.5-flash (10K/day)
                  </SelectItem>
                  <SelectItem value="gemini-2.0-flash-exp">
                    gemini-2.0-flash-exp (500/day)
                  </SelectItem>
                </SelectContent>
              </Select>
              <p className="text-sm text-gray-500">
                AI model for video analysis. Use &quot;gemini-2.0-flash&quot; for unlimited requests.
              </p>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setProcessModalOpen(false)}>
              Cancel
            </Button>
            <Button
              onClick={handleProcessSubmit}
              disabled={processMutation.isPending}
            >
              {processMutation.isPending ? (
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              ) : (
                <Play className="h-4 w-4 mr-2" />
              )}
              Process {selectedVideoIds.size} Videos
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
