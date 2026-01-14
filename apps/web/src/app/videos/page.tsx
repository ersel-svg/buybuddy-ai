"use client";

import { useState } from "react";
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

export default function VideosPage() {
  const queryClient = useQueryClient();
  const [selectedVideoIds, setSelectedVideoIds] = useState<Set<number>>(
    new Set()
  );
  const [syncModalOpen, setSyncModalOpen] = useState(false);
  const [syncLimit, setSyncLimit] = useState<string>("");
  const [syncAll, setSyncAll] = useState(false);

  // Fetch videos
  const { data: videos, isLoading: isLoadingVideos } = useQuery({
    queryKey: ["videos"],
    queryFn: () => apiClient.getVideos(),
    refetchInterval: 10000, // Refresh every 10 seconds
  });

  // Fetch running jobs
  const { data: jobs } = useQuery({
    queryKey: ["jobs", "video_processing"],
    queryFn: () => apiClient.getJobs("video_processing"),
    refetchInterval: 5000, // Refresh every 5 seconds
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
    mutationFn: (videoIds: number[]) => apiClient.processVideos(videoIds),
    onSuccess: (result) => {
      toast.success(`Started processing ${result.length} videos`);
      setSelectedVideoIds(new Set());
      queryClient.invalidateQueries({ queryKey: ["videos"] });
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
    },
    onError: () => {
      toast.error("Failed to start processing");
    },
  });

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
  const toggleSelectAll = () => {
    if (selectedVideoIds.size === pendingVideos.length) {
      setSelectedVideoIds(new Set());
    } else {
      setSelectedVideoIds(new Set(pendingVideos.map((v) => v.id)));
    }
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
            onClick={() =>
              processMutation.mutate(Array.from(selectedVideoIds))
            }
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
            <CardTitle className="flex items-center gap-2">
              <Loader2 className="h-5 w-5 animate-spin text-blue-500" />
              Currently Processing
            </CardTitle>
            <CardDescription>
              Videos being processed by Runpod workers
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {processingVideos.map((video) => {
                const job = jobs?.find(
                  (j) =>
                    j.type === "video_processing" &&
                    j.config?.video_id === video.id
                );
                return (
                  <div key={video.id} className="flex items-center gap-4">
                    <Loader2 className="h-5 w-5 text-blue-500 animate-spin" />
                    <div className="flex-1">
                      <p className="font-medium font-mono">{video.barcode}</p>
                      <Progress
                        value={job?.progress || 0}
                        className="h-2 mt-1"
                      />
                    </div>
                    <span className="text-sm text-gray-500">
                      {job?.progress || 0}%
                    </span>
                  </div>
                );
              })}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Pending Videos */}
      <Card>
        <CardHeader>
          <CardTitle>Pending Videos</CardTitle>
          <CardDescription>
            Videos ready to be processed. Select and click &quot;Process
            Selected&quot; to start.
          </CardDescription>
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
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-12">
                    <Checkbox
                      checked={
                        pendingVideos.length > 0 &&
                        selectedVideoIds.size === pendingVideos.length
                      }
                      onCheckedChange={toggleSelectAll}
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
                {pendingVideos.map((video) => (
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
    </div>
  );
}
