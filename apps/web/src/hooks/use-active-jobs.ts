"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "@/lib/api-client";
import { toast } from "sonner";

export interface Job {
  id: string;
  type: string;
  status: "pending" | "queued" | "running" | "completed" | "failed" | "cancelled";
  progress: number;
  error?: string | null;
  result?: {
    stage?: string;
    message?: string;
    can_resume?: boolean;
    images_imported?: number;
    annotations_imported?: number;
    processed_count?: number;
    total_images?: number;
    failed_count?: number;
    checkpoint?: {
      stage?: string;
      error_message?: string;
      download_complete?: boolean;
      processed_ids?: string[];
      total_images?: number;
    };
    [key: string]: unknown;
  } | null;
  config?: {
    workspace?: string;
    project?: string;
    version?: number;
    dataset_id?: string;
    use_streaming?: boolean;
    [key: string]: unknown;
  } | null;
  created_at: string;
  updated_at?: string;
}

export function useActiveJobs(options?: {
  jobType?: string;
  refetchInterval?: number;
  enabled?: boolean;
}) {
  const { jobType, refetchInterval = 3000, enabled = true } = options || {};

  return useQuery({
    queryKey: ["active-jobs", jobType],
    queryFn: async () => {
      const jobs = await apiClient.getJobs(jobType);
      // Filter to recent jobs (last 24 hours) and sort by created_at desc
      const oneDayAgo = new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString();
      return jobs
        .filter((job: Job) => job.created_at > oneDayAgo)
        .sort((a: Job, b: Job) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())
        .slice(0, 20); // Limit to 20 most recent
    },
    refetchInterval: (query) => {
      // Only poll if there are active jobs
      const data = query.state.data as Job[] | undefined;
      const hasActiveJobs = data?.some(
        (job) => job.status === "running" || job.status === "pending" || job.status === "queued"
      );
      return hasActiveJobs ? refetchInterval : false;
    },
    enabled,
  });
}

export function useActiveJobsCount(jobType?: string) {
  return useQuery({
    queryKey: ["active-jobs-count", jobType],
    queryFn: () => apiClient.getActiveJobsCount(jobType),
    refetchInterval: 5000, // Poll every 5 seconds
  });
}

export function useCancelJob() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (jobId: string) => apiClient.cancelJob(jobId),
    onSuccess: () => {
      toast.success("Job cancelled");
      queryClient.invalidateQueries({ queryKey: ["active-jobs"] });
      queryClient.invalidateQueries({ queryKey: ["active-jobs-count"] });
    },
    onError: (error) => {
      toast.error(`Failed to cancel job: ${error}`);
    },
  });
}

export function useRetryRoboflowImport() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (jobId: string) => apiClient.retryRoboflowImport(jobId),
    onSuccess: (data) => {
      toast.success(data.message || "Import retry started");
      queryClient.invalidateQueries({ queryKey: ["active-jobs"] });
      queryClient.invalidateQueries({ queryKey: ["active-jobs-count"] });
    },
    onError: (error) => {
      toast.error(`Failed to retry import: ${error}`);
    },
  });
}

// Helper to get job display info
export function getJobDisplayInfo(job: Job): {
  title: string;
  subtitle: string;
  icon: "import" | "training" | "export" | "ai" | "sync" | "default";
  canRetry: boolean;
  canCancel: boolean;
  checkpointInfo?: string;
} {
  const status = job.status;
  const canCancel = status === "running" || status === "pending" || status === "queued";

  switch (job.type) {
    case "roboflow_import": {
      // Get checkpoint progress info
      const processedCount = job.result?.processed_count ||
        job.result?.checkpoint?.processed_ids?.length || 0;
      const totalImages = job.result?.total_images ||
        job.result?.checkpoint?.total_images || 0;

      const checkpointInfo = totalImages > 0
        ? `${processedCount}/${totalImages} images`
        : undefined;

      return {
        title: "Roboflow Import",
        subtitle: job.config?.project
          ? `${job.config.project} v${job.config.version}`
          : "Importing dataset",
        icon: "import",
        canRetry: status === "failed" && job.result?.can_resume !== false,
        canCancel,
        checkpointInfo,
      };
    }
    case "od_training":
      return {
        title: "OD Training",
        subtitle: job.result?.message || "Training model",
        icon: "training",
        canRetry: false,
        canCancel,
      };
    case "od_export":
      return {
        title: "Dataset Export",
        subtitle: job.result?.message || "Exporting dataset",
        icon: "export",
        canRetry: false,
        canCancel,
      };
    case "od_ai_annotation":
      return {
        title: "AI Annotation",
        subtitle: job.result?.message || "Auto-annotating images",
        icon: "ai",
        canRetry: false,
        canCancel,
      };
    case "buybuddy_sync":
      return {
        title: "BuyBuddy Sync",
        subtitle: job.result?.message || "Syncing images",
        icon: "sync",
        canRetry: false,
        canCancel,
      };
    default:
      return {
        title: job.type.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()),
        subtitle: job.result?.message || "",
        icon: "default",
        canRetry: false,
        canCancel,
      };
  }
}
