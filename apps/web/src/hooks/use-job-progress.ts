"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef } from "react";
import { apiClient } from "@/lib/api-client";

export interface JobProgress {
  id: string;
  type: string;
  status: "pending" | "queued" | "running" | "completed" | "failed" | "cancelled";
  progress: number;
  current_step?: string;
  result?: {
    added?: number;
    skipped?: number;
    deleted?: number;
    total?: number;
    processed?: number;
    errors?: string[];
    message?: string;
    // Export job fields
    download_url?: string;
    total_images?: number;
    total_annotations?: number;
    total_classes?: number;
    [key: string]: unknown;
  };
  error?: string;
}

interface UseJobProgressOptions {
  /** Called when job completes successfully */
  onComplete?: (result: JobProgress["result"]) => void;
  /** Called when job fails */
  onError?: (error: string) => void;
  /** Called when job is cancelled */
  onCancel?: () => void;
  /** Polling interval in ms (default: 1000) */
  pollingInterval?: number;
  /** Query keys to invalidate on completion */
  invalidateOnComplete?: string[][];
}

/**
 * Hook to track background job progress.
 *
 * @param jobId - Job ID to track (null to disable)
 * @param options - Callbacks and configuration
 *
 * @example
 * const { data: job, isPolling } = useJobProgress(jobId, {
 *   onComplete: (result) => {
 *     toast.success(`Added ${result.added} images`);
 *     setJobId(null);
 *   },
 *   onError: (error) => toast.error(error),
 *   invalidateOnComplete: [["od-datasets"], ["od-images"]],
 * });
 */
export function useJobProgress(
  jobId: string | null,
  options: UseJobProgressOptions = {}
) {
  const {
    onComplete,
    onError,
    onCancel,
    pollingInterval = 1000,
    invalidateOnComplete = [],
  } = options;

  const queryClient = useQueryClient();
  const completedRef = useRef(false);

  // Reset completed ref when jobId changes
  useEffect(() => {
    completedRef.current = false;
  }, [jobId]);

  const query = useQuery({
    queryKey: ["job-progress", jobId],
    queryFn: async (): Promise<JobProgress | null> => {
      if (!jobId) return null;
      return apiClient.getJob(jobId) as Promise<JobProgress>;
    },
    enabled: !!jobId,
    refetchInterval: (query) => {
      const data = query.state.data;
      if (!data) return pollingInterval;

      // Stop polling when job is done
      if (["completed", "failed", "cancelled"].includes(data.status)) {
        return false;
      }

      return pollingInterval;
    },
    staleTime: 0, // Always refetch
  });

  // Handle job completion/failure
  useEffect(() => {
    if (!query.data || completedRef.current) return;

    const { status, result, error } = query.data;

    if (status === "completed") {
      completedRef.current = true;
      onComplete?.(result);

      // Invalidate specified queries
      invalidateOnComplete.forEach((queryKey) => {
        queryClient.invalidateQueries({ queryKey });
      });
    } else if (status === "failed") {
      completedRef.current = true;
      onError?.(error || "Job failed");
    } else if (status === "cancelled") {
      completedRef.current = true;
      onCancel?.();
    }
  }, [query.data, onComplete, onError, onCancel, queryClient, invalidateOnComplete]);

  return {
    /** Current job data */
    job: query.data,
    /** Whether the job is being polled */
    isPolling: query.isFetching && !completedRef.current,
    /** Whether the job is in a terminal state */
    isDone: query.data
      ? ["completed", "failed", "cancelled"].includes(query.data.status)
      : false,
    /** Raw query object */
    query,
  };
}
