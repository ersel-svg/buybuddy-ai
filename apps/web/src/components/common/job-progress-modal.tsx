"use client";

import { Loader2, CheckCircle, XCircle, AlertCircle } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Progress } from "@/components/ui/progress";
import { Button } from "@/components/ui/button";
import { useJobProgress, JobProgress } from "@/hooks/use-job-progress";
import { apiClient } from "@/lib/api-client";
import { useMutation } from "@tanstack/react-query";

interface JobProgressModalProps {
  /** Job ID to track */
  jobId: string | null;
  /** Modal title */
  title: string;
  /** Called when modal should close */
  onClose: () => void;
  /** Called when job completes successfully */
  onComplete?: (result: JobProgress["result"]) => void;
  /** Query keys to invalidate on completion */
  invalidateOnComplete?: string[][];
}

/**
 * Modal that displays background job progress with a progress bar.
 *
 * @example
 * <JobProgressModal
 *   jobId={activeJobId}
 *   title="Adding images to dataset"
 *   onClose={() => setActiveJobId(null)}
 *   onComplete={(result) => toast.success(`Added ${result.added} images`)}
 *   invalidateOnComplete={[["od-datasets"]]}
 * />
 */
export function JobProgressModal({
  jobId,
  title,
  onClose,
  onComplete,
  invalidateOnComplete = [],
}: JobProgressModalProps) {
  const { job, isDone } = useJobProgress(jobId, {
    onComplete,
    invalidateOnComplete,
  });

  const cancelMutation = useMutation({
    mutationFn: async () => {
      if (!jobId) return;
      return apiClient.cancelJob(jobId);
    },
  });

  const handleCancel = () => {
    cancelMutation.mutate();
  };

  const getStatusIcon = () => {
    if (!job) return <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />;

    switch (job.status) {
      case "completed":
        return <CheckCircle className="h-5 w-5 text-green-500" />;
      case "failed":
        return <XCircle className="h-5 w-5 text-red-500" />;
      case "cancelled":
        return <AlertCircle className="h-5 w-5 text-yellow-500" />;
      default:
        return <Loader2 className="h-5 w-5 animate-spin text-blue-500" />;
    }
  };

  const getStatusText = () => {
    if (!job) return "Starting...";

    switch (job.status) {
      case "pending":
        return "Waiting to start...";
      case "queued":
        return "In queue...";
      case "running":
        return job.current_step || "Processing...";
      case "completed":
        return job.result?.message || "Completed successfully";
      case "failed":
        return job.error || "Job failed";
      case "cancelled":
        return "Job was cancelled";
      default:
        return "Unknown status";
    }
  };

  const getResultSummary = () => {
    if (!job?.result) return null;

    const { added, skipped, deleted, total, processed, errors } = job.result;
    const parts: string[] = [];

    if (typeof added === "number") parts.push(`${added} added`);
    if (typeof skipped === "number") parts.push(`${skipped} skipped`);
    if (typeof deleted === "number") parts.push(`${deleted} deleted`);
    if (typeof processed === "number" && typeof total === "number") {
      parts.push(`${processed}/${total} processed`);
    }

    if (errors && errors.length > 0) {
      parts.push(`${errors.length} errors`);
    }

    return parts.length > 0 ? parts.join(" | ") : null;
  };

  return (
    <Dialog open={!!jobId} onOpenChange={(open) => !open && isDone && onClose()}>
      <DialogContent className="sm:max-w-md" onPointerDownOutside={(e) => !isDone && e.preventDefault()}>
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            {getStatusIcon()}
            {title}
          </DialogTitle>
          <DialogDescription>{getStatusText()}</DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-4">
          {/* Progress bar */}
          <div className="space-y-2">
            <Progress value={job?.progress ?? 0} className="h-2" />
            <div className="flex justify-between text-sm text-muted-foreground">
              <span>{job?.progress ?? 0}%</span>
              {job?.result?.processed !== undefined && job?.result?.total !== undefined && (
                <span>
                  {job.result.processed} / {job.result.total}
                </span>
              )}
            </div>
          </div>

          {/* Result summary */}
          {isDone && getResultSummary() && (
            <div className="rounded-md bg-muted p-3 text-sm">
              {getResultSummary()}
            </div>
          )}

          {/* Errors */}
          {job?.result?.errors && job.result.errors.length > 0 && (
            <div className="rounded-md bg-red-50 p-3 text-sm text-red-800 dark:bg-red-900/20 dark:text-red-200">
              <p className="font-medium">Errors:</p>
              <ul className="list-inside list-disc">
                {job.result.errors.slice(0, 5).map((error, i) => (
                  <li key={i} className="truncate">
                    {error}
                  </li>
                ))}
                {job.result.errors.length > 5 && (
                  <li>...and {job.result.errors.length - 5} more</li>
                )}
              </ul>
            </div>
          )}
        </div>

        {/* Actions */}
        <div className="flex justify-end gap-2">
          {!isDone && (
            <Button
              variant="outline"
              onClick={handleCancel}
              disabled={cancelMutation.isPending}
            >
              {cancelMutation.isPending ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Cancelling...
                </>
              ) : (
                "Cancel"
              )}
            </Button>
          )}
          {isDone && (
            <Button onClick={onClose}>Close</Button>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}
