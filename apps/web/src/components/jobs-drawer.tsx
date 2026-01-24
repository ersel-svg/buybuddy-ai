"use client";

import { useState } from "react";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import {
  Loader2,
  CheckCircle,
  XCircle,
  Clock,
  Package,
  Brain,
  Download,
  Cloud,
  Sparkles,
  X,
  RefreshCw,
  Bell,
  Inbox,
  CircleDot,
  History,
} from "lucide-react";
import {
  useActiveJobs,
  useActiveJobsCount,
  useCancelJob,
  useRetryRoboflowImport,
  getJobDisplayInfo,
  type Job,
} from "@/hooks/use-active-jobs";
import { formatDistanceToNow } from "date-fns";

const statusConfig: Record<
  string,
  {
    icon: typeof Clock;
    color: string;
    bgColor: string;
    borderColor: string;
    label: string;
    animate?: boolean;
  }
> = {
  pending: {
    icon: Clock,
    color: "text-amber-600 dark:text-amber-400",
    bgColor: "bg-amber-50 dark:bg-amber-950/30",
    borderColor: "border-amber-200 dark:border-amber-800/50",
    label: "Pending",
  },
  queued: {
    icon: Clock,
    color: "text-blue-600 dark:text-blue-400",
    bgColor: "bg-blue-50 dark:bg-blue-950/30",
    borderColor: "border-blue-200 dark:border-blue-800/50",
    label: "Queued",
  },
  running: {
    icon: Loader2,
    color: "text-blue-600 dark:text-blue-400",
    bgColor: "bg-blue-50 dark:bg-blue-950/30",
    borderColor: "border-blue-200 dark:border-blue-800/50",
    label: "Running",
    animate: true,
  },
  completed: {
    icon: CheckCircle,
    color: "text-emerald-600 dark:text-emerald-400",
    bgColor: "bg-emerald-50 dark:bg-emerald-950/30",
    borderColor: "border-emerald-200 dark:border-emerald-800/50",
    label: "Completed",
  },
  failed: {
    icon: XCircle,
    color: "text-red-600 dark:text-red-400",
    bgColor: "bg-red-50 dark:bg-red-950/30",
    borderColor: "border-red-200 dark:border-red-800/50",
    label: "Failed",
  },
  cancelled: {
    icon: X,
    color: "text-gray-500 dark:text-gray-400",
    bgColor: "bg-gray-50 dark:bg-gray-900/50",
    borderColor: "border-gray-200 dark:border-gray-700",
    label: "Cancelled",
  },
};

const iconMap = {
  import: Package,
  training: Brain,
  export: Download,
  ai: Sparkles,
  sync: Cloud,
  default: Clock,
};

function JobCard({ job }: { job: Job }) {
  const cancelJob = useCancelJob();
  const retryImport = useRetryRoboflowImport();
  const displayInfo = getJobDisplayInfo(job);
  const status = statusConfig[job.status] || statusConfig.pending;
  const StatusIcon = status.icon;
  const TypeIcon = iconMap[displayInfo.icon] || iconMap.default;
  const isActive =
    job.status === "running" ||
    job.status === "pending" ||
    job.status === "queued";

  return (
    <div
      className={`
        group relative p-4 rounded-xl border transition-all duration-200
        ${status.borderColor} ${status.bgColor}
        hover:shadow-sm hover:border-opacity-80
      `}
    >
      <div className="flex items-start gap-3">
        {/* Type Icon */}
        <div
          className={`
            flex-shrink-0 p-2 rounded-lg
            bg-background/60 dark:bg-background/40
            border border-border/50
          `}
        >
          <TypeIcon className="h-4 w-4 text-muted-foreground" />
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0 space-y-2">
          {/* Header */}
          <div className="flex items-start justify-between gap-2">
            <div className="min-w-0 flex-1">
              <h4 className="font-medium text-sm leading-tight truncate">
                {displayInfo.title}
              </h4>
              {displayInfo.subtitle && (
                <p className="text-xs text-muted-foreground mt-0.5 truncate">
                  {displayInfo.subtitle}
                </p>
              )}
            </div>
            <Badge
              variant="secondary"
              className={`
                flex-shrink-0 text-[11px] font-medium px-2 py-0.5
                ${status.color} ${status.bgColor} border ${status.borderColor}
              `}
            >
              <StatusIcon
                className={`h-3 w-3 mr-1 ${status.animate ? "animate-spin" : ""}`}
              />
              {status.label}
            </Badge>
          </div>

          {/* Progress Bar */}
          {isActive && job.progress > 0 && (
            <div className="space-y-1.5">
              <div className="flex items-center justify-between text-xs">
                <span className="text-muted-foreground">
                  {job.result?.stage || "Processing"}
                </span>
                <span className="font-medium tabular-nums">{job.progress}%</span>
              </div>
              <Progress value={job.progress} className="h-1.5" />
            </div>
          )}

          {/* Error Message */}
          {job.status === "failed" && job.error && (
            <p className="text-xs text-red-600 dark:text-red-400 line-clamp-2 bg-red-50 dark:bg-red-950/30 rounded-md px-2 py-1.5">
              {job.error}
            </p>
          )}

          {/* Checkpoint Info */}
          {job.status === "failed" && displayInfo.checkpointInfo && (
            <p className="text-xs text-amber-600 dark:text-amber-400 bg-amber-50 dark:bg-amber-950/30 rounded-md px-2 py-1.5">
              Progress saved: {displayInfo.checkpointInfo}
            </p>
          )}

          {/* Success Result */}
          {job.status === "completed" && job.result && (
            <p className="text-xs text-emerald-700 dark:text-emerald-400">
              {job.result.images_imported !== undefined && (
                <span>{job.result.images_imported} images imported</span>
              )}
              {job.result.annotations_imported !== undefined && (
                <span> with {job.result.annotations_imported} annotations</span>
              )}
            </p>
          )}

          {/* Footer */}
          <div className="flex items-center justify-between pt-1">
            <span className="text-[11px] text-muted-foreground">
              {formatDistanceToNow(new Date(job.created_at), { addSuffix: true })}
            </span>
            <div className="flex gap-1.5">
              {displayInfo.canRetry && (
                <Button
                  variant="outline"
                  size="sm"
                  className="h-7 px-2.5 text-xs font-medium"
                  onClick={() => retryImport.mutate(job.id)}
                  disabled={retryImport.isPending}
                >
                  {retryImport.isPending ? (
                    <Loader2 className="h-3 w-3 animate-spin" />
                  ) : (
                    <>
                      <RefreshCw className="h-3 w-3 mr-1.5" />
                      {displayInfo.checkpointInfo ? "Resume" : "Retry"}
                    </>
                  )}
                </Button>
              )}
              {displayInfo.canCancel && (
                <Button
                  variant="outline"
                  size="sm"
                  className="h-7 px-2.5 text-xs font-medium text-red-600 dark:text-red-400 border-red-200 dark:border-red-800/50 hover:bg-red-50 dark:hover:bg-red-950/30"
                  onClick={() => cancelJob.mutate(job.id)}
                  disabled={cancelJob.isPending}
                >
                  {cancelJob.isPending ? (
                    <Loader2 className="h-3 w-3 animate-spin" />
                  ) : (
                    <>
                      <X className="h-3 w-3 mr-1.5" />
                      Cancel
                    </>
                  )}
                </Button>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export function JobsDrawer() {
  const [open, setOpen] = useState(false);
  const { data: countData } = useActiveJobsCount();
  const { data: jobs, isLoading } = useActiveJobs({ enabled: open });

  const activeCount = countData?.count || 0;
  const activeJobs =
    jobs?.filter(
      (j) =>
        j.status === "running" || j.status === "pending" || j.status === "queued"
    ) || [];
  const recentJobs =
    jobs?.filter(
      (j) =>
        j.status === "completed" ||
        j.status === "failed" ||
        j.status === "cancelled"
    ) || [];

  return (
    <Sheet open={open} onOpenChange={setOpen}>
      <SheetTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          className="relative h-9 w-9 rounded-lg"
        >
          <Bell className="h-[18px] w-[18px]" />
          {activeCount > 0 && (
            <span
              className={`
                absolute -top-0.5 -right-0.5
                min-w-[18px] h-[18px] px-1
                rounded-full bg-blue-500
                text-white text-[10px] font-semibold
                flex items-center justify-center
                ring-2 ring-background
                animate-pulse
              `}
            >
              {activeCount > 9 ? "9+" : activeCount}
            </span>
          )}
        </Button>
      </SheetTrigger>
      <SheetContent className="w-[400px] sm:w-[440px] p-0 flex flex-col">
        {/* Header */}
        <div className="flex-shrink-0 px-6 py-5 border-b">
          <SheetHeader>
            <SheetTitle className="flex items-center justify-between">
              <div className="flex items-center gap-2.5">
                <div className="p-2 rounded-lg bg-primary/10">
                  <Bell className="h-4 w-4 text-primary" />
                </div>
                <span className="text-lg">Jobs</span>
              </div>
              {activeCount > 0 && (
                <Badge
                  variant="secondary"
                  className="bg-blue-50 dark:bg-blue-950/50 text-blue-600 dark:text-blue-400 border-blue-200 dark:border-blue-800/50"
                >
                  <div className="h-1.5 w-1.5 rounded-full bg-blue-500 animate-pulse mr-1.5" />
                  {activeCount} active
                </Badge>
              )}
            </SheetTitle>
          </SheetHeader>
        </div>

        {/* Content */}
        <ScrollArea className="flex-1">
          <div className="pl-6 pr-8 py-4">
            {isLoading ? (
              <div className="flex flex-col items-center justify-center py-16 gap-3">
                <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
                <p className="text-sm text-muted-foreground">Loading jobs...</p>
              </div>
            ) : !jobs || jobs.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-16 text-center">
                <div className="p-4 rounded-full bg-muted/50 mb-4">
                  <Inbox className="h-10 w-10 text-muted-foreground/60" />
                </div>
                <h3 className="font-medium text-foreground mb-1">
                  No recent jobs
                </h3>
                <p className="text-sm text-muted-foreground max-w-[240px]">
                  Jobs from the last 24 hours will appear here
                </p>
              </div>
            ) : (
              <div className="space-y-6">
                {/* Active Jobs Section */}
                {activeJobs.length > 0 && (
                  <section>
                    <div className="flex items-center gap-2 mb-3">
                      <CircleDot className="h-4 w-4 text-blue-500" />
                      <h3 className="text-sm font-semibold text-foreground">
                        Active
                      </h3>
                      <span className="text-xs text-muted-foreground">
                        ({activeJobs.length})
                      </span>
                    </div>
                    <div className="space-y-3">
                      {activeJobs.map((job) => (
                        <JobCard key={job.id} job={job} />
                      ))}
                    </div>
                  </section>
                )}

                {/* Separator */}
                {activeJobs.length > 0 && recentJobs.length > 0 && (
                  <Separator className="my-2" />
                )}

                {/* Recent Jobs Section */}
                {recentJobs.length > 0 && (
                  <section>
                    <div className="flex items-center gap-2 mb-3">
                      <History className="h-4 w-4 text-muted-foreground" />
                      <h3 className="text-sm font-semibold text-foreground">
                        Recent
                      </h3>
                      <span className="text-xs text-muted-foreground">
                        ({recentJobs.length})
                      </span>
                    </div>
                    <div className="space-y-3">
                      {recentJobs.map((job) => (
                        <JobCard key={job.id} job={job} />
                      ))}
                    </div>
                  </section>
                )}
              </div>
            )}
          </div>
        </ScrollArea>
      </SheetContent>
    </Sheet>
  );
}
