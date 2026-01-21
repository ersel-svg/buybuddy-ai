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
  ChevronRight,
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

const statusConfig: Record<string, { icon: typeof Clock; color: string; bgColor: string; borderColor: string; label: string; animate?: boolean }> = {
  pending: { icon: Clock, color: "text-yellow-500", bgColor: "bg-yellow-50", borderColor: "border-yellow-200", label: "Pending" },
  queued: { icon: Clock, color: "text-blue-500", bgColor: "bg-blue-50", borderColor: "border-blue-200", label: "Queued" },
  running: { icon: Loader2, color: "text-blue-500", bgColor: "bg-blue-50", borderColor: "border-blue-200", label: "Running", animate: true },
  completed: { icon: CheckCircle, color: "text-green-500", bgColor: "bg-green-50", borderColor: "border-green-200", label: "Completed" },
  failed: { icon: XCircle, color: "text-red-500", bgColor: "bg-red-50", borderColor: "border-red-200", label: "Failed" },
  cancelled: { icon: X, color: "text-gray-500", bgColor: "bg-gray-50", borderColor: "border-gray-200", label: "Cancelled" },
};

const iconMap = { import: Package, training: Brain, export: Download, ai: Sparkles, sync: Cloud, default: Clock };

function JobCard({ job }: { job: Job }) {
  const cancelJob = useCancelJob();
  const retryImport = useRetryRoboflowImport();
  const displayInfo = getJobDisplayInfo(job);
  const status = statusConfig[job.status] || statusConfig.pending;
  const StatusIcon = status.icon;
  const TypeIcon = iconMap[displayInfo.icon] || iconMap.default;
  const isActive = job.status === "running" || job.status === "pending" || job.status === "queued";

  return (
    <div className={`p-3 rounded-lg border ${status.borderColor} ${status.bgColor} transition-all`}>
      <div className="flex items-start gap-3">
        <div className="mt-0.5"><TypeIcon className="h-5 w-5 text-muted-foreground" /></div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="font-medium text-sm truncate">{displayInfo.title}</span>
            <Badge variant="outline" className={`text-xs ${status.color} border-current`}>
              <StatusIcon className={`h-3 w-3 mr-1 ${"animate" in status && status.animate ? "animate-spin" : ""}`} />
              {status.label}
            </Badge>
          </div>
          {displayInfo.subtitle && <p className="text-xs text-muted-foreground mt-0.5 truncate">{displayInfo.subtitle}</p>}
          {isActive && job.progress > 0 && (
            <div className="mt-2">
              <div className="flex items-center justify-between text-xs mb-1">
                <span className="text-muted-foreground">{job.result?.stage || "Processing"}</span>
                <span className="font-medium">{job.progress}%</span>
              </div>
              <Progress value={job.progress} className="h-1.5" />
            </div>
          )}
          {job.status === "failed" && job.error && <p className="text-xs text-red-600 mt-1 line-clamp-2">{job.error}</p>}
          {job.status === "completed" && job.result && (
            <p className="text-xs text-green-700 mt-1">
              {job.result.images_imported !== undefined && <span>{job.result.images_imported} images imported</span>}
              {job.result.annotations_imported !== undefined && <span> with {job.result.annotations_imported} annotations</span>}
            </p>
          )}
          <div className="flex items-center justify-between mt-2">
            <span className="text-xs text-muted-foreground">{formatDistanceToNow(new Date(job.created_at), { addSuffix: true })}</span>
            <div className="flex gap-1">
              {displayInfo.canRetry && (
                <Button variant="ghost" size="sm" className="h-6 px-2 text-xs" onClick={() => retryImport.mutate(job.id)} disabled={retryImport.isPending}>
                  {retryImport.isPending ? <Loader2 className="h-3 w-3 animate-spin" /> : <><RefreshCw className="h-3 w-3 mr-1" />Retry</>}
                </Button>
              )}
              {displayInfo.canCancel && (
                <Button variant="ghost" size="sm" className="h-6 px-2 text-xs text-red-600 hover:text-red-700 hover:bg-red-50" onClick={() => cancelJob.mutate(job.id)} disabled={cancelJob.isPending}>
                  {cancelJob.isPending ? <Loader2 className="h-3 w-3 animate-spin" /> : <><X className="h-3 w-3 mr-1" />Cancel</>}
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
  const activeJobs = jobs?.filter((j) => j.status === "running" || j.status === "pending" || j.status === "queued") || [];
  const recentJobs = jobs?.filter((j) => j.status === "completed" || j.status === "failed" || j.status === "cancelled") || [];

  return (
    <Sheet open={open} onOpenChange={setOpen}>
      <SheetTrigger asChild>
        <Button variant="ghost" size="sm" className="relative">
          <Bell className="h-5 w-5" />
          {activeCount > 0 && (
            <span className="absolute -top-1 -right-1 h-5 w-5 rounded-full bg-blue-500 text-white text-xs flex items-center justify-center font-medium animate-pulse">
              {activeCount > 9 ? "9+" : activeCount}
            </span>
          )}
        </Button>
      </SheetTrigger>
      <SheetContent className="w-[400px] sm:w-[450px]">
        <SheetHeader>
          <SheetTitle className="flex items-center gap-2">
            <Bell className="h-5 w-5" />
            Jobs
            {activeCount > 0 && <Badge variant="secondary" className="ml-2">{activeCount} active</Badge>}
          </SheetTitle>
        </SheetHeader>
        <ScrollArea className="h-[calc(100vh-100px)] mt-4 pr-4">
          {isLoading ? (
            <div className="flex items-center justify-center py-12"><Loader2 className="h-8 w-8 animate-spin text-muted-foreground" /></div>
          ) : !jobs || jobs.length === 0 ? (
            <div className="text-center py-12 text-muted-foreground">
              <Clock className="h-12 w-12 mx-auto mb-4 opacity-50" />
              <p className="font-medium">No recent jobs</p>
              <p className="text-sm mt-1">Jobs from the last 24 hours will appear here</p>
            </div>
          ) : (
            <div className="space-y-6">
              {activeJobs.length > 0 && (
                <div>
                  <div className="flex items-center gap-2 mb-3">
                    <div className="h-2 w-2 rounded-full bg-blue-500 animate-pulse" />
                    <h3 className="text-sm font-medium text-muted-foreground">Active ({activeJobs.length})</h3>
                  </div>
                  <div className="space-y-2">{activeJobs.map((job) => <JobCard key={job.id} job={job} />)}</div>
                </div>
              )}
              {recentJobs.length > 0 && (
                <div>
                  <div className="flex items-center gap-2 mb-3">
                    <ChevronRight className="h-4 w-4 text-muted-foreground" />
                    <h3 className="text-sm font-medium text-muted-foreground">Recent ({recentJobs.length})</h3>
                  </div>
                  <div className="space-y-2">{recentJobs.map((job) => <JobCard key={job.id} job={job} />)}</div>
                </div>
              )}
            </div>
          )}
        </ScrollArea>
      </SheetContent>
    </Sheet>
  );
}
