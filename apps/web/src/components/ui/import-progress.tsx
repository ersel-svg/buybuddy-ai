"use client";

import { Progress } from "@/components/ui/progress";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  AlertCircle,
  CheckCircle,
  Loader2,
  RefreshCw,
  XCircle,
  ChevronDown,
  ChevronUp,
} from "lucide-react";
import { useState } from "react";
import type { ImportProgress } from "@/hooks/use-chunked-import";

interface ImportProgressUIProps {
  progress: ImportProgress;
  onRetry?: () => void;
  onCancel?: () => void;
  className?: string;
}

export function ImportProgressUI({
  progress,
  onRetry,
  onCancel,
  className,
}: ImportProgressUIProps) {
  const [showErrors, setShowErrors] = useState(false);

  const percentage =
    progress.totalItems > 0
      ? Math.round((progress.processedItems / progress.totalItems) * 100)
      : 0;

  if (progress.status === "idle") return null;

  const getStatusIcon = () => {
    switch (progress.status) {
      case "importing":
      case "fetching":
        return <Loader2 className="h-4 w-4 animate-spin text-primary" />;
      case "completed":
        return <CheckCircle className="h-4 w-4 text-green-600" />;
      case "error":
        return <AlertCircle className="h-4 w-4 text-amber-600" />;
      case "cancelled":
        return <XCircle className="h-4 w-4 text-red-600" />;
      default:
        return null;
    }
  };

  const getStatusColor = () => {
    switch (progress.status) {
      case "completed":
        return "border-green-200 bg-green-50 dark:border-green-900 dark:bg-green-950";
      case "error":
        return "border-amber-200 bg-amber-50 dark:border-amber-900 dark:bg-amber-950";
      case "cancelled":
        return "border-red-200 bg-red-50 dark:border-red-900 dark:bg-red-950";
      default:
        return "border-border bg-muted/50";
    }
  };

  return (
    <div className={`space-y-3 p-4 rounded-lg border ${getStatusColor()} ${className || ""}`}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {getStatusIcon()}
          <span className="font-medium text-sm">{progress.stage}</span>
        </div>
        <span className="text-xs text-muted-foreground">
          {progress.currentChunk}/{progress.totalChunks} chunks
        </span>
      </div>

      {/* Progress Bar */}
      {(progress.status === "importing" || progress.status === "fetching") && (
        <div className="space-y-1">
          <Progress value={percentage} className="h-2" />
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>
              {progress.processedItems.toLocaleString()} / {progress.totalItems.toLocaleString()} items
            </span>
            <span>{percentage}%</span>
          </div>
        </div>
      )}

      {/* Stats */}
      <div className="flex flex-wrap gap-2">
        {progress.imported > 0 && (
          <Badge variant="secondary" className="text-xs">
            {progress.imported.toLocaleString()} imported
          </Badge>
        )}
        {progress.skipped > 0 && (
          <Badge variant="outline" className="text-xs">
            {progress.skipped.toLocaleString()} skipped
          </Badge>
        )}
        {progress.duplicates > 0 && (
          <Badge variant="outline" className="text-xs text-amber-600 border-amber-300">
            {progress.duplicates.toLocaleString()} duplicates
          </Badge>
        )}
        {progress.labelsCreated > 0 && (
          <Badge variant="outline" className="text-xs text-blue-600 border-blue-300">
            {progress.labelsCreated.toLocaleString()} labels
          </Badge>
        )}
        {progress.classesCreated > 0 && (
          <Badge variant="outline" className="text-xs text-purple-600 border-purple-300">
            {progress.classesCreated.toLocaleString()} classes
          </Badge>
        )}
        {progress.errors.length > 0 && (
          <Badge variant="destructive" className="text-xs">
            {progress.errors.length} errors
          </Badge>
        )}
      </div>

      {/* Actions */}
      <div className="flex items-center gap-2">
        {progress.status === "importing" && onCancel && (
          <Button variant="outline" size="sm" onClick={onCancel}>
            <XCircle className="h-3 w-3 mr-1" />
            Cancel
          </Button>
        )}

        {progress.failedChunks.length > 0 && onRetry && (
          <Button variant="outline" size="sm" onClick={onRetry}>
            <RefreshCw className="h-3 w-3 mr-1" />
            Retry {progress.failedChunks.length} Failed
          </Button>
        )}
      </div>

      {/* Error Details (collapsible) */}
      {progress.errors.length > 0 && (
        <div className="border-t pt-2 mt-2">
          <button
            type="button"
            onClick={() => setShowErrors(!showErrors)}
            className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
          >
            {showErrors ? (
              <ChevronUp className="h-3 w-3" />
            ) : (
              <ChevronDown className="h-3 w-3" />
            )}
            {showErrors ? "Hide" : "Show"} error details ({progress.errors.length})
          </button>

          {showErrors && (
            <div className="mt-2 space-y-1 text-xs max-h-32 overflow-y-auto">
              {progress.errors.slice(0, 10).map((err, i) => (
                <p key={i} className="text-red-600 dark:text-red-400">
                  <span className="font-medium">Chunk {err.chunk + 1}:</span> {err.message}
                </p>
              ))}
              {progress.errors.length > 10 && (
                <p className="text-muted-foreground italic">
                  ...and {progress.errors.length - 10} more errors
                </p>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
