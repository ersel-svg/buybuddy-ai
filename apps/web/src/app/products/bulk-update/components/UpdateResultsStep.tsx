"use client";

import {
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Clock,
  RefreshCw,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

interface ExecuteResponse {
  success: boolean;
  updated_count: number;
  failed: Array<{ product_id: string; error: string }>;
  execution_time_ms: number;
}

interface PreviewSummary {
  total_rows: number;
  matched: number;
  not_found: number;
  validation_errors: number;
  will_update: number;
}

interface UpdateResultsStepProps {
  result: ExecuteResponse;
  previewSummary: PreviewSummary;
  onReset: () => void;
}

export function UpdateResultsStep({
  result,
  previewSummary,
  onReset,
}: UpdateResultsStepProps) {
  const successRate =
    previewSummary.will_update > 0
      ? Math.round((result.updated_count / previewSummary.will_update) * 100)
      : 0;

  const formatDuration = (ms: number) => {
    if (ms < 1000) return `${ms}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  };

  return (
    <div className="space-y-6">
      {/* Result Banner */}
      {result.success ? (
        <Alert className="border-green-200 bg-green-50 dark:bg-green-950/20">
          <CheckCircle2 className="h-5 w-5 text-green-600" />
          <AlertDescription className="text-green-800 dark:text-green-200 font-medium">
            Bulk update completed successfully! {result.updated_count} products
            were updated.
          </AlertDescription>
        </Alert>
      ) : (
        <Alert variant="destructive">
          <AlertTriangle className="h-5 w-5" />
          <AlertDescription>
            Bulk update completed with errors. {result.updated_count} products
            updated, {result.failed.length} failed.
          </AlertDescription>
        </Alert>
      )}

      {/* Summary Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="pt-6">
            <div className="text-2xl font-bold">{previewSummary.will_update}</div>
            <div className="text-sm text-muted-foreground">Attempted</div>
          </CardContent>
        </Card>
        <Card className="border-green-200 bg-green-50 dark:bg-green-950/20">
          <CardContent className="pt-6">
            <div className="text-2xl font-bold text-green-600">
              {result.updated_count}
            </div>
            <div className="text-sm text-muted-foreground">Updated</div>
          </CardContent>
        </Card>
        <Card className="border-red-200 bg-red-50 dark:bg-red-950/20">
          <CardContent className="pt-6">
            <div className="text-2xl font-bold text-red-600">
              {result.failed.length}
            </div>
            <div className="text-sm text-muted-foreground">Failed</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center gap-2">
              <Clock className="h-5 w-5 text-muted-foreground" />
              <div className="text-2xl font-bold">
                {formatDuration(result.execution_time_ms)}
              </div>
            </div>
            <div className="text-sm text-muted-foreground">Duration</div>
          </CardContent>
        </Card>
      </div>

      {/* Success Rate */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Update Progress</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span>Success Rate</span>
              <span className="font-medium">{successRate}%</span>
            </div>
            <div className="h-2 bg-muted rounded-full overflow-hidden">
              <div
                className={`h-full transition-all ${
                  successRate === 100
                    ? "bg-green-500"
                    : successRate >= 80
                      ? "bg-yellow-500"
                      : "bg-red-500"
                }`}
                style={{ width: `${successRate}%` }}
              />
            </div>
            <div className="text-xs text-muted-foreground">
              {result.updated_count} of {previewSummary.will_update} products
              updated successfully
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Failed Updates */}
      {result.failed.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <XCircle className="h-5 w-5 text-red-500" />
              Failed Updates ({result.failed.length})
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="border rounded-lg overflow-auto max-h-[300px]">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-[200px]">Product ID</TableHead>
                    <TableHead>Error</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {result.failed.map((failure, index) => (
                    <TableRow key={index}>
                      <TableCell className="font-mono text-sm">
                        {failure.product_id}
                      </TableCell>
                      <TableCell className="text-red-600">
                        {failure.error}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Skipped Items Info */}
      {(previewSummary.not_found > 0 || previewSummary.validation_errors > 0) && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Skipped Items</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2 text-sm">
              {previewSummary.not_found > 0 && (
                <div className="flex items-center gap-2 text-yellow-600">
                  <AlertTriangle className="h-4 w-4" />
                  <span>
                    {previewSummary.not_found} products not found in database
                    (skipped)
                  </span>
                </div>
              )}
              {previewSummary.validation_errors > 0 && (
                <div className="flex items-center gap-2 text-red-600">
                  <XCircle className="h-4 w-4" />
                  <span>
                    {previewSummary.validation_errors} rows had validation errors
                    (skipped)
                  </span>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Actions */}
      <div className="flex justify-center pt-4">
        <Button onClick={onReset} size="lg">
          <RefreshCw className="h-4 w-4 mr-2" />
          Start New Update
        </Button>
      </div>
    </div>
  );
}
