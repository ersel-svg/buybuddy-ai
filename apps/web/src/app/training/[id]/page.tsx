"use client";

import { useParams, useRouter } from "next/navigation";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "@/lib/api-client";
import type {
  TrainingCheckpoint,
  TrainingMetricsHistory,
  TrainingRunStatus,
} from "@/types";
import { useState, useMemo } from "react";
import {
  ArrowLeft,
  Play,
  Square,
  Trash2,
  Download,
  Clock,
  CheckCircle2,
  XCircle,
  Loader2,
  AlertTriangle,
  TrendingDown,
  Target,
  Zap,
  BarChart3,
  RefreshCw,
} from "lucide-react";
import Link from "next/link";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

// Status badge component
function StatusBadge({ status }: { status: TrainingRunStatus }) {
  const statusConfig: Record<
    TrainingRunStatus,
    { color: string; icon: React.ReactNode; label: string }
  > = {
    pending: {
      color: "bg-gray-100 text-gray-700 border-gray-200",
      icon: <Clock className="w-3 h-3" />,
      label: "Pending",
    },
    preparing: {
      color: "bg-blue-100 text-blue-700 border-blue-200",
      icon: <Loader2 className="w-3 h-3 animate-spin" />,
      label: "Preparing",
    },
    running: {
      color: "bg-green-100 text-green-700 border-green-200",
      icon: <Play className="w-3 h-3" />,
      label: "Running",
    },
    completed: {
      color: "bg-emerald-100 text-emerald-700 border-emerald-200",
      icon: <CheckCircle2 className="w-3 h-3" />,
      label: "Completed",
    },
    failed: {
      color: "bg-red-100 text-red-700 border-red-200",
      icon: <XCircle className="w-3 h-3" />,
      label: "Failed",
    },
    cancelled: {
      color: "bg-orange-100 text-orange-700 border-orange-200",
      icon: <AlertTriangle className="w-3 h-3" />,
      label: "Cancelled",
    },
  };

  const config = statusConfig[status];

  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border ${config.color}`}
    >
      {config.icon}
      {config.label}
    </span>
  );
}

// Progress bar component
function ProgressBar({
  current,
  total,
  showLabel = true,
}: {
  current: number;
  total: number;
  showLabel?: boolean;
}) {
  const percentage = total > 0 ? Math.round((current / total) * 100) : 0;

  return (
    <div className="w-full">
      {showLabel && (
        <div className="flex justify-between text-xs text-gray-500 mb-1">
          <span>
            Epoch {current} / {total}
          </span>
          <span>{percentage}%</span>
        </div>
      )}
      <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
        <div
          className="bg-blue-600 h-full rounded-full transition-all duration-500"
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}

// Metrics chart component using recharts
function MetricsChart({
  metricsHistory,
  selectedMetrics,
}: {
  metricsHistory: TrainingMetricsHistory[];
  selectedMetrics: string[];
}) {
  const chartData = useMemo(
    () => [...metricsHistory].sort((a, b) => a.epoch - b.epoch),
    [metricsHistory]
  );

  if (chartData.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-400">
        No metrics data yet
      </div>
    );
  }

  const metricColors: Record<string, string> = {
    train_loss: "#ef4444",
    val_loss: "#3b82f6",
    val_recall_at_1: "#10b981",
    val_recall_at_5: "#8b5cf6",
    learning_rate: "#f59e0b",
  };

  const metricLabels: Record<string, string> = {
    train_loss: "Train Loss",
    val_loss: "Val Loss",
    val_recall_at_1: "Recall@1",
    val_recall_at_5: "Recall@5",
    learning_rate: "Learning Rate",
  };

  return (
    <div className="w-full h-64">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis
            dataKey="epoch"
            tick={{ fontSize: 11 }}
            tickLine={{ stroke: "#9ca3af" }}
          />
          <YAxis
            tick={{ fontSize: 11 }}
            tickLine={{ stroke: "#9ca3af" }}
            tickFormatter={(value) => value < 1 ? value.toFixed(3) : value.toFixed(1)}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "white",
              border: "1px solid #e5e7eb",
              borderRadius: "8px",
              fontSize: "12px",
            }}
            formatter={(value, name) => {
              const numValue = typeof value === "number" ? value : 0;
              return [
                numValue < 1 ? numValue.toFixed(4) : numValue.toFixed(2),
                metricLabels[name as string] || name,
              ];
            }}
          />
          <Legend
            wrapperStyle={{ fontSize: "12px" }}
            formatter={(value) => metricLabels[value] || value}
          />
          {selectedMetrics.map((metric) => (
            <Line
              key={metric}
              type="monotone"
              dataKey={metric}
              stroke={metricColors[metric] || "#6b7280"}
              strokeWidth={2}
              dot={{ r: 3 }}
              activeDot={{ r: 5 }}
              connectNulls
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

// Checkpoint card component
function CheckpointCard({
  checkpoint,
  onDelete,
  isDeleting,
}: {
  checkpoint: TrainingCheckpoint;
  onDelete: (id: string) => void;
  isDeleting: boolean;
}) {
  return (
    <div
      className={`p-4 rounded-lg border ${
        checkpoint.is_best
          ? "border-green-300 bg-green-50"
          : "border-gray-200 bg-white"
      }`}
    >
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="font-medium">Epoch {checkpoint.epoch}</span>
          {checkpoint.is_best && (
            <span className="text-xs bg-green-100 text-green-700 px-2 py-0.5 rounded">
              Best
            </span>
          )}
          {checkpoint.is_final && (
            <span className="text-xs bg-blue-100 text-blue-700 px-2 py-0.5 rounded">
              Final
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {checkpoint.checkpoint_url && (
            <a
              href={checkpoint.checkpoint_url}
              target="_blank"
              rel="noopener noreferrer"
              className="p-1.5 text-gray-500 hover:text-blue-600 hover:bg-blue-50 rounded"
              title="Download checkpoint"
            >
              <Download className="w-4 h-4" />
            </a>
          )}
          <button
            onClick={() => onDelete(checkpoint.id)}
            disabled={isDeleting}
            className="p-1.5 text-gray-500 hover:text-red-600 hover:bg-red-50 rounded disabled:opacity-50"
            title="Delete checkpoint"
          >
            {isDeleting ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Trash2 className="w-4 h-4" />
            )}
          </button>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-2 text-sm text-gray-600">
        {checkpoint.val_loss !== undefined && (
          <div>
            <span className="text-gray-400">Val Loss:</span>{" "}
            {checkpoint.val_loss.toFixed(4)}
          </div>
        )}
        {checkpoint.val_recall_at_1 !== undefined && (
          <div>
            <span className="text-gray-400">Recall@1:</span>{" "}
            {(checkpoint.val_recall_at_1 * 100).toFixed(1)}%
          </div>
        )}
        {checkpoint.file_size_bytes && (
          <div>
            <span className="text-gray-400">Size:</span>{" "}
            {(checkpoint.file_size_bytes / 1024 / 1024).toFixed(1)} MB
          </div>
        )}
      </div>
    </div>
  );
}

export default function TrainingDetailPage() {
  const params = useParams();
  const router = useRouter();
  const queryClient = useQueryClient();
  const runId = params.id as string;

  const [selectedMetrics, setSelectedMetrics] = useState<string[]>([
    "train_loss",
    "val_loss",
    "val_recall_at_1",
  ]);
  const [deletingCheckpointId, setDeletingCheckpointId] = useState<string | null>(null);

  // Fetch training run
  const {
    data: run,
    isLoading: isLoadingRun,
    error: runError,
  } = useQuery({
    queryKey: ["training-run", runId],
    queryFn: () => apiClient.getTrainingRun(runId),
    refetchInterval: (query) => {
      const data = query.state.data;
      // Refetch every 3 seconds if running, otherwise stop
      if (data?.status === "running" || data?.status === "preparing") {
        return 3000;
      }
      return false;
    },
  });

  // Fetch metrics history
  const { data: metricsHistory = [], isLoading: isLoadingMetrics } = useQuery({
    queryKey: ["training-metrics-history", runId],
    queryFn: () => apiClient.getTrainingMetricsHistory(runId),
    refetchInterval: () => {
      // Refetch every 5 seconds if run is active
      if (run?.status === "running" || run?.status === "preparing") {
        return 5000;
      }
      return false;
    },
    enabled: !!run,
  });

  // Fetch checkpoints
  const { data: checkpoints = [], isLoading: isLoadingCheckpoints } = useQuery({
    queryKey: ["training-checkpoints", runId],
    queryFn: () => apiClient.getTrainingCheckpoints(runId),
    refetchInterval: () => {
      if (run?.status === "running" || run?.status === "preparing") {
        return 10000;
      }
      return false;
    },
    enabled: !!run,
  });

  // Cancel mutation
  const cancelMutation = useMutation({
    mutationFn: () => apiClient.cancelTrainingRun(runId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["training-run", runId] });
    },
  });

  // Delete checkpoint mutation
  const deleteCheckpointMutation = useMutation({
    mutationFn: (checkpointId: string) =>
      apiClient.deleteTrainingCheckpoint(checkpointId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["training-checkpoints", runId] });
      setDeletingCheckpointId(null);
    },
    onError: () => {
      setDeletingCheckpointId(null);
    },
  });

  // Delete run mutation
  const deleteRunMutation = useMutation({
    mutationFn: () => apiClient.deleteTrainingRun(runId),
    onSuccess: () => {
      router.push("/training");
    },
  });

  const handleDeleteCheckpoint = (checkpointId: string) => {
    setDeletingCheckpointId(checkpointId);
    deleteCheckpointMutation.mutate(checkpointId);
  };

  const isRunning = run?.status === "running" || run?.status === "preparing";

  if (isLoadingRun) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <Loader2 className="w-8 h-8 animate-spin text-blue-600" />
      </div>
    );
  }

  if (runError || !run) {
    // Log the actual error for debugging (browser console)
    if (runError) {
      console.error("Training run fetch error:", runError);
    }

    return (
      <div className="flex flex-col items-center justify-center min-h-screen gap-4">
        <XCircle className="w-12 h-12 text-red-500" />
        <p className="text-gray-600">
          {runError instanceof Error && runError.message.includes("fetch")
            ? "Unable to connect to server"
            : "Training run not found"}
        </p>
        <Link
          href="/training"
          className="text-blue-600 hover:underline flex items-center gap-1"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Training
        </Link>
      </div>
    );
  }

  const latestMetrics = metricsHistory[metricsHistory.length - 1];

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-6xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link
              href="/training"
              className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
            >
              <ArrowLeft className="w-5 h-5" />
            </Link>
            <div>
              <h1 className="text-2xl font-bold text-gray-900">{run.name}</h1>
              {run.description && (
                <p className="text-gray-500 mt-1">{run.description}</p>
              )}
            </div>
          </div>
          <div className="flex items-center gap-3">
            <StatusBadge status={run.status} />
            {isRunning && (
              <button
                onClick={() => cancelMutation.mutate()}
                disabled={cancelMutation.isPending}
                className="flex items-center gap-2 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:opacity-50"
              >
                {cancelMutation.isPending ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Square className="w-4 h-4" />
                )}
                Cancel
              </button>
            )}
            {!isRunning && run.status !== "pending" && (
              <button
                onClick={() => {
                  if (confirm("Are you sure you want to delete this training run?")) {
                    deleteRunMutation.mutate();
                  }
                }}
                disabled={deleteRunMutation.isPending}
                className="flex items-center gap-2 px-4 py-2 border border-red-300 text-red-600 rounded-lg hover:bg-red-50 disabled:opacity-50"
              >
                {deleteRunMutation.isPending ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Trash2 className="w-4 h-4" />
                )}
                Delete
              </button>
            )}
          </div>
        </div>

        {/* Progress Section (for running jobs) */}
        {isRunning && (
          <div className="bg-white rounded-xl border border-gray-200 p-6">
            <div className="flex items-center gap-2 mb-4">
              <RefreshCw className="w-5 h-5 text-blue-600 animate-spin" />
              <h2 className="text-lg font-semibold">Training Progress</h2>
            </div>
            <ProgressBar current={run.current_epoch} total={run.total_epochs} />

            {latestMetrics && (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4">
                {latestMetrics.train_loss !== undefined && (
                  <div className="p-3 bg-red-50 rounded-lg">
                    <div className="text-xs text-red-600 font-medium">Train Loss</div>
                    <div className="text-lg font-bold text-red-700">
                      {latestMetrics.train_loss.toFixed(4)}
                    </div>
                  </div>
                )}
                {latestMetrics.val_loss !== undefined && (
                  <div className="p-3 bg-blue-50 rounded-lg">
                    <div className="text-xs text-blue-600 font-medium">Val Loss</div>
                    <div className="text-lg font-bold text-blue-700">
                      {latestMetrics.val_loss.toFixed(4)}
                    </div>
                  </div>
                )}
                {latestMetrics.val_recall_at_1 !== undefined && (
                  <div className="p-3 bg-green-50 rounded-lg">
                    <div className="text-xs text-green-600 font-medium">Recall@1</div>
                    <div className="text-lg font-bold text-green-700">
                      {(latestMetrics.val_recall_at_1 * 100).toFixed(1)}%
                    </div>
                  </div>
                )}
                {latestMetrics.learning_rate !== undefined && (
                  <div className="p-3 bg-amber-50 rounded-lg">
                    <div className="text-xs text-amber-600 font-medium">Learning Rate</div>
                    <div className="text-lg font-bold text-amber-700">
                      {latestMetrics.learning_rate.toExponential(2)}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Stats Cards */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-white rounded-xl border border-gray-200 p-4">
            <div className="flex items-center gap-2 text-gray-500 mb-1">
              <Target className="w-4 h-4" />
              <span className="text-sm">Base Model</span>
            </div>
            <div className="font-semibold">{run.base_model_type}</div>
          </div>
          <div className="bg-white rounded-xl border border-gray-200 p-4">
            <div className="flex items-center gap-2 text-gray-500 mb-1">
              <BarChart3 className="w-4 h-4" />
              <span className="text-sm">Classes</span>
            </div>
            <div className="font-semibold">{run.num_classes?.toLocaleString() || "-"}</div>
          </div>
          <div className="bg-white rounded-xl border border-gray-200 p-4">
            <div className="flex items-center gap-2 text-gray-500 mb-1">
              <TrendingDown className="w-4 h-4" />
              <span className="text-sm">Best Val Loss</span>
            </div>
            <div className="font-semibold">
              {run.best_val_loss?.toFixed(4) || "-"}
            </div>
          </div>
          <div className="bg-white rounded-xl border border-gray-200 p-4">
            <div className="flex items-center gap-2 text-gray-500 mb-1">
              <Zap className="w-4 h-4" />
              <span className="text-sm">Best Recall@1</span>
            </div>
            <div className="font-semibold">
              {run.best_val_recall_at_1
                ? `${(run.best_val_recall_at_1 * 100).toFixed(1)}%`
                : "-"}
            </div>
          </div>
        </div>

        {/* Two Column Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Metrics Chart */}
          <div className="bg-white rounded-xl border border-gray-200 p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold flex items-center gap-2">
                <BarChart3 className="w-5 h-5 text-blue-600" />
                Training Metrics
              </h2>
              {isLoadingMetrics && (
                <Loader2 className="w-4 h-4 animate-spin text-gray-400" />
              )}
            </div>

            {/* Metric toggles */}
            <div className="flex flex-wrap gap-2 mb-4">
              {["train_loss", "val_loss", "val_recall_at_1", "val_recall_at_5", "learning_rate"].map(
                (metric) => (
                  <button
                    key={metric}
                    onClick={() =>
                      setSelectedMetrics((prev) =>
                        prev.includes(metric)
                          ? prev.filter((m) => m !== metric)
                          : [...prev, metric]
                      )
                    }
                    className={`px-3 py-1 rounded-full text-xs font-medium transition-colors ${
                      selectedMetrics.includes(metric)
                        ? "bg-blue-100 text-blue-700"
                        : "bg-gray-100 text-gray-500 hover:bg-gray-200"
                    }`}
                  >
                    {metric.replace(/_/g, " ").replace("val ", "Val ")}
                  </button>
                )
              )}
            </div>

            <MetricsChart
              metricsHistory={metricsHistory}
              selectedMetrics={selectedMetrics}
            />
          </div>

          {/* Checkpoints */}
          <div className="bg-white rounded-xl border border-gray-200 p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold">
                Checkpoints ({checkpoints.length})
              </h2>
              {isLoadingCheckpoints && (
                <Loader2 className="w-4 h-4 animate-spin text-gray-400" />
              )}
            </div>

            {checkpoints.length === 0 ? (
              <div className="text-center py-8 text-gray-400">
                No checkpoints yet
              </div>
            ) : (
              <div className="space-y-3 max-h-96 overflow-y-auto">
                {[...checkpoints]
                  .sort((a, b) => b.epoch - a.epoch)
                  .map((checkpoint) => (
                    <CheckpointCard
                      key={checkpoint.id}
                      checkpoint={checkpoint}
                      onDelete={handleDeleteCheckpoint}
                      isDeleting={deletingCheckpointId === checkpoint.id}
                    />
                  ))}
              </div>
            )}
          </div>
        </div>

        {/* Configuration Details */}
        <div className="bg-white rounded-xl border border-gray-200 p-6">
          <h2 className="text-lg font-semibold mb-4">Training Configuration</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <div className="text-sm text-gray-500">Epochs</div>
              <div className="font-medium">
                {run.current_epoch} / {run.total_epochs}
              </div>
            </div>
            <div>
              <div className="text-sm text-gray-500">Batch Size</div>
              <div className="font-medium">
                {run.training_config?.batch_size || "-"}
              </div>
            </div>
            <div>
              <div className="text-sm text-gray-500">Learning Rate</div>
              <div className="font-medium">
                {run.training_config?.learning_rate?.toExponential(2) || "-"}
              </div>
            </div>
            <div>
              <div className="text-sm text-gray-500">Embedding Dim</div>
              <div className="font-medium">
                {run.training_config?.embedding_dim || "-"}
              </div>
            </div>
            <div>
              <div className="text-sm text-gray-500">Train Products</div>
              <div className="font-medium">
                {run.train_product_count?.toLocaleString() || "-"}
              </div>
            </div>
            <div>
              <div className="text-sm text-gray-500">Val Products</div>
              <div className="font-medium">
                {run.val_product_count?.toLocaleString() || "-"}
              </div>
            </div>
            <div>
              <div className="text-sm text-gray-500">Test Products</div>
              <div className="font-medium">
                {run.test_product_count?.toLocaleString() || "-"}
              </div>
            </div>
            <div>
              <div className="text-sm text-gray-500">Data Source</div>
              <div className="font-medium capitalize">
                {run.data_source?.replace(/_/g, " ") || "-"}
              </div>
            </div>
          </div>

          {/* SOTA Config */}
          {run.sota_config?.enabled && (
            <div className="mt-6 pt-6 border-t border-gray-200">
              <h3 className="text-sm font-semibold text-gray-700 mb-3">
                SOTA Features
              </h3>
              <div className="flex flex-wrap gap-2">
                {run.sota_config.use_combined_loss && (
                  <span className="px-2 py-1 bg-purple-100 text-purple-700 rounded text-xs">
                    Combined Loss
                  </span>
                )}
                {run.sota_config.use_pk_sampling && (
                  <span className="px-2 py-1 bg-indigo-100 text-indigo-700 rounded text-xs">
                    P-K Sampling
                  </span>
                )}
                {run.sota_config.use_curriculum && (
                  <span className="px-2 py-1 bg-cyan-100 text-cyan-700 rounded text-xs">
                    Curriculum Learning
                  </span>
                )}
                {run.sota_config.use_domain_adaptation && (
                  <span className="px-2 py-1 bg-teal-100 text-teal-700 rounded text-xs">
                    Domain Adaptation
                  </span>
                )}
                {run.sota_config.use_early_stopping && (
                  <span className="px-2 py-1 bg-amber-100 text-amber-700 rounded text-xs">
                    Early Stopping
                  </span>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Error Message */}
        {run.error_message && (
          <div className="bg-red-50 border border-red-200 rounded-xl p-6">
            <h2 className="text-lg font-semibold text-red-700 mb-2">Error</h2>
            <p className="text-red-600">{run.error_message}</p>
            {run.error_traceback && (
              <pre className="mt-4 p-4 bg-red-100 rounded-lg text-xs text-red-800 overflow-x-auto">
                {run.error_traceback}
              </pre>
            )}
          </div>
        )}

        {/* Timestamps */}
        <div className="text-sm text-gray-400 flex gap-6">
          <div>
            Created:{" "}
            {new Date(run.created_at).toLocaleString()}
          </div>
          {run.started_at && (
            <div>
              Started:{" "}
              {new Date(run.started_at).toLocaleString()}
            </div>
          )}
          {run.completed_at && (
            <div>
              Completed:{" "}
              {new Date(run.completed_at).toLocaleString()}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
