"use client";

import { useState, use } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { useRouter } from "next/navigation";
import { apiClient } from "@/lib/api-client";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  ArrowLeft,
  RefreshCw,
  Loader2,
  Brain,
  CheckCircle,
  XCircle,
  Clock,
  Square,
  StopCircle,
  Activity,
  BarChart3,
  Terminal,
  Settings,
  Database,
  Cpu,
  Timer,
  TrendingUp,
  Award,
  HelpCircle,
} from "lucide-react";
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
import { useMemo } from "react";

// Metric info for OD training
const OD_METRIC_INFO: Record<string, {
  name: string;
  description: string;
  goodRange: string;
  badRange: string;
  direction: "lower" | "higher";
}> = {
  loss: {
    name: "Loss",
    description: "Combined detection loss (classification + box regression + auxiliary losses). Lower values indicate better model fit.",
    goodRange: "< 1.0",
    badRange: "> 5.0",
    direction: "lower",
  },
  map: {
    name: "mAP",
    description: "Mean Average Precision across all IoU thresholds (0.5:0.95). The primary metric for object detection quality.",
    goodRange: "> 50%",
    badRange: "< 20%",
    direction: "higher",
  },
  map_50: {
    name: "mAP@50",
    description: "Mean Average Precision at IoU threshold 0.5. More lenient metric, good for initial training assessment.",
    goodRange: "> 70%",
    badRange: "< 40%",
    direction: "higher",
  },
  map_75: {
    name: "mAP@75",
    description: "Mean Average Precision at IoU threshold 0.75. Stricter metric requiring more precise bounding boxes.",
    goodRange: "> 40%",
    badRange: "< 15%",
    direction: "higher",
  },
};

// Metric tooltip component
function MetricTooltip({ metricKey, children }: { metricKey: string; children: React.ReactNode }) {
  const info = OD_METRIC_INFO[metricKey];
  if (!info) return <>{children}</>;

  return (
    <div className="group relative inline-flex items-center gap-1">
      {children}
      <HelpCircle className="w-3.5 h-3.5 text-gray-400 cursor-help" />
      <div className="absolute bottom-full left-0 mb-2 w-72 p-3 bg-gray-900 text-white text-xs rounded-lg shadow-lg opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-50">
        <div className="font-semibold mb-1">{info.name}</div>
        <p className="text-gray-300 mb-2">{info.description}</p>
        <div className="flex gap-3 text-xs">
          <span className="text-green-400">Good: {info.goodRange}</span>
          <span className="text-red-400">Bad: {info.badRange}</span>
        </div>
        <div className="absolute bottom-0 left-4 transform translate-y-full">
          <div className="border-8 border-transparent border-t-gray-900" />
        </div>
      </div>
    </div>
  );
}

// Metrics chart component for OD training
function ODMetricsChart({
  metrics,
  selectedMetrics,
}: {
  metrics: Array<{ epoch: number; loss?: number; map?: number; map_50?: number; timestamp: string }>;
  selectedMetrics: string[];
}) {
  const chartData = useMemo(
    () => [...metrics].sort((a, b) => a.epoch - b.epoch),
    [metrics]
  );

  if (chartData.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-400 dark:text-gray-500">
        <div className="text-center">
          <BarChart3 className="h-12 w-12 mx-auto mb-2 opacity-50" />
          <p>No metrics data yet</p>
          <p className="text-sm mt-1">Metrics will appear once training starts</p>
        </div>
      </div>
    );
  }

  const metricColors: Record<string, string> = {
    loss: "#ef4444",
    map: "#10b981",
    map_50: "#3b82f6",
    map_75: "#8b5cf6",
  };

  const metricLabels: Record<string, string> = {
    loss: "Loss",
    map: "mAP",
    map_50: "mAP@50",
    map_75: "mAP@75",
  };

  // Determine if we need dual Y-axis (loss vs mAP metrics)
  const hasLoss = selectedMetrics.includes("loss");
  const hasMAPMetrics = selectedMetrics.some(m => m.startsWith("map"));

  return (
    <div className="w-full h-72">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData} margin={{ top: 5, right: 30, left: 10, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#4b5563" strokeOpacity={0.3} />
          <XAxis
            dataKey="epoch"
            tick={{ fontSize: 11, fill: "#9ca3af" }}
            tickLine={{ stroke: "#6b7280" }}
            axisLine={{ stroke: "#6b7280" }}
            label={{ value: "Epoch", position: "insideBottom", offset: -5, fontSize: 11, fill: "#9ca3af" }}
          />
          <YAxis
            yAxisId="left"
            tick={{ fontSize: 11, fill: "#9ca3af" }}
            tickLine={{ stroke: "#6b7280" }}
            axisLine={{ stroke: "#6b7280" }}
            tickFormatter={(value) => value < 1 ? value.toFixed(2) : value.toFixed(1)}
            label={{ value: hasLoss ? "Loss" : "mAP (%)", angle: -90, position: "insideLeft", fontSize: 11, fill: "#9ca3af" }}
          />
          {hasLoss && hasMAPMetrics && (
            <YAxis
              yAxisId="right"
              orientation="right"
              tick={{ fontSize: 11, fill: "#9ca3af" }}
              tickLine={{ stroke: "#6b7280" }}
              axisLine={{ stroke: "#6b7280" }}
              tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
              label={{ value: "mAP (%)", angle: 90, position: "insideRight", fontSize: 11, fill: "#9ca3af" }}
              domain={[0, 1]}
            />
          )}
          <Tooltip
            contentStyle={{
              backgroundColor: "#1f2937",
              border: "1px solid #374151",
              borderRadius: "8px",
              fontSize: "12px",
              color: "#f9fafb",
            }}
            formatter={(value, name) => {
              const numValue = typeof value === "number" ? value : 0;
              const metricName = name as string;
              if (metricName.startsWith("map")) {
                return [`${(numValue * 100).toFixed(2)}%`, metricLabels[metricName] || metricName];
              }
              return [numValue.toFixed(4), metricLabels[metricName] || metricName];
            }}
          />
          <Legend
            wrapperStyle={{ fontSize: "12px", color: "#9ca3af" }}
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
              yAxisId={metric === "loss" ? "left" : (hasLoss && hasMAPMetrics ? "right" : "left")}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export default function ODTrainingDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const router = useRouter();
  const queryClient = useQueryClient();
  const [activeTab, setActiveTab] = useState("overview");
  const [selectedMetrics, setSelectedMetrics] = useState<string[]>(["loss", "map", "map_50"]);

  // Fetch training run details
  const {
    data: training,
    isLoading,
    isFetching,
  } = useQuery({
    queryKey: ["od-training-run", id],
    queryFn: () => apiClient.getODTrainingRun(id),
    refetchInterval: (query) => {
      const data = query.state.data;
      // Auto-refresh while training is in progress
      if (data && ["training", "preparing", "queued", "pending"].includes(data.status)) {
        return 5000;
      }
      return false;
    },
  });

  // Fetch metrics separately for the chart
  const { data: metrics } = useQuery({
    queryKey: ["od-training-metrics", id],
    queryFn: () => apiClient.getODTrainingMetrics(id),
    enabled: !!training,
    refetchInterval: (query) => {
      if (training && ["training"].includes(training.status)) {
        return 10000;
      }
      return false;
    },
  });

  // Fetch logs
  const { data: logs } = useQuery({
    queryKey: ["od-training-logs", id],
    queryFn: () => apiClient.getODTrainingLogs(id, 200),
    enabled: activeTab === "logs",
    refetchInterval: (query) => {
      if (training && ["training", "preparing"].includes(training.status)) {
        return 5000;
      }
      return false;
    },
  });

  // Cancel training mutation
  const cancelTrainingMutation = useMutation({
    mutationFn: () => apiClient.cancelODTrainingRun(id),
    onSuccess: () => {
      toast.success("Training cancelled");
      queryClient.invalidateQueries({ queryKey: ["od-training-run", id] });
      queryClient.invalidateQueries({ queryKey: ["od-training-runs"] });
    },
    onError: (error: Error) => {
      toast.error(`Failed to cancel training: ${error.message}`);
    },
  });

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "completed":
        return (
          <Badge variant="default" className="bg-green-600">
            <CheckCircle className="h-3 w-3 mr-1" />
            Completed
          </Badge>
        );
      case "training":
        return (
          <Badge variant="default" className="bg-blue-600">
            <Loader2 className="h-3 w-3 mr-1 animate-spin" />
            Training
          </Badge>
        );
      case "preparing":
        return (
          <Badge variant="default" className="bg-yellow-600">
            <Loader2 className="h-3 w-3 mr-1 animate-spin" />
            Preparing
          </Badge>
        );
      case "queued":
        return (
          <Badge variant="secondary">
            <Clock className="h-3 w-3 mr-1" />
            Queued
          </Badge>
        );
      case "pending":
        return (
          <Badge variant="outline">
            <Clock className="h-3 w-3 mr-1" />
            Pending
          </Badge>
        );
      case "started":
        return (
          <Badge variant="default" className="bg-blue-600">
            <Loader2 className="h-3 w-3 mr-1 animate-spin" />
            Starting
          </Badge>
        );
      case "downloading":
        return (
          <Badge variant="default" className="bg-purple-600">
            <Loader2 className="h-3 w-3 mr-1 animate-spin" />
            Downloading
          </Badge>
        );
      case "failed":
        return (
          <Badge variant="destructive">
            <XCircle className="h-3 w-3 mr-1" />
            Failed
          </Badge>
        );
      case "cancelled":
        return (
          <Badge variant="outline">
            <Square className="h-3 w-3 mr-1" />
            Cancelled
          </Badge>
        );
      default:
        return <Badge variant="outline">{status}</Badge>;
    }
  };

  const formatDuration = (start?: string, end?: string) => {
    if (!start) return "-";
    const startDate = new Date(start);
    const endDate = end ? new Date(end) : new Date();
    const diff = endDate.getTime() - startDate.getTime();
    const hours = Math.floor(diff / (1000 * 60 * 60));
    const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
    const seconds = Math.floor((diff % (1000 * 60)) / 1000);
    if (hours > 0) return `${hours}h ${minutes}m`;
    if (minutes > 0) return `${minutes}m ${seconds}s`;
    return `${seconds}s`;
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-24">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (!training) {
    return (
      <div className="text-center py-24">
        <Brain className="h-16 w-16 mx-auto text-muted-foreground mb-4" />
        <h3 className="text-lg font-medium">Training run not found</h3>
        <p className="text-muted-foreground mt-1">
          The requested training run could not be found.
        </p>
        <Button className="mt-4" onClick={() => router.push("/od/training")}>
          <ArrowLeft className="h-4 w-4 mr-2" />
          Back to Training
        </Button>
      </div>
    );
  }

  const progress = (training.current_epoch / training.total_epochs) * 100;
  const isRunning = ["training", "preparing", "queued", "pending"].includes(training.status);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-start">
        <div className="flex items-center gap-4">
          <Button variant="ghost" size="icon" onClick={() => router.push("/od/training")}>
            <ArrowLeft className="h-5 w-5" />
          </Button>
          <div>
            <div className="flex items-center gap-3">
              <h1 className="text-2xl font-bold">{training.name}</h1>
              {getStatusBadge(training.status)}
            </div>
            {training.description && (
              <p className="text-muted-foreground mt-1">{training.description}</p>
            )}
          </div>
        </div>
        <div className="flex gap-2">
          <Button
            variant="outline"
            onClick={() => {
              queryClient.invalidateQueries({ queryKey: ["od-training-run", id] });
              queryClient.invalidateQueries({ queryKey: ["od-training-metrics", id] });
            }}
            disabled={isFetching}
          >
            <RefreshCw className={`h-4 w-4 mr-2 ${isFetching ? "animate-spin" : ""}`} />
            Refresh
          </Button>
          {isRunning && (
            <Button
              variant="destructive"
              onClick={() => cancelTrainingMutation.mutate()}
              disabled={cancelTrainingMutation.isPending}
            >
              <StopCircle className="h-4 w-4 mr-2" />
              Cancel Training
            </Button>
          )}
        </div>
      </div>

      {/* Progress Card */}
      <Card>
        <CardContent className="pt-6">
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-sm font-medium">Training Progress</span>
              <span className="text-sm text-muted-foreground">
                Epoch {training.current_epoch} / {training.total_epochs}
              </span>
            </div>
            <Progress value={progress} className="h-3" />
            <div className="grid grid-cols-4 gap-4 pt-2">
              <div className="text-center">
                <p className="text-2xl font-bold">{progress.toFixed(1)}%</p>
                <p className="text-xs text-muted-foreground">Progress</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold">
                  {training.best_map !== null && training.best_map !== undefined
                    ? `${(training.best_map * 100).toFixed(1)}%`
                    : "-"}
                </p>
                <p className="text-xs text-muted-foreground">Best mAP</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold">{training.best_epoch ?? "-"}</p>
                <p className="text-xs text-muted-foreground">Best Epoch</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold">
                  {formatDuration(training.started_at, training.completed_at)}
                </p>
                <p className="text-xs text-muted-foreground">Duration</p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Error Message */}
      {training.error_message && (
        <Card className="border-red-500 bg-red-50 dark:bg-red-950/20">
          <CardContent className="pt-6">
            <div className="flex items-start gap-3">
              <XCircle className="h-5 w-5 text-red-600 mt-0.5" />
              <div>
                <p className="font-medium text-red-600">Training Failed</p>
                <p className="text-sm text-red-600/80 mt-1">{training.error_message}</p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList>
          <TabsTrigger value="overview" className="flex items-center gap-2">
            <Activity className="h-4 w-4" />
            Overview
          </TabsTrigger>
          <TabsTrigger value="metrics" className="flex items-center gap-2">
            <BarChart3 className="h-4 w-4" />
            Metrics
          </TabsTrigger>
          <TabsTrigger value="logs" className="flex items-center gap-2">
            <Terminal className="h-4 w-4" />
            Logs
          </TabsTrigger>
          <TabsTrigger value="config" className="flex items-center gap-2">
            <Settings className="h-4 w-4" />
            Configuration
          </TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="mt-6 space-y-6">
          {/* Stats Grid */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm flex items-center gap-2">
                  <Cpu className="h-4 w-4" />
                  Model
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-lg font-semibold">{training.model_type.toUpperCase()}</p>
                <p className="text-xs text-muted-foreground capitalize">{training.model_size}</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm flex items-center gap-2">
                  <Database className="h-4 w-4" />
                  Dataset
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-lg font-semibold truncate">{training.dataset_id.slice(0, 8)}...</p>
                <p className="text-xs text-muted-foreground">
                  {training.dataset_version_id ? `Version: ${training.dataset_version_id.slice(0, 8)}` : "Latest"}
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm flex items-center gap-2">
                  <Timer className="h-4 w-4" />
                  Started
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-lg font-semibold">
                  {training.started_at
                    ? new Date(training.started_at).toLocaleDateString()
                    : "-"}
                </p>
                <p className="text-xs text-muted-foreground">
                  {training.started_at
                    ? new Date(training.started_at).toLocaleTimeString()
                    : "Not started"}
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm flex items-center gap-2">
                  <Award className="h-4 w-4" />
                  Best Result
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-lg font-semibold">
                  {training.best_map !== null && training.best_map !== undefined
                    ? `${(training.best_map * 100).toFixed(2)}% mAP`
                    : "-"}
                </p>
                <p className="text-xs text-muted-foreground">
                  {training.best_epoch !== null ? `at epoch ${training.best_epoch}` : "No results yet"}
                </p>
              </CardContent>
            </Card>
          </div>

          {/* Recent Metrics */}
          {metrics && metrics.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Recent Metrics</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {metrics.slice(-5).reverse().map((m, idx) => (
                    <div
                      key={idx}
                      className="flex justify-between items-center py-2 border-b last:border-0"
                    >
                      <span className="text-sm font-medium">Epoch {m.epoch}</span>
                      <div className="flex gap-4 text-sm">
                        {m.loss != null && (
                          <span className="text-muted-foreground">
                            Loss: <span className="font-medium">{m.loss.toFixed(4)}</span>
                          </span>
                        )}
                        {m.map != null && (
                          <span className="text-muted-foreground">
                            mAP: <span className="font-medium text-green-600">{(m.map * 100).toFixed(2)}%</span>
                          </span>
                        )}
                        {m.map_50 != null && (
                          <span className="text-muted-foreground">
                            mAP@50: <span className="font-medium">{(m.map_50 * 100).toFixed(2)}%</span>
                          </span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="metrics" className="mt-6 space-y-6">
          {/* Metrics Chart */}
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="flex items-center gap-2">
                    <TrendingUp className="h-5 w-5" />
                    Training Metrics
                  </CardTitle>
                  <CardDescription>Loss and mAP over training epochs</CardDescription>
                </div>
              </div>

              {/* Metric toggles */}
              {metrics && metrics.length > 0 && (
                <div className="flex flex-wrap gap-2 mt-4">
                  {["loss", "map", "map_50", "map_75"].map((metric) => {
                    const info = OD_METRIC_INFO[metric];
                    return (
                      <button
                        key={metric}
                        onClick={() =>
                          setSelectedMetrics((prev) =>
                            prev.includes(metric)
                              ? prev.filter((m) => m !== metric)
                              : [...prev, metric]
                          )
                        }
                        className={`px-3 py-1.5 rounded-full text-xs font-medium transition-colors ${
                          selectedMetrics.includes(metric)
                            ? metric === "loss"
                              ? "bg-red-100 text-red-700 dark:bg-red-900/50 dark:text-red-300"
                              : metric === "map"
                                ? "bg-green-100 text-green-700 dark:bg-green-900/50 dark:text-green-300"
                                : metric === "map_50"
                                  ? "bg-blue-100 text-blue-700 dark:bg-blue-900/50 dark:text-blue-300"
                                  : "bg-purple-100 text-purple-700 dark:bg-purple-900/50 dark:text-purple-300"
                            : "bg-gray-100 text-gray-500 hover:bg-gray-200 dark:bg-gray-700 dark:text-gray-400 dark:hover:bg-gray-600"
                        }`}
                      >
                        {info?.name || metric}
                      </button>
                    );
                  })}
                </div>
              )}
            </CardHeader>
            <CardContent>
              <ODMetricsChart metrics={metrics || []} selectedMetrics={selectedMetrics} />
            </CardContent>
          </Card>

          {/* Metrics Table */}
          {metrics && metrics.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Metrics History</CardTitle>
                <CardDescription>Detailed metrics for each epoch</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto max-h-96 overflow-y-auto">
                  <table className="w-full text-sm">
                    <thead className="sticky top-0 bg-background">
                      <tr className="border-b">
                        <th className="text-left py-2 px-2 font-medium">Epoch</th>
                        <th className="text-left py-2 px-2 font-medium">
                          <MetricTooltip metricKey="loss">Loss</MetricTooltip>
                        </th>
                        <th className="text-left py-2 px-2 font-medium">
                          <MetricTooltip metricKey="map">mAP</MetricTooltip>
                        </th>
                        <th className="text-left py-2 px-2 font-medium">
                          <MetricTooltip metricKey="map_50">mAP@50</MetricTooltip>
                        </th>
                        <th className="text-left py-2 px-2 font-medium">Timestamp</th>
                      </tr>
                    </thead>
                    <tbody>
                      {[...metrics].reverse().map((m, idx) => (
                        <tr key={idx} className="border-b last:border-0 hover:bg-muted/50">
                          <td className="py-2 px-2 font-medium">{m.epoch}</td>
                          <td className="py-2 px-2 font-mono text-red-600 dark:text-red-400">
                            {m.loss != null ? m.loss.toFixed(4) : "-"}
                          </td>
                          <td className="py-2 px-2 font-mono text-green-600 dark:text-green-400">
                            {m.map != null ? `${(m.map * 100).toFixed(2)}%` : "-"}
                          </td>
                          <td className="py-2 px-2 font-mono text-blue-600 dark:text-blue-400">
                            {m.map_50 != null ? `${(m.map_50 * 100).toFixed(2)}%` : "-"}
                          </td>
                          <td className="py-2 px-2 text-muted-foreground">
                            {new Date(m.timestamp).toLocaleTimeString()}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="logs" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Terminal className="h-5 w-5" />
                Training Logs
              </CardTitle>
              <CardDescription>Real-time training output</CardDescription>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-[500px] w-full rounded-md border bg-slate-950 p-4">
                {!logs || logs.length === 0 ? (
                  <p className="text-slate-400 text-sm">No logs available yet...</p>
                ) : (
                  <pre className="text-xs text-slate-300 font-mono whitespace-pre-wrap">
                    {logs.join("\n")}
                  </pre>
                )}
              </ScrollArea>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="config" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Settings className="h-5 w-5" />
                Training Configuration
              </CardTitle>
              <CardDescription>Parameters used for this training run</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-6">
                <div className="space-y-4">
                  <h4 className="font-medium text-sm text-muted-foreground">Model Settings</h4>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm">Model Type</span>
                      <span className="text-sm font-medium">{training.model_type.toUpperCase()}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm">Model Size</span>
                      <span className="text-sm font-medium capitalize">{training.model_size}</span>
                    </div>
                  </div>
                </div>

                <div className="space-y-4">
                  <h4 className="font-medium text-sm text-muted-foreground">Training Parameters</h4>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm">Total Epochs</span>
                      <span className="text-sm font-medium">{training.total_epochs}</span>
                    </div>
                    {training.config && (
                      <>
                        <div className="flex justify-between">
                          <span className="text-sm">Batch Size</span>
                          <span className="text-sm font-medium">
                            {String((training.config as Record<string, unknown>).batch_size ?? "-")}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm">Learning Rate</span>
                          <span className="text-sm font-medium">
                            {String((training.config as Record<string, unknown>).learning_rate ?? "-")}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm">Image Size</span>
                          <span className="text-sm font-medium">
                            {String((training.config as Record<string, unknown>).image_size ?? "-")}px
                          </span>
                        </div>
                      </>
                    )}
                  </div>
                </div>
              </div>

              {/* Raw Config JSON */}
              <div className="mt-6 pt-6 border-t">
                <h4 className="font-medium text-sm text-muted-foreground mb-2">Raw Configuration</h4>
                <pre className="text-xs bg-muted p-4 rounded-md overflow-x-auto">
                  {JSON.stringify(training.config, null, 2)}
                </pre>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
