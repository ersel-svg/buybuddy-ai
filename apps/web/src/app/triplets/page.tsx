"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
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
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
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
  DialogHeader,
  DialogTitle,
  DialogTrigger,
  DialogFooter,
} from "@/components/ui/dialog";
import {
  Triangle,
  Play,
  Download,
  Loader2,
  Info,
  Database,
  Trash2,
  Eye,
  RefreshCw,
  CheckCircle,
  XCircle,
  Clock,
  AlertCircle,
  BarChart3,
  ThumbsUp,
  ThumbsDown,
  HelpCircle,
} from "lucide-react";
import type { TripletMiningRun, MinedTriplet, TripletMiningStats } from "@/types";
import { formatDistanceToNow } from "date-fns";
import { toast } from "sonner";

function StatusBadge({ status }: { status: string }) {
  const variants: Record<string, { variant: "default" | "secondary" | "destructive" | "outline"; icon: React.ReactNode }> = {
    pending: { variant: "secondary", icon: <Clock className="h-3 w-3" /> },
    running: { variant: "default", icon: <Loader2 className="h-3 w-3 animate-spin" /> },
    completed: { variant: "outline", icon: <CheckCircle className="h-3 w-3 text-green-500" /> },
    failed: { variant: "destructive", icon: <XCircle className="h-3 w-3" /> },
    cancelled: { variant: "secondary", icon: <AlertCircle className="h-3 w-3" /> },
  };

  const { variant, icon } = variants[status] || variants.pending;

  return (
    <Badge variant={variant} className="flex items-center gap-1">
      {icon}
      {status}
    </Badge>
  );
}

function DifficultyBadge({ difficulty }: { difficulty: string }) {
  const colors: Record<string, string> = {
    hard: "bg-red-100 text-red-800 border-red-200",
    semi_hard: "bg-orange-100 text-orange-800 border-orange-200",
    easy: "bg-green-100 text-green-800 border-green-200",
  };

  return (
    <Badge variant="outline" className={colors[difficulty] || ""}>
      {difficulty.replace("_", " ")}
    </Badge>
  );
}

export default function TripletsPage() {
  const queryClient = useQueryClient();
  const [activeTab, setActiveTab] = useState("runs");

  // Mining configuration
  const [selectedDataset, setSelectedDataset] = useState<string>("");
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [miningName, setMiningName] = useState("");
  const [miningDescription, setMiningDescription] = useState("");
  const [hardNegativeThreshold, setHardNegativeThreshold] = useState([0.7]);
  const [positiveThreshold, setPositiveThreshold] = useState([0.9]);
  const [maxTripletsPerAnchor, setMaxTripletsPerAnchor] = useState([10]);
  const [includeCrossDomain, setIncludeCrossDomain] = useState(true);

  // View triplets state
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [difficultyFilter, setDifficultyFilter] = useState<string>("all");

  // Fetch datasets
  const { data: datasets, isLoading: datasetsLoading } = useQuery({
    queryKey: ["datasets"],
    queryFn: () => apiClient.getDatasets(),
  });

  // Fetch embedding models
  const { data: embeddingModels } = useQuery({
    queryKey: ["embedding-models"],
    queryFn: () => apiClient.getEmbeddingModels(),
  });

  // Fetch mining runs
  const { data: miningRuns, isLoading: runsLoading, refetch: refetchRuns } = useQuery({
    queryKey: ["triplet-mining-runs"],
    queryFn: () => apiClient.getTripletMiningRuns(),
    refetchInterval: 5000, // Poll every 5s for running jobs
  });

  // Fetch triplets for selected run
  const { data: tripletsData, isLoading: tripletsLoading } = useQuery({
    queryKey: ["mined-triplets", selectedRunId, difficultyFilter],
    queryFn: () => apiClient.getMinedTriplets(selectedRunId!, {
      difficulty: difficultyFilter !== "all" ? difficultyFilter : undefined,
      limit: 50,
    }),
    enabled: !!selectedRunId,
  });

  // Fetch stats for selected run
  const { data: runStats } = useQuery({
    queryKey: ["triplet-stats", selectedRunId],
    queryFn: () => apiClient.getTripletMiningStats(selectedRunId!),
    enabled: !!selectedRunId,
  });

  // Fetch feedback stats
  const { data: feedbackStats } = useQuery({
    queryKey: ["feedback-stats"],
    queryFn: () => apiClient.getFeedbackStats(),
  });

  // Start mining mutation
  const startMiningMutation = useMutation({
    mutationFn: () => apiClient.startTripletMining({
      name: miningName || `Mining ${new Date().toLocaleDateString()}`,
      description: miningDescription || undefined,
      dataset_id: selectedDataset || undefined,
      embedding_model_id: selectedModel || undefined,
      hard_negative_threshold: hardNegativeThreshold[0],
      positive_threshold: positiveThreshold[0],
      max_triplets_per_anchor: maxTripletsPerAnchor[0],
      include_cross_domain: includeCrossDomain,
    }),
    onSuccess: () => {
      toast.success("Triplet mining started");
      refetchRuns();
      setMiningName("");
      setMiningDescription("");
    },
    onError: (error: Error) => {
      toast.error(`Failed to start mining: ${error.message}`);
    },
  });

  // Delete run mutation
  const deleteRunMutation = useMutation({
    mutationFn: (runId: string) => apiClient.deleteTripletMiningRun(runId),
    onSuccess: () => {
      toast.success("Mining run deleted");
      refetchRuns();
      if (selectedRunId) setSelectedRunId(null);
    },
    onError: (error: Error) => {
      toast.error(`Failed to delete: ${error.message}`);
    },
  });

  // Export mutation
  const exportMutation = useMutation({
    mutationFn: ({ runId, format }: { runId: string; format: "json" | "csv" }) =>
      apiClient.exportTriplets(runId, format),
    onSuccess: (data) => {
      toast.success(`Exported ${data.total_triplets} triplets`);
      if (data.file_url) {
        window.open(data.file_url, "_blank");
      }
    },
    onError: (error: Error) => {
      toast.error(`Export failed: ${error.message}`);
    },
  });

  const selectedDatasetInfo = datasets?.find((d) => d.id === selectedDataset);
  const activeModel = embeddingModels?.find((m) => m.is_matching_active);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold">Hard Triplet Mining</h1>
          <p className="text-muted-foreground">
            Find confusing product pairs to improve model training
          </p>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={() => refetchRuns()}
        >
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh
        </Button>
      </div>

      {/* Info Card */}
      <Card className="bg-blue-50 border-blue-200">
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center gap-2 text-blue-800">
            <Info className="h-4 w-4" />
            What is Hard Triplet Mining?
          </CardTitle>
        </CardHeader>
        <CardContent className="text-blue-700 text-sm">
          Hard triplet mining finds products that look similar but are different (hard negatives).
          Training on these challenging examples significantly improves model accuracy.
          A triplet consists of: <strong>Anchor</strong> (reference product),
          <strong> Positive</strong> (same product, different view), and
          <strong> Hard Negative</strong> (different product that looks similar).
        </CardContent>
      </Card>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList>
          <TabsTrigger value="runs">Mining Runs</TabsTrigger>
          <TabsTrigger value="new">New Mining Job</TabsTrigger>
          <TabsTrigger value="feedback">Feedback Stats</TabsTrigger>
        </TabsList>

        {/* Mining Runs Tab */}
        <TabsContent value="runs" className="space-y-4">
          {runsLoading ? (
            <div className="flex justify-center py-8">
              <Loader2 className="h-8 w-8 animate-spin" />
            </div>
          ) : !miningRuns?.length ? (
            <Card>
              <CardContent className="text-center py-12">
                <Triangle className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                <p className="text-muted-foreground">No mining runs yet</p>
                <Button
                  className="mt-4"
                  onClick={() => setActiveTab("new")}
                >
                  <Play className="h-4 w-4 mr-2" />
                  Start First Mining Job
                </Button>
              </CardContent>
            </Card>
          ) : (
            <div className="grid gap-4">
              {/* Runs List */}
              <Card>
                <CardHeader>
                  <CardTitle>Mining Runs</CardTitle>
                  <CardDescription>
                    View and manage triplet mining jobs
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Name</TableHead>
                        <TableHead>Status</TableHead>
                        <TableHead>Triplets</TableHead>
                        <TableHead>Hard</TableHead>
                        <TableHead>Cross-Domain</TableHead>
                        <TableHead>Created</TableHead>
                        <TableHead className="text-right">Actions</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {miningRuns.map((run) => (
                        <TableRow key={run.id} className={selectedRunId === run.id ? "bg-muted/50" : ""}>
                          <TableCell>
                            <div>
                              <div className="font-medium">{run.name}</div>
                              {run.description && (
                                <div className="text-xs text-muted-foreground">{run.description}</div>
                              )}
                            </div>
                          </TableCell>
                          <TableCell>
                            <StatusBadge status={run.status} />
                          </TableCell>
                          <TableCell>{run.total_triplets?.toLocaleString() || "-"}</TableCell>
                          <TableCell>{run.hard_triplets?.toLocaleString() || "-"}</TableCell>
                          <TableCell>{run.cross_domain_triplets?.toLocaleString() || "-"}</TableCell>
                          <TableCell className="text-sm text-muted-foreground">
                            {formatDistanceToNow(new Date(run.created_at), { addSuffix: true })}
                          </TableCell>
                          <TableCell className="text-right">
                            <div className="flex gap-2 justify-end">
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => setSelectedRunId(run.id)}
                                disabled={run.status !== "completed"}
                              >
                                <Eye className="h-4 w-4" />
                              </Button>
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => exportMutation.mutate({ runId: run.id, format: "json" })}
                                disabled={run.status !== "completed"}
                              >
                                <Download className="h-4 w-4" />
                              </Button>
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => deleteRunMutation.mutate(run.id)}
                                disabled={run.status === "running"}
                              >
                                <Trash2 className="h-4 w-4 text-destructive" />
                              </Button>
                            </div>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </CardContent>
              </Card>

              {/* Selected Run Details */}
              {selectedRunId && runStats && (
                <Card>
                  <CardHeader>
                    <div className="flex justify-between items-center">
                      <CardTitle>Triplet Statistics</CardTitle>
                      <Select value={difficultyFilter} onValueChange={setDifficultyFilter}>
                        <SelectTrigger className="w-[150px]">
                          <SelectValue placeholder="Filter by difficulty" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="all">All Difficulties</SelectItem>
                          <SelectItem value="hard">Hard Only</SelectItem>
                          <SelectItem value="semi_hard">Semi-Hard</SelectItem>
                          <SelectItem value="easy">Easy</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </CardHeader>
                  <CardContent>
                    {/* Stats Grid */}
                    <div className="grid grid-cols-4 gap-4 mb-6">
                      <div className="text-center p-4 bg-muted rounded-lg">
                        <div className="text-2xl font-bold">{runStats.total_triplets.toLocaleString()}</div>
                        <div className="text-sm text-muted-foreground">Total Triplets</div>
                      </div>
                      <div className="text-center p-4 bg-red-50 rounded-lg">
                        <div className="text-2xl font-bold text-red-700">{runStats.hard_count.toLocaleString()}</div>
                        <div className="text-sm text-muted-foreground">Hard</div>
                      </div>
                      <div className="text-center p-4 bg-orange-50 rounded-lg">
                        <div className="text-2xl font-bold text-orange-700">{runStats.semi_hard_count.toLocaleString()}</div>
                        <div className="text-sm text-muted-foreground">Semi-Hard</div>
                      </div>
                      <div className="text-center p-4 bg-blue-50 rounded-lg">
                        <div className="text-2xl font-bold text-blue-700">{runStats.cross_domain_count.toLocaleString()}</div>
                        <div className="text-sm text-muted-foreground">Cross-Domain</div>
                      </div>
                    </div>

                    <div className="grid grid-cols-3 gap-4 mb-6 text-sm">
                      <div className="flex justify-between p-2 bg-muted/50 rounded">
                        <span className="text-muted-foreground">Avg Margin:</span>
                        <span className="font-medium">{runStats.avg_margin.toFixed(3)}</span>
                      </div>
                      <div className="flex justify-between p-2 bg-muted/50 rounded">
                        <span className="text-muted-foreground">Min Margin:</span>
                        <span className="font-medium">{runStats.min_margin.toFixed(3)}</span>
                      </div>
                      <div className="flex justify-between p-2 bg-muted/50 rounded">
                        <span className="text-muted-foreground">Max Margin:</span>
                        <span className="font-medium">{runStats.max_margin.toFixed(3)}</span>
                      </div>
                    </div>

                    {/* Triplets Table */}
                    {tripletsLoading ? (
                      <div className="flex justify-center py-8">
                        <Loader2 className="h-6 w-6 animate-spin" />
                      </div>
                    ) : (
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>Anchor</TableHead>
                            <TableHead>Positive</TableHead>
                            <TableHead>Negative</TableHead>
                            <TableHead>A-P Sim</TableHead>
                            <TableHead>A-N Sim</TableHead>
                            <TableHead>Margin</TableHead>
                            <TableHead>Difficulty</TableHead>
                            <TableHead>Cross-Domain</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {tripletsData?.triplets.map((triplet) => (
                            <TableRow key={triplet.id}>
                              <TableCell className="font-mono text-xs">
                                {triplet.anchor_product_id.slice(0, 8)}...
                              </TableCell>
                              <TableCell className="font-mono text-xs">
                                {triplet.positive_product_id.slice(0, 8)}...
                              </TableCell>
                              <TableCell className="font-mono text-xs">
                                {triplet.negative_product_id.slice(0, 8)}...
                              </TableCell>
                              <TableCell>{triplet.anchor_positive_sim.toFixed(3)}</TableCell>
                              <TableCell>{triplet.anchor_negative_sim.toFixed(3)}</TableCell>
                              <TableCell>{triplet.margin.toFixed(3)}</TableCell>
                              <TableCell>
                                <DifficultyBadge difficulty={triplet.difficulty} />
                              </TableCell>
                              <TableCell>
                                {triplet.is_cross_domain ? (
                                  <Badge variant="secondary">Yes</Badge>
                                ) : (
                                  <span className="text-muted-foreground">-</span>
                                )}
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    )}
                    {tripletsData && tripletsData.total > 50 && (
                      <p className="text-sm text-muted-foreground text-center mt-4">
                        Showing 50 of {tripletsData.total.toLocaleString()} triplets
                      </p>
                    )}
                  </CardContent>
                </Card>
              )}
            </div>
          )}
        </TabsContent>

        {/* New Mining Job Tab */}
        <TabsContent value="new" className="space-y-4">
          <div className="grid grid-cols-2 gap-6">
            {/* Basic Configuration */}
            <Card>
              <CardHeader>
                <CardTitle>Mining Configuration</CardTitle>
                <CardDescription>
                  Configure the triplet mining job
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label>Job Name</Label>
                  <Input
                    value={miningName}
                    onChange={(e) => setMiningName(e.target.value)}
                    placeholder="e.g., Hard negatives for training v2"
                  />
                </div>

                <div className="space-y-2">
                  <Label>Description (Optional)</Label>
                  <Textarea
                    value={miningDescription}
                    onChange={(e) => setMiningDescription(e.target.value)}
                    placeholder="Describe the purpose of this mining run..."
                    rows={2}
                  />
                </div>

                <div className="space-y-2">
                  <Label>Dataset (Optional)</Label>
                  <Select
                    value={selectedDataset}
                    onValueChange={setSelectedDataset}
                    disabled={datasetsLoading}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Use all products or select a dataset..." />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="">All Products</SelectItem>
                      {datasets?.map((dataset) => (
                        <SelectItem key={dataset.id} value={dataset.id}>
                          <div className="flex items-center gap-2">
                            <Database className="h-4 w-4" />
                            {dataset.name}
                            <Badge variant="secondary" className="ml-2">
                              {dataset.product_count} products
                            </Badge>
                          </div>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label>Embedding Model</Label>
                  <Select
                    value={selectedModel}
                    onValueChange={setSelectedModel}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder={activeModel ? `Active: ${activeModel.name}` : "Select model..."} />
                    </SelectTrigger>
                    <SelectContent>
                      {activeModel && (
                        <SelectItem value="">
                          Active Model: {activeModel.name}
                        </SelectItem>
                      )}
                      {embeddingModels?.map((model) => (
                        <SelectItem key={model.id} value={model.id}>
                          {model.name}
                          {model.is_matching_active && (
                            <Badge variant="default" className="ml-2">Active</Badge>
                          )}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div className="flex items-center justify-between py-2">
                  <div>
                    <Label>Include Cross-Domain Triplets</Label>
                    <p className="text-xs text-muted-foreground">
                      Synthetic anchor with real negative
                    </p>
                  </div>
                  <Switch
                    checked={includeCrossDomain}
                    onCheckedChange={setIncludeCrossDomain}
                  />
                </div>
              </CardContent>
            </Card>

            {/* Threshold Configuration */}
            <Card>
              <CardHeader>
                <CardTitle>Mining Parameters</CardTitle>
                <CardDescription>
                  Configure similarity thresholds for triplet selection
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <Label>Hard Negative Threshold</Label>
                    <span className="text-sm font-medium">{hardNegativeThreshold[0]}</span>
                  </div>
                  <Slider
                    value={hardNegativeThreshold}
                    onValueChange={setHardNegativeThreshold}
                    min={0.5}
                    max={0.95}
                    step={0.05}
                  />
                  <p className="text-xs text-muted-foreground">
                    Products with similarity above this threshold are considered &quot;hard&quot; negatives
                  </p>
                </div>

                <div className="space-y-3">
                  <div className="flex justify-between">
                    <Label>Positive Threshold</Label>
                    <span className="text-sm font-medium">{positiveThreshold[0]}</span>
                  </div>
                  <Slider
                    value={positiveThreshold}
                    onValueChange={setPositiveThreshold}
                    min={0.8}
                    max={1.0}
                    step={0.02}
                  />
                  <p className="text-xs text-muted-foreground">
                    Same-product pairs must have similarity above this to be positive examples
                  </p>
                </div>

                <div className="space-y-3">
                  <div className="flex justify-between">
                    <Label>Max Triplets Per Anchor</Label>
                    <span className="text-sm font-medium">{maxTripletsPerAnchor[0]}</span>
                  </div>
                  <Slider
                    value={maxTripletsPerAnchor}
                    onValueChange={setMaxTripletsPerAnchor}
                    min={1}
                    max={50}
                    step={1}
                  />
                  <p className="text-xs text-muted-foreground">
                    Maximum number of triplets to generate per anchor product
                  </p>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Start Button */}
          <Card>
            <CardContent className="pt-6">
              <div className="flex gap-4">
                <Button
                  className="flex-1"
                  onClick={() => startMiningMutation.mutate()}
                  disabled={startMiningMutation.isPending}
                >
                  {startMiningMutation.isPending ? (
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <Play className="h-4 w-4 mr-2" />
                  )}
                  Start Mining
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Feedback Stats Tab */}
        <TabsContent value="feedback" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="h-5 w-5" />
                Matching Feedback Statistics
              </CardTitle>
              <CardDescription>
                User feedback on matching results for active learning
              </CardDescription>
            </CardHeader>
            <CardContent>
              {feedbackStats ? (
                <div className="space-y-6">
                  <div className="grid grid-cols-4 gap-4">
                    <div className="text-center p-4 bg-muted rounded-lg">
                      <div className="text-2xl font-bold">{feedbackStats.total}</div>
                      <div className="text-sm text-muted-foreground">Total Feedback</div>
                    </div>
                    <div className="text-center p-4 bg-green-50 rounded-lg">
                      <div className="flex items-center justify-center gap-2">
                        <ThumbsUp className="h-5 w-5 text-green-600" />
                        <span className="text-2xl font-bold text-green-700">{feedbackStats.correct}</span>
                      </div>
                      <div className="text-sm text-muted-foreground">Correct</div>
                    </div>
                    <div className="text-center p-4 bg-red-50 rounded-lg">
                      <div className="flex items-center justify-center gap-2">
                        <ThumbsDown className="h-5 w-5 text-red-600" />
                        <span className="text-2xl font-bold text-red-700">{feedbackStats.wrong}</span>
                      </div>
                      <div className="text-sm text-muted-foreground">Wrong</div>
                    </div>
                    <div className="text-center p-4 bg-yellow-50 rounded-lg">
                      <div className="flex items-center justify-center gap-2">
                        <HelpCircle className="h-5 w-5 text-yellow-600" />
                        <span className="text-2xl font-bold text-yellow-700">{feedbackStats.uncertain}</span>
                      </div>
                      <div className="text-sm text-muted-foreground">Uncertain</div>
                    </div>
                  </div>

                  <div className="flex items-center justify-center gap-4 p-6 bg-muted/50 rounded-lg">
                    <div className="text-center">
                      <div className="text-4xl font-bold">
                        {(feedbackStats.accuracy_rate * 100).toFixed(1)}%
                      </div>
                      <div className="text-sm text-muted-foreground">Model Accuracy Rate</div>
                    </div>
                    <div className="h-16 w-px bg-border" />
                    <div className="text-sm text-muted-foreground max-w-xs">
                      Based on user feedback, the model correctly matched{" "}
                      <strong>{feedbackStats.correct}</strong> out of{" "}
                      <strong>{feedbackStats.correct + feedbackStats.wrong}</strong> evaluated predictions.
                    </div>
                  </div>

                  <div className="text-sm text-muted-foreground">
                    <p>
                      <strong>How it works:</strong> When users verify matching predictions in the Matching page,
                      their feedback is recorded here. Wrong predictions become &quot;hard examples&quot; that can be
                      used to generate new triplets for training, improving the model on its weakest predictions.
                    </p>
                  </div>
                </div>
              ) : (
                <div className="text-center py-12">
                  <BarChart3 className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                  <p className="text-muted-foreground">No feedback collected yet</p>
                  <p className="text-sm text-muted-foreground mt-1">
                    Feedback is collected when users verify matching predictions
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
