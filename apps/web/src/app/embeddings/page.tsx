"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
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
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
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
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  Play,
  Loader2,
  RefreshCw,
  Layers,
  CheckCircle,
  Clock,
  XCircle,
  AlertCircle,
  ImageIcon,
  Package,
} from "lucide-react";
import type { EmbeddingJob, EmbeddingModel } from "@/types";

export default function EmbeddingsPage() {
  const queryClient = useQueryClient();
  const [isStartJobOpen, setIsStartJobOpen] = useState(false);
  const [jobSource, setJobSource] = useState<"cutouts" | "products" | "both">("both");
  const [jobType, setJobType] = useState<"full" | "incremental">("incremental");

  // Fetch cutout stats for progress
  const { data: cutoutStats } = useQuery({
    queryKey: ["cutout-stats"],
    queryFn: () => apiClient.getCutoutStats(),
  });

  // Fetch embedding models
  const { data: models, isLoading: modelsLoading } = useQuery({
    queryKey: ["embedding-models"],
    queryFn: () => apiClient.getEmbeddingModels(),
  });

  // Fetch active model
  const { data: activeModel } = useQuery({
    queryKey: ["active-embedding-model"],
    queryFn: () => apiClient.getActiveEmbeddingModel(),
  });

  // Fetch embedding jobs
  const {
    data: jobs,
    isLoading: jobsLoading,
    isFetching,
  } = useQuery({
    queryKey: ["embedding-jobs"],
    queryFn: () => apiClient.getEmbeddingJobs(),
    refetchInterval: 5000, // Poll every 5 seconds for job updates
  });

  // Start job mutation
  const startJobMutation = useMutation({
    mutationFn: (params: { model_id: string; job_type: "full" | "incremental"; source: "cutouts" | "products" | "both" }) =>
      apiClient.startEmbeddingJob(params),
    onSuccess: () => {
      toast.success("Embedding extraction job started");
      queryClient.invalidateQueries({ queryKey: ["embedding-jobs"] });
      setIsStartJobOpen(false);
    },
    onError: (error) => {
      toast.error(`Failed to start job: ${error.message}`);
    },
  });

  // Activate model mutation
  const activateModelMutation = useMutation({
    mutationFn: (modelId: string) => apiClient.activateEmbeddingModel(modelId),
    onSuccess: () => {
      toast.success("Model activated");
      queryClient.invalidateQueries({ queryKey: ["embedding-models"] });
      queryClient.invalidateQueries({ queryKey: ["active-embedding-model"] });
    },
    onError: (error) => {
      toast.error(`Failed to activate model: ${error.message}`);
    },
  });

  const getJobStatusBadge = (status: string) => {
    switch (status) {
      case "completed":
        return (
          <Badge variant="default" className="bg-green-600">
            <CheckCircle className="h-3 w-3 mr-1" />
            Completed
          </Badge>
        );
      case "running":
        return (
          <Badge variant="default" className="bg-blue-600">
            <Loader2 className="h-3 w-3 mr-1 animate-spin" />
            Running
          </Badge>
        );
      case "pending":
        return (
          <Badge variant="secondary">
            <Clock className="h-3 w-3 mr-1" />
            Pending
          </Badge>
        );
      case "failed":
        return (
          <Badge variant="destructive">
            <XCircle className="h-3 w-3 mr-1" />
            Failed
          </Badge>
        );
      default:
        return <Badge variant="outline">{status}</Badge>;
    }
  };

  const getSourceIcon = (source: string) => {
    switch (source) {
      case "cutouts":
        return <ImageIcon className="h-4 w-4" />;
      case "products":
        return <Package className="h-4 w-4" />;
      default:
        return <Layers className="h-4 w-4" />;
    }
  };

  // Calculate product embedding coverage (would need API endpoint)
  const cutoutCoverage = cutoutStats
    ? Math.round((cutoutStats.with_embedding / cutoutStats.total) * 100) || 0
    : 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold">Embeddings</h1>
          <p className="text-muted-foreground">
            Extract DINOv2 embeddings for similarity matching
          </p>
        </div>
        <div className="flex gap-2">
          <Button
            variant="outline"
            onClick={() => {
              queryClient.invalidateQueries({ queryKey: ["embedding-jobs"] });
              queryClient.invalidateQueries({ queryKey: ["cutout-stats"] });
            }}
            disabled={isFetching}
          >
            <RefreshCw className={`h-4 w-4 mr-2 ${isFetching ? "animate-spin" : ""}`} />
            Refresh
          </Button>
          <Dialog open={isStartJobOpen} onOpenChange={setIsStartJobOpen}>
            <DialogTrigger asChild>
              <Button disabled={!activeModel}>
                <Play className="h-4 w-4 mr-2" />
                Start Extraction
              </Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Start Embedding Extraction</DialogTitle>
                <DialogDescription>
                  Extract embeddings using the active model ({activeModel?.name || "none"})
                </DialogDescription>
              </DialogHeader>
              <div className="space-y-4 py-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium">Source</label>
                  <Select value={jobSource} onValueChange={(v: "cutouts" | "products" | "both") => setJobSource(v)}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="cutouts">Cutouts only</SelectItem>
                      <SelectItem value="products">Products only</SelectItem>
                      <SelectItem value="both">Both (Cutouts + Products)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">Job Type</label>
                  <Select value={jobType} onValueChange={(v: "full" | "incremental") => setJobType(v)}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="incremental">Incremental (new items only)</SelectItem>
                      <SelectItem value="full">Full (re-extract all)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
              <DialogFooter>
                <Button variant="outline" onClick={() => setIsStartJobOpen(false)}>
                  Cancel
                </Button>
                <Button
                  onClick={() =>
                    activeModel &&
                    startJobMutation.mutate({
                      model_id: activeModel.id,
                      job_type: jobType,
                      source: jobSource,
                    })
                  }
                  disabled={startJobMutation.isPending || !activeModel}
                >
                  {startJobMutation.isPending ? (
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <Play className="h-4 w-4 mr-2" />
                  )}
                  Start
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-3 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">Active Model</CardTitle>
          </CardHeader>
          <CardContent>
            {activeModel ? (
              <div>
                <p className="text-lg font-bold">{activeModel.name}</p>
                <p className="text-sm text-muted-foreground">{activeModel.embedding_dim}d vectors</p>
              </div>
            ) : (
              <p className="text-lg font-bold text-orange-600">No active model</p>
            )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">Cutout Coverage</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">{cutoutCoverage}%</p>
            <Progress value={cutoutCoverage} className="h-2 mt-2" />
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">Running Jobs</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">
              {jobs?.filter((j) => j.status === "running").length || 0}
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Models Section */}
      <Card>
        <CardHeader>
          <CardTitle>Embedding Models</CardTitle>
          <CardDescription>
            DINOv2 models available for embedding extraction
          </CardDescription>
        </CardHeader>
        <CardContent>
          {modelsLoading ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-6 w-6 animate-spin" />
            </div>
          ) : models?.length === 0 ? (
            <div className="text-center py-8">
              <Layers className="h-12 w-12 mx-auto text-muted-foreground mb-2" />
              <p className="text-muted-foreground">No embedding models available</p>
              <p className="text-sm text-muted-foreground mt-1">
                Models are auto-registered when first used
              </p>
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Name</TableHead>
                  <TableHead>Dimension</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {models?.map((model: EmbeddingModel) => (
                  <TableRow key={model.id}>
                    <TableCell className="font-medium">{model.name}</TableCell>
                    <TableCell>{model.embedding_dim}d</TableCell>
                    <TableCell>
                      {model.is_matching_active ? (
                        <Badge variant="default" className="bg-green-600">
                          <CheckCircle className="h-3 w-3 mr-1" />
                          Active
                        </Badge>
                      ) : (
                        <Badge variant="secondary">Inactive</Badge>
                      )}
                    </TableCell>
                    <TableCell className="text-right">
                      {!model.is_matching_active && (
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => activateModelMutation.mutate(model.id)}
                          disabled={activateModelMutation.isPending}
                        >
                          Activate
                        </Button>
                      )}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      {/* Jobs Section */}
      <Card>
        <CardHeader>
          <CardTitle>Extraction Jobs</CardTitle>
          <CardDescription>
            Recent embedding extraction jobs and their status
          </CardDescription>
        </CardHeader>
        <CardContent>
          {jobsLoading ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-6 w-6 animate-spin" />
            </div>
          ) : jobs?.length === 0 ? (
            <div className="text-center py-8">
              <AlertCircle className="h-12 w-12 mx-auto text-muted-foreground mb-2" />
              <p className="text-muted-foreground">No extraction jobs yet</p>
              <p className="text-sm text-muted-foreground mt-1">
                Start a job to extract embeddings from cutouts or products
              </p>
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Source</TableHead>
                  <TableHead>Type</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Progress</TableHead>
                  <TableHead>Started</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {jobs?.map((job: EmbeddingJob) => (
                  <TableRow key={job.id}>
                    <TableCell>
                      <div className="flex items-center gap-2">
                        {getSourceIcon(job.source)}
                        <span className="capitalize">{job.source}</span>
                      </div>
                    </TableCell>
                    <TableCell>
                      <Badge variant="outline" className="capitalize">
                        {job.job_type}
                      </Badge>
                    </TableCell>
                    <TableCell>{getJobStatusBadge(job.status)}</TableCell>
                    <TableCell>
                      {job.status === "running" && job.total_images > 0 ? (
                        <div className="w-32">
                          <Progress value={(job.processed_images / job.total_images) * 100} className="h-2" />
                          <span className="text-xs text-muted-foreground">
                            {Math.round((job.processed_images / job.total_images) * 100)}%
                          </span>
                        </div>
                      ) : job.status === "completed" ? (
                        <span className="text-green-600">100%</span>
                      ) : (
                        "-"
                      )}
                    </TableCell>
                    <TableCell className="text-muted-foreground text-sm">
                      {new Date(job.created_at).toLocaleString()}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
