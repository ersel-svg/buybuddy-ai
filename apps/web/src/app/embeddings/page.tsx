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
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Loader2,
  RefreshCw,
  Layers,
  CheckCircle,
  Clock,
  XCircle,
  AlertCircle,
  ImageIcon,
  Package,
  Database,
  Trash2,
  GraduationCap,
  FlaskConical,
  Download,
} from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import type { EmbeddingJob, EmbeddingModel, CollectionInfo } from "@/types";

// Import new extraction tabs
import { MatchingExtractionTab } from "./components/MatchingExtractionTab";
import { TrainingExtractionTab } from "./components/TrainingExtractionTab";
import { EvaluationExtractionTab } from "./components/EvaluationExtractionTab";

export default function EmbeddingsPage() {
  const queryClient = useQueryClient();
  const [activeTab, setActiveTab] = useState("matching");

  // Export dialog state
  const [exportDialogOpen, setExportDialogOpen] = useState(false);
  const [selectedCollectionForExport, setSelectedCollectionForExport] = useState<string | null>(null);
  const [exportFormat, setExportFormat] = useState<"json" | "numpy" | "faiss">("json");

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

  // Fetch trained models for dropdown
  const { data: trainedModels } = useQuery({
    queryKey: ["trained-models"],
    queryFn: () => apiClient.getTrainedModels(),
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
    refetchInterval: 5000,
  });

  // Fetch Qdrant collections
  const {
    data: collections,
    isLoading: collectionsLoading,
    isFetching: collectionsFetching,
  } = useQuery({
    queryKey: ["qdrant-collections"],
    queryFn: () => apiClient.getQdrantCollections(),
    enabled: activeTab === "collections",
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

  // Delete collection mutation
  const deleteCollectionMutation = useMutation({
    mutationFn: (collectionName: string) => apiClient.deleteQdrantCollection(collectionName),
    onSuccess: () => {
      toast.success("Collection deleted");
      queryClient.invalidateQueries({ queryKey: ["qdrant-collections"] });
    },
    onError: (error) => {
      toast.error(`Failed to delete collection: ${error.message}`);
    },
  });

  // Export collection mutation
  const exportCollectionMutation = useMutation({
    mutationFn: ({ collectionName, format }: { collectionName: string; format: "json" | "numpy" | "faiss" }) =>
      apiClient.exportCollection(collectionName, format),
    onSuccess: (data) => {
      toast.success(`Export completed: ${data.vector_count.toLocaleString()} vectors`);
      setExportDialogOpen(false);
      // Open download in new tab
      if (data.file_url) {
        window.open(data.file_url, "_blank");
      }
    },
    onError: (error) => {
      toast.error(`Export failed: ${error.message}`);
    },
  });

  const handleExportCollection = (collectionName: string) => {
    setSelectedCollectionForExport(collectionName);
    setExportFormat("json");
    setExportDialogOpen(true);
  };

  const handleStartExport = () => {
    if (selectedCollectionForExport) {
      exportCollectionMutation.mutate({
        collectionName: selectedCollectionForExport,
        format: exportFormat,
      });
    }
  };

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

  const getCollectionStatusBadge = (status: string) => {
    switch (status) {
      case "green":
        return <Badge variant="default" className="bg-green-600">Healthy</Badge>;
      case "yellow":
        return <Badge variant="default" className="bg-yellow-600">Warning</Badge>;
      case "red":
        return <Badge variant="destructive">Error</Badge>;
      default:
        return <Badge variant="outline">{status}</Badge>;
    }
  };

  const formatBytes = (bytes: number | undefined | null) => {
    if (!bytes) return "-";
    const gb = bytes / (1024 * 1024 * 1024);
    if (gb >= 1) return `${gb.toFixed(2)} GB`;
    const mb = bytes / (1024 * 1024);
    if (mb >= 1) return `${mb.toFixed(2)} MB`;
    const kb = bytes / 1024;
    return `${kb.toFixed(2)} KB`;
  };

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
            Extract DINOv2 embeddings for matching, training, and evaluation
          </p>
        </div>
        <Button
          variant="outline"
          onClick={() => {
            queryClient.invalidateQueries({ queryKey: ["embedding-jobs"] });
            queryClient.invalidateQueries({ queryKey: ["cutout-stats"] });
            queryClient.invalidateQueries({ queryKey: ["qdrant-collections"] });
            queryClient.invalidateQueries({ queryKey: ["matched-products-stats"] });
          }}
          disabled={isFetching || collectionsFetching}
        >
          <RefreshCw className={`h-4 w-4 mr-2 ${(isFetching || collectionsFetching) ? "animate-spin" : ""}`} />
          Refresh
        </Button>
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

      {/* Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-6">
          <TabsTrigger value="matching" className="flex items-center gap-1">
            <Package className="h-4 w-4" />
            Matching
          </TabsTrigger>
          <TabsTrigger value="training" className="flex items-center gap-1">
            <GraduationCap className="h-4 w-4" />
            Training
          </TabsTrigger>
          <TabsTrigger value="evaluation" className="flex items-center gap-1">
            <FlaskConical className="h-4 w-4" />
            Evaluation
          </TabsTrigger>
          <TabsTrigger value="models" className="flex items-center gap-1">
            <Layers className="h-4 w-4" />
            Models
          </TabsTrigger>
          <TabsTrigger value="jobs" className="flex items-center gap-1">
            <Clock className="h-4 w-4" />
            Jobs
          </TabsTrigger>
          <TabsTrigger value="collections" className="flex items-center gap-1">
            <Database className="h-4 w-4" />
            Collections
          </TabsTrigger>
        </TabsList>

        {/* Matching Extraction Tab */}
        <TabsContent value="matching">
          <MatchingExtractionTab activeModel={activeModel ?? null} models={models} trainedModels={trainedModels} />
        </TabsContent>

        {/* Training Extraction Tab */}
        <TabsContent value="training">
          <TrainingExtractionTab activeModel={activeModel ?? null} models={models} />
        </TabsContent>

        {/* Evaluation Extraction Tab */}
        <TabsContent value="evaluation">
          <EvaluationExtractionTab models={models} />
        </TabsContent>

        {/* Models Tab */}
        <TabsContent value="models">
          <Card>
            <CardHeader>
              <CardTitle>Embedding Models</CardTitle>
              <CardDescription>
                DINOv2, DINOv3, CLIP, and custom fine-tuned models for embedding extraction
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
                      <TableHead>Family</TableHead>
                      <TableHead>Type</TableHead>
                      <TableHead>Dimension</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead className="text-right">Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {models?.map((model: EmbeddingModel) => (
                      <TableRow key={model.id}>
                        <TableCell>
                          <div>
                            <p className="font-medium">{model.name}</p>
                            {model.is_pretrained === false && (
                              <p className="text-xs text-muted-foreground">Fine-tuned</p>
                            )}
                          </div>
                        </TableCell>
                        <TableCell>
                          <Badge variant="outline" className="capitalize">
                            {model.model_family || model.model_type?.split("-")[0] || "custom"}
                          </Badge>
                        </TableCell>
                        <TableCell>
                          <span className="text-sm text-muted-foreground">
                            {model.model_type}
                          </span>
                        </TableCell>
                        <TableCell>{model.embedding_dim}d</TableCell>
                        <TableCell>
                          {model.is_matching_active ? (
                            <Badge variant="default" className="bg-green-600">
                              <CheckCircle className="h-3 w-3 mr-1" />
                              Active
                            </Badge>
                          ) : model.is_default ? (
                            <Badge variant="secondary">Default</Badge>
                          ) : (
                            <Badge variant="outline">Inactive</Badge>
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
        </TabsContent>

        {/* Jobs Tab */}
        <TabsContent value="jobs">
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
        </TabsContent>

        {/* Collections Tab */}
        <TabsContent value="collections">
          <Card>
            <CardHeader>
              <CardTitle>Qdrant Collections</CardTitle>
              <CardDescription>
                Vector database collections storing embeddings
              </CardDescription>
            </CardHeader>
            <CardContent>
              {collectionsLoading ? (
                <div className="flex items-center justify-center py-8">
                  <Loader2 className="h-6 w-6 animate-spin" />
                </div>
              ) : !collections || collections.length === 0 ? (
                <div className="text-center py-8">
                  <Database className="h-12 w-12 mx-auto text-muted-foreground mb-2" />
                  <p className="text-muted-foreground">No collections yet</p>
                  <p className="text-sm text-muted-foreground mt-1">
                    Collections are created when embeddings are extracted
                  </p>
                </div>
              ) : (
                <div className="space-y-4">
                  {collections.map((collection: CollectionInfo) => (
                    <Card key={collection.name}>
                      <CardHeader className="pb-2">
                        <div className="flex items-center justify-between">
                          <CardTitle className="text-lg flex items-center gap-2">
                            <Database className="h-5 w-5" />
                            {collection.name}
                          </CardTitle>
                          {getCollectionStatusBadge(collection.status)}
                        </div>
                        {collection.model_name && (
                          <CardDescription>
                            Model: {collection.model_name}
                          </CardDescription>
                        )}
                      </CardHeader>
                      <CardContent>
                        <div className="grid grid-cols-4 gap-4 text-sm mb-4">
                          <div>
                            <p className="text-muted-foreground">Vectors</p>
                            <p className="font-medium">{collection.vectors_count?.toLocaleString() || 0}</p>
                          </div>
                          <div>
                            <p className="text-muted-foreground">Points</p>
                            <p className="font-medium">{collection.points_count?.toLocaleString() || 0}</p>
                          </div>
                          <div>
                            <p className="text-muted-foreground">Dimension</p>
                            <p className="font-medium">{collection.vector_size}d</p>
                          </div>
                          <div>
                            <p className="text-muted-foreground">Size</p>
                            <p className="font-medium">{formatBytes(collection.size_bytes)}</p>
                          </div>
                        </div>
                        <div className="flex gap-2">
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => handleExportCollection(collection.name)}
                          >
                            <Download className="h-4 w-4 mr-2" />
                            Export
                          </Button>
                          <AlertDialog>
                            <AlertDialogTrigger asChild>
                              <Button variant="destructive" size="sm">
                                <Trash2 className="h-4 w-4 mr-2" />
                                Delete
                              </Button>
                            </AlertDialogTrigger>
                            <AlertDialogContent>
                              <AlertDialogHeader>
                                <AlertDialogTitle>Delete Collection?</AlertDialogTitle>
                                <AlertDialogDescription>
                                  This will permanently delete the collection &quot;{collection.name}&quot; and all
                                  its {collection.vectors_count?.toLocaleString() || 0} embeddings.
                                  This action cannot be undone.
                                </AlertDialogDescription>
                              </AlertDialogHeader>
                              <AlertDialogFooter>
                                <AlertDialogCancel>Cancel</AlertDialogCancel>
                                <AlertDialogAction
                                  onClick={() => deleteCollectionMutation.mutate(collection.name)}
                                  className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                                >
                                  {deleteCollectionMutation.isPending ? (
                                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                                  ) : (
                                    <Trash2 className="h-4 w-4 mr-2" />
                                  )}
                                  Delete
                                </AlertDialogAction>
                              </AlertDialogFooter>
                            </AlertDialogContent>
                          </AlertDialog>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Export Collection Dialog */}
      <Dialog open={exportDialogOpen} onOpenChange={setExportDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Export Collection</DialogTitle>
            <DialogDescription>
              Export embeddings from &quot;{selectedCollectionForExport}&quot; to a file.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label>Export Format</Label>
              <Select
                value={exportFormat}
                onValueChange={(v: "json" | "numpy" | "faiss") => setExportFormat(v)}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="json">
                    <div className="flex flex-col">
                      <span>JSON</span>
                      <span className="text-xs text-muted-foreground">Human-readable, includes metadata</span>
                    </div>
                  </SelectItem>
                  <SelectItem value="numpy">
                    <div className="flex flex-col">
                      <span>NumPy (.npz)</span>
                      <span className="text-xs text-muted-foreground">Compressed, for Python/ML</span>
                    </div>
                  </SelectItem>
                  <SelectItem value="faiss">
                    <div className="flex flex-col">
                      <span>FAISS Index</span>
                      <span className="text-xs text-muted-foreground">Fast similarity search</span>
                    </div>
                  </SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="bg-muted p-3 rounded-md text-sm">
              <p className="font-medium">Format Details:</p>
              {exportFormat === "json" && (
                <p className="text-muted-foreground mt-1">
                  JSON file with vectors, product IDs, and full metadata. Best for inspection and custom processing.
                </p>
              )}
              {exportFormat === "numpy" && (
                <p className="text-muted-foreground mt-1">
                  Compressed .npz file with separate arrays for vectors, IDs, and payloads. Best for ML pipelines.
                </p>
              )}
              {exportFormat === "faiss" && (
                <p className="text-muted-foreground mt-1">
                  FAISS index file + ID mapping JSON. Best for deploying fast similarity search.
                </p>
              )}
            </div>
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setExportDialogOpen(false)}>
              Cancel
            </Button>
            <Button
              onClick={handleStartExport}
              disabled={exportCollectionMutation.isPending}
            >
              {exportCollectionMutation.isPending ? (
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              ) : (
                <Download className="h-4 w-4 mr-2" />
              )}
              Export
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
