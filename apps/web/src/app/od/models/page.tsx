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
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  RefreshCw,
  Loader2,
  Cpu,
  Search,
  Download,
  MoreHorizontal,
  Trash2,
  Star,
  StarOff,
  ExternalLink,
  BarChart3,
  Clock,
  FileCode,
  Layers,
  Rocket,
  Copy,
  CheckCircle,
} from "lucide-react";
import Link from "next/link";

interface TrainedModel {
  id: string;
  name: string;
  version: string;
  architecture: string;
  dataset_id: string;
  dataset_name?: string;
  training_run_id: string;
  metrics: {
    map50: number;
    map50_95: number;
    precision: number;
    recall: number;
    inference_time_ms: number;
  };
  file_size_mb: number;
  class_count: number;
  is_deployed: boolean;
  is_starred: boolean;
  created_at: string;
  export_formats: string[];
}

export default function ODModelsPage() {
  const queryClient = useQueryClient();
  const [search, setSearch] = useState("");
  const [filterArchitecture, setFilterArchitecture] = useState<string>("all");
  const [selectedModel, setSelectedModel] = useState<TrainedModel | null>(null);
  const [isDetailsOpen, setIsDetailsOpen] = useState(false);

  // Fetch models
  const { data: models, isLoading, isFetching } = useQuery({
    queryKey: ["od-models", filterArchitecture],
    queryFn: async () => {
      // This would fetch from API - for now return mock data
      return [] as TrainedModel[];
    },
  });

  // Filter models
  const filteredModels = models?.filter((m) =>
    m.name.toLowerCase().includes(search.toLowerCase()) ||
    m.architecture.toLowerCase().includes(search.toLowerCase())
  );

  // Toggle star mutation (placeholder)
  const toggleStarMutation = useMutation({
    mutationFn: async (modelId: string) => {
      toast.info("Model starring not yet implemented");
      return { id: modelId };
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["od-models"] });
    },
  });

  // Delete model mutation (placeholder)
  const deleteModelMutation = useMutation({
    mutationFn: async (modelId: string) => {
      toast.info("Model deletion not yet implemented");
      return { id: modelId };
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["od-models"] });
    },
  });

  const handleViewDetails = (model: TrainedModel) => {
    setSelectedModel(model);
    setIsDetailsOpen(true);
  };

  const handleDelete = (model: TrainedModel) => {
    if (!confirm(`Delete model "${model.name}"? This cannot be undone.`)) return;
    deleteModelMutation.mutate(model.id);
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    toast.success("Copied to clipboard");
  };

  // Stats
  const totalModels = models?.length ?? 0;
  const deployedModels = models?.filter((m) => m.is_deployed).length ?? 0;
  const starredModels = models?.filter((m) => m.is_starred).length ?? 0;
  const avgMap50 = models && models.length > 0
    ? (models.reduce((sum, m) => sum + m.metrics.map50, 0) / models.length).toFixed(3)
    : "-";

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold">Models</h1>
          <p className="text-muted-foreground">
            View and manage trained detection models
          </p>
        </div>
        <div className="flex gap-2">
          <Button
            variant="outline"
            onClick={() => queryClient.invalidateQueries({ queryKey: ["od-models"] })}
            disabled={isFetching}
          >
            <RefreshCw className={`h-4 w-4 mr-2 ${isFetching ? "animate-spin" : ""}`} />
            Refresh
          </Button>
          <Button asChild>
            <Link href="/od/training">
              <Rocket className="h-4 w-4 mr-2" />
              Train New Model
            </Link>
          </Button>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">Total Models</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">{totalModels}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">Deployed</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-green-600">{deployedModels}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">Starred</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-yellow-600">{starredModels}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">Avg mAP50</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-blue-600">{avgMap50}</p>
          </CardContent>
        </Card>
      </div>

      {/* Models Table */}
      <Card>
        <CardHeader>
          <CardTitle>Model Registry</CardTitle>
          <CardDescription>
            All trained models ready for deployment
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Filters */}
          <div className="flex gap-4">
            <div className="flex-1">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search models..."
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  className="pl-10"
                />
              </div>
            </div>
            <Select value={filterArchitecture} onValueChange={setFilterArchitecture}>
              <SelectTrigger className="w-[180px]">
                <SelectValue placeholder="Architecture" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Architectures</SelectItem>
                <SelectItem value="rf-detr">RF-DETR</SelectItem>
                <SelectItem value="rt-detr">RT-DETR</SelectItem>
                <SelectItem value="yolo-nas">YOLO-NAS</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Table */}
          {isLoading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
            </div>
          ) : filteredModels?.length === 0 ? (
            <div className="text-center py-12">
              <Cpu className="h-16 w-16 mx-auto text-muted-foreground mb-4" />
              <h3 className="text-lg font-medium">No models found</h3>
              <p className="text-muted-foreground mt-1">
                {search || filterArchitecture !== "all"
                  ? "Try adjusting your filters"
                  : "Train your first model to see it here"}
              </p>
              <Button className="mt-4" asChild>
                <Link href="/od/training">
                  <Rocket className="h-4 w-4 mr-2" />
                  Start Training
                </Link>
              </Button>
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-8"></TableHead>
                  <TableHead>Name</TableHead>
                  <TableHead>Architecture</TableHead>
                  <TableHead>mAP50</TableHead>
                  <TableHead>mAP50-95</TableHead>
                  <TableHead>Inference</TableHead>
                  <TableHead>Size</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead className="w-8"></TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filteredModels?.map((model) => (
                  <TableRow
                    key={model.id}
                    className="cursor-pointer"
                    onClick={() => handleViewDetails(model)}
                  >
                    <TableCell onClick={(e) => e.stopPropagation()}>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-8 w-8"
                        onClick={() => toggleStarMutation.mutate(model.id)}
                      >
                        {model.is_starred ? (
                          <Star className="h-4 w-4 fill-yellow-500 text-yellow-500" />
                        ) : (
                          <StarOff className="h-4 w-4 text-muted-foreground" />
                        )}
                      </Button>
                    </TableCell>
                    <TableCell>
                      <div>
                        <p className="font-medium">{model.name}</p>
                        <p className="text-xs text-muted-foreground">v{model.version}</p>
                      </div>
                    </TableCell>
                    <TableCell>
                      <Badge variant="outline">{model.architecture}</Badge>
                    </TableCell>
                    <TableCell className="font-mono">
                      {model.metrics.map50.toFixed(3)}
                    </TableCell>
                    <TableCell className="font-mono">
                      {model.metrics.map50_95.toFixed(3)}
                    </TableCell>
                    <TableCell className="font-mono">
                      {model.metrics.inference_time_ms}ms
                    </TableCell>
                    <TableCell>
                      {model.file_size_mb.toFixed(1)} MB
                    </TableCell>
                    <TableCell>
                      {model.is_deployed ? (
                        <Badge variant="default" className="bg-green-600">
                          <CheckCircle className="h-3 w-3 mr-1" />
                          Deployed
                        </Badge>
                      ) : (
                        <Badge variant="secondary">Ready</Badge>
                      )}
                    </TableCell>
                    <TableCell onClick={(e) => e.stopPropagation()}>
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button variant="ghost" size="icon">
                            <MoreHorizontal className="h-4 w-4" />
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end">
                          <DropdownMenuItem onClick={() => handleViewDetails(model)}>
                            <BarChart3 className="h-4 w-4 mr-2" />
                            View Details
                          </DropdownMenuItem>
                          <DropdownMenuItem>
                            <Download className="h-4 w-4 mr-2" />
                            Download
                          </DropdownMenuItem>
                          <DropdownMenuSeparator />
                          <DropdownMenuItem
                            className="text-destructive"
                            onClick={() => handleDelete(model)}
                          >
                            <Trash2 className="h-4 w-4 mr-2" />
                            Delete
                          </DropdownMenuItem>
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      {/* Model Details Dialog */}
      <Dialog open={isDetailsOpen} onOpenChange={setIsDetailsOpen}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Cpu className="h-5 w-5" />
              {selectedModel?.name}
              <Badge variant="outline" className="ml-2">v{selectedModel?.version}</Badge>
            </DialogTitle>
            <DialogDescription>
              Model details and performance metrics
            </DialogDescription>
          </DialogHeader>

          {selectedModel && (
            <div className="space-y-6 py-4">
              {/* Info Grid */}
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1">
                  <p className="text-sm text-muted-foreground">Architecture</p>
                  <p className="font-medium">{selectedModel.architecture}</p>
                </div>
                <div className="space-y-1">
                  <p className="text-sm text-muted-foreground">Training Dataset</p>
                  <p className="font-medium">{selectedModel.dataset_name || selectedModel.dataset_id}</p>
                </div>
                <div className="space-y-1">
                  <p className="text-sm text-muted-foreground">Classes</p>
                  <p className="font-medium">{selectedModel.class_count}</p>
                </div>
                <div className="space-y-1">
                  <p className="text-sm text-muted-foreground">Created</p>
                  <p className="font-medium">{new Date(selectedModel.created_at).toLocaleString()}</p>
                </div>
              </div>

              {/* Metrics */}
              <div>
                <h4 className="font-medium mb-3 flex items-center gap-2">
                  <BarChart3 className="h-4 w-4" />
                  Performance Metrics
                </h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="bg-muted rounded-lg p-3 text-center">
                    <p className="text-2xl font-bold text-blue-600">
                      {selectedModel.metrics.map50.toFixed(3)}
                    </p>
                    <p className="text-xs text-muted-foreground">mAP50</p>
                  </div>
                  <div className="bg-muted rounded-lg p-3 text-center">
                    <p className="text-2xl font-bold text-green-600">
                      {selectedModel.metrics.map50_95.toFixed(3)}
                    </p>
                    <p className="text-xs text-muted-foreground">mAP50-95</p>
                  </div>
                  <div className="bg-muted rounded-lg p-3 text-center">
                    <p className="text-2xl font-bold">
                      {selectedModel.metrics.precision.toFixed(3)}
                    </p>
                    <p className="text-xs text-muted-foreground">Precision</p>
                  </div>
                  <div className="bg-muted rounded-lg p-3 text-center">
                    <p className="text-2xl font-bold">
                      {selectedModel.metrics.recall.toFixed(3)}
                    </p>
                    <p className="text-xs text-muted-foreground">Recall</p>
                  </div>
                </div>
              </div>

              {/* Export Formats */}
              <div>
                <h4 className="font-medium mb-3 flex items-center gap-2">
                  <FileCode className="h-4 w-4" />
                  Export Formats
                </h4>
                <div className="flex flex-wrap gap-2">
                  {selectedModel.export_formats.map((format) => (
                    <Badge key={format} variant="secondary">
                      {format}
                    </Badge>
                  ))}
                </div>
              </div>

              {/* Model ID */}
              <div>
                <h4 className="font-medium mb-2">Model ID</h4>
                <div className="flex items-center gap-2 bg-muted rounded-lg p-3">
                  <code className="flex-1 text-sm font-mono">{selectedModel.id}</code>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => copyToClipboard(selectedModel.id)}
                  >
                    <Copy className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            </div>
          )}

          <DialogFooter>
            <Button variant="outline" onClick={() => setIsDetailsOpen(false)}>
              Close
            </Button>
            <Button>
              <Download className="h-4 w-4 mr-2" />
              Download Model
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
