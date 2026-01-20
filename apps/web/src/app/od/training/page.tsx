"use client";

import { useState } from "react";
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
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
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
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Slider } from "@/components/ui/slider";
import {
  RefreshCw,
  Loader2,
  Brain,
  Play,
  Square,
  CheckCircle,
  XCircle,
  Clock,
  AlertTriangle,
  Cpu,
  BarChart3,
  Layers,
  Settings,
  Rocket,
} from "lucide-react";

interface TrainingRun {
  id: string;
  name: string;
  dataset_id: string;
  dataset_name?: string;
  model_architecture: string;
  status: string;
  progress: number;
  current_epoch?: number;
  total_epochs: number;
  metrics?: {
    loss?: number;
    map50?: number;
    map50_95?: number;
  };
  created_at: string;
  started_at?: string;
  completed_at?: string;
  error_message?: string;
}

const MODEL_ARCHITECTURES = [
  { id: "rf-detr-base", name: "RF-DETR Base", description: "Balanced accuracy and speed" },
  { id: "rf-detr-large", name: "RF-DETR Large", description: "Higher accuracy, slower inference" },
  { id: "rt-detr-l", name: "RT-DETR-L", description: "Real-time detection, good accuracy" },
  { id: "rt-detr-x", name: "RT-DETR-X", description: "Best accuracy, requires more resources" },
  { id: "yolo-nas-s", name: "YOLO-NAS Small", description: "Fast inference, compact model" },
  { id: "yolo-nas-m", name: "YOLO-NAS Medium", description: "Good balance for production" },
  { id: "yolo-nas-l", name: "YOLO-NAS Large", description: "High accuracy detection" },
];

export default function ODTrainingPage() {
  const router = useRouter();
  const queryClient = useQueryClient();

  // Dialog state
  const [isNewTrainingOpen, setIsNewTrainingOpen] = useState(false);

  // Form state
  const [formName, setFormName] = useState("");
  const [formDatasetId, setFormDatasetId] = useState("");
  const [formArchitecture, setFormArchitecture] = useState("rf-detr-base");
  const [formEpochs, setFormEpochs] = useState(100);
  const [formBatchSize, setFormBatchSize] = useState(16);
  const [formLearningRate, setFormLearningRate] = useState(0.001);
  const [formImageSize, setFormImageSize] = useState(640);

  // Fetch training runs
  const { data: trainingRuns, isLoading, isFetching } = useQuery({
    queryKey: ["od-training-runs"],
    queryFn: async () => {
      // This would fetch from API - for now return mock data
      return [] as TrainingRun[];
    },
    refetchInterval: 10000, // Refresh every 10s to update progress
  });

  // Fetch datasets for selection
  const { data: datasets } = useQuery({
    queryKey: ["od-datasets"],
    queryFn: () => apiClient.getODDatasets(),
  });

  // Start training mutation (placeholder)
  const startTrainingMutation = useMutation({
    mutationFn: async (data: {
      name: string;
      dataset_id: string;
      model_architecture: string;
      config: {
        epochs: number;
        batch_size: number;
        learning_rate: number;
        image_size: number;
      };
    }) => {
      // This would call the API to start training
      toast.info("Training API not yet implemented");
      return { id: "mock-id" };
    },
    onSuccess: () => {
      toast.success("Training job queued");
      queryClient.invalidateQueries({ queryKey: ["od-training-runs"] });
      setIsNewTrainingOpen(false);
      resetForm();
    },
    onError: (error) => {
      toast.error(`Failed to start training: ${error.message}`);
    },
  });

  const resetForm = () => {
    setFormName("");
    setFormDatasetId("");
    setFormArchitecture("rf-detr-base");
    setFormEpochs(100);
    setFormBatchSize(16);
    setFormLearningRate(0.001);
    setFormImageSize(640);
  };

  const handleStartTraining = () => {
    if (!formName || !formDatasetId) {
      toast.error("Please fill in all required fields");
      return;
    }
    startTrainingMutation.mutate({
      name: formName,
      dataset_id: formDatasetId,
      model_architecture: formArchitecture,
      config: {
        epochs: formEpochs,
        batch_size: formBatchSize,
        learning_rate: formLearningRate,
        image_size: formImageSize,
      },
    });
  };

  const getStatusBadge = (status: string) => {
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
      case "queued":
        return (
          <Badge variant="secondary">
            <Clock className="h-3 w-3 mr-1" />
            Queued
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

  const selectedDataset = datasets?.find((d) => d.id === formDatasetId);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold">Training</h1>
          <p className="text-muted-foreground">
            Train object detection models with RF-DETR, RT-DETR, or YOLO-NAS
          </p>
        </div>
        <div className="flex gap-2">
          <Button
            variant="outline"
            onClick={() => queryClient.invalidateQueries({ queryKey: ["od-training-runs"] })}
            disabled={isFetching}
          >
            <RefreshCw className={`h-4 w-4 mr-2 ${isFetching ? "animate-spin" : ""}`} />
            Refresh
          </Button>
          <Button onClick={() => {
            resetForm();
            setIsNewTrainingOpen(true);
          }}>
            <Rocket className="h-4 w-4 mr-2" />
            New Training
          </Button>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">Total Runs</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">{trainingRuns?.length ?? 0}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">Running</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-blue-600">
              {trainingRuns?.filter((r) => r.status === "running").length ?? 0}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">Completed</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-green-600">
              {trainingRuns?.filter((r) => r.status === "completed").length ?? 0}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">Failed</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-red-600">
              {trainingRuns?.filter((r) => r.status === "failed").length ?? 0}
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Training Runs Table */}
      <Card>
        <CardHeader>
          <CardTitle>Training Runs</CardTitle>
          <CardDescription>
            History of all training jobs
          </CardDescription>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
            </div>
          ) : trainingRuns?.length === 0 ? (
            <div className="text-center py-12">
              <Brain className="h-16 w-16 mx-auto text-muted-foreground mb-4" />
              <h3 className="text-lg font-medium">No training runs yet</h3>
              <p className="text-muted-foreground mt-1">
                Start your first training job to train a detection model
              </p>
              <Button className="mt-4" onClick={() => {
                resetForm();
                setIsNewTrainingOpen(true);
              }}>
                <Rocket className="h-4 w-4 mr-2" />
                Start Training
              </Button>
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Name</TableHead>
                  <TableHead>Dataset</TableHead>
                  <TableHead>Model</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Progress</TableHead>
                  <TableHead>Metrics</TableHead>
                  <TableHead>Created</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {trainingRuns?.map((run) => (
                  <TableRow key={run.id} className="cursor-pointer hover:bg-muted/50">
                    <TableCell className="font-medium">{run.name}</TableCell>
                    <TableCell>{run.dataset_name || run.dataset_id}</TableCell>
                    <TableCell>
                      <Badge variant="outline">{run.model_architecture}</Badge>
                    </TableCell>
                    <TableCell>{getStatusBadge(run.status)}</TableCell>
                    <TableCell>
                      <div className="w-32">
                        <Progress value={run.progress} className="h-2" />
                        <p className="text-xs text-muted-foreground mt-1">
                          {run.current_epoch ?? 0}/{run.total_epochs} epochs
                        </p>
                      </div>
                    </TableCell>
                    <TableCell>
                      {run.metrics ? (
                        <div className="text-xs">
                          <p>mAP50: {(run.metrics.map50 ?? 0).toFixed(3)}</p>
                          <p>mAP50-95: {(run.metrics.map50_95 ?? 0).toFixed(3)}</p>
                        </div>
                      ) : (
                        <span className="text-muted-foreground">-</span>
                      )}
                    </TableCell>
                    <TableCell className="text-muted-foreground">
                      {new Date(run.created_at).toLocaleDateString()}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      {/* Model Architectures Info */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Cpu className="h-5 w-5" />
            Available Model Architectures
          </CardTitle>
          <CardDescription>
            All models are Apache 2.0 licensed for commercial use
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {MODEL_ARCHITECTURES.map((model) => (
              <div key={model.id} className="border rounded-lg p-4">
                <h4 className="font-medium">{model.name}</h4>
                <p className="text-sm text-muted-foreground mt-1">{model.description}</p>
                <Badge variant="outline" className="mt-2">
                  {model.id.split("-")[0].toUpperCase()}
                </Badge>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* New Training Dialog */}
      <Dialog open={isNewTrainingOpen} onOpenChange={setIsNewTrainingOpen}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Rocket className="h-5 w-5" />
              New Training Job
            </DialogTitle>
            <DialogDescription>
              Configure and start a new model training run
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-6 py-4">
            {/* Basic Info */}
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label>Training Name *</Label>
                <Input
                  placeholder="e.g., Shelf Detection v1"
                  value={formName}
                  onChange={(e) => setFormName(e.target.value)}
                />
              </div>
              <div className="space-y-2">
                <Label>Dataset *</Label>
                <Select value={formDatasetId} onValueChange={setFormDatasetId}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select dataset..." />
                  </SelectTrigger>
                  <SelectContent>
                    {datasets?.map((ds) => (
                      <SelectItem key={ds.id} value={ds.id}>
                        {ds.name} ({ds.image_count} images)
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>

            {selectedDataset && (
              <div className="bg-muted rounded-lg p-4">
                <p className="text-sm font-medium">Dataset Info</p>
                <div className="grid grid-cols-3 gap-4 mt-2 text-sm">
                  <div>
                    <span className="text-muted-foreground">Images:</span>{" "}
                    <span className="font-medium">{selectedDataset.image_count}</span>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Annotated:</span>{" "}
                    <span className="font-medium">{selectedDataset.annotated_image_count}</span>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Annotations:</span>{" "}
                    <span className="font-medium">{selectedDataset.annotation_count}</span>
                  </div>
                </div>
                {selectedDataset.annotated_image_count < selectedDataset.image_count && (
                  <div className="flex items-center gap-2 mt-3 text-orange-600 text-sm">
                    <AlertTriangle className="h-4 w-4" />
                    Not all images are annotated. Consider completing annotations first.
                  </div>
                )}
              </div>
            )}

            {/* Model Selection */}
            <div className="space-y-2">
              <Label>Model Architecture</Label>
              <Select value={formArchitecture} onValueChange={setFormArchitecture}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {MODEL_ARCHITECTURES.map((model) => (
                    <SelectItem key={model.id} value={model.id}>
                      <div>
                        <span className="font-medium">{model.name}</span>
                        <span className="text-muted-foreground ml-2">- {model.description}</span>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Training Parameters */}
            <div className="space-y-4">
              <div className="flex items-center gap-2">
                <Settings className="h-4 w-4" />
                <Label className="text-base">Training Parameters</Label>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>Epochs: {formEpochs}</Label>
                  <Slider
                    value={[formEpochs]}
                    onValueChange={(v) => setFormEpochs(v[0])}
                    min={10}
                    max={300}
                    step={10}
                  />
                </div>
                <div className="space-y-2">
                  <Label>Batch Size: {formBatchSize}</Label>
                  <Slider
                    value={[formBatchSize]}
                    onValueChange={(v) => setFormBatchSize(v[0])}
                    min={4}
                    max={64}
                    step={4}
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>Learning Rate</Label>
                  <Select
                    value={formLearningRate.toString()}
                    onValueChange={(v) => setFormLearningRate(parseFloat(v))}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="0.01">0.01 (High)</SelectItem>
                      <SelectItem value="0.001">0.001 (Default)</SelectItem>
                      <SelectItem value="0.0001">0.0001 (Low)</SelectItem>
                      <SelectItem value="0.00001">0.00001 (Very Low)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label>Image Size</Label>
                  <Select
                    value={formImageSize.toString()}
                    onValueChange={(v) => setFormImageSize(parseInt(v))}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="416">416px</SelectItem>
                      <SelectItem value="512">512px</SelectItem>
                      <SelectItem value="640">640px (Default)</SelectItem>
                      <SelectItem value="800">800px</SelectItem>
                      <SelectItem value="1024">1024px</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </div>
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setIsNewTrainingOpen(false)}>
              Cancel
            </Button>
            <Button
              onClick={handleStartTraining}
              disabled={!formName || !formDatasetId || startTrainingMutation.isPending}
            >
              {startTrainingMutation.isPending ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Starting...
                </>
              ) : (
                <>
                  <Play className="h-4 w-4 mr-2" />
                  Start Training
                </>
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
