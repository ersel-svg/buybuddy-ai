"use client";

import { useState, useEffect, useCallback, useMemo } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { toast } from "sonner";
import { apiClient } from "@/lib/api-client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import { Progress } from "@/components/ui/progress";
import { Slider } from "@/components/ui/slider";
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
import {
  Card,
  CardContent,
} from "@/components/ui/card";
import {
  Loader2,
  Sparkles,
  CheckCircle,
  XCircle,
  AlertTriangle,
  ArrowRight,
  Info,
} from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface BulkAnnotateModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  datasetId: string;
  totalImages: number;
  unannotatedCount: number;
  selectedImageIds?: string[];
  existingClasses: Array<{
    id: string;
    name: string;
    display_name?: string;
    color: string;
  }>;
  onSuccess?: () => void;
}

interface ClassMapping {
  detected: string;
  targetClassId: string | null;
  createNew: boolean;
}

// Roboflow model class configuration
interface RoboflowClassConfig {
  className: string;
  enabled: boolean;  // Include this class in predictions
  targetClassId: string | null;  // Map to existing class ID, or null for create new
}

type ImageSelectionMode = "unannotated" | "selected" | "all";
type JobStatus = "idle" | "running" | "completed" | "failed";

export function BulkAnnotateModal({
  open,
  onOpenChange,
  datasetId,
  totalImages,
  unannotatedCount,
  selectedImageIds = [],
  existingClasses,
  onSuccess,
}: BulkAnnotateModalProps) {
  // Form state
  const [selectionMode, setSelectionMode] = useState<ImageSelectionMode>("unannotated");
  const [selectedModel, setSelectedModel] = useState<string>("grounding_dino");
  const [textPrompt, setTextPrompt] = useState("");
  const [confidence, setConfidence] = useState(0.3);
  const [autoAccept, setAutoAccept] = useState(false);
  const [enableClassMapping, setEnableClassMapping] = useState(true);
  const [classMappings, setClassMappings] = useState<ClassMapping[]>([]);

  // Roboflow model class configuration
  const [rfClassConfigs, setRfClassConfigs] = useState<RoboflowClassConfig[]>([]);

  // Limit options
  const [enableLimit, setEnableLimit] = useState(false);
  const [imageLimit, setImageLimit] = useState(100);

  // Job state
  const [jobStatus, setJobStatus] = useState<JobStatus>("idle");
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobProgress, setJobProgress] = useState(0);
  const [jobTotalImages, setJobTotalImages] = useState(0);
  const [predictionsGenerated, setPredictionsGenerated] = useState(0);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Fetch available AI models
  const { data: modelsData } = useQuery({
    queryKey: ["od-ai-models"],
    queryFn: () => apiClient.getODAIModels(),
    enabled: open,
  });

  // Reset state when modal opens
  useEffect(() => {
    if (open) {
      setJobStatus("idle");
      setJobId(null);
      setJobProgress(0);
      setPredictionsGenerated(0);
      setErrorMessage(null);
      // Keep form values for convenience
    }
  }, [open]);

  // Parse prompt to get class names for mapping
  const parsePromptClasses = useCallback((prompt: string): string[] => {
    if (!prompt.trim()) return [];
    // Split by " . " (Grounding DINO format)
    return prompt
      .split(/\s*\.\s*/)
      .map((s) => s.trim().toLowerCase())
      .filter((s) => s.length > 0);
  }, []);

  // Update class mappings when prompt changes
  useEffect(() => {
    const promptClasses = parsePromptClasses(textPrompt);
    const newMappings: ClassMapping[] = promptClasses.map((className) => {
      // Find existing mapping
      const existingMapping = classMappings.find(
        (m) => m.detected === className
      );
      if (existingMapping) return existingMapping;

      // Try to find matching existing class
      const matchingClass = existingClasses.find(
        (c) =>
          c.name.toLowerCase() === className ||
          c.display_name?.toLowerCase() === className
      );

      return {
        detected: className,
        targetClassId: matchingClass?.id || null,
        createNew: !matchingClass,
      };
    });
    setClassMappings(newMappings);
  }, [textPrompt, existingClasses, parsePromptClasses]);

  // Get image count based on selection mode
  const getImageCount = useCallback((): number => {
    let count = 0;
    switch (selectionMode) {
      case "unannotated":
        count = unannotatedCount;
        break;
      case "selected":
        count = selectedImageIds.length;
        break;
      case "all":
        count = totalImages;
        break;
      default:
        count = 0;
    }

    // Apply limit if enabled
    if (enableLimit && imageLimit < count) {
      return imageLimit;
    }
    return count;
  }, [selectionMode, unannotatedCount, selectedImageIds.length, totalImages, enableLimit, imageLimit]);

  // Detection models and selected model info - memoized to prevent infinite loops
  const detectionModels = useMemo(() => {
    return modelsData?.detection_models || [];
  }, [modelsData?.detection_models]);
  
  const selectedModelInfo = useMemo(() => {
    return detectionModels.find((m) => m.id === selectedModel);
  }, [detectionModels, selectedModel]);
  
  // Check if selected model is a closed-vocabulary model (Roboflow or trained)
  const isClosedVocabModel = selectedModel.startsWith("rf:") || selectedModel.startsWith("trained:");

  // Memoize modelClasses to prevent infinite useEffect loops
  const modelClasses = useMemo(() => {
    return selectedModelInfo?.classes || [];
  }, [selectedModelInfo]);

  // Initialize closed-vocab class configs when model changes
  useEffect(() => {
    const isClosedVocab = selectedModel.startsWith("rf:") || selectedModel.startsWith("trained:");
    
    if (isClosedVocab) {
      // Find the model info for this model
      const modelInfo = detectionModels.find((m) => m.id === selectedModel);
      const classes = modelInfo?.classes || [];

      if (classes.length > 0) {
        // Create new configs for this model
        const newConfigs: RoboflowClassConfig[] = classes.map((cls) => {
          // Try to find matching existing class in dataset
          const matchingClass = existingClasses.find(
            (c) =>
              c.name.toLowerCase() === cls.toLowerCase() ||
              c.display_name?.toLowerCase() === cls.toLowerCase()
          );

          return {
            className: cls,
            enabled: true,  // Include by default
            targetClassId: matchingClass?.id || null,  // Auto-map if found, otherwise create new
          };
        });
        setRfClassConfigs(newConfigs);
      } else {
        setRfClassConfigs([]);
      }
    } else {
      // Clear class configs when switching to open-vocab model
      setRfClassConfigs([]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedModel]);
  // Note: detectionModels and existingClasses intentionally excluded - we only want to initialize once when model changes
  
  // Update Roboflow class config
  const updateRfClassConfig = (className: string, updates: Partial<RoboflowClassConfig>) => {
    setRfClassConfigs((prev) =>
      prev.map((config) =>
        config.className === className ? { ...config, ...updates } : config
      )
    );
  };

  // Start batch annotation mutation
  const startBatchMutation = useMutation({
    mutationFn: async () => {
      // Build class mapping and filter_classes based on model type
      const classMapping: Record<string, string> = {};
      let filterClasses: string[] | undefined;
      
      if (isClosedVocabModel) {
        // Roboflow model: use rfClassConfigs for filtering and mapping
        const enabledConfigs = rfClassConfigs.filter((c) => c.enabled);
        
        // Set filter_classes to only include enabled classes
        filterClasses = enabledConfigs.map((c) => c.className);
        
        // Build class mapping from enabled configs
        enabledConfigs.forEach((config) => {
          if (config.targetClassId) {
            // Map to existing class
            classMapping[config.className] = config.targetClassId;
          } else {
            // Create new class with same name
            classMapping[config.className] = `__new__:${config.className}`;
          }
        });
      } else if (enableClassMapping) {
        // Open-vocab model: use text prompt based class mappings
        classMappings.forEach((m) => {
          if (m.targetClassId) {
            classMapping[m.detected] = m.targetClassId;
          } else if (m.createNew) {
            // Use special marker for creating new class
            classMapping[m.detected] = `__new__:${m.detected}`;
          }
        });
      }

      // Determine image IDs
      let imageIds: string[] | undefined;
      if (selectionMode === "selected" && selectedImageIds.length > 0) {
        imageIds = selectedImageIds;
      }
      // For "unannotated" and "all", the backend handles filtering

      return apiClient.batchAnnotateODAI({
        dataset_id: datasetId,
        image_ids: imageIds,
        model: selectedModel,
        text_prompt: isClosedVocabModel ? undefined : textPrompt,  // Not needed for Roboflow
        box_threshold: confidence,
        auto_accept: autoAccept,
        class_mapping: Object.keys(classMapping).length > 0 ? classMapping : undefined,
        filter_classes: filterClasses,  // For Roboflow models
        limit: enableLimit ? imageLimit : undefined,  // Limit number of images if enabled
      });
    },
    onSuccess: (data) => {
      setJobId(data.job_id);
      setJobStatus("running");
      setJobTotalImages(data.total_images);
      toast.success(`Started processing ${data.total_images} images`);
    },
    onError: (error: Error) => {
      setJobStatus("failed");
      setErrorMessage(error.message);
      toast.error(`Failed to start batch annotation: ${error.message}`);
    },
  });

  // Poll job status
  useEffect(() => {
    if (!jobId || jobStatus !== "running") return;

    const pollInterval = setInterval(async () => {
      try {
        const status = await apiClient.getODAIJobStatus(jobId);
        setJobProgress(status.progress);
        setPredictionsGenerated(status.predictions_generated);

        if (status.status === "completed") {
          setJobStatus("completed");
          clearInterval(pollInterval);
          toast.success(
            `Batch annotation completed! Generated ${status.predictions_generated} predictions.`
          );
          onSuccess?.();
        } else if (status.status === "failed") {
          setJobStatus("failed");
          setErrorMessage(status.error_message || "Unknown error");
          clearInterval(pollInterval);
          toast.error(`Batch annotation failed: ${status.error_message}`);
        }
      } catch (error) {
        console.error("Error polling job status:", error);
      }
    }, 2000);

    return () => clearInterval(pollInterval);
  }, [jobId, jobStatus, onSuccess]);

  // Handle form submission
  const handleSubmit = () => {
    // Validation based on model type
    if (isClosedVocabModel) {
      // Roboflow model: check if at least one class is enabled
      const enabledCount = rfClassConfigs.filter((c) => c.enabled).length;
      if (enabledCount === 0) {
        toast.error("Please enable at least one class to detect");
        return;
      }
    } else {
      // Open-vocab model: require text prompt
      if (!textPrompt.trim()) {
        toast.error("Please enter a detection prompt");
        return;
      }
    }

    const imageCount = getImageCount();
    if (imageCount === 0) {
      toast.error("No images to process");
      return;
    }

    startBatchMutation.mutate();
  };

  // Update class mapping
  const updateClassMapping = (index: number, updates: Partial<ClassMapping>) => {
    setClassMappings((prev) => {
      const updated = [...prev];
      updated[index] = { ...updated[index], ...updates };
      return updated;
    });
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-purple-500" />
            AI Bulk Auto-Annotate
          </DialogTitle>
          <DialogDescription>
            Automatically detect and annotate objects across multiple images using AI models.
          </DialogDescription>
        </DialogHeader>

        {jobStatus === "idle" ? (
          <div className="space-y-6 py-4">
            {/* Image Selection */}
            <div className="space-y-3">
              <Label className="text-sm font-medium">Images to process</Label>
              <div className="space-y-2">
                <label className="flex items-center gap-3 p-3 border rounded-lg cursor-pointer hover:bg-muted/50 transition-colors">
                  <input
                    type="radio"
                    name="selectionMode"
                    value="unannotated"
                    checked={selectionMode === "unannotated"}
                    onChange={() => setSelectionMode("unannotated")}
                    className="h-4 w-4"
                  />
                  <div className="flex-1">
                    <p className="font-medium">Unannotated images only</p>
                    <p className="text-sm text-muted-foreground">
                      Process images that have no annotations yet
                    </p>
                  </div>
                  <Badge variant="secondary">{unannotatedCount}</Badge>
                </label>

                {selectedImageIds.length > 0 && (
                  <label className="flex items-center gap-3 p-3 border rounded-lg cursor-pointer hover:bg-muted/50 transition-colors">
                    <input
                      type="radio"
                      name="selectionMode"
                      value="selected"
                      checked={selectionMode === "selected"}
                      onChange={() => setSelectionMode("selected")}
                      className="h-4 w-4"
                    />
                    <div className="flex-1">
                      <p className="font-medium">Selected images</p>
                      <p className="text-sm text-muted-foreground">
                        Process only the images you selected
                      </p>
                    </div>
                    <Badge variant="secondary">{selectedImageIds.length}</Badge>
                  </label>
                )}

                <label className="flex items-center gap-3 p-3 border rounded-lg cursor-pointer hover:bg-muted/50 transition-colors">
                  <input
                    type="radio"
                    name="selectionMode"
                    value="all"
                    checked={selectionMode === "all"}
                    onChange={() => setSelectionMode("all")}
                    className="h-4 w-4"
                  />
                  <div className="flex-1">
                    <p className="font-medium">All images</p>
                    <p className="text-sm text-muted-foreground">
                      Re-process all images (will add new predictions)
                    </p>
                  </div>
                  <Badge variant="secondary">{totalImages}</Badge>
                </label>
              </div>
            </div>

            {/* Model Selection */}
            <div className="space-y-2">
              <Label htmlFor="model">Detection Model</Label>
              <Select value={selectedModel} onValueChange={setSelectedModel}>
                <SelectTrigger>
                  <SelectValue placeholder="Select a model" />
                </SelectTrigger>
                <SelectContent>
                  {detectionModels.map((model) => (
                    <SelectItem key={model.id} value={model.id}>
                      <div className="flex items-center gap-2">
                        <span>{model.name}</span>
                        {model.id === "grounding_dino" && (
                          <Badge variant="secondary" className="text-xs">
                            Recommended
                          </Badge>
                        )}
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {selectedModelInfo && (
                <p className="text-xs text-muted-foreground">
                  {selectedModelInfo.description}
                </p>
              )}
            </div>

            {/* Text Prompt - Only for open-vocab models */}
            {!isClosedVocabModel && (
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <Label htmlFor="prompt">Detection Prompt</Label>
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger>
                        <Info className="h-4 w-4 text-muted-foreground" />
                      </TooltipTrigger>
                      <TooltipContent className="max-w-xs">
                        <p>
                          Separate class names with &quot; . &quot; (space dot space).
                          Example: &quot;shelf . product . price tag&quot;
                        </p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>
                <Input
                  id="prompt"
                  placeholder="shelf . product . price tag . promotional material"
                  value={textPrompt}
                  onChange={(e) => setTextPrompt(e.target.value)}
                />
                <p className="text-xs text-muted-foreground">
                  Separate class names with &quot; . &quot; (space dot space)
                </p>
              </div>
            )}

            {/* Roboflow Model Class Configuration */}
            {isClosedVocabModel && rfClassConfigs.length > 0 && (
              <div className="space-y-3">
                <div className="flex items-center gap-2">
                  <Label>Model Classes</Label>
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger>
                        <Info className="h-4 w-4 text-muted-foreground" />
                      </TooltipTrigger>
                      <TooltipContent className="max-w-xs">
                        <p>
                          Select which classes to detect and map them to your dataset classes.
                          Uncheck a class to exclude it from detection.
                        </p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>
                <Card>
                  <CardContent className="p-3 space-y-3">
                    <div className="grid grid-cols-[auto,1fr,auto,1fr] gap-2 text-xs font-medium text-muted-foreground pb-2 border-b">
                      <span></span>
                      <span>Model Class</span>
                      <span></span>
                      <span>Map to Dataset Class</span>
                    </div>
                    {rfClassConfigs.map((config) => (
                      <div
                        key={config.className}
                        className={`grid grid-cols-[auto,1fr,auto,1fr] gap-2 items-center ${
                          !config.enabled ? "opacity-50" : ""
                        }`}
                      >
                        <Checkbox
                          checked={config.enabled}
                          onCheckedChange={(checked) =>
                            updateRfClassConfig(config.className, { enabled: !!checked })
                          }
                        />
                        <Badge
                          variant={config.enabled ? "secondary" : "outline"}
                          className="justify-start"
                        >
                          {config.className}
                        </Badge>
                        <ArrowRight className="h-4 w-4 text-muted-foreground" />
                        <Select
                          value={config.targetClassId || "__new__"}
                          onValueChange={(value) =>
                            updateRfClassConfig(config.className, {
                              targetClassId: value === "__new__" ? null : value,
                            })
                          }
                          disabled={!config.enabled}
                        >
                          <SelectTrigger className="h-8">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="__new__">
                              <span className="text-purple-600">+ Create &quot;{config.className}&quot;</span>
                            </SelectItem>
                            {existingClasses.map((cls) => (
                              <SelectItem key={cls.id} value={cls.id}>
                                <div className="flex items-center gap-2">
                                  <div
                                    className="w-3 h-3 rounded"
                                    style={{ backgroundColor: cls.color }}
                                  />
                                  {cls.display_name || cls.name}
                                </div>
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                    ))}
                    <p className="text-xs text-muted-foreground pt-2 border-t">
                      {rfClassConfigs.filter((c) => c.enabled).length} of {rfClassConfigs.length} classes enabled
                    </p>
                  </CardContent>
                </Card>
              </div>
            )}

            {/* Confidence Threshold */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <Label>Confidence Threshold</Label>
                <span className="text-sm font-mono">{confidence.toFixed(2)}</span>
              </div>
              <Slider
                value={[confidence]}
                onValueChange={([v]) => setConfidence(v)}
                min={0.1}
                max={0.9}
                step={0.05}
                className="w-full"
              />
              <p className="text-xs text-muted-foreground">
                Lower values detect more objects but may include false positives
              </p>
            </div>

            {/* Image Limit Option */}
            <div className="space-y-3 p-3 border rounded-lg">
              <div className="flex items-center gap-3">
                <Checkbox
                  id="enableLimit"
                  checked={enableLimit}
                  onCheckedChange={(checked) => setEnableLimit(checked === true)}
                />
                <div className="flex-1">
                  <Label htmlFor="enableLimit" className="cursor-pointer font-medium">
                    Limit number of images
                  </Label>
                  <p className="text-sm text-muted-foreground">
                    Process only the first N images (useful for testing)
                  </p>
                </div>
              </div>
              {enableLimit && (
                <div className="flex items-center gap-2 pl-7">
                  <Label htmlFor="imageLimit" className="text-sm whitespace-nowrap">
                    Process first
                  </Label>
                  <Input
                    id="imageLimit"
                    type="number"
                    min={1}
                    max={10000}
                    value={imageLimit}
                    onChange={(e) => setImageLimit(Math.max(1, parseInt(e.target.value) || 1))}
                    className="w-24"
                  />
                  <span className="text-sm text-muted-foreground">images</span>
                </div>
              )}
            </div>

            {/* Auto-accept Option */}
            <div className="flex items-center gap-3 p-3 border rounded-lg">
              <Checkbox
                id="autoAccept"
                checked={autoAccept}
                onCheckedChange={(checked) => setAutoAccept(checked === true)}
              />
              <div className="flex-1">
                <Label htmlFor="autoAccept" className="cursor-pointer font-medium">
                  Auto-accept predictions
                </Label>
                <p className="text-sm text-muted-foreground">
                  Skip review and save predictions directly as annotations
                </p>
              </div>
            </div>

            {/* Class Mapping - Only for open-vocab models */}
            {!isClosedVocabModel && classMappings.length > 0 && (
              <div className="space-y-3">
                <div className="flex items-center gap-3">
                  <Checkbox
                    id="enableMapping"
                    checked={enableClassMapping}
                    onCheckedChange={(checked) => setEnableClassMapping(checked === true)}
                  />
                  <Label htmlFor="enableMapping" className="cursor-pointer font-medium">
                    Map to existing classes
                  </Label>
                </div>

                {enableClassMapping && (
                  <Card>
                    <CardContent className="p-3 space-y-2">
                      <div className="grid grid-cols-[1fr,auto,1fr] gap-2 text-xs font-medium text-muted-foreground mb-2">
                        <span>Detected</span>
                        <span></span>
                        <span>Map to</span>
                      </div>
                      {classMappings.map((mapping, index) => (
                        <div
                          key={mapping.detected}
                          className="grid grid-cols-[1fr,auto,1fr] gap-2 items-center"
                        >
                          <Badge variant="outline" className="justify-start">
                            {mapping.detected}
                          </Badge>
                          <ArrowRight className="h-4 w-4 text-muted-foreground" />
                          <Select
                            value={mapping.targetClassId || "__new__"}
                            onValueChange={(value) =>
                              updateClassMapping(index, {
                                targetClassId: value === "__new__" ? null : value,
                                createNew: value === "__new__",
                              })
                            }
                          >
                            <SelectTrigger className="h-8">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="__new__">
                                <span className="text-purple-600">+ Create new class</span>
                              </SelectItem>
                              {existingClasses.map((cls) => (
                                <SelectItem key={cls.id} value={cls.id}>
                                  <div className="flex items-center gap-2">
                                    <div
                                      className="w-3 h-3 rounded"
                                      style={{ backgroundColor: cls.color }}
                                    />
                                    {cls.display_name || cls.name}
                                  </div>
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        </div>
                      ))}
                    </CardContent>
                  </Card>
                )}
              </div>
            )}
          </div>
        ) : (
          /* Progress View */
          <div className="py-8 space-y-6">
            <div className="text-center space-y-2">
              {jobStatus === "running" && (
                <>
                  <Loader2 className="h-12 w-12 animate-spin mx-auto text-purple-500" />
                  <h3 className="text-lg font-semibold">Processing images...</h3>
                </>
              )}
              {jobStatus === "completed" && (
                <>
                  <CheckCircle className="h-12 w-12 mx-auto text-green-500" />
                  <h3 className="text-lg font-semibold text-green-600">
                    Batch annotation completed!
                  </h3>
                </>
              )}
              {jobStatus === "failed" && (
                <>
                  <XCircle className="h-12 w-12 mx-auto text-red-500" />
                  <h3 className="text-lg font-semibold text-red-600">
                    Batch annotation failed
                  </h3>
                  {errorMessage && (
                    <p className="text-sm text-muted-foreground">{errorMessage}</p>
                  )}
                </>
              )}
            </div>

            {jobStatus === "running" && (
              <div className="space-y-4">
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Progress</span>
                    <span className="font-mono">{jobProgress}%</span>
                  </div>
                  <Progress value={jobProgress} className="h-3" />
                </div>

                <div className="grid grid-cols-2 gap-4 text-center">
                  <Card>
                    <CardContent className="p-4">
                      <p className="text-2xl font-bold">{jobTotalImages}</p>
                      <p className="text-sm text-muted-foreground">Total images</p>
                    </CardContent>
                  </Card>
                  <Card>
                    <CardContent className="p-4">
                      <p className="text-2xl font-bold text-purple-600">
                        {predictionsGenerated}
                      </p>
                      <p className="text-sm text-muted-foreground">
                        Predictions generated
                      </p>
                    </CardContent>
                  </Card>
                </div>

                <div className="flex items-center justify-center gap-2 text-sm text-muted-foreground">
                  <AlertTriangle className="h-4 w-4" />
                  <span>Do not close this window while processing</span>
                </div>
              </div>
            )}

            {jobStatus === "completed" && (
              <div className="text-center space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <Card>
                    <CardContent className="p-4">
                      <p className="text-2xl font-bold">{jobTotalImages}</p>
                      <p className="text-sm text-muted-foreground">Images processed</p>
                    </CardContent>
                  </Card>
                  <Card>
                    <CardContent className="p-4">
                      <p className="text-2xl font-bold text-green-600">
                        {predictionsGenerated}
                      </p>
                      <p className="text-sm text-muted-foreground">
                        Predictions generated
                      </p>
                    </CardContent>
                  </Card>
                </div>
                {!autoAccept && (
                  <p className="text-sm text-muted-foreground">
                    Review predictions in the annotation editor to accept or reject them.
                  </p>
                )}
              </div>
            )}
          </div>
        )}

        <DialogFooter>
          {jobStatus === "idle" && (
            <>
              <Button variant="outline" onClick={() => onOpenChange(false)}>
                Cancel
              </Button>
              <Button
                onClick={handleSubmit}
                disabled={
                  (isClosedVocabModel
                    ? rfClassConfigs.filter((c) => c.enabled).length === 0
                    : !textPrompt.trim()) ||
                  getImageCount() === 0 ||
                  startBatchMutation.isPending
                }
              >
                {startBatchMutation.isPending ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Starting...
                  </>
                ) : (
                  <>
                    <Sparkles className="h-4 w-4 mr-2" />
                    Start Processing ({getImageCount()} images)
                  </>
                )}
              </Button>
            </>
          )}
          {jobStatus === "running" && (
            <Button variant="outline" disabled>
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              Processing...
            </Button>
          )}
          {(jobStatus === "completed" || jobStatus === "failed") && (
            <Button onClick={() => onOpenChange(false)}>Close</Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
