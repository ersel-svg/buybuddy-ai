"use client";

import { useState, useCallback, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { useRetryRoboflowImport, type Job } from "@/hooks/use-active-jobs";
import { apiClient } from "@/lib/api-client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import { Progress } from "@/components/ui/progress";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";
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
  Upload,
  Link as LinkIcon,
  Package,
  Loader2,
  CheckCircle,
  XCircle,
  AlertTriangle,
  FileArchive,
  ArrowRight,
  Plus,
  Minus,
  Eye,
  Cloud,
  RefreshCw,
  Key,
  FolderOpen,
  ChevronRight,
} from "lucide-react";

interface ImportModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  datasetId?: string; // If provided, imports directly to this dataset
  onSuccess?: () => void;
}

interface ClassMappingItem {
  source_name: string;
  target_class_id: string | null;
  create_new: boolean;
  skip: boolean;
  color: string;
}

interface UploadQueueItem {
  file: File;
  status: "pending" | "uploading" | "success" | "error" | "duplicate";
  error?: string;
  duplicateId?: string;
}

// Color palette for new classes
const CLASS_COLORS = [
  "#ef4444", "#f97316", "#eab308", "#22c55e", "#14b8a6",
  "#3b82f6", "#8b5cf6", "#ec4899", "#6366f1", "#06b6d4",
];

export function ImportModal({ open, onOpenChange, datasetId, onSuccess }: ImportModalProps) {
  const queryClient = useQueryClient();
  const [activeTab, setActiveTab] = useState("upload");

  // Upload Tab State
  const [uploadQueue, setUploadQueue] = useState<UploadQueueItem[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [skipDuplicates, setSkipDuplicates] = useState(true);

  // URL Tab State
  const [urlInput, setUrlInput] = useState("");
  const [urlFolder, setUrlFolder] = useState("");

  // Annotated Import Tab State
  const [annotatedFile, setAnnotatedFile] = useState<File | null>(null);
  const [importPreview, setImportPreview] = useState<{
    format_detected: string;
    total_images: number;
    total_annotations: number;
    classes_found: string[];
    sample_images: Array<{ filename: string; annotation_count: number; classes: string[] }>;
    errors: string[];
  } | null>(null);
  const [classMapping, setClassMapping] = useState<ClassMappingItem[]>([]);
  const [isPreviewLoading, setIsPreviewLoading] = useState(false);
  const [mergeAnnotations, setMergeAnnotations] = useState(false);

  // BuyBuddy Sync Tab State
  const [bbMaxImages, setBbMaxImages] = useState<number>(100);
  const [bbFolder, setBbFolder] = useState("buybuddy_sync");
  const [bbStartDate, setBbStartDate] = useState<string>("");
  const [bbEndDate, setBbEndDate] = useState<string>("");
  const [bbStoreId, setBbStoreId] = useState<string>("");
  const [bbIsAnnotated, setBbIsAnnotated] = useState<boolean | undefined>(undefined);
  const [bbIsApproved, setBbIsApproved] = useState<boolean | undefined>(undefined);

  // Roboflow Tab State
  const [rfApiKey, setRfApiKey] = useState("");
  const [rfKeyValidated, setRfKeyValidated] = useState(false);
  const [rfWorkspaces, setRfWorkspaces] = useState<Array<{ name: string; url: string; projects: number }>>([]);
  const [rfProjects, setRfProjects] = useState<Array<{ id: string; name: string; type: string; images: number; versions?: number }>>([]);
  const [rfSelectedWorkspace, setRfSelectedWorkspace] = useState<string>("");
  const [rfSelectedProject, setRfSelectedProject] = useState<string>("");
  const [rfVersions, setRfVersions] = useState<Array<{ id: string; name: string; version: number; images: Record<string, number>; classes: string[] }>>([]);
  const [rfSelectedVersion, setRfSelectedVersion] = useState<number | null>(null);
  const [rfPreview, setRfPreview] = useState<{
    workspace: string;
    project: string;
    version: number;
    version_name: string;
    total_images: number;
    splits: Record<string, number>;
    classes: Array<{ name: string; count: number }>;
    class_count: number;
  } | null>(null);
  const [rfClassMapping, setRfClassMapping] = useState<ClassMappingItem[]>([]);
  const [rfFormat, setRfFormat] = useState<"coco" | "yolov8">("coco");
  const [rfImportJobId, setRfImportJobId] = useState<string | null>(null);
  const [rfImportProgress, setRfImportProgress] = useState(0);
  const [rfImportStatus, setRfImportStatus] = useState<string | null>(null);
  const [rfImportMessage, setRfImportMessage] = useState<string | null>(null);
  const [rfResumableJob, setRfResumableJob] = useState<Job | null>(null);

  // Hook for retrying/resuming roboflow imports
  const retryImport = useRetryRoboflowImport();

  // Check for resumable jobs when modal opens on Roboflow tab
  useEffect(() => {
    if (open && activeTab === "roboflow" && datasetId) {
      const checkResumable = async () => {
        try {
          const jobs = await apiClient.getJobs("roboflow_import");
          const resumable = jobs.find(
            (j: Job) =>
              j.type === "roboflow_import" &&
              j.status === "failed" &&
              j.config?.dataset_id === datasetId &&
              j.result?.can_resume === true
          );
          setRfResumableJob(resumable || null);
        } catch (error) {
          console.error("Failed to check resumable jobs:", error);
        }
      };
      checkResumable();
    }
  }, [open, activeTab, datasetId]);

  // Fetch existing classes for mapping
  const { data: existingClasses } = useQuery({
    queryKey: ["od-classes", datasetId],
    queryFn: () => apiClient.getODClasses({ dataset_id: datasetId || undefined }),
    enabled: open && !!datasetId,
  });

  // BuyBuddy API status check
  const { data: bbStatus } = useQuery({
    queryKey: ["buybuddy-status"],
    queryFn: () => apiClient.checkBuyBuddySyncStatus(),
    enabled: open && activeTab === "buybuddy",
  });

  // Build filter params for BuyBuddy
  const bbFilterParams = {
    start_date: bbStartDate || undefined,
    end_date: bbEndDate || undefined,
    store_id: bbStoreId ? parseInt(bbStoreId) : undefined,
    is_annotated: bbIsAnnotated,
    is_approved: bbIsApproved,
  };

  // BuyBuddy preview
  const { data: bbPreview, isLoading: bbPreviewLoading, refetch: refetchBbPreview } = useQuery({
    queryKey: ["buybuddy-preview", bbFilterParams],
    queryFn: () => apiClient.previewBuyBuddySync({ ...bbFilterParams, limit: 10 }),
    enabled: open && activeTab === "buybuddy" && bbStatus?.accessible === true,
  });

  // BuyBuddy sync mutation
  const bbSyncMutation = useMutation({
    mutationFn: async () => {
      return apiClient.syncFromBuyBuddy({
        ...bbFilterParams,
        max_images: bbMaxImages,
        dataset_id: datasetId,
        tags: bbFolder ? [bbFolder] : undefined,
      });
    },
    onSuccess: (data) => {
      toast.success(`Synced ${data.synced} images from BuyBuddy${data.skipped > 0 ? `, ${data.skipped} duplicates skipped` : ""}`);
      if (data.errors && data.errors.length > 0) {
        toast.warning(`${data.errors.length} errors occurred`);
      }
      queryClient.invalidateQueries({ queryKey: ["od-images"] });
      queryClient.invalidateQueries({ queryKey: ["buybuddy-preview"] });
      if (datasetId) {
        queryClient.invalidateQueries({ queryKey: ["od-dataset-images", datasetId] });
      }
      onSuccess?.();
    },
    onError: (error) => {
      toast.error(`Sync failed: ${error}`);
    },
  });

  // ===========================================
  // Roboflow Tab Functions
  // ===========================================

  const rfValidateKeyMutation = useMutation({
    mutationFn: async (apiKey: string) => {
      return apiClient.validateRoboflowKey(apiKey);
    },
    onSuccess: (data) => {
      if (data.valid) {
        setRfKeyValidated(true);
        setRfWorkspaces(data.workspaces);
        // If projects are included in validation response, use them
        if (data.projects && data.projects.length > 0) {
          setRfProjects(data.projects);
          // Auto-select first workspace if only one
          if (data.workspaces.length === 1) {
            setRfSelectedWorkspace(data.workspaces[0].url);
          }
        }
        toast.success("API key validated successfully");
      } else {
        toast.error(data.error || "Invalid API key");
      }
    },
    onError: (error) => {
      toast.error(`Validation failed: ${error}`);
    },
  });

  const rfLoadProjectsMutation = useMutation({
    mutationFn: async (workspace: string) => {
      return apiClient.getRoboflowProjects(rfApiKey, workspace);
    },
    onSuccess: (data) => {
      setRfProjects(data);
      setRfSelectedProject("");
      setRfVersions([]);
      setRfSelectedVersion(null);
      setRfPreview(null);
    },
    onError: (error) => {
      toast.error(`Failed to load projects: ${error}`);
    },
  });

  const rfLoadVersionsMutation = useMutation({
    mutationFn: async (project: string) => {
      return apiClient.getRoboflowVersions(rfApiKey, rfSelectedWorkspace, project);
    },
    onSuccess: (data) => {
      setRfVersions(data);
      setRfSelectedVersion(null);
      setRfPreview(null);
    },
    onError: (error) => {
      toast.error(`Failed to load versions: ${error}`);
    },
  });

  const rfLoadPreviewMutation = useMutation({
    mutationFn: async (version: number) => {
      return apiClient.previewRoboflowImport(rfApiKey, rfSelectedWorkspace, rfSelectedProject, version);
    },
    onSuccess: (data) => {
      setRfPreview(data);
      // Initialize class mapping for Roboflow classes
      const mapping: ClassMappingItem[] = data.classes.map((cls, index) => {
        const existingMatch = existingClasses?.find(
          c => c.name.toLowerCase() === cls.name.toLowerCase()
        );
        return {
          source_name: cls.name,
          target_class_id: existingMatch?.id || null,
          create_new: !existingMatch,
          skip: false,
          color: existingMatch?.color || CLASS_COLORS[index % CLASS_COLORS.length],
        };
      });
      setRfClassMapping(mapping);
    },
    onError: (error) => {
      toast.error(`Failed to load preview: ${error}`);
    },
  });

  const rfImportMutation = useMutation({
    mutationFn: async () => {
      if (!datasetId || !rfSelectedVersion) throw new Error("Missing dataset or version");

      const validMapping = rfClassMapping.filter(m => !m.skip).map(m => ({
        source_name: m.source_name,
        target_class_id: m.create_new ? undefined : m.target_class_id || undefined,
        create_new: m.create_new,
        skip: m.skip,
        color: m.color,
      }));

      // Start the import job
      const response = await apiClient.importFromRoboflow({
        api_key: rfApiKey,
        workspace: rfSelectedWorkspace,
        project: rfSelectedProject,
        version: rfSelectedVersion,
        dataset_id: datasetId,
        format: rfFormat,
        class_mapping: validMapping,
      });

      // Set job ID and start polling
      setRfImportJobId(response.job_id);
      setRfImportProgress(0);
      setRfImportStatus("downloading");
      setRfImportMessage("Starting download...");

      // Poll for status
      return new Promise<{ result?: { images_imported?: number; annotations_imported?: number; errors?: string[] } }>((resolve, reject) => {
        const pollInterval = setInterval(async () => {
          try {
            const status = await apiClient.getRoboflowImportStatus(response.job_id);
            setRfImportProgress(status.progress);
            setRfImportStatus(status.result?.stage || status.status);
            setRfImportMessage(status.result?.message || null);

            if (status.status === "completed") {
              clearInterval(pollInterval);
              resolve(status as { result?: { images_imported?: number; annotations_imported?: number; errors?: string[] } });
            } else if (status.status === "failed") {
              clearInterval(pollInterval);
              reject(new Error(status.error || "Import failed"));
            }
          } catch (err) {
            clearInterval(pollInterval);
            reject(err);
          }
        }, 2000); // Poll every 2 seconds
      });
    },
    onSuccess: (data) => {
      const result = data.result || {};
      toast.success(`Imported ${result.images_imported || 0} images with ${result.annotations_imported || 0} annotations`);
      if (result.errors && result.errors.length > 0) {
        toast.warning(`${result.errors.length} errors occurred`);
      }
      queryClient.invalidateQueries({ queryKey: ["od-images"] });
      queryClient.invalidateQueries({ queryKey: ["od-classes"] });
      if (datasetId) {
        queryClient.invalidateQueries({ queryKey: ["od-dataset-images", datasetId] });
      }
      // Reset import state
      setRfImportJobId(null);
      setRfImportProgress(0);
      setRfImportStatus(null);
      setRfImportMessage(null);
      handleOpenChange(false);
      onSuccess?.();
    },
    onError: (error) => {
      toast.error(`Import failed: ${error}`);
      setRfImportJobId(null);
      setRfImportProgress(0);
      setRfImportStatus(null);
      setRfImportMessage(null);
    },
  });

  const handleRfWorkspaceChange = (workspace: string) => {
    setRfSelectedWorkspace(workspace);
    setRfSelectedProject("");
    setRfVersions([]);
    setRfSelectedVersion(null);
    setRfPreview(null);
    rfLoadProjectsMutation.mutate(workspace);
  };

  const handleRfProjectChange = (project: string) => {
    setRfSelectedProject(project);
    setRfSelectedVersion(null);
    setRfPreview(null);
    rfLoadVersionsMutation.mutate(project);
  };

  const handleRfVersionChange = (version: number) => {
    setRfSelectedVersion(version);
    rfLoadPreviewMutation.mutate(version);
  };

  const updateRfClassMapping = (index: number, updates: Partial<ClassMappingItem>) => {
    setRfClassMapping(prev => prev.map((item, i) =>
      i === index ? { ...item, ...updates } : item
    ));
  };

  // Reset state when modal closes
  const handleOpenChange = (open: boolean) => {
    if (!open) {
      setUploadQueue([]);
      setUrlInput("");
      setAnnotatedFile(null);
      setImportPreview(null);
      setClassMapping([]);
      setIsUploading(false);
      setUploadProgress(0);
      // Reset Roboflow state (but keep job tracking if import is in progress)
      if (!rfImportMutation.isPending) {
        setRfApiKey("");
        setRfKeyValidated(false);
        setRfWorkspaces([]);
        setRfProjects([]);
        setRfSelectedWorkspace("");
        setRfSelectedProject("");
        setRfVersions([]);
        setRfSelectedVersion(null);
        setRfPreview(null);
        setRfClassMapping([]);
        setRfImportJobId(null);
        setRfImportProgress(0);
        setRfImportStatus(null);
        setRfImportMessage(null);
      }
    }
    onOpenChange(open);
  };

  // ===========================================
  // Upload Tab
  // ===========================================

  const handleFileDrop = useCallback(async (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const files = Array.from(e.dataTransfer.files).filter(
      f => f.type.startsWith("image/")
    );
    addFilesToQueue(files);
  }, []);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    addFilesToQueue(files);
    e.target.value = ""; // Reset input
  }, []);

  const addFilesToQueue = (files: File[]) => {
    const newItems: UploadQueueItem[] = files.map(file => ({
      file,
      status: "pending",
    }));
    setUploadQueue(prev => [...prev, ...newItems]);
  };

  const removeFromQueue = (index: number) => {
    setUploadQueue(prev => prev.filter((_, i) => i !== index));
  };

  const uploadMutation = useMutation({
    mutationFn: async () => {
      setIsUploading(true);
      const total = uploadQueue.length;
      let uploaded = 0;
      let skipped = 0;

      for (let i = 0; i < uploadQueue.length; i++) {
        const item = uploadQueue[i];
        if (item.status !== "pending") continue;

        setUploadQueue(prev => prev.map((p, idx) =>
          idx === i ? { ...p, status: "uploading" } : p
        ));

        try {
          // Check for duplicates first if enabled
          if (skipDuplicates) {
            const dupCheck = await apiClient.checkODImageDuplicate(item.file);
            if (dupCheck.is_duplicate) {
              setUploadQueue(prev => prev.map((p, idx) =>
                idx === i ? { ...p, status: "duplicate", duplicateId: dupCheck.similar_images[0]?.id } : p
              ));
              skipped++;
              setUploadProgress(Math.round(((i + 1) / total) * 100));
              continue;
            }
          }

          await apiClient.uploadODImage(item.file);

          setUploadQueue(prev => prev.map((p, idx) =>
            idx === i ? { ...p, status: "success" } : p
          ));
          uploaded++;
        } catch (error) {
          setUploadQueue(prev => prev.map((p, idx) =>
            idx === i ? { ...p, status: "error", error: String(error) } : p
          ));
        }

        setUploadProgress(Math.round(((i + 1) / total) * 100));
      }

      return { uploaded, skipped };
    },
    onSuccess: (data) => {
      toast.success(`Uploaded ${data.uploaded} images${data.skipped > 0 ? `, skipped ${data.skipped} duplicates` : ""}`);
      queryClient.invalidateQueries({ queryKey: ["od-images"] });
      if (datasetId) {
        queryClient.invalidateQueries({ queryKey: ["od-dataset-images", datasetId] });
      }
      onSuccess?.();
    },
    onError: (error) => {
      toast.error(`Upload failed: ${error}`);
    },
    onSettled: () => {
      setIsUploading(false);
    },
  });

  // ===========================================
  // URL Tab
  // ===========================================

  const urlImportMutation = useMutation({
    mutationFn: async () => {
      const urls = urlInput.split("\n").map(u => u.trim()).filter(Boolean);
      if (urls.length === 0) throw new Error("No URLs provided");

      return apiClient.importODImagesFromUrls({
        urls,
        folder: urlFolder || undefined,
        skip_duplicates: skipDuplicates,
        dataset_id: datasetId,
      });
    },
    onSuccess: (data) => {
      toast.success(`Imported ${data.images_imported} images${data.duplicates_found > 0 ? `, ${data.duplicates_found} duplicates skipped` : ""}`);
      if (data.errors.length > 0) {
        toast.warning(`${data.errors.length} errors occurred`);
      }
      queryClient.invalidateQueries({ queryKey: ["od-images"] });
      setUrlInput("");
      onSuccess?.();
    },
    onError: (error) => {
      toast.error(`Import failed: ${error}`);
    },
  });

  // ===========================================
  // Annotated Import Tab
  // ===========================================

  const handleAnnotatedFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setAnnotatedFile(file);
    setIsPreviewLoading(true);

    try {
      const preview = await apiClient.previewODImport(file);
      setImportPreview(preview);

      // Initialize class mapping
      const mapping: ClassMappingItem[] = preview.classes_found.map((className, index) => {
        // Try to find matching existing class
        const existingMatch = existingClasses?.find(
          c => c.name.toLowerCase() === className.toLowerCase()
        );

        return {
          source_name: className,
          target_class_id: existingMatch?.id || null,
          create_new: !existingMatch,
          skip: false,
          color: existingMatch?.color || CLASS_COLORS[index % CLASS_COLORS.length],
        };
      });

      setClassMapping(mapping);
    } catch (error) {
      toast.error(`Failed to preview file: ${error}`);
      setAnnotatedFile(null);
    } finally {
      setIsPreviewLoading(false);
    }
  };

  const updateClassMapping = (index: number, updates: Partial<ClassMappingItem>) => {
    setClassMapping(prev => prev.map((item, i) =>
      i === index ? { ...item, ...updates } : item
    ));
  };

  const annotatedImportMutation = useMutation({
    mutationFn: async () => {
      if (!annotatedFile || !datasetId) throw new Error("Missing file or dataset");

      const validMapping = classMapping.filter(m => !m.skip).map(m => ({
        source_name: m.source_name,
        target_class_id: m.create_new ? undefined : m.target_class_id || undefined,
        create_new: m.create_new,
        skip: m.skip,
        color: m.color,
      }));

      return apiClient.importODAnnotatedDataset(
        annotatedFile,
        datasetId,
        validMapping,
        { skip_duplicates: skipDuplicates, merge_annotations: mergeAnnotations }
      );
    },
    onSuccess: (data) => {
      toast.success(`Imported ${data.images_imported} images with ${data.annotations_imported} annotations`);
      if (data.errors.length > 0) {
        toast.warning(`${data.errors.length} errors occurred`);
      }
      queryClient.invalidateQueries({ queryKey: ["od-images"] });
      queryClient.invalidateQueries({ queryKey: ["od-classes"] });
      queryClient.invalidateQueries({ queryKey: ["od-dataset-images", datasetId] });
      handleOpenChange(false);
      onSuccess?.();
    },
    onError: (error) => {
      toast.error(`Import failed: ${error}`);
    },
  });

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="max-w-3xl max-h-[90vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Package className="h-5 w-5" />
            Import Images
          </DialogTitle>
          <DialogDescription>
            {datasetId
              ? "Import images directly into this dataset"
              : "Import images to your library"}
          </DialogDescription>
        </DialogHeader>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1 overflow-hidden flex flex-col">
          <TabsList className="grid w-full grid-cols-5">
            <TabsTrigger value="upload" className="flex items-center gap-1.5 text-xs">
              <Upload className="h-3.5 w-3.5" />
              Upload
            </TabsTrigger>
            <TabsTrigger value="url" className="flex items-center gap-1.5 text-xs">
              <LinkIcon className="h-3.5 w-3.5" />
              URL
            </TabsTrigger>
            <TabsTrigger value="annotated" className="flex items-center gap-1.5 text-xs" disabled={!datasetId}>
              <FileArchive className="h-3.5 w-3.5" />
              Annotated
            </TabsTrigger>
            <TabsTrigger value="roboflow" className="flex items-center gap-1.5 text-xs" disabled={!datasetId}>
              <Package className="h-3.5 w-3.5" />
              Roboflow
            </TabsTrigger>
            <TabsTrigger value="buybuddy" className="flex items-center gap-1.5 text-xs">
              <Cloud className="h-3.5 w-3.5" />
              BuyBuddy
            </TabsTrigger>
          </TabsList>

          {/* Upload Tab */}
          <TabsContent value="upload" className="flex-1 overflow-auto space-y-4">
            {/* Drop Zone */}
            <div
              className="border-2 border-dashed rounded-lg p-8 text-center hover:border-primary transition-colors cursor-pointer"
              onDrop={handleFileDrop}
              onDragOver={(e) => e.preventDefault()}
              onClick={() => document.getElementById("file-input")?.click()}
            >
              <Upload className="h-10 w-10 mx-auto text-muted-foreground mb-4" />
              <p className="text-lg font-medium">Drop images here or click to browse</p>
              <p className="text-sm text-muted-foreground mt-1">
                Supports JPG, PNG, WebP, BMP
              </p>
              <input
                id="file-input"
                type="file"
                multiple
                accept="image/*"
                className="hidden"
                onChange={handleFileSelect}
              />
            </div>

            {/* Options */}
            <div className="flex items-center gap-2">
              <Checkbox
                id="skip-dup"
                checked={skipDuplicates}
                onCheckedChange={(v) => setSkipDuplicates(!!v)}
              />
              <Label htmlFor="skip-dup">Skip duplicate images (by content hash)</Label>
            </div>

            {/* Upload Queue */}
            {uploadQueue.length > 0 && (
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label>Upload Queue ({uploadQueue.length} files)</Label>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setUploadQueue([])}
                    disabled={isUploading}
                  >
                    Clear All
                  </Button>
                </div>

                {isUploading && (
                  <Progress value={uploadProgress} className="h-2" />
                )}

                <div className="max-h-48 overflow-auto border rounded-md">
                  <Table>
                    <TableBody>
                      {uploadQueue.map((item, index) => (
                        <TableRow key={index}>
                          <TableCell className="py-2">
                            <span className="truncate max-w-[200px] block">{item.file.name}</span>
                          </TableCell>
                          <TableCell className="py-2 text-right text-muted-foreground">
                            {(item.file.size / 1024 / 1024).toFixed(1)} MB
                          </TableCell>
                          <TableCell className="py-2 w-24">
                            {item.status === "pending" && (
                              <Badge variant="secondary">Pending</Badge>
                            )}
                            {item.status === "uploading" && (
                              <Badge variant="default" className="flex items-center gap-1">
                                <Loader2 className="h-3 w-3 animate-spin" />
                                Uploading
                              </Badge>
                            )}
                            {item.status === "success" && (
                              <Badge variant="default" className="bg-green-600">
                                <CheckCircle className="h-3 w-3 mr-1" />
                                Done
                              </Badge>
                            )}
                            {item.status === "duplicate" && (
                              <Badge variant="outline" className="text-amber-600 border-amber-600">
                                <AlertTriangle className="h-3 w-3 mr-1" />
                                Duplicate
                              </Badge>
                            )}
                            {item.status === "error" && (
                              <Badge variant="destructive">
                                <XCircle className="h-3 w-3 mr-1" />
                                Error
                              </Badge>
                            )}
                          </TableCell>
                          <TableCell className="py-2 w-10">
                            {item.status === "pending" && !isUploading && (
                              <Button
                                variant="ghost"
                                size="sm"
                                className="h-6 w-6 p-0"
                                onClick={() => removeFromQueue(index)}
                              >
                                <Minus className="h-4 w-4" />
                              </Button>
                            )}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>

                <Button
                  className="w-full"
                  onClick={() => uploadMutation.mutate()}
                  disabled={isUploading || uploadQueue.every(i => i.status !== "pending")}
                >
                  {isUploading ? (
                    <>
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      Uploading... {uploadProgress}%
                    </>
                  ) : (
                    <>
                      <Upload className="h-4 w-4 mr-2" />
                      Upload {uploadQueue.filter(i => i.status === "pending").length} Images
                    </>
                  )}
                </Button>
              </div>
            )}
          </TabsContent>

          {/* URL Tab */}
          <TabsContent value="url" className="flex-1 overflow-auto space-y-4">
            <div className="space-y-2">
              <Label>Image URLs (one per line)</Label>
              <Textarea
                value={urlInput}
                onChange={(e) => setUrlInput(e.target.value)}
                placeholder="https://example.com/image1.jpg&#10;https://example.com/image2.png&#10;..."
                rows={8}
              />
              <p className="text-xs text-muted-foreground">
                {urlInput.split("\n").filter(u => u.trim()).length} URLs detected
              </p>
            </div>

            <div className="space-y-2">
              <Label>Folder (optional)</Label>
              <Input
                value={urlFolder}
                onChange={(e) => setUrlFolder(e.target.value)}
                placeholder="e.g., imported/batch-1"
              />
            </div>

            <div className="flex items-center gap-2">
              <Checkbox
                id="skip-dup-url"
                checked={skipDuplicates}
                onCheckedChange={(v) => setSkipDuplicates(!!v)}
              />
              <Label htmlFor="skip-dup-url">Skip duplicate images</Label>
            </div>

            <Button
              className="w-full"
              onClick={() => urlImportMutation.mutate()}
              disabled={urlImportMutation.isPending || !urlInput.trim()}
            >
              {urlImportMutation.isPending ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Importing...
                </>
              ) : (
                <>
                  <LinkIcon className="h-4 w-4 mr-2" />
                  Import from URLs
                </>
              )}
            </Button>
          </TabsContent>

          {/* Annotated Import Tab */}
          <TabsContent value="annotated" className="flex-1 overflow-auto space-y-4">
            {!datasetId ? (
              <div className="text-center py-8 text-muted-foreground">
                <Package className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>Select a dataset to import annotated data</p>
              </div>
            ) : (
              <>
                {/* File Selection */}
                {!annotatedFile && (
                  <div
                    className="border-2 border-dashed rounded-lg p-8 text-center hover:border-primary transition-colors cursor-pointer"
                    onClick={() => document.getElementById("annotated-input")?.click()}
                  >
                    <FileArchive className="h-10 w-10 mx-auto text-muted-foreground mb-4" />
                    <p className="text-lg font-medium">Select annotation file or ZIP</p>
                    <p className="text-sm text-muted-foreground mt-1">
                      Supports COCO JSON, YOLO, Pascal VOC, LabelMe
                    </p>
                    <input
                      id="annotated-input"
                      type="file"
                      accept=".json,.zip,.xml,.yaml,.yml"
                      className="hidden"
                      onChange={handleAnnotatedFileSelect}
                    />
                  </div>
                )}

                {/* Loading Preview */}
                {isPreviewLoading && (
                  <div className="text-center py-8">
                    <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4" />
                    <p>Analyzing file...</p>
                  </div>
                )}

                {/* Preview Results */}
                {importPreview && (
                  <div className="space-y-4">
                    {/* File Info */}
                    <div className="flex items-center justify-between bg-muted/50 rounded-lg p-3">
                      <div className="flex items-center gap-3">
                        <FileArchive className="h-8 w-8 text-muted-foreground" />
                        <div>
                          <p className="font-medium">{annotatedFile?.name}</p>
                          <p className="text-sm text-muted-foreground">
                            Format: <Badge variant="outline">{importPreview.format_detected.toUpperCase()}</Badge>
                          </p>
                        </div>
                      </div>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => {
                          setAnnotatedFile(null);
                          setImportPreview(null);
                          setClassMapping([]);
                        }}
                      >
                        Change
                      </Button>
                    </div>

                    {/* Stats */}
                    <div className="grid grid-cols-3 gap-4">
                      <div className="bg-muted/30 rounded-lg p-3 text-center">
                        <p className="text-2xl font-bold">{importPreview.total_images}</p>
                        <p className="text-sm text-muted-foreground">Images</p>
                      </div>
                      <div className="bg-muted/30 rounded-lg p-3 text-center">
                        <p className="text-2xl font-bold">{importPreview.total_annotations}</p>
                        <p className="text-sm text-muted-foreground">Annotations</p>
                      </div>
                      <div className="bg-muted/30 rounded-lg p-3 text-center">
                        <p className="text-2xl font-bold">{importPreview.classes_found.length}</p>
                        <p className="text-sm text-muted-foreground">Classes</p>
                      </div>
                    </div>

                    {/* Class Mapping */}
                    <div className="space-y-2">
                      <Label>Class Mapping</Label>
                      <div className="border rounded-lg max-h-48 overflow-auto">
                        <Table>
                          <TableHeader>
                            <TableRow>
                              <TableHead>Source Class</TableHead>
                              <TableHead>
                                <ArrowRight className="h-4 w-4" />
                              </TableHead>
                              <TableHead>Target Class</TableHead>
                              <TableHead className="w-20">Skip</TableHead>
                            </TableRow>
                          </TableHeader>
                          <TableBody>
                            {classMapping.map((mapping, index) => (
                              <TableRow key={mapping.source_name}>
                                <TableCell className="py-2">
                                  <div className="flex items-center gap-2">
                                    <div
                                      className="w-4 h-4 rounded"
                                      style={{ backgroundColor: mapping.color }}
                                    />
                                    {mapping.source_name}
                                  </div>
                                </TableCell>
                                <TableCell className="py-2">
                                  <ArrowRight className="h-4 w-4 text-muted-foreground" />
                                </TableCell>
                                <TableCell className="py-2">
                                  {mapping.skip ? (
                                    <span className="text-muted-foreground">Skipped</span>
                                  ) : mapping.create_new ? (
                                    <Badge variant="outline" className="bg-green-50">
                                      <Plus className="h-3 w-3 mr-1" />
                                      Create New
                                    </Badge>
                                  ) : (
                                    <Select
                                      value={mapping.target_class_id || ""}
                                      onValueChange={(v) => updateClassMapping(index, {
                                        target_class_id: v || null,
                                        create_new: !v
                                      })}
                                    >
                                      <SelectTrigger className="h-8">
                                        <SelectValue placeholder="Select class..." />
                                      </SelectTrigger>
                                      <SelectContent>
                                        <SelectItem value="__new__">
                                          <Plus className="h-3 w-3 mr-1 inline" />
                                          Create New
                                        </SelectItem>
                                        {existingClasses?.map(cls => (
                                          <SelectItem key={cls.id} value={cls.id}>
                                            <div className="flex items-center gap-2">
                                              <div
                                                className="w-3 h-3 rounded"
                                                style={{ backgroundColor: cls.color }}
                                              />
                                              {cls.name}
                                            </div>
                                          </SelectItem>
                                        ))}
                                      </SelectContent>
                                    </Select>
                                  )}
                                </TableCell>
                                <TableCell className="py-2">
                                  <Checkbox
                                    checked={mapping.skip}
                                    onCheckedChange={(v) => updateClassMapping(index, { skip: !!v })}
                                  />
                                </TableCell>
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      </div>
                    </div>

                    {/* Options */}
                    <div className="space-y-2">
                      <div className="flex items-center gap-2">
                        <Checkbox
                          id="skip-dup-ann"
                          checked={skipDuplicates}
                          onCheckedChange={(v) => setSkipDuplicates(!!v)}
                        />
                        <Label htmlFor="skip-dup-ann">Skip duplicate images</Label>
                      </div>
                      <div className="flex items-center gap-2">
                        <Checkbox
                          id="merge-ann"
                          checked={mergeAnnotations}
                          onCheckedChange={(v) => setMergeAnnotations(!!v)}
                        />
                        <Label htmlFor="merge-ann">Merge annotations if image exists</Label>
                      </div>
                    </div>

                    {/* Errors */}
                    {importPreview.errors.length > 0 && (
                      <div className="bg-red-50 border border-red-200 rounded-lg p-3">
                        <p className="font-medium text-red-800 mb-1">Warnings</p>
                        {importPreview.errors.map((err, i) => (
                          <p key={i} className="text-sm text-red-600">{err}</p>
                        ))}
                      </div>
                    )}

                    {/* Import Button */}
                    <Button
                      className="w-full"
                      onClick={() => annotatedImportMutation.mutate()}
                      disabled={annotatedImportMutation.isPending}
                    >
                      {annotatedImportMutation.isPending ? (
                        <>
                          <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                          Importing...
                        </>
                      ) : (
                        <>
                          <Package className="h-4 w-4 mr-2" />
                          Import {importPreview.total_images} Images with Annotations
                        </>
                      )}
                    </Button>
                  </div>
                )}
              </>
            )}
          </TabsContent>

          {/* Roboflow Tab */}
          <TabsContent value="roboflow" className="flex-1 overflow-auto space-y-4">
            {!datasetId ? (
              <div className="text-center py-8 text-muted-foreground">
                <Package className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>Select a dataset to import from Roboflow</p>
              </div>
            ) : (
              <div className="space-y-4">
                {/* Step 1: API Key */}
                {!rfKeyValidated ? (
                  <div className="space-y-4">
                    <div className="bg-muted/50 rounded-lg p-4">
                      <div className="flex items-center gap-3 mb-3">
                        <div className="h-8 w-8 rounded-full bg-primary/10 flex items-center justify-center">
                          <Key className="h-4 w-4 text-primary" />
                        </div>
                        <div>
                          <p className="font-medium">Connect to Roboflow</p>
                          <p className="text-sm text-muted-foreground">
                            Enter your Roboflow API key to browse projects
                          </p>
                        </div>
                      </div>
                      <div className="flex gap-2">
                        <Input
                          type="password"
                          value={rfApiKey}
                          onChange={(e) => setRfApiKey(e.target.value)}
                          placeholder="Enter your Roboflow API key"
                          className="flex-1"
                        />
                        <Button
                          onClick={() => rfValidateKeyMutation.mutate(rfApiKey)}
                          disabled={!rfApiKey || rfValidateKeyMutation.isPending}
                        >
                          {rfValidateKeyMutation.isPending ? (
                            <Loader2 className="h-4 w-4 animate-spin" />
                          ) : (
                            "Connect"
                          )}
                        </Button>
                      </div>
                      <p className="text-xs text-muted-foreground mt-2">
                        Find your API key at{" "}
                        <a
                          href="https://app.roboflow.com/settings/api"
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-primary hover:underline"
                        >
                          roboflow.com/settings/api
                        </a>
                      </p>
                    </div>
                  </div>
                ) : (
                  <>
                    {/* Connected Status */}
                    <div className="flex items-center justify-between bg-green-50 border border-green-200 rounded-lg p-3">
                      <div className="flex items-center gap-2">
                        <CheckCircle className="h-5 w-5 text-green-600" />
                        <span className="text-green-800 font-medium">Connected to Roboflow</span>
                      </div>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => {
                          setRfKeyValidated(false);
                          setRfWorkspaces([]);
                          setRfProjects([]);
                          setRfSelectedWorkspace("");
                          setRfSelectedProject("");
                          setRfVersions([]);
                          setRfSelectedVersion(null);
                          setRfPreview(null);
                        }}
                      >
                        Disconnect
                      </Button>
                    </div>

                    {/* Step 2: Select Workspace/Project/Version */}
                    <div className="space-y-3">
                      {/* Workspace Selection */}
                      {rfWorkspaces.length > 1 && (
                        <div className="space-y-2">
                          <Label className="text-sm">Workspace</Label>
                          <Select
                            value={rfSelectedWorkspace}
                            onValueChange={handleRfWorkspaceChange}
                          >
                            <SelectTrigger>
                              <SelectValue placeholder="Select workspace..." />
                            </SelectTrigger>
                            <SelectContent>
                              {rfWorkspaces.map((ws) => (
                                <SelectItem key={ws.url} value={ws.url}>
                                  <div className="flex items-center gap-2">
                                    <FolderOpen className="h-4 w-4" />
                                    {ws.name}
                                    <span className="text-muted-foreground text-xs">
                                      ({ws.projects} projects)
                                    </span>
                                  </div>
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        </div>
                      )}

                      {/* Project Selection */}
                      <div className="space-y-2">
                        <Label className="text-sm">Project</Label>
                        <Select
                          value={rfSelectedProject}
                          onValueChange={handleRfProjectChange}
                          disabled={rfProjects.length === 0 || rfLoadProjectsMutation.isPending}
                        >
                          <SelectTrigger>
                            <SelectValue placeholder={
                              rfLoadProjectsMutation.isPending ? "Loading projects..." : "Select project..."
                            } />
                          </SelectTrigger>
                          <SelectContent>
                            {rfProjects.map((proj) => (
                              <SelectItem key={proj.id} value={proj.id}>
                                <div className="flex items-center justify-between w-full">
                                  <span>{proj.name}</span>
                                  <span className="text-muted-foreground text-xs ml-2">
                                    {proj.images} images{proj.versions !== undefined ? ` • ${proj.versions} versions` : ''}
                                  </span>
                                </div>
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>

                      {/* Version Selection */}
                      {rfSelectedProject && (
                        <div className="space-y-2">
                          <Label className="text-sm">Version</Label>
                          <Select
                            value={rfSelectedVersion?.toString() || ""}
                            onValueChange={(v) => handleRfVersionChange(parseInt(v))}
                            disabled={rfVersions.length === 0 || rfLoadVersionsMutation.isPending}
                          >
                            <SelectTrigger>
                              <SelectValue placeholder={
                                rfLoadVersionsMutation.isPending ? "Loading versions..." : "Select version..."
                              } />
                            </SelectTrigger>
                            <SelectContent>
                              {rfVersions.map((ver) => (
                                <SelectItem key={ver.id} value={ver.version.toString()}>
                                  <div className="flex items-center justify-between w-full">
                                    <span>v{ver.version}</span>
                                    <span className="text-muted-foreground text-xs ml-2">
                                      {Object.values(ver.images || {}).reduce((a, b) => a + b, 0)} images
                                    </span>
                                  </div>
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        </div>
                      )}
                    </div>

                    {/* Loading Preview */}
                    {rfLoadPreviewMutation.isPending && (
                      <div className="text-center py-8">
                        <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4" />
                        <p>Loading dataset preview...</p>
                      </div>
                    )}

                    {/* Preview Results */}
                    {rfPreview && (
                      <div className="space-y-4">
                        {/* Stats */}
                        <div className="grid grid-cols-3 gap-4">
                          <div className="bg-muted/30 rounded-lg p-3 text-center">
                            <p className="text-2xl font-bold">{rfPreview.total_images}</p>
                            <p className="text-sm text-muted-foreground">Images</p>
                          </div>
                          <div className="bg-muted/30 rounded-lg p-3 text-center">
                            <p className="text-2xl font-bold">
                              {rfPreview.classes.reduce((sum, c) => sum + c.count, 0)}
                            </p>
                            <p className="text-sm text-muted-foreground">Annotations</p>
                          </div>
                          <div className="bg-muted/30 rounded-lg p-3 text-center">
                            <p className="text-2xl font-bold">{rfPreview.class_count}</p>
                            <p className="text-sm text-muted-foreground">Classes</p>
                          </div>
                        </div>

                        {/* Splits Info */}
                        {Object.keys(rfPreview.splits || {}).length > 0 && (
                          <div className="flex gap-2 flex-wrap">
                            {Object.entries(rfPreview.splits).map(([split, count]) => (
                              <Badge key={split} variant="outline">
                                {split}: {count}
                              </Badge>
                            ))}
                          </div>
                        )}

                        {/* Export Format */}
                        <div className="space-y-2">
                          <Label className="text-sm">Export Format</Label>
                          <Select
                            value={rfFormat}
                            onValueChange={(v) => setRfFormat(v as "coco" | "yolov8")}
                          >
                            <SelectTrigger className="w-40">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="coco">COCO JSON</SelectItem>
                              <SelectItem value="yolov8">YOLOv8</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>

                        {/* Class Mapping */}
                        <div className="space-y-2">
                          <Label>Class Mapping</Label>
                          <div className="border rounded-lg max-h-48 overflow-auto">
                            <Table>
                              <TableHeader>
                                <TableRow>
                                  <TableHead>Roboflow Class</TableHead>
                                  <TableHead>
                                    <ArrowRight className="h-4 w-4" />
                                  </TableHead>
                                  <TableHead>Target Class</TableHead>
                                  <TableHead className="w-20">Skip</TableHead>
                                </TableRow>
                              </TableHeader>
                              <TableBody>
                                {rfClassMapping.map((mapping, index) => (
                                  <TableRow key={mapping.source_name}>
                                    <TableCell className="py-2">
                                      <div className="flex items-center gap-2">
                                        <div
                                          className="w-4 h-4 rounded"
                                          style={{ backgroundColor: mapping.color }}
                                        />
                                        {mapping.source_name}
                                        <span className="text-xs text-muted-foreground">
                                          ({rfPreview.classes.find(c => c.name === mapping.source_name)?.count || 0})
                                        </span>
                                      </div>
                                    </TableCell>
                                    <TableCell className="py-2">
                                      <ArrowRight className="h-4 w-4 text-muted-foreground" />
                                    </TableCell>
                                    <TableCell className="py-2">
                                      {mapping.skip ? (
                                        <span className="text-muted-foreground">Skipped</span>
                                      ) : mapping.create_new ? (
                                        <Badge variant="outline" className="bg-green-50">
                                          <Plus className="h-3 w-3 mr-1" />
                                          Create New
                                        </Badge>
                                      ) : (
                                        <Select
                                          value={mapping.target_class_id || ""}
                                          onValueChange={(v) => {
                                            if (v === "__new__") {
                                              updateRfClassMapping(index, { target_class_id: null, create_new: true });
                                            } else {
                                              updateRfClassMapping(index, { target_class_id: v, create_new: false });
                                            }
                                          }}
                                        >
                                          <SelectTrigger className="h-8">
                                            <SelectValue placeholder="Select class..." />
                                          </SelectTrigger>
                                          <SelectContent>
                                            <SelectItem value="__new__">
                                              <Plus className="h-3 w-3 mr-1 inline" />
                                              Create New
                                            </SelectItem>
                                            {existingClasses?.map(cls => (
                                              <SelectItem key={cls.id} value={cls.id}>
                                                <div className="flex items-center gap-2">
                                                  <div
                                                    className="w-3 h-3 rounded"
                                                    style={{ backgroundColor: cls.color }}
                                                  />
                                                  {cls.name}
                                                </div>
                                              </SelectItem>
                                            ))}
                                          </SelectContent>
                                        </Select>
                                      )}
                                    </TableCell>
                                    <TableCell className="py-2">
                                      <Checkbox
                                        checked={mapping.skip}
                                        onCheckedChange={(v) => updateRfClassMapping(index, { skip: !!v })}
                                      />
                                    </TableCell>
                                  </TableRow>
                                ))}
                              </TableBody>
                            </Table>
                          </div>
                        </div>

                        {/* Resume Banner for Interrupted Import */}
                        {rfResumableJob && !rfImportMutation.isPending && (
                          <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
                            <div className="flex items-start justify-between gap-4">
                              <div className="flex-1">
                                <p className="font-medium text-amber-800 flex items-center gap-2">
                                  <AlertTriangle className="h-4 w-4" />
                                  Interrupted import found
                                </p>
                                <p className="text-sm text-amber-700 mt-1">
                                  {rfResumableJob.result?.processed_count || 0}/
                                  {rfResumableJob.result?.total_images || "?"} images were imported from{" "}
                                  {rfResumableJob.config?.project}
                                </p>
                              </div>
                              <div className="flex gap-2 flex-shrink-0">
                                <Button
                                  variant="outline"
                                  size="sm"
                                  onClick={() => setRfResumableJob(null)}
                                >
                                  Start Fresh
                                </Button>
                                <Button
                                  size="sm"
                                  onClick={() => {
                                    retryImport.mutate(rfResumableJob.id);
                                    setRfResumableJob(null);
                                    handleOpenChange(false);
                                  }}
                                  disabled={retryImport.isPending}
                                >
                                  {retryImport.isPending ? (
                                    <Loader2 className="h-4 w-4 animate-spin" />
                                  ) : (
                                    <>
                                      <RefreshCw className="h-4 w-4 mr-1" />
                                      Resume Import
                                    </>
                                  )}
                                </Button>
                              </div>
                            </div>
                          </div>
                        )}

                        {/* Import Progress */}
                        {rfImportMutation.isPending && (
                          <div className="space-y-2 bg-muted/50 rounded-lg p-4">
                            <div className="flex items-center justify-between">
                              <span className="text-sm font-medium">
                                {rfImportMessage || (
                                  <>
                                    {rfImportStatus === "downloading" && "Downloading from Roboflow..."}
                                    {rfImportStatus === "processing" && "Processing images..."}
                                    {rfImportStatus === "running" && "Importing..."}
                                    {rfImportStatus === "streaming" && "Streaming images..."}
                                    {!rfImportStatus && "Starting import..."}
                                  </>
                                )}
                              </span>
                              <span className="text-sm text-muted-foreground">{rfImportProgress}%</span>
                            </div>
                            <Progress value={rfImportProgress} className="h-2" />
                            <p className="text-xs text-muted-foreground text-center">
                              You can close this dialog - the import will continue in the background
                            </p>
                          </div>
                        )}

                        {/* Import Button */}
                        <Button
                          className="w-full"
                          onClick={() => rfImportMutation.mutate()}
                          disabled={rfImportMutation.isPending}
                        >
                          {rfImportMutation.isPending ? (
                            <>
                              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                              Importing... {rfImportProgress}%
                            </>
                          ) : (
                            <>
                              <Package className="h-4 w-4 mr-2" />
                              Import {rfPreview.total_images} Images from Roboflow
                            </>
                          )}
                        </Button>
                      </div>
                    )}
                  </>
                )}
              </div>
            )}
          </TabsContent>

          {/* BuyBuddy Sync Tab */}
          <TabsContent value="buybuddy" className="flex-1 overflow-auto space-y-4">
            {!bbStatus?.configured ? (
              <div className="text-center py-8 text-muted-foreground">
                <Cloud className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p className="font-medium">BuyBuddy API Not Configured</p>
                <p className="text-sm mt-1">
                  Add BuyBuddy API credentials in environment variables to enable sync.
                </p>
              </div>
            ) : !bbStatus?.accessible ? (
              <div className="text-center py-8 text-muted-foreground">
                <AlertTriangle className="h-12 w-12 mx-auto mb-4 text-amber-500" />
                <p className="font-medium">Cannot Connect to BuyBuddy API</p>
                <p className="text-sm mt-1">{bbStatus?.message}</p>
              </div>
            ) : bbPreviewLoading ? (
              <div className="text-center py-8">
                <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4" />
                <p>Loading available images...</p>
              </div>
            ) : (
              <div className="space-y-4">
                {/* Status */}
                <div className="flex items-center justify-between bg-green-50 border border-green-200 rounded-lg p-3">
                  <div className="flex items-center gap-2">
                    <CheckCircle className="h-5 w-5 text-green-600" />
                    <span className="text-green-800 font-medium">BuyBuddy API Connected</span>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => refetchBbPreview()}
                  >
                    <RefreshCw className="h-4 w-4 mr-1" />
                    Refresh
                  </Button>
                </div>

                {/* Available Images Count */}
                <div className="bg-muted/50 rounded-lg p-4 text-center">
                  <p className="text-3xl font-bold">{bbPreview?.total_available ?? 0}</p>
                  <p className="text-sm text-muted-foreground">Images available for sync</p>
                </div>

                {/* Preview Grid */}
                {bbPreview?.sample_images && bbPreview.sample_images.length > 0 && (
                  <div className="space-y-2">
                    <Label>Preview ({bbPreview.sample_count} shown)</Label>
                    <div className="grid grid-cols-5 gap-2 max-h-32 overflow-hidden">
                      {bbPreview.sample_images.slice(0, 10).map((img, i) => (
                        <div key={i} className="aspect-square rounded-md overflow-hidden bg-muted">
                          {/* eslint-disable-next-line @next/next/no-img-element */}
                          <img
                            src={img.image_url}
                            alt={`Preview ${i + 1}`}
                            className="w-full h-full object-cover"
                          />
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Filters */}
                <div className="space-y-4 border rounded-lg p-4 bg-muted/30">
                  <Label className="text-sm font-medium">Filters</Label>

                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label className="text-xs text-muted-foreground">Start Date</Label>
                      <Input
                        type="date"
                        value={bbStartDate}
                        onChange={(e) => setBbStartDate(e.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label className="text-xs text-muted-foreground">End Date</Label>
                      <Input
                        type="date"
                        value={bbEndDate}
                        onChange={(e) => setBbEndDate(e.target.value)}
                      />
                    </div>
                  </div>

                  <div className="space-y-2">
                    <Label className="text-xs text-muted-foreground">Store ID (optional)</Label>
                    <Input
                      type="number"
                      value={bbStoreId}
                      onChange={(e) => setBbStoreId(e.target.value)}
                      placeholder="e.g., 123"
                    />
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label className="text-xs text-muted-foreground">Annotation Status</Label>
                      <Select
                        value={bbIsAnnotated === undefined ? "all" : bbIsAnnotated ? "annotated" : "not_annotated"}
                        onValueChange={(v) => setBbIsAnnotated(v === "all" ? undefined : v === "annotated")}
                      >
                        <SelectTrigger>
                          <SelectValue placeholder="All" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="all">All</SelectItem>
                          <SelectItem value="annotated">Annotated Only</SelectItem>
                          <SelectItem value="not_annotated">Not Annotated</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <Label className="text-xs text-muted-foreground">Approval Status</Label>
                      <Select
                        value={bbIsApproved === undefined ? "all" : bbIsApproved ? "approved" : "not_approved"}
                        onValueChange={(v) => setBbIsApproved(v === "all" ? undefined : v === "approved")}
                      >
                        <SelectTrigger>
                          <SelectValue placeholder="All" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="all">All</SelectItem>
                          <SelectItem value="approved">Approved Only</SelectItem>
                          <SelectItem value="not_approved">Not Approved</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                </div>

                {/* Options */}
                <div className="space-y-4">
                  <div className="space-y-2">
                    <Label>Max Images to Sync</Label>
                    <Input
                      type="number"
                      value={bbMaxImages}
                      onChange={(e) => setBbMaxImages(parseInt(e.target.value) || 100)}
                      min={1}
                      max={10000}
                    />
                    <p className="text-xs text-muted-foreground">
                      Limit how many images to import in this sync
                    </p>
                  </div>

                  <div className="space-y-2">
                    <Label>Tag (optional)</Label>
                    <Input
                      value={bbFolder}
                      onChange={(e) => setBbFolder(e.target.value)}
                      placeholder="e.g., buybuddy_sync"
                    />
                    <p className="text-xs text-muted-foreground">
                      Tag synced images for organization
                    </p>
                  </div>

                  {datasetId && (
                    <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                      <p className="text-sm text-blue-800">
                        Images will be added directly to the current dataset after sync.
                      </p>
                    </div>
                  )}
                </div>

                {/* Sync Button */}
                <Button
                  className="w-full"
                  onClick={() => bbSyncMutation.mutate()}
                  disabled={bbSyncMutation.isPending || (bbPreview?.total_available ?? 0) === 0}
                >
                  {bbSyncMutation.isPending ? (
                    <>
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      Syncing...
                    </>
                  ) : (
                    <>
                      <Cloud className="h-4 w-4 mr-2" />
                      Sync {Math.min(bbMaxImages, bbPreview?.total_available ?? 0)} Images from BuyBuddy
                    </>
                  )}
                </Button>
              </div>
            )}
          </TabsContent>
        </Tabs>
      </DialogContent>
    </Dialog>
  );
}
