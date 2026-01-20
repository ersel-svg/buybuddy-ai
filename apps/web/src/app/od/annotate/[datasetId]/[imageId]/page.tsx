"use client";

import { useState, useRef, useEffect, useCallback, use } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { useRouter } from "next/navigation";
import { apiClient } from "@/lib/api-client";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Loader2,
  ChevronLeft,
  ChevronRight,
  Trash2,
  CheckCircle,
  Layers,
  Eye,
  EyeOff,
  Home,
  Plus,
  MousePointer2,
  Square,
  Undo2,
  Redo2,
  Copy,
  ZoomIn,
  ZoomOut,
  Maximize2,
  Repeat,
  Keyboard,
  X,
  Crosshair,
} from "lucide-react";
import Link from "next/link";
import { AIPanel } from "@/components/od/ai-panel";
import type { AIPrediction } from "@/types/od";

// ============================================
// TYPES
// ============================================

interface Annotation {
  id: string;
  class_id: string;
  class_name: string;
  class_color: string;
  bbox: { x: number; y: number; width: number; height: number };
  is_ai_generated: boolean;
  confidence?: number;
}

interface DrawingBox {
  startX: number;
  startY: number;
  endX: number;
  endY: number;
}

type ToolMode = "select" | "bbox" | "sam";
type ResizeHandle = "nw" | "n" | "ne" | "e" | "se" | "s" | "sw" | "w" | null;

interface HistoryState {
  annotations: Annotation[];
  description: string;
}

// ============================================
// UTILITY FUNCTIONS
// ============================================

function generateRandomColor(): string {
  const colors = [
    "#EF4444", "#F97316", "#F59E0B", "#EAB308", "#84CC16",
    "#22C55E", "#10B981", "#14B8A6", "#06B6D4", "#0EA5E9",
    "#3B82F6", "#6366F1", "#8B5CF6", "#A855F7", "#D946EF",
    "#EC4899", "#F43F5E",
  ];
  return colors[Math.floor(Math.random() * colors.length)];
}

// ============================================
// MAIN COMPONENT
// ============================================

export default function ODAnnotationEditorPage({
  params,
}: {
  params: Promise<{ datasetId: string; imageId: string }>;
}) {
  const { datasetId, imageId } = use(params);
  const router = useRouter();
  const queryClient = useQueryClient();

  // Refs
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const classInputRef = useRef<HTMLInputElement>(null);

  // ============================================
  // STATE
  // ============================================

  // Tool mode
  const [toolMode, setToolMode] = useState<ToolMode>("bbox");
  const [continuousDrawMode, setContinuousDrawMode] = useState(true);

  // Annotations
  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [hoveredId, setHoveredId] = useState<string | null>(null);

  // Drawing
  const [isDrawing, setIsDrawing] = useState(false);
  const [drawingBox, setDrawingBox] = useState<DrawingBox | null>(null);
  const [mousePos, setMousePos] = useState<{ canvasX: number; canvasY: number } | null>(null);

  // Dragging & Resizing
  const [isDragging, setIsDragging] = useState(false);
  const [isResizing, setIsResizing] = useState(false);
  const [resizeHandle, setResizeHandle] = useState<ResizeHandle>(null);
  const [dragStart, setDragStart] = useState<{ x: number; y: number } | null>(null);
  const [originalBboxes, setOriginalBboxes] = useState<Map<string, Annotation["bbox"]>>(new Map());

  // Zoom & Pan
  const [scale, setScale] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState<{ x: number; y: number; offsetX: number; offsetY: number } | null>(null);
  const [spacePressed, setSpacePressed] = useState(false);

  // Display
  const [showAnnotations, setShowAnnotations] = useState(true);
  const [imageLoaded, setImageLoaded] = useState(false);

  // Class input
  const [pendingBbox, setPendingBbox] = useState<{ x: number; y: number; width: number; height: number } | null>(null);
  const [classInput, setClassInput] = useState("");
  const [showClassInput, setShowClassInput] = useState(false);
  const [lastUsedClassId, setLastUsedClassId] = useState<string | null>(null);

  // Quick class change (inline at bbox position)
  const [showClassChange, setShowClassChange] = useState(false);
  const [classChangeInput, setClassChangeInput] = useState("");
  const [classChangePosition, setClassChangePosition] = useState<{ x: number; y: number } | null>(null);

  // History (Undo/Redo)
  const [history, setHistory] = useState<HistoryState[]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const historyInitialized = useRef(false);

  // Clipboard
  const [clipboard, setClipboard] = useState<Annotation[]>([]);

  // Base scale (for fit-to-view)
  const [baseScale, setBaseScale] = useState(1);
  const [baseOffset, setBaseOffset] = useState({ x: 0, y: 0 });

  // Keyboard shortcuts panel
  const [showShortcuts, setShowShortcuts] = useState(false);

  // AI Predictions
  const [aiPredictions, setAiPredictions] = useState<AIPrediction[]>([]);

  // SAM Interactive Mode
  const [samLoading, setSamLoading] = useState(false);
  const [samPreview, setSamPreview] = useState<AIPrediction | null>(null);

  // ============================================
  // DATA FETCHING
  // ============================================

  const { data: dataset } = useQuery({
    queryKey: ["od-dataset", datasetId],
    queryFn: () => apiClient.getODDataset(datasetId),
  });

  const { data: image, isLoading: imageInfoLoading } = useQuery({
    queryKey: ["od-image", imageId],
    queryFn: () => apiClient.getODImage(imageId),
  });

  const { data: datasetImages } = useQuery({
    queryKey: ["od-dataset-images-nav", datasetId],
    queryFn: () => apiClient.getODDatasetImages(datasetId, { page: 1, limit: 1000 }),
  });

  const { data: classes } = useQuery({
    queryKey: ["od-classes", datasetId],
    queryFn: () => apiClient.getODDatasetClasses(datasetId),
  });

  const { data: existingAnnotations, isLoading: annotationsLoading } = useQuery({
    queryKey: ["od-annotations", datasetId, imageId],
    queryFn: () => apiClient.getODAnnotations(datasetId, imageId),
  });

  // Set annotations when loaded (only initialize history once per image)
  useEffect(() => {
    if (existingAnnotations) {
      setAnnotations(existingAnnotations);
      // Only initialize history on first load, not on every refetch
      if (!historyInitialized.current) {
        setHistory([{ annotations: existingAnnotations, description: "Initial state" }]);
        setHistoryIndex(0);
        historyInitialized.current = true;
      }
    }
  }, [existingAnnotations]);

  // Reset history initialization when image changes
  useEffect(() => {
    historyInitialized.current = false;
    setHistory([]);
    setHistoryIndex(-1);
  }, [imageId]);

  // ============================================
  // MUTATIONS
  // ============================================

  const createClassMutation = useMutation({
    mutationFn: async (data: { name: string; color: string }) => {
      return apiClient.createODDatasetClass(datasetId, data);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["od-classes", datasetId] });
    },
  });

  const createMutation = useMutation({
    mutationFn: async (data: { class_id: string; bbox: { x: number; y: number; width: number; height: number } }) => {
      return apiClient.createODAnnotation(datasetId, imageId, data);
    },
    onSuccess: (newAnnotation) => {
      queryClient.invalidateQueries({ queryKey: ["od-annotations", datasetId, imageId] });
      setLastUsedClassId(newAnnotation.class_id);
    },
    onError: (error) => {
      toast.error(`Failed to create annotation: ${error.message}`);
    },
  });

  const updateMutation = useMutation({
    mutationFn: async ({ id, data }: { id: string; data: { bbox?: Annotation["bbox"]; class_id?: string } }) => {
      return apiClient.updateODAnnotation(id, data);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["od-annotations", datasetId, imageId] });
    },
    onError: (error) => {
      toast.error(`Failed to update annotation: ${error.message}`);
    },
  });

  const deleteMutation = useMutation({
    mutationFn: async (annotationId: string) => {
      return apiClient.deleteODAnnotation(annotationId);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["od-annotations", datasetId, imageId] });
    },
    onError: (error) => {
      toast.error(`Failed to delete annotation: ${error.message}`);
    },
  });

  const markCompletedMutation = useMutation({
    mutationFn: async () => {
      return apiClient.updateODDatasetImageStatus(datasetId, imageId, "completed");
    },
    onSuccess: () => {
      toast.success("Marked as completed - advancing to next");
      goToNextImage();
    },
    onError: (error) => {
      toast.error(`Failed to update status: ${error.message}`);
    },
  });

  // ============================================
  // NAVIGATION
  // ============================================

  const currentImageIndex = datasetImages?.images.findIndex((img) => img.image_id === imageId) ?? -1;
  const totalImages = datasetImages?.images.length ?? 0;
  const prevImage = currentImageIndex > 0 ? datasetImages?.images[currentImageIndex - 1] : null;
  const nextImage = currentImageIndex < totalImages - 1 ? datasetImages?.images[currentImageIndex + 1] : null;

  const goToPrevImage = useCallback(() => {
    if (prevImage) {
      router.push(`/od/annotate/${datasetId}/${prevImage.image_id}`);
    }
  }, [prevImage, router, datasetId]);

  const goToNextImage = useCallback(() => {
    if (nextImage) {
      router.push(`/od/annotate/${datasetId}/${nextImage.image_id}`);
    }
  }, [nextImage, router, datasetId]);

  // ============================================
  // HISTORY (UNDO/REDO)
  // History stores STATES (not diffs). historyIndex points to current state.
  // Initial: history = [initialState], historyIndex = 0
  // After action: push NEW state, increment historyIndex
  // Undo: decrement historyIndex, apply that state
  // Redo: increment historyIndex, apply that state
  // ============================================

  const pushHistory = useCallback((newAnnotations: Annotation[], description: string) => {
    setHistory((prev) => {
      // Cut off any redo history when new action is performed
      const newHistory = prev.slice(0, historyIndex + 1);
      // Deep clone to prevent reference issues
      const clonedAnnotations = JSON.parse(JSON.stringify(newAnnotations));
      newHistory.push({ annotations: clonedAnnotations, description });
      // Limit to 50 entries
      if (newHistory.length > 50) {
        newHistory.shift();
        return newHistory;
      }
      return newHistory;
    });
    setHistoryIndex((prev) => Math.min(prev + 1, 49));
  }, [historyIndex]);

  const undo = useCallback(() => {
    if (historyIndex > 0 && history.length > 0) {
      const newIndex = historyIndex - 1;
      const currentState = history[historyIndex];
      const prevState = history[newIndex];

      if (!prevState || !currentState) return;

      setHistoryIndex(newIndex);
      // Deep clone to prevent mutation
      setAnnotations(JSON.parse(JSON.stringify(prevState.annotations)));
      setSelectedIds(new Set());
      toast.info(`Undo: ${currentState.description}`);
    }
  }, [history, historyIndex]);

  const redo = useCallback(() => {
    if (historyIndex < history.length - 1) {
      const newIndex = historyIndex + 1;
      const nextState = history[newIndex];

      if (!nextState) return;

      setHistoryIndex(newIndex);
      // Deep clone to prevent mutation
      setAnnotations(JSON.parse(JSON.stringify(nextState.annotations)));
      setSelectedIds(new Set());
      toast.info(`Redo: ${nextState.description}`);
    }
  }, [history, historyIndex]);

  // ============================================
  // AI PREDICTION HANDLERS
  // ============================================

  const handleAIPredictionsReceived = useCallback((predictions: AIPrediction[]) => {
    setAiPredictions(predictions);
  }, []);

  const handleAcceptAIPrediction = useCallback(async (prediction: AIPrediction) => {
    // Find or create class for the label
    const className = prediction.label.toLowerCase().trim();
    const existingClass = classes?.find(
      (c) => c.name.toLowerCase() === className || c.display_name?.toLowerCase() === className
    );

    let classId: string;

    if (existingClass) {
      classId = existingClass.id;
    } else {
      try {
        const newClass = await createClassMutation.mutateAsync({
          name: className,
          color: generateRandomColor(),
        });
        classId = newClass.id;
      } catch {
        toast.error("Failed to create class");
        return;
      }
    }

    // Create annotation from prediction
    try {
      const newAnnotation = await createMutation.mutateAsync({
        class_id: classId,
        bbox: prediction.bbox,
      });

      const updatedAnnotations = [...annotations, newAnnotation];
      setAnnotations(updatedAnnotations);
      pushHistory(updatedAnnotations, "Accept AI prediction");

      // Remove from predictions
      setAiPredictions((prev) => prev.filter((p) => p !== prediction));
      toast.success(`Added: ${prediction.label}`);
    } catch {
      toast.error("Failed to create annotation");
    }
  }, [classes, createClassMutation, createMutation, annotations, pushHistory]);

  const handleRejectAIPrediction = useCallback((index: number) => {
    setAiPredictions((prev) => prev.filter((_, i) => i !== index));
  }, []);

  const handleAcceptAllAIPredictions = useCallback(async () => {
    for (const prediction of aiPredictions) {
      await handleAcceptAIPrediction(prediction);
    }
  }, [aiPredictions, handleAcceptAIPrediction]);

  const handleRejectAllAIPredictions = useCallback(() => {
    setAiPredictions([]);
  }, []);

  // ============================================
  // SAM INTERACTIVE MODE
  // ============================================

  const handleSAMClick = useCallback(async (normalizedX: number, normalizedY: number) => {
    if (samLoading) return;

    setSamLoading(true);
    setSamPreview(null);

    try {
      const result = await apiClient.segmentODAI({
        image_id: imageId,
        model: "sam2",
        prompt_type: "point",
        point: [normalizedX, normalizedY],
        label: 1, // foreground
      });

      // Create preview from result
      const preview: AIPrediction = {
        bbox: result.bbox,
        label: "SAM segment",
        confidence: result.confidence,
        mask: result.mask,
      };

      setSamPreview(preview);
      toast.success("Segment found! Click Accept or press Enter to add.");
    } catch (error) {
      toast.error("SAM segmentation failed");
      console.error("SAM error:", error);
    } finally {
      setSamLoading(false);
    }
  }, [imageId, samLoading]);

  const handleAcceptSAMPreview = useCallback(async () => {
    if (!samPreview) return;

    // Use last used class or prompt for class
    if (lastUsedClassId) {
      try {
        const newAnnotation = await createMutation.mutateAsync({
          class_id: lastUsedClassId,
          bbox: samPreview.bbox,
        });

        const updatedAnnotations = [...annotations, newAnnotation];
        setAnnotations(updatedAnnotations);
        pushHistory(updatedAnnotations, "Accept SAM segment");
        toast.success("Annotation added!");
      } catch {
        toast.error("Failed to create annotation");
      }
    } else {
      // Show class input for new bbox
      setPendingBbox(samPreview.bbox);
      setShowClassInput(true);
      setClassInput("");
      setTimeout(() => classInputRef.current?.focus(), 50);
    }

    setSamPreview(null);
  }, [samPreview, lastUsedClassId, createMutation, annotations, pushHistory]);

  const handleRejectSAMPreview = useCallback(() => {
    setSamPreview(null);
  }, []);

  // ============================================
  // CLIPBOARD (COPY/PASTE)
  // ============================================

  const copySelected = useCallback(() => {
    if (selectedIds.size === 0) return;
    const selected = annotations.filter((a) => selectedIds.has(a.id));
    setClipboard(selected);
    toast.success(`Copied ${selected.length} annotation(s)`);
  }, [annotations, selectedIds]);

  const paste = useCallback(async () => {
    if (clipboard.length === 0) return;

    const newAnnotations: Annotation[] = [];
    for (const ann of clipboard) {
      const newAnnotation = await createMutation.mutateAsync({
        class_id: ann.class_id,
        bbox: {
          x: Math.min(ann.bbox.x + 0.02, 1 - ann.bbox.width),
          y: Math.min(ann.bbox.y + 0.02, 1 - ann.bbox.height),
          width: ann.bbox.width,
          height: ann.bbox.height,
        },
      });
      newAnnotations.push(newAnnotation);
    }

    // Add to local state and history
    const updatedAnnotations = [...annotations, ...newAnnotations];
    setAnnotations(updatedAnnotations);
    pushHistory(updatedAnnotations, `Paste ${newAnnotations.length} annotation(s)`);

    toast.success(`Pasted ${newAnnotations.length} annotation(s)`);
  }, [clipboard, createMutation, annotations, pushHistory]);

  // ============================================
  // IMAGE LOAD & ZOOM
  // ============================================

  const handleImageLoad = useCallback(() => {
    const img = imageRef.current;
    const container = containerRef.current;
    if (!img || !container) return;

    const containerWidth = container.clientWidth;
    const containerHeight = container.clientHeight;
    const imgWidth = img.naturalWidth;
    const imgHeight = img.naturalHeight;

    const scaleX = containerWidth / imgWidth;
    const scaleY = containerHeight / imgHeight;
    const newScale = Math.min(scaleX, scaleY, 1);

    const scaledWidth = imgWidth * newScale;
    const scaledHeight = imgHeight * newScale;
    const offsetX = (containerWidth - scaledWidth) / 2;
    const offsetY = (containerHeight - scaledHeight) / 2;

    setBaseScale(newScale);
    setBaseOffset({ x: offsetX, y: offsetY });
    setScale(newScale);
    setOffset({ x: offsetX, y: offsetY });
    setImageLoaded(true);
  }, []);

  const zoomIn = useCallback(() => {
    setScale((prev) => Math.min(prev * 1.25, 5));
  }, []);

  const zoomOut = useCallback(() => {
    setScale((prev) => Math.max(prev / 1.25, 0.1));
  }, []);

  const resetZoom = useCallback(() => {
    setScale(baseScale);
    setOffset(baseOffset);
  }, [baseScale, baseOffset]);

  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const container = containerRef.current;
    if (!container) return;

    const rect = container.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
    const newScale = Math.max(0.1, Math.min(5, scale * zoomFactor));

    const newOffsetX = mouseX - (mouseX - offset.x) * (newScale / scale);
    const newOffsetY = mouseY - (mouseY - offset.y) * (newScale / scale);

    setScale(newScale);
    setOffset({ x: newOffsetX, y: newOffsetY });
  }, [scale, offset]);

  // ============================================
  // COORDINATE HELPERS
  // ============================================

  const getMousePos = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    const img = imageRef.current;
    if (!canvas || !img) return null;

    const rect = canvas.getBoundingClientRect();
    const canvasX = e.clientX - rect.left;
    const canvasY = e.clientY - rect.top;

    const imgX = (canvasX - offset.x) / scale;
    const imgY = (canvasY - offset.y) / scale;

    if (imgX < 0 || imgX > img.naturalWidth || imgY < 0 || imgY > img.naturalHeight) {
      return { canvasX, canvasY, imgX, imgY, normalized: null };
    }

    return {
      canvasX,
      canvasY,
      imgX,
      imgY,
      normalized: { x: imgX / img.naturalWidth, y: imgY / img.naturalHeight },
    };
  }, [offset, scale]);

  // ============================================
  // HIT TESTING
  // ============================================

  const getAnnotationAtPoint = useCallback((imgX: number, imgY: number): Annotation | null => {
    const img = imageRef.current;
    if (!img) return null;

    for (let i = annotations.length - 1; i >= 0; i--) {
      const ann = annotations[i];
      const x = ann.bbox.x * img.naturalWidth;
      const y = ann.bbox.y * img.naturalHeight;
      const width = ann.bbox.width * img.naturalWidth;
      const height = ann.bbox.height * img.naturalHeight;

      if (imgX >= x && imgX <= x + width && imgY >= y && imgY <= y + height) {
        return ann;
      }
    }
    return null;
  }, [annotations]);

  const getResizeHandleAtPoint = useCallback((imgX: number, imgY: number, ann: Annotation): ResizeHandle => {
    const img = imageRef.current;
    if (!img) return null;

    const x = ann.bbox.x * img.naturalWidth;
    const y = ann.bbox.y * img.naturalHeight;
    const width = ann.bbox.width * img.naturalWidth;
    const height = ann.bbox.height * img.naturalHeight;

    const handleSize = 10 / scale;

    if (Math.abs(imgX - x) < handleSize && Math.abs(imgY - y) < handleSize) return "nw";
    if (Math.abs(imgX - (x + width)) < handleSize && Math.abs(imgY - y) < handleSize) return "ne";
    if (Math.abs(imgX - x) < handleSize && Math.abs(imgY - (y + height)) < handleSize) return "sw";
    if (Math.abs(imgX - (x + width)) < handleSize && Math.abs(imgY - (y + height)) < handleSize) return "se";

    if (Math.abs(imgX - x) < handleSize && imgY > y && imgY < y + height) return "w";
    if (Math.abs(imgX - (x + width)) < handleSize && imgY > y && imgY < y + height) return "e";
    if (Math.abs(imgY - y) < handleSize && imgX > x && imgX < x + width) return "n";
    if (Math.abs(imgY - (y + height)) < handleSize && imgX > x && imgX < x + width) return "s";

    return null;
  }, [scale]);

  // ============================================
  // MOUSE HANDLERS
  // ============================================

  const handleMouseDown = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (showClassInput || showClassChange) return;

    const pos = getMousePos(e);
    if (!pos) return;

    // Right-click or middle-click or space+click = pan
    if (spacePressed || e.button === 1 || e.button === 2) {
      e.preventDefault();
      setIsPanning(true);
      setPanStart({ x: e.clientX, y: e.clientY, offsetX: offset.x, offsetY: offset.y });
      return;
    }

    if (toolMode === "select") {
      if (selectedIds.size === 1) {
        const selectedAnn = annotations.find((a) => selectedIds.has(a.id));
        if (selectedAnn) {
          const handle = getResizeHandleAtPoint(pos.imgX, pos.imgY, selectedAnn);
          if (handle) {
            setIsResizing(true);
            setResizeHandle(handle);
            setDragStart({ x: pos.imgX, y: pos.imgY });
            setOriginalBboxes(new Map([[selectedAnn.id, { ...selectedAnn.bbox }]]));
            return;
          }
        }
      }

      const clickedAnn = getAnnotationAtPoint(pos.imgX, pos.imgY);
      if (clickedAnn) {
        if (e.shiftKey) {
          setSelectedIds((prev) => {
            const next = new Set(prev);
            if (next.has(clickedAnn.id)) {
              next.delete(clickedAnn.id);
            } else {
              next.add(clickedAnn.id);
            }
            return next;
          });
        } else if (!selectedIds.has(clickedAnn.id)) {
          setSelectedIds(new Set([clickedAnn.id]));
        }

        setIsDragging(true);
        setDragStart({ x: pos.imgX, y: pos.imgY });
        const selected = annotations.filter((a) => selectedIds.has(a.id) || a.id === clickedAnn.id);
        const bboxMap = new Map<string, Annotation["bbox"]>();
        selected.forEach((a) => bboxMap.set(a.id, { ...a.bbox }));
        setOriginalBboxes(bboxMap);
      } else {
        setSelectedIds(new Set());
      }
    } else if (toolMode === "bbox") {
      const clickedAnn = getAnnotationAtPoint(pos.imgX, pos.imgY);
      if (clickedAnn) {
        if (e.shiftKey) {
          setSelectedIds((prev) => {
            const next = new Set(prev);
            if (next.has(clickedAnn.id)) {
              next.delete(clickedAnn.id);
            } else {
              next.add(clickedAnn.id);
            }
            return next;
          });
        } else {
          setSelectedIds(new Set([clickedAnn.id]));
        }
        return;
      }

      setIsDrawing(true);
      setSelectedIds(new Set());
      setDrawingBox({
        startX: pos.canvasX,
        startY: pos.canvasY,
        endX: pos.canvasX,
        endY: pos.canvasY,
      });
    } else if (toolMode === "sam") {
      // SAM click-to-segment mode
      if (samLoading) return;

      // Clear any existing selection
      setSelectedIds(new Set());

      // Call SAM with normalized coordinates
      handleSAMClick(pos.imgX, pos.imgY);
    }
  }, [showClassInput, showClassChange, getMousePos, spacePressed, offset, toolMode, selectedIds, annotations, getAnnotationAtPoint, getResizeHandleAtPoint, samLoading, handleSAMClick]);

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const pos = getMousePos(e);

    // Track mouse position for crosshair
    if (pos) {
      setMousePos({ canvasX: pos.canvasX, canvasY: pos.canvasY });
    }

    if (isPanning && panStart) {
      const dx = e.clientX - panStart.x;
      const dy = e.clientY - panStart.y;
      setOffset({ x: panStart.offsetX + dx, y: panStart.offsetY + dy });
      return;
    }

    if (isDragging && dragStart && pos) {
      const img = imageRef.current;
      if (!img) return;

      const dx = (pos.imgX - dragStart.x) / img.naturalWidth;
      const dy = (pos.imgY - dragStart.y) / img.naturalHeight;

      setAnnotations((prev) =>
        prev.map((ann) => {
          const original = originalBboxes.get(ann.id);
          if (!original) return ann;

          return {
            ...ann,
            bbox: {
              ...ann.bbox,
              x: Math.max(0, Math.min(1 - ann.bbox.width, original.x + dx)),
              y: Math.max(0, Math.min(1 - ann.bbox.height, original.y + dy)),
            },
          };
        })
      );
      return;
    }

    if (isResizing && resizeHandle && dragStart && pos) {
      const img = imageRef.current;
      if (!img) return;

      const selectedAnn = annotations.find((a) => selectedIds.has(a.id));
      if (!selectedAnn) return;

      const original = originalBboxes.get(selectedAnn.id);
      if (!original) return;

      const dx = (pos.imgX - dragStart.x) / img.naturalWidth;
      const dy = (pos.imgY - dragStart.y) / img.naturalHeight;

      let newBbox = { ...original };

      switch (resizeHandle) {
        case "nw":
          newBbox.x = Math.max(0, original.x + dx);
          newBbox.y = Math.max(0, original.y + dy);
          newBbox.width = Math.max(0.01, original.width - dx);
          newBbox.height = Math.max(0.01, original.height - dy);
          break;
        case "n":
          newBbox.y = Math.max(0, original.y + dy);
          newBbox.height = Math.max(0.01, original.height - dy);
          break;
        case "ne":
          newBbox.y = Math.max(0, original.y + dy);
          newBbox.width = Math.max(0.01, original.width + dx);
          newBbox.height = Math.max(0.01, original.height - dy);
          break;
        case "e":
          newBbox.width = Math.max(0.01, original.width + dx);
          break;
        case "se":
          newBbox.width = Math.max(0.01, original.width + dx);
          newBbox.height = Math.max(0.01, original.height + dy);
          break;
        case "s":
          newBbox.height = Math.max(0.01, original.height + dy);
          break;
        case "sw":
          newBbox.x = Math.max(0, original.x + dx);
          newBbox.width = Math.max(0.01, original.width - dx);
          newBbox.height = Math.max(0.01, original.height + dy);
          break;
        case "w":
          newBbox.x = Math.max(0, original.x + dx);
          newBbox.width = Math.max(0.01, original.width - dx);
          break;
      }

      newBbox.x = Math.max(0, Math.min(1 - 0.01, newBbox.x));
      newBbox.y = Math.max(0, Math.min(1 - 0.01, newBbox.y));
      newBbox.width = Math.min(1 - newBbox.x, newBbox.width);
      newBbox.height = Math.min(1 - newBbox.y, newBbox.height);

      setAnnotations((prev) =>
        prev.map((ann) => (ann.id === selectedAnn.id ? { ...ann, bbox: newBbox } : ann))
      );
      return;
    }

    if (isDrawing && drawingBox && pos) {
      setDrawingBox({
        ...drawingBox,
        endX: pos.canvasX,
        endY: pos.canvasY,
      });
      return;
    }

    if (pos && !isDragging && !isResizing && !isDrawing) {
      const hovered = getAnnotationAtPoint(pos.imgX, pos.imgY);
      setHoveredId(hovered?.id ?? null);

      const canvas = canvasRef.current;
      if (canvas) {
        if (toolMode === "select" && selectedIds.size === 1) {
          const selectedAnn = annotations.find((a) => selectedIds.has(a.id));
          if (selectedAnn) {
            const handle = getResizeHandleAtPoint(pos.imgX, pos.imgY, selectedAnn);
            if (handle) {
              const cursors: Record<NonNullable<ResizeHandle>, string> = {
                nw: "nwse-resize",
                ne: "nesw-resize",
                sw: "nesw-resize",
                se: "nwse-resize",
                n: "ns-resize",
                s: "ns-resize",
                e: "ew-resize",
                w: "ew-resize",
              };
              canvas.style.cursor = cursors[handle] || "default";
              return;
            }
          }
        }
        if (hovered && toolMode === "select") {
          canvas.style.cursor = "move";
        } else if (spacePressed) {
          canvas.style.cursor = "grab";
        } else {
          canvas.style.cursor = toolMode === "bbox" ? "crosshair" : toolMode === "sam" ? "crosshair" : "default";
        }
      }
    }
  }, [getMousePos, isPanning, panStart, isDragging, dragStart, originalBboxes, isResizing, resizeHandle, isDrawing, drawingBox, annotations, selectedIds, toolMode, getAnnotationAtPoint, getResizeHandleAtPoint, spacePressed]);

  const handleMouseUp = useCallback(async () => {
    if (isPanning) {
      setIsPanning(false);
      setPanStart(null);
      return;
    }

    if (isDragging && originalBboxes.size > 0) {
      const movedAnnotations = annotations.filter((a) => {
        const original = originalBboxes.get(a.id);
        if (!original) return false;
        return original.x !== a.bbox.x || original.y !== a.bbox.y;
      });

      if (movedAnnotations.length > 0) {
        // Push current state (after move) to history
        pushHistory(annotations, `Move ${movedAnnotations.length} annotation(s)`);
        for (const ann of movedAnnotations) {
          if (!ann.id.startsWith("temp-")) {
            await updateMutation.mutateAsync({ id: ann.id, data: { bbox: ann.bbox } });
          }
        }
      }

      setIsDragging(false);
      setDragStart(null);
      setOriginalBboxes(new Map());
      return;
    }

    if (isResizing && originalBboxes.size > 0) {
      const resizedAnnotations = annotations.filter((a) => {
        const original = originalBboxes.get(a.id);
        if (!original) return false;
        return (
          original.x !== a.bbox.x ||
          original.y !== a.bbox.y ||
          original.width !== a.bbox.width ||
          original.height !== a.bbox.height
        );
      });

      if (resizedAnnotations.length > 0) {
        // Push current state (after resize) to history
        pushHistory(annotations, "Resize annotation");
        for (const ann of resizedAnnotations) {
          if (!ann.id.startsWith("temp-")) {
            await updateMutation.mutateAsync({ id: ann.id, data: { bbox: ann.bbox } });
          }
        }
      }

      setIsResizing(false);
      setResizeHandle(null);
      setDragStart(null);
      setOriginalBboxes(new Map());
      return;
    }

    if (!isDrawing || !drawingBox) {
      setIsDrawing(false);
      setDrawingBox(null);
      return;
    }

    const img = imageRef.current;
    if (!img) {
      setIsDrawing(false);
      setDrawingBox(null);
      return;
    }

    const imgWidth = img.naturalWidth;
    const imgHeight = img.naturalHeight;

    const startImgX = (Math.min(drawingBox.startX, drawingBox.endX) - offset.x) / scale;
    const startImgY = (Math.min(drawingBox.startY, drawingBox.endY) - offset.y) / scale;
    const boxWidth = Math.abs(drawingBox.endX - drawingBox.startX) / scale;
    const boxHeight = Math.abs(drawingBox.endY - drawingBox.startY) / scale;

    const x = startImgX / imgWidth;
    const y = startImgY / imgHeight;
    const width = boxWidth / imgWidth;
    const height = boxHeight / imgHeight;

    if (isNaN(x) || isNaN(y) || isNaN(width) || isNaN(height)) {
      setIsDrawing(false);
      setDrawingBox(null);
      return;
    }

    if (width > 0.01 && height > 0.01) {
      if (lastUsedClassId) {
        const newAnnotation = await createMutation.mutateAsync({
          class_id: lastUsedClassId,
          bbox: { x, y, width, height },
        });
        // Add to local state and history
        const updatedAnnotations = [...annotations, newAnnotation];
        setAnnotations(updatedAnnotations);
        pushHistory(updatedAnnotations, "Create annotation");
        // Continuous draw mode - stay in bbox mode
      } else {
        setPendingBbox({ x, y, width, height });
        setShowClassInput(true);
        setClassInput("");
        setTimeout(() => classInputRef.current?.focus(), 50);
      }
    }

    setIsDrawing(false);
    setDrawingBox(null);
  }, [isPanning, isDragging, isResizing, isDrawing, drawingBox, originalBboxes, annotations, pushHistory, updateMutation, offset, scale, lastUsedClassId, createMutation]);

  // Prevent context menu on right-click
  const handleContextMenu = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
  }, []);

  // ============================================
  // DOUBLE CLICK FOR INLINE CLASS CHANGE
  // ============================================

  const handleDoubleClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const pos = getMousePos(e);
    if (!pos) return;

    const clickedAnn = getAnnotationAtPoint(pos.imgX, pos.imgY);
    if (clickedAnn) {
      setSelectedIds(new Set([clickedAnn.id]));

      // Position the class change popup at the bbox
      const img = imageRef.current;
      if (img) {
        const bboxX = offset.x + clickedAnn.bbox.x * img.naturalWidth * scale;
        const bboxY = offset.y + (clickedAnn.bbox.y + clickedAnn.bbox.height) * img.naturalHeight * scale + 8;
        setClassChangePosition({ x: bboxX, y: bboxY });
      }

      setShowClassChange(true);
      setClassChangeInput("");
      setTimeout(() => document.getElementById("class-change-input")?.focus(), 50);
    }
  }, [getMousePos, getAnnotationAtPoint, offset, scale]);

  // ============================================
  // CLASS INPUT HANDLERS
  // ============================================

  const handleClassSubmit = async () => {
    if (!pendingBbox || !classInput.trim()) return;

    const className = classInput.trim().toLowerCase();
    let existingClass = classes?.find(
      (c) => c.name.toLowerCase() === className || c.display_name?.toLowerCase() === className
    );

    let classId: string;

    if (existingClass) {
      classId = existingClass.id;
    } else {
      try {
        const newClass = await createClassMutation.mutateAsync({
          name: className,
          color: generateRandomColor(),
        });
        classId = newClass.id;
      } catch {
        toast.error("Failed to create class");
        return;
      }
    }

    const newAnnotation = await createMutation.mutateAsync({
      class_id: classId,
      bbox: pendingBbox,
    });

    // Add to local state and history
    const updatedAnnotations = [...annotations, newAnnotation];
    setAnnotations(updatedAnnotations);
    pushHistory(updatedAnnotations, "Create annotation");

    setPendingBbox(null);
    setShowClassInput(false);
    setClassInput("");
  };

  const handleClassInputKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      e.preventDefault();
      handleClassSubmit();
    } else if (e.key === "Escape") {
      setPendingBbox(null);
      setShowClassInput(false);
      setClassInput("");
    }
  };

  // ============================================
  // QUICK CLASS CHANGE
  // ============================================

  const handleQuickClassChange = async (classIdToUse?: string) => {
    if (selectedIds.size !== 1) return;

    let classId = classIdToUse;

    if (!classId) {
      if (!classChangeInput.trim()) return;

      const className = classChangeInput.trim().toLowerCase();
      const existingClass = classes?.find(
        (c) => c.name.toLowerCase() === className || c.display_name?.toLowerCase() === className
      );

      if (existingClass) {
        classId = existingClass.id;
      } else {
        try {
          const newClass = await createClassMutation.mutateAsync({
            name: className,
            color: generateRandomColor(),
          });
          classId = newClass.id;
        } catch {
          toast.error("Failed to create class");
          return;
        }
      }
    }

    const selectedAnn = annotations.find((a) => selectedIds.has(a.id));
    if (selectedAnn && !selectedAnn.id.startsWith("temp-") && classId) {
      await updateMutation.mutateAsync({ id: selectedAnn.id, data: { class_id: classId } });
      // Update local state with new class info
      const selectedClass = classes?.find((c) => c.id === classId);
      const updatedAnnotations = annotations.map((a) =>
        a.id === selectedAnn.id
          ? {
              ...a,
              class_id: classId,
              class_name: selectedClass?.name || a.class_name,
              class_color: selectedClass?.color || a.class_color,
            }
          : a
      );
      setAnnotations(updatedAnnotations);
      pushHistory(updatedAnnotations, "Change class");
    }

    setShowClassChange(false);
    setClassChangeInput("");
    setClassChangePosition(null);
  };

  // ============================================
  // DELETE SELECTED
  // ============================================

  const deleteSelected = useCallback(async () => {
    if (selectedIds.size === 0) return;

    const toDelete = annotations.filter((a) => selectedIds.has(a.id));
    const remaining = annotations.filter((a) => !selectedIds.has(a.id));

    for (const ann of toDelete) {
      if (!ann.id.startsWith("temp-")) {
        await deleteMutation.mutateAsync(ann.id);
      }
    }

    setAnnotations(remaining);
    // Push state AFTER delete to history
    pushHistory(remaining, `Delete ${toDelete.length} annotation(s)`);
    setSelectedIds(new Set());
    toast.success(`Deleted ${toDelete.length} annotation(s)`);
  }, [annotations, selectedIds, pushHistory, deleteMutation]);

  // ============================================
  // SELECT ALL
  // ============================================

  const selectAll = useCallback(() => {
    setSelectedIds(new Set(annotations.map((a) => a.id)));
  }, [annotations]);

  // ============================================
  // TAB NAVIGATION
  // ============================================

  const selectNextAnnotation = useCallback(() => {
    if (annotations.length === 0) return;

    if (selectedIds.size === 0) {
      setSelectedIds(new Set([annotations[0].id]));
      return;
    }

    const currentId = Array.from(selectedIds)[0];
    const currentIndex = annotations.findIndex((a) => a.id === currentId);
    const nextIndex = (currentIndex + 1) % annotations.length;
    setSelectedIds(new Set([annotations[nextIndex].id]));
  }, [annotations, selectedIds]);

  const selectPrevAnnotation = useCallback(() => {
    if (annotations.length === 0) return;

    if (selectedIds.size === 0) {
      setSelectedIds(new Set([annotations[annotations.length - 1].id]));
      return;
    }

    const currentId = Array.from(selectedIds)[0];
    const currentIndex = annotations.findIndex((a) => a.id === currentId);
    const prevIndex = currentIndex === 0 ? annotations.length - 1 : currentIndex - 1;
    setSelectedIds(new Set([annotations[prevIndex].id]));
  }, [annotations, selectedIds]);

  // ============================================
  // KEYBOARD SHORTCUTS
  // ============================================

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (showClassInput || showClassChange) {
        if (e.key === "Escape") {
          setShowClassInput(false);
          setShowClassChange(false);
          setPendingBbox(null);
          setClassChangePosition(null);
        }
        return;
      }

      if (e.key === " " && !e.repeat) {
        e.preventDefault();
        setSpacePressed(true);
        return;
      }

      if (e.key === "v" || e.key === "V") {
        setToolMode("select");
        return;
      }
      if (e.key === "b" || e.key === "B") {
        setToolMode("bbox");
        return;
      }
      if (e.key === "s" || e.key === "S") {
        setToolMode("sam");
        return;
      }

      // Tab navigation
      if (e.key === "Tab") {
        e.preventDefault();
        if (e.shiftKey) {
          selectPrevAnnotation();
        } else {
          selectNextAnnotation();
        }
        return;
      }

      if (e.key === "ArrowRight" && !e.shiftKey) {
        e.preventDefault();
        goToNextImage();
        return;
      }
      if (e.key === "ArrowLeft" && !e.shiftKey) {
        e.preventDefault();
        goToPrevImage();
        return;
      }

      if ((e.key === "Delete" || e.key === "Backspace" || e.key === "d" || e.key === "D") && selectedIds.size > 0) {
        e.preventDefault();
        deleteSelected();
        return;
      }

      // Ctrl+S - Save (mark as completed and advance)
      if ((e.ctrlKey || e.metaKey) && e.key === "s") {
        e.preventDefault();
        markCompletedMutation.mutate();
        return;
      }

      if ((e.ctrlKey || e.metaKey) && e.key === "z" && !e.shiftKey) {
        e.preventDefault();
        undo();
        return;
      }
      if ((e.ctrlKey || e.metaKey) && (e.key === "y" || (e.key === "z" && e.shiftKey))) {
        e.preventDefault();
        redo();
        return;
      }

      if ((e.ctrlKey || e.metaKey) && e.key === "c") {
        e.preventDefault();
        copySelected();
        return;
      }
      if ((e.ctrlKey || e.metaKey) && e.key === "v") {
        e.preventDefault();
        paste();
        return;
      }

      if ((e.ctrlKey || e.metaKey) && e.key === "a") {
        e.preventDefault();
        selectAll();
        return;
      }

      // Enter key - accept SAM preview
      if (e.key === "Enter" && samPreview && !showClassInput && !showClassChange) {
        e.preventDefault();
        handleAcceptSAMPreview();
        return;
      }

      if (e.key === "Escape") {
        // Reject SAM preview first
        if (samPreview) {
          handleRejectSAMPreview();
          return;
        }
        setSelectedIds(new Set());
        setIsDrawing(false);
        setDrawingBox(null);
        setPendingBbox(null);
        setShowClassInput(false);
        setShowClassChange(false);
        setClassChangePosition(null);
        setShowShortcuts(false);
        return;
      }

      if (e.key === "+" || e.key === "=") {
        e.preventDefault();
        zoomIn();
        return;
      }
      if (e.key === "-" || e.key === "_") {
        e.preventDefault();
        zoomOut();
        return;
      }
      if (e.key === "0") {
        e.preventDefault();
        resetZoom();
        return;
      }

      // Numbers 1-9: If annotation selected, change its class. Otherwise select class for drawing.
      if (/^[1-9]$/.test(e.key) && classes && classes.length > 0) {
        const index = parseInt(e.key) - 1;
        if (index < classes.length) {
          const selectedClass = classes[index];

          if (selectedIds.size === 1) {
            // Change selected annotation's class
            handleQuickClassChange(selectedClass.id);
            toast.success(`Changed class to: ${selectedClass.display_name || selectedClass.name}`);
          } else {
            // Select class for drawing
            setLastUsedClassId(selectedClass.id);
            toast.info(`Selected class: ${selectedClass.display_name || selectedClass.name}`);
          }
        }
        return;
      }

      if (e.key === "c" && selectedIds.size === 1 && !(e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        const selectedAnn = annotations.find((a) => selectedIds.has(a.id));
        if (selectedAnn) {
          const img = imageRef.current;
          if (img) {
            const bboxX = offset.x + selectedAnn.bbox.x * img.naturalWidth * scale;
            const bboxY = offset.y + (selectedAnn.bbox.y + selectedAnn.bbox.height) * img.naturalHeight * scale + 8;
            setClassChangePosition({ x: bboxX, y: bboxY });
          }
        }
        setShowClassChange(true);
        setClassChangeInput("");
        setTimeout(() => document.getElementById("class-change-input")?.focus(), 50);
        return;
      }

      // ? - Toggle keyboard shortcuts panel
      if (e.key === "?" || (e.shiftKey && e.key === "/")) {
        e.preventDefault();
        setShowShortcuts((prev) => !prev);
        return;
      }
    };

    const handleKeyUp = (e: KeyboardEvent) => {
      if (e.key === " ") {
        setSpacePressed(false);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
    };
  }, [
    showClassInput,
    showClassChange,
    selectedIds,
    deleteSelected,
    undo,
    redo,
    copySelected,
    paste,
    selectAll,
    zoomIn,
    zoomOut,
    resetZoom,
    classes,
    annotations,
    offset,
    scale,
    goToNextImage,
    goToPrevImage,
    selectNextAnnotation,
    selectPrevAnnotation,
    handleQuickClassChange,
    markCompletedMutation,
    samPreview,
    handleAcceptSAMPreview,
    handleRejectSAMPreview,
  ]);

  // ============================================
  // DRAWING
  // ============================================

  const drawAnnotations = useCallback(() => {
    const canvas = canvasRef.current;
    const img = imageRef.current;
    if (!canvas || !img || !imageLoaded) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const container = containerRef.current;
    if (!container) return;

    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const imgWidth = img.naturalWidth;
    const imgHeight = img.naturalHeight;

    // Draw crosshair when in bbox mode and drawing or hovering
    if (toolMode === "bbox" && mousePos && !isPanning) {
      ctx.strokeStyle = "#ffffff80";
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);

      // Vertical line
      ctx.beginPath();
      ctx.moveTo(mousePos.canvasX, 0);
      ctx.lineTo(mousePos.canvasX, canvas.height);
      ctx.stroke();

      // Horizontal line
      ctx.beginPath();
      ctx.moveTo(0, mousePos.canvasY);
      ctx.lineTo(canvas.width, mousePos.canvasY);
      ctx.stroke();

      ctx.setLineDash([]);
    }

    if (!showAnnotations) return;

    // Draw existing annotations
    annotations.forEach((ann) => {
      const isSelected = selectedIds.has(ann.id);
      const isHovered = ann.id === hoveredId;

      const x = offset.x + ann.bbox.x * imgWidth * scale;
      const y = offset.y + ann.bbox.y * imgHeight * scale;
      const width = ann.bbox.width * imgWidth * scale;
      const height = ann.bbox.height * imgHeight * scale;

      ctx.fillStyle = ann.class_color + (isHovered ? "40" : "20");
      ctx.fillRect(x, y, width, height);

      ctx.strokeStyle = ann.class_color;
      ctx.lineWidth = isSelected ? 3 : isHovered ? 2.5 : 2;
      ctx.setLineDash([]);
      ctx.strokeRect(x, y, width, height);

      const label = ann.class_name;
      ctx.font = "bold 11px system-ui, sans-serif";
      const labelWidth = ctx.measureText(label).width + 8;
      const labelHeight = 18;

      ctx.fillStyle = ann.class_color;
      ctx.fillRect(x, y - labelHeight, labelWidth, labelHeight);

      ctx.fillStyle = "white";
      ctx.fillText(label, x + 4, y - 5);

      if (isSelected) {
        const handleSize = 8;
        ctx.fillStyle = "white";
        ctx.strokeStyle = ann.class_color;
        ctx.lineWidth = 2;

        const handles = [
          { px: x, py: y },
          { px: x + width / 2, py: y },
          { px: x + width, py: y },
          { px: x + width, py: y + height / 2 },
          { px: x + width, py: y + height },
          { px: x + width / 2, py: y + height },
          { px: x, py: y + height },
          { px: x, py: y + height / 2 },
        ];

        handles.forEach(({ px, py }) => {
          ctx.fillRect(px - handleSize / 2, py - handleSize / 2, handleSize, handleSize);
          ctx.strokeRect(px - handleSize / 2, py - handleSize / 2, handleSize, handleSize);
        });
      }
    });

    // Draw current drawing box with size display
    if (drawingBox) {
      const color = lastUsedClassId
        ? classes?.find((c) => c.id === lastUsedClassId)?.color || "#3B82F6"
        : "#3B82F6";
      const x = Math.min(drawingBox.startX, drawingBox.endX);
      const y = Math.min(drawingBox.startY, drawingBox.endY);
      const width = Math.abs(drawingBox.endX - drawingBox.startX);
      const height = Math.abs(drawingBox.endY - drawingBox.startY);

      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      ctx.strokeRect(x, y, width, height);
      ctx.setLineDash([]);

      ctx.fillStyle = color + "20";
      ctx.fillRect(x, y, width, height);

      // Draw size label
      const sizeWidth = Math.round(width / scale);
      const sizeHeight = Math.round(height / scale);
      const sizeText = `${sizeWidth} × ${sizeHeight}`;

      ctx.font = "bold 12px system-ui, sans-serif";
      const textWidth = ctx.measureText(sizeText).width + 8;
      const textHeight = 20;

      const labelX = x + width / 2 - textWidth / 2;
      const labelY = y + height / 2 - textHeight / 2;

      ctx.fillStyle = "rgba(0,0,0,0.7)";
      ctx.fillRect(labelX, labelY, textWidth, textHeight);

      ctx.fillStyle = "white";
      ctx.fillText(sizeText, labelX + 4, labelY + 14);
    }

    // Draw pending bbox
    if (pendingBbox && img) {
      const color = "#10B981";
      const x = offset.x + pendingBbox.x * imgWidth * scale;
      const y = offset.y + pendingBbox.y * imgHeight * scale;
      const width = pendingBbox.width * imgWidth * scale;
      const height = pendingBbox.height * imgHeight * scale;

      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.strokeRect(x, y, width, height);

      ctx.fillStyle = color + "30";
      ctx.fillRect(x, y, width, height);
    }

    // Draw AI predictions (dashed border, purple color)
    aiPredictions.forEach((pred) => {
      const color = "#A855F7"; // Purple for AI predictions
      const x = offset.x + pred.bbox.x * imgWidth * scale;
      const y = offset.y + pred.bbox.y * imgHeight * scale;
      const width = pred.bbox.width * imgWidth * scale;
      const height = pred.bbox.height * imgHeight * scale;

      // Dashed border
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.strokeRect(x, y, width, height);
      ctx.setLineDash([]);

      // Semi-transparent fill
      ctx.fillStyle = color + "15";
      ctx.fillRect(x, y, width, height);

      // Label with confidence
      const label = `${pred.label} ${(pred.confidence * 100).toFixed(0)}%`;
      ctx.font = "bold 11px system-ui, sans-serif";
      const labelWidth = ctx.measureText(label).width + 8;
      const labelHeight = 18;

      ctx.fillStyle = color;
      ctx.fillRect(x, y - labelHeight, labelWidth, labelHeight);

      ctx.fillStyle = "white";
      ctx.fillText(label, x + 4, y - 5);

      // AI badge
      const badgeText = "AI";
      const badgeWidth = ctx.measureText(badgeText).width + 6;
      ctx.fillStyle = color;
      ctx.fillRect(x + labelWidth + 2, y - labelHeight, badgeWidth, labelHeight);
      ctx.fillStyle = "white";
      ctx.font = "bold 9px system-ui, sans-serif";
      ctx.fillText(badgeText, x + labelWidth + 5, y - 6);
    });

    // Draw SAM preview (cyan color, animated border)
    if (samPreview) {
      const color = "#06B6D4"; // Cyan for SAM
      const x = offset.x + samPreview.bbox.x * imgWidth * scale;
      const y = offset.y + samPreview.bbox.y * imgHeight * scale;
      const width = samPreview.bbox.width * imgWidth * scale;
      const height = samPreview.bbox.height * imgHeight * scale;

      // Animated dashed border
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.setLineDash([8, 4]);
      ctx.strokeRect(x, y, width, height);
      ctx.setLineDash([]);

      // Semi-transparent fill
      ctx.fillStyle = color + "20";
      ctx.fillRect(x, y, width, height);

      // Label
      const label = `SAM ${(samPreview.confidence * 100).toFixed(0)}%`;
      ctx.font = "bold 12px system-ui, sans-serif";
      const labelWidth = ctx.measureText(label).width + 12;
      const labelHeight = 22;

      ctx.fillStyle = color;
      ctx.fillRect(x, y - labelHeight - 4, labelWidth, labelHeight);

      ctx.fillStyle = "white";
      ctx.fillText(label, x + 6, y - 10);

      // Instructions
      ctx.font = "11px system-ui, sans-serif";
      const instructionText = "Enter to accept | Esc to cancel";
      const instructionWidth = ctx.measureText(instructionText).width + 12;
      ctx.fillStyle = "rgba(0,0,0,0.7)";
      ctx.fillRect(x, y + height + 4, instructionWidth, 20);
      ctx.fillStyle = "white";
      ctx.fillText(instructionText, x + 6, y + height + 17);
    }

    // Draw SAM loading indicator
    if (samLoading && toolMode === "sam") {
      ctx.fillStyle = "rgba(0,0,0,0.5)";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      ctx.font = "bold 16px system-ui, sans-serif";
      ctx.fillStyle = "white";
      ctx.textAlign = "center";
      ctx.fillText("Segmenting...", canvas.width / 2, canvas.height / 2);
      ctx.textAlign = "start";
    }
  }, [annotations, selectedIds, hoveredId, drawingBox, pendingBbox, showAnnotations, imageLoaded, scale, offset, lastUsedClassId, classes, toolMode, mousePos, isPanning, aiPredictions, samPreview, samLoading]);

  useEffect(() => {
    drawAnnotations();
  }, [drawAnnotations]);

  // ============================================
  // FILTER CLASSES
  // ============================================

  const filteredClasses = classes?.filter((c) =>
    c.name.toLowerCase().includes(classInput.toLowerCase()) ||
    c.display_name?.toLowerCase().includes(classInput.toLowerCase())
  ) ?? [];

  const filteredClassesForChange = classes?.filter((c) =>
    c.name.toLowerCase().includes(classChangeInput.toLowerCase()) ||
    c.display_name?.toLowerCase().includes(classChangeInput.toLowerCase())
  ) ?? [];

  // ============================================
  // LOADING
  // ============================================

  if (imageInfoLoading || annotationsLoading) {
    return (
      <div className="flex items-center justify-center h-[calc(100vh-4rem)]">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  // ============================================
  // RENDER
  // ============================================

  return (
    <div className="h-[calc(100vh-4rem)] flex flex-col bg-background">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 border-b bg-card">
        <div className="flex items-center gap-4">
          <Button variant="ghost" size="sm" asChild>
            <Link href={`/od/datasets/${datasetId}`}>
              <Home className="h-4 w-4 mr-1" />
              Back
            </Link>
          </Button>
          <div className="text-sm">
            <span className="text-muted-foreground">{dataset?.name}</span>
            <span className="mx-2 text-muted-foreground">/</span>
            <span className="font-medium">{image?.filename}</span>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-sm text-muted-foreground">
            {currentImageIndex + 1} / {totalImages}
          </span>
          <Button variant="outline" size="sm" onClick={goToPrevImage} disabled={!prevImage}>
            <ChevronLeft className="h-4 w-4" />
          </Button>
          <Button variant="outline" size="sm" onClick={goToNextImage} disabled={!nextImage}>
            <ChevronRight className="h-4 w-4" />
          </Button>
        </div>

        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowAnnotations(!showAnnotations)}
          >
            {showAnnotations ? <Eye className="h-4 w-4" /> : <EyeOff className="h-4 w-4" />}
          </Button>

          <Button
            variant="default"
            size="sm"
            onClick={() => markCompletedMutation.mutate()}
            disabled={markCompletedMutation.isPending}
          >
            {markCompletedMutation.isPending ? (
              <Loader2 className="h-4 w-4 mr-1 animate-spin" />
            ) : (
              <CheckCircle className="h-4 w-4 mr-1" />
            )}
            Done
          </Button>
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 flex overflow-hidden min-h-0">
        {/* Left toolbar */}
        <div className="w-12 border-r bg-card flex flex-col items-center py-2 gap-1">
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant={toolMode === "select" ? "default" : "ghost"}
                size="icon"
                className="h-9 w-9"
                onClick={() => setToolMode("select")}
              >
                <MousePointer2 className="h-4 w-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent side="right">Select (V)</TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant={toolMode === "bbox" ? "default" : "ghost"}
                size="icon"
                className="h-9 w-9"
                onClick={() => setToolMode("bbox")}
              >
                <Square className="h-4 w-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent side="right">Bounding Box (B)</TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant={toolMode === "sam" ? "default" : "ghost"}
                size="icon"
                className="h-9 w-9"
                onClick={() => setToolMode("sam")}
              >
                <Crosshair className="h-4 w-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent side="right">SAM Click-to-Segment (S)</TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant={continuousDrawMode ? "secondary" : "ghost"}
                size="icon"
                className="h-9 w-9"
                onClick={() => setContinuousDrawMode(!continuousDrawMode)}
              >
                <Repeat className="h-4 w-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent side="right">
              Continuous Draw: {continuousDrawMode ? "ON" : "OFF"}
            </TooltipContent>
          </Tooltip>

          <div className="h-px w-6 bg-border my-1" />

          <Tooltip>
            <TooltipTrigger asChild>
              <Button variant="ghost" size="icon" className="h-9 w-9" onClick={undo} disabled={historyIndex <= 0}>
                <Undo2 className="h-4 w-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent side="right">Undo (Ctrl+Z)</TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="ghost"
                size="icon"
                className="h-9 w-9"
                onClick={redo}
                disabled={historyIndex >= history.length - 1}
              >
                <Redo2 className="h-4 w-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent side="right">Redo (Ctrl+Y)</TooltipContent>
          </Tooltip>

          <div className="h-px w-6 bg-border my-1" />

          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="ghost"
                size="icon"
                className="h-9 w-9"
                onClick={copySelected}
                disabled={selectedIds.size === 0}
              >
                <Copy className="h-4 w-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent side="right">Copy (Ctrl+C)</TooltipContent>
          </Tooltip>

          <div className="flex-1" />

          <Tooltip>
            <TooltipTrigger asChild>
              <Button variant="ghost" size="icon" className="h-9 w-9" onClick={zoomOut}>
                <ZoomOut className="h-4 w-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent side="right">Zoom Out (-)</TooltipContent>
          </Tooltip>

          <div className="text-xs text-muted-foreground font-mono w-9 text-center">
            {Math.round(scale * 100)}%
          </div>

          <Tooltip>
            <TooltipTrigger asChild>
              <Button variant="ghost" size="icon" className="h-9 w-9" onClick={zoomIn}>
                <ZoomIn className="h-4 w-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent side="right">Zoom In (+)</TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <Button variant="ghost" size="icon" className="h-9 w-9" onClick={resetZoom}>
                <Maximize2 className="h-4 w-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent side="right">Fit to View (0)</TooltipContent>
          </Tooltip>
        </div>

        {/* Canvas area */}
        <div
          ref={containerRef}
          className="flex-1 relative bg-neutral-900 overflow-hidden"
          onWheel={handleWheel}
          onContextMenu={handleContextMenu}
        >
          {/* Actual image */}
          {image?.image_url && (
            <img
              ref={imageRef}
              src={image.image_url}
              alt={image.filename}
              onLoad={handleImageLoad}
              onError={() => toast.error("Failed to load image")}
              className="absolute"
              style={{
                left: offset.x,
                top: offset.y,
                width: imageRef.current ? imageRef.current.naturalWidth * scale : "auto",
                height: imageRef.current ? imageRef.current.naturalHeight * scale : "auto",
                maxWidth: "none",
              }}
              draggable={false}
            />
          )}

          {/* Canvas overlay */}
          <canvas
            ref={canvasRef}
            className="absolute inset-0"
            style={{ cursor: spacePressed || isPanning ? (isPanning ? "grabbing" : "grab") : (toolMode === "bbox" || toolMode === "sam") ? "crosshair" : "default" }}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={() => {
              handleMouseUp();
              setMousePos(null);
            }}
            onDoubleClick={handleDoubleClick}
          />

          {/* Loading indicator */}
          {!imageLoaded && image?.image_url && (
            <div className="absolute inset-0 flex items-center justify-center bg-neutral-900">
              <div className="flex flex-col items-center gap-2">
                <Loader2 className="h-8 w-8 animate-spin text-white" />
                <p className="text-sm text-white/70">Loading image...</p>
              </div>
            </div>
          )}

          {/* Class input popup (for new bbox) */}
          {showClassInput && pendingBbox && imageRef.current && (
            <div
              className="absolute z-50 bg-card border rounded-lg shadow-xl p-3 min-w-[220px]"
              style={{
                left: Math.min(
                  offset.x + pendingBbox.x * imageRef.current.naturalWidth * scale,
                  (containerRef.current?.clientWidth || 400) - 240
                ),
                top: Math.min(
                  offset.y + (pendingBbox.y + pendingBbox.height) * imageRef.current.naturalHeight * scale + 8,
                  (containerRef.current?.clientHeight || 300) - 200
                ),
              }}
            >
              <p className="text-xs text-muted-foreground mb-2">Enter class name:</p>
              <Input
                ref={classInputRef}
                value={classInput}
                onChange={(e) => setClassInput(e.target.value)}
                onKeyDown={handleClassInputKeyDown}
                placeholder="e.g., product, shelf..."
                className="h-8 text-sm"
                autoFocus
              />
              {filteredClasses.length > 0 && (
                <div className="mt-2 border-t pt-2">
                  <p className="text-xs text-muted-foreground mb-1">Existing classes:</p>
                  <div className="flex flex-wrap gap-1 max-h-24 overflow-y-auto">
                    {filteredClasses.slice(0, 10).map((c) => (
                      <button
                        key={c.id}
                        onClick={() => {
                          setClassInput(c.name);
                          setTimeout(handleClassSubmit, 0);
                        }}
                        className="px-2 py-1 text-xs rounded-md flex items-center gap-1 hover:bg-muted border"
                        style={{ borderColor: c.color + "60" }}
                      >
                        <div className="w-2 h-2 rounded-full" style={{ backgroundColor: c.color }} />
                        {c.display_name || c.name}
                      </button>
                    ))}
                  </div>
                </div>
              )}
              <div className="mt-2 flex gap-2">
                <Button size="sm" className="h-7 text-xs flex-1" onClick={handleClassSubmit}>
                  <Plus className="h-3 w-3 mr-1" />
                  Add
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  className="h-7 text-xs"
                  onClick={() => {
                    setPendingBbox(null);
                    setShowClassInput(false);
                  }}
                >
                  Cancel
                </Button>
              </div>
            </div>
          )}

          {/* Inline class change popup (positioned at bbox) */}
          {showClassChange && selectedIds.size === 1 && (
            <div
              className="absolute z-50 bg-card border rounded-lg shadow-xl p-3 min-w-[220px]"
              style={{
                left: classChangePosition
                  ? Math.min(classChangePosition.x, (containerRef.current?.clientWidth || 400) - 240)
                  : "50%",
                top: classChangePosition
                  ? Math.min(classChangePosition.y, (containerRef.current?.clientHeight || 300) - 200)
                  : "50%",
                transform: classChangePosition ? undefined : "translate(-50%, -50%)",
              }}
            >
              <p className="text-xs text-muted-foreground mb-2">Change class:</p>
              <Input
                id="class-change-input"
                value={classChangeInput}
                onChange={(e) => setClassChangeInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    e.preventDefault();
                    handleQuickClassChange();
                  } else if (e.key === "Escape") {
                    setShowClassChange(false);
                    setClassChangePosition(null);
                  }
                }}
                placeholder="Class name..."
                className="h-8 text-sm"
                autoFocus
              />
              {filteredClassesForChange.length > 0 && (
                <div className="mt-2 border-t pt-2">
                  <div className="flex flex-wrap gap-1 max-h-24 overflow-y-auto">
                    {filteredClassesForChange.slice(0, 10).map((c) => (
                      <button
                        key={c.id}
                        onClick={() => handleQuickClassChange(c.id)}
                        className="px-2 py-1 text-xs rounded-md flex items-center gap-1 hover:bg-muted border"
                        style={{ borderColor: c.color + "60" }}
                      >
                        <div className="w-2 h-2 rounded-full" style={{ backgroundColor: c.color }} />
                        {c.display_name || c.name}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Right sidebar */}
        <div className="w-64 border-l bg-card flex flex-col overflow-hidden min-h-0">
          <div className="p-3 border-b flex items-center justify-between">
            <h3 className="font-medium text-sm flex items-center gap-2">
              <Layers className="h-4 w-4" />
              Annotations
              <Badge variant="secondary">{annotations.length}</Badge>
            </h3>
            {selectedIds.size > 0 && (
              <Button variant="ghost" size="sm" className="h-6 text-xs" onClick={deleteSelected}>
                <Trash2 className="h-3 w-3 mr-1" />
                {selectedIds.size}
              </Button>
            )}
          </div>

          <ScrollArea className="flex-1 min-h-0">
            <div className="p-2 space-y-1">
              {annotations.length === 0 ? (
                <p className="text-sm text-muted-foreground text-center py-4">
                  Draw a box to annotate
                </p>
              ) : (
                annotations.map((ann, index) => (
                  <div
                    key={ann.id}
                    onClick={(e) => {
                      if (e.shiftKey) {
                        setSelectedIds((prev) => {
                          const next = new Set(prev);
                          if (next.has(ann.id)) {
                            next.delete(ann.id);
                          } else {
                            next.add(ann.id);
                          }
                          return next;
                        });
                      } else {
                        setSelectedIds(new Set([ann.id]));
                      }
                    }}
                    onDoubleClick={() => {
                      setSelectedIds(new Set([ann.id]));
                      const img = imageRef.current;
                      if (img) {
                        const bboxX = offset.x + ann.bbox.x * img.naturalWidth * scale;
                        const bboxY = offset.y + (ann.bbox.y + ann.bbox.height) * img.naturalHeight * scale + 8;
                        setClassChangePosition({ x: bboxX, y: bboxY });
                      }
                      setShowClassChange(true);
                      setClassChangeInput("");
                      setTimeout(() => document.getElementById("class-change-input")?.focus(), 50);
                    }}
                    onMouseEnter={() => setHoveredId(ann.id)}
                    onMouseLeave={() => setHoveredId(null)}
                    className={`flex items-center gap-2 px-3 py-2 rounded-md cursor-pointer transition-colors ${
                      selectedIds.has(ann.id)
                        ? "bg-primary/10 border border-primary"
                        : hoveredId === ann.id
                        ? "bg-muted/80"
                        : "hover:bg-muted"
                    }`}
                  >
                    <span className="text-xs text-muted-foreground w-4">{index + 1}</span>
                    <div
                      className="w-3 h-3 rounded"
                      style={{ backgroundColor: ann.class_color }}
                    />
                    <span className="flex-1 text-sm truncate">{ann.class_name}</span>
                    {ann.is_ai_generated && (
                      <Badge variant="outline" className="text-xs">AI</Badge>
                    )}
                    {selectedIds.has(ann.id) && (
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-6 w-6"
                        onClick={(e) => {
                          e.stopPropagation();
                          deleteMutation.mutate(ann.id);
                        }}
                      >
                        <Trash2 className="h-3 w-3 text-destructive" />
                      </Button>
                    )}
                  </div>
                ))
              )}
            </div>
          </ScrollArea>

          {/* Classes section */}
          <div className="border-t p-3">
            <p className="text-xs text-muted-foreground mb-2">Quick class (1-9):</p>
            <div className="flex flex-wrap gap-1">
              {classes?.slice(0, 9).map((c, i) => (
                <button
                  key={c.id}
                  onClick={() => {
                    if (selectedIds.size === 1) {
                      handleQuickClassChange(c.id);
                    } else {
                      setLastUsedClassId(c.id);
                      toast.info(`Selected: ${c.display_name || c.name}`);
                    }
                  }}
                  className={`px-2 py-1 text-xs rounded-md flex items-center gap-1 border transition-colors ${
                    lastUsedClassId === c.id ? "ring-2 ring-primary" : "hover:bg-muted"
                  }`}
                  style={{ borderColor: c.color + "60" }}
                >
                  <span className="text-muted-foreground w-3">{i + 1}</span>
                  <div className="w-2 h-2 rounded-full" style={{ backgroundColor: c.color }} />
                  <span className="truncate max-w-16">{c.display_name || c.name}</span>
                </button>
              ))}
            </div>
          </div>

          {/* AI Assist Panel */}
          <AIPanel
            imageId={imageId}
            datasetId={datasetId}
            predictions={aiPredictions}
            onPredictionsReceived={handleAIPredictionsReceived}
            onAcceptPrediction={handleAcceptAIPrediction}
            onRejectPrediction={handleRejectAIPrediction}
            onAcceptAll={handleAcceptAllAIPredictions}
            onRejectAll={handleRejectAllAIPredictions}
          />

          {/* Shortcuts toggle button */}
          <div className="p-3 border-t">
            <Button
              variant="outline"
              size="sm"
              className="w-full"
              onClick={() => setShowShortcuts(!showShortcuts)}
            >
              <Keyboard className="h-4 w-4 mr-2" />
              Keyboard Shortcuts
            </Button>
          </div>
        </div>
      </div>

      {/* Floating Keyboard Shortcuts Panel */}
      {showShortcuts && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50" onClick={() => setShowShortcuts(false)}>
          <div
            className="bg-card border rounded-xl shadow-2xl p-6 max-w-2xl w-full mx-4 max-h-[80vh] overflow-y-auto"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-semibold flex items-center gap-2">
                <Keyboard className="h-5 w-5" />
                Keyboard Shortcuts
              </h2>
              <Button variant="ghost" size="icon" onClick={() => setShowShortcuts(false)}>
                <X className="h-5 w-5" />
              </Button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Tools */}
              <div>
                <h3 className="font-medium text-sm text-muted-foreground mb-3 uppercase tracking-wide">Tools</h3>
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span>Select Mode</span>
                    <kbd className="px-2 py-1 bg-muted rounded text-sm font-mono">V</kbd>
                  </div>
                  <div className="flex justify-between items-center">
                    <span>Bounding Box Mode</span>
                    <kbd className="px-2 py-1 bg-muted rounded text-sm font-mono">B</kbd>
                  </div>
                  <div className="flex justify-between items-center">
                    <span>SAM Click-to-Segment</span>
                    <kbd className="px-2 py-1 bg-muted rounded text-sm font-mono">S</kbd>
                  </div>
                  <div className="flex justify-between items-center">
                    <span>Select Class (1-9)</span>
                    <kbd className="px-2 py-1 bg-muted rounded text-sm font-mono">1-9</kbd>
                  </div>
                </div>
              </div>

              {/* Selection */}
              <div>
                <h3 className="font-medium text-sm text-muted-foreground mb-3 uppercase tracking-wide">Selection</h3>
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span>Select All</span>
                    <kbd className="px-2 py-1 bg-muted rounded text-sm font-mono">Ctrl+A</kbd>
                  </div>
                  <div className="flex justify-between items-center">
                    <span>Deselect</span>
                    <kbd className="px-2 py-1 bg-muted rounded text-sm font-mono">Escape</kbd>
                  </div>
                  <div className="flex justify-between items-center">
                    <span>Next Annotation</span>
                    <kbd className="px-2 py-1 bg-muted rounded text-sm font-mono">Tab</kbd>
                  </div>
                  <div className="flex justify-between items-center">
                    <span>Previous Annotation</span>
                    <kbd className="px-2 py-1 bg-muted rounded text-sm font-mono">Shift+Tab</kbd>
                  </div>
                </div>
              </div>

              {/* Edit */}
              <div>
                <h3 className="font-medium text-sm text-muted-foreground mb-3 uppercase tracking-wide">Edit</h3>
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span>Delete Selected</span>
                    <div className="flex gap-1">
                      <kbd className="px-2 py-1 bg-muted rounded text-sm font-mono">D</kbd>
                      <span className="text-muted-foreground">or</span>
                      <kbd className="px-2 py-1 bg-muted rounded text-sm font-mono">Del</kbd>
                    </div>
                  </div>
                  <div className="flex justify-between items-center">
                    <span>Copy</span>
                    <kbd className="px-2 py-1 bg-muted rounded text-sm font-mono">Ctrl+C</kbd>
                  </div>
                  <div className="flex justify-between items-center">
                    <span>Paste</span>
                    <kbd className="px-2 py-1 bg-muted rounded text-sm font-mono">Ctrl+V</kbd>
                  </div>
                  <div className="flex justify-between items-center">
                    <span>Undo</span>
                    <kbd className="px-2 py-1 bg-muted rounded text-sm font-mono">Ctrl+Z</kbd>
                  </div>
                  <div className="flex justify-between items-center">
                    <span>Redo</span>
                    <kbd className="px-2 py-1 bg-muted rounded text-sm font-mono">Ctrl+Y</kbd>
                  </div>
                  <div className="flex justify-between items-center">
                    <span>Change Class</span>
                    <kbd className="px-2 py-1 bg-muted rounded text-sm font-mono">C</kbd>
                  </div>
                </div>
              </div>

              {/* View */}
              <div>
                <h3 className="font-medium text-sm text-muted-foreground mb-3 uppercase tracking-wide">View & Navigation</h3>
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span>Pan Mode (hold)</span>
                    <kbd className="px-2 py-1 bg-muted rounded text-sm font-mono">Space</kbd>
                  </div>
                  <div className="flex justify-between items-center">
                    <span>Zoom In/Out</span>
                    <div className="flex gap-1">
                      <kbd className="px-2 py-1 bg-muted rounded text-sm font-mono">+</kbd>
                      <kbd className="px-2 py-1 bg-muted rounded text-sm font-mono">-</kbd>
                    </div>
                  </div>
                  <div className="flex justify-between items-center">
                    <span>Reset Zoom</span>
                    <kbd className="px-2 py-1 bg-muted rounded text-sm font-mono">0</kbd>
                  </div>
                  <div className="flex justify-between items-center">
                    <span>Next Image</span>
                    <kbd className="px-2 py-1 bg-muted rounded text-sm font-mono">→</kbd>
                  </div>
                  <div className="flex justify-between items-center">
                    <span>Previous Image</span>
                    <kbd className="px-2 py-1 bg-muted rounded text-sm font-mono">←</kbd>
                  </div>
                </div>
              </div>

              {/* Save */}
              <div className="md:col-span-2">
                <h3 className="font-medium text-sm text-muted-foreground mb-3 uppercase tracking-wide">Save</h3>
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span>Mark Complete & Next</span>
                    <kbd className="px-2 py-1 bg-muted rounded text-sm font-mono">Ctrl+S</kbd>
                  </div>
                </div>
              </div>

              {/* Mouse */}
              <div className="md:col-span-2">
                <h3 className="font-medium text-sm text-muted-foreground mb-3 uppercase tracking-wide">Mouse</h3>
                <div className="grid grid-cols-2 gap-2">
                  <div className="flex justify-between items-center">
                    <span>Pan</span>
                    <span className="text-sm text-muted-foreground">Right-click drag</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span>Zoom</span>
                    <span className="text-sm text-muted-foreground">Scroll wheel</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span>Multi-select</span>
                    <span className="text-sm text-muted-foreground">Shift+click</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span>Quick Class Change</span>
                    <span className="text-sm text-muted-foreground">Double-click</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
