"use client";

import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { Eye, RotateCcw, Play, Plus, Trash2, Loader2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

import { apiClient } from "@/lib/api-client";
import { FrameCanvas, Point } from "./FrameCanvas";

interface Product {
  id: string;
  video_url?: string;
  grounding_prompt?: string;
  status?: string;
}

interface SegmentationTabProps {
  product: Product;
}

interface MaskStats {
  pixel_count: number;
  coverage_percent: number;
  width: number;
  height: number;
}

export function SegmentationTab({ product }: SegmentationTabProps) {
  const queryClient = useQueryClient();

  // State
  const [textPrompts, setTextPrompts] = useState<string[]>(
    product.grounding_prompt ? [product.grounding_prompt] : []
  );
  const [newPrompt, setNewPrompt] = useState("");
  const [points, setPoints] = useState<Point[]>([]);
  const [previewMask, setPreviewMask] = useState<string | null>(null);
  const [maskStats, setMaskStats] = useState<MaskStats | null>(null);

  // Preview mutation
  const previewMutation = useMutation({
    mutationFn: async () => {
      return apiClient.previewSegmentation(product.id, {
        text_prompts: textPrompts.filter(Boolean),
        points: points,
      });
    },
    onSuccess: (data) => {
      setPreviewMask(data.mask_image);
      setMaskStats(data.mask_stats);
      toast.success("Preview generated successfully");
    },
    onError: (error: Error) => {
      toast.error(error.message || "Preview failed");
    },
  });

  // Reprocess mutation
  const reprocessMutation = useMutation({
    mutationFn: async () => {
      return apiClient.reprocessProduct(product.id, {
        custom_prompts: textPrompts.filter(Boolean),
        points: points,
      });
    },
    onSuccess: (result) => {
      toast.success(result.message || "Reprocessing started");
      queryClient.invalidateQueries({ queryKey: ["product", product.id] });
      queryClient.invalidateQueries({ queryKey: ["product-frames", product.id] });
      // Clear preview after starting reprocess
      setPreviewMask(null);
      setMaskStats(null);
    },
    onError: (error: Error) => {
      toast.error(error.message || "Reprocess failed");
    },
  });

  // Handlers
  const handleAddPrompt = () => {
    if (newPrompt.trim()) {
      setTextPrompts([...textPrompts, newPrompt.trim()]);
      setNewPrompt("");
      // Clear preview when prompts change
      setPreviewMask(null);
      setMaskStats(null);
    }
  };

  const handleRemovePrompt = (index: number) => {
    setTextPrompts(textPrompts.filter((_, i) => i !== index));
    setPreviewMask(null);
    setMaskStats(null);
  };

  const handleAddPoint = (point: Point) => {
    setPoints([...points, point]);
    setPreviewMask(null);
    setMaskStats(null);
  };

  const handleRemovePoint = (index: number) => {
    setPoints(points.filter((_, i) => i !== index));
    setPreviewMask(null);
    setMaskStats(null);
  };

  const handleClearAll = () => {
    setPoints([]);
    setPreviewMask(null);
    setMaskStats(null);
  };

  const canPreview = textPrompts.length > 0 || points.length > 0;
  const canReprocess = previewMask !== null;
  // Don't block on processing status - user may want to manually segment
  const isProcessing = false; // Was: product.status === "processing"

  if (!product.video_url) {
    return (
      <Card>
        <CardContent className="pt-6">
          <p className="text-center text-gray-500">
            No video available for this product. Upload a video first.
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      {/* Processing status banner */}
      {isProcessing && (
        <Card className="border-yellow-200 bg-yellow-50">
          <CardContent className="pt-4 pb-4">
            <div className="flex items-center gap-2 text-yellow-800">
              <Loader2 className="h-4 w-4 animate-spin" />
              <span>Video is currently being processed. Please wait...</span>
            </div>
          </CardContent>
        </Card>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Left: Canvas */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-lg">First Frame</CardTitle>
            <CardDescription>
              Click to add points. Left = include, Right = exclude.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <FrameCanvas
              videoUrl={product.video_url}
              points={points}
              maskOverlay={previewMask}
              onAddPoint={handleAddPoint}
              onRemovePoint={handleRemovePoint}
              isLoading={previewMutation.isPending}
            />
          </CardContent>
        </Card>

        {/* Right: Controls */}
        <div className="space-y-4">
          {/* Text Prompts */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-lg">Text Prompts</CardTitle>
              <CardDescription>
                Describe the object to segment (e.g., "the red can on the left")
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              {/* Existing prompts */}
              {textPrompts.length > 0 && (
                <div className="flex flex-wrap gap-2">
                  {textPrompts.map((prompt, index) => (
                    <Badge
                      key={index}
                      variant="secondary"
                      className="flex items-center gap-1 py-1"
                    >
                      <span className="max-w-[200px] truncate">{prompt}</span>
                      <button
                        onClick={() => handleRemovePrompt(index)}
                        className="hover:bg-gray-300 rounded-full p-0.5"
                      >
                        <Trash2 className="h-3 w-3" />
                      </button>
                    </Badge>
                  ))}
                </div>
              )}

              {/* Add new prompt */}
              <div className="flex gap-2">
                <Input
                  placeholder="Enter text prompt..."
                  value={newPrompt}
                  onChange={(e) => setNewPrompt(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleAddPrompt()}
                />
                <Button
                  variant="outline"
                  size="icon"
                  onClick={handleAddPrompt}
                  disabled={!newPrompt.trim()}
                >
                  <Plus className="h-4 w-4" />
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* Points Summary */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-lg">Point Prompts</CardTitle>
              <CardDescription>
                {points.length} point{points.length !== 1 ? "s" : ""} selected
              </CardDescription>
            </CardHeader>
            <CardContent>
              {points.length > 0 ? (
                <div className="space-y-2">
                  <div className="flex flex-wrap gap-2">
                    {points.map((point, index) => (
                      <Badge
                        key={index}
                        variant={point.label === 1 ? "default" : "destructive"}
                        className="flex items-center gap-1"
                      >
                        {point.label === 1 ? "+" : "-"} (
                        {(point.x * 100).toFixed(0)}%,{" "}
                        {(point.y * 100).toFixed(0)}%)
                        <button
                          onClick={() => handleRemovePoint(index)}
                          className="hover:bg-white/20 rounded-full p-0.5"
                        >
                          <Trash2 className="h-3 w-3" />
                        </button>
                      </Badge>
                    ))}
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={handleClearAll}
                    className="text-red-500 hover:text-red-700"
                  >
                    Clear all points
                  </Button>
                </div>
              ) : (
                <p className="text-sm text-gray-500">
                  Click on the frame to add points
                </p>
              )}
            </CardContent>
          </Card>

          {/* Preview Button */}
          <Button
            onClick={() => previewMutation.mutate()}
            disabled={!canPreview || previewMutation.isPending || isProcessing}
            className="w-full"
            size="lg"
          >
            {previewMutation.isPending ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Generating Preview...
              </>
            ) : (
              <>
                <Eye className="mr-2 h-4 w-4" />
                Preview Segmentation
              </>
            )}
          </Button>

          {/* Preview Results */}
          {maskStats && (
            <Card className="border-green-200 bg-green-50">
              <CardHeader className="pb-2">
                <CardTitle className="text-lg text-green-800">
                  Preview Result
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <div className="grid grid-cols-2 gap-2 text-sm">
                  <div>
                    <Label className="text-green-700">Pixels</Label>
                    <p className="font-medium">
                      {maskStats.pixel_count.toLocaleString()}
                    </p>
                  </div>
                  <div>
                    <Label className="text-green-700">Coverage</Label>
                    <p className="font-medium">{maskStats.coverage_percent}%</p>
                  </div>
                  <div>
                    <Label className="text-green-700">Frame Size</Label>
                    <p className="font-medium">
                      {maskStats.width} x {maskStats.height}
                    </p>
                  </div>
                </div>

                <div className="flex gap-2 pt-2">
                  <Button
                    onClick={() => reprocessMutation.mutate()}
                    disabled={reprocessMutation.isPending || isProcessing}
                    className="flex-1"
                    variant="default"
                  >
                    {reprocessMutation.isPending ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Starting...
                      </>
                    ) : (
                      <>
                        <Play className="mr-2 h-4 w-4" />
                        Process Full Video
                      </>
                    )}
                  </Button>
                  <Button
                    onClick={() => {
                      setPreviewMask(null);
                      setMaskStats(null);
                    }}
                    variant="outline"
                  >
                    <RotateCcw className="mr-2 h-4 w-4" />
                    Try Again
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
