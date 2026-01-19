"use client";

import { useRef, useState, useCallback } from "react";
import { X, Pause, Play } from "lucide-react";

export interface Point {
  x: number; // Normalized 0-1
  y: number; // Normalized 0-1
  label: number; // 1 = positive, 0 = negative
}

interface FrameCanvasProps {
  videoUrl: string;
  primaryImageUrl?: string | null; // Fallback image
  points: Point[];
  maskOverlay?: string | null; // Base64 PNG
  onAddPoint: (point: Point) => void;
  onRemovePoint: (index: number) => void;
  isLoading?: boolean;
}

export function FrameCanvas({
  videoUrl,
  primaryImageUrl,
  points,
  maskOverlay,
  onAddPoint,
  onRemovePoint,
  isLoading = false,
}: FrameCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isPaused, setIsPaused] = useState(true);
  const [isVideoReady, setIsVideoReady] = useState(false);
  const [videoError, setVideoError] = useState<string | null>(null);

  // Handle video loaded
  const handleVideoLoaded = useCallback(() => {
    setIsVideoReady(true);
    setVideoError(null);
    // Pause at first frame
    if (videoRef.current) {
      videoRef.current.currentTime = 0;
      videoRef.current.pause();
    }
  }, []);

  // Handle video error
  const handleVideoError = useCallback(() => {
    setVideoError("Failed to load video");
    setIsVideoReady(false);
  }, []);

  // Toggle play/pause
  const togglePlay = useCallback(() => {
    if (!videoRef.current) return;
    if (isPaused) {
      videoRef.current.play();
      setIsPaused(false);
    } else {
      videoRef.current.pause();
      setIsPaused(true);
    }
  }, [isPaused]);

  // Handle click on video/container
  const handleClick = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      if (isLoading || !isPaused) return;

      const container = containerRef.current;
      if (!container) return;

      const rect = container.getBoundingClientRect();
      const x = (e.clientX - rect.left) / rect.width;
      const y = (e.clientY - rect.top) / rect.height;

      // Clamp to 0-1
      const clampedX = Math.max(0, Math.min(1, x));
      const clampedY = Math.max(0, Math.min(1, y));

      // Left click = positive
      onAddPoint({ x: clampedX, y: clampedY, label: 1 });
    },
    [isLoading, isPaused, onAddPoint]
  );

  // Handle right click for negative points
  const handleContextMenu = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      e.preventDefault();
      if (isLoading || !isPaused) return;

      const container = containerRef.current;
      if (!container) return;

      const rect = container.getBoundingClientRect();
      const x = (e.clientX - rect.left) / rect.width;
      const y = (e.clientY - rect.top) / rect.height;

      const clampedX = Math.max(0, Math.min(1, x));
      const clampedY = Math.max(0, Math.min(1, y));

      // Right click = negative
      onAddPoint({ x: clampedX, y: clampedY, label: 0 });
    },
    [isLoading, isPaused, onAddPoint]
  );

  return (
    <div className="space-y-2">
      {/* Video container with overlay */}
      <div
        ref={containerRef}
        className="relative rounded-lg overflow-hidden border border-gray-200 bg-black"
        onClick={handleClick}
        onContextMenu={handleContextMenu}
        style={{ cursor: isPaused && !isLoading ? "crosshair" : "default" }}
      >
        {/* Video element */}
        <video
          ref={videoRef}
          src={videoUrl}
          poster={primaryImageUrl || undefined}
          className="w-full"
          style={{ maxHeight: "400px", objectFit: "contain" }}
          onLoadedData={handleVideoLoaded}
          onError={handleVideoError}
          onPlay={() => setIsPaused(false)}
          onPause={() => setIsPaused(true)}
          muted
          playsInline
        />

        {/* Video error fallback */}
        {videoError && primaryImageUrl && (
          <div className="absolute inset-0 flex items-center justify-center">
            <img
              src={primaryImageUrl}
              alt="Product frame"
              className="w-full h-full object-contain"
            />
          </div>
        )}

        {/* Mask overlay */}
        {maskOverlay && (
          <img
            src={`data:image/png;base64,${maskOverlay}`}
            alt="Segmentation mask"
            className="absolute inset-0 w-full h-full object-contain pointer-events-none"
            style={{ mixBlendMode: "multiply", opacity: 0.6 }}
          />
        )}

        {/* Point markers */}
        {points.map((point, index) => (
          <div
            key={index}
            className={`absolute w-6 h-6 rounded-full border-2 border-white flex items-center justify-center text-white text-sm font-bold shadow-lg ${
              point.label === 1 ? "bg-green-500" : "bg-red-500"
            }`}
            style={{
              left: `${point.x * 100}%`,
              top: `${point.y * 100}%`,
              transform: "translate(-50%, -50%)",
              pointerEvents: "none",
            }}
          >
            {point.label === 1 ? "+" : "-"}
          </div>
        ))}

        {/* Loading overlay */}
        {isLoading && (
          <div className="absolute inset-0 bg-black/50 flex items-center justify-center">
            <div className="text-white text-sm">Processing...</div>
          </div>
        )}

        {/* Play/Pause button */}
        {isVideoReady && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              togglePlay();
            }}
            className="absolute bottom-2 right-2 p-2 bg-black/70 rounded-full text-white hover:bg-black/90 transition-colors"
          >
            {isPaused ? (
              <Play className="h-4 w-4" />
            ) : (
              <Pause className="h-4 w-4" />
            )}
          </button>
        )}

        {/* Pause hint */}
        {!isPaused && (
          <div className="absolute top-2 left-2 px-2 py-1 bg-yellow-500 text-black text-xs rounded">
            Pause video to add points
          </div>
        )}
      </div>

      {/* Point list */}
      {points.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {points.map((point, index) => (
            <div
              key={index}
              className={`flex items-center gap-1 px-2 py-1 rounded-full text-xs text-white ${
                point.label === 1 ? "bg-green-500" : "bg-red-500"
              }`}
            >
              <span>
                {point.label === 1 ? "+" : "-"} ({(point.x * 100).toFixed(0)}%,{" "}
                {(point.y * 100).toFixed(0)}%)
              </span>
              <button
                onClick={() => onRemovePoint(index)}
                className="hover:bg-white/20 rounded-full p-0.5"
              >
                <X className="h-3 w-3" />
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Instructions */}
      <p className="text-xs text-gray-500">
        {isPaused
          ? "Left click = include (green) | Right click = exclude (red)"
          : "Pause the video to add points"}
      </p>
    </div>
  );
}
