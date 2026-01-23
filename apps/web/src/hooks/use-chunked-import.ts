"use client";

import { useState, useCallback, useRef } from "react";

// ============================================
// Types
// ============================================

export interface ChunkResult {
  success: boolean;
  images_imported: number;
  images_skipped?: number;
  duplicates_found?: number;
  labels_created?: number;
  classes_created?: number;
  errors: string[];
}

export interface ImportError {
  chunk: number;
  message: string;
  ids?: string[];
}

export interface FailedChunk {
  chunkIndex: number;
  ids: string[];
  error: string;
  retryCount: number;
}

export interface ImportProgress {
  status: "idle" | "fetching" | "importing" | "completed" | "error" | "cancelled";
  currentChunk: number;
  totalChunks: number;
  processedItems: number;
  totalItems: number;
  stage: string;

  // Results
  imported: number;
  skipped: number;
  duplicates: number;
  labelsCreated: number;
  classesCreated: number;
  errors: ImportError[];

  // For retry
  failedChunks: FailedChunk[];
}

export interface UseChunkedImportOptions {
  chunkSize?: number;
  maxRetries?: number;
  onProgress?: (progress: ImportProgress) => void;
  onChunkComplete?: (chunkIndex: number, result: ChunkResult) => void;
  onComplete?: (finalResult: ImportProgress) => void;
}

const DEFAULT_CHUNK_SIZE = 500;
const DEFAULT_MAX_RETRIES = 2;

// ============================================
// Hook
// ============================================

export function useChunkedImport(options: UseChunkedImportOptions = {}) {
  const {
    chunkSize = DEFAULT_CHUNK_SIZE,
    maxRetries = DEFAULT_MAX_RETRIES,
    onProgress,
    onChunkComplete,
    onComplete,
  } = options;

  const [progress, setProgress] = useState<ImportProgress>({
    status: "idle",
    currentChunk: 0,
    totalChunks: 0,
    processedItems: 0,
    totalItems: 0,
    stage: "",
    imported: 0,
    skipped: 0,
    duplicates: 0,
    labelsCreated: 0,
    classesCreated: 0,
    errors: [],
    failedChunks: [],
  });

  const abortRef = useRef(false);
  const progressRef = useRef(progress);

  // Keep ref in sync
  progressRef.current = progress;

  const updateProgress = useCallback(
    (updates: Partial<ImportProgress>) => {
      setProgress((prev) => {
        const newProgress = { ...prev, ...updates };
        onProgress?.(newProgress);
        return newProgress;
      });
    },
    [onProgress]
  );

  const importChunked = useCallback(
    async <T extends string>(
      ids: T[],
      importFn: (chunkIds: T[]) => Promise<ChunkResult>
    ): Promise<ImportProgress> => {
      abortRef.current = false;

      // Split into chunks
      const chunks: T[][] = [];
      for (let i = 0; i < ids.length; i += chunkSize) {
        chunks.push(ids.slice(i, i + chunkSize));
      }

      const initialProgress: ImportProgress = {
        status: "importing",
        currentChunk: 0,
        totalChunks: chunks.length,
        processedItems: 0,
        totalItems: ids.length,
        stage: `Processing chunk 1 of ${chunks.length}...`,
        imported: 0,
        skipped: 0,
        duplicates: 0,
        labelsCreated: 0,
        classesCreated: 0,
        errors: [],
        failedChunks: [],
      };

      setProgress(initialProgress);
      onProgress?.(initialProgress);

      let currentProgress = { ...initialProgress };

      for (let i = 0; i < chunks.length; i++) {
        if (abortRef.current) {
          currentProgress = {
            ...currentProgress,
            status: "cancelled",
            stage: `Cancelled after ${i} chunks`,
          };
          setProgress(currentProgress);
          onProgress?.(currentProgress);
          return currentProgress;
        }

        const chunk = chunks[i];
        let result: ChunkResult | null = null;
        let lastError = "";

        // Retry logic with exponential backoff
        for (let retry = 0; retry <= maxRetries; retry++) {
          try {
            result = await importFn(chunk);
            break;
          } catch (error) {
            lastError = error instanceof Error ? error.message : String(error);
            if (retry < maxRetries) {
              // Exponential backoff: 1s, 2s, 4s...
              await new Promise((r) => setTimeout(r, 1000 * Math.pow(2, retry)));
            }
          }
        }

        // Update progress
        currentProgress = {
          ...currentProgress,
          currentChunk: i + 1,
          processedItems: currentProgress.processedItems + chunk.length,
          stage:
            i < chunks.length - 1
              ? `Processing chunk ${i + 2} of ${chunks.length}...`
              : "Finalizing...",
        };

        if (result) {
          currentProgress = {
            ...currentProgress,
            imported: currentProgress.imported + (result.images_imported || 0),
            skipped: currentProgress.skipped + (result.images_skipped || 0),
            duplicates: currentProgress.duplicates + (result.duplicates_found || 0),
            labelsCreated: currentProgress.labelsCreated + (result.labels_created || 0),
            classesCreated: currentProgress.classesCreated + (result.classes_created || 0),
          };

          if (result.errors && result.errors.length > 0) {
            currentProgress = {
              ...currentProgress,
              errors: [
                ...currentProgress.errors,
                {
                  chunk: i,
                  message: result.errors.join("; "),
                },
              ],
            };
          }

          onChunkComplete?.(i, result);
        } else {
          // Chunk failed after all retries
          currentProgress = {
            ...currentProgress,
            failedChunks: [
              ...currentProgress.failedChunks,
              {
                chunkIndex: i,
                ids: chunk as string[],
                error: lastError,
                retryCount: maxRetries,
              },
            ],
            errors: [
              ...currentProgress.errors,
              {
                chunk: i,
                message: lastError,
                ids: chunk as string[],
              },
            ],
          };
        }

        setProgress(currentProgress);
        onProgress?.(currentProgress);
      }

      // Finalize
      const finalStatus =
        currentProgress.failedChunks.length > 0 ? "error" : "completed";
      const finalStage =
        finalStatus === "completed"
          ? `Import completed: ${currentProgress.imported} images imported`
          : `Completed with ${currentProgress.failedChunks.length} failed chunks`;

      currentProgress = {
        ...currentProgress,
        status: finalStatus,
        stage: finalStage,
      };

      setProgress(currentProgress);
      onProgress?.(currentProgress);
      onComplete?.(currentProgress);

      return currentProgress;
    },
    [chunkSize, maxRetries, onProgress, onChunkComplete, onComplete]
  );

  const abort = useCallback(() => {
    abortRef.current = true;
  }, []);

  const retryFailed = useCallback(
    async (importFn: (chunkIds: string[]) => Promise<ChunkResult>) => {
      const failedChunks = [...progressRef.current.failedChunks];
      if (failedChunks.length === 0) return progressRef.current;

      let currentProgress: ImportProgress = {
        ...progressRef.current,
        status: "importing",
        stage: `Retrying ${failedChunks.length} failed chunks...`,
        failedChunks: [],
      };

      setProgress(currentProgress);
      onProgress?.(currentProgress);

      for (const failed of failedChunks) {
        if (abortRef.current) break;

        try {
          const result = await importFn(failed.ids);
          currentProgress = {
            ...currentProgress,
            imported: currentProgress.imported + (result.images_imported || 0),
            skipped: currentProgress.skipped + (result.images_skipped || 0),
            duplicates: currentProgress.duplicates + (result.duplicates_found || 0),
            labelsCreated: currentProgress.labelsCreated + (result.labels_created || 0),
            classesCreated: currentProgress.classesCreated + (result.classes_created || 0),
          };
        } catch (error) {
          currentProgress = {
            ...currentProgress,
            failedChunks: [
              ...currentProgress.failedChunks,
              {
                ...failed,
                retryCount: failed.retryCount + 1,
                error: error instanceof Error ? error.message : String(error),
              },
            ],
          };
        }

        setProgress(currentProgress);
        onProgress?.(currentProgress);
      }

      const finalStatus =
        currentProgress.failedChunks.length > 0 ? "error" : "completed";
      currentProgress = {
        ...currentProgress,
        status: finalStatus,
        stage:
          finalStatus === "completed"
            ? "Retry completed successfully"
            : `Retry completed with ${currentProgress.failedChunks.length} failures`,
      };

      setProgress(currentProgress);
      onProgress?.(currentProgress);
      onComplete?.(currentProgress);

      return currentProgress;
    },
    [onProgress, onComplete]
  );

  const reset = useCallback(() => {
    abortRef.current = false;
    setProgress({
      status: "idle",
      currentChunk: 0,
      totalChunks: 0,
      processedItems: 0,
      totalItems: 0,
      stage: "",
      imported: 0,
      skipped: 0,
      duplicates: 0,
      labelsCreated: 0,
      classesCreated: 0,
      errors: [],
      failedChunks: [],
    });
  }, []);

  return {
    progress,
    importChunked,
    abort,
    retryFailed,
    reset,
    isImporting: progress.status === "importing",
    isCompleted: progress.status === "completed",
    hasErrors: progress.failedChunks.length > 0 || progress.errors.length > 0,
  };
}
