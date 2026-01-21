/**
 * Smart Defaults Utility
 *
 * Generates intelligent training configuration recommendations based on
 * dataset characteristics.
 */

import type {
  SmartDefaultsInput,
  SmartDefaults,
  PreprocessStepData,
  OfflineAugStepData,
  OnlineAugStepData,
  ModelStepData,
  HyperparamsStepData,
  AugmentationPreset,
} from "../types/wizard";

/**
 * Dataset size thresholds for recommendations
 */
const DATASET_THRESHOLDS = {
  TINY: 100,      // < 100 images
  SMALL: 500,     // 100-500 images
  MEDIUM: 2000,   // 500-2000 images
  LARGE: 10000,   // 2000-10000 images
  // > 10000 is considered huge
};

/**
 * Get dataset size category
 */
function getDatasetSizeCategory(count: number): "tiny" | "small" | "medium" | "large" | "huge" {
  if (count < DATASET_THRESHOLDS.TINY) return "tiny";
  if (count < DATASET_THRESHOLDS.SMALL) return "small";
  if (count < DATASET_THRESHOLDS.MEDIUM) return "medium";
  if (count < DATASET_THRESHOLDS.LARGE) return "large";
  return "huge";
}

/**
 * Recommend preprocessing settings
 */
function recommendPreprocess(input: SmartDefaultsInput): Partial<PreprocessStepData> {
  const { avgImageSize, hasSmallObjects } = input;
  const avgDim = Math.max(avgImageSize.width, avgImageSize.height);

  let targetSize = 640;
  let resizeStrategy: PreprocessStepData["resizeStrategy"] = "letterbox";
  const tiling = { enabled: false, tileSize: 640, overlap: 0.2, minObjectArea: 0.1 };

  // Large images with small objects benefit from tiling
  if (avgDim > 1500 && hasSmallObjects) {
    tiling.enabled = true;
    tiling.tileSize = 640;
    tiling.overlap = 0.25;
    targetSize = 640;
  } else if (avgDim > 1200) {
    targetSize = 800;
  } else if (avgDim < 500) {
    targetSize = 512;
  }

  return {
    targetSize,
    resizeStrategy,
    tiling,
    autoOrient: true,
    normalizePixels: true,
  };
}

/**
 * Recommend offline augmentation settings
 */
function recommendOfflineAug(input: SmartDefaultsInput): Partial<OfflineAugStepData> {
  const sizeCategory = getDatasetSizeCategory(input.datasetSize);

  // Only recommend offline augmentation for small datasets
  if (sizeCategory === "tiny") {
    return {
      enabled: true,
      multiplier: 5,
    };
  }

  if (sizeCategory === "small") {
    return {
      enabled: true,
      multiplier: 3,
    };
  }

  // Medium+ datasets don't need offline augmentation
  return {
    enabled: false,
    multiplier: 1,
  };
}

/**
 * Recommend online augmentation preset
 */
function recommendOnlineAug(input: SmartDefaultsInput): Partial<OnlineAugStepData> {
  const sizeCategory = getDatasetSizeCategory(input.datasetSize);

  let preset: AugmentationPreset = "sota-v2";

  switch (sizeCategory) {
    case "tiny":
      preset = "heavy"; // Maximum augmentation for tiny datasets
      break;
    case "small":
      preset = "heavy"; // Heavy for small datasets
      break;
    case "medium":
      preset = "sota-v2"; // SOTA-v2 for medium
      break;
    case "large":
      preset = "sota-v2"; // SOTA-v2 for large
      break;
    case "huge":
      preset = "medium"; // Less augmentation for huge datasets (already diverse)
      break;
  }

  return { preset };
}

/**
 * Recommend model settings
 */
function recommendModel(input: SmartDefaultsInput): Partial<ModelStepData> {
  const sizeCategory = getDatasetSizeCategory(input.datasetSize);
  const { classCount, hasSmallObjects, hasDenseAnnotations } = input;

  // Default to RT-DETR Large
  let modelType: ModelStepData["modelType"] = "rt-detr";
  let modelSize: ModelStepData["modelSize"] = "l";
  let freezeBackbone = false;
  let freezeEpochs = 0;

  // For small datasets, consider D-FINE (better with less data)
  if (sizeCategory === "tiny" || sizeCategory === "small") {
    modelType = "d-fine";
    modelSize = "m";
    freezeBackbone = true;
    freezeEpochs = 10;
  }

  // For small objects, larger model sizes help
  if (hasSmallObjects) {
    modelSize = "l";
  }

  // For many classes or dense scenes, consider larger models
  if (classCount > 50 || hasDenseAnnotations) {
    modelSize = "l";
    if (modelType === "d-fine") {
      modelSize = "x";
    }
  }

  // For huge datasets, RT-DETR is faster
  if (sizeCategory === "huge") {
    modelType = "rt-detr";
    modelSize = "l";
    freezeBackbone = false;
  }

  return {
    modelType,
    modelSize,
    pretrained: true,
    freezeBackbone,
    freezeEpochs,
  };
}

/**
 * Recommend hyperparameters
 */
function recommendHyperparams(input: SmartDefaultsInput): Partial<HyperparamsStepData> {
  const sizeCategory = getDatasetSizeCategory(input.datasetSize);

  // Base recommendations
  let epochs = 100;
  let batchSize = 16;
  let learningRate = 0.0001;
  let patience = 20;
  let warmupEpochs = 3;

  switch (sizeCategory) {
    case "tiny":
      epochs = 200; // More epochs for small data
      batchSize = 8;
      learningRate = 0.00005; // Lower LR
      patience = 30;
      warmupEpochs = 5;
      break;
    case "small":
      epochs = 150;
      batchSize = 8;
      learningRate = 0.0001;
      patience = 25;
      warmupEpochs = 5;
      break;
    case "medium":
      epochs = 100;
      batchSize = 16;
      learningRate = 0.0001;
      patience = 20;
      warmupEpochs = 3;
      break;
    case "large":
      epochs = 80;
      batchSize = 16;
      learningRate = 0.0001;
      patience = 15;
      warmupEpochs = 3;
      break;
    case "huge":
      epochs = 50;
      batchSize = 32;
      learningRate = 0.0002; // Slightly higher LR
      patience = 10;
      warmupEpochs = 2;
      break;
  }

  return {
    epochs,
    batchSize,
    learningRate,
    patience,
    warmupEpochs,
    // SOTA features always recommended
    useEma: true,
    emaDecay: 0.9999,
    useLlrd: true,
    llrdDecay: 0.9,
    useMixedPrecision: true,
    gradientClip: 1.0,
    scheduler: "cosine",
    minLrRatio: 0.01,
  };
}

/**
 * Generate reasoning for recommendations
 */
function generateReasoning(input: SmartDefaultsInput): string[] {
  const sizeCategory = getDatasetSizeCategory(input.datasetSize);
  const reasoning: string[] = [];

  // Dataset size reasoning
  reasoning.push(
    `Dataset size: ${input.datasetSize} images (${sizeCategory})`
  );

  if (sizeCategory === "tiny" || sizeCategory === "small") {
    reasoning.push(
      "Small dataset detected: Enabled heavy augmentation and longer training"
    );
    reasoning.push(
      "Recommended offline augmentation to increase effective dataset size"
    );
  }

  if (input.hasSmallObjects) {
    reasoning.push(
      "Small objects detected: Using larger image size and model"
    );
  }

  if (input.hasDenseAnnotations) {
    reasoning.push(
      "Dense annotations detected: Using larger model for better accuracy"
    );
  }

  if (input.classCount > 30) {
    reasoning.push(
      `Many classes (${input.classCount}): Extended training epochs`
    );
  }

  // Image size reasoning
  const avgDim = Math.max(input.avgImageSize.width, input.avgImageSize.height);
  if (avgDim > 1500 && input.hasSmallObjects) {
    reasoning.push(
      "Large images with small objects: Enabled tiling for better detection"
    );
  }

  return reasoning;
}

/**
 * Main function to generate smart defaults
 */
export function generateSmartDefaults(input: SmartDefaultsInput): SmartDefaults {
  return {
    preprocess: recommendPreprocess(input),
    offlineAug: recommendOfflineAug(input),
    onlineAug: recommendOnlineAug(input),
    model: recommendModel(input),
    hyperparams: recommendHyperparams(input),
    reasoning: generateReasoning(input),
  };
}

/**
 * Analyze dataset to get SmartDefaultsInput
 */
export function analyzeDatasetForDefaults(dataset: {
  imageCount: number;
  annotatedImageCount: number;
  annotationCount: number;
  classCount: number;
  avgImageSize?: { width: number; height: number };
  avgAnnotationsPerImage?: number;
  minBboxArea?: number;
}): SmartDefaultsInput {
  const avgAnnotationsPerImage =
    dataset.avgAnnotationsPerImage ??
    (dataset.annotatedImageCount > 0
      ? dataset.annotationCount / dataset.annotatedImageCount
      : 0);

  return {
    datasetSize: dataset.annotatedImageCount || dataset.imageCount,
    annotationCount: dataset.annotationCount,
    classCount: dataset.classCount || 1,
    avgImageSize: dataset.avgImageSize || { width: 800, height: 600 },
    avgAnnotationsPerImage,
    hasSmallObjects: (dataset.minBboxArea ?? 1000) < 500, // Rough heuristic
    hasDenseAnnotations: avgAnnotationsPerImage > 20,
  };
}

/**
 * Get preset description for UI
 */
export function getPresetRecommendation(
  datasetSize: number
): { preset: AugmentationPreset; reason: string } {
  const sizeCategory = getDatasetSizeCategory(datasetSize);

  switch (sizeCategory) {
    case "tiny":
      return {
        preset: "heavy",
        reason: "Maximum augmentation recommended for very small datasets",
      };
    case "small":
      return {
        preset: "heavy",
        reason: "Heavy augmentation to prevent overfitting on small dataset",
      };
    case "medium":
      return {
        preset: "sota-v2",
        reason: "SOTA-v2 provides optimal accuracy-speed balance",
      };
    case "large":
      return {
        preset: "sota-v2",
        reason: "SOTA-v2 for best accuracy with modern techniques",
      };
    case "huge":
      return {
        preset: "medium",
        reason: "Dataset is already diverse, lighter augmentation is sufficient",
      };
  }
}

/**
 * Get batch size recommendation based on model and GPU memory
 */
export function recommendBatchSize(
  modelType: "rt-detr" | "d-fine",
  modelSize: "s" | "m" | "l" | "x",
  imageSize: number,
  gpuMemoryGB: number = 16
): number {
  // Rough VRAM estimates (GB) per batch of 1
  const vramPerBatch: Record<string, Record<string, number>> = {
    "rt-detr": { s: 0.5, m: 0.8, l: 1.2 },
    "d-fine": { s: 0.6, m: 1.0, l: 1.5, x: 2.0 },
  };

  const baseVram = vramPerBatch[modelType]?.[modelSize] ?? 1.0;

  // Adjust for image size (640 is baseline)
  const sizeMultiplier = (imageSize / 640) ** 2;
  const adjustedVram = baseVram * sizeMultiplier;

  // Leave 2GB headroom for other operations
  const availableVram = gpuMemoryGB - 2;

  // Calculate max batch size
  const maxBatch = Math.floor(availableVram / adjustedVram);

  // Round to nearest power of 2 or common batch size
  const commonSizes = [4, 8, 12, 16, 24, 32, 48, 64];
  return commonSizes.reduce((prev, curr) =>
    curr <= maxBatch && curr > prev ? curr : prev
  , 4);
}
