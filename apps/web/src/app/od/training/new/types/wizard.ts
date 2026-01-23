/**
 * Training Wizard Types
 *
 * Complete type definitions for the 7-step training wizard.
 */

// ============================================================================
// STEP 1: Dataset Types
// ============================================================================

export interface DatasetStepData {
  datasetId: string;
  versionId?: string;
  trainSplit: number; // 0.7 - 0.9
  valSplit: number;   // 0.1 - 0.3
  testSplit: number;  // 0 - 0.1
  seed: number;
}

export interface DatasetInfo {
  id: string;
  name: string;
  imageCount: number;
  annotatedImageCount: number;
  annotationCount: number;
  classNames: string[];
  classDistribution: Record<string, number>;
  avgAnnotationsPerImage: number;
  minImageSize: { width: number; height: number };
  maxImageSize: { width: number; height: number };
  avgImageSize: { width: number; height: number };
}

export interface DatasetVersion {
  id: string;
  versionNumber: number;
  name: string | null;
  imageCount: number;
  createdAt: string;
}

// ============================================================================
// STEP 2: Preprocessing Types
// ============================================================================

export type ResizeStrategy =
  | "stretch"      // Stretch to target size (may distort)
  | "letterbox"    // Maintain aspect ratio with padding
  | "crop"         // Center crop to target size
  | "fit"          // Fit within target, no padding
  | "tile";        // Split large images into tiles

export interface TilingConfig {
  enabled: boolean;
  tileSize: number;        // e.g., 640
  overlap: number;         // 0.0 - 0.5
  minObjectArea: number;   // Minimum object area to keep (0.0 - 1.0)
}

export interface PreprocessStepData {
  targetSize: number;      // 416, 512, 640, 800, 1024
  resizeStrategy: ResizeStrategy;
  tiling: TilingConfig;
  autoOrient: boolean;     // Fix EXIF orientation
  normalizePixels: boolean; // 0-255 to 0-1
}

// ============================================================================
// STEP 3: Offline Augmentation Types
// ============================================================================

export type OfflineMultiplier = 1 | 2 | 3 | 5 | 10;

export interface OfflineAugmentationConfig {
  enabled: boolean;
  probability: number;
  // Additional params per augmentation type
  [key: string]: unknown;
}

export interface OfflineAugStepData {
  enabled: boolean;
  multiplier: OfflineMultiplier;
  augmentations: {
    // Geometric
    horizontalFlip: OfflineAugmentationConfig;
    verticalFlip: OfflineAugmentationConfig;
    rotate90: OfflineAugmentationConfig;
    randomRotate: OfflineAugmentationConfig & { limit: number };
    // Color
    brightnessContrast: OfflineAugmentationConfig & { brightnessLimit: number; contrastLimit: number };
    hueSaturation: OfflineAugmentationConfig & { hueShift: number; satShift: number; valShift: number };
    // Noise
    gaussianNoise: OfflineAugmentationConfig & { varLimit: number };
    // Blur
    gaussianBlur: OfflineAugmentationConfig & { blurLimit: number };
  };
}

// ============================================================================
// STEP 4: Online Augmentation Types
// ============================================================================

export type AugmentationPreset =
  | "sota-v2"   // New: Best of YOLOv8 + RT-DETR + D-FINE
  | "sota"      // Legacy: Mosaic + MixUp + CopyPaste
  | "heavy"     // All augmentations
  | "medium"    // Balanced
  | "light"     // Basic only
  | "none"      // No augmentation
  | "custom";   // User-defined

export interface AugmentationCategory {
  id: string;
  name: string;
  description: string;
  augmentations: string[];
}

// Individual augmentation config with full params
export interface AugmentationConfig {
  enabled: boolean;
  probability: number;
  params: Record<string, number | boolean | string>;
}

// Multi-image augmentations
export interface MultiImageAugmentations {
  mosaic: AugmentationConfig & { params: { gridSize: 2 | 3 } };
  mosaic9: AugmentationConfig;
  mixup: AugmentationConfig & { params: { alpha: number } };
  cutmix: AugmentationConfig & { params: { alpha: number } };
  copypaste: AugmentationConfig & { params: { maxObjects: number; scaleRange: [number, number] } };
}

// Geometric augmentations
export interface GeometricAugmentations {
  horizontalFlip: AugmentationConfig;
  verticalFlip: AugmentationConfig;
  rotate90: AugmentationConfig;
  randomRotate: AugmentationConfig & { params: { limit: number; borderMode: string } };
  shiftScaleRotate: AugmentationConfig & { params: { shiftLimit: number; scaleLimit: number; rotateLimit: number } };
  affine: AugmentationConfig & { params: { scale: [number, number]; translate: number; rotate: number; shear: number } };
  perspective: AugmentationConfig & { params: { scale: number } };
  safeRotate: AugmentationConfig & { params: { limit: number } };
  randomCrop: AugmentationConfig & { params: { scale: [number, number] } };
  randomScale: AugmentationConfig & { params: { scaleLimit: number } };
  gridDistortion: AugmentationConfig & { params: { numSteps: number; distortLimit: number } };
  elasticTransform: AugmentationConfig & { params: { alpha: number; sigma: number } };
  opticalDistortion: AugmentationConfig & { params: { distortLimit: number; shiftLimit: number } };
  piecewiseAffine: AugmentationConfig & { params: { scale: number; nbRows: number; nbCols: number } };
}

// Color augmentations
export interface ColorAugmentations {
  brightnessContrast: AugmentationConfig & { params: { brightnessLimit: number; contrastLimit: number } };
  hueSaturation: AugmentationConfig & { params: { hueShift: number; satShift: number; valShift: number } };
  rgbShift: AugmentationConfig & { params: { rShift: number; gShift: number; bShift: number } };
  channelShuffle: AugmentationConfig;
  clahe: AugmentationConfig & { params: { clipLimit: number; tileGridSize: number } };
  equalize: AugmentationConfig;
  toGray: AugmentationConfig;
  randomGamma: AugmentationConfig & { params: { gammaLimit: [number, number] } };
  randomToneCurve: AugmentationConfig & { params: { scale: number } };
  posterize: AugmentationConfig & { params: { numBits: number } };
  solarize: AugmentationConfig & { params: { threshold: number } };
  sharpen: AugmentationConfig & { params: { alpha: [number, number]; lightness: [number, number] } };
  unsharpMask: AugmentationConfig & { params: { blurLimit: [number, number]; alpha: [number, number] } };
  fancyPCA: AugmentationConfig & { params: { alpha: number } };
  invertImg: AugmentationConfig;
  colorJitter: AugmentationConfig & { params: { brightness: number; contrast: number; saturation: number; hue: number } };
}

// Blur augmentations
export interface BlurAugmentations {
  gaussianBlur: AugmentationConfig & { params: { blurLimit: [number, number] } };
  motionBlur: AugmentationConfig & { params: { blurLimit: number } };
  medianBlur: AugmentationConfig & { params: { blurLimit: number } };
  defocus: AugmentationConfig & { params: { radius: [number, number]; aliasBlur: [number, number] } };
  zoomBlur: AugmentationConfig & { params: { maxFactor: number } };
  glassBlur: AugmentationConfig & { params: { sigma: number; maxDelta: number; iterations: number } };
  advancedBlur: AugmentationConfig & { params: { blurLimit: [number, number]; noiseLimit: [number, number] } };
}

// Noise augmentations
export interface NoiseAugmentations {
  gaussianNoise: AugmentationConfig & { params: { varLimit: [number, number] } };
  isoNoise: AugmentationConfig & { params: { colorShift: [number, number]; intensity: [number, number] } };
  multiplicativeNoise: AugmentationConfig & { params: { multiplier: [number, number] } };
}

// Quality augmentations
export interface QualityAugmentations {
  imageCompression: AugmentationConfig & { params: { qualityLower: number; qualityUpper: number } };
  downscale: AugmentationConfig & { params: { scaleMin: number; scaleMax: number } };
}

// Dropout augmentations
export interface DropoutAugmentations {
  coarseDropout: AugmentationConfig & { params: { maxHoles: number; maxHeight: number; maxWidth: number; fillValue: number } };
  gridDropout: AugmentationConfig & { params: { ratio: number; unit_size_min: number; unit_size_max: number } };
  pixelDropout: AugmentationConfig & { params: { dropoutProb: number } };
  maskDropout: AugmentationConfig & { params: { maxObjects: number } };
}

// Weather augmentations
export interface WeatherAugmentations {
  randomRain: AugmentationConfig & { params: { slantLower: number; slantUpper: number; dropLength: number; dropWidth: number; dropColor: [number, number, number]; blurValue: number } };
  randomFog: AugmentationConfig & { params: { fogCoefLower: number; fogCoefUpper: number; alphaCoef: number } };
  randomShadow: AugmentationConfig & { params: { numShadowsLower: number; numShadowsUpper: number; shadowDimension: number; shadowRoi: [number, number, number, number] } };
  randomSunFlare: AugmentationConfig & { params: { flareRoi: [number, number, number, number]; angleLower: number; angleUpper: number; numFlareCirclesLower: number; numFlareCirclesUpper: number; srcRadius: number } };
  randomSnow: AugmentationConfig & { params: { snowPointLower: number; snowPointUpper: number; brightnessCoeff: number } };
  spatter: AugmentationConfig & { params: { mode: "rain" | "mud" } };
  plasmaBrightness: AugmentationConfig & { params: { roughness: number } };
}

// Complete online augmentation config
export interface OnlineAugStepData {
  preset: AugmentationPreset;
  // Only used when preset === "custom"
  customConfig?: {
    multiImage: Partial<MultiImageAugmentations>;
    geometric: Partial<GeometricAugmentations>;
    color: Partial<ColorAugmentations>;
    blur: Partial<BlurAugmentations>;
    noise: Partial<NoiseAugmentations>;
    quality: Partial<QualityAugmentations>;
    dropout: Partial<DropoutAugmentations>;
    weather: Partial<WeatherAugmentations>;
  };
}

// ============================================================================
// STEP 5: Model Types
// ============================================================================

export type ModelType = "rt-detr" | "d-fine";
export type ModelSize = "s" | "m" | "l" | "x";

export interface ModelInfo {
  type: ModelType;
  size: ModelSize;
  name: string;
  description: string;
  params: string;        // e.g., "32M"
  speed: string;         // e.g., "45 FPS"
  accuracy: string;      // e.g., "52.3 mAP"
  vramRequired: string;  // e.g., "8GB"
}

export interface ModelStepData {
  modelType: ModelType;
  modelSize: ModelSize;
  pretrained: boolean;
  freezeBackbone: boolean;
  freezeEpochs: number;  // Epochs to keep backbone frozen
}

// ============================================================================
// STEP 6: Hyperparameters Types
// ============================================================================

export type OptimizerType = "adamw" | "sgd" | "adam" | "lion";
export type SchedulerType = "cosine" | "step" | "linear" | "onecycle" | "plateau";

export interface HyperparamsStepData {
  // Basic
  epochs: number;
  batchSize: number;
  accumulationSteps: number;

  // Optimizer
  optimizer: OptimizerType;
  learningRate: number;
  weightDecay: number;
  momentum: number;  // For SGD

  // Scheduler
  scheduler: SchedulerType;
  warmupEpochs: number;
  minLrRatio: number;

  // SOTA Features
  useEma: boolean;
  emaDecay: number;
  emaWarmupSteps: number;

  useLlrd: boolean;
  llrdDecay: number;
  headLrFactor: number;

  useMixedPrecision: boolean;
  gradientClip: number;
  labelSmoothing: number;

  // Early Stopping
  patience: number;
  minDelta: number;

  // Multi-scale
  multiScale: boolean;
  multiScaleRange: [number, number];
}

// ============================================================================
// STEP 7: Review Types (Computed from other steps)
// ============================================================================

export interface TrainingEstimate {
  estimatedEpochTime: string;     // e.g., "2m 30s"
  estimatedTotalTime: string;     // e.g., "4h 10m"
  estimatedVramUsage: string;     // e.g., "12GB"
  estimatedDiskUsage: string;     // e.g., "2.5GB"
  warnings: string[];
  recommendations: string[];
}

export interface ReviewStepData {
  name: string;
  description: string;
  tags: string[];
}

// ============================================================================
// Wizard State Types
// ============================================================================

export type WizardStep =
  | "dataset"
  | "preprocess"
  | "offline-aug"
  | "online-aug"
  | "model"
  | "hyperparams"
  | "review";

export const WIZARD_STEPS: WizardStep[] = [
  "dataset",
  "preprocess",
  "offline-aug",
  "online-aug",
  "model",
  "hyperparams",
  "review",
];

export const STEP_TITLES: Record<WizardStep, string> = {
  "dataset": "Dataset",
  "preprocess": "Preprocessing",
  "offline-aug": "Offline Augmentation",
  "online-aug": "Online Augmentation",
  "model": "Model",
  "hyperparams": "Hyperparameters",
  "review": "Review & Start",
};

export const STEP_DESCRIPTIONS: Record<WizardStep, string> = {
  "dataset": "Select your dataset and configure train/val split",
  "preprocess": "Configure image preprocessing and resizing",
  "offline-aug": "Generate augmented copies before training",
  "online-aug": "Configure real-time augmentations during training",
  "model": "Choose model architecture and size",
  "hyperparams": "Fine-tune training parameters",
  "review": "Review configuration and start training",
};

export interface WizardState {
  currentStep: WizardStep;
  completedSteps: Set<WizardStep>;

  // Step data
  dataset: DatasetStepData;
  preprocess: PreprocessStepData;
  offlineAug: OfflineAugStepData;
  onlineAug: OnlineAugStepData;
  model: ModelStepData;
  hyperparams: HyperparamsStepData;
  review: ReviewStepData;

  // Metadata
  datasetInfo: DatasetInfo | null;
  isSubmitting: boolean;
  errors: Partial<Record<WizardStep, string[]>>;
}

// ============================================================================
// Smart Defaults Types
// ============================================================================

export interface SmartDefaultsInput {
  datasetSize: number;
  annotationCount: number;
  classCount: number;
  avgImageSize: { width: number; height: number };
  avgAnnotationsPerImage: number;
  hasSmallObjects: boolean;
  hasDenseAnnotations: boolean;
}

export interface SmartDefaults {
  preprocess: Partial<PreprocessStepData>;
  offlineAug: Partial<OfflineAugStepData>;
  onlineAug: Partial<OnlineAugStepData>;
  model: Partial<ModelStepData>;
  hyperparams: Partial<HyperparamsStepData>;
  reasoning: string[];
}

// ============================================================================
// API Types
// ============================================================================

export interface CreateTrainingRequest {
  name: string;
  description?: string;
  tags?: string[];
  dataset_id: string;
  dataset_version_id?: string;
  model_type: ModelType;
  model_size: ModelSize;
  config: {
    // Preprocessing
    image_size: number;
    resize_strategy: ResizeStrategy;
    tiling?: TilingConfig;

    // Training
    epochs: number;
    batch_size: number;
    accumulation_steps: number;
    learning_rate: number;
    weight_decay: number;

    // Optimizer
    optimizer: OptimizerType;

    // Scheduler
    scheduler: SchedulerType;
    warmup_epochs: number;
    min_lr_ratio: number;

    // SOTA
    use_ema: boolean;
    ema_decay: number;
    ema_warmup_steps: number;
    mixed_precision: boolean;
    llrd_decay: number;
    head_lr_factor: number;
    gradient_clip: number;
    label_smoothing: number;

    // Early stopping
    patience: number;

    // Augmentation
    augmentation_preset: AugmentationPreset;
    augmentation_config?: Record<string, unknown>;

    // Offline augmentation
    offline_augmentation?: {
      enabled: boolean;
      multiplier: OfflineMultiplier;
      config: Record<string, unknown>;
    };

    // Train/Val split
    train_split: number;
    val_split: number;
    seed: number;

    // Model
    pretrained: boolean;
    freeze_backbone: boolean;
    freeze_epochs: number;

    // Multi-scale
    multi_scale: boolean;
    multi_scale_range?: [number, number];
  };
}

// ============================================================================
// Default Values
// ============================================================================

export const DEFAULT_DATASET_STEP: DatasetStepData = {
  datasetId: "",
  versionId: undefined,
  trainSplit: 0.8,
  valSplit: 0.15,
  testSplit: 0.05,
  seed: 42,
};

export const DEFAULT_PREPROCESS_STEP: PreprocessStepData = {
  targetSize: 640,
  resizeStrategy: "letterbox",
  tiling: {
    enabled: false,
    tileSize: 640,
    overlap: 0.2,
    minObjectArea: 0.1,
  },
  autoOrient: true,
  normalizePixels: true,
};

export const DEFAULT_OFFLINE_AUG_STEP: OfflineAugStepData = {
  enabled: false,
  multiplier: 2,
  augmentations: {
    horizontalFlip: { enabled: true, probability: 0.5 },
    verticalFlip: { enabled: false, probability: 0.5 },
    rotate90: { enabled: true, probability: 0.5 },
    randomRotate: { enabled: true, probability: 0.5, limit: 15 },
    brightnessContrast: { enabled: true, probability: 0.5, brightnessLimit: 0.2, contrastLimit: 0.2 },
    hueSaturation: { enabled: true, probability: 0.5, hueShift: 20, satShift: 30, valShift: 20 },
    gaussianNoise: { enabled: false, probability: 0.3, varLimit: 25 },
    gaussianBlur: { enabled: false, probability: 0.3, blurLimit: 5 },
  },
};

export const DEFAULT_ONLINE_AUG_STEP: OnlineAugStepData = {
  preset: "sota-v2",
  customConfig: undefined,
};

export const DEFAULT_MODEL_STEP: ModelStepData = {
  modelType: "rt-detr",
  modelSize: "l",
  pretrained: true,
  freezeBackbone: false,
  freezeEpochs: 0,
};

export const DEFAULT_HYPERPARAMS_STEP: HyperparamsStepData = {
  epochs: 100,
  batchSize: 16,
  accumulationSteps: 1,
  optimizer: "adamw",
  learningRate: 0.0001,
  weightDecay: 0.0001,
  momentum: 0.9,
  scheduler: "cosine",
  warmupEpochs: 3,
  minLrRatio: 0.01,
  useEma: true,
  emaDecay: 0.9999,
  emaWarmupSteps: 2000,
  useLlrd: true,
  llrdDecay: 0.9,
  headLrFactor: 10.0,
  useMixedPrecision: true,
  gradientClip: 1.0,
  labelSmoothing: 0.0,
  patience: 20,
  minDelta: 0.001,
  multiScale: false,
  multiScaleRange: [0.5, 1.5],
};

export const DEFAULT_REVIEW_STEP: ReviewStepData = {
  name: "",
  description: "",
  tags: [],
};

export const DEFAULT_WIZARD_STATE: Omit<WizardState, "completedSteps"> & { completedSteps: WizardStep[] } = {
  currentStep: "dataset",
  completedSteps: [],
  dataset: DEFAULT_DATASET_STEP,
  preprocess: DEFAULT_PREPROCESS_STEP,
  offlineAug: DEFAULT_OFFLINE_AUG_STEP,
  onlineAug: DEFAULT_ONLINE_AUG_STEP,
  model: DEFAULT_MODEL_STEP,
  hyperparams: DEFAULT_HYPERPARAMS_STEP,
  review: DEFAULT_REVIEW_STEP,
  datasetInfo: null,
  isSubmitting: false,
  errors: {},
};
