/**
 * Wizard Validation Utility
 *
 * Validates wizard step data and provides user-friendly error messages.
 */

import type {
  WizardStep,
  WizardState,
  DatasetStepData,
  PreprocessStepData,
  OfflineAugStepData,
  OnlineAugStepData,
  ModelStepData,
  HyperparamsStepData,
  ReviewStepData,
  DatasetInfo,
} from "../types/wizard";

export interface ValidationResult {
  isValid: boolean;
  errors: string[];
  warnings: string[];
}

/**
 * Validate dataset step
 */
export function validateDatasetStep(
  data: DatasetStepData,
  datasetInfo: DatasetInfo | null
): ValidationResult {
  const errors: string[] = [];
  const warnings: string[] = [];

  // Required fields
  if (!data.datasetId) {
    errors.push("Please select a dataset");
  }

  // Dataset info checks
  if (datasetInfo) {
    if (datasetInfo.annotationCount === 0) {
      errors.push("Selected dataset has no annotations. Please annotate images first.");
    }

    if (datasetInfo.annotatedImageCount < 10) {
      warnings.push(
        `Dataset has only ${datasetInfo.annotatedImageCount} annotated images. ` +
        "Consider adding more data for better results."
      );
    }

    if (datasetInfo.annotatedImageCount < datasetInfo.imageCount) {
      const unannotated = datasetInfo.imageCount - datasetInfo.annotatedImageCount;
      warnings.push(
        `${unannotated} images are not annotated and won't be used for training.`
      );
    }

    // Class distribution check
    if (datasetInfo.classDistribution) {
      const counts = Object.values(datasetInfo.classDistribution);
      const minCount = Math.min(...counts);
      const maxCount = Math.max(...counts);
      if (maxCount > minCount * 10) {
        warnings.push(
          "Class imbalance detected. Some classes have significantly fewer examples."
        );
      }
    }
  }

  // Split validation
  const totalSplit = data.trainSplit + data.valSplit + data.testSplit;
  if (Math.abs(totalSplit - 1.0) > 0.001) {
    errors.push("Train + Val + Test splits must equal 100%");
  }

  if (data.trainSplit < 0.5) {
    warnings.push("Training split is below 50%. Consider increasing for better model training.");
  }

  if (data.valSplit < 0.1) {
    warnings.push("Validation split is below 10%. May not accurately evaluate model performance.");
  }

  return {
    isValid: errors.length === 0,
    errors,
    warnings,
  };
}

/**
 * Validate preprocessing step
 */
export function validatePreprocessStep(data: PreprocessStepData): ValidationResult {
  const errors: string[] = [];
  const warnings: string[] = [];

  // Target size validation
  if (data.targetSize < 320) {
    errors.push("Target size must be at least 320 pixels");
  }

  if (data.targetSize > 1280) {
    warnings.push("Large image sizes (>1280) require significant GPU memory");
  }

  // Tiling validation
  if (data.tiling.enabled) {
    if (data.tiling.tileSize < 320) {
      errors.push("Tile size must be at least 320 pixels");
    }

    if (data.tiling.overlap < 0 || data.tiling.overlap > 0.5) {
      errors.push("Tile overlap must be between 0% and 50%");
    }

    if (data.tiling.minObjectArea < 0 || data.tiling.minObjectArea > 1) {
      errors.push("Minimum object area must be between 0% and 100%");
    }
  }

  return {
    isValid: errors.length === 0,
    errors,
    warnings,
  };
}

/**
 * Validate offline augmentation step
 */
export function validateOfflineAugStep(
  data: OfflineAugStepData,
  datasetInfo: DatasetInfo | null
): ValidationResult {
  const errors: string[] = [];
  const warnings: string[] = [];

  if (data.enabled) {
    // Check multiplier
    if (![1, 2, 3, 5, 10].includes(data.multiplier)) {
      errors.push("Invalid multiplier. Choose from 1x, 2x, 3x, 5x, or 10x");
    }

    // Check if any augmentations are enabled
    const enabledAugs = Object.values(data.augmentations).filter(
      (aug) => aug.enabled
    );
    if (enabledAugs.length === 0) {
      errors.push("Enable at least one offline augmentation");
    }

    // Check resulting dataset size
    if (datasetInfo) {
      const resultingSize = datasetInfo.annotatedImageCount * data.multiplier;
      if (resultingSize > 50000) {
        warnings.push(
          `Offline augmentation will create ${resultingSize.toLocaleString()} images. ` +
          "This may significantly increase storage and preprocessing time."
        );
      }
    }

    // Probability validation
    for (const [name, config] of Object.entries(data.augmentations)) {
      if (config.enabled && (config.probability < 0 || config.probability > 1)) {
        errors.push(`${name}: probability must be between 0 and 1`);
      }
    }
  }

  return {
    isValid: errors.length === 0,
    errors,
    warnings,
  };
}

/**
 * Validate online augmentation step
 */
export function validateOnlineAugStep(data: OnlineAugStepData): ValidationResult {
  const errors: string[] = [];
  const warnings: string[] = [];

  const validPresets = ["sota-v2", "sota", "heavy", "medium", "light", "none", "custom"];
  if (!validPresets.includes(data.preset)) {
    errors.push("Invalid augmentation preset");
  }

  // Custom config validation
  if (data.preset === "custom" && data.customConfig) {
    // Check if at least some augmentations are enabled
    const allConfigs = [
      ...Object.values(data.customConfig.multiImage || {}),
      ...Object.values(data.customConfig.geometric || {}),
      ...Object.values(data.customConfig.color || {}),
      ...Object.values(data.customConfig.blur || {}),
      ...Object.values(data.customConfig.noise || {}),
      ...Object.values(data.customConfig.quality || {}),
      ...Object.values(data.customConfig.dropout || {}),
      ...Object.values(data.customConfig.weather || {}),
    ];

    const enabledCount = allConfigs.filter((c) => c?.enabled).length;
    if (enabledCount === 0) {
      warnings.push("No augmentations enabled. Consider enabling some for better generalization.");
    }
  }

  return {
    isValid: errors.length === 0,
    errors,
    warnings,
  };
}

/**
 * Validate model step
 */
export function validateModelStep(data: ModelStepData): ValidationResult {
  const errors: string[] = [];
  const warnings: string[] = [];

  // Model type validation
  const validTypes = ["rt-detr", "d-fine"];
  if (!validTypes.includes(data.modelType)) {
    errors.push("Invalid model type");
  }

  // Model size validation
  const validSizes: Record<string, string[]> = {
    "rt-detr": ["s", "m", "l"],
    "d-fine": ["s", "m", "l", "x"],
  };

  if (!validSizes[data.modelType]?.includes(data.modelSize)) {
    errors.push(`Invalid model size for ${data.modelType}`);
  }

  // Freeze validation
  if (data.freezeBackbone && data.freezeEpochs <= 0) {
    warnings.push("Freeze epochs should be > 0 when freezing backbone");
  }

  if (data.freezeEpochs > 50) {
    warnings.push("Freezing backbone for more than 50 epochs may limit model adaptation");
  }

  return {
    isValid: errors.length === 0,
    errors,
    warnings,
  };
}

/**
 * Validate hyperparameters step
 */
export function validateHyperparamsStep(
  data: HyperparamsStepData,
  datasetInfo: DatasetInfo | null
): ValidationResult {
  const errors: string[] = [];
  const warnings: string[] = [];

  // Epochs
  if (data.epochs < 1) {
    errors.push("Epochs must be at least 1");
  }
  if (data.epochs > 500) {
    warnings.push("Training for more than 500 epochs may lead to overfitting");
  }

  // Batch size
  if (data.batchSize < 1) {
    errors.push("Batch size must be at least 1");
  }
  if (data.batchSize > 128) {
    warnings.push("Very large batch sizes may require significant GPU memory");
  }

  // Learning rate
  if (data.learningRate <= 0) {
    errors.push("Learning rate must be positive");
  }
  if (data.learningRate > 0.01) {
    warnings.push("Learning rate > 0.01 may cause training instability");
  }

  // Weight decay
  if (data.weightDecay < 0) {
    errors.push("Weight decay cannot be negative");
  }

  // Warmup epochs
  if (data.warmupEpochs < 0) {
    errors.push("Warmup epochs cannot be negative");
  }
  if (data.warmupEpochs >= data.epochs) {
    errors.push("Warmup epochs must be less than total epochs");
  }

  // Patience
  if (data.patience < 1) {
    errors.push("Patience must be at least 1");
  }
  if (data.patience > data.epochs / 2) {
    warnings.push("Patience is more than half the epochs. Early stopping may not trigger.");
  }

  // EMA
  if (data.useEma) {
    if (data.emaDecay < 0.9 || data.emaDecay > 0.9999) {
      warnings.push("EMA decay is typically between 0.9 and 0.9999");
    }
  }

  // LLRD
  if (data.useLlrd) {
    if (data.llrdDecay < 0.5 || data.llrdDecay > 1.0) {
      warnings.push("LLRD decay is typically between 0.5 and 1.0");
    }
  }

  // Dataset-specific warnings
  if (datasetInfo) {
    const stepsPerEpoch = Math.ceil(datasetInfo.annotatedImageCount / data.batchSize);
    if (stepsPerEpoch < 10) {
      warnings.push(
        `Only ${stepsPerEpoch} steps per epoch. Consider reducing batch size.`
      );
    }
  }

  return {
    isValid: errors.length === 0,
    errors,
    warnings,
  };
}

/**
 * Validate review step
 */
export function validateReviewStep(data: ReviewStepData): ValidationResult {
  const errors: string[] = [];
  const warnings: string[] = [];

  // Name is required
  if (!data.name || data.name.trim().length === 0) {
    errors.push("Training name is required");
  }

  if (data.name && data.name.length > 100) {
    errors.push("Training name must be 100 characters or less");
  }

  // Description length
  if (data.description && data.description.length > 500) {
    warnings.push("Description is quite long. Consider shortening it.");
  }

  // Tags validation
  if (data.tags && data.tags.length > 10) {
    warnings.push("Too many tags. Consider using fewer, more relevant tags.");
  }

  return {
    isValid: errors.length === 0,
    errors,
    warnings,
  };
}

/**
 * Validate a specific step
 */
export function validateStep(
  step: WizardStep,
  state: WizardState
): ValidationResult {
  switch (step) {
    case "dataset":
      return validateDatasetStep(state.dataset, state.datasetInfo);
    case "preprocess":
      return validatePreprocessStep(state.preprocess);
    case "offline-aug":
      return validateOfflineAugStep(state.offlineAug, state.datasetInfo);
    case "online-aug":
      return validateOnlineAugStep(state.onlineAug);
    case "model":
      return validateModelStep(state.model);
    case "hyperparams":
      return validateHyperparamsStep(state.hyperparams, state.datasetInfo);
    case "review":
      return validateReviewStep(state.review);
    default:
      return { isValid: true, errors: [], warnings: [] };
  }
}

/**
 * Validate all steps
 */
export function validateAllSteps(state: WizardState): Record<WizardStep, ValidationResult> {
  return {
    dataset: validateDatasetStep(state.dataset, state.datasetInfo),
    preprocess: validatePreprocessStep(state.preprocess),
    "offline-aug": validateOfflineAugStep(state.offlineAug, state.datasetInfo),
    "online-aug": validateOnlineAugStep(state.onlineAug),
    model: validateModelStep(state.model),
    hyperparams: validateHyperparamsStep(state.hyperparams, state.datasetInfo),
    review: validateReviewStep(state.review),
  };
}

/**
 * Check if a step can be navigated to (all previous steps valid)
 */
export function canNavigateToStep(
  targetStep: WizardStep,
  state: WizardState,
  steps: WizardStep[]
): boolean {
  const targetIndex = steps.indexOf(targetStep);

  // Can always go to first step
  if (targetIndex === 0) return true;

  // Can go back to completed steps
  if (state.completedSteps.has(targetStep)) return true;

  // Can only go forward if all previous steps are valid
  for (let i = 0; i < targetIndex; i++) {
    const stepName = steps[i];
    const validation = validateStep(stepName, state);
    if (!validation.isValid) {
      return false;
    }
  }

  return true;
}

/**
 * Get first invalid step (for navigation)
 */
export function getFirstInvalidStep(
  state: WizardState,
  steps: WizardStep[]
): WizardStep | null {
  for (const step of steps) {
    const validation = validateStep(step, state);
    if (!validation.isValid) {
      return step;
    }
  }
  return null;
}
