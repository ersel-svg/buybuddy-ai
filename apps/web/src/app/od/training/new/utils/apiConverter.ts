/**
 * API Converter Utility
 *
 * Converts wizard state to API request format.
 */

import type {
  WizardState,
  CreateTrainingRequest,
  AugmentationPreset,
} from "../types/wizard";

/**
 * Convert wizard state to API request format
 */
export function convertWizardStateToApiRequest(
  state: WizardState
): CreateTrainingRequest {
  const {
    dataset,
    preprocess,
    offlineAug,
    onlineAug,
    model,
    hyperparams,
    review,
  } = state;

  // Build augmentation config for custom presets
  let augmentationConfig: Record<string, unknown> | undefined;

  if (onlineAug.preset === "custom" && onlineAug.customConfig) {
    augmentationConfig = convertCustomAugConfig(onlineAug.customConfig);
  }

  // Build offline augmentation config
  let offlineAugmentation: CreateTrainingRequest["config"]["offline_augmentation"];

  if (offlineAug.enabled) {
    offlineAugmentation = {
      enabled: true,
      multiplier: offlineAug.multiplier,
      config: convertOfflineAugConfig(offlineAug),
    };
  }

  return {
    name: review.name,
    description: review.description || undefined,
    tags: review.tags.length > 0 ? review.tags : undefined,
    dataset_id: dataset.datasetId,
    dataset_version_id: dataset.versionId || undefined,
    model_type: model.modelType,
    model_size: model.modelSize,
    config: {
      // Preprocessing
      image_size: preprocess.targetSize,
      resize_strategy: preprocess.resizeStrategy,
      tiling: preprocess.tiling.enabled ? preprocess.tiling : undefined,

      // Training
      epochs: hyperparams.epochs,
      batch_size: hyperparams.batchSize,
      accumulation_steps: hyperparams.accumulationSteps,
      learning_rate: hyperparams.learningRate,
      weight_decay: hyperparams.weightDecay,

      // Scheduler
      scheduler: hyperparams.scheduler,
      warmup_epochs: hyperparams.warmupEpochs,
      min_lr_ratio: hyperparams.minLrRatio,

      // SOTA features
      use_ema: hyperparams.useEma,
      ema_decay: hyperparams.emaDecay,
      mixed_precision: hyperparams.useMixedPrecision,
      llrd_decay: hyperparams.useLlrd ? hyperparams.llrdDecay : 1.0,
      head_lr_factor: hyperparams.headLrFactor,
      gradient_clip: hyperparams.gradientClip,
      label_smoothing: hyperparams.labelSmoothing,

      // Early stopping
      patience: hyperparams.patience,

      // Augmentation
      augmentation_preset: onlineAug.preset,
      augmentation_config: augmentationConfig,

      // Offline augmentation
      offline_augmentation: offlineAugmentation,

      // Train/Val split
      train_split: dataset.trainSplit,
      val_split: dataset.valSplit,
      seed: dataset.seed,

      // Model
      pretrained: model.pretrained,
      freeze_backbone: model.freezeBackbone,
      freeze_epochs: model.freezeEpochs,

      // Multi-scale
      multi_scale: hyperparams.multiScale,
      multi_scale_range: hyperparams.multiScale
        ? hyperparams.multiScaleRange
        : undefined,
    },
  };
}

/**
 * Convert custom augmentation config to API format
 */
function convertCustomAugConfig(
  customConfig: NonNullable<WizardState["onlineAug"]["customConfig"]>
): Record<string, unknown> {
  const result: Record<string, unknown> = {};

  // Helper to convert aug config
  const convertAug = (
    key: string,
    config: { enabled: boolean; probability: number; params?: Record<string, unknown> } | undefined
  ) => {
    if (config?.enabled) {
      result[key] = {
        enabled: true,
        probability: config.probability,
        ...config.params,
      };
    }
  };

  // Multi-image
  if (customConfig.multiImage) {
    convertAug("mosaic", customConfig.multiImage.mosaic);
    convertAug("mosaic9", customConfig.multiImage.mosaic9);
    convertAug("mixup", customConfig.multiImage.mixup);
    convertAug("cutmix", customConfig.multiImage.cutmix);
    convertAug("copypaste", customConfig.multiImage.copypaste);
  }

  // Geometric
  if (customConfig.geometric) {
    convertAug("horizontal_flip", customConfig.geometric.horizontalFlip);
    convertAug("vertical_flip", customConfig.geometric.verticalFlip);
    convertAug("rotate90", customConfig.geometric.rotate90);
    convertAug("random_rotate", customConfig.geometric.randomRotate);
    convertAug("shift_scale_rotate", customConfig.geometric.shiftScaleRotate);
    convertAug("affine", customConfig.geometric.affine);
    convertAug("perspective", customConfig.geometric.perspective);
    convertAug("safe_rotate", customConfig.geometric.safeRotate);
    convertAug("random_crop", customConfig.geometric.randomCrop);
    convertAug("random_scale", customConfig.geometric.randomScale);
    convertAug("grid_distortion", customConfig.geometric.gridDistortion);
    convertAug("elastic_transform", customConfig.geometric.elasticTransform);
    convertAug("optical_distortion", customConfig.geometric.opticalDistortion);
    convertAug("piecewise_affine", customConfig.geometric.piecewiseAffine);
  }

  // Color
  if (customConfig.color) {
    convertAug("brightness_contrast", customConfig.color.brightnessContrast);
    convertAug("color_jitter", customConfig.color.colorJitter);
    convertAug("hue_saturation", customConfig.color.hueSaturation);
    convertAug("random_gamma", customConfig.color.randomGamma);
    convertAug("rgb_shift", customConfig.color.rgbShift);
    convertAug("channel_shuffle", customConfig.color.channelShuffle);
    convertAug("clahe", customConfig.color.clahe);
    convertAug("equalize", customConfig.color.equalize);
    convertAug("random_tone_curve", customConfig.color.randomToneCurve);
    convertAug("posterize", customConfig.color.posterize);
    convertAug("solarize", customConfig.color.solarize);
    convertAug("sharpen", customConfig.color.sharpen);
    convertAug("unsharp_mask", customConfig.color.unsharpMask);
    convertAug("fancy_pca", customConfig.color.fancyPCA);
    convertAug("invert_img", customConfig.color.invertImg);
    convertAug("to_gray", customConfig.color.toGray);
  }

  // Blur
  if (customConfig.blur) {
    convertAug("gaussian_blur", customConfig.blur.gaussianBlur);
    convertAug("motion_blur", customConfig.blur.motionBlur);
    convertAug("median_blur", customConfig.blur.medianBlur);
    convertAug("defocus", customConfig.blur.defocus);
    convertAug("zoom_blur", customConfig.blur.zoomBlur);
    convertAug("glass_blur", customConfig.blur.glassBlur);
    convertAug("advanced_blur", customConfig.blur.advancedBlur);
  }

  // Noise
  if (customConfig.noise) {
    convertAug("gaussian_noise", customConfig.noise.gaussianNoise);
    convertAug("iso_noise", customConfig.noise.isoNoise);
    convertAug("multiplicative_noise", customConfig.noise.multiplicativeNoise);
  }

  // Quality
  if (customConfig.quality) {
    convertAug("image_compression", customConfig.quality.imageCompression);
    convertAug("downscale", customConfig.quality.downscale);
  }

  // Dropout
  if (customConfig.dropout) {
    convertAug("coarse_dropout", customConfig.dropout.coarseDropout);
    convertAug("grid_dropout", customConfig.dropout.gridDropout);
    convertAug("pixel_dropout", customConfig.dropout.pixelDropout);
    convertAug("mask_dropout", customConfig.dropout.maskDropout);
  }

  // Weather
  if (customConfig.weather) {
    convertAug("random_rain", customConfig.weather.randomRain);
    convertAug("random_fog", customConfig.weather.randomFog);
    convertAug("random_shadow", customConfig.weather.randomShadow);
    convertAug("random_sun_flare", customConfig.weather.randomSunFlare);
    convertAug("random_snow", customConfig.weather.randomSnow);
    convertAug("spatter", customConfig.weather.spatter);
    convertAug("plasma_brightness", customConfig.weather.plasmaBrightness);
  }

  return result;
}

/**
 * Convert offline augmentation config to API format
 */
function convertOfflineAugConfig(
  offlineAug: WizardState["offlineAug"]
): Record<string, unknown> {
  const result: Record<string, unknown> = {};

  const { augmentations } = offlineAug;

  if (augmentations.horizontalFlip.enabled) {
    result.horizontal_flip = {
      enabled: true,
      probability: augmentations.horizontalFlip.probability,
    };
  }

  if (augmentations.verticalFlip.enabled) {
    result.vertical_flip = {
      enabled: true,
      probability: augmentations.verticalFlip.probability,
    };
  }

  if (augmentations.rotate90.enabled) {
    result.rotate90 = {
      enabled: true,
      probability: augmentations.rotate90.probability,
    };
  }

  if (augmentations.randomRotate.enabled) {
    result.random_rotate = {
      enabled: true,
      probability: augmentations.randomRotate.probability,
      limit: augmentations.randomRotate.limit,
    };
  }

  if (augmentations.brightnessContrast.enabled) {
    result.brightness_contrast = {
      enabled: true,
      probability: augmentations.brightnessContrast.probability,
      brightness_limit: augmentations.brightnessContrast.brightnessLimit,
      contrast_limit: augmentations.brightnessContrast.contrastLimit,
    };
  }

  if (augmentations.hueSaturation.enabled) {
    result.hue_saturation = {
      enabled: true,
      probability: augmentations.hueSaturation.probability,
      hue_shift: augmentations.hueSaturation.hueShift,
      sat_shift: augmentations.hueSaturation.satShift,
      val_shift: augmentations.hueSaturation.valShift,
    };
  }

  if (augmentations.gaussianNoise.enabled) {
    result.gaussian_noise = {
      enabled: true,
      probability: augmentations.gaussianNoise.probability,
      var_limit: augmentations.gaussianNoise.varLimit,
    };
  }

  if (augmentations.gaussianBlur.enabled) {
    result.gaussian_blur = {
      enabled: true,
      probability: augmentations.gaussianBlur.probability,
      blur_limit: augmentations.gaussianBlur.blurLimit,
    };
  }

  return result;
}

/**
 * Get preset display info
 */
export function getPresetDisplayInfo(preset: AugmentationPreset): {
  name: string;
  description: string;
  icon: string;
  boost: string;
} {
  const presets: Record<AugmentationPreset, {
    name: string;
    description: string;
    icon: string;
    boost: string;
  }> = {
    "sota-v2": {
      name: "SOTA-v2 (Recommended)",
      description: "Next-gen: YOLOv8 + RT-DETR + D-FINE best practices",
      icon: "rocket",
      boost: "+4-6% mAP",
    },
    sota: {
      name: "SOTA (Legacy)",
      description: "Mosaic, MixUp, CopyPaste + standard augmentations",
      icon: "star",
      boost: "+3-5% mAP",
    },
    heavy: {
      name: "Heavy",
      description: "All augmentations, ideal for small datasets (<1000 images)",
      icon: "fire",
      boost: "+5-8% mAP",
    },
    medium: {
      name: "Medium",
      description: "Balanced augmentations for general use",
      icon: "zap",
      boost: "+2-3% mAP",
    },
    light: {
      name: "Light",
      description: "Basic augmentations only, fast training",
      icon: "feather",
      boost: "+1% mAP",
    },
    none: {
      name: "None",
      description: "No augmentation, for baseline comparison",
      icon: "x",
      boost: "Baseline",
    },
    custom: {
      name: "Custom",
      description: "Configure augmentations manually",
      icon: "settings",
      boost: "Variable",
    },
  };

  return presets[preset];
}

/**
 * Get model display info
 */
export function getModelDisplayInfo(
  modelType: "rt-detr" | "d-fine",
  modelSize: "s" | "m" | "l" | "x"
): {
  name: string;
  description: string;
  params: string;
  vram: string;
} {
  const models: Record<string, Record<string, {
    name: string;
    description: string;
    params: string;
    vram: string;
  }>> = {
    "rt-detr": {
      s: {
        name: "RT-DETR Small",
        description: "ResNet-18 backbone, fastest inference",
        params: "20M",
        vram: "~4GB",
      },
      m: {
        name: "RT-DETR Medium",
        description: "ResNet-50 backbone, balanced",
        params: "32M",
        vram: "~6GB",
      },
      l: {
        name: "RT-DETR Large",
        description: "ResNet-101 backbone, best accuracy",
        params: "42M",
        vram: "~8GB",
      },
    },
    "d-fine": {
      s: {
        name: "D-FINE Small",
        description: "Lightweight, fast inference",
        params: "25M",
        vram: "~5GB",
      },
      m: {
        name: "D-FINE Medium",
        description: "Balanced speed and accuracy",
        params: "52M",
        vram: "~8GB",
      },
      l: {
        name: "D-FINE Large",
        description: "High accuracy model",
        params: "62M",
        vram: "~10GB",
      },
      x: {
        name: "D-FINE XLarge",
        description: "Maximum accuracy, slower",
        params: "82M",
        vram: "~12GB",
      },
    },
  };

  return models[modelType]?.[modelSize] || {
    name: `${modelType.toUpperCase()} ${modelSize.toUpperCase()}`,
    description: "Unknown configuration",
    params: "N/A",
    vram: "N/A",
  };
}

/**
 * Estimate training time (rough estimate)
 */
export function estimateTrainingTime(
  datasetSize: number,
  epochs: number,
  batchSize: number,
  modelType: "rt-detr" | "d-fine",
  modelSize: "s" | "m" | "l" | "x"
): string {
  // Very rough estimates (seconds per step)
  const timePerStep: Record<string, Record<string, number>> = {
    "rt-detr": { s: 0.05, m: 0.08, l: 0.12 },
    "d-fine": { s: 0.06, m: 0.1, l: 0.15, x: 0.2 },
  };

  const baseTime = timePerStep[modelType]?.[modelSize] ?? 0.1;
  const stepsPerEpoch = Math.ceil(datasetSize / batchSize);
  const totalSteps = stepsPerEpoch * epochs;
  const totalSeconds = totalSteps * baseTime;

  // Add validation time (~10% of training)
  const totalWithVal = totalSeconds * 1.1;

  // Format time
  if (totalWithVal < 3600) {
    return `~${Math.ceil(totalWithVal / 60)} minutes`;
  } else {
    const hours = Math.floor(totalWithVal / 3600);
    const minutes = Math.ceil((totalWithVal % 3600) / 60);
    return `~${hours}h ${minutes}m`;
  }
}

/**
 * Calculate effective dataset size with offline augmentation
 */
export function calculateEffectiveDatasetSize(
  baseSize: number,
  offlineMultiplier: number,
  offlineEnabled: boolean
): number {
  return offlineEnabled ? baseSize * offlineMultiplier : baseSize;
}
