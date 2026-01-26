/**
 * Detection Block Definition
 *
 * Object detection using YOLO, RT-DETR, D-FINE, Grounding DINO, etc.
 */

import type { BlockDefinition, ValidationResult } from "../types";

export const detectionBlock: BlockDefinition = {
  type: "detection",
  displayName: "Object Detection",
  description: "Detect objects in images using AI models",
  category: "model",
  icon: "ScanSearch",

  inputs: [
    {
      name: "image",
      type: "image",
      required: true,
      description: "Input image to detect objects in",
    },
    {
      name: "text_prompt",
      type: "string",
      required: false,
      description: "Text prompt for open-vocab detection (Grounding DINO, Florence)",
    },
    {
      name: "class_filter",
      type: "array",
      required: false,
      description: "Filter to specific class names",
    },
    {
      name: "roi",
      type: "object",
      required: false,
      description: "Region of interest to detect within",
    },
  ],

  outputs: [
    { name: "detections", type: "array", description: "All detected objects with bbox, class, confidence" },
    { name: "boxes", type: "array", description: "Bounding boxes only [x1,y1,x2,y2]" },
    { name: "labels", type: "array", description: "Class labels for each detection" },
    { name: "scores", type: "array", description: "Confidence scores for each detection" },
    { name: "annotated_image", type: "image", description: "Image with bounding boxes drawn" },
    { name: "crops", type: "array", description: "Cropped images for each detection" },
    { name: "count", type: "number", description: "Total number of detections" },
    { name: "class_counts", type: "object", description: "Count per class {class: count}" },
  ],

  configFields: [
    // Model Selection
    {
      key: "model_id",
      type: "model",
      label: "Model",
      description: "Detection model to use",
      category: "detection",
      default: "yolo11n",
      required: true,
    },
    {
      key: "model_source",
      type: "select",
      label: "Model Source",
      description: "Whether using pretrained or custom trained model",
      default: "pretrained",
      options: [
        { value: "pretrained", label: "Pretrained" },
        { value: "trained", label: "Custom Trained" },
      ],
      advanced: true,
    },

    // Text Prompt (for open-vocab models)
    {
      key: "text_prompt",
      type: "string",
      label: "Text Prompt",
      description: "Describe what to detect (for Grounding DINO, Florence)",
      placeholder: "person. car. bottle.",
      supportsParams: true,
      showWhen: {
        field: "model_id",
        value: ["grounding-dino", "florence2", "owl"],
        operator: "includes",
      },
    },

    // Thresholds for open-vocab
    {
      key: "box_threshold",
      type: "slider",
      label: "Box Threshold",
      description: "Minimum confidence for detected boxes",
      default: 0.35,
      min: 0,
      max: 1,
      step: 0.05,
      formatValue: (v) => v.toFixed(2),
      supportsParams: true,
      showWhen: {
        field: "model_id",
        value: ["grounding-dino", "florence2", "owl"],
        operator: "includes",
      },
    },
    {
      key: "text_threshold",
      type: "slider",
      label: "Text Threshold",
      description: "Minimum text-box matching score",
      default: 0.25,
      min: 0,
      max: 1,
      step: 0.05,
      formatValue: (v) => v.toFixed(2),
      supportsParams: true,
      showWhen: {
        field: "model_id",
        value: ["grounding-dino", "florence2", "owl"],
        operator: "includes",
      },
    },

    // Standard Detection Settings
    {
      key: "confidence_threshold",
      type: "slider",
      label: "Confidence Threshold",
      description: "Minimum confidence to keep a detection",
      default: 0.25,
      min: 0,
      max: 1,
      step: 0.05,
      formatValue: (v) => v.toFixed(2),
      supportsParams: true,
    },
    {
      key: "nms_threshold",
      type: "slider",
      label: "NMS Threshold",
      description: "IoU threshold for Non-Maximum Suppression",
      default: 0.45,
      min: 0,
      max: 1,
      step: 0.05,
      formatValue: (v) => v.toFixed(2),
      supportsParams: true,
      advanced: true,
    },

    // Input Resolution
    {
      key: "input_size",
      type: "select",
      label: "Input Resolution",
      description: "Image size for model input",
      default: "640",
      options: [
        { value: "320", label: "320px (Fast)" },
        { value: "480", label: "480px" },
        { value: "640", label: "640px (Balanced)" },
        { value: "800", label: "800px" },
        { value: "1024", label: "1024px" },
        { value: "1280", label: "1280px (Quality)" },
      ],
    },

    // Class Filtering
    {
      key: "class_filter",
      type: "array",
      label: "Class Filter",
      description: "Only detect these classes (empty = all)",
      default: [],
      itemType: "string",
      supportsParams: true,
      advanced: true,
    },

    // Advanced Options
    {
      key: "max_detections",
      type: "number",
      label: "Max Detections",
      description: "Maximum number of detections to return",
      default: 300,
      min: 1,
      max: 1000,
      advanced: true,
    },
    {
      key: "half_precision",
      type: "boolean",
      label: "Half Precision (FP16)",
      description: "Use FP16 for faster inference",
      default: true,
      advanced: true,
    },
    {
      key: "augment",
      type: "boolean",
      label: "Test-Time Augmentation",
      description: "Apply TTA for better accuracy (slower)",
      default: false,
      advanced: true,
    },
    {
      key: "agnostic_nms",
      type: "boolean",
      label: "Agnostic NMS",
      description: "Class-agnostic non-maximum suppression",
      default: false,
      advanced: true,
    },

    // Output Options
    {
      key: "return_crops",
      type: "boolean",
      label: "Return Crops",
      description: "Include cropped images for each detection",
      default: false,
      advanced: true,
    },
    {
      key: "draw_boxes",
      type: "boolean",
      label: "Draw Boxes",
      description: "Return annotated image with bounding boxes",
      default: true,
      advanced: true,
    },
  ],

  // Compute default config from fields
  get defaultConfig() {
    const config: Record<string, unknown> = {};
    for (const field of this.configFields) {
      if (field.default !== undefined) {
        config[field.key] = field.default;
      }
    }
    return config;
  },

  validate(config: Record<string, unknown>): ValidationResult {
    const errors: Array<{ field: string; message: string }> = [];
    const warnings: Array<{ field: string; message: string }> = [];

    // Required: model_id
    if (!config.model_id) {
      errors.push({ field: "model_id", message: "Model is required" });
    }

    // Check open-vocab models need text_prompt
    const modelId = (config.model_id as string)?.toLowerCase() ?? "";
    const isOpenVocab =
      modelId.includes("grounding") ||
      modelId.includes("dino") ||
      modelId.includes("owl") ||
      modelId.includes("florence");

    if (isOpenVocab && !config.text_prompt) {
      warnings.push({
        field: "text_prompt",
        message: "Open-vocabulary models work best with a text prompt",
      });
    }

    // Validate thresholds
    const confidence = config.confidence_threshold as number;
    if (confidence !== undefined && (confidence < 0 || confidence > 1)) {
      errors.push({
        field: "confidence_threshold",
        message: "Confidence must be between 0 and 1",
      });
    }

    return {
      valid: errors.length === 0,
      errors,
      warnings,
    };
  },
};
