/**
 * Classification Block Definition
 *
 * Image classification using ViT, ConvNeXt, EfficientNet, Swin, etc.
 */

import type { BlockDefinition, ValidationResult } from "../types";

export const classificationBlock: BlockDefinition = {
  type: "classification",
  displayName: "Classification",
  description: "Classify images into categories",
  category: "model",
  icon: "Tags",

  inputs: [
    {
      name: "image",
      type: "image",
      required: false,
      description: "Single image to classify",
    },
    {
      name: "images",
      type: "array",
      required: false,
      description: "Array of images to classify (batch)",
    },
    {
      name: "crops",
      type: "array",
      required: false,
      description: "Detection crops to classify",
    },
  ],

  outputs: [
    { name: "predictions", type: "array", description: "Full predictions with top-k classes" },
    { name: "label", type: "string", description: "Top predicted class label" },
    { name: "labels", type: "array", description: "Top labels for each image (batch)" },
    { name: "confidence", type: "number", description: "Confidence of top prediction" },
    { name: "probabilities", type: "object", description: "All class probabilities" },
    { name: "is_uncertain", type: "boolean", description: "Whether prediction is uncertain" },
    { name: "decision", type: "string", description: "Binary decision output (if configured)" },
  ],

  configFields: [
    // Model Selection
    {
      key: "model_id",
      type: "model",
      label: "Model",
      description: "Classification model to use",
      category: "classification",
      default: "vit-base",
      required: true,
    },
    {
      key: "model_source",
      type: "select",
      label: "Model Source",
      default: "pretrained",
      options: [
        { value: "pretrained", label: "Pretrained (ImageNet)" },
        { value: "trained", label: "Custom Trained" },
      ],
      advanced: true,
    },

    // Inference Settings
    {
      key: "top_k",
      type: "number",
      label: "Top-K",
      description: "Number of top predictions to return",
      default: 5,
      min: 1,
      max: 100,
    },
    {
      key: "confidence_threshold",
      type: "slider",
      label: "Confidence Threshold",
      description: "Minimum confidence to accept a prediction",
      default: 0.0,
      min: 0,
      max: 1,
      step: 0.05,
      formatValue: (v) => v.toFixed(2),
      supportsParams: true,
    },

    // Binary Decision Mode
    {
      key: "binary_mode",
      type: "boolean",
      label: "Binary Decision Mode",
      description: "Output yes/no decision based on threshold",
      default: false,
    },
    {
      key: "positive_classes",
      type: "array",
      label: "Positive Classes",
      description: "Classes that count as 'positive' in binary mode",
      default: [],
      itemType: "string",
      showWhen: { field: "binary_mode", value: true },
    },
    {
      key: "decision_threshold",
      type: "slider",
      label: "Decision Threshold",
      description: "Threshold for positive decision",
      default: 0.5,
      min: 0,
      max: 1,
      step: 0.05,
      formatValue: (v) => v.toFixed(2),
      showWhen: { field: "binary_mode", value: true },
    },

    // Advanced
    {
      key: "uncertainty_threshold",
      type: "slider",
      label: "Uncertainty Threshold",
      description: "Mark as uncertain if top confidence below this",
      default: 0.3,
      min: 0,
      max: 1,
      step: 0.05,
      formatValue: (v) => v.toFixed(2),
      advanced: true,
    },
    {
      key: "softmax_temperature",
      type: "number",
      label: "Softmax Temperature",
      description: "Temperature for probability scaling",
      default: 1.0,
      min: 0.1,
      max: 10,
      step: 0.1,
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
  ],

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

    if (!config.model_id) {
      errors.push({ field: "model_id", message: "Model is required" });
    }

    if (config.binary_mode) {
      const positiveClasses = config.positive_classes as string[];
      if (!positiveClasses || positiveClasses.length === 0) {
        errors.push({
          field: "positive_classes",
          message: "Positive classes required for binary mode",
        });
      }
    }

    return { valid: errors.length === 0, errors };
  },
};
