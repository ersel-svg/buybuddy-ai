/**
 * Embedding Block Definition
 *
 * Generate embeddings using DINOv2, CLIP, SigLIP, etc.
 */

import type { BlockDefinition, ValidationResult } from "../types";

export const embeddingBlock: BlockDefinition = {
  type: "embedding",
  displayName: "Embedding",
  description: "Generate image embeddings for similarity search",
  category: "model",
  icon: "Binary",

  inputs: [
    {
      name: "image",
      type: "image",
      required: false,
      description: "Single image to embed",
    },
    {
      name: "images",
      type: "array",
      required: false,
      description: "Array of images to embed (batch)",
    },
    {
      name: "crops",
      type: "array",
      required: false,
      description: "Detection crops to embed",
    },
    {
      name: "mask",
      type: "image",
      required: false,
      description: "Mask to focus embedding region",
    },
  ],

  outputs: [
    { name: "embedding", type: "array", description: "Single embedding vector" },
    { name: "embeddings", type: "array", description: "Array of embedding vectors (batch)" },
    { name: "dimension", type: "number", description: "Embedding dimension size" },
    { name: "norm", type: "number", description: "L2 norm of embedding" },
    { name: "attention_map", type: "image", description: "Attention visualization (if enabled)" },
  ],

  configFields: [
    // Model Selection
    {
      key: "model_id",
      type: "model",
      label: "Model",
      description: "Embedding model to use",
      category: "embedding",
      default: "dinov2-base",
      required: true,
    },
    {
      key: "model_source",
      type: "select",
      label: "Model Source",
      default: "pretrained",
      options: [
        { value: "pretrained", label: "Pretrained" },
        { value: "trained", label: "Fine-tuned" },
      ],
      advanced: true,
    },

    // Processing Options
    {
      key: "pooling",
      type: "select",
      label: "Pooling Strategy",
      description: "How to aggregate spatial features",
      default: "cls",
      options: [
        { value: "cls", label: "CLS Token", description: "Use [CLS] token embedding" },
        { value: "mean", label: "Mean Pooling", description: "Average all patch embeddings" },
        { value: "max", label: "Max Pooling", description: "Max across patches" },
        { value: "gem", label: "GeM Pooling", description: "Generalized mean pooling" },
      ],
    },
    {
      key: "normalize",
      type: "boolean",
      label: "L2 Normalize",
      description: "Normalize embeddings to unit length",
      default: true,
    },

    // Advanced
    {
      key: "return_attention",
      type: "boolean",
      label: "Return Attention Map",
      description: "Include attention visualization",
      default: false,
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
      key: "batch_size",
      type: "number",
      label: "Batch Size",
      description: "Process images in batches",
      default: 32,
      min: 1,
      max: 128,
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

    return { valid: errors.length === 0, errors };
  },
};
