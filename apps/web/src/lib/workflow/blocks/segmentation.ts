/**
 * Segmentation Block Definition
 *
 * Image segmentation using SAM2, SAM3, etc.
 */

import type { BlockDefinition, ValidationResult } from "../types";

export const segmentationBlock: BlockDefinition = {
  type: "segmentation",
  displayName: "Segmentation",
  description: "Segment objects in images",
  category: "model",
  icon: "PenTool",

  inputs: [
    { name: "image", type: "image", required: true, description: "Image to segment" },
    { name: "boxes", type: "array", required: false, description: "Bounding boxes as prompts" },
    { name: "points", type: "array", required: false, description: "Point coordinates as prompts" },
    { name: "point_labels", type: "array", required: false, description: "Point labels (1=foreground, 0=background)" },
    { name: "text_prompt", type: "string", required: false, description: "Text prompt for SAM3" },
  ],

  outputs: [
    { name: "masks", type: "array", description: "Binary segmentation masks" },
    { name: "polygons", type: "array", description: "Mask contours as polygons" },
    { name: "rle_masks", type: "array", description: "Run-length encoded masks" },
    { name: "masked_image", type: "image", description: "Image with masks overlaid" },
    { name: "cropped_objects", type: "array", description: "Cropped objects using masks" },
    { name: "mask_scores", type: "array", description: "Confidence scores" },
    { name: "areas", type: "array", description: "Pixel area of each mask" },
  ],

  configFields: [
    // Model Selection
    {
      key: "model_id",
      type: "model",
      label: "Model",
      description: "Segmentation model to use",
      category: "segmentation",
      default: "sam2-large",
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

    // Prompt Type
    {
      key: "prompt_type",
      type: "select",
      label: "Prompt Type",
      description: "How to prompt the model",
      default: "boxes",
      options: [
        { value: "boxes", label: "Box Prompts", description: "Use bounding boxes from detection" },
        { value: "points", label: "Point Prompts", description: "Use click points" },
        { value: "text", label: "Text Prompts", description: "Use text description (SAM3 only)" },
        { value: "auto", label: "Automatic", description: "Segment everything" },
      ],
    },

    // Text Prompt (for SAM3)
    {
      key: "text_prompt",
      type: "string",
      label: "Text Prompt",
      description: "What to segment (SAM3 only)",
      placeholder: "person, car, bottle...",
      supportsParams: true,
      showWhen: { field: "prompt_type", value: "text" },
    },

    // Mask Settings
    {
      key: "mask_threshold",
      type: "slider",
      label: "Mask Threshold",
      description: "Threshold for binary mask",
      default: 0.5,
      min: 0,
      max: 1,
      step: 0.05,
      formatValue: (v) => v.toFixed(2),
    },
    {
      key: "points_per_side",
      type: "number",
      label: "Points Per Side",
      description: "Grid density for auto mode",
      default: 32,
      min: 8,
      max: 64,
      showWhen: { field: "prompt_type", value: "auto" },
    },

    // Output Options
    {
      key: "multimask_output",
      type: "boolean",
      label: "Multi-Mask Output",
      description: "Return multiple mask hypotheses",
      default: false,
      advanced: true,
    },
    {
      key: "return_polygons",
      type: "boolean",
      label: "Return Polygons",
      description: "Convert masks to polygon contours",
      default: true,
      advanced: true,
    },
    {
      key: "simplify_tolerance",
      type: "number",
      label: "Polygon Simplification",
      description: "Simplify polygon points (0 = none)",
      default: 2,
      min: 0,
      max: 10,
      advanced: true,
    },

    // Advanced
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
    const warnings: Array<{ field: string; message: string }> = [];

    if (!config.model_id) {
      errors.push({ field: "model_id", message: "Model is required" });
    }

    // Check if SAM3 is selected for text prompts
    const modelId = (config.model_id as string)?.toLowerCase() ?? "";
    const promptType = config.prompt_type as string;

    if (promptType === "text" && !modelId.includes("sam3")) {
      errors.push({
        field: "model_id",
        message: "Text prompts require SAM3 model",
      });
    }

    if (promptType === "text" && !config.text_prompt) {
      warnings.push({
        field: "text_prompt",
        message: "Text prompt is recommended when using text prompt mode",
      });
    }

    return { valid: errors.length === 0, errors, warnings };
  },
};
