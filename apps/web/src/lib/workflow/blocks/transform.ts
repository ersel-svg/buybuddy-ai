/**
 * Transform Block Definitions
 *
 * Image transformation blocks: crop, resize, tile, stitch, etc.
 */

import type { BlockDefinition } from "../types";

export const cropBlock: BlockDefinition = {
  type: "crop",
  displayName: "Crop",
  description: "Crop regions from images",
  category: "transform",
  icon: "Crop",

  inputs: [
    { name: "image", type: "image", required: true, description: "Image to crop from" },
    { name: "detections", type: "array", required: false, description: "Detection results with bboxes" },
    { name: "boxes", type: "array", required: false, description: "Raw bounding boxes [x1,y1,x2,y2]" },
    { name: "masks", type: "array", required: false, description: "Segmentation masks" },
  ],

  outputs: [
    { name: "crops", type: "array", description: "Cropped image regions" },
    { name: "crop_boxes", type: "array", description: "Final crop coordinates" },
    { name: "crop_sizes", type: "array", description: "Width/height of each crop" },
    { name: "crop_count", type: "number", description: "Number of crops produced" },
  ],

  configFields: [
    {
      key: "padding",
      type: "number",
      label: "Padding",
      description: "Extra pixels around each crop",
      default: 0,
      min: 0,
      max: 100,
    },
    {
      key: "padding_percent",
      type: "slider",
      label: "Padding %",
      description: "Padding as percentage of box size",
      default: 0,
      min: 0,
      max: 50,
      step: 1,
      formatValue: (v) => `${v}%`,
    },
    {
      key: "min_size",
      type: "number",
      label: "Minimum Size",
      description: "Skip crops smaller than this",
      default: 0,
      min: 0,
      max: 1000,
      advanced: true,
    },
    {
      key: "square",
      type: "boolean",
      label: "Square Crops",
      description: "Make crops square by expanding shorter side",
      default: false,
      advanced: true,
    },
    {
      key: "resize_to",
      type: "number",
      label: "Resize To",
      description: "Resize all crops to this size (0 = no resize)",
      default: 0,
      min: 0,
      max: 1024,
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
};

export const resizeBlock: BlockDefinition = {
  type: "resize",
  displayName: "Resize",
  description: "Resize images to target dimensions",
  category: "transform",
  icon: "Scaling",

  inputs: [
    { name: "image", type: "image", required: true, description: "Image to resize" },
    { name: "images", type: "array", required: false, description: "Array of images (batch)" },
  ],

  outputs: [
    { name: "image", type: "image", description: "Resized image" },
    { name: "images", type: "array", description: "Resized images (batch)" },
    { name: "width", type: "number", description: "Final width" },
    { name: "height", type: "number", description: "Final height" },
    { name: "scale_x", type: "number", description: "Horizontal scale factor" },
    { name: "scale_y", type: "number", description: "Vertical scale factor" },
  ],

  configFields: [
    {
      key: "width",
      type: "number",
      label: "Width",
      description: "Target width (0 = auto from aspect ratio)",
      default: 640,
      min: 0,
      max: 4096,
      supportsParams: true,
    },
    {
      key: "height",
      type: "number",
      label: "Height",
      description: "Target height (0 = auto from aspect ratio)",
      default: 0,
      min: 0,
      max: 4096,
      supportsParams: true,
    },
    {
      key: "keep_aspect",
      type: "boolean",
      label: "Keep Aspect Ratio",
      description: "Preserve original aspect ratio",
      default: true,
    },
    {
      key: "interpolation",
      type: "select",
      label: "Interpolation",
      description: "Resampling method",
      default: "bilinear",
      options: [
        { value: "nearest", label: "Nearest" },
        { value: "bilinear", label: "Bilinear" },
        { value: "bicubic", label: "Bicubic" },
        { value: "lanczos", label: "Lanczos" },
      ],
      advanced: true,
    },
    {
      key: "pad_color",
      type: "string",
      label: "Padding Color",
      description: "Color for letterbox padding",
      default: "#000000",
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
};

export const tileBlock: BlockDefinition = {
  type: "tile",
  displayName: "Tile",
  description: "Split image into overlapping tiles",
  category: "transform",
  icon: "LayoutGrid",

  inputs: [
    { name: "image", type: "image", required: true, description: "Image to tile" },
  ],

  outputs: [
    { name: "tiles", type: "array", description: "Array of tile images" },
    { name: "tile_coords", type: "array", description: "Coordinates of each tile" },
    { name: "tile_count", type: "number", description: "Total number of tiles" },
    { name: "grid_size", type: "object", description: "Grid dimensions {rows, cols}" },
    { name: "original_size", type: "object", description: "Original image size" },
  ],

  configFields: [
    {
      key: "tile_size",
      type: "number",
      label: "Tile Size",
      description: "Width and height of each tile",
      default: 640,
      min: 128,
      max: 2048,
      supportsParams: true,
    },
    {
      key: "overlap",
      type: "slider",
      label: "Overlap",
      description: "Overlap between adjacent tiles",
      default: 64,
      min: 0,
      max: 256,
      step: 8,
      formatValue: (v) => `${v}px`,
      supportsParams: true,
    },
    {
      key: "min_coverage",
      type: "slider",
      label: "Min Coverage",
      description: "Minimum tile coverage to include edge tiles",
      default: 0.5,
      min: 0,
      max: 1,
      step: 0.1,
      formatValue: (v) => `${(v * 100).toFixed(0)}%`,
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
};

export const stitchBlock: BlockDefinition = {
  type: "stitch",
  displayName: "Stitch",
  description: "Merge tile results back together",
  category: "transform",
  icon: "Combine",

  inputs: [
    { name: "tiles", type: "array", required: true, description: "Array of tile images or results" },
    { name: "tile_coords", type: "array", required: true, description: "Coordinates from tile node" },
    { name: "detections", type: "array", required: false, description: "Detections to merge from tiles" },
    { name: "original_size", type: "object", required: false, description: "Original image size" },
  ],

  outputs: [
    { name: "image", type: "image", description: "Stitched image" },
    { name: "merged_detections", type: "array", description: "Merged detections with NMS" },
    { name: "detection_count", type: "number", description: "Final detection count" },
  ],

  configFields: [
    {
      key: "nms_threshold",
      type: "slider",
      label: "NMS Threshold",
      description: "IoU threshold for merging overlapping detections",
      default: 0.5,
      min: 0,
      max: 1,
      step: 0.05,
      formatValue: (v) => v.toFixed(2),
    },
    {
      key: "min_confidence",
      type: "slider",
      label: "Min Confidence",
      description: "Filter low-confidence detections before merge",
      default: 0.1,
      min: 0,
      max: 1,
      step: 0.05,
      formatValue: (v) => v.toFixed(2),
    },
    {
      key: "merge_strategy",
      type: "select",
      label: "Merge Strategy",
      description: "How to merge overlapping detections",
      default: "nms",
      options: [
        { value: "nms", label: "Non-Maximum Suppression" },
        { value: "soft_nms", label: "Soft NMS" },
        { value: "wbf", label: "Weighted Box Fusion" },
      ],
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
};

export const filterBlock: BlockDefinition = {
  type: "filter",
  displayName: "Filter",
  description: "Filter arrays by conditions",
  category: "transform",
  icon: "Filter",

  inputs: [
    { name: "items", type: "array", required: true, description: "Array to filter" },
    { name: "detections", type: "array", required: false, description: "Detection results" },
  ],

  outputs: [
    { name: "passed", type: "array", description: "Items that passed filter" },
    { name: "rejected", type: "array", description: "Items that failed filter" },
    { name: "passed_count", type: "number", description: "Number of passed items" },
    { name: "pass_rate", type: "number", description: "Percentage that passed" },
  ],

  configFields: [
    {
      key: "field",
      type: "string",
      label: "Field",
      description: "Field to filter on",
      default: "confidence",
      placeholder: "confidence, label, area...",
    },
    {
      key: "operator",
      type: "select",
      label: "Operator",
      description: "Comparison operator",
      default: "gte",
      options: [
        { value: "eq", label: "Equals (==)" },
        { value: "neq", label: "Not Equals (!=)" },
        { value: "gt", label: "Greater Than (>)" },
        { value: "gte", label: "Greater or Equal (>=)" },
        { value: "lt", label: "Less Than (<)" },
        { value: "lte", label: "Less or Equal (<=)" },
        { value: "in", label: "In List" },
        { value: "nin", label: "Not In List" },
        { value: "contains", label: "Contains" },
      ],
    },
    {
      key: "value",
      type: "string",
      label: "Value",
      description: "Value to compare against",
      default: "0.5",
      supportsParams: true,
    },
    {
      key: "top_n",
      type: "number",
      label: "Top N",
      description: "Keep only top N items (0 = all)",
      default: 0,
      min: 0,
      max: 1000,
      advanced: true,
    },
    {
      key: "sort_by",
      type: "string",
      label: "Sort By",
      description: "Field to sort by before top-N",
      placeholder: "confidence",
      advanced: true,
    },
    {
      key: "sort_order",
      type: "select",
      label: "Sort Order",
      default: "desc",
      options: [
        { value: "asc", label: "Ascending" },
        { value: "desc", label: "Descending" },
      ],
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
};

export const rotateFlipBlock: BlockDefinition = {
  type: "rotate_flip",
  displayName: "Rotate/Flip",
  description: "Rotate and flip images",
  category: "transform",
  icon: "RotateCw",

  inputs: [
    { name: "image", type: "image", required: true, description: "Image to transform" },
    { name: "detections", type: "array", required: false, description: "Detections to transform" },
  ],

  outputs: [
    { name: "image", type: "image", description: "Transformed image" },
    { name: "detections", type: "array", description: "Transformed detections" },
    { name: "transform_matrix", type: "object", description: "Transformation matrix" },
  ],

  configFields: [
    {
      key: "rotation",
      type: "select",
      label: "Rotation",
      description: "Rotation angle",
      default: "none",
      options: [
        { value: "none", label: "No Rotation" },
        { value: "90", label: "90° Clockwise" },
        { value: "180", label: "180°" },
        { value: "270", label: "270° (90° Counter-clockwise)" },
        { value: "auto_exif", label: "Auto (from EXIF)" },
        { value: "custom", label: "Custom Angle" },
      ],
    },
    {
      key: "custom_angle",
      type: "slider",
      label: "Angle (degrees)",
      description: "Custom rotation angle",
      default: 0,
      min: -180,
      max: 180,
      step: 1,
      showWhen: { field: "rotation", value: "custom" },
    },
    {
      key: "flip_horizontal",
      type: "boolean",
      label: "Flip Horizontal",
      description: "Mirror left-right",
      default: false,
    },
    {
      key: "flip_vertical",
      type: "boolean",
      label: "Flip Vertical",
      description: "Mirror top-bottom",
      default: false,
    },
    {
      key: "transform_boxes",
      type: "boolean",
      label: "Transform Boxes",
      description: "Apply same transform to bboxes",
      default: true,
    },
    {
      key: "expand_canvas",
      type: "boolean",
      label: "Expand Canvas",
      description: "Fit rotated image without crop",
      default: true,
      showWhen: { field: "rotation", value: "custom" },
      advanced: true,
    },
    {
      key: "background_color",
      type: "string",
      label: "Background Color",
      description: "Fill color for expanded areas",
      default: "#000000",
      showWhen: { field: "rotation", value: "custom" },
      advanced: true,
    },
    {
      key: "interpolation",
      type: "select",
      label: "Interpolation",
      description: "Resampling method",
      default: "bilinear",
      options: [
        { value: "nearest", label: "Nearest" },
        { value: "bilinear", label: "Bilinear" },
        { value: "bicubic", label: "Bicubic" },
      ],
      advanced: true,
    },
    {
      key: "center",
      type: "select",
      label: "Rotation Center",
      description: "Center point for rotation",
      default: "center",
      options: [
        { value: "center", label: "Image Center" },
        { value: "top_left", label: "Top Left" },
        { value: "custom", label: "Custom Point" },
      ],
      advanced: true,
    },
    {
      key: "output_matrix",
      type: "boolean",
      label: "Output Transform Matrix",
      description: "For coordinate mapping",
      default: false,
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
};

export const normalizeBlock: BlockDefinition = {
  type: "normalize",
  displayName: "Normalize",
  description: "Normalize image values for model input",
  category: "transform",
  icon: "Sliders",

  inputs: [
    { name: "image", type: "image", required: true, description: "Image to normalize" },
    { name: "images", type: "array", required: false, description: "Batch of images" },
  ],

  outputs: [
    { name: "image", type: "image", description: "Normalized image" },
    { name: "tensor", type: "object", description: "Tensor output [C,H,W]" },
    { name: "stats", type: "object", description: "Normalization statistics" },
  ],

  configFields: [
    {
      key: "preset",
      type: "select",
      label: "Normalization Preset",
      description: "Pre-defined normalization values",
      default: "imagenet",
      options: [
        { value: "imagenet", label: "ImageNet (most models)" },
        { value: "clip", label: "CLIP / SigLIP" },
        { value: "dinov2", label: "DINOv2" },
        { value: "0_1", label: "0-1 Range (divide by 255)" },
        { value: "-1_1", label: "-1 to 1 Range" },
        { value: "none", label: "None (keep 0-255)" },
        { value: "custom", label: "Custom Mean/Std" },
      ],
    },
    {
      key: "mean_r",
      type: "number",
      label: "Mean R",
      description: "Red channel mean",
      default: 0.485,
      showWhen: { field: "preset", value: "custom" },
    },
    {
      key: "mean_g",
      type: "number",
      label: "Mean G",
      description: "Green channel mean",
      default: 0.456,
      showWhen: { field: "preset", value: "custom" },
    },
    {
      key: "mean_b",
      type: "number",
      label: "Mean B",
      description: "Blue channel mean",
      default: 0.406,
      showWhen: { field: "preset", value: "custom" },
    },
    {
      key: "std_r",
      type: "number",
      label: "Std R",
      description: "Red channel std",
      default: 0.229,
      showWhen: { field: "preset", value: "custom" },
    },
    {
      key: "std_g",
      type: "number",
      label: "Std G",
      description: "Green channel std",
      default: 0.224,
      showWhen: { field: "preset", value: "custom" },
    },
    {
      key: "std_b",
      type: "number",
      label: "Std B",
      description: "Blue channel std",
      default: 0.225,
      showWhen: { field: "preset", value: "custom" },
    },
    {
      key: "output_format",
      type: "select",
      label: "Output Format",
      description: "Output data format",
      default: "image",
      options: [
        { value: "image", label: "Image (for display)" },
        { value: "tensor", label: "Tensor [C,H,W] (for models)" },
        { value: "both", label: "Both" },
      ],
    },
    {
      key: "channel_order",
      type: "select",
      label: "Channel Order",
      description: "Color channel order",
      default: "rgb",
      options: [
        { value: "rgb", label: "RGB (most models)" },
        { value: "bgr", label: "BGR (OpenCV style)" },
      ],
    },
    {
      key: "dtype",
      type: "select",
      label: "Output Data Type",
      description: "Numeric precision",
      default: "float32",
      options: [
        { value: "float32", label: "float32 (default)" },
        { value: "float16", label: "float16 (half precision)" },
        { value: "uint8", label: "uint8 (image only)" },
      ],
      advanced: true,
    },
    {
      key: "output_denormalized",
      type: "boolean",
      label: "Also Output Denormalized",
      description: "Include original scale image",
      default: false,
      advanced: true,
    },
    {
      key: "clip_values",
      type: "boolean",
      label: "Clip Values",
      description: "Clamp to valid range",
      default: true,
      advanced: true,
    },
    {
      key: "output_stats",
      type: "boolean",
      label: "Output Channel Stats",
      description: "Include mean/std info",
      default: false,
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
};

export const smoothingBlock: BlockDefinition = {
  type: "smoothing",
  displayName: "Smoothing",
  description: "Apply blur/smoothing filters",
  category: "transform",
  icon: "Droplet",

  inputs: [
    { name: "image", type: "image", required: true, description: "Image to smooth" },
  ],

  outputs: [
    { name: "image", type: "image", description: "Smoothed image" },
  ],

  configFields: [
    {
      key: "smoothing_type",
      type: "select",
      label: "Smoothing Type",
      description: "Type of blur to apply",
      default: "gaussian",
      options: [
        { value: "gaussian", label: "Gaussian Blur" },
        { value: "median", label: "Median Blur" },
        { value: "bilateral", label: "Bilateral Filter" },
        { value: "box", label: "Box Blur" },
        { value: "motion", label: "Motion Blur" },
      ],
    },
    {
      key: "kernel_size",
      type: "slider",
      label: "Kernel Size",
      description: "Blur kernel size (must be odd)",
      default: 5,
      min: 1,
      max: 31,
      step: 2,
    },
    {
      key: "sigma_x",
      type: "slider",
      label: "Sigma X",
      description: "Gaussian sigma (0 = auto)",
      default: 0,
      min: 0,
      max: 10,
      step: 0.5,
      showWhen: { field: "smoothing_type", value: "gaussian" },
    },
    {
      key: "d",
      type: "slider",
      label: "Diameter",
      description: "Bilateral filter diameter",
      default: 9,
      min: 1,
      max: 25,
      step: 2,
      showWhen: { field: "smoothing_type", value: "bilateral" },
    },
    {
      key: "sigma_color",
      type: "slider",
      label: "Sigma Color",
      description: "Color space sigma",
      default: 75,
      min: 10,
      max: 200,
      step: 5,
      showWhen: { field: "smoothing_type", value: "bilateral" },
    },
    {
      key: "sigma_space",
      type: "slider",
      label: "Sigma Space",
      description: "Coordinate space sigma",
      default: 75,
      min: 10,
      max: 200,
      step: 5,
      showWhen: { field: "smoothing_type", value: "bilateral" },
    },
    {
      key: "motion_length",
      type: "slider",
      label: "Motion Length",
      description: "Length of motion blur",
      default: 15,
      min: 3,
      max: 51,
      step: 2,
      showWhen: { field: "smoothing_type", value: "motion" },
    },
    {
      key: "motion_angle",
      type: "slider",
      label: "Motion Angle",
      description: "Direction of motion blur",
      default: 0,
      min: 0,
      max: 360,
      step: 15,
      showWhen: { field: "smoothing_type", value: "motion" },
    },
    {
      key: "iterations",
      type: "number",
      label: "Iterations",
      description: "Apply filter multiple times",
      default: 1,
      min: 1,
      max: 5,
      advanced: true,
    },
    {
      key: "apply_roi",
      type: "boolean",
      label: "Apply to ROI Only",
      description: "Smooth only a region",
      default: false,
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
};
