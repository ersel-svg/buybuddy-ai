/**
 * Output Block Definitions
 *
 * Output blocks: JSON output, grid builder, draw boxes, blur region.
 */

import type { BlockDefinition } from "../types";

export const jsonOutputBlock: BlockDefinition = {
  type: "json_output",
  displayName: "JSON Output",
  description: "Format and output JSON data",
  category: "output",
  icon: "FileJson",
  canBeEnd: true,

  inputs: [
    { name: "data", type: "any", required: true, description: "Data to output" },
  ],

  outputs: [
    { name: "json", type: "object", description: "Formatted JSON output" },
  ],

  configFields: [
    {
      key: "schema",
      type: "code",
      label: "Output Schema",
      description: "Transform data to specific schema",
      language: "json",
    },
    {
      key: "pretty",
      type: "boolean",
      label: "Pretty Print",
      description: "Format output with indentation",
      default: true,
    },
    {
      key: "include_metadata",
      type: "boolean",
      label: "Include Metadata",
      description: "Add execution metadata to output",
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

export const gridBuilderBlock: BlockDefinition = {
  type: "grid_builder",
  displayName: "Grid Builder",
  description: "Build shelf grid from detections",
  category: "output",
  icon: "Grid3X3",
  canBeEnd: true,

  inputs: [
    { name: "detections", type: "array", required: true, description: "Detection data with bboxes" },
    { name: "image", type: "image", required: false, description: "Original image (for shelf edge detection)" },
    { name: "expected_layout", type: "array", required: false, description: "Reference planogram" },
  ],

  outputs: [
    { name: "grid", type: "array", description: "2D grid [[row1], [row2], ...]" },
    { name: "realogram", type: "object", description: "Full realogram with metadata" },
    { name: "shelves", type: "array", description: "Shelf-grouped items" },
    { name: "row_count", type: "number", description: "Number of detected rows" },
    { name: "col_count", type: "number", description: "Maximum columns per row" },
    { name: "facings", type: "object", description: "Facing counts per product" },
    { name: "comparison", type: "object", description: "Planogram comparison results" },
  ],

  configFields: [
    {
      key: "detection_mode",
      type: "select",
      label: "Row Detection Mode",
      description: "How to detect shelf rows",
      default: "clustering",
      options: [
        { value: "clustering", label: "Y-Coordinate Clustering" },
        { value: "edge_detection", label: "Shelf Edge Detection" },
        { value: "fixed", label: "Fixed Row Count" },
      ],
    },
    {
      key: "row_tolerance",
      type: "number",
      label: "Row Tolerance",
      description: "Y-distance tolerance for same row",
      default: 50,
      min: 10,
      max: 200,
    },
    {
      key: "fixed_rows",
      type: "number",
      label: "Fixed Row Count",
      description: "Number of rows (for fixed mode)",
      default: 4,
      min: 1,
      max: 20,
      showWhen: { field: "detection_mode", value: "fixed" },
    },
    {
      key: "sort_by",
      type: "select",
      label: "Sort Items By",
      description: "How to order items within rows",
      default: "x",
      options: [
        { value: "x", label: "Left to Right (X)" },
        { value: "area", label: "By Area" },
        { value: "confidence", label: "By Confidence" },
      ],
    },
    {
      key: "merge_overlapping",
      type: "boolean",
      label: "Merge Overlapping",
      description: "Combine highly overlapping detections",
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
};

export const drawBoxesBlock: BlockDefinition = {
  type: "draw_boxes",
  displayName: "Draw Boxes",
  description: "Draw bounding boxes on image",
  category: "output",
  icon: "PenTool",
  canBeEnd: true,

  inputs: [
    { name: "image", type: "image", required: true, description: "Image to draw on" },
    { name: "detections", type: "array", required: true, description: "Detections with bboxes" },
  ],

  outputs: [
    { name: "image", type: "image", description: "Image with boxes drawn" },
  ],

  configFields: [
    {
      key: "line_width",
      type: "number",
      label: "Line Width",
      description: "Box border thickness",
      default: 2,
      min: 1,
      max: 10,
    },
    {
      key: "show_labels",
      type: "boolean",
      label: "Show Labels",
      description: "Display class labels",
      default: true,
    },
    {
      key: "show_confidence",
      type: "boolean",
      label: "Show Confidence",
      description: "Display confidence scores",
      default: true,
    },
    {
      key: "font_scale",
      type: "slider",
      label: "Font Scale",
      description: "Label font size",
      default: 0.5,
      min: 0.3,
      max: 2,
      step: 0.1,
      formatValue: (v) => v.toFixed(1),
    },
    {
      key: "color_by",
      type: "select",
      label: "Color By",
      description: "How to color boxes",
      default: "class",
      options: [
        { value: "class", label: "By Class" },
        { value: "confidence", label: "By Confidence" },
        { value: "fixed", label: "Single Color" },
      ],
    },
    {
      key: "fixed_color",
      type: "string",
      label: "Box Color",
      description: "Fixed color for all boxes",
      default: "#00ff00",
      showWhen: { field: "color_by", value: "fixed" },
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

export const blurRegionBlock: BlockDefinition = {
  type: "blur_region",
  displayName: "Blur Region",
  description: "Blur detected regions in image",
  category: "output",
  icon: "Eraser",
  canBeEnd: true,

  inputs: [
    { name: "image", type: "image", required: true, description: "Image to blur" },
    { name: "regions", type: "array", required: true, description: "Regions to blur (detections)" },
  ],

  outputs: [
    { name: "image", type: "image", description: "Image with blurred regions" },
  ],

  configFields: [
    {
      key: "blur_strength",
      type: "slider",
      label: "Blur Strength",
      description: "Blur kernel size",
      default: 51,
      min: 3,
      max: 101,
      step: 2,
      formatValue: (v) => `${v}px`,
    },
    {
      key: "blur_type",
      type: "select",
      label: "Blur Type",
      description: "Type of blur to apply",
      default: "gaussian",
      options: [
        { value: "gaussian", label: "Gaussian Blur" },
        { value: "pixelate", label: "Pixelate" },
        { value: "box", label: "Box Blur" },
      ],
    },
    {
      key: "class_filter",
      type: "array",
      label: "Class Filter",
      description: "Only blur these classes (empty = all)",
      default: [],
      itemType: "string",
      advanced: true,
    },
    {
      key: "padding",
      type: "number",
      label: "Padding",
      description: "Extra pixels around region",
      default: 0,
      min: 0,
      max: 50,
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

export const drawMasksBlock: BlockDefinition = {
  type: "draw_masks",
  displayName: "Draw Masks",
  description: "Visualize segmentation masks",
  category: "output",
  icon: "Palette",
  canBeEnd: true,

  inputs: [
    { name: "image", type: "image", required: true, description: "Base image" },
    { name: "masks", type: "array", required: true, description: "Segmentation masks" },
  ],

  outputs: [
    { name: "image", type: "image", description: "Image with masks overlaid" },
  ],

  configFields: [
    {
      key: "mode",
      type: "select",
      label: "Rendering Mode",
      description: "How to display masks",
      default: "overlay",
      options: [
        { value: "overlay", label: "Overlay (transparent)" },
        { value: "contour", label: "Contour Only" },
        { value: "filled", label: "Filled (solid)" },
        { value: "side_by_side", label: "Side by Side" },
      ],
    },
    {
      key: "palette",
      type: "select",
      label: "Color Palette",
      description: "Color scheme for masks",
      default: "rainbow",
      options: [
        { value: "rainbow", label: "Rainbow (vibrant)" },
        { value: "category", label: "Category (COCO style)" },
        { value: "pastel", label: "Pastel (soft)" },
        { value: "viridis", label: "Viridis (scientific)" },
      ],
    },
    {
      key: "opacity",
      type: "slider",
      label: "Opacity",
      description: "Mask transparency",
      default: 0.5,
      min: 0,
      max: 1,
      step: 0.05,
      formatValue: (v) => `${Math.round(v * 100)}%`,
    },
    {
      key: "draw_contours",
      type: "boolean",
      label: "Draw Contours",
      description: "Show mask edges",
      default: true,
    },
    {
      key: "contour_thickness",
      type: "slider",
      label: "Contour Thickness",
      description: "Edge line width",
      default: 2,
      min: 1,
      max: 5,
      step: 1,
      showWhen: { field: "draw_contours", value: true },
      advanced: true,
    },
    {
      key: "show_labels",
      type: "boolean",
      label: "Show Labels",
      description: "Display class names",
      default: false,
      advanced: true,
    },
    {
      key: "show_areas",
      type: "boolean",
      label: "Show Areas",
      description: "Display pixel counts",
      default: false,
      advanced: true,
    },
    {
      key: "show_iou",
      type: "boolean",
      label: "Show IoU Scores",
      description: "Display confidence",
      default: false,
      advanced: true,
    },
    {
      key: "min_area",
      type: "number",
      label: "Min Area (pixels)",
      description: "Filter small masks",
      default: 0,
      min: 0,
      advanced: true,
    },
    {
      key: "max_masks",
      type: "number",
      label: "Max Masks to Display",
      description: "Limit number of masks",
      default: 100,
      min: 1,
      max: 500,
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

export const heatmapBlock: BlockDefinition = {
  type: "heatmap",
  displayName: "Heatmap",
  description: "Visualize attention/activation maps",
  category: "output",
  icon: "Flame",
  canBeEnd: true,

  inputs: [
    { name: "image", type: "image", required: true, description: "Base image" },
    { name: "heatmap", type: "object", required: true, description: "Heatmap data" },
  ],

  outputs: [
    { name: "image", type: "image", description: "Image with heatmap overlay" },
    { name: "colorbar", type: "image", description: "Colorbar legend" },
  ],

  configFields: [
    {
      key: "colormap",
      type: "select",
      label: "Colormap",
      description: "Color scheme",
      default: "jet",
      options: [
        { value: "jet", label: "Jet (blue→red)" },
        { value: "viridis", label: "Viridis (purple→yellow)" },
        { value: "hot", label: "Hot (black→white)" },
        { value: "cool", label: "Cool (cyan→magenta)" },
        { value: "plasma", label: "Plasma" },
        { value: "inferno", label: "Inferno" },
        { value: "magma", label: "Magma" },
        { value: "turbo", label: "Turbo" },
      ],
    },
    {
      key: "opacity",
      type: "slider",
      label: "Opacity",
      description: "Heatmap transparency",
      default: 0.6,
      min: 0,
      max: 1,
      step: 0.05,
      formatValue: (v) => `${Math.round(v * 100)}%`,
    },
    {
      key: "blend_mode",
      type: "select",
      label: "Blend Mode",
      description: "How to blend with image",
      default: "overlay",
      options: [
        { value: "overlay", label: "Overlay (standard)" },
        { value: "multiply", label: "Multiply (darker)" },
        { value: "screen", label: "Screen (lighter)" },
        { value: "add", label: "Add (bright)" },
      ],
    },
    {
      key: "threshold",
      type: "slider",
      label: "Threshold",
      description: "Hide low values",
      default: 0,
      min: 0,
      max: 0.5,
      step: 0.05,
      formatValue: (v) => `${Math.round(v * 100)}%`,
    },
    {
      key: "normalize",
      type: "boolean",
      label: "Normalize Values",
      description: "Scale to 0-1 range",
      default: true,
      advanced: true,
    },
    {
      key: "normalize_method",
      type: "select",
      label: "Normalization Method",
      description: "How to normalize",
      default: "minmax",
      options: [
        { value: "minmax", label: "Min-Max (full range)" },
        { value: "percentile", label: "Percentile (robust)" },
        { value: "zscore", label: "Z-Score (standard)" },
      ],
      showWhen: { field: "normalize", value: true },
      advanced: true,
    },
    {
      key: "smooth",
      type: "boolean",
      label: "Smooth Heatmap",
      description: "Apply Gaussian blur",
      default: true,
      advanced: true,
    },
    {
      key: "smooth_sigma",
      type: "slider",
      label: "Smooth Sigma",
      description: "Blur amount",
      default: 2,
      min: 1,
      max: 10,
      step: 1,
      showWhen: { field: "smooth", value: true },
      advanced: true,
    },
    {
      key: "output_colorbar",
      type: "boolean",
      label: "Generate Colorbar",
      description: "Output legend image",
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

export const comparisonBlock: BlockDefinition = {
  type: "comparison",
  displayName: "Comparison",
  description: "Compare two images side by side",
  category: "output",
  icon: "GitCompare",
  canBeEnd: true,

  inputs: [
    { name: "image_a", type: "image", required: true, description: "First image" },
    { name: "image_b", type: "image", required: true, description: "Second image" },
  ],

  outputs: [
    { name: "image", type: "image", description: "Comparison image" },
    { name: "diff_stats", type: "object", description: "Difference statistics" },
  ],

  configFields: [
    {
      key: "mode",
      type: "select",
      label: "Comparison Mode",
      description: "How to compare images",
      default: "side_by_side",
      options: [
        { value: "side_by_side", label: "Side by Side" },
        { value: "top_bottom", label: "Top / Bottom" },
        { value: "overlay", label: "Overlay Blend" },
        { value: "difference", label: "Difference Map" },
        { value: "checkerboard", label: "Checkerboard" },
        { value: "grid", label: "Grid Layout" },
      ],
    },
    {
      key: "show_labels",
      type: "boolean",
      label: "Show Labels",
      description: "Display image labels",
      default: true,
    },
    {
      key: "label_a",
      type: "string",
      label: "Label A",
      description: "First image label",
      default: "Before",
      showWhen: { field: "show_labels", value: true },
    },
    {
      key: "label_b",
      type: "string",
      label: "Label B",
      description: "Second image label",
      default: "After",
      showWhen: { field: "show_labels", value: true },
    },
    {
      key: "overlay_opacity",
      type: "slider",
      label: "Blend Opacity",
      description: "Overlay blend amount",
      default: 0.5,
      min: 0,
      max: 1,
      step: 0.05,
      showWhen: { field: "mode", value: "overlay" },
    },
    {
      key: "diff_type",
      type: "select",
      label: "Difference Type",
      description: "Type of difference",
      default: "absolute",
      options: [
        { value: "absolute", label: "Absolute Difference" },
        { value: "signed", label: "Signed Difference" },
        { value: "edge", label: "Edge Difference" },
      ],
      showWhen: { field: "mode", value: "difference" },
    },
    {
      key: "diff_colorize",
      type: "boolean",
      label: "Colorize Difference",
      description: "Green=same, Red=different",
      default: true,
      showWhen: { field: "mode", value: "difference" },
    },
    {
      key: "diff_amplify",
      type: "slider",
      label: "Amplify Difference",
      description: "Enhance visibility",
      default: 1,
      min: 1,
      max: 10,
      step: 1,
      showWhen: { field: "mode", value: "difference" },
    },
    {
      key: "checker_size",
      type: "slider",
      label: "Tile Size",
      description: "Checkerboard tile size",
      default: 50,
      min: 20,
      max: 200,
      step: 10,
      showWhen: { field: "mode", value: "checkerboard" },
    },
    {
      key: "grid_cols",
      type: "slider",
      label: "Grid Columns",
      description: "Number of columns",
      default: 2,
      min: 2,
      max: 6,
      step: 1,
      showWhen: { field: "mode", value: "grid" },
    },
    {
      key: "show_border",
      type: "boolean",
      label: "Show Border",
      description: "Border between images",
      default: true,
      advanced: true,
    },
    {
      key: "border_width",
      type: "slider",
      label: "Border Width",
      description: "Border thickness",
      default: 2,
      min: 1,
      max: 10,
      step: 1,
      showWhen: { field: "show_border", value: true },
      advanced: true,
    },
    {
      key: "border_color",
      type: "string",
      label: "Border Color",
      description: "Border color",
      default: "#ffffff",
      showWhen: { field: "show_border", value: true },
      advanced: true,
    },
    {
      key: "compute_diff_stats",
      type: "boolean",
      label: "Compute Diff Stats",
      description: "MSE, PSNR, changed %",
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

export const webhookBlock: BlockDefinition = {
  type: "webhook",
  displayName: "Webhook",
  description: "Send results to external endpoint",
  category: "output",
  icon: "Webhook",
  canBeEnd: true,

  inputs: [
    { name: "data", type: "any", required: true, description: "Data to send" },
  ],

  outputs: [
    { name: "response", type: "object", description: "API response body" },
    { name: "status_code", type: "number", description: "HTTP status code" },
    { name: "success", type: "boolean", description: "Request success status" },
    { name: "request_id", type: "string", description: "Tracking ID" },
  ],

  configFields: [
    {
      key: "url",
      type: "string",
      label: "Webhook URL",
      description: "Endpoint to receive results",
      required: true,
      placeholder: "https://api.example.com/webhook",
    },
    {
      key: "method",
      type: "select",
      label: "HTTP Method",
      description: "Request method",
      default: "POST",
      options: [
        { value: "POST", label: "POST" },
        { value: "PUT", label: "PUT" },
        { value: "PATCH", label: "PATCH" },
      ],
    },
    {
      key: "auth_type",
      type: "select",
      label: "Authentication",
      description: "Authentication method",
      default: "none",
      options: [
        { value: "none", label: "No Authentication" },
        { value: "api_key", label: "API Key" },
        { value: "bearer", label: "Bearer Token" },
        { value: "basic", label: "Basic Auth" },
      ],
    },
    {
      key: "auth_value",
      type: "string",
      label: "Auth Value",
      description: "Authentication credential",
      showWhen: { field: "auth_type", operator: "neq", value: "none" },
    },
    {
      key: "retry_enabled",
      type: "boolean",
      label: "Enable Retry",
      description: "Retry on failure with backoff",
      default: true,
    },
    {
      key: "timeout",
      type: "number",
      label: "Timeout (seconds)",
      description: "Request timeout",
      default: 30,
      min: 1,
      max: 300,
      advanced: true,
    },
    {
      key: "retry_max",
      type: "number",
      label: "Max Retries",
      description: "Maximum retry attempts",
      default: 3,
      min: 0,
      max: 10,
      showWhen: { field: "retry_enabled", value: true },
      advanced: true,
    },
    {
      key: "batch_enabled",
      type: "boolean",
      label: "Enable Batching",
      description: "Split large datasets into batches",
      default: false,
      advanced: true,
    },
    {
      key: "batch_size",
      type: "number",
      label: "Batch Size",
      description: "Items per batch",
      default: 100,
      min: 1,
      max: 10000,
      showWhen: { field: "batch_enabled", value: true },
      advanced: true,
    },
    {
      key: "fire_and_forget",
      type: "boolean",
      label: "Fire and Forget",
      description: "Don't wait for response",
      default: false,
      advanced: true,
    },
    {
      key: "include_metadata",
      type: "boolean",
      label: "Include Metadata",
      description: "Add workflow ID, timestamp",
      default: true,
      advanced: true,
    },
    {
      key: "data_field",
      type: "string",
      label: "Data Field Name",
      description: "Field name in request body",
      default: "data",
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

export const aggregationBlock: BlockDefinition = {
  type: "aggregation",
  displayName: "Aggregation",
  description: "Aggregate and summarize data",
  category: "output",
  icon: "BarChart3",
  canBeEnd: true,

  inputs: [
    { name: "data", type: "array", required: true, description: "Data to aggregate" },
  ],

  outputs: [
    { name: "result", type: "any", description: "Aggregated result" },
    { name: "stats", type: "object", description: "Statistics if calculated" },
    { name: "count", type: "number", description: "Number of items processed" },
  ],

  configFields: [
    {
      key: "operation",
      type: "select",
      label: "Operation",
      description: "Aggregation operation",
      default: "flatten",
      options: [
        { value: "flatten", label: "Flatten Nested Arrays" },
        { value: "group", label: "Group By Field" },
        { value: "stats", label: "Calculate Statistics" },
        { value: "top_n", label: "Top N Items" },
        { value: "pivot", label: "Pivot Table" },
        { value: "dedupe", label: "Remove Duplicates" },
      ],
    },
    // Flatten config
    {
      key: "flatten_child_field",
      type: "string",
      label: "Child Field",
      description: "Field containing nested array",
      default: "matches",
      showWhen: { field: "operation", value: "flatten" },
    },
    {
      key: "flatten_parent_prefix",
      type: "string",
      label: "Parent Prefix",
      description: "Prefix for parent fields",
      default: "parent_",
      showWhen: { field: "operation", value: "flatten" },
    },
    {
      key: "flatten_add_index",
      type: "boolean",
      label: "Add Index Columns",
      description: "Include row indices",
      default: true,
      showWhen: { field: "operation", value: "flatten" },
    },
    // Group config
    {
      key: "group_by",
      type: "string",
      label: "Group By Field",
      description: "Field to group by",
      default: "class_name",
      showWhen: { field: "operation", value: "group" },
    },
    {
      key: "group_agg_func",
      type: "select",
      label: "Aggregation Function",
      description: "How to aggregate groups",
      default: "count",
      options: [
        { value: "count", label: "Count" },
        { value: "sum", label: "Sum" },
        { value: "avg", label: "Average" },
        { value: "min", label: "Minimum" },
        { value: "max", label: "Maximum" },
        { value: "collect", label: "Collect Values" },
      ],
      showWhen: { field: "operation", value: "group" },
    },
    {
      key: "group_agg_field",
      type: "string",
      label: "Aggregate Field",
      description: "Field to aggregate (optional)",
      placeholder: "confidence",
      showWhen: { field: "operation", value: "group" },
    },
    {
      key: "group_sort_order",
      type: "select",
      label: "Sort Order",
      description: "Result ordering",
      default: "desc",
      options: [
        { value: "desc", label: "Descending (highest first)" },
        { value: "asc", label: "Ascending (lowest first)" },
      ],
      showWhen: { field: "operation", value: "group" },
    },
    // Stats config
    {
      key: "stats_fields",
      type: "string",
      label: "Fields to Analyze",
      description: "Comma-separated numeric fields",
      default: "confidence, score",
      showWhen: { field: "operation", value: "stats" },
    },
    {
      key: "stats_include_distribution",
      type: "boolean",
      label: "Include Distribution",
      description: "Add percentiles (p25, p50, p75, p90)",
      default: false,
      showWhen: { field: "operation", value: "stats" },
    },
    // Top N config
    {
      key: "top_n_count",
      type: "number",
      label: "Number of Items (N)",
      description: "How many items to return",
      default: 10,
      min: 1,
      max: 10000,
      showWhen: { field: "operation", value: "top_n" },
    },
    {
      key: "top_n_by",
      type: "string",
      label: "Sort By Field",
      description: "Field to sort by",
      default: "confidence",
      showWhen: { field: "operation", value: "top_n" },
    },
    {
      key: "top_n_order",
      type: "select",
      label: "Order",
      description: "Sort direction",
      default: "desc",
      options: [
        { value: "desc", label: "Highest First" },
        { value: "asc", label: "Lowest First" },
      ],
      showWhen: { field: "operation", value: "top_n" },
    },
    {
      key: "top_n_group_by",
      type: "string",
      label: "Per Group (optional)",
      description: "Get top N per group",
      placeholder: "class_name",
      showWhen: { field: "operation", value: "top_n" },
    },
    // Dedupe config
    {
      key: "dedupe_by",
      type: "string",
      label: "Deduplicate By Field",
      description: "Field for uniqueness check",
      default: "id",
      showWhen: { field: "operation", value: "dedupe" },
    },
    {
      key: "dedupe_keep",
      type: "select",
      label: "Keep Strategy",
      description: "Which duplicate to keep",
      default: "first",
      options: [
        { value: "first", label: "Keep First" },
        { value: "last", label: "Keep Last" },
        { value: "highest", label: "Highest Value" },
        { value: "lowest", label: "Lowest Value" },
      ],
      showWhen: { field: "operation", value: "dedupe" },
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
