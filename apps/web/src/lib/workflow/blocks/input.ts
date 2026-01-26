/**
 * Input Block Definitions
 *
 * Image Input and Parameter Input blocks.
 */

import type { BlockDefinition } from "../types";

export const imageInputBlock: BlockDefinition = {
  type: "image_input",
  displayName: "Image Input",
  description: "Receive image input for the workflow",
  category: "input",
  icon: "Image",
  canBeStart: true,

  inputs: [],

  outputs: [
    { name: "image", type: "image", description: "The input image" },
    { name: "image_url", type: "string", description: "URL of the image (if provided)" },
    { name: "width", type: "number", description: "Image width in pixels" },
    { name: "height", type: "number", description: "Image height in pixels" },
    { name: "original_width", type: "number", description: "Original width before processing" },
    { name: "original_height", type: "number", description: "Original height before processing" },
  ],

  configFields: [
    {
      key: "max_size",
      type: "number",
      label: "Max Size",
      description: "Maximum dimension (resize if larger)",
      default: 1920,
      min: 256,
      max: 4096,
    },
    {
      key: "preserve_aspect",
      type: "boolean",
      label: "Preserve Aspect Ratio",
      description: "Keep original aspect ratio when resizing",
      default: true,
    },
    {
      key: "format",
      type: "select",
      label: "Output Format",
      description: "Image format for processing",
      default: "rgb",
      options: [
        { value: "rgb", label: "RGB" },
        { value: "bgr", label: "BGR" },
        { value: "grayscale", label: "Grayscale" },
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

export const parameterInputBlock: BlockDefinition = {
  type: "parameter_input",
  displayName: "Parameters",
  description: "Receive workflow parameters",
  category: "input",
  icon: "Settings2",
  canBeStart: true,

  inputs: [],

  outputs: [
    { name: "parameters", type: "object", description: "All workflow parameters as object" },
  ],

  configFields: [],

  defaultConfig: {},
};
