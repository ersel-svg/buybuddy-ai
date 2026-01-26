/**
 * Logic Block Definitions
 *
 * Control flow blocks: condition, foreach, collect, map.
 */

import type { BlockDefinition } from "../types";

export const conditionBlock: BlockDefinition = {
  type: "condition",
  displayName: "Condition",
  description: "Branch based on conditions",
  category: "logic",
  icon: "GitBranch",

  inputs: [
    { name: "value", type: "any", required: true, description: "Value to evaluate" },
    { name: "compare_to", type: "any", required: false, description: "Value to compare against" },
    { name: "context", type: "object", required: false, description: "Additional context" },
  ],

  outputs: [
    { name: "true_output", type: "any", description: "Output when condition is true" },
    { name: "false_output", type: "any", description: "Output when condition is false" },
    { name: "result", type: "boolean", description: "Condition result" },
    { name: "matched_conditions", type: "array", description: "Which conditions matched" },
  ],

  configFields: [
    {
      key: "field",
      type: "string",
      label: "Field",
      description: "Field to evaluate (use dot notation for nested)",
      placeholder: "confidence, label, detections.length",
      supportsParams: true,
    },
    {
      key: "operator",
      type: "select",
      label: "Operator",
      description: "Comparison operator",
      default: "gt",
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
        { value: "matches", label: "Matches Regex" },
        { value: "exists", label: "Exists" },
        { value: "empty", label: "Is Empty" },
      ],
    },
    {
      key: "value",
      type: "string",
      label: "Compare Value",
      description: "Value to compare against",
      supportsParams: true,
    },
    {
      key: "true_value",
      type: "string",
      label: "True Output",
      description: "Value to output when condition is true",
      default: "pass",
      supportsParams: true,
      advanced: true,
    },
    {
      key: "false_value",
      type: "string",
      label: "False Output",
      description: "Value to output when condition is false",
      default: "fail",
      supportsParams: true,
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

export const foreachBlock: BlockDefinition = {
  type: "foreach",
  displayName: "For Each",
  description: "Iterate over array items",
  category: "logic",
  icon: "Repeat",

  inputs: [
    { name: "items", type: "array", required: true, description: "Array to iterate over" },
    { name: "context", type: "any", required: false, description: "Context passed to each iteration" },
  ],

  outputs: [
    { name: "item", type: "any", description: "Current item in iteration" },
    { name: "index", type: "number", description: "Current index (0-based)" },
    { name: "total", type: "number", description: "Total number of items" },
    { name: "context", type: "any", description: "Passed through context" },
    { name: "is_first", type: "boolean", description: "True if first item" },
    { name: "is_last", type: "boolean", description: "True if last item" },
  ],

  configFields: [
    {
      key: "max_iterations",
      type: "number",
      label: "Max Iterations",
      description: "Maximum items to process (0 = all)",
      default: 0,
      min: 0,
      max: 10000,
    },
    {
      key: "parallel",
      type: "boolean",
      label: "Parallel Processing",
      description: "Process items in parallel",
      default: false,
      advanced: true,
    },
    {
      key: "batch_size",
      type: "number",
      label: "Batch Size",
      description: "Items per batch for parallel processing",
      default: 10,
      min: 1,
      max: 100,
      showWhen: { field: "parallel", value: true },
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

export const collectBlock: BlockDefinition = {
  type: "collect",
  displayName: "Collect",
  description: "Collect iteration results",
  category: "logic",
  icon: "ListPlus",

  inputs: [
    { name: "item", type: "any", required: true, description: "Item from each iteration" },
    { name: "index", type: "number", required: false, description: "Index from ForEach" },
  ],

  outputs: [
    { name: "results", type: "array", description: "Collected results array" },
    { name: "count", type: "number", description: "Number of collected items" },
  ],

  configFields: [
    {
      key: "filter_nulls",
      type: "boolean",
      label: "Filter Nulls",
      description: "Exclude null/undefined values",
      default: true,
    },
    {
      key: "flatten",
      type: "boolean",
      label: "Flatten",
      description: "Flatten nested arrays",
      default: false,
    },
    {
      key: "unique",
      type: "boolean",
      label: "Unique Only",
      description: "Remove duplicate values",
      default: false,
      advanced: true,
    },
    {
      key: "unique_key",
      type: "string",
      label: "Unique Key",
      description: "Field to use for uniqueness check",
      placeholder: "id",
      showWhen: { field: "unique", value: true },
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

export const mapBlock: BlockDefinition = {
  type: "map",
  displayName: "Map",
  description: "Transform array items",
  category: "logic",
  icon: "Shuffle",

  inputs: [
    { name: "items", type: "array", required: true, description: "Array to transform" },
  ],

  outputs: [
    { name: "results", type: "array", description: "Transformed items" },
    { name: "count", type: "number", description: "Number of items" },
  ],

  configFields: [
    {
      key: "expression",
      type: "code",
      label: "Transform Expression",
      description: "JMESPath or JavaScript expression",
      default: "item",
      language: "jmespath",
    },
    {
      key: "extract_field",
      type: "string",
      label: "Extract Field",
      description: "Simple field extraction (alternative to expression)",
      placeholder: "confidence, label",
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
