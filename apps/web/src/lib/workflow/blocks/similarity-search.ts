/**
 * Similarity Search Block Definition
 *
 * Search vector database for similar items using embeddings.
 */

import type { BlockDefinition, ValidationResult } from "../types";

export const similaritySearchBlock: BlockDefinition = {
  type: "similarity_search",
  displayName: "Similarity Search",
  description: "Find similar items in vector database",
  category: "model",
  icon: "Search",

  inputs: [
    {
      name: "embedding",
      type: "array",
      required: false,
      description: "Single query embedding",
    },
    {
      name: "embeddings",
      type: "array",
      required: false,
      description: "Multiple query embeddings (batch)",
    },
    {
      name: "filter",
      type: "object",
      required: false,
      description: "Metadata filter criteria",
    },
    {
      name: "text_query",
      type: "string",
      required: false,
      description: "Text query for hybrid search",
    },
  ],

  outputs: [
    { name: "matches", type: "array", description: "All matching items with scores" },
    { name: "top_match", type: "object", description: "Best matching item" },
    { name: "match_ids", type: "array", description: "IDs of matched items" },
    { name: "match_scores", type: "array", description: "Similarity scores (0-1)" },
    { name: "match_payloads", type: "array", description: "Metadata of matched items" },
    { name: "match_count", type: "number", description: "Number of matches found" },
    { name: "has_matches", type: "boolean", description: "Whether any matches found" },
    { name: "avg_score", type: "number", description: "Average similarity score" },
  ],

  configFields: [
    // Collection Selection
    {
      key: "collection",
      type: "string",
      label: "Collection",
      description: "Vector collection to search",
      default: "products",
      required: true,
    },

    // Search Settings
    {
      key: "top_k",
      type: "number",
      label: "Top-K Results",
      description: "Maximum number of results to return",
      default: 10,
      min: 1,
      max: 100,
    },
    {
      key: "min_score",
      type: "slider",
      label: "Minimum Score",
      description: "Only return results above this similarity",
      default: 0.0,
      min: 0,
      max: 1,
      step: 0.05,
      formatValue: (v) => v.toFixed(2),
      supportsParams: true,
    },

    // Filtering
    {
      key: "filter_field",
      type: "string",
      label: "Filter Field",
      description: "Metadata field to filter on",
      placeholder: "category",
      advanced: true,
    },
    {
      key: "filter_value",
      type: "string",
      label: "Filter Value",
      description: "Value to match in filter field",
      supportsParams: true,
      advanced: true,
    },

    // Advanced Options
    {
      key: "distance_metric",
      type: "select",
      label: "Distance Metric",
      description: "How to measure similarity",
      default: "cosine",
      options: [
        { value: "cosine", label: "Cosine Similarity" },
        { value: "euclidean", label: "Euclidean Distance" },
        { value: "dot", label: "Dot Product" },
      ],
      advanced: true,
    },
    {
      key: "deduplicate",
      type: "boolean",
      label: "Deduplicate",
      description: "Remove duplicate matches",
      default: true,
      advanced: true,
    },
    {
      key: "include_vectors",
      type: "boolean",
      label: "Include Vectors",
      description: "Return embedding vectors in results",
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

  validate(config: Record<string, unknown>): ValidationResult {
    const errors: Array<{ field: string; message: string }> = [];

    if (!config.collection) {
      errors.push({ field: "collection", message: "Collection is required" });
    }

    return { valid: errors.length === 0, errors };
  },
};
