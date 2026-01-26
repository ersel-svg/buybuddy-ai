/**
 * Block Definitions Index
 *
 * Export all block definitions for the registry.
 */

// Input blocks
export { imageInputBlock, parameterInputBlock } from "./input";

// Model blocks
export { detectionBlock } from "./detection";
export { classificationBlock } from "./classification";
export { embeddingBlock } from "./embedding";
export { segmentationBlock } from "./segmentation";
export { similaritySearchBlock } from "./similarity-search";

// Transform blocks
export {
  cropBlock,
  resizeBlock,
  tileBlock,
  stitchBlock,
  filterBlock,
  rotateFlipBlock,
  normalizeBlock,
  smoothingBlock,
} from "./transform";

// Logic blocks
export {
  conditionBlock,
  foreachBlock,
  collectBlock,
  mapBlock,
} from "./logic";

// Output blocks
export {
  jsonOutputBlock,
  gridBuilderBlock,
  drawBoxesBlock,
  blurRegionBlock,
  drawMasksBlock,
  heatmapBlock,
  comparisonBlock,
  webhookBlock,
  aggregationBlock,
} from "./output";

// Re-export all blocks as array for easy registration
import { imageInputBlock, parameterInputBlock } from "./input";
import { detectionBlock } from "./detection";
import { classificationBlock } from "./classification";
import { embeddingBlock } from "./embedding";
import { segmentationBlock } from "./segmentation";
import { similaritySearchBlock } from "./similarity-search";
import { cropBlock, resizeBlock, tileBlock, stitchBlock, filterBlock, rotateFlipBlock, normalizeBlock, smoothingBlock } from "./transform";
import { conditionBlock, foreachBlock, collectBlock, mapBlock } from "./logic";
import { jsonOutputBlock, gridBuilderBlock, drawBoxesBlock, blurRegionBlock, drawMasksBlock, heatmapBlock, comparisonBlock, webhookBlock, aggregationBlock } from "./output";

import type { BlockDefinition } from "../types";

export const ALL_BLOCKS: BlockDefinition[] = [
  // Input
  imageInputBlock,
  parameterInputBlock,
  // Model
  detectionBlock,
  classificationBlock,
  embeddingBlock,
  segmentationBlock,
  similaritySearchBlock,
  // Transform
  cropBlock,
  resizeBlock,
  tileBlock,
  stitchBlock,
  filterBlock,
  rotateFlipBlock,
  normalizeBlock,
  smoothingBlock,
  // Logic
  conditionBlock,
  foreachBlock,
  collectBlock,
  mapBlock,
  // Output
  jsonOutputBlock,
  gridBuilderBlock,
  drawBoxesBlock,
  blurRegionBlock,
  drawMasksBlock,
  heatmapBlock,
  comparisonBlock,
  webhookBlock,
  aggregationBlock,
];
