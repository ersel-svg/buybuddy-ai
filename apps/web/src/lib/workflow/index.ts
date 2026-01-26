/**
 * Workflow Library
 *
 * Central exports for the workflow block system.
 */

// Registry and utilities
export {
  blockRegistry,
  getBlockDefaults,
  validateBlockConfig,
  getBlockPalette,
  getBlocksByCategory,
  shouldShowField,
  getConfigFields,
  mergeWithDefaults,
} from "./registry";

// Types
export type {
  BlockDefinition,
  BlockCategory,
  BlockRegistry,
  ValidationResult,
  ConfigField,
  PortDefinition,
  PortType,
  WorkflowNodeData,
  WorkflowEdge,
  // Config field types
  StringConfigField,
  NumberConfigField,
  BooleanConfigField,
  SelectConfigField,
  ModelConfigField,
  SliderConfigField,
  ArrayConfigField,
  ObjectConfigField,
  CodeConfigField,
} from "./types";

// Individual blocks (for direct access if needed)
export * from "./blocks";
