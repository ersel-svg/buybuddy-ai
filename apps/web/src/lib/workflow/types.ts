/**
 * Workflow Block Registry - Type Definitions
 *
 * Central type definitions for the workflow block system.
 * All blocks, configs, and validations are typed here.
 */

// =============================================================================
// Port Types
// =============================================================================

/** Supported data types for block ports */
export type PortType =
  | "image"
  | "array"
  | "number"
  | "string"
  | "boolean"
  | "object"
  | "any";

/** Port definition for inputs/outputs */
export interface PortDefinition {
  name: string;
  type: PortType;
  required?: boolean;
  description?: string;
  /** Default value for optional inputs */
  default?: unknown;
}

// =============================================================================
// Config Field Types
// =============================================================================

/** Types of config fields */
export type ConfigFieldType =
  | "string"
  | "number"
  | "boolean"
  | "select"
  | "model"
  | "slider"
  | "array"
  | "object"
  | "code";

/** Base config field definition */
interface BaseConfigField {
  key: string;
  label: string;
  description?: string;
  required?: boolean;
  advanced?: boolean;
  /** Condition to show this field */
  showWhen?: {
    field: string;
    value: unknown;
    operator?: "eq" | "neq" | "includes" | "gt" | "lt";
  };
}

/** String input field */
export interface StringConfigField extends BaseConfigField {
  type: "string";
  default?: string;
  placeholder?: string;
  /** Support parameter references like {{ params.x }} */
  supportsParams?: boolean;
}

/** Number input field */
export interface NumberConfigField extends BaseConfigField {
  type: "number";
  default?: number;
  min?: number;
  max?: number;
  step?: number;
  supportsParams?: boolean;
}

/** Boolean toggle field */
export interface BooleanConfigField extends BaseConfigField {
  type: "boolean";
  default?: boolean;
}

/** Dropdown select field */
export interface SelectConfigField extends BaseConfigField {
  type: "select";
  default?: string;
  options: Array<{
    value: string;
    label: string;
    description?: string;
  }>;
}

/** Model selector field */
export interface ModelConfigField extends BaseConfigField {
  type: "model";
  default?: string;
  /** Model category filter */
  category: "detection" | "classification" | "embedding" | "segmentation";
  /** Default model source */
  defaultSource?: "pretrained" | "trained";
}

/** Slider field */
export interface SliderConfigField extends BaseConfigField {
  type: "slider";
  default?: number;
  min: number;
  max: number;
  step: number;
  /** Format function for display */
  formatValue?: (value: number) => string;
  supportsParams?: boolean;
}

/** Array input field */
export interface ArrayConfigField extends BaseConfigField {
  type: "array";
  default?: unknown[];
  itemType?: PortType;
  supportsParams?: boolean;
}

/** Object/JSON field */
export interface ObjectConfigField extends BaseConfigField {
  type: "object";
  default?: Record<string, unknown>;
}

/** Code/expression field */
export interface CodeConfigField extends BaseConfigField {
  type: "code";
  default?: string;
  language?: "javascript" | "json" | "jmespath";
}

/** Union of all config field types */
export type ConfigField =
  | StringConfigField
  | NumberConfigField
  | BooleanConfigField
  | SelectConfigField
  | ModelConfigField
  | SliderConfigField
  | ArrayConfigField
  | ObjectConfigField
  | CodeConfigField;

// =============================================================================
// Block Definition
// =============================================================================

/** Block categories */
export type BlockCategory = "input" | "model" | "transform" | "logic" | "output";

/** Complete block definition */
export interface BlockDefinition {
  /** Unique block type identifier */
  type: string;
  /** Display name in UI */
  displayName: string;
  /** Short description */
  description: string;
  /** Block category for grouping */
  category: BlockCategory;
  /** Icon name from lucide-react */
  icon: string;
  /** Input port definitions */
  inputs: PortDefinition[];
  /** Output port definitions */
  outputs: PortDefinition[];
  /** Config field definitions */
  configFields: ConfigField[];
  /** Default config values (computed from configFields) */
  defaultConfig: Record<string, unknown>;
  /** Validation function */
  validate?: (config: Record<string, unknown>) => ValidationResult;
  /** Whether this block can be the start of a workflow */
  canBeStart?: boolean;
  /** Whether this block can be the end of a workflow */
  canBeEnd?: boolean;
}

/** Validation result */
export interface ValidationResult {
  valid: boolean;
  errors: Array<{
    field: string;
    message: string;
  }>;
  warnings?: Array<{
    field: string;
    message: string;
  }>;
}

// =============================================================================
// Runtime Types (for workflow execution)
// =============================================================================

/** Node instance in a workflow */
export interface WorkflowNodeData {
  label: string;
  type: string;
  category: BlockCategory;
  config: Record<string, unknown>;
}

/** Edge connection in a workflow */
export interface WorkflowEdge {
  id: string;
  source: string;
  target: string;
  sourceHandle?: string;
  targetHandle?: string;
}

// =============================================================================
// Registry Types
// =============================================================================

/** Block registry interface */
export interface BlockRegistry {
  /** Get all registered blocks */
  getAll(): BlockDefinition[];
  /** Get block by type */
  get(type: string): BlockDefinition | undefined;
  /** Get blocks by category */
  getByCategory(category: BlockCategory): BlockDefinition[];
  /** Get default config for a block type */
  getDefaultConfig(type: string): Record<string, unknown>;
  /** Validate config for a block type */
  validateConfig(type: string, config: Record<string, unknown>): ValidationResult;
  /** Register a new block */
  register(block: BlockDefinition): void;
}
