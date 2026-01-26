/**
 * Block Registry
 *
 * Central registry for all workflow blocks.
 * Single source of truth for block definitions, defaults, and validation.
 */

import type {
  BlockDefinition,
  BlockCategory,
  BlockRegistry,
  ValidationResult,
  ConfigField,
} from "./types";
import { ALL_BLOCKS } from "./blocks";

// =============================================================================
// Registry Implementation
// =============================================================================

class BlockRegistryImpl implements BlockRegistry {
  private blocks: Map<string, BlockDefinition> = new Map();

  constructor(initialBlocks: BlockDefinition[] = []) {
    for (const block of initialBlocks) {
      this.register(block);
    }
  }

  register(block: BlockDefinition): void {
    if (this.blocks.has(block.type)) {
      console.warn(`Block type "${block.type}" is being overwritten`);
    }
    this.blocks.set(block.type, block);
  }

  get(type: string): BlockDefinition | undefined {
    return this.blocks.get(type);
  }

  getAll(): BlockDefinition[] {
    return Array.from(this.blocks.values());
  }

  getByCategory(category: BlockCategory): BlockDefinition[] {
    return this.getAll().filter((block) => block.category === category);
  }

  getDefaultConfig(type: string): Record<string, unknown> {
    const block = this.get(type);
    if (!block) {
      console.warn(`Unknown block type: ${type}`);
      return {};
    }
    return { ...block.defaultConfig };
  }

  validateConfig(type: string, config: Record<string, unknown>): ValidationResult {
    const block = this.get(type);
    if (!block) {
      return {
        valid: false,
        errors: [{ field: "_type", message: `Unknown block type: ${type}` }],
      };
    }

    // Run block's custom validation if exists
    if (block.validate) {
      return block.validate(config);
    }

    // Default validation: check required fields
    const errors: Array<{ field: string; message: string }> = [];

    for (const field of block.configFields) {
      if (field.required && (config[field.key] === undefined || config[field.key] === "")) {
        errors.push({
          field: field.key,
          message: `${field.label} is required`,
        });
      }
    }

    return { valid: errors.length === 0, errors };
  }
}

// =============================================================================
// Singleton Instance
// =============================================================================

// Create and populate the registry
const registry = new BlockRegistryImpl(ALL_BLOCKS);

// Export singleton
export const blockRegistry: BlockRegistry = registry;

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Get default config for a block type.
 * Use this when creating new nodes.
 */
export function getBlockDefaults(type: string): Record<string, unknown> {
  return blockRegistry.getDefaultConfig(type);
}

/**
 * Validate config for a block type.
 * Use this before saving or executing.
 */
export function validateBlockConfig(
  type: string,
  config: Record<string, unknown>
): ValidationResult {
  return blockRegistry.validateConfig(type, config);
}

/**
 * Get all blocks as palette items for UI.
 * Returns blocks grouped by category with display info.
 */
export function getBlockPalette(): Array<{
  type: string;
  display_name: string;
  description: string;
  category: BlockCategory;
  icon: string;
}> {
  return blockRegistry.getAll().map((block) => ({
    type: block.type,
    display_name: block.displayName,
    description: block.description,
    category: block.category,
    icon: block.icon,
  }));
}

/**
 * Get blocks grouped by category for palette UI.
 */
export function getBlocksByCategory(): Record<BlockCategory, BlockDefinition[]> {
  const categories: BlockCategory[] = ["input", "model", "transform", "logic", "output"];
  const result: Record<BlockCategory, BlockDefinition[]> = {
    input: [],
    model: [],
    transform: [],
    logic: [],
    output: [],
  };

  for (const category of categories) {
    result[category] = blockRegistry.getByCategory(category);
  }

  return result;
}

/**
 * Check if a field should be shown based on showWhen condition.
 */
export function shouldShowField(
  field: ConfigField,
  config: Record<string, unknown>
): boolean {
  if (!field.showWhen) return true;

  const { field: dependentField, value, operator = "eq" } = field.showWhen;
  const currentValue = config[dependentField];

  switch (operator) {
    case "eq":
      return currentValue === value;
    case "neq":
      return currentValue !== value;
    case "includes":
      if (Array.isArray(value)) {
        return value.some((v) =>
          String(currentValue).toLowerCase().includes(String(v).toLowerCase())
        );
      }
      return String(currentValue).toLowerCase().includes(String(value).toLowerCase());
    case "gt":
      return Number(currentValue) > Number(value);
    case "lt":
      return Number(currentValue) < Number(value);
    default:
      return true;
  }
}

/**
 * Get config fields for a block, optionally filtered for advanced.
 */
export function getConfigFields(
  type: string,
  options: { includeAdvanced?: boolean } = {}
): ConfigField[] {
  const { includeAdvanced = true } = options;
  const block = blockRegistry.get(type);

  if (!block) return [];

  if (includeAdvanced) {
    return block.configFields;
  }

  return block.configFields.filter((field) => !field.advanced);
}

/**
 * Merge partial config with defaults.
 * Use this when loading saved workflows.
 */
export function mergeWithDefaults(
  type: string,
  partialConfig: Record<string, unknown>
): Record<string, unknown> {
  const defaults = getBlockDefaults(type);
  return { ...defaults, ...partialConfig };
}

// =============================================================================
// Export Types
// =============================================================================

export type {
  BlockDefinition,
  BlockCategory,
  BlockRegistry,
  ValidationResult,
  ConfigField,
  PortDefinition,
  PortType,
} from "./types";
