"use client";

import { useState, useMemo } from "react";
import { ChevronDown, Settings2 } from "lucide-react";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { blockRegistry } from "@/lib/workflow/registry";
import type { ConfigField } from "@/lib/workflow/types";
import { ConfigFieldRenderer, shouldShowField } from "./ConfigFieldRenderer";

// =============================================================================
// Types
// =============================================================================

export interface NodeConfig {
  [key: string]: unknown;
}

export interface NodeConfigPanelProps {
  nodeType: string;
  config: NodeConfig;
  onConfigChange: (keyOrObject: string | Record<string, unknown>, value?: unknown) => void;
}

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Group config fields into normal and advanced sections
 */
function groupFields(fields: ConfigField[]): {
  normalFields: ConfigField[];
  advancedFields: ConfigField[];
} {
  const normalFields: ConfigField[] = [];
  const advancedFields: ConfigField[] = [];

  for (const field of fields) {
    if (field.advanced) {
      advancedFields.push(field);
    } else {
      normalFields.push(field);
    }
  }

  return { normalFields, advancedFields };
}

// =============================================================================
// Main Component
// =============================================================================

/**
 * NodeConfigPanel - Dynamic config panel using Block Registry
 *
 * Features:
 * - Renders config fields based on block definition
 * - Groups fields into normal and advanced sections
 * - Handles showWhen conditions
 * - Supports parameter references
 * - Batch updates for related fields (e.g., model_id + model_source)
 */
export function NodeConfigPanel({
  nodeType,
  config,
  onConfigChange,
}: NodeConfigPanelProps) {
  const [advancedOpen, setAdvancedOpen] = useState(false);

  // Get block definition from registry
  const blockDef = useMemo(() => {
    return blockRegistry.get(nodeType);
  }, [nodeType]);

  // Group fields
  const { normalFields, advancedFields } = useMemo(() => {
    if (!blockDef) return { normalFields: [], advancedFields: [] };
    return groupFields(blockDef.configFields);
  }, [blockDef]);

  // Filter visible fields based on showWhen conditions
  const visibleNormalFields = useMemo(() => {
    return normalFields.filter((field) => shouldShowField(field, config));
  }, [normalFields, config]);

  const visibleAdvancedFields = useMemo(() => {
    return advancedFields.filter((field) => shouldShowField(field, config));
  }, [advancedFields, config]);

  // Handle single field change
  const handleFieldChange = (key: string, value: unknown) => {
    onConfigChange(key, value);
  };

  // Handle batch change (for model fields that need to update multiple keys)
  const handleBatchChange = (updates: Record<string, unknown>) => {
    onConfigChange(updates);
  };

  // Unknown block type fallback
  if (!blockDef) {
    return (
      <div className="p-4 text-center text-muted-foreground">
        <p className="text-sm">Unknown block type: {nodeType}</p>
        <p className="text-xs mt-1">No configuration available</p>
      </div>
    );
  }

  // No config fields
  if (normalFields.length === 0 && advancedFields.length === 0) {
    return (
      <div className="p-4 text-center text-muted-foreground">
        <p className="text-sm">No configuration options</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Normal Fields */}
      {visibleNormalFields.map((field) => (
        <ConfigFieldRenderer
          key={field.key}
          field={field}
          value={config[field.key]}
          onChange={(value) => handleFieldChange(field.key, value)}
          allValues={config}
          onBatchChange={handleBatchChange}
        />
      ))}

      {/* Advanced Section */}
      {visibleAdvancedFields.length > 0 && (
        <Collapsible open={advancedOpen} onOpenChange={setAdvancedOpen}>
          <CollapsibleTrigger className="flex w-full items-center justify-between py-2 text-xs font-medium text-muted-foreground hover:text-foreground transition-colors">
            <div className="flex items-center gap-2">
              <Settings2 className="h-3.5 w-3.5" />
              Advanced Settings
            </div>
            <ChevronDown
              className={`h-4 w-4 transition-transform ${
                advancedOpen ? "rotate-180" : ""
              }`}
            />
          </CollapsibleTrigger>
          <CollapsibleContent className="space-y-4 pt-2">
            {visibleAdvancedFields.map((field) => (
              <ConfigFieldRenderer
                key={field.key}
                field={field}
                value={config[field.key]}
                onChange={(value) => handleFieldChange(field.key, value)}
                allValues={config}
                onBatchChange={handleBatchChange}
              />
            ))}
          </CollapsibleContent>
        </Collapsible>
      )}
    </div>
  );
}
