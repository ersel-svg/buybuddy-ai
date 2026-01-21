"use client";

import { useState, useMemo } from "react";
import { Plus, Trash2, ArrowRight, GripVertical } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import type { ParsedFile } from "./FileUploadStep";

export interface MatchRule {
  id: string;
  sourceColumn: string;
  targetField: string;
  priority: number;
}

export interface MappingConfig {
  matchRules: MatchRule[];
}

interface FieldMappingStepProps {
  parsedFile: ParsedFile;
  onComplete: (config: MappingConfig) => void;
  onBack: () => void;
}

const SYSTEM_FIELDS = [
  // Primary identifiers
  { value: "barcode", label: "Barcode", description: "Product barcode" },
  { value: "sku", label: "SKU", description: "Stock Keeping Unit" },
  { value: "upc", label: "UPC", description: "Universal Product Code" },
  { value: "ean", label: "EAN", description: "European Article Number" },
  { value: "short_code", label: "Short Code", description: "Short product code" },
  // Product info fields
  { value: "product_name", label: "Product Name", description: "Full product name" },
  { value: "brand_name", label: "Brand Name", description: "Product brand" },
  { value: "sub_brand", label: "Sub Brand", description: "Sub-brand or product line" },
  { value: "category", label: "Category", description: "Product category" },
  { value: "variant_flavor", label: "Variant/Flavor", description: "Product variant or flavor" },
  { value: "container_type", label: "Container Type", description: "Package type (bottle, can, box, etc.)" },
  { value: "net_quantity", label: "Net Quantity", description: "Product size/weight" },
  { value: "manufacturer_country", label: "Country", description: "Manufacturing country" },
  { value: "marketing_description", label: "Description", description: "Product marketing description" },
];

export function FieldMappingStep({
  parsedFile,
  onComplete,
  onBack,
}: FieldMappingStepProps) {
  // Auto-suggest initial mapping based on column names
  const initialRules = useMemo(() => {
    const suggestions: MatchRule[] = [];
    let priority = 1;

    const columnLower = parsedFile.columns.map((c) => c.toLowerCase());

    // Try to auto-map common column names
    const mappings: Record<string, string[]> = {
      barcode: ["barcode", "barkod", "bar_code", "ean", "upc", "gtin", "product_code", "productcode", "ürün_kodu", "urun_kodu"],
      sku: ["sku", "sku_code", "stock_code", "stok_kodu", "model_code"],
      product_name: ["product_name", "productname", "urun_adi", "ürün_adı", "name", "title", "product", "urun"],
      brand_name: ["brand", "brand_name", "marka", "marka_adi"],
    };

    for (const [targetField, keywords] of Object.entries(mappings)) {
      for (const keyword of keywords) {
        const index = columnLower.findIndex((c) => c.includes(keyword));
        if (index !== -1) {
          // Check if this column is already mapped
          const alreadyMapped = suggestions.some(
            (s) => s.sourceColumn === parsedFile.columns[index]
          );
          if (!alreadyMapped) {
            suggestions.push({
              id: crypto.randomUUID(),
              sourceColumn: parsedFile.columns[index],
              targetField,
              priority: priority++,
            });
            break;
          }
        }
      }
    }

    if (suggestions.length > 0) {
      return suggestions;
    }
    // Return one empty rule
    return [
      {
        id: crypto.randomUUID(),
        sourceColumn: "",
        targetField: "",
        priority: 1,
      },
    ];
  }, [parsedFile.columns]);

  const [matchRules, setMatchRules] = useState<MatchRule[]>(initialRules);

  const addRule = () => {
    setMatchRules((prev) => [
      ...prev,
      {
        id: crypto.randomUUID(),
        sourceColumn: "",
        targetField: "",
        priority: prev.length + 1,
      },
    ]);
  };

  const removeRule = (id: string) => {
    setMatchRules((prev) => {
      const filtered = prev.filter((r) => r.id !== id);
      return filtered.map((r, i) => ({ ...r, priority: i + 1 }));
    });
  };

  const updateRule = (
    id: string,
    field: "sourceColumn" | "targetField",
    value: string
  ) => {
    setMatchRules((prev) =>
      prev.map((r) => (r.id === id ? { ...r, [field]: value } : r))
    );
  };

  const handleContinue = () => {
    const validRules = matchRules.filter(
      (r) => r.sourceColumn && r.targetField
    );
    if (validRules.length === 0) {
      return;
    }
    onComplete({ matchRules: validRules });
  };

  const validRulesCount = matchRules.filter(
    (r) => r.sourceColumn && r.targetField
  ).length;

  // Get used columns and fields for disabling
  const usedColumns = new Set(matchRules.map((r) => r.sourceColumn).filter(Boolean));
  const usedFields = new Set(matchRules.map((r) => r.targetField).filter(Boolean));

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-xl font-semibold">Map Fields</h2>
        <p className="text-muted-foreground mt-1">
          Map your file columns to system fields for matching
        </p>
      </div>

      {/* File Info */}
      <Card>
        <CardHeader className="py-3">
          <CardTitle className="text-sm font-medium flex items-center gap-2">
            <Badge variant="secondary">{parsedFile.fileName}</Badge>
            <span className="text-muted-foreground font-normal">
              {parsedFile.totalRows.toLocaleString()} rows,{" "}
              {parsedFile.columns.length} columns
            </span>
          </CardTitle>
        </CardHeader>
      </Card>

      {/* Mapping Rules */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Match Rules</CardTitle>
          <p className="text-sm text-muted-foreground">
            Rules are checked in priority order. First match wins.
          </p>
        </CardHeader>
        <CardContent className="space-y-4">
          {matchRules.map((rule, index) => (
            <div
              key={rule.id}
              className="flex items-center gap-3 p-3 bg-muted/50 rounded-lg"
            >
              <div className="flex items-center gap-2 text-muted-foreground">
                <GripVertical className="h-4 w-4" />
                <span className="text-sm font-medium w-6">{index + 1}</span>
              </div>

              <Select
                value={rule.sourceColumn}
                onValueChange={(v) => updateRule(rule.id, "sourceColumn", v)}
              >
                <SelectTrigger className="w-[200px]">
                  <SelectValue placeholder="Select column" />
                </SelectTrigger>
                <SelectContent>
                  {parsedFile.columns.map((col) => (
                    <SelectItem
                      key={col}
                      value={col}
                      disabled={usedColumns.has(col) && rule.sourceColumn !== col}
                    >
                      {col}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <ArrowRight className="h-4 w-4 text-muted-foreground flex-shrink-0" />

              <Select
                value={rule.targetField}
                onValueChange={(v) => updateRule(rule.id, "targetField", v)}
              >
                <SelectTrigger className="w-[200px]">
                  <SelectValue placeholder="Select system field" />
                </SelectTrigger>
                <SelectContent>
                  {SYSTEM_FIELDS.map((field) => (
                    <SelectItem
                      key={field.value}
                      value={field.value}
                      disabled={usedFields.has(field.value) && rule.targetField !== field.value}
                    >
                      <div className="flex flex-col">
                        <span>{field.label}</span>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Button
                variant="ghost"
                size="icon"
                onClick={() => removeRule(rule.id)}
                disabled={matchRules.length === 1}
                className="text-muted-foreground hover:text-destructive"
              >
                <Trash2 className="h-4 w-4" />
              </Button>
            </div>
          ))}

          <Button
            variant="outline"
            size="sm"
            onClick={addRule}
            className="w-full"
            disabled={matchRules.length >= Math.min(parsedFile.columns.length, SYSTEM_FIELDS.length)}
          >
            <Plus className="h-4 w-4 mr-2" />
            Add Match Rule
          </Button>
        </CardContent>
      </Card>

      {/* Preview */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Data Preview</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="rounded-md border overflow-auto max-h-[300px]">
            <Table>
              <TableHeader>
                <TableRow>
                  {parsedFile.columns.map((col) => (
                    <TableHead key={col} className="whitespace-nowrap">
                      {col}
                      {usedColumns.has(col) && (
                        <Badge variant="secondary" className="ml-2 text-xs">
                          Mapped
                        </Badge>
                      )}
                    </TableHead>
                  ))}
                </TableRow>
              </TableHeader>
              <TableBody>
                {parsedFile.preview.map((row, i) => (
                  <TableRow key={i}>
                    {parsedFile.columns.map((col) => (
                      <TableCell key={col} className="whitespace-nowrap">
                        {String(row[col] || "")}
                      </TableCell>
                    ))}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>

      {/* Actions */}
      <div className="flex justify-between">
        <Button variant="outline" onClick={onBack}>
          Back
        </Button>
        <Button onClick={handleContinue} disabled={validRulesCount === 0}>
          Start Matching ({validRulesCount} rule{validRulesCount !== 1 ? "s" : ""})
        </Button>
      </div>
    </div>
  );
}
