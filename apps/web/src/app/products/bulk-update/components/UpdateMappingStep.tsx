"use client";

import { useState, useMemo, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { Plus, Trash2, ArrowLeft, ArrowRight, AlertCircle } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
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
import { Alert, AlertDescription } from "@/components/ui/alert";
import { apiClient } from "@/lib/api-client";

// Auto-mapping rules for column detection (supports Turkish)
const AUTO_MAPPINGS: Record<string, string[]> = {
  // Identifiers
  barcode: ["barcode", "barkod", "bar_code", "ean13", "gtin", "upc_code"],
  sku: ["sku", "stock_code", "stok_kodu", "model_code", "urun_kodu", "product_code"],
  upc: ["upc", "upc_code", "upc_kodu"],
  ean: ["ean", "ean_code", "ean_kodu"],
  short_code: ["short_code", "kisa_kod", "code", "kod"],

  // Product Info
  product_name: ["product_name", "name", "title", "urun_adi", "ürün_adı", "product", "urun"],
  brand_name: ["brand", "brand_name", "marka", "marka_adi", "marka_adı"],
  sub_brand: ["sub_brand", "subbrand", "alt_marka", "sub_marka"],
  category: ["category", "kategori", "cat", "product_category", "urun_kategorisi"],
  variant_flavor: ["flavor", "variant", "aroma", "tat", "cesit", "çeşit", "variant_flavor"],
  container_type: ["container", "ambalaj", "package", "paket_tipi", "container_type"],
  net_quantity: ["quantity", "miktar", "net_weight", "agirlik", "size", "net_quantity", "net_miktar"],
  manufacturer_country: ["country", "ulke", "ülke", "origin", "mensei", "manufacturer_country"],
  marketing_description: ["description", "aciklama", "açıklama", "desc", "tanim", "tanım"],

  // Structured
  pack_configuration: ["pack", "paket", "multipack", "pack_config"],
  claims: ["claims", "iddialar", "ozellikler", "özellikler"],
  visibility_score: ["score", "puan", "visibility", "visibility_score"],
};

export interface FieldMapping {
  sourceColumn: string;
  targetField: string;
}

export interface UpdateMappingConfig {
  identifierColumn: string;
  fieldMappings: FieldMapping[];
}

interface UpdateMappingStepProps {
  columns: string[];
  preview: Record<string, unknown>[];
  onComplete: (config: UpdateMappingConfig) => void;
  onBack: () => void;
}

export function UpdateMappingStep({
  columns,
  preview,
  onComplete,
  onBack,
}: UpdateMappingStepProps) {
  const [identifierColumn, setIdentifierColumn] = useState<string>("");
  const [fieldMappings, setFieldMappings] = useState<FieldMapping[]>([]);

  // Fetch system fields
  const { data: systemFields } = useQuery({
    queryKey: ["bulk-update-system-fields"],
    queryFn: () => apiClient.getBulkUpdateSystemFields(),
  });

  // Auto-detect identifier column (barcode) on mount
  useEffect(() => {
    if (identifierColumn || columns.length === 0) return;

    const barcodeMatches = AUTO_MAPPINGS.barcode;
    for (const col of columns) {
      const normalized = col.toLowerCase().replace(/[_\s-]/g, "");
      if (barcodeMatches.some((m) => normalized.includes(m.replace(/[_\s-]/g, "")))) {
        setIdentifierColumn(col);
        break;
      }
    }

    // Fallback to first column
    if (!identifierColumn && columns.length > 0) {
      setIdentifierColumn(columns[0]);
    }
  }, [columns, identifierColumn]);

  // Auto-detect field mappings on mount
  useEffect(() => {
    if (fieldMappings.length > 0 || columns.length === 0 || !systemFields) return;

    const autoMappings: FieldMapping[] = [];
    const usedColumns = new Set<string>();
    const usedFields = new Set<string>();

    // Skip barcode as it's the identifier
    usedFields.add("barcode");

    for (const col of columns) {
      const normalized = col.toLowerCase().replace(/[_\s-]/g, "");

      for (const [field, patterns] of Object.entries(AUTO_MAPPINGS)) {
        if (field === "barcode") continue; // Skip barcode
        if (usedFields.has(field)) continue;
        if (usedColumns.has(col)) continue;

        const fieldInfo = systemFields.find((f) => f.id === field);
        if (!fieldInfo?.editable) continue;

        if (patterns.some((p) => normalized.includes(p.replace(/[_\s-]/g, "")))) {
          autoMappings.push({
            sourceColumn: col,
            targetField: field,
          });
          usedColumns.add(col);
          usedFields.add(field);
          break;
        }
      }
    }

    if (autoMappings.length > 0) {
      setFieldMappings(autoMappings);
    }
  }, [columns, systemFields, fieldMappings.length]);

  // Group system fields by category
  const groupedFields = useMemo(() => {
    if (!systemFields) return {};

    const groups: Record<string, typeof systemFields> = {};
    for (const field of systemFields) {
      if (!groups[field.group]) {
        groups[field.group] = [];
      }
      groups[field.group].push(field);
    }
    return groups;
  }, [systemFields]);

  // Get available columns (not used in other mappings)
  const getAvailableColumns = (currentMapping?: FieldMapping) => {
    const usedColumns = new Set(
      fieldMappings
        .filter((m) => m !== currentMapping)
        .map((m) => m.sourceColumn)
    );
    return columns.filter((col) => !usedColumns.has(col) && col !== identifierColumn);
  };

  // Get available fields (not used in other mappings)
  const getAvailableFields = (currentMapping?: FieldMapping) => {
    const usedFields = new Set(
      fieldMappings
        .filter((m) => m !== currentMapping)
        .map((m) => m.targetField)
    );
    return (systemFields || []).filter(
      (f) => f.editable && !usedFields.has(f.id)
    );
  };

  const addMapping = () => {
    const availableCols = getAvailableColumns();
    const availableFields = getAvailableFields();

    if (availableCols.length === 0 || availableFields.length === 0) return;

    setFieldMappings([
      ...fieldMappings,
      {
        sourceColumn: availableCols[0],
        targetField: availableFields[0].id,
      },
    ]);
  };

  const updateMapping = (index: number, field: keyof FieldMapping, value: string) => {
    const newMappings = [...fieldMappings];
    newMappings[index] = { ...newMappings[index], [field]: value };
    setFieldMappings(newMappings);
  };

  const removeMapping = (index: number) => {
    setFieldMappings(fieldMappings.filter((_, i) => i !== index));
  };

  const handleSubmit = () => {
    if (!identifierColumn || fieldMappings.length === 0) return;

    onComplete({
      identifierColumn,
      fieldMappings,
    });
  };

  const isValid = identifierColumn && fieldMappings.length > 0;

  const groupLabels: Record<string, string> = {
    identifiers: "Identifiers",
    product_info: "Product Information",
    structured: "Structured Data",
    quality: "Quality Metrics",
  };

  return (
    <div className="space-y-6">
      {/* Identifier Column Selection */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">1. Select Identifier Column</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-sm text-muted-foreground">
            Choose the column that contains the product barcode to match against existing products.
          </p>
          <div className="max-w-sm">
            <Label htmlFor="identifier-column">Identifier Column</Label>
            <Select value={identifierColumn} onValueChange={setIdentifierColumn}>
              <SelectTrigger id="identifier-column" className="mt-1.5">
                <SelectValue placeholder="Select column" />
              </SelectTrigger>
              <SelectContent>
                {columns.map((col) => (
                  <SelectItem key={col} value={col}>
                    {col}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {/* Field Mappings */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="text-lg">2. Map Fields to Update</CardTitle>
          <Button
            variant="outline"
            size="sm"
            onClick={addMapping}
            disabled={
              getAvailableColumns().length === 0 ||
              getAvailableFields().length === 0
            }
          >
            <Plus className="h-4 w-4 mr-1" />
            Add Mapping
          </Button>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-sm text-muted-foreground">
            Map your file columns to the product fields you want to update.
          </p>

          {fieldMappings.length === 0 ? (
            <Alert>
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>
                No field mappings configured. Add at least one mapping to continue.
              </AlertDescription>
            </Alert>
          ) : (
            <div className="space-y-3">
              {fieldMappings.map((mapping, index) => (
                <div
                  key={index}
                  className="flex items-center gap-3 p-3 border rounded-lg bg-muted/30"
                >
                  <div className="flex-1 grid grid-cols-2 gap-3">
                    <div>
                      <Label className="text-xs text-muted-foreground">
                        Source Column
                      </Label>
                      <Select
                        value={mapping.sourceColumn}
                        onValueChange={(v) => updateMapping(index, "sourceColumn", v)}
                      >
                        <SelectTrigger className="mt-1">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {Array.from(new Set([mapping.sourceColumn, ...getAvailableColumns(mapping)])).map(
                            (col) => (
                              <SelectItem key={col} value={col}>
                                {col}
                              </SelectItem>
                            )
                          )}
                        </SelectContent>
                      </Select>
                    </div>
                    <div>
                      <Label className="text-xs text-muted-foreground">
                        Target Field
                      </Label>
                      <Select
                        value={mapping.targetField}
                        onValueChange={(v) => updateMapping(index, "targetField", v)}
                      >
                        <SelectTrigger className="mt-1">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {Object.entries(groupedFields).map(([group, fields]) => (
                            <div key={group}>
                              <div className="px-2 py-1.5 text-xs font-semibold text-muted-foreground">
                                {groupLabels[group] || group}
                              </div>
                              {fields
                                .filter(
                                  (f) =>
                                    f.editable &&
                                    (f.id === mapping.targetField ||
                                      !fieldMappings.some(
                                        (m, i) =>
                                          i !== index && m.targetField === f.id
                                      ))
                                )
                                .map((field) => (
                                  <SelectItem key={field.id} value={field.id}>
                                    {field.label}
                                  </SelectItem>
                                ))}
                            </div>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => removeMapping(index)}
                    className="text-destructive hover:text-destructive"
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Data Preview */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Data Preview</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="border rounded-lg overflow-auto max-h-[300px]">
            <Table>
              <TableHeader>
                <TableRow>
                  {columns.map((col) => {
                    const isIdentifier = col === identifierColumn;
                    const mapping = fieldMappings.find(
                      (m) => m.sourceColumn === col
                    );

                    return (
                      <TableHead key={col} className="whitespace-nowrap">
                        <div className="flex flex-col gap-1">
                          <span>{col}</span>
                          {isIdentifier && (
                            <Badge variant="secondary" className="w-fit text-xs">
                              Identifier
                            </Badge>
                          )}
                          {mapping && (
                            <Badge className="w-fit text-xs">
                              {systemFields?.find((f) => f.id === mapping.targetField)
                                ?.label || mapping.targetField}
                            </Badge>
                          )}
                        </div>
                      </TableHead>
                    );
                  })}
                </TableRow>
              </TableHeader>
              <TableBody>
                {preview.slice(0, 5).map((row, i) => (
                  <TableRow key={i}>
                    {columns.map((col) => (
                      <TableCell key={col} className="whitespace-nowrap">
                        {String(row[col] ?? "")}
                      </TableCell>
                    ))}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
          <p className="text-xs text-muted-foreground mt-2">
            Showing first 5 rows of your data
          </p>
        </CardContent>
      </Card>

      {/* Actions */}
      <div className="flex justify-between">
        <Button variant="outline" onClick={onBack}>
          <ArrowLeft className="h-4 w-4 mr-2" />
          Back
        </Button>
        <Button onClick={handleSubmit} disabled={!isValid}>
          Preview Changes
          <ArrowRight className="h-4 w-4 ml-2" />
        </Button>
      </div>
    </div>
  );
}
