"use client";

import { useState } from "react";
import {
  ArrowLeft,
  ArrowRight,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  ChevronDown,
  ChevronRight,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Alert, AlertDescription } from "@/components/ui/alert";

interface PreviewResponse {
  matches: Array<{
    row_index: number;
    product_id: string;
    barcode: string;
    current_values: Record<string, unknown>;
    new_values: Record<string, unknown>;
    product_field_changes: string[];
    identifier_field_changes: string[];
  }>;
  not_found: Array<{
    row_index: number;
    identifier_value: string;
    source_row: Record<string, unknown>;
  }>;
  validation_errors: Array<{
    row_index: number;
    field: string;
    value: unknown;
    error: string;
  }>;
  summary: {
    total_rows: number;
    matched: number;
    not_found: number;
    validation_errors: number;
    will_update: number;
  };
}

interface UpdatePreviewStepProps {
  previewData: PreviewResponse;
  columns: string[];
  onExecute: () => void;
  onBack: () => void;
}

// Field labels for display
const FIELD_LABELS: Record<string, string> = {
  product_name: "Product Name",
  brand_name: "Brand Name",
  sub_brand: "Sub Brand",
  category: "Category",
  variant_flavor: "Variant/Flavor",
  container_type: "Container Type",
  net_quantity: "Net Quantity",
  manufacturer_country: "Country",
  marketing_description: "Description",
  pack_configuration: "Pack Config",
  nutrition_facts: "Nutrition Facts",
  claims: "Claims",
  visibility_score: "Visibility Score",
  sku: "SKU",
  upc: "UPC",
  ean: "EAN",
  short_code: "Short Code",
};

export function UpdatePreviewStep({
  previewData,
  columns,
  onExecute,
  onBack,
}: UpdatePreviewStepProps) {
  const [expandedRows, setExpandedRows] = useState<Set<number>>(new Set());

  const toggleRow = (index: number) => {
    const newExpanded = new Set(expandedRows);
    if (newExpanded.has(index)) {
      newExpanded.delete(index);
    } else {
      newExpanded.add(index);
    }
    setExpandedRows(newExpanded);
  };

  const { summary, matches, not_found, validation_errors } = previewData;

  return (
    <div className="space-y-6">
      {/* Summary Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="pt-6">
            <div className="text-2xl font-bold">{summary.total_rows}</div>
            <div className="text-sm text-muted-foreground">Total Rows</div>
          </CardContent>
        </Card>
        <Card className="border-green-200 bg-green-50 dark:bg-green-950/20">
          <CardContent className="pt-6">
            <div className="text-2xl font-bold text-green-600">
              {summary.will_update}
            </div>
            <div className="text-sm text-muted-foreground">Will Update</div>
          </CardContent>
        </Card>
        <Card className="border-yellow-200 bg-yellow-50 dark:bg-yellow-950/20">
          <CardContent className="pt-6">
            <div className="text-2xl font-bold text-yellow-600">
              {summary.not_found}
            </div>
            <div className="text-sm text-muted-foreground">Not Found</div>
          </CardContent>
        </Card>
        <Card className="border-red-200 bg-red-50 dark:bg-red-950/20">
          <CardContent className="pt-6">
            <div className="text-2xl font-bold text-red-600">
              {summary.validation_errors}
            </div>
            <div className="text-sm text-muted-foreground">Validation Errors</div>
          </CardContent>
        </Card>
      </div>

      {/* Tabs for different views */}
      <Tabs defaultValue="updates" className="space-y-4">
        <TabsList>
          <TabsTrigger value="updates" className="gap-2">
            <CheckCircle2 className="h-4 w-4" />
            Updates ({matches.length})
          </TabsTrigger>
          <TabsTrigger value="not-found" className="gap-2">
            <XCircle className="h-4 w-4" />
            Not Found ({not_found.length})
          </TabsTrigger>
          {validation_errors.length > 0 && (
            <TabsTrigger value="errors" className="gap-2">
              <AlertTriangle className="h-4 w-4" />
              Errors ({validation_errors.length})
            </TabsTrigger>
          )}
        </TabsList>

        {/* Updates Tab */}
        <TabsContent value="updates">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Changes to Apply</CardTitle>
            </CardHeader>
            <CardContent>
              {matches.length === 0 ? (
                <Alert>
                  <AlertDescription>
                    No products will be updated. Check your field mappings.
                  </AlertDescription>
                </Alert>
              ) : (
                <div className="space-y-2 max-h-[400px] overflow-auto">
                  {matches.map((match, index) => {
                    const isExpanded = expandedRows.has(index);
                    const allChanges = [
                      ...match.product_field_changes,
                      ...match.identifier_field_changes,
                    ];

                    return (
                      <Collapsible
                        key={index}
                        open={isExpanded}
                        onOpenChange={() => toggleRow(index)}
                      >
                        <CollapsibleTrigger asChild>
                          <div className="flex items-center justify-between p-3 border rounded-lg cursor-pointer hover:bg-muted/50">
                            <div className="flex items-center gap-3">
                              {isExpanded ? (
                                <ChevronDown className="h-4 w-4" />
                              ) : (
                                <ChevronRight className="h-4 w-4" />
                              )}
                              <div>
                                <div className="font-medium">
                                  {match.barcode}
                                </div>
                                <div className="text-sm text-muted-foreground">
                                  Row {match.row_index + 1}
                                </div>
                              </div>
                            </div>
                            <div className="flex items-center gap-2">
                              <Badge variant="outline">
                                {allChanges.length} field
                                {allChanges.length !== 1 ? "s" : ""}
                              </Badge>
                            </div>
                          </div>
                        </CollapsibleTrigger>
                        <CollapsibleContent>
                          <div className="mt-2 ml-7 p-3 border rounded-lg bg-muted/30">
                            <Table>
                              <TableHeader>
                                <TableRow>
                                  <TableHead className="w-[200px]">Field</TableHead>
                                  <TableHead>Current Value</TableHead>
                                  <TableHead>New Value</TableHead>
                                </TableRow>
                              </TableHeader>
                              <TableBody>
                                {allChanges.map((field) => (
                                  <TableRow key={field}>
                                    <TableCell className="font-medium">
                                      {FIELD_LABELS[field] || field}
                                    </TableCell>
                                    <TableCell className="text-muted-foreground">
                                      {formatValue(match.current_values[field])}
                                    </TableCell>
                                    <TableCell className="text-green-600 font-medium">
                                      {formatValue(match.new_values[field])}
                                    </TableCell>
                                  </TableRow>
                                ))}
                              </TableBody>
                            </Table>
                          </div>
                        </CollapsibleContent>
                      </Collapsible>
                    );
                  })}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Not Found Tab */}
        <TabsContent value="not-found">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Products Not Found</CardTitle>
            </CardHeader>
            <CardContent>
              {not_found.length === 0 ? (
                <Alert>
                  <AlertDescription>
                    All products were found in the database.
                  </AlertDescription>
                </Alert>
              ) : (
                <div className="border rounded-lg overflow-auto max-h-[400px]">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead className="w-[80px]">Row</TableHead>
                        <TableHead>Identifier</TableHead>
                        <TableHead>Source Data</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {not_found.map((item, index) => (
                        <TableRow key={index}>
                          <TableCell>{item.row_index + 1}</TableCell>
                          <TableCell className="font-mono">
                            {item.identifier_value || "(empty)"}
                          </TableCell>
                          <TableCell className="text-sm text-muted-foreground">
                            {Object.entries(item.source_row)
                              .slice(0, 3)
                              .map(([k, v]) => `${k}: ${v}`)
                              .join(", ")}
                            {Object.keys(item.source_row).length > 3 && "..."}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Validation Errors Tab */}
        {validation_errors.length > 0 && (
          <TabsContent value="errors">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Validation Errors</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="border rounded-lg overflow-auto max-h-[400px]">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead className="w-[80px]">Row</TableHead>
                        <TableHead>Field</TableHead>
                        <TableHead>Value</TableHead>
                        <TableHead>Error</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {validation_errors.map((error, index) => (
                        <TableRow key={index}>
                          <TableCell>{error.row_index + 1}</TableCell>
                          <TableCell className="font-medium">
                            {FIELD_LABELS[error.field] || error.field}
                          </TableCell>
                          <TableCell className="font-mono text-sm">
                            {String(error.value)}
                          </TableCell>
                          <TableCell className="text-red-600">
                            {error.error}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        )}
      </Tabs>

      {/* Warning if no updates */}
      {matches.length === 0 && (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>
            No products will be updated. Please check your identifier column and
            field mappings, then go back to make changes.
          </AlertDescription>
        </Alert>
      )}

      {/* Actions */}
      <div className="flex justify-between">
        <Button variant="outline" onClick={onBack}>
          <ArrowLeft className="h-4 w-4 mr-2" />
          Back to Mapping
        </Button>
        <Button
          onClick={onExecute}
          disabled={matches.length === 0}
          className="bg-green-600 hover:bg-green-700"
        >
          Apply {matches.length} Update{matches.length !== 1 ? "s" : ""}
          <ArrowRight className="h-4 w-4 ml-2" />
        </Button>
      </div>
    </div>
  );
}

function formatValue(value: unknown): string {
  if (value === null || value === undefined) {
    return "(empty)";
  }
  if (typeof value === "object") {
    return JSON.stringify(value);
  }
  return String(value);
}
