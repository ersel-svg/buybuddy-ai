"use client";

import { useState, useEffect } from "react";
import { Plus, Trash2, Star, StarOff } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import type {
  ProductIdentifier,
  IdentifierType,
  ProductIdentifierCreate,
} from "@/types";
import { IDENTIFIER_TYPE_LABELS } from "@/types";

const IDENTIFIER_TYPES: { value: IdentifierType; label: string }[] = [
  { value: "barcode", label: "Barcode" },
  { value: "short_code", label: "Short Code" },
  { value: "sku", label: "SKU" },
  { value: "upc", label: "UPC" },
  { value: "ean", label: "EAN" },
  { value: "custom", label: "Custom" },
];

interface IdentifiersEditorProps {
  identifiers: ProductIdentifier[];
  isEditing: boolean;
  onUpdate: (identifiers: ProductIdentifierCreate[]) => void;
}

export function IdentifiersEditor({
  identifiers,
  isEditing,
  onUpdate,
}: IdentifiersEditorProps) {
  const [localIdentifiers, setLocalIdentifiers] = useState<
    ProductIdentifierCreate[]
  >([]);

  // Initialize from props when entering edit mode
  useEffect(() => {
    if (isEditing) {
      setLocalIdentifiers(
        identifiers.map((i) => ({
          identifier_type: i.identifier_type,
          identifier_value: i.identifier_value,
          custom_label: i.custom_label,
          is_primary: i.is_primary,
        }))
      );
    }
  }, [identifiers, isEditing]);

  const handleAdd = () => {
    const newIdentifier: ProductIdentifierCreate = {
      identifier_type: "barcode",
      identifier_value: "",
      is_primary: localIdentifiers.length === 0, // First one is primary
    };
    const updated = [...localIdentifiers, newIdentifier];
    setLocalIdentifiers(updated);
    onUpdate(updated);
  };

  const handleRemove = (index: number) => {
    const updated = localIdentifiers.filter((_, i) => i !== index);
    // If removing primary, make first one primary
    if (localIdentifiers[index].is_primary && updated.length > 0) {
      updated[0].is_primary = true;
    }
    setLocalIdentifiers(updated);
    onUpdate(updated);
  };

  const handleChange = (
    index: number,
    field: keyof ProductIdentifierCreate,
    value: string | boolean
  ) => {
    const updated = [...localIdentifiers];
    updated[index] = { ...updated[index], [field]: value };
    setLocalIdentifiers(updated);
    onUpdate(updated);
  };

  const handleSetPrimary = (index: number) => {
    const updated = localIdentifiers.map((i, idx) => ({
      ...i,
      is_primary: idx === index,
    }));
    setLocalIdentifiers(updated);
    onUpdate(updated);
  };

  // View mode
  if (!isEditing) {
    return (
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Product Identifiers</CardTitle>
          <CardDescription>
            All identifiers associated with this product
          </CardDescription>
        </CardHeader>
        <CardContent>
          {identifiers.length === 0 ? (
            <p className="text-sm text-muted-foreground">No identifiers</p>
          ) : (
            <div className="space-y-2">
              {identifiers.map((identifier) => (
                <div
                  key={identifier.id}
                  className="flex items-center justify-between p-2 bg-gray-50 rounded"
                >
                  <div className="flex items-center gap-2">
                    <Badge variant="outline" className="text-xs">
                      {identifier.identifier_type === "custom"
                        ? identifier.custom_label || "Custom"
                        : IDENTIFIER_TYPE_LABELS[identifier.identifier_type]}
                    </Badge>
                    <span className="font-mono text-sm">
                      {identifier.identifier_value}
                    </span>
                  </div>
                  {identifier.is_primary && (
                    <Badge className="bg-yellow-100 text-yellow-800 hover:bg-yellow-100">
                      <Star className="h-3 w-3 mr-1 fill-yellow-600" />
                      Primary
                    </Badge>
                  )}
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    );
  }

  // Edit mode
  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-base">Product Identifiers</CardTitle>
            <CardDescription>
              Manage product identifiers (barcode, SKU, UPC, etc.)
            </CardDescription>
          </div>
          <Button variant="outline" size="sm" onClick={handleAdd}>
            <Plus className="h-4 w-4 mr-1" />
            Add
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        {localIdentifiers.length === 0 ? (
          <div className="text-center py-4">
            <p className="text-sm text-muted-foreground mb-2">
              No identifiers yet
            </p>
            <Button variant="outline" size="sm" onClick={handleAdd}>
              <Plus className="h-4 w-4 mr-1" />
              Add First Identifier
            </Button>
          </div>
        ) : (
          <div className="space-y-3">
            {localIdentifiers.map((identifier, index) => (
              <div
                key={index}
                className="flex items-start gap-2 p-3 border rounded-lg"
              >
                {/* Type selector */}
                <div className="w-28">
                  <Select
                    value={identifier.identifier_type}
                    onValueChange={(value) =>
                      handleChange(
                        index,
                        "identifier_type",
                        value as IdentifierType
                      )
                    }
                  >
                    <SelectTrigger className="h-9">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {IDENTIFIER_TYPES.map((type) => (
                        <SelectItem key={type.value} value={type.value}>
                          {type.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                {/* Custom label (only for custom type) */}
                {identifier.identifier_type === "custom" && (
                  <div className="w-24">
                    <Input
                      value={identifier.custom_label || ""}
                      onChange={(e) =>
                        handleChange(index, "custom_label", e.target.value)
                      }
                      placeholder="Label"
                      className="h-9"
                    />
                  </div>
                )}

                {/* Value input */}
                <div className="flex-1">
                  <Input
                    value={identifier.identifier_value}
                    onChange={(e) =>
                      handleChange(index, "identifier_value", e.target.value)
                    }
                    placeholder="Enter identifier value"
                    className="h-9 font-mono"
                  />
                </div>

                {/* Primary toggle */}
                <Button
                  variant={identifier.is_primary ? "default" : "ghost"}
                  size="icon"
                  className="h-9 w-9"
                  onClick={() => handleSetPrimary(index)}
                  title={
                    identifier.is_primary
                      ? "Primary identifier"
                      : "Set as primary"
                  }
                >
                  {identifier.is_primary ? (
                    <Star className="h-4 w-4 fill-yellow-400 text-yellow-400" />
                  ) : (
                    <StarOff className="h-4 w-4" />
                  )}
                </Button>

                {/* Delete button */}
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-9 w-9 text-red-500 hover:text-red-600 hover:bg-red-50"
                  onClick={() => handleRemove(index)}
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
