"use client";

import { useState, useEffect } from "react";
import { Plus, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

interface FieldEntry {
  key: string;
  value: string;
}

interface CustomFieldsEditorProps {
  fields: Record<string, string>;
  isEditing: boolean;
  onUpdate: (fields: Record<string, string>) => void;
}

export function CustomFieldsEditor({
  fields,
  isEditing,
  onUpdate,
}: CustomFieldsEditorProps) {
  const [entries, setEntries] = useState<FieldEntry[]>([]);

  // Initialize entries from fields when entering edit mode
  useEffect(() => {
    if (isEditing) {
      setEntries(
        Object.entries(fields || {}).map(([key, value]) => ({ key, value }))
      );
    }
  }, [fields, isEditing]);

  const handleAdd = () => {
    const updated = [...entries, { key: "", value: "" }];
    setEntries(updated);
  };

  const handleRemove = (index: number) => {
    const updated = entries.filter((_, i) => i !== index);
    setEntries(updated);
    // Convert to object and notify parent
    const fieldsObj: Record<string, string> = {};
    updated.forEach((e) => {
      if (e.key.trim()) {
        fieldsObj[e.key.trim()] = e.value;
      }
    });
    onUpdate(fieldsObj);
  };

  const handleChange = (
    index: number,
    field: "key" | "value",
    newValue: string
  ) => {
    const updated = [...entries];
    updated[index] = { ...updated[index], [field]: newValue };
    setEntries(updated);

    // Convert to object and notify parent
    const fieldsObj: Record<string, string> = {};
    updated.forEach((e) => {
      if (e.key.trim()) {
        fieldsObj[e.key.trim()] = e.value;
      }
    });
    onUpdate(fieldsObj);
  };

  // View mode
  if (!isEditing) {
    const fieldEntries = Object.entries(fields || {});

    return (
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Custom Fields</CardTitle>
          <CardDescription>Additional product attributes</CardDescription>
        </CardHeader>
        <CardContent>
          {fieldEntries.length === 0 ? (
            <p className="text-sm text-muted-foreground">No custom fields</p>
          ) : (
            <div className="space-y-2">
              {fieldEntries.map(([key, value]) => (
                <div
                  key={key}
                  className="flex items-center justify-between p-2 bg-gray-50 rounded"
                >
                  <span className="text-sm font-medium text-gray-600">
                    {key}
                  </span>
                  <span className="text-sm font-mono">{value}</span>
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
            <CardTitle className="text-base">Custom Fields</CardTitle>
            <CardDescription>Add custom key-value pairs</CardDescription>
          </div>
          <Button variant="outline" size="sm" onClick={handleAdd}>
            <Plus className="h-4 w-4 mr-1" />
            Add
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        {entries.length === 0 ? (
          <div className="text-center py-4">
            <p className="text-sm text-muted-foreground mb-2">
              No custom fields yet
            </p>
            <Button variant="outline" size="sm" onClick={handleAdd}>
              <Plus className="h-4 w-4 mr-1" />
              Add First Field
            </Button>
          </div>
        ) : (
          <div className="space-y-2">
            {entries.map((entry, index) => (
              <div key={index} className="flex items-center gap-2">
                <Input
                  value={entry.key}
                  onChange={(e) => handleChange(index, "key", e.target.value)}
                  placeholder="Field name"
                  className="w-1/3"
                />
                <span className="text-gray-400">=</span>
                <Input
                  value={entry.value}
                  onChange={(e) => handleChange(index, "value", e.target.value)}
                  placeholder="Value"
                  className="flex-1"
                />
                <Button
                  variant="ghost"
                  size="icon"
                  className="text-red-500 hover:text-red-600 hover:bg-red-50"
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
