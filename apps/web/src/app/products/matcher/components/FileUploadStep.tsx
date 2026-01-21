"use client";

import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, FileSpreadsheet, AlertCircle, Check } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { apiClient } from "@/lib/api-client";

export interface ParsedFile {
  fileName: string;
  columns: string[];
  totalRows: number;
  preview: Record<string, unknown>[];
  rows: Record<string, unknown>[];
}

interface FileUploadStepProps {
  onComplete: (data: ParsedFile) => void;
}

export function FileUploadStep({ onComplete }: FileUploadStepProps) {
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploadedFile, setUploadedFile] = useState<{
    name: string;
    size: number;
  } | null>(null);

  const onDrop = useCallback(
    async (acceptedFiles: File[]) => {
      const file = acceptedFiles[0];
      if (!file) return;

      setError(null);
      setIsUploading(true);
      setUploadedFile({ name: file.name, size: file.size });

      try {
        const result = await apiClient.uploadProductMatcherFile(file);

        // Parse the file locally to get all rows (not just preview)
        const allRows = await parseFileLocally(file);

        onComplete({
          fileName: result.file_name,
          columns: result.columns,
          totalRows: result.total_rows,
          preview: result.preview,
          rows: allRows,
        });
      } catch (err) {
        setError(err instanceof Error ? err.message : "Upload failed");
        setUploadedFile(null);
      } finally {
        setIsUploading(false);
      }
    },
    [onComplete]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "text/csv": [".csv"],
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [
        ".xlsx",
      ],
      "application/vnd.ms-excel": [".xls"],
    },
    maxFiles: 1,
    maxSize: 10 * 1024 * 1024, // 10MB
  });

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-xl font-semibold">Upload Product List</h2>
        <p className="text-muted-foreground mt-1">
          Upload a CSV or Excel file containing product information
        </p>
      </div>

      <Card>
        <CardContent className="pt-6">
          <div
            {...getRootProps()}
            className={`
              border-2 border-dashed rounded-lg p-12 text-center cursor-pointer
              transition-colors duration-200
              ${isDragActive ? "border-primary bg-primary/5" : "border-muted-foreground/25 hover:border-primary/50"}
              ${isUploading ? "pointer-events-none opacity-50" : ""}
            `}
          >
            <input {...getInputProps()} />

            {isUploading ? (
              <div className="flex flex-col items-center gap-4">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary" />
                <p className="text-muted-foreground">
                  Parsing {uploadedFile?.name}...
                </p>
              </div>
            ) : uploadedFile ? (
              <div className="flex flex-col items-center gap-4">
                <div className="h-12 w-12 rounded-full bg-green-100 dark:bg-green-900/30 flex items-center justify-center">
                  <Check className="h-6 w-6 text-green-600 dark:text-green-400" />
                </div>
                <div>
                  <p className="font-medium">{uploadedFile.name}</p>
                  <p className="text-sm text-muted-foreground">
                    {formatFileSize(uploadedFile.size)}
                  </p>
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center gap-4">
                {isDragActive ? (
                  <Upload className="h-12 w-12 text-primary" />
                ) : (
                  <FileSpreadsheet className="h-12 w-12 text-muted-foreground" />
                )}
                <div>
                  <p className="font-medium">
                    {isDragActive
                      ? "Drop the file here"
                      : "Drag & drop your file here"}
                  </p>
                  <p className="text-sm text-muted-foreground mt-1">
                    or click to browse
                  </p>
                </div>
                <p className="text-xs text-muted-foreground">
                  Supported formats: .csv, .xlsx, .xls (max 10MB)
                </p>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}
    </div>
  );
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

async function parseFileLocally(file: File): Promise<Record<string, unknown>[]> {
  const text = await file.text();

  // Simple CSV parsing (for Excel, the backend already parsed it)
  if (file.name.toLowerCase().endsWith(".csv")) {
    const lines = text.split("\n").filter((line) => line.trim());
    if (lines.length < 2) return [];

    // Detect delimiter
    const firstLine = lines[0];
    let delimiter = ",";
    if (firstLine.split(";").length > firstLine.split(",").length) {
      delimiter = ";";
    } else if (firstLine.split("\t").length > firstLine.split(",").length) {
      delimiter = "\t";
    }

    const headers = parseCSVLine(lines[0], delimiter);
    const rows: Record<string, unknown>[] = [];

    for (let i = 1; i < lines.length; i++) {
      const values = parseCSVLine(lines[i], delimiter);
      const row: Record<string, unknown> = {};
      headers.forEach((header, index) => {
        row[header] = values[index] || "";
      });
      rows.push(row);
    }

    return rows;
  }

  // For Excel files, we need to use the xlsx library
  // For now, return empty and rely on backend parsing
  return [];
}

function parseCSVLine(line: string, delimiter: string): string[] {
  const result: string[] = [];
  let current = "";
  let inQuotes = false;

  for (let i = 0; i < line.length; i++) {
    const char = line[i];

    if (char === '"') {
      if (inQuotes && line[i + 1] === '"') {
        current += '"';
        i++;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (char === delimiter && !inQuotes) {
      result.push(current.trim());
      current = "";
    } else {
      current += char;
    }
  }

  result.push(current.trim());
  return result;
}
