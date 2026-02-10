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
        // For small files (<1000 rows), request all rows from backend
        // For larger files, parse locally (CSV) or use backend async processing
        const isExcel = file.name.toLowerCase().endsWith('.xlsx') || 
                        file.name.toLowerCase().endsWith('.xls');
        
        // Request all rows from backend for Excel files or small CSV files
        const includeAllRows = isExcel || file.size < 500 * 1024; // < 500KB
        
        const result = await apiClient.uploadProductMatcherFile(file, includeAllRows);

        let allRows: Record<string, unknown>[] = [];

        // If backend returned all rows, use them
        if (result.all_rows && result.all_rows.length > 0) {
          allRows = result.all_rows;
        } else {
          // Otherwise, parse locally (for CSV files)
          allRows = await parseFileLocally(file);
          
          // If local parsing failed and we have no rows, use preview as fallback
          // This shouldn't happen, but provides a safety net
          if (allRows.length === 0 && result.preview.length > 0) {
            console.warn("Local parsing returned no rows, using preview as fallback");
            allRows = result.preview;
          }
        }

        // Validate we have rows
        if (allRows.length === 0) {
          throw new Error("No data rows found in file. Please check the file format.");
        }

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
  // Only parse CSV files locally - Excel files should be parsed by backend
  if (!file.name.toLowerCase().endsWith(".csv")) {
    // For Excel files, return empty array - backend should have provided all_rows
    return [];
  }

  try {
    const text = await file.text();
    const lines = text.split("\n").filter((line) => line.trim());
    
    if (lines.length < 2) {
      return [];
    }

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
        // Preserve values as-is (including leading zeros for barcodes)
        const value = values[index];
        row[header] = value !== undefined ? value : "";
      });
      
      // Only add non-empty rows
      if (Object.values(row).some(v => v && String(v).trim())) {
        rows.push(row);
      }
    }

    return rows;
  } catch (error) {
    console.error("Error parsing CSV file locally:", error);
    return [];
  }
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
