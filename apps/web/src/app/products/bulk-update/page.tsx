"use client";

import { useState } from "react";
import { FileSpreadsheet, GitCompare, Eye, CheckCircle, Loader2, Upload } from "lucide-react";
import { FileUploadStep, type ParsedFile } from "../matcher/components/FileUploadStep";
import { UpdateMappingStep, type UpdateMappingConfig } from "./components/UpdateMappingStep";
import { UpdatePreviewStep } from "./components/UpdatePreviewStep";
import { UpdateResultsStep } from "./components/UpdateResultsStep";
import { JobProgressModal } from "@/components/common/job-progress-modal";
import { apiClient } from "@/lib/api-client";
import { toast } from "sonner";

type Step = "upload" | "mapping" | "preview" | "executing" | "results";

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

interface ExecuteResponse {
  success: boolean;
  updated_count: number;
  failed: Array<{ product_id: string; error: string }>;
  execution_time_ms: number;
}

const STEPS = [
  { id: "upload", label: "Upload", icon: FileSpreadsheet },
  { id: "mapping", label: "Mapping", icon: GitCompare },
  { id: "preview", label: "Preview", icon: Eye },
  { id: "results", label: "Results", icon: CheckCircle },
] as const;

const ASYNC_THRESHOLD = 50;

export default function BulkUpdatePage() {
  const [currentStep, setCurrentStep] = useState<Step>("upload");
  const [parsedFile, setParsedFile] = useState<ParsedFile | null>(null);
  const [mappingConfig, setMappingConfig] = useState<UpdateMappingConfig | null>(null);
  const [previewData, setPreviewData] = useState<PreviewResponse | null>(null);
  const [executeResult, setExecuteResult] = useState<ExecuteResponse | null>(null);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);

  const handleFileUploaded = (data: ParsedFile) => {
    setParsedFile(data);
    setCurrentStep("mapping");
  };

  const handleMappingComplete = async (config: UpdateMappingConfig) => {
    if (!parsedFile) return;

    setMappingConfig(config);

    try {
      // Call preview API
      const response = await apiClient.previewBulkUpdate({
        rows: parsedFile.rows,
        identifier_column: config.identifierColumn,
        field_mappings: config.fieldMappings.map((m) => ({
          source_column: m.sourceColumn,
          target_field: m.targetField,
        })),
      });

      setPreviewData(response);
      setCurrentStep("preview");
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : "Preview failed"
      );
    }
  };

  const handleExecuteUpdate = async () => {
    if (!previewData || !previewData.matches.length) return;

    // Convert preview matches to update format
    const updates = previewData.matches.map((match) => ({
      product_id: match.product_id,
      fields: match.new_values,
    }));

    // Use async for large batches
    if (updates.length >= ASYNC_THRESHOLD) {
      try {
        const result = await apiClient.executeBulkUpdateAsync({
          updates,
          mode: "lenient",
        });
        setActiveJobId(result.job_id);
        toast.info(result.message);
      } catch (error) {
        toast.error(
          error instanceof Error ? error.message : "Failed to start update job"
        );
      }
      return;
    }

    // Use sync for small batches
    setCurrentStep("executing");

    try {
      const result = await apiClient.executeBulkUpdate({
        updates,
        mode: "lenient", // Skip invalid rows
      });

      setExecuteResult(result);
      setCurrentStep("results");

      if (result.success) {
        toast.success(`Successfully updated ${result.updated_count} products`);
      } else {
        toast.warning(
          `Updated ${result.updated_count} products, ${result.failed.length} failed`
        );
      }
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : "Update failed"
      );
      setCurrentStep("preview");
    }
  };

  const handleJobComplete = (jobResult?: Record<string, unknown>) => {
    setActiveJobId(null);
    if (!jobResult) return;

    // Convert job result to ExecuteResponse format
    const result: ExecuteResponse = {
      success: (jobResult.failed as number) === 0,
      updated_count: jobResult.updated as number,
      failed: (jobResult.errors as Array<{ product_id: string; error: string }>) || [],
      execution_time_ms: 0,
    };

    setExecuteResult(result);
    setCurrentStep("results");

    if (result.success) {
      toast.success(`Successfully updated ${result.updated_count} products`);
    } else {
      toast.warning(
        `Updated ${result.updated_count} products, ${result.failed.length} failed`
      );
    }
  };

  const handleReset = () => {
    setParsedFile(null);
    setMappingConfig(null);
    setPreviewData(null);
    setExecuteResult(null);
    setCurrentStep("upload");
  };

  const handleBack = () => {
    if (currentStep === "mapping") {
      setCurrentStep("upload");
    } else if (currentStep === "preview") {
      setCurrentStep("mapping");
    }
  };

  const getStepStatus = (stepId: string) => {
    const stepOrder = ["upload", "mapping", "preview", "executing", "results"];
    const currentIndex = stepOrder.indexOf(currentStep);
    const stepIndex = stepOrder.indexOf(stepId);

    if (stepIndex < currentIndex) return "completed";
    if (stepIndex === currentIndex) return "current";
    return "upcoming";
  };

  return (
    <div className="container max-w-5xl py-8 space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold">Bulk Product Update</h1>
        <p className="text-muted-foreground mt-1">
          Update product information in bulk from Excel or CSV files
        </p>
      </div>

      {/* Stepper */}
      <div className="flex items-center justify-center gap-4">
        {STEPS.map((step, index) => {
          const status = getStepStatus(step.id);
          const Icon = step.icon;

          return (
            <div key={step.id} className="flex items-center">
              <div className="flex flex-col items-center gap-2">
                <div
                  className={`
                    flex items-center justify-center w-10 h-10 rounded-full border-2 transition-colors
                    ${status === "completed" ? "bg-primary border-primary text-primary-foreground" : ""}
                    ${status === "current" ? "border-primary text-primary" : ""}
                    ${status === "upcoming" ? "border-muted text-muted-foreground" : ""}
                  `}
                >
                  {status === "completed" ? (
                    <CheckCircle className="h-5 w-5" />
                  ) : currentStep === "executing" && step.id === "preview" ? (
                    <Loader2 className="h-5 w-5 animate-spin" />
                  ) : (
                    <Icon className="h-5 w-5" />
                  )}
                </div>
                <span
                  className={`text-sm font-medium ${
                    status === "upcoming"
                      ? "text-muted-foreground"
                      : "text-foreground"
                  }`}
                >
                  {step.label}
                </span>
              </div>
              {index < STEPS.length - 1 && (
                <div
                  className={`w-24 h-0.5 mx-4 ${
                    getStepStatus(STEPS[index + 1].id) !== "upcoming"
                      ? "bg-primary"
                      : "bg-muted"
                  }`}
                />
              )}
            </div>
          );
        })}
      </div>

      {/* Step Content */}
      <div className="min-h-[400px]">
        {currentStep === "upload" && (
          <FileUploadStep onComplete={handleFileUploaded} />
        )}

        {currentStep === "mapping" && parsedFile && (
          <UpdateMappingStep
            columns={parsedFile.columns}
            preview={parsedFile.preview}
            onComplete={handleMappingComplete}
            onBack={handleBack}
          />
        )}

        {currentStep === "preview" && previewData && parsedFile && (
          <UpdatePreviewStep
            previewData={previewData}
            columns={parsedFile.columns}
            onExecute={handleExecuteUpdate}
            onBack={handleBack}
          />
        )}

        {currentStep === "executing" && (
          <div className="flex flex-col items-center justify-center py-16 space-y-4">
            <Loader2 className="h-12 w-12 animate-spin text-primary" />
            <p className="text-lg font-medium">Updating products...</p>
            <p className="text-sm text-muted-foreground">
              This may take a moment for large batches
            </p>
          </div>
        )}

        {currentStep === "results" && executeResult && previewData && (
          <UpdateResultsStep
            result={executeResult}
            previewSummary={previewData.summary}
            onReset={handleReset}
          />
        )}
      </div>

      {/* Job Progress Modal */}
      <JobProgressModal
        jobId={activeJobId}
        title="Updating Products"
        onComplete={handleJobComplete}
        onClose={() => setActiveJobId(null)}
      />
    </div>
  );
}
