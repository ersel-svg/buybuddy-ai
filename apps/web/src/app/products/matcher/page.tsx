"use client";

import { useState } from "react";
import { FileSpreadsheet, GitCompare, CheckCircle, Loader2 } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { FileUploadStep, type ParsedFile } from "./components/FileUploadStep";
import { FieldMappingStep, type MappingConfig } from "./components/FieldMappingStep";
import { MatchingResults } from "./components/MatchingResults";
import { apiClient } from "@/lib/api-client";
import { toast } from "sonner";

type Step = "upload" | "mapping" | "matching" | "results";

interface MatchResponse {
  matched: Array<{
    source_row: Record<string, unknown>;
    product: {
      id: string;
      barcode: string;
      product_name?: string;
      brand_name?: string;
      category?: string;
      status: string;
    };
    matched_by: string;
  }>;
  unmatched: Array<{ source_row: Record<string, unknown> }>;
  summary: {
    total: number;
    matched_count: number;
    unmatched_count: number;
    match_rate: number;
  };
}

const STEPS = [
  { id: "upload", label: "Upload", icon: FileSpreadsheet },
  { id: "mapping", label: "Mapping", icon: GitCompare },
  { id: "results", label: "Results", icon: CheckCircle },
] as const;

export default function ProductMatcherPage() {
  const [currentStep, setCurrentStep] = useState<Step>("upload");
  const [parsedFile, setParsedFile] = useState<ParsedFile | null>(null);
  const [matchResults, setMatchResults] = useState<MatchResponse | null>(null);

  const handleFileUploaded = (data: ParsedFile) => {
    setParsedFile(data);
    setCurrentStep("mapping");
  };

  const handleMappingComplete = async (config: MappingConfig) => {
    if (!parsedFile) return;

    setCurrentStep("matching");

    try {
      // Convert mapping config to API format
      const matchRules = config.matchRules.map((rule) => ({
        source_column: rule.sourceColumn,
        target_field: rule.targetField,
        priority: rule.priority,
      }));

      const results = await apiClient.matchProducts(parsedFile.rows, matchRules);
      setMatchResults(results);
      setCurrentStep("results");
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : "Matching failed"
      );
      setCurrentStep("mapping");
    }
  };

  const handleReset = () => {
    setParsedFile(null);
    setMatchResults(null);
    setCurrentStep("upload");
  };

  const getStepStatus = (stepId: string) => {
    const stepOrder = ["upload", "mapping", "matching", "results"];
    const currentIndex = stepOrder.indexOf(currentStep);
    const stepIndex = stepOrder.indexOf(stepId);

    if (stepIndex < currentIndex) return "completed";
    if (stepIndex === currentIndex) return "current";
    return "upcoming";
  };

  return (
    <div className="container max-w-4xl py-8 space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold">Product Matcher</h1>
        <p className="text-muted-foreground mt-1">
          Match your product list against the system database
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
                  ) : currentStep === "matching" && step.id === "mapping" ? (
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

      {/* Content */}
      <Card>
        <CardContent className="pt-6">
          {currentStep === "upload" && (
            <FileUploadStep onComplete={handleFileUploaded} />
          )}

          {currentStep === "mapping" && parsedFile && (
            <FieldMappingStep
              parsedFile={parsedFile}
              onComplete={handleMappingComplete}
              onBack={() => setCurrentStep("upload")}
            />
          )}

          {currentStep === "matching" && (
            <div className="flex flex-col items-center justify-center py-16 gap-4">
              <Loader2 className="h-12 w-12 animate-spin text-primary" />
              <div className="text-center">
                <p className="font-medium">Matching products...</p>
                <p className="text-sm text-muted-foreground mt-1">
                  Comparing {parsedFile?.totalRows.toLocaleString()} rows against
                  the database
                </p>
              </div>
            </div>
          )}

          {currentStep === "results" && matchResults && parsedFile && (
            <MatchingResults
              results={matchResults}
              fileName={parsedFile.fileName}
              columns={parsedFile.columns}
              onReset={handleReset}
            />
          )}
        </CardContent>
      </Card>
    </div>
  );
}
