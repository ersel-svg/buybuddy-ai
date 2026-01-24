/**
 * Training Wizard Page
 *
 * 7-step wizard for configuring and starting object detection training.
 */

"use client";

import { useState, useCallback, useEffect } from "react";
import { useRouter } from "next/navigation";
import { ArrowLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";
import { apiClient } from "@/lib/api-client";

// Components
import { WizardStepper, WizardStepperCompact } from "./components/WizardStepper";
import { WizardNavigation } from "./components/WizardNavigation";
import { SmartRecommendationCard } from "./components/SmartRecommendationCard";
import {
  DatasetStep,
  PreprocessStep,
  OfflineAugStep,
  OnlineAugStep,
  ModelStep,
  HyperparamsStep,
  ReviewStep,
} from "./components/steps";

// Hooks
import { useWizardState } from "./hooks/useWizardState";
import { useSmartDefaults } from "./hooks/useSmartDefaults";

// Utils
import { convertWizardStateToApiRequest } from "./utils/apiConverter";
import { validateStep } from "./utils/validation";

// Types
import { WIZARD_STEPS, STEP_DESCRIPTIONS } from "./types/wizard";
import type { WizardStep } from "./types/wizard";

export default function TrainingWizardPage() {
  const router = useRouter();
  const [smartDefaultsApplied, setSmartDefaultsApplied] = useState(false);

  // Wizard state management
  const {
    state,
    stepIndex,
    isFirstStep,
    isLastStep,
    currentStepValidation,
    goToStep,
    nextStep,
    prevStep,
    updateDataset,
    updatePreprocess,
    updateOfflineAug,
    updateOnlineAug,
    updateModel,
    updateHyperparams,
    updateReview,
    setDatasetInfo,
    setSubmitting,
    applySmartDefaults,
  } = useWizardState();

  // Smart defaults
  const { smartDefaults, presetRecommendation } = useSmartDefaults({
    datasetInfo: state.datasetInfo,
  });

  // Apply smart defaults when dataset changes
  const handleApplySmartDefaults = useCallback(() => {
    if (smartDefaults) {
      applySmartDefaults({
        preprocess: smartDefaults.preprocess,
        offlineAug: smartDefaults.offlineAug,
        onlineAug: smartDefaults.onlineAug,
        model: smartDefaults.model,
        hyperparams: smartDefaults.hyperparams,
      });
      setSmartDefaultsApplied(true);
      toast.success("Smart defaults applied!");
    }
  }, [smartDefaults, applySmartDefaults]);

  // Reset smart defaults flag when dataset changes
  useEffect(() => {
    setSmartDefaultsApplied(false);
  }, [state.dataset.datasetId]);

  // Handle step click from stepper
  const handleStepClick = useCallback(
    (step: WizardStep) => {
      const targetIndex = WIZARD_STEPS.indexOf(step);
      const currentIndex = WIZARD_STEPS.indexOf(state.currentStep);

      // Can always go back
      if (targetIndex < currentIndex) {
        goToStep(step);
        return;
      }

      // Can only go forward if current and all previous steps are valid
      for (let i = 0; i <= currentIndex; i++) {
        const validation = validateStep(WIZARD_STEPS[i], state);
        if (!validation.isValid) {
          toast.error(`Please complete ${WIZARD_STEPS[i]} step first`);
          return;
        }
      }

      goToStep(step);
    },
    [state, goToStep]
  );

  // Handle form submission
  const handleSubmit = useCallback(async () => {
    // Validate review step
    const validation = validateStep("review", state);
    if (!validation.isValid) {
      toast.error(validation.errors[0]);
      return;
    }

    setSubmitting(true);

    try {
      const request = convertWizardStateToApiRequest(state);

      // Use apiClient for proper backend URL
      const result = await apiClient.createODTrainingRun(request as Parameters<typeof apiClient.createODTrainingRun>[0]);

      toast.success("Training started successfully!");
      router.push(`/od/training/${result.id}`);
    } catch (error) {
      console.error("Training submission error:", error);
      toast.error(
        error instanceof Error ? error.message : "Failed to start training"
      );
    } finally {
      setSubmitting(false);
    }
  }, [state, setSubmitting, router]);

  // Render current step content
  const renderStepContent = () => {
    const stepErrors = state.errors[state.currentStep];

    switch (state.currentStep) {
      case "dataset":
        return (
          <DatasetStep
            data={state.dataset}
            onChange={updateDataset}
            datasetInfo={state.datasetInfo}
            onDatasetInfoChange={setDatasetInfo}
            errors={stepErrors}
          />
        );
      case "preprocess":
        return (
          <PreprocessStep
            data={state.preprocess}
            onChange={updatePreprocess}
            errors={stepErrors}
          />
        );
      case "offline-aug":
        return (
          <OfflineAugStep
            data={state.offlineAug}
            onChange={updateOfflineAug}
            datasetInfo={state.datasetInfo}
            errors={stepErrors}
          />
        );
      case "online-aug":
        return (
          <OnlineAugStep
            data={state.onlineAug}
            onChange={updateOnlineAug}
            recommendedPreset={presetRecommendation?.preset}
            errors={stepErrors}
          />
        );
      case "model":
        return (
          <ModelStep
            data={state.model}
            onChange={updateModel}
            errors={stepErrors}
          />
        );
      case "hyperparams":
        return (
          <HyperparamsStep
            data={state.hyperparams}
            onChange={updateHyperparams}
            errors={stepErrors}
          />
        );
      case "review":
        return (
          <ReviewStep
            data={state.review}
            onChange={updateReview}
            wizardState={state}
            errors={stepErrors}
          />
        );
      default:
        return null;
    }
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="sticky top-0 z-50 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container flex h-14 items-center gap-4">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => router.push("/od/training")}
            className="gap-2"
          >
            <ArrowLeft className="h-4 w-4" />
            Back to Trainings
          </Button>
          <div className="flex-1" />
          <h1 className="text-lg font-semibold">New Training</h1>
        </div>
      </header>

      {/* Stepper - Desktop */}
      <div className="container py-6 hidden md:block">
        <WizardStepper
          currentStep={state.currentStep}
          completedSteps={state.completedSteps}
          onStepClick={handleStepClick}
        />
      </div>

      {/* Stepper - Mobile */}
      <div className="container py-4 md:hidden">
        <WizardStepperCompact
          currentStep={state.currentStep}
          completedSteps={state.completedSteps}
        />
      </div>

      {/* Main Content */}
      <main className="container pb-24">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Step Content */}
          <div className="lg:col-span-3">
            <div className="mb-4">
              <p className="text-sm text-muted-foreground">
                {STEP_DESCRIPTIONS[state.currentStep]}
              </p>
            </div>
            {renderStepContent()}
          </div>

          {/* Sidebar - Smart Recommendations */}
          <div className="hidden lg:block">
            {state.currentStep === "dataset" && smartDefaults && (
              <SmartRecommendationCard
                smartDefaults={smartDefaults}
                onApply={handleApplySmartDefaults}
                isApplied={smartDefaultsApplied}
              />
            )}
          </div>
        </div>
      </main>

      {/* Navigation */}
      <WizardNavigation
        isFirstStep={isFirstStep}
        isLastStep={isLastStep}
        isSubmitting={state.isSubmitting}
        canProceed={currentStepValidation.isValid}
        onBack={prevStep}
        onNext={nextStep}
        onSubmit={handleSubmit}
        className="fixed bottom-0 left-0 right-0"
      />
    </div>
  );
}
