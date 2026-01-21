/**
 * WizardStepper Component
 *
 * Visual stepper showing current progress through the wizard steps.
 */

"use client";

import { Check, Circle } from "lucide-react";
import { cn } from "@/lib/utils";
import type { WizardStep } from "../types/wizard";
import { WIZARD_STEPS, STEP_TITLES } from "../types/wizard";

interface WizardStepperProps {
  currentStep: WizardStep;
  completedSteps: Set<WizardStep>;
  onStepClick?: (step: WizardStep) => void;
  className?: string;
}

export function WizardStepper({
  currentStep,
  completedSteps,
  onStepClick,
  className,
}: WizardStepperProps) {
  const currentIndex = WIZARD_STEPS.indexOf(currentStep);

  return (
    <nav aria-label="Progress" className={cn("w-full", className)}>
      <ol className="flex items-center justify-between">
        {WIZARD_STEPS.map((step, index) => {
          const isCompleted = completedSteps.has(step);
          const isCurrent = step === currentStep;
          const isClickable = isCompleted || index <= currentIndex;

          return (
            <li key={step} className="relative flex-1">
              {/* Connector line */}
              {index > 0 && (
                <div
                  className={cn(
                    "absolute left-0 top-4 -translate-y-1/2 h-0.5 w-full -ml-1/2",
                    index <= currentIndex ? "bg-primary" : "bg-muted"
                  )}
                  style={{ width: "calc(100% - 1.5rem)", marginLeft: "-50%" }}
                />
              )}

              {/* Step indicator */}
              <div className="relative flex flex-col items-center">
                <button
                  type="button"
                  onClick={() => isClickable && onStepClick?.(step)}
                  disabled={!isClickable}
                  className={cn(
                    "relative z-10 flex h-8 w-8 items-center justify-center rounded-full border-2 transition-all",
                    isCompleted && "bg-primary border-primary text-primary-foreground",
                    isCurrent && !isCompleted && "border-primary bg-background",
                    !isCurrent && !isCompleted && "border-muted bg-background text-muted-foreground",
                    isClickable && "cursor-pointer hover:border-primary/80",
                    !isClickable && "cursor-not-allowed"
                  )}
                >
                  {isCompleted ? (
                    <Check className="h-4 w-4" />
                  ) : (
                    <span className="text-sm font-medium">{index + 1}</span>
                  )}
                </button>

                {/* Step title */}
                <span
                  className={cn(
                    "mt-2 text-xs font-medium text-center",
                    isCurrent && "text-primary",
                    !isCurrent && !isCompleted && "text-muted-foreground",
                    isCompleted && "text-foreground"
                  )}
                >
                  {STEP_TITLES[step]}
                </span>
              </div>
            </li>
          );
        })}
      </ol>
    </nav>
  );
}

/**
 * Compact stepper for mobile
 */
export function WizardStepperCompact({
  currentStep,
  completedSteps,
  className,
}: Omit<WizardStepperProps, "onStepClick">) {
  const currentIndex = WIZARD_STEPS.indexOf(currentStep);
  const totalSteps = WIZARD_STEPS.length;
  const completedCount = completedSteps.size;

  return (
    <div className={cn("flex items-center gap-2", className)}>
      <span className="text-sm font-medium">
        Step {currentIndex + 1} of {totalSteps}
      </span>
      <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
        <div
          className="h-full bg-primary transition-all duration-300"
          style={{ width: `${((currentIndex + 1) / totalSteps) * 100}%` }}
        />
      </div>
      <span className="text-sm text-muted-foreground">
        {STEP_TITLES[currentStep]}
      </span>
    </div>
  );
}
