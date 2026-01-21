/**
 * WizardNavigation Component
 *
 * Bottom navigation with Back, Next, and Submit buttons.
 */

"use client";

import { ArrowLeft, ArrowRight, Loader2, Rocket } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface WizardNavigationProps {
  isFirstStep: boolean;
  isLastStep: boolean;
  isSubmitting: boolean;
  canProceed: boolean;
  onBack: () => void;
  onNext: () => void;
  onSubmit: () => void;
  className?: string;
}

export function WizardNavigation({
  isFirstStep,
  isLastStep,
  isSubmitting,
  canProceed,
  onBack,
  onNext,
  onSubmit,
  className,
}: WizardNavigationProps) {
  return (
    <div
      className={cn(
        "flex items-center justify-between border-t bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 px-6 py-4",
        className
      )}
    >
      {/* Back button */}
      <Button
        type="button"
        variant="outline"
        onClick={onBack}
        disabled={isFirstStep || isSubmitting}
        className="gap-2"
      >
        <ArrowLeft className="h-4 w-4" />
        Back
      </Button>

      {/* Next / Submit button */}
      {isLastStep ? (
        <Button
          type="button"
          onClick={onSubmit}
          disabled={!canProceed || isSubmitting}
          className="gap-2"
        >
          {isSubmitting ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" />
              Starting Training...
            </>
          ) : (
            <>
              <Rocket className="h-4 w-4" />
              Start Training
            </>
          )}
        </Button>
      ) : (
        <Button
          type="button"
          onClick={onNext}
          disabled={!canProceed || isSubmitting}
          className="gap-2"
        >
          Next
          <ArrowRight className="h-4 w-4" />
        </Button>
      )}
    </div>
  );
}

/**
 * Inline navigation variant
 */
export function WizardNavigationInline({
  isFirstStep,
  isLastStep,
  isSubmitting,
  canProceed,
  onBack,
  onNext,
  onSubmit,
  className,
}: WizardNavigationProps) {
  return (
    <div className={cn("flex items-center gap-4", className)}>
      {!isFirstStep && (
        <Button
          type="button"
          variant="ghost"
          size="sm"
          onClick={onBack}
          disabled={isSubmitting}
          className="gap-1"
        >
          <ArrowLeft className="h-3 w-3" />
          Back
        </Button>
      )}

      {isLastStep ? (
        <Button
          type="button"
          size="sm"
          onClick={onSubmit}
          disabled={!canProceed || isSubmitting}
          className="gap-1"
        >
          {isSubmitting ? (
            <>
              <Loader2 className="h-3 w-3 animate-spin" />
              Starting...
            </>
          ) : (
            <>
              <Rocket className="h-3 w-3" />
              Start
            </>
          )}
        </Button>
      ) : (
        <Button
          type="button"
          size="sm"
          onClick={onNext}
          disabled={!canProceed || isSubmitting}
          className="gap-1"
        >
          Next
          <ArrowRight className="h-3 w-3" />
        </Button>
      )}
    </div>
  );
}
