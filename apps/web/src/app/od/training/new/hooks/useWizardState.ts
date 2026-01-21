/**
 * Wizard State Hook
 *
 * Central state management for the training wizard.
 */

"use client";

import { useCallback, useMemo, useReducer } from "react";
import type {
  WizardState,
  WizardStep,
  DatasetStepData,
  PreprocessStepData,
  OfflineAugStepData,
  OnlineAugStepData,
  ModelStepData,
  HyperparamsStepData,
  ReviewStepData,
  DatasetInfo,
  WIZARD_STEPS,
} from "../types/wizard";
import {
  DEFAULT_WIZARD_STATE,
  DEFAULT_DATASET_STEP,
  DEFAULT_PREPROCESS_STEP,
  DEFAULT_OFFLINE_AUG_STEP,
  DEFAULT_ONLINE_AUG_STEP,
  DEFAULT_MODEL_STEP,
  DEFAULT_HYPERPARAMS_STEP,
  DEFAULT_REVIEW_STEP,
} from "../types/wizard";
import { validateStep } from "../utils/validation";

// Smart defaults payload type
interface SmartDefaultsPayload {
  preprocess?: Partial<PreprocessStepData>;
  offlineAug?: Partial<OfflineAugStepData>;
  onlineAug?: Partial<OnlineAugStepData>;
  model?: Partial<ModelStepData>;
  hyperparams?: Partial<HyperparamsStepData>;
}

// Action types
type WizardAction =
  | { type: "SET_STEP"; payload: WizardStep }
  | { type: "COMPLETE_STEP"; payload: WizardStep }
  | { type: "UPDATE_DATASET"; payload: Partial<DatasetStepData> }
  | { type: "UPDATE_PREPROCESS"; payload: Partial<PreprocessStepData> }
  | { type: "UPDATE_OFFLINE_AUG"; payload: Partial<OfflineAugStepData> }
  | { type: "UPDATE_ONLINE_AUG"; payload: Partial<OnlineAugStepData> }
  | { type: "UPDATE_MODEL"; payload: Partial<ModelStepData> }
  | { type: "UPDATE_HYPERPARAMS"; payload: Partial<HyperparamsStepData> }
  | { type: "UPDATE_REVIEW"; payload: Partial<ReviewStepData> }
  | { type: "SET_DATASET_INFO"; payload: DatasetInfo | null }
  | { type: "SET_SUBMITTING"; payload: boolean }
  | { type: "SET_ERRORS"; payload: Partial<Record<WizardStep, string[]>> }
  | { type: "APPLY_SMART_DEFAULTS"; payload: SmartDefaultsPayload }
  | { type: "RESET" };

// Reducer
function wizardReducer(state: WizardState, action: WizardAction): WizardState {
  switch (action.type) {
    case "SET_STEP":
      return { ...state, currentStep: action.payload };

    case "COMPLETE_STEP":
      const newCompleted = new Set(state.completedSteps);
      newCompleted.add(action.payload);
      return { ...state, completedSteps: newCompleted };

    case "UPDATE_DATASET":
      return {
        ...state,
        dataset: { ...state.dataset, ...action.payload },
      };

    case "UPDATE_PREPROCESS":
      return {
        ...state,
        preprocess: { ...state.preprocess, ...action.payload },
      };

    case "UPDATE_OFFLINE_AUG":
      return {
        ...state,
        offlineAug: { ...state.offlineAug, ...action.payload },
      };

    case "UPDATE_ONLINE_AUG":
      return {
        ...state,
        onlineAug: { ...state.onlineAug, ...action.payload },
      };

    case "UPDATE_MODEL":
      return {
        ...state,
        model: { ...state.model, ...action.payload },
      };

    case "UPDATE_HYPERPARAMS":
      return {
        ...state,
        hyperparams: { ...state.hyperparams, ...action.payload },
      };

    case "UPDATE_REVIEW":
      return {
        ...state,
        review: { ...state.review, ...action.payload },
      };

    case "SET_DATASET_INFO":
      return { ...state, datasetInfo: action.payload };

    case "SET_SUBMITTING":
      return { ...state, isSubmitting: action.payload };

    case "SET_ERRORS":
      return { ...state, errors: action.payload };

    case "APPLY_SMART_DEFAULTS":
      return {
        ...state,
        preprocess: { ...state.preprocess, ...action.payload.preprocess },
        offlineAug: { ...state.offlineAug, ...action.payload.offlineAug },
        onlineAug: { ...state.onlineAug, ...action.payload.onlineAug },
        model: { ...state.model, ...action.payload.model },
        hyperparams: { ...state.hyperparams, ...action.payload.hyperparams },
      };

    case "RESET":
      return {
        ...DEFAULT_WIZARD_STATE,
        completedSteps: new Set(),
      };

    default:
      return state;
  }
}

// Initial state with proper Set
const initialState: WizardState = {
  ...DEFAULT_WIZARD_STATE,
  completedSteps: new Set(),
};

/**
 * Main wizard state hook
 */
export function useWizardState() {
  const [state, dispatch] = useReducer(wizardReducer, initialState);

  // Step navigation
  const goToStep = useCallback((step: WizardStep) => {
    dispatch({ type: "SET_STEP", payload: step });
  }, []);

  const completeStep = useCallback((step: WizardStep) => {
    dispatch({ type: "COMPLETE_STEP", payload: step });
  }, []);

  const nextStep = useCallback(() => {
    const steps: WizardStep[] = [
      "dataset",
      "preprocess",
      "offline-aug",
      "online-aug",
      "model",
      "hyperparams",
      "review",
    ];
    const currentIndex = steps.indexOf(state.currentStep);

    // Validate current step
    const validation = validateStep(state.currentStep, state);
    if (!validation.isValid) {
      dispatch({
        type: "SET_ERRORS",
        payload: { [state.currentStep]: validation.errors },
      });
      return false;
    }

    // Mark current step as completed
    dispatch({ type: "COMPLETE_STEP", payload: state.currentStep });

    // Go to next step
    if (currentIndex < steps.length - 1) {
      dispatch({ type: "SET_STEP", payload: steps[currentIndex + 1] });
    }

    return true;
  }, [state]);

  const prevStep = useCallback(() => {
    const steps: WizardStep[] = [
      "dataset",
      "preprocess",
      "offline-aug",
      "online-aug",
      "model",
      "hyperparams",
      "review",
    ];
    const currentIndex = steps.indexOf(state.currentStep);

    if (currentIndex > 0) {
      dispatch({ type: "SET_STEP", payload: steps[currentIndex - 1] });
    }
  }, [state.currentStep]);

  // Step data updaters
  const updateDataset = useCallback((data: Partial<DatasetStepData>) => {
    dispatch({ type: "UPDATE_DATASET", payload: data });
  }, []);

  const updatePreprocess = useCallback((data: Partial<PreprocessStepData>) => {
    dispatch({ type: "UPDATE_PREPROCESS", payload: data });
  }, []);

  const updateOfflineAug = useCallback((data: Partial<OfflineAugStepData>) => {
    dispatch({ type: "UPDATE_OFFLINE_AUG", payload: data });
  }, []);

  const updateOnlineAug = useCallback((data: Partial<OnlineAugStepData>) => {
    dispatch({ type: "UPDATE_ONLINE_AUG", payload: data });
  }, []);

  const updateModel = useCallback((data: Partial<ModelStepData>) => {
    dispatch({ type: "UPDATE_MODEL", payload: data });
  }, []);

  const updateHyperparams = useCallback((data: Partial<HyperparamsStepData>) => {
    dispatch({ type: "UPDATE_HYPERPARAMS", payload: data });
  }, []);

  const updateReview = useCallback((data: Partial<ReviewStepData>) => {
    dispatch({ type: "UPDATE_REVIEW", payload: data });
  }, []);

  // Dataset info
  const setDatasetInfo = useCallback((info: DatasetInfo | null) => {
    dispatch({ type: "SET_DATASET_INFO", payload: info });
  }, []);

  // Submitting state
  const setSubmitting = useCallback((isSubmitting: boolean) => {
    dispatch({ type: "SET_SUBMITTING", payload: isSubmitting });
  }, []);

  // Errors
  const setErrors = useCallback((errors: Partial<Record<WizardStep, string[]>>) => {
    dispatch({ type: "SET_ERRORS", payload: errors });
  }, []);

  const clearErrors = useCallback(() => {
    dispatch({ type: "SET_ERRORS", payload: {} });
  }, []);

  // Smart defaults
  const applySmartDefaults = useCallback((defaults: SmartDefaultsPayload) => {
    dispatch({ type: "APPLY_SMART_DEFAULTS", payload: defaults });
  }, []);

  // Reset
  const reset = useCallback(() => {
    dispatch({ type: "RESET" });
  }, []);

  // Current step info
  const stepIndex = useMemo(() => {
    const steps: WizardStep[] = [
      "dataset",
      "preprocess",
      "offline-aug",
      "online-aug",
      "model",
      "hyperparams",
      "review",
    ];
    return steps.indexOf(state.currentStep);
  }, [state.currentStep]);

  const isFirstStep = stepIndex === 0;
  const isLastStep = stepIndex === 6;

  // Current step validation
  const currentStepValidation = useMemo(() => {
    return validateStep(state.currentStep, state);
  }, [state]);

  return {
    // State
    state,
    stepIndex,
    isFirstStep,
    isLastStep,
    currentStepValidation,

    // Navigation
    goToStep,
    completeStep,
    nextStep,
    prevStep,

    // Updaters
    updateDataset,
    updatePreprocess,
    updateOfflineAug,
    updateOnlineAug,
    updateModel,
    updateHyperparams,
    updateReview,

    // Misc
    setDatasetInfo,
    setSubmitting,
    setErrors,
    clearErrors,
    applySmartDefaults,
    reset,
  };
}

export type WizardStateHook = ReturnType<typeof useWizardState>;
