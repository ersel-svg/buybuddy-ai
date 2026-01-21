# Training Wizard - QA Test Plan

## Overview
Comprehensive test plan for the 7-step Training Wizard feature covering:
- Backend (Python): presets.py, albumentations_wrapper.py, handler.py
- Frontend (TypeScript/React): types, hooks, utils, components

---

## 1. CHECKLIST

### 1.1 Backend Checklist

#### presets.py
- [ ] All 40+ augmentations defined in AugmentationPreset dataclass
- [ ] SOTA-v2 preset created with correct configuration
- [ ] SOTA, Heavy, Medium, Light, None presets working
- [ ] `get_preset()` function returns correct preset with overrides
- [ ] `get_preset_info()` returns info for all presets
- [ ] `get_augmentation_categories()` returns all 8 categories
- [ ] `validate_custom_config()` catches invalid configs
- [ ] All augmentation params have correct default values

#### albumentations_wrapper.py
- [ ] `build_albumentations_pipeline()` handles all augmentation types
- [ ] Geometric transforms with correct border modes
- [ ] Color transforms with correct parameters
- [ ] Blur transforms working
- [ ] Noise transforms working
- [ ] Quality degradation transforms working
- [ ] Dropout transforms working
- [ ] Weather transforms working
- [ ] Multi-image transforms (mosaic, mixup, copypaste) handled separately

#### handler.py
- [ ] `convert_frontend_augmentation_config()` maps all 40+ augmentations
- [ ] Legacy aliases (copy_paste -> copypaste) working
- [ ] Probability conversion (probability -> prob) working
- [ ] Params extraction working correctly
- [ ] Returns None for empty/invalid config

### 1.2 Frontend Checklist

#### Types (wizard.ts)
- [ ] WizardStep type includes all 7 steps
- [ ] DatasetStepData interface complete
- [ ] PreprocessStepData interface complete
- [ ] OfflineAugStepData interface complete
- [ ] OnlineAugStepData interface complete
- [ ] ModelStepData interface complete
- [ ] HyperparamsStepData interface complete
- [ ] ReviewStepData interface complete
- [ ] WizardState interface complete
- [ ] CreateTrainingRequest type matches API spec
- [ ] Default values properly defined

#### Hooks
- [ ] useWizardState returns all required functions
- [ ] useSmartDefaults generates recommendations
- [ ] useDatasetStats fetches and formats data

#### Utils
- [ ] smartDefaults.ts generates correct defaults for all dataset sizes
- [ ] validation.ts validates all 7 steps
- [ ] apiConverter.ts converts state to API format correctly

#### Components
- [ ] WizardStepper shows all 7 steps
- [ ] WizardNavigation handles back/next/submit
- [ ] All step components render without errors
- [ ] Error states displayed correctly

---

## 2. USER SCENARIOS

### Scenario 1: First-Time User - Quick Start
**Goal:** User wants to train a model with minimal configuration
**Steps:**
1. User selects a dataset
2. System shows smart recommendations
3. User clicks "Apply Smart Defaults"
4. User reviews and submits

**Expected:** Training starts with optimal defaults

### Scenario 2: Power User - Custom Configuration
**Goal:** User wants full control over all settings
**Steps:**
1. User selects dataset
2. User configures preprocessing (custom image size, tiling)
3. User enables offline augmentation with custom multiplier
4. User selects "Custom" augmentation preset
5. User enables specific augmentations (mosaic, mixup, shift_scale_rotate)
6. User selects D-FINE XL model
7. User adjusts hyperparameters (epochs, LR, batch size)
8. User adds tags and description
9. User submits

**Expected:** Training starts with all custom settings

### Scenario 3: Small Dataset User
**Goal:** User with <500 images wants maximum augmentation
**Steps:**
1. User selects small dataset
2. Smart defaults recommend "Heavy" preset
3. User enables offline augmentation (3x multiplier)
4. User enables all weather augmentations
5. User submits

**Expected:** Heavy augmentation applied, dataset multiplied

### Scenario 4: Large Dataset User
**Goal:** User with >10000 images wants fast training
**Steps:**
1. User selects large dataset
2. Smart defaults recommend "Light" preset
3. User keeps defaults
4. User submits

**Expected:** Light augmentation, fast training

### Scenario 5: Navigation Test
**Goal:** Test wizard navigation behavior
**Steps:**
1. User completes step 1-3
2. User clicks back to step 1
3. User changes dataset
4. User tries to jump to step 5

**Expected:** Blocked, must complete steps in order

### Scenario 6: Validation Error Handling
**Goal:** Test error states
**Steps:**
1. User tries to proceed without selecting dataset
2. User enters invalid batch size (0)
3. User enters invalid learning rate (negative)
4. User submits without training name

**Expected:** Clear error messages, blocked progression

---

## 3. PRODUCTION BATTLE TEST CASES

### 3.1 Edge Cases

#### TC-001: Empty Dataset Selection
- Input: No dataset selected
- Expected: Validation error, cannot proceed

#### TC-002: Extremely Large Image Size
- Input: image_size = 2048
- Expected: Warning about VRAM, suggest tiling

#### TC-003: Zero Batch Size
- Input: batch_size = 0
- Expected: Validation error

#### TC-004: Negative Learning Rate
- Input: learning_rate = -0.001
- Expected: Validation error

#### TC-005: All Augmentations Enabled
- Input: Custom preset with all 40+ augmentations at 1.0 probability
- Expected: Warning about slow training, but allowed

#### TC-006: No Augmentations
- Input: preset = "none"
- Expected: Warning about potential overfitting

#### TC-007: Conflicting Augmentations
- Input: Both mosaic and mosaic9 enabled
- Expected: Should work (applied sequentially)

#### TC-008: Invalid Augmentation Params
- Input: blur_limit = -5
- Expected: Validation error or clamped to valid range

### 3.2 Stress Tests

#### TC-009: Rapid Step Navigation
- Action: Click through all steps rapidly
- Expected: No state corruption, smooth transitions

#### TC-010: Form State Persistence
- Action: Fill step 3, go back to step 1, return to step 3
- Expected: Step 3 data preserved

#### TC-011: Smart Defaults Re-application
- Action: Apply smart defaults, modify, apply again
- Expected: Clean re-application

#### TC-012: Long Training Name
- Input: 500 character training name
- Expected: Truncated or validation error

### 3.3 API Integration Tests

#### TC-013: API Request Format
- Check: convertWizardStateToApiRequest output matches API spec
- Verify: All required fields present, correct types

#### TC-014: Augmentation Config Conversion
- Check: Custom augmentation config properly converted
- Verify: probability -> prob, params extracted

#### TC-015: Model Configuration
- Check: All model types (rt-detr, d-fine) and sizes (s, m, l, x) valid

### 3.4 Accessibility Tests

#### TC-016: Keyboard Navigation
- Action: Navigate wizard using only keyboard
- Expected: All steps accessible, focus management correct

#### TC-017: Screen Reader Compatibility
- Check: All form labels associated, ARIA attributes correct

---

## 4. TEST EXECUTION COMMANDS

```bash
# TypeScript compilation check
cd buybuddy-ai/apps/web && npx tsc --noEmit

# Python syntax check
cd buybuddy-ai/workers/od-training && python -m py_compile handler.py
cd buybuddy-ai/workers/od-training/src/augmentations && python -m py_compile presets.py
cd buybuddy-ai/workers/od-training/src/augmentations && python -m py_compile albumentations_wrapper.py

# Run Python unit tests
cd buybuddy-ai/workers/od-training && python -m pytest tests/ -v

# Run frontend tests
cd buybuddy-ai/apps/web && npm test

# E2E tests
cd buybuddy-ai && npm run test:e2e
```

---

## 5. SIGN-OFF CRITERIA

- [ ] All backend Python files pass syntax check
- [ ] All frontend TypeScript files compile without errors
- [ ] All unit tests pass
- [ ] All user scenarios complete successfully
- [ ] All edge cases handled gracefully
- [ ] No console errors during wizard flow
- [ ] API request format validated
- [ ] Performance acceptable (< 100ms state updates)
