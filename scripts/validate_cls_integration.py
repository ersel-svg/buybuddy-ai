#!/usr/bin/env python3
"""
CLS Integration Validator

Validates that API and Workers are properly integrated by analyzing source code.
Run this script to check for integration issues before deployment.

Usage:
    python scripts/validate_cls_integration.py
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.RESET}")


def print_pass(text: str):
    print(f"  {Colors.GREEN}[PASS]{Colors.RESET} {text}")


def print_fail(text: str):
    print(f"  {Colors.RED}[FAIL]{Colors.RESET} {text}")


def print_warn(text: str):
    print(f"  {Colors.YELLOW}[WARN]{Colors.RESET} {text}")


def print_info(text: str):
    print(f"  {Colors.BLUE}[INFO]{Colors.RESET} {text}")


def find_project_root() -> Path:
    """Find project root by looking for buybuddy-ai directory."""
    current = Path.cwd()
    while current != current.parent:
        if (current / "buybuddy-ai").exists():
            return current / "buybuddy-ai"
        if current.name == "buybuddy-ai":
            return current
        current = current.parent
    return Path.cwd()


def read_file(path: Path) -> str:
    """Read file contents."""
    try:
        return path.read_text()
    except Exception as e:
        return ""


def check_cls_annotation_api(root: Path) -> List[Tuple[str, bool, str]]:
    """Check CLS Annotation API integration."""
    results = []

    ai_py = root / "apps/api/src/api/v1/classification/ai.py"
    content = read_file(ai_py)

    if not content:
        results.append(("ai.py exists", False, "File not found"))
        return results

    results.append(("ai.py exists", True, ""))

    # Check that we use 'classes' not 'class_names' in RunPod input
    # Pattern: should find '"classes":' in runpod_input context
    if '"classes":' in content or "'classes':" in content:
        results.append(("Uses 'classes' (not 'class_names')", True, ""))
    else:
        results.append(("Uses 'classes' (not 'class_names')", False,
                       "RunPod input should use 'classes' key"))

    # Check that we send images array for single prediction
    if '"images":' in content and '[{"id":' in content.replace(" ", "").replace("'", '"'):
        results.append(("Single predict sends 'images' array", True, ""))
    else:
        # More lenient check
        if 'images' in content and 'request.image_id' in content:
            results.append(("Single predict sends 'images' array", True, ""))
        else:
            results.append(("Single predict sends 'images' array", False,
                           "Should send images as array"))

    # Check response parsing handles 'class' field
    if 'pred.get("class"' in content or "pred.get('class'" in content:
        results.append(("Response parsing handles 'class' field", True, ""))
    else:
        results.append(("Response parsing handles 'class' field", False,
                       "Should parse 'class' field from worker response"))

    # Check results array parsing
    if 'results[0].get("predictions"' in content or "results[0].get('predictions'" in content:
        results.append(("Parses results array correctly", True, ""))
    else:
        results.append(("Parses results array correctly", False,
                       "Should parse results[0].predictions"))

    return results


def check_cls_training_api(root: Path) -> List[Tuple[str, bool, str]]:
    """Check CLS Training API integration."""
    results = []

    training_py = root / "apps/api/src/api/v1/classification/training.py"
    content = read_file(training_py)

    if not content:
        results.append(("training.py exists", False, "File not found"))
        return results

    results.append(("training.py exists", True, ""))

    # Check submit_training_job function exists
    if "async def submit_training_job" in content:
        results.append(("submit_training_job function exists", True, ""))
    else:
        results.append(("submit_training_job function exists", False,
                       "Missing submit_training_job function"))

    # Check background task is enabled (not commented)
    if "background_tasks.add_task(submit_training_job" in content:
        # Check it's not commented
        lines = content.split('\n')
        for line in lines:
            if "background_tasks.add_task(submit_training_job" in line:
                if line.strip().startswith('#'):
                    results.append(("Background task enabled", False,
                                   "Line is commented out"))
                else:
                    results.append(("Background task enabled", True, ""))
                break
    else:
        results.append(("Background task enabled", False,
                       "Missing background_tasks.add_task call"))

    # Check train_urls/val_urls format (not images/labels)
    if '"train_urls":' in content or "'train_urls':" in content:
        results.append(("Uses train_urls format", True, ""))
    else:
        results.append(("Uses train_urls format", False,
                       "Should use train_urls/val_urls format"))

    # Check model_name mapping exists
    if "model_name_map" in content:
        results.append(("Model name mapping defined", True, ""))
    else:
        results.append(("Model name mapping defined", False,
                       "Missing model_name_map"))

    # Check RunPod service import
    if "from services.runpod import" in content:
        results.append(("RunPod service imported", True, ""))
    else:
        results.append(("RunPod service imported", False,
                       "Missing runpod service import"))

    # Check EndpointType.CLS_TRAINING usage
    if "EndpointType.CLS_TRAINING" in content:
        results.append(("Uses CLS_TRAINING endpoint", True, ""))
    else:
        results.append(("Uses CLS_TRAINING endpoint", False,
                       "Should use EndpointType.CLS_TRAINING"))

    return results


def check_cls_annotation_worker(root: Path) -> List[Tuple[str, bool, str]]:
    """Check CLS Annotation Worker."""
    results = []

    handler_py = root / "workers/cls-annotation/handler.py"
    content = read_file(handler_py)

    if not content:
        results.append(("handler.py exists", False, "File not found"))
        return results

    results.append(("handler.py exists", True, ""))

    # Check expected input parsing
    if 'job_input.get("classes"' in content or "job_input.get('classes'" in content:
        results.append(("Expects 'classes' input", True, ""))
    else:
        results.append(("Expects 'classes' input", False,
                       "Should expect 'classes' key"))

    # Check images parsing
    if 'job_input.get("images"' in content or "job_input.get('images'" in content:
        results.append(("Expects 'images' input", True, ""))
    else:
        results.append(("Expects 'images' input", False,
                       "Should expect 'images' key"))

    # Check output format has 'class' field
    if '"class":' in content or "'class':" in content:
        results.append(("Output uses 'class' field", True, ""))
    else:
        results.append(("Output uses 'class' field", False,
                       "Should output 'class' field in predictions"))

    return results


def check_cls_training_worker(root: Path) -> List[Tuple[str, bool, str]]:
    """Check CLS Training Worker."""
    results = []

    handler_py = root / "workers/cls-training/handler.py"
    content = read_file(handler_py)

    if not content:
        results.append(("handler.py exists", False, "File not found"))
        return results

    results.append(("handler.py exists", True, ""))

    # Check expected input format
    if 'dataset["train_urls"]' in content or "dataset['train_urls']" in content:
        results.append(("Expects train_urls format", True, ""))
    else:
        results.append(("Expects train_urls format", False,
                       "Should expect train_urls in dataset"))

    # Check class_names expected
    if 'dataset.get("class_names"' in content or "dataset.get('class_names'" in content:
        results.append(("Expects class_names", True, ""))
    else:
        if '"class_names"' in content or "'class_names'" in content:
            results.append(("Expects class_names", True, ""))
        else:
            results.append(("Expects class_names", False,
                           "Should expect class_names in dataset"))

    # Check URLImageDataset expects url and label
    if '"url"' in content and '"label"' in content:
        results.append(("Expects url/label format", True, ""))
    else:
        results.append(("Expects url/label format", False,
                       "Should expect url and label in image data"))

    return results


def check_runpod_service(root: Path) -> List[Tuple[str, bool, str]]:
    """Check RunPod service configuration."""
    results = []

    runpod_py = root / "apps/api/src/services/runpod.py"
    content = read_file(runpod_py)

    if not content:
        results.append(("runpod.py exists", False, "File not found"))
        return results

    results.append(("runpod.py exists", True, ""))

    # Check CLS endpoint types
    if "CLS_ANNOTATION" in content:
        results.append(("CLS_ANNOTATION endpoint defined", True, ""))
    else:
        results.append(("CLS_ANNOTATION endpoint defined", False,
                       "Missing CLS_ANNOTATION in EndpointType"))

    if "CLS_TRAINING" in content:
        results.append(("CLS_TRAINING endpoint defined", True, ""))
    else:
        results.append(("CLS_TRAINING endpoint defined", False,
                       "Missing CLS_TRAINING in EndpointType"))

    return results


def check_config(root: Path) -> List[Tuple[str, bool, str]]:
    """Check configuration settings."""
    results = []

    config_py = root / "apps/api/src/config.py"
    content = read_file(config_py)

    if not content:
        results.append(("config.py exists", False, "File not found"))
        return results

    results.append(("config.py exists", True, ""))

    # Check CLS endpoint settings
    if "runpod_endpoint_cls_annotation" in content:
        results.append(("CLS annotation setting defined", True, ""))
    else:
        results.append(("CLS annotation setting defined", False,
                       "Missing runpod_endpoint_cls_annotation"))

    if "runpod_endpoint_cls_training" in content:
        results.append(("CLS training setting defined", True, ""))
    else:
        results.append(("CLS training setting defined", False,
                       "Missing runpod_endpoint_cls_training"))

    return results


def main():
    """Run all integration checks."""
    print_header("CLS Integration Validator")

    root = find_project_root()
    print_info(f"Project root: {root}")

    all_results: Dict[str, List[Tuple[str, bool, str]]] = {}

    # Run all checks
    print_header("1. CLS Annotation API")
    all_results["CLS Annotation API"] = check_cls_annotation_api(root)

    print_header("2. CLS Training API")
    all_results["CLS Training API"] = check_cls_training_api(root)

    print_header("3. CLS Annotation Worker")
    all_results["CLS Annotation Worker"] = check_cls_annotation_worker(root)

    print_header("4. CLS Training Worker")
    all_results["CLS Training Worker"] = check_cls_training_worker(root)

    print_header("5. RunPod Service")
    all_results["RunPod Service"] = check_runpod_service(root)

    print_header("6. Configuration")
    all_results["Configuration"] = check_config(root)

    # Print results
    total_pass = 0
    total_fail = 0

    for category, results in all_results.items():
        print(f"\n{Colors.BOLD}{category}:{Colors.RESET}")
        for name, passed, msg in results:
            if passed:
                print_pass(name)
                total_pass += 1
            else:
                print_fail(f"{name}: {msg}")
                total_fail += 1

    # Summary
    print_header("Summary")
    print(f"  Total: {total_pass + total_fail}")
    print(f"  {Colors.GREEN}Passed: {total_pass}{Colors.RESET}")
    print(f"  {Colors.RED}Failed: {total_fail}{Colors.RESET}")

    if total_fail == 0:
        print(f"\n{Colors.GREEN}{Colors.BOLD}All integration checks passed!{Colors.RESET}")
        return 0
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}Some integration checks failed!{Colors.RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
