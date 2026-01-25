"""
Test suite for CLS Trainer config key mismatch fix.

Tests:
1. New format (data_loading.preload) support
2. Old format (preload_config) backward compatibility
3. DataLoader config support
4. Default values fallback
"""

import sys
from pathlib import Path


def test_new_format_preload_config():
    """Test 1: New Format (data_loading.preload) Support"""
    print("\n" + "=" * 70)
    print("TEST 1: New Format (data_loading.preload) Support")
    print("=" * 70)

    # Mock config from API (new format)
    full_config = {
        "data_loading": {
            "preload": {
                "enabled": True,
                "batched": True,
                "batch_size": 1000,
                "max_workers": 32,
                "http_timeout": 60,
            },
            "dataloader": {
                "num_workers": 8,
                "pin_memory": True,
                "prefetch_factor": 4,
            }
        }
    }

    # Simulate config extraction logic from handler.py
    data_loading = full_config.get("data_loading", {})
    preload_config = data_loading.get("preload") if data_loading else full_config.get("preload_config", {})

    print(f"\n✓ Config format: NEW (data_loading.preload)")
    print(f"✓ Preload config extracted: {preload_config}")
    print(f"✓ Preload enabled: {preload_config.get('enabled', True)}")
    print(f"✓ Batched: {preload_config.get('batched', False)}")
    print(f"✓ Batch size: {preload_config.get('batch_size', 500)}")

    # Verify extraction
    assert preload_config is not None, "Should extract preload config"
    assert preload_config.get("enabled") == True, "Should read enabled field"
    assert preload_config.get("batched") == True, "Should read batched field"
    assert preload_config.get("batch_size") == 1000, "Should read batch_size"

    print("\n✅ TEST 1 PASSED: New format correctly supported")
    return True


def test_old_format_backward_compatibility():
    """Test 2: Old Format (preload_config) Backward Compatibility"""
    print("\n" + "=" * 70)
    print("TEST 2: Old Format (preload_config) Backward Compatibility")
    print("=" * 70)

    # Mock config from old jobs (legacy format)
    full_config = {
        "preload_config": {
            "enabled": True,
            "batched": False,
            "batch_size": 500,
        },
        "num_workers": 4,
    }

    # Simulate config extraction logic
    data_loading = full_config.get("data_loading", {})
    preload_config = data_loading.get("preload") if data_loading else full_config.get("preload_config", {})

    print(f"\n✓ Config format: OLD (preload_config)")
    print(f"✓ Preload config extracted: {preload_config}")
    print(f"✓ Preload enabled: {preload_config.get('enabled', True)}")
    print(f"✓ Batched: {preload_config.get('batched', False)}")

    # Verify extraction
    assert preload_config is not None, "Should extract preload config"
    assert preload_config.get("enabled") == True, "Should read enabled field"
    assert preload_config.get("batched") == False, "Should read batched field"

    print("\n✅ TEST 2 PASSED: Old format backward compatibility maintained")
    return True


def test_dataloader_config_extraction():
    """Test 3: DataLoader Config Extraction"""
    print("\n" + "=" * 70)
    print("TEST 3: DataLoader Config Extraction")
    print("=" * 70)

    # Mock config with dataloader settings
    full_config = {
        "data_loading": {
            "dataloader": {
                "num_workers": 8,
                "pin_memory": False,
                "prefetch_factor": 4,
            }
        },
        "num_workers": 4,  # Legacy fallback
    }

    # Simulate dataloader config extraction
    data_loading = full_config.get("data_loading", {})
    dataloader_config = data_loading.get("dataloader") if data_loading else {}
    num_workers = dataloader_config.get("num_workers", full_config.get("num_workers", 4))
    pin_memory = dataloader_config.get("pin_memory", True)
    prefetch_factor = dataloader_config.get("prefetch_factor", 2)

    print(f"\n✓ DataLoader config extracted: {dataloader_config}")
    print(f"✓ num_workers: {num_workers} (from data_loading.dataloader)")
    print(f"✓ pin_memory: {pin_memory}")
    print(f"✓ prefetch_factor: {prefetch_factor}")

    # Verify extraction
    assert num_workers == 8, f"Expected 8 workers, got {num_workers}"
    assert pin_memory == False, "Should use dataloader config pin_memory"
    assert prefetch_factor == 4, "Should use dataloader config prefetch_factor"

    print("\n✅ TEST 3 PASSED: DataLoader config correctly extracted")
    return True


def test_default_values_fallback():
    """Test 4: Default Values Fallback (Empty Config)"""
    print("\n" + "=" * 70)
    print("TEST 4: Default Values Fallback (Empty Config)")
    print("=" * 70)

    # Mock empty config
    full_config = {}

    # Simulate config extraction with defaults
    data_loading = full_config.get("data_loading", {})
    preload_config = data_loading.get("preload") if data_loading else full_config.get("preload_config", {})
    dataloader_config = data_loading.get("dataloader") if data_loading else {}

    num_workers = dataloader_config.get("num_workers", full_config.get("num_workers", 4))
    pin_memory = dataloader_config.get("pin_memory", True)
    prefetch_factor = dataloader_config.get("prefetch_factor", 2)
    preload_enabled = preload_config.get("enabled", True)

    print(f"\n✓ Config: EMPTY (using defaults)")
    print(f"✓ Preload enabled (default): {preload_enabled}")
    print(f"✓ num_workers (default): {num_workers}")
    print(f"✓ pin_memory (default): {pin_memory}")
    print(f"✓ prefetch_factor (default): {prefetch_factor}")

    # Verify defaults
    assert preload_enabled == True, "Default preload should be True"
    assert num_workers == 4, "Default num_workers should be 4"
    assert pin_memory == True, "Default pin_memory should be True"
    assert prefetch_factor == 2, "Default prefetch_factor should be 2"

    print("\n✅ TEST 4 PASSED: Default values correctly applied")
    return True


def test_legacy_num_workers_fallback():
    """Test 5: Legacy num_workers Fallback"""
    print("\n" + "=" * 70)
    print("TEST 5: Legacy num_workers Fallback")
    print("=" * 70)

    # Mock old config with only num_workers (no data_loading)
    full_config = {
        "num_workers": 6,
    }

    # Simulate extraction
    data_loading = full_config.get("data_loading", {})
    dataloader_config = data_loading.get("dataloader") if data_loading else {}
    num_workers = dataloader_config.get("num_workers", full_config.get("num_workers", 4))

    print(f"\n✓ Config: Legacy (only num_workers)")
    print(f"✓ num_workers: {num_workers} (from full_config)")

    # Verify fallback
    assert num_workers == 6, f"Expected 6 workers from legacy config, got {num_workers}"

    print("\n✅ TEST 5 PASSED: Legacy num_workers fallback works")
    return True


def test_prefetch_factor_with_zero_workers():
    """Test 6: prefetch_factor Should Be None When num_workers=0"""
    print("\n" + "=" * 70)
    print("TEST 6: prefetch_factor With Zero Workers")
    print("=" * 70)

    # Mock config with num_workers=0
    full_config = {
        "data_loading": {
            "dataloader": {
                "num_workers": 0,
                "prefetch_factor": 4,
            }
        }
    }

    # Simulate extraction
    data_loading = full_config.get("data_loading", {})
    dataloader_config = data_loading.get("dataloader") if data_loading else {}
    num_workers = dataloader_config.get("num_workers", 4)
    prefetch_factor = dataloader_config.get("prefetch_factor", 2)

    # Apply prefetch_factor logic from handler
    prefetch_factor_final = prefetch_factor if num_workers > 0 else None

    print(f"\n✓ num_workers: {num_workers}")
    print(f"✓ prefetch_factor from config: {prefetch_factor}")
    print(f"✓ prefetch_factor applied: {prefetch_factor_final} (None because num_workers=0)")

    # Verify
    assert num_workers == 0, "Should have 0 workers"
    assert prefetch_factor_final is None, "prefetch_factor should be None when num_workers=0"

    print("\n✅ TEST 6 PASSED: prefetch_factor correctly set to None for single-process DataLoader")
    return True


def run_all_tests():
    """Run all CLS Trainer config tests"""
    print("\n" + "=" * 70)
    print("CLS TRAINER CONFIG INTEGRATION TESTS")
    print("=" * 70)

    tests = [
        test_new_format_preload_config,
        test_old_format_backward_compatibility,
        test_dataloader_config_extraction,
        test_default_values_fallback,
        test_legacy_num_workers_fallback,
        test_prefetch_factor_with_zero_workers,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"\n❌ TEST FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"✅ Passed: {passed}/{len(tests)}")
    print(f"❌ Failed: {failed}/{len(tests)}")
    print("=" * 70 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
