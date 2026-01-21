"""
Unit tests for augmentation presets.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.augmentations.presets import (
    get_preset,
    get_preset_info,
    get_augmentation_categories,
    get_all_augmentation_names,
    validate_custom_config,
    AugmentationPreset,
    AugmentationConfig,
    PresetLevel,
)


def test_all_presets_exist():
    """Test that all presets can be loaded."""
    presets = ["sota-v2", "sota", "heavy", "medium", "light", "none"]
    for preset_name in presets:
        preset = get_preset(preset_name)
        assert preset is not None, f"Preset {preset_name} should exist"
        assert isinstance(preset, AugmentationPreset), f"Preset {preset_name} should be AugmentationPreset"
    print("✅ TC-PRESET-001: All presets exist - PASSED")


def test_sota_v2_preset_configuration():
    """Test SOTA-v2 preset has expected configuration."""
    preset = get_preset("sota-v2")

    # Core SOTA-v2 features should be enabled
    assert preset.mosaic.enabled, "SOTA-v2 should have mosaic enabled"
    assert preset.mixup.enabled, "SOTA-v2 should have mixup enabled"
    assert preset.copypaste.enabled, "SOTA-v2 should have copypaste enabled"
    assert preset.horizontal_flip.enabled, "SOTA-v2 should have horizontal_flip enabled"
    assert preset.shift_scale_rotate.enabled, "SOTA-v2 should have shift_scale_rotate enabled"
    assert preset.color_jitter.enabled, "SOTA-v2 should have color_jitter enabled"
    assert preset.image_compression.enabled, "SOTA-v2 should have image_compression enabled"

    # Check probabilities are reasonable
    assert 0 < preset.mosaic.prob <= 1, "Mosaic prob should be between 0 and 1"
    assert 0 < preset.mixup.prob <= 1, "MixUp prob should be between 0 and 1"

    print("✅ TC-PRESET-002: SOTA-v2 configuration - PASSED")


def test_preset_overrides():
    """Test that preset overrides work correctly."""
    preset = get_preset("sota", overrides={
        "mosaic": {"prob": 0.8},
        "mixup": {"enabled": False},
    })

    assert preset.mosaic.prob == 0.8, "Mosaic prob should be overridden to 0.8"
    assert not preset.mixup.enabled, "MixUp should be disabled"

    print("✅ TC-PRESET-003: Preset overrides - PASSED")


def test_custom_preset():
    """Test custom preset starts empty and accepts overrides."""
    preset = get_preset("custom", overrides={
        "horizontal_flip": {"enabled": True, "prob": 0.5},
    })

    assert preset.horizontal_flip.enabled, "Horizontal flip should be enabled"
    assert preset.horizontal_flip.prob == 0.5, "Horizontal flip prob should be 0.5"
    assert not preset.mosaic.enabled, "Mosaic should be disabled by default"

    print("✅ TC-PRESET-004: Custom preset - PASSED")


def test_get_preset_info():
    """Test preset info retrieval."""
    info = get_preset_info()

    expected_presets = ["sota-v2", "sota", "heavy", "medium", "light", "none"]
    for preset_name in expected_presets:
        assert preset_name in info, f"Info should contain {preset_name}"
        assert "name" in info[preset_name], f"{preset_name} should have name"
        assert "description" in info[preset_name], f"{preset_name} should have description"
        assert "features" in info[preset_name], f"{preset_name} should have features"

    print("✅ TC-PRESET-005: Get preset info - PASSED")


def test_get_augmentation_categories():
    """Test augmentation categories retrieval."""
    categories = get_augmentation_categories()

    expected_categories = [
        "multi_image", "geometric", "color", "blur",
        "noise", "quality", "dropout", "weather"
    ]

    for cat_name in expected_categories:
        assert cat_name in categories, f"Category {cat_name} should exist"
        assert "name" in categories[cat_name], f"{cat_name} should have name"
        assert "augmentations" in categories[cat_name], f"{cat_name} should have augmentations"

    # Check specific augmentations exist
    assert "mosaic" in categories["multi_image"]["augmentations"]
    assert "horizontal_flip" in categories["geometric"]["augmentations"]
    assert "shift_scale_rotate" in categories["geometric"]["augmentations"]
    assert "color_jitter" in categories["color"]["augmentations"]
    assert "gaussian_blur" in categories["blur"]["augmentations"]
    assert "gaussian_noise" in categories["noise"]["augmentations"]
    assert "image_compression" in categories["quality"]["augmentations"]
    assert "coarse_dropout" in categories["dropout"]["augmentations"]
    assert "random_rain" in categories["weather"]["augmentations"]

    print("✅ TC-PRESET-006: Get augmentation categories - PASSED")


def test_get_all_augmentation_names():
    """Test all augmentation names retrieval."""
    names = get_all_augmentation_names()

    # Should have 40+ augmentations
    assert len(names) >= 40, f"Should have at least 40 augmentations, got {len(names)}"

    # Check some specific names
    expected_augs = [
        "mosaic", "mosaic9", "mixup", "cutmix", "copypaste",
        "horizontal_flip", "vertical_flip", "shift_scale_rotate",
        "color_jitter", "brightness_contrast", "random_gamma",
        "gaussian_blur", "defocus", "zoom_blur",
        "gaussian_noise", "iso_noise",
        "image_compression", "downscale",
        "coarse_dropout", "grid_dropout",
        "random_rain", "random_fog", "random_sun_flare"
    ]

    for aug in expected_augs:
        assert aug in names, f"Augmentation {aug} should exist"

    print(f"✅ TC-PRESET-007: Get all augmentation names ({len(names)} augmentations) - PASSED")


def test_validate_custom_config_valid():
    """Test validation accepts valid config."""
    valid_config = {
        "mosaic": {"enabled": True, "prob": 0.5},
        "horizontal_flip": {"enabled": True, "prob": 0.5},
    }

    errors = validate_custom_config(valid_config)
    assert len(errors) == 0, f"Valid config should have no errors, got: {errors}"

    print("✅ TC-PRESET-008: Validate valid config - PASSED")


def test_validate_custom_config_invalid():
    """Test validation catches invalid config."""
    # Invalid augmentation name
    invalid_config = {
        "nonexistent_aug": {"enabled": True},
    }
    errors = validate_custom_config(invalid_config)
    assert len(errors) > 0, "Should catch unknown augmentation"

    # Invalid probability
    invalid_config2 = {
        "mosaic": {"enabled": True, "prob": 1.5},  # > 1
    }
    errors2 = validate_custom_config(invalid_config2)
    assert len(errors2) > 0, "Should catch invalid probability"

    print("✅ TC-PRESET-009: Validate invalid config - PASSED")


def test_heavy_preset_has_all_categories():
    """Test heavy preset enables augmentations from all categories."""
    preset = get_preset("heavy")
    enabled = preset.get_enabled_augmentations()

    # Heavy should have many augmentations enabled
    assert len(enabled) >= 20, f"Heavy should have many augmentations, got {len(enabled)}"

    # Should have augmentations from multiple categories
    has_multi_image = any(a in enabled for a in ["mosaic", "mixup", "copypaste"])
    has_geometric = any(a in enabled for a in ["horizontal_flip", "shift_scale_rotate"])
    has_color = any(a in enabled for a in ["brightness_contrast", "color_jitter"])
    has_blur = any(a in enabled for a in ["gaussian_blur", "motion_blur"])

    assert has_multi_image, "Heavy should have multi-image augs"
    assert has_geometric, "Heavy should have geometric augs"
    assert has_color, "Heavy should have color augs"
    assert has_blur, "Heavy should have blur augs"

    print(f"✅ TC-PRESET-010: Heavy preset coverage ({len(enabled)} augmentations) - PASSED")


def test_none_preset_has_nothing_enabled():
    """Test none preset has no augmentations enabled."""
    preset = get_preset("none")
    enabled = preset.get_enabled_augmentations()

    assert len(enabled) == 0, f"None preset should have no augmentations enabled, got {enabled}"

    print("✅ TC-PRESET-011: None preset is empty - PASSED")


def test_preset_to_dict():
    """Test preset can be serialized to dict."""
    preset = get_preset("sota-v2")
    data = preset.to_dict()

    assert isinstance(data, dict), "to_dict should return dict"
    assert "mosaic" in data, "Dict should contain mosaic"
    assert "enabled" in data["mosaic"], "Mosaic should have enabled"
    assert "prob" in data["mosaic"], "Mosaic should have prob"

    print("✅ TC-PRESET-012: Preset to_dict - PASSED")


def test_preset_from_dict():
    """Test preset can be created from dict."""
    data = {
        "mosaic": {"enabled": True, "prob": 0.7, "params": {"img_size": 640}},
        "mixup": {"enabled": False, "prob": 0.0, "params": {}},
    }

    preset = AugmentationPreset.from_dict(data)

    assert preset.mosaic.enabled, "Mosaic should be enabled"
    assert preset.mosaic.prob == 0.7, "Mosaic prob should be 0.7"
    assert not preset.mixup.enabled, "MixUp should be disabled"

    print("✅ TC-PRESET-013: Preset from_dict - PASSED")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("RUNNING PRESET UNIT TESTS")
    print("="*60 + "\n")

    test_all_presets_exist()
    test_sota_v2_preset_configuration()
    test_preset_overrides()
    test_custom_preset()
    test_get_preset_info()
    test_get_augmentation_categories()
    test_get_all_augmentation_names()
    test_validate_custom_config_valid()
    test_validate_custom_config_invalid()
    test_heavy_preset_has_all_categories()
    test_none_preset_has_nothing_enabled()
    test_preset_to_dict()
    test_preset_from_dict()

    print("\n" + "="*60)
    print("✅ ALL PRESET TESTS PASSED!")
    print("="*60 + "\n")
