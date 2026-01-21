"""
Unit tests for handler.py augmentation config conversion.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from handler import convert_frontend_augmentation_config


def test_empty_config():
    """Test empty config returns None."""
    result = convert_frontend_augmentation_config(None)
    assert result is None, "None input should return None"

    result2 = convert_frontend_augmentation_config({})
    assert result2 is None, "Empty dict should return None"

    print("✅ TC-HANDLER-001: Empty config handling - PASSED")


def test_basic_augmentation_conversion():
    """Test basic augmentation config conversion."""
    frontend_config = {
        "mosaic": {"enabled": True, "probability": 0.5},
        "mixup": {"enabled": True, "probability": 0.3},
    }

    result = convert_frontend_augmentation_config(frontend_config)

    assert result is not None, "Should return result"
    assert "mosaic" in result, "Should have mosaic"
    assert "mixup" in result, "Should have mixup"
    assert result["mosaic"]["enabled"] is True
    assert result["mosaic"]["prob"] == 0.5
    assert result["mixup"]["prob"] == 0.3

    print("✅ TC-HANDLER-002: Basic conversion - PASSED")


def test_legacy_alias_conversion():
    """Test legacy key aliases are converted."""
    frontend_config = {
        "copy_paste": {"enabled": True, "probability": 0.2},  # Legacy name
    }

    result = convert_frontend_augmentation_config(frontend_config)

    assert "copypaste" in result, "copy_paste should be converted to copypaste"
    assert result["copypaste"]["enabled"] is True
    assert result["copypaste"]["prob"] == 0.2

    print("✅ TC-HANDLER-003: Legacy alias conversion - PASSED")


def test_params_extraction():
    """Test augmentation params are properly extracted."""
    frontend_config = {
        "mosaic": {
            "enabled": True,
            "probability": 0.5,
            "img_size": 640,
            "center_ratio": 0.5,
        },
        "shift_scale_rotate": {
            "enabled": True,
            "probability": 0.4,
            "shift_limit": 0.1,
            "scale_limit": 0.2,
            "rotate_limit": 15,
        },
    }

    result = convert_frontend_augmentation_config(frontend_config)

    assert "params" in result["mosaic"], "Mosaic should have params"
    assert result["mosaic"]["params"]["img_size"] == 640
    assert result["mosaic"]["params"]["center_ratio"] == 0.5

    assert "params" in result["shift_scale_rotate"]
    assert result["shift_scale_rotate"]["params"]["shift_limit"] == 0.1
    assert result["shift_scale_rotate"]["params"]["scale_limit"] == 0.2
    assert result["shift_scale_rotate"]["params"]["rotate_limit"] == 15

    print("✅ TC-HANDLER-004: Params extraction - PASSED")


def test_all_augmentation_types():
    """Test all augmentation types are mapped correctly."""
    # Test one from each category
    frontend_config = {
        # Multi-image
        "mosaic": {"enabled": True, "probability": 0.5},
        "mosaic9": {"enabled": True, "probability": 0.2},
        "mixup": {"enabled": True, "probability": 0.3},
        "cutmix": {"enabled": True, "probability": 0.2},
        "copypaste": {"enabled": True, "probability": 0.2},

        # Geometric
        "horizontal_flip": {"enabled": True, "probability": 0.5},
        "vertical_flip": {"enabled": True, "probability": 0.2},
        "rotate90": {"enabled": True, "probability": 0.3},
        "random_rotate": {"enabled": True, "probability": 0.3},
        "shift_scale_rotate": {"enabled": True, "probability": 0.5},
        "affine": {"enabled": True, "probability": 0.3},
        "perspective": {"enabled": True, "probability": 0.2},
        "safe_rotate": {"enabled": True, "probability": 0.2},
        "random_crop": {"enabled": True, "probability": 0.3},
        "random_scale": {"enabled": True, "probability": 0.5},
        "grid_distortion": {"enabled": True, "probability": 0.1},
        "elastic_transform": {"enabled": True, "probability": 0.1},
        "optical_distortion": {"enabled": True, "probability": 0.1},
        "piecewise_affine": {"enabled": True, "probability": 0.1},

        # Color
        "brightness_contrast": {"enabled": True, "probability": 0.4},
        "color_jitter": {"enabled": True, "probability": 0.4},
        "hue_saturation": {"enabled": True, "probability": 0.3},
        "random_gamma": {"enabled": True, "probability": 0.2},
        "rgb_shift": {"enabled": True, "probability": 0.2},
        "channel_shuffle": {"enabled": True, "probability": 0.1},
        "clahe": {"enabled": True, "probability": 0.2},
        "equalize": {"enabled": True, "probability": 0.1},
        "random_tone_curve": {"enabled": True, "probability": 0.1},
        "posterize": {"enabled": True, "probability": 0.1},
        "solarize": {"enabled": True, "probability": 0.1},
        "sharpen": {"enabled": True, "probability": 0.1},
        "unsharp_mask": {"enabled": True, "probability": 0.1},
        "fancy_pca": {"enabled": True, "probability": 0.1},
        "invert_img": {"enabled": True, "probability": 0.05},
        "to_gray": {"enabled": True, "probability": 0.1},

        # Blur
        "gaussian_blur": {"enabled": True, "probability": 0.1},
        "motion_blur": {"enabled": True, "probability": 0.1},
        "median_blur": {"enabled": True, "probability": 0.1},
        "defocus": {"enabled": True, "probability": 0.05},
        "zoom_blur": {"enabled": True, "probability": 0.05},
        "glass_blur": {"enabled": True, "probability": 0.05},
        "advanced_blur": {"enabled": True, "probability": 0.05},

        # Noise
        "gaussian_noise": {"enabled": True, "probability": 0.1},
        "iso_noise": {"enabled": True, "probability": 0.1},
        "multiplicative_noise": {"enabled": True, "probability": 0.1},

        # Quality
        "image_compression": {"enabled": True, "probability": 0.2},
        "downscale": {"enabled": True, "probability": 0.1},

        # Dropout
        "coarse_dropout": {"enabled": True, "probability": 0.1},
        "grid_dropout": {"enabled": True, "probability": 0.1},
        "pixel_dropout": {"enabled": True, "probability": 0.05},
        "mask_dropout": {"enabled": True, "probability": 0.1},

        # Weather
        "random_rain": {"enabled": True, "probability": 0.1},
        "random_fog": {"enabled": True, "probability": 0.1},
        "random_shadow": {"enabled": True, "probability": 0.1},
        "random_sun_flare": {"enabled": True, "probability": 0.05},
        "random_snow": {"enabled": True, "probability": 0.05},
        "spatter": {"enabled": True, "probability": 0.05},
        "plasma_brightness": {"enabled": True, "probability": 0.05},
    }

    result = convert_frontend_augmentation_config(frontend_config)

    assert len(result) == len(frontend_config), f"Should convert all {len(frontend_config)} augmentations"

    # Verify each one has correct structure
    for key, val in result.items():
        assert "enabled" in val, f"{key} should have enabled"
        assert val["enabled"] is True, f"{key} should be enabled"
        assert "prob" in val, f"{key} should have prob"

    print(f"✅ TC-HANDLER-005: All {len(result)} augmentation types mapped - PASSED")


def test_prob_vs_probability():
    """Test both prob and probability are accepted."""
    # Using 'probability'
    config1 = {"mosaic": {"enabled": True, "probability": 0.5}}
    result1 = convert_frontend_augmentation_config(config1)
    assert result1["mosaic"]["prob"] == 0.5

    # Using 'prob' directly
    config2 = {"mosaic": {"enabled": True, "prob": 0.6}}
    result2 = convert_frontend_augmentation_config(config2)
    assert result2["mosaic"]["prob"] == 0.6

    print("✅ TC-HANDLER-006: Both prob and probability accepted - PASSED")


def test_disabled_augmentation():
    """Test disabled augmentations are handled."""
    frontend_config = {
        "mosaic": {"enabled": False, "probability": 0.5},
    }

    result = convert_frontend_augmentation_config(frontend_config)

    assert result["mosaic"]["enabled"] is False, "Should preserve enabled=False"

    print("✅ TC-HANDLER-007: Disabled augmentation handling - PASSED")


def test_unknown_augmentation_passthrough():
    """Test unknown augmentations are passed through."""
    frontend_config = {
        "unknown_custom_aug": {"enabled": True, "probability": 0.3, "custom_param": 42},
    }

    result = convert_frontend_augmentation_config(frontend_config)

    # Unknown augs should be passed through with frontend key
    assert "unknown_custom_aug" in result, "Unknown aug should pass through"
    assert result["unknown_custom_aug"]["enabled"] is True
    assert result["unknown_custom_aug"]["prob"] == 0.3
    assert result["unknown_custom_aug"]["params"]["custom_param"] == 42

    print("✅ TC-HANDLER-008: Unknown augmentation passthrough - PASSED")


def test_non_dict_values_ignored():
    """Test non-dict values are ignored."""
    frontend_config = {
        "mosaic": {"enabled": True, "probability": 0.5},
        "invalid": "not_a_dict",
        "also_invalid": 123,
    }

    result = convert_frontend_augmentation_config(frontend_config)

    assert "mosaic" in result, "Valid aug should be included"
    assert "invalid" not in result, "String value should be ignored"
    assert "also_invalid" not in result, "Number value should be ignored"

    print("✅ TC-HANDLER-009: Non-dict values ignored - PASSED")


def test_complex_params():
    """Test complex nested params."""
    frontend_config = {
        "color_jitter": {
            "enabled": True,
            "probability": 0.4,
            "brightness": 0.2,
            "contrast": 0.2,
            "saturation": 0.3,
            "hue": 0.015,
        },
        "defocus": {
            "enabled": True,
            "probability": 0.05,
            "radius": [3, 7],  # Tuple as list
            "alias_blur": [0.1, 0.3],
        },
    }

    result = convert_frontend_augmentation_config(frontend_config)

    assert result["color_jitter"]["params"]["brightness"] == 0.2
    assert result["color_jitter"]["params"]["saturation"] == 0.3
    assert result["defocus"]["params"]["radius"] == [3, 7]

    print("✅ TC-HANDLER-010: Complex params handling - PASSED")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("RUNNING HANDLER AUGMENTATION CONVERSION TESTS")
    print("="*60 + "\n")

    test_empty_config()
    test_basic_augmentation_conversion()
    test_legacy_alias_conversion()
    test_params_extraction()
    test_all_augmentation_types()
    test_prob_vs_probability()
    test_disabled_augmentation()
    test_unknown_augmentation_passthrough()
    test_non_dict_values_ignored()
    test_complex_params()

    print("\n" + "="*60)
    print("✅ ALL HANDLER TESTS PASSED!")
    print("="*60 + "\n")
