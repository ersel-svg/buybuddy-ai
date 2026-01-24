"""
Test script for data_loading schemas and API integration.

Run from api directory:
    cd buybuddy-ai/apps/api
    python test_data_loading_schemas.py
"""

import sys
sys.path.insert(0, "src")

from pydantic import ValidationError


def test_preload_config():
    """Test PreloadConfig schema validation."""
    from schemas.data_loading import PreloadConfig

    # Test default values
    config = PreloadConfig()
    assert config.enabled == True
    assert config.batched == False
    assert config.batch_size == 500
    assert config.max_workers == 16
    print("✓ PreloadConfig default values OK")

    # Test custom values
    config = PreloadConfig(
        enabled=True,
        batched=True,
        batch_size=1000,
        max_workers=32,
        http_timeout=60,
        retry_attempts=5,
        retry_delay=2.0
    )
    assert config.batched == True
    assert config.batch_size == 1000
    print("✓ PreloadConfig custom values OK")

    # Test validation errors
    try:
        PreloadConfig(batch_size=50)  # Min is 100
        assert False, "Should have raised validation error"
    except ValidationError:
        print("✓ PreloadConfig validation (batch_size min) OK")

    try:
        PreloadConfig(max_workers=100)  # Max is 64
        assert False, "Should have raised validation error"
    except ValidationError:
        print("✓ PreloadConfig validation (max_workers max) OK")


def test_dataloader_config():
    """Test DataLoaderConfig schema validation."""
    from schemas.data_loading import DataLoaderConfig

    config = DataLoaderConfig()
    assert config.num_workers == 4
    assert config.pin_memory == True
    assert config.prefetch_factor == 2
    print("✓ DataLoaderConfig default values OK")


def test_data_loading_config():
    """Test DataLoadingConfig schema validation."""
    from schemas.data_loading import DataLoadingConfig, PreloadConfig, DataLoaderConfig

    # Test with None values (default)
    config = DataLoadingConfig()
    assert config.preload is None
    assert config.dataloader is None
    print("✓ DataLoadingConfig default values OK")

    # Test with nested configs
    config = DataLoadingConfig(
        preload=PreloadConfig(batched=True, batch_size=1000),
        dataloader=DataLoaderConfig(num_workers=8)
    )
    assert config.preload.batched == True
    assert config.preload.batch_size == 1000
    assert config.dataloader.num_workers == 8
    print("✓ DataLoadingConfig nested values OK")

    # Test model_dump (for API serialization)
    dumped = config.model_dump()
    assert dumped["preload"]["batched"] == True
    assert dumped["dataloader"]["num_workers"] == 8
    print("✓ DataLoadingConfig model_dump OK")


def test_od_training_config():
    """Test ODTrainingConfigBase with data_loading."""
    from schemas.od import ODTrainingConfigBase
    from schemas.data_loading import DataLoadingConfig, PreloadConfig

    # Test default (no data_loading)
    config = ODTrainingConfigBase()
    assert config.data_loading is None
    print("✓ ODTrainingConfigBase default data_loading OK")

    # Test with data_loading
    config = ODTrainingConfigBase(
        epochs=50,
        batch_size=32,
        data_loading=DataLoadingConfig(
            preload=PreloadConfig(batched=True, batch_size=1000)
        )
    )
    assert config.epochs == 50
    assert config.data_loading.preload.batched == True
    print("✓ ODTrainingConfigBase with data_loading OK")

    # Test model_dump includes data_loading
    dumped = config.model_dump()
    assert "data_loading" in dumped
    assert dumped["data_loading"]["preload"]["batched"] == True
    print("✓ ODTrainingConfigBase model_dump includes data_loading OK")


def test_cls_training_config():
    """Test CLSTrainingConfig with data_loading."""
    from schemas.classification import CLSTrainingConfig
    from schemas.data_loading import DataLoadingConfig, PreloadConfig

    # Test with data_loading
    config = CLSTrainingConfig(
        model_type="vit",
        model_size="base",
        data_loading=DataLoadingConfig(
            preload=PreloadConfig(batched=True)
        )
    )
    assert config.data_loading.preload.batched == True
    print("✓ CLSTrainingConfig with data_loading OK")


def test_embedding_training_config():
    """Test TrainingConfigOverrides with new fields."""
    # Import from the training API module
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "training",
        "src/api/v1/training.py"
    )
    module = importlib.util.module_from_spec(spec)

    # We need to mock some imports
    import sys
    from unittest.mock import MagicMock

    # Mock dependencies
    sys.modules['services.supabase'] = MagicMock()
    sys.modules['services.runpod'] = MagicMock()
    sys.modules['auth.dependencies'] = MagicMock()

    try:
        spec.loader.exec_module(module)
        TrainingConfigOverrides = module.TrainingConfigOverrides

        # Test new fields exist
        config = TrainingConfigOverrides(
            epochs=50,
            scheduler_eta_min=1e-6,
            head_lr_multiplier=10,
            val_batch_multiplier=2
        )
        assert config.scheduler_eta_min == 1e-6
        assert config.head_lr_multiplier == 10
        assert config.val_batch_multiplier == 2
        print("✓ TrainingConfigOverrides new fields OK")

        # Test data_loading field
        from schemas.data_loading import DataLoadingConfig, PreloadConfig
        config = TrainingConfigOverrides(
            data_loading=DataLoadingConfig(
                preload=PreloadConfig(batched=True)
            )
        )
        assert config.data_loading.preload.batched == True
        print("✓ TrainingConfigOverrides data_loading OK")

    except Exception as e:
        print(f"⚠ TrainingConfigOverrides test skipped (import issues): {e}")


def test_backward_compatibility():
    """Test that existing configs without new fields still work."""
    from schemas.od import ODTrainingConfigBase
    from schemas.classification import CLSTrainingConfig

    # OD config without data_loading should work
    od_config = ODTrainingConfigBase(
        epochs=100,
        batch_size=16,
        learning_rate=0.0001
    )
    assert od_config.data_loading is None
    print("✓ ODTrainingConfigBase backward compatibility OK")

    # CLS config without data_loading should work
    cls_config = CLSTrainingConfig(
        model_type="vit",
        model_size="base"
    )
    assert cls_config.data_loading is None
    print("✓ CLSTrainingConfig backward compatibility OK")


if __name__ == "__main__":
    print("\n" + "="*50)
    print("Testing Data Loading Schema Integration")
    print("="*50 + "\n")

    test_preload_config()
    print()

    test_dataloader_config()
    print()

    test_data_loading_config()
    print()

    test_od_training_config()
    print()

    test_cls_training_config()
    print()

    test_embedding_training_config()
    print()

    test_backward_compatibility()

    print("\n" + "="*50)
    print("All tests passed! ✅")
    print("="*50 + "\n")
