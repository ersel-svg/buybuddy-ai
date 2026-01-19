#!/usr/bin/env python3
"""Quick test for checkpoint upload."""
import sys
sys.path.insert(0, '/workspace/training/src')
import torch
import os

print("Creating test model...")
from trainer import EmbeddingModel

model = EmbeddingModel(
    model_type="dinov2-base",
    embedding_dim=512,
    num_classes=20,
    use_arcface=True,
)

# Save checkpoint
checkpoint_path = "/workspace/test_checkpoint.pth"
checkpoint_data = {
    "model_state_dict": model.state_dict(),
    "epoch": 0,
    "val_loss": 0.5,
    "val_recall_at_1": 0.8,
}

print("Saving checkpoint...")
torch.save(checkpoint_data, checkpoint_path)
size = os.path.getsize(checkpoint_path)
print(f"Checkpoint saved: {size / 1024 / 1024:.1f}MB")

# Now test upload
print("\nTesting checkpoint upload...")
from checkpoint_upload import upload_checkpoint_to_storage, MAX_UPLOAD_SIZE
from supabase import create_client

print(f"MAX_UPLOAD_SIZE: {MAX_UPLOAD_SIZE / 1024 / 1024:.0f}MB")

client = create_client(os.environ['SUPABASE_URL'], os.environ['SUPABASE_SERVICE_ROLE_KEY'])

url = upload_checkpoint_to_storage(
    client=client,
    checkpoint_path=checkpoint_path,
    training_run_id='qa-test-upload',
    epoch=0,
    is_best=True,
)

if url:
    print(f"\nSUCCESS! Checkpoint uploaded to: {url}")
else:
    print("\nFAILED: Upload returned None")

# Cleanup
os.remove(checkpoint_path)
print("Test checkpoint removed")
