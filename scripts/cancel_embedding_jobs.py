#!/usr/bin/env python3
"""
Cancel all active embedding jobs in both Supabase and RunPod.

Usage:
    python scripts/cancel_embedding_jobs.py

Environment variables required:
    - SUPABASE_URL
    - SUPABASE_SERVICE_ROLE_KEY
    - RUNPOD_API_KEY
    - RUNPOD_ENDPOINT_EMBEDDING
"""

import os
import sys
import httpx
from datetime import datetime, timezone

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to load from .env file
def load_env():
    env_paths = [
        os.path.join(os.path.dirname(__file__), '..', 'apps', 'api', '.env'),
        os.path.join(os.path.dirname(__file__), '..', '.env'),
        '.env',
    ]

    for env_path in env_paths:
        if os.path.exists(env_path):
            print(f"Loading environment from: {env_path}")
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key and key not in os.environ:
                            os.environ[key] = value
            break

load_env()

# Configuration
SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_KEY = os.environ.get('SUPABASE_SERVICE_ROLE_KEY')
RUNPOD_API_KEY = os.environ.get('RUNPOD_API_KEY')
RUNPOD_ENDPOINT_EMBEDDING = os.environ.get('RUNPOD_ENDPOINT_EMBEDDING')

RUNPOD_BASE_URL = "https://api.runpod.ai/v2"


def get_active_jobs():
    """Get all active embedding jobs from Supabase."""
    print("\n📋 Fetching active embedding jobs...")

    response = httpx.get(
        f"{SUPABASE_URL}/rest/v1/embedding_jobs",
        params={
            "select": "id,status,runpod_job_id,created_at,total_images,processed_images",
            "status": "in.(queued,running,pending)",
            "order": "created_at.desc",
        },
        headers={
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
        },
        timeout=30,
    )

    if response.status_code != 200:
        print(f"❌ Failed to fetch jobs: {response.status_code} - {response.text}")
        return []

    jobs = response.json()
    print(f"   Found {len(jobs)} active jobs")
    return jobs


def cancel_runpod_job(runpod_job_id: str) -> bool:
    """Cancel a job on RunPod."""
    if not runpod_job_id:
        return False

    if not RUNPOD_ENDPOINT_EMBEDDING:
        print(f"   ⚠️  RUNPOD_ENDPOINT_EMBEDDING not configured, skipping RunPod cancel")
        return False

    try:
        response = httpx.post(
            f"{RUNPOD_BASE_URL}/{RUNPOD_ENDPOINT_EMBEDDING}/cancel/{runpod_job_id}",
            headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
            timeout=30,
        )

        if response.status_code == 200:
            print(f"   ✅ RunPod job {runpod_job_id} cancelled")
            return True
        else:
            print(f"   ⚠️  RunPod cancel returned {response.status_code}: {response.text}")
            return False

    except Exception as e:
        print(f"   ⚠️  Failed to cancel RunPod job: {e}")
        return False


def update_job_status(job_id: str) -> bool:
    """Update job status to cancelled in Supabase."""
    try:
        response = httpx.patch(
            f"{SUPABASE_URL}/rest/v1/embedding_jobs",
            params={"id": f"eq.{job_id}"},
            headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal",
            },
            json={
                "status": "cancelled",
                "error_message": "Cancelled via cleanup script",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            timeout=30,
        )

        if response.status_code in [200, 204]:
            print(f"   ✅ DB status updated to cancelled")
            return True
        else:
            print(f"   ⚠️  DB update failed: {response.status_code}")
            return False

    except Exception as e:
        print(f"   ⚠️  Failed to update DB: {e}")
        return False


def purge_runpod_queue():
    """Purge all jobs from RunPod queue."""
    if not RUNPOD_ENDPOINT_EMBEDDING:
        print("⚠️  RUNPOD_ENDPOINT_EMBEDDING not configured")
        return

    print("\n🗑️  Purging RunPod queue...")

    try:
        response = httpx.post(
            f"{RUNPOD_BASE_URL}/{RUNPOD_ENDPOINT_EMBEDDING}/purge-queue",
            headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
            timeout=30,
        )

        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Queue purged: {result}")
        else:
            print(f"   ⚠️  Purge returned {response.status_code}: {response.text}")

    except Exception as e:
        print(f"   ⚠️  Failed to purge queue: {e}")


def main():
    print("=" * 60)
    print("EMBEDDING JOBS CLEANUP SCRIPT")
    print("=" * 60)

    # Validate configuration
    missing = []
    if not SUPABASE_URL:
        missing.append("SUPABASE_URL")
    if not SUPABASE_KEY:
        missing.append("SUPABASE_SERVICE_ROLE_KEY")
    if not RUNPOD_API_KEY:
        missing.append("RUNPOD_API_KEY")

    if missing:
        print(f"\n❌ Missing environment variables: {', '.join(missing)}")
        print("   Please set them or create a .env file")
        sys.exit(1)

    print(f"\n✅ Configuration loaded")
    print(f"   Supabase: {SUPABASE_URL[:30]}...")
    print(f"   RunPod Endpoint: {RUNPOD_ENDPOINT_EMBEDDING or 'NOT CONFIGURED'}")

    # Get active jobs
    jobs = get_active_jobs()

    if not jobs:
        print("\n✨ No active jobs to cancel!")
    else:
        print(f"\n🔄 Cancelling {len(jobs)} jobs...\n")

        for i, job in enumerate(jobs, 1):
            job_id = job['id']
            runpod_job_id = job.get('runpod_job_id')
            status = job['status']

            print(f"[{i}/{len(jobs)}] Job {job_id[:8]}...")
            print(f"   Status: {status}")
            print(f"   RunPod ID: {runpod_job_id or 'N/A'}")
            print(f"   Progress: {job.get('processed_images', 0)}/{job.get('total_images', 0)}")

            # Cancel on RunPod first
            if runpod_job_id:
                cancel_runpod_job(runpod_job_id)

            # Update DB status
            update_job_status(job_id)
            print()

    # Purge the RunPod queue to clear any pending jobs
    purge_runpod_queue()

    print("\n" + "=" * 60)
    print("✅ CLEANUP COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
