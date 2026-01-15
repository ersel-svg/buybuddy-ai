"""Resource locks API router for multi-user editing support."""

from datetime import datetime, timezone, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from services.supabase import SupabaseService, supabase_service
from auth.dependencies import get_current_user
from auth.service import UserInfo

# Router with authentication required for all endpoints
router = APIRouter(dependencies=[Depends(get_current_user)])


# ===========================================
# Schemas
# ===========================================


class AcquireLockRequest(BaseModel):
    """Request to acquire a resource lock."""

    resource_type: str  # "product" or "dataset"
    resource_id: str


class LockResponse(BaseModel):
    """Lock information response."""

    id: str
    resource_type: str
    resource_id: str
    user_id: str
    user_email: Optional[str] = None
    locked_at: str
    expires_at: str


class LockStatusResponse(BaseModel):
    """Lock status check response."""

    is_locked: bool
    lock: Optional[LockResponse] = None
    can_edit: bool  # True if not locked or locked by current user


# ===========================================
# Dependency
# ===========================================


def get_supabase() -> SupabaseService:
    """Get Supabase service instance."""
    return supabase_service


# ===========================================
# Helper Functions
# ===========================================


LOCK_DURATION_MINUTES = 5  # Lock expires after 5 minutes


def cleanup_expired_locks(db: SupabaseService) -> int:
    """Remove expired locks. Returns count of removed locks."""
    now = datetime.now(timezone.utc).isoformat()

    # supabase-py is sync, no await needed
    result = db.client.table("resource_locks").delete().lt("expires_at", now).execute()

    return len(result.data) if result.data else 0


def get_lock_for_resource(
    db: SupabaseService,
    resource_type: str,
    resource_id: str,
) -> Optional[dict]:
    """Get active lock for a resource (if any)."""
    # First cleanup expired locks
    cleanup_expired_locks(db)

    # supabase-py is sync, no await needed
    result = db.client.table("resource_locks").select("*").eq("resource_type", resource_type).eq("resource_id", resource_id).execute()

    if result.data and len(result.data) > 0:
        return result.data[0]
    return None


# ===========================================
# Endpoints
# ===========================================


@router.post("/acquire", response_model=LockResponse)
async def acquire_lock(
    request: AcquireLockRequest,
    user: UserInfo = Depends(get_current_user),
    db: SupabaseService = Depends(get_supabase),
) -> LockResponse:
    """
    Acquire a lock on a resource for editing.

    - Returns the lock if successfully acquired
    - Returns 409 Conflict if resource is already locked by another user
    - If already locked by current user, extends the lock
    """
    # Validate resource type
    if request.resource_type not in ["product", "dataset"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid resource type. Must be 'product' or 'dataset'",
        )

    # Check for existing lock
    existing_lock = get_lock_for_resource(db, request.resource_type, request.resource_id)

    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(minutes=LOCK_DURATION_MINUTES)

    if existing_lock:
        # If locked by same user, extend the lock
        if existing_lock["user_id"] == user.username:
            result = db.client.table("resource_locks").update({"expires_at": expires_at.isoformat()}).eq("id", existing_lock["id"]).execute()

            lock_data = result.data[0] if result.data else existing_lock
            return LockResponse(
                id=lock_data["id"],
                resource_type=lock_data["resource_type"],
                resource_id=lock_data["resource_id"],
                user_id=lock_data["user_id"],
                user_email=lock_data.get("user_email"),
                locked_at=lock_data["locked_at"],
                expires_at=expires_at.isoformat(),
            )

        # Locked by another user
        raise HTTPException(
            status_code=409,
            detail=f"Resource is locked by {existing_lock['user_id']}. Try again in a few minutes.",
        )

    # Create new lock
    lock_data = {
        "resource_type": request.resource_type,
        "resource_id": request.resource_id,
        "user_id": user.username,
        "user_email": None,  # Could be added if we have email in UserInfo
        "locked_at": now.isoformat(),
        "expires_at": expires_at.isoformat(),
    }

    result = db.client.table("resource_locks").insert(lock_data).execute()

    if not result.data:
        raise HTTPException(status_code=500, detail="Failed to create lock")

    created_lock = result.data[0]
    return LockResponse(
        id=created_lock["id"],
        resource_type=created_lock["resource_type"],
        resource_id=created_lock["resource_id"],
        user_id=created_lock["user_id"],
        user_email=created_lock.get("user_email"),
        locked_at=created_lock["locked_at"],
        expires_at=created_lock["expires_at"],
    )


@router.delete("/{lock_id}")
async def release_lock(
    lock_id: str,
    user: UserInfo = Depends(get_current_user),
    db: SupabaseService = Depends(get_supabase),
) -> dict:
    """
    Release a lock.

    - Only the user who owns the lock can release it
    """
    # Get the lock
    result = db.client.table("resource_locks").select("*").eq("id", lock_id).execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Lock not found")

    lock = result.data[0]

    # Verify ownership
    if lock["user_id"] != user.username:
        raise HTTPException(
            status_code=403,
            detail="You can only release locks you own",
        )

    # Delete the lock
    db.client.table("resource_locks").delete().eq("id", lock_id).execute()

    return {"status": "released", "lock_id": lock_id}


@router.get("/{resource_type}/{resource_id}", response_model=LockStatusResponse)
async def get_lock_status(
    resource_type: str,
    resource_id: str,
    user: UserInfo = Depends(get_current_user),
    db: SupabaseService = Depends(get_supabase),
) -> LockStatusResponse:
    """
    Check if a resource is locked and who owns the lock.

    - Returns lock info if locked
    - Indicates if current user can edit (unlocked or owns the lock)
    """
    # Validate resource type
    if resource_type not in ["product", "dataset"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid resource type. Must be 'product' or 'dataset'",
        )

    existing_lock = get_lock_for_resource(db, resource_type, resource_id)

    if not existing_lock:
        return LockStatusResponse(
            is_locked=False,
            lock=None,
            can_edit=True,
        )

    lock_response = LockResponse(
        id=existing_lock["id"],
        resource_type=existing_lock["resource_type"],
        resource_id=existing_lock["resource_id"],
        user_id=existing_lock["user_id"],
        user_email=existing_lock.get("user_email"),
        locked_at=existing_lock["locked_at"],
        expires_at=existing_lock["expires_at"],
    )

    can_edit = existing_lock["user_id"] == user.username

    return LockStatusResponse(
        is_locked=True,
        lock=lock_response,
        can_edit=can_edit,
    )


@router.post("/{lock_id}/refresh", response_model=LockResponse)
async def refresh_lock(
    lock_id: str,
    user: UserInfo = Depends(get_current_user),
    db: SupabaseService = Depends(get_supabase),
) -> LockResponse:
    """
    Extend a lock's expiration time.

    - Only the user who owns the lock can refresh it
    - Extends lock by another LOCK_DURATION_MINUTES
    """
    # Get the lock
    result = db.client.table("resource_locks").select("*").eq("id", lock_id).execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Lock not found or expired")

    lock = result.data[0]

    # Verify ownership
    if lock["user_id"] != user.username:
        raise HTTPException(
            status_code=403,
            detail="You can only refresh locks you own",
        )

    # Extend expiration
    new_expires_at = datetime.now(timezone.utc) + timedelta(minutes=LOCK_DURATION_MINUTES)

    result = db.client.table("resource_locks").update({"expires_at": new_expires_at.isoformat()}).eq("id", lock_id).execute()

    if not result.data:
        raise HTTPException(status_code=500, detail="Failed to refresh lock")

    updated_lock = result.data[0]
    return LockResponse(
        id=updated_lock["id"],
        resource_type=updated_lock["resource_type"],
        resource_id=updated_lock["resource_id"],
        user_id=updated_lock["user_id"],
        user_email=updated_lock.get("user_email"),
        locked_at=updated_lock["locked_at"],
        expires_at=updated_lock["expires_at"],
    )
