"""Store Map API router - proxies requests to BuyBuddy Legacy API."""

from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from pydantic import BaseModel

from services.buybuddy import BuybuddyService, buybuddy_service
from auth.dependencies import get_current_user

# Router with authentication required for all endpoints
router = APIRouter(dependencies=[Depends(get_current_user)])


# ===========================================
# Schemas
# ===========================================


class CreateMapRequest(BaseModel):
    """Request to create a new store map."""

    store_id: int
    name: str
    base_ratio: float = 1.0
    grid: float = 50.0


class UpdateMapRequest(BaseModel):
    """Request to update an existing map."""

    name: Optional[str] = None
    base_ratio: Optional[float] = None
    grid: Optional[float] = None


class CreateAreaRequest(BaseModel):
    """Request to create an area."""

    name: str
    floor_id: int


class UpdateAreaRequest(BaseModel):
    """Request to update an area."""

    id: int
    name: str


class CreateCoordinateRequest(BaseModel):
    """Request to create a coordinate."""

    area_id: int
    x: float
    y: float
    z: float = 0.0
    r: float = 0.0
    circle: bool = False


class UpdateCoordinateRequest(BaseModel):
    """Request to update a coordinate."""

    id: int
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None
    r: Optional[float] = None


# ===========================================
# Dependencies
# ===========================================


def get_buybuddy() -> BuybuddyService:
    """Get BuyBuddy service instance."""
    return buybuddy_service


# ===========================================
# Map Endpoints
# ===========================================


@router.post("")
async def create_map(
    request: CreateMapRequest,
    buybuddy: BuybuddyService = Depends(get_buybuddy),
):
    """Create a new store map."""
    if not buybuddy.is_configured():
        raise HTTPException(
            status_code=400,
            detail="BuyBuddy API credentials not configured.",
        )

    try:
        return await buybuddy.create_map(
            store_id=request.store_id,
            name=request.name,
            base_ratio=request.base_ratio,
            grid=request.grid,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{map_id}")
async def update_map(
    map_id: int,
    request: UpdateMapRequest,
    buybuddy: BuybuddyService = Depends(get_buybuddy),
):
    """Update an existing map."""
    if not buybuddy.is_configured():
        raise HTTPException(
            status_code=400,
            detail="BuyBuddy API credentials not configured.",
        )

    try:
        return await buybuddy.update_map(
            map_id=map_id,
            name=request.name,
            base_ratio=request.base_ratio,
            grid=request.grid,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{map_id}")
async def get_map(
    map_id: int,
    buybuddy: BuybuddyService = Depends(get_buybuddy),
):
    """Retrieve a single map by ID."""
    if not buybuddy.is_configured():
        raise HTTPException(
            status_code=400,
            detail="BuyBuddy API credentials not configured.",
        )

    try:
        return await buybuddy.get_map(map_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("")
async def list_maps(
    buybuddy: BuybuddyService = Depends(get_buybuddy),
):
    """List all maps for the authenticated merchant."""
    if not buybuddy.is_configured():
        raise HTTPException(
            status_code=400,
            detail="BuyBuddy API credentials not configured.",
        )

    try:
        result = await buybuddy.list_maps()
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{map_id}")
async def delete_map(
    map_id: int,
    buybuddy: BuybuddyService = Depends(get_buybuddy),
):
    """Delete a map by ID."""
    if not buybuddy.is_configured():
        raise HTTPException(
            status_code=400,
            detail="BuyBuddy API credentials not configured.",
        )

    try:
        await buybuddy.delete_map(map_id)
        return {"message": "Map deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===========================================
# Store-based Map Endpoints
# ===========================================


@router.get("/store/{store_id}/map")
async def get_map_by_store(
    store_id: int,
    buybuddy: BuybuddyService = Depends(get_buybuddy),
):
    """Retrieve a map filtered by store ID."""
    if not buybuddy.is_configured():
        raise HTTPException(
            status_code=400,
            detail="BuyBuddy API credentials not configured.",
        )

    try:
        return await buybuddy.get_map_by_store(store_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===========================================
# Floor Endpoints
# ===========================================


@router.post("/{map_id}/floor")
async def create_floor(
    map_id: int,
    floor: int = Form(...),
    file: UploadFile = File(...),
    buybuddy: BuybuddyService = Depends(get_buybuddy),
):
    """Create a floor for a map with floor plan image upload."""
    if not buybuddy.is_configured():
        raise HTTPException(
            status_code=400,
            detail="BuyBuddy API credentials not configured.",
        )

    try:
        # Read file content
        file_content = await file.read()
        filename = file.filename or "floor_plan.jpg"

        return await buybuddy.create_floor(
            map_id=map_id,
            floor=floor,
            file_content=file_content,
            filename=filename,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{map_id}/floor")
async def list_floors(
    map_id: int,
    buybuddy: BuybuddyService = Depends(get_buybuddy),
):
    """List all floors for a given map."""
    if not buybuddy.is_configured():
        raise HTTPException(
            status_code=400,
            detail="BuyBuddy API credentials not configured.",
        )

    try:
        return await buybuddy.list_floors(map_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===========================================
# Area Endpoints
# ===========================================


@router.post("/area")
async def create_area(
    request: CreateAreaRequest,
    buybuddy: BuybuddyService = Depends(get_buybuddy),
):
    """Create an area within a floor."""
    if not buybuddy.is_configured():
        raise HTTPException(
            status_code=400,
            detail="BuyBuddy API credentials not configured.",
        )

    try:
        return await buybuddy.create_area(
            name=request.name,
            floor_id=request.floor_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/area")
async def update_area(
    request: UpdateAreaRequest,
    buybuddy: BuybuddyService = Depends(get_buybuddy),
):
    """Update an area (name)."""
    if not buybuddy.is_configured():
        raise HTTPException(
            status_code=400,
            detail="BuyBuddy API credentials not configured.",
        )

    try:
        return await buybuddy.update_area(
            area_id=request.id,
            name=request.name,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/area/{area_id}")
async def delete_area(
    area_id: int,
    buybuddy: BuybuddyService = Depends(get_buybuddy),
):
    """Delete an area and all its associated coordinates."""
    if not buybuddy.is_configured():
        raise HTTPException(
            status_code=400,
            detail="BuyBuddy API credentials not configured.",
        )

    try:
        await buybuddy.delete_area(area_id)
        return {"message": "Area deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===========================================
# Coordinate Endpoints
# ===========================================


@router.post("/area/coordinate")
async def create_coordinate(
    request: CreateCoordinateRequest,
    buybuddy: BuybuddyService = Depends(get_buybuddy),
):
    """Create a coordinate point for an area."""
    if not buybuddy.is_configured():
        raise HTTPException(
            status_code=400,
            detail="BuyBuddy API credentials not configured.",
        )

    try:
        return await buybuddy.create_coordinate(
            area_id=request.area_id,
            x=request.x,
            y=request.y,
            z=request.z,
            r=request.r,
            circle=request.circle,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/area/coordinate")
async def update_coordinate(
    request: UpdateCoordinateRequest,
    buybuddy: BuybuddyService = Depends(get_buybuddy),
):
    """Update a coordinate."""
    if not buybuddy.is_configured():
        raise HTTPException(
            status_code=400,
            detail="BuyBuddy API credentials not configured.",
        )

    try:
        return await buybuddy.update_coordinate(
            coord_id=request.id,
            x=request.x,
            y=request.y,
            z=request.z,
            r=request.r,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/area/coordinate")
async def list_coordinates(
    map_id: Optional[int] = None,
    store_id: Optional[int] = None,
    area_id: Optional[int] = None,
    floor_id: Optional[int] = None,
    buybuddy: BuybuddyService = Depends(get_buybuddy),
):
    """List coordinates with full hierarchy."""
    if not buybuddy.is_configured():
        raise HTTPException(
            status_code=400,
            detail="BuyBuddy API credentials not configured.",
        )

    try:
        return await buybuddy.list_coordinates(
            map_id=map_id,
            store_id=store_id,
            area_id=area_id,
            floor_id=floor_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/area/coordinate/{coord_id}")
async def delete_coordinate(
    coord_id: int,
    buybuddy: BuybuddyService = Depends(get_buybuddy),
):
    """Delete a coordinate by ID."""
    if not buybuddy.is_configured():
        raise HTTPException(
            status_code=400,
            detail="BuyBuddy API credentials not configured.",
        )

    try:
        await buybuddy.delete_coordinate(coord_id)
        return {"message": "Coordinate deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
