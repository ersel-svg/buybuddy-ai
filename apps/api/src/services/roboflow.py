"""
Roboflow API Client for importing datasets.

Provides functionality to:
- List workspaces and projects
- Download datasets in COCO/YOLO format
- Stream ZIP files for import
"""

import io
import os
import tempfile
import httpx
import logging
from typing import Any, Optional, Literal
from dataclasses import dataclass

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class RoboflowWorkspace:
    """Roboflow workspace info."""
    name: str
    url: str
    project_count: int


@dataclass
class RoboflowProject:
    """Roboflow project info."""
    id: str
    name: str
    type: str  # object-detection, classification, etc.
    created: str
    updated: str
    images: int
    classes: list[str]
    versions: int


@dataclass
class RoboflowVersion:
    """Roboflow dataset version info."""
    id: str
    name: str
    version: int
    images: dict  # {train: X, valid: Y, test: Z}
    classes: list[str]
    preprocessing: dict
    augmentation: dict
    created: str
    exports: list[str]  # Available export formats


class RoboflowService:
    """Client for Roboflow API."""

    BASE_URL = "https://api.roboflow.com"

    def __init__(self, api_key: Optional[str] = None):
        """Initialize with optional API key (falls back to config)."""
        self.api_key = api_key or settings.roboflow_api_key

    def _get_headers(self, api_key: Optional[str] = None) -> dict:
        """Get request headers."""
        return {
            "Content-Type": "application/json",
        }

    def _get_params(self, api_key: Optional[str] = None) -> dict:
        """Get query params with API key."""
        key = api_key or self.api_key
        return {"api_key": key}

    async def validate_api_key(self, api_key: str) -> dict[str, Any]:
        """
        Validate an API key and return workspace info.

        Returns:
            {
                "valid": bool,
                "workspaces": [{"name": str, "url": str, "projects": int}]
            }
        """
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                # Step 1: Get workspace slug from root endpoint
                resp = await client.get(
                    f"{self.BASE_URL}/",
                    params={"api_key": api_key},
                )

                if resp.status_code == 401:
                    return {"valid": False, "error": "Invalid API key", "workspaces": []}

                resp.raise_for_status()
                data = resp.json()

                # Root endpoint returns workspace as a string (slug)
                workspace_slug = data.get("workspace", "")
                if isinstance(workspace_slug, dict):
                    # Some API versions might return full object
                    workspace_slug = workspace_slug.get("url", workspace_slug.get("name", ""))

                if not workspace_slug:
                    return {"valid": False, "error": "No workspace found", "workspaces": []}

                # Step 2: Get full workspace details
                ws_resp = await client.get(
                    f"{self.BASE_URL}/{workspace_slug}",
                    params={"api_key": api_key},
                )
                ws_resp.raise_for_status()
                ws_data = ws_resp.json()

                workspace = ws_data.get("workspace", {})
                if not isinstance(workspace, dict):
                    workspace = {}

                raw_projects = workspace.get("projects", [])

                workspaces = [{
                    "name": workspace.get("name", workspace_slug),
                    "url": workspace.get("url", workspace_slug),
                    "projects": len(raw_projects) if isinstance(raw_projects, list) else 0,
                }]

                # Format projects to match expected structure
                formatted_projects = []
                if isinstance(raw_projects, list):
                    for proj in raw_projects:
                        # Handle both dict and string formats
                        if isinstance(proj, dict):
                            images_data = proj.get("images", 0)
                            if isinstance(images_data, dict):
                                images_count = images_data.get("total", 0)
                            else:
                                images_count = images_data if isinstance(images_data, int) else 0

                            formatted_projects.append({
                                "id": proj.get("id", proj.get("name", "")),
                                "name": proj.get("name", ""),
                                "type": proj.get("type", "object-detection"),
                                "images": images_count,
                                "versions": proj.get("versions", 0),
                            })
                        elif isinstance(proj, str):
                            # If project is just a string (project ID/name)
                            formatted_projects.append({
                                "id": proj,
                                "name": proj,
                                "type": "object-detection",
                                "images": 0,
                                "versions": 0,
                            })

                return {
                    "valid": True,
                    "workspaces": workspaces,
                    "projects": formatted_projects,
                }

        except httpx.HTTPStatusError as e:
            return {"valid": False, "error": f"API error: {e.response.status_code}", "workspaces": []}
        except Exception as e:
            return {"valid": False, "error": str(e), "workspaces": []}

    async def list_projects(self, api_key: str, workspace: str) -> list[dict]:
        """
        List projects in a workspace.

        Args:
            api_key: Roboflow API key
            workspace: Workspace URL/slug

        Returns:
            List of project info dicts
        """
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{self.BASE_URL}/{workspace}",
                params={"api_key": api_key},
            )
            resp.raise_for_status()
            data = resp.json()

            projects = []
            for proj in data.get("workspace", {}).get("projects", []):
                # Handle both dict and string formats
                if isinstance(proj, dict):
                    images_data = proj.get("images", 0)
                    if isinstance(images_data, dict):
                        images_count = images_data.get("total", 0)
                    else:
                        images_count = images_data if isinstance(images_data, int) else 0

                    classes_data = proj.get("classes", {})
                    if isinstance(classes_data, dict):
                        classes_list = list(classes_data.keys())
                    elif isinstance(classes_data, list):
                        classes_list = classes_data
                    else:
                        classes_list = []

                    projects.append({
                        "id": proj.get("id", proj.get("name", "")),
                        "name": proj.get("name", ""),
                        "type": proj.get("type", "object-detection"),
                        "created": proj.get("created", ""),
                        "updated": proj.get("updated", ""),
                        "images": images_count,
                        "classes": classes_list,
                        "versions": proj.get("versions", 0),
                    })
                elif isinstance(proj, str):
                    # If project is just a string (project ID/name)
                    projects.append({
                        "id": proj,
                        "name": proj,
                        "type": "object-detection",
                        "created": "",
                        "updated": "",
                        "images": 0,
                        "classes": [],
                        "versions": 0,
                    })

            return projects

    async def get_project(self, api_key: str, workspace: str, project: str) -> dict:
        """
        Get detailed project info.

        Args:
            api_key: Roboflow API key
            workspace: Workspace URL/slug
            project: Project URL/slug

        Returns:
            Project details including versions
        """
        async with httpx.AsyncClient(timeout=30) as client:
            # Handle full project ID (workspace/project) or just project name
            if "/" in project:
                # Full project ID already contains workspace
                url = f"{self.BASE_URL}/{project}"
            else:
                # Just project name, need to add workspace
                url = f"{self.BASE_URL}/{workspace}/{project}"

            resp = await client.get(url, params={"api_key": api_key})
            resp.raise_for_status()
            data = resp.json()

            project_data = data.get("project", {})
            if not isinstance(project_data, dict):
                project_data = {}

            # Get classes from PROJECT level (not version level!)
            # Format: {"class_name": annotation_count}
            project_classes = project_data.get("classes", {})
            if isinstance(project_classes, dict):
                project_classes_list = list(project_classes.keys())
            elif isinstance(project_classes, list):
                project_classes_list = project_classes
            else:
                project_classes_list = []

            # Parse versions - they are at ROOT level, not inside project!
            versions = []
            raw_versions = data.get("versions", [])  # Note: data, not project_data
            if isinstance(raw_versions, list):
                for ver in raw_versions:
                    if isinstance(ver, dict):
                        # Version-level classes are often empty, use project-level classes
                        classes_data = ver.get("classes", {})
                        if isinstance(classes_data, dict) and classes_data:
                            classes_list = list(classes_data.keys())
                        elif isinstance(classes_data, list) and classes_data:
                            classes_list = classes_data
                        else:
                            # Fallback to project-level classes
                            classes_list = project_classes_list

                        # Extract version number from id (e.g., "workspace/project/15" -> 15)
                        version_id = ver.get("id", "")
                        version_num = 0
                        if version_id and "/" in version_id:
                            try:
                                version_num = int(version_id.split("/")[-1])
                            except (ValueError, IndexError):
                                version_num = 0

                        images_data = ver.get("images", 0)
                        if isinstance(images_data, dict):
                            images_info = images_data
                        else:
                            images_info = {"total": images_data if isinstance(images_data, int) else 0}

                        versions.append({
                            "id": version_id,
                            "name": ver.get("name", f"Version {version_num}"),
                            "version": version_num,
                            "images": images_info,
                            "splits": ver.get("splits", {}),
                            "classes": classes_list,
                            "preprocessing": ver.get("preprocessing", {}),
                            "augmentation": ver.get("augmentation", {}),
                            "created": ver.get("created", ""),
                            "exports": ver.get("exports", []),
                        })

            return {
                "id": project_data.get("id", ""),
                "name": project_data.get("name", ""),
                "type": project_data.get("type", "object-detection"),
                "classes": project_data.get("classes", {}),
                "versions": versions,
            }

    async def list_versions(self, api_key: str, workspace: str, project: str) -> list[dict]:
        """
        List versions of a project.

        Args:
            api_key: Roboflow API key
            workspace: Workspace URL/slug
            project: Project URL/slug

        Returns:
            List of version info dicts
        """
        project_data = await self.get_project(api_key, workspace, project)
        return project_data.get("versions", [])

    async def get_version(self, api_key: str, workspace: str, project: str, version: int) -> dict:
        """
        Get specific version details.

        Args:
            api_key: Roboflow API key
            workspace: Workspace URL/slug
            project: Project URL/slug
            version: Version number

        Returns:
            Version details
        """
        async with httpx.AsyncClient(timeout=30) as client:
            # Handle full project ID (workspace/project) or just project name
            if "/" in project:
                url = f"{self.BASE_URL}/{project}/{version}"
            else:
                url = f"{self.BASE_URL}/{workspace}/{project}/{version}"

            resp = await client.get(url, params={"api_key": api_key})
            resp.raise_for_status()
            data = resp.json()

            version_data = data.get("version", {})
            if not isinstance(version_data, dict):
                version_data = {}

            project_data = data.get("project", {})
            if not isinstance(project_data, dict):
                project_data = {}

            # Classes come from PROJECT level, not version level
            # Format: {"class_name": annotation_count}
            classes = project_data.get("classes", {})
            if not isinstance(classes, dict):
                # Fallback to version classes if project classes not available
                classes = version_data.get("classes", {})

            # Colors also come from project level
            colors = project_data.get("colors", {})

            return {
                "id": version_data.get("id", ""),
                "name": version_data.get("name", f"Version {version}"),
                "version": version_data.get("version", version),
                "images": version_data.get("images", {}),
                "splits": version_data.get("splits", {}),
                "classes": classes,
                "colors": colors,
                "preprocessing": version_data.get("preprocessing", {}),
                "augmentation": version_data.get("augmentation", {}),
                "created": version_data.get("created", ""),
                "exports": version_data.get("exports", []),
            }

    async def get_download_url(
        self,
        api_key: str,
        workspace: str,
        project: str,
        version: int,
        format: Literal["coco", "yolov8", "yolov5pytorch", "voc"] = "coco",
    ) -> str:
        """
        Get the download URL for a dataset version.

        Args:
            api_key: Roboflow API key
            workspace: Workspace URL/slug
            project: Project URL/slug
            version: Version number
            format: Export format (coco, yolov8, yolov5pytorch, voc)

        Returns:
            Download URL for the ZIP file
        """
        # Roboflow download URL format
        # https://api.roboflow.com/{workspace}/{project}/{version}/{format}?api_key=xxx
        if "/" in project:
            return f"{self.BASE_URL}/{project}/{version}/{format}?api_key={api_key}"
        else:
            return f"{self.BASE_URL}/{workspace}/{project}/{version}/{format}?api_key={api_key}"

    async def download_dataset(
        self,
        api_key: str,
        workspace: str,
        project: str,
        version: int,
        format: Literal["coco", "yolov8", "yolov5pytorch", "voc"] = "coco",
    ) -> bytes:
        """
        Download a dataset version as ZIP.

        Args:
            api_key: Roboflow API key
            workspace: Workspace URL/slug
            project: Project URL/slug
            version: Version number
            format: Export format (coco, yolov8, yolov5pytorch, voc)

        Returns:
            ZIP file bytes
        """
        url = await self.get_download_url(api_key, workspace, project, version, format)

        # Use longer timeout for large datasets (30 minutes)
        async with httpx.AsyncClient(timeout=1800, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()

            # Check if it's a redirect/link response
            if resp.headers.get("content-type", "").startswith("application/json"):
                data = resp.json()
                # Roboflow returns a link in the response
                if "export" in data and "link" in data["export"]:
                    download_link = data["export"]["link"]
                    resp = await client.get(download_link)
                    resp.raise_for_status()

            return resp.content

    async def stream_download_dataset(
        self,
        api_key: str,
        workspace: str,
        project: str,
        version: int,
        format: Literal["coco", "yolov8", "yolov5pytorch", "voc"] = "coco",
        max_retries: int = 3,
    ):
        """
        Stream download a dataset version with retry support.

        Yields chunks of the ZIP file for memory-efficient downloading.
        Includes automatic retry on timeout/network errors.

        Args:
            api_key: Roboflow API key
            workspace: Workspace URL/slug
            project: Project URL/slug
            version: Version number
            format: Export format
            max_retries: Maximum number of retry attempts (default: 3)

        Yields:
            Bytes chunks of the ZIP file
        """
        url = await self.get_download_url(api_key, workspace, project, version, format)

        # Use separate timeouts for different operations:
        # - connect: 60 seconds to establish connection (increased for slow networks)
        # - read: 10 minutes per chunk (handles very slow networks and large files)
        # - write: 60 seconds
        # - pool: 60 seconds
        timeout = httpx.Timeout(
            connect=60.0,
            read=600.0,  # 10 minutes read timeout per chunk
            write=60.0,
            pool=60.0,
        )

        last_error = None

        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                    # First, get the export link
                    resp = await client.get(url)
                    resp.raise_for_status()

                    download_link = url
                    if resp.headers.get("content-type", "").startswith("application/json"):
                        data = resp.json()
                        if "export" in data and "link" in data["export"]:
                            download_link = data["export"]["link"]

                    # Stream the actual download with 1MB chunks for efficiency
                    async with client.stream("GET", download_link) as response:
                        response.raise_for_status()
                        async for chunk in response.aiter_bytes(chunk_size=1048576):  # 1MB chunks
                            yield chunk

                    # If we get here, download completed successfully
                    return

            except (httpx.TimeoutException, httpx.NetworkError) as e:
                last_error = e
                if attempt < max_retries - 1:
                    # Wait before retry (exponential backoff: 5s, 10s, 20s)
                    import asyncio
                    wait_time = 5 * (2 ** attempt)
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise
            except httpx.HTTPStatusError:
                # Don't retry on HTTP errors (4xx, 5xx)
                raise

        # If we exhausted all retries
        if last_error:
            raise last_error

    async def download_dataset_to_file(
        self,
        api_key: str,
        workspace: str,
        project: str,
        version: int,
        format: Literal["coco", "yolov8", "yolov5pytorch", "voc"] = "coco",
        max_retries: int = 3,
        progress_callback: Optional[callable] = None,
    ) -> tuple[str, int]:
        """
        Download dataset directly to a temporary file on disk.

        This is memory-efficient for large datasets. The file is automatically
        cleaned up if download fails or is incomplete.

        Args:
            api_key: Roboflow API key
            workspace: Workspace URL/slug
            project: Project URL/slug
            version: Version number
            format: Export format
            max_retries: Maximum number of retry attempts
            progress_callback: Optional callback(downloaded_bytes, total_bytes) for progress updates

        Returns:
            Tuple of (temp_file_path, total_bytes_downloaded)

        Raises:
            httpx.TimeoutException: If download times out after all retries
            httpx.NetworkError: If network error occurs after all retries
            httpx.HTTPStatusError: If HTTP error occurs (4xx, 5xx)
        """
        url = await self.get_download_url(api_key, workspace, project, version, format)

        timeout = httpx.Timeout(
            connect=60.0,
            read=600.0,  # 10 minutes read timeout per chunk
            write=60.0,
            pool=60.0,
        )

        # Create temp file - we'll clean it up on failure
        temp_file = tempfile.NamedTemporaryFile(
            mode='wb',
            suffix='.zip',
            delete=False,
            prefix='roboflow_download_'
        )
        temp_path = temp_file.name
        download_complete = False
        total_downloaded = 0

        try:
            last_error = None

            for attempt in range(max_retries):
                try:
                    # Reset file for retry
                    temp_file.seek(0)
                    temp_file.truncate()
                    total_downloaded = 0

                    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                        # First, get the export link
                        resp = await client.get(url)
                        resp.raise_for_status()

                        download_link = url
                        content_length = None

                        if resp.headers.get("content-type", "").startswith("application/json"):
                            data = resp.json()
                            if "export" in data and "link" in data["export"]:
                                download_link = data["export"]["link"]

                        # Stream the actual download directly to file
                        async with client.stream("GET", download_link) as response:
                            response.raise_for_status()
                            content_length = response.headers.get("content-length")
                            if content_length:
                                content_length = int(content_length)

                            async for chunk in response.aiter_bytes(chunk_size=1048576):  # 1MB chunks
                                temp_file.write(chunk)
                                total_downloaded += len(chunk)

                                if progress_callback:
                                    try:
                                        progress_callback(total_downloaded, content_length)
                                    except Exception:
                                        pass  # Don't fail on callback errors

                        # Verify download completed (if we have content-length)
                        if content_length and total_downloaded < content_length:
                            raise httpx.NetworkError(
                                f"Incomplete download: got {total_downloaded} bytes, expected {content_length}"
                            )

                        # Download completed successfully
                        download_complete = True
                        temp_file.flush()
                        temp_file.close()

                        logger.info(f"Roboflow download complete: {total_downloaded} bytes to {temp_path}")
                        return temp_path, total_downloaded

                except (httpx.TimeoutException, httpx.NetworkError) as e:
                    last_error = e
                    logger.warning(
                        f"Roboflow download attempt {attempt + 1}/{max_retries} failed: {e}"
                    )
                    if attempt < max_retries - 1:
                        import asyncio
                        wait_time = 5 * (2 ** attempt)
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        raise

                except httpx.HTTPStatusError as e:
                    logger.error(f"Roboflow HTTP error: {e.response.status_code}")
                    raise

            if last_error:
                raise last_error

        except Exception:
            # Clean up temp file on any failure
            download_complete = False
            raise

        finally:
            # Always clean up if download was not complete
            if not download_complete:
                try:
                    temp_file.close()
                except Exception:
                    pass
                try:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                        logger.info(f"Cleaned up incomplete download: {temp_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temp file {temp_path}: {e}")

        # Should not reach here, but just in case
        return temp_path, total_downloaded

    async def preview_import(
        self,
        api_key: str,
        workspace: str,
        project: str,
        version: int,
    ) -> dict:
        """
        Preview what will be imported from a Roboflow dataset.

        Returns class info, image counts, etc. without downloading.

        Args:
            api_key: Roboflow API key
            workspace: Workspace URL/slug
            project: Project URL/slug
            version: Version number

        Returns:
            Preview information
        """
        version_data = await self.get_version(api_key, workspace, project, version)

        # Get class info - handle different formats
        classes = version_data.get("classes", {})
        class_list = []

        if isinstance(classes, dict):
            for class_name, class_info in classes.items():
                if isinstance(class_info, int):
                    count = class_info
                elif isinstance(class_info, dict):
                    count = class_info.get("count", 0)
                else:
                    count = 0
                class_list.append({
                    "name": class_name,
                    "count": count,
                })
        elif isinstance(classes, list):
            # If classes is a list of strings
            for cls in classes:
                if isinstance(cls, str):
                    class_list.append({"name": cls, "count": 0})
                elif isinstance(cls, dict):
                    class_list.append({
                        "name": cls.get("name", ""),
                        "count": cls.get("count", 0),
                    })

        # Get image counts
        splits = version_data.get("splits", {})
        images = version_data.get("images", {})

        total_images = 0
        if isinstance(images, dict):
            total_images = sum(v for v in images.values() if isinstance(v, (int, float)))
        elif isinstance(images, int):
            total_images = images

        return {
            "workspace": workspace,
            "project": project,
            "version": version,
            "version_name": version_data.get("name", f"Version {version}"),
            "total_images": total_images,
            "splits": splits if isinstance(splits, dict) else {},
            "classes": class_list,
            "class_count": len(class_list),
            "preprocessing": version_data.get("preprocessing", {}),
            "augmentation": version_data.get("augmentation", {}),
        }


# Singleton instance (for server-side default key usage)
roboflow_service = RoboflowService()
