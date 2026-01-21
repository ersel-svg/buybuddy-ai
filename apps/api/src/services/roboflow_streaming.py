"""
Roboflow Streaming Import Service.

Imports datasets from Roboflow by streaming images directly to Supabase,
without downloading the entire ZIP file first.

This is much faster and more memory-efficient than the ZIP-based approach:
- No large ZIP download required
- Each image streams directly: Roboflow URL -> Supabase Storage
- Parallel processing with controlled concurrency
- Fine-grained checkpoint/resume support
"""

import asyncio
import logging
import uuid
import httpx
from dataclasses import dataclass, field
from typing import Optional, Callable, Any

from services.roboflow import roboflow_service
from services.supabase import supabase_service

logger = logging.getLogger(__name__)


@dataclass
class StreamingImportResult:
    """Result of a streaming import operation."""
    success: bool
    images_imported: int = 0
    annotations_imported: int = 0
    images_skipped: int = 0
    images_failed: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class StreamingCheckpoint:
    """Checkpoint data for resumable streaming imports."""
    job_id: str
    total_images: int = 0
    processed_ids: set = field(default_factory=set)
    failed_ids: dict = field(default_factory=dict)  # {id: error_message}
    storage_map: dict = field(default_factory=dict)  # {roboflow_id: storage_filename}
    db_image_ids: dict = field(default_factory=dict)  # {roboflow_id: db_image_id}

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "total_images": self.total_images,
            "processed_count": len(self.processed_ids),
            "failed_count": len(self.failed_ids),
        }


class RoboflowStreamingImporter:
    """
    Imports Roboflow datasets by streaming images directly to Supabase.

    Instead of downloading a large ZIP file, this:
    1. Lists all image IDs from Roboflow project
    2. Fetches each image's URL and annotations
    3. Streams image data directly to Supabase Storage
    4. Inserts database records

    Benefits:
    - No disk space required
    - Faster (parallel streaming)
    - Better progress tracking (per-image)
    - Resume from any point
    """

    def __init__(
        self,
        api_key: str,
        workspace: str,
        project: str,
        dataset_id: str,
        class_mapping: list[dict],
    ):
        self.api_key = api_key
        self.workspace = workspace
        self.project = project
        self.dataset_id = dataset_id
        self.class_mapping = {m["source_name"]: m for m in class_mapping}

        # Build class name to ID map
        self.class_id_map: dict[str, str] = {}
        self._http_client: Optional[httpx.AsyncClient] = None

    async def _ensure_http_client(self):
        """Ensure HTTP client is initialized."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(connect=30.0, read=120.0, write=30.0, pool=30.0),
                follow_redirects=True,
            )
        return self._http_client

    async def _close_http_client(self):
        """Close HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    async def _setup_class_mapping(self):
        """
        Setup class ID mapping from class_mapping config.
        Creates new classes if needed.
        """
        for source_name, mapping in self.class_mapping.items():
            if mapping.get("skip"):
                continue

            if mapping.get("target_class_id"):
                self.class_id_map[source_name] = mapping["target_class_id"]
            elif mapping.get("create_new"):
                # Create new class using direct table access
                try:
                    result = supabase_service.client.table("od_classes").insert({
                        "name": source_name,
                        "color": mapping.get("color", self._generate_color(source_name)),
                    }).execute()
                    if result.data:
                        self.class_id_map[source_name] = result.data[0]["id"]
                        logger.info(f"Created new class: {source_name} -> {result.data[0]['id']}")
                except Exception as e:
                    logger.error(f"Failed to create class {source_name}: {e}")

    def _generate_color(self, class_name: str) -> str:
        """Generate a consistent color for a class name."""
        import hashlib
        hash_val = int(hashlib.md5(class_name.encode()).hexdigest()[:6], 16)
        r = (hash_val >> 16) & 0xFF
        g = (hash_val >> 8) & 0xFF
        b = hash_val & 0xFF
        return f"#{r:02x}{g:02x}{b:02x}"

    async def _stream_image_to_storage(
        self,
        image_url: str,
        filename: str,
        max_retries: int = 3,
    ) -> tuple[bool, int, str]:
        """
        Stream an image from URL directly to Supabase Storage.

        Args:
            image_url: Source URL (Roboflow)
            filename: Target filename in storage
            max_retries: Number of retry attempts

        Returns:
            Tuple of (success, file_size, error_or_content_type)
        """
        client = await self._ensure_http_client()

        for attempt in range(max_retries):
            try:
                # Download image to memory
                print(f"[DEBUG] [STREAM] Downloading from Roboflow...")
                response = await client.get(image_url)
                response.raise_for_status()

                content = response.content
                content_type = response.headers.get("content-type", "image/jpeg")
                file_size = len(content)
                print(f"[DEBUG] [STREAM] Downloaded {file_size} bytes, uploading to storage...")

                # Upload to Supabase Storage
                try:
                    supabase_service.client.storage.from_("od-images").upload(
                        filename,
                        content,
                        {"content-type": content_type}
                    )
                    return (True, file_size, content_type)
                except Exception as e:
                    error_msg = str(e).lower()
                    # 409 = already exists, treat as success
                    if "409" in str(e) or "duplicate" in error_msg or "already exists" in error_msg:
                        return (True, file_size, content_type)
                    raise

            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
                return (False, 0, str(e))

        return (False, 0, "Max retries exceeded")

    async def _process_single_image(
        self,
        image_data: dict,
        checkpoint: StreamingCheckpoint,
        semaphore: asyncio.Semaphore,
    ) -> tuple[bool, int, str]:
        """
        Process a single image: upload to storage and insert to DB.

        Args:
            image_data: Full image data from Roboflow API
            checkpoint: Current checkpoint state
            semaphore: Concurrency limiter

        Returns:
            Tuple of (success, annotations_count, error_message)
        """
        async with semaphore:
            roboflow_id = image_data.get("id", "")

            # Skip if already processed
            if roboflow_id in checkpoint.processed_ids:
                return (True, 0, "already_processed")

            try:
                # Extract data
                urls = image_data.get("urls", {})
                original_url = urls.get("original", "")
                annotation_data = image_data.get("annotation", {})
                original_name = image_data.get("name", f"{roboflow_id}.jpg")
                split = image_data.get("split", "train")

                if not original_url:
                    return (False, 0, "No image URL")

                # Generate storage filename
                storage_filename = f"{uuid.uuid4()}.jpg"

                # 1. Upload image to storage
                print(f"[DEBUG] [UPLOAD] Starting upload for {roboflow_id[:8]}...")
                success, file_size, content_type_or_error = await self._stream_image_to_storage(
                    original_url, storage_filename
                )
                print(f"[DEBUG] [UPLOAD] Completed {roboflow_id[:8]}: success={success}, size={file_size}")

                if not success:
                    return (False, 0, f"Upload failed: {content_type_or_error}")

                # 2. Get image dimensions
                width = annotation_data.get("width", 0)
                height = annotation_data.get("height", 0)

                # 3. Build storage URL
                storage_url = f"{supabase_service.client.storage.from_('od-images').get_public_url(storage_filename)}"

                # 4. Insert image to database
                image_record = {
                    "filename": storage_filename,
                    "original_filename": original_name,
                    "image_url": storage_url,
                    "width": width,
                    "height": height,
                    "file_size_bytes": file_size,
                    "mime_type": "image/jpeg",
                    "source": "import",  # Valid values: "url", "import"
                    "folder": split,
                    "tags": [f"roboflow:{self.project}", f"split:{split}"],
                    "metadata": {
                        "roboflow_id": roboflow_id,
                        "roboflow_project": self.project,
                        "import_method": "streaming",
                    }
                }

                db_result = supabase_service.client.table("od_images").insert(image_record).execute()
                if not db_result.data:
                    return (False, 0, "Failed to insert image to database")
                db_image_id = db_result.data[0]["id"]

                # 5. Link image to dataset
                supabase_service.client.table("od_dataset_images").insert({
                    "dataset_id": self.dataset_id,
                    "image_id": db_image_id,
                }).execute()

                # 6. Insert annotations
                annotations_count = 0
                boxes = annotation_data.get("boxes", [])

                for box in boxes:
                    label = box.get("label", "")

                    # Check class mapping
                    mapping = self.class_mapping.get(label, {})
                    if mapping.get("skip"):
                        continue

                    class_id = self.class_id_map.get(label)
                    if not class_id:
                        # Skip unmapped classes
                        continue

                    # Parse bbox (Roboflow uses center-x, center-y, width, height)
                    try:
                        cx = float(box.get("x", 0))
                        cy = float(box.get("y", 0))
                        bw = float(box.get("width", 0))
                        bh = float(box.get("height", 0))

                        # Convert to top-left corner format (normalized)
                        if width > 0 and height > 0:
                            bbox_x = (cx - bw / 2) / width
                            bbox_y = (cy - bh / 2) / height
                            bbox_w = bw / width
                            bbox_h = bh / height
                        else:
                            # Fallback if dimensions unknown
                            bbox_x = cx - bw / 2
                            bbox_y = cy - bh / 2
                            bbox_w = bw
                            bbox_h = bh
                    except (ValueError, TypeError):
                        continue

                    annotation_record = {
                        "dataset_id": self.dataset_id,
                        "image_id": db_image_id,
                        "class_id": class_id,
                        "bbox_x": bbox_x,
                        "bbox_y": bbox_y,
                        "bbox_width": bbox_w,
                        "bbox_height": bbox_h,
                        "is_ai_generated": False,
                    }

                    supabase_service.client.table("od_annotations").insert(annotation_record).execute()
                    annotations_count += 1

                # Update checkpoint
                checkpoint.processed_ids.add(roboflow_id)
                checkpoint.storage_map[roboflow_id] = storage_filename
                checkpoint.db_image_ids[roboflow_id] = db_image_id

                return (True, annotations_count, "")

            except Exception as e:
                checkpoint.failed_ids[roboflow_id] = str(e)
                return (False, 0, str(e))

    async def import_dataset(
        self,
        concurrency: int = 10,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        checkpoint: Optional[StreamingCheckpoint] = None,
    ) -> StreamingImportResult:
        """
        Import the dataset using streaming.

        Args:
            concurrency: Max parallel image processing
            progress_callback: Callback(processed, total, message)
            checkpoint: Optional checkpoint to resume from

        Returns:
            StreamingImportResult
        """
        result = StreamingImportResult(success=False)

        try:
            # Setup
            await self._setup_class_mapping()

            if progress_callback:
                progress_callback(0, 0, "Listing images from Roboflow...")

            # 1. List all images
            logger.info(f"Listing images from {self.workspace}/{self.project}")
            image_list = await roboflow_service.list_project_images(
                self.api_key, self.workspace, self.project
            )

            total_images = len(image_list)
            logger.info(f"Found {total_images} images to import")

            if not image_list:
                result.success = True
                return result

            # Initialize or use checkpoint
            if checkpoint is None:
                checkpoint = StreamingCheckpoint(job_id="", total_images=total_images)
            checkpoint.total_images = total_images

            if progress_callback:
                progress_callback(0, total_images, f"Found {total_images} images. Starting import...")

            # 2. Process images with concurrency control
            print(f"[DEBUG] Creating semaphore with concurrency={concurrency}")
            semaphore = asyncio.Semaphore(concurrency)
            processed = len(checkpoint.processed_ids)

            # Filter out already processed
            print(f"[DEBUG] Filtering pending images from {len(image_list)} total")
            pending_images = [
                img for img in image_list
                if img.get("id") not in checkpoint.processed_ids
            ]

            print(f"[DEBUG] Processing {len(pending_images)} pending images ({processed} already done)")
            logger.info(f"Processing {len(pending_images)} pending images ({processed} already done)")

            # Process in batches for better progress reporting
            batch_size = 50
            print(f"[DEBUG] Starting batch loop with batch_size={batch_size}, total batches={len(pending_images) // batch_size + 1}")

            for batch_start in range(0, len(pending_images), batch_size):
                batch = pending_images[batch_start:batch_start + batch_size]
                batch_image_ids = [img.get("id") for img in batch]
                batch_num = batch_start // batch_size + 1
                print(f"[DEBUG] [BATCH {batch_num}] Starting with {len(batch_image_ids)} images")
                logger.info(f"[BATCH {batch_num}] Starting with {len(batch_image_ids)} images")

                # Fetch full details for this batch
                if progress_callback:
                    progress_callback(
                        processed, total_images,
                        f"Fetching details for batch {batch_num}..."
                    )

                # Get full image data with annotations using asyncio.gather
                # (more reliable than async generators in background tasks)
                batch_results = []
                print(f"[DEBUG] [BATCH {batch_num}] Calling fetch_images_with_details...")
                logger.info(f"[BATCH {batch_num}] Fetching image details from Roboflow...")

                fetch_results = await roboflow_service.fetch_images_with_details(
                    self.api_key, self.workspace, self.project, batch_image_ids, concurrency=concurrency
                )

                print(f"[DEBUG] [BATCH {batch_num}] fetch_images_with_details returned {len(fetch_results)} results")
                logger.info(f"[BATCH {batch_num}] Fetched {len(fetch_results)}/{len(batch_image_ids)} images")

                print(f"[DEBUG] [BATCH {batch_num}] Processing fetch results...")
                for img_id, img_data, error in fetch_results:
                    if error:
                        checkpoint.failed_ids[img_id] = error
                        result.images_failed += 1
                        result.errors.append(f"Failed to fetch {img_id}: {error}")
                    elif img_data:
                        batch_results.append(img_data)

                # Process the batch
                print(f"[DEBUG] [BATCH {batch_num}] Creating {len(batch_results)} processing tasks...")
                tasks = [
                    self._process_single_image(img_data, checkpoint, semaphore)
                    for img_data in batch_results
                ]

                print(f"[DEBUG] [BATCH {batch_num}] Starting asyncio.as_completed loop...")
                completed_count = 0
                for coro in asyncio.as_completed(tasks):
                    success, ann_count, error = await coro
                    processed += 1
                    completed_count += 1

                    if completed_count <= 3:
                        print(f"[DEBUG] [BATCH {batch_num}] Image {completed_count} completed: success={success}, ann_count={ann_count}, error={error[:50] if error else 'None'}")

                    if success:
                        if error != "already_processed":
                            result.images_imported += 1
                            result.annotations_imported += ann_count
                        else:
                            result.images_skipped += 1
                    else:
                        result.images_failed += 1
                        if error:
                            result.errors.append(error)

                    if progress_callback and processed % 10 == 0:
                        print(f"[DEBUG] Calling progress_callback: processed={processed}, total={total_images}")
                        progress_callback(
                            processed, total_images,
                            f"Imported {result.images_imported} images, {result.annotations_imported} annotations"
                        )

                # Small delay between batches
                print(f"[DEBUG] [BATCH {batch_num}] COMPLETE: {result.images_imported} imported, {result.images_failed} failed")
                await asyncio.sleep(0.5)

            result.success = result.images_failed == 0 or result.images_imported > 0

            logger.info(
                f"Import complete: {result.images_imported} images, "
                f"{result.annotations_imported} annotations, "
                f"{result.images_failed} failed"
            )

            return result

        except Exception as e:
            import traceback
            print(f"[DEBUG] EXCEPTION in streaming import: {e}")
            print(f"[DEBUG] Traceback: {traceback.format_exc()}")
            logger.exception(f"Streaming import failed: {e}")
            result.errors.append(str(e))
            return result

        finally:
            await self._close_http_client()


# Convenience function for direct use
async def streaming_import_from_roboflow(
    api_key: str,
    workspace: str,
    project: str,
    dataset_id: str,
    class_mapping: list[dict],
    concurrency: int = 10,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> StreamingImportResult:
    """
    Import a Roboflow dataset using streaming (no ZIP download).

    Args:
        api_key: Roboflow API key
        workspace: Workspace slug
        project: Project slug
        dataset_id: Target dataset ID
        class_mapping: List of class mapping configs
        concurrency: Max parallel processing
        progress_callback: Optional progress callback

    Returns:
        StreamingImportResult
    """
    importer = RoboflowStreamingImporter(
        api_key=api_key,
        workspace=workspace,
        project=project,
        dataset_id=dataset_id,
        class_mapping=class_mapping,
    )

    return await importer.import_dataset(
        concurrency=concurrency,
        progress_callback=progress_callback,
    )
