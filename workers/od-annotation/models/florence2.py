"""
Florence-2 Model Wrapper.

Microsoft's versatile vision foundation model.
https://huggingface.co/microsoft/Florence-2-large
"""

import torch
from typing import Any, Optional
from loguru import logger

from .base import BaseModel


class Florence2Model(BaseModel):
    """
    Florence-2: A Versatile Vision Foundation Model.

    Supports multiple tasks:
    - Object Detection (<OD>)
    - Phrase Grounding (<CAPTION_TO_PHRASE_GROUNDING>)
    - Dense Region Caption (<DENSE_REGION_CAPTION>)
    - Region to Category (<REGION_TO_CATEGORY>)

    Can work with or without text prompts.
    """

    # Task tokens
    TASK_OD = "<OD>"
    TASK_PHRASE_GROUNDING = "<CAPTION_TO_PHRASE_GROUNDING>"
    TASK_DENSE_REGION_CAPTION = "<DENSE_REGION_CAPTION>"
    TASK_REGION_TO_CATEGORY = "<REGION_TO_CATEGORY>"

    def __init__(
        self,
        model_id: str = "microsoft/Florence-2-large",
        device: str = "cuda",
        cache_dir: Optional[str] = None,
    ):
        super().__init__(device)
        self.model_id = model_id
        self.cache_dir = cache_dir
        self._processor = None

    def _load_model(self) -> Any:
        """Load Florence-2 model from HuggingFace."""
        logger.info(f"Loading Florence-2 model: {self.model_id}")

        try:
            from transformers import AutoProcessor, AutoModelForCausalLM

            # Load processor
            self._processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                cache_dir=self.cache_dir,
            )

            # Load model (disable SDPA for compatibility)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True,
                cache_dir=self.cache_dir,
                attn_implementation="eager",  # Disable Flash Attention for compatibility
            ).to(self.device)

            model.eval()

            logger.info("Florence-2 model loaded successfully")
            return model

        except Exception as e:
            logger.error(f"Failed to load Florence-2: {e}")
            raise

    @property
    def processor(self):
        """Get processor (lazy load)."""
        if self._processor is None:
            self._load_model()
        return self._processor

    def _run_inference(
        self,
        image,
        task_prompt: str,
        text_input: str = "",
    ) -> dict:
        """
        Run Florence-2 inference.

        Args:
            image: PIL Image
            task_prompt: Task token (e.g., <OD>)
            text_input: Optional text input for grounding tasks

        Returns:
            Parsed result dict
        """
        # Prepare prompt
        if text_input:
            prompt = f"{task_prompt}{text_input}"
        else:
            prompt = task_prompt

        # Process inputs
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        ).to(self.device)

        # Convert pixel_values to model dtype (float16 on CUDA)
        pixel_values = inputs["pixel_values"]
        if self.device == "cuda":
            pixel_values = pixel_values.half()

        # Generate (use greedy decoding with cache disabled for compatibility)
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=pixel_values,
                max_new_tokens=1024,
                num_beams=1,  # Greedy decoding for stability
                do_sample=False,
                use_cache=False,  # Disable KV cache for compatibility
            )

        # Decode
        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=False,
        )[0]

        # Post-process
        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image.width, image.height),
        )

        return parsed_answer

    def predict(
        self,
        image_url: str,
        text_prompt: str = "",
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
    ) -> list[dict]:
        """
        Run detection using Florence-2.

        If text_prompt is provided, uses phrase grounding.
        Otherwise, uses general object detection.

        Args:
            image_url: URL of the image
            text_prompt: Optional caption/phrase for grounding
            box_threshold: Not used (Florence-2 doesn't have confidence filtering)
            text_threshold: Not used

        Returns:
            List of predictions with bbox and label
        """
        # Download image
        image = self.download_image(image_url)
        width, height = image.size

        if text_prompt:
            # Use phrase grounding
            result = self._run_inference(
                image,
                task_prompt=self.TASK_PHRASE_GROUNDING,
                text_input=text_prompt,
            )
            key = self.TASK_PHRASE_GROUNDING
        else:
            # Use general object detection
            result = self._run_inference(
                image,
                task_prompt=self.TASK_OD,
            )
            key = self.TASK_OD

        # Parse results
        predictions = []

        if key in result:
            data = result[key]
            bboxes = data.get("bboxes", [])
            labels = data.get("labels", [])

            for bbox, label in zip(bboxes, labels):
                # Florence returns bboxes in [x1, y1, x2, y2] absolute pixels
                x1, y1, x2, y2 = bbox

                predictions.append({
                    "bbox": {
                        "x": x1 / width,
                        "y": y1 / height,
                        "width": (x2 - x1) / width,
                        "height": (y2 - y1) / height,
                    },
                    "label": label.strip(),
                    "confidence": 1.0,  # Florence doesn't provide confidence
                })

        logger.debug(f"Florence-2 found {len(predictions)} objects")

        return predictions

    def detect_all(self, image_url: str) -> list[dict]:
        """
        Detect all objects in image without text prompt.

        Args:
            image_url: URL of the image

        Returns:
            List of predictions
        """
        return self.predict(image_url=image_url, text_prompt="")

    def phrase_grounding(
        self,
        image_url: str,
        caption: str,
    ) -> list[dict]:
        """
        Ground phrases in a caption to image regions.

        Args:
            image_url: URL of the image
            caption: Caption containing objects to ground

        Returns:
            List of predictions with bbox and label
        """
        return self.predict(image_url=image_url, text_prompt=caption)

    def dense_region_caption(
        self,
        image_url: str,
    ) -> list[dict]:
        """
        Generate dense captions for all regions in image.

        Args:
            image_url: URL of the image

        Returns:
            List of regions with bbox and caption
        """
        image = self.download_image(image_url)
        width, height = image.size

        result = self._run_inference(
            image,
            task_prompt=self.TASK_DENSE_REGION_CAPTION,
        )

        predictions = []

        if self.TASK_DENSE_REGION_CAPTION in result:
            data = result[self.TASK_DENSE_REGION_CAPTION]
            bboxes = data.get("bboxes", [])
            labels = data.get("labels", [])

            for bbox, label in zip(bboxes, labels):
                x1, y1, x2, y2 = bbox

                predictions.append({
                    "bbox": {
                        "x": x1 / width,
                        "y": y1 / height,
                        "width": (x2 - x1) / width,
                        "height": (y2 - y1) / height,
                    },
                    "caption": label.strip(),
                    "confidence": 1.0,
                })

        return predictions
