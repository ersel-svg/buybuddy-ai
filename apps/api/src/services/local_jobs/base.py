"""
Base classes for local background job handlers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class JobProgress:
    """Progress update for a job."""

    progress: int  # 0-100
    current_step: str
    processed: int = 0
    total: int = 0
    errors: list[str] = field(default_factory=list)


class BaseJobHandler(ABC):
    """
    Abstract base class for local job handlers.

    Subclasses must:
    1. Set `job_type` class attribute
    2. Implement `execute()` method

    Example:
        @job_registry.register
        class MyHandler(BaseJobHandler):
            job_type = "local_my_operation"

            async def execute(self, job_id, config, update_progress):
                # Do work...
                update_progress(JobProgress(progress=50, current_step="Halfway"))
                # More work...
                return {"result": "success"}
    """

    # Subclasses must define this
    job_type: str = None

    @abstractmethod
    async def execute(
        self,
        job_id: str,
        config: dict,
        update_progress: Callable[[JobProgress], None],
    ) -> dict:
        """
        Execute the job.

        Args:
            job_id: Unique job identifier
            config: Job configuration from jobs.config column
            update_progress: Callback to update progress in database

        Returns:
            Result dict to store in jobs.result column

        Raises:
            Exception: Job will be marked as failed with error message
        """
        pass

    async def on_cancel(self, job_id: str, config: dict) -> None:
        """
        Called when job is cancelled.

        Override this method to perform cleanup (delete temp files, etc).
        Default implementation does nothing.
        """
        pass

    def validate_config(self, config: dict) -> Optional[str]:
        """
        Validate job config before execution.

        Args:
            config: Job configuration dict

        Returns:
            Error message string if invalid, None if valid
        """
        return None
