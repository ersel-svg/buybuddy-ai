"""
Registry for local job handlers.

Handlers register themselves using the @job_registry.register decorator.
"""

from typing import Type

from .base import BaseJobHandler


class JobRegistry:
    """
    Registry for local job handlers.

    Usage:
        from services.local_jobs import job_registry

        @job_registry.register
        class MyHandler(BaseJobHandler):
            job_type = "local_my_operation"
            ...
    """

    _handlers: dict[str, Type[BaseJobHandler]] = {}

    @classmethod
    def register(cls, handler_class: Type[BaseJobHandler]) -> Type[BaseJobHandler]:
        """
        Decorator to register a job handler.

        Args:
            handler_class: Handler class to register

        Returns:
            The same handler class (allows use as decorator)

        Raises:
            ValueError: If handler doesn't define job_type
        """
        if not handler_class.job_type:
            raise ValueError(
                f"Handler {handler_class.__name__} must define job_type class attribute"
            )

        if not handler_class.job_type.startswith("local_"):
            raise ValueError(
                f"Handler {handler_class.__name__} job_type must start with 'local_': "
                f"got '{handler_class.job_type}'"
            )

        cls._handlers[handler_class.job_type] = handler_class
        print(f"[JobRegistry] Registered handler: {handler_class.job_type}")
        return handler_class

    @classmethod
    def get_handler(cls, job_type: str) -> Type[BaseJobHandler] | None:
        """
        Get handler class for a job type.

        Args:
            job_type: Job type string

        Returns:
            Handler class or None if not found
        """
        return cls._handlers.get(job_type)

    @classmethod
    def get_all_types(cls) -> list[str]:
        """Get all registered job types."""
        return list(cls._handlers.keys())

    @classmethod
    def is_registered(cls, job_type: str) -> bool:
        """Check if a job type is registered."""
        return job_type in cls._handlers


# Singleton instance
job_registry = JobRegistry()
