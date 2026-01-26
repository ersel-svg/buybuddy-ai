"""
Local job handlers.

All handlers in this package are automatically registered when imported.
Add new handler imports here to register them.
"""

# Import handlers to register them with the registry
# OD Handlers
from .bulk_add_to_dataset import BulkAddToDatasetHandler
from .bulk_remove_from_dataset import BulkRemoveFromDatasetHandler
from .bulk_update_status import BulkUpdateStatusHandler
from .bulk_delete_images import BulkDeleteImagesHandler
from .export_dataset import ExportDatasetHandler

# Classification Handlers
from .cls_bulk_delete_images import BulkDeleteCLSImagesHandler
from .cls_bulk_add_to_dataset import BulkAddToCLSDatasetHandler
from .cls_bulk_remove_from_dataset import BulkRemoveFromCLSDatasetHandler
from .cls_bulk_set_labels import BulkSetCLSLabelsHandler
from .cls_bulk_clear_labels import BulkClearCLSLabelsHandler
from .cls_bulk_update_tags import BulkUpdateCLSTagsHandler

# Product Handlers
from .bulk_update_products import BulkUpdateProductsHandler
from .bulk_delete_products import BulkDeleteProductsHandler
from .bulk_product_matcher import BulkProductMatcherHandler

__all__ = [
    # OD Handlers
    "BulkAddToDatasetHandler",
    "BulkRemoveFromDatasetHandler",
    "BulkUpdateStatusHandler",
    "BulkDeleteImagesHandler",
    "ExportDatasetHandler",
    # Classification Handlers
    "BulkDeleteCLSImagesHandler",
    "BulkAddToCLSDatasetHandler",
    "BulkRemoveFromCLSDatasetHandler",
    "BulkSetCLSLabelsHandler",
    "BulkClearCLSLabelsHandler",
    "BulkUpdateCLSTagsHandler",
    # Product Handlers
    "BulkUpdateProductsHandler",
    "BulkDeleteProductsHandler",
    "BulkProductMatcherHandler",
]
