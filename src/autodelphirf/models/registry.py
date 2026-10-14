"""Combined primary-model and comparator catalog used by the dispatcher."""

from .revroute_delphirf import PRIMARY_METHODS
from ..comparators.registry import COMPARATOR_METHODS
from ..comparators.delphirf import COMPARATOR_DELPHIRF_METHODS

DELPHIRF_METHODS = {**PRIMARY_METHODS, **COMPARATOR_DELPHIRF_METHODS}
MODEL_METHODS = tuple(DELPHIRF_METHODS) + tuple(
    method for method in COMPARATOR_METHODS if method not in DELPHIRF_METHODS)
DEFAULT_MODEL_METHODS = tuple(PRIMARY_METHODS)

__all__ = ["DEFAULT_MODEL_METHODS", "DELPHIRF_METHODS", "MODEL_METHODS"]
