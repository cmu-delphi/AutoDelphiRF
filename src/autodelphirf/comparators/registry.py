"""Canonical comparator roles exposed to configuration and reporting."""

from .delphirf import COMPARATOR_DELPHIRF_METHODS

COMPARATOR_METHODS = tuple(COMPARATOR_DELPHIRF_METHODS) + ("red",)

__all__ = ["COMPARATOR_DELPHIRF_METHODS", "COMPARATOR_METHODS"]
