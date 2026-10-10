"""Canonical roles and labels for the primary method and comparators."""

PRIMARY_DELPHIRF_METHODS = {
    "delphirf": "RevRoute DelphiRF",
}

COMPARATOR_DELPHIRF_METHODS = {
    "naive_delphirf": "Naive DelphiRF",
    "similarity_weighted_delphirf": "Similarity-weighted DelphiRF",
    "global_delphirf": "Global DelphiRF",
}

DELPHIRF_METHODS = {**PRIMARY_DELPHIRF_METHODS, **COMPARATOR_DELPHIRF_METHODS}
COMPARATOR_METHODS = tuple(COMPARATOR_DELPHIRF_METHODS) + ("red",)
MODEL_METHODS = tuple(DELPHIRF_METHODS) + ("red",)
DEFAULT_MODEL_METHODS = tuple(PRIMARY_DELPHIRF_METHODS)

__all__ = ["COMPARATOR_DELPHIRF_METHODS", "COMPARATOR_METHODS",
           "DEFAULT_MODEL_METHODS", "DELPHIRF_METHODS", "MODEL_METHODS",
           "PRIMARY_DELPHIRF_METHODS"]
