"""DelphiRF comparator strategies.

Naive and similarity-weighted DelphiRF retain DelphiRF's configured lag-band
fits; Global DelphiRF fits its location-and-lag fixed-effect model globally.
None of these strategies is part of the primary RevRoute--DelphiRF method.
"""

COMPARATOR_DELPHIRF_METHODS = {
    "naive_delphirf": "Naive DelphiRF",
    "similarity_weighted_delphirf": "Similarity-weighted DelphiRF",
    "global_delphirf": "Global DelphiRF",
}

__all__ = ["COMPARATOR_DELPHIRF_METHODS"]
