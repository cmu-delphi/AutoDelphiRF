"""Stage 2: delegate raw-triangle feature construction to DelphiRF."""

from .prepare import (PreparationError, PreparationSpec, build_spec,
                      prepare_dataset, run_r_bridge)

__all__ = ["PreparationError", "PreparationSpec", "build_spec",
           "prepare_dataset", "run_r_bridge"]
