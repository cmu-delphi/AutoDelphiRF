"""Stage 5: prospective calibration, diagnostics, and report generation."""

from .calibration import prospective_calibration
from .report import build_report
from .validation import *  # noqa: F401,F403 - public assessment surface

__all__ = ["prospective_calibration", "build_report"]
