"""Poker44 V4 coherent-chunk detector.

V4 is an additive challenger.  It intentionally leaves the deployed V3
artifacts and implementation untouched until an honest promotion gate passes.
"""

from .inference import Poker44V4Detector

__all__ = ["Poker44V4Detector"]
