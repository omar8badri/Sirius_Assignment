"""
Spatial Join Module
==================

This module provides spatial join functionality for matching PV assets
to irradiance data pixels.
"""

from .irradiance_matcher import IrradianceMatcher

__all__ = ["IrradianceMatcher"]
