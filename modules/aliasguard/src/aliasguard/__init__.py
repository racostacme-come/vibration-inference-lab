"""Streaming decimation with explicit sampling and delay conventions."""

from .signal import Decimator, alias_frequency, frequency_response, lowpass

__all__ = ["Decimator", "alias_frequency", "frequency_response", "lowpass"]
