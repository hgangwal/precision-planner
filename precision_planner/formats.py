"""Numeric formats and rounding modes that a plan can assign."""

from __future__ import annotations

from enum import Enum


class Format(str, Enum):
    """Precision a single layer can run at.

    Subclassing ``str`` lets formats serialize to JSON as plain strings
    (``"fp16"``) and compare equal to them.
    """

    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    FP8 = "fp8"

    @property
    def bytes_per_param(self) -> float:
        """Storage cost of one weight in this format (ignores scales/zero-points)."""
        return _BYTES_PER_PARAM[self]

    @property
    def label(self) -> str:
        """Display name, e.g. ``"FP16"``."""
        return self.value.upper()


_BYTES_PER_PARAM = {
    Format.FP32: 4.0,
    Format.FP16: 2.0,
    Format.INT8: 1.0,
    Format.FP8: 1.0,
}


class Rounding(str, Enum):
    """How a value is rounded when cast to a lower-precision format.

    Nearest is deterministic. Stochastic rounds up with probability equal to
    the fractional distance, which is unbiased in expectation and prevented
    oscillation in the mixed-precision Newton's method work this is based on.
    """

    NEAREST = "nearest"
    STOCHASTIC = "stochastic"
