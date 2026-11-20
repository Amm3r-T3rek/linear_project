"""Exact arithmetic for LP solving.

Provides :class:`BigM`, a symbolic number of the form ``a + b*M`` where ``a``
and ``b`` are :class:`fractions.Fraction` and ``M`` is a symbol standing for an
arbitrarily large positive constant.  ``M`` is *never* given a numeric value;
all comparisons are lexicographic (the ``M``-coefficient compared first).

This lets the Big-M method run with exact rational arithmetic while still
correctly recognising that, e.g. ``-4 + 7M`` is larger than ``-1 + 4M``.

``BigM`` also doubles as an exact rational number: ``BigM(5)`` behaves exactly
like ``Fraction(5)``.  The plain Simplex / Two-Phase / Dual-Simplex engines
therefore run on ``BigM`` values whose ``b`` part is always zero, giving exact
fractional results.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Union

Number = Union[int, Fraction, "BigM"]


class BigM:
    """A number of the form ``a + b * M`` with exact rational ``a``, ``b``.

    Comparison is lexicographic with the ``M``-coefficient taking priority:
    ``a1 + b1*M  <  a2 + b2*M``  iff  ``b1 < b2`` or ``(b1 == b2 and a1 < a2)``.
    """

    __slots__ = ("a", "b")

    def __init__(self, a: Number = 0, b: Number = 0) -> None:
        if isinstance(a, BigM):
            self.a = a.a
            self.b = a.b + Fraction(b)
        else:
            self.a = Fraction(a)
            self.b = Fraction(b)

    # -- conversions -------------------------------------------------------
    @classmethod
    def coerce(cls, x: Number) -> "BigM":
        if isinstance(x, BigM):
            return cls(x.a, x.b)
        return cls(Fraction(x), 0)

    @property
    def is_finite(self) -> bool:
        """True when the ``M``-part is zero (i.e. a plain rational number)."""
        return self.b == 0

    # -- arithmetic --------------------------------------------------------
    def __add__(self, other: Number) -> "BigM":
        o = BigM.coerce(other)
        return BigM(self.a + o.a, self.b + o.b)

    __radd__ = __add__

    def __sub__(self, other: Number) -> "BigM":
        o = BigM.coerce(other)
        return BigM(self.a - o.a, self.b - o.b)

    def __rsub__(self, other: Number) -> "BigM":
        o = BigM.coerce(other)
        return BigM(o.a - self.a, o.b - self.b)

    def __neg__(self) -> "BigM":
        return BigM(-self.a, -self.b)

    def __mul__(self, other: Number) -> "BigM":
        o = BigM.coerce(other)
        # (a1 + b1 M)(a2 + b2 M) = a1 a2 + (a1 b2 + a2 b1) M + b1 b2 M^2.
        # M^2 only arises when *both* factors carry an M-term; this never
        # happens in a valid Big-M tableau (constraint rows stay M-free), so
        # we raise rather than silently producing a wrong result.
        if self.b != 0 and o.b != 0:
            raise ValueError(
                "BigM * BigM with both M-coefficients nonzero (M^2) is undefined"
            )
        return BigM(self.a * o.a, self.a * o.b + o.a * self.b)

    __rmul__ = __mul__

    def __truediv__(self, other: Number) -> "BigM":
        o = BigM.coerce(other)
        if o.b != 0:
            raise ValueError("Division by a BigM with nonzero M-part is undefined")
        if o.a == 0:
            raise ZeroDivisionError("BigM division by zero")
        return BigM(self.a / o.a, self.b / o.a)

    def __rtruediv__(self, other: Number) -> "BigM":
        o = BigM.coerce(other)
        return o.__truediv__(self)

    # -- comparison (lexicographic, M first) -------------------------------
    def _key(self):
        return (self.b, self.a)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, (BigM, int, Fraction)):
            return NotImplemented
        o = BigM.coerce(other)  # type: ignore[arg-type]
        return self.a == o.a and self.b == o.b

    def __hash__(self) -> int:
        return hash((self.a, self.b))

    def __lt__(self, other: Number) -> bool:
        o = BigM.coerce(other)
        return self._key() < o._key()

    def __le__(self, other: Number) -> bool:
        o = BigM.coerce(other)
        return self._key() <= o._key()

    def __gt__(self, other: Number) -> bool:
        o = BigM.coerce(other)
        return self._key() > o._key()

    def __ge__(self, other: Number) -> bool:
        o = BigM.coerce(other)
        return self._key() >= o._key()

    def __bool__(self) -> bool:
        return self.a != 0 or self.b != 0

    # -- formatting --------------------------------------------------------
    def to_fraction(self) -> Fraction:
        """Return the rational value, requiring the M-part to be zero."""
        if self.b != 0:
            raise ValueError(f"{self} has a nonzero M-part; cannot convert to Fraction")
        return self.a

    def format(self, mode: str = "frac") -> str:
        """Human-readable string.

        ``mode='frac'``  -> ``5/3``, ``-2``, ``4 - 7M`` ...
        ``mode='dec'``   -> decimal approximations (M kept symbolic).
        """
        a, b = self.a, self.b
        if mode == "dec":
            a_s = _format_fraction_dec(a)
        else:
            a_s = _format_fraction(a)

        if b == 0:
            return a_s
        if mode == "dec":
            b_s = _format_fraction_dec(b)
        else:
            b_s = _format_fraction(b)

        # Build the M-term string.
        if b == 1:
            m_term = "M"
        elif b == -1:
            m_term = "-M"
        else:
            m_term = f"{b_s}M"

        if a == 0:
            return m_term
        # sign of the M term
        if b > 0:
            return f"{a_s} + {m_term}" if b == 1 else f"{a_s} + {b_s}M"
        # b < 0
        if b == -1:
            return f"{a_s} - M"
        return f"{a_s} - {b_s}M"

    def __str__(self) -> str:
        return self.format("frac")

    def __repr__(self) -> str:
        return f"BigM({self.a!r}, {self.b!r})"


def _format_fraction(f: Fraction) -> str:
    """Format a :class:`Fraction` as a clean string (``5``, ``-2/3`` ...)."""
    if f.denominator == 1:
        return str(f.numerator)
    return f"{f.numerator}/{f.denominator}"


def _format_fraction_dec(f: Fraction) -> str:
    """Format a :class:`Fraction` as a rounded decimal string."""
    if f.denominator == 1:
        return str(f.numerator)
    # show up to 4 decimals, trimming trailing zeros
    val = float(f)
    s = f"{val:.4f}".rstrip("0").rstrip(".")
    return s


def to_bigm(x: Number) -> BigM:
    """Convenience converter."""
    return BigM.coerce(x)
