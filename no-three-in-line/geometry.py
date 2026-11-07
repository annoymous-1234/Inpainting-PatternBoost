from __future__ import annotations

from math import gcd

Point = tuple[int, int]
LineKey = tuple[int, int, int]


def normalize_line(A: int, B: int, C: int) -> LineKey:
    """Return a canonical integer key for the line ``Ax + By = C``.

    The inputs are integer coefficients for a 2D line. The function divides by
    the joint gcd when possible and fixes the sign convention so equivalent
    lines map to the same hashable ``(A, B, C)`` tuple. It returns that tuple,
    does not mutate external state, and assumes ``A`` and ``B`` are not both
    zero.
    """
    if A == 0 and B == 0:
        raise ValueError("invalid line coefficients: A and B cannot both be zero")

    divisor = gcd(gcd(abs(A), abs(B)), abs(C))
    if divisor:
        A //= divisor
        B //= divisor
        C //= divisor

    if A < 0 or (A == 0 and B < 0):
        A, B, C = -A, -B, -C

    return (A, B, C)


def line_key(p: Point, q: Point) -> LineKey:
    """Return the canonical line key through two distinct lattice points.

    ``p`` and ``q`` are integer grid points ``(x, y)``. The function derives
    integer line coefficients, normalizes them with :func:`normalize_line`, and
    returns the resulting ``LineKey`` without mutating state. It assumes the
    two input points are different and raises ``ValueError`` otherwise.
    """
    if p == q:
        raise ValueError("line_key requires two distinct points")

    x1, y1 = p
    x2, y2 = q
    A = y2 - y1
    B = x1 - x2
    C = A * x1 + B * y1
    return normalize_line(A, B, C)


def point_on_line(p: Point, L: LineKey) -> bool:
    """Return whether an integer point satisfies a canonical line equation.

    ``p`` is a lattice point and ``L`` is a normalized ``(A, B, C)`` line key.
    The function evaluates the integer equation ``Ax + By = C`` and returns a
    boolean result. It does not mutate state and assumes ``L`` is a valid line
    key produced by this module.
    """
    x, y = p
    A, B, C = L
    return A * x + B * y == C
