import numpy as np

from ._jit import jit


@jit(nopython=True)
def jit_parse_float(
    chars: np.ndarray,
    start: int,
    stop: int,
) -> float:
    """Parse a float from ``chars[start:stop]``.

    Supports optional sign, integer part, fractional part, and
    scientific notation (e/E with optional signed exponent).

    Parameters
    ----------
    chars : np.ndarray
        uint8 byte array (e.g., from ``str.encode("ascii")``).
    start : int
        Start index (inclusive) into *chars*.
    stop : int
        Stop index (exclusive) into *chars*.

    Returns
    -------
    float
        The parsed value, or ``nan`` if the range is empty or
        contains only whitespace.
    """
    ZERO = np.uint8(ord("0"))
    NINE = np.uint8(ord("9"))
    MINUS = np.uint8(ord("-"))
    PLUS = np.uint8(ord("+"))
    DOT = np.uint8(ord("."))
    E_LOWER = np.uint8(ord("e"))
    E_UPPER = np.uint8(ord("E"))

    # skip leading whitespace
    i = start
    while i < stop and chars[i] == np.uint8(32):
        i += 1

    if i >= stop:
        return np.nan

    # sign
    negative = False
    if chars[i] == MINUS:
        negative = True
        i += 1
    elif chars[i] == PLUS:
        i += 1

    # integer part
    result = 0.0
    while i < stop and ZERO <= chars[i] <= NINE:
        result = result * 10.0 + np.float64(chars[i] - ZERO)
        i += 1

    # fractional part
    if i < stop and chars[i] == DOT:
        i += 1
        frac = 0.0
        frac_digits = 0
        while i < stop and ZERO <= chars[i] <= NINE:
            frac = frac * 10.0 + np.float64(chars[i] - ZERO)
            frac_digits += 1
            i += 1
        if frac_digits > 0:
            divisor = 1.0
            for _d in range(frac_digits):
                divisor *= 10.0
            result += frac / divisor

    # exponent
    if i < stop and (chars[i] == E_LOWER or chars[i] == E_UPPER):
        i += 1
        exp_neg = False
        if i < stop and chars[i] == MINUS:
            exp_neg = True
            i += 1
        elif i < stop and chars[i] == PLUS:
            i += 1
        exp = 0
        while i < stop and ZERO <= chars[i] <= NINE:
            exp = exp * 10 + (chars[i] - ZERO)
            i += 1
        if exp_neg:
            for _e in range(exp):
                result /= 10.0
        else:
            for _e in range(exp):
                result *= 10.0

    if negative:
        result = -result

    return result
