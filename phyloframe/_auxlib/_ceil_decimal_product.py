from decimal import Decimal
import math


def ceil_decimal_product(value: float, count: int) -> int:
    """Return `value * count`, rounded up, by exact decimal arithmetic on
    `value`'s shortest decimal representation (unlike
    `math.ceil(value * count)`, which rounds through floating point, giving,
    e.g., 8, not 7, for 0.07 * 100)."""
    return math.ceil(Decimal(str(value)) * count)
