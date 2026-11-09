def ceil_div(numerator: int, denominator: int) -> int:
    """Return `numerator / denominator`, rounded up, by exact integer
    arithmetic (unlike `math.ceil(numerator / denominator)`, which rounds
    through floating point)."""
    return -(-numerator // denominator)
