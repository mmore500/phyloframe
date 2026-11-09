def ceil_div(numerator: int, denominator: int) -> int:
    """Return `numerator / denominator`, rounded up, by exact integer
    arithmetic."""
    return -(-numerator // denominator)
