import typing


def identity(x: typing.Any) -> typing.Any:
    """Return `x` unchanged.

    A picklable stand-in for `lambda x: x`, e.g., as a default no-op
    `progress_wrap` that can be passed to process-based parallel backends.
    """
    return x
