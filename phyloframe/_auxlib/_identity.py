import typing


def identity(x: typing.Any) -> typing.Any:
    """Return `x` unchanged.

    A picklable stand-in for `lambda x: x`.
    """
    return x
