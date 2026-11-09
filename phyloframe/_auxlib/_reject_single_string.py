import typing


def reject_single_string(value: typing.Any, *, name: str) -> None:
    """Raise TypeError if `value`, argument `name`, which should be a
    sequence of column names, is a single string, which would otherwise be
    iterated as characters."""
    if isinstance(value, str):
        raise TypeError(
            f"{name} must be a sequence of column names, not a single "
            f"string; did you mean [{value!r}]?",
        )
