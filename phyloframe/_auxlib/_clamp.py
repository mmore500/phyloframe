import typing

from ._jit import jit

_T = typing.TypeVar("_T")


# adapted from https://github.com/mmore500/hstrat/blob/56cafcac/hstrat/_auxiliary_lib/_clamp.py
@jit(nopython=True)
def clamp(
    value: _T,
    lo: typing.Optional[_T] = None,
    hi: typing.Optional[_T] = None,
) -> _T:
    """Clamp `value` to [`lo`, `hi`], with `std::clamp` semantics.

    Returns `lo` if `value < lo`, `hi` if `hi < value`, and `value`
    otherwise; a bound of None, the default, is not applied. Requires
    `not hi < lo`. Unlike `np.clip`, supports scalars within numba-jitted
    functions.
    """
    assert lo is None or hi is None or not hi < lo
    if lo is not None and value < lo:
        return lo
    elif hi is not None and hi < value:
        return hi
    else:
        return value
