import typing

from ._jit import jit

_T = typing.TypeVar("_T")


# adapted from https://github.com/mmore500/hstrat/blob/56cafcac/hstrat/_auxiliary_lib/_clamp.py
@jit(nopython=True)
def clamp(value: _T, lo: _T, hi: _T) -> _T:
    """Clamp `value` to [`lo`, `hi`], with `std::clamp` semantics.

    Returns `lo` if `value < lo`, `hi` if `hi < value`, and `value`
    otherwise. Requires `not hi < lo`. Unlike `np.clip`, supports scalars
    within numba-jitted functions.
    """
    assert not hi < lo
    if value < lo:
        return lo
    elif hi < value:
        return hi
    else:
        return value
