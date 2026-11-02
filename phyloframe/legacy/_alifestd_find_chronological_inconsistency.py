import numbers
import typing

import numpy as np
import pandas as pd

from .._auxlib._jit import jit
from .._auxlib._jit_TypingError import jit_TypingError
from ._alifestd_has_compact_ids import alifestd_has_compact_ids
from ._alifestd_has_contiguous_ids import alifestd_has_contiguous_ids
from ._alifestd_parse_ancestor_ids import alifestd_parse_ancestor_ids


@jit(nopython=True)
def _alifestd_find_chronological_inconsistency_contiguous(
    ancestor_ids: np.ndarray, origin_times: np.ndarray, epsilon: numbers.Real
) -> typing.Optional[int]:
    for id_, ancestor_id in enumerate(ancestor_ids):
        if ancestor_id == id_:
            continue  # handle root cases
        if origin_times[ancestor_id] + epsilon > origin_times[id_]:
            return id_
    return None


@jit(nopython=True)
def _alifestd_find_chronological_inconsistency_compact(
    ids: np.ndarray,
    ancestor_ids: np.ndarray,
    origin_times: np.ndarray,
    epsilon: numbers.Real,
) -> typing.Optional[int]:
    origin_time_lookup = np.empty(len(ids))
    for id_, origin_time in zip(ids, origin_times):
        origin_time_lookup[id_] = origin_time

    for id_, ancestor_id in zip(ids, ancestor_ids):
        if ancestor_id == id_:
            continue  # handle root cases
        if origin_time_lookup[ancestor_id] + epsilon > origin_time_lookup[id_]:
            return id_
    return None


@jit(nopython=True)
def _alifestd_find_chronological_inconsistency_arbitrary(
    ids: np.ndarray,
    ancestor_ids: np.ndarray,
    origin_times: np.ndarray,
    epsilon: numbers.Real,
) -> typing.Optional[int]:
    origin_time_lookup = dict()
    for id_, origin_time in zip(ids, origin_times):
        origin_time_lookup[id_] = origin_time

    for id_, ancestor_id in zip(ids, ancestor_ids):
        if ancestor_id == id_:
            continue  # handle root cases
        if origin_time_lookup[ancestor_id] + epsilon > origin_time_lookup[id_]:
            return id_
    return None


def _alifestd_find_chronological_inconsistency_asexual(
    phylogeny_df: pd.DataFrame, unwrap: typing.Callable, epsilon: numbers.Real
) -> typing.Optional[int]:
    if alifestd_has_contiguous_ids(phylogeny_df):
        call = unwrap(_alifestd_find_chronological_inconsistency_contiguous)
        return call(
            phylogeny_df["ancestor_id"].to_numpy(),
            phylogeny_df["origin_time"].to_numpy(),
            epsilon,
        )
    elif alifestd_has_compact_ids(phylogeny_df):
        call = unwrap(_alifestd_find_chronological_inconsistency_compact)
        return call(
            phylogeny_df["id"].to_numpy(),
            phylogeny_df["ancestor_id"].to_numpy(),
            phylogeny_df["origin_time"].to_numpy(),
            epsilon,
        )
    else:
        call = unwrap(_alifestd_find_chronological_inconsistency_arbitrary)
        return call(
            phylogeny_df["id"].to_numpy(),
            phylogeny_df["ancestor_id"].to_numpy(),
            phylogeny_df["origin_time"].to_numpy(),
            epsilon,
        )


def alifestd_find_chronological_inconsistency(
    phylogeny_df: pd.DataFrame,
    *,
    epsilon: numbers.Real = 0,
) -> typing.Optional[int]:
    """Return the id of a taxon with origin time preceding its parent's
    origin time plus `epsilon`, if any are present.

    Root taxa are never reported.
    """

    if "ancestor_id" in phylogeny_df.columns:
        for unwrap in lambda x: x, lambda x: x.__wrapped__:
            try:
                return _alifestd_find_chronological_inconsistency_asexual(
                    phylogeny_df, unwrap, epsilon
                )
            except jit_TypingError:  # pragma: no cover
                # this does get exercised in tests, but not in coverage
                # because jit is disabled for better visibility of jit code
                continue
            except ValueError:
                continue  # if origin_times are sequences i.e., tuples
            except TypeError:
                break  # e.g., can't add epsilon to str origin_times

    phylogeny_df = phylogeny_df.set_index("id", drop=False)

    origin_time_loc = phylogeny_df.columns.get_loc("origin_time")
    for _pos, (_idx, row) in enumerate(phylogeny_df.iterrows()):
        for ancestor_id in alifestd_parse_ancestor_ids(row["ancestor_list"]):
            ancestor_loc = phylogeny_df.index.get_loc(ancestor_id)
            ancestor_time = phylogeny_df.iat[ancestor_loc, origin_time_loc]
            if epsilon != 0:  # skip for non-numeric types, e.g., str
                ancestor_time = ancestor_time + epsilon
            if ancestor_time > row["origin_time"]:
                return _idx

    return None
