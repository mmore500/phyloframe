import argparse
import functools
import logging
import os
import typing
import warnings

import joinem
from joinem._dataframe_cli import _add_parser_base, _run_dataframe_cli
import numpy as np
import pandas as pd

from .._auxlib._add_bool_arg import add_bool_arg
from .._auxlib._begin_prod_logging import begin_prod_logging
from .._auxlib._delegate_polars_implementation import (
    delegate_polars_implementation,
)
from .._auxlib._format_cli_description import format_cli_description
from .._auxlib._get_phyloframe_version import get_phyloframe_version
from .._auxlib._log_context_duration import log_context_duration
from ._alifestd_find_leaf_ids import alifestd_find_leaf_ids
from ._alifestd_has_contiguous_ids import alifestd_has_contiguous_ids
from ._alifestd_try_add_ancestor_id_col import alifestd_try_add_ancestor_id_col


def _deprecate_num_tips(
    fn: typing.Callable,
) -> typing.Callable:
    import inspect

    params = inspect.signature(fn).parameters
    target = "n_sample" if "n_sample" in params else "n_downsample"

    @functools.wraps(fn)
    def wrapper(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        if "num_tips" in kwargs:
            warnings.warn(
                f"num_tips is deprecated in favor of {target} and "
                "will be removed in a future release of phyloframe.",
                DeprecationWarning,
                stacklevel=2,
            )
            if target in kwargs:
                raise TypeError(
                    f"cannot specify both {target} and num_tips",
                )
            kwargs[target] = kwargs.pop("num_tips")
        return fn(*args, **kwargs)

    return wrapper


@_deprecate_num_tips
def alifestd_mark_sample_tips_canopy_asexual(
    phylogeny_df: pd.DataFrame,
    n_sample: typing.Optional[int] = None,
    mutate: bool = False,
    criterion: str = "origin_time",
    *,
    mark_as: str = "alifestd_mark_sample_tips_canopy_asexual",
) -> pd.DataFrame:
    """Mark the `n_sample` leaves with the largest `criterion` values.

    Adds a boolean column ``mark_as`` indicating retained tips.

    If `n_sample` is ``None``, it defaults to the number of leaves that
    share the maximum value of the `criterion` column. If `n_sample` is
    greater than or equal to the number of leaves in the phylogeny, all
    leaves are marked.  Ties are broken arbitrarily.

    Only supports asexual phylogenies.

    Parameters
    ----------
    phylogeny_df : pandas.DataFrame
        The phylogeny as a dataframe in alife standard format.

        Must represent an asexual phylogeny.
    n_sample : int, optional
        Number of tips to mark. If ``None``, defaults to the count of
        leaves with the maximum `criterion` value.
    mutate : bool, default False
        Are side effects on the input argument `phylogeny_df` allowed?
    criterion : str, default "origin_time"
        Column name used to rank leaves. The `n_sample` leaves with the
        largest values in this column are marked. Ties are broken
        arbitrarily.
    mark_as : str, default "alifestd_mark_sample_tips_canopy_asexual"
        Column name for the boolean mark.

    Raises
    ------
    ValueError
        If `criterion` is not a column in `phylogeny_df`.

    Returns
    -------
    pandas.DataFrame
        The phylogeny with an added boolean mark column.
    """
    if criterion not in phylogeny_df.columns:
        raise ValueError(
            f"criterion column {criterion!r} not found in phylogeny_df",
        )

    if not mutate:
        phylogeny_df = phylogeny_df.copy()

    phylogeny_df = alifestd_try_add_ancestor_id_col(phylogeny_df)
    if "ancestor_id" not in phylogeny_df.columns:
        raise ValueError(
            "alifestd_mark_sample_tips_canopy_asexual only supports "
            "asexual phylogenies.",
        )

    if phylogeny_df.empty:
        phylogeny_df[mark_as] = pd.Series(dtype=bool)
        return phylogeny_df

    if alifestd_has_contiguous_ids(phylogeny_df):
        # With contiguous IDs, id == row index so we can use direct
        # numpy array indexing instead of expensive .isin() calls.
        leaf_positions = alifestd_find_leaf_ids(phylogeny_df)
        leaf_df = phylogeny_df.iloc[leaf_positions]
        if n_sample is None:
            max_val = leaf_df[criterion].max()
            n_sample = int((leaf_df[criterion] == max_val).sum())
        kept_ids = leaf_df.nlargest(n_sample, criterion)["id"]
        phylogeny_df[mark_as] = np.bincount(
            kept_ids.to_numpy().astype(np.intp), minlength=len(phylogeny_df)
        ).astype(bool)
    else:
        tips = alifestd_find_leaf_ids(phylogeny_df)
        leaf_df = phylogeny_df.loc[phylogeny_df["id"].isin(tips)]
        if n_sample is None:
            max_val = leaf_df[criterion].max()
            n_sample = int((leaf_df[criterion] == max_val).sum())
        kept_ids = leaf_df.nlargest(n_sample, criterion)["id"]
        phylogeny_df[mark_as] = phylogeny_df["id"].isin(kept_ids)

    return phylogeny_df


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Mark the `-n` leaves with the largest `--criterion` values.

If `-n` is greater than or equal to the number of leaves in the phylogeny, all leaves are marked. Ties are broken arbitrarily.

Data is assumed to be in alife standard format.
Only supports asexual phylogenies.

Additional Notes
================
- Requires 'ancestor_id' column to be present in input DataFrame.
Otherwise, no action is taken.

- Use `--eager-read` if modifying data file inplace.

- This CLI entrypoint is experimental and may be subject to change.

See Also
========
phyloframe.legacy._alifestd_mark_sample_tips_canopy_polars :
    Entrypoint for high-performance Polars-based implementation.
"""


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        add_help=False,
        allow_abbrev=False,
        description=format_cli_description(_raw_description),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser = _add_parser_base(
        parser=parser,
        dfcli_module="phyloframe.legacy._alifestd_mark_sample_tips_canopy_asexual",
        dfcli_version=get_phyloframe_version(),
    )
    parser.add_argument(
        "-n",
        default=None,
        type=int,
        help="Number of tips to mark. If omitted, defaults to the count of leaves with the maximum criterion value.",
    )
    parser.add_argument(
        "--criterion",
        default="origin_time",
        type=str,
        help="Column name used to rank leaves; ties broken arbitrarily (default: origin_time).",
    )
    parser.add_argument(
        "--mark-as",
        default="alifestd_mark_sample_tips_canopy_asexual",
        type=str,
        help="Column name for the boolean mark (default: alifestd_mark_sample_tips_canopy_asexual).",
    )
    add_bool_arg(
        parser,
        "ignore-topological-sensitivity",
        default=False,
        help="suppress topological sensitivity warning (default: False)",
    )
    add_bool_arg(
        parser,
        "drop-topological-sensitivity",
        default=False,
        help="drop topology-sensitive columns from output (default: False)",
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()
    with log_context_duration(
        "phyloframe.legacy._alifestd_mark_sample_tips_canopy_asexual",
        logging.info,
    ):
        _run_dataframe_cli(
            base_parser=parser,
            output_dataframe_op=delegate_polars_implementation()(
                functools.partial(
                    alifestd_mark_sample_tips_canopy_asexual,
                    n_sample=args.n,
                    criterion=args.criterion,
                    mark_as=args.mark_as,
                ),
            ),
        )
