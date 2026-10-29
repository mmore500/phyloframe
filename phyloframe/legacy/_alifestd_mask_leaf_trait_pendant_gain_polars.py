import argparse
import functools
import logging
import os
import typing
import warnings

import joinem
from joinem._dataframe_cli import _add_parser_base, _run_dataframe_cli
import numpy as np
import polars as pl

from .._auxlib._begin_prod_logging import begin_prod_logging
from .._auxlib._format_cli_description import format_cli_description
from .._auxlib._get_phyloframe_version import get_phyloframe_version
from .._auxlib._log_context_duration import log_context_duration
from .._auxlib._warn_experimental import warn_experimental
from ._alifestd_has_contiguous_ids_polars import (
    alifestd_has_contiguous_ids_polars,
)
from ._alifestd_is_topologically_sorted_polars import (
    alifestd_is_topologically_sorted_polars,
)
from ._alifestd_mask_leaf_trait_pendant_gain_asexual import (
    _alifestd_mask_leaf_trait_pendant_gain_asexual_fast_path,
)
from ._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)


def _alifestd_mask_leaf_trait_pendant_gain_polars_fill_null(
    trait_mask: pl.Series, name: str = "trait_mask"
) -> np.ndarray:
    """Implementation detail for
    `alifestd_mask_leaf_trait_pendant_gain_polars` and
    `alifestd_estimate_trait_gain_selection_pendant_polars`.

    Returns boolean `trait_mask` as an array, treating null values as False
    with a warning.
    """
    if trait_mask.has_nulls():
        warnings.warn(
            f"{name} contains null values, "
            "which will be treated as trait absent (False).",
        )
    return trait_mask.fill_null(False).to_numpy()


@warn_experimental
def alifestd_mask_leaf_trait_pendant_gain_polars(
    phylogeny_df: pl.DataFrame,
    *,
    trait_mask: typing.Union[str, pl.Expr],
    mark_as: str = "alifestd_mask_leaf_trait_pendant_gain_polars",
) -> pl.DataFrame:
    """Mark leaves that gained a trait on their own pendant branch, i.e.,
    leaves with the trait that is not shared by every leaf descending from
    their parent.

    Under parsimony, such a leaf's trait arose on its pendant branch; a
    leaf whose trait is shared by every leaf of its parent's clade could
    have gained it above its parent node, so is marked False. Root leaves
    and leaves without the trait are also marked False. Unifurcations are
    bypassed, so a leaf's parent is its nearest branching ancestor.

    Only supports asexual phylogenies.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame or polars.LazyFrame
        The phylogeny as a dataframe in alife standard format.

        Must represent an asexual phylogeny with contiguous,
        topologically sorted ids.
    trait_mask : str or polars.Expr
        Column name or boolean expression marking taxa with the trait.
        Values at inner nodes are ignored. Null values are treated as trait
        absent, with a warning.
    mark_as : str, default "alifestd_mask_leaf_trait_pendant_gain_polars"
        Column name for the boolean mask.

    Raises
    ------
    NotImplementedError
        If `phylogeny_df` has no "ancestor_id" column or if ids are
        non-contiguous or not topologically sorted.

    Returns
    -------
    polars.DataFrame
        The phylogeny with an added boolean mask column.

    See Also
    --------
    alifestd_mask_leaf_trait_pendant_gain_asexual :
        Pandas-based implementation.
    """
    if isinstance(trait_mask, str):
        trait_mask = pl.col(trait_mask)

    phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)
    schema_names = phylogeny_df.lazy().collect_schema().names()
    if "ancestor_id" not in schema_names:
        raise NotImplementedError(
            "alifestd_mask_leaf_trait_pendant_gain_polars only "
            "supports asexual phylogenies.",
        )

    if phylogeny_df.lazy().limit(1).collect().is_empty():
        return phylogeny_df.with_columns(
            pl.lit(False, dtype=pl.Boolean).alias(mark_as),
        )

    if not alifestd_has_contiguous_ids_polars(phylogeny_df):
        raise NotImplementedError(
            "non-contiguous ids not supported",
        )

    if not alifestd_is_topologically_sorted_polars(phylogeny_df):
        raise NotImplementedError(
            "non-topologically-sorted data not supported",
        )

    selected = (
        phylogeny_df.lazy()
        .select(
            pl.col("ancestor_id"),
            trait_mask=trait_mask.cast(pl.Boolean),
        )
        .collect()
    )
    result = _alifestd_mask_leaf_trait_pendant_gain_asexual_fast_path(
        selected["ancestor_id"].to_numpy(),
        _alifestd_mask_leaf_trait_pendant_gain_polars_fill_null(
            selected["trait_mask"]
        ),
    )

    return phylogeny_df.with_columns(
        pl.Series(name=mark_as, values=result),
    )


_raw_description = f"""\
{os.path.basename(__file__)} | \
(phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Mark leaves that gained a trait on their own pendant branch, i.e., leaves with
the trait that is not shared by every leaf descending from their parent.

Root leaves and leaves without the trait are marked False.

Data is assumed to be in alife standard format.
Only supports asexual phylogenies.

Additional Notes
================
- Requires contiguous, topologically sorted ids.

- Use `--eager-read` if modifying data file inplace.

- This CLI entrypoint is experimental and may be subject to change.

See Also
========
phyloframe.legacy._alifestd_mask_leaf_trait_pendant_gain_asexual :
    Pandas-based implementation.
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
        dfcli_module=(
            "phyloframe.legacy._alifestd_mask_leaf_trait_pendant_gain_polars"
        ),
        dfcli_version=get_phyloframe_version(),
    )
    parser.add_argument(
        "--trait-mask",
        required=True,
        type=str,
        help="Name of boolean column marking taxa with the trait.",
    )
    parser.add_argument(
        "--mark-as",
        default="alifestd_mask_leaf_trait_pendant_gain_polars",
        type=str,
        help=(
            "Column name for the boolean mask "
            "(default: alifestd_mask_leaf_trait_pendant_gain_polars)."
        ),
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()

    try:
        with log_context_duration(
            "phyloframe.legacy._alifestd_mask_leaf_trait_pendant_gain_polars",
            logging.info,
        ):
            _run_dataframe_cli(
                base_parser=parser,
                output_dataframe_op=functools.partial(
                    alifestd_mask_leaf_trait_pendant_gain_polars,
                    trait_mask=args.trait_mask,
                    mark_as=args.mark_as,
                ),
            )
    except NotImplementedError as e:
        logging.error(
            "- polars op not yet implemented, use pandas op instead",
        )
        raise e
