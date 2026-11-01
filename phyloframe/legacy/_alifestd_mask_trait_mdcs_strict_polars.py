import argparse
import functools
import logging
import os
import typing

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
from ._alifestd_mask_trait_mdcs_strict_asexual import (
    _alifestd_mask_trait_mdcs_strict_asexual_fast_path,
)
from ._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)


def _alifestd_mask_trait_polars_impl(
    phylogeny_df: pl.DataFrame,
    *,
    trait_mask: typing.Union[str, pl.Expr],
    mark_as: str,
    fast_path: typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
    caller: str,
) -> pl.DataFrame:
    """Implementation detail for `alifestd_mask_trait_mdcs_strict_polars`
    and `alifestd_mask_trait_singletons_polars`.

    Validates `phylogeny_df` and `trait_mask`, then marks nodes by
    `fast_path(ancestor_ids, trait_mask)`.
    """
    if isinstance(trait_mask, str):
        trait_mask = pl.col(trait_mask)

    phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)
    schema_names = phylogeny_df.lazy().collect_schema().names()
    if "ancestor_id" not in schema_names:
        raise NotImplementedError(
            f"{caller} only supports asexual phylogenies.",
        )
    if phylogeny_df.lazy().limit(1).collect().is_empty():
        return phylogeny_df.with_columns(
            pl.lit(False, dtype=pl.Boolean).alias(mark_as),
        )
    if not alifestd_has_contiguous_ids_polars(phylogeny_df):
        raise NotImplementedError("non-contiguous ids not supported")
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
    if selected["trait_mask"].has_nulls():
        raise NotImplementedError("null trait_mask values not supported")
    result = fast_path(
        selected["ancestor_id"].to_numpy(),
        selected["trait_mask"].to_numpy(),
    )
    return phylogeny_df.with_columns(pl.Series(name=mark_as, values=result))


@warn_experimental
def alifestd_mask_trait_mdcs_strict_polars(
    phylogeny_df: pl.DataFrame,
    *,
    trait_mask: typing.Union[str, pl.Expr],
    mark_as: str = "alifestd_mask_trait_mdcs_strict_polars",
) -> pl.DataFrame:
    """Mark the root of each mutationally defined clade (MDC): a maximal
    clade whose leaves all have a trait, i.e., a clade whose every leaf has
    the trait but whose parent's clade does not.

    Under parsimony without loss, an MDC gained the trait on its stem
    branch. Roots are marked False; along a chain of unifurcations, only the
    rootmost node is marked. See `alifestd_mask_trait_mdcs_strict_asexual`.

    Only supports asexual phylogenies.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame or polars.LazyFrame
        The phylogeny as a dataframe in alife standard format.

        Must represent an asexual phylogeny with contiguous,
        topologically sorted ids.
    trait_mask : str or polars.Expr
        Column name or boolean expression marking taxa with the trait.
        Values at inner nodes are ignored. Null values are not supported.
    mark_as : str, default "alifestd_mask_trait_mdcs_strict_polars"
        Column name for the boolean mask.

    Raises
    ------
    NotImplementedError
        If `phylogeny_df` has no "ancestor_id" column, if ids are
        non-contiguous or not topologically sorted, or if `trait_mask` has
        null values.

    Returns
    -------
    polars.DataFrame
        The phylogeny with an added boolean mask column.

    See Also
    --------
    alifestd_mask_monomorphic_clades_polars :
        Marks every node whose clade's leaves share a trait value, present
        or absent; this function marks only the rootmost node of each clade
        whose leaves all have the trait.
    alifestd_mask_trait_mdcs_strict_asexual :
        Pandas-based implementation.
    """
    return _alifestd_mask_trait_polars_impl(
        phylogeny_df,
        trait_mask=trait_mask,
        mark_as=mark_as,
        fast_path=_alifestd_mask_trait_mdcs_strict_asexual_fast_path,
        caller="alifestd_mask_trait_mdcs_strict_polars",
    )


_raw_description = f"""\
{os.path.basename(__file__)} | \
(phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Mark the root of each mutationally defined clade (MDC): a maximal clade
whose leaves all have a trait, which gained it on its stem branch.

Roots are marked False.

Data is assumed to be in alife standard format.
Only supports asexual phylogenies.

Additional Notes
================
- Requires contiguous, topologically sorted ids.

- Use `--eager-read` if modifying data file inplace.

- This CLI entrypoint is experimental and may be subject to change.

See Also
========
phyloframe.legacy._alifestd_mask_trait_mdcs_strict_asexual :
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
            "phyloframe.legacy._alifestd_mask_trait_mdcs_strict_polars"
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
        default="alifestd_mask_trait_mdcs_strict_polars",
        type=str,
        help=(
            "Column name for the boolean mask "
            "(default: alifestd_mask_trait_mdcs_strict_polars)."
        ),
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()

    try:
        with log_context_duration(
            "phyloframe.legacy._alifestd_mask_trait_mdcs_strict_polars",
            logging.info,
        ):
            _run_dataframe_cli(
                base_parser=parser,
                output_dataframe_op=functools.partial(
                    alifestd_mask_trait_mdcs_strict_polars,
                    trait_mask=args.trait_mask,
                    mark_as=args.mark_as,
                ),
            )
    except NotImplementedError as e:
        logging.error(
            "- polars op not yet implemented, use pandas op instead",
        )
        raise e
