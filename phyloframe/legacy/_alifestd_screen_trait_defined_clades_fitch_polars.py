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
from ._alifestd_has_contiguous_ids_polars import (
    alifestd_has_contiguous_ids_polars,
)
from ._alifestd_is_topologically_sorted_polars import (
    alifestd_is_topologically_sorted_polars,
)
from ._alifestd_screen_trait_defined_clades_fitch_asexual import (
    _alifestd_screen_trait_defined_clades_fitch_asexual_fast_path,
)
from ._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)


def alifestd_screen_trait_defined_clades_fitch_polars(
    phylogeny_df: pl.DataFrame,
    *,
    mask_trait_absent: typing.Union[str, pl.Expr],
    mask_trait_present: typing.Union[str, pl.Expr],
    mark_as: str = "alifestd_screen_trait_defined_clades_fitch_polars",
) -> pl.DataFrame:
    """Perform a maximum parsimony screen for trait-defined clades using
    Fitch's algorithm.

    Marks nodes where the trait is gained, i.e., nodes inferred to have the
    trait whose parent is inferred not to. At polytomies, ambiguity is
    resolved by majority vote among children. Ambiguous root state is
    resolved as trait absent.

    Only supports asexual phylogenies.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame or polars.LazyFrame
        The phylogeny as a dataframe in alife standard format.

        Must represent an asexual phylogeny with contiguous,
        topologically sorted ids.
    mask_trait_absent : str or polars.Expr
        Column name or boolean expression marking taxa observed to lack the
        trait. Null values are treated as unobserved, with a warning.
    mask_trait_present : str or polars.Expr
        Column name or boolean expression marking taxa observed to have the
        trait. Null values are treated as unobserved, with a warning.

        Taxa marked neither absent nor present are unobserved, and have
        their trait state inferred. For instance, &'ing with `is_leaf` (see
        `alifestd_mark_leaves_polars`) can be used to only consider traits
        on leaves.
    mark_as : str, default "alifestd_screen_trait_defined_clades_fitch_polars"
        Column name for the boolean mark.

    Raises
    ------
    NotImplementedError
        If `phylogeny_df` has no "ancestor_id" column or if ids are
        non-contiguous or not topologically sorted.
    ValueError
        If any taxon is marked both trait absent and trait present.

    Returns
    -------
    polars.DataFrame
        The phylogeny with an added boolean mark column.

    See Also
    --------
    alifestd_screen_trait_defined_clades_fitch_asexual :
        Pandas-based implementation.
    """
    if isinstance(mask_trait_absent, str):
        mask_trait_absent = pl.col(mask_trait_absent)
    if isinstance(mask_trait_present, str):
        mask_trait_present = pl.col(mask_trait_present)

    phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)
    schema_names = phylogeny_df.lazy().collect_schema().names()
    if "ancestor_id" not in schema_names:
        raise NotImplementedError(
            "alifestd_screen_trait_defined_clades_fitch_polars only "
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
            mask_trait_absent=mask_trait_absent.cast(pl.Boolean),
            mask_trait_present=mask_trait_present.cast(pl.Boolean),
        )
        .collect()
    )
    for mask_name in "mask_trait_absent", "mask_trait_present":
        if selected[mask_name].has_nulls():
            warnings.warn(
                f"{mask_name} contains null values, "
                "which will be treated as unobserved (False).",
            )

    ancestor_ids = selected["ancestor_id"].to_numpy()
    mask_trait_absent = (
        selected["mask_trait_absent"].fill_null(False).to_numpy()
    )
    mask_trait_present = (
        selected["mask_trait_present"].fill_null(False).to_numpy()
    )
    del selected

    if (mask_trait_absent & mask_trait_present).any():
        raise ValueError("conflicting trait presence/absence masks")

    logging.info(
        "- alifestd_screen_trait_defined_clades_fitch_polars: "
        "performing parsimony screen...",
    )
    result = _alifestd_screen_trait_defined_clades_fitch_asexual_fast_path(
        ancestor_ids,
        (mask_trait_absent + 2 * mask_trait_present).astype(np.uint8),
    )

    return phylogeny_df.with_columns(
        pl.Series(name=mark_as, values=result),
    )


_raw_description = f"""\
{os.path.basename(__file__)} | \
(phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Perform a maximum parsimony screen for trait-defined clades using Fitch's
algorithm, marking nodes where the trait is gained.

Ambiguous root state is resolved as trait absent.

Data is assumed to be in alife standard format.
Only supports asexual phylogenies.

Additional Notes
================
- Requires contiguous, topologically sorted ids.

- Taxa marked by neither `--mask-trait-absent` nor `--mask-trait-present`
are treated as unobserved, and have their trait state inferred.

- Use `--eager-read` if modifying data file inplace.

- This CLI entrypoint is experimental and may be subject to change.

See Also
========
phyloframe.legacy._alifestd_screen_trait_defined_clades_fitch_asexual :
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
            "phyloframe.legacy._alifestd_screen_trait_defined_clades_fitch_polars"
        ),
        dfcli_version=get_phyloframe_version(),
    )
    parser.add_argument(
        "--mask-trait-absent",
        required=True,
        type=str,
        help="Name of boolean column marking taxa observed to lack the trait.",
    )
    parser.add_argument(
        "--mask-trait-present",
        required=True,
        type=str,
        help="Name of boolean column marking taxa observed to have the trait.",
    )
    parser.add_argument(
        "--mark-as",
        default="alifestd_screen_trait_defined_clades_fitch_polars",
        type=str,
        help=(
            "Column name for the boolean mark "
            "(default: alifestd_screen_trait_defined_clades_fitch_polars)."
        ),
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()

    try:
        with log_context_duration(
            "phyloframe.legacy._alifestd_screen_trait_defined_clades_fitch_polars",
            logging.info,
        ):
            _run_dataframe_cli(
                base_parser=parser,
                output_dataframe_op=functools.partial(
                    alifestd_screen_trait_defined_clades_fitch_polars,
                    mask_trait_absent=args.mask_trait_absent,
                    mask_trait_present=args.mask_trait_present,
                    mark_as=args.mark_as,
                ),
            )
    except NotImplementedError as e:
        logging.error(
            "- polars op not yet implemented, use pandas op CLI instead",
        )
        raise e
