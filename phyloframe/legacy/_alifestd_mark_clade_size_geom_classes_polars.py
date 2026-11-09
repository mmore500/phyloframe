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
from .._auxlib._calc_geometric_bins import calc_geometric_bins
from .._auxlib._clamp import clamp
from .._auxlib._format_cli_description import format_cli_description
from .._auxlib._get_phyloframe_version import get_phyloframe_version
from .._auxlib._log_context_duration import log_context_duration
from ._alifestd_mark_num_leaves_polars import alifestd_mark_num_leaves_polars
from ._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)


def alifestd_mark_clade_size_geom_classes_polars(
    phylogeny_df: pl.DataFrame,
    *,
    size_ratio: float = 2.0,
    weight_by: typing.Union[str, typing.Sequence[str]] = (),
    min_class_weight: float = 0.0,
    mark_as: str = "clade_size_geom_class",
) -> pl.DataFrame:
    """Add column `clade_size_geom_class`, classing nodes by clade size
    (i.e., leaf count) on a log scale.

    The default `size_ratio` of 2 classes clades of 1, 2, 3-4, 5-8, 9-16,
    ... leaves. Classes are numbered consecutively from 0, skipping empty
    ones, so leaves are class 0.

    If `min_class_weight` is given, the first class above class 0 with less
    summed `weight_by` and all classes of larger clades merge into the class
    below. Class 1 is always kept, so leaves stay apart from inner nodes.

    The output column name can be changed via the ``mark_as`` parameter.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame or polars.LazyFrame
        The phylogeny as a dataframe in alife standard format.

        Must represent an asexual phylogeny with contiguous, topologically
        sorted ids.
    size_ratio : float, default 2.0
        Ratio between successive classes' clade size bounds.
    weight_by : str or Sequence[str], default ()
        Numeric columns summed as node weights.
    min_class_weight : float, default 0.0
        Least weight for a class to avoid merging.
    mark_as : str, default "clade_size_geom_class"
        Output column name.

    See Also
    --------
    alifestd_mark_num_leaves_polars :
        Marks clade sizes.
    """
    if not size_ratio > 1:
        raise ValueError(f"size_ratio must exceed 1, got {size_ratio}")
    if isinstance(weight_by, str):
        weight_by = [weight_by]

    if "num_leaves" not in phylogeny_df.lazy().collect_schema().names():
        phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)
        phylogeny_df = alifestd_mark_num_leaves_polars(phylogeny_df)

    logging.info(
        "- alifestd_mark_clade_size_geom_classes_polars: classing...",
    )
    selected = (
        phylogeny_df.lazy()
        .select(
            "num_leaves",
            pl.sum_horizontal(*weight_by, pl.lit(0.0)).alias("weight"),
        )
        .collect()
    )
    __, classes = np.unique(
        calc_geometric_bins(
            selected["num_leaves"].to_numpy(), ratio=size_ratio
        ),
        return_inverse=True,
    )
    class_weights = np.bincount(classes, selected["weight"].to_numpy())
    light_classes = np.flatnonzero(class_weights[1:] < min_class_weight) + 1
    if len(light_classes):  # merge down, keeping class 1
        max_class = clamp(light_classes[0] - 1, lo=1)
        classes = np.clip(classes, a_min=None, a_max=max_class)

    return phylogeny_df.with_columns(
        pl.Series(name=mark_as, values=classes, dtype=pl.Int64),
    )


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Add column `clade_size_geom_class`, classing nodes by clade size (i.e., leaf count) on a log scale.

Data is assumed to be in alife standard format.

Additional Notes
================
- Requires asexual phylogeny with contiguous, topologically sorted ids.

- Use `--eager-read` if modifying data file inplace.

- This CLI entrypoint is experimental and may be subject to change.
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
        dfcli_module="phyloframe.legacy._alifestd_mark_clade_size_geom_classes_polars",
        dfcli_version=get_phyloframe_version(),
    )
    parser.add_argument(
        "--size-ratio",
        default=2.0,
        type=float,
        help="Ratio between successive clade size classes (default: 2.0).",
    )
    parser.add_argument(
        "--weight-by",
        nargs="*",
        default=(),
        type=str,
        help="Columns summed into each node's weight (default: none).",
    )
    parser.add_argument(
        "--min-class-weight",
        default=0.0,
        type=float,
        help="Least weight for a class to avoid merging (default: 0.0).",
    )
    parser.add_argument(
        "--mark-as",
        default="clade_size_geom_class",
        type=str,
        help="Output column name (default: clade_size_geom_class).",
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()

    with log_context_duration(
        "phyloframe.legacy._alifestd_mark_clade_size_geom_classes_polars",
        logging.info,
    ):
        _run_dataframe_cli(
            base_parser=parser,
            output_dataframe_op=functools.partial(
                alifestd_mark_clade_size_geom_classes_polars,
                size_ratio=args.size_ratio,
                weight_by=args.weight_by,
                min_class_weight=args.min_class_weight,
                mark_as=args.mark_as,
            ),
        )
