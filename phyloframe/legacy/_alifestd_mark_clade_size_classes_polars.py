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
from .._auxlib._format_cli_description import format_cli_description
from .._auxlib._get_phyloframe_version import get_phyloframe_version
from .._auxlib._log_context_duration import log_context_duration
from ._alifestd_has_contiguous_ids_polars import (
    alifestd_has_contiguous_ids_polars,
)
from ._alifestd_is_topologically_sorted_polars import (
    alifestd_is_topologically_sorted_polars,
)
from ._alifestd_mark_num_leaves_asexual import (
    _alifestd_mark_num_leaves_asexual_fast_path,
)
from ._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)


def _alifestd_mark_clade_size_classes_fast_path(
    *,
    clade_sizes: np.ndarray,
    weights: np.ndarray,
    size_ratio: float,
    min_class_weight: float,
) -> np.ndarray:
    """Implementation detail for `alifestd_mark_clade_size_classes_polars`.

    Returns each node's clade size class, given its clade size (number of
    leaves) and weight.
    """
    clade_sizes = np.asarray(clade_sizes)
    # number occupied classes consecutively, skipping empty ones
    __, classes = np.unique(
        calc_geometric_bins(clade_sizes, ratio=size_ratio),
        return_inverse=True,
    )
    num_classes = classes.max(initial=0) + 1
    class_weights = np.bincount(classes, weights, minlength=num_classes)
    # the first class above class 0 that is too light, and every class of
    # larger clades, join the class below it
    (light,) = np.nonzero(class_weights[1:] < min_class_weight)
    if len(light):
        cap = max(light[0], 1)
        if cap < num_classes - 1:
            logging.info(
                "- alifestd_mark_clade_size_classes_polars: clade size "
                f"classes above {cap} weigh less than {min_class_weight}, so "
                f"join class {cap}",
            )
        classes = np.minimum(classes, cap)
        num_classes = cap + 1
        class_weights = np.bincount(classes, weights, minlength=num_classes)

    for class_ in range(num_classes):
        sizes = clade_sizes[classes == class_]
        logging.info(
            "- alifestd_mark_clade_size_classes_polars: class "
            f"{class_} has clade sizes {sizes.min()}-{sizes.max()}, "
            f"{len(sizes)} nodes, and weight {class_weights[class_]:g}",
        )
    return classes


def alifestd_mark_clade_size_classes_polars(
    phylogeny_df: pl.DataFrame,
    *,
    size_ratio: float = 2.0,
    weight_by: typing.Sequence[str] = (),
    min_class_weight: float = 0.0,
    mark_as: str = "clade_size_class",
) -> pl.DataFrame:
    """Add column `clade_size_class`, each node's class of clade size (number
    of leaves below it, itself included if a leaf) on a log scale.

    A node with `n` leaves falls in bin ceil(log(n) / log(`size_ratio`)),
    so the default ratio of 2 gives bins of 1, 2, 3-4, 5-8, 9-16, ...
    leaves. Classes number occupied bins consecutively from 0, so leaves are
    always class 0.

    These classes can partition nodes for calibration in
    `alifestd_tgs_allmdc_estimate_polars`, since neutral traits' gain rates
    per unit branch length may differ between small and large clades
    (e.g., between leaves and inner nodes). Calibrating within a class
    needs enough neutral gains there, so weight each node by its neutral
    traits' transitions (`weight_by`) and set `min_class_weight`.

    If `min_class_weight` is positive, this function merges classes from
    the largest clades down: with each node's weight the sum of its
    `weight_by` columns, the first class above class 0 weighing less than
    `min_class_weight`, and every class of larger clades, joins the class
    below it. If even class 1 is too light, all classes but 0 form one
    class.

    The output column name can be changed via the ``mark_as`` parameter.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame or polars.LazyFrame
        The phylogeny as a dataframe in alife standard format.

        Must represent an asexual phylogeny with contiguous,
        topologically sorted ids.
    size_ratio : float, default 2.0
        Ratio between successive classes' clade size bounds; must exceed 1.
        Larger ratios give fewer, larger classes.
    weight_by : Sequence[str], default ()
        Names of numeric columns summed into each node's weight, e.g.,
        neutral traits' transitions.
    min_class_weight : float, default 0.0
        Least weight a class above class 0 may hold before merging; 0
        disables merging.
    mark_as : str, default "clade_size_class"
        Output column name.

    Returns
    -------
    polars.DataFrame
        The phylogeny with an added integer column of clade size classes.

    Raises
    ------
    NotImplementedError
        If `phylogeny_df` is not asexual, or its ids are non-contiguous or
        not topologically sorted, or a `weight_by` column has null values.
    ValueError
        If `size_ratio` does not exceed 1, if `min_class_weight` is negative,
        or if a `weight_by` column is missing.

    See Also
    --------
    alifestd_mark_num_leaves_polars :
        Marks clade sizes.
    alifestd_mark_joint_quantile_bins_polars :
        Bins nodes within these classes, e.g., for calibration.
    alifestd_tgs_allmdc_estimate_polars :
        Calibrates within these classes, e.g., with `weight_by` its
        `neutral_traits`.
    """
    if not size_ratio > 1:
        raise ValueError(f"size_ratio must exceed 1, got {size_ratio}")
    if not min_class_weight >= 0:
        raise ValueError(
            f"min_class_weight must be nonnegative, got {min_class_weight}",
        )
    if isinstance(weight_by, str):
        raise TypeError(
            "weight_by must be a sequence of column names, not a single "
            f"string; did you mean [{weight_by!r}]?",
        )

    phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)
    schema_names = phylogeny_df.lazy().collect_schema().names()
    if "ancestor_id" not in schema_names:
        raise NotImplementedError(
            "alifestd_mark_clade_size_classes_polars only supports asexual "
            "phylogenies.",
        )
    for column in weight_by:
        if column not in schema_names:
            raise ValueError(f"phylogeny_df has no column {column}")

    selected = (
        phylogeny_df.lazy()
        .select(
            "id",
            "ancestor_id",
            pl.sum_horizontal(*weight_by, pl.lit(0.0)).alias("weight"),
            pl.any_horizontal(
                *(pl.col(c).is_null() for c in weight_by), pl.lit(False)
            ).alias("has_null"),
        )
        .collect()
    )
    if selected.is_empty():
        return phylogeny_df.with_columns(
            pl.Series(name=mark_as, values=[], dtype=pl.Int64),
        )
    if not alifestd_has_contiguous_ids_polars(selected):
        raise NotImplementedError("non-contiguous ids not supported")
    if not alifestd_is_topologically_sorted_polars(selected):
        raise NotImplementedError(
            "non-topologically-sorted data not supported",
        )
    if selected["has_null"].any():
        raise NotImplementedError("null weight_by values not supported")

    return phylogeny_df.with_columns(
        pl.Series(
            name=mark_as,
            values=_alifestd_mark_clade_size_classes_fast_path(
                clade_sizes=_alifestd_mark_num_leaves_asexual_fast_path(
                    selected["ancestor_id"].to_numpy()
                ),
                weights=selected["weight"].to_numpy(),
                size_ratio=size_ratio,
                min_class_weight=min_class_weight,
            ),
            dtype=pl.Int64,
        ),
    )


_raw_description = f"""\
{os.path.basename(__file__)} | \
(phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Add column `clade_size_class`, each node's class of clade size (number of
leaves below it) on a log scale, optionally merging classes of large clades
that carry too little weight (e.g., neutral trait gains).

Data is assumed to be in alife standard format.
Only supports asexual phylogenies.

Additional Notes
================
- Requires contiguous, topologically sorted ids.

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
        dfcli_module="phyloframe.legacy._alifestd_mark_clade_size_classes_polars",
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
        help=(
            "Least weight a class above class 0 may hold before merging; 0 "
            "disables merging (default: 0.0)."
        ),
    )
    parser.add_argument(
        "--mark-as",
        default="clade_size_class",
        type=str,
        help="Output column name (default: clade_size_class).",
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()

    try:
        with log_context_duration(
            "phyloframe.legacy._alifestd_mark_clade_size_classes_polars",
            logging.info,
        ):
            _run_dataframe_cli(
                base_parser=parser,
                output_dataframe_op=functools.partial(
                    alifestd_mark_clade_size_classes_polars,
                    size_ratio=args.size_ratio,
                    weight_by=args.weight_by,
                    min_class_weight=args.min_class_weight,
                    mark_as=args.mark_as,
                ),
            )
    except NotImplementedError as e:
        logging.error(
            "- polars op not yet implemented, use pandas op instead",
        )
        raise e
