import argparse
import functools
import logging
import os

import joinem
from joinem._dataframe_cli import _add_parser_base, _run_dataframe_cli
import numpy as np
import pandas as pd

from .._auxlib._begin_prod_logging import begin_prod_logging
from .._auxlib._delegate_polars_implementation import (
    delegate_polars_implementation,
)
from .._auxlib._format_cli_description import format_cli_description
from .._auxlib._get_phyloframe_version import get_phyloframe_version
from .._auxlib._log_context_duration import log_context_duration
from ._alifestd_mark_colless_index_asexual import (
    alifestd_mark_colless_index_asexual,
)
from ._alifestd_mark_num_leaves_asexual import alifestd_mark_num_leaves_asexual


def alifestd_mark_colless_index_corrected_asexual(
    phylogeny_df: pd.DataFrame,
    mutate: bool = False,
    *,
    mark_as: str = "colless_index_corrected",
) -> pd.DataFrame:
    """Add column `colless_index_corrected` with the corrected Colless
    index for each subtree.

    The output column name can be changed via the ``mark_as`` parameter.

    The corrected Colless index IC(T) normalizes the Colless index by
    tree size. For a subtree with n leaves:

        IC(T) = 0                          if n <= 2
        IC(T) = 2 * C(T) / ((n-1)*(n-2))  if n > 2

    where C(T) is the Colless index of the subtree.

    This function delegates to `alifestd_mark_colless_index_asexual` to
    compute the Colless index, and therefore requires strictly
    bifurcating trees.

    Raises ValueError if the tree is not strictly bifurcating. For
    trees with polytomies, consider computing the generalized Colless
    index and normalizing separately.

    A topological sort will be applied if `phylogeny_df` is not
    topologically sorted. Dataframe reindexing (e.g., df.index) may
    be applied.

    Input dataframe is not mutated by this operation unless `mutate`
    set True. If mutate set True, operation does not occur in place;
    still use return value to get transformed phylogeny dataframe.

    Parameters
    ----------
    phylogeny_df : pd.DataFrame
        Alife standard DataFrame containing the phylogenetic
        relationships.

    mutate : bool, optional
        If True, modify the input DataFrame in place. Default is
        False.

    Returns
    -------
    pd.DataFrame
        Phylogeny DataFrame with an additional column
        "colless_index_corrected" containing the corrected Colless
        imbalance index for the subtree rooted at each node.

    Raises
    ------
    ValueError
        If phylogeny_df is not strictly bifurcating.

    See Also
    --------
    alifestd_mark_colless_index_asexual :
        Unnormalized Colless index for strictly bifurcating trees.
    alifestd_mark_colless_like_index_mdm_asexual :
        Colless-like index (MDM) that supports polytomies.
    alifestd_mark_colless_like_index_var_asexual :
        Colless-like index (variance) that supports polytomies.
    alifestd_mark_colless_like_index_sd_asexual :
        Colless-like index (std dev) that supports polytomies.
    """
    if not mutate:
        phylogeny_df = phylogeny_df.copy()

    if phylogeny_df.empty:
        phylogeny_df[mark_as] = pd.Series(
            dtype=float,
        )
        return phylogeny_df

    if "colless_index" not in phylogeny_df.columns:
        phylogeny_df = alifestd_mark_colless_index_asexual(
            phylogeny_df, mutate=True
        )

    if "num_leaves" not in phylogeny_df.columns:
        phylogeny_df = alifestd_mark_num_leaves_asexual(
            phylogeny_df, mutate=True
        )

    n = phylogeny_df["num_leaves"].to_numpy(dtype=np.float64)
    c = phylogeny_df["colless_index"].to_numpy(dtype=np.float64)

    result = np.zeros_like(n)
    mask = n > 2
    result[mask] = 2.0 * c[mask] / ((n[mask] - 1.0) * (n[mask] - 2.0))
    phylogeny_df[mark_as] = result

    return phylogeny_df


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Add column `colless_index_corrected` with the corrected Colless index for each subtree.

Data is assumed to be in alife standard format.

Additional Notes
================
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
        dfcli_module="phyloframe.legacy._alifestd_mark_colless_index_corrected_asexual",
        dfcli_version=get_phyloframe_version(),
    )
    parser.add_argument(
        "--mark-as",
        default="colless_index_corrected",
        type=str,
        help="output column name (default: colless_index_corrected)",
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()
    with log_context_duration(
        "phyloframe.legacy._alifestd_mark_colless_index_corrected_asexual",
        logging.info,
    ):
        _run_dataframe_cli(
            base_parser=parser,
            output_dataframe_op=delegate_polars_implementation()(
                functools.partial(
                    alifestd_mark_colless_index_corrected_asexual,
                    mark_as=args.mark_as,
                ),
            ),
        )
