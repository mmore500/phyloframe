# Instructions: Implement Remaining Polars Implementations for phyloframe.legacy

## Overview

The `phyloframe.legacy` module has ~129 pandas implementation files and ~49 polars implementation files. About 80 pandas functions still lack polars equivalents. Your job is to implement polars versions for the remaining functions, following the exact patterns established in the codebase.

## Key Requirements

1. **Reuse jitted fast paths**: Many pandas implementations have `@jit(nopython=True)` helper functions that take numpy arrays (e.g., `ancestor_ids`) as arguments. Import and reuse these in the polars implementation — do NOT duplicate the algorithm. If a pandas implementation doesn't have a suitable fast path function, refactor it to extract one (taking the `ancestor_id` array as argument), then use that shared function in the polars implementation.

2. **Raise NotImplementedError for unsupported inputs**: Polars implementations must raise `NotImplementedError` if IDs are not contiguous or data is not topologically sorted.

3. **Assume asexual phylogeny**: Polars implementations assume the phylogeny is asexual (single parent per node, `ancestor_id` column instead of `ancestor_list`).

4. **Add both Python API unit tests AND CLI smoke tests** following existing patterns.

5. **Share code between polars and pandas implementations** as reasonable (via the jitted fast paths).

6. **Register new CLI entrypoints** in `phyloframe/__main__.py`.

7. **Check polars implementations against pandas implementations** using `enforce_dtype_stability_polars` and `matches_pandas` tests.

8. **Run linters before pushing**: `isort==8.0.1`, `ruff==0.1.11`, `black==22.10.0` (pinned versions). Install with `uv pip install --system isort==8.0.1 ruff==0.1.11 black==22.10.0`.

9. **Run tests locally** for all new functions before pushing.

10. **Install dev dependencies** with: `uv pip install --system -r requirements-dev/py311/requirements-testing.txt`

11. **Push simple functions first** before working on complex ones and reusable fast paths.

---

## Functions Still Needing Polars Implementations

### Tier 1: Functions With Existing Jitted Fast Paths (Straightforward)

These pandas files already have `@jit(nopython=True)` fast path functions. The polars implementation simply needs to: validate inputs → extract numpy arrays → call the existing fast path → return with `with_columns`.

| # | Pandas File | Fast Path Function | Inputs | Output Column |
|---|---|---|---|---|
| 1 | `_alifestd_mark_left_child_asexual.py` | `_alifestd_mark_left_child_asexual_fast_path` | `ancestor_ids` | `left_child` |
| 2 | `_alifestd_mark_is_left_child_asexual.py` | (check if has fast path; may need `left_child` first) | `ancestor_ids` | `is_left_child` |
| 3 | `_alifestd_mark_is_right_child_asexual.py` | (check if has fast path; may need sibling info) | `ancestor_ids` | `is_right_child` |
| 4 | `_alifestd_mark_right_child_asexual.py` | (check for fast path) | `ancestor_ids` | `right_child` |
| 5 | `_alifestd_mark_sister_asexual.py` | (check for fast path) | `ancestor_ids` | `sister` |
| 6 | `_alifestd_mark_num_leaves_sibling_asexual.py` | (no jit — may need `num_leaves` + sibling info) | depends | `num_leaves_sibling` |
| 7 | `_alifestd_mark_num_preceding_leaves_asexual.py` | `_alifestd_mark_num_preceding_leaves_asexual_fast_path` | `ancestor_ids`, `num_leaves`, `is_right_child` | `num_preceding_leaves` |
| 8 | `_alifestd_mark_colless_index_asexual.py` | `alifestd_mark_colless_index_asexual_fast_path` | `ancestor_ids`, `num_leaves`, `left_child_ids` | `colless_index` |
| 9 | `_alifestd_mark_colless_index_corrected_asexual.py` | (check — may depend on colless_index) | depends | `colless_index_corrected` |
| 10 | `_alifestd_mark_colless_like_index_mdm_asexual.py` | `_colless_like_fast_path` | `ancestor_ids`, `diss_type` | `colless_like_index_mdm` |
| 11 | `_alifestd_mark_colless_like_index_sd_asexual.py` | (shares with mdm — different `diss_type`) | same | `colless_like_index_sd` |
| 12 | `_alifestd_mark_colless_like_index_var_asexual.py` | (shares with mdm — different `diss_type`) | same | `colless_like_index_var` |
| 13 | `_alifestd_mark_clade_faithpd_asexual.py` | `_alifestd_mark_clade_faithpd_asexual_fast_path` | `ancestor_ids`, `origin_time_deltas` | `clade_faithpd` |
| 14 | `_alifestd_mark_clade_duration_asexual.py` | (no jit — check pandas impl) | depends | `clade_duration` |
| 15 | `_alifestd_mark_ot_mrca_asexual.py` | (check pandas impl) | depends | `ot_mrca` |
| 16 | `_alifestd_mask_descendants_asexual.py` | `_alifestd_mask_descendants_asexual_fast_path` | `ancestor_ids`, `ancestor_mask` | mask column |
| 17 | `_alifestd_mask_monomorphic_clades_asexual.py` | `_alifestd_mask_monomorphic_clades_asexual_fast_path` | `ancestor_ids`, `trait_mask`, `trait_values` | mask column |
| 18 | `_alifestd_calc_clade_trait_count_asexual.py` | `_alifestd_calc_clade_trait_count_asexual_fast_path` | `ancestor_ids`, `trait_mask` | `clade_trait_count` |
| 19 | `_alifestd_sum_origin_time_deltas_asexual.py` | (check pandas impl) | depends | scalar result |
| 20 | `_alifestd_unfurl_traversal_semiorder_asexual.py` | `_alifestd_unfurl_traversal_semiorder_asexual_fast_path` | `ancestor_ids`, `num_descendants`, `leaf_ids` | traversal order |

### Tier 2: Functions That Need New Fast Paths Extracted

These pandas implementations do NOT have jitted fast paths yet. You should refactor the pandas file to extract a `@jit(nopython=True)` function, then reuse it in both pandas and polars.

| # | Pandas File | Notes |
|---|---|---|
| 1 | `_alifestd_mark_clade_duration_asexual.py` | May need origin_time and ancestor_origin_time |
| 2 | `_alifestd_mark_clade_duration_ratio_sister_asexual.py` | Depends on clade_duration and sister |
| 3 | `_alifestd_mark_clade_fblr_growth_children_asexual.py` | Complex growth metric |
| 4 | `_alifestd_mark_clade_fblr_growth_sister_asexual.py` | Complex growth metric |
| 5 | `_alifestd_mark_clade_leafcount_ratio_sister_asexual.py` | Depends on num_leaves and sister |
| 6 | `_alifestd_mark_clade_logistic_growth_children_asexual.py` | Has `_calc_boundaries` jit function |
| 7 | `_alifestd_mark_clade_logistic_growth_sister_asexual.py` | Complex growth metric |
| 8 | `_alifestd_mark_clade_nodecount_ratio_sister_asexual.py` | Depends on num_descendants and sister |
| 9 | `_alifestd_mark_clade_subtended_duration_asexual.py` | Depends on origin_time |
| 10 | `_alifestd_mark_clade_subtended_duration_ratio_sister_asexual.py` | Depends on subtended_duration and sister |
| 11 | `_alifestd_mark_num_leaves_sibling_asexual.py` | Depends on num_leaves and sibling |

### Tier 3: Structural/Transform Operations (More Complex)

These functions transform the structure of the dataframe (add/remove rows, reorder, etc.) rather than just adding columns. They are more complex to implement in polars.

| # | Pandas File | Notes |
|---|---|---|
| 1 | `_alifestd_topological_sort.py` | Reorder rows; already has jit variants |
| 2 | `_alifestd_chronological_sort.py` | Reorder rows by origin_time |
| 3 | `_alifestd_assign_root_ancestor_token.py` | Modify ancestor_list for roots |
| 4 | `_alifestd_convert_root_ancestor_token.py` | Change root ancestor token format |
| 5 | `_alifestd_splay_polytomies.py` | Add rows; has jit fast path |
| 6 | `_alifestd_join_roots.py` | Add a global root connecting all roots |
| 7 | `_alifestd_add_global_root.py` | Add a global root |
| 8 | `_alifestd_add_inner_knuckles_asexual.py` | Add intermediate nodes |
| 9 | `_alifestd_add_inner_leaves.py` | Add leaf nodes |
| 10 | `_alifestd_add_inner_niblings_asexual.py` | Add nodes |
| 11 | `_alifestd_collapse_trunk_asexual.py` | Remove trunk nodes |
| 12 | `_alifestd_delete_unifurcating_roots_asexual.py` | Remove rows |
| 13 | `_alifestd_coarsen_taxa_asexual.py` | Has jit fast path; restructures |
| 14 | `_alifestd_coarsen_mask.py` | Filter/mask based |
| 15 | `_alifestd_reroot_at_id_asexual.py` | Reroot tree |
| 16 | `_alifestd_coerce_chronological_consistency.py` | Fix inconsistencies |

### Tier 4: Query/Predicate Functions (Simple, Pure Polars)

These return scalars or booleans and can often be implemented with pure polars expressions (no numpy/jit needed).

| # | Pandas File | Returns | Notes |
|---|---|---|---|
| 1 | `_alifestd_has_compact_ids.py` | `bool` | Check if ids are 0..n-1 (any order) |
| 2 | `_alifestd_has_increasing_ids.py` | `bool` | Check if ids are strictly increasing |
| 3 | `_alifestd_is_asexual.py` | `bool` | Check if all ancestor_list entries have ≤1 parent |
| 4 | `_alifestd_is_sexual.py` | `bool` | Opposite of is_asexual |
| 5 | `_alifestd_is_chronologically_ordered.py` | `bool` | Check origin_time ordering |
| 6 | `_alifestd_is_chronologically_sorted.py` | `bool` | Check chronological sort |
| 7 | `_alifestd_is_strictly_bifurcating_asexual.py` | `bool` | Check all internal nodes have exactly 2 children |
| 8 | `_alifestd_is_working_format_asexual.py` | `bool` | Check working format properties |
| 9 | `_alifestd_count_children_of_asexual.py` | `int` | Count children of a specific node |
| 10 | `_alifestd_count_inner_nodes.py` | `int` | Already has polars! (skip) |
| 11 | `_alifestd_count_leaf_nodes.py` | `int` | Already has polars! (skip) |
| 12 | `_alifestd_count_polytomies.py` | `int` | Already has polars! (skip) |
| 13 | `_alifestd_count_root_nodes.py` | `int` | Already has polars! (skip) |
| 14 | `_alifestd_count_unifurcating_roots_asexual.py` | `int` | Count unifurcating roots |
| 15 | `_alifestd_count_unifurcations.py` | `int` | Already has polars! (skip) |
| 16 | `_alifestd_find_leaf_ids.py` | `np.ndarray` | Already has polars! (skip) |
| 17 | `_alifestd_find_chronological_inconsistency.py` | `Optional[int]` | Has jit variants |
| 18 | `_alifestd_calc_polytomic_index.py` | scalar | Polytomy metric |

### Tier 5: Complex/Specialized Operations (Lowest Priority)

| # | Pandas File | Notes |
|---|---|---|
| 1 | `_alifestd_as_newick_asexual.py` | Newick serialization (polars version exists for different variant) |
| 2 | `_alifestd_aggregate_phylogenies.py` | Combine multiple phylogenies |
| 3 | `_alifestd_calc_mrca_id_matrix_asexual.py` | O(n²) MRCA matrix |
| 4 | `_alifestd_calc_triplet_distance_asexual.py` | Triplet distance metric |
| 5 | `_alifestd_categorize_triplet_asexual.py` | Triplet classification |
| 6 | `_alifestd_estimate_triplet_distance_asexual.py` | Approximate triplet distance |
| 7 | `_alifestd_sample_triplet_comparisons_asexual.py` | Sampling for triplet distance |
| 8 | `_alifestd_test_leaves_isomorphic_asexual.py` | Tree isomorphism test |
| 9 | `_alifestd_screen_trait_defined_clades_fisher_asexual.py` | Statistical test |
| 10 | `_alifestd_screen_trait_defined_clades_fitch_asexual.py` | Fitch parsimony |
| 11 | `_alifestd_screen_trait_defined_clades_naive_asexual.py` | Naive clade screening |
| 12 | `_alifestd_downsample_tips_asexual.py` | Tip downsampling |
| 13 | `_alifestd_find_mrca_id_asexual.py` | Find MRCA for set of taxa |
| 14 | `_alifestd_find_pair_mrca_id_asexual.py` | Already has polars! (skip) |
| 15 | `_alifestd_unfurl_lineage_asexual.py` | Trace lineage |
| 16 | `_alifestd_unfurl_traversal_inorder_asexual.py` | Inorder traversal |
| 17 | `_alifestd_unfurl_traversal_postorder_asexual.py` | Postorder traversal |
| 18 | `_alifestd_make_balanced_bifurcating.py` | Generate test trees |
| 19 | `_alifestd_make_comb.py` | Generate test trees |
| 20 | `_alifestd_make_empty.py` | Generate empty frame |
| 21 | `_alifestd_make_ancestor_id_col.py` | Parse ancestor_list → ancestor_id |
| 22 | `_alifestd_parse_ancestor_id.py` | Parse single ancestor_id |
| 23 | `_alifestd_parse_ancestor_ids.py` | Parse ancestor_ids |
| 24 | `_alifestd_validate.py` | Full validation suite |
| 25 | `_alifestd_prune_extinct_lineages_asexual.py` | Already has polars! (skip) |
| 26 | `_alifestd_to_working_format.py` | Full pipeline to convert to working format |
| 27 | `_alifestd_check_topological_sensitivity.py` | Already has polars! (skip) |

---

## Exact Patterns to Follow

### Pattern A: Polars Implementation File (with jitted fast path)

Reference: `phyloframe/legacy/_alifestd_mark_num_descendants_polars.py`

```python
import argparse
import logging
import os

import joinem
from joinem._dataframe_cli import _add_parser_base, _run_dataframe_cli
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
# Import the shared jitted fast path from the pandas implementation file:
from ._alifestd_mark_XXXX_asexual import (
    _alifestd_mark_XXXX_asexual_fast_path,
)
from ._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)


def alifestd_mark_XXXX_polars(
    phylogeny_df: pl.DataFrame,
) -> pl.DataFrame:
    """Add column `XXXX`."""

    logging.info(
        "- alifestd_mark_XXXX_polars: " "adding ancestor_id col...",
    )
    phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)

    if phylogeny_df.lazy().limit(1).collect().is_empty():
        return phylogeny_df.with_columns(
            XXXX=pl.lit(0).cast(pl.Int64),  # appropriate default
        )

    logging.info(
        "- alifestd_mark_XXXX_polars: "
        "checking contiguous ids...",
    )
    if not alifestd_has_contiguous_ids_polars(phylogeny_df):
        raise NotImplementedError(
            "non-contiguous ids not yet supported",
        )

    logging.info(
        "- alifestd_mark_XXXX_polars: "
        "checking topological sort...",
    )
    if not alifestd_is_topologically_sorted_polars(phylogeny_df):
        raise NotImplementedError(
            "topologically unsorted rows not yet supported",
        )

    logging.info(
        "- alifestd_mark_XXXX_polars: "
        "extracting ancestor ids...",
    )
    ancestor_ids = (
        phylogeny_df.lazy()
        .select("ancestor_id")
        .collect()
        .to_series()
        .to_numpy()
    )

    logging.info(
        "- alifestd_mark_XXXX_polars: "
        "computing XXXX...",
    )
    result = _alifestd_mark_XXXX_asexual_fast_path(ancestor_ids)

    return phylogeny_df.with_columns(
        XXXX=result,
    )


_raw_description = f"""\
{os.path.basename(__file__)} | \
(phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Add column `XXXX`.

Data is assumed to be in alife standard format.

Additional Notes
================
- Use `--eager-read` if modifying data file inplace.

- This CLI entrypoint is experimental and may be subject to change.

See Also
========
phyloframe.legacy._alifestd_mark_XXXX_asexual :
    CLI entrypoint for Pandas-based implementation.
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
            "phyloframe.legacy._alifestd_mark_XXXX_polars"
        ),
        dfcli_version=get_phyloframe_version(),
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()

    try:
        with log_context_duration(
            "phyloframe.legacy._alifestd_mark_XXXX_polars",
            logging.info,
        ):
            _run_dataframe_cli(
                base_parser=parser,
                output_dataframe_op=(alifestd_mark_XXXX_polars),
            )
    except NotImplementedError as e:
        logging.error(
            "- polars op not yet implemented, use pandas op CLI instead",
        )
        raise e
```

### Pattern B: Unit Test File

Reference: `tests/test_phyloframe/test_legacy/test_alifestd_mark_num_descendants_polars.py`

```python
import os
import typing

import pandas as pd
import polars as pl
import pytest

from phyloframe.legacy import (
    alifestd_mark_XXXX_asexual,  # pandas version
    alifestd_to_working_format,
)
from phyloframe.legacy._alifestd_mark_XXXX_polars import (
    alifestd_mark_XXXX_polars as alifestd_mark_XXXX_polars_,
)

from ._impl import enforce_dtype_stability_polars

alifestd_mark_XXXX_polars = enforce_dtype_stability_polars(
    alifestd_mark_XXXX_polars_
)

assets_path = os.path.join(os.path.dirname(__file__), "assets")


@pytest.mark.parametrize(
    "phylogeny_df",
    [
        alifestd_to_working_format(
            pd.read_csv(
                f"{assets_path}/example-standard-toy-asexual-phylogeny.csv"
            )
        ),
        alifestd_to_working_format(
            pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv")
        ),
        alifestd_to_working_format(
            pd.read_csv(f"{assets_path}/nk_lexicaseselection.csv")
        ),
        alifestd_to_working_format(
            pd.read_csv(f"{assets_path}/nk_tournamentselection.csv")
        ),
    ],
)
@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_XXXX_polars_matches_pandas(
    phylogeny_df: pd.DataFrame, apply: typing.Callable
):
    """Verify polars result matches pandas result."""
    result_pd = alifestd_mark_XXXX_asexual(phylogeny_df, mutate=False)

    df_pl = apply(pl.from_pandas(phylogeny_df))
    result_pl = alifestd_mark_XXXX_polars(df_pl).lazy().collect()

    assert result_pd["XXXX"].tolist() == (
        result_pl["XXXX"].to_list()
    )


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_XXXX_polars_simple_chain(
    apply: typing.Callable,
):
    """Test a simple chain: 0 -> 1 -> 2."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_id": [0, 0, 1],
            }
        ),
    )

    result = alifestd_mark_XXXX_polars(df_pl).lazy().collect()

    assert result["XXXX"].to_list() == [expected0, expected1, expected2]


# ... additional tests for simple_tree, single_node, empty, etc.


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_XXXX_polars_empty(apply: typing.Callable):
    """Empty dataframe gets XXXX column."""
    df_pl = apply(
        pl.DataFrame(
            {"id": [], "ancestor_id": []},
            schema={"id": pl.Int64, "ancestor_id": pl.Int64},
        ),
    )

    result = alifestd_mark_XXXX_polars(df_pl).lazy().collect()

    assert "XXXX" in result.columns
    assert result.is_empty()


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_XXXX_polars_non_contiguous_ids(
    apply: typing.Callable,
):
    """Verify NotImplementedError for non-contiguous ids."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 2, 5],
                "ancestor_id": [0, 0, 2],
            }
        ),
    )
    with pytest.raises(NotImplementedError):
        alifestd_mark_XXXX_polars(df_pl).lazy().collect()


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_XXXX_polars_unsorted(
    apply: typing.Callable,
):
    """Verify NotImplementedError for topologically unsorted data."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_id": [0, 2, 0],
            }
        ),
    )
    with pytest.raises(NotImplementedError):
        alifestd_mark_XXXX_polars(df_pl).lazy().collect()
```

### Pattern C: CLI Smoke Test File

Reference: `tests/test_phyloframe/test_legacy/test_alifestd_mark_num_descendants_polars_cli.py`

```python
import os
import pathlib
import subprocess

import pandas as pd

assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def test_alifestd_mark_XXXX_polars_cli_help():
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_mark_XXXX_polars",
            "--help",
        ],
        check=True,
    )


def test_alifestd_mark_XXXX_polars_cli_version():
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_mark_XXXX_polars",
            "--version",
        ],
        check=True,
    )


def test_alifestd_mark_XXXX_polars_cli_csv():
    output_file = "/tmp/phyloframe_alifestd_mark_XXXX_polars.csv"  # nosec B108
    pathlib.Path(output_file).unlink(missing_ok=True)
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_mark_XXXX_polars",
            "--eager-write",
            output_file,
        ],
        check=True,
        input=f"{assets}/trunktestphylo.csv".encode(),
    )
    assert os.path.exists(output_file)
    result_df = pd.read_csv(output_file)
    assert len(result_df) > 0
    assert "XXXX" in result_df.columns


def test_alifestd_mark_XXXX_polars_cli_parquet():
    output_file = "/tmp/phyloframe_alifestd_mark_XXXX_polars.pqt"  # nosec B108
    pathlib.Path(output_file).unlink(missing_ok=True)
    subprocess.run(  # nosec B603
        [
            "python3",
            "-m",
            "phyloframe.legacy._alifestd_mark_XXXX_polars",
            "--eager-write",
            output_file,
        ],
        check=True,
        input=f"{assets}/trunktestphylo.csv".encode(),
    )
    assert os.path.exists(output_file)
    result_df = pd.read_parquet(output_file)
    assert len(result_df) > 0
    assert "XXXX" in result_df.columns
```

---

## Test Asset Selection for CLI Tests

- **`trunktestphylo.csv`**: Use for functions that only need `id` and `ancestor_id` columns (no `origin_time` needed). Already has contiguous IDs and topological sort.
- **`nk_ecoeaselection-workingformat.csv`**: Use for functions that need `origin_time` column. Already in working format with contiguous IDs.
- Do NOT use raw `nk_ecoeaselection.csv` for CLI tests — it uses `ancestor_list` format and doesn't have contiguous IDs when read via polars.

---

## CLI Entrypoint Registration

Add each new polars CLI to `phyloframe/__main__.py` in **alphabetical order** among the existing entries. Format:

```python
$ python3 -m phyloframe.legacy._alifestd_mark_XXXX_polars
```

---

## Workflow

1. **Install dependencies**: `uv pip install --system -r requirements-dev/py311/requirements-testing.txt`
2. **For each function** (start with simplest):
   a. Read the pandas implementation to understand the algorithm
   b. Check if it already has a jitted fast path; if not, refactor one out
   c. Create the polars implementation file in `phyloframe/legacy/`
   d. Create the unit test file in `tests/test_phyloframe/test_legacy/`
   e. Create the CLI smoke test file (if the function has CLI)
   f. Add CLI entrypoint to `phyloframe/__main__.py`
3. **Run linters**: `python3 -m isort . && python3 -m ruff check --fix . && python3 -m black .`
4. **Run tests**: `python3 -m pytest tests/test_phyloframe/test_legacy/test_alifestd_mark_XXXX_polars.py tests/test_phyloframe/test_legacy/test_alifestd_mark_XXXX_polars_cli.py -x -v`
5. **Push simple functions first**, then complex ones.

---

## Important Implementation Notes

- **LazyFrame support**: All polars functions must accept both `pl.DataFrame` and `pl.LazyFrame`. Use `.lazy()` and `.collect()` as needed.
- **Empty frame handling**: Always check for empty frames early and return with an appropriately typed dummy column.
- **`try_add_ancestor_id_col`**: Call `alifestd_try_add_ancestor_id_col_polars(phylogeny_df)` at the start if the function needs `ancestor_id`. This converts `ancestor_list` to `ancestor_id` if needed.
- **Column casting**: Use `pl.lit(0).cast(pl.Int64)` for integer defaults, `pl.lit(0.0).cast(pl.Float64)` for float defaults.
- **`# nosec B603` and `# nosec B108`**: Add these bandit suppression comments on subprocess calls and `/tmp/` paths in test files.
- **Some functions depend on other polars functions**: e.g., `mark_sackin_index` needs `num_leaves`. If a dependency doesn't have a polars version, implement it first. Use existing polars helpers like `alifestd_mark_num_leaves_polars`, `alifestd_mark_roots_polars`, etc.
- **Functions returning scalars** (like `count_*`, `has_*`, `is_*`) may not need CLI entrypoints or the full validation pattern. Check the pandas version's interface.
- **When comparing floats in matches_pandas tests**: Use `pytest.approx()` or compare with tolerance for floating-point columns.
- **The `enforce_dtype_stability_polars` wrapper** in tests ensures that `id` and `ancestor_id` column dtypes are preserved through the function. Always wrap the polars function with it in tests.

---

## Already Implemented (Do Not Reimplement)

These already have polars versions — skip them:

- `assign_contiguous_ids` → `_polars` exists
- `as_newick` → `_polars` exists (different variant from `_asexual`)
- `calc_mrca_id_vector_asexual` → `_polars` exists
- `check_topological_sensitivity` → `_polars` exists
- `coarsen_dilate` → `_polars` exists
- `collapse_unifurcations` → `_polars` exists
- `count_inner_nodes` → `_polars` exists
- `count_leaf_nodes` → `_polars` exists
- `count_polytomies` → `_polars` exists
- `count_root_nodes` → `_polars` exists
- `count_unifurcating_roots` → `_polars` exists
- `count_unifurcations` → `_polars` exists
- `delete_trunk_asexual` → `_polars` exists
- `downsample_tips_canopy` → `_polars` exists
- `downsample_tips_clade` → `_polars` exists
- `downsample_tips_lineage` → `_polars` exists
- `downsample_tips_lineage_stratified` → `_polars` exists
- `downsample_tips` → `_polars` exists
- `drop_topological_sensitivity` → `_polars` exists
- `find_leaf_ids` → `_polars` exists
- `find_pair_mrca_id` → `_polars` exists
- `find_root_ids` → `_polars` exists
- `from_newick` → `_polars` exists
- `has_contiguous_ids` → `_polars` exists
- `has_multiple_roots` → `_polars` exists
- `is_topologically_sorted` → `_polars` exists
- `make_ancestor_list_col` → `_polars` exists
- `mark_ancestor_origin_time` → `_polars` exists
- `mark_first_child_id` → `_polars` exists
- `mark_leaves` → `_polars` exists
- `mark_max_descendant_origin_time` → `_polars` exists
- `mark_next_sibling_id` → `_polars` exists
- `mark_node_depth` → `_polars` exists
- `mark_num_children` → `_polars` exists
- `mark_num_descendants` → `_polars` exists
- `mark_num_leaves` → `_polars` exists
- `mark_oldest_root` → `_polars` exists
- `mark_origin_time_delta` → `_polars` exists
- `mark_prev_sibling_id` → `_polars` exists
- `mark_root_id` → `_polars` exists
- `mark_roots` → `_polars` exists
- `mark_sackin_index` → `_polars` exists
- `mask_descendants` → `_polars` exists
- `prefix_roots` → `_polars` exists
- `prune_extinct_lineages` → `_polars` exists
- `topological_sensitivity_warned` → `_polars` exists
- `try_add_ancestor_id_col` → `_polars` exists
- `try_add_ancestor_list_col` → `_polars` exists
- `warn_topological_sensitivity` → `_polars` exists
