# phyloframe

[
![PyPI](https://img.shields.io/pypi/v/phyloframe.svg)
](https://pypi.python.org/pypi/phyloframe)
[
![CI](https://github.com/mmore500/phyloframe/actions/workflows/ci.yaml/badge.svg)
](https://github.com/mmore500/phyloframe/actions)
[
![codecov](https://codecov.io/gh/mmore500/phyloframe/graph/badge.svg?token=YyQ34WbkqT)
](https://codecov.io/gh/mmore500/phyloframe)
[
![GitHub stars](https://img.shields.io/github/stars/mmore500/phyloframe.svg?style=round-square&logo=github&label=Stars&logoColor=white)](https://github.com/mmore500/phyloframe)
[![DOI](https://zenodo.org/badge/1170914158.svg)](https://zenodo.org/doi/10.5281/zenodo.18842673)

Dataframe-based tools for working with phylogenetic trees.

- Free software: MIT license
- Documentation: <https://phyloframe.readthedocs.io>
- Repository: <https://github.com/mmore500/phyloframe>

## Why a DataFrame-based Tree Representation?

DataFrames are scripting-friendly and end-user extensible, enabling a composable, interoperable ecosystem for phylogenetic analysis.
The R ecosystem's success with the `ape` data structure demonstrates the value of accessible tree representations --- phyloframe pushes this idea further with a fully tabular format.

**Fast and highly portable load/save.**
Use `pandas.read_csv`, `polars.read_parquet`, R's `read.table`, etc. --- libraries transparently fetch from URLs, cloud providers (S3, Google Cloud, etc.).
Serialized and in-memory representations are unified.

**Benefit from modern tabular data formats.**
Granular deserialization of selected columns (e.g., Parquet), transparent compression configuration (e.g., Parquet), columnar compression for efficient storage, categorical strings, and explicit column typing.
Options exist for both binary and text formats.

**Benefit from modern high-performance dataframe tooling.**
Memory-efficient representation, larger-than-memory streaming operations (e.g., Polars), distributed computing operations (e.g., Dask), multithreaded operations (e.g., Polars), vectorized operations (e.g., NumPy), and just-in-time compilation (e.g., Numba).

**Rich interoperative ecosystem.**
Multi-language interoperation (e.g., zero-copy interop between R and Python via `reticulate` and Arrow; zero-copy Polars DataFrames shared between Rust and Python).
Multi-library interoperation (e.g., highly-optimized or zero-copy interoperation between Polars and Pandas; Python dataframe protocol).

## Install

```bash
python3 -m pip install "phyloframe[jit]==0.6.1"
```

The `[jit]` extra installs [Numba](https://numba.pydata.org/) for just-in-time compilation, providing native-level performance for many operations.
Omit it if you do not need JIT acceleration.

A containerized release of `phyloframe` is available via [ghcr.io](https://ghcr.io/mmore500/phyloframe)

```bash
singularity exec docker://ghcr.io/mmore500/phyloframe:v0.6.1 python3 -m phyloframe --help
```

## Quickstart

### Import Convention

```python3
from phyloframe import legacy as pfl
```

The `legacy` module provides all current phyloframe operations.
As phyloframe evolves, `legacy` will continue to be maintained for backward compatibility while new API designs are developed.

### The Alife Standard Format

Phyloframe represents phylogenies as DataFrames in the **alife standard format**.
Each row is an organism (or taxon), with columns:

| Column | Type | Description |
|--------|------|-------------|
| `id` | int | Unique, non-negative identifier for this organism |
| `ancestor_list` | str | JSON-encoded list of ancestor IDs, e.g., `"[0]"` |
| `ancestor_id` | int | *(optional)* Direct ancestor ID for asexual phylogenies |

Root nodes (organisms with no ancestor) use `"[None]"` or `"[none]"` as their `ancestor_list`.
For asexual phylogenies, the `ancestor_id` column provides an optimized integer representation.

```python3
import pandas as pd

# A simple three-node chain: root -> internal -> leaf
phylogeny_df = pd.DataFrame({
    "id": [0, 1, 2],
    "ancestor_list": ["[None]", "[0]", "[1]"],
})
```

### Creating Phylogenies

```python3
# Empty phylogeny
empty_df = pfl.alifestd_make_empty()

# Balanced bifurcating tree (depth=3 gives 7 nodes, 4 leaves)
balanced_df = pfl.alifestd_make_balanced_bifurcating(depth=3)

# Comb (caterpillar) tree with 10 leaves
comb_df = pfl.alifestd_make_comb(n_leaves=10)

# Parse from Newick format
newick_df = pfl.alifestd_from_newick("((A,B),(C,D));")
```

### Working Format

Many operations run fastest on data in **working format**: topologically sorted, with contiguous IDs and an `ancestor_id` column.
Convert once, then chain operations:

```python3
df = pfl.alifestd_from_newick("((A:1,B:2):3,(C:4,D:5):6);")
df = pfl.alifestd_to_working_format(df)
```

### Marking Properties

"Mark" functions add computed columns to a phylogeny DataFrame:

```python3
df = pfl.alifestd_from_newick("((A,B),(C,D));")
df = pfl.alifestd_to_working_format(df)

# Which nodes are leaves?
df = pfl.alifestd_mark_leaves(df)

# How deep is each node from the root?
df = pfl.alifestd_mark_node_depth_asexual(df)

# How many descendants does each node have?
df = pfl.alifestd_mark_num_descendants_asexual(df)

print(df[["id", "ancestor_id", "is_leaf", "node_depth", "num_descendants"]])
```

Column names can be customized via the `mark_as` parameter:

```python3
df = pfl.alifestd_mark_leaves(df, mark_as="is_tip")
```

### Composed Example: Downsampling with Combined Masks

A common workflow is to select tips using multiple criteria, combine them with boolean OR, and then prune.
Here, we keep tips that are *either* in the canopy (most recent) *or* along a focal lineage, then remove extinct lineages:

```python3
import pandas as pd
from phyloframe import legacy as pfl

# Create a tree with origin times
df = pfl.alifestd_from_newick(
    "((A:1,B:2):3,(C:4,(D:5,E:6):7):8);",
)
df = pfl.alifestd_to_working_format(df)

# Assign origin times for demonstration
df["origin_time"] = range(len(df))

# Mark canopy tips (most recent leaves by origin_time)
df = pfl.alifestd_mark_sample_tips_canopy_asexual(
    df, n_sample=2, mark_as="keep_canopy",
)

# Mark lineage tips (closest to a focal lineage)
df = pfl.alifestd_mark_sample_tips_lineage_asexual(
    df, n_sample=2, mark_as="keep_lineage",
)

# Combine masks with boolean OR
df["extant"] = df["keep_canopy"] | df["keep_lineage"]

# Prune non-extant lineages
pruned_df = pfl.alifestd_prune_extinct_lineages_asexual(df)
```

### Counting and Querying

```python3
df = pfl.alifestd_from_newick("((A,B),(C,D));")

n_leaves = pfl.alifestd_count_leaf_nodes(df)
n_inner = pfl.alifestd_count_inner_nodes(df)
is_asexual = pfl.alifestd_is_asexual(df)
is_sorted = pfl.alifestd_is_topologically_sorted(df)
```

### Tree Transformations

```python3
df = pfl.alifestd_from_newick("((A,B),(C,D));")
df = pfl.alifestd_to_working_format(df)

# Collapse single-child nodes
df = pfl.alifestd_collapse_unifurcations(df)

# Add a synthetic global root above all existing roots
df = pfl.alifestd_add_global_root(df)

# Export back to Newick
newick_str = pfl.alifestd_as_newick_asexual(df)
```

### Mutation Semantics

By default, operations return a new DataFrame without modifying the input.
Set `mutate=True` to allow in-place modification for better performance in pipelines.
Even with `mutate=True`, always use the return value:

```python3
# Safe default: input is not modified
result = pfl.alifestd_mark_leaves(df)

# Faster in pipelines: allows reuse of input memory
result = pfl.alifestd_mark_leaves(df, mutate=True)
```

### Piping Operations

Chain multiple operations using `alifestd_pipe_unary_ops`:

```python3
result = pfl.alifestd_pipe_unary_ops(
    df,
    pfl.alifestd_collapse_unifurcations,
    pfl.alifestd_mark_leaves,
    pfl.alifestd_mark_node_depth_asexual,
)
```

### Pandas vs. Polars

Many phyloframe operations have Polars implementations, available via
``_polars`` suffixed functions:

```python3
import polars as pl
from phyloframe import legacy as pfl

# Use _polars suffixed functions for Polars DataFrames
df_polars = pfl.alifestd_from_newick_polars("((A,B),(C,D));")
df_polars = pfl.alifestd_mark_leaves_polars(df_polars)
```

Polars implementations can be more performant but are more restrictive: they exclusively support asexual phylogenies and require topological sortedness and contiguous IDs.

### Command-line Interface

Every phyloframe operation is available as a CLI command, reading and writing CSV or Parquet:

```bash
# Apply an operation to a file
python3 -m phyloframe.legacy._alifestd_mark_leaves output.csv < input.csv

# Pipe operations together
python3 -m phyloframe.legacy._alifestd_collapse_unifurcations /dev/stdout \
    < input.csv \
  | python3 -m phyloframe.legacy._alifestd_mark_leaves output.csv

# Use the pipe_unary_ops CLI for multi-step pipelines
python3 -m phyloframe.legacy._alifestd_pipe_unary_ops \
    --op "pfl.alifestd_collapse_unifurcations" \
    --op "pfl.alifestd_mark_leaves" \
    output.csv < input.csv
```

Prefer Polars CLI entrypoints (modules ending in `_polars`) to avoid Pandas-to-Polars conversion overhead when working with Parquet data:

```bash
python3 -m phyloframe.legacy._alifestd_mark_leaves_polars output.pqt < input.pqt
```

List all available commands with:

```bash
python3 -m phyloframe --help
```

### JIT Compilation for Custom Code

Use phyloframe's `jit` utility to write native-speed custom operations with Numba:

```python3
import numpy as np
from phyloframe._auxlib import jit
from phyloframe import legacy as pfl

@jit(nopython=True, cache=False)
def count_deep_nodes(ancestor_ids: np.ndarray, threshold: int) -> int:
    """Count nodes deeper than `threshold` in an asexual phylogeny.

    Requires contiguous IDs and topological sorting.
    """
    n = len(ancestor_ids)
    depths = np.zeros(n, dtype=np.int64)
    count = 0
    for i in range(n):
        if ancestor_ids[i] != i:  # not a root
            depths[i] = depths[ancestor_ids[i]] + 1
        if depths[i] > threshold:
            count += 1
    return count

# Use with phyloframe data
df = pfl.alifestd_make_balanced_bifurcating(depth=10)
df = pfl.alifestd_to_working_format(df)
n_deep = count_deep_nodes(df["ancestor_id"].values, threshold=5)
```

The `jit` decorator gracefully falls back to pure Python if Numba is not installed, and automatically disables compilation during coverage runs for better source visibility.

## Citing

If phyloframe contributes to a scholarly work, please cite it as

> Matthew Andres Moreno. (2026). mmore500/phyloframe. Zenodo. https://doi.org/10.5281/zenodo.18842674

```bibtex
@software{moreno2026phyloframe,
  author = {Matthew Andres Moreno},
  title = {mmore500/phyloframe},
  month = mar,
  year = 2026,
  publisher = {Zenodo},
  doi = {10.5281/zenodo.18842674},
  url = {https://doi.org/10.5281/zenodo.18842674}
}
```

And don't forget to leave a [star on GitHub](https://github.com/mmore500/phyloframe/stargazers)!
