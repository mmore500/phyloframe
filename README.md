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

The R ecosystem's success with the `ape` data structure demonstrates the value of edge matrix tree representations --- phyloframe pushes this idea further with a fully tabular format hosted within DataFrame objects (e.g., `pd.DataFrame`, `pl.LazyFrame`, `pl.DataFrame`, etc.).

DataFrames are scripting-friendly and end-user extensible, enabling a composable, interoperable, high-performance ecosystem for phylogenetic analysis --- in applications to our work, scalable to billion-tip phylogenies.

**Fast and highly portable load/save.**
Use `pandas.read_csv`, `polars.read_parquet`, R's `read.table`, etc. --- libraries transparently fetch from URLs, cloud providers (S3, Google Cloud, etc.).

**Benefit from modern tabular data formats.**
Granular deserialization of selected columns (e.g., Parquet), transparent compression configuration (e.g., Parquet), columnar compression for efficient storage, categorical strings, and explicit column typing.

**Benefit from modern high-performance dataframe tooling.**
Memory-efficient representation, larger-than-memory streaming operations (e.g., Polars), distributed computing operations (e.g., Dask), multithreaded operations (e.g., Polars), vectorized operations (e.g., NumPy), and just-in-time compilation (e.g., Numba).

**Rich interoperative ecosystem.**
Multi-language interoperation (e.g., possible future support for zero-copy interop between R and Python via `reticulate` and Arrow; possible future support for zero-copy Polars DataFrames shared between Rust and Python).
Multi-library interoperation (e.g., highly-optimized or zero-copy interoperation between Polars and Pandas; Python dataframe protocol).

## Install

```bash
python3 -m pip install "phyloframe[jit]==0.6.1"
```

The `[jit]` extra installs [Numba](https://numba.pydata.org/) for just-in-time compilation, providing native-level performance for many operations.
Jit dependency is strongly recommended.

A containerized release of `phyloframe` is available via [ghcr.io](https://ghcr.io/mmore500/phyloframe)

```bash
singularity exec docker://ghcr.io/mmore500/phyloframe:v0.6.1 python3 -m phyloframe --help
```

## Quickstart

Phyloframe represents phylogenies as DataFrames in the [**alife standard format**](https://alife-data-standards.github.io/alife-data-standards/phylogeny.html).

```python3
from phyloframe import legacy as pfl

# Parse a Newick tree (already in working format)
df = pfl.alifestd_from_newick("((A:1,B:2):3,(C:4,D:5):6);")

# Mark properties and transform using df.pipe() (pandas syntactic sugar)
df = (
    df.pipe(pfl.alifestd_mark_leaves)
    .pipe(pfl.alifestd_mark_node_depth_asexual)
    .pipe(pfl.alifestd_collapse_unifurcations)
)

print("leaf count:", pfl.alifestd_count_leaf_nodes(df))
print(df[["id", "ancestor_id", "is_leaf", "node_depth"]].head())
```

The `legacy` module (`from phyloframe import legacy`) provides all current phyloframe operations.
The legacy API is stable and will continue to be maintained for backward compatibility.
A redesigned API will accompany phyloframe v1.0.0.

For the full quickstart covering tree representation semantics, tree creation, tree computations, tree transforms, Polars, CLI, JIT compilation, and more, see the [documentation quickstart](https://phyloframe.readthedocs.io/en/latest/quickstart.html).

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
