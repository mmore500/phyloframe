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

## Install

`python3 -m pip install phyloframe`

A containerized release of `phyloframe` is available via [ghcr.io](https://ghcr.io/mmore500/phyloframe)

```bash
singularity exec docker://ghcr.io/mmore500/phyloframe:v0.2.0 python3 -m phyloframe --help
```

## Usage

```python3
from phyloframe import legacy as pfl

df = pfl.alifestd_make_empty()
print(df)
```

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
