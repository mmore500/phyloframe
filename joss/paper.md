---
title: 'PhyloFrame: A DataFrame-based Library for Fast, Flexible Phylogenetic Computation'
tags:
  - Python
  - phylogenetics
  - dataframe
  - evolutionary biology
authors:
  - name: Matthew Andres Moreno
    orcid: 0000-0003-4726-4479
    affiliation: "1, 2, 3, 4"
affiliations:
  - name: Department of Ecology and Evolutionary Biology
    index: 1
  - name: Center for the Study of Complex Systems
    index: 2
  - name: Michigan Institute for Data and AI in Society
    index: 3
  - name: University of Michigan, Ann Arbor, United States
    index: 4
date: 14 May 2026
bibliography: paper.bib
---

# Summary

PhyloFrame is a Python library for phylogenetic computation targeting the gap between specialist, compiler-optimized operations and flexible, script-based workflows --- with emphasis on fast, memory-efficient operations for very large tree sizes (e.g., $\geq$ 300,000 taxa).

PhyloFrame is built around a DataFrame-based tree representation, where each row corresponds to a node and columns record ancestor relationships, branch lengths, and arbitrary user-defined attributes.
Crucial for scalability, such array-backed storage allows library features and end-user code alike to seamlessly harness Just-in-Time (JIT) compilation (e.g., Numba) and vectorized execution (e.g., NumPy, Polars).
At large tree sizes, performance generally matches or exceeds libraries backed by native code --- notably, achieving up to $10\times$ faster topological-order traversals, up to $10\times$ faster Newick reads, as well as nearly $2\times$ faster Newick writes.

DataFrame-based representation affords additional conveniences, notably including:

- succinct bulk operations (e.g., NumPy);
- powerful queries and transformations (e.g., Polars expressions, Pandas indexing, SQL-style joins and merges);
- compatibility with modern tabular data formats that are compression-friendly, type-aware, nullable, and highly portable (e.g., Parquet); and
- broad interoperation with table-oriented data science tools (e.g., Seaborn, Plotly, Vega-Altair, tidyverse, Excel).

Current library features include tree input/output, synthetic tree generation, taxon-based queries, tree traversals, tree metrics, tree manipulation, tree downsampling, and tree comparison.
Most functionality supports both Pandas and Polars DataFrames and is available through package- and CLI-based interfaces.

# Statement of Need

In addition to the ever-growing influx of high-throughput sequence data [@stephens2015big], recent years have seen the advent of powerful biotechnologies for cell-lineage tracing [@mckenna2016whole;@nguyenba2019high], ultra-high-throughput workflows capable of estimating ancestry among hundreds of millions of taxa [@konno2022deep], and ultra-scale simulations generating billion-taxa lineage histories [@singhvi2025scalable].
These emerging sources of phylogeny data (evolutionary trees) offer unprecedented visibility into developmental and eco-evolutionary processes [@faith1992conservation;@STAMATAKIS2005phylogenetics;@frenchHostPhylogenyShapes2023;@kim2006discovery;@lewinsohnStatedependentEvolutionaryModels2023a;@lenski2003evolutionary;@nozoe2017inferring;@chan2019molecular], but dataset scales far exceed the capabilities of traditional tools supporting bioinformatics workflow development [@moreno2024dendropy;@cock2009biopython;@huertacepas2016ete3].

This unmet need has prompted development of several performance-first libraries for phylogenetic computing in Python [@moshiri2020treeswift;@moshiri2025compacttree;@ryneches2018suchtree].
These libraries have greatly improved the state of the art, and provide powerful compiler-optimized implementations of standard phylogeny-based calculations [@moshiri2025compacttree;@ryneches2018suchtree].
End-user code requiring custom iterative operations, however, typically remains as slower, interpreted Python bytecode.
Achieving full compiler optimization for custom operations thus requires writing extensions (or working fully) in less familiar systems-level languages (e.g., C/C++).

As such, PhyloFrame seeks to complement this existing software landscape by filling the gap between fully-optimized native code and custom end-user operations.
To this end, PhyloFrame sacrifices a typical object-oriented interface in favor of a DataFrame-based representation of tree nodes.
A tree with $n$ nodes, in this framework, boils down to a set of length-$n$ arrays --- each storing a particular node-attribute (e.g., ancestor index, origin time, taxon name, etc.).
Importantly, such array-backing is friendly to vectorized bulk operations (e.g., NumPy [@harris2020array]) and JIT compilation (e.g., Numba [@lam2015numba]) --- ultimately, yielding competitive performance while retaining Python-level expressiveness and productivity.

![Benchmark comparison. Left: throughput (tips processed per second) for each operation across tree sizes. Right: memory efficiency (tips stored per byte of RSS) across tree sizes. Higher is better in both panels.\label{fig:benchmark}](benchmark-panel.png)

\autoref{fig:benchmark} compares throughput and memory efficiency across these libraries on balanced binary trees with up to 30 million tips.[^bench]
For most benchmarked operations, PhyloFrame matches or exceeds the throughput and efficiency of native-backed libraries (e.g., CompactTree, SuchTree) beyond tree sizes of around 300,000 tips.
For workloads involving large quantities of smaller trees, performance benefit can be achieved by consolidating data as a "forest" within a single DataFrame.

Surprisingly, at very large tree sizes (e.g., $\geq$ 1 million tips) PhyloFrame substantially accelerates throughput beyond the state of the art.
For traversal operations, this likely stems from JIT-based capability to materialize iteration within a fully-native context.
Topological order traversals are particularly efficient, as they simply correspond to a sequential scan over array memory.
Newick parsing, on the other hand, likely benefits from streamlined per-array (as opposed to per-node) memory allocation, while Newick generation leverages the Polars engine to accelerate string-building.

An important performance trade-off not captured in these benchmarks is tree manipulation.
While DataFrame-based representation does support mutable tree reconfiguration after construction, unlike allocated node-and-pointer representations, one-off node creation and deletion is not guaranteed $\mathcal{O}(1)$.

[^bench]:
Timings were conducted on GitHub action `ubuntu-24.04` runners (4-core x86/16GB memory circa May 2026), with cross-library comparisons restricted to common job instances for parity.
Raw data is archived at <https://osf.io/knw8x> [@foster2017open].
Benchmark design follows [@moshiri2025compacttree].

Beyond high-performance end-user extensibility, DataFrame-based representation affords several notable conveniences that complement the existing phylogenetic computing landscape --- briefly described in Appendix A.

# Features

PhyloFrame supports the following operations on both Pandas and Polars DataFrames [@mckinney2010pandas;@vink2024polars]:

- __tree input/output__: Newick and ALife Data Standard formats [@lalejini2019alife];
- __tree synthesis__: structured (e.g., comb, balanced, star) and random (e.g., edge-adding, node-adding);
- __taxon-based queries__: pairwise and all-pairs MRCA/patristic distance;
- __tree traversals__: preorder, postorder, inorder, levelorder, semiorder, topological;
- __tree metrics__: Colless imbalance [@colless1982phylogenetics], Sackin index [@sackin1972good], Faith's phylogenetic diversity [@faith1992conservation];
- __tree manipulation__: collapsing unifurcations, rerooting, ladderizing;
- __tree downsampling__: lineage sampling, tip sampling, clade sampling, custom pruning;
- __tree comparison__: triplet/quartet distance [@sand2014tqdist;@estabrook1985comparison;@dobson1975comparing] and topological isomorphism; and
- __tree visualization__: integration with Matplotlib-based iplotx [@zanini2025iplotx] and interactive Taxonium web application [@sanderson2022taxonium].[^fork]

[^fork]: via an experimental fork at <https://mmore500.github.io/taxonium>.

A quickstart guide and full API listing are included in the [PhyloFrame documentation](https://phyloframe.readthedocs.io).
PhyloFrame is installable from the Python Package Index (PyPI) via `pip` (e.g., `python3 -m pip install phyloframe[jit]`).

# Demo: End-user JIT Compilation and Tidy Plotting

Example code applies a pipeline pattern to (1) generate a 100-tip tree, (2) apply JIT-compiled end-user code to simulate trait inheritance, (3) mark attributes `node_depth` and `is_leaf`, and (4) visualize using Seaborn and Matplotlib.

```python
import numpy as np; import polars as pl; import seaborn as sns
from phyloframe import _auxlib as pfa
from phyloframe import legacy as pfl

@pfa.jit(cache=False, nopython=True)  # JIT compile via Numba, if available
def simulate_trait(ancestor_ids: np.ndarray) -> np.ndarray:
    trait = np.zeros(ancestor_ids.size, dtype=float)
    for id_, anc_id in enumerate(ancestor_ids):
        if id_ == anc_id: continue  # exclude root
        trait[id_] = trait[anc_id] + np.random.normal()
    return trait

def plot_trait(data: pl.DataFrame) -> None:
    selectors = dict(x="node_depth", y="trait", hue="is_leaf")
    style = dict(legend=False, palette=["gray", "steelblue"], s=15)
    ax = sns.scatterplot(data, **selectors, **style)

    depth, ancestor, trait = data[["node_depth", "ancestor_id", "trait"]]
    segments = [[depth, depth[ancestor]], [trait, trait[ancestor]]]
    ax.plot(*segments, color="#CCCCCCCC", zorder=-2)  # link parent/child

pfl.alifestd_make_edge_split_polars(n_leaves=100, seed=42,  # random tree
    ).pipe(pfl.alifestd_topological_sort_polars,  # parents before children
    ).pipe(pfl.alifestd_assign_contiguous_ids_polars,  # reassign ids
    ).pipe(pfl.alifestd_mark_node_depth_polars,  # add node_depth col
    ).pipe(pfl.alifestd_mark_leaves_polars,  # add is_leaf col
    ).with_columns(trait=pl.col("ancestor_id").map_batches(  # add trait col
        lambda x: simulate_trait(x.to_numpy()), return_dtype=float,
)).pipe(plot_trait)  # draw tree
```

![Trait simulation visualization \label{fig:demo2}](demo2.png)


# Related Software

A rich ensemble of established libraries support Python-based phylogenetic computing.

- DendroPy [@moreno2024dendropy] offers a comprehensive object-oriented framework for phylogenetic simulation and analysis.
- Biopython [@cock2009biopython] includes a `Bio.Phylo` module supporting multiple tree formats with a focus on interoperability.
- ETE [@huertacepas2016ete3] combines tree analysis with visualization capabilities.
- scikit-bio [@aton2026scikitbio] provides a broad bioinformatics toolkit, including tree manipulation, reconstruction, and phylogenetic diversity metrics.
- tskit [@wong2024args;@kelleher2016msprime] uses a specialized data structure to compactly store millions of related gene trees.
- CompactTree [@moshiri2025compacttree] achieves minimal memory footprint through a header-only C++ implementation with a Python wrapper.
- TreeSwift [@moshiri2020treeswift] is a performance-oriented pure-Python library using a compact linked-node data structure, designed to scale to very large trees.
- SuchTree [@ryneches2018suchtree] uses a Cython-based array data structure, focusing on fast pairwise distance queries and co-phylogenetic analyses; operations release the Python GIL (Global Interpreter Lock) to allow multithread parallelism.
- ToyTree [@eaton2019toytree] is an object-oriented library, providing integrated visualization functionality.
- PhyloTrack [@dolson2024phylotrack] uses a node-and-pointer data structure to record lineage histories in forward-time agent-based models, with support for on-the-fly extinction pruning and metric calculations.

In the Julia [@bezanson2017julia] ecosystem, PhyloNetworks [@solislemus2017phylonetworks] emphasizes support for generalized phylogenetic networks incorporating reticulation events.
The R-based ecosystem has largely coalesced around ape's edge matrix tree representation [@paradis2018ape].
Other work has applied graph databases to manage large-scale phylogeny data [@loureno2024phylodb].

Except for commonality in name, the PhyloFrame library presented here is unrelated to recent machine learning methodology developed to counteract ancestral bias in precision medicine [@smith2025equitable].

# Projects Using the Software

PhyloFrame originated from phylogeny-tracking components developed for the hstrat library [@moreno2022hstrat], which enables phylogenetic inference over distributed digital evolution populations.
The alifestd operations now in PhyloFrame provide the core tree analysis and manipulation layer used by `hstrat` and downstream digital evolution experiments.
Underlying software (earlier, a submodule of hstrat) has contributed substantially to several projects [@moreno2025ecology;@singhvi2025scalable;@moreno2025testing;@moreno2024trackable;@moreno2022hereditary].

# Development Roadmap

Much future work remains in development of the PhyloFrame library.

Feature-level improvements (e.g., tree metrics, manipulations, etc.) are planned on an as-needed basis, with requests welcome via the project [issue tracker](https://github.com/mmore500/phyloframe).

Looking further ahead, a redesigned API is planned to accompany PhyloFrame's v1 release.
In anticipation of this release, all current PhyloFrame operations are packaged in `phyloframe.legacy`.
This API is stable and will continue to be maintained for backward compatibility.

Identified design and development priorities include:

- better-standardized naming schemes for library-function-generated columns,
- automatic cleanup of columns invalidated by tree manipulation,
- caching contiguous id and topological order safety checks (currently, bypassable via environment variable),
- first-class support for unrooted trees,
- first-class support for reticulated ancestry graphs,
- automatic repair of canonical representation invariants (i.e., contiguous ids, topological order),
- support for GPU-based computations via CuPy and RAPIDS' Pandas integration [@cupy_learningsys2017;@rapids],
- wheel-based distribution of pre-compiled Numba artifacts,
- high-level visualization utilities leveraging Seaborn and iplotx [@waskom2021seaborn;@zanini2025iplotx], and
- greater API symmetry between Pandas and Polars functionality.

Beyond Python, DataFrame-based phylogenetic computing may prove useful in other language ecosystems, such as Julia and R.
Julia appears especially well-suited, given its mature, tightly-integrated DataFrame stack [@bouchetvalat2023dataframesjl] and first-class language support for JIT compilation [@bezanson2017julia].

# Acknowledgements

This material is based upon work supported by the Eric and Wendy Schmidt AI in Science Postdoctoral Fellowship, a Schmidt Sciences program.
This material is based upon work supported by the U.S. Department of Energy, Office of Science, Office of Advanced Scientific Computing Research (ASCR), under Award Number DE-SC0025634.
This report was prepared as an account of work sponsored by an agency of the United States Government.
Neither the United States Government nor any agency thereof, nor any of their employees, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness of any information, apparatus, product, or process disclosed, or represents that its use would not infringe privately owned rights.
Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof.
The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.

**AI Use Declaration:**
During the preparation of this work, AI tools were used to assemble manuscript boilerplate and draft benchmarking scripts.
Closely supervised agentic software development is used for refactoring code, drafting documentation, and scoped library feature development.
Such contributions are tracked via commit co-authorship.
Tools used include Claude Code, Google Gemini, and OpenAI ChatGPT.

# References

<div id="refs"></div>

\pagebreak
\appendix

# Appendix A: Why a DataFrame-based Tree Representation?

DataFrames are scripting-friendly and end-user extensible, enabling a composable, interoperable, high-performance ecosystem for much of modern data science [@wu2020is].
PhyloFrame seeks to go further with a fully tabular format hosted within DataFrame objects (e.g., pandas.DataFrame, pl.LazyFrame, pl.DataFrame, etc.).

**Fast and highly portable load/save.**
Use `pandas.read_csv`, `polars.read_parquet`, R's `read.table`, etc.
Many implementations can automatically fetch from URLs, cloud providers (e.g., AWS S3, Google Cloud, etc.), and online repositories [@foster2017open;@singh2011figshare].
Contiguous allocations allow fast tree deserialization (e.g., Newick) and tree generation.

**Benefit from modern tabular file formats.**
Granular deserialization of selected columns, columnar compression for efficient storage, categorical strings, and explicit column typing with first-class null representation (e.g., Parquet [@vohra2016parquet]).
Data layout optimization for fast load/save (e.g., Feather [@wickham2016feather]).

**Benefit from modern high-performance dataframe tooling.**
Memory-efficient representation, larger-than-memory streaming operations (e.g., Polars), distributed computing operations (e.g., Dask [@rocklin2015dask]), multithreaded operations (e.g., Polars), vectorized operations (e.g., NumPy), and just-in-time compilation (e.g., Numba).

**Benefit from rich, expressive DataFrame functionality.**
Leverage powerful querying and transformation APIs (e.g., Polars expressions, Pandas indexing) for flexible filtering, bulk calculations, grouped aggregations, join/merge operations, and chained transformations directly over tree data without manual loops.

**Cache-friendly, memory-efficient, flexible data structure.**
Data occupies contiguous arrays, expediting tree creation and topological order traversals (e.g., parents before children or vice versa).
Base memory footprint is lightweight (e.g., as little as 32 bits per node), but can be dynamically augmented to expedite traversals and calculations (e.g., "linked list" of children via columns for first child/next sibling indices).

**Interoperation.**
Multi-language interoperation (e.g., possible future support for zero-copy interop between R and Python via reticulate and Arrow [@reticulate;@arrow], possible future support for zero-copy Polars DataFrames shared between Rust and Python).
Multi-library interoperation (e.g., highly-optimized or zero-copy interoperation between Polars and Pandas; Python dataframe protocol [@meurer2023python]).
Interoperation with broader Python DataFrame ecosystem [@vallat2018pingouin;@vanderplas2018altair;@waskom2021seaborn;@skrub] and ALife data standard tooling [@lalejini2019alife].

# Appendix B: Tree Manipulation Pipeline Demo

Example code shows sequential tree transforms applied using a pipeline pattern.
Such complex tree manipulations, including custom operations, can often be performed succinctly without loops or recursion.

```python
import numpy as np; from pandas import DataFrame
from phyloframe import legacy as pfl

df_raw: DataFrame = pfl.alifestd_from_newick("(((r:1)c:2,(x:2,y:1)e:1.5)s:2)a;")
df_res: DataFrame = df_raw.drop(columns=["branch_length", "origin_time_delta"],
    ).pipe(pfl.alifestd_reroot_at_id_asexual,  # reroot at node "r"
      new_root_id=df_raw.query("taxon_label == 'r'")["id"].item(),
    ).pipe(lambda df: df.assign(branch_length=np.where(  # flip rerooted lengths
      df_raw.loc[df["id"], "ancestor_id"] != df["ancestor_id"],  # where flipped
      df_raw.loc[df["ancestor_id"], "branch_length"],  # take ancestor's value
      df_raw.loc[df["id"], "branch_length"],  # ...otherwise keep own
    ))).pipe(pfl.alifestd_to_working_format,  # reassign id values
    ).pipe(pfl.alifestd_mark_lineage_cumsum_asexual,  # accumulate branch length
      mark_as="origin_time", values="branch_length",  # ...to mark origin time
    ).pipe(pfl.alifestd_sort_children_asexual, criterion="taxon_label",
    ).pipe(pfl.alifestd_to_working_format,  # reassign id values
    ).pipe(pfl.alifestd_ultrametricize, method="extend")  # align tip times
```

![Before and after tree plots, rendered via integration with iplotx. \label{fig:demo1}](demo1.png)

# Appendix C: Compound Downsampling via Command-Line Interface Demo

Integration with the Joinem DataFrame CLI engine provides direct access to most library functionality [@moreno2024joinem].
GitHub Container Registry releases allow zero-install execution via Singularity [@kurtzer2017singularity].
The following example shows a compound downsample operation combining canopy and lineage masks.

```bash
ls -1 "input.csv" `# input path, in alife standard format` \
| singularity exec docker://ghcr.io/mmore500/phyloframe:v0.9.0 `# container` \
  python3 -m phyloframe.legacy._alifestd_pipe_unary_ops `# apply ops in turn` \
  --op "lambda df: pfl.alifestd_mark_sample_tips_canopy_asexual(" \
                         "df, n_sample=5, mark_as='keep_canopy')" \
  --op "lambda df: pfl.alifestd_mark_sample_tips_lineage_asexual(" \
                         "df, n_sample=5, mark_as='keep_lineage')" \
  --op "lambda df: df.assign(extant=df['keep_canopy'] | df['keep_lineage'])" \
  --op "pfl.alifestd_prune_extinct_lineages_asexual" \
  "output.parquet"  # output path
```

A full CLI listing is available via `python3 -m phyloframe --help`.
