---
title: 'phyloframe: Dataframe-based tools for working with phylogenetic trees'
tags:
  - Python
  - phylogenetics
  - dataframes
  - evolutionary biology
authors:
  - name: Matthew Andres Moreno
    orcid: 0000-0003-4726-4479
    affiliation: "1, 2, 3, 4"
  - name: Luis Zaman
    orcid: 0000-0001-6838-7385
    affiliation: "1, 2, 4"
  - name: Emily Dolson
    orcid: 0000-0001-8616-4898
    affiliation: "5, 6, 7"
affiliations:
  - name: Department of Ecology and Evolutionary Biology
    index: 1
  - name: Center for the Study of Complex Systems
    index: 2
  - name: Michigan Institute for Data and AI in Society
    index: 3
  - name: University of Michigan, Ann Arbor, United States
    index: 4
  - name: Department of Computer Science and Engineering
    index: 5
  - name: Program in Ecology, Evolution, and Behavior
    index: 6
  - name: Michigan State University, East Lansing, United States
    index: 7
date: 14 March 2026
bibliography: paper.bib
---

# Summary

`phyloframe` is a Python library for phylogenetic tree analysis that represents trees as dataframes rather than linked node objects.
Trees are stored in the Artificial Life Data Standard (alifestd) tabular format [@lalejini2019alife], where each row corresponds to a node and columns record node identifiers, ancestor relationships, branch lengths, and arbitrary user-defined attributes.
Built on pandas [@mckinney2010pandas] and Polars [@vink2024polars], `phyloframe` provides over 190 operations spanning tree construction, traversal, manipulation, topology metrics, and format conversion.
Performance-critical routines are JIT-compiled via Numba [@lam2015numba], and the library is available on PyPI as `phyloframe[jit]`.

# Statement of Need

Phylogenetic trees are a central data structure in evolutionary biology, epidemiology, and artificial life research.
Most existing Python tree libraries represent phylogenies as pointer-linked node objects, which limits interoperability with the broader data science ecosystem and incurs per-node Python object overhead at large scale.
Researchers working with large-scale digital evolution simulations or genomic surveillance pipelines increasingly need to analyze trees with hundreds of thousands to millions of tips alongside rich per-node metadata.
`phyloframe` addresses this gap by storing phylogenies as column-oriented dataframes, enabling vectorized computation, zero-copy interoperation with analytics tools, and natural integration of metadata as additional columns.

# Features

**Tree construction and I/O.**
`phyloframe` reads and writes Newick strings via `alifestd_from_newick_polars` and `alifestd_as_newick_polars`, with a JIT-compiled parser that handles trees with millions of tips.
Synthetic trees (balanced bifurcating, comb) can be generated for testing and benchmarking.

**Traversals.**
Preorder, postorder, inorder, levelorder, and semiorder traversals are available, returning node index arrays suitable for downstream vectorized operations.

**Topology metrics.**
The library computes Colless imbalance, Sackin index, Faith's phylogenetic diversity, mean pairwise distance, and polytomic index, among others.

**Tree manipulation.**
Operations include collapsing unifurcations, pruning extinct lineages, downsampling tips by lineage or canopy strategies, rerooting, ladderizing, and coarsening.

**All-pairs MRCA.**
A most-recent common ancestor matrix can be computed for all tip pairs, enabling downstream comparative analyses.

**Dual dataframe backends.**
All I/O and traversal operations support both Polars and pandas, with automatic delegation to the faster Polars path when available.

**Command-line interface.**
Each public function is also accessible as a CLI command, facilitating use in shell pipelines.

# Projects Using the Software

`phyloframe` originated from phylogeny-tracking components developed for the `hstrat` hereditary stratigraphy library [@moreno2022hstrat], which enables phylogenetic inference over distributed digital evolution populations.
The alifestd operations now in `phyloframe` provide the core tree analysis and manipulation layer used by `hstrat` and downstream digital evolution experiments.

# Related Software

Several established Python libraries provide phylogenetic tree functionality.
TreeSwift [@moshiri2020treeswift] is a high-performance library using compact linked-node structures, designed to scale to very large trees.
DendroPy [@sukumaran2010dendropy] offers a comprehensive object-oriented framework for phylogenetic simulation and analysis.
Biopython [@cock2009biopython] includes a `Bio.Phylo` module supporting multiple tree formats with a focus on interoperability.
ETE [@huertacepas2016ete3] combines tree analysis with visualization capabilities.
CompactTree [@moshiri2025compacttree] achieves minimal memory footprint through a header-only C++ implementation with a Python wrapper.
scikit-bio [@aton2026scikitbio] provides a broad bioinformatics toolkit including tree data structures and ecological diversity analyses.

All of these libraries represent trees as pointer-linked node objects.
`phyloframe` takes a fundamentally different approach by storing trees as column-oriented dataframes.
This design enables direct integration with pandas and Polars analytics workflows, vectorized computation over node attributes, and natural attachment of per-node metadata as additional columns without custom data structures.
The dataframe representation also facilitates JIT compilation of inner loops via Numba, yielding competitive traversal and parsing performance while retaining Python-level expressiveness.

# Acknowledgements

Computational resources were provided by the MSU Institute for Cyber-Enabled Research and from PSC Neocortex via the ByteBoost training program.
Thank you also to Mathias Jacquelin and Leighton Wilson at Cerebras Systems.
This material is based upon work supported by the Eric and Wendy Schmidt AI in Science Postdoctoral Fellowship, a Schmidt Sciences program.
This material is based upon work supported by the U.S. Department of Energy, Office of Science, Office of Advanced Scientific Computing Research (ASCR), under Award Number DE-SC0025634.
This report was prepared as an account of work sponsored by an agency of the United States Government.
Neither the United States Government nor any agency thereof, nor any of their employees, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness of any information, apparatus, product, or process disclosed, or represents that its use would not infringe privately owned rights.
Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof.
The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.

# References

<div id="refs"></div>

\pagebreak
\appendix
