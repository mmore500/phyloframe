==========================
Tree Visualization (Legacy)
==========================

.. note::

   This documentation covers the **legacy** API (from phyloframe import legacy).
   The legacy API is stable and will continue to be maintained for backward
   compatibility.
   A redesigned API will accompany phyloframe v1.0.0.


This guide covers visualizing phylogenetic trees from phyloframe DataFrames using `iplotx <https://iplotx.readthedocs.io/>`_.

IplotX Integration
==================

Phyloframe provides ``TreeDataProvider`` shims that allow iplotx to directly consume alife-standard phylogeny DataFrames.
Three shim classes are available, corresponding to different input types:

- ``AlifestdIplotxShimPandas`` — for pandas DataFrames
- ``AlifestdIplotxShimPolars`` — for polars DataFrames
- ``AlifestdIplotxShimNumpy`` — for raw numpy ``ancestor_id`` arrays

Requirements
------------

The input data must be **asexual** (single parent per node), have **contiguous IDs** (``id == row index``), and be **topologically sorted** (ancestors before descendants).
Use ``alifestd_to_working_format`` to prepare data that does not already meet these requirements.

Basic Usage
-----------

.. skip: next

.. code-block:: python

   from phyloframe import legacy as pfl
   import plotly.graph_objects as go

   # Create a tree
   df = pfl.alifestd_from_newick("((A:1,B:2):3,(C:4,D:5):6);")
   df = pfl.alifestd_to_working_format(df)

   # Build the iplotx provider
   shim = pfl.AlifestdIplotxShimPandas(df)

   # Compute layout and draw with plotly
   tree_data = shim(layout="horizontal")
   fig = go.Figure()
   # ... add traces from tree_data["vertex_df"], tree_data["edge_df"] ...

The ``shim(layout=...)`` call returns a ``TreeData`` dict containing ``vertex_df``, ``edge_df``, and ``leaf_df`` DataFrames with x/y coordinates for plotting.

Layout Options
--------------

IplotX supports several layout algorithms:

- ``"horizontal"`` — left-to-right dendrogram
- ``"vertical"`` — top-to-bottom dendrogram
- ``"radial"`` — circular layout

.. skip: next

.. code-block:: python

   tree_data = shim(layout="radial")

Branch Lengths
--------------

Branch lengths are extracted automatically from the DataFrame:

1. **``origin_time_delta``** column (if present) — used directly as branch lengths.
2. **``origin_time``** column (if present) — branch lengths are computed as ``origin_time[child] - origin_time[parent]``.
3. **Neither** — all branch lengths are ``None``, and iplotx uses a unit-length layout.

Leaf Labels
-----------

When the DataFrame contains a ``taxon_label`` column, those labels are used as node names.
Pass ``leaf_labels=True`` to the provider call to include them in the layout:

.. skip: next

.. code-block:: python

   tree_data = shim(layout="horizontal", leaf_labels=True)
   # tree_data["leaf_df"]["label"] contains the taxon labels

Polars DataFrames
-----------------

The polars shim works identically:

.. skip: next

.. code-block:: python

   import polars as pl
   from phyloframe import legacy as pfl

   df_polars = pfl.alifestd_from_newick_polars("((A:1,B:2):3,(C:4,D:5):6);")
   shim = pfl.AlifestdIplotxShimPolars(df_polars)
   tree_data = shim(layout="horizontal")

NumPy Arrays
------------

For direct numpy usage when you already have an ``ancestor_id`` array:

.. skip: next

.. code-block:: python

   import numpy as np
   from phyloframe import legacy as pfl

   ancestor_ids = np.array([0, 0, 0, 1, 1, 2, 2])
   branch_lengths = np.array([np.nan, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0])
   names = np.array(["root", "A", "B", "C", "D", "E", "F"])

   shim = pfl.AlifestdIplotxShimNumpy(
       ancestor_ids, names=names, branch_lengths=branch_lengths,
   )
   tree_data = shim(layout="horizontal")

Entry Point Autodiscovery
-------------------------

The pandas and polars shims are registered as `iplotx entry points <https://iplotx.readthedocs.io/en/latest/api/providers.html>`_, so iplotx can automatically discover and use them when a phyloframe DataFrame is passed directly to iplotx plotting functions.
