==========================
Tree Visualization (Legacy)
==========================

.. note::

   This documentation covers the **legacy** API (from phyloframe import legacy).
   The legacy API is stable and will continue to be maintained for backward
   compatibility.
   A redesigned API will accompany phyloframe v1.0.0.

For interactive in-browser visualization of trees up to millions of nodes, an `experimental fork of taxonium <https://mmore500.com/taxonium>`_ is available that supports alife standard CSV, TSV, and Parquet files.
Usage information can be found in `Taxonium's documentation <https://docs.taxonium.org/en/latest/>`_.

For programmatic visualizations, phyloframe integrates with `iplotx <https://iplotx.readthedocs.io/>`_ to visualize phylogenetic trees from DataFrames.

iplotx Quick Start
===========

.. skip: next

.. code-block:: python

   import iplotx
   from matplotlib import pyplot as plt
   from phyloframe import legacy as pfl

   df = pfl.alifestd_from_newick("((A:1,B:2):3,(C:4,D:5):6);")
   df = pfl.alifestd_to_working_format(df)

   iplotx.tree(pfl.alifestd_to_iplotx_pandas(df), leaf_labels=True)
   plt.show()

The input data must be **asexual**, have **contiguous IDs**, and be **topologically sorted**.
Use ``alifestd_to_working_format`` to ensure this.

Branch lengths are extracted automatically from ``origin_time_delta`` or ``origin_time`` columns when present.
Taxon labels are read from the ``taxon_label`` column if it exists.

For layout options, styling, leaf labels, and other features, see the `iplotx tree documentation <https://iplotx.readthedocs.io/en/latest/api/tree.html>`_.
More advanced phyloframe/iplotx examples --- including plots that leverage DataFrame-based Seaborn interoperation --- can be found in the `iplotx gallery <https://iplotx.readthedocs.io/en/latest/gallery/tree/plot_phyloframe_tree.html>`_.
