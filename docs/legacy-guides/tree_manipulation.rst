========================================
Tree Manipulation and Pruning (Legacy)
========================================

.. note::

   This documentation covers the **legacy** API (from phyloframe import legacy).
   The legacy API is stable and will continue to be maintained for backward
   compatibility.
   A redesigned API is under development.


This guide covers operations that transform the structure of a phylogeny:
collapsing, splaying, pruning, downsampling, and masking.

.. note::

   Structural transforms may invalidate previously computed columns (e.g.,
   ``node_depth``, ``num_descendants``).
   See :doc:`concepts` for details on the topological sensitivity system.

Structural Transformations
==========================

Collapsing Unifurcations
------------------------

Remove single-child (unifurcating) nodes, connecting their parent directly
to their child:

.. code-block:: python

   from phyloframe import legacy as pfl

   df = pfl.alifestd_from_newick("((A,B),(C,D));")
   df = (
       df.pipe(pfl.alifestd_to_working_format)
       .pipe(pfl.alifestd_collapse_unifurcations)
   )

Splaying Polytomies
-------------------

Expand multi-child (polytomy) nodes into a cascade of bifurcations:

.. code-block:: python

   df = pfl.alifestd_splay_polytomies(df)

Root Operations
---------------

.. code-block:: python

   # Add a synthetic global root above all existing roots
   df = pfl.alifestd_add_global_root(df)

   # Set attributes on the new root
   df = pfl.alifestd_add_global_root(
       df, root_attrs={"origin_time": 0.0, "taxon_label": "global_root"},
   )

   # Join multiple roots to the oldest root
   df = pfl.alifestd_join_roots(df)

Rerooting
---------

Change the root of a tree to a specified node:

.. code-block:: python

   df_reroot = pfl.alifestd_from_newick(
       "((A,B),(C,D));", create_ancestor_list=True,
   )
   df_reroot = pfl.alifestd_to_working_format(df_reroot)
   leaf_ids = pfl.alifestd_find_leaf_ids(df_reroot)
   df_reroot = pfl.alifestd_reroot_at_id_asexual(
       df_reroot, int(leaf_ids[0]),
   )

Ladderization
-------------

Reorder children for consistent visual presentation:

.. code-block:: python

   df_ladder = pfl.alifestd_from_newick(
       "((A,B),(C,D));", create_ancestor_list=True,
   )
   df_ladder = pfl.alifestd_to_working_format(df_ladder)
   df_ladder = pfl.alifestd_ladderize_asexual(df_ladder)

Trunk Operations
----------------

.. code-block:: python

   df_trunk = pfl.alifestd_make_comb(n_leaves=5)
   df_trunk = pfl.alifestd_to_working_format(df_trunk)

   # Delete unifurcating root nodes
   df_trunk = pfl.alifestd_delete_unifurcating_roots_asexual(df_trunk)

Aggregating Multiple Phylogenies
---------------------------------

Concatenate independent phylogenies with ID reassignment:

.. code-block:: python

   tree1 = pfl.alifestd_from_newick(
       "(A,B);", create_ancestor_list=True,
   )
   tree2 = pfl.alifestd_from_newick(
       "(C,D);", create_ancestor_list=True,
   )
   combined = pfl.alifestd_aggregate_phylogenies([tree1, tree2])

Chronological Sorting
---------------------

Sort rows by ``origin_time`` for time-based analyses:

.. code-block:: python

   df_chrono = pfl.alifestd_from_newick("((A:1,B:2):3,(C:4,D:5):6);")
   df_chrono = pfl.alifestd_to_working_format(df_chrono)
   df_chrono["origin_time"] = range(len(df_chrono))
   df_chrono = pfl.alifestd_chronological_sort(df_chrono)

Tip Sampling (Mark Functions)
=============================

Tip sampling mark functions add a boolean column indicating which tips to
retain.
They do **not** remove any rows --- use a pruning step afterward.

Uniform Random Sampling
------------------------

.. code-block:: python

   df_sample = pfl.alifestd_make_balanced_bifurcating(depth=5)
   df_sample = pfl.alifestd_to_working_format(df_sample)
   df_sample["origin_time"] = range(len(df_sample))

   # Mark 10 randomly selected tips
   df_sample = pfl.alifestd_mark_sample_tips_uniform_asexual(
       df_sample, n_sample=10, seed=42, mark_as="keep",
   )

Canopy Sampling
---------------

Retain tips with the largest values in a criterion column (e.g., the most
recent tips by ``origin_time``):

.. code-block:: python

   df_sample = pfl.alifestd_mark_sample_tips_canopy_asexual(
       df_sample,
       criterion="origin_time",
       mark_as="keep_canopy",
       n_sample=5,
   )

Lineage Sampling
----------------

Retain tips closest to a focal lineage (the lineage of the tip with the
largest criterion value):

.. code-block:: python

   df_sample = pfl.alifestd_mark_sample_tips_lineage_asexual(
       df_sample, n_sample=5, mark_as="keep_lineage",
   )

Combining Sample Masks
======================

Because sample marks are boolean columns, they compose naturally with
standard boolean operations:

.. code-block:: python

   # Keep tips matching EITHER criterion (union)
   df_sample["keep"] = (
       df_sample["keep_canopy"] | df_sample["keep_lineage"]
   )

   # Keep tips matching BOTH criteria (intersection)
   df_sample["keep"] = (
       df_sample["keep_canopy"] & df_sample["keep_lineage"]
   )

   # Invert a selection
   df_sample["keep"] = ~df_sample["keep_canopy"]

Pruning
=======

Pruning Extinct Lineages
------------------------

Remove lineages that have no extant descendants.
The ``criterion`` parameter specifies which boolean column marks extant
taxa (default: ``"extant"``):

.. code-block:: python

   df_prune = pfl.alifestd_make_balanced_bifurcating(depth=4)
   df_prune = pfl.alifestd_to_working_format(df_prune)
   df_prune["origin_time"] = range(len(df_prune))

   # Mark which taxa are extant
   threshold = len(df_prune) // 2
   df_prune["extant"] = df_prune["origin_time"] >= threshold

   # Remove lineages without extant descendants
   df_prune = pfl.alifestd_prune_extinct_lineages_asexual(df_prune)

You can also use a custom criterion column name:

.. code-block:: python

   df_prune2 = pfl.alifestd_make_balanced_bifurcating(depth=4)
   df_prune2 = pfl.alifestd_to_working_format(df_prune2)
   df_prune2["is_alive"] = True
   df_prune2 = pfl.alifestd_prune_extinct_lineages_asexual(
       df_prune2, criterion="is_alive",
   )

Coarsening with a Mask
----------------------

Keep only rows matching a boolean mask, re-wiring ancestor relationships
to maintain tree connectivity:

.. code-block:: python

   df_coarsen = pfl.alifestd_from_newick(
       "((A,B),(C,D));", create_ancestor_list=True,
   )
   df_coarsen = (
       df_coarsen.pipe(pfl.alifestd_to_working_format)
       .pipe(pfl.alifestd_mark_node_depth_asexual)
       .pipe(pfl.alifestd_mark_leaves)
   )
   mask = (df_coarsen["node_depth"] % 2 == 0) | df_coarsen["is_leaf"]
   df_coarsen = pfl.alifestd_coarsen_mask(df_coarsen, mask)

Composed Example: Multi-criteria Downsampling
=============================================

A complete workflow combining multiple sampling strategies and pruning:

.. code-block:: python

   import pandas as pd
   from phyloframe import legacy as pfl

   # Load or create a phylogeny with origin times
   df = pfl.alifestd_from_newick(
       "((A:1,B:2):3,(C:4,(D:5,E:6):7):8);",
   )
   df = df.pipe(pfl.alifestd_to_working_format)
   df["origin_time"] = range(len(df))

   # Strategy 1: canopy --- keep the 2 most recent tips
   df = df.pipe(
       pfl.alifestd_mark_sample_tips_canopy_asexual,
       n_sample=2, mark_as="keep_canopy",
   )

   # Strategy 2: lineage --- keep the 2 tips closest to focal lineage
   df = df.pipe(
       pfl.alifestd_mark_sample_tips_lineage_asexual,
       n_sample=2, mark_as="keep_lineage",
   )

   # Combine: OR the masks to keep tips matching either criterion
   df["extant"] = df["keep_canopy"] | df["keep_lineage"]

   # Prune lineages with no extant descendants
   result = df.pipe(pfl.alifestd_prune_extinct_lineages_asexual)

Downsampling (Combined Mark + Prune)
=====================================

For convenience, ``alifestd_downsample_tips_*`` functions combine the
mark and prune steps:

.. code-block:: python

   df_ds = pfl.alifestd_make_balanced_bifurcating(depth=5)
   df_ds = pfl.alifestd_to_working_format(df_ds)
   df_ds["origin_time"] = range(len(df_ds))

   # Uniform random downsampling
   df_u = pfl.alifestd_downsample_tips_uniform_asexual(
       df_ds, n_downsample=10,
   )

   # Canopy downsampling
   df_c = pfl.alifestd_downsample_tips_canopy_asexual(
       df_ds, n_downsample=5,
   )

   # Lineage-based downsampling
   df_l = pfl.alifestd_downsample_tips_lineage_asexual(
       df_ds, n_downsample=5,
   )

These are equivalent to calling the corresponding mark function, setting
the ``"extant"`` column, and then pruning extinct lineages.
Use the separate mark + prune workflow (above) when you need to combine
multiple sampling criteria.
