======================================
Tree Properties and Metrics (Legacy)
======================================

This guide covers phyloframe's operations for marking node properties,
counting tree features, and computing phylogenetic metrics.

Marking Properties
==================

"Mark" functions add a new column to the phylogeny DataFrame.
All mark functions share these conventions:

- ``mark_as`` parameter to customize the output column name.
- ``mutate`` parameter (default ``False``) to control whether the input
  DataFrame is modified.
- Return the modified DataFrame (always use the return value).

Leaf and Root Detection
-----------------------

.. code-block:: python

   from phyloframe import legacy as pfl

   df = pfl.alifestd_from_newick("((A,B),(C,D));")
   df = pfl.alifestd_to_working_format(df)

   # Mark leaf nodes (no descendants)
   df = pfl.alifestd_mark_leaves(df)
   # df["is_leaf"]: True for A, B, C, D

   # Mark root nodes (no ancestors)
   df = pfl.alifestd_mark_roots(df)
   # df["is_root"]: True for the root node

Node Depth
----------

Number of edges between a node and the root:

.. code-block:: python

   df = pfl.alifestd_mark_node_depth_asexual(df)
   # df["node_depth"]: 0 for root, 1 for root's children, etc.

Descendant and Children Counts
------------------------------

.. code-block:: python

   # Total descendants (excluding self)
   df = pfl.alifestd_mark_num_descendants_asexual(df)

   # Direct children count
   df = pfl.alifestd_mark_num_children_asexual(df)

Child Identification
--------------------

For binary trees, identify left and right children:

.. code-block:: python

   # Requires strictly bifurcating tree
   df = pfl.alifestd_mark_is_left_child_asexual(df)
   df = pfl.alifestd_mark_is_right_child_asexual(df)

Time-based Properties
---------------------

These require an ``origin_time`` column in the input:

.. code-block:: python

   df_timed = pfl.alifestd_from_newick("((A:1,B:2):3,(C:4,D:5):6);")
   df_timed = pfl.alifestd_to_working_format(df_timed)
   df_timed["origin_time"] = range(len(df_timed))

   # Time of this node's ancestor
   df_timed = pfl.alifestd_mark_ancestor_origin_time_asexual(df_timed)

   # Time elapsed since ancestor (branch length equivalent)
   df_timed = pfl.alifestd_mark_origin_time_delta_asexual(df_timed)

Counting Operations
===================

Count functions return scalars:

.. code-block:: python

   df = pfl.alifestd_from_newick("((A,B),(C,(D,E)));")

   pfl.alifestd_count_leaf_nodes(df)      # 5
   pfl.alifestd_count_inner_nodes(df)     # 4
   pfl.alifestd_count_root_nodes(df)      # 1
   pfl.alifestd_count_unifurcations(df)   # 0
   pfl.alifestd_count_polytomies(df)      # 0

Finding Specific Nodes
======================

.. code-block:: python

   import numpy as np

   # Get arrays of IDs
   leaf_ids = pfl.alifestd_find_leaf_ids(df)     # np.ndarray
   root_ids = pfl.alifestd_find_root_ids(df)     # np.ndarray

Validation
==========

Check that a DataFrame conforms to the alife standard format:

.. code-block:: python

   is_valid = pfl.alifestd_validate(df)

   # Check specific structural properties
   pfl.alifestd_is_asexual(df)
   pfl.alifestd_is_topologically_sorted(df)
   pfl.alifestd_has_contiguous_ids(df)
   pfl.alifestd_is_strictly_bifurcating_asexual(df)

Balance Metrics
===============

.. code-block:: python

   df = pfl.alifestd_from_newick("((A,B),(C,D));")
   df = pfl.alifestd_to_working_format(df)

   # Colless balance index (per-node)
   df = pfl.alifestd_mark_colless_index_asexual(df)

   # Sackin balance index (per-node)
   df = pfl.alifestd_mark_sackin_index_asexual(df)

MRCA (Most Recent Common Ancestor)
===================================

.. code-block:: python

   df_mrca = pfl.alifestd_from_newick("((A,B),(C,D));")
   df_mrca = pfl.alifestd_to_working_format(df_mrca)

   # Find IDs
   leaf_ids = pfl.alifestd_find_leaf_ids(df_mrca)

   # MRCA of two specific nodes
   mrca_id = pfl.alifestd_find_pair_mrca_id_asexual(
       df_mrca, leaf_ids[0], leaf_ids[1],
   )

Triplet Distance
================

Compare two trees using triplet-based distance:

.. code-block:: python

   tree1 = pfl.alifestd_from_newick("((A,B),(C,D));")
   tree2 = pfl.alifestd_from_newick("((A,C),(B,D));")

   dist = pfl.alifestd_calc_triplet_distance_asexual(tree1, tree2)

Distance Matrix
===============

Compute pairwise distances between leaf nodes:

.. code-block:: python

   df_dist = pfl.alifestd_from_newick("((A:1,B:2):3,(C:4,D:5):6);")
   df_dist = pfl.alifestd_to_working_format(df_dist)
   df_dist["origin_time"] = range(len(df_dist))

   # Returns a NumPy distance matrix
   dist_matrix = pfl.alifestd_calc_distance_matrix_asexual(df_dist)
