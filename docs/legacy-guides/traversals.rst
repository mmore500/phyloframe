=======================
Tree Traversal (Legacy)
=======================

This guide covers phyloframe's tree traversal operations and the
supplemental data structures that enable them.

Traversal Orders
================

Phyloframe provides functions that return node IDs in standard traversal
orders as NumPy arrays.

Preorder (Depth-first, Parent Before Children)
----------------------------------------------

Visit each node before its descendants:

.. code-block:: python

   from phyloframe import legacy as pfl
   import numpy as np

   df = pfl.alifestd_from_newick("((A,B),(C,D));")
   df = pfl.alifestd_to_working_format(df)

   order = pfl.alifestd_unfurl_traversal_preorder_asexual(df)
   # Returns np.ndarray of node IDs in preorder

Postorder (Depth-first, Children Before Parent)
------------------------------------------------

Visit each node after all its descendants:

.. code-block:: python

   order = pfl.alifestd_unfurl_traversal_postorder_asexual(df)

Inorder (Left, Root, Right)
----------------------------

For binary trees, visit left subtree, then node, then right subtree:

.. code-block:: python

   order = pfl.alifestd_unfurl_traversal_inorder_asexual(df)

Level-order (Breadth-first)
----------------------------

Visit nodes level by level from root to leaves:

.. code-block:: python

   order = pfl.alifestd_unfurl_traversal_levelorder_asexual(df)

Topological Order
-----------------

Visit nodes in topological order (ancestors before descendants).
For data in working format, this is simply a forward iteration along rows
and can be much faster than other traversal orderings:

.. code-block:: python

   order = pfl.alifestd_unfurl_traversal_topological_asexual(df)

Semiorder
---------

A hybrid traversal that interleaves preorder and postorder visits:

.. code-block:: python

   order = pfl.alifestd_unfurl_traversal_semiorder_asexual(df)

Lineage Unfurling
-----------------

Trace the lineage from a given node back to the root:

.. code-block:: python

   # Get the path from node 3 to the root
   lineage_ids = pfl.alifestd_unfurl_lineage_asexual(df, 3)

Supplemental Data Structures for Traversal
===========================================

The alife standard format stores only parent pointers (``ancestor_id``).
Traversal algorithms need to navigate from parents to children.
Phyloframe provides two supplemental structures for this purpose.

CSR (Compressed Sparse Row) Representation
-------------------------------------------

The CSR format represents the parent-to-children mapping as two flat
arrays.
This is the most common structure used internally by traversal and
distance algorithms.

.. code-block:: python

   df = pfl.alifestd_from_newick("((A,B),(C,D));")
   df = pfl.alifestd_to_working_format(df)

   # Step 1: mark number of children
   df = pfl.alifestd_mark_num_children_asexual(df)

   # Step 2: compute CSR offsets and children arrays
   df = pfl.alifestd_mark_csr_offsets_asexual(df)
   df = pfl.alifestd_mark_csr_children_asexual(df)

   # Access children of any node in O(1)
   offsets = df["csr_offsets"].values
   children = df["csr_children"].values
   num_children = df["num_children"].values

   def get_children(node_id):
       start = offsets[node_id]
       return children[start:start + num_children[node_id]]

**How CSR works:**

Consider a tree with nodes ``{0, 1, 2, 3, 4}`` where node 0 has
children ``{1, 2}`` and node 1 has children ``{3, 4}``::

   csr_offsets:  [0, 2, 4, 4, 4]
   csr_children: [1, 2, 3, 4]
   num_children: [2, 2, 0, 0, 0]

   Node 0's children: csr_children[0:0+2] = [1, 2]
   Node 1's children: csr_children[2:2+2] = [3, 4]
   Node 2's children: csr_children[4:4+0] = []  (leaf)

First-Child/Next-Sibling Linked List
--------------------------------------

An alternative structure that uses two integer columns to form a linked
list through the tree.
This uses less memory and avoids constructing auxiliary arrays.

.. code-block:: python

   df = pfl.alifestd_from_newick("((A,B),(C,D));")
   df = pfl.alifestd_to_working_format(df)

   # Add linked list columns
   df = pfl.alifestd_mark_first_child_id_asexual(df)
   df = pfl.alifestd_mark_next_sibling_id_asexual(df)

   first_child = df["first_child_id"].values
   next_sibling = df["next_sibling_id"].values

   def get_children_linked(node_id):
       """Walk the linked list of children."""
       result = []
       child = first_child[node_id]
       if child == node_id:  # leaf, no children
           return result
       while True:
           result.append(child)
           nxt = next_sibling[child]
           if nxt == child:  # no more siblings
               break
           child = nxt
       return result

**Sentinel convention:** a node points to itself when there is no
first child (leaf) or no next sibling (last sibling).

**How it works:**

For the same tree as above::

   first_child:  [1, 3, -, -, -]   (node 0 -> child 1, node 1 -> child 3)
   next_sibling: [-, 2, -, 4, -]   (node 1 -> sibling 2, node 3 -> sibling 4)

   ('-' means self-reference, i.e., no child/sibling)

Choosing Between CSR and Linked List
--------------------------------------

- **CSR** is better for algorithms that need random access to any node's
  children (e.g., distance matrix computation).
- **Linked list** is better for sequential traversals (e.g., DFS) where
  you visit children in order and want to minimize memory.

In practice, the traversal functions choose automatically based on what
columns are already available.

Using Traversals for Custom Algorithms
======================================

.. code-block:: python

   import numpy as np
   from phyloframe._auxlib import jit
   from phyloframe import legacy as pfl

   @jit(nopython=True, cache=False)
   def sum_subtree_values(
       ancestor_ids: np.ndarray,
       values: np.ndarray,
   ) -> np.ndarray:
       """Compute sum of values in each node's subtree.

       Processes nodes in reverse order (postorder for topologically
       sorted, contiguous-ID data).
       """
       n = len(ancestor_ids)
       subtree_sums = values.copy()
       # Reverse iteration is postorder for topologically sorted data
       for i in range(n - 1, -1, -1):
           parent = ancestor_ids[i]
           if parent != i:  # not a root
               subtree_sums[parent] += subtree_sums[i]
       return subtree_sums

   df = pfl.alifestd_make_balanced_bifurcating(depth=4)
   df = pfl.alifestd_to_working_format(df)
   df["value"] = np.ones(len(df))

   sums = sum_subtree_values(
       df["ancestor_id"].values, df["value"].values,
   )
