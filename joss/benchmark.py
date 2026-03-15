#!/usr/bin/env python3
"""Benchmark phyloframe against other phylogenetics libraries.

Reproduces the TreeSwift paper benchmark: preorder, postorder, inorder,
levelorder traversals, all-pairs MRCA, and all-pairs pairwise distances
on binary trees with 100 to 1,000,000 leaves.  Also benchmarks newick
load/save.  Any single operation exceeding TIMEOUT seconds is skipped.
"""

import csv
import io
import signal
import sys
import time

TIMEOUT = 10  # seconds per operation
SIZES = [100, 1_000, 10_000, 100_000, 1_000_000]
OPERATIONS = [
    "load_newick",
    "save_newick",
    "preorder",
    "postorder",
    "inorder",
    "levelorder",
    "mrca_allpairs",
    "pairwise_dist",
]


class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Timed out")


def timed(fn, timeout=TIMEOUT):
    """Run fn() and return elapsed seconds, or None if it times out."""
    old = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)
    try:
        t0 = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - t0
    except TimeoutError:
        elapsed = None
    except Exception as exc:
        print(f"    error: {exc}", file=sys.stderr)
        elapsed = None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    return elapsed


# ── tree generation ──────────────────────────────────────────────────
def _random_binary_newick(n_leaves):
    """Build a random binary tree with n_leaves as a Newick string.

    Uses treeswift's coalescent simulator when available, otherwise
    builds a balanced binary tree newick string directly.
    """
    try:
        import dendropy

        taxa = dendropy.TaxonNamespace([f"t{i}" for i in range(n_leaves)])
        tree = dendropy.simulate.treesim.birth_death_tree(
            birth_rate=1.0,
            death_rate=0.0,
            num_extant_tips=n_leaves,
            taxon_namespace=taxa,
        )
        return tree.as_string(schema="newick")
    except Exception:
        pass
    # fallback: balanced binary tree
    return _balanced_newick(n_leaves)


def _balanced_newick(n):
    """Build a balanced binary newick string with n leaves."""
    if n <= 0:
        return "();"
    labels = [f"t{i}" for i in range(n)]

    def _build(lo, hi):
        if lo == hi:
            return f"{labels[lo]}:1"
        mid = (lo + hi) // 2
        return f"({_build(lo, mid)},{_build(mid + 1, hi)}):1"

    return _build(0, n - 1) + ";"


# ── library adapters ────────────────────────────────────────────────
class PhyloframeBench:
    name = "phyloframe"

    def __init__(self, newick):
        from phyloframe.legacy import (
            alifestd_from_newick,
            alifestd_to_working_format,
        )

        self._newick = newick
        self._from_newick = alifestd_from_newick
        self._to_working = alifestd_to_working_format
        self._df = None

    def load_newick(self):
        self._df = self._from_newick(self._newick)

    def save_newick(self):
        from phyloframe.legacy import alifestd_as_newick_asexual

        df = self._ensure_df()
        alifestd_as_newick_asexual(df, mutate=True)

    def preorder(self):
        # phyloframe has no dedicated preorder; postorder reversed is
        # not equivalent, so report as unsupported
        raise NotImplementedError("preorder not available")

    def postorder(self):
        from phyloframe.legacy import (
            alifestd_unfurl_traversal_postorder_asexual,
        )

        df = self._ensure_df()
        alifestd_unfurl_traversal_postorder_asexual(df, mutate=True)

    def inorder(self):
        from phyloframe.legacy import (
            alifestd_unfurl_traversal_inorder_asexual,
        )

        df = self._ensure_df()
        alifestd_unfurl_traversal_inorder_asexual(df, mutate=True)

    def levelorder(self):
        raise NotImplementedError("levelorder not available")

    def mrca_allpairs(self):
        from phyloframe.legacy import alifestd_calc_mrca_id_matrix_asexual

        df = self._ensure_working()
        alifestd_calc_mrca_id_matrix_asexual(df, mutate=True)

    def pairwise_dist(self):
        raise NotImplementedError("pairwise distances not available")

    def _ensure_df(self):
        if self._df is None:
            self._df = self._from_newick(self._newick)
        return self._df

    def _ensure_working(self):
        df = self._ensure_df()
        return self._to_working(df)


class TreeswiftBench:
    name = "treeswift"

    def __init__(self, newick):
        import treeswift

        self._newick = newick
        self._treeswift = treeswift
        self._tree = None

    def load_newick(self):
        self._tree = self._treeswift.read_tree_newick(self._newick)

    def save_newick(self):
        t = self._ensure_tree()
        t.newick()

    def preorder(self):
        t = self._ensure_tree()
        for _ in t.traverse_preorder():
            pass

    def postorder(self):
        t = self._ensure_tree()
        for _ in t.traverse_postorder():
            pass

    def inorder(self):
        t = self._ensure_tree()
        for _ in t.traverse_inorder():
            pass

    def levelorder(self):
        t = self._ensure_tree()
        for _ in t.traverse_levelorder():
            pass

    def mrca_allpairs(self):
        t = self._ensure_tree()
        labels = [n.label for n in t.traverse_leaves()]
        for i, a in enumerate(labels):
            for b in labels[i + 1 :]:
                t.mrca({a, b})

    def pairwise_dist(self):
        t = self._ensure_tree()
        t.distance_matrix(leaf_labels=True)

    def _ensure_tree(self):
        if self._tree is None:
            self._tree = self._treeswift.read_tree_newick(self._newick)
        return self._tree


class BiopythonBench:
    name = "biopython"

    def __init__(self, newick):
        self._newick = newick
        self._tree = None

    def load_newick(self):
        from Bio import Phylo

        self._tree = Phylo.read(io.StringIO(self._newick), "newick")

    def save_newick(self):
        from Bio import Phylo

        t = self._ensure_tree()
        buf = io.StringIO()
        Phylo.write(t, buf, "newick")

    def preorder(self):
        t = self._ensure_tree()
        for _ in t.find_clades(order="preorder"):
            pass

    def postorder(self):
        t = self._ensure_tree()
        for _ in t.find_clades(order="postorder"):
            pass

    def inorder(self):
        raise NotImplementedError("inorder not available in Bio.Phylo")

    def levelorder(self):
        t = self._ensure_tree()
        for _ in t.find_clades(order="level"):
            pass

    def mrca_allpairs(self):
        t = self._ensure_tree()
        terminals = t.get_terminals()
        for i, a in enumerate(terminals):
            for b in terminals[i + 1 :]:
                t.common_ancestor(a, b)

    def pairwise_dist(self):
        t = self._ensure_tree()
        terminals = t.get_terminals()
        for i, a in enumerate(terminals):
            for b in terminals[i + 1 :]:
                t.distance(a, b)

    def _ensure_tree(self):
        if self._tree is None:
            from Bio import Phylo

            self._tree = Phylo.read(io.StringIO(self._newick), "newick")
        return self._tree


class DendropyBench:
    name = "dendropy"

    def __init__(self, newick):
        self._newick = newick
        self._tree = None

    def load_newick(self):
        import dendropy

        self._tree = dendropy.Tree.get(data=self._newick, schema="newick")

    def save_newick(self):
        t = self._ensure_tree()
        t.as_string(schema="newick")

    def preorder(self):
        t = self._ensure_tree()
        for _ in t.preorder_node_iter():
            pass

    def postorder(self):
        t = self._ensure_tree()
        for _ in t.postorder_node_iter():
            pass

    def inorder(self):
        t = self._ensure_tree()
        for _ in t.inorder_node_iter():
            pass

    def levelorder(self):
        t = self._ensure_tree()
        for _ in t.levelorder_node_iter():
            pass

    def mrca_allpairs(self):

        t = self._ensure_tree()
        pdm = t.phylogenetic_distance_matrix()
        leaf_list = list(t.leaf_node_iter())
        for i, a in enumerate(leaf_list):
            for b in leaf_list[i + 1 :]:
                pdm.mrca(a.taxon, b.taxon)

    def pairwise_dist(self):
        t = self._ensure_tree()
        pdm = t.phylogenetic_distance_matrix()
        leaf_list = list(t.leaf_node_iter())
        for i, a in enumerate(leaf_list):
            for b in leaf_list[i + 1 :]:
                pdm(a.taxon, b.taxon)

    def _ensure_tree(self):
        if self._tree is None:
            import dendropy

            self._tree = dendropy.Tree.get(data=self._newick, schema="newick")
        return self._tree


class EteBench:
    name = "ete"

    def __init__(self, newick):
        self._newick = newick
        self._tree = None

    def load_newick(self):
        from ete3 import Tree

        self._tree = Tree(self._newick)

    def save_newick(self):
        t = self._ensure_tree()
        t.write()

    def preorder(self):
        t = self._ensure_tree()
        for _ in t.traverse("preorder"):
            pass

    def postorder(self):
        t = self._ensure_tree()
        for _ in t.traverse("postorder"):
            pass

    def inorder(self):
        raise NotImplementedError("inorder not available in ete")

    def levelorder(self):
        t = self._ensure_tree()
        for _ in t.traverse("levelorder"):
            pass

    def mrca_allpairs(self):
        t = self._ensure_tree()
        leaves = list(t.get_leaves())
        for i, a in enumerate(leaves):
            for b in leaves[i + 1 :]:
                t.get_common_ancestor(a, b)

    def pairwise_dist(self):
        t = self._ensure_tree()
        leaves = list(t.get_leaves())
        for i, a in enumerate(leaves):
            for b in leaves[i + 1 :]:
                a.get_distance(b)

    def _ensure_tree(self):
        if self._tree is None:
            from ete3 import Tree

            self._tree = Tree(self._newick)
        return self._tree


class CompactTreeBench:
    name = "compacttree"

    def __init__(self, newick):
        self._newick = newick
        self._tree = None

    def load_newick(self):
        from compact_tree import CompactTree

        self._tree = CompactTree(self._newick)

    def save_newick(self):
        t = self._ensure_tree()
        t.to_newick()

    def preorder(self):
        t = self._ensure_tree()
        for _ in t.traverse_preorder():
            pass

    def postorder(self):
        t = self._ensure_tree()
        for _ in t.traverse_postorder():
            pass

    def inorder(self):
        t = self._ensure_tree()
        for _ in t.traverse_inorder():
            pass

    def levelorder(self):
        t = self._ensure_tree()
        for _ in t.traverse_levelorder():
            pass

    def mrca_allpairs(self):
        raise NotImplementedError("MRCA not available in CompactTree")

    def pairwise_dist(self):
        raise NotImplementedError(
            "pairwise distances not available in CompactTree"
        )

    def _ensure_tree(self):
        if self._tree is None:
            from compact_tree import CompactTree

            self._tree = CompactTree(self._newick)
        return self._tree


LIBRARIES = [
    PhyloframeBench,
    TreeswiftBench,
    BiopythonBench,
    DendropyBench,
    EteBench,
    CompactTreeBench,
]


def run_benchmarks():
    results = []
    for n_leaves in SIZES:
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"Generating tree with {n_leaves:,} leaves...", file=sys.stderr)
        newick = _balanced_newick(n_leaves)
        print(f"  newick length: {len(newick):,} chars", file=sys.stderr)

        for LibClass in LIBRARIES:
            print(f"  {LibClass.name}:", file=sys.stderr)
            bench = LibClass(newick)

            for op in OPERATIONS:
                fn = getattr(bench, op, None)
                if fn is None:
                    elapsed = None
                else:
                    elapsed = timed(fn)
                status = f"{elapsed:.4f}s" if elapsed is not None else "SKIP"
                print(f"    {op}: {status}", file=sys.stderr)
                results.append(
                    {
                        "library": LibClass.name,
                        "n_leaves": n_leaves,
                        "operation": op,
                        "seconds": elapsed,
                    }
                )

    return results


def print_summary_table(results):
    """Print a pivoted summary table to stderr."""
    import pandas as pd

    df = pd.DataFrame(results)
    df["seconds"] = pd.to_numeric(df["seconds"])
    for n_leaves, grp in df.groupby("n_leaves"):
        pivot = grp.pivot(
            index="operation", columns="library", values="seconds"
        )
        # reorder columns to match LIBRARIES order
        lib_order = [L.name for L in LIBRARIES]
        pivot = pivot[[c for c in lib_order if c in pivot.columns]]
        print(f"\n--- {n_leaves:,} leaves ---", file=sys.stderr)
        with pd.option_context(
            "display.float_format",
            "{:.6f}".format,
            "display.max_columns",
            20,
            "display.width",
            200,
        ):
            print(pivot.to_string(na_rep="--"), file=sys.stderr)


def main():
    results = run_benchmarks()
    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=["library", "n_leaves", "operation", "seconds"],
    )
    writer.writeheader()
    writer.writerows(results)
    print_summary_table(results)


if __name__ == "__main__":
    main()
