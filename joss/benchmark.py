#!/usr/bin/env python3
"""Benchmark phyloframe against other phylogenetics libraries.

Reproduces the TreeSwift paper benchmark: preorder, postorder, inorder,
levelorder traversals, all-pairs MRCA, and all-pairs pairwise distances
on binary trees with 100 to 1,000,000 leaves.  Also benchmarks newick
load/save.  Any single operation exceeding TIMEOUT seconds is skipped.
"""

import csv
import gc
import io
import multiprocessing
import sys
import time

# Use forkserver to avoid deadlocks caused by forking after polars/numba
# have initialized background threads during JIT warmup.
multiprocessing.set_start_method("forkserver", force=True)

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
    "memory_bytes",
]


def _run_in_child(conn, fn, return_value):
    """Target for subprocess: run fn and send result over pipe."""
    try:
        t0 = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - t0
        if return_value:
            conn.send((elapsed, result))
        else:
            conn.send((elapsed, None))
    except Exception as exc:
        conn.send(("error", str(exc)))
    finally:
        conn.close()


def timed(fn, timeout=TIMEOUT):
    """Run fn() in a subprocess and return elapsed seconds.

    Using a subprocess protects against C++ crashes (e.g. std::bad_alloc)
    that would otherwise abort the entire benchmark process.
    """
    parent_conn, child_conn = multiprocessing.Pipe()
    proc = multiprocessing.Process(
        target=_run_in_child, args=(child_conn, fn, False)
    )
    proc.start()
    proc.join(timeout=timeout)
    if proc.is_alive():
        proc.kill()
        proc.join()
        parent_conn.close()
        return None
    if proc.exitcode != 0:
        print(
            f"    error: child process exited with code {proc.exitcode}",
            file=sys.stderr,
        )
        parent_conn.close()
        return None
    if parent_conn.poll():
        result = parent_conn.recv()
        parent_conn.close()
        if result[0] == "error":
            print(f"    error: {result[1]}", file=sys.stderr)
            return None
        return result[0]
    parent_conn.close()
    return None


def timed_call(fn, timeout=TIMEOUT):
    """Run fn() in a subprocess and return its result, or None."""
    parent_conn, child_conn = multiprocessing.Pipe()
    proc = multiprocessing.Process(
        target=_run_in_child, args=(child_conn, fn, True)
    )
    proc.start()
    proc.join(timeout=timeout)
    if proc.is_alive():
        proc.kill()
        proc.join()
        parent_conn.close()
        return None
    if proc.exitcode != 0:
        print(
            f"    error: child process exited with code {proc.exitcode}",
            file=sys.stderr,
        )
        parent_conn.close()
        return None
    if parent_conn.poll():
        result = parent_conn.recv()
        parent_conn.close()
        if result[0] == "error":
            print(f"    error: {result[1]}", file=sys.stderr)
            return None
        return result[1]
    parent_conn.close()
    return None


def _measure_memory(load_fn):
    """Measure memory held by the data structure returned by load_fn().

    Returns the current traced memory (bytes) after loading, which
    reflects the in-memory size of the loaded data structure (excluding
    temporary allocations that have been freed).
    """
    import tracemalloc

    gc.collect()
    tracemalloc.start()
    result = load_fn()  # keep reference alive during measurement
    current, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del result
    return current


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
    engine_affinity = None

    def __init__(self, newick):
        from phyloframe.legacy import alifestd_from_newick_polars

        self._newick = newick
        self._from_newick = alifestd_from_newick_polars
        self._df = None
        self._pdf = None

    _env_overrides = {
        "PHYLOFRAME_DANGEROUSLY_ASSUME_LEGACY_ALIFESTD_HAS_CONTIGUOUS_IDS_POLARS": "1",
        "PHYLOFRAME_DANGEROUSLY_ASSUME_LEGACY_ALIFESTD_IS_TOPOLOGICALLY_SORTED_POLARS": "1",
    }

    def _set_engine(self):
        import os

        import polars as pl

        pl.Config.set_engine_affinity(self.engine_affinity)
        self._saved_env = {}
        for key, val in self._env_overrides.items():
            self._saved_env[key] = os.environ.get(key)
            os.environ[key] = val

    def _reset_engine(self):
        import os

        import polars as pl

        for key in self._env_overrides:
            prev = self._saved_env.get(key)
            if prev is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prev
        pl.Config.set_engine_affinity(None)

    def load_newick(self):
        self._set_engine()
        try:
            self._df = self._from_newick(self._newick)
        finally:
            self._reset_engine()

    def save_newick(self):
        from phyloframe.legacy import alifestd_as_newick_polars

        df = self._ensure_df()
        self._set_engine()
        try:
            alifestd_as_newick_polars(df)
        finally:
            self._reset_engine()

    def preorder(self):
        from phyloframe.legacy import (
            alifestd_unfurl_traversal_preorder_polars,
        )

        df = self._ensure_df()
        alifestd_unfurl_traversal_preorder_polars(df)

    def postorder(self):
        from phyloframe.legacy._alifestd_mark_node_depth_asexual import (
            _alifestd_calc_node_depth_asexual_contiguous,
        )
        from phyloframe.legacy._alifestd_unfurl_traversal_postorder_asexual import (
            _alifestd_unfurl_traversal_postorder_asexual_fast_path,
        )

        df = self._ensure_df()
        ancestor_ids = df.get_column("ancestor_id").to_numpy()
        node_depths = _alifestd_calc_node_depth_asexual_contiguous(
            ancestor_ids,
        )
        _alifestd_unfurl_traversal_postorder_asexual_fast_path(
            ancestor_ids,
            node_depths,
        )

    def inorder(self):
        from phyloframe.legacy import (
            alifestd_unfurl_traversal_inorder_asexual,
        )

        pdf = self._ensure_pdf()
        alifestd_unfurl_traversal_inorder_asexual(pdf, mutate=True)

    def levelorder(self):
        raise NotImplementedError("levelorder not available")

    def mrca_allpairs(self):
        from phyloframe.legacy import alifestd_calc_mrca_id_matrix_asexual

        pdf = self._ensure_working_pdf()
        alifestd_calc_mrca_id_matrix_asexual(pdf, mutate=True)

    def pairwise_dist(self):
        raise NotImplementedError("pairwise distances not available")

    def memory_bytes(self):
        from phyloframe.legacy import alifestd_from_newick_polars

        newick = self._newick
        self._set_engine()
        try:
            return _measure_memory(
                lambda: alifestd_from_newick_polars(newick),
            )
        finally:
            self._reset_engine()

    def _ensure_df(self):
        if self._df is None:
            self._set_engine()
            try:
                self._df = self._from_newick(self._newick)
            finally:
                self._reset_engine()
        return self._df

    def _ensure_pdf(self):
        if self._pdf is None:
            from phyloframe.legacy import alifestd_from_newick

            self._pdf = alifestd_from_newick(self._newick)
        return self._pdf

    def _ensure_working_pdf(self):
        from phyloframe.legacy import alifestd_to_working_format

        return alifestd_to_working_format(self._ensure_pdf())


class PhyloframeInMemoryBench(PhyloframeBench):
    name = "phyloframe (in-memory)"
    engine_affinity = "in-memory"


class PhyloframeStreamingBench(PhyloframeBench):
    name = "phyloframe (streaming)"
    engine_affinity = "streaming"


class TreeswiftBench:
    name = "treeswift"

    def __init__(self, newick):
        self._newick = newick
        self._tree = None

    def load_newick(self):
        import treeswift

        self._tree = treeswift.read_tree_newick(self._newick)

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

    def memory_bytes(self):
        newick = self._newick

        def _load():
            import treeswift

            return treeswift.read_tree_newick(newick)

        return _measure_memory(_load)

    def _ensure_tree(self):
        if self._tree is None:
            import treeswift

            self._tree = treeswift.read_tree_newick(self._newick)
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

    def memory_bytes(self):
        newick = self._newick

        def _load():
            from Bio import Phylo

            Phylo.read(io.StringIO(newick), "newick")

        return _measure_memory(_load)

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

    def memory_bytes(self):
        newick = self._newick

        def _load():
            import dendropy

            dendropy.Tree.get(data=newick, schema="newick")

        return _measure_memory(_load)

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

    def memory_bytes(self):
        newick = self._newick

        def _load():
            from ete3 import Tree

            Tree(newick)

        return _measure_memory(_load)

    def _ensure_tree(self):
        if self._tree is None:
            from ete3 import Tree

            self._tree = Tree(self._newick)
        return self._tree


class CompactTreeBench:
    name = "compacttree"

    def __init__(self, newick):
        import tempfile

        tmpfile = tempfile.NamedTemporaryFile(
            mode="w", suffix=".nwk", delete=False
        )
        tmpfile.write(newick)
        tmpfile.close()
        self._tmppath = tmpfile.name
        self._tree = None

    def load_newick(self):
        from CompactTree import compact_tree

        self._tree = compact_tree(self._tmppath)

    def save_newick(self):
        t = self._ensure_tree()
        t.get_newick()

    def preorder(self):
        from CompactTree import traverse_preorder

        t = self._ensure_tree()
        for _ in traverse_preorder(t):
            pass

    def postorder(self):
        from CompactTree import traverse_postorder

        t = self._ensure_tree()
        for _ in traverse_postorder(t):
            pass

    def inorder(self):
        raise NotImplementedError("inorder not available in CompactTree")

    def levelorder(self):
        from CompactTree import traverse_levelorder

        t = self._ensure_tree()
        for _ in traverse_levelorder(t):
            pass

    def mrca_allpairs(self):
        from CompactTree import traverse_leaves

        t = self._ensure_tree()
        leaves = list(traverse_leaves(t))

        def _mrca(a, b):
            ancestors_a = set()
            while not t.is_root(a):
                ancestors_a.add(a)
                a = t.get_parent(a)
            ancestors_a.add(a)
            while b not in ancestors_a:
                b = t.get_parent(b)
            return b

        for i, a in enumerate(leaves):
            for b in leaves[i + 1 :]:
                _mrca(a, b)

    def pairwise_dist(self):
        t = self._ensure_tree()
        t.calc_distance_matrix()

    def memory_bytes(self):
        tmpname = self._tmppath

        def _load():
            from CompactTree import compact_tree

            compact_tree(tmpname)

        return _measure_memory(_load)

    def _ensure_tree(self):
        if self._tree is None:
            from CompactTree import compact_tree

            self._tree = compact_tree(self._tmppath)
        return self._tree


LIBRARIES = [
    PhyloframeBench,
    PhyloframeInMemoryBench,
    PhyloframeStreamingBench,
    TreeswiftBench,
    BiopythonBench,
    DendropyBench,
    EteBench,
    CompactTreeBench,
]


def _warmup_jit():
    """Run phyloframe operations on a tiny tree to trigger JIT compilation."""
    print("Warming up JIT...", file=sys.stderr)
    from phyloframe.legacy import (
        alifestd_from_newick,
        alifestd_from_newick_polars,
        alifestd_to_working_format,
        alifestd_unfurl_traversal_inorder_asexual,
        alifestd_unfurl_traversal_postorder_asexual,
        alifestd_unfurl_traversal_preorder_polars,
    )
    from phyloframe.legacy._alifestd_mark_node_depth_asexual import (
        _alifestd_calc_node_depth_asexual_contiguous,
    )
    from phyloframe.legacy._alifestd_unfurl_traversal_postorder_asexual import (
        _alifestd_unfurl_traversal_postorder_asexual_fast_path,
    )

    tiny = _balanced_newick(8)
    # polars path
    pldf = alifestd_from_newick_polars(tiny)
    ancestor_ids = pldf.get_column("ancestor_id").to_numpy()
    node_depths = _alifestd_calc_node_depth_asexual_contiguous(ancestor_ids)
    _alifestd_unfurl_traversal_postorder_asexual_fast_path(
        ancestor_ids,
        node_depths,
    )
    alifestd_unfurl_traversal_preorder_polars(pldf)
    # pandas path
    pdf = alifestd_from_newick(tiny)
    alifestd_unfurl_traversal_postorder_asexual(pdf, mutate=True)
    alifestd_unfurl_traversal_inorder_asexual(pdf, mutate=True)
    from phyloframe.legacy import alifestd_calc_mrca_id_matrix_asexual

    wdf = alifestd_to_working_format(pdf)
    alifestd_calc_mrca_id_matrix_asexual(wdf, mutate=True)
    print("JIT warmup complete.", file=sys.stderr)


def run_benchmarks():
    _warmup_jit()
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
                    value = None
                elif op == "memory_bytes":
                    value = timed_call(fn)
                else:
                    value = timed(fn)
                if op == "memory_bytes":
                    status = f"{value:,} B" if value is not None else "SKIP"
                else:
                    status = f"{value:.4f}s" if value is not None else "SKIP"
                print(f"    {op}: {status}", file=sys.stderr)
                results.append(
                    {
                        "library": LibClass.name,
                        "n_leaves": n_leaves,
                        "operation": op,
                        "seconds": value,
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
