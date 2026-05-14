// External shim that exposes compact_tree::find_mrca to Python.
//
// The CompactTree PyPI wheel's SWIG interface does not include
// std_unordered_set.i, so the native find_mrca(std::unordered_set<...>)
// is unreachable from Python. This shim builds the unordered_set on the
// C++ side from a flat uint32_t array. It is compiled at benchmark
// warmup against the compact_tree.h header that ships with the pip
// install, so no patching of the installed package is required.
//
// Compile: g++ -O3 -fPIC -shared -std=c++17 ${this} \
//              -I<dir-of-compact_tree.h> -o ${shim.so}

#include "compact_tree.h"
#include <cstddef>
#include <cstdint>
#include <unordered_set>

extern "C" std::uint32_t phyloframe_compacttree_find_mrca(
    void* tree_ptr,
    const std::uint32_t* nodes,
    std::size_t n
) {
    auto* t = static_cast<compact_tree*>(tree_ptr);
    std::unordered_set<std::uint32_t> node_set(nodes, nodes + n);
    return t->find_mrca(node_set);
}
