# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.23.5",
#     "matplotlib==3.10.9",
#     "numpy==2.2.6",
#     "phyloframe[jit]==0.9.0",
#     "polars==1.34.0",
#     "seaborn==0.13.2",
#     "teeplot==1.5.0",
# ]
# ///

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _():
    from matplotlib import pyplot as plt
    import numpy as np
    import polars as pl
    import seaborn as sns
    from teeplot import teeplot as tp

    from phyloframe import _auxlib as pfa
    from phyloframe import legacy as pfl

    return np, pfa, pfl, pl, plt, sns, tp


@app.cell
def _(np, pfa):
    @pfa.jit(cache=False, nopython=True)  # JIT compile via Numba, if available
    def simulate_trait(ancestor_ids: np.ndarray) -> np.ndarray:
        trait = np.zeros(ancestor_ids.size, dtype=float)
        for id_, anc_id in enumerate(ancestor_ids):
            if id_ == anc_id:
                continue  # exclude root
            trait[id_] = trait[anc_id] + np.random.normal()
        return trait

    return (simulate_trait,)


@app.cell
def _(pl, plt, sns):
    def plot_trait(data: pl.DataFrame) -> plt.Axes:
        selectors = dict(x="node_depth", y="trait", hue="is_leaf")
        style = dict(legend=False, palette=["gray", "steelblue"], s=15)
        ax = sns.scatterplot(data, **selectors, **style)

        depth, ancestor, trait = data[["node_depth", "ancestor_id", "trait"]]
        segments = [[depth, depth[ancestor]], [trait, trait[ancestor]]]
        ax.plot(*segments, color="#CCCCCCCC", zorder=-2)  # link parent/child

        sns.despine()
        return ax

    return (plot_trait,)


@app.cell
def _(pfl, pl, plot_trait, simulate_trait, tp):
    def run():
        return (
            pfl.alifestd_make_edge_split_polars(n_leaves=100, seed=42)  # random tree
            .pipe(pfl.alifestd_topological_sort_polars)  # parents before children
            .pipe(pfl.alifestd_assign_contiguous_ids_polars)  # reassign ids
            .pipe(pfl.alifestd_mark_node_depth_polars)  # add node_depth col
            .pipe(pfl.alifestd_mark_leaves_polars)  # add is_leaf col
            .with_columns(  # add trait col
                trait=pl.col("ancestor_id").map_batches(
                    lambda x: simulate_trait(x.to_numpy()),
                    return_dtype=float,
                ),
            )
            .pipe(plot_trait)  # draw tree
        )

    tp.tee(run)
    return


if __name__ == "__main__":
    app.run()
