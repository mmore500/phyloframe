# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "iplotx==1.8.0",
#     "marimo>=0.23.5",
#     "matplotlib==3.10.9",
#     "numpy==2.2.6",
#     "pandas==3.0.2",
#     "phyloframe==0.9.0",
#     "seaborn==0.13.2",
#     "teeplot==1.5.0",
# ]
# ///

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import iplotx as ipx
    from matplotlib import pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from teeplot import teeplot as tp

    from phyloframe import legacy as pfl

    return ipx, np, pd, pfl, plt, sns, tp


@app.cell
def _(ipx, pd, pfl, plt, sns):
    def plot(df_raw: pd.DataFrame, df_res: pd.DataFrame) -> plt.Figure:
        style = dict(
            layout="vertical",
            layout_orientation="ascending",
            margins=0.1,
            strip_axes=False,
            vertex_facecolor="steelblue",
            vertex_labels=True,
            vertex_label_hmargin=-8,
            vertex_label_vmargin=-7,
            vertex_size=8,
            vertex_zorder=10,
        )
        ipx_wrap = pfl.alifestd_to_iplotx_pandas

        fig, (ax_raw, ax_res) = plt.subplots(
            1, 2, figsize=(4, 2), layout="constrained", sharey=True
        )
        ipx.tree(df_raw.pipe(ipx_wrap), ax=ax_raw, **style, title="before")
        ipx.tree(df_res.pipe(ipx_wrap), ax=ax_res, **style, title="after")

        sns.despine(fig, bottom=True)
        ax_raw.get_xaxis().set_visible(False)
        ax_res.get_xaxis().set_visible(False)
        ax_res.invert_yaxis()
        return fig

    return (plot,)


@app.cell
def _(np, pfl, plot, tp):
    df_raw = pfl.alifestd_from_newick("(((r:1)c:2,(x:2,y:1)e:1.5)s:2)a;")
    df_res = (
        df_raw.drop(columns=["branch_length", "origin_time_delta"])
        .pipe(  # reroot at node "r"
            pfl.alifestd_reroot_at_id_asexual,
            new_root_id=df_raw.query("taxon_label == 'r'")["id"].item(),
        )
        .pipe(  # flip rerooted lengths
            lambda df: df.assign(
                branch_length=np.where(
                    df_raw.loc[df["id"], "ancestor_id"] != df["ancestor_id"],
                    df_raw.loc[df["ancestor_id"], "branch_length"],
                    df_raw.loc[df["id"], "branch_length"],
                ),
            ),
        )
        .pipe(pfl.alifestd_to_working_format)  # reassign id values
        .pipe(  # accumulate branch lengths to mark origin times
            pfl.alifestd_mark_lineage_cumsum_asexual,
            mark_as="origin_time",
            values="branch_length",
        )
        .pipe(pfl.alifestd_sort_children_asexual, criterion="taxon_label")
        .pipe(pfl.alifestd_to_working_format)  # reassign id values
        .pipe(pfl.alifestd_ultrametricize, method="extend")
    )

    tp.tee(plot, df_raw, df_res)
    return


if __name__ == "__main__":
    app.run()
