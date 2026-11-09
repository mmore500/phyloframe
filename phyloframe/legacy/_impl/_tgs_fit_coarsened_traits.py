import math
import typing
import warnings

import numpy as np
import polars as pl

from ._tgs_calc_nll import tgs_calc_nll
from ._tgs_fit_trait import tgs_fit_trait
from ._tgs_minimize_nll import tgs_minimize_nll, tgs_pick_selection_rescale


def _tgs_calc_lrt(*, null_nll: float, free_nll: float) -> float:
    """Implementation detail for `_tgs_calc_trait_gain_ratio_lrt` and
    `_tgs_calc_trait_selection_lrt`. Acronym tgs stands for trait
    gain/selection.

    Returns the likelihood-ratio statistic, 2 * (`null_nll` - `free_nll`),
    clipped at 0, warning if the null fit is better beyond optimizer
    tolerance, as then the free fit missed its optimum.
    """
    if null_nll < free_nll and not math.isclose(
        null_nll, free_nll, rel_tol=1e-8, abs_tol=1e-8
    ):
        warnings.warn(
            "tgs fit optimizer found a worse free fit than constrained fit; "
            "clipping likelihood-ratio statistic to 0, but it may be "
            "unreliable.",
        )
    return float(np.clip(2 * (null_nll - free_nll), 0.0, None))


def _tgs_calc_trait_gain_ratio_lrt(
    *,
    nll_kwargs: typing.Dict[str, typing.Any],
    gain_rate_trait: float,
    gain_rate_neutral: float,
    selection_coef: float,
    loss_rate: float,
    selection_coef_starts: typing.Sequence[float],
) -> float:
    """Implementation detail for `_tgs_calc_trait_gain_ratio_lrts`. Acronym
    tgs stands for trait gain/selection.

    Returns one trait's gain ratio likelihood-ratio statistic (see
    `_tgs_calc_trait_gain_ratio_lrts`), given its fitted gain rate,
    `gain_rate_trait`, and selection coefficient, and the mean gain rate of
    its neutral reference, `gain_rate_neutral`; nan if its gain ratio is
    undefined.
    """
    if not (gain_rate_trait > 0 and np.isfinite(gain_rate_neutral)):
        return np.nan
    selection_rescale = tgs_pick_selection_rescale(nll_kwargs["coarsened_df"])
    free_nll, __, __ = tgs_calc_nll(
        **nll_kwargs,
        gain_rate=gain_rate_trait,
        growth_rate=selection_coef - loss_rate,
    )
    null = tgs_minimize_nll(
        nll_kwargs=nll_kwargs,
        # from the free fit, then from each of `selection_coef_starts`
        starts=[
            (np.log(gain_rate_neutral), start)
            for start in (
                selection_coef / selection_rescale,
                *selection_coef_starts,
            )
        ],
        fit_gain=False,
        fit_selection=True,
        selection_rescale=selection_rescale,
        loss_rate=loss_rate,
    )
    if not null.num_converged:
        warnings.warn(
            "tgs fit optimizer did not converge with gain rate fixed at the "
            "mean gain rate of its neutral reference traits; gain ratio "
            "likelihood-ratio statistic may be unreliable.",
        )
    return _tgs_calc_lrt(null_nll=null.fun, free_nll=free_nll)


def _tgs_calc_trait_gain_ratio_lrts(
    *,
    fits_df: pl.DataFrame,
    trait_nll_kwargs: typing.Sequence[typing.Dict[str, typing.Any]],
    loss_rate: float,
    selection_coef_starts: typing.Sequence[float],
) -> pl.DataFrame:
    """Implementation detail for `alifestd_tgs_estimate_polars`. Acronym tgs
    stands for trait gain/selection.

    Returns, per trait, the likelihood-ratio test (LRT) statistic for gain
    ratio = 1, a profile likelihood test: twice the increase in negative
    log likelihood from holding the trait's gain rate at the mean fitted
    gain rate of its neutral reference, `gain_rate_neutral`,
    and refitting the selection coefficient alone; see Output Schema.
    `fits_df` holds each trait's fitted `gain_rate` and `selection_coef`,
    and its neutral reference's `gain_rate_neutral`, and `trait_nll_kwargs`
    each trait's `tgs_calc_nll`
    arguments but the rate parameters.

    The statistic naively treats the reference gain rate as known, and
    exposure episodes as independent, so overstates significance.

    Input Schema
    ------------
    `fits_df` columns, one row per trait, e.g., from `tgs_fit_trait`:

    'gain_rate' : float
        Fitted gain rate.
    'selection_coef' : float
        Fitted selection coefficient.
    'gain_rate_neutral' : float
        Mean gain rate of the trait's neutral reference; nan if undefined
        (see `_tgs_calc_neutral_reference_gain_rates`).

    Output Schema
    -------------
    'trait_id' : integer
        Trait index.
    'gain_ratio_lrt_naive' : float
        The statistic; nan if the gain ratio is undefined, or the trait has
        no gains.
    """
    return pl.DataFrame(
        [
            {
                "trait_id": trait_id,
                "gain_ratio_lrt_naive": _tgs_calc_trait_gain_ratio_lrt(
                    nll_kwargs=nll_kwargs,
                    gain_rate_trait=fit["gain_rate"],
                    gain_rate_neutral=fit["gain_rate_neutral"],
                    selection_coef=fit["selection_coef"],
                    loss_rate=loss_rate,
                    selection_coef_starts=selection_coef_starts,
                ),
            }
            for trait_id, (nll_kwargs, fit) in enumerate(
                zip(trait_nll_kwargs, fits_df.iter_rows(named=True))
            )
        ],
        schema={"trait_id": pl.Int64, "gain_ratio_lrt_naive": pl.Float64},
    )


def _tgs_calc_trait_selection_lrt(
    *,
    nll_kwargs: typing.Dict[str, typing.Any],
    gain_rate: float,
    selection_coef: float,
    loss_rate: float,
) -> float:
    """Implementation detail for `_tgs_calc_trait_selection_lrts`. Acronym
    tgs stands for trait gain/selection.

    Returns one trait's selection likelihood-ratio statistic (see
    `_tgs_calc_trait_selection_lrts`), given its fitted `gain_rate` and
    `selection_coef`, or nan if it has no gains.
    """
    # without gains, the selection coefficient is undefined, so untestable
    if not gain_rate > 0:
        return np.nan
    selection_rescale = tgs_pick_selection_rescale(nll_kwargs["coarsened_df"])
    free_nll, __, __ = tgs_calc_nll(
        **nll_kwargs,
        gain_rate=gain_rate,
        growth_rate=selection_coef - loss_rate,
    )
    # under the null hypothesis of no selection, refit the log gain rate
    # alone, with selection held at 0; one start, from the free fit,
    # suffices, as the negative log likelihood is then convex in log gain
    # rate (each bin's gain probability is gain rate times a fixed exposure)
    null = tgs_minimize_nll(
        nll_kwargs=nll_kwargs,
        starts=[(np.log(gain_rate), 0.0)],  # (log gain rate, selection)
        fit_gain=True,
        fit_selection=False,
        selection_rescale=selection_rescale,
        loss_rate=loss_rate,
    )
    if not null.num_converged:
        warnings.warn(
            "tgs fit optimizer did not converge under the null hypothesis "
            f"of no selection ({null.message}); likelihood-ratio statistic "
            "may be unreliable.",
        )
    # two-tailed, testing selection coefficient != 0 in either direction
    return _tgs_calc_lrt(null_nll=null.fun, free_nll=free_nll)


def _tgs_calc_trait_selection_lrts(
    *,
    fits_df: pl.DataFrame,
    trait_nll_kwargs: typing.Sequence[typing.Dict[str, typing.Any]],
    loss_rate: float,
) -> pl.DataFrame:
    """Implementation detail for `alifestd_tgs_estimate_polars`. Acronym tgs
    stands for trait gain/selection.

    Returns, per trait, the likelihood-ratio test (LRT) statistic for
    selection coefficient = 0: twice the increase in negative log
    likelihood from fixing the selection coefficient at 0 and refitting
    the gain rate alone; see Output Schema. `fits_df` holds each trait's
    fitted `gain_rate` and `selection_coef`, and `trait_nll_kwargs` each
    trait's `tgs_calc_nll` arguments but the rate parameters.

    The statistic naively treats exposure episodes as independent, so
    overstates significance.

    Input Schema
    ------------
    `fits_df` columns, one row per trait, e.g., from `tgs_fit_trait`:

    'gain_rate' : float
        Fitted gain rate.
    'selection_coef' : float
        Fitted selection coefficient.

    Output Schema
    -------------
    'trait_id' : integer
        Trait index.
    'selection_coef_lrt_naive' : float
        The statistic; nan if the trait has no gains.
    """
    return pl.DataFrame(
        [
            {
                "trait_id": trait_id,
                "selection_coef_lrt_naive": _tgs_calc_trait_selection_lrt(
                    nll_kwargs=nll_kwargs,
                    gain_rate=fit["gain_rate"],
                    selection_coef=fit["selection_coef"],
                    loss_rate=loss_rate,
                ),
            }
            for trait_id, (nll_kwargs, fit) in enumerate(
                zip(trait_nll_kwargs, fits_df.iter_rows(named=True))
            )
        ],
        schema={"trait_id": pl.Int64, "selection_coef_lrt_naive": pl.Float64},
    )


def _tgs_calc_neutral_reference_gain_rate(
    *,
    trait_id: int,
    gain_rates: np.ndarray,
    neutral_traits: typing.Sequence[int],
) -> float:
    """Implementation detail for `_tgs_calc_neutral_reference_gain_rates`.
    Acronym tgs stands for trait gain/selection.

    Returns trait `trait_id`'s neutral reference gain rate, the mean gain
    rate of the neutral traits other than itself; nan if none of them
    gained their trait.
    """
    # a neutral trait is not compared against itself
    refs = [ref for ref in neutral_traits if ref != trait_id]
    ref_gain_rate = gain_rates[refs].mean()
    return ref_gain_rate if ref_gain_rate > 0 else np.nan


def _tgs_calc_neutral_reference_gain_rates(
    *,
    gain_rates: np.ndarray,
    neutral_traits: typing.Sequence[int],
) -> np.ndarray:
    """Implementation detail for `alifestd_tgs_estimate_polars`. Acronym tgs
    stands for trait gain/selection.

    Returns each trait's neutral reference gain rate, the mean gain rate of
    the neutral traits other than itself, against which its gain ratio is
    taken; nan if none of them gained their trait.

    Raises ValueError with fewer than two distinct neutral traits, so that
    every trait has a neutral reference.
    """
    if len(set(neutral_traits)) < 2:
        raise ValueError(
            "at least two distinct neutral traits are required, so that "
            "every trait has a neutral reference",
        )
    gain_rates = np.asarray(gain_rates, dtype=np.float64)
    return np.array(
        [
            _tgs_calc_neutral_reference_gain_rate(
                trait_id=trait_id,
                gain_rates=gain_rates,
                neutral_traits=neutral_traits,
            )
            for trait_id in range(len(gain_rates))
        ],
        dtype=np.float64,
    )


def tgs_fit_coarsened_traits(
    *,
    trait_nll_kwargs: typing.Sequence[typing.Dict[str, typing.Any]],
    neutral_traits: typing.Sequence[int],
    loss_rate: float,
    selection_coef_starts: typing.Sequence[float],
    progress_wrap: typing.Callable,
) -> pl.DataFrame:
    """Implementation detail for `alifestd_tgs_estimate_polars`. Acronym tgs
    stands for trait gain/selection.

    Fits each trait's gain rate and selection coefficient, given each
    trait's `tgs_calc_nll` arguments but the rate parameters,
    `trait_nll_kwargs`, e.g., its coarsened episodes from `_tgs_coarsen_episodes` (see
    Input Schema of `tgs_calc_nll`).

    Fits one trait at a time, wrapped in `progress_wrap` (e.g., tqdm).
    Returns one row per trait, in trait index order; see Output Schema.

    Output Schema
    -------------
    'gain_rate' : float
        Fitted gain rate, per unit branch length; 0 without gains.
    'selection_coef' : float
        Fitted selection coefficient; nan without gains.
    'selection_coef_lrt_naive' : float
        Likelihood-ratio statistic for selection coefficient = 0; see
        `_tgs_calc_trait_selection_lrts`.
    'gain_ratio' : float
        Gain rate relative to the mean gain rate of the trait's neutral
        reference; see `_tgs_calc_neutral_reference_gain_rates`.
    'gain_ratio_lrt_naive' : float
        Likelihood-ratio statistic for gain ratio = 1; see
        `_tgs_calc_trait_gain_ratio_lrts`.
    """
    fits_df = pl.DataFrame(
        [
            tgs_fit_trait(
                nll_kwargs=nll_kwargs,
                loss_rate=loss_rate,
                selection_coef_starts=selection_coef_starts,
            )
            for nll_kwargs in progress_wrap(trait_nll_kwargs)
        ],
        schema={"gain_rate": pl.Float64, "selection_coef": pl.Float64},
    )
    # normalize each gain rate by the mean gain rate of its neutral reference;
    # undefined (nan) without neutral reference traits, or if none gained the
    # trait
    fits_df = fits_df.with_columns(
        trait_id=pl.int_range(pl.len(), dtype=pl.Int64),
        gain_rate_neutral=_tgs_calc_neutral_reference_gain_rates(
            gain_rates=fits_df["gain_rate"].to_numpy(),
            neutral_traits=neutral_traits,
        ),
    ).with_columns(
        gain_ratio=pl.col("gain_rate") / pl.col("gain_rate_neutral")
    )
    lrt_kwargs = dict(
        fits_df=fits_df, trait_nll_kwargs=trait_nll_kwargs, loss_rate=loss_rate
    )
    return (
        fits_df.join(
            _tgs_calc_trait_selection_lrts(**lrt_kwargs),
            on="trait_id",
            maintain_order="left",
        )
        .join(
            _tgs_calc_trait_gain_ratio_lrts(
                **lrt_kwargs, selection_coef_starts=selection_coef_starts
            ),
            on="trait_id",
            maintain_order="left",
        )
        .select(
            "gain_rate",
            "selection_coef",
            "selection_coef_lrt_naive",
            "gain_ratio",
            "gain_ratio_lrt_naive",
        )
    )
