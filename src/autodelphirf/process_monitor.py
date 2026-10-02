"""Target-free monitoring of changes in the reporting process."""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_monitor_index(prepared: pd.DataFrame, cutoff: pd.Timestamp) -> dict:
    """Group cutoff-visible rows by location once per origin.

    The reporting process is a property of a location's releases, not of one
    particular forecast lag.  A triangle normally contains exactly one row for
    each ``(location, lag, report_date)``.  Grouping by lag before making daily
    summaries therefore makes report volume and active-episode count constants
    (the failure observed on NHSN-28).  The index deliberately retains every
    lag for a location; ``lag`` remains an argument of the public monitor API
    only because alerts are attached to lag-specific forecast cases.
    """
    if "report_date" not in prepared:
        return {}
    visible = prepared[prepared.report_date <= cutoff]
    # Store row POSITIONS per group, not materialised frames. Holding ~3,340
    # DataFrames per origin costs enough memory/GC pressure to measurably slow
    # the RED loop that runs alongside it; positional arrays are ~2 orders of
    # magnitude lighter and the per-pair slice is rebuilt cheaply on demand.
    return {"__frame__": visible,
            "__groups__": visible.groupby("geo_value", observed=True).indices}


#: A prior-state spread at or below this multiple of the states' own magnitude
#: counts as NO historical variation. Prespecified and frozen: it exists to
#: recognise a constant feature, never to tune sensitivity.
RELATIVE_VARIATION_TOLERANCE = 1e-9


def _historical_scale(values: np.ndarray) -> float:
    """1.4826*MAD -> IQR/1.349 -> SD -> unavailable (NaN).

    "A feature with no historical variation is marked unavailable rather than
    stabilized by an arbitrarily small denominator."

    Constancy is judged RELATIVE to the magnitude of the states themselves.
    A strict ``> 0`` test is not enough: a rolling-window summary of a feature
    that is genuinely constant still differs between windows at floating-point
    size, and dividing by a 1e-17 spread produced scores of order 1e15 on real
    nhsn28 data -- trading the old saturation defect for an explosion. Scaling
    the tolerance by the states' own level keeps genuinely small-magnitude
    features (signed revision magnitudes of order 1e-5) available while
    recognising a constant one.
    """
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float("nan")
    level = float(np.median(np.abs(values)))
    floor = RELATIVE_VARIATION_TOLERANCE * (level if level > 0 else 1.)
    mad = float(np.median(np.abs(values - np.median(values))))
    if mad > floor:
        return 1.4826 * mad
    q1, q3 = np.quantile(values, [.25, .75])
    if q3 - q1 > floor:
        return float(q3 - q1) / 1.349
    sd = float(np.std(values, ddof=1))
    return sd if sd > floor else float("nan")


def process_state_monitor(prepared: pd.DataFrame, cutoff: pd.Timestamp, location: str, lag: int,
                          recent_days: int = 28, history_days: int = 84, *,
                          min_recent_dates: int = 4, min_history_dates: int = 8,
                          watch_threshold: float = 3., rediagnose_threshold: float = 5.,
                          epsilon_scale: float = 1e-8,
                          monitor_index: dict | None = None,
                          return_details: bool = False):
    """Target-free released-report monitor; it never changes the point forecast.

    Implements the monitor of ``RRDelphiRF_post_prediction_revised_harm.tex``.
    The object monitored is the SEQUENCE of process states

        Z_sr = g_r(X_ur : u in W_s),

    one window summary per report date, compared against the distribution of
    its OWN earlier states:

        mu_sr    = median{Z_ur : u < s}
        sigma_sr = 1.4826*MAD -> IQR/1.349 -> SD -> unavailable
        d_sr     = |Z_sr - mu_sr| / sigma_sr
        D_s      = max_r d_sr,   r* = argmax_r d_sr.

    Two deliberate departures from the previous implementation, both required
    by the spec:

    * It no longer compares a recent window median with a reference window
      median. "The previous recent-median-versus-reference-median construction
      is not used as the default because discrete or stable reporting features
      can produce identical medians and hence a degenerate all-zero shift
      score even when their distributions differ."
    * The score is UNCAPPED. "The scientific score is explicitly uncapped: no
      saturation such as min(c, D_s^shift) is permitted before storage or
      validation." The old cap at the re-diagnose boundary put 21-43% of cases
      exactly at the maximum across six datasets, which destroyed all ordering
      in precisely the tail the monitor exists to rank.

    ``watch_threshold``/``rediagnose_threshold`` now label the returned state
    for display only; they never truncate the stored score. ``epsilon_scale``
    is retained for signature compatibility and is unused: a feature with no
    historical variation is reported unavailable instead.
    """
    if "report_date" not in prepared:
        return "insufficient-history", np.nan
    if monitor_index is not None:
        positions = monitor_index["__groups__"].get(location)
        if positions is None:
            positions = monitor_index["__groups__"].get(str(location))
        if positions is None or not len(positions):
            result = ("insufficient-history", np.nan, {})
            return result if return_details else result[:2]
        x = monitor_index["__frame__"].take(positions)
    else:
        x = prepared[(prepared.geo_value.eq(location)) & (prepared.report_date <= cutoff)]
    x = x.copy()
    # One target-free process-summary vector per released report date: X_sr.
    aggregation = {"report_volume": ("geo_value", "size")}
    if "reference_date" in x: aggregation["active_reference_dates"] = ("reference_date", "nunique")
    if "genuine_event" in x: aggregation["revision_event_share"] = ("genuine_event", "mean")
    daily = x.groupby("report_date", observed=True).agg(**aggregation)
    source = "log_delta_value_7dav_lag7"
    if source in x:
        delta = pd.to_numeric(x[source], errors="coerce")
        x = x.assign(_delta=delta, _abs_delta=delta.abs(), _zero_delta=(delta.abs() <= 1e-12))
        grouped = x.groupby("report_date", observed=True)
        # Means/shares retain sparse changes. Daily medians make an otherwise
        # real revision process identically zero whenever most rows are zero.
        daily["zero_change_share"] = grouped["_zero_delta"].mean()
        daily["signed_revision_magnitude"] = grouped["_delta"].mean()
        daily["absolute_revision_magnitude"] = grouped["_abs_delta"].mean()
    if len(daily):
        gaps = pd.Series(daily.index, index=daily.index).diff().dt.days
        daily["report_gap_days"] = gaps.fillna(1.).astype(float)
        daily["gap_frequency"] = (daily.report_gap_days > 1).astype(float)
    daily = daily.sort_index()
    if not len(daily):
        result = ("insufficient-history", np.nan, {})
        return result if return_details else result[:2]
    # Z_sr: the prespecified window summary g_r is the mean over the recent
    # window ending at each report date, for the reason given above -- a median
    # collapses a sparse-but-real revision process to a constant.
    window = f"{int(recent_days)}D"
    states = daily.astype(float).rolling(window, min_periods=int(min_recent_dates)).mean()
    current_date = states.index[-1]
    current = states.loc[current_date]
    prior = states.loc[states.index < current_date]
    # Only states whose own window lies inside the retained history contribute.
    prior = prior.loc[prior.index > current_date - pd.Timedelta(days=int(recent_days + history_days))]
    if not len(prior):
        result = ("insufficient-history", np.nan, {})
        return result if return_details else result[:2]
    features = {}
    for column in states.columns:
        value = float(current.get(column, np.nan))
        earlier = prior[column].to_numpy(float)
        earlier = earlier[np.isfinite(earlier)]
        if not np.isfinite(value) or earlier.size < int(min_history_dates):
            continue
        scale = _historical_scale(earlier)
        if not np.isfinite(scale) or scale <= 0:
            continue                      # no historical variation: unavailable
        features[column] = float(abs(value - float(np.median(earlier))) / scale)
    if not features:
        result = ("insufficient-history", np.nan, {})
        return result if return_details else result[:2]
    score = float(max(features.values()))
    state = "stable" if score < watch_threshold else "watch" if score < rediagnose_threshold else "re-diagnose"
    result = (state, score, features)
    return result if return_details else result[:2]
