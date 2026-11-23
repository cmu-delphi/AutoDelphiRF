"""Re-assessment of issued forecasts against newer forecasts of the same target.

On every release ``r``, each forecast of a target issued at ``s`` with age
``a = r - s`` in ``1..horizon`` days is re-assessed against the forecast of the
same target issued on ``r``:

* flag  ``F_{s,r} = 1`` if the newest median lies outside the earlier forecast's
  central 80% interval (``median_r < lower_s`` or ``median_r > upper_s``);
* score ``J_{s,r} = |median_r - median_s| / (upper_s - lower_s)``, undefined
  (NaN) when the earlier interval has zero width.

Each ``(s, r)`` pair is one decision, graded by the earlier forecast's interval
miss (the target value outside its 80% interval). The forecast issued on ``r``
is not assessed on ``r``; it is the reference for earlier forecasts and is
re-assessed on later releases. Flags use only forecasts issued on or before
``r``; target values enter only through the ``miss`` column used to evaluate
them.

Typical use::

    decisions = reassess(forecasts)                 # one row per (s, r) decision
    by_age = evaluate_by_age(decisions)             # metrics by days since issue

Interval decisions are invariant to any strictly increasing transform, so the
columns may be on the working (e.g. log) scale; ``J`` is scale dependent and
should be computed on the scale used for model fitting.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

__all__ = ["reassess", "age_counts", "evaluate_by_age", "average_precision",
           "run_reassessment", "write_reassessment"]


def reassess(
    forecasts: pd.DataFrame,
    *,
    horizon: int = 14,
    keys: Sequence[str] = ("geo_value", "reference_date"),
    issue_col: str = "report_date",
    lower: str = "q0.1_working",
    median: str = "q0.5_working",
    upper: str = "q0.9_working",
    miss: str = "miss",
    target_col: str | None = "target_date",
) -> pd.DataFrame:
    """Return one row per re-assessment decision.

    Parameters
    ----------
    forecasts
        One row per issued forecast: target ``keys``, issue date ``issue_col``,
        the earlier forecast's ``lower``/``median``/``upper`` quantiles and its
        interval-miss indicator ``miss`` (1/0, NaN if unknown). Each target must
        have at most one forecast per issue date.
    horizon
        Largest age (days since issue) at which a forecast is re-assessed.
    target_col
        Date at which the target value becomes available; if given, the output
        includes ``lead_days = target_date - release_date``.

    Returns
    -------
    DataFrame with columns ``release_date``, ``issue_date``, ``age`` (days),
    ``miss``, ``flag`` (0/1), ``jump`` (J, NaN for zero-width intervals) and,
    when ``target_col`` is given, ``lead_days``.
    """
    keys = list(keys)
    cols = keys + [issue_col, lower, median, upper, miss] + ([target_col] if target_col else [])
    f = forecasts[cols].copy()
    f[issue_col] = pd.to_datetime(f[issue_col])
    if f.duplicated(keys + [issue_col]).any():
        raise ValueError("forecasts must have at most one row per target and issue date")
    f = f.sort_values(keys + [issue_col]).reset_index(drop=True)
    lo, med, hi = (f[c].to_numpy(float) for c in (lower, median, upper))
    width = hi - lo
    ok_old = np.isfinite(lo) & np.isfinite(med) & np.isfinite(hi)
    g = f.groupby(keys, sort=False)
    parts = []
    k = 0
    while True:
        k += 1
        nxt = g[[issue_col, median]].shift(-k)
        age = (nxt[issue_col] - f[issue_col]).dt.days.to_numpy(float)
        in_window = np.isfinite(age) & (age <= horizon)
        if not in_window.any():
            break
        new = nxt[median].to_numpy(float)
        ok = in_window & (age >= 1) & ok_old & np.isfinite(new)
        i = np.flatnonzero(ok)
        if not len(i):
            continue
        out = pd.DataFrame({
            "release_date": nxt[issue_col].to_numpy()[i],
            "issue_date": f[issue_col].to_numpy()[i],
            "age": age[i].astype(np.int16),
            "miss": f[miss].to_numpy(float)[i].astype(np.float32),
            "flag": ((new[i] < lo[i]) | (new[i] > hi[i])).astype(np.int8),
            "jump": np.where(width[i] > 0, np.abs(new[i] - med[i]) / np.where(width[i] > 0, width[i], 1.0),
                             np.nan).astype(np.float32),
        })
        for key in keys:
            out[key] = f[key].to_numpy()[i]
        if target_col:
            out["lead_days"] = (pd.to_datetime(f[target_col].to_numpy()[i]) - out.release_date).dt.days.astype(np.int32)
        parts.append(out)
    if not parts:
        empty = {c: [] for c in [*keys, "release_date", "issue_date", "age", "miss", "flag", "jump"]}
        if target_col:
            empty["lead_days"] = []
        return pd.DataFrame(empty)
    return pd.concat(parts, ignore_index=True)


def age_counts(decisions: pd.DataFrame) -> pd.DataFrame:
    """Per (release date, age) counts ``n`` (graded decisions), ``y`` (misses),
    ``f`` (flags) and ``tp`` (flagged misses): enough to bootstrap over release dates."""
    d = decisions[decisions.miss.notna()]
    y = d.miss.to_numpy() == 1
    fl = d.flag.to_numpy() == 1
    c = pd.DataFrame({"release_date": d.release_date.to_numpy(), "age": d.age.to_numpy(), "n": 1,
                      "y": y.astype(int), "f": fl.astype(int), "tp": (y & fl).astype(int)})
    return c.groupby(["release_date", "age"], as_index=False).sum()


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    """Average precision (step-wise area under the precision-recall curve), with
    tied scores treated as one threshold; equals ``sklearn.metrics.average_precision_score``."""
    y = np.asarray(y, bool)
    score = np.asarray(score, float)
    if not y.any():
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    s, yy = score[order], y[order]
    last = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]       # last index of each distinct score
    tps = np.cumsum(yy)[last]
    precision = tps / (last + 1)
    recall = tps / tps[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def evaluate_by_age(
    decisions: pd.DataFrame,
    *,
    n_boot: int = 1000,
    seed: int | None = 20261121,
    rng: np.random.Generator | None = None,
    min_scored: int = 50,
) -> pd.DataFrame:
    """Flag and score metrics by age, with bootstrap intervals over release dates.

    Precision, false omission rate (FOR) and sensitivity get 95% percentile
    intervals from ``n_boot`` resamples of complete release dates. PR-AUC ranks
    decisions by ``jump`` (decisions with undefined ``jump`` are excluded;
    reported only with more than ``min_scored`` scored decisions and both
    outcomes present); its no-skill value is ``base``. ``lead`` is the median
    ``lead_days`` of correctly flagged decisions. Pass ``rng`` to share one
    random stream across several calls.
    """
    rng = rng if rng is not None else np.random.default_rng(seed)
    graded = decisions[decisions.miss.notna()]
    counts = age_counts(graded)
    rows = []
    for a, q in counts.groupby("age"):
        n, y, f, tp = (q[k].to_numpy(float) for k in ("n", "y", "f", "tp"))
        m = len(q)
        prec, sens, fo = [], [], []
        for _ in range(n_boot):
            w = np.bincount(rng.integers(0, m, m), minlength=m)
            prec.append((w @ tp) / max(w @ f, 1))
            sens.append((w @ tp) / max(w @ y, 1))
            fo.append((w @ (y - tp)) / max(w @ (n - f), 1))
        N, Y, F, TP = n.sum(), y.sum(), f.sum(), tp.sum()
        d = graded[graded.age == a]
        yy, ff, sc = d.miss.to_numpy() == 1, d.flag.to_numpy() == 1, d.jump.to_numpy(float)
        s_ok = np.isfinite(sc)
        pr_auc = (average_precision(yy[s_ok], sc[s_ok])
                  if s_ok.sum() > min_scored and 0 < yy[s_ok].sum() < s_ok.sum() else np.nan)
        lead = (float(np.median(d.lead_days.to_numpy()[ff & yy]))
                if "lead_days" in d and (ff & yy).any() else np.nan)
        q2 = lambda v, p: float(np.quantile(v, p)) if n_boot else np.nan
        rows.append(dict(
            age=int(a), n=int(N), releases=m, base=Y / N, flag_rate=F / N,
            precision=TP / F if F else np.nan, prec_lo=q2(prec, .025), prec_hi=q2(prec, .975),
            false_omission=(Y - TP) / (N - F) if N > F else np.nan, for_lo=q2(fo, .025), for_hi=q2(fo, .975),
            sensitivity=TP / Y if Y else np.nan, sens_lo=q2(sens, .025), sens_hi=q2(sens, .975),
            specificity=(N - Y - (F - TP)) / (N - Y) if N > Y else np.nan,
            pr_auc=pr_auc, lead=lead))
    return pd.DataFrame(rows, columns=[
        "age", "n", "releases", "base", "flag_rate", "precision", "prec_lo", "prec_hi",
        "false_omission", "for_lo", "for_hi", "sensitivity", "sens_lo", "sens_hi",
        "specificity", "pr_auc", "lead"])


def run_reassessment(forecasts: pd.DataFrame, *, method: str = "delphirf",
                     horizon: int = 14, n_boot: int = 1000,
                     release_dates=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reassess genuine, originally issued forecasts; grade only resolved targets.

    ``release_dates`` limits decisions to newly issued report dates in an
    operational update. Earlier forecasts remain eligible even when their
    target is not yet known. Truth is used only for the retrospective ``miss``.
    """
    required = {"method", "geo_value", "reference_date", "report_date", "target_date",
                "genuine_event", "model_available", "truth", "q0.1", "q0.9",
                "q0.1_working", "q0.5_working", "q0.9_working"}
    missing = required - set(forecasts)
    if missing:
        raise ValueError(f"reassessment forecasts missing columns: {sorted(missing)}")
    f = forecasts[(forecasts.method == method) &
                  forecasts.genuine_event.fillna(False).astype(bool) &
                  forecasts.model_available.fillna(False).astype(bool)].copy()
    if release_dates is not None:
        dates = pd.to_datetime(pd.Index(release_dates)).unique()
        if not len(dates):
            f = f.iloc[:0]
        else:
            recent = f[pd.to_datetime(f.report_date).isin(dates)]
            targets = recent[["geo_value", "reference_date"]].drop_duplicates()
            f = f.merge(targets, on=["geo_value", "reference_date"], how="inner")
            earliest = min(dates) - pd.Timedelta(days=horizon)
            f = f[pd.to_datetime(f.report_date).ge(earliest) &
                  pd.to_datetime(f.report_date).le(max(dates))]
    quantiles = [f"q{tau:g}_working" for tau in
                 (.01, .025, .1, .25, .5, .75, .9, .975, .99)]
    if all(column in f for column in quantiles):
        values = f[quantiles].to_numpy(float)
        ordered = np.where(np.isfinite(values).all(axis=1, keepdims=True),
                           np.sort(values, axis=1), np.nan)
        f[["q0.1_working", "q0.5_working", "q0.9_working"]] = ordered[:, [2, 4, 6]]
    f = f.dropna(subset=["q0.1_working", "q0.5_working", "q0.9_working"])
    truth = pd.to_numeric(f.truth, errors="coerce").to_numpy(float)
    lower = pd.to_numeric(f["q0.1"], errors="coerce").to_numpy(float)
    upper = pd.to_numeric(f["q0.9"], errors="coerce").to_numpy(float)
    tolerance = 1e-9 * np.maximum.reduce([np.abs(truth), np.abs(lower), np.abs(upper)])
    f["miss"] = np.where(np.isfinite(truth) & np.isfinite(lower) & np.isfinite(upper),
                         ((truth < lower - tolerance) | (truth > upper + tolerance)).astype(float),
                         np.nan)
    decisions = reassess(f, horizon=horizon)
    if release_dates is not None:
        decisions = decisions[pd.to_datetime(decisions.release_date).isin(dates)].copy()
    decisions.insert(0, "method", method)
    by_age = evaluate_by_age(decisions, n_boot=n_boot)
    return decisions, by_age


def write_reassessment(forecasts: pd.DataFrame, output, **kwargs) -> dict:
    """Write independently reproducible decisions and retrospective age summaries."""
    from pathlib import Path
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    decisions, by_age = run_reassessment(forecasts, **kwargs)
    decisions.to_csv(output / "reassessment_decisions.csv.gz", index=False, compression="gzip")
    by_age.to_csv(output / "reassessment_by_age.csv", index=False)
    return {"decisions": len(decisions), "graded_decisions": int(decisions.miss.notna().sum()),
            "decisions_file": str(output / "reassessment_decisions.csv.gz"),
            "by_age_file": str(output / "reassessment_by_age.csv")}
