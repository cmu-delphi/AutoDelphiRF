from __future__ import annotations
from pathlib import Path
import html
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LEVELS = {50: (.25, .75), 80: (.1, .9), 95: (.025, .975)}
TAUS = [.01, .025, .1, .25, .5, .75, .9, .975, .99]
LABELS = {"baseline_null": "Latest reported value", "delphirf": "RevRoute pooling + DelphiRF", "naive_delphirf": "Naive DelphiRF", "similarity_weighted_delphirf": "Similarity-weighted DelphiRF",
          "red": "Revision-pattern matching", "residual_calendar": "Revision-pattern matching + calendar residual RF",
          "residual_full": "Revision-pattern matching + full residual RF",
          "global_delphirf": "Global DelphiRF (location+lag FE)",
          "rr_delphirf_hard": "RevRoute Delphi-RF (task pooling)",
          "rr_delphirf3": "RevRoute Delphi-RF3 (vectorized geometry)",
          "rr_delphirf2": "RevRoute Delphi-RF2 (NN threshold)",
          "graph_fused_delphirf": "Graph-fused Delphi-RF"}
COLORS = ["#222222", "#4daf4a", "#377eb8", "#ff7f00", "#984ea3", "#e41a1c",
          "#a65628", "#17becf"]

def label(x): return LABELS.get(x, x.replace("_", " ").title())

def table(frame):
    if frame is None or frame.empty: return "<p class='muted'>No eligible rows.</p>"
    shown=frame.copy()
    if "method" in shown: shown["method"]=shown.method.map(label)
    return shown.to_html(index=False,escape=True,float_format=lambda x:f"{x:.4f}")

def validation_figures(validation, figures, palette, methods):
    """Three-axis post-prediction figures (RRDelphiRF_post_prediction_three_axis).

    The previous set was built around ``S^risk``, the categorical ``Q^pred``
    and a merged alert grid, none of which the three-axis spec forms. These
    figures instead ask, one axis at a time, whether the issued quantity
    actually captures the outcome it claims:

      A1  q_err  -- upper-tail coverage against the claimed eta, by lag
      A2  q_err  -- does realized AE rise across bins of q_err?
      B   p_harm -- reliability diagram against the identity line (+ Brier)
      C1  Dshift -- is the monitor nondegenerate?
      C2  Dshift -- does realized AE rise across bins of Dshift?
      D   the opposed-objective diagnostic: q_err ranks error but ANTI-ranks
          harm, which is why the axes are not combined
      E1  V4 heatmap, q_err x Dshift -> realized AE
      E2  V4 heatmap, p_harm x Dshift -> observed harm frequency
      F   scorecard: one verdict per axis

    Each axis is judged against its own outcome: q_err against absolute error,
    p_harm against the harm indicator, Dshift against both. No figure mixes
    them into a single score.
    """
    made = {}
    def colour(name): return palette.get(name, "#777777")
    def save(fig, key, filename):
        fig.tight_layout(); fig.savefig(figures/filename, dpi=170); plt.close(fig)
        made[key] = filename
    def nonempty(name):
        frame = validation.get(name)
        return frame if frame is not None and not frame.empty else None

    # ---- A1: does q_err hold its upper-tail claim, overall and by lag? ----
    frame = nonempty("v2_reliability_coverage")
    if frame is not None and "coverage_error_quantity" in frame:
        by_lag = frame[frame.subset.astype(str).str.startswith("lag=")].copy()
        fig, ax = plt.subplots(figsize=(7.6, 3.4))
        eta = float(frame.eta.iloc[0]) if "eta" in frame else .9
        if len(by_lag):
            by_lag["lag"] = by_lag.subset.astype(str).str.replace("lag=", "").astype(int)
            for method, z in by_lag.sort_values("lag").groupby("method", observed=True):
                ax.plot(z.lag, z.coverage_error_quantity, "o-", ms=4, lw=1.4,
                        color=colour(method), label=f"{label(method)}  $q^{{err}}$")
                if "coverage_regret_quantity" in z:
                    ax.plot(z.lag, z.coverage_regret_quantity, "s--", ms=3.4, lw=1.1,
                            color=colour(method), alpha=.55,
                            label=f"{label(method)}  $q^{{reg}}$ (diagnostic)")
        overall = frame[frame.subset.eq("overall")]
        for _, row in overall.iterrows():
            ax.axhline(row.coverage_error_quantity, color=colour(row.method), lw=.8, alpha=.35)
        ax.axhline(eta, color="#c00000", lw=1.3)
        ax.annotate(f"claimed $\\eta$ = {eta:g}", xy=(.99, eta), xycoords=("axes fraction", "data"),
                    ha="right", va="bottom", fontsize=8, color="#c00000")
        ax.set(xlabel="lag", ylabel="realized coverage", ylim=(0, 1.02),
               title="A1  Axis 1 ($q^{err}$): is the upper-tail claim honest?")
        ax.legend(fontsize=7, ncol=2); ax.grid(alpha=.25, lw=.5)
        save(fig, "a1_qerr_coverage", "a1_qerr_coverage.png")

    # ---- A2: does realized AE rise across bins of q_err? ----
    frame = nonempty("v2_qerr_discrimination")
    if frame is not None and {"signal", "bin", "mean_ae"}.issubset(frame.columns):
        z = frame[frame.subset.eq("overall") & frame.signal.eq("q_error")].sort_values("bin")
        if len(z):
            fig, ax = plt.subplots(figsize=(6.4, 3.4))
            for method, g in z.groupby("method", observed=True):
                g = g.sort_values("bin")
                ax.plot(g.bin + 1, g.mean_ae, "o-", color=colour(method), lw=1.6, ms=5,
                        label=label(method))
                rho = g.get("spearman_signal_vs_ae")
                if rho is not None and len(rho.dropna()):
                    ax.annotate(f"$\\rho$ = {rho.dropna().iloc[0]:+.3f}",
                                xy=(.03, .92), xycoords="axes fraction", fontsize=9,
                                color=colour(method))
            ax.set(xlabel="bin of issued $q^{err}$  (low $\\rightarrow$ high)",
                   ylabel="realized mean AE",
                   title="A2  Axis 1 ($q^{err}$): does it rank prediction difficulty?")
            ax.legend(fontsize=8); ax.grid(alpha=.25, lw=.5)
            save(fig, "a2_qerr_discrimination", "a2_qerr_discrimination.png")

    # ---- B: p_harm reliability diagram ----
    frame = nonempty("v2_pharm_calibration")
    if frame is not None and {"mean_predicted_harm", "observed_harm_frequency"}.issubset(frame.columns):
        z = frame[frame.subset.eq("overall")]
        if len(z):
            fig, ax = plt.subplots(figsize=(4.9, 4.6))
            ax.plot([0, 1], [0, 1], color="#c00000", lw=1.2, zorder=1)
            ax.annotate("perfect calibration", xy=(.62, .58), fontsize=8, color="#c00000",
                        rotation=38)
            for method, g in z.groupby("method", observed=True):
                g = g.sort_values("mean_predicted_harm")
                sizes = 14 + 150 * g.n / max(g.n.max(), 1)
                ax.plot(g.mean_predicted_harm, g.observed_harm_frequency, "-",
                        color=colour(method), lw=1.4, zorder=2)
                ax.scatter(g.mean_predicted_harm, g.observed_harm_frequency, s=sizes,
                           color=colour(method), edgecolor="white", linewidth=.6, zorder=3,
                           label=label(method))
                skill = g.brier_skill_score.dropna()
                brier = g.brier_score.dropna()
                if len(brier):
                    ax.annotate(f"Brier = {brier.iloc[0]:.3f}"
                                + (f"\nskill = {skill.iloc[0]:+.3f}" if len(skill) else ""),
                                xy=(.03, .86), xycoords="axes fraction", fontsize=8,
                                color=colour(method))
            ax.set(xlabel="issued $p_{harm}$", ylabel="observed $P(\\Delta_q>0)$",
                   xlim=(0, 1), ylim=(0, 1),
                   title="B  Axis 2 ($p_{harm}$):\nis the harm probability calibrated?")
            ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=.25, lw=.5)
            ax.set_aspect("equal")
            save(fig, "b_pharm_reliability", "b_pharm_reliability.png")

    # ---- C1/C2: the process axis ----
    nondeg = nonempty("v3_process_monitor_nondegeneracy")
    process = nonempty("v3_process_alert_validation")
    if nondeg is not None or process is not None:
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.5))
        ax = axes[0]
        if nondeg is not None:
            row = nondeg.iloc[0]
            stats = [("min", row.get("min")), ("q25", row.get("q25")), ("median", row.get("median")),
                     ("q75", row.get("q75")), ("max", row.get("max"))]
            stats = [(k, v) for k, v in stats if v is not None and np.isfinite(v)]
            ax.bar([k for k, _ in stats], [v for _, v in stats], color="#4576a8", width=.62)
            verdict = str(row.get("verdict", ""))
            degenerate = bool(row.get("monitor_degenerate", False))
            ax.annotate(("DEGENERATE — monitor failure" if degenerate
                         else f"nondegenerate · {int(row.get('distinct_values', 0))} distinct values"),
                        xy=(.5, .93), xycoords="axes fraction", ha="center", fontsize=8.5,
                        color="#c00000" if degenerate else "#2e7d32")
            share = row.get("fraction_at_max")
            if share is not None and np.isfinite(share):
                ax.annotate(f"{share:.0%} of cases sit at the capped maximum",
                            xy=(.5, .84), xycoords="axes fraction", ha="center",
                            fontsize=7.5, color="#555555")
            ax.set(ylabel="$D^{shift}$", title="C1  Axis 3: is the monitor nondegenerate?")
        ax = axes[1]
        if process is not None and {"process_alert", "mean_ae", "mean_regret"}.issubset(process.columns):
            z = process[process.subset.eq("overall")]
            order = ["stable", "watch", "re-diagnose"]
            for method, g in z.groupby("method", observed=True):
                g = g.set_index("process_alert").reindex(order).dropna(subset=["mean_ae"])
                if not len(g):
                    continue
                ax.plot(range(len(g)), g.mean_ae, "o-", color=colour(method), lw=1.6, ms=5,
                        label=f"{label(method)} — mean AE")
                if g.mean_regret.abs().sum() > 0:
                    twin = ax.twinx() if not hasattr(ax, "_twin") else ax._twin
                    ax._twin = twin
                    twin.plot(range(len(g)), g.mean_regret, "s--", color=colour(method),
                              alpha=.6, lw=1.2, ms=4, label=f"{label(method)} — mean regret")
                    twin.axhline(0, color="#888888", lw=.7)
                    twin.set_ylabel("mean regret  (>0 = correction harmed)")
                ax.set_xticks(range(len(g))); ax.set_xticklabels(g.index)
            ax.set(ylabel="realized mean AE",
                   title="C2  Axis 3: does process shift track outcome?")
            ax.legend(fontsize=7, loc="upper left"); ax.grid(alpha=.25, lw=.5)
        save(fig, "c_process_axis", "c_process_axis.png")

    # ---- D: the opposed-objective diagnostic ----
    frame = nonempty("v2_opposed_objective_diagnostic")
    if frame is not None:
        z = frame[frame.subset.eq("overall")]
        if len(z):
            keys = [("spearman_qerr_vs_ae", "$q^{err}$ vs AE\n(want +)"),
                    ("spearman_qerr_vs_regret", "$q^{err}$ vs regret"),
                    ("spearman_qerr_vs_harm", "$q^{err}$ vs harm"),
                    ("spearman_pharm_vs_harm", "$p_{harm}$ vs harm\n(want +)")]
            keys = [(k, t) for k, t in keys if k in z.columns]
            fig, ax = plt.subplots(figsize=(7.2, 3.4))
            x = np.arange(len(keys)); width = .8/max(len(z), 1)
            for index, (_, row) in enumerate(z.iterrows()):
                values = [row[k] for k, _ in keys]
                ax.bar(x + (index - (len(z)-1)/2)*width, values, width,
                       color=colour(row.method), label=label(row.method))
            ax.axhline(0, color="#333333", lw=.9)
            ax.set(xticks=x, xticklabels=[t for _, t in keys], ylabel="Spearman $\\rho$",
                   ylim=(-1, 1),
                   title="D  Why the axes stay separate: difficulty ranks error but anti-ranks harm")
            ax.legend(fontsize=8); ax.grid(alpha=.25, lw=.5, axis="y")
            save(fig, "d_opposed_objectives", "d_opposed_objectives.png")

    # ---- E1/E2: the two V4 displays, deliberately not merged ----
    for key, name, value, title, cmap in (
            ("e1_difficulty_vs_process", "v4_difficulty_vs_process", "mean_ae",
             "E1  $q^{err}$ x $D^{shift}$ $\\rightarrow$ realized AE", "viridis"),
            ("e2_harm_vs_process", "v4_harm_vs_process", "observed_harm_frequency",
             "E2  $p_{harm}$ x $D^{shift}$ $\\rightarrow$ observed harm rate", "magma")):
        frame = nonempty(name)
        if frame is None or value not in frame.columns:
            continue
        axis_col = next((c for c in frame.columns if c.endswith("_bin") and c != "process_bin"), None)
        if axis_col is None:
            continue
        for method, g in frame.groupby("method", observed=True):
            grid = g.pivot_table(index=axis_col, columns="process_bin", values=value)
            counts = g.pivot_table(index=axis_col, columns="process_bin", values="n")
            if grid.empty:
                continue
            fig, ax = plt.subplots(figsize=(5.4, 4.1))
            im = ax.imshow(grid.to_numpy(float), cmap=cmap, aspect="auto", origin="lower")
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    v = grid.to_numpy(float)[i, j]
                    if np.isfinite(v):
                        n = counts.to_numpy(float)[i, j]
                        ax.text(j, i, f"{v:.3g}\nn={int(n) if np.isfinite(n) else 0}",
                                ha="center", va="center", fontsize=7, color="white")
            ax.set(xticks=range(grid.shape[1]), xticklabels=[f"P{c+1}" for c in grid.columns],
                   yticks=range(grid.shape[0]), yticklabels=[f"Q{r+1}" for r in grid.index],
                   xlabel="$D^{shift}$ quantile", ylabel=axis_col.replace("_bin", "") + " quantile",
                   title=f"{title}\n{label(method)}")
            fig.colorbar(im, ax=ax, shrink=.85, label=value.replace("_", " "))
            save(fig, f"{key}_{method}", f"{key}_{method}.png")

    # ---- F: one verdict per axis ----
    verdicts = []
    cov = nonempty("v2_reliability_coverage")
    disc = nonempty("v2_qerr_discrimination")
    if cov is not None and disc is not None:
        overall = cov[cov.subset.eq("overall")]
        d = disc[disc.subset.eq("overall") & disc.signal.eq("q_error")]
        gap = float(overall.coverage_error_gap.iloc[0]) if len(overall) else np.nan
        rho = float(d.spearman_signal_vs_ae.dropna().iloc[0]) if len(d.spearman_signal_vs_ae.dropna()) else np.nan
        verdicts.append(("Axis 1  $q^{err}$", f"coverage gap {gap:+.3f}", f"$\\rho$ = {rho:+.3f}",
                         abs(gap) <= .05 and rho >= .2))
    ph = nonempty("v2_pharm_calibration")
    if ph is not None:
        z = ph[ph.subset.eq("overall")]
        skill = float(z.brier_skill_score.dropna().iloc[0]) if len(z.brier_skill_score.dropna()) else np.nan
        rho = float(z.spearman_pharm_vs_harm.dropna().iloc[0]) if len(z.spearman_pharm_vs_harm.dropna()) else np.nan
        verdicts.append(("Axis 2  $p_{harm}$", f"Brier skill {skill:+.3f}", f"$\\rho$ = {rho:+.3f}",
                         (np.isfinite(skill) and skill > 0) or (np.isfinite(rho) and rho >= .2)))
    nd = nonempty("v3_process_monitor_nondegeneracy")
    if nd is not None:
        row = nd.iloc[0]
        verdicts.append(("Axis 3  $D^{shift}$",
                         "nondegenerate" if not row.monitor_degenerate else "DEGENERATE",
                         f"{int(row.distinct_values)} distinct", not bool(row.monitor_degenerate)))
    if verdicts:
        fig, ax = plt.subplots(figsize=(7.4, .9 + .72*len(verdicts)))
        ax.axis("off")
        for index, (axis, left, right, ok) in enumerate(reversed(verdicts)):
            y = index * .9
            ax.add_patch(plt.Rectangle((0, y-.3), 10, .62,
                                       color="#e8f5e9" if ok else "#fdecea", zorder=0))
            ax.text(.15, y, axis, fontsize=11, va="center", weight="bold")
            ax.text(3.3, y, left, fontsize=9.5, va="center")
            ax.text(6.0, y, right, fontsize=9.5, va="center")
            ax.text(8.6, y, "supported" if ok else "not supported", fontsize=9.5, va="center",
                    color="#2e7d32" if ok else "#c00000", weight="bold")
        ax.set(xlim=(0, 10), ylim=(-.5, len(verdicts)*.9 - .3))
        ax.set_title("F  Do the three axes capture prediction quality?", fontsize=11, loc="left")
        save(fig, "f_scorecard", "f_axis_scorecard.png")
    return made


def build_report(data: pd.DataFrame, output: Path, *, wide: pd.DataFrame|None=None,
                 runtime: pd.DataFrame|None=None, interval_calibration: pd.DataFrame|None=None,
                 reliability: pd.DataFrame|None=None, bootstrap: pd.DataFrame|None=None,
                 rr_regimes: pd.DataFrame|None=None,
                 rr_clusters: pd.DataFrame|None=None,
                 validation: dict|None=None) -> None:
    output.mkdir(parents=True,exist_ok=True); tables=output/"tables";figures=output/"figures"
    tables.mkdir(exist_ok=True);figures.mkdir(exist_ok=True);data=data.copy();data["cutoff"]=pd.to_datetime(data.cutoff)
    methods=list(dict.fromkeys(data.method));palette=dict(zip(methods,COLORS*3));plt.style.use("seaborn-v0_8-whitegrid")
    match_keys=["fold","geo_value","reference_date","report_date","lag"]
    grouped_cases=data.groupby(match_keys,observed=True)
    complete=(grouped_cases.method.nunique().eq(len(methods)) &
              grouped_cases.model_available.all())
    complete_keys=complete[complete].index
    matched=data[pd.MultiIndex.from_frame(data[match_keys]).isin(complete_keys)].copy()
    point=matched.groupby("method",observed=True).agg(n=("absolute_error","size"),mean_ae=("absolute_error","mean"),median_ae=("absolute_error","median"),p90_ae=("absolute_error",lambda x:x.quantile(.9))).reset_index()
    null=point[point.method.eq("baseline_null")]
    if not null.empty:
        point["mean_ae_gain_vs_null_pct"]=100*(null.mean_ae.iloc[0]-point.mean_ae)/null.mean_ae.iloc[0]
        point["median_ae_gain_vs_null_pct"]=100*(null.median_ae.iloc[0]-point.median_ae)/null.median_ae.iloc[0]
    # Same comparison on the model's working scale. Raw-scale AE is dominated by
    # high-volume locations (error grows with level); working-scale AE weights
    # locations roughly equally, so the two can rank methods differently. Both
    # are reported rather than picking one.
    point_working=pd.DataFrame()
    if "absolute_error_working" in matched:
        point_working=matched.groupby("method",observed=True).agg(
            n=("absolute_error_working","size"),mean_ae=("absolute_error_working","mean"),
            median_ae=("absolute_error_working","median"),
            p90_ae=("absolute_error_working",lambda x:x.quantile(.9))).reset_index()
        nw=point_working[point_working.method.eq("baseline_null")]
        if not nw.empty and nw.mean_ae.iloc[0]:
            point_working["mean_ae_gain_vs_null_pct"]=100*(nw.mean_ae.iloc[0]-point_working.mean_ae)/nw.mean_ae.iloc[0]
            point_working["median_ae_gain_vs_null_pct"]=100*(nw.median_ae.iloc[0]-point_working.median_ae)/nw.median_ae.iloc[0]
    availability=data.groupby("method",observed=True).agg(n=("model_available","size"),available_n=("model_available","sum"),availability=("model_available","mean"),probabilistic_n=("has_distribution","sum")).reset_index()
    # Both scales by lag, for the same reason the overall tables carry both:
    # raw-scale error grows with the level of the series so high-volume
    # locations dominate it, while working-scale error weights locations
    # roughly equally. At a fixed lag the two can rank methods differently.
    # Mean AND median, for AE and WIS, on BOTH scales: the full 2x2x2 set at
    # every lag. Means and medians disagree badly here because revision error
    # is heavy-tailed -- a method can win on the mean while losing on the
    # median -- and the two scales disagree because raw-scale error grows with
    # the level of the series. Reporting one of the eight would hide the others.
    by_lag_metrics={"n":("absolute_error","size")}
    for metric,source in (("ae","absolute_error"),("wis","wis"),
                          ("ae_working","absolute_error_working"),
                          ("wis_working","wis_working")):
        if source in matched:
            by_lag_metrics[f"mean_{metric}"]=(source,"mean")
            by_lag_metrics[f"median_{metric}"]=(source,"median")
    by_lag=matched.groupby(["lag","method"],observed=True).agg(**by_lag_metrics).reset_index()
    by_time=matched.groupby(["cutoff","method"],observed=True).agg(n=("absolute_error","size"),mean_ae=("absolute_error","mean"),median_ae=("absolute_error","median")).reset_index()
    by_location=matched[matched.lag.le(14)].groupby(["geo_value","method"],observed=True).agg(n=("absolute_error","size"),mean_ae=("absolute_error","mean"),median_ae=("absolute_error","median")).reset_index()
    uncertainty=[];calibration=[]
    for method,group in data.groupby("method",observed=True):
        x=group[group.has_distribution&group.model_available]
        if x.empty:continue
        row={"method":method,"n":len(x),"mean_wis":x.wis.mean(),"median_wis":x.wis.median()}
        if "wis_working" in x:
            row.update(mean_wis_working=x.wis_working.mean(),
                       median_wis_working=x.wis_working.median())
        for level,(lo,hi) in LEVELS.items():
            lower,upper=x[f"q{lo:g}"],x[f"q{hi:g}"]
            row.update({f"coverage_{level}":((lower<=x.truth)&(x.truth<=upper)).mean(),f"mean_width_{level}":(upper-lower).mean(),f"median_width_{level}":(upper-lower).median(),f"lower_miss_{level}":(x.truth<lower).mean(),f"upper_miss_{level}":(x.truth>upper).mean()})
        uncertainty.append(row)
        calibration += [{"method":method,"nominal_quantile":tau,"empirical_cdf":(x.truth<=x[f"q{tau:g}"]).mean(),"n":len(x)} for tau in TAUS]
    uncertainty=pd.DataFrame(uncertainty);calibration=pd.DataFrame(calibration)
    magnitude=matched.copy();magnitude["absolute_revision"]=magnitude.true_remaining_revision.abs();magnitude["revision_bin"]=pd.qcut(magnitude.absolute_revision,5,duplicates="drop")
    by_magnitude=magnitude.groupby(["revision_bin","method"],observed=True).agg(n=("absolute_error","size"),mean_ae=("absolute_error","mean"),median_ae=("absolute_error","median")).reset_index();by_magnitude["revision_bin"]=by_magnitude.revision_bin.astype(str)
    process=pd.DataFrame();routing=pd.DataFrame()
    if wide is not None and "process_state" in data:
        # The monitor is target-free and identical for every method, so read it
        # off whichever method the dataset actually ran, not a hardcoded "red".
        audit=data[data.method.eq("red" if "red" in set(data.method) else methods[0])]
        process=audit.groupby("process_state",observed=True).agg(n=("absolute_error","size"),mean_ae=("absolute_error","mean"),median_ae=("absolute_error","median"),mean_shift_score=("process_shift_score","mean")).reset_index()
        fields=[c for c in ["cold_start","point_ess","max_normalized_donor_weight","mean_curve_distance","mean_stage_distance"] if c in wide]
        if fields:
            routing=wide.groupby("lag",observed=True)[fields].agg(["count","mean","median"]).reset_index();routing.columns=["_".join(str(z) for z in c if z) if isinstance(c,tuple) else c for c in routing.columns]
    outputs={"prediction_comparison":point,"prediction_comparison_working_scale":point_working,"model_availability":availability,"comparison_by_lag":by_lag,"comparison_by_origin":by_time,"early_lag_comparison_by_location":by_location,"uncertainty_analysis":uncertainty,"quantile_calibration":calibration,"comparison_by_revision_magnitude":by_magnitude,"process_monitor_audit":process,"routing_audit_by_lag":routing}
    for name,frame in outputs.items():frame.to_csv(tables/f"{name}.csv",index=False)

    fig,axes=plt.subplots(1,2,figsize=(12,4.5));x=np.arange(len(point))
    for ax,metric,title in zip(axes,["mean_ae","median_ae"],["Mean absolute error","Median absolute error"]):ax.bar(x,point[metric],color=[palette[m] for m in point.method]);ax.set(ylabel=title,xticks=x,xticklabels=[label(m) for m in point.method]);ax.tick_params(axis="x",rotation=25)
    fig.suptitle("Matched accuracy on the raw reporting scale");fig.tight_layout();fig.savefig(figures/"overall_accuracy.png",dpi=180);plt.close(fig)
    def lag_figure(specs,title,filename):
        panels=[(c,t) for c,t in specs if c in by_lag and by_lag[c].notna().any()]
        if not panels: return
        rows=(len(panels)+1)//2
        fig,axes=plt.subplots(rows,2,figsize=(13,4.8*rows));axes=np.atleast_1d(axes).ravel()
        for ax,(col,name) in zip(axes,panels):
            for method in methods:
                z=by_lag[by_lag.method.eq(method)].sort_values("lag")
                if z[col].notna().any():
                    ax.plot(z.lag,z[col],"o-",ms=3,label=label(method),color=palette[method])
            ax.set(xlabel="Exact lag",ylabel=name)
        axes[0].legend(fontsize=8)
        for ax in axes[len(panels):]: ax.axis("off")
        fig.suptitle(title);fig.tight_layout();fig.savefig(figures/filename,dpi=180);plt.close(fig)
    lag_figure([("mean_ae","Mean AE (raw scale)"),("median_ae","Median AE (raw scale)"),
                ("mean_ae_working","Mean AE (working scale)"),
                ("median_ae_working","Median AE (working scale)")],
               "Absolute error by exact lag, both evaluation scales","accuracy_by_lag.png")
    lag_figure([("mean_wis","Mean WIS (reporting scale)"),
                ("median_wis","Median WIS (reporting scale)"),
                ("mean_wis_working","Mean WIS (working scale diagnostic)"),
                ("median_wis_working","Median WIS (working scale diagnostic)")],
               "Weighted interval score by exact lag, both evaluation scales","wis_by_lag.png")
    fig,ax=plt.subplots(figsize=(11,5))
    for method in methods:
        z=by_time[by_time.method.eq(method)].sort_values("cutoff");ax.plot(z.cutoff,z.mean_ae,marker="o",ms=2.5,label=label(method),color=palette[method])
    ax.set(xlabel="Forecast origin",ylabel="Mean AE",title="Performance stability over prospective origins");ax.tick_params(axis="x",rotation=30);ax.legend(fontsize=8,ncol=2);fig.tight_layout();fig.savefig(figures/"accuracy_over_time.png",dpi=180);plt.close(fig)
    pivot=by_location.pivot(index="geo_value",columns="method",values="mean_ae")
    if "baseline_null" in pivot:
        fig,ax=plt.subplots(figsize=(10,max(8,.22*len(pivot))));y=np.arange(len(pivot));others=[m for m in methods if m!="baseline_null"]
        for off,method in zip(np.linspace(-.3,.3,len(others)),others):ax.scatter(100*(pivot["baseline_null"]-pivot[method])/pivot["baseline_null"].replace(0,np.nan),y+off,s=15,label=label(method),color=palette[method])
        ax.axvline(0,color="black");ax.set(yticks=y,yticklabels=pivot.index,xlabel="Mean-AE gain versus Null (%)",ylabel="Location",title="Early-lag performance by location (lag ≤ 14)");ax.legend(fontsize=8);fig.tight_layout();fig.savefig(figures/"location_gain.png",dpi=180);plt.close(fig)
    if not calibration.empty:
        fig,ax=plt.subplots(figsize=(6.5,5.5));ax.plot([0,1],[0,1],"--",color="black",label="Ideal")
        for method in calibration.method.unique():
            z=calibration[calibration.method.eq(method)];ax.plot(z.nominal_quantile,z.empirical_cdf,"o-",label=label(method),color=palette[method])
        ax.set(xlabel="Nominal quantile",ylabel="Observed proportion below quantile",title="Quantile calibration");ax.legend(fontsize=8);fig.tight_layout();fig.savefig(figures/"quantile_calibration.png",dpi=180);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(12,4.5));levels=np.array(list(LEVELS));width=.7/len(uncertainty)
        for i,row in uncertainty.reset_index(drop=True).iterrows():
            pos=np.arange(3)+(i-(len(uncertainty)-1)/2)*width;axes[0].bar(pos,[row[f"coverage_{x}"] for x in levels],width,label=label(row.method),color=palette[row.method]);axes[1].bar(pos,[row[f"median_width_{x}"] for x in levels],width,label=label(row.method),color=palette[row.method])
        axes[0].plot(np.arange(3),levels/100,"kx",ms=8,label="Nominal");axes[0].set(ylabel="Empirical coverage",xticks=np.arange(3),xticklabels=[f"{x}%" for x in levels]);axes[1].set(ylabel="Median interval width",xticks=np.arange(3),xticklabels=[f"{x}%" for x in levels]);axes[1].legend(fontsize=7);fig.suptitle("Calibration and sharpness");fig.tight_layout();fig.savefig(figures/"coverage_and_width.png",dpi=180);plt.close(fig)
    if runtime is not None and runtime.standalone_total_seconds.notna().any():
        z=runtime[runtime.standalone_total_seconds.notna()];fig,ax=plt.subplots(figsize=(8,4));ax.bar([label(x) for x in z["method"]],z.standalone_total_seconds/60,color=[palette.get(x,"#777") for x in z["method"]]);ax.set(ylabel="Minutes",title="Measured standalone-equivalent runtime");ax.tick_params(axis="x",rotation=25);fig.tight_layout();fig.savefig(figures/"runtime_by_method.png",dpi=180);plt.close(fig)
    if not process.empty:
        fig,ax=plt.subplots(figsize=(7,4));ax.bar(process.process_state.astype(str),process.mean_ae,color="#377eb8");ax.set(xlabel="Process state",ylabel="RED mean AE",title="Does monitoring identify difficult cases?");fig.tight_layout();fig.savefig(figures/"process_monitor.png",dpi=180);plt.close(fig)

    # Section 4: matured interval calibration, forecast-quality risk score/
    # alert, and episode-clustered bootstrap CIs. Written for every run
    # (not an optional side computation); tables are empty rather than
    # absent when a dataset has no distribution-issuing methods configured.
    interval_calibration = pd.DataFrame() if interval_calibration is None else interval_calibration
    reliability = pd.DataFrame() if reliability is None else reliability
    bootstrap = pd.DataFrame() if bootstrap is None else bootstrap
    interval_calibration.to_csv(tables/"interval_calibration.csv", index=False)
    bootstrap.to_csv(tables/"bootstrap_pairwise_comparisons.csv", index=False)
    risk_group = pd.DataFrame()
    if not reliability.empty and "reliability_level" in reliability:
        risk_group = reliability.groupby(["method", "reliability_level"], observed=True).agg(
            n=("realized_ae", "size"), mean_ae=("realized_ae", "mean"), median_ae=("realized_ae", "median"),
            harm_frequency=("realized_regret", lambda x: (x > 0).mean())).reset_index()
    risk_group.to_csv(tables/"risk_group_validation.csv", index=False)
    if not interval_calibration.empty:
        fig,ax=plt.subplots(figsize=(7,5))
        for (method,status),g in interval_calibration.groupby(["method","status"],observed=True):
            g=g.sort_values("nominal");ax.plot(g.nominal,g.coverage,marker="o",label=f"{label(method)} ({status})")
        ax.plot([.5,.95],[.5,.95],"--",color="black",label="Ideal")
        ax.set(xlabel="Nominal coverage",ylabel="Empirical coverage",title="Interval coverage before/after matured calibration");ax.legend(fontsize=7);fig.tight_layout();fig.savefig(figures/"interval_calibration.png",dpi=180);plt.close(fig)

    # RevRoute Delphi-RF: how many revision-state regimes each origin selected
    # and why, plus where test cases were actually routed. Rendered only when
    # an rr_delphirf_* layer ran, so other datasets' reports are unchanged.
    rr_regimes = pd.DataFrame() if rr_regimes is None else rr_regimes
    rr_routing = pd.DataFrame()
    if wide is not None and "rr_regime" in wide:
        grouped=wide.groupby("cutoff",observed=True)
        rr_routing=pd.DataFrame({"cases":grouped.size(),"regimes_available":grouped.rr_clusters.max(),
            "regimes_used":grouped.rr_regime.nunique(),
            "process_component_share":grouped.rr_process_component_active.mean(),
            "curve_only_share":grouped.rr_route_components.apply(lambda x:(x=="curve").mean()),
            "routing_fallback_share":grouped.rr_routing_fallback.apply(lambda x:(x!="none").mean()),
            "median_route_distance":grouped.rr_route_distance.median()}).reset_index()
    rr_clusters = pd.DataFrame() if rr_clusters is None else rr_clusters
    rr_regimes.to_csv(tables/"rr_delphirf_regime_selection.csv",index=False)
    rr_clusters.to_csv(tables/"rr_delphirf_cluster_profile.csv",index=False)
    rr_routing.to_csv(tables/"rr_delphirf_routing_by_origin.csv",index=False)
    rr_section=("" if rr_regimes.empty and rr_routing.empty else
        "<h2>RevRoute Delphi-RF: revision-state regimes</h2><p class='note'>Regimes are chosen from training geometry alone, before any Delphi-RF is fit: outlying relative merge-height gaps in one average-linkage hierarchy propose candidate cuts, silhouette picks among the candidates whose every cluster holds the required number of training rows, and K=1 is reported when no candidate qualifies. Test cases are routed to the nearest regime medoid using only releases observable at the current report date; the mature-stage component is a training component and is never constructed for an unresolved case. A routing fallback is not an error: a medoid and a test case are compared only on integer lags inside <em>both</em> observed supports, never by extrapolation, so a case whose current lag falls before a medoid's first observed release shares no support with it and no component is defined. Read <code>routing_fallback_share</code> together with <code>medoid_first_lag</code> in the selection table below; when a regime has only one selected medoid, a fallback case is routed to the largest regime, which for K=1 is the same regime every other case receives.</p>"
        f"<h3>Selection per retraining origin</h3>{table(rr_regimes)}"
        + ("" if rr_clusters.empty else
           "<h3>What each regime pooled</h3><p class='note'>One row per origin and regime. Regime membership is learned from revision behaviour, not location identity, so a low <code>top_location_share</code> means the regime pools cases across many locations. <code>abs_remaining_revision</code> is the mature stage component |r| = |y - b| that the training dissimilarity uses.</p>"
           + table(rr_clusters))
        + f"<h3>Where test cases were routed</h3>{table(rr_routing)}")

    validation = validation or {}
    for name, frame in validation.items():
        if not name.startswith("_") and frame is not None and not frame.empty:
            frame.to_csv(tables/f"{name}.csv", index=False)
    TITLES={"v1_flat_line_comparison":"V1 &mdash; better/worse/tied against the flat-line baseline",
            "v1_pairwise_point":"V1 &mdash; pairwise point comparisons (mean AE)",
            "v1_pairwise_distribution":"V1 &mdash; pairwise distribution comparisons (WIS)",
            "v1_interval_calibration_by_lag":"V1 &mdash; interval calibration, base vs calibrated",
            "v1_regime_stratified_differences":"V1 &mdash; loss differences by prospective stratifier",
            "v2_reliability_coverage":"V2 &mdash; do the risk quantities have their stated upper-tail meaning?",
            "v2_alert_stratification":"V2 &mdash; realized performance by forecast alert",
            "v2_risk_discrimination":"V2 &mdash; continuous risk discrimination and rank association",
            "v2_operational_alert_summary":"V2 &mdash; operational summaries at the frozen thresholds",
            "v3_process_alert_validation":"V3 &mdash; issued process alerts and downstream forecast association",
            "v3_process_coordinate_attribution":"V3 &mdash; process-coordinate attribution",
            "v4_joint_alert_table":"V4 &mdash; joint forecast-alert x process-alert grid",
            "v4_dual_alert_contrasts":"V4 &mdash; conditional adjacent alert contrasts"}
    figure_files=validation_figures(validation, figures, palette, methods)
    # Each table is preceded by the figure that answers the same question, so
    # the section reads as pictures first and numbers second.
    FIGURE_FOR={"v4_joint_alert_table":"vC_joint_alert",
                "v1_flat_line_comparison":"v1_flat_line","v1_pairwise_point":"v1_pairwise",
                "v1_interval_calibration_by_lag":"v1_calibration",
                "v2_reliability_coverage":"v2_coverage","v2_alert_stratification":"v2_alert",
                "v2_risk_discrimination":"v2_discrimination",
                "v2_operational_alert_summary":"v2_operational"}
    def block(name, frame):
        picture=figure_files.get(FIGURE_FOR.get(name, ""))
        image=f"<img src='figures/{picture}'>" if picture else ""
        return f"<h3>{TITLES.get(name, name)}</h3>{image}{table(frame.head(200))}"
    validation_html="".join(block(name, frame) for name, frame in validation.items()
                            if not name.startswith("_") and frame is not None and not frame.empty)
    validation_section=("" if not validation_html else
        "<h2>Prospective validation (V1&ndash;V2)</h2><p class='note'>Pure post-processing of forecasts "
        "that were already issued and scored. Matured outcomes are used only to judge them: nothing is "
        "re-predicted, and no alert threshold is tuned on these tables. The alert assigned at issuance "
        "is read, never recomputed.</p>" + validation_html)

    report=f"""<!doctype html><html><head><meta charset='utf-8'><title>AutoDelphiRF report</title><style>body{{font:14px system-ui;max-width:1320px;margin:30px auto;padding:0 18px;line-height:1.45}}table{{border-collapse:collapse;font-size:12px;display:block;overflow-x:auto;margin-bottom:20px}}th,td{{border:1px solid #ddd;padding:5px 7px;text-align:right}}th{{background:#f1f3f5}}td:first-child,th:first-child{{text-align:left}}img{{max-width:100%;border:1px solid #ddd;margin:8px 0 28px}}.note{{background:#fff7df;border-left:4px solid #d99800;padding:12px}}.muted{{color:#666}}</style></head><body><h1>{html.escape(str(data.dataset.iloc[0]))}: AutoDelphiRF forecast evaluation report</h1><p class='note'>Point accuracy is reported on BOTH the raw reporting scale (primary) and the model's working scale; they can rank methods differently because raw-scale error grows with the level of the series, so high-volume locations dominate it, while working-scale error weights locations roughly equally. Remaining metrics use the raw reporting scale. Point-only methods are excluded from WIS and calibration. Residual uncertainty includes only cases where that residual model was available.</p><h2>Coverage and availability</h2>{table(availability)}<h2>Overall prediction comparison (raw reporting scale)</h2>{table(point)}<img src='figures/overall_accuracy.png'><h2>Overall prediction comparison (working scale)</h2><p class='note'>Same matched cases and same predictions as above, scored before the inverse transform. A method that helps most on high-volume locations looks better on the raw scale; one that helps uniformly across locations looks better here.</p>{table(point_working)}<h2>Accuracy by exact lag</h2><p class='note'>Mean and median, for AE and WIS, on both scales -- eight columns per lag in <code>comparison_by_lag.csv</code>. Means and medians disagree because revision error is heavy-tailed, so a method can win on one and lose the other. Raw-scale error grows with the level of the series, so high-volume locations dominate it; working-scale error weights locations roughly equally. At a given lag the two can rank methods differently, so read them together.</p><img src='figures/accuracy_by_lag.png'><img src='figures/wis_by_lag.png'>{table(by_lag)}<h2>Stability over time</h2><img src='figures/accuracy_over_time.png'><h2>Early-lag behavior by location</h2><img src='figures/location_gain.png'>{table(by_location)}<h2>Revision-magnitude analysis</h2>{table(by_magnitude)}<h2>Uncertainty: accuracy, calibration, sharpness, and miss direction</h2>{table(uncertainty)}<img src='figures/quantile_calibration.png'><img src='figures/coverage_and_width.png'><h2>Routing audit</h2>{table(routing)}<h2>Process-monitor audit</h2>{table(process)}<img src='figures/process_monitor.png'><h2>Uncertainty and reliability: calibration, risk score, alerts, and bootstrap intervals</h2><p class='note'>Interval calibration expands each base quantile endpoint using matured historical errors (asymmetric, per side); the risk-score/alert layer stratifies forecasts by historical trustworthiness in comparable cases; bootstrap CIs use the reference-date episode as the resampling unit. None of this changes a point forecast.</p><h3>Interval coverage before/after calibration</h3>{table(interval_calibration)}{"<img src='figures/interval_calibration.png'>" if not interval_calibration.empty else ""}<h3>Risk-group validation</h3>{table(risk_group)}<h3>Reference-date clustered bootstrap comparisons</h3>{table(bootstrap)}{validation_section}{rr_section}<h2>Computation</h2><p class='note'>Shared RED routing/history cost is separated from residual-model incremental fit and prediction cost. Imported comparators are marked unavailable when no runtime was recorded.</p>{table(runtime)}<img src='figures/runtime_by_method.png'><h2>Artifacts</h2><p>All tables are under <code>report/tables</code>; figures under <code>report/figures</code>. Per-origin and pipeline-stage timings are stored beside this report.</p></body></html>"""
    (output/"report.html").write_text(report)
