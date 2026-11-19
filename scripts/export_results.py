"""Export numerical tables, manuscript figures, and an exposure timeline.

CSV tables are written to results/tables; figure PDFs to results/figures.
"""

from pathlib import Path
import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results/tables"
FIGURES = ROOT / "results/figures"
LLAMA = {
    "Llama 3.1 8B": ("Llama 3.1 8B", "Llama 3.1 8B (hosted)"),
    "Llama 3.1 70B": ("Llama 3.1 70B", "Llama 3.1 70B"),
}
LL = tuple(LLAMA)
COLORS = {"Llama 3.1 8B": "#1f5fa8", "Llama 3.1 70B": "#c4561d"}


def read_report(relative_path):
    return json.loads((ROOT / relative_path).read_text())


class AnalysisResults:
    """Saved estimates shared by the numerical tables and figures."""

    def __init__(self):
        self.quarterly_confirmation = pd.read_csv(
            ROOT / "results/quarterly_confirmation/contrasts.csv"
        )
        self.quarterly_pilot = pd.read_csv(ROOT / "results/quarterly/contrasts.csv")
        self.annual_pilot = pd.read_csv(ROOT / "results/pilot/contrasts.csv")
        self.annual = read_report("results/annual/summary.json")
        self.controls = read_report("results/controls/report.json")
        self.primary = read_report("results/primary/report.json")
        self.audit_checks = read_report("results/audit_checks/report.json")
        self.sensitivity = read_report("results/sensitivity/report.json")
        self.presentation = read_report("results/presentation/report.json")
        for model, values in read_report("results/presentation/magnitudes.json").items():
            self.presentation[model].update(values)

    def quarterly_contrast(self, model, contrast="NC against N_p1", split=False, frame=None):
        frame = self.quarterly_confirmation if frame is None else frame
        key = (
            f"{model} placebo in time (to 2022Q2 vs 2022Q3-2023Q3)"
            if split
            else f"{model} within-model"
        )
        rows = frame[(frame.analysis == key) & (frame.contrast == contrast)]
        return rows.iloc[0] if len(rows) else None


def short_model(m):
    return (
        m.replace("Llama 3.1 ", "Llama ")
        .replace("Muse Glimmer 30B", "Muse")
        .replace("gpt-oss-120b", "gpt-oss")
    )


def splits(D, model):
    """Declared placebo-in-time splits of NC against N_p1 for one Llama model."""
    out = []
    _, hosted = LLAMA[model]
    for label, frame in (
        ("confirmation, quarterly", D.quarterly_confirmation),
        ("pilot, quarterly", D.quarterly_pilot),
    ):
        r = D.quarterly_contrast(model, split=True, frame=frame)
        if r is not None:
            out.append(
                {
                    "panel": label,
                    "estimate": r.estimate,
                    "low": r.wcr_low,
                    "high": r.wcr_high,
                    "interval": "WCR",
                }
            )
    if model == "Llama 3.1 8B":
        r = D.annual_pilot[
            (D.annual_pilot.model == "Llama 3.1 8B")
            & (D.annual_pilot.contrast == "Primary: NC against N_p1")
            & (D.annual_pilot.variant == "placebo-in-time FY2012-16 vs FY2017-22")
        ]
        if len(r):
            r = r.iloc[0]
            lo, hi = (
                (r.wcr_low, r.wcr_high)
                if "wcr_low" in r and not pd.isna(r.wcr_low)
                else (r.firm_low, r.firm_high)
            )
            out.append(
                {
                    "panel": "pilot, annual",
                    "estimate": r.estimate,
                    "low": lo,
                    "high": hi,
                    "interval": "WCR" if "wcr_low" in r and not pd.isna(r.wcr_low) else "firm",
                }
            )
    else:
        pt = (
            D.annual.get("Llama 3.1 70B", {})
            .get("contrasts", {})
            .get("NC against N_p1", {})
            .get("placebo_in_time")
        )
        if pt:
            out.append(
                {
                    "panel": "pilot, annual",
                    "estimate": pt["estimate"],
                    "low": pt["firm_low"],
                    "high": pt["firm_high"],
                    "interval": "firm",
                }
            )
    pt = (
        D.annual.get(f"{hosted} confirmation", {})
        .get("contrasts", {})
        .get("NC against N_p1", {})
        .get("placebo_in_time")
    )
    if pt:
        out.append(
            {
                "panel": "confirmation, annual",
                "estimate": pt["estimate"],
                "low": pt["firm_low"],
                "high": pt["firm_high"],
                "interval": "firm",
            }
        )
    return out


def m_rule(D):
    res = {}
    for model in LLAMA:
        s = splits(D, model)
        p = D.quarterly_contrast(model)
        M = max(abs(x["estimate"]) for x in s)
        Mext = max(max(abs(x["low"]), abs(x["high"])) for x in s)
        res[model] = {
            "splits": s,
            "M": M,
            "M_ext": Mext,
            "theta": p.estimate,
            "l": p.wcr_low,
            "u": p.wcr_high,
            "mde": p.mde80,
            "signflip_p": p.signflip_p,
            "bounds": {"0": p.wcr_high, "M": p.wcr_high + M, "M_ext": p.wcr_high + Mext},
            "negative_split": min(x["estimate"] for x in s),
        }
    return res


def bounds(D, M):
    """Scenario bounds (u + M)/(1 - rho) over drift calibrations and placebo-removal ratios."""
    controls, primary = D.controls, D.primary
    rho = controls["rho"]
    out = {}
    for model, m in M.items():
        sp = controls["splits"][model]
        u = primary["primary"][model]["primary"]["wcr_high"]
        prim = [x for x in sp if x["panel"] == "confirmation, quarterly"]
        Ms = {
            "panel_min3": D.sensitivity["drift_min3"][model]["M"],
            "panel_any": D.audit_checks[model]["M_primary_panel"],
            "rule": m["M"],
            "ext": m["M_ext"],
            "scaled": max(x["scaled"] for x in prim),
            "unexposed": max([m["M"]] + [abs(x["estimate"]) for x in sp]),
        }
        rhos = {"zero": 0.0, "point": max(rho["rho"], 0.0), "upper": min(rho["high"], 0.95)}
        out[model] = {
            "u": u,
            "M": Ms,
            "rho": rhos,
            "grid": {
                mk: {rk: (u + mv) / (1 - rv) for rk, rv in rhos.items()} for mk, mv in Ms.items()
            },
            "joint_p95": primary["joint_bound"][model]["p95"],
        }
    return out


def figure_bound(D, M, B):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    xs = np.linspace(0, 0.12, 60)
    for model in LL:
        b = B[model]
        c = COLORS[model]
        for rk, ls, lw in (("zero", "--", 0.7), ("point", "-", 1.4), ("upper", ":", 1.0)):
            ax.plot(
                xs,
                (b["u"] + xs) / (1 - b["rho"][rk]),
                color=c,
                lw=lw,
                ls=ls,
                label=f"{short_model(model)}" if rk == "point" else None,
            )
        for s in D.controls["splits"][model]:
            v = abs(s["estimate"])
            ax.plot(
                [v],
                [(b["u"] + v) / (1 - b["rho"]["point"])],
                marker="o" if s["period"] == "exposed" else "s",
                ms=3,
                color=c,
                mfc="white" if s["panel"] != "confirmation, quarterly" else c,
                mew=0.8,
            )
        for mk in ("rule", "ext"):
            ax.axvline(b["M"][mk], color=c, lw=0.5, ls=":")
        leading_m = b["M"]["panel_min3"]
        leading_upper = b["grid"]["panel_min3"]["point"]
        ax.plot(leading_m, leading_upper, marker="*", ms=8, color=c, zorder=5)
        ax.annotate(
            f"{leading_upper:.3f}",
            (leading_m, leading_upper),
            xytext=(8, 22 if model == LL[0] else 9),
            textcoords="offset points",
            color=c,
            fontsize=6,
            arrowprops={"arrowstyle": "-", "color": c, "lw": 0.5},
        )
    ax.set_xlabel("Assumed drift bound $M$ (Brier units)", fontsize=7)
    ax.set_ylabel("Upper end of 95% confidence set", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=6, frameon=False, loc="upper left")
    ax.set_xlim(0, 0.12)
    ax.set_ylim(0, None)
    fig.tight_layout(pad=0.3)
    fig.savefig(ROOT / "results/figures/upper_bound.pdf")
    plt.close(fig)


def figure_removal(D):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rd = D.controls["removal_by_dose"]
    p2 = D.controls.get("representation_retention")
    fig, axes = plt.subplots(
        1, 2 if p2 else 1, figsize=(3.4, 2.1), gridspec_kw={"width_ratios": [3, 2]} if p2 else None
    )
    ax = axes[0] if p2 else axes
    arms = [
        ("NC", "Converted (NC)", "#c4561d"),
        ("N_p1", "Reworded (N$_{p1}$)", "#1f5fa8"),
        ("NP", "Rows shuffled (NP)", "#6a9a3a"),
    ]
    doses = ["60", "120", "240"]
    for i, (k, label, c) in enumerate(arms):
        xs = np.arange(3) + (i - 1) * 0.22
        y = [rd[d][k]["share"] for d in doses]
        lo = [rd[d][k]["share_low"] for d in doses]
        hi = [rd[d][k]["share_high"] for d in doses]
        ax.errorbar(
            xs,
            y,
            yerr=[np.array(y) - np.array(lo), np.array(hi) - np.array(y)],
            fmt="o",
            ms=3,
            color=c,
            lw=0.8,
            capsize=1.5,
            label=label,
        )
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["60", "120", "240"], fontsize=6)
    ax.set_xlabel("Passes per row", fontsize=7)
    ax.set_ylabel("Share of planted gain removed", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=5.5, frameon=False, loc="upper right")
    ax.set_ylim(-0.6, 1.25)
    if p2:
        ax2 = axes[1]
        order = [("N", "N"), ("U", "U"), ("E", "E"), ("Q", "Q"), ("R", "R")]
        a = p2["arms"]
        ys = [a[k]["retained_share"] for k, _ in order if k in a]
        lo = [a[k]["retained_low"] for k, _ in order if k in a]
        hi = [a[k]["retained_high"] for k, _ in order if k in a]
        ax2.errorbar(
            range(len(ys)),
            ys,
            yerr=[np.array(ys) - np.array(lo), np.array(hi) - np.array(ys)],
            fmt="s",
            ms=3,
            color="#444",
            lw=0.8,
            capsize=1.5,
        )
        ax2.axhline(0, color="grey", lw=0.5)
        ax2.axhline(1, color="grey", lw=0.5, ls=":")
        ax2.set_xticks(range(len(ys)))
        ax2.set_xticklabels([l for k, l in order if k in a], fontsize=6)
        ax2.tick_params(labelsize=6)
        ax2.set_ylabel("Share of planted gain kept", fontsize=7)
        ax2.set_xlabel("Arm (60 passes)", fontsize=7)
    fig.tight_layout(pad=0.3)
    fig.savefig(ROOT / "results/figures/removal.pdf")
    plt.close(fig)


def figure_quarterly(D):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = lambda s: int(s[:4]) + (int(s[-1]) - 1) / 4
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    for k, model in enumerate(LL):
        q = D.presentation[model]["by_quarter"]
        x = np.array([t(r["period"]) for r in q]) + (k - 0.5) * 0.06
        y = np.array([r["gap"] for r in q])
        lo = np.array([r["low"] for r in q])
        hi = np.array([r["high"] for r in q])
        ax.errorbar(
            x,
            y,
            yerr=[y - lo, hi - y],
            fmt="o",
            ms=2.5,
            lw=0.7,
            capsize=1,
            color=COLORS[model],
            label=short_model(model),
        )
        for cell in ("exposed", "unexposed"):
            sel = [r for r in q if r["cell"] == cell]
            m = np.average([r["gap"] for r in sel], weights=[r["n"] for r in sel])
            xs = [t(sel[0]["period"]), t(sel[-1]["period"]) + 0.1]
            ax.plot(xs, [m, m], color=COLORS[model], lw=0.8, ls="--")
    ax.axvline(2023 + 11 / 12, color="black", lw=0.8)
    ax.axhline(0, color="grey", lw=0.5)
    ax.text(
        2023 + 11 / 12 + 0.05, ax.get_ylim()[1] * 0.9, "Llama 3.1 cutoff", fontsize=5.5, va="top"
    )
    ax.set_ylabel("Brier gap, converted\nminus reworded header", fontsize=6.5)
    ax.set_xticks(range(2021, 2027))
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=6, frameon=False, loc="lower right")
    fig.tight_layout(pad=0.3)
    fig.savefig(ROOT / "results/figures/quarterly_gap.pdf")
    plt.close(fig)


def figure_timeline():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    q = [f"{y}Q{k}" for y in range(2021, 2027) for k in (1, 2, 3)][:17]
    t = lambda s: int(s[:4]) + (int(s[-1]) - 1) / 4
    rows = [
        ("Llama 3.1 8B / 70B", "2023-12-31", "Dec 2023"),
        ("gpt-oss-120b", "2024-06-01", "Jun 2024"),
        ("Muse Glimmer 30B", "2026-01-04", "Jan 2026"),
    ]
    # filing lags: outcomes are filed about 40 days after quarter end; exposure follows the panel's cells
    cells = {
        "Llama 3.1 8B / 70B": lambda s: "exposed" if t(s) <= 2023.5 else "unexposed",
        "gpt-oss-120b": lambda s: (
            "exposed" if t(s) <= 2024.0 else ("donut" if t(s) < 2024.5 else "unexposed")
        ),
        "Muse Glimmer 30B": lambda s: (
            "exposed" if t(s) <= 2025.5 else ("donut" if t(s) < 2026.25 else "unexposed")
        ),
    }
    col = {"exposed": "#c9d7ea", "donut": "#eeeeee", "unexposed": "#f2c9b1"}
    fig, ax = plt.subplots(figsize=(3.4, 1.45))
    for i, (name, cut, lab_) in enumerate(rows):
        for s in q:
            ax.add_patch(
                plt.Rectangle((t(s), i - 0.35), 0.22, 0.7, color=col[cells[name](s)], lw=0)
            )
        y, mth, d = map(int, cut.split("-"))
        ax.plot([y + (mth - 1) / 12 + d / 365] * 2, [i - 0.45, i + 0.45], color="black", lw=1)
        ax.text(2020.9, i, name, ha="right", va="center", fontsize=6)
    ax.set_xlim(2020.9, 2026.6)
    ax.set_ylim(-0.6, 2.9)
    ax.set_yticks([])
    ax.invert_yaxis()
    ax.set_xticks(range(2021, 2027))
    ax.tick_params(labelsize=6)
    for k, c in col.items():
        ax.add_patch(plt.Rectangle((0, 0), 0, 0, color=c, label=k))
    ax.legend(fontsize=5.5, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.55, 1.32))
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    fig.tight_layout(pad=0.2)
    fig.savefig(ROOT / "results/figures/timeline.pdf")
    plt.close(fig)


def flatten_fields(values, prefix=""):
    """Expand nested statistics into named CSV columns without rounding."""
    row = {}
    for key, value in values.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            row.update(flatten_fields(value, name + "."))
        elif isinstance(value, list):
            row.update({f"{name}.{i}": item for i, item in enumerate(value)})
        else:
            row[name] = value
    return row


def write_table(name, rows):
    pd.DataFrame(rows).to_csv(TABLES / f"{name}.csv", index=False)


def export_tables(data, drift, scenarios):
    rows = []
    for model, analyses in data.primary["primary"].items():
        for name, result in analyses.items():
            rows.append({"model": model, "analysis": name, **flatten_fields(result)})
        calibrated = data.primary["validation"][model].get("primary_calibrated")
        if calibrated is not None:
            rows.append({"model": model, "analysis": "calibrated_elicitation", **calibrated})
    write_table("primary_contrasts", rows)

    drift_names = {
        "panel_min3": "primary_panel_minimum_three_quarters",
        "panel_any": "primary_panel_any_segment_length",
        "rule": "exposed_splits",
        "ext": "exposed_interval_extremes",
        "scaled": "primary_splits_distance_scaled",
        "unexposed": "all_splits",
    }
    rows = []
    for model, scenario in scenarios.items():
        for drift_key, values in scenario["grid"].items():
            for rho_key, upper in values.items():
                rows.append(
                    {
                        "model": model,
                        "drift_scenario": drift_names[drift_key],
                        "rho_scenario": rho_key,
                        "M": scenario["M"][drift_key],
                        "rho": scenario["rho"][rho_key],
                        "wcr_upper": scenario["u"],
                        "conditional_upper": upper,
                    }
                )
    write_table("conditional_bounds", rows)
    write_table(
        "joint_bound_diagnostic",
        [{"model": model, **values} for model, values in data.primary["joint_bound"].items()],
    )
    write_table(
        "exposed_temporal_splits",
        [{"model": model, **row} for model, values in drift.items() for row in values["splits"]],
    )
    write_table(
        "temporal_splits",
        [
            {"model": model, **row}
            for model, values in data.controls["splits"].items()
            for row in values
        ],
    )
    write_table(
        "known_target_validation",
        [
            {
                "model": model,
                **{
                    key: value
                    for key, value in result.items()
                    if key.startswith("known_") or key == "n"
                },
            }
            for model, result in data.primary["validation"].items()
        ],
    )
    write_table(
        "exposure_counts",
        [{"model": model, **values} for model, values in data.primary["exposure_counts"].items()],
    )
    write_table(
        "decision_flips",
        [
            {"model": model, "arm": arm, **values}
            for model, result in data.presentation.items()
            for arm, values in result["flips"].items()
        ],
    )
    write_table(
        "forecast_changes",
        [
            {"model": model, "elicitation": name, **result[key]}
            for model, result in data.presentation.items()
            for key, name in (("magnitudes", "raw"), ("magnitudes_calibrated", "calibrated"))
            if key in result
        ],
    )
    write_table(
        "quarterly_gaps",
        [
            {"model": model, **values}
            for model, result in data.presentation.items()
            for values in result.get("by_quarter", [])
        ],
    )
    write_table(
        "planted_attenuation",
        [
            {
                "passes_per_row": int(dose),
                "arm": arm,
                "trained_arm_effect": result["T_N"],
                **result[arm],
            }
            for dose, result in data.controls["removal_by_dose"].items()
            for arm in ("NC", "N_p1", "NP")
        ],
    )
    write_table(
        "planted_placebo_ratios",
        [
            {"passes_per_row": int(dose), "placebo": arm, **result[f"rho_{arm}"]}
            for dose, result in data.controls["removal_by_dose"].items()
            for arm in ("N_p1", "NP")
        ],
    )
    write_table(
        "representation_retention",
        [
            {"arm": arm, **values}
            for arm, values in data.controls["representation_retention"]["arms"].items()
        ],
    )
    (TABLES / "drift_calibration.json").write_text(
        json.dumps(drift, indent=2, default=float) + "\n"
    )
    export_elicitation()


def export_elicitation():
    """Elicitation checks (Table VII of the paper), if the analysis has been run."""
    path = ROOT / "results/elicitation/report.json"
    if not path.exists():
        return
    rep = json.loads(path.read_text())
    rows = []

    def row(setting, model, t, m):
        rows.append({"setting": setting, "model": model, "theta": t["estimate"], "wcr_low": t["wcr_low"],
                     "wcr_high": t["wcr_high"], "mde80": t["mde80"], "flips_conversion": m["NC"]["flip_share"],
                     "flips_header": m["N_p1"]["flip_share"]})

    for model, e in rep["instruction_ablation"].items():
        row("single token, without instruction", model, e["theta_without_instruction"], e["moves_without_instruction"])
    g = rep["gptoss_single_token"]
    row("single token, with instruction (own cutoff)", "gpt-oss-120b", g["theta"], g["moves"])
    r = rep["generated_reasoning"]
    row("generated reasoning, without instruction", "gpt-oss-120b", r["theta"], r["moves"])
    write_table("elicitation_checks", rows)


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    data = AnalysisResults()
    drift = m_rule(data)
    scenarios = bounds(data, drift)
    export_tables(data, drift, scenarios)
    figure_bound(data, drift, scenarios)
    figure_removal(data)
    figure_quarterly(data)
    figure_timeline()
    print("Exported numerical tables to results/tables and four figures to results/figures.")


if __name__ == "__main__":
    main()
