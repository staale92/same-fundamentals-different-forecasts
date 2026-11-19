#!/usr/bin/env python3
"""Score firm-identification answers against SEC entity names and tickers.

Report free-text and forced-choice accuracy by arm and exposure cell, with
firm-clustered intervals and exposed-minus-unexposed and N-minus-arm contrasts.
The probe design is specified in configs/identification_probe.json.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from numeraire.schema import exposure_cell, read_jsonl
from numeraire.scoring import cluster_ols

ROOT = Path(__file__).resolve().parents[1]
SUFFIX = (
    r"\b(the|inc|incorporated|corp|corporation|company|co|plc|ltd|limited|"
    r"llc|lp|group|holdings?|parent|sa|nv|se|ag)\b"
)


def norm(s):
    """Normalize case, punctuation, and legal suffixes for entity-name matching."""
    text = re.sub(r"[^a-z0-9 ]", " ", s.lower().replace("&", " and "))
    return re.sub(r"\s+", " ", re.sub(SUFFIX, " ", text)).strip()


def correct(answer, ident):
    toks = set(re.findall(r"\b[A-Z][A-Z.\-]{0,5}\b", answer))
    if any(t in toks for t in ident["tickers"]):
        return True
    a = " " + norm(answer) + " "
    names = [ident["name"], *ident.get("former_names", [])]
    return any(n and (" " + n + " ") in a for n in (norm(x) for x in names if x))


def rate(d, mask=None):
    f = d if mask is None else d[mask]
    s = cluster_ols(f.ok.to_numpy(float), np.ones((len(f), 1)), f.firm_id, f.id, 0)
    return {
        "rate": s["estimate"],
        "low": s["firm_low"],
        "high": s["firm_high"],
        "n": int(len(f)),
    }


def contrast(d, x):
    s = cluster_ols(
        d.ok.to_numpy(float),
        np.column_stack([np.ones(len(d)), x]),
        d.firm_id,
        d.id,
        1,
    )
    return {"estimate": s["estimate"], "low": s["firm_low"], "high": s["firm_high"]}


def cell_rates(g):
    """Accuracy and exposure contrast for a single presentation."""
    eligible = g[g.cell.isin(["exposed", "unexposed"])]
    return {
        "all": rate(g),
        "exposed": rate(g, g.cell.eq("exposed")),
        "unexposed": rate(g, g.cell.eq("unexposed")),
        "exposed_minus_unexposed": contrast(
            eligible, eligible.cell.eq("exposed").to_numpy(float)
        ),
    }


def reference_contrast(d, arm):
    paired = d[d.arm.isin(["N", arm])]
    return contrast(paired, paired.arm.eq("N").to_numpy(float))


def industry_matching(c, sic):
    """Is a correct choice only industry matching? Two-digit SIC of the true firm against the four distractors."""
    letters = "ABCDE"
    c = c.copy()
    c["true_sic"] = c.firm_id.map(sic)
    c["same_industry_distractors"] = [
        sum(sic.get(f) == t for f in cand if f != fid) for cand, fid, t in zip(c.candidates, c.firm_id, c.true_sic)
    ]
    c["pick_same_industry"] = [
        float(sic.get(cand[letters.index(p)]) == t) for cand, p, t in zip(c.candidates, c.pred, c.true_sic)
    ]
    out = {}
    for arm, g in c.groupby("arm"):
        shared = g.same_industry_distractors > 0
        wrong = g[~g.ok].assign(ok=lambda x: x.pick_same_industry)
        out[arm] = {
            "share_with_same_industry_distractor": float(shared.mean()),
            "accuracy_no_same_industry_distractor": rate(g, ~shared),
            "accuracy_with_same_industry_distractor": rate(g, shared),
            "wrong_answers_same_industry": rate(wrong),
            "wrong_answers_same_industry_chance": float((wrong.same_industry_distractors / 4).mean()),
        }
    return out


def main():
    config = json.loads((ROOT / "configs/identification_probe.json").read_text())
    ident = json.loads((ROOT / config["identities"]).read_text())
    recs = {r["id"]: r for r in read_jsonl(ROOT / config["panel"])}

    def cell(observation_id):
        return exposure_cell(recs[observation_id], config["exposure_cutoff"])

    rep = {}
    for folder in sorted((ROOT / "results/identification").glob("llama31-*")):
        rows = read_jsonl(folder / "answers.jsonl")
        d = pd.DataFrame(rows)
        d["ok"] = [correct(a, ident[f]) for a, f in zip(d.answer, d.firm_id)]
        d["cell"] = d.id.map(cell)
        e = {
            "answers": len(d),
            "firms_ever_identified_N": int(d[(d.arm == "N") & d.ok].firm_id.nunique()),
        }
        for arm, g in d.groupby("arm"):
            e[arm] = cell_rates(g)
            if arm != "N":
                e[arm]["N_minus_arm"] = reference_contrast(d, arm)
        e["examples_correct_N"] = d[(d.arm == "N") & d.ok].answer.head(5).tolist()
        e["refusal_share"] = float(
            d.answer.str.contains(
                r"can.?t|unable|cannot|couldn.?t|not possible|not able",
                case=False,
                regex=True,
            ).mean()
        )
        e["correct_share_all_arms"] = float(d.ok.mean())
        cf = folder / "choice_scores.jsonl"
        if cf.exists():
            c = pd.DataFrame(read_jsonl(cf))
            c["pred"] = c.logp.map(lambda scores: max(scores, key=scores.get))
            c["ok"] = c.pred.eq(c.true_letter)
            c["p_true"] = [
                np.exp(scores[letter]) / sum(np.exp(v) for v in scores.values())
                for scores, letter in zip(c.logp, c.true_letter)
            ]
            c["cell"] = c.id.map(cell)
            ch = {
                "scores": len(c),
                "letter_shares": c.pred.value_counts(normalize=True).round(3).to_dict(),
            }
            for arm, g in c.groupby("arm"):
                ch[arm] = cell_rates(g) | {"mean_p_true": float(g.p_true.mean())}
                if arm != "N":
                    ch[arm]["N_minus_arm"] = reference_contrast(c, arm)
            sel = json.loads((ROOT / "data/raw/sec_confirmation/selection.json").read_text())["candidates"]
            sic = {f'{int(x["cik"]):010d}': str(x["sic"])[:2] for x in sel if x.get("sic")}
            ch["industry_matching"] = industry_matching(c, sic)
            e["forced_choice"] = ch
        rep[folder.name] = e
    output = ROOT / "results/identification/report.json"
    output.write_text(json.dumps(rep, indent=2, default=float) + "\n")
    for m, e in rep.items():
        print(m, e["answers"], "answers; firms identified on N:", e["firms_ever_identified_N"])


if __name__ == "__main__":
    main()
