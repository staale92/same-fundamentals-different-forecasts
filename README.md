# Same Fundamentals, Different Forecasts

Replication code for **Same Fundamentals, Different Forecasts: Auditing Memory in LLM Financial Statement Analysis**, by Alessio Staffini, Trust Co., Ltd.

This research was carried out while the author was employed at Trust Co., Ltd.

The study tests whether economically equivalent presentations of SEC financial statements change language-model forecasts of earnings direction. This repository contains the financial panels, saved model probabilities, statement transformations, statistical analyses, and figures. Reproducing the statistics requires no model download, inference service, or credentials.

Contact: alessio_staffini@trust-partner.co.jp

## Installation and reproduction

Use Python 3.11 or later and run these commands from the repository root:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
python scripts/verify.py
python reproduce.py
```

`verify.py` checks the distribution checksums, financial panels, outcome labels, exposure classifications, prompt hashes, answer-token probabilities, generated-reasoning answers and their parsed forecasts, identification-probe coverage and candidate assignments, inversion scores, and reference statistics. Run it before regeneration to validate the downloaded archive. `reproduce.py` exports numerical tables to `results/tables/` and figures to `results/figures/` from the saved analysis results. It also checks the results against `expected/`.

To recompute the primary estimates, sensitivities, validation probes, benchmarks, and forecast-change statistics from the saved predictions:

```sh
python reproduce.py --primary
```

To recompute all analyses, including the additional diagnostics described below and seeded cluster resampling:

```sh
python reproduce.py --all
```

The full run can take tens of minutes on a laptop. `requirements-lock.txt` records the tested dependency versions. Small floating-point differences can occur across environments. The analysis and verification commands operate offline.

## Repository contents

| Path | Contents |
| --- | --- |
| `src/numeraire/` | SEC extraction, statement renderings, prompts, scoring, statistical inference, and benchmarks |
| `scripts/` | Executable analyses, data preparation, numerical export, and verification |
| `configs/` | Sample definitions, model and tokenizer settings, planted-control training settings, and identification-probe specifications |
| `data/processed/` | Financial panels and the planted-control rank reference |
| `results/` | Saved model outputs and numerical analysis results |
| `expected/` | Reference statistics checked after reproduction |
| `results/figures/` | Figures generated from saved analysis results |
| `tests/` | Checks of transformations, scoring, cluster inference, and control assignment |

## Analyses and outputs

| Script | Analysis | Output |
| --- | --- | --- |
| `analyze_primary.py` | Primary quarterly Brier contrasts with their exposure-cell means, sensitivity checks, calibrated elicitation, and benchmarks | `results/primary/` |
| `analyze_presentation.py` | Directional flips, probability changes, and quarterly loss gaps | `results/presentation/` |
| `analyze_diagnostics.py` | Matched annual benchmark forecasts and paired benchmark-minus-model differences, primary-panel mean shifts, repeated scoring, and planted-prompt overlap | `results/diagnostics/` |
| `analyze_audit_checks.py` | Primary-panel moving-cutoff drift, alternative placebos, firm-prominence triple difference, aggregation rule with mean forecasts, and forecast-probe correlation | `results/audit_checks/` |
| `analyze_identification.py` | Firm-identification probe: free-text answers and five-candidate choice scored by answer-letter probabilities, by arm and exposure cell, and industry matching by two-digit SIC code | `results/identification/report.json` |
| `analyze_elicitation.py` | Elicitation checks: the primary contrast without the no-memory instruction, the instruction's effect on as-filed forecasts by exposure cell, gpt-oss-120b generated reasoning, and gpt-oss single-token contrasts under its own cutoff | `results/elicitation/report.json` |
| `analyze_sensitivity.py` | Post hoc drift with at least three quarters per segment; exclusion of 2021Q1 or 2021Q1–Q3; drift recalibrated on the restricted panel; identification by arm, exposure cell, and firm size; forecast contrasts stratified by identification | `results/sensitivity/report.json` |
| `analyze_forecasts.py` | Known-target probes, elicitation, extrapolation benchmarks, nuisance arms, and template dates | `results/forecast_diagnostics/` |
| `analyze_annual.py` | Annual confirmation contrasts and repeated-scoring checks | `results/annual/` |
| `analyze_quarterly.py` | Quarterly pilot contrasts and staggered-cutoff rectangles | `results/quarterly/` |
| `analyze_quarterly.py --confirmation` | Quarterly confirmation contrasts and rectangles | `results/quarterly_confirmation/` |
| `analyze_pilot.py` | Annual pilot contrasts, mean-shift diagnostics, and conversion inversion | `results/pilot/` |
| `analyze_planted.py` | Planted-key manipulation checks and performance contrasts | `results/planted_control/validation.json` |
| `analyze_controls.py` | Temporal placebo splits, attenuation by training dose, and representation retention | `results/controls/` |
| `export_results.py` | Numerical CSV tables and manuscript figures | `results/tables/`, `results/figures/` |

All scripts in this table are under `scripts/`. Numerical exports retain full precision. The paper reports rounded values. The reported confidence sets concern conversion-removable recall conditional on fixed drift and placebo-removal assumptions; they do not bound total memory. The joint-bootstrap result is a diagnostic without a coverage guarantee. The three-quarter minimum segment length and exclusions of 2021Q1 and 2021Q1–Q3 were chosen after inspecting results. The `drift_restricted` results recalibrate the maximum absolute within-cell contrast after excluding 2021Q1–Q3, with a minimum of three quarters per segment and with any nonempty segment; this recalibration is also post hoc. The size comparison uses Llama 70B's as-filed identification scores and the top tercile of primary-panel firms by 2018 total assets; arm-by-cell identification rates are reported for both Llama models.

### Additional diagnostic outputs

The analysis scripts also produce useful checks beyond the results reported in the manuscript:

| Diagnostic | Saved output | Purpose |
| --- | --- | --- |
| Binned Brier-score decomposition | `results/pilot/discrimination.csv`, `results/quarterly/discrimination.csv`, `results/quarterly_confirmation/discrimination.csv`, and the `murphy_N`/`murphy_E` entries in `results/annual/summary.json` | Separates reliability, resolution, outcome uncertainty, and the within-bin remainder |
| Llama 70B minus 8B exposure contrast | `scale_gradient` in `results/quarterly/report.json` and `results/quarterly_confirmation/report.json` | Exploratory paired comparison of the two models' contrasts, with firm-bootstrap intervals |
| Maverick pilot scores and filing-date checks | Maverick entries in `results/annual/summary.json`, `results/quarterly/report.json`, and `results/quarterly/discrimination.csv` | Documents the excluded model's pilot performance and checks filing dates around its cutoff |
| Distance-scaled drift scenario | Rows labeled `primary_splits_distance_scaled` in `results/tables/conditional_bounds.csv` | An alternative sensitivity calculation for the conditional bound |
| Quarterly exposure timeline | `results/figures/timeline.pdf` | Displays stated-cutoff exposure classifications and the supported model-period comparisons |

These diagnostics are exploratory and do not establish additional confirmatory findings. In particular, Maverick's serving noise prevents an inferential interpretation of its scores. `reproduce.py --all` recomputes these outputs together with the manuscript analyses. The checks under `tests/` verify software correctness.

## Financial panels and saved predictions

| Panel in `data/processed/` | Observations | Firms | Use |
| --- | ---: | ---: | --- |
| `sec_pilot.jsonl` | 127 | 11 | Annual pilot |
| `sec_quarterly.jsonl` | 184 | 11 | Quarterly pilot |
| `sec_confirmation.jsonl` | 1,186 | 128 | Annual confirmation |
| `sec_quarterly_confirmation.jsonl` | 1,592 | 107 | Primary quarterly analysis |
| `planted_rank_reference.jsonl` | 1,587 | 107 | Fixed sampling frame and percentile-rank reference for the planted control |

Each panel row contains an observation identifier (`id`), firm identifier (`firm_id`), filing dates, two-period financial values, binary outcome (`label`), and SEC provenance. Monetary values are stored as decimal strings in whole USD; EPS is USD per share. Provenance includes tags, accessions, source values, and available source hashes. Identifiers and filing dates support verification and are excluded from the model prompts. The raw SEC snapshots are not bundled.

Prediction files contain one row per observation and arm, including the mean UP probability, the two label-swapped answer-token scores, prompt hashes, outcome label, and exposure cell. A probability of at least 0.5 is classified as UP. Saved `manifest.json` and `checkpoint_manifest.json` files identify the scoring configurations. Local and hosted checkpoints are treated as distinct configurations. Scoring manifests retain the input and source hashes recorded at inference (`inference_driver_sha256` identifies the scoring driver); `MANIFEST.sha256` gives the checksums of the files in this distribution.

`results/identification/` holds the identification-probe answers and letter scores; `configs/identification_probe.json` specifies the questions, candidate selection, and scoring. `data/processed/firm_identities.json` supplies the SEC names and tickers for matching free-text answers and constructing the five-candidate identification question. These identities never enter an earnings-forecast prompt. `results/inversion_hosted/` holds the hosted inversion probes for Llama 3.1 70B and gpt-oss-120b.

Hosted Llama scoring used the pinned `unsloth/Meta-Llama-3.1-70B-Instruct` tokenizer mirror for both model sizes. `configs/tokenizer_provenance.json` records the template and configuration hashes. The manifest's `matches_official` flags concern whole-file checksums. The template supplies a fixed “Today Date: 26 Jul 2024” header; the date-removal sensitivity is a separate scoring configuration.

Hosted forecasts are under `results/hosted/`; local pilot forecasts are in the model-named directories under `results/`. Folder suffixes identify the panel and scoring batch. `_native` and `_placebos` identify rendering batches, `_representations` includes rank and ratio evaluations, `_N_repeat` contains repeated reference-arm scores, and `_boundary` contains exact one-cent EPS-change cases. Analysis scripts assemble the required batches automatically. Actual arm coverage is recorded in each batch's manifest. Elicitation inputs are the `scores.jsonl` files under `results/forecast_diagnostics/` and `results/primary/`. `_no_instruction` holds the instruction ablation and `_reasoning` the generated-reasoning answers (`answers.jsonl`, with the reasoning text, the answer, and the parsed probability).

The primary models are Llama 3.1 8B and 70B. gpt-oss-120b and Muse Glimmer 30B provide additional comparisons. Maverick outputs support the reported serving-noise check. Cutoff classifications do not establish training-set membership. The study was not publicly preregistered; the paper identifies exploratory analyses and choices made after results were observed.

## Elicitation checks

Two checks were added after peer review. Both are exploratory and were declared before scoring. `configs/elicitation_checks.json` specifies them.

- **Instruction ablation.** Llama 3.1 8B and 70B rescore the primary panel's as-filed, reworded-header and yen arms. The two knowledge-restricting sentences are removed from the system message. Everything else matches the primary run.
- **Generated reasoning.** gpt-oss-120b first reasons at medium effort, then gives a short evidence summary and a probability in JSON.

The provider renders gpt-oss's chat template. That template states the model's knowledge cutoff and the run date. Each scoring manifest records its system message, and `verify.py` re-derives every prompt hash from it. `results/tables/elicitation_checks.csv` collects the contrasts reported in the paper.

## Data preparation and planted control

`select_confirmation_sample.py`, `build_confirmation_panel.py`, and `build_quarterly_panel.py` implement the SEC extraction recipe. Selection requests require a genuine project/contact identity in `SEC_USER_AGENT`. The financial panels are the fixed inputs for reproduction; live SEC responses can differ from the recorded source snapshots.

`results/planted_control/` contains the 40-firm sample, treatment assignments, training examples, base and adapter predictions, and single-item training checks. To reconstruct the sample, assignments, and both branches' training records:

```sh
python scripts/prepare_control_data.py --output /path/to/empty-directory
```

`configs/planted_control.json` specifies training settings. `scores/` contains the dose experiment; `scores_representations/` contains the alternative-representation experiment. The fixed rank reference in the panel table determines the planted-control percentile inputs. Model weights and adapters are not bundled. The archive reproduces analyses of saved outputs; it does not rerun neural training. Provider-internal optimizer defaults are not fully specified, so another training stack need not produce identical adapters.

Training doses denote 60, 120, and 240 passes per unique training row. Each branch used a separate 60-epoch job for each dose, with one, two, or four copies of its 40-row training file. Dose labels such as `e120` in saved output directories identify effective passes per row.

## Citation and license

Citation metadata are in `CITATION.cff`. Code is licensed under the MIT License; see `LICENSE`.
