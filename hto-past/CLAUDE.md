# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Healthcare Time-series Outcome (HTO) prediction on MIMIC-IV — benchmark of 16 baselines for a binary hypertension-outcome prediction from multi-domain patient histories (diagnoses, medications, labs, vitals).

## Baselines (16 total)

| group | baselines | variant | notes |
|---|---|---|---|
| **original** (paper-faithful) | `medbert`, `tlstm`, `time2vec`, `transformehr`, `behrt`, `cehrbert`, `multimodal_behrt` | `codes_flat` / `codes_visit` / `multimodal` | diag-only-style inputs, paper-original time handling |
| **multidomain** (our adaptation) | `medbert_md`, `tlstm_md`, `time2vec_md`, `transformehr_md`, `behrt_md`, `cehrbert_md`, `multimodal_behrt_md` | `multidomain_visit` | all 4 domains + **composite PE** + **optional relative-day attention bias** |
| **irregular-TS baselines** | `mtan`, `strats` | `multidomain_visit` | event-triplet models (Shukla 2021, Tipirneni 2022) |

## Temporal modeling — composite PE + relative-day bias

The multidomain baselines replace a baseline-specific scalar time embedding with a decomposable composite positional encoding:

```
PE(event) =  sinusoidal(position)               # always on
          +  T_abs(day_offset)                  # Time2Vec, 1/day ... 1/(10y)
          +  T_gap(delta_prev_days)             # Time2Vec over visit gaps
          +  D(domain_id)                       # {diag, med, lab, vital, struct}
          +  DT(domain_id, day_offset)          # MLP per-domain time modulation
```

Each component can be ablated via `--pe-parts abs,gap,domain,interaction`.

`--rel-day-bias on` additionally threads a per-head ALiBi-style bias `φ(|Δday_i − Δday_j|)` into every attention layer.

See `src/hto/models/composite_pe.py` and `docs/transformations.md`.

## Layout

```
mimic_data/                    # source parquets (do not touch)
processed_data/<tag>/          # canonical + per-variant pickles + split.json
predictions/<tag>/<baseline>/  # metrics.json (incl. AUC, AUPRC, Brier, ECE, by_target)
                               #   + predictions.npz + ROC/PR/confusion/prob_hist plots
reports/                       # aggregated summary.md + summary.tsv + bars
docs/transformations.md        # per-baseline input transforms + PE spec
src/hto/
  data/build.py                # raw parquets → canonical records
  data/splits.py               # stable 85/15 patient split
  data/variants.py             # canonical → codes_visit / codes_flat / multimodal / multidomain_visit
  models/
    medbert.py etc.            # original 7 baselines (unchanged)
    composite_pe.py            # CompositePE + RelativeDayBias + BiasableEncoder
    multidomain_batcher.py     # shared flat / visit batch builders
    md_backbone.py             # MDBackboneFlat (medbert_md/time2vec_md/behrt_md/cehrbert_md)
    md_hier.py                 # TransformEHR_MD, MultimodalBEHRT_MD, TLSTM_MD
    medbert_md.py / time2vec_md.py / behrt_md.py / cehrbert_md.py
    mtan.py / strats.py        # new irregular-TS baselines
  train.py                     # shared training / evaluation loop
  eval/metrics.py              # metrics + plots + NPZ persistence
  registry.py                  # baseline → (variant, class, factory)
build_data.py                  # CLI: produce processed_data/<tag>/
run.py                         # CLI: python run.py --baseline X --dataset Y [--pe-parts ...] [--rel-day-bias on|off]
build_report.py                # aggregate predictions/ into reports/
submit_HPC_jobs/
  run_one.sh                   # SLURM template (forwards PE flags)
  submit_all.sh                # fan out baseline × dataset, optional PE_ABLATION sweep
tests/smoke_all.sh             # run every baseline end-to-end on a small dataset
```

## Local development

```bash
# one-time: venv
python3 -m venv .venv && source .venv/bin/activate
pip install numpy pandas pyarrow scikit-learn matplotlib seaborn torch tabulate

# build smoke dataset (200 patients) — produces all 4 variants + split.json
python build_data.py --n-patients 200 --tag smoke_200 --skip-text

# build a 1K-patient smoke dataset (still CPU-friendly)
python build_data.py --n-patients 1000 --tag smoke_1000 --skip-text

# one baseline with composite PE + rel-day bias
python run.py --baseline medbert_md --dataset smoke_1000 \
    --epochs 2 --batch-size 16 --d-model 64 --device cpu \
    --pe-parts "abs,gap,domain,interaction" --rel-day-bias on

# PE ablation (same baseline, different PE configs via --run-suffix)
python run.py --baseline medbert_md --dataset smoke_1000 --pe-parts "" --rel-day-bias off --run-suffix no_pe
python run.py --baseline medbert_md --dataset smoke_1000 --pe-parts "abs" --rel-day-bias off --run-suffix abs_only
python run.py --baseline medbert_md --dataset smoke_1000 --pe-parts "abs,gap" --rel-day-bias off --run-suffix abs_gap

# run every baseline end-to-end (smoke test before HPC)
bash tests/smoke_all.sh smoke_1000

# aggregate results
python build_report.py --root predictions --out reports/summary.md
```

**MPS note**: on macOS, `torch.nn.TransformerEncoder` hits an unimplemented MPS kernel. Smoke tests default to CPU. HPC uses CUDA.

## HPC (SLURM)

```bash
# once, on a node with the parquets
python build_data.py --tag full

# fan out all 16 baselines on one dataset (16 jobs)
bash submit_HPC_jobs/submit_all.sh full

# subset of baselines
BASELINES="medbert_md strats" bash submit_HPC_jobs/submit_all.sh full

# PE ablation sweep (submits each _md baseline with 4 ablation configs)
PE_ABLATION=1 bash submit_HPC_jobs/submit_all.sh full

# aggregate after jobs complete
python build_report.py
```

## Data contract

- 30,423 patients, binary label (77% positive), stable patient-level 85/15 split.
- `multidomain_visit` mirrors the `final_sequences.parquet` grammar (`[EOV] [EOD] [EOM] [EOL] [EOVI]` delimiters) but re-attaches day_offsets and visit gaps (days) that the text form drops. Every visit exposes its 4 domain blocks + a parallel numeric value/mask stream.
- Time resolution is days.

## Adding a new baseline

1. Create `src/hto/models/<name>.py` with a class exposing `name`, `needs_variant`, `build_batch(items) -> dict[str, Tensor]`, `forward(batch) -> logits[B,2]`.
2. Register in `src/hto/registry.py` (include the class on `Entry`).
3. If it consumes a new variant, add a `make_<variant>` in `src/hto/data/variants.py` and a line in `build_data.py`.
4. Add to `DEFAULT_BASELINES` in `tests/smoke_all.sh` and `submit_HPC_jobs/submit_all.sh`.
5. Run `bash tests/smoke_all.sh smoke_200` — must finish every run green before shipping.
