# hto-ker-diag: HTR-Former with a knowledge-graph diagnosis tower

This folder predicts **hypertension therapy response** on MIMIC-IV: whether a patient's blood pressure is controlled (median SBP < 120 and DBP < 80) in the 90 days after their last antihypertensive prescription. It's a copy of `hto-past` (HTR-Former plus 16 baselines) with three additions:

1. **A local extraction pipeline** (`extraction/`). It rebuilds the cohort from the original Colab/BigQuery notebook, using DuckDB on the PhysioNet CSVs, with timestamps and leakage checks.
2. **A SNOMED CT knowledge graph** (`build_kg.py`, `build_patient_graphs.py`). It's built from ICD codes, and each patient gets a time-aware graph over their own diagnoses, following KerPrint (Yang et al., AAAI-23).
3. **`htr_former_kg`**, an HTR-Former whose diagnosis embeddings are fused with that graph through a learned gate. Every other baseline is unchanged.

`hto-past/` stays as the untouched reference.

---

## Pipeline at a glance

```
MIMIC-IV CSVs ──► extraction/run_extraction.py ──► data/processed/<dataset>/
                                                     mimic_data/final_*.parquet   (cohort + timestamped events)
                                                     tables/icd_inventory.csv     (diagnosis codes in the cohort)
                                                     tables/splits.csv            (train / val / test)
                         │                                  │
                         ▼                                  ▼
         build_data.py ──► processed_data/<tag>/      build_kg.py ◄── SNOMED CT release + CMS GEMs
           (hto-past variants, incl. allcodes_timeaware)    │  processed_data/kg/<kg>/ (graph + TransR embeddings)
                         │                                  ▼
                         │                       build_patient_graphs.py ──► processed_data/<tag>/patient_graphs.<kg>.pkl
                         ▼                                  │
                     run.py --baseline <name> [--kg-tag <kg>] ◄┘ ──► predictions/<tag>/<run>/metrics.json
```

## Setup

```bash
cd Knowledge-HTO/hto-ker-diag
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
source .venv/bin/activate
python -m pytest tests/test_kg.py -q        # 15 tests, about 5 s, no data needed (one test uses the demo build if present)
```

**External inputs.** None of these go in git; all of them are covered by `.gitignore`.

| input | where to get it | used by |
|---|---|---|
| MIMIC-IV 3.1 (full) | PhysioNet, credentialed: `wget -r -N -c -np --user <you> --ask-password https://physionet.org/files/mimiciv/3.1/` | extraction |
| MIMIC-IV demo 2.2 (100 patients, public) | `wget -r -N -c -np https://physionet.org/files/mimic-iv-demo/2.2/` | development |
| SNOMED CT US Edition (RF2) | NLM, free in the US with a UMLS account | `build_kg.py` |
| CMS 2018 diagnosis GEMs (ICD-9 → ICD-10) | `https://www.cms.gov/medicare/coding/icd10/downloads/2018-icd-10-cm-general-equivalence-mappings.zip`, unzipped into `data/external/cms-gems-2018/unzipped/` | `build_kg.py` |

## Running it

All commands run from this folder. Swap `demo` / `mimic-iv-demo-2.2` for `full` / `mimic-iv-3.1` on the full data.

```bash
DS=mimic-iv-3.1                     # extraction dataset name (mimic-iv-3.1 also turns on the BigQuery cross-check)
P=data/processed/$DS
SNOMED=/path/to/SnomedCT_ManagedServiceUS_PRODUCTION_US1000124_20260901T120000Z
GEMS=data/external/cms-gems-2018/unzipped/2018_I9gem.txt

# 1. extraction: cohort + timestamped events (first run caches the CSVs as parquet: tens of minutes on full data)
python extraction/run_extraction.py --raw-dir /path/to/physionet.org/files/mimiciv/3.1 --dataset $DS
#    add --no-icu-vitals if chartevents is too slow, --memory-limit 8GB on a 16 GB machine

# 2. hto-past datasets, using the extraction's split
python build_data.py --tag full --skip-text --mimic-dir $P/mimic_data --split-csv $P/tables/splits.csv

# 3. knowledge graphs: the real one + two ablation graphs (add --device cuda for TransR on a GPU)
KG="--codes $P/tables/icd_inventory.csv --snomed-dir $SNOMED --gems $GEMS"
python build_kg.py all --tag full $KG
python build_kg.py all --tag full_isa --isa-only $KG
python build_kg.py all --tag full_rewired --rewire-seed 1 $KG

# 4. per-patient graphs
for kg in full full_isa full_rewired; do python build_patient_graphs.py --dataset full --kg-tag $kg --mimic-dir $P/mimic_data; done

# 5a. one run
python run.py --baseline htr_former_kg --dataset full --kg-tag full --epochs 10 --batch-size 32 --device cuda
# 5b. or the experiment matrix on SLURM: 8 configs x 5 seeds = 40 jobs
HTO_ROOT=$(pwd) EPOCHS=10 bash submit_HPC_jobs/submit_kg_experiments.sh full full
# 5c. any other baseline
python run.py --baseline htr_former_ca --dataset full --epochs 10 --device cuda
python build_report.py --root predictions --out reports/summary.md
```

**Check before training:**
- `data/processed/$DS/manifest.json`: every leakage check `true`, and the cohort within 0.5% of the BigQuery cohort (38,364).
- `processed_data/kg/full/coverage.json`: `all_usage_weighted` ≥ 0.85 (the demo gives 0.97).
- `processed_data/full/patient_graphs.full.stats.json`: how many patients have a graph.

**Check after training:** the `kg_diagnostics` entry in each `metrics.json` (see "Reading the results").

**On a laptop:** use `--device cpu`. macOS MPS lacks a kernel `nn.TransformerEncoder` needs. `bash tests/smoke_all.sh demo` runs every baseline end to end; with `KG_TAG=demo` it includes `htr_former_kg`.

### Coming from the Colab notebook (`MIMIC_IV_Analysis.ipynb`)

You don't need to extend the Colab notebook. `extraction/MIMIC_IV_Extraction.ipynb` is that notebook rebuilt: same cohort, same antihypertensive regex, same label. It adds timestamps, labs, vitals and leakage checks, and writes everything the rest of the pipeline needs. The one difference is the data source: it reads the PhysioNet CSV files, not BigQuery. To run it in Colab:

```python
# Colab cell 1: code + packages
!git clone -b diagnosis-tower-mods https://github.com/LazarCankovic/Knowledge-HTO.git
%cd Knowledge-HTO/hto-ker-diag
!pip install -q -r requirements.txt

# Colab cell 2: download MIMIC-IV 3.1 onto the Colab machine (~10 GB, credentialed PhysioNet account)
import getpass
user = input("PhysioNet username: "); pw = getpass.getpass("PhysioNet password: ")
!wget -q -r -N -c -np --user "{user}" --password "{pw}" https://physionet.org/files/mimiciv/3.1/hosp/ https://physionet.org/files/mimiciv/3.1/icu/

# Colab cell 3: extraction (skip ICU charts to fit Colab's RAM and time limits)
!python extraction/run_extraction.py --raw-dir physionet.org/files/mimiciv/3.1 --dataset mimic-iv-3.1 \
    --no-icu-vitals --memory-limit 10GB
```

- **Check the run:** `extraction/runs/MIMIC_IV_Extraction_mimic-iv-3.1.ipynb` should show the cohort within 0.5% of the Colab notebook's 38,364, with every leakage check passing.
- **Continue:** run steps 2–5 above. Steps 3–4 also need the SNOMED release uploaded to the Colab machine.
- **Storage:** Colab's disk is wiped when the session ends. Copy `data/processed/` somewhere your PhysioNet data use agreement allows before you disconnect.

---

## 1. Data extraction (`extraction/`)

`MIMIC_IV_Extraction.ipynb` ports the cohort logic of the team's Colab notebook (`MIMIC_IV_Analysis.ipynb`, BigQuery) to DuckDB. `run_extraction.py` runs it against any MIMIC-IV folder (`.csv` or `.csv.gz`) and saves an executed copy to `extraction/runs/`. Settings can come from flags or from `HTO_*` environment variables.

**Cohort** (same rules as the Colab notebook):
- broad hypertension diagnosis: ICD-10 I10–I16 or ICD-9 401–405 / 437.2
- an antihypertensive prescription; the last one is the index time
- a diagnosis history before the index
- at least one outpatient (OMR) BP reading 1–90 days after the index

**Label:** median SBP < 120 and DBP < 80 over those readings.

**What's different from the Colab version:**
- Every feature row has a time, and only events available at the index time are kept.
- Diagnoses count as available at discharge, so the index admission's codes are excluded. The Colab notebook leaked them; `--dx-time-policy admission` reproduces that behaviour.
- OMR vitals must be dated strictly before the index date.
- 12 leakage checks must pass before outputs are moved into place.
- `--require-usable-dx` drops patients with no usable diagnoses. On the demo that's 10 of 26.

**Outputs** (`data/processed/<dataset>/`):
- `mimic_data/final_{patients_table,diagnoses,medications,labs,vitals,sequences}.parquet`: the input contract `build_data.py` expects
- `tables/visits.parquet`, `tables/icd_inventory.csv`, `tables/splits.csv` (70/15/15, stratified)
- `hto_eval/bp_patient_sequences.csv`
- `manifest.json` (aggregates only) and `DATA_DICTIONARY.md` (every column)

**Built for full data:**
- Large tables go from DuckDB straight to parquet, not through pandas.
- The CSV format is declared explicitly. Guessing it from the first rows breaks on full `prescriptions`.
- The executed notebook hides patient-level rows by default, so it can be shared.

**Tested:** on the demo; on gzipped input; and on a 30× synthetic copy (20M chart rows, 1 GB memory cap), where every count came out exactly 30×.

## 2. The knowledge graph (`src/hto/kg/`)

**ICD → SNOMED** (`icd_snomed_map.py`):
- **ICD-10-CM:** the SNOMED → ICD-10-CM extended map (US refset 6011000124106), read in reverse.
- **ICD-9:** CMS GEMs to ICD-10-CM first, then the same map.
- **Codes with no direct map:** fall back to more specific codes, then to parent codes.
- **Coverage:** 97% of the demo's diagnosis rows. The unmapped ones are mostly BMI (Z68/V85) and external-cause E-codes.

**Global graph** (`snomed_graph.py`):
- **Nodes:** each ICD code and its ICD category, plus up to 10 SNOMED concepts per code (unconditional map rules first, then the more general concepts).
- **SNOMED neighbourhood:** the concepts' IS-A ancestors (2 levels) and 9 clinical attributes: finding site, morphology, pathological process, due to, causative agent, definitional manifestation, associated with, after, interprets.
- **Filtering:** generic top-level concepts and hubs are dropped. Every relation also gets an inverse.
- **Ablation graphs:** `--isa-only` keeps only IS-A edges. `--rewire-seed` shuffles which concepts are linked while keeping each node's edge counts (a content control).
- **Demo size:** 10,212 nodes and 66,524 edges.

**TransR pretraining** (`transr.py`): starting embeddings for the nodes. A held-out link-prediction check reports hits@10, 0.86 on the demo against 0.10 by chance.

**Per-patient graphs** (`patient_graphs.py`):
- Each observed code keeps its own 1-hop knowledge (τ = 0).
- Pairs of the patient's codes connected within 4 hops add the edges on those paths. Other ICD codes are never used as intermediates.
- **τ** is the gap in days between the two codes' most recent visits. The closest pair of occurrences would put τ near 0 for every pair, because chronic codes recur at almost every visit.

## 3. The model: `htr_former_kg` (`src/hto/models/`)

`kg_code_embedding.py` implements KerPrint's time-aware graph attention (Eqs. 1–5) in plain PyTorch. Each patient's graph is kept separate within a batch.

```
attention over a node's incoming edges = softmax( FFN( LN(M_r e_h) ‖ LN(M_r e_u) ‖ LN(f_τ) ) ),   f_τ = W(1 − tanh²(W' τ/180 + b')) + b''
node embedding after layer l           = LeakyReLU( W_l (e^(l−1) + Σ attention · neighbour embeddings) ),   output = Linear(all layers concatenated)
```

`htr_former_kg.py` subclasses `htr_former_ca` and fuses each diagnosis-code embedding with its node's vector:

```
gate (default):  x = emb + s · sigmoid(W_g [emb ‖ LN(kg)]) ⊙ LN(kg) · has_node      s: learnable scale, starts at 0
add:             x = emb + kg                                                       kg from a zero-initialised output layer
```

**Design decisions** (and where they differ from KerPrint):
- **Starts at the baseline.** The KG contribution starts at exactly 0 in both modes, so training begins from plain `htr_former_ca`.
- **The gate.** It learns how much to trust the KG for each code. On the demo it opens wider for codes the vocab maps to `[UNK]` than for known codes. The vocab's `min_count=2` and shared 30k cap turned 29% of the demo's training diagnosis codes into `[UNK]`. The KG lookup uses the raw code strings, so those codes keep their identity.
- **The scale's learning rate.** s gets 30× the base learning rate and no weight decay (`--kg-scale-lr-mult`). At the base rate it barely moved: in a 150-step test the KG reached ~1% of the embedding's size, against ~21% at ×30. `train.py` uses a model's `param_groups()` when it defines one.
- **The attention trains.** KerPrint computed it under `torch.no_grad()`, so its time network never learned.
- **LayerNorm on the attention inputs** (not in KerPrint). TransR embeddings reach the attention network at a tiny scale, so without it the scores barely differ across edges and the attention is uniform. That made τ's influence ~0.
- **τ is scaled by 180 days,** as in the paper (the KerPrint code used 28).

## Reading the results

Each `htr_former_kg` run writes `kg_diagnostics` into `metrics.json`. Read it before comparing AUCs:

| field | meaning | warning sign |
|---|---|---|
| `kg_to_emb_ratio` | size of what the KG adds, relative to the code embedding | ≲ 0.01: the KG branch never engaged |
| `max_dp_kg_removed` | largest change in P(controlled) when the KG term is removed | ≈ 0: the KG isn't used |
| `max_dp_time_off` | largest change in P(controlled) when τ is dropped | ≈ 0: `kg_no_time` / `kg_tau_shuffled` can't differ from `kg` |
| `gate_scale`, `gate_mean_unk`, `gate_mean_known` | the scale s; the mean gate at `[UNK]` vs known codes | s ≈ 0 or both gates ≈ 0.5: the gate didn't learn |

**The experiment matrix** (`submit_kg_experiments.sh`). Results go to `predictions/<tag>/<model>_<config>_s<seed>/`.

| config | question |
|---|---|
| `baseline` (`htr_former_ca`) | reference |
| `kg` | the full model |
| `kg_add` | does the gate beat plain addition? |
| `kg_no_time`, `kg_tau_shuffled` | does τ matter, and is it the real τ? |
| `kg_no_transr` | does TransR initialisation help? |
| `kg_isa` | are the attribute relations needed? |
| `kg_rewired` | is it SNOMED's content, or just an extra graph and more parameters? |

**Noise caveat:** configs whose graphs have different edge counts also draw a different number of dropout random numbers, which changes their training runs. On the demo that effect was about 100× the KG's own effect. Only differences that hold up across all 5 seeds count.

**Small cohorts:** `cv_compare.py` runs the same configs as a paired, repeated, stratified k-fold comparison. Out-of-fold predictions are pooled, and each config's difference from the baseline gets a patient-bootstrap CI. Outputs go to `reports/cv_<dataset>/`.

## What has been verified, and what hasn't

**On the 100-patient demo:**
- **Extraction:** the leakage checks pass; `build_data.py` writes every variant.
- **Existing baselines:** `htr_former`, `htr_former_ca`, `a1_time`, `a2_no_time`, `medbert_md`, `strats` and `tlstm` all still train.
- **The KG pipeline** runs end to end.
- **Tests:** the 15 tests pass. Several were mutation-checked: re-adding KerPrint's `no_grad` bug, ignoring τ, misaligning codes and graph nodes, and leaking the gate into padding each make a test fail.
- **The KG gets used.** In a fixed-step test, the gated KG reached ~21% of the embedding's size.

**Not yet known:** speed and memory on the full data, and whether any configuration beats the baseline. With 26 patients every AUC confidence interval spans most of the range, so the demo can't answer that.

## Layout

```
extraction/                 MIMIC-IV → data/processed/<dataset>/  (notebook + runner)
build_data.py               data/processed/... → processed_data/<tag>/ (all hto-past variants)
build_kg.py                 ICD → SNOMED map, global graph, TransR → processed_data/kg/<kg>/
build_patient_graphs.py     per-patient time-aware graphs
run.py                      train + evaluate one model (val-based checkpointing; test set used once)
cv_compare.py               paired k-fold comparison for small cohorts
build_report.py             aggregate predictions/ into reports/
src/hto/data/               canonical records (build.py), splits, variants, batch building
src/hto/kg/                 knowledge-graph construction
src/hto/models/             all models; registry.py maps --baseline names to classes
src/hto/train.py            shared training loop      src/hto/eval/metrics.py  AUC, AUPRC, Brier, ECE, per-drug
submit_HPC_jobs/            run_one.sh (one SLURM job), submit_all.sh (all baselines), submit_kg_experiments.sh
tests/                      test_kg.py (unit + integration), smoke_all.sh (every baseline end to end)
data/, processed_data/, predictions/, reports/   local outputs (gitignored)
```

**Baselines available through `run.py --baseline`:**
- **Original:** `medbert`, `tlstm`, `time2vec`, `transformehr`, `behrt`, `cehrbert`, `multimodal_behrt`
- **Multidomain versions:** each of the above with `_md`, adding a composite time encoding and an optional relative-day bias (`--pe-parts`, `--rel-day-bias`)
- **Irregular time series:** `mtan`, `strats`
- **HTR-Former family:** `htr_former`, `htr_former_ca`, `a1_time`, `a2_no_time`, and **`htr_former_kg`**

## Data and licensing

Everything under `data/`, `processed_data/`, `predictions/` and `extraction/runs/` is derived from MIMIC-IV and/or SNOMED CT. It stays out of git, and only people covered by the PhysioNet data use agreement and a SNOMED/UMLS licence may have it. Aggregate files are fine to share: `manifest.json`, `coverage.json`, `*.stats.json`, and `metrics.json`.
