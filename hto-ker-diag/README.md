# hto-ker-diag: HTR-Former with knowledge-graph diagnosis and lab towers

This folder predicts **hypertension therapy response** on MIMIC-IV: whether a patient's blood pressure is controlled (median SBP < 120 and DBP < 80) in the 90 days after their last antihypertensive prescription. It's a copy of `hto-past` (HTR-Former plus 16 baselines) with three additions:

1. **A local extraction pipeline** (`extraction/`). It rebuilds the cohort from the original Colab/BigQuery notebook, using DuckDB on the PhysioNet CSVs, with timestamps and leakage checks.
2. **A SNOMED CT knowledge graph** (`build_kg.py`, `build_patient_graphs.py`). It's built from ICD codes, and each patient gets a time-aware graph over their own diagnoses, following KerPrint (Yang et al., AAAI-23).
3. **`htr_former_kg`**, an HTR-Former whose diagnosis embeddings are fused with that graph through a learned gate.
4. **`htr_former_lab_kg`**, a separate model that enriches only the lab tower using a graph joined to the diagnosis SNOMED CT graph through LOINC. The diagnosis, medication and vital towers remain the usual HTR-Former towers.

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
python -m pytest tests/test_kg.py tests/test_lab_kg.py -q
```

**External inputs.** None of these go in git; all of them are covered by `.gitignore`.

| input | where to get it | used by |
|---|---|---|
| MIMIC-IV 3.1 (full) | PhysioNet, credentialed: `wget -r -N -c -np --user <you> --ask-password https://physionet.org/files/mimiciv/3.1/` | extraction |
| MIMIC-IV demo 2.2 (100 patients, public) | `wget -r -N -c -np https://physionet.org/files/mimic-iv-demo/2.2/` | development |
| SNOMED CT US Edition (RF2) | NLM, free in the US with a UMLS account | `build_kg.py` |
| CMS 2018 diagnosis GEMs (ICD-9 → ICD-10) | `https://www.cms.gov/medicare/coding/icd10/downloads/2018-icd-10-cm-general-equivalence-mappings.zip`, unzipped into `data/external/cms-gems-2018/unzipped/` | `build_kg.py` |
| MIMIC-IV `d_labitems` | Export from BigQuery with `extraction/export_from_bigquery.py` or use the matching raw CSV | `build_lab_kg.py` |
| MIMIC item-to-LOINC map | MIT-LCP `mimic-code/mimic-iv/mapping/d_labitems_to_loinc.csv` | `build_lab_kg.py` |
| LOINC Ontology RF2 | Compatible LOINC Ontology release | `build_lab_kg.py` |

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

### Lab knowledge-graph tower

Regenerate the event files: lab `time` now uses `storetime`, when the result was available, rather than the earlier `charttime`. Rows without `storetime` are omitted. If your source is BigQuery, use `extraction/export_from_bigquery.py` as described below. Build a *new* processed tag with the same split CSV; do not use `--reuse-canonical` from an older run, because it lacks aligned lab `itemid` values. Rerun the plain `htr_former_ca` baseline on this same new tag for a comparable result. A result on the older charttime-based dataset is not directly comparable.

```bash
DS=mimic-iv-3.1
P=data/processed/$DS
python extraction/run_extraction.py --raw-dir /path/to/physionet.org/files/mimiciv/3.1 --dataset $DS
python build_data.py --tag full_storetime --skip-text \
  --mimic-dir $P/mimic_data --split-csv $P/tables/splits.csv

python build_lab_kg.py all --tag lab_snomed --diagnosis-kg-tag full \
  --labitems $P/tables/d_labitems.csv \
  --loinc-map /path/to/mimic-code/mimic-iv/mapping/d_labitems_to_loinc.csv \
  --snomed-dir /path/to/SNOMED_US_release \
  --loinc-extension-dir /path/to/compatible_LOINC_Ontology_RF2
python build_lab_patient_graphs.py --dataset full_storetime --kg-tag lab_snomed \
  --mimic-dir $P/mimic_data

python run.py --baseline htr_former_lab_kg --dataset full_storetime \
  --kg-tag lab_snomed --epochs 1 --batch-size 32 --device cuda \
  --max-train-batches 2 --max-eval-batches 2 --run-suffix preflight

python run.py --baseline htr_former_ca --dataset full_storetime \
  --epochs 20 --batch-size 32 --device cuda
python run.py --baseline htr_former_lab_kg --dataset full_storetime \
  --kg-tag lab_snomed --epochs 20 --batch-size 32 --device cuda
```

The lab graph copies the diagnosis graph and adds `itemid → LOINC → SNOMED CT` links. MIMIC-IV 3.1 `d_labitems` does **not** contain `loinc_code`; download MIT-LCP's `mimic-iv/mapping/d_labitems_to_loinc.csv` or supply a reviewed `itemid,loinc_code` CSV. The script filters the repository file to LOINC mappings. The official LOINC Ontology RF2 Identifier file resolves each LOINC code to its active SNOMED CT concept. Its relationships and the compatible SNOMED release provide shared concept nodes; lab metadata links remain available. Obtain the matching LOINC Ontology release and check its compatibility with the SNOMED release used for the diagnosis graph. The script never guesses a LOINC code from a lab label.

Patient graphs contain each patient's observed lab item IDs **and diagnoses** and use the joined global graph. They contain no outcomes or measurements; the lab value is still represented by the model's value-bin token. Read `processed_data/kg/lab_snomed/graph_stats.json` for the count of linked lab items and SNOMED concepts shared with the diagnosis graph. Read `processed_data/full_storetime/patient_graphs.lab_snomed.stats.json` for `lab_row_snomed_coverage` and `patients_with_lab_diagnosis_path`. If the latter is zero, the global graph has links to SNOMED but no observed cross-domain paths under the local graph's four-hop limit. The patient-graph builder requires at least 90% of lab rows to match `d_labitems` and at least one cohort lab result to map through LOINC to SNOMED. The model maps raw item IDs to KG nodes even when the lab token is `[UNK]`, and fuses KG vectors only into measurement code tokens (position `4*k+1`). Check `kg_diagnostics.lab_kg_coverage` and `max_dp_kg_removed` after training.

**Check before training:**
- `data/processed/$DS/manifest.json`: every leakage check `true`, and the cohort within 0.5% of the BigQuery cohort (38,364).
- `processed_data/kg/full/coverage.json`: `all_usage_weighted` ≥ 0.85 (the demo gives 0.97).
- `processed_data/full/patient_graphs.full.stats.json`: how many patients have a graph.

**Check after training:** the `kg_diagnostics` entry in each `metrics.json` (see "Reading the results").

**On a laptop:** use `--device cpu`. macOS MPS lacks a kernel `nn.TransformerEncoder` needs. `bash tests/smoke_all.sh demo` runs every baseline end to end; with `KG_TAG=demo` it includes `htr_former_kg`.

### Coming from the Colab notebook (`MIMIC_IV_Analysis.ipynb`)

Your original Colab `mimic_iv_analysis.py` builds the LAST-HTO cohort and label, but the exported sequence CSV lacks event times and lab item IDs. Run its cohort cell (`df_last_cohort`), then use the BigQuery companion to export the input tables. It creates a short-lived table of patient IDs and index times in your writable Google Cloud project; it deletes that table at the end. Run from the authenticated notebook after checking out this branch:

```python
%pip install -q google-cloud-bigquery pyarrow pandas numpy
from extraction.export_from_bigquery import export_cohort
export_cohort(client, df_last_cohort, "/content/hto_export",
              scratch_dataset="hto-mimic.hto_extract")
```

Copy `/content/hto_export/{mimic_data,tables}` to `data/processed/mimic-iv-3.1/` on the HPC using approved storage. The export includes the final patient table, pre-index discharge diagnoses, medications, storetime labs, OMR/ICU vitals, `d_labitems.csv`, `icd_inventory.csv`, and stratified patient splits. The graph build still needs the SNOMED CT release, CMS GEMs, the separate MIT-LCP item-to-LOINC mapping, and the compatible LOINC Ontology RF2 extension. Keep MIMIC patient tables out of Git.

If your source is the PhysioNet CSV archive, `extraction/MIMIC_IV_Extraction.ipynb` provides the equivalent local DuckDB extraction. To run it in Colab:

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

## 4. Graph enrichment: ClinicalBERT features and the PrimeKG treatment block

Two separate structures, so drug knowledge does not dilute the time-aware diagnosis graph:
- **g_y, the patient disease graph:** the SNOMED CT patient graph above, with ClinicalBERT node features.
- **g_a, the candidate treatment block:** a 1-hop PrimeKG lookup for each patient and each of the 24 candidate antihypertensives in `src/hto/kg/resources/antihypertensive_candidates.tsv`.

Extra inputs, all public: MONDO (`mondo.json`), PrimeKG (`kg.csv` and `disease_features.tab` from Harvard Dataverse, doi:10.7910/DVN/IXA7BM), the RxNorm full release (`rrf/`) and `emilyalsentzer/Bio_ClinicalBERT`. Install `transformers` and `torch_geometric` as well as `requirements.txt`.

```bash
# Stage 1: ICD -> MONDO -> PrimeKG, then ClinicalBERT text features   (GPU helps; --device auto)
python 1_extract_text_embeddings.py --codes $P/tables/icd_inventory.csv --mondo mondo.json \
  --primekg kg.csv --disease-features disease_features.tab --kg-dir processed_data/kg/full \
  --snomed-dir $SNOMED --gems $GEMS
# rebuild the patient graphs once: they now store edge_hops, which stage 2 needs
python build_patient_graphs.py --dataset full_storetime --kg-tag full --mimic-dir $P/mimic_data
# Stage 2: PyG patient graphs with node / edge features
python 2_enrich_patient_graphs.py --dataset full_storetime --kg-tag full --mimic-dir $P/mimic_data
# Stage 3: treatment context tensor
python 3_build_treatment_block.py --dataset full_storetime --mimic-dir $P/mimic_data --rxnorm-dir rxnorm/rrf
# hand-off checks: 0% leakage, >= 90% code coverage, < 20% empty context
python eval_graph_enrichment.py --dataset full_storetime --kg-tag full --mimic-dir $P/mimic_data \
  --codes $P/tables/icd_inventory.csv
```

On SLURM, `submit_HPC_jobs/run_graph_enrichment.sh` runs one stage per job (`STAGE=stage1|graphs|stage2|stage3|eval`).

**Stage 1** (`src/hto/kg/disease_map.py`, `text_features.py`). MONDO links few ICD codes directly but about 9k SNOMED concepts, so each code takes the first route that reaches MONDO:
1. its own ICD xref
2. for ICD-9, its GEM targets
3. its SNOMED concepts
4. its nearest parent code
5. its SNOMED IS-A ancestors (up to 3 levels)
6. for a history or status concept, its SNOMED "Associated finding" (family-history concepts are skipped)

A MONDO term that PrimeKG lacks resolves to its nearest MONDO parent in PrimeKG, up to 2 levels. Each text is `Disease: <ICD title> (<MONDO name>). Symptoms: <Mayo Clinic symptoms>`; a code without symptoms gets a definition instead. Unmapped codes keep their ICD title, so every observed code gets an embedding. SNOMED nodes are encoded from their names, and ICD category nodes use the mean of their codes. Embeddings use masked mean pooling. Outputs in `processed_data/kg/`:
- `node_features_bert.pt`: `load_text_features()` gives `{key: 768-d tensor}`.
- `icd_mondo_map.parquet`: the code → MONDO → PrimeKG map.
- `icd_mondo_coverage.json`: aggregate coverage.

**Stage 2** (`src/hto/kg/enrich.py`). Features:
- Node: `x = [e_text (768) ‖ log1p(recency_days) ‖ log1p(frequency)]`, which is 770-d.
- Edge: `edge_attr = [sin/cos of τ at 8 periods from 1 day to 10 years (16) ‖ edge_hops / 4]`, which is 17-d. `edge_hops` is the length of the shortest observed-code-to-observed-code path the edge lies on, and 0 for an edge that belongs only to one code's own 1-hop knowledge.
- Recency and frequency use only diagnoses strictly before t0. A graph code seen only at t0 stays a plain node (`observed_mask = False`).

Storing x for every node of every patient would take tens of GB, so `patient_graphs_enriched.<kg>.pt` keeps compact PyG `Data` objects (global node ids, τ, hops, recency, frequency) plus one text table. `EnrichedPatientGraphs(path)[i]` returns the full `x` and `edge_attr`.

**Stage 3** (`src/hto/kg/treatment.py`). Inputs and matching:
- `V_p`: the PrimeKG disease nodes of the patient's diagnoses before t0.
- `M_active`: the DrugBank ingredients of prescriptions started in the 30 days before t0. Medications map through RxNorm: the NDC first, then the name, then to ingredients and their DrugBank ids.
- For each candidate T, the block intersects T's contraindication, indication and off-label edges with `V_p`, and its `drug_drug` edges with `M_active` (excluding T itself).

`treatment_context.pt` holds:
- `features [P, 24, 16]`, with names in `feature_names`
- `is_index_candidate [P, 24]`
- `patient_ids`
- `candidates`

`treatment_context_matches.parquet` lists every match with its evidence time. PrimeKG DDIs carry no severity, so they count as unrated unless `--severity` provides a `drug_a, drug_b, level` table.

**Full cohort** (`full_storetime`, `kg.csv`). Results:
- Temporal leakage is 0% in both g_y and g_a. Graph features are 770 / 17-d.
- 88.96% of diagnosis rows map to MONDO and have an embedding; every observed code has an embedding. This misses the 90% target. Excluding the supplementary V–Z / E codes (20% of rows), coverage is 97.07%. Most remaining unmapped rows are status, encounter or external-cause codes, or symptoms and abnormal findings, which MONDO does not model as diseases.
- The empty-context rate is 4.33% (4.46% for index pairs). 93.85% of medication rows map to DrugBank.

### Labs, vitals and symptoms: the context graph (`KG_TAG=ctx`)

The context KG adds measurements and symptoms to the team's diagnosis + lab SNOMED graph (`processed_data/kg/lab_snomed`, kept verbatim). Extra public input: HPO (`hp.json`, https://hpo.jax.org).

```bash
python build_context_kg.py --base-kg-dir processed_data/kg/lab_snomed \
  --icd-snomed-map processed_data/kg/full/icd_snomed_map.parquet --codes $P/tables/icd_inventory.csv \
  --labitems $P/tables/d_labitems.csv --hpo hp.json --primekg kg.csv --mondo mondo.json
python 1_extract_text_embeddings.py ... --kg-dir processed_data/kg/ctx     # texts for the new nodes too
python build_context_patient_graphs.py --dataset full_storetime --kg-tag ctx --mimic-dir $P/mimic_data
python 2_enrich_patient_graphs.py --dataset full_storetime --kg-tag ctx --mimic-dir $P/mimic_data
python 3_build_treatment_block.py --dataset full_storetime --mimic-dir $P/mimic_data --rxnorm-dir rxnorm/rrf
python eval_graph_enrichment.py --dataset full_storetime --kg-tag ctx --mimic-dir $P/mimic_data \
  --codes $P/tables/icd_inventory.csv
```

SLURM: `STAGE=ctxkg`, then `stage1` / `ctxgraphs` / `stage2` / `eval` with `KG_TAG=ctx`, then `stage3`, which picks the context up by itself.

**Measurements.** `src/hto/kg/resources/measurement_panel.tsv` holds 31 measurements: 26 labs (electrolytes, kidney, glucose / HbA1c, blood count, liver, lipids, TSH, NT-proBNP, urine albumin and protein ratios) and 5 vitals (systolic and diastolic BP from OMR and ICU, BMI, weight, height). The vitals table has no heart rate. Each measurement has optional low / high thresholds. A value outside them is an abnormal result with a state, `low` or `high`, and each state names the HPO phenotype and the PrimeKG disease it indicates. Names are resolved at build time, never guessed, and unresolved ones are listed in `graph_stats.json`.

**New nodes and edges** (each relation also gets its inverse):

| Node | Example | Edges |
|---|---|---|
| measure | `meas:potassium` | `measure_maps_to_snomed` → the SNOMED concepts its lab items map to (LOINC) |
| abnormal result | `meas:potassium:high` | `state_of` → measure; `indicates` → HPO phenotype |
| phenotype (HPO) | `HP:0002153` Hyperkalemia | `phenotype_is_a` → HPO parent |
| diagnosis (existing) | `10_E875` | `manifests_as` → its own symptom; `has_phenotype` → phenotypes of its PrimeKG disease |

`manifests_as` comes from the code's core SNOMED finding / disorder concept, or else its ICD title, matched exactly to HPO labels and exact synonyms. Current HPO releases no longer carry SNOMED xrefs. The core concept is the one named most like the ICD title, with a token Jaccard of at least 0.5. The ICD → SNOMED map is many-to-one, so without this rule `I10` picked up every concept that rolls up into it, e.g. "Hypertensive crisis". `has_phenotype` uses PrimeKG `disease_phenotype_positive`, at most 10 per code: phenotypes that an abnormal result or a symptom code can reach come first, then the most specific ones. An abnormal result and a diagnosis therefore meet at a shared phenotype, and the patient graph builder links them with the usual τ.

**Patient graphs.** The observed codes are the diagnoses plus `meas:<m>` for every measurement and `meas:<m>:<state>` for every abnormal one. Measurements are used only when strictly before t0. `x` stays 770-d and `edge_attr` 17-d. Stage 2 adds `value_attr [n, 4] = [has_value, z(last value), z(last − previous value), abnormal −1 / 0 / +1]` for measurement nodes, which is 0 elsewhere. z uses a robust median / IQR scale fitted on training patients' pre-t0 values, stored in `value_scale`.

**Treatment block.** The extra features, appended after the original 16:
- `side_effect_overlap_flag` and `matched_side_effect_count`: the candidate's PrimeKG `drug_effect` phenotypes ∩ the patient's symptoms before t0. Symptoms come from symptom codes and from current abnormal results.
- `measurement_contraindication_flag`, `matched_measurement_contraindication_count` and `measurement_indication_count`: the candidate's contraindications / indications ∩ the diseases indicated by current abnormal results. "Current" means the last value within 365 days before t0 is abnormal.
- `n_patient_phenotypes` and `n_abnormal_measurements`.

`treatment_context.pt` also gets `measurements [P, 31, 4]`, the same four value features per panel measurement, with `measurement_names`. Matches carry `evidence_source` (diagnosis / medication / symptom / measurement). The original empty-context rate keeps its definition; the rate that also counts side effects and measurement matches is reported next to it.

**Full cohort, context graph** (`graph_enrichment_eval.ctx.json`):
- **Context KG:** 99,848 entities, 76 relations, 812,814 triples. It adds 31 measures, 41 abnormal-result states (all resolved to HPO) and 3,379 HPO phenotypes. 1,063 ICD codes have `manifests_as` (21.8% of diagnosis rows, 49.1% of symptom-chapter rows), and 3,860 have `has_phenotype`.
- **Patient graphs:** 19.6M pre-t0 panel measurements over 38,311 patients. The observed codes are 775k diagnosis, 850k measurement and 431k abnormal-result codes. Every patient has measurement nodes and 99.2% have phenotype nodes. Median per patient: 22 measures, 32 abnormal-result nodes, 38 phenotype nodes. 37,813 patients have a lab–diagnosis path.
- **Leakage:** 0% in both g_y and g_a, measurements included. Dimensions are x 770, edge_attr 17, value_attr 4, all finite.
- **Treatment block:** the empty-context rate is 4.33% (original definition) and 2.06% counting side effects and measurement matches. 34.5k patients have symptoms before t0, and 34.3k have a current abnormal result that indicates a PrimeKG disease. Measurement coverage ranges from 99% (basic metabolic panel, blood count) through 89% (BP) and 51–58% (lipids, HbA1c) down to 9% (eGFR, which is rarely reported in MIMIC; creatinine covers 99%).
- **Diagnosis-code coverage is unchanged:** 88.96%, or 97.07% on disease codes.

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
build_context_kg.py         + labs / vitals / HPO symptoms → processed_data/kg/ctx/
build_context_patient_graphs.py  per-patient graphs on the context KG (diagnoses + pre-t0 measurements)
1_extract_text_embeddings.py, 2_enrich_patient_graphs.py, 3_build_treatment_block.py, eval_graph_enrichment.py
                            graph enrichment (section 4)
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
