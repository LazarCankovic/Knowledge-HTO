# Dataset transformations per baseline

This document lists, for each baseline, (a) what the original paper / code
expected as input, (b) what we emit from the shared MIMIC-IV preprocessing
pipeline, (c) where the processed files are stored, and (d) any notable
adaptations.

The shared pipeline has two stages:

1. `build_data.py` aggregates raw per-event rows from `mimic_data/*.parquet`
   into **canonical records**: one dict per patient containing a list of
   `visits`, each with diagnosis codes, medication codes, numeric lab
   values, numeric vital values, and a `day_offset` (days since the
   patient's first observed event).
2. For each **variant** below, the canonical records are re-encoded into a
   baseline-specific tensor layout. Vocabularies are fit on TRAIN only and
   reused at TEST time. All baselines share the same patient-level 85/15
   split (`processed_data/<tag>/split.json`).

## Variants

| variant             | contents                                                                 | baselines using it                       |
|---------------------|--------------------------------------------------------------------------|------------------------------------------|
| `codes_visit`       | list of visits, each a list of code-ids (diag + med + lab-bin + vital-bin) with `day_offset` | T-LSTM, BEHRT, CEHR-BEHRT, TransformEHR (originals) |
| `codes_flat`        | single flat code stream per patient, plus `day_offset` per token         | Med-BERT, Time2Vec (originals)           |
| `multimodal`        | `codes_visit` + per-visit numeric vectors (labs + vitals) with masks     | Multimodal-BEHRT (original)              |
| `multidomain_visit` | **all 4 domains separated by delimiter tokens** (`[EOD]/[EOM]/[EOL]/[EOVI]/[EOV]`) + `day_offset` + `delta_prev_days` + per-token `domain_id` + numeric value/mask | All *_md baselines + mTAN + STraTS |
| `text_sequence`     | pre-tokenized text from `final_sequences.parquet`, whitespace+bracket    | optional Med-BERT variant                |

`multidomain_visit` mirrors the grammar of `final_sequences.parquet` (domain
blocks terminated by EO* markers) but **re-attaches day offsets and
visit-gap days that the text form drops**, and exposes the per-token
domain id + a parallel numeric stream so models can pick the encoding
they want.

Lab values and vital values are binned into pseudo-tokens
(`L_<key>_<log10 bucket>` and `V_<key>_<bucket>`) when added to the
token-based variants, so a uniform vocabulary covers everything.

## Original-input expectations vs. our adapters

### Med-BERT (Rasmy 2020)
- **Original**: diagnosis-code sequences per patient, BERT masked-LM + LOS
  pretraining, then fine-tuning. Uses code + segment (visit) + position.
- **Our adapter**: `codes_flat` stream with a `[CLS]` prefix. Segment
  embedding is the visit-index reconstructed from unique `day_offset`
  values. We skip MLM pretraining and train supervised from scratch.
- **Files**: `processed_data/<tag>/codes_flat.{train,test}.pkl`,
  `codes_flat.vocab.json`.
- **Caveats**: we include medications + binned labs + vitals as tokens
  (not just diagnoses) to exploit the richer MIMIC data, and we add a
  learned visit-segment embedding.

### Time-aware LSTM (Baytas 2017)
- **Original**: T-LSTM cell with g(Δt) = 1 / log(e + Δt) decaying short-term
  memory between visits. Input is a per-visit feature vector.
- **Our adapter**: `codes_visit`. Each visit's feature vector is the mean of
  its code embeddings; `Δt` is the gap in days between consecutive visits
  (first visit Δt=0). We keep the exact cell equations from the paper.
- **Files**: `processed_data/<tag>/codes_visit.{train,test}.pkl`,
  `codes_visit.vocab.json`.

### Time2Vec (Kazemi 2019)
- **Original**: learnable time embedding `t2v(t) = [ω₀t+φ₀, sin(ωᵢt+φᵢ)]`,
  usually plugged into a downstream seq model. Scale of `t` is dataset-
  dependent.
- **Our adapter**: scale `day_offset / 365` so the base frequency is yearly.
  Fuse `t2v(t)` (projected to `d_model`) with a small Transformer over
  `codes_flat`. `[CLS]` readout for classification.
- **Files**: same as Med-BERT (`codes_flat.*`).

### TransformEHR (Yang 2023)
- **Original**: hierarchical transformer — visit encoder, then patient
  encoder over visit embeddings. Uses a time embedding at the patient level.
- **Our adapter**: `codes_visit` fed directly. Visit encoder reads `[VCLS]`
  + codes; patient encoder reads `[PCLS]` + (visit_emb + linear(day_offset/
  365) + sinusoidal position).
- **Files**: `codes_visit.*`.

### BEHRT (Li 2020)
- **Original**: BERT over codes with token + age-at-visit + visit-position +
  alternating-segment embeddings. Visits are joined with `[SEP]`.
- **Our adapter**: age is not available in our preprocessed data, so we
  drop the age embedding and retain the other three. Visits are flattened
  with `[SEP]` separators in `codes_visit`.
- **Files**: `codes_visit.*`.

### CEHR-BEHRT (Pang 2021)
- **Original**: extends BEHRT with "ATT" (artificial time tokens) between
  visits — W (<1 wk) / M (<1 mo) / Y otherwise — plus continuous time
  embedding.
- **Our adapter**: ATT tokens are produced from the visit-to-visit `Δdays`
  using the paper's W/M/Y bucketing, stored via a side `att_ids` tensor
  (rather than expanding the vocab). A continuous `Linear(day_offset/365)`
  is also added to every token's embedding.
- **Files**: `codes_visit.*`.

### Multimodal-BEHRT (Mbaye 2025)
- **Original**: BEHRT with an additional numeric modality (lab / vital
  values) fused at the visit level.
- **Our adapter**: `multimodal` variant. Per visit we emit a fixed-size
  numeric vector (12 labs + 3 vitals, see `LAB_KEYS` / `VITAL_KEYS` in
  `hto.data.variants`) and a missingness mask. A two-layer MLP projects
  `[values | mask]` to `d_model`, added to the visit embedding before the
  patient transformer. Feature-wise z-score normalisation uses statistics
  computed from the first batch seen during training.
- **Files**: `processed_data/<tag>/multimodal.{train,test}.pkl`,
  `multimodal.vocab.json`.

## Raw predictions

Every run writes `predictions/<dataset>/<baseline>/`:

- `metrics.json` — summary metrics (AUC, accuracy, F1, precision, recall,
  confusion matrix) + per-epoch history + the full train config.
- `predictions.npz` — patient_ids, y_true, logits (float32, shape
  `[N, 2]`), y_prob, y_pred. Nothing is thresholded irreversibly.
- `confusion.png`, `roc.png`, `prob_hist.png` — diagnostic plots.

## Multidomain baselines (`*_md`)

Each original baseline has a multidomain counterpart that:

1. Reads `multidomain_visit` (so every visit exposes all 4 domains with
   explicit domain ids and numeric values where applicable).
2. Replaces its paper-specific time embedding with the **composite PE**
   below, with every component independently ablatable.
3. Optionally threads a **relative-day attention bias** through every
   transformer layer.

### Composite PE

```
PE(event) =  P_pos(visit_idx)                       # within-sequence position (always on)
          +  T_abs(day_offset)                      # Time2Vec over days (log-spaced ω: 1/day .. 1/10y)
          +  T_gap(delta_prev_days)                 # Time2Vec over visit-to-visit gaps
          +  D(domain_id)                           # learned embedding over {diag, med, lab, vital, struct}
          +  DT(domain_id, day_offset)              # MLP on concat([Time2Vec(day), onehot(domain)])
```

Select components at runtime via `--pe-parts` (e.g. `"abs,gap"`, `""` to
disable all optional components and keep only sinusoidal position).

### Relative-day attention bias

```
A_ij = (Q_i K_j^T) / sqrt(d)  +  φ_h(|Δday_i − Δday_j|)
```

`φ_h` is a small MLP on `[Δd, log(1+Δd), 1/(1+Δd)]` with a per-head
output. Toggled via `--rel-day-bias {on,off}`. Implemented in
`BiasableEncoder` (a small custom transformer stack that adds the bias to
attention logits every layer).

### Per-baseline adaptations

| baseline            | flat/hier | extra signal on top of CompositePE                              |
|---------------------|-----------|-----------------------------------------------------------------|
| `medbert_md`        | flat      | none (PE handles segment replacement)                           |
| `time2vec_md`       | flat      | defaults to `pe_parts=abs` (isolates Time2Vec mechanism)        |
| `behrt_md`          | flat      | visit-index + alternating A/B segment embedding                 |
| `cehrbert_md`       | flat      | W/M/Y "ATT" embedding at the first token of each new visit      |
| `transformehr_md`   | hier      | PE applied at patient level between visits                      |
| `multimodal_behrt_md`| hier     | hier + per-token numeric projector (labs/vitals)                |
| `tlstm_md`          | hier      | T-LSTM cell consumes visit embeddings + Δt                      |

## New irregular-time baselines

### mTAN (Shukla & Marlin, ICLR 2021)
R learnable reference time points cross-attend over observed events. Each
event is `tok_emb + domain_emb + num_proj(value) + time_proj(Time2Vec(t))`.
Reference queries: `ref_query + time_proj(Time2Vec(learnable_ref_times))`.
Pooling is mean over reference outputs → classifier.

### STraTS (Tipirneni & Reddy, KDD 2022)
Every observation is a triplet `(feature, value, time)`. Encoded as
`tok_emb(feature) + domain_emb + time_proj(Time2Vec(t)) + val_proj(Fourier(z(value)))`.
A learned `[CLS]` attends over the set; its hidden state is the patient
summary. Value standardisation uses running mean/std fit on the first
training batch.

## Metrics

`metrics.json` per run:
- `auc`, `auprc`, `brier`, `ece` (equal-width 15-bin)
- `accuracy`, `f1`, `precision`, `recall`, `confusion` at threshold 0.5
- `by_target`: stratified AUC / AUPRC / accuracy / F1 for each
  `latest_htn_med_name` bucket (metoprolol / hydrochlorothiazide /
  carvedilol / atenolol) — the population is skewed 77/23 positive and the
  4 target-meds have very different support (25,304 / 1,923 / 1,903 /
  1,293), so per-target stratification is important.
- `extra.pe_parts`, `extra.rel_day_bias`: captured so the ablation table
  writes itself from `predictions/**/metrics.json`.

Plots: `confusion.png`, `roc.png`, `pr.png`, `prob_hist.png`.

## Reproducibility

- Seed: 42 everywhere unless `--seed` is passed. Applies to
  `random`, `numpy`, and `torch` (+ CUDA).
- The 85/15 patient split is seeded and persisted as `split.json`, so the
  same patients land in train vs. test across every variant and baseline.
