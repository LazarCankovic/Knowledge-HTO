"""Export the LAST-HTO cohort in mimic_iv_analysis.py to hto-ker-diag inputs.

Run from the authenticated Colab notebook after df_last_cohort has been computed:

    from extraction.export_from_bigquery import export_cohort
    export_cohort(client, df_last_cohort, '/content/hto_export')

The input is the cohort DataFrame, not the notebook's truncated sequence CSV.
Requires permission to create a temporary table in the billing project. The
temporary cohort table is deleted even when an event query fails. Keep the
export on approved storage for credentialed MIMIC-IV data.
"""
from __future__ import annotations

import re
import shutil
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from google.cloud import bigquery


REQUIRED = {"patient_id", "index_time", "index_hto_drugs", "median_sbp_90", "median_dbp_90"}


def prepare_patients(cohort: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED - set(cohort.columns)
    if missing:
        raise ValueError(f"Run the LAST-HTO cohort cell first; missing {sorted(missing)}")
    if cohort["patient_id"].isna().any() or cohort["patient_id"].duplicated().any():
        raise ValueError("Expected one non-null row per patient_id")
    df = cohort.copy()
    df["patient_id"] = df["patient_id"].astype("int64")
    df["index_time"] = pd.to_datetime(df["index_time"])
    if df["index_time"].isna().any() or df["index_time"].dt.tz is not None:
        raise ValueError("index_time must be a non-null, timezone-naive MIMIC DATETIME")
    for col in ("median_sbp_90", "median_dbp_90"):
        df[col] = pd.to_numeric(df[col], errors="raise")
    if df[["median_sbp_90", "median_dbp_90"]].isna().any().any():
        raise ValueError("The LAST-HTO cohort must have both post-index BP medians")
    df = df.sort_values("patient_id").reset_index(drop=True)
    df["label"] = ((df["median_sbp_90"] < 120) & (df["median_dbp_90"] < 80)).astype("int64")
    df["latest_htn_med_name"] = df["index_hto_drugs"]
    df["bps"] = df["median_sbp_90"].astype(float)
    df["bpd"] = df["median_dbp_90"].astype(float)
    df["latest_bp_time"] = df["index_time"]
    rng = np.random.default_rng(42)
    df["split"] = ""
    for _, positions in sorted(df.groupby("label").indices.items()):
        positions = rng.permutation(np.sort(positions))
        n_train = round(0.70 * len(positions))
        n_val = round(0.15 * len(positions))
        df.loc[positions[:n_train], "split"] = "train"
        df.loc[positions[n_train:n_train + n_val], "split"] = "val"
        df.loc[positions[n_train + n_val:], "split"] = "test"
    return df


def _normalize_drug(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"\b\d+(\.\d+)?\s*(mg|mcg|g|ml|meq|units|unit|%)(\s*/\s*(ml|l|hr|hour))?\b", " ", text)
    text = re.sub(r"\b\d+(\.\d+)?\s*%", " ", text)
    text = re.sub(r"\b(tablet|tablets|tab|capsule|capsules|cap|solution|suspension|injection|inj|vial|syringe|ampule|amp|oral|po|iv|im|sc|subcutaneous|extended release|immediate release|delayed release|er|xr|xl|sr|dr|mini bag plus|mini bag|bag|cream|ointment|gel|patch|spray|neb|nebulizer|inhalation)\b", " ", text)
    text = re.sub(r"[()\[\]{},;]", " ", text)
    text = re.sub(r"[/\\]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return {"morphine sulfate": "morphine", "docusate sodium": "docusate",
            "metoprolol tartrate": "metoprolol", "metoprolol succinate": "metoprolol",
            "amlodipine besylate": "amlodipine"}.get(text, text)


def _write_query(client, sql: str, destination: Path) -> int:
    """Stream BigQuery results to an atomic Parquet file, bounded by one batch."""
    temporary = destination.with_suffix(".parquet.partial")
    writer = None
    count = 0
    try:
        for batch in client.query(sql).result().to_arrow_iterable(max_queue_size=2):
            if writer is None:
                writer = pq.ParquetWriter(temporary, batch.schema, compression="snappy")
            writer.write_batch(batch)
            count += batch.num_rows
        if writer is None:
            raise ValueError(f"No rows returned for {destination.name}; check the cohort and source tables")
        writer.close()
        writer = None
        temporary.replace(destination)
    finally:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
    print(f"{destination.name}: {count:,} rows")
    return count


def export_cohort(client: bigquery.Client, df_last_cohort: pd.DataFrame,
                  out_dir: str | Path, scratch_dataset: str | None = None,
                  hosp: str = "physionet-data.mimiciv_3_1_hosp",
                  icu: str = "physionet-data.mimiciv_3_1_icu") -> None:
    """Export five final_* tables, d_labitems, ICD inventory and patient splits.

    ``scratch_dataset`` is a writable dataset in the billing project; if absent,
    ``<client.project>.hto_extract`` is created in US (MIMIC BigQuery location).
    """
    df = prepare_patients(df_last_cohort)
    if not re.fullmatch(r"[\w-]+\.[\w]+", scratch_dataset or f"{client.project}.hto_extract"):
        raise ValueError("scratch_dataset must be project.dataset")
    scratch_dataset = scratch_dataset or f"{client.project}.hto_extract"
    out = Path(out_dir).expanduser().resolve()
    if (out / "mimic_data").exists() or (out / "tables").exists():
        raise FileExistsError(f"{out} already has an export; choose a fresh directory")
    stage = out / f"_staging_{uuid.uuid4().hex}"
    data_dir, table_dir = stage / "mimic_data", stage / "tables"
    data_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    # Do not write any patient files before the BigQuery upload succeeds.
    dataset = bigquery.Dataset(scratch_dataset)
    dataset.location = "US"
    client.create_dataset(dataset, exists_ok=True)
    cohort_table = f"{scratch_dataset}.cohort_{uuid.uuid4().hex}"
    med_table = f"{scratch_dataset}.meds_{uuid.uuid4().hex}"
    created = []
    try:
        config = bigquery.LoadJobConfig(schema=[bigquery.SchemaField("patient_id", "INT64"),
                                                 bigquery.SchemaField("index_time", "DATETIME")])
        client.load_table_from_dataframe(df[["patient_id", "index_time"]], cohort_table,
                                         job_config=config).result()
        created.append(cohort_table)
        c = f"`{cohort_table}`"
        h = f"`{hosp}"
        dx_sql = f"""
        SELECT c.patient_id, a.hadm_id AS visit_id, a.dischtime AS time,
          CONCAT(CAST(d.icd_version AS STRING), '_', UPPER(TRIM(d.icd_code))) AS diag,
          CONCAT('DX_', CAST(d.icd_version AS STRING), '_', UPPER(TRIM(d.icd_code))) AS token,
          d.icd_version, UPPER(TRIM(d.icd_code)) AS icd_code,
          d.seq_num, dd.long_title,
          (a.admittime <= c.index_time AND a.dischtime > c.index_time) AS is_index_admission
        FROM {c} c
        JOIN {h}.admissions` a ON a.subject_id = c.patient_id
             AND a.dischtime <= c.index_time
        JOIN {h}.diagnoses_icd` d ON d.hadm_id = a.hadm_id
        LEFT JOIN {h}.d_icd_diagnoses` dd
          ON dd.icd_code = d.icd_code AND dd.icd_version = d.icd_version
        """
        _write_query(client, dx_sql, data_dir / "final_diagnoses.parquet")

        raw_med_sql = f"""
        SELECT DISTINCT LOWER(p.drug) AS drug
        FROM {c} c JOIN {h}.prescriptions` p ON p.subject_id = c.patient_id
        WHERE p.drug IS NOT NULL AND p.starttime IS NOT NULL
              AND p.starttime <= c.index_time
        """
        names = client.query(raw_med_sql).result().to_dataframe()
        names["medication"] = names["drug"].map(_normalize_drug)
        names = names[names["medication"] != ""].copy()
        names["token"] = "MED_" + names["medication"].str.upper().str.replace(r"[^A-Z0-9]+", "_", regex=True).str.strip("_")
        client.load_table_from_dataframe(names, med_table).result()
        created.append(med_table)
        meds_sql = f"""
        SELECT patient_id, visit_id, time, medication, token, drug, route, ndc
        FROM (
          SELECT c.patient_id, p.hadm_id AS visit_id, p.starttime AS time,
                 m.medication, m.token, LOWER(p.drug) AS drug, p.route, p.ndc,
                 ROW_NUMBER() OVER (PARTITION BY c.patient_id, p.hadm_id, m.token
                                    ORDER BY p.starttime, p.drug, p.route, p.ndc) AS rn
          FROM {c} c JOIN {h}.prescriptions` p ON p.subject_id = c.patient_id
          JOIN {h}.admissions` a ON a.hadm_id = p.hadm_id
                                AND a.admittime <= c.index_time
          JOIN `{med_table}` m ON m.drug = LOWER(p.drug)
          WHERE p.starttime IS NOT NULL AND p.starttime <= c.index_time
        ) WHERE rn = 1
        """
        _write_query(client, meds_sql, data_dir / "final_medications.parquet")

        labs_sql = f"""
        SELECT c.patient_id, l.hadm_id AS visit_id, l.storetime AS time,
               'storetime' AS time_source, l.itemid, di.label AS lab_key,
               di.fluid, di.category, l.valuenum AS lab_value,
               l.valueuom, l.ref_range_lower, l.ref_range_upper, l.flag
        FROM {c} c JOIN {h}.labevents` l ON l.subject_id = c.patient_id
        LEFT JOIN {h}.d_labitems` di ON di.itemid = l.itemid
        WHERE l.valuenum IS NOT NULL AND l.storetime IS NOT NULL
              AND l.storetime <= c.index_time
        """
        _write_query(client, labs_sql, data_dir / "final_labs.parquet")

        vitals_sql = f"""
        WITH omr AS (
          SELECT c.patient_id, o.chartdate, o.result_name, o.result_value
          FROM {c} c JOIN {h}.omr` o ON o.subject_id = c.patient_id
          WHERE o.chartdate < DATE(c.index_time)
        ), bp AS (
          SELECT *, SAFE_CAST(REGEXP_EXTRACT(result_value, r'^\\s*(\\d{{2,3}})\\s*/') AS INT64) AS sbp,
                    SAFE_CAST(REGEXP_EXTRACT(result_value, r'/\\s*(\\d{{1,3}})\\s*$') AS INT64) AS dbp
          FROM omr WHERE LOWER(result_name) LIKE '%blood pressure%'
        ), clean_bp AS (
          SELECT * FROM bp WHERE sbp > 0 AND dbp > 0 AND sbp > dbp
        )
        SELECT patient_id, CAST(NULL AS INT64) AS visit_id,
               DATETIME(chartdate) AS time, 'bps' AS vital_key,
               CAST(sbp AS FLOAT64) AS vital_value, 'omr' AS source FROM clean_bp
        UNION ALL
        SELECT patient_id, NULL, DATETIME(chartdate), 'bpd', CAST(dbp AS FLOAT64), 'omr' FROM clean_bp
        UNION ALL
        SELECT patient_id, NULL, DATETIME(chartdate),
               CASE WHEN result_name LIKE 'BMI%' THEN 'bmi'
                    WHEN result_name LIKE 'Weight%' THEN 'weight_lbs'
                    WHEN result_name LIKE 'Height%' THEN 'height_in' END,
               SAFE_CAST(result_value AS FLOAT64), 'omr'
        FROM omr
        WHERE (result_name LIKE 'BMI%' OR result_name LIKE 'Weight%' OR result_name LIKE 'Height%')
          AND SAFE_CAST(result_value AS FLOAT64) IS NOT NULL
        UNION ALL
        SELECT c.patient_id, ce.hadm_id, ce.storetime,
               CASE WHEN ce.itemid IN (220179, 220050) THEN 'bps' ELSE 'bpd' END,
               ce.valuenum, 'icu'
        FROM {c} c JOIN `{icu}.chartevents` ce ON ce.subject_id = c.patient_id
        WHERE ce.itemid IN (220179, 220050, 220180, 220051)
          AND ce.valuenum IS NOT NULL AND ce.storetime IS NOT NULL
          AND ce.storetime <= c.index_time
          AND ((ce.itemid IN (220179, 220050) AND ce.valuenum BETWEEN 40 AND 300)
            OR (ce.itemid IN (220180, 220051) AND ce.valuenum BETWEEN 20 AND 200))
        """
        _write_query(client, vitals_sql, data_dir / "final_vitals.parquet")

        labitems = client.query(f"SELECT itemid, label, fluid, category FROM {h}.d_labitems`").result().to_dataframe()
        labitems.to_csv(table_dir / "d_labitems.csv", index=False)
        inventory = client.query(f"""
          SELECT icd_version, icd_code, ANY_VALUE(long_title) AS long_title,
                 COUNT(*) AS n_rows_usable, COUNT(DISTINCT patient_id) AS n_patients_usable
          FROM ({dx_sql})
          GROUP BY icd_version, icd_code
        """).result().to_dataframe()
        inventory.to_csv(table_dir / "icd_inventory.csv", index=False)

        df[["patient_id", "label", "split"]].to_csv(table_dir / "splits.csv", index=False)
        df.to_parquet(data_dir / "final_patients_table.parquet", index=False)
        data_dir.replace(out / "mimic_data")
        table_dir.replace(out / "tables")
        print(f"Exported {len(df):,} patients to {out}; train/val/test splits in tables/splits.csv")
    finally:
        for table in created:
            client.delete_table(table, not_found_ok=True)
        shutil.rmtree(stage, ignore_errors=True)
