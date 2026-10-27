"""Execute MIMIC_IV_Extraction.ipynb against a given MIMIC-IV folder and save the executed copy.

    python extraction/run_extraction.py --raw-dir /path/to/physionet.org/files/mimiciv/3.1 --dataset mimic-iv-3.1

Settings reach the notebook as HTO_* environment variables. The template notebook is never modified.
The kernel is always the Python interpreter running this script, so run it from the environment that
has requirements.txt installed. Outputs go to data/processed/<dataset>/ (or --out-dir). The executed
notebook, with every printed count, is saved to extraction/runs/MIMIC_IV_Extraction_<dataset>.ipynb.
"""
import argparse
import os
import sys
from pathlib import Path

import nbformat
from jupyter_client.kernelspec import KernelSpec
from jupyter_client.manager import KernelManager
from nbclient import NotebookClient

PIPELINE_DIR = Path(__file__).resolve().parent
TEMPLATE = PIPELINE_DIR / "MIMIC_IV_Extraction.ipynb"


class ThisPythonKernel(KernelManager):
    """Launch ipykernel with sys.executable instead of whatever 'python3' kernelspec is installed."""

    @property
    def kernel_spec(self):
        return KernelSpec(argv=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                          display_name="Python (run_extraction)", language="python")


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--raw-dir", required=True, help="folder containing hosp/ and icu/ (.csv or .csv.gz)")
ap.add_argument("--dataset", required=True, help="output name, e.g. mimic-iv-3.1 (also selects reference numbers)")
ap.add_argument("--out-dir", help="default: data/processed/<dataset>")
ap.add_argument("--cache-dir", help="parquet cache, default: data/cache/<dataset>")
ap.add_argument("--no-cache", action="store_true", help="query the CSVs directly (slower on re-runs)")
ap.add_argument("--memory-limit", help='DuckDB memory limit, e.g. "8GB" (default: 80%% of RAM)')
ap.add_argument("--threads", type=int, help="DuckDB threads (default: all cores)")
ap.add_argument("--dx-time-policy", choices=["discharge", "admission"], default="discharge")
ap.add_argument("--require-usable-dx", action="store_true")
ap.add_argument("--no-icu-vitals", action="store_true", help="skip ICU chartevents")
ap.add_argument("--lab-lookback-days", type=int, help="keep only labs from the last N days before index")
ap.add_argument("--show-sample-rows", action="store_true",
                help="include patient-level sample rows in the saved notebook (off by default so the executed "
                     "notebook holds aggregates only and can be shared outside the credentialed team)")
args = ap.parse_args()

raw_dir = Path(args.raw_dir).expanduser().resolve()
missing = [m for m in ("hosp",) + (() if args.no_icu_vitals else ("icu",)) if not (raw_dir / m).is_dir()]
if missing:
    raise SystemExit(f"{raw_dir} is missing {missing} folder(s)")

settings = {
    "HTO_RAW_DIR": str(raw_dir),
    "HTO_DATASET": args.dataset,
    "HTO_DX_TIME_POLICY": args.dx_time_policy,
    "HTO_REQUIRE_USABLE_DX": str(args.require_usable_dx),
    "HTO_INCLUDE_ICU_VITALS": str(not args.no_icu_vitals),
    "HTO_USE_PARQUET_CACHE": str(not args.no_cache),
    "HTO_SHOW_SAMPLE_ROWS": str(args.show_sample_rows),
}
optional = {"HTO_OUT_DIR": args.out_dir, "HTO_CACHE_DIR": args.cache_dir,
            "HTO_DUCKDB_MEMORY_LIMIT": args.memory_limit, "HTO_DUCKDB_THREADS": args.threads,
            "HTO_LAB_LOOKBACK_DAYS": args.lab_lookback_days}
settings.update({k: str(Path(v).expanduser().resolve()) if k in ("HTO_OUT_DIR", "HTO_CACHE_DIR") else str(v)
                 for k, v in optional.items() if v is not None})
os.environ.update(settings)   # inherited by the kernel process

nb = nbformat.read(TEMPLATE, as_version=4)
out_nb = PIPELINE_DIR / "runs" / f"MIMIC_IV_Extraction_{args.dataset}.ipynb"
out_nb.parent.mkdir(exist_ok=True)

print(f"Python: {sys.executable}")
for k, v in settings.items():
    print(f"  {k}={v}")
try:
    NotebookClient(nb, km=ThisPythonKernel(), timeout=None,
                   resources={"metadata": {"path": str(PIPELINE_DIR)}}).execute()
    print("Finished OK")
finally:
    nbformat.write(nb, out_nb)
    print("Executed notebook saved to", out_nb)
