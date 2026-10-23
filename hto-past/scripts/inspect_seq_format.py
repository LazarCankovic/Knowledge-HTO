"""Map the token vocabulary in final_sequences.parquet: bracket tokens (domain
markers) and their frequencies, plus whether any diagnosis markers appear."""
from pathlib import Path
import re
from collections import Counter

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent / "mimic_data"
seq = pd.read_parquet(ROOT / "final_sequences.parquet")

# unique bracket tokens across ALL patients (cheap: just scan each string)
bracket_pat = re.compile(r"\[[A-Za-z/][A-Za-z0-9_]*\]")

bracket_counts = Counter()
for s in seq["sequence"]:
    bracket_counts.update(bracket_pat.findall(s))
print(f"n patients: {len(seq)}")
print(f"distinct bracket tokens: {len(bracket_counts)}")
for tok, n in bracket_counts.most_common(40):
    print(f"  {tok:<12} {n:>10,}")

# sanity: show a couple fully decoded small-size examples (short patient)
short_idx = seq["sequence_length"].astype(int).idxmin()
r = seq.iloc[short_idx]
print(f"\n=== Shortest sequence (patient {r['patient_id']}, len={r['sequence_length']}) ===")
print(r["sequence"][:2000])

# is there a diag-related token anywhere?
diag_like = [t for t in bracket_counts if "DIAG" in t.upper() or t in ("[D]", "[ED]", "[BOD]", "[EOD]")]
print(f"\nDiagnosis-related bracket tokens found: {diag_like}")

# header (left of "|"): see how labels and targets are delimited
sample_heads = [s.split("|", 1)[0] + " | " for s in seq["sequence"].head(8)]
print("\nSample heads:")
for h in sample_heads:
    print(f"  {h!r}")

# What are the distinct "target med" strings the header carries?
targets = seq["latest_htn_med_name"].value_counts()
print(f"\ntarget med distribution:\n{targets.to_string()}")

# Section breakdown: count visits per patient = count of [EOV] occurrences
seq["n_eov"] = seq["sequence"].str.count(r"\[EOV\]")
print(f"\n[EOV] per patient: min={seq['n_eov'].min()} mean={seq['n_eov'].mean():.1f} max={seq['n_eov'].max()}")
print(f"sequence_length vs n_eov correlation: {seq[['sequence_length', 'n_eov']].corr().iloc[0,1]:.3f}")
