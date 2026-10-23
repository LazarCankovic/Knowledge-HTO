import json, os

BASELINES = [
    "medbert", "tlstm", "time2vec", "transformehr",
    "behrt", "cehrbert", "multimodal_behrt",
    "medbert_md", "tlstm_md", "time2vec_md", "transformehr_md",
    "behrt_md", "cehrbert_md", "multimodal_behrt_md",
    "mtan", "strats", "htr_former", "a1_time", "a2_no_time", "htr_former_ca",
]

rows, missing = [], []
for b in BASELINES:
    path = f"predictions/full_70_10_20/{b}/metrics.json"
    if not os.path.exists(path):
        missing.append(b)
        continue
    m = json.load(open(path))
    cfg = m.get("extra", {}).get("config", {})
    rows.append({
        "baseline": b,
        "auroc":    m.get("auc"),
        "auprc":    m.get("auprc"),
        "accuracy": m.get("accuracy"),
        "f1":       m.get("f1"),
        "brier":    m.get("brier"),
        "ece":      m.get("ece"),
        "epochs":   cfg.get("epochs", "?"),
        "n_params": m.get("extra", {}).get("n_params", "?"),
    })

rows.sort(key=lambda r: r.get("auroc") or 0, reverse=True)

def f(v): return f"{v:.4f}" if v is not None else "-"

header = f"{'Baseline':<25} {'AUROC':>7} {'AUPRC':>7} {'ACC':>7} {'F1':>7} {'Brier':>7} {'ECE':>7} {'Epochs':>7} {'Params':>10}"
print("\n" + "-"*len(header))
print(header)
print("-"*len(header))
for r in rows:
    print(f"{r['baseline']:<25} {f(r['auroc']):>7} {f(r['auprc']):>7} {f(r['accuracy']):>7} {f(r['f1']):>7} {f(r['brier']):>7} {f(r['ece']):>7} {str(r['epochs']):>7} {str(r['n_params']):>10}")
print("-"*len(header))

if missing:
    print(f"\nMissing: {', '.join(missing)}")
