#!/usr/bin/env python
"""
inspect_patients.py  (was: four_towers_timeaware_all_codes_attention_weights.py)

Qualitative and quantitative inspection of FourTowersTimeAwareAllCodes on
real test patients.  Uses model.forward_explain() — no hooks, no re-entrancy.

Console output per patient:
  - Per-visit table: tower embedding norms, fused norm, visit_attn, temporal PE norm
  - Temporal encoding analysis table
  - Top-K attended codes for all 4 domains (diag / med / lab / vital)

File output:
  - explain_output/<patient_id>.explain.json  (all raw tensors as lists)

Optional:
  - --run-ablation: domain ablation AUC table (baseline + 4 domain-zeroed passes)

Usage:
  python inspect_patients.py
  python inspect_patients.py --run-ablation
  python inspect_patients.py --n 10 --top-k 8 --seed 7 --device cuda
  python inspect_patients.py --ckpt /path/to/ckpt.pt --vocab /path/vocab.json \\
      --test /path/test.pkl
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path

import torch
from tabulate import tabulate

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT / "src"))

from hto.models.four_towers_timeaware_all_codes_backup import FourTowersTimeAwareAllCodes
from hto.data.lab_vital_tokenizer import build_lab_sequence, build_vital_sequence

VOCAB_PATH = ROOT / "processed_data" / "full_80_20" / "allcodes_timeaware.vocab.json"
TEST_PATH  = ROOT / "processed_data" / "full_80_20" / "multidomain_visit.test.pkl"
CKPT_PATH  = ROOT / "checkpoints" / "four_towers_timeaware_allcodes_best.pt"


# ── batch building ────────────────────────────────────────────────────────────
# Replicated locally to avoid importing batch_builder_all_codes which has a
# hardcoded HPC vocab path.

def _pad_code_visits(visit_lists, max_visits, max_codes):
    B = len(visit_lists)
    ids   = torch.zeros((B, max_visits, max_codes), dtype=torch.long)
    mask  = torch.zeros((B, max_visits, max_codes), dtype=torch.float32)
    vmask = torch.zeros((B, max_visits),             dtype=torch.float32)
    for i, visits in enumerate(visit_lists):
        visits = visits[-max_visits:]
        for j, codes in enumerate(visits):
            vmask[i, j] = 1.0
            codes = codes[:max_codes]
            if codes:
                ids[i, j, :len(codes)]  = torch.tensor(codes, dtype=torch.long)
                mask[i, j, :len(codes)] = 1.0
    return ids, mask, vmask


def _ids_from_tokens(tokens, token_to_id, unk_id):
    return [token_to_id.get(t, unk_id) for t in tokens]


def build_single_batch(item, vocab, token_to_id,
                       max_visits=64, max_diag=64, max_med=64,
                       max_lab=128, max_vit=128):
    """Build a B=1 batch for one patient dict.
    Returns (batch, n_real_visits, days_arr).
    """
    unk_id = token_to_id.get("[UNK]", 1)
    visits = item["visits"][-max_visits:]
    V = len(visits)

    days_arr, diag_v, med_v, lab_v, vit_v = [], [], [], [], []
    for visit in visits:
        days_arr.append(float(visit["day_offset"]) / 365.0)
        diag_v.append(list(map(int, visit["blocks"].get("diag", []))))
        med_v.append(list(map(int, visit["blocks"].get("med", []))))

        lab_raw  = visit["blocks"].get("lab", [])
        lab_vals = visit["block_numeric"].get("lab", [])
        lab_toks = [vocab[t] for t in lab_raw if 0 <= t < len(vocab)]
        lab_v.append(_ids_from_tokens(build_lab_sequence(lab_toks, lab_vals), token_to_id, unk_id))

        vit_raw  = visit["blocks"].get("vital", [])
        vit_vals = visit["block_numeric"].get("vital", [])
        vit_toks = [vocab[t] for t in vit_raw if 0 <= t < len(vocab)]
        vit_v.append(_ids_from_tokens(build_vital_sequence(vit_toks, vit_vals), token_to_id, unk_id))

    diag_ids, diag_mask, _ = _pad_code_visits([diag_v], max_visits, max_diag)
    med_ids,  med_mask,  _ = _pad_code_visits([med_v],  max_visits, max_med)
    lab_ids,  lab_mask,  _ = _pad_code_visits([lab_v],  max_visits, max_lab)
    vit_ids,  vit_mask,  _ = _pad_code_visits([vit_v],  max_visits, max_vit)

    vmask        = torch.zeros(1, max_visits)
    vmask[0, :V] = 1.0
    days         = torch.zeros(1, max_visits)
    days[0, :V]  = torch.tensor(days_arr)

    return {
        "diag_ids": diag_ids, "diag_mask": diag_mask,
        "med_ids":  med_ids,  "med_mask":  med_mask,
        "lab_ids":  lab_ids,  "lab_mask":  lab_mask,
        "vit_ids":  vit_ids,  "vit_mask":  vit_mask,
        "days": days, "vmask": vmask,
    }, V, days_arr


def build_multi_batch(items, vocab, token_to_id,
                      max_visits=64, max_diag=64, max_med=64,
                      max_lab=128, max_vit=128):
    """Build a B>1 batch dict from a list of patient items.

    Replicates batch_builder_all_codes.build_allcodes_batch without the
    hardcoded HPC vocab path.
    """
    unk_id = token_to_id.get("[UNK]", 1)
    B = len(items)
    labels = torch.zeros(B, dtype=torch.long)
    days   = torch.zeros(B, max_visits)

    diag_v_all, med_v_all, lab_v_all, vit_v_all = [], [], [], []

    for i, item in enumerate(items):
        labels[i] = int(item["label"])
        visits = item["visits"][-max_visits:]

        diag_v, med_v, lab_v, vit_v = [], [], [], []
        for j, visit in enumerate(visits):
            days[i, j] = float(visit["day_offset"]) / 365.0

            diag_v.append(list(map(int, visit["blocks"].get("diag", []))))
            med_v.append(list(map(int, visit["blocks"].get("med", []))))

            lab_raw  = visit["blocks"].get("lab", [])
            lab_vals = visit["block_numeric"].get("lab", [])
            lab_toks = [vocab[t] for t in lab_raw if 0 <= t < len(vocab)]
            lab_v.append(_ids_from_tokens(build_lab_sequence(lab_toks, lab_vals), token_to_id, unk_id))

            vit_raw  = visit["blocks"].get("vital", [])
            vit_vals = visit["block_numeric"].get("vital", [])
            vit_toks = [vocab[t] for t in vit_raw if 0 <= t < len(vocab)]
            vit_v.append(_ids_from_tokens(build_vital_sequence(vit_toks, vit_vals), token_to_id, unk_id))

        diag_v_all.append(diag_v)
        med_v_all.append(med_v)
        lab_v_all.append(lab_v)
        vit_v_all.append(vit_v)

    diag_ids, diag_mask, diag_vmask = _pad_code_visits(diag_v_all, max_visits, max_diag)
    med_ids,  med_mask,  med_vmask  = _pad_code_visits(med_v_all,  max_visits, max_med)
    lab_ids,  lab_mask,  lab_vmask  = _pad_code_visits(lab_v_all,  max_visits, max_lab)
    vit_ids,  vit_mask,  vit_vmask  = _pad_code_visits(vit_v_all,  max_visits, max_vit)

    vmask = torch.maximum(diag_vmask, med_vmask)
    vmask = torch.maximum(vmask,      lab_vmask)
    vmask = torch.maximum(vmask,      vit_vmask)

    return {
        "diag_ids": diag_ids, "diag_mask": diag_mask,
        "med_ids":  med_ids,  "med_mask":  med_mask,
        "lab_ids":  lab_ids,  "lab_mask":  lab_mask,
        "vit_ids":  vit_ids,  "vit_mask":  vit_mask,
        "days": days, "vmask": vmask, "labels": labels,
    }


# ── attention display ─────────────────────────────────────────────────────────

def top_codes_per_visit(domain_attn, ids, mask, vocab, V, days_arr, top_k):
    """Return tabulate-ready rows for the top-k attended codes across real visits.

    domain_attn: [1, max_V, L]  CLS→code attention from forward_explain
    ids:         [1, max_V, L]
    mask:        [1, max_V, L]
    V:           number of real visits to display
    days_arr:    list of normalised day values (multiply by 365 for display)
    Returns list of [visit_num, day_str, token_str, weight_str].
    """
    L = ids.shape[2]
    rows = []
    for j in range(V):
        scores    = domain_attn[0, j, :L].clone()
        code_mask = mask[0, j, :L].bool()
        if not code_mask.any():
            continue
        scores[~code_mask] = -1.0
        k = min(top_k, int(code_mask.sum().item()))
        topk_vals, topk_idxs = torch.topk(scores, k)
        for idx, w in zip(topk_idxs.tolist(), topk_vals.tolist()):
            vid = ids[0, j, idx].item()
            tok = vocab[vid] if 0 <= vid < len(vocab) else f"<id:{vid}>"
            rows.append([j + 1, f"{days_arr[j] * 365:.0f}", tok, f"{w:.4f}"])
    return rows


# ── patient sampling ──────────────────────────────────────────────────────────

def sample_patients(items, n, seed):
    pos = [it for it in items if int(it["label"]) == 1]
    neg = [it for it in items if int(it["label"]) == 0]
    rng = random.Random(seed)
    n_pos = max(1, (n + 1) // 2)
    n_neg = n - n_pos
    chosen = (
        rng.sample(pos, min(n_pos, len(pos)))
        + rng.sample(neg, min(n_neg, len(neg)))
    )
    rng.shuffle(chosen)
    return chosen


# ── printing ──────────────────────────────────────────────────────────────────

def print_patient(item, explain, batch, V, days_arr, vocab, top_k):
    pid     = item["patient_id"]
    label   = int(item["label"])
    p_pos   = float(explain["prob_pos"][0].item())
    pred    = int(p_pos > 0.5)
    correct = "✓" if pred == label else "✗"

    cls_norm = explain["cls_embedding"][0].norm().item()
    print("\n" + "═" * 80)
    print(f"  Patient {pid}  |  label={label}  pred={pred} {correct}  P(pos)={p_pos:.4f}")
    print(f"  Patient CLS  ‖·‖={cls_norm:.4f}   visits={V}")
    print("═" * 80)

    # ── per-visit summary table ───────────────────────────────────────────────
    rows = []
    for j in range(V):
        day = days_arr[j] * 365.0
        rows.append([
            j + 1,
            f"{day:.0f}",
            f"{explain['diag_embedding_norm'][0, j].item():.3f}",
            f"{explain['med_embedding_norm'][0, j].item():.3f}",
            f"{explain['lab_embedding_norm'][0, j].item():.3f}",
            f"{explain['vit_embedding_norm'][0, j].item():.3f}",
            f"{explain['fused_visit'][0, j].norm().item():.3f}",
            f"{explain['visit_attn'][0, j].item():.4f}",
            f"{explain['temporal_pe'][0, j].norm().item():.4f}",
        ])
    print(tabulate(
        rows,
        headers=["#", "Day", "‖diag‖", "‖med‖", "‖lab‖",
                 "‖vit‖", "‖fused‖", "visit_attn", "‖tpe‖"],
        tablefmt="rounded_outline",
    ))

    # ── temporal encoding analysis ────────────────────────────────────────────
    print("\n  Temporal encoding analysis:")
    t_rows = []
    for j in range(V):
        day_abs = days_arr[j] * 365.0
        dsf     = (days_arr[j] - days_arr[0]) * 365.0
        t_rows.append([
            j + 1,
            f"{day_abs:.0f}",
            f"{dsf:.0f}",
            f"{explain['temporal_pe'][0, j].norm().item():.4f}",
            f"{explain['visit_attn'][0, j].item():.4f}",
            f"{p_pos:.4f}",
        ])
    print(tabulate(
        t_rows,
        headers=["Visit", "Day", "days_since_first", "‖tpe‖", "visit_attn", "P(pos)"],
        tablefmt="simple",
    ))

    # ── per-domain top-K code attention ───────────────────────────────────────
    DOMAIN_KEYS = [
        ("diag", "diag_ids", "diag_mask"),
        ("med",  "med_ids",  "med_mask"),
        ("lab",  "lab_ids",  "lab_mask"),
        ("vit",  "vit_ids",  "vit_mask"),
    ]
    for domain, ids_key, mask_key in DOMAIN_KEYS:
        attn_rows = top_codes_per_visit(
            explain["domain_attn"][domain],
            batch[ids_key].cpu(),
            batch[mask_key].cpu(),
            vocab, V, days_arr, top_k,
        )
        if attn_rows:
            print(f"\n  Top-{top_k} {domain} codes by attention weight:")
            print(tabulate(attn_rows,
                           headers=["Visit", "Day", "Token", "Attn Weight"],
                           tablefmt="simple"))
        else:
            print(f"\n  (no {domain} codes found in these visits)")


# ── JSON dump ─────────────────────────────────────────────────────────────────

def write_explain_json(item, explain, days_arr, V, out_dir: Path):
    """Write a per-patient JSON file with all ExplainOutput tensors."""
    pid = item["patient_id"]
    out_dir.mkdir(parents=True, exist_ok=True)

    def _t(tensor):
        return tensor.tolist()

    data = {
        "meta": {
            "patient_id": int(pid),
            "label":      int(item["label"]),
            "prob_pos":   float(explain["prob_pos"][0].item()),
            "n_visits":   V,
            "days_arr":   days_arr,
        },
        "logits":               _t(explain["logits"][0]),
        "cls_embedding":        _t(explain["cls_embedding"][0]),
        "fused_visit":          _t(explain["fused_visit"][0]),
        "h_diag":               _t(explain["h_diag"][0]),
        "h_med":                _t(explain["h_med"][0]),
        "h_lab":                _t(explain["h_lab"][0]),
        "h_vit":                _t(explain["h_vit"][0]),
        "diag_embedding_norm":  _t(explain["diag_embedding_norm"][0]),
        "med_embedding_norm":   _t(explain["med_embedding_norm"][0]),
        "lab_embedding_norm":   _t(explain["lab_embedding_norm"][0]),
        "vit_embedding_norm":   _t(explain["vit_embedding_norm"][0]),
        "temporal_pe":          _t(explain["temporal_pe"][0]),
        "days_enc":             _t(explain["days_enc"][0]),
        "visit_attn":           _t(explain["visit_attn"][0]),
        "domain_attn": {
            "diag": _t(explain["domain_attn"]["diag"][0]),
            "med":  _t(explain["domain_attn"]["med"][0]),
            "lab":  _t(explain["domain_attn"]["lab"][0]),
            "vit":  _t(explain["domain_attn"]["vit"][0]),
        },
    }

    out_path = out_dir / f"{pid}.explain.json"
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n  → wrote {out_path}")


# ── domain ablation ───────────────────────────────────────────────────────────

def run_ablation(model, items, vocab, token_to_id, device, batch_size=32):
    """Run 5 passes (baseline + 4 domain-zeroed) over all items.

    Returns dict: pass_name → {'auc': float, 'delta': float | None}
    """
    from sklearn.metrics import roc_auc_score

    PASSES = [
        ("baseline", None),
        ("no_diag",  ("diag_ids", "diag_mask")),
        ("no_med",   ("med_ids",  "med_mask")),
        ("no_lab",   ("lab_ids",  "lab_mask")),
        ("no_vital", ("vit_ids",  "vit_mask")),
    ]

    results = {}
    n = len(items)
    for pass_name, zero_keys in PASSES:
        all_probs, all_labels = [], []

        for start in range(0, n, batch_size):
            chunk = items[start: start + batch_size]
            batch = build_multi_batch(chunk, vocab, token_to_id)

            if zero_keys is not None:
                ids_key, mask_key = zero_keys
                batch = dict(batch)
                batch[ids_key]  = torch.zeros_like(batch[ids_key])
                batch[mask_key] = torch.zeros_like(batch[mask_key])

            labels = batch.pop("labels")
            batch_dev = {k: v.to(device) for k, v in batch.items()}

            with torch.no_grad():
                logits = model(batch_dev)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
            all_probs.extend(probs)
            all_labels.extend(labels.tolist())

        auc = roc_auc_score(all_labels, all_probs)
        results[pass_name] = auc
        print(f"  {pass_name:12s}  AUC={auc:.4f}")

    baseline_auc = results["baseline"]
    out = {}
    for pass_name, auc in results.items():
        delta = None if pass_name == "baseline" else auc - baseline_auc
        out[pass_name] = {"auc": auc, "delta": delta}
    return out


def print_ablation(ablation_results, n_patients):
    """Print domain ablation AUC table with ASCII bars."""
    max_bar  = 20
    max_drop = max(
        abs(v["delta"]) for v in ablation_results.values() if v["delta"] is not None
    ) or 1.0

    print(f"\nDomain ablation (full test set, N={n_patients})")
    rows = []
    for pass_name, vals in ablation_results.items():
        auc   = vals["auc"]
        delta = vals["delta"]
        if delta is None:
            bar   = "---"
            d_str = "---"
        else:
            bar   = "█" * int(round(abs(delta) / max_drop * max_bar))
            d_str = f"{delta:+.4f}"
        rows.append([pass_name, f"{auc:.4f}", d_str, bar])

    print(tabulate(rows, headers=["Domain", "AUC", "ΔAUC", ""], tablefmt="simple"))


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Inspect FourTowersTimeAwareAllCodes on test patients"
    )
    p.add_argument("--n",                   type=int,   default=5,
                   help="number of test patients to inspect (default 5)")
    p.add_argument("--top-k",               type=int,   default=5,
                   help="top-K codes per domain per visit (default 5)")
    p.add_argument("--seed",                type=int,   default=42)
    p.add_argument("--device",              default="cpu")
    p.add_argument("--vocab",               default=str(VOCAB_PATH))
    p.add_argument("--test",                default=str(TEST_PATH))
    p.add_argument("--ckpt",                default=str(CKPT_PATH))
    p.add_argument("--out-dir",             default="explain_output",
                   help="directory for per-patient JSON dumps (default explain_output/)")
    p.add_argument("--run-ablation",        action="store_true",
                   help="run domain ablation AUC table before patient inspection")
    p.add_argument("--ablation-batch-size", type=int,   default=32,
                   help="batch size for ablation pass (default 32)")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)

    # vocab
    with open(args.vocab) as f:
        vocab = json.load(f)                        # list[str]
    token_to_id = {tok: i for i, tok in enumerate(vocab)}
    print(f"Vocab size : {len(vocab)}")

    # test data
    with open(args.test, "rb") as f:
        items = pickle.load(f)
    if isinstance(items, dict) and "items" in items:
        items = items["items"]
    print(f"Test set   : {len(items)} patients")

    # sample
    patients  = sample_patients(items, args.n, args.seed)
    pos_count = sum(int(p["label"]) for p in patients)
    print(
        f"Sampled    : {len(patients)} patients "
        f"({pos_count} pos / {len(patients) - pos_count} neg)  seed={args.seed}\n"
    )

    # model
    model = FourTowersTimeAwareAllCodes(
        vocab_size=len(vocab),
        d_model=128, n_heads=4,
        n_visit_layers=2, n_patient_layers=2,
        max_visits=64, dropout=0.0, n_classes=2,
    ).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # ── optional domain ablation ──────────────────────────────────────────────
    if args.run_ablation:
        print("Running domain ablation over full test set …")
        ablation = run_ablation(
            model, items, vocab, token_to_id,
            device=device, batch_size=args.ablation_batch_size,
        )
        print_ablation(ablation, n_patients=len(items))

    # ── per-patient inspection ────────────────────────────────────────────────
    with torch.no_grad():
        for item in patients:
            batch, V, days_arr = build_single_batch(
                item, vocab, token_to_id,
                max_visits=64, max_diag=64, max_med=64,
                max_lab=128, max_vit=128,
            )
            batch_dev = {k: v.to(device) for k, v in batch.items()}
            explain   = model.forward_explain(batch_dev)

            print_patient(item, explain, batch, V, days_arr, vocab, args.top_k)
            write_explain_json(item, explain, days_arr, V,
                               out_dir=Path(args.out_dir))

    print("\nDone.")


if __name__ == "__main__":
    main()
