"""Shared training / evaluation loop used by every baseline."""
from __future__ import annotations

import math
import random
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class TrainConfig:
    epochs: int = 3
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 1e-2
    warmup_frac: float = 0.1
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    grad_clip: float = 1.0
    seed: int = 42
    device: str = "cpu"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _move(batch: dict, device: str) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def _batch_iter(items, model, batch_size, shuffle, device):
    idx = list(range(len(items)))
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        chunk = [items[k] for k in idx[i: i + batch_size]]
        batch = model.build_batch(chunk)
        moved = _move(batch, device)
        moved["_target_meds"] = [str(it.get("target_med", "-")) for it in chunk]
        yield moved


def _class_weights(items, device):
    labels = np.asarray([int(it["label"]) for it in items])
    n = len(labels)
    n1 = int(labels.sum())
    n0 = int(n - n1)
    if n0 == 0 or n1 == 0:
        return torch.tensor([1.0, 1.0], device=device)
    w0 = n / (2 * n0)
    w1 = n / (2 * n1)
    return torch.tensor([w0, w1], dtype=torch.float32, device=device)


def train_and_eval(
    model: nn.Module,
    train_items: list[dict],
    test_items: list[dict],
    cfg: TrainConfig,
    val_items: list[dict] | None = None,
) -> dict:
    _set_seed(cfg.seed)
    device = cfg.device
    model.to(device)

    _best_val_auc = -1.0
    _best_state = None
    _patience = 5
    _no_improve = 0

    # a model may ask for its own parameter groups (e.g. a higher lr for one parameter); groups keep their
    # own lr / weight_decay and follow the same warmup + decay schedule (LambdaLR scales each group)
    params = (model.param_groups(cfg.lr, cfg.weight_decay) if hasattr(model, "param_groups")
              else model.parameters())
    optim = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps_per_epoch = max(1, math.ceil(len(train_items) / cfg.batch_size))
    if cfg.max_train_batches is not None:
        total_steps_per_epoch = min(total_steps_per_epoch, cfg.max_train_batches)
    total_steps = max(1, total_steps_per_epoch * cfg.epochs)
    warmup = max(1, int(total_steps * cfg.warmup_frac))

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        return max(0.05, 1.0 - (step - warmup) / max(1, total_steps - warmup))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    loss_fn = nn.CrossEntropyLoss(weight=_class_weights(train_items, device))

    es_items = val_items if val_items is not None else test_items
    es_label = "val" if val_items is not None else "test"

    history = []
    t0 = time.time()
    step = 0
    stopped_epoch = cfg.epochs

    for ep in range(cfg.epochs):
        model.train()
        running = 0.0
        n_batches = 0
        for b_idx, batch in enumerate(_batch_iter(train_items, model, cfg.batch_size, shuffle=True, device=device)):
            if cfg.max_train_batches is not None and b_idx >= cfg.max_train_batches:
                break
            batch.pop("_target_meds", None)
            optim.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = loss_fn(logits, batch["labels"])
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()
            sched.step()
            step += 1
            running += float(loss.item())
            n_batches += 1
        avg_loss = running / max(1, n_batches)

        # evaluate on val only — test set NOT touched here
        es_out = evaluate(model, es_items, cfg)
        cur_auc = es_out["metrics"]["auc"]

        if cur_auc > _best_val_auc:
            _best_val_auc = cur_auc
            _best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            _no_improve = 0
        else:
            _no_improve += 1

        history.append({
            "epoch": ep + 1,
            "train_loss": avg_loss,
            f"{es_label}_auc": cur_auc,
            f"{es_label}_acc": es_out["metrics"]["accuracy"],
            f"{es_label}_f1":  es_out["metrics"]["f1"],
            "elapsed_sec": round(time.time() - t0, 1),
        })
        print(f"  epoch {ep+1}/{cfg.epochs}  train_loss={avg_loss:.4f}  "
              f"{es_label}_auc={cur_auc:.3f}  "
              f"{es_label}_f1={es_out['metrics']['f1']:.3f}  "
              f"elapsed={history[-1]['elapsed_sec']}s", flush=True)

        if _no_improve >= _patience:
            stopped_epoch = ep + 1
            print(f"  [early stop at epoch {stopped_epoch}, best {es_label}_auc={_best_val_auc:.4f}]", flush=True)
            break

    best_ep = max(range(len(history)), key=lambda i: history[i][f"{es_label}_auc"])
    print(f"  [best epoch: {best_ep+1}, {es_label}_auc={_best_val_auc:.4f}]", flush=True)

    # reload best checkpoint — test set touched exactly once here
    model.load_state_dict(_best_state)
    final = evaluate(model, test_items, cfg, return_predictions=True)
    print(f"  [final test_auc={final['metrics']['auc']:.4f}  test_f1={final['metrics']['f1']:.4f}]", flush=True)

    return {
        "history":       history,
        "metrics":       final["metrics"],
        "val_best_auc":  _best_val_auc,
        "best_epoch":    best_ep + 1,
        "stopped_epoch": stopped_epoch,
        "predictions":   final["predictions"],
        "config":        asdict(cfg),
    }


@torch.no_grad()
def evaluate(model, items, cfg, return_predictions=False):
    from .eval.metrics import compute_metrics

    device = cfg.device
    model.eval()
    all_logits, all_labels, all_pids, all_tgts = [], [], [], []
    running, n = 0.0, 0
    loss_fn = nn.CrossEntropyLoss()

    for b_idx, batch in enumerate(_batch_iter(items, model, cfg.batch_size, shuffle=False, device=device)):
        if cfg.max_eval_batches is not None and b_idx >= cfg.max_eval_batches:
            break
        tgts = batch.pop("_target_meds", None)
        logits = model(batch)
        running += float(loss_fn(logits, batch["labels"]).item()) * batch["labels"].size(0)
        n += batch["labels"].size(0)
        all_logits.append(logits.cpu().numpy())
        all_labels.append(batch["labels"].cpu().numpy())
        all_pids.append(batch["patient_ids"].cpu().numpy())
        if tgts is not None:
            all_tgts.extend(tgts)

    if not all_logits:
        return {"loss": float("nan"), "metrics": {"auc": float("nan")}}

    logits = np.concatenate(all_logits, 0)
    probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()[:, 1]
    labels = np.concatenate(all_labels, 0)
    pids = np.concatenate(all_pids, 0)
    target_meds = np.asarray(all_tgts) if all_tgts else None
    metrics = compute_metrics(labels, probs, target_meds=target_meds)
    out = {"loss": running / max(1, n), "metrics": metrics}
    if return_predictions:
        out["predictions"] = {
            "patient_ids": pids, "y_true": labels, "logits": logits, "y_prob": probs,
            "y_pred": (probs >= 0.5).astype(int),
            "target_meds": target_meds,
        }
    return out
