from __future__ import annotations

from typing import Dict, Tuple
import os
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    total_n = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        logits = model(batch)
        loss = loss_fn(logits, batch["labels"])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = batch["labels"].size(0)
        total_loss += loss.item() * bs
        total_n += bs

    return total_loss / max(total_n, 1)


@torch.no_grad()
def evaluate_model(model, loader, loss_fn, device) -> Tuple[float, float, float, float, float]:
    model.eval()

    total_loss = 0.0
    total_n = 0

    all_probs = []
    all_preds = []
    all_labels = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        logits = model(batch)
        loss = loss_fn(logits, batch["labels"])

        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)

        bs = batch["labels"].size(0)
        total_loss += loss.item() * bs
        total_n += bs

        all_probs.append(probs.cpu())
        all_preds.append(preds.cpu())
        all_labels.append(batch["labels"].cpu())

    avg_loss = total_loss / max(total_n, 1)

    all_probs = torch.cat(all_probs).numpy()
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    acc = (all_preds == all_labels).mean()

    try:
        auroc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auroc = 0.0

    try:
        auprc = average_precision_score(all_labels, all_probs)
    except Exception:
        auprc = 0.0

    try:
        f1 = f1_score(all_labels, all_preds)
    except Exception:
        f1 = 0.0

    return avg_loss, acc, auroc, auprc, f1


@torch.no_grad()
def collect_logits(model, loader, device):
    model.eval()

    all_logits = []
    all_labels = []
    all_patient_ids = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        logits = model(batch)

        all_logits.append(logits.cpu())
        all_labels.append(batch["labels"].cpu())
        all_patient_ids.append(batch["patient_ids"].cpu())

    return torch.cat(all_logits), torch.cat(all_labels), torch.cat(all_patient_ids)


def save_checkpoint(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)