import json, sys
from pathlib import Path
sys.path.insert(0, "/fs/atipa/data/nsf-jg/hto/hto2/src")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from hto.data.dataset import PatientDataset
from hto.data.batch_builder_all_codes import allcodes_collate_fn
from hto.models.four_towers_timeaware_all_codes import FourTowersTimeAwareAllCodes
from hto.train_utils import evaluate_model, save_checkpoint

TRAIN_PATH = "/fs/atipa/data/nsf-jg/hto/hto2/processed_data/full_80_20/multidomain_visit.train.pkl"
TEST_PATH  = "/fs/atipa/data/nsf-jg/hto/hto2/processed_data/full_80_20/multidomain_visit.test.pkl"
VOCAB_PATH = "/fs/atipa/data/nsf-jg/hto/hto2/processed_data/full_80_20/allcodes_timeaware.vocab.json"
CKPT_PATH  = "/fs/atipa/data/nsf-jg/hto/hto2/checkpoints/four_towers_timeaware_allcodes_best.pt"

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS     = 20
LR         = 3e-4

print(f"Device: {DEVICE}", flush=True)

with open(VOCAB_PATH) as f:
    vocab = json.load(f)
vocab_size = len(vocab)
print(f"Vocab size: {vocab_size}", flush=True)

collate = allcodes_collate_fn(max_visits=64, max_diag_codes=64, max_med_codes=64, max_lab_codes=128, max_vit_codes=128)
train_ds = PatientDataset(TRAIN_PATH)
test_ds  = PatientDataset(TEST_PATH)
print(f"Train: {len(train_ds)}  Test: {len(test_ds)}", flush=True)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate, num_workers=4, pin_memory=True)

model = FourTowersTimeAwareAllCodes(
    vocab_size=vocab_size, d_model=128, n_heads=4,
    n_visit_layers=2, n_patient_layers=2,
    max_visits=64, dropout=0.1, n_classes=2, time_fn="log",
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
loss_fn   = nn.CrossEntropyLoss()
best_auroc = -1.0

for epoch in range(EPOCHS):
    model.train()
    total_loss, total_n = 0.0, 0
    for i, batch in enumerate(train_loader):
        batch = {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in batch.items()}
        logits = model(batch)
        loss = loss_fn(logits, batch["labels"])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        bs = batch["labels"].size(0)
        total_loss += loss.item() * bs
        total_n += bs
        if (i + 1) % 100 == 0:
            print(f"  epoch {epoch+1} batch {i+1}/{len(train_loader)} loss={total_loss/total_n:.4f}", flush=True)

    train_loss = total_loss / max(total_n, 1)
    test_loss, test_acc, test_auroc, test_auprc, test_f1 = evaluate_model(model, test_loader, loss_fn, DEVICE)
    print(f"[Epoch {epoch+1}/{EPOCHS}] train={train_loss:.4f} auroc={test_auroc:.4f} auprc={test_auprc:.4f} acc={test_acc:.4f} f1={test_f1:.4f}", flush=True)

    if test_auroc > best_auroc:
        best_auroc = test_auroc
        save_checkpoint(model, CKPT_PATH)
        print(f"  -> saved checkpoint (auroc={best_auroc:.4f})", flush=True)

print(f"Done. Best AUROC: {best_auroc:.4f}", flush=True)
