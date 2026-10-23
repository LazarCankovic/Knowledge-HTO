import json
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from hto.data.dataset import PatientDataset
from hto.data.batch_builder_visit_based import visit_collate_fn
from hto.models.single_domain import SingleDomainSequenceModel
from hto.train_utils import train_one_epoch, evaluate_model, collect_logits, save_checkpoint

TRAIN_PATH = "/data/nsf-jg/hto/hto2/processed_data/full_80_20/multidomain_visit.train.pkl"
TEST_PATH = "/data/nsf-jg/hto/hto2/processed_data/full_80_20/multidomain_visit.test.pkl"
VOCAB_PATH = "/data/nsf-jg/hto/hto2/processed_data/full_80_20/multidomain_visit.vocab.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LAB_DIM = 6
VIT_DIM = 33

BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-4
SAVE_DIR = "/data/nsf-jg/hto/hto2/checkpoints/approach2"


def load_vocab_size():
    with open(VOCAB_PATH, "r") as f:
        vocab = json.load(f)
    return len(vocab)


def train_domain(domain_name, model, train_loader, test_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    best_auroc = -1.0
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, DEVICE)
        test_loss, test_acc, test_auroc, test_auprc, test_f1 = evaluate_model(model, test_loader, loss_fn, DEVICE)

        print(
            f"[A2-{domain_name}] Epoch {epoch+1}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"test_loss={test_loss:.4f} acc={test_acc:.4f} "
            f"auroc={test_auroc:.4f} auprc={test_auprc:.4f} f1={test_f1:.4f}"
        )

        if test_auroc > best_auroc:
            best_auroc = test_auroc
            save_checkpoint(model, os.path.join(SAVE_DIR, f"{domain_name}_best.pt"))

    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, f"{domain_name}_best.pt"), map_location=DEVICE))
    logits, labels, patient_ids = collect_logits(model, test_loader, DEVICE)
    torch.save(logits, os.path.join(SAVE_DIR, f"{domain_name}_logits.pt"))
    torch.save(labels, os.path.join(SAVE_DIR, f"{domain_name}_labels.pt"))
    torch.save(patient_ids, os.path.join(SAVE_DIR, f"{domain_name}_patient_ids.pt"))


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    vocab_size = load_vocab_size()

    train_ds = PatientDataset(TRAIN_PATH)
    test_ds = PatientDataset(TEST_PATH)

    collate = visit_collate_fn(
        max_visits=64,
        max_diag_codes=64,
        max_med_codes=64,
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

    diag_model = SingleDomainSequenceModel("diag", vocab_size=vocab_size).to(DEVICE)
    med_model = SingleDomainSequenceModel("med", vocab_size=vocab_size).to(DEVICE)
    lab_model = SingleDomainSequenceModel("lab", n_feat=LAB_DIM).to(DEVICE)
    vit_model = SingleDomainSequenceModel("vit", n_feat=VIT_DIM).to(DEVICE)

    train_domain("diag", diag_model, train_loader, test_loader)
    train_domain("med", med_model, train_loader, test_loader)
    train_domain("lab", lab_model, train_loader, test_loader)
    train_domain("vit", vit_model, train_loader, test_loader)


if __name__ == "__main__":
    main()
