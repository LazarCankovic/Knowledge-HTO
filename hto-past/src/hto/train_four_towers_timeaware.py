import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from hto.data.dataset import PatientDataset
from hto.data.batch_builder_visit_based import visit_collate_fn
from hto.models.four_towers_timeaware import FourTowersTimeAwareModel
from hto.train_utils import train_one_epoch, evaluate_model, save_checkpoint

TRAIN_PATH = "/fs/atipa/data/nsf-jg/hto/hto2/processed_data/full_80_20/multidomain_visit.train.pkl"
TEST_PATH = "/fs/atipa/data/nsf-jg/hto/hto2/processed_data/full_80_20/multidomain_visit.test.pkl"
VOCAB_PATH = "/fs/atipa/data/nsf-jg/hto/hto2/processed_data/full_80_20/multidomain_visit.vocab.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LAB_DIM = 6
VIT_DIM = 33

BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-4


def load_vocab_size():
    with open(VOCAB_PATH, "r") as f:
        vocab = json.load(f)
    return len(vocab)


def main():
    vocab_size = load_vocab_size()

    train_ds = PatientDataset(TRAIN_PATH)
    test_ds = PatientDataset(TEST_PATH)

    collate = visit_collate_fn(
        max_visits=64,
        max_diag_codes=64,
        max_med_codes=64,
        n_lab=LAB_DIM,
        n_vit=VIT_DIM,
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

    model = FourTowersTimeAwareModel(
        diag_vocab_size=vocab_size,
        med_vocab_size=vocab_size,
        n_lab=LAB_DIM,
        n_vit=VIT_DIM,
        d_model=128,
        n_heads=4,
        n_visit_layers=2,
        n_patient_layers=2,
        max_visits=64,
        dropout=0.1,
        n_classes=2,
        time_mode="linear",
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    best_auroc = -1.0
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, DEVICE)
        test_loss, test_acc, test_auroc, test_auprc, test_f1 = evaluate_model(model, test_loader, loss_fn, DEVICE)

        print(
            f"[A1+Time] Epoch {epoch+1}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"test_loss={test_loss:.4f} acc={test_acc:.4f} "
            f"auroc={test_auroc:.4f} auprc={test_auprc:.4f} f1={test_f1:.4f}"
        )

        if test_auroc > best_auroc:
            best_auroc = test_auroc
            save_checkpoint(model, "/fs/atipa/data/nsf-jg/hto/hto2/checkpoints/four_towers_timeaware_best.pt")


if __name__ == "__main__":
    main()
