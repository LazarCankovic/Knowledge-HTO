import json
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from hto.data.dataset import PatientDataset
from hto.data.batch_builder_ordered import ordered_collate_fn
from hto.models.ordered_fusion import OrderedFusionModel
from hto.train_utils import train_one_epoch, evaluate_model, save_checkpoint

TRAIN_PATH = "/data/nsf-jg/hto/hto2/processed_data/full_80_20/multidomain_visit.train.pkl"
TEST_PATH = "/data/nsf-jg/hto/hto2/processed_data/full_80_20/multidomain_visit.test.pkl"
VOCAB_PATH = "/data/nsf-jg/hto/hto2/processed_data/full_80_20/multidomain_visit.vocab.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-4


def load_vocab_size():
    with open(VOCAB_PATH, "r") as f:
        vocab = json.load(f)
    return len(vocab)


def infer_num_domains():
    with open(TRAIN_PATH, "rb") as f:
        data = pickle.load(f)

    max_domain = -1
    for item in data[:500]:   # enough to infer quickly
        if len(item["flat_domain"]) > 0:
            max_domain = max(max_domain, max(item["flat_domain"]))

    return max_domain + 1


def main():
    vocab_size = load_vocab_size()
    n_domains = infer_num_domains()

    train_ds = PatientDataset(TRAIN_PATH)
    test_ds = PatientDataset(TEST_PATH)

    collate = ordered_collate_fn(max_events=512)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

    model = OrderedFusionModel(
        vocab_size=vocab_size,
        n_domains=n_domains,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    best_auroc = -1.0
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, DEVICE)
        test_loss, test_acc, test_auroc, test_auprc, test_f1 = evaluate_model(model, test_loader, loss_fn, DEVICE)

        print(
            f"[A3] Epoch {epoch+1}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"test_loss={test_loss:.4f} acc={test_acc:.4f} "
            f"auroc={test_auroc:.4f} auprc={test_auprc:.4f} f1={test_f1:.4f}"
        )

        if test_auroc > best_auroc:
            best_auroc = test_auroc
            save_checkpoint(model, "/data/nsf-jg/hto/hto2/checkpoints/approach3_best.pt")


if __name__ == "__main__":
    main()
