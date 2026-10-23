import torch
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score

from hto.models.vote import majority_vote_from_logits, soft_vote_from_logits

SAVE_DIR = "/data/nsf-jg/hto/hto2/checkpoints/approach2"


def main():
    diag_logits = torch.load(f"{SAVE_DIR}/diag_logits.pt")
    med_logits = torch.load(f"{SAVE_DIR}/med_logits.pt")
    lab_logits = torch.load(f"{SAVE_DIR}/lab_logits.pt")
    vit_logits = torch.load(f"{SAVE_DIR}/vit_logits.pt")

    labels = torch.load(f"{SAVE_DIR}/diag_labels.pt").numpy()

    majority_preds = majority_vote_from_logits([diag_logits, med_logits, lab_logits, vit_logits]).numpy()
    soft_preds = soft_vote_from_logits([diag_logits, med_logits, lab_logits, vit_logits]).numpy()

    avg_probs = torch.softmax(
        torch.stack([diag_logits, med_logits, lab_logits, vit_logits], dim=0).mean(dim=0),
        dim=1
    )[:, 1].numpy()

    print("Approach 2 Majority Vote")
    print("acc:", accuracy_score(labels, majority_preds))
    print("f1:", f1_score(labels, majority_preds))

    print("\nApproach 2 Soft Vote")
    print("acc:", accuracy_score(labels, soft_preds))
    print("f1:", f1_score(labels, soft_preds))
    print("auroc:", roc_auc_score(labels, avg_probs))
    print("auprc:", average_precision_score(labels, avg_probs))


if __name__ == "__main__":
    main()