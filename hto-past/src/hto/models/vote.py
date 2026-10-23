import torch


def majority_vote_from_logits(logits_list):
    """
    logits_list: list of tensors, each [N, C]
    returns: final predicted class [N]
    """
    preds = [logits.argmax(dim=1) for logits in logits_list]
    votes = torch.stack(preds, dim=1)   # [N, num_models]

    final_preds = []
    for i in range(votes.size(0)):
        vals, counts = votes[i].unique(return_counts=True)
        final_preds.append(vals[counts.argmax()])

    return torch.stack(final_preds)


def soft_vote_from_logits(logits_list):
    """
    Average logits, then argmax.
    Usually stronger than hard majority vote.
    """
    avg_logits = torch.stack(logits_list, dim=0).mean(dim=0)   # [N, C]
    return avg_logits.argmax(dim=1)