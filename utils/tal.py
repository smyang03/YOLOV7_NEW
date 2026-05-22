import torch


def assignment_cost(cost, pair_wise_iou, cls_scores=None, hyp=None, assign='simota', default_topk=10):
    hyp = hyp or {}
    if assign != 'tal':
        top_k, _ = torch.topk(pair_wise_iou, min(default_topk, pair_wise_iou.shape[1]), dim=1)
        dynamic_ks = torch.clamp(top_k.sum(1).int(), min=1)
        return cost, dynamic_ks

    topk = min(int(hyp.get('tal_topk', default_topk)), pair_wise_iou.shape[1])
    alpha = float(hyp.get('tal_alpha', 1.0))
    beta = float(hyp.get('tal_beta', 6.0))
    if cls_scores is None:
        cls_scores = torch.ones_like(pair_wise_iou)
    metric = cls_scores.detach().clamp(0.0, 1.0).pow(alpha) * pair_wise_iou.detach().clamp(0.0, 1.0).pow(beta)
    dynamic_ks = torch.full((pair_wise_iou.shape[0],), max(topk, 1),
                            dtype=torch.int32, device=pair_wise_iou.device)
    return -metric, dynamic_ks
