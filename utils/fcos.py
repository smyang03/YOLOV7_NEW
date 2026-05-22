import torch
import torch.nn.functional as F


def make_locations(height, width, stride, device=None):
    y, x = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
    )
    return torch.stack(((x + 0.5) * stride, (y + 0.5) * stride), dim=-1).view(-1, 2)


def decode_fcos_raw(raw, stride=4, img_size=None, conf_thres=0.25, topk=1000,
                    score_mode='sqrt_cls_centerness'):
    if raw.ndim != 4:
        raise ValueError(f'FCOS raw output must be BCHW, got shape {list(raw.shape)}')
    bs, channels, height, width = raw.shape
    if channels < 6:
        raise ValueError('FCOS raw output requires 4 box channels, 1 centerness channel, and at least 1 class')

    nc = channels - 5
    device = raw.device
    locations = make_locations(height, width, stride, device=device)
    ltrb = F.relu(raw[:, :4]).permute(0, 2, 3, 1).reshape(bs, -1, 4) * stride
    centerness = raw[:, 4:5].sigmoid().permute(0, 2, 3, 1).reshape(bs, -1)
    cls_scores = raw[:, 5:].sigmoid().permute(0, 2, 3, 1).reshape(bs, -1, nc)
    class_scores, classes = cls_scores.max(dim=-1)

    if score_mode == 'sqrt_cls_centerness':
        scores = torch.sqrt((class_scores * centerness).clamp(min=0.0))
    elif score_mode == 'mul_cls_centerness':
        scores = class_scores * centerness
    else:
        raise ValueError(f'Unsupported score_mode: {score_mode}')

    centers = locations.unsqueeze(0).expand(bs, -1, 2)
    boxes = torch.stack((
        centers[..., 0] - ltrb[..., 0],
        centers[..., 1] - ltrb[..., 1],
        centers[..., 0] + ltrb[..., 2],
        centers[..., 1] + ltrb[..., 3],
    ), dim=-1)
    if img_size is not None:
        h, w = img_size
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clamp(0, w)
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clamp(0, h)

    decoded = []
    for i in range(bs):
        keep = scores[i] >= conf_thres
        image_boxes = boxes[i, keep]
        image_scores = scores[i, keep]
        image_classes = classes[i, keep].float()
        if topk and image_scores.numel() > topk:
            top_scores, idx = image_scores.topk(topk)
            image_boxes = image_boxes[idx]
            image_scores = top_scores
            image_classes = image_classes[idx]
        decoded.append(torch.cat((image_boxes, image_scores[:, None], image_classes[:, None]), dim=1)
                       if image_scores.numel() else raw.new_zeros((0, 6)))
    return decoded


def fcos_contract(raw, stride=4, img_size=None, conf_thres=0.25, topk=1000,
                  score_mode='sqrt_cls_centerness'):
    decoded = decode_fcos_raw(raw, stride=stride, img_size=img_size, conf_thres=conf_thres,
                              topk=topk, score_mode=score_mode)
    return {
        'raw_shape': list(raw.shape),
        'stride': int(stride),
        'num_classes': int(raw.shape[1] - 5),
        'score_mode': score_mode,
        'decoded_box_count': [int(x.shape[0]) for x in decoded],
        'decoded_shape': [list(x.shape) for x in decoded],
    }
