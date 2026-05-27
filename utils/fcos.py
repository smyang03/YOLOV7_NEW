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


def default_level_ranges(strides):
    base = [
        (0, 64),
        (64, 128),
        (128, 256),
        (256, 512),
        (512, 1e8),
    ]
    if len(strides) <= 3:
        return [(0, 64), (64, 128), (128, 1e8)]
    return base[:len(strides)]


def compute_centerness_targets(ltrb_targets):
    left_right = ltrb_targets[:, [0, 2]]
    top_bottom = ltrb_targets[:, [1, 3]]
    centerness = (
        left_right.min(dim=-1).values / left_right.max(dim=-1).values.clamp(min=1e-6) *
        top_bottom.min(dim=-1).values / top_bottom.max(dim=-1).values.clamp(min=1e-6)
    ).clamp(min=0.0)
    return torch.sqrt(centerness)


def _targets_to_xyxy(targets, img_size):
    h, w = img_size
    boxes = targets[:, 2:6].clone()
    boxes[:, [0, 2]] *= w
    boxes[:, [1, 3]] *= h
    xyxy = torch.empty_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] * 0.5
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] * 0.5
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] * 0.5
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] * 0.5
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clamp(0, w)
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clamp(0, h)
    return xyxy


def fcos_targets(targets, feature_shapes, strides, img_size, num_classes,
                 batch_size, center_radius=1.5, level_ranges=None):
    device = targets.device
    level_ranges = level_ranges or default_level_ranges(strides)
    results = []
    for height, width in feature_shapes:
        points = height * width
        results.append({
            'labels': torch.full((batch_size, points), num_classes, dtype=torch.long, device=device),
            'ltrb': torch.zeros((batch_size, points, 4), device=device),
            'centerness': torch.zeros((batch_size, points), device=device),
            'pos_mask': torch.zeros((batch_size, points), dtype=torch.bool, device=device),
        })
    if targets.numel() == 0:
        return results

    img_targets = targets.detach()
    for b in range(batch_size):
        per_image = img_targets[img_targets[:, 0].long() == b]
        if per_image.numel() == 0:
            continue
        classes = per_image[:, 1].long().clamp(0, num_classes - 1)
        boxes = _targets_to_xyxy(per_image, img_size)
        wh = (boxes[:, 2:] - boxes[:, :2]).clamp(min=1e-6)
        areas = wh[:, 0] * wh[:, 1]
        centers = (boxes[:, :2] + boxes[:, 2:]) * 0.5

        for level, ((height, width), stride) in enumerate(zip(feature_shapes, strides)):
            locations = make_locations(height, width, stride, device=device)
            xs, ys = locations[:, 0:1], locations[:, 1:2]
            l = xs - boxes[:, 0]
            t = ys - boxes[:, 1]
            r = boxes[:, 2] - xs
            btm = boxes[:, 3] - ys
            ltrb = torch.stack((l, t, r, btm), dim=-1)
            inside_box = ltrb.min(dim=-1).values > 0

            radius = center_radius * stride
            center_boxes = torch.stack((
                torch.max(boxes[:, 0], centers[:, 0] - radius),
                torch.max(boxes[:, 1], centers[:, 1] - radius),
                torch.min(boxes[:, 2], centers[:, 0] + radius),
                torch.min(boxes[:, 3], centers[:, 1] + radius),
            ), dim=-1)
            cl = xs - center_boxes[:, 0]
            ct = ys - center_boxes[:, 1]
            cr = center_boxes[:, 2] - xs
            cb = center_boxes[:, 3] - ys
            inside_center = torch.stack((cl, ct, cr, cb), dim=-1).min(dim=-1).values > 0

            max_ltrb = ltrb.max(dim=-1).values
            low, high = level_ranges[level]
            in_range = (max_ltrb >= low) & (max_ltrb <= high)
            candidates = inside_box & inside_center & in_range
            candidate_areas = areas.unsqueeze(0).expand_as(candidates).clone()
            candidate_areas[~candidates] = float('inf')
            min_area, matched = candidate_areas.min(dim=1)
            pos = torch.isfinite(min_area)
            if not pos.any():
                continue
            point_index = pos.nonzero(as_tuple=False).view(-1)
            gt_index = matched[pos]
            matched_ltrb = ltrb[point_index, gt_index]
            results[level]['labels'][b, point_index] = classes[gt_index]
            results[level]['ltrb'][b, point_index] = matched_ltrb
            results[level]['centerness'][b, point_index] = compute_centerness_targets(matched_ltrb)
            results[level]['pos_mask'][b, point_index] = True
    return results


def decode_fcos_outputs(raw_outputs, strides, img_size=None, conf_thres=0.25, topk=1000,
                        score_mode='sqrt_cls_centerness'):
    decoded = []
    for raw, stride in zip(raw_outputs, strides):
        per_level = decode_fcos_raw(raw, stride=stride, img_size=img_size, conf_thres=conf_thres,
                                    topk=topk, score_mode=score_mode)
        if not decoded:
            decoded = per_level
        else:
            decoded = [torch.cat((a, b), 0) for a, b in zip(decoded, per_level)]
    return decoded
