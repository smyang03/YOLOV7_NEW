import json
import random
from pathlib import Path

import cv2
import numpy as np


def _clip_labels(labels, width, height):
    if labels is None or len(labels) == 0:
        return labels
    labels[:, [1, 3]] = labels[:, [1, 3]].clip(0, width)
    labels[:, [2, 4]] = labels[:, [2, 4]].clip(0, height)
    wh = labels[:, 3:5] - labels[:, 1:3]
    keep = (wh[:, 0] > 2) & (wh[:, 1] > 2)
    return labels[keep]


def _box_iou_xyxy(box, boxes):
    if boxes is None or len(boxes) == 0:
        return np.zeros((0,), dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area1 = max((box[2] - box[0]) * (box[3] - box[1]), 1.0)
    area2 = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)
    return inter / (area1 + area2 - inter + 1e-6)


def spider_web(img, max_lines=8):
    h, w = img.shape[:2]
    overlay = img.copy()
    center = (random.randint(0, max(w - 1, 0)), random.randint(0, max(h - 1, 0)))
    color = random.choice([(210, 210, 210), (230, 230, 230), (180, 180, 180)])
    thickness = 1
    for _ in range(random.randint(3, max_lines)):
        angle = random.random() * 2 * np.pi
        length = random.randint(max(8, min(h, w) // 4), max(12, max(h, w)))
        end = (int(center[0] + np.cos(angle) * length), int(center[1] + np.sin(angle) * length))
        cv2.line(overlay, center, end, color, thickness, cv2.LINE_AA)
    for radius in range(max(8, min(h, w) // 10), max(9, min(h, w) // 2), max(8, min(h, w) // 8)):
        cv2.ellipse(overlay, center, (radius, max(3, radius // 3)), 0, 0, 360, color, thickness, cv2.LINE_AA)
    return cv2.addWeighted(overlay, 0.25, img, 0.75, 0)


def to_gray3(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe_op = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l = clahe_op.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def motion_blur(img, max_kernel=9):
    k = random.choice([3, 5, 7, max_kernel])
    kernel = np.zeros((k, k), dtype=np.float32)
    if random.random() < 0.5:
        kernel[k // 2, :] = 1.0
    else:
        kernel[:, k // 2] = 1.0
    kernel /= kernel.sum()
    return cv2.filter2D(img, -1, kernel)


def compression_noise(img, quality_min=35, quality_max=80):
    quality = random.randint(quality_min, quality_max)
    ok, encoded = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return img
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return decoded if decoded is not None else img


def overexposure(img):
    h, w = img.shape[:2]
    overlay = img.copy()
    cx = random.randint(0, max(w - 1, 0))
    cy = random.randint(0, max(h - 1, 0))
    radius = random.randint(max(8, min(h, w) // 10), max(12, min(h, w) // 3))
    cv2.circle(overlay, (cx, cy), radius, (255, 255, 255), -1, cv2.LINE_AA)
    return cv2.addWeighted(overlay, 0.18, img, 0.82, 8)


def patch_paste(img, labels, probability=0.0, max_attempts=20, max_iou=0.30):
    info = {'patch_paste_applied': 0, 'patch_paste_failures': 0}
    if probability <= 0.0 or random.random() >= probability or labels is None or len(labels) == 0:
        return img, labels, info

    h, w = img.shape[:2]
    labels = labels.copy()
    source_order = np.argsort((labels[:, 3] - labels[:, 1]) * (labels[:, 4] - labels[:, 2]))
    for src_idx in source_order:
        cls, x1, y1, x2, y2 = labels[src_idx]
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        bw, bh = x2 - x1, y2 - y1
        if bw < 4 or bh < 4:
            continue
        patch = img[y1:y2, x1:x2].copy()
        if patch.size == 0:
            continue
        for _ in range(max_attempts):
            scale = random.uniform(0.7, 1.2)
            nw, nh = max(3, int(bw * scale)), max(3, int(bh * scale))
            if nw >= w or nh >= h:
                continue
            px1 = random.randint(0, w - nw)
            py1 = random.randint(0, h - nh)
            new_box = np.array([px1, py1, px1 + nw, py1 + nh], dtype=np.float32)
            if _box_iou_xyxy(new_box, labels[:, 1:5]).max(initial=0.0) > max_iou:
                continue
            resized = cv2.resize(patch, (nw, nh), interpolation=cv2.INTER_LINEAR)
            img[py1:py1 + nh, px1:px1 + nw] = resized
            new_label = np.array([[cls, px1, py1, px1 + nw, py1 + nh]], dtype=labels.dtype)
            labels = np.concatenate((labels, new_label), 0)
            info['patch_paste_applied'] += 1
            return img, _clip_labels(labels, w, h), info
    info['patch_paste_failures'] += 1
    return img, labels, info


def hard_negative_paste(img, labels, probability=0.0, max_attempts=20):
    info = {'hard_negative_pastes': 0, 'hard_negative_failures': 0}
    if probability <= 0.0 or random.random() >= probability:
        return img, labels, info

    h, w = img.shape[:2]
    boxes = labels[:, 1:5] if labels is not None and len(labels) else np.zeros((0, 4), dtype=np.float32)
    for _ in range(max_attempts):
        pw = random.randint(max(8, w // 16), max(9, w // 4))
        ph = random.randint(max(8, h // 16), max(9, h // 4))
        if pw >= w or ph >= h:
            continue
        sx1 = random.randint(0, w - pw)
        sy1 = random.randint(0, h - ph)
        src_box = np.array([sx1, sy1, sx1 + pw, sy1 + ph], dtype=np.float32)
        if _box_iou_xyxy(src_box, boxes).max(initial=0.0) > 0.05:
            continue
        dx1 = random.randint(0, w - pw)
        dy1 = random.randint(0, h - ph)
        dst_box = np.array([dx1, dy1, dx1 + pw, dy1 + ph], dtype=np.float32)
        if _box_iou_xyxy(dst_box, boxes).max(initial=0.0) > 0.10:
            continue
        img[dy1:dy1 + ph, dx1:dx1 + pw] = img[sy1:sy1 + ph, sx1:sx1 + pw]
        info['hard_negative_pastes'] += 1
        return img, labels, info
    info['hard_negative_failures'] += 1
    return img, labels, info


def _read_manifest_patch(entry):
    image_path = entry.get('image') or entry.get('path')
    bbox = entry.get('bbox_xyxy') or entry.get('bbox')
    if not image_path or not bbox or len(bbox) != 4:
        return None
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    return image[y1:y2, x1:x2].copy()


def hard_negative_manifest_paste(img, labels, crops, probability=0.0, max_attempts=20):
    info = {'hard_negative_pastes': 0, 'hard_negative_failures': 0}
    if not crops or probability <= 0.0 or random.random() >= probability:
        return img, labels, info

    h, w = img.shape[:2]
    boxes = labels[:, 1:5] if labels is not None and len(labels) else np.zeros((0, 4), dtype=np.float32)
    for _ in range(max_attempts):
        patch = _read_manifest_patch(random.choice(crops))
        if patch is None:
            continue
        ph0, pw0 = patch.shape[:2]
        max_pw, max_ph = max(8, w // 4), max(8, h // 4)
        scale = min(max_pw / max(pw0, 1), max_ph / max(ph0, 1), 1.0) * random.uniform(0.7, 1.1)
        pw, ph = max(3, int(pw0 * scale)), max(3, int(ph0 * scale))
        if pw >= w or ph >= h:
            continue
        dx1 = random.randint(0, w - pw)
        dy1 = random.randint(0, h - ph)
        dst_box = np.array([dx1, dy1, dx1 + pw, dy1 + ph], dtype=np.float32)
        if _box_iou_xyxy(dst_box, boxes).max(initial=0.0) > 0.10:
            continue
        resized = cv2.resize(patch, (pw, ph), interpolation=cv2.INTER_LINEAR)
        img[dy1:dy1 + ph, dx1:dx1 + pw] = resized
        info['hard_negative_pastes'] += 1
        return img, labels, info
    info['hard_negative_failures'] += 1
    return img, labels, info


def load_hard_negative_manifest(path):
    if not path:
        return []
    manifest = Path(path)
    if not manifest.is_file():
        return []
    data = json.loads(manifest.read_text(encoding='utf-8'))
    crops = data.get('crops', data if isinstance(data, list) else [])
    normalized = []
    for crop in crops:
        if not isinstance(crop, dict):
            continue
        crop = dict(crop)
        image_path = crop.get('image') or crop.get('path')
        if image_path:
            p = Path(image_path)
            if not p.is_absolute() and not p.exists():
                p = manifest.parent / p
            crop['image'] = str(p)
        normalized.append(crop)
    return normalized


def apply_cctv_augmentations(img, labels, policy, hard_negative_crops=None):
    info = {
        'spider_web': 0,
        'gray': 0,
        'clahe': 0,
        'blur': 0,
        'compression': 0,
        'overexposure': 0,
        'patch_paste_applied': 0,
        'patch_paste_failures': 0,
        'hard_negative_pastes': 0,
        'hard_negative_failures': 0,
    }
    if policy is None or not policy.enabled:
        return img, labels, info

    if random.random() < policy.spider_web_p:
        img = spider_web(img)
        info['spider_web'] += 1
    if random.random() < policy.gray_p:
        img = to_gray3(img)
        info['gray'] += 1
    if random.random() < policy.clahe_p:
        img = clahe(img)
        info['clahe'] += 1
    if random.random() < policy.blur_p:
        img = motion_blur(img)
        info['blur'] += 1
    if random.random() < policy.compression_p:
        img = compression_noise(img)
        info['compression'] += 1
    if random.random() < policy.overexposure_p:
        img = overexposure(img)
        info['overexposure'] += 1

    img, labels, patch_info = patch_paste(img, labels, policy.patch_paste_p)
    info.update({k: info.get(k, 0) + v for k, v in patch_info.items()})
    if policy.hard_negative_p > 0.0 and random.random() < policy.hard_negative_p:
        if hard_negative_crops:
            img, labels, hard_info = hard_negative_manifest_paste(
                img, labels, hard_negative_crops, probability=1.0)
            if hard_info['hard_negative_pastes'] == 0:
                img, labels, fallback_info = hard_negative_paste(img, labels, probability=1.0)
                hard_info.update({k: hard_info.get(k, 0) + v for k, v in fallback_info.items()})
        else:
            img, labels, hard_info = hard_negative_paste(img, labels, probability=1.0)
        info.update({k: info.get(k, 0) + v for k, v in hard_info.items()})
    return img, labels, info
