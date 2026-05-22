import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.dataset_manifest import read_text_fallback, resolve_path
from utils.datasets import create_dataloader


def draw_labels(img_rgb, labels, names):
    img = img_rgb[:, :, ::-1].copy()
    h, w = img.shape[:2]
    errors = {'bbox_range_errors': 0, 'class_id_errors': 0}
    for row in labels:
        cls, xc, yc, bw, bh = row.tolist()
        if cls < 0 or int(cls) >= len(names):
            errors['class_id_errors'] += 1
            continue
        if min(xc, yc, bw, bh) < 0 or max(xc, yc, bw, bh) > 1 or bw <= 0 or bh <= 0:
            errors['bbox_range_errors'] += 1
            continue
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, str(names[int(cls)]), (x1, max(0, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    return img, errors


def main(opt):
    data_path = Path(opt.data)
    data = yaml.safe_load(read_text_fallback(data_path))
    hyp = yaml.safe_load(read_text_fallback(opt.hyp)) if opt.hyp else {}
    names = data.get('names') or [str(i) for i in range(int(data['nc']))]
    train_source = data['train']
    train_path = [str(resolve_path(x, data_path.parent, ROOT)) for x in train_source] \
        if isinstance(train_source, list) else str(resolve_path(train_source, data_path.parent, ROOT))
    output = Path(opt.output)
    sample_dir = output / 'aug_samples'
    sample_dir.mkdir(parents=True, exist_ok=True)

    loader_opt = SimpleNamespace(
        single_cls=False,
        cache_images=False,
        image_weights=False,
        quad=False,
        workers=0,
        world_size=1,
        aug_profile=opt.aug_profile,
        sampler_mode='off',
        hard_negative_manifest=opt.hard_negative_manifest,
        nc=int(data['nc']),
        seed=0,
    )
    dataloader, dataset = create_dataloader(
        train_path, opt.img, 1, 32, loader_opt, hyp=hyp, augment=True, cache=False,
        rect=False, rank=-1, world_size=1, workers=0, prefix='aug_check: ',
        aug_phase=opt.phase)

    bbox_range_errors = 0
    class_id_errors = 0
    saved = 0
    iterator = iter(dataloader)
    for i in range(opt.samples):
        try:
            imgs, labels, paths, _ = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            imgs, labels, paths, _ = next(iterator)
        img = imgs[0].numpy().transpose(1, 2, 0)
        labels_np = labels[labels[:, 0] == 0][:, 1:].numpy() if len(labels) else np.zeros((0, 5), dtype=np.float32)
        drawn, errors = draw_labels(img, labels_np, names)
        bbox_range_errors += errors['bbox_range_errors']
        class_id_errors += errors['class_id_errors']
        cv2.imwrite(str(sample_dir / f'sample_{i:04d}.jpg'), drawn)
        saved += 1

    result = {
        'aug_profile': opt.aug_profile,
        'samples': saved,
        'label_preserving': opt.aug_profile != 'cctv_paste',
        'bbox_range_errors': bbox_range_errors,
        'class_id_errors': class_id_errors,
        'paste_failures': int(dataset.aug_stats.get('patch_paste_failures', 0)),
        'hard_negative_pastes': int(dataset.aug_stats.get('hard_negative_pastes', 0)),
        'aug_stats': dataset.aug_stats,
        'manual_review_required': opt.aug_profile == 'cctv_paste',
    }
    result['status'] = 'pass' if bbox_range_errors == 0 and class_id_errors == 0 else 'fail'
    (output / 'aug_check.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result['status'] != 'pass':
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--hyp', type=str, default='data/hyp_phase1.yaml')
    parser.add_argument('--aug-profile', choices=['off', 'cctv_pixel', 'cctv_paste'], default='cctv_pixel')
    parser.add_argument('--samples', type=int, default=16)
    parser.add_argument('--img', type=int, default=640)
    parser.add_argument('--phase', type=str, default='phase1')
    parser.add_argument('--hard-negative-manifest', type=str, default='')
    parser.add_argument('--output', type=str, required=True)
    main(parser.parse_args())
