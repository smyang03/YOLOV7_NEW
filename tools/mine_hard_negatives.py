import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.dataset_manifest import read_image_list, read_text_fallback
from utils.datasets import img2label_paths


def main(opt):
    data_path = Path(opt.data)
    data = yaml.safe_load(read_text_fallback(data_path))
    images = read_image_list(data['train'], data_path.parent, ROOT)
    labels = img2label_paths(images)
    crops = []
    rng = random.Random(opt.seed)
    for image_path, label_path in zip(images, labels):
        if Path(label_path).is_file() and read_text_fallback(label_path).strip():
            continue
        img = cv2.imread(str(image_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        for _ in range(opt.crops_per_image):
            cw = rng.randint(max(16, w // 12), max(17, w // 4))
            ch = rng.randint(max(16, h // 12), max(17, h // 4))
            if cw >= w or ch >= h:
                continue
            x1 = rng.randint(0, w - cw)
            y1 = rng.randint(0, h - ch)
            crops.append({'image': str(image_path), 'bbox_xyxy': [x1, y1, x1 + cw, y1 + ch], 'source': 'empty_label_image'})
            if len(crops) >= opt.max_crops:
                break
        if len(crops) >= opt.max_crops:
            break
    result = {'data': str(data_path), 'crops': crops, 'count': len(crops)}
    output = Path(opt.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'hard negative manifest saved to {output} ({len(crops)} crops)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--max-crops', type=int, default=200)
    parser.add_argument('--crops-per-image', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    main(parser.parse_args())
