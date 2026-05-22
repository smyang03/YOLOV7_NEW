import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.dataset_manifest import read_image_list, read_text_fallback
from utils.datasets import img2label_paths


def validate_label_file(path, nc, tiny_thr=0.001):
    result = {
        'path': str(path),
        'rows': 0,
        'empty': True,
        'format_errors': 0,
        'class_id_errors': 0,
        'bbox_range_errors': 0,
        'tiny_boxes': 0,
    }
    if not Path(path).is_file():
        return result
    lines = [x.strip() for x in read_text_fallback(path).splitlines() if x.strip()]
    result['rows'] = len(lines)
    result['empty'] = len(lines) == 0
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            result['format_errors'] += 1
            continue
        try:
            values = np.array([float(x) for x in parts], dtype=np.float32)
        except ValueError:
            result['format_errors'] += 1
            continue
        cls, x, y, w, h = values.tolist()
        if cls < 0 or cls >= nc or int(cls) != cls:
            result['class_id_errors'] += 1
        if min(x, y, w, h) < 0 or max(x, y, w, h) > 1 or w <= 0 or h <= 0:
            result['bbox_range_errors'] += 1
        if w * h < tiny_thr:
            result['tiny_boxes'] += 1
    return result


def main(opt):
    data_path = Path(opt.data)
    data = yaml.safe_load(read_text_fallback(data_path))
    nc = int(data['nc'])
    data_dir = data_path.parent
    split_names = opt.splits.split(',')

    files = []
    for split in split_names:
        split = split.strip()
        if split and split in data and data[split]:
            files.extend(img2label_paths(read_image_list(data[split], data_dir, ROOT)))

    per_file = [validate_label_file(path, nc, opt.tiny_thr) for path in files]
    summary = {
        'data': str(data_path),
        'splits': split_names,
        'label_files_checked': len(per_file),
        'rows': sum(x['rows'] for x in per_file),
        'empty_label_files': sum(1 for x in per_file if x['empty']),
        'format_errors': sum(x['format_errors'] for x in per_file),
        'class_id_errors': sum(x['class_id_errors'] for x in per_file),
        'bbox_range_errors': sum(x['bbox_range_errors'] for x in per_file),
        'tiny_boxes': sum(x['tiny_boxes'] for x in per_file),
    }
    summary['status'] = 'pass' if summary['format_errors'] == 0 and summary['class_id_errors'] == 0 and \
        summary['bbox_range_errors'] == 0 else 'fail'

    result = {'summary': summary, 'files': per_file if opt.verbose else []}
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if opt.output:
        output = Path(opt.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + '\n', encoding='utf-8')
    print(text)
    if summary['status'] != 'pass':
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--splits', type=str, default='train,val')
    parser.add_argument('--tiny-thr', type=float, default=0.001)
    parser.add_argument('--output', type=str, default='')
    parser.add_argument('--verbose', action='store_true')
    main(parser.parse_args())
