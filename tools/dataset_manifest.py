import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

TEXT_ENCODINGS = ('utf-8', 'cp949', 'euc-kr', 'latin1', 'utf-16')
img_formats = ('bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo')


def img2label_paths(img_paths):
    replacements = (
        ('/images/', '/labels/'),
        ('/JPEGImages/', '/labels/'),
        ('\\images\\', '\\labels\\'),
        ('\\JPEGImages\\', '\\labels\\'),
    )
    label_paths = []
    for x in img_paths:
        x = str(x)
        for src, dst in replacements:
            if src in x:
                x = x.replace(src, dst, 1)
                break
        label_paths.append(str(Path(x).with_suffix('.txt')))
    return label_paths


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_text_fallback(path):
    last_error = None
    for encoding in TEXT_ENCODINGS:
        try:
            return Path(path).read_text(encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error


def resolve_path(value, data_dir, root):
    path = Path(value)
    if path.is_absolute():
        return path
    data_relative = data_dir / path
    return data_relative if data_relative.exists() else root / path


def read_image_list(source, data_dir, root):
    if isinstance(source, dict):
        source = source.get('path') or source.get('images') or source.get('source')
        if not source:
            return []
    if isinstance(source, (list, tuple)):
        images = []
        for item in source:
            images.extend(read_image_list(item, data_dir, root))
        return images

    path = resolve_path(source, data_dir, root)
    if path.is_dir():
        return sorted(str(p) for p in path.rglob('*') if p.suffix[1:].lower() in img_formats)
    if path.is_file():
        lines = [x.strip() for x in read_text_fallback(path).splitlines() if x.strip()]
        images = []
        for line in lines:
            item = Path(line)
            if not item.is_absolute():
                item = path.parent / item if line.startswith('.') else root / item
            images.append(str(item))
        return sorted(images)
    return []


def split_summary(name, source, data_dir):
    images = read_image_list(source, data_dir, ROOT)
    labels = img2label_paths(images)
    existing_labels = [x for x in labels if Path(x).is_file()]
    missing_labels = len(labels) - len(existing_labels)
    label_rows = 0
    for label in existing_labels:
        rows = [x for x in read_text_fallback(label).splitlines() if x.strip()]
        label_rows += len(rows)

    aggregate = hashlib.sha256()
    for path in sorted(images + existing_labels):
        p = Path(path)
        if p.is_file():
            aggregate.update(str(p).encode())
            aggregate.update(file_sha256(p).encode())

    return {
        'split': name,
        'source': source,
        'image_count': len(images),
        'label_count': len(existing_labels),
        'missing_label_count': missing_labels,
        'label_row_count': label_rows,
        'checksum': aggregate.hexdigest(),
    }


def main(opt):
    data_path = Path(opt.data)
    data = yaml.safe_load(read_text_fallback(data_path))
    data_dir = data_path.parent

    splits = {}
    for name in ('train', 'val', 'test'):
        if name in data and data[name]:
            splits[name] = split_summary(name, data[name], data_dir)

    result = {
        'schema_version': opt.schema_version,
        'data': str(data_path),
        'stage': opt.stage,
        'nc': data.get('nc'),
        'names': data.get('names'),
        'splits': splits,
    }
    aggregate = hashlib.sha256()
    for split in splits.values():
        aggregate.update(split['checksum'].encode())
    result['aggregate_checksum'] = aggregate.hexdigest()

    output = Path(opt.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'dataset manifest saved to {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='data yaml path')
    parser.add_argument('--output', type=str, required=True, help='dataset_manifest.json output path')
    parser.add_argument('--schema-version', type=str, default='1.3.1')
    parser.add_argument('--stage', type=str, default='')
    main(parser.parse_args())
