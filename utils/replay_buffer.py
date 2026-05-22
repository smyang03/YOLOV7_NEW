import hashlib
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from utils.datasets import img2label_paths, img_formats


TEXT_ENCODINGS = ('utf-8', 'cp949', 'euc-kr', 'latin1', 'utf-16')


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
        images = []
        for line in read_text_fallback(path).splitlines():
            line = line.strip()
            if not line:
                continue
            item = Path(line)
            if not item.is_absolute():
                item = path.parent / item if line.startswith('.') else root / item
            images.append(str(item))
        return sorted(images)
    return []


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def label_classes(path):
    path = Path(path)
    classes = []
    counts = Counter()
    errors = 0
    if not path.is_file():
        return classes, errors, counts
    for line in read_text_fallback(path).splitlines():
        line = line.strip().lstrip('\ufeff')
        if not line:
            continue
        try:
            cls = int(float(line.split()[0]))
        except (IndexError, ValueError):
            errors += 1
            continue
        classes.append(cls)
        counts[cls] += 1
    return sorted(set(classes)), errors, counts


def _copy_selected(selected, output_dir):
    output_dir = Path(output_dir)
    image_dir = output_dir / 'images'
    label_dir = output_dir / 'labels'
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for item in selected:
        image = Path(item['image'])
        label = Path(item['label'])
        stem = hashlib.sha1(str(image).encode()).hexdigest()[:10] + '_' + image.stem
        dst_image = image_dir / f'{stem}{image.suffix}'
        dst_label = label_dir / f'{stem}.txt'
        if image.is_file():
            shutil.copy2(image, dst_image)
        if label.is_file():
            shutil.copy2(label, dst_label)
        copied_item = dict(item)
        copied_item['copied_image'] = str(dst_image)
        copied_item['copied_label'] = str(dst_label)
        copied.append(copied_item)
    return copied


class ReplayBufferBuilder:
    def __init__(self, data, split='train', replay_ratio=0.3, seed=0, root=None):
        self.data = Path(data)
        self.split = split
        self.replay_ratio = float(replay_ratio)
        self.seed = int(seed)
        self.root = Path(root) if root else Path(__file__).resolve().parents[1]

    def _load_images(self):
        data_dict = yaml.safe_load(read_text_fallback(self.data)) or {}
        data_dir = self.data.parent
        source = data_dict.get(self.split)
        return data_dict, read_image_list(source, data_dir, self.root) if source else []

    def build(self, output='', copy_dir=''):
        data_dict, images = self._load_images()
        labels = img2label_paths(images)
        per_image = []
        class_counts = Counter()
        label_errors = 0
        aggregate = hashlib.sha256()
        for image, label in zip(images, labels):
            classes, errors, counts = label_classes(label)
            label_errors += errors
            class_counts.update(counts)
            if Path(image).is_file():
                aggregate.update(str(image).encode())
                aggregate.update(file_sha256(image).encode())
            if Path(label).is_file():
                aggregate.update(str(label).encode())
                aggregate.update(file_sha256(label).encode())
            per_image.append({
                'image': str(image),
                'label': str(label),
                'classes': classes,
                'label_exists': Path(label).is_file(),
            })

        if self.replay_ratio <= 0 or not per_image:
            target_count = 0
        elif self.replay_ratio <= 1:
            target_count = max(1, int(math.ceil(len(per_image) * self.replay_ratio)))
        else:
            target_count = min(len(per_image), int(self.replay_ratio))

        rng = random.Random(self.seed)
        class_to_count = defaultdict(lambda: 1)
        class_to_count.update(class_counts)

        def score(item):
            rare_score = sum(1.0 / class_to_count[c] for c in item['classes'])
            label_bonus = 0.1 if item['label_exists'] else 0.0
            return rare_score + label_bonus + rng.random() * 1e-6

        selected = sorted(per_image, key=score, reverse=True)[:target_count]
        for item in selected:
            item['selection_reason'] = 'class_balanced_replay'
        if copy_dir:
            selected = _copy_selected(selected, copy_dir)

        manifest = {
            'schema_version': '1.3.7',
            'source_data': str(self.data),
            'source_split': self.split,
            'source_train': data_dict.get(self.split),
            'replay_ratio': self.replay_ratio,
            'image_count': len(per_image),
            'selected_count': len(selected),
            'class_counts': {str(k): int(v) for k, v in sorted(class_counts.items())},
            'selected_images': selected,
            'selection_seed': self.seed,
            'label_errors': label_errors,
            'source_hash': aggregate.hexdigest(),
            'status': 'pass' if label_errors == 0 else 'warn',
        }
        if output:
            output = Path(output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        return manifest
