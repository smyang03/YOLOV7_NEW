import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.pseudo_label import merge_label_files, write_yolo_label_file


IMG_FORMATS = ('bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo')
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
        return sorted(str(p) for p in path.rglob('*') if p.suffix[1:].lower() in IMG_FORMATS)
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


def load_yaml(path):
    return yaml.safe_load(read_text_fallback(path)) or {}


def save_yaml(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding='utf-8')


def unique_stem(image):
    image = Path(image)
    digest = hashlib.sha1(str(image).encode()).hexdigest()[:12]
    return f'{digest}_{image.stem}'


def link_image(src, dst, mode='symlink'):
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return 'exists'
    if mode == 'copy':
        shutil.copy2(src, dst)
        return 'copy'
    if mode == 'hardlink':
        try:
            dst.hardlink_to(src)
            return 'hardlink'
        except OSError:
            shutil.copy2(src, dst)
            return 'copy_fallback'
    if mode == 'symlink':
        try:
            dst.symlink_to(src)
            return 'symlink'
        except OSError:
            shutil.copy2(src, dst)
            return 'copy_fallback'
    raise ValueError(f'unsupported link mode: {mode}')


def data_nc(data):
    return int(data['nc']) if 'nc' in data else None


def pseudo_path_from_dir(pseudo_dir, image):
    if not pseudo_dir:
        return None
    pseudo_dir = Path(pseudo_dir)
    candidates = [
        pseudo_dir / f'{Path(image).stem}.txt',
        pseudo_dir / f'{unique_stem(image)}.txt',
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def model_predict_rows(model, device, image, img_size, stride, conf_thres, iou_thres):
    import torch
    from utils.datasets import LoadImages
    from utils.general import non_max_suppression, scale_coords, xyxy2xywh

    rows = []
    dataset = LoadImages(str(image), img_size=img_size, stride=stride)
    for _, img, im0, _ in dataset:
        img_tensor = torch.from_numpy(img).to(device).float() / 255.0
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)
        with torch.no_grad():
            pred = model(img_tensor)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres)
        det = pred[0]
        if len(det):
            det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], im0.shape).round()
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            for *xyxy, conf, cls in det:
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                rows.append({
                    'class_id': int(cls),
                    'x': float(xywh[0]),
                    'y': float(xywh[1]),
                    'w': float(xywh[2]),
                    'h': float(xywh[3]),
                    'conf': float(conf),
                })
    return rows


def prepare_model(opt):
    if opt.pseudo_labels:
        return None, None, None, None
    if not opt.weights:
        raise SystemExit('--weights is required when --pseudo-labels is not provided')
    from models.experimental import attempt_load
    from utils.general import check_img_size
    from utils.torch_utils import select_device

    device = select_device(opt.device)
    model = attempt_load(opt.weights, map_location=device).eval()
    stride = int(model.stride.max())
    img_size = check_img_size(opt.img_size, s=stride)
    return model, device, stride, img_size


def main(opt):
    data_path = Path(opt.data)
    data = load_yaml(data_path)
    images = read_image_list(data.get('train'), data_path.parent, ROOT) if data.get('train') else []
    if opt.max_images:
        images = images[:opt.max_images]
    output_dir = Path(opt.output)
    image_dir = output_dir / 'images'
    label_dir = output_dir / 'labels'
    pseudo_dir = output_dir / 'teacher_pseudo_labels'
    train_list = output_dir / 'train_pseudo_old_labels.txt'
    data_output = output_dir / 'pseudo_old_data.yaml'
    manifest_output = Path(opt.manifest) if opt.manifest else output_dir / 'pseudo_old_label_manifest.json'

    if opt.dry_run:
        manifest = {
            'schema_version': '1.3.7',
            'status': 'dry_run',
            'data': str(data_path),
            'output': str(output_dir),
            'train_images': len(images),
            'weights': opt.weights,
            'pseudo_labels': opt.pseudo_labels,
        }
        manifest_output.parent.mkdir(parents=True, exist_ok=True)
        manifest_output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return

    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    pseudo_dir.mkdir(parents=True, exist_ok=True)
    model, device, stride, img_size = prepare_model(opt)
    labels = img2label_paths(images)
    old_nc = opt.old_nc if opt.old_nc >= 0 else data_nc(data)
    class_counts = Counter()
    link_counts = Counter()
    reports = []
    output_images = []

    for index, (image, gt_label) in enumerate(zip(images, labels), 1):
        stem = unique_stem(image)
        shadow_image = image_dir / f'{stem}{Path(image).suffix}'
        shadow_label = label_dir / f'{stem}.txt'
        pseudo_label = pseudo_path_from_dir(opt.pseudo_labels, image)

        if pseudo_label is None:
            pseudo_label = pseudo_dir / f'{stem}.txt'
            rows = model_predict_rows(
                model, device, image, img_size, stride,
                opt.conf_thres, opt.iou_thres)
            write_yolo_label_file(pseudo_label, rows, include_conf=True)

        link_counts[link_image(image, shadow_image, mode=opt.link_mode)] += 1
        report = merge_label_files(
            gt_label, pseudo_label, shadow_label,
            pseudo_conf=opt.pseudo_conf,
            pseudo_iou_dedup=opt.dedupe_iou,
            old_nc=old_nc)
        reports.append(report)
        output_images.append(str(shadow_image))
        class_counts.update([int(row.split()[0]) for row in shadow_label.read_text(encoding='utf-8').splitlines() if row.strip()])
        if opt.progress_interval > 0 and index % opt.progress_interval == 0:
            print(f'prepared {index}/{len(images)} images')

    train_list.write_text('\n'.join(output_images) + ('\n' if output_images else ''), encoding='utf-8')
    output_data = dict(data)
    output_data['train'] = str(train_list)
    save_yaml(data_output, output_data)

    manifest = {
        'schema_version': '1.3.7',
        'status': 'pass',
        'data': str(data_path),
        'output': str(output_dir),
        'data_output': str(data_output),
        'train_list': str(train_list),
        'weights': opt.weights,
        'pseudo_labels': opt.pseudo_labels,
        'train_images': len(images),
        'pseudo_input': sum(x['input'] for x in reports),
        'pseudo_kept': sum(x['kept'] for x in reports),
        'deduped_with_gt': sum(x['deduped_with_gt'] for x in reports),
        'merged_rows': sum(x['merged_rows'] for x in reports),
        'class_counts': {str(k): int(v) for k, v in sorted(class_counts.items())},
        'link_counts': {str(k): int(v) for k, v in sorted(link_counts.items())},
        'conf_thres': opt.conf_thres,
        'iou_thres': opt.iou_thres,
        'pseudo_conf': opt.pseudo_conf,
        'dedupe_iou': opt.dedupe_iou,
        'old_nc': old_nc,
        'reports': reports if opt.verbose else [],
    }
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--weights', type=str, default='')
    parser.add_argument('--pseudo-labels', type=str, default='',
                        help='existing pseudo label directory; if omitted, --weights is used to generate labels')
    parser.add_argument('--manifest', type=str, default='')
    parser.add_argument('--conf-thres', type=float, default=0.6)
    parser.add_argument('--iou-thres', type=float, default=0.45)
    parser.add_argument('--pseudo-conf', type=float, default=0.6)
    parser.add_argument('--dedupe-iou', type=float, default=0.8)
    parser.add_argument('--old-nc', type=int, default=-1)
    parser.add_argument('--img-size', '--img', dest='img_size', type=int, default=640)
    parser.add_argument('--device', type=str, default='')
    parser.add_argument('--link-mode', choices=['symlink', 'hardlink', 'copy'], default='symlink')
    parser.add_argument('--max-images', type=int, default=0)
    parser.add_argument('--progress-interval', type=int, default=100)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    main(parser.parse_args())
