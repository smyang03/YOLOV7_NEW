import json
from pathlib import Path


def read_yolo_label_file(path, with_conf=False):
    path = Path(path)
    rows = []
    if not path.is_file():
        return rows
    for line_no, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        parts = line.strip().split()
        if not parts:
            continue
        expected = 6 if with_conf else 5
        if len(parts) < 5:
            rows.append({'error': f'{path}:{line_no}: expected at least 5 columns'})
            continue
        try:
            values = [float(x) for x in parts[:max(expected, len(parts))]]
        except ValueError:
            rows.append({'error': f'{path}:{line_no}: non-numeric label row'})
            continue
        row = {
            'class_id': int(values[0]),
            'x': values[1],
            'y': values[2],
            'w': values[3],
            'h': values[4],
        }
        if len(values) > 5:
            row['conf'] = values[5]
        rows.append(row)
    return rows


def write_yolo_label_file(path, rows, include_conf=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for row in rows:
        if 'error' in row:
            continue
        values = [row['class_id'], row['x'], row['y'], row['w'], row['h']]
        if include_conf and 'conf' in row:
            values.append(row['conf'])
        lines.append(('%g ' * len(values)).rstrip() % tuple(values))
    path.write_text(('\n'.join(lines) + '\n') if lines else '', encoding='utf-8')


def xywhn_to_xyxy(row):
    x, y, w, h = row['x'], row['y'], row['w'], row['h']
    return [x - w / 2, y - h / 2, x + w / 2, y + h / 2]


def bbox_iou_xywhn(row, others):
    box = xywhn_to_xyxy(row)
    best = 0.0
    for other in others:
        if 'error' in other:
            continue
        other_box = xywhn_to_xyxy(other)
        ix1 = max(box[0], other_box[0])
        iy1 = max(box[1], other_box[1])
        ix2 = min(box[2], other_box[2])
        iy2 = min(box[3], other_box[3])
        iw = max(ix2 - ix1, 0.0)
        ih = max(iy2 - iy1, 0.0)
        inter = iw * ih
        area1 = max(box[2] - box[0], 0.0) * max(box[3] - box[1], 0.0)
        area2 = max(other_box[2] - other_box[0], 0.0) * max(other_box[3] - other_box[1], 0.0)
        union = area1 + area2 - inter
        if union > 0:
            best = max(best, inter / union)
    return best


def valid_row(row, old_nc=None):
    if 'error' in row:
        return False
    if old_nc is not None and not (0 <= int(row['class_id']) < int(old_nc)):
        return False
    return (
        0.0 <= row['x'] <= 1.0 and
        0.0 <= row['y'] <= 1.0 and
        0.0 < row['w'] <= 1.0 and
        0.0 < row['h'] <= 1.0
    )


def filter_pseudo_labels(pseudo_rows, gt_rows=None, pseudo_conf=0.5, pseudo_iou_dedup=0.8, old_nc=None):
    gt_rows = gt_rows or []
    kept = []
    stats = {
        'input': len(pseudo_rows),
        'kept': 0,
        'low_conf': 0,
        'invalid': 0,
        'deduped_with_gt': 0,
    }
    for row in pseudo_rows:
        conf = float(row.get('conf', 1.0)) if 'error' not in row else 0.0
        if conf < pseudo_conf:
            stats['low_conf'] += 1
            continue
        if not valid_row(row, old_nc=old_nc):
            stats['invalid'] += 1
            continue
        if gt_rows and bbox_iou_xywhn(row, gt_rows) > pseudo_iou_dedup:
            stats['deduped_with_gt'] += 1
            continue
        kept.append(row)
    stats['kept'] = len(kept)
    return kept, stats


def merge_label_files(gt_label, pseudo_label, output, pseudo_conf=0.5, pseudo_iou_dedup=0.8, old_nc=None):
    gt_rows = [x for x in read_yolo_label_file(gt_label) if valid_row(x)]
    pseudo_rows = read_yolo_label_file(pseudo_label, with_conf=True)
    kept_pseudo, stats = filter_pseudo_labels(
        pseudo_rows, gt_rows, pseudo_conf=pseudo_conf,
        pseudo_iou_dedup=pseudo_iou_dedup, old_nc=old_nc)
    merged = gt_rows + [{k: v for k, v in row.items() if k != 'conf'} for row in kept_pseudo]
    write_yolo_label_file(output, merged, include_conf=False)
    stats.update({
        'gt_rows': len(gt_rows),
        'merged_rows': len(merged),
        'gt_label': str(gt_label),
        'pseudo_label': str(pseudo_label),
        'output': str(output),
    })
    return stats


class PseudoLabelGenerator:
    def __init__(self, pseudo_conf=0.5, pseudo_iou_dedup=0.8, old_nc=None):
        self.pseudo_conf = float(pseudo_conf)
        self.pseudo_iou_dedup = float(pseudo_iou_dedup)
        self.old_nc = old_nc

    def filter(self, pseudo_rows, gt_rows=None):
        return filter_pseudo_labels(
            pseudo_rows, gt_rows,
            pseudo_conf=self.pseudo_conf,
            pseudo_iou_dedup=self.pseudo_iou_dedup,
            old_nc=self.old_nc)

    def save_image_labels(self, image_path, rows, output_dir, gt_label=''):
        output = Path(output_dir) / f'{Path(image_path).stem}.txt'
        gt_rows = read_yolo_label_file(gt_label) if gt_label else []
        kept, stats = self.filter(rows, gt_rows)
        write_yolo_label_file(output, kept, include_conf=True)
        stats.update({'image': str(image_path), 'output': str(output)})
        return stats


def write_manifest(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {'schema_version': '1.3.7', **data}
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
