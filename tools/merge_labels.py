import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.pseudo_label import merge_label_files


def _label_files(path):
    path = Path(path)
    if path.is_file():
        return {Path(path.name): path}
    return {p.relative_to(path): p for p in sorted(path.rglob('*.txt'))}


def main(opt):
    gt_files = _label_files(opt.gt_labels)
    pseudo_files = _label_files(opt.pseudo_labels)
    rel_paths = sorted(set(gt_files) | set(pseudo_files))
    output_dir = Path(opt.output)
    reports = []
    for rel in rel_paths:
        gt = gt_files.get(rel, Path('__missing_gt__'))
        pseudo = pseudo_files.get(rel, Path('__missing_pseudo__'))
        reports.append(merge_label_files(
            gt, pseudo, output_dir / rel,
            pseudo_conf=opt.pseudo_conf,
            pseudo_iou_dedup=opt.dedupe_iou,
            old_nc=opt.old_nc if opt.old_nc >= 0 else None))
    result = {
        'schema_version': '1.3.7',
        'gt_labels': opt.gt_labels,
        'pseudo_labels': opt.pseudo_labels,
        'output': opt.output,
        'files': len(reports),
        'pseudo_input': sum(x['input'] for x in reports),
        'pseudo_kept': sum(x['kept'] for x in reports),
        'deduped_with_gt': sum(x['deduped_with_gt'] for x in reports),
        'report': reports if opt.verbose else [],
        'status': 'pass',
    }
    report_path = Path(opt.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt-labels', type=str, required=True)
    parser.add_argument('--pseudo-labels', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--dedupe-iou', type=float, default=0.8)
    parser.add_argument('--pseudo-conf', type=float, default=0.5)
    parser.add_argument('--old-nc', type=int, default=-1)
    parser.add_argument('--report', type=str, default='merge_report.json')
    parser.add_argument('--verbose', action='store_true')
    main(parser.parse_args())
