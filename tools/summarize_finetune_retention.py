import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


NUM = r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?'
OVERALL_RE = re.compile(
    rf'^\s*(?P<epoch>\d+)/(?P<epochs>\d+).*?\[\s*(?P<scenario>[^\]]+)\]\s+'
    rf'(?P<precision>{NUM})\s+(?P<recall>{NUM})\s+(?P<map50>{NUM})\s+(?P<map50_95>{NUM})'
)
CLASS_RE = re.compile(
    rf'^\s*\[\s*(?P<scenario>[^\]]+)\]\[\s*(?P<class_name>[^\]]+)\]\s+'
    rf'Images:\s*(?P<images>\d+),\s*P:\s*(?P<precision>{NUM}),\s*R:\s*(?P<recall>{NUM}),\s*'
    rf'mAP@\.5:\s*(?P<map50>{NUM}),\s*mAP@\.5:\.95:\s*(?P<map50_95>{NUM})'
)


def _float(row, key):
    value = row.get(key)
    return None if value in ('', None) else float(value)


def _metric_dict(match):
    return {
        'precision': float(match.group('precision')),
        'recall': float(match.group('recall')),
        'map50': float(match.group('map50')),
        'map50_95': float(match.group('map50_95')),
    }


def parse_results_detail(path):
    path = Path(path)
    overall = []
    per_class = []
    current_epoch = None
    if not path.is_file():
        return overall, per_class
    for line in path.read_text(encoding='utf-8-sig', errors='ignore').splitlines():
        match = OVERALL_RE.match(line)
        if match:
            current_epoch = int(match.group('epoch'))
            row = {
                'epoch': current_epoch,
                'scenario': match.group('scenario').strip(),
                **_metric_dict(match),
            }
            overall.append(row)
            continue
        match = CLASS_RE.match(line)
        if match and current_epoch is not None:
            per_class.append({
                'epoch': current_epoch,
                'scenario': match.group('scenario').strip(),
                'class_name': match.group('class_name').strip(),
                'images': int(match.group('images')),
                **_metric_dict(match),
            })
    return overall, per_class


def scenario_names(overall):
    names = []
    for row in overall:
        if row['scenario'] not in names:
            names.append(row['scenario'])
    return names


def epoch_scores(overall, select_scenario='combined'):
    by_epoch = defaultdict(list)
    for row in overall:
        by_epoch[row['epoch']].append(row)
    scores = {}
    for epoch, rows in by_epoch.items():
        if select_scenario.lower() == 'combined':
            scores[epoch] = sum(row['map50_95'] for row in rows) / max(len(rows), 1)
            continue
        selected = [row for row in rows if row['scenario'] == select_scenario]
        if selected:
            scores[epoch] = selected[0]['map50_95']
    return scores


def best_epoch(overall, select_scenario='combined'):
    scores = epoch_scores(overall, select_scenario)
    if not scores:
        return None, None
    epoch = max(scores, key=lambda x: scores[x])
    return epoch, scores[epoch]


def row_at(overall, epoch, scenario):
    for row in overall:
        if row['epoch'] == epoch and row['scenario'] == scenario:
            return row
    return {}


def best_baseline_by_scenario(overall):
    result = {}
    for scenario in scenario_names(overall):
        rows = [row for row in overall if row['scenario'] == scenario]
        if rows:
            result[scenario] = max(rows, key=lambda row: row['map50_95'])
    return result


def worst_class_drop(baseline_class_rows, run_class_rows, epoch, scenario):
    base_best = {}
    for row in baseline_class_rows:
        if row['scenario'] != scenario:
            continue
        current = base_best.get(row['class_name'])
        if current is None or row['map50_95'] > current['map50_95']:
            base_best[row['class_name']] = row
    run_rows = {
        row['class_name']: row for row in run_class_rows
        if row['epoch'] == epoch and row['scenario'] == scenario
    }
    drops = []
    for class_name, base in base_best.items():
        current = run_rows.get(class_name)
        if not current:
            continue
        drops.append({
            'class_name': class_name,
            'baseline_map50_95': base['map50_95'],
            'current_map50_95': current['map50_95'],
            'drop': current['map50_95'] - base['map50_95'],
        })
    if not drops:
        return {}
    return min(drops, key=lambda row: row['drop'])


def detail_path(run_dir):
    run_dir = Path(run_dir)
    if run_dir.is_file():
        return run_dir
    return run_dir / 'results_detail.txt'


def summarize_run(run_dir, baseline, baseline_classes, opt):
    path = detail_path(run_dir)
    overall, per_class = parse_results_detail(path)
    if not overall:
        return {
            'run': str(run_dir),
            'results_detail': str(path),
            'status': 'missing_results_detail' if not path.is_file() else 'empty_results_detail',
            'best_weight': str(Path(run_dir) / 'weights' / 'best.pt'),
        }
    select_epoch, select_score = best_epoch(overall, opt.select_scenario)
    scenarios = scenario_names(overall)
    finetune_scenario = opt.finetune_scenario or (scenarios[0] if scenarios else '')
    base_scenario = opt.base_scenario or (scenarios[-1] if scenarios else '')
    finetune_row = row_at(overall, select_epoch, finetune_scenario)
    base_row = row_at(overall, select_epoch, base_scenario)
    baseline_base = baseline.get(base_scenario, {})
    baseline_finetune = baseline.get(finetune_scenario, {})
    base_current = _float(base_row, 'map50_95')
    base_before = _float(baseline_base, 'map50_95')
    finetune_current = _float(finetune_row, 'map50_95')
    finetune_before = _float(baseline_finetune, 'map50_95')
    retention = None if base_current is None or not base_before else base_current / base_before * 100.0
    finetune_delta = None if finetune_current is None or finetune_before is None else finetune_current - finetune_before
    worst_drop = worst_class_drop(baseline_classes, per_class, select_epoch, base_scenario) if baseline_classes else {}
    status = 'keep_candidate'
    if retention is not None and retention < opt.min_base_retention:
        status = 'drop_base_retention'
    if opt.require_finetune_gain and finetune_delta is not None and finetune_delta < opt.min_finetune_delta:
        status = 'drop_finetune_gain'
    return {
        'run': str(run_dir),
        'results_detail': str(path),
        'status': status,
        'best_epoch': select_epoch,
        'selection_scenario': opt.select_scenario,
        'selection_map50_95': select_score,
        'finetune_scenario': finetune_scenario,
        'finetune_map50_95': finetune_current,
        'finetune_baseline_map50_95': finetune_before,
        'finetune_delta': finetune_delta,
        'base_scenario': base_scenario,
        'base_map50_95': base_current,
        'base_baseline_map50_95': base_before,
        'base_retention_percent': retention,
        'base_worst_class': worst_drop.get('class_name', ''),
        'base_worst_class_delta': worst_drop.get('drop'),
        'base_worst_class_current': worst_drop.get('current_map50_95'),
        'base_worst_class_baseline': worst_drop.get('baseline_map50_95'),
        'best_weight': str(Path(run_dir) / 'weights' / 'best.pt'),
    }


def write_csv(rows, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with output.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(opt):
    baseline_overall = []
    baseline_classes = []
    if opt.baseline:
        baseline_overall, baseline_classes = parse_results_detail(detail_path(opt.baseline))
    baseline = best_baseline_by_scenario(baseline_overall)
    rows = [summarize_run(run, baseline, baseline_classes, opt) for run in opt.runs]
    write_csv(rows, opt.output)
    if opt.json_output:
        Path(opt.json_output).parent.mkdir(parents=True, exist_ok=True)
        Path(opt.json_output).write_text(json.dumps(rows, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(yaml.safe_dump({'output': opt.output, 'runs': len(rows), 'rows': rows}, sort_keys=False, allow_unicode=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', required=True, help='run directories or results_detail.txt files')
    parser.add_argument('--baseline', type=str, default='', help='baseline run dir or results_detail.txt')
    parser.add_argument('--output', type=str, default='finetune_retention_summary.csv')
    parser.add_argument('--json-output', type=str, default='')
    parser.add_argument('--select-scenario', type=str, default='combined')
    parser.add_argument('--finetune-scenario', type=str, default='')
    parser.add_argument('--base-scenario', type=str, default='')
    parser.add_argument('--min-base-retention', type=float, default=95.0)
    parser.add_argument('--min-finetune-delta', type=float, default=0.0)
    parser.add_argument('--require-finetune-gain', action='store_true')
    main(parser.parse_args())
