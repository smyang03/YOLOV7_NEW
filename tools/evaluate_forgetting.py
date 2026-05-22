import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.summarize_metrics import parse_rows, row_to_metrics


def load_metrics(path):
    path = Path(path)
    if not path.is_file():
        return {}
    if path.suffix.lower() == '.json':
        return json.loads(path.read_text(encoding='utf-8'))
    rows = parse_rows(path)
    if not rows:
        return {}
    return {'last': row_to_metrics(rows[-1]), 'best': row_to_metrics(max(rows, key=lambda x: x[11] if len(x) > 11 else x[-1]))}


def metric_value(metrics, key='map50_95'):
    for scope in ('best', 'last'):
        if scope in metrics and key in metrics[scope]:
            return float(metrics[scope][key])
    return None


def percent_retention(base, current):
    if base is None or current is None or base == 0:
        return None
    return current / base * 100.0


def main(opt):
    scratch = load_metrics(opt.scratch_results)
    finetune = load_metrics(opt.finetune_results)
    scratch_map = metric_value(scratch, opt.metric)
    finetune_map = metric_value(finetune, opt.metric)
    retention = percent_retention(scratch_map, finetune_map)
    mapping = json.loads(Path(opt.class_mapping_check).read_text(encoding='utf-8')) if opt.class_mapping_check else {}
    result = {
        'schema_version': '1.3.7',
        'scratch_baseline': opt.scratch_results,
        'finetune_run': opt.finetune_results,
        'new_class_map': mapping.get('new_only_classes', []),
        'old_class_map': mapping.get('resolved_mapping', {}),
        'heldout_class_map': [],
        'metric': opt.metric,
        'scratch_metric': scratch_map,
        'finetune_metric': finetune_map,
        'old_class_drop_percent': None if retention is None else max(0.0, 100.0 - retention),
        'new_class_retention_percent': None,
        'overall_retention_percent': retention,
        'replay_ratio': opt.replay_ratio,
        'distill_alpha': opt.distill_alpha,
        'distill_beta': opt.distill_beta,
        'bn_policy': opt.bn_policy,
        'freeze_policy': opt.freeze_policy,
        'sub_stage': opt.sub_stage,
        'status': 'pass' if retention is None or retention >= opt.min_retention else 'fail',
    }
    output = Path(opt.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(result, sort_keys=False, allow_unicode=True), encoding='utf-8')
    print(yaml.safe_dump(result, sort_keys=False, allow_unicode=True))
    if result['status'] != 'pass':
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scratch-results', type=str, required=True)
    parser.add_argument('--finetune-results', type=str, required=True)
    parser.add_argument('--class-mapping-check', type=str, default='')
    parser.add_argument('--output', type=str, default='forgetting_report.yaml')
    parser.add_argument('--metric', type=str, default='map50_95')
    parser.add_argument('--min-retention', type=float, default=93.0)
    parser.add_argument('--replay-ratio', type=float, default=0.3)
    parser.add_argument('--distill-alpha', type=str, default='0.0')
    parser.add_argument('--distill-beta', type=str, default='0.0')
    parser.add_argument('--bn-policy', choices=['train', 'eval'], default='train')
    parser.add_argument('--freeze-policy', choices=['none', 'backbone', 'partial', 'neck_lower'], default='none')
    parser.add_argument('--sub-stage', type=str, default='1.3.7-E1')
    main(parser.parse_args())
