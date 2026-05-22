import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.summarize_metrics import parse_rows, row_to_metrics
from utils.stage_schema import StageConfig, StageResult


def _read_json(path):
    path = Path(path)
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding='utf-8-sig'))


def _as_float(value):
    if value in ('', None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _best_from_results_csv(path):
    path = Path(path)
    if not path.is_file():
        return {}
    with path.open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    key = 'map50_95' if 'map50_95' in rows[0] else ('metrics/mAP_0.5:0.95' if 'metrics/mAP_0.5:0.95' in rows[0] else None)
    if key:
        best = max(rows, key=lambda row: _as_float(row.get(key)) if _as_float(row.get(key)) is not None else -1.0)
    else:
        best = rows[-1]
    return {k: _as_float(v) for k, v in best.items()}


def _best_from_results_txt(path):
    path = Path(path)
    if not path.is_file():
        return {}
    rows = parse_rows(path)
    if not rows:
        return {}
    best_row = max(rows, key=lambda x: x[11] if len(x) > 11 else x[-1])
    return row_to_metrics(best_row)


def _extract_export_status(export_check):
    if not export_check:
        return 'skip', None
    comparison = export_check.get('comparison') or {}
    status = comparison.get('status') or export_check.get('status') or 'unknown'
    return status, comparison.get('max_abs_diff')


def collect_stage_result(stage_dir):
    stage_dir = Path(stage_dir)
    config_path = stage_dir / 'stage_config.yaml'
    result_path = stage_dir / 'stage_result.yaml'
    config = StageConfig.load(config_path).to_dict() if config_path.is_file() else {}
    result = StageResult.load(result_path).to_dict() if result_path.is_file() else {}

    metrics = {}
    csv_metrics = _best_from_results_csv(stage_dir / 'results.csv')
    txt_metrics = _best_from_results_txt(stage_dir / 'results.txt')
    source_metrics = csv_metrics or txt_metrics
    metrics['primary_mAP'] = source_metrics.get('map50_95', source_metrics.get('metrics/mAP_0.5:0.95'))
    metrics['mAP50'] = source_metrics.get('map50', source_metrics.get('metrics/mAP_0.5'))
    metrics['precision'] = source_metrics.get('precision', source_metrics.get('metrics/precision'))
    metrics['recall'] = source_metrics.get('recall', source_metrics.get('metrics/recall'))

    profile = _read_json(stage_dir / 'profile.json')
    metrics['GFLOPs'] = profile.get('gflops', profile.get('current_gflops'))
    metrics['GFLOPs_delta_percent'] = profile.get('gflops_delta_percent')
    metrics['python_infer_ms'] = profile.get('python_infer_ms')
    metrics['python_nms_ms'] = profile.get('python_nms_ms')
    metrics['small_AP'] = profile.get('small_AP')
    metrics['rare_recall'] = profile.get('rare_recall')
    metrics['FP_per_image'] = profile.get('FP_per_image')
    metrics['FN_per_image'] = profile.get('FN_per_image')

    nms = _read_json(stage_dir / 'nms_cost.json')
    metrics['python_nms_ms'] = metrics['python_nms_ms'] or nms.get('nms_ms')

    export_check = _read_json(stage_dir / 'export_check.json')
    export_status, onnx_diff = _extract_export_status(export_check)
    metrics['export_status'] = export_status
    metrics['onnx_max_abs_diff'] = onnx_diff

    if result.get('metrics'):
        merged_metrics = dict(metrics)
        merged_metrics.update({k: v for k, v in result.get('metrics', {}).items() if v is not None})
        metrics = merged_metrics
    for key in ('primary_mAP', 'mAP50', 'small_AP', 'rare_recall', 'GFLOPs', 'python_nms_ms', 'onnx_max_abs_diff'):
        if result.get(key) is not None:
            metrics[key] = result.get(key)

    best = stage_dir / 'weights' / 'best.pt'
    last = stage_dir / 'weights' / 'last.pt'
    best_weight = result.get('best_weight') or (str(best) if best.is_file() else (str(last) if last.is_file() else ''))
    collected = {
        'schema_version': '1.3.8',
        'stage_dir': str(stage_dir),
        'stage_config': config,
        'stage_result': result,
        'stage_id': result.get('stage_id') or config.get('stage_id') or stage_dir.name[:2],
        'stage_name': config.get('stage_name') or stage_dir.name,
        'model_family': config.get('model_family'),
        'dataset_profile': config.get('dataset_profile'),
        'train_type': result.get('train_type') or config.get('train_type', ''),
        'decision': result.get('decision', 'defer'),
        'reason': result.get('reason', ''),
        'best_weight': best_weight,
        'fallback_weight': result.get('fallback_weight', ''),
        'hard_fail': bool(result.get('hard_fail', False)),
        'soft_fail': bool(result.get('soft_fail', False)),
        'failed_category': result.get('failed_category'),
        'missing_artifacts': list(result.get('missing_artifacts') or []),
        'log_paths': dict(result.get('log_paths') or {}),
        'metrics': metrics,
        'artifacts': {
            'stage_config': str(config_path) if config_path.is_file() else '',
            'stage_result': str(result_path) if result_path.is_file() else '',
            'results_csv': str(stage_dir / 'results.csv') if (stage_dir / 'results.csv').is_file() else '',
            'results_txt': str(stage_dir / 'results.txt') if (stage_dir / 'results.txt').is_file() else '',
            'loss_detail_csv': str(stage_dir / 'loss_detail.csv') if (stage_dir / 'loss_detail.csv').is_file() else '',
            'debug_trace': str(stage_dir / 'debug_trace.log') if (stage_dir / 'debug_trace.log').is_file() else '',
            'error_trace': str(stage_dir / 'error_trace.log') if (stage_dir / 'error_trace.log').is_file() else '',
            'run_summary': str(stage_dir / 'run_summary.md') if (stage_dir / 'run_summary.md').is_file() else '',
            'profile_json': str(stage_dir / 'profile.json') if (stage_dir / 'profile.json').is_file() else '',
            'export_check_json': str(stage_dir / 'export_check.json') if (stage_dir / 'export_check.json').is_file() else '',
        },
    }
    return collected


def main(opt):
    result = collect_stage_result(opt.stage_dir)
    output = Path(opt.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() in ('.yaml', '.yml'):
        output.write_text(yaml.safe_dump(result, sort_keys=False, allow_unicode=True), encoding='utf-8')
    else:
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'stage results collected to {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage-dir', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    main(parser.parse_args())
