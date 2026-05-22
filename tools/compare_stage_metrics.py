import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.collect_stage_results import collect_stage_result


METRICS = (
    'primary_mAP', 'mAP50', 'small_AP', 'rare_recall', 'FP_per_image', 'FN_per_image',
    'GFLOPs', 'GFLOPs_delta_percent', 'python_infer_ms', 'python_nms_ms', 'onnx_max_abs_diff')


def _num(value):
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _delta(current, ref):
    current = _num(current)
    ref = _num(ref)
    return None if current is None or ref is None else current - ref


def _percent_delta(current, ref):
    current = _num(current)
    ref = _num(ref)
    if current is None or ref in (None, 0.0):
        return None
    return (current - ref) / ref * 100.0


def _same_family(result, family):
    return result.get('model_family') == family


def _best_previous(results, upto_index, family):
    previous = [
        r for r in results[:upto_index]
        if _same_family(r, family) and r.get('decision') in ('keep', 'keep_candidate')]
    if not previous:
        return None
    return max(previous, key=lambda r: _num(r.get('metrics', {}).get('primary_mAP')) if _num(r.get('metrics', {}).get('primary_mAP')) is not None else -1.0)


def _previous_success(results, upto_index, family):
    for result in reversed(results[:upto_index]):
        if _same_family(result, family) and result.get('decision') in ('keep', 'keep_candidate'):
            return result
    return None


def _baseline_for_family(results, family):
    for result in results:
        if _same_family(result, family):
            return result
    return results[0] if results else None


def compute_stage_delta(results):
    if not results:
        return []
    rows = []
    for i, result in enumerate(results):
        family = result.get('model_family')
        baseline = _baseline_for_family(results, family)
        previous = _previous_success(results, i, family)
        best_previous = _best_previous(results, i, family)
        for metric in METRICS:
            current = result.get('metrics', {}).get(metric)
            baseline_value = baseline.get('metrics', {}).get(metric)
            previous_value = previous.get('metrics', {}).get(metric) if previous else None
            best_value = best_previous.get('metrics', {}).get(metric) if best_previous else None
            if metric == 'GFLOPs_delta_percent':
                current = result.get('metrics', {}).get('GFLOPs')
                baseline_value = baseline.get('metrics', {}).get('GFLOPs')
                previous_value = previous.get('metrics', {}).get('GFLOPs') if previous else None
                best_value = best_previous.get('metrics', {}).get('GFLOPs') if best_previous else None
                explicit_percent = result.get('metrics', {}).get('GFLOPs_delta_percent')
                delta_baseline = explicit_percent if explicit_percent is not None else _percent_delta(current, baseline_value)
                current_value = delta_baseline
                delta_previous = _percent_delta(current, previous_value)
                delta_best = _percent_delta(current, best_value)
            else:
                current_value = current
                delta_baseline = _delta(current, baseline_value)
                delta_previous = _delta(current, previous_value)
                delta_best = _delta(current, best_value)
            rows.append({
                'stage_id': result.get('stage_id'),
                'stage_name': result.get('stage_name'),
                'model_family': result.get('model_family'),
                'decision': result.get('decision'),
                'metric': metric,
                'current': current_value,
                'baseline': baseline_value,
                'previous_success': previous_value,
                'best_previous': best_value,
                'delta_vs_baseline': delta_baseline,
                'delta_vs_previous_success': delta_previous,
                'delta_vs_best_previous': delta_best,
            })
    return rows


def load_results(paths):
    results = []
    for path in paths:
        path = Path(path)
        if path.is_dir():
            results.append(collect_stage_result(path))
        elif path.suffix.lower() in ('.yaml', '.yml'):
            results.append(yaml.safe_load(path.read_text(encoding='utf-8-sig')) or {})
        else:
            results.append(json.loads(path.read_text(encoding='utf-8-sig')))
    return results


def write_delta_csv(rows, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'stage_id', 'stage_name', 'model_family', 'decision', 'metric',
        'current', 'baseline', 'previous_success', 'best_previous',
        'delta_vs_baseline', 'delta_vs_previous_success', 'delta_vs_best_previous']
    with output.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(opt):
    results = load_results(opt.inputs)
    rows = compute_stage_delta(results)
    write_delta_csv(rows, opt.output)
    if opt.json_output:
        Path(opt.json_output).write_text(json.dumps(rows, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'metrics delta saved to {opt.output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', nargs='+', required=True, help='stage dirs or collected result json/yaml files')
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--json-output', type=str, default='')
    main(parser.parse_args())
