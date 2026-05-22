import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.collect_stage_results import collect_stage_result
from tools.compare_stage_metrics import METRICS, compute_stage_delta, write_delta_csv


def _fmt(value):
    if value is None:
        return ''
    if isinstance(value, float):
        return f'{value:.6g}'
    return str(value)


def _delta_map(delta_rows, stage_id, model_family):
    grouped = {}
    for row in delta_rows:
        if row.get('stage_id') == stage_id and row.get('model_family') == model_family:
            grouped[row['metric']] = row
    return grouped


class TrainingReportWriter:
    def __init__(self, sequence_dir):
        self.sequence_dir = Path(sequence_dir)
        self.final_dir = self.sequence_dir / 'final_report'
        self.final_dir.mkdir(parents=True, exist_ok=True)

    def write_stage_summary(self, stage_config, stage_result, delta_rows=None):
        stage_dir = Path(stage_config.get('output_dir') or stage_result.get('stage_dir') or self.sequence_dir)
        stage_dir.mkdir(parents=True, exist_ok=True)
        metrics = stage_result.get('metrics', {})
        delta = _delta_map(delta_rows or [], stage_result.get('stage_id'), stage_result.get('model_family'))
        lines = [
            f"# Stage {stage_result.get('stage_id')} Summary - {stage_config.get('stage_name', stage_result.get('stage_name', ''))}",
            '',
            '## Decision',
            '',
            f"- decision: {stage_result.get('decision', '')}",
            f"- reason: {stage_result.get('reason', '')}",
            f"- train_type: {stage_result.get('train_type', stage_config.get('train_type', ''))}",
            f"- start_weight: {stage_config.get('start_weight', '')}",
            f"- best_weight: {stage_result.get('best_weight', '')}",
            f"- failed_flag: {stage_result.get('failed_category', '')}",
            f"- fallback_weight: {stage_result.get('fallback_weight', '')}",
            '',
            '## Metric Delta',
            '',
            '| Metric | Baseline | Previous | Current | Delta vs Baseline | Delta vs Previous |',
            '| --- | ---: | ---: | ---: | ---: | ---: |',
        ]
        for metric in METRICS:
            row = delta.get(metric, {})
            lines.append(
                f"| {metric} | {_fmt(row.get('baseline'))} | {_fmt(row.get('previous_success'))} | "
                f"{_fmt(metrics.get(metric))} | {_fmt(row.get('delta_vs_baseline'))} | "
                f"{_fmt(row.get('delta_vs_previous_success'))} |")
        lines.extend([
            '',
            '## What Changed',
            '',
            f"- increased: {stage_result.get('increased', '')}",
            f"- decreased: {stage_result.get('decreased', '')}",
            f"- cost_increased: {stage_result.get('cost_increased', '')}",
            f"- unchanged: {stage_result.get('unchanged', '')}",
            '',
            '## Risk Check',
            '',
            f"- loss_stability: {stage_result.get('loss_stability', '')}",
            f"- export_status: {metrics.get('export_status', stage_result.get('export_status', ''))}",
            f"- label_status: {stage_result.get('label_status', '')}",
            f"- per_class_regression: {stage_result.get('per_class_regression', '')}",
            f"- runtime_cost: {stage_result.get('runtime_cost', '')}",
            '',
            '## Next Action',
            '',
            f"- next_stage: {stage_result.get('next_stage', '')}",
            f"- carry_flags: {stage_result.get('carry_flags', {})}",
            f"- disabled_flags: {stage_result.get('disabled_flags', {})}",
            f"- notes: {stage_result.get('notes', '')}",
            '',
        ])
        output = stage_dir / 'stage_summary.md'
        output.write_text('\n'.join(lines), encoding='utf-8')
        return output

    def write_stage_delta_csv(self, stage_result, delta_rows):
        stage_dir = Path(stage_result.get('stage_dir') or stage_result.get('stage_config', {}).get('output_dir') or self.sequence_dir)
        stage_dir.mkdir(parents=True, exist_ok=True)
        output = stage_dir / 'metrics_delta.csv'
        rows = [
            row for row in delta_rows
            if row.get('stage_id') == stage_result.get('stage_id')
            and row.get('model_family') == stage_result.get('model_family')
        ]
        if not rows:
            rows = []
        fieldnames = ['stage_id', 'model_family', 'metric', 'baseline', 'previous_success',
                      'best_previous', 'current', 'delta_vs_baseline',
                      'delta_vs_previous_success', 'delta_vs_best_previous']
        with output.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k) for k in fieldnames})
        return output

    def write_decision_table_csv(self, results):
        output = self.final_dir / 'decision_table.csv'
        fieldnames = [
            'stage', 'model', 'enabled_flags', 'decision', 'reason',
            'primary_mAP_delta', 'small_AP_delta', 'rare_recall_delta',
            'FP_delta', 'FN_delta', 'GFLOPs_delta_percent', 'NMS_delta_ms',
            'export_status', 'risk_level', 'next_action']
        delta_rows = compute_stage_delta(results)
        with output.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                delta = _delta_map(delta_rows, result.get('stage_id'), result.get('model_family'))
                metrics = result.get('metrics', {})
                risk = 'blocker' if result.get('hard_fail') else ('high' if result.get('decision') in ('drop', 'retry_tune') else 'low')
                writer.writerow({
                    'stage': result.get('stage_id'),
                    'model': result.get('model_family'),
                    'enabled_flags': result.get('stage_config', {}).get('enabled_flags', {}),
                    'decision': result.get('decision'),
                    'reason': result.get('reason'),
                    'primary_mAP_delta': delta.get('primary_mAP', {}).get('delta_vs_baseline'),
                    'small_AP_delta': delta.get('small_AP', {}).get('delta_vs_baseline'),
                    'rare_recall_delta': delta.get('rare_recall', {}).get('delta_vs_baseline'),
                    'FP_delta': delta.get('FP_per_image', {}).get('delta_vs_baseline'),
                    'FN_delta': delta.get('FN_per_image', {}).get('delta_vs_baseline'),
                    'GFLOPs_delta_percent': delta.get('GFLOPs_delta_percent', {}).get('delta_vs_baseline'),
                    'NMS_delta_ms': delta.get('python_nms_ms', {}).get('delta_vs_baseline'),
                    'export_status': metrics.get('export_status', result.get('export_status')),
                    'risk_level': risk,
                    'next_action': result.get('next_action', ''),
                })
        return output

    def write_metrics_delta_csv(self, results):
        output = self.final_dir / 'metrics_delta_all.csv'
        write_delta_csv(compute_stage_delta(results), output)
        return output

    def write_sequence_summary(self, results):
        output = self.final_dir / 'sequence_summary.md'
        lines = [
            '# Training Sequence Summary',
            '',
            f'- sequence_dir: {self.sequence_dir}',
            f'- stages: {len(results)}',
            '',
            '| Stage | Model | Decision | primary_mAP | mAP50 | GFLOPs | Export | Reason |',
            '| --- | --- | --- | ---: | ---: | ---: | --- | --- |',
        ]
        for result in results:
            metrics = result.get('metrics', {})
            lines.append(
                f"| {result.get('stage_id')} {result.get('stage_name', '')} | {result.get('model_family', '')} | "
                f"{result.get('decision', '')} | {_fmt(metrics.get('primary_mAP'))} | "
                f"{_fmt(metrics.get('mAP50'))} | {_fmt(metrics.get('GFLOPs'))} | "
                f"{metrics.get('export_status', result.get('export_status', ''))} | {result.get('reason', '')} |")
        output.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        return output

    def write_train_type_summaries(self, results, delta_rows=None):
        delta_rows = delta_rows or compute_stage_delta(results)
        grouped = {}
        for result in results:
            train_type = result.get('train_type') or result.get('stage_config', {}).get('train_type') or 'unknown'
            grouped.setdefault(train_type, []).append(result)
        outputs = []
        for train_type, items in sorted(grouped.items()):
            output = self.final_dir / f'{train_type}_summary.md'
            lines = [
                f"# Train Type Summary - {train_type}",
                '',
                f'- stages: {len(items)}',
                '',
                '| Stage | Model | Decision | primary_mAP | mAP50 | GFLOPs | Risk | Reason |',
                '| --- | --- | --- | ---: | ---: | ---: | --- | --- |',
            ]
            for result in items:
                metrics = result.get('metrics', {})
                risk = 'blocker' if result.get('hard_fail') else ('high' if result.get('decision') in ('drop', 'retry_tune') else 'low')
                lines.append(
                    f"| {result.get('stage_id')} {result.get('stage_name', '')} | {result.get('model_family', '')} | "
                    f"{result.get('decision', '')} | {_fmt(metrics.get('primary_mAP'))} | "
                    f"{_fmt(metrics.get('mAP50'))} | {_fmt(metrics.get('GFLOPs'))} | {risk} | {result.get('reason', '')} |")
            lines.extend([
                '',
                '## Delta Notes',
                '',
            ])
            for result in items:
                delta = _delta_map(delta_rows, result.get('stage_id'), result.get('model_family'))
                primary = delta.get('primary_mAP', {})
                gfops = delta.get('GFLOPs_delta_percent', {})
                lines.append(
                    f"- {result.get('stage_id')} {result.get('model_family', '')}: "
                    f"primary_mAP_delta={_fmt(primary.get('delta_vs_baseline'))}, "
                    f"GFLOPs_delta_percent={_fmt(gfops.get('delta_vs_baseline'))}")
            lines.extend([
                '',
                '## Next Action',
                '',
                '- COCO128 quick이면 산출물, crash, stage 전환, report 생성 여부만 판단한다.',
                '- target full이면 baseline 대비 성능과 비용을 같이 판단한다.',
                '',
            ])
            output.write_text('\n'.join(lines), encoding='utf-8')
            outputs.append(output)
        return outputs

    def write_final_report(self, results, output_path):
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        keep = [r for r in results if r.get('decision') in ('keep', 'keep_candidate')]
        drop = [r for r in results if r.get('decision') == 'drop']
        retry = [r for r in results if r.get('decision') == 'retry_tune']
        blockers = [r for r in results if r.get('decision') == 'blocker']
        lines = [
            f"# Final Training Report v1.8 - {date.today().isoformat()}",
            '',
            '## 실행 요약',
            '',
            f"- sequence_dir: {self.sequence_dir}",
            f"- total_stages: {len(results)}",
            f"- blockers: {len(blockers)}",
            '',
            '## Stage별 결정표',
            '',
            '| Stage | Model | Decision | Reason |',
            '| --- | --- | --- | --- |',
        ]
        for result in results:
            lines.append(f"| {result.get('stage_id')} {result.get('stage_name', '')} | {result.get('model_family', '')} | {result.get('decision', '')} | {result.get('reason', '')} |")
        lines.extend([
            '',
            '## 유지',
            '',
            *(f"- {r.get('stage_id')} {r.get('stage_name')}: {r.get('reason', '')}" for r in keep),
            '',
            '## 제거',
            '',
            *(f"- {r.get('stage_id')} {r.get('stage_name')}: {r.get('reason', '')}" for r in drop),
            '',
            '## 재실험',
            '',
            *(f"- {r.get('stage_id')} {r.get('stage_name')}: {r.get('reason', '')}" for r in retry),
            '',
            '## 원인',
            '',
            *(f"- {r.get('stage_id')} {r.get('stage_name')}: {r.get('failed_category') or 'none'}" for r in blockers + drop + retry),
            '',
            '## 다음 액션',
            '',
            '- COCO128 quick 결과는 orchestration, 산출물, hard fail 판정 확인에만 사용한다.',
            '- target full run 결과에서 최종 유지/제거/재실험 판단을 확정한다.',
            '- blocker stage가 있으면 해당 stage의 stderr와 stage_result.yaml을 먼저 확인한다.',
            '',
        ])
        output.write_text('\n'.join(lines), encoding='utf-8')
        return output


def collect_sequence_results(sequence_dir):
    sequence_dir = Path(sequence_dir)
    results = []
    for result_path in sorted(sequence_dir.glob('*/stage_result.yaml')):
        if result_path.parent.name == 'final_report':
            continue
        results.append(collect_stage_result(result_path.parent))
    return results


def main(opt):
    results = collect_sequence_results(opt.sequence_dir)
    writer = TrainingReportWriter(opt.sequence_dir)
    delta_rows = compute_stage_delta(results)
    for result in results:
        writer.write_stage_summary(result.get('stage_config', {}), result, delta_rows)
        writer.write_stage_delta_csv(result, delta_rows)
    writer.write_metrics_delta_csv(results)
    writer.write_decision_table_csv(results)
    writer.write_sequence_summary(results)
    writer.write_train_type_summaries(results, delta_rows)
    report_output = opt.output or f'doc/REPORT/final_training_report_v1.8_{date.today().isoformat()}.md'
    writer.write_final_report(results, report_output)
    print(f'training report saved to {report_output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence-dir', type=str, required=True)
    parser.add_argument('--output', type=str, default='')
    main(parser.parse_args())
