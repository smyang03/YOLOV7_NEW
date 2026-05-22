import csv
from pathlib import Path

import yaml


RESULTS_HEADER = [
    'epoch', 'phase', 'lr',
    'train/box_loss', 'train/cls_loss', 'train/obj_loss', 'train/aux_loss', 'train/total_loss',
    'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
    'val/box_loss', 'val/cls_loss', 'val/obj_loss',
    'x/lr0', 'x/lr1', 'x/lr2', 'gpu_mem_gb', 'epoch_time_sec',
]

LOSS_HEADER = [
    'epoch', 'phase', 'box_loss', 'cls_loss', 'obj_loss', 'aux_loss',
    'free_loss', 'total_loss', 'lambda_aux', 'lambda_free', 'positive_count',
    'assigner', 'loss_box', 'loss_cls', 'head',
]

PER_CLASS_HEADER = [
    'epoch', 'phase', 'class_id', 'class_name', 'precision', 'recall',
    'AP_0.5', 'AP_0.5:0.95', 'is_rare',
]

SCENARIO_HEADER = [
    'epoch', 'phase', 'scenario', 'AP_0.5', 'recall',
    'false_positive_per_image', 'sample_count',
]


class TrainLogger:
    def __init__(self, save_dir, log_format='both', per_class_log_interval=10):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_format = log_format
        self.per_class_log_interval = max(int(per_class_log_interval), 1)
        self.results_csv = self.save_dir / 'results.csv'
        self.loss_detail_csv = self.save_dir / 'loss_detail.csv'
        self.per_class_csv = self.save_dir / 'results_per_class.csv'
        self.scenario_metrics_csv = self.save_dir / 'scenario_metrics.csv'
        self.phase_log = self.save_dir / 'phase_transition.log'
        self.train_log = self.save_dir / 'train_log.txt'
        self.hyp_used = self.save_dir / 'hyp_used.yaml'
        self.stage_result = self.save_dir / 'stage_result.yaml'
        self.run_summary = self.save_dir / 'run_summary.md'

        if log_format in ('csv', 'both'):
            self._ensure_csv(self.results_csv, RESULTS_HEADER)
            self._ensure_csv(self.loss_detail_csv, LOSS_HEADER)
            self._ensure_csv(self.per_class_csv, PER_CLASS_HEADER)
            self._ensure_csv(self.scenario_metrics_csv, SCENARIO_HEADER)
        self.log_text('train_logger initialized')

    def _ensure_csv(self, path, header):
        if not path.exists():
            with open(path, 'w', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow(header)

    def _append_csv(self, path, row):
        if self.log_format not in ('csv', 'both'):
            return
        with open(path, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(row)

    def log_text(self, message):
        with open(self.train_log, 'a', encoding='utf-8') as f:
            f.write(str(message).rstrip() + '\n')

    def log_phase_transition(self, epoch, from_phase, to_phase, imgsz, rect, mosaic, hyp_path,
                             train_loader_rebuilt, val_loader_rebuilt, persistent_workers, reason):
        line = (
            f'epoch={epoch} from_phase={from_phase} to_phase={to_phase} imgsz={imgsz} '
            f'rect={rect} mosaic={mosaic} hyp_path={hyp_path} '
            f'train_loader_rebuilt={train_loader_rebuilt} val_loader_rebuilt={val_loader_rebuilt} '
            f'persistent_workers={persistent_workers} reason={reason}'
        )
        with open(self.phase_log, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
        self.log_text(line)

    def write_hyp_snapshot(self, epoch, phase, hyp):
        snapshot = {'epoch': int(epoch), 'phase': phase, 'hyp': dict(hyp)}
        data = {'snapshots': []}
        if self.hyp_used.exists():
            data = yaml.safe_load(self.hyp_used.read_text(encoding='utf-8')) or data
        data.setdefault('snapshots', []).append(snapshot)
        self.hyp_used.write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')

    def log_epoch(self, epoch, phase, mloss, results, lr, gpu_mem_gb=0.0, epoch_time_sec=0.0):
        losses = [float(x) for x in list(mloss)]
        while len(losses) < 4:
            losses.append(0.0)
        metrics = [float(x) for x in list(results)]
        while len(metrics) < 7:
            metrics.append(0.0)
        lr_values = [float(x) for x in list(lr)]
        while len(lr_values) < 3:
            lr_values.append(0.0)
        self._append_csv(self.results_csv, [
            epoch, phase, lr_values[0],
            losses[0], losses[2], losses[1], 0.0, losses[3],
            metrics[0], metrics[1], metrics[2], metrics[3],
            metrics[4], metrics[6], metrics[5],
            lr_values[0], lr_values[1], lr_values[2], gpu_mem_gb, epoch_time_sec,
        ])

    def log_loss_detail(self, epoch, phase, mloss, lambda_aux=0.0, lambda_free=0.0, positive_count=0,
                        assigner='simota', loss_box='ciou', loss_cls='bce', head='coupled'):
        losses = [float(x) for x in list(mloss)]
        while len(losses) < 4:
            losses.append(0.0)
        self._append_csv(self.loss_detail_csv, [
            epoch, phase, losses[0], losses[2], losses[1], 0.0, 0.0, losses[3],
            lambda_aux, lambda_free, positive_count, assigner, loss_box, loss_cls, head,
        ])

    def log_per_class(self, epoch, phase, per_class, is_best=False):
        if per_class is None:
            return
        if not is_best and (epoch + 1) % self.per_class_log_interval != 0:
            return
        names = per_class['names']
        for i, class_id in enumerate(per_class['ap_class']):
            nt = int(per_class['nt'][class_id])
            self._append_csv(self.per_class_csv, [
                epoch, phase, int(class_id), names[class_id],
                float(per_class['p'][i]), float(per_class['r'][i]),
                float(per_class['ap50'][i]), float(per_class['ap'][i]),
                nt < 10,
            ])

    def log_scenario_metrics(self, epoch, phase, val_results):
        if not val_results:
            return
        for val_result in val_results:
            results = list(val_result.get('results') or [])
            while len(results) < 4:
                results.append(0.0)
            self._append_csv(self.scenario_metrics_csv, [
                epoch,
                phase,
                val_result.get('name', 'val'),
                float(results[2]),
                float(results[1]),
                '',
                val_result.get('sample_count', ''),
            ])

    def _train_type(self, stage_id, data):
        if stage_id == '1.3.1':
            return 'baseline_export'
        if stage_id == '1.3.2':
            return 'phase_training'
        if stage_id == '1.3.3':
            return 'core_loss_model'
        if stage_id == '1.3.4':
            return 'augmentation_data'
        if stage_id == '1.3.5':
            return 'model_family_export'
        if stage_id == '1.3.6':
            return 'optional_gate'
        if stage_id == '1.3.7':
            return 'finetune_distill'
        if data.get('optional_decision'):
            return 'optional_gate'
        if data.get('p2_head') not in (None, '', 'none') or data.get('neck_mod') not in (None, '', 'none'):
            return 'model_family_export'
        return 'sequence_report'

    def _stage_name(self, stage_id, train_type):
        names = {
            '1.3.1': 'baseline_export',
            '1.3.2': 'phase_training',
            '1.3.3': 'core_loss_model',
            '1.3.4': 'augmentation_data',
            '1.3.5': 'model_family_export',
            '1.3.6': 'optional_gate',
            '1.3.7': 'finetune_distill',
            '1.3.8': 'sequence_report',
        }
        return names.get(stage_id, train_type or 'training')

    def _artifact_status(self, path):
        path = Path(path)
        return 'ok' if path.is_file() else 'missing'

    def _path_name(self, path):
        path = Path(path)
        try:
            return str(path.relative_to(self.save_dir))
        except ValueError:
            return str(path)

    def write_stage_result(self, **kwargs):
        legacy_stage = str(kwargs.pop('stage', kwargs.get('stage_id', '1.3.2')))
        data = dict(kwargs)
        if 'mAP_0_5' in data:
            data['mAP_0.5'] = data.pop('mAP_0_5')

        stage_id = str(data.pop('stage_id', legacy_stage))
        train_type = str(data.pop('train_type', self._train_type(stage_id, data)))
        stage_name = str(data.pop('stage_name', self._stage_name(stage_id, train_type)))
        status = str(data.get('status', 'completed'))
        hard_fail = bool(data.pop('hard_fail', status not in ('completed', 'ok', 'success')))
        soft_fail = bool(data.pop('soft_fail', False))
        decision = str(data.pop('decision', 'blocker' if hard_fail else 'keep'))
        reason = str(data.pop('reason', 'training failed' if hard_fail else 'training completed'))
        failed_category = data.pop('failed_category', None)

        mAP50 = data.get('mAP50', data.get('mAP_0.5'))
        metrics = dict(data.pop('metrics', {}) or {})
        metrics.setdefault('primary_mAP', data.get('primary_mAP', data.get('best_map_50_95')))
        metrics.setdefault('mAP50', mAP50)
        metrics.setdefault('small_AP', data.get('small_AP'))
        metrics.setdefault('rare_recall', data.get('rare_recall'))
        metrics.setdefault('GFLOPs', data.get('current_gflops', data.get('GFLOPs')))
        metrics.setdefault('GFLOPs_delta_percent', data.get('gflops_delta_percent'))
        metrics.setdefault('export_status', data.get('export_status', 'skip'))

        artifacts = dict(data.pop('artifacts', {}) or {})
        artifacts.setdefault('best_pt', self._path_name(self.save_dir / 'weights' / 'best.pt'))
        artifacts.setdefault('last_pt', self._path_name(self.save_dir / 'weights' / 'last.pt'))
        artifacts.setdefault('results_csv', self._path_name(self.results_csv))
        artifacts.setdefault('loss_detail_csv', self._path_name(self.loss_detail_csv))
        artifacts.setdefault('stage_result', self._path_name(self.stage_result))
        artifacts.setdefault('run_summary', self._path_name(self.run_summary))

        log_paths = dict(data.pop('log_paths', {}) or {})
        log_paths.setdefault('train_log', self._path_name(self.train_log))
        log_paths.setdefault('phase_transition', self._path_name(self.phase_log))
        log_paths.setdefault('debug_trace', 'debug_trace.log')
        log_paths.setdefault('error_trace', 'error_trace.log')

        result = {
            'schema_version': '1.3',
            'stage': legacy_stage,
            'stage_id': stage_id,
            'stage_name': stage_name,
            'train_type': train_type,
            'decision': decision,
            'reason': reason,
            'status': status,
            'hard_fail': hard_fail,
            'soft_fail': soft_fail,
            'failed_category': failed_category,
            'metrics': metrics,
            'artifacts': artifacts,
            'log_paths': log_paths,
        }
        result.update(data)
        self.stage_result.write_text(yaml.safe_dump(result, sort_keys=False, allow_unicode=True), encoding='utf-8')
        return result

    def write_run_summary(self, stage_result=None):
        if stage_result is None:
            stage_result = yaml.safe_load(self.stage_result.read_text(encoding='utf-8')) if self.stage_result.exists() else {}
        metrics = stage_result.get('metrics', {}) or {}
        artifacts = stage_result.get('artifacts', {}) or {}
        checks = [
            ('best.pt', self.save_dir / 'weights' / 'best.pt', artifacts.get('best_pt', 'weights/best.pt')),
            ('last.pt', self.save_dir / 'weights' / 'last.pt', artifacts.get('last_pt', 'weights/last.pt')),
            ('results.csv', self.results_csv, artifacts.get('results_csv', 'results.csv')),
            ('loss_detail.csv', self.loss_detail_csv, artifacts.get('loss_detail_csv', 'loss_detail.csv')),
            ('stage_result.yaml', self.stage_result, artifacts.get('stage_result', 'stage_result.yaml')),
        ]
        lines = [
            f"# Run Summary - {stage_result.get('train_type', '')}",
            '',
            '## Decision',
            '',
            f"- decision: {stage_result.get('decision', '')}",
            f"- reason: {stage_result.get('reason', '')}",
            f"- stage_id: {stage_result.get('stage_id', '')}",
            f"- stage_name: {stage_result.get('stage_name', '')}",
            f"- train_type: {stage_result.get('train_type', '')}",
            f"- current_run: {stage_result.get('current_run', str(self.save_dir))}",
            '',
            '## Artifact Check',
            '',
            '| Artifact | Status | Path |',
            '| --- | --- | --- |',
        ]
        for name, path, display in checks:
            lines.append(f"| {name} | {self._artifact_status(path)} | {display} |")
        lines.extend([
            '',
            '## Metric Summary',
            '',
            '| Metric | Value |',
            '| --- | ---: |',
            f"| primary_mAP | {metrics.get('primary_mAP', '')} |",
            f"| mAP50 | {metrics.get('mAP50', '')} |",
            f"| precision | {metrics.get('precision', '')} |",
            f"| recall | {metrics.get('recall', '')} |",
            f"| GFLOPs | {metrics.get('GFLOPs', '')} |",
            f"| GFLOPs_delta_percent | {metrics.get('GFLOPs_delta_percent', '')} |",
            '',
            '## Stability And Risk',
            '',
            f"- hard_fail: {stage_result.get('hard_fail', False)}",
            f"- soft_fail: {stage_result.get('soft_fail', False)}",
            f"- failed_category: {stage_result.get('failed_category', '')}",
            f"- export_status: {metrics.get('export_status', stage_result.get('export_status', ''))}",
            '',
            '## Next Action',
            '',
            f"- next_stage: {stage_result.get('next_stage', '')}",
            f"- carry_flags: {stage_result.get('carry_flags', {})}",
            f"- rollback_flags: {stage_result.get('rollback_flags', {})}",
            f"- code_area: {stage_result.get('code_area', '')}",
            '',
        ])
        self.run_summary.write_text('\n'.join(lines), encoding='utf-8')
        return self.run_summary
