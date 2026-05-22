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

    def write_stage_result(self, **kwargs):
        data = {'stage': kwargs.pop('stage', '1.3.2')}
        data.update(kwargs)
        if 'mAP_0_5' in data:
            data['mAP_0.5'] = data.pop('mAP_0_5')
        self.stage_result.write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')
