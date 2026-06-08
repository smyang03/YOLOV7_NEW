import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


FREEZE_POLICY_LAYERS = {
    'none': [0],
    'backbone': [50],
    'partial': [75],
    'neck_lower': [75],
}


@dataclass
class ExperimentSpec:
    key: str
    replay_ratio: float
    bn_policy: str = 'train'
    freeze_policy: str = 'none'
    distill_alpha: str = '0.0'
    distill_beta: str = '0.0'
    distill_conf_thres: float = 0.5
    sub_stage: str = '1.3.7-E1'
    pseudo_old_labels: bool = False
    note: str = ''


CORE_EXPERIMENTS = [
    ExperimentSpec(
        'e00_no_replay_short',
        0.0,
        note='Plain short fine-tune baseline. Confirms the forgetting level without retention controls.'),
    ExperimentSpec(
        'e01_no_replay_bn_eval',
        0.0,
        bn_policy='eval',
        note='Freeze BN statistics to reduce distribution drift.'),
    ExperimentSpec(
        'e02_no_replay_freeze_neck',
        0.0,
        bn_policy='eval',
        freeze_policy='neck_lower',
        note='Limit trainable layers when the new data is small or narrow.'),
    ExperimentSpec(
        'e10_replay_005_bn_eval',
        0.05,
        bn_policy='eval',
        note='Very small replay to keep old distribution visible without dominating tuning data.'),
    ExperimentSpec(
        'e11_replay_010_bn_eval',
        0.10,
        bn_policy='eval',
        note='Primary retention candidate: small replay plus frozen BN.'),
    ExperimentSpec(
        'e12_replay_030_bn_eval',
        0.30,
        bn_policy='eval',
        note='Original design replay level, checked after smaller replay ratios.'),
    ExperimentSpec(
        'e20_replay_010_cls_distill',
        0.10,
        bn_policy='eval',
        distill_alpha='0.2:0.5',
        sub_stage='1.3.7-E2',
        note='Replay plus class/objectness distillation from the base model.'),
    ExperimentSpec(
        'e21_replay_030_cls_distill',
        0.30,
        bn_policy='eval',
        distill_alpha='0.2:0.5',
        sub_stage='1.3.7-E2',
        note='Higher replay with class/objectness distillation.'),
    ExperimentSpec(
        'e30_replay_010_cls_reg_light',
        0.10,
        bn_policy='eval',
        distill_alpha='0.2:0.5',
        distill_beta='0.05:0.15',
        distill_conf_thres=0.6,
        sub_stage='1.3.7-E3',
        note='Light regression distillation only on high-confidence teacher boxes.'),
    ExperimentSpec(
        'e31_replay_010_freeze_neck_cls',
        0.10,
        bn_policy='eval',
        freeze_policy='neck_lower',
        distill_alpha='0.2:0.5',
        sub_stage='1.3.7-E2',
        note='Most conservative retention candidate: replay, BN eval, limited trainable layers, cls distill.'),
]


FULL_EXTRA_EXPERIMENTS = [
    ExperimentSpec(
        'e03_no_replay_freeze_backbone',
        0.0,
        bn_policy='eval',
        freeze_policy='backbone',
        note='Checks whether head/neck-only tuning is enough without replay.'),
    ExperimentSpec(
        'e13_replay_020_bn_eval',
        0.20,
        bn_policy='eval',
        note='Middle replay ratio between 0.10 and 0.30.'),
    ExperimentSpec(
        'e14_replay_010_bn_train',
        0.10,
        bn_policy='train',
        note='Separates replay benefit from BN freezing benefit.'),
    ExperimentSpec(
        'e15_replay_010_freeze_neck',
        0.10,
        bn_policy='eval',
        freeze_policy='neck_lower',
        note='Replay and limited trainable layers without distillation.'),
    ExperimentSpec(
        'e22_replay_010_cls_stronger',
        0.10,
        bn_policy='eval',
        distill_alpha='0.5:0.8',
        sub_stage='1.3.7-E2',
        note='Stronger class/objectness retention pressure.'),
    ExperimentSpec(
        'e23_replay_005_cls_distill',
        0.05,
        bn_policy='eval',
        distill_alpha='0.2:0.5',
        sub_stage='1.3.7-E2',
        note='Lower replay cost with distillation.'),
    ExperimentSpec(
        'e32_replay_030_cls_reg_light',
        0.30,
        bn_policy='eval',
        distill_alpha='0.2:0.5',
        distill_beta='0.05:0.15',
        distill_conf_thres=0.6,
        sub_stage='1.3.7-E3',
        note='Higher replay plus light regression distillation.'),
    ExperimentSpec(
        'e33_replay_010_cls_reg_default',
        0.10,
        bn_policy='eval',
        distill_alpha='0.2:0.5',
        distill_beta='0.1:0.3',
        distill_conf_thres=0.5,
        sub_stage='1.3.7-E3',
        note='Documented default cls/reg distillation weights.'),
]


PSEUDO_EXPERIMENTS = [
    ExperimentSpec(
        'p00_pseudo_only',
        0.0,
        bn_policy='eval',
        pseudo_old_labels=True,
        note='Teacher pseudo old-label completion without replay.'),
    ExperimentSpec(
        'p01_pseudo_replay005',
        0.05,
        bn_policy='eval',
        pseudo_old_labels=True,
        note='Pseudo old-label completion with very small replay.'),
    ExperimentSpec(
        'p02_pseudo_replay010',
        0.10,
        bn_policy='eval',
        pseudo_old_labels=True,
        note='Primary pseudo+replay candidate.'),
    ExperimentSpec(
        'p03_pseudo_replay010_cls',
        0.10,
        bn_policy='eval',
        distill_alpha='0.2:0.5',
        sub_stage='1.3.7-E2',
        pseudo_old_labels=True,
        note='Pseudo+replay plus class/objectness distillation.'),
    ExperimentSpec(
        'p04_pseudo_replay010_freeze_cls',
        0.10,
        bn_policy='eval',
        freeze_policy='neck_lower',
        distill_alpha='0.2:0.5',
        sub_stage='1.3.7-E2',
        pseudo_old_labels=True,
        note='Conservative pseudo+replay+freeze+class distillation candidate.'),
    ExperimentSpec(
        'p05_pseudo_replay030',
        0.30,
        bn_policy='eval',
        pseudo_old_labels=True,
        note='Pseudo old-label completion with original replay level.'),
]


def experiments_for_preset(preset):
    if preset == 'core':
        return list(CORE_EXPERIMENTS)
    if preset == 'full':
        return list(CORE_EXPERIMENTS) + list(FULL_EXTRA_EXPERIMENTS)
    if preset == 'pseudo':
        return list(PSEUDO_EXPERIMENTS)
    if preset == 'core_pseudo':
        return list(CORE_EXPERIMENTS) + list(PSEUDO_EXPERIMENTS)
    raise ValueError(f'unsupported preset: {preset}')


def is_distill_active(spec):
    return max(float(x) for x in str(spec.distill_alpha).split(':')) > 0 or \
        max(float(x) for x in str(spec.distill_beta).split(':')) > 0


def unique_name(project, name, allow_existing=False):
    project = Path(project)
    if allow_existing:
        return name
    candidate = name
    index = 2
    while (project / candidate).exists():
        candidate = f'{name}_{index}'
        index += 1
    return candidate


def command_text(command):
    return shlex.join([str(x) for x in command])


def run_command(command, execute=True):
    print(command_text(command))
    if not execute:
        return 0
    completed = subprocess.run([str(x) for x in command], cwd=str(ROOT), check=False)
    return completed.returncode


def write_run_status(sweep_dir, statuses):
    path = Path(sweep_dir) / 'sweep_run_status.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        'schema_version': '1.3.7-sweep-status',
        'statuses': statuses,
    }, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return path


def finetune_prep_command(opt, spec, name):
    data = pseudo_data_yaml(opt, spec, name) if spec.pseudo_old_labels else opt.data
    command = [
        sys.executable, str(ROOT / 'finetune.py'),
        '--weights', opt.weights,
        '--base-data', opt.base_data,
        '--data', str(data),
        '--replay-ratio', str(spec.replay_ratio),
        '--replay-ratio-source', opt.replay_ratio_source,
        '--hyp', opt.hyp,
        '--epochs', str(opt.epochs),
        '--batch-size', str(opt.batch_size),
        '--img-size', *[str(x) for x in opt.img_size],
        '--workers', str(opt.workers),
        '--freeze', spec.freeze_policy,
        '--bn-policy', spec.bn_policy,
        '--best-val-set', opt.best_val_set,
        '--distill-alpha', spec.distill_alpha,
        '--distill-beta', spec.distill_beta,
        '--distill-conf-thres', str(spec.distill_conf_thres),
        '--sub-stage', spec.sub_stage,
        '--project', opt.project,
        '--name', name,
        '--exist-ok',
        '--dry-run',
    ]
    if opt.cfg:
        command.extend(['--cfg', opt.cfg])
    if opt.device:
        command.extend(['--device', opt.device])
    if opt.save_best_only:
        command.append('--save-best-only')
    if opt.replay_count >= 0:
        command.extend(['--replay-count', str(opt.replay_count)])
    if is_distill_active(spec):
        command.extend(['--teacher-weights', opt.teacher_weights or opt.weights])
    return command


def pseudo_data_yaml(opt, spec, name):
    return Path(opt.project) / name / 'pseudo_old_labels' / 'pseudo_old_data.yaml'


def pseudo_prep_command(opt, spec, name):
    if not spec.pseudo_old_labels:
        return []
    output = Path(opt.project) / name / 'pseudo_old_labels'
    command = [
        sys.executable, str(ROOT / 'tools' / 'prepare_pseudo_old_labels.py'),
        '--data', opt.data,
        '--output', str(output),
        '--weights', opt.teacher_weights or opt.weights,
        '--conf-thres', str(opt.pseudo_conf_thres),
        '--iou-thres', str(opt.pseudo_iou_thres),
        '--pseudo-conf', str(opt.pseudo_conf),
        '--dedupe-iou', str(opt.pseudo_dedupe_iou),
        '--img-size', str(opt.img_size[0]),
        '--link-mode', opt.pseudo_link_mode,
        '--manifest', str(output / 'pseudo_old_label_manifest.json'),
    ]
    if opt.device:
        command.extend(['--device', opt.device])
    if opt.pseudo_max_images:
        command.extend(['--max-images', str(opt.pseudo_max_images)])
    return command


def launcher_prefix(opt):
    if opt.launcher == 'none':
        return [sys.executable]
    if opt.launcher == 'distributed':
        command = [
            sys.executable, '-m', 'torch.distributed.launch',
            '--nproc_per_node', str(opt.nproc_per_node),
        ]
        if opt.master_port:
            command.extend(['--master_port', str(opt.master_port)])
        return command
    raise ValueError(f'unsupported launcher: {opt.launcher}')


def train_command(opt, spec, name):
    data_yaml = Path(opt.project) / name / 'finetune_data.yaml'
    command = launcher_prefix(opt) + [
        str(ROOT / 'train.py'),
        '--weights', opt.weights,
        '--data', str(data_yaml),
        '--hyp', opt.hyp,
        '--epochs', str(opt.epochs),
        '--batch-size', str(opt.batch_size),
        '--img-size', *[str(x) for x in opt.img_size],
        '--workers', str(opt.workers),
        '--freeze', *[str(x) for x in FREEZE_POLICY_LAYERS[spec.freeze_policy]],
        '--bn-policy', spec.bn_policy,
        '--best-val-set', opt.best_val_set,
        '--project', opt.project,
        '--name', name,
        '--exist-ok',
    ]
    if opt.cfg:
        command.extend(['--cfg', opt.cfg])
    if opt.device:
        command.extend(['--device', opt.device])
    if opt.save_best_only:
        command.append('--save-best-only')
    if is_distill_active(spec):
        command.extend([
            '--teacher-weights', opt.teacher_weights or opt.weights,
            '--distill-alpha', spec.distill_alpha,
            '--distill-beta', spec.distill_beta,
            '--distill-conf-thres', str(spec.distill_conf_thres),
        ])
    return command


def write_sweep_artifacts(opt, rows):
    sweep_dir = Path(opt.project) / f'{opt.prefix}_sweep'
    sweep_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        'schema_version': '1.3.7-sweep',
        'preset': opt.preset,
        'goal': 'improve finetune validation while retaining base validation performance',
        'selection_rule': {
            'best_val_set': opt.best_val_set,
            'prefer': 'highest finetune mAP50-95 subject to base retention threshold',
            'base_retention_threshold_percent': opt.base_retention,
            'finetune_min_delta': opt.finetune_min_delta,
        },
        'common': {
            'weights': opt.weights,
            'teacher_weights': opt.teacher_weights or opt.weights,
            'base_data': opt.base_data,
            'data': opt.data,
            'hyp': opt.hyp,
            'replay_ratio_source': opt.replay_ratio_source,
            'replay_count': opt.replay_count,
            'epochs': opt.epochs,
            'batch_size': opt.batch_size,
            'img_size': opt.img_size,
            'launcher': opt.launcher,
            'nproc_per_node': opt.nproc_per_node,
            'pseudo_conf': opt.pseudo_conf,
            'pseudo_conf_thres': opt.pseudo_conf_thres,
            'pseudo_iou_thres': opt.pseudo_iou_thres,
            'pseudo_dedupe_iou': opt.pseudo_dedupe_iou,
        },
        'experiments': rows,
    }
    (sweep_dir / 'sweep_plan.yaml').write_text(
        yaml.safe_dump(plan, sort_keys=False, allow_unicode=True), encoding='utf-8')
    (sweep_dir / 'sweep_plan.json').write_text(
        json.dumps(plan, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    with (sweep_dir / 'commands.sh').open('w', encoding='utf-8') as f:
        f.write('#!/usr/bin/env bash\nset -euo pipefail\n\n')
        for row in rows:
            if row.get('pseudo_command'):
                f.write(command_text(row['pseudo_command']) + '\n')
            f.write(command_text(row['prep_command']) + '\n')
            f.write(command_text(row['train_command']) + '\n\n')
    return sweep_dir


def summary_command(opt, sweep_dir, rows):
    output = Path(opt.summary_output) if opt.summary_output else Path(sweep_dir) / 'retention_summary.csv'
    json_output = Path(opt.summary_json_output) if opt.summary_json_output else Path(sweep_dir) / 'retention_summary.json'
    command = [
        sys.executable, str(ROOT / 'tools' / 'summarize_finetune_retention.py'),
        '--runs', *[str(row['save_dir']) for row in rows],
        '--output', str(output),
        '--json-output', str(json_output),
        '--select-scenario', opt.summary_select_scenario,
        '--min-base-retention', str(opt.base_retention),
        '--min-finetune-delta', str(opt.finetune_min_delta),
    ]
    if opt.baseline_run:
        command.extend(['--baseline', opt.baseline_run])
    if opt.finetune_scenario:
        command.extend(['--finetune-scenario', opt.finetune_scenario])
    if opt.base_scenario:
        command.extend(['--base-scenario', opt.base_scenario])
    if opt.require_finetune_gain:
        command.append('--require-finetune-gain')
    return command


def main(opt):
    specs = experiments_for_preset(opt.preset)
    rows = []
    project = Path(opt.project)
    for spec in specs:
        name = unique_name(project, f'{opt.prefix}_{spec.key}', opt.allow_existing)
        pseudo = pseudo_prep_command(opt, spec, name)
        prep = finetune_prep_command(opt, spec, name)
        train = train_command(opt, spec, name)
        rows.append({
            **asdict(spec),
            'name': name,
            'save_dir': str(project / name),
            'pseudo_command': pseudo,
            'prep_command': prep,
            'train_command': train,
        })

    sweep_dir = write_sweep_artifacts(opt, rows)
    print(f'sweep plan saved to {sweep_dir}')
    statuses = []

    if opt.print_only:
        for row in rows:
            print(f'\n# {row["name"]}: {row["note"]}')
            if row.get('pseudo_command'):
                print(command_text(row['pseudo_command']))
            print(command_text(row['prep_command']))
            print(command_text(row['train_command']))
        if opt.summarize:
            print('\n# summary')
            print(command_text(summary_command(opt, sweep_dir, rows)))
        return

    for row in rows:
        status = {'name': row['name'], 'save_dir': row['save_dir'], 'status': 'pending'}
        if row.get('pseudo_command'):
            print(f'\n# prepare pseudo old labels {row["name"]}')
            rc = run_command(row['pseudo_command'], execute=True)
            if rc:
                status.update({'status': 'fail', 'step': 'pseudo', 'returncode': rc})
                statuses.append(status)
                write_run_status(sweep_dir, statuses)
                if not opt.continue_on_error:
                    raise SystemExit(rc)
                continue
        print(f'\n# prepare {row["name"]}')
        rc = run_command(row['prep_command'], execute=True)
        if rc:
            status.update({'status': 'fail', 'step': 'finetune_dry_run', 'returncode': rc})
            statuses.append(status)
            write_run_status(sweep_dir, statuses)
            if not opt.continue_on_error:
                raise SystemExit(rc)
            continue
        if opt.execute:
            print(f'\n# train {row["name"]}')
            rc = run_command(row['train_command'], execute=True)
            if rc:
                status.update({'status': 'fail', 'step': 'train', 'returncode': rc})
                statuses.append(status)
                write_run_status(sweep_dir, statuses)
                if not opt.continue_on_error:
                    raise SystemExit(rc)
                continue
            status.update({'status': 'pass', 'step': 'train', 'returncode': 0})
        else:
            status.update({'status': 'prepared', 'step': 'finetune_dry_run', 'returncode': 0})
        statuses.append(status)
        write_run_status(sweep_dir, statuses)

    if opt.summarize:
        print('\n# summarize')
        rc = run_command(summary_command(opt, sweep_dir, rows), execute=True)
        if rc and not opt.continue_on_error:
            raise SystemExit(rc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--teacher-weights', type=str, default='')
    parser.add_argument('--base-data', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--cfg', type=str, default='')
    parser.add_argument('--hyp', type=str, default='data/hyp_finetune.yaml')
    parser.add_argument('--project', type=str, default='runs/finetune')
    parser.add_argument('--prefix', type=str, default=f'retention_{date.today().isoformat()}')
    parser.add_argument('--preset', choices=['core', 'full', 'pseudo', 'core_pseudo'], default='core')
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch-size', '--batch', dest='batch_size', type=int, default=16)
    parser.add_argument('--img-size', '--img', dest='img_size', nargs='+', type=int, default=[640, 640])
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='')
    parser.add_argument('--best-val-set', type=str, default='combined')
    parser.add_argument('--save-best-only', action='store_true')
    parser.add_argument('--launcher', choices=['none', 'distributed'], default='none')
    parser.add_argument('--nproc-per-node', type=int, default=1)
    parser.add_argument('--master-port', type=int, default=9527)
    parser.add_argument('--base-retention', type=float, default=95.0)
    parser.add_argument('--finetune-min-delta', type=float, default=0.0)
    parser.add_argument('--replay-ratio-source', choices=['base', 'finetune'], default='base')
    parser.add_argument('--replay-count', type=int, default=-1)
    parser.add_argument('--pseudo-conf-thres', type=float, default=0.6)
    parser.add_argument('--pseudo-iou-thres', type=float, default=0.45)
    parser.add_argument('--pseudo-conf', type=float, default=0.6)
    parser.add_argument('--pseudo-dedupe-iou', type=float, default=0.8)
    parser.add_argument('--pseudo-max-images', type=int, default=0)
    parser.add_argument('--pseudo-link-mode', choices=['symlink', 'hardlink', 'copy'], default='symlink')
    parser.add_argument('--allow-existing', action='store_true',
                        help='reuse project/prefix experiment names instead of choosing unused names')
    parser.add_argument('--print-only', action='store_true',
                        help='write the sweep plan and print commands without materializing dry-run data')
    parser.add_argument('--execute', action='store_true',
                        help='after dry-run preparation, execute each train command sequentially')
    parser.add_argument('--continue-on-error', action='store_true',
                        help='continue remaining experiments when one experiment fails')
    parser.add_argument('--summarize', action='store_true',
                        help='run summarize_finetune_retention.py after preparation/training')
    parser.add_argument('--baseline-run', type=str, default='',
                        help='baseline run directory or results_detail.txt used for retention calculation')
    parser.add_argument('--finetune-scenario', type=str, default='',
                        help='scenario name for the finetune validation set in results_detail.txt')
    parser.add_argument('--base-scenario', type=str, default='',
                        help='scenario name for the base/original validation set in results_detail.txt')
    parser.add_argument('--summary-select-scenario', type=str, default='combined',
                        help='validation scenario used to choose each run best epoch for summary')
    parser.add_argument('--summary-output', type=str, default='',
                        help='CSV summary path; default is <project>/<prefix>_sweep/retention_summary.csv')
    parser.add_argument('--summary-json-output', type=str, default='',
                        help='JSON summary path; default is <project>/<prefix>_sweep/retention_summary.json')
    parser.add_argument('--require-finetune-gain', action='store_true',
                        help='mark runs as drop when finetune validation does not improve over baseline')
    main(parser.parse_args())
