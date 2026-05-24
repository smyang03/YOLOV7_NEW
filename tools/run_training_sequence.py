import argparse
import json
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.collect_stage_results import collect_stage_result
from tools.compare_stage_metrics import compute_stage_delta, write_delta_csv
from utils.debug_logging import get_debug_logger
from tools.generate_training_report import TrainingReportWriter
from utils.stage_schema import Decision, StageConfig, StageResult


@dataclass
class StageSpec:
    stage_id: str
    stage_name: str
    enabled_flags: Dict[str, Any] = field(default_factory=dict)
    disabled_flags: Dict[str, Any] = field(default_factory=dict)
    families: str = 'all'
    defer: bool = False
    train_type: str = ''


STAGES = [
    StageSpec('00', 'baseline', train_type='baseline_export'),
    StageSpec('01', 'phase_logging', {'phase-train': 'on'}, train_type='phase_training'),
    StageSpec('02', 'head_decoupled', {'head': 'decoupled'}, train_type='core_loss_model'),
    StageSpec('03', 'wiou_v3', {'loss-box': 'wiou_v3'}, train_type='core_loss_model'),
    StageSpec('04', 'tal_vfl', {'assign': 'tal', 'loss-cls': 'vfl'}, train_type='core_loss_model'),
    StageSpec('05', 'core_cumulative', {'head': 'decoupled', 'loss-box': 'wiou_v3', 'assign': 'tal', 'loss-cls': 'vfl'}, train_type='core_loss_model'),
    StageSpec('06', 'cctv_pixel_aug', {'aug-profile': 'cctv_pixel'}, train_type='augmentation_data'),
    StageSpec('07', 'patch_paste_hard_negative', {'aug-profile': 'cctv_paste'}, train_type='augmentation_data'),
    StageSpec('08', 'weighted_sampler', {'sampler-mode': 'weighted'}, train_type='augmentation_data'),
    StageSpec('09', 'w6_scdown', {'neck-mod': 'scdown'}, families='w6', train_type='model_family_export'),
    StageSpec('10', 'w6_p2_anchor', {'p2-head': 'anchor'}, families='w6', train_type='model_family_export'),
    StageSpec('11', 'w6_p2_anchor_scdown', {'p2-head': 'anchor', 'neck-mod': 'scdown'}, families='w6', train_type='model_family_export'),
    StageSpec('12', 'optional_gate', {}, defer=True, train_type='optional_gate'),
    StageSpec('13', 'finetune_continual', {}, defer=True, train_type='finetune_distill'),
]


def parse_families(value):
    value = value.strip()
    if value == 'l,w6':
        return ['l', 'w6']
    if value == 'l_only':
        return ['l']
    if value == 'w6_only':
        return ['w6']
    families = [x.strip() for x in value.split(',') if x.strip()]
    if not families:
        raise ValueError('--model-family is empty')
    for family in families:
        if family not in ('l', 'w6'):
            raise ValueError(f'unsupported model family: {family}')
    return families


def stage_in_range(spec, start_stage='', end_stage=''):
    if start_stage and spec.stage_id < start_stage:
        return False
    if end_stage and spec.stage_id > end_stage:
        return False
    return True


def stage_for_family(spec, family):
    return spec.families == 'all' or spec.families == family


def quick_stage_overrides(spec, opt):
    flags = dict(spec.enabled_flags)
    epochs = opt.epochs
    if opt.dataset_profile == 'coco128_quick' and spec.stage_id == '01':
        phase_img = opt.img if len(opt.img) == 2 else [opt.img[0], opt.img[0]]
        if getattr(opt, 'phase1_epochs', None) is None:
            flags.setdefault('phase1-epochs', 1)
        if getattr(opt, 'phase2_epochs', None) is None:
            flags.setdefault('phase2-epochs', 1)
        if getattr(opt, 'phase3_epochs', None) is None:
            flags.setdefault('phase3-epochs', 1)
        flags.setdefault('phase2-img', phase_img)
        flags.setdefault('phase3-img', phase_img)
        epochs = max(epochs, 3)
    return flags, epochs


def flag_to_args(key, value):
    if value is None or value is False:
        return []
    flag = f'--{key}'
    if value is True:
        return [flag]
    if isinstance(value, (list, tuple)):
        return [flag] + [str(x) for x in value]
    return [flag, str(value)]


def family_weight(opt, family):
    if family == 'l':
        return opt.l_weights or opt.weights
    return opt.w6_weights or opt.weights


def family_cfg(opt, family):
    if family == 'l':
        return opt.l_cfg or opt.cfg
    return opt.w6_cfg or opt.cfg


def launcher_prefix(opt):
    if getattr(opt, 'launcher', 'none') == 'none':
        return [sys.executable]
    if opt.launcher == 'distributed':
        prefix = [
            sys.executable, '-m', 'torch.distributed.launch',
            '--nproc_per_node', str(opt.nproc_per_node),
        ]
        if opt.nnodes != 1:
            prefix.extend(['--nnodes', str(opt.nnodes)])
            prefix.extend(['--node_rank', str(opt.node_rank)])
            prefix.extend(['--master_addr', opt.master_addr])
        if opt.master_port:
            prefix.extend(['--master_port', str(opt.master_port)])
        return prefix
    raise ValueError(f'unsupported launcher: {opt.launcher}')


def build_train_command(opt, config, family):
    command = launcher_prefix(opt) + [
        str(ROOT / 'train.py'),
        '--data', config.data,
        '--project', str(Path(config.output_dir).parent),
        '--name', Path(config.output_dir).name,
        '--exist-ok',
        '--epochs', str(config.epochs),
        '--batch-size', str(opt.batch_size),
        '--img-size', *[str(x) for x in opt.img],
    ]
    if config.start_weight:
        command.extend(['--weights', config.start_weight])
    elif family_cfg(opt, family):
        command.extend(['--weights', '', '--cfg', family_cfg(opt, family)])
    elif opt.dry_run:
        command.extend(['--weights', f'<{family}_weights_required_for_real_run>'])
    if opt.hyp:
        command.extend(['--hyp', opt.hyp])
    if opt.device:
        command.extend(['--device', opt.device])
    if opt.workers >= 0:
        command.extend(['--workers', str(opt.workers)])
    if getattr(opt, 'sync_bn', False):
        command.append('--sync-bn')
    if getattr(opt, 'image_weights', False):
        command.append('--image-weights')
    if getattr(opt, 'multi_scale', False):
        command.append('--multi-scale')
    if getattr(opt, 'save_best_only', False):
        command.append('--save-best-only')
    for phase_arg in ('phase1_epochs', 'phase2_epochs', 'phase3_epochs'):
        value = getattr(opt, phase_arg, None)
        if value is not None:
            command.extend([f'--{phase_arg.replace("_", "-")}', str(value)])
    if getattr(opt, 'progress_log_interval', None) is not None:
        command.extend(['--progress-log-interval', str(opt.progress_log_interval)])
    if getattr(opt, 'debug_log', 'off') != 'off':
        command.extend(['--debug-log', opt.debug_log])
        command.extend(['--debug-log-file', opt.debug_log_file])
        command.extend(['--debug-log-modules', opt.debug_log_modules])
    for key, value in config.enabled_flags.items():
        command.extend(flag_to_args(key, value))
    return command


def write_stage_plan(config):
    stage_dir = Path(config.output_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    config.save(stage_dir / 'stage_config.yaml')


def best_weight_path(stage_dir):
    stage_dir = Path(stage_dir)
    best = stage_dir / 'weights' / 'best.pt'
    last = stage_dir / 'weights' / 'last.pt'
    if best.is_file():
        return str(best)
    if last.is_file():
        return str(last)
    return ''


def _stream_pipe(pipe, log_handle, console_handle=None):
    while True:
        line = pipe.readline()
        if not line:
            break
        log_handle.write(line)
        log_handle.flush()
        if console_handle:
            console_handle.write(line)
            console_handle.flush()


def _console_handle(console_log, stream_name):
    if console_log == 'all' or console_log == stream_name:
        return sys.stderr if stream_name == 'stderr' else sys.stdout
    return None


def run_command(command, stage_dir, debug_logger=None, console_log='stderr'):
    stdout = Path(stage_dir) / 'stdout.log'
    stderr = Path(stage_dir) / 'stderr.log'
    if debug_logger:
        debug_logger.log_event(
            'debug', 'runner', 'run_command', 'command_start', 'stage command started',
            summary={'command': command, 'stdout_path': str(stdout), 'stderr_path': str(stderr)})
    if console_log != 'off':
        print(f'\n[runner] stage start: {Path(stage_dir).name}', flush=True)
        print(f'[runner] stdout: {stdout}', flush=True)
        print(f'[runner] stderr: {stderr}', flush=True)
    with stdout.open('w', encoding='utf-8') as out, stderr.open('w', encoding='utf-8') as err:
        if console_log == 'off':
            completed = subprocess.run(command, cwd=str(ROOT), stdout=out, stderr=err, check=False)
            exit_code = completed.returncode
        else:
            process = subprocess.Popen(
                command, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding='utf-8', errors='replace', bufsize=1)
            threads = [
                threading.Thread(
                    target=_stream_pipe,
                    args=(process.stdout, out, _console_handle(console_log, 'stdout')),
                    daemon=True),
                threading.Thread(
                    target=_stream_pipe,
                    args=(process.stderr, err, _console_handle(console_log, 'stderr')),
                    daemon=True),
            ]
            for thread in threads:
                thread.start()
            exit_code = process.wait()
            for thread in threads:
                thread.join()
    if debug_logger:
        level = 'error' if exit_code else 'debug'
        debug_logger.log_event(
            level, 'runner', 'run_command', 'command_end', 'stage command completed',
            summary={'exit_code': exit_code, 'stdout_path': str(stdout), 'stderr_path': str(stderr)})
    if console_log != 'off':
        print(f'\n[runner] stage end: {Path(stage_dir).name} exit_code={exit_code}', flush=True)
    return exit_code, stdout, stderr


def missing_artifacts(stage_dir, require_export=False):
    stage_dir = Path(stage_dir)
    missing = []
    if not (stage_dir / 'results.csv').is_file():
        missing.append('results.csv')
    if not (stage_dir / 'loss_detail.csv').is_file():
        missing.append('loss_detail.csv')
    if not ((stage_dir / 'weights' / 'best.pt').is_file() or (stage_dir / 'weights' / 'last.pt').is_file()):
        missing.append('weights/best.pt|weights/last.pt')
    if require_export and not (stage_dir / 'export_check.json').is_file():
        missing.append('export_check.json')
    return missing


def run_profile(opt, stage_dir, weights):
    if opt.skip_profile or not weights:
        return
    profile = Path(stage_dir) / 'profile.json'
    command = [
        sys.executable, str(ROOT / 'tools' / 'profile_model.py'),
        '--weights', weights,
        '--img', *[str(x) for x in opt.img],
        '--batch', '1',
        '--device', opt.profile_device,
        '--output', str(profile),
    ]
    if opt.baseline_gflops:
        command.extend(['--baseline-gflops', str(opt.baseline_gflops)])
    subprocess.run(command, cwd=str(ROOT), check=False)


def placeholder_export(stage_dir, require_export=False):
    export_path = Path(stage_dir) / 'export_check.json'
    if export_path.exists():
        return
    export_path.write_text(json.dumps({
        'schema_version': '1.3.8',
        'status': 'skip' if not require_export else 'missing',
        'comparison': {'status': 'skip' if not require_export else 'missing'},
        'reason': 'export check is optional for 1.3.8 runner unless --require-export is set',
    }, indent=2) + '\n', encoding='utf-8')


def _metric(result, key):
    return (result or {}).get('metrics', {}).get(key)


def _numeric_delta(current, previous):
    try:
        if current is None or previous is None:
            return None
        return float(current) - float(previous)
    except (TypeError, ValueError):
        return None


def _percent_delta(current, baseline):
    try:
        if current is None or baseline in (None, 0):
            return None
        return (float(current) - float(baseline)) / float(baseline) * 100.0
    except (TypeError, ValueError):
        return None


def previous_success(results, family):
    for result in reversed(results):
        if result.get('model_family') == family and result.get('decision') in (Decision.KEEP, Decision.KEEP_CANDIDATE):
            return result
    return None


def baseline_for_family(results, family):
    for result in results:
        if result.get('model_family') == family:
            return result
    return None


def decide_result(opt, config, collected, exit_code=0, stdout_path='', stderr_path='', dry_run=False,
                  baseline=None, previous=None, missing=None):
    metrics = collected.get('metrics', {})
    best_weight = best_weight_path(config.output_dir)
    export_status = metrics.get('export_status', 'skip')
    missing = list(missing or [])
    log_paths = {
        'stdout': str(stdout_path),
        'stderr': str(stderr_path),
        'debug_trace': str(Path(config.output_dir) / getattr(opt, 'debug_log_file', 'debug_trace.log')),
        'error_trace': str(Path(config.output_dir) / 'error_trace.log'),
    }
    if dry_run:
        return StageResult(
            stage_id=config.stage_id,
            train_type=config.train_type,
            decision=Decision.DEFER,
            reason='dry-run command planned',
            best_weight='',
            fallback_weight=config.start_weight,
            metrics=metrics,
            export_status='skip',
            command=config.command,
            log_paths=log_paths,
        )
    if exit_code:
        return StageResult(
            stage_id=config.stage_id,
            train_type=config.train_type,
            decision=Decision.BLOCKER,
            reason=f'training command failed with exit code {exit_code}',
            best_weight='',
            fallback_weight=config.start_weight,
            metrics=metrics,
            export_status=export_status,
            hard_fail=True,
            failed_category='train',
            exit_code=exit_code,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            command=config.command,
            missing_artifacts=missing,
            log_paths=log_paths,
        )
    if not best_weight:
        return StageResult(
            stage_id=config.stage_id,
            train_type=config.train_type,
            decision=Decision.BLOCKER,
            reason='best.pt/last.pt missing after stage run',
            fallback_weight=config.start_weight,
            metrics=metrics,
            export_status=export_status,
            hard_fail=True,
            failed_category='artifact',
            exit_code=exit_code,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            command=config.command,
            missing_artifacts=missing or ['weights/best.pt|weights/last.pt'],
            log_paths=log_paths,
        )
    if opt.require_export and export_status not in ('ok', 'pass'):
        return StageResult(
            stage_id=config.stage_id,
            train_type=config.train_type,
            decision=Decision.BLOCKER,
            reason=f'export status is {export_status}',
            best_weight=best_weight,
            fallback_weight=config.start_weight,
            metrics=metrics,
            export_status=export_status,
            hard_fail=True,
            failed_category='export',
            exit_code=exit_code,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            command=config.command,
            missing_artifacts=missing,
            log_paths=log_paths,
        )
    if config.dataset_profile == 'target_full' and metrics.get('primary_mAP') is None:
        return StageResult(
            stage_id=config.stage_id,
            train_type=config.train_type,
            decision=Decision.BLOCKER,
            reason='target_full primary_mAP is missing',
            best_weight=best_weight,
            fallback_weight=config.start_weight,
            metrics=metrics,
            export_status=export_status,
            hard_fail=True,
            failed_category='metric',
            exit_code=exit_code,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            command=config.command,
            missing_artifacts=missing,
            log_paths=log_paths,
        )
    if config.dataset_profile == 'target_full':
        map_delta = _numeric_delta(metrics.get('primary_mAP'), _metric(previous, 'primary_mAP'))
        if map_delta is not None and map_delta < opt.min_primary_map_delta:
            return StageResult(
                stage_id=config.stage_id,
                train_type=config.train_type,
                decision=Decision.DROP,
                reason=f'primary_mAP delta vs previous success {map_delta:.4f} below {opt.min_primary_map_delta:.4f}',
                best_weight=best_weight,
                fallback_weight=config.start_weight,
                metrics=metrics,
                export_status=export_status,
                soft_fail=True,
                failed_category='metric',
                exit_code=exit_code,
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
                command=config.command,
                missing_artifacts=missing,
                log_paths=log_paths,
            )
        gfops_delta = _percent_delta(metrics.get('GFLOPs'), _metric(baseline, 'GFLOPs'))
        if gfops_delta is not None and gfops_delta > opt.max_gflops_delta_percent:
            metrics['GFLOPs_delta_percent'] = gfops_delta
            return StageResult(
                stage_id=config.stage_id,
                train_type=config.train_type,
                decision=Decision.DROP,
                reason=f'GFLOPs delta vs baseline {gfops_delta:.2f}% above {opt.max_gflops_delta_percent:.2f}%',
                best_weight=best_weight,
                fallback_weight=config.start_weight,
                metrics=metrics,
                export_status=export_status,
                soft_fail=True,
                failed_category='runtime_cost',
                exit_code=exit_code,
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
                command=config.command,
                missing_artifacts=missing,
                log_paths=log_paths,
            )
    decision = Decision.KEEP if config.dataset_profile == 'coco128_quick' else Decision.KEEP_CANDIDATE
    reason = 'COCO128 quick run passed. Artifacts generated.' if config.dataset_profile == 'coco128_quick' else 'target full run completed; review deltas before final keep'
    return StageResult(
        stage_id=config.stage_id,
        train_type=config.train_type,
        decision=decision,
        reason=reason,
        best_weight=best_weight,
        fallback_weight=config.start_weight,
        metrics=metrics,
        export_status=export_status,
        hard_fail=False,
        soft_fail=False,
        exit_code=exit_code,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        command=config.command,
        missing_artifacts=missing,
        log_paths=log_paths,
    )


def build_stage_configs(opt):
    output = Path(opt.output)
    families = parse_families(opt.model_family)
    current_weights = {family: family_weight(opt, family) for family in families}
    configs = []
    for spec in STAGES:
        if not stage_in_range(spec, opt.start_stage, opt.end_stage):
            continue
        for family in families:
            if not stage_for_family(spec, family):
                continue
            stage_dir = output / f'{spec.stage_id}_{spec.stage_name}_{family}'
            start_weight = current_weights.get(family, '')
            enabled_flags, epochs = quick_stage_overrides(spec, opt)
            config = StageConfig(
                stage_id=spec.stage_id,
                stage_name=spec.stage_name,
                dataset_profile=opt.dataset_profile,
                model_family=family,
                train_type=spec.train_type,
                data=opt.data,
                start_weight=start_weight,
                enabled_flags=enabled_flags,
                disabled_flags=dict(spec.disabled_flags),
                seed=opt.seed,
                epochs=epochs,
                output_dir=str(stage_dir),
                dry_run=opt.dry_run or spec.defer,
            )
            config.command = build_train_command(opt, config, family)
            configs.append((config, spec))
            if opt.dry_run:
                planned_weight = stage_dir / 'weights' / 'best.pt'
                current_weights[family] = str(planned_weight)
    return configs


class TrainingSequenceRunner:
    def __init__(self, opt):
        self.opt = opt
        self.output = Path(opt.output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.results = []
        self.current_weights = {family: family_weight(opt, family) for family in parse_families(opt.model_family)}

    def run(self):
        configs = build_stage_configs(self.opt)
        for config, spec in configs:
            if not self.opt.dry_run and config.stage_id != '00':
                config.start_weight = self.current_weights.get(config.model_family, config.start_weight)
                config.command = build_train_command(self.opt, config, config.model_family)
            config.validate(strict_paths=not self.opt.dry_run and not spec.defer)
            write_stage_plan(config)
            stage_dir = Path(config.output_dir)
            if self.opt.resume_sequence and (stage_dir / 'stage_result.yaml').is_file():
                collected = collect_stage_result(stage_dir)
                self.results.append(collected)
                if collected.get('decision') in (Decision.KEEP, Decision.KEEP_CANDIDATE):
                    self.current_weights[config.model_family] = collected.get('best_weight', '')
                if collected.get('hard_fail') and self.opt.stop_on_hard_fail:
                    break
                continue
            if self.opt.dry_run or spec.defer:
                result = decide_result(self.opt, config, {'metrics': {}}, dry_run=True)
                result.reason = 'deferred optional stage' if spec.defer and not self.opt.dry_run else result.reason
                result.save(stage_dir / 'stage_result.yaml')
                self.results.append(collect_stage_result(stage_dir))
                continue

            debug_logger = get_debug_logger(stage_dir,
                                            getattr(self.opt, 'debug_log', 'off'),
                                            getattr(self.opt, 'debug_log_modules', ''),
                                            debug_file=getattr(self.opt, 'debug_log_file', 'debug_trace.log'))
            exit_code, stdout_path, stderr_path = run_command(
                config.command, stage_dir, debug_logger,
                console_log=getattr(self.opt, 'console_log', 'stderr'))
            best = best_weight_path(stage_dir)
            run_profile(self.opt, stage_dir, best)
            placeholder_export(stage_dir, require_export=self.opt.require_export)
            collected = collect_stage_result(stage_dir)
            baseline = baseline_for_family(self.results, config.model_family)
            previous = previous_success(self.results, config.model_family)
            missing = missing_artifacts(stage_dir, require_export=self.opt.require_export)
            result = decide_result(
                self.opt, config, collected, exit_code=exit_code,
                stdout_path=stdout_path, stderr_path=stderr_path,
                baseline=baseline, previous=previous, missing=missing)
            if result.hard_fail or result.soft_fail:
                debug_logger.log_event(
                    'error', 'runner', 'decide_result', 'stage_failed', result.reason,
                    summary={
                        'stage_id': config.stage_id,
                        'stage_name': config.stage_name,
                        'train_type': config.train_type,
                        'decision': result.decision,
                        'failed_category': result.failed_category,
                        'missing_artifacts': result.missing_artifacts,
                    })
            result.save(stage_dir / 'stage_result.yaml')
            collected = collect_stage_result(stage_dir)
            self.results.append(collected)
            if result.decision in (Decision.KEEP, Decision.KEEP_CANDIDATE):
                self.current_weights[config.model_family] = result.best_weight
            if result.hard_fail and self.opt.stop_on_hard_fail:
                break

        self.write_reports()
        return self.results

    def write_reports(self):
        writer = TrainingReportWriter(self.output)
        delta_rows = compute_stage_delta(self.results)
        for result in self.results:
            writer.write_stage_summary(result.get('stage_config', {}), result, delta_rows)
            writer.write_stage_delta_csv(result, delta_rows)
        final_dir = self.output / 'final_report'
        final_dir.mkdir(parents=True, exist_ok=True)
        write_delta_csv(delta_rows, final_dir / 'metrics_delta_all.csv')
        writer.write_decision_table_csv(self.results)
        writer.write_sequence_summary(self.results)
        writer.write_train_type_summaries(self.results, delta_rows)
        if self.opt.report_output:
            report_output = self.opt.report_output
        elif self.opt.dry_run:
            report_output = str(final_dir / f'final_training_report_v1.8_dry_run_{date.today().isoformat()}.md')
        else:
            report_output = f'doc/REPORT/final_training_report_v1.8_{date.today().isoformat()}.md'
        writer.write_final_report(self.results, report_output)
        manifest = {
            'schema_version': '1.3.8',
            'plan': self.opt.plan,
            'dataset_profile': self.opt.dataset_profile,
            'model_family': self.opt.model_family,
            'output': str(self.output),
            'dry_run': self.opt.dry_run,
            'launcher': self.opt.launcher,
            'nproc_per_node': self.opt.nproc_per_node,
            'console_log': self.opt.console_log,
            'stage_count': len(self.results),
            'report_output': report_output,
        }
        (final_dir / 'sequence_manifest.yaml').write_text(
            yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding='utf-8')


def main(opt):
    if not Path(opt.plan).is_file():
        raise SystemExit(f'plan file not found: {opt.plan}')
    if not opt.output:
        if opt.project and opt.name:
            opt.output = str(Path(opt.project) / opt.name)
        else:
            raise SystemExit('the following arguments are required: --output or --project plus --name')
    if opt.launcher == 'distributed':
        if opt.nproc_per_node < 1:
            raise SystemExit('--nproc-per-node must be >= 1 when --launcher distributed is used')
        world_size = opt.nnodes * opt.nproc_per_node
        if opt.batch_size % world_size != 0:
            raise SystemExit('--batch-size must be a multiple of distributed world size')
    if not opt.dry_run and not opt.weights and not opt.l_weights and not opt.w6_weights and not opt.cfg and not opt.l_cfg and not opt.w6_cfg:
        raise SystemExit('real sequence run requires --weights/--l-weights/--w6-weights or --cfg/--l-cfg/--w6-cfg')
    runner = TrainingSequenceRunner(opt)
    results = runner.run()
    print(f'training sequence complete: {len(results)} stage records in {opt.output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--dataset-profile', choices=['coco128_quick', 'target_full'], required=True)
    parser.add_argument('--model-family', type=str, required=True, help='l,w6,l_only,w6_only,l or w6')
    parser.add_argument('--output', type=str, default='')
    parser.add_argument('--project', type=str, default='', help='compatibility alias used with --name when --output is omitted')
    parser.add_argument('--name', type=str, default='', help='compatibility alias used with --project when --output is omitted')
    parser.add_argument('--exist-ok', action='store_true', help='accepted for train.py command compatibility')
    parser.add_argument('--stop-on-hard-fail', action='store_true')
    parser.add_argument('--start-stage', type=str, default='')
    parser.add_argument('--end-stage', type=str, default='')
    parser.add_argument('--resume-sequence', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--max-retry-per-stage', type=int, default=1)
    parser.add_argument('--skip-plots', action='store_true')
    parser.add_argument('--skip-profile', action='store_true')
    parser.add_argument('--require-export', action='store_true')
    parser.add_argument('--epochs', '--epoch', dest='epochs', type=int, default=3)
    parser.add_argument('--batch-size', '--batch', dest='batch_size', type=int, default=16)
    parser.add_argument('--img', '--img-size', dest='img', nargs='+', type=int, default=[640, 640])
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.p5.yaml')
    parser.add_argument('--weights', type=str, default='')
    parser.add_argument('--l-weights', type=str, default='')
    parser.add_argument('--w6-weights', type=str, default='')
    parser.add_argument('--cfg', type=str, default='')
    parser.add_argument('--l-cfg', type=str, default='')
    parser.add_argument('--w6-cfg', type=str, default='')
    parser.add_argument('--device', type=str, default='')
    parser.add_argument('--profile-device', type=str, default='cpu')
    parser.add_argument('--workers', '--worker', dest='workers', type=int, default=8)
    parser.add_argument('--sync-bn', action='store_true', help='pass SyncBatchNorm flag to each training stage')
    parser.add_argument('--image-weights', action='store_true', help='pass image-weights flag to each training stage')
    parser.add_argument('--multi-scale', action='store_true', help='pass multi-scale flag to each training stage')
    parser.add_argument('--save-best-only', action='store_true',
                        help='pass save-best-only flag to each training stage')
    parser.add_argument('--phase1-epochs', type=int, default=None,
                        help='pass phase1 epoch count to each training stage')
    parser.add_argument('--phase2-epochs', type=int, default=None,
                        help='pass phase2 epoch count to each training stage')
    parser.add_argument('--phase3-epochs', type=int, default=None,
                        help='pass phase3 epoch count to each training stage')
    parser.add_argument('--launcher', choices=['none', 'distributed'], default='none',
                        help='run each stage command directly or through torch.distributed.launch')
    parser.add_argument('--nproc-per-node', '--nproc_per_node', dest='nproc_per_node', type=int, default=1)
    parser.add_argument('--nnodes', type=int, default=1)
    parser.add_argument('--node-rank', '--node_rank', dest='node_rank', type=int, default=0)
    parser.add_argument('--master-addr', '--master_addr', dest='master_addr', type=str, default='127.0.0.1')
    parser.add_argument('--master-port', '--master_port', dest='master_port', type=str, default='')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--baseline-gflops', type=float, default=None)
    parser.add_argument('--min-primary-map-delta', type=float, default=-0.02,
                        help='target_full soft fail threshold versus previous success')
    parser.add_argument('--max-gflops-delta-percent', type=float, default=10.0,
                        help='target_full soft fail threshold versus family baseline')
    parser.add_argument('--debug-log', choices=['off', 'error', 'debug', 'trace'], default='off')
    parser.add_argument('--debug-log-file', type=str, default='debug_trace.log')
    parser.add_argument('--debug-log-modules', type=str,
                        default='dataset,phase,model,loss,aug,sampler,finetune,runner')
    parser.add_argument('--console-log', choices=['off', 'stderr', 'stdout', 'all'], default='stderr',
                        help='stream stage logs to console while still writing stdout.log/stderr.log')
    parser.add_argument('--progress-log-interval', type=int, default=50,
                        help='pass rank0 batch progress interval to each training stage; 0 disables')
    parser.add_argument('--report-output', type=str, default='')
    main(parser.parse_args())
