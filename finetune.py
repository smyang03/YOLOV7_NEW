import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import yaml

from utils.class_mapping import validate_class_mapping
from utils.continual_loss import max_schedule_value, parse_float_or_schedule
from utils.general import increment_path
from utils.replay_buffer import ReplayBufferBuilder, read_image_list, read_text_fallback
from tools.dataset_manifest import split_summary


ROOT = Path(__file__).resolve().parent
FREEZE_POLICY_LAYERS = {
    'none': [0],
    'backbone': [50],
    'partial': [75],
    'neck_lower': [75],
}


def load_yaml(path):
    return yaml.safe_load(read_text_fallback(path)) or {}


def save_yaml(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding='utf-8')


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def write_dataset_manifest(data_yaml, output, stage='1.3.7'):
    data_yaml = Path(data_yaml)
    data = load_yaml(data_yaml)
    data_dir = data_yaml.parent
    splits = {}
    for name in ('train', 'val', 'test'):
        if name in data and data[name]:
            splits[name] = split_summary(name, data[name], data_dir)
    import hashlib
    aggregate = hashlib.sha256()
    for split in splits.values():
        aggregate.update(split['checksum'].encode())
    result = {
        'schema_version': '1.3.7',
        'stage': stage,
        'data': str(data_yaml),
        'nc': data.get('nc'),
        'names': data.get('names'),
        'splits': splits,
        'aggregate_checksum': aggregate.hexdigest(),
    }
    save_json(output, result)
    return result


def write_skip_artifacts(save_dir, opt):
    pseudo_path = save_dir / 'pseudo_label_manifest.json'
    merge_path = save_dir / 'merge_report.json'
    if not pseudo_path.exists():
        save_json(pseudo_path, {
            'schema_version': '1.3.7',
            'status': 'skip',
            'reason': 'pseudo label generation is executed by tools/generate_pseudo_labels.py when needed',
            'pseudo_conf': opt.pseudo_conf,
            'pseudo_iou_dedup': opt.pseudo_iou_dedup,
        })
    if not merge_path.exists():
        save_json(merge_path, {
            'schema_version': '1.3.7',
            'status': 'skip',
            'reason': 'label merge is executed by tools/merge_labels.py when needed',
        })
    return pseudo_path, merge_path


def load_replay_images(replay_buffer):
    if not replay_buffer:
        return []
    path = Path(replay_buffer)
    if path.is_file():
        manifest = json.loads(path.read_text(encoding='utf-8'))
        return [x['image'] for x in manifest.get('selected_images', [])]
    if path.is_dir():
        manifest = path / 'replay_manifest.json'
        if manifest.is_file():
            return load_replay_images(manifest)
        return sorted(str(p) for p in path.rglob('*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))
    return []


def replay_request(opt, train_image_count):
    if opt.replay_count >= 0:
        return float(opt.replay_count), int(opt.replay_count)
    ratio = float(opt.replay_ratio)
    if opt.replay_ratio_source == 'finetune' and 0 < ratio <= 1:
        count = max(1, int(math.ceil(int(train_image_count) * ratio)))
        return float(count), count
    return ratio, None


def build_train_data(opt, save_dir):
    data_path = Path(opt.data)
    data = load_yaml(data_path)
    train_images = read_image_list(data.get('train'), data_path.parent, ROOT) if data.get('train') else []
    replay_images = load_replay_images(opt.replay_buffer)
    replay_manifest_path = save_dir / 'replay_manifest.json'

    replay_value, replay_requested_count = replay_request(opt, len(train_images))

    if not replay_images and replay_value > 0:
        manifest = ReplayBufferBuilder(
            opt.base_data,
            replay_ratio=replay_value,
            seed=opt.seed).build(output=replay_manifest_path)
        manifest['replay_ratio_source'] = opt.replay_ratio_source
        manifest['replay_requested_count'] = replay_requested_count
        save_json(replay_manifest_path, manifest)
        replay_images = [x['image'] for x in manifest.get('selected_images', [])]
    elif replay_images:
        save_json(replay_manifest_path, {
            'schema_version': '1.3.7',
            'source_data': opt.base_data,
            'source_split': 'external_replay_buffer',
            'replay_ratio': opt.replay_ratio,
            'replay_ratio_source': opt.replay_ratio_source,
            'replay_requested_count': replay_requested_count,
            'selected_count': len(replay_images),
            'selected_images': [{'image': x, 'label': '', 'classes': []} for x in replay_images],
            'selection_seed': opt.seed,
            'status': 'pass',
        })
    else:
        save_json(replay_manifest_path, {
            'schema_version': '1.3.7',
            'source_data': opt.base_data,
            'source_split': 'train',
            'replay_ratio': opt.replay_ratio,
            'replay_ratio_source': opt.replay_ratio_source,
            'replay_requested_count': replay_requested_count,
            'selected_count': 0,
            'selected_images': [],
            'selection_seed': opt.seed,
            'status': 'pass',
        })

    if train_images or replay_images:
        train_list = save_dir / 'finetune_train_with_replay.txt'
        lines = [str(x) for x in train_images + replay_images]
        train_list.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
        data = dict(data)
        data['train'] = str(train_list)

    output_data = save_dir / 'finetune_data.yaml'
    save_yaml(output_data, data)
    return output_data, len(train_images), len(replay_images), replay_manifest_path, replay_requested_count


def write_finetune_results(path, opt, train_images, replay_images, status):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'sub_stage', 'status', 'train_images', 'replay_images', 'replay_ratio',
            'replay_ratio_source', 'replay_requested_count',
            'pseudo_conf', 'pseudo_iou_dedup', 'distill_alpha', 'distill_beta',
            'bn_policy', 'freeze_policy', 'best_val_set'])
        writer.writeheader()
        writer.writerow({
            'sub_stage': opt.sub_stage,
            'status': status,
            'train_images': train_images,
            'replay_images': replay_images,
            'replay_ratio': opt.replay_ratio,
            'replay_ratio_source': opt.replay_ratio_source,
            'replay_requested_count': opt.replay_requested_count,
            'pseudo_conf': opt.pseudo_conf,
            'pseudo_iou_dedup': opt.pseudo_iou_dedup,
            'distill_alpha': opt.distill_alpha,
            'distill_beta': opt.distill_beta,
            'bn_policy': opt.bn_policy,
            'freeze_policy': opt.freeze_policy,
            'best_val_set': opt.best_val_set,
        })


def write_forgetting_template(path, opt, status):
    data = {
        'schema_version': '1.3.7',
        'scratch_baseline': opt.base_weights,
        'finetune_run': '',
        'new_class_map': [],
        'old_class_map': {},
        'heldout_class_map': [],
        'old_class_drop_percent': None,
        'new_class_retention_percent': None,
        'replay_ratio': opt.replay_ratio,
        'pseudo_conf': opt.pseudo_conf,
        'pseudo_iou_dedup': opt.pseudo_iou_dedup,
        'distill_alpha': opt.distill_alpha,
        'distill_beta': opt.distill_beta,
        'bn_policy': opt.bn_policy,
        'freeze_policy': opt.freeze_policy,
        'sub_stage': opt.sub_stage,
        'status': status,
    }
    save_yaml(path, data)


def train_command(opt, data_yaml, save_dir):
    cmd = [
        sys.executable, str(ROOT / 'train.py'),
        '--weights', opt.base_weights,
        '--data', str(data_yaml),
        '--hyp', opt.hyp,
        '--epochs', str(opt.epochs),
        '--batch-size', str(opt.batch_size),
        '--project', str(save_dir.parent),
        '--name', save_dir.name,
        '--exist-ok',
        '--best-val-set', opt.best_val_set,
        '--bn-policy', opt.bn_policy,
        '--freeze', *[str(x) for x in FREEZE_POLICY_LAYERS[opt.freeze_policy]],
    ]
    if opt.save_best_only:
        cmd.append('--save-best-only')
    if opt.cfg:
        cmd.extend(['--cfg', opt.cfg])
    if opt.device:
        cmd.extend(['--device', opt.device])
    if opt.workers >= 0:
        cmd.extend(['--workers', str(opt.workers)])
    if opt.img_size:
        cmd.extend(['--img-size', *[str(x) for x in opt.img_size]])
    if opt.teacher_weights and (
            max_schedule_value(parse_float_or_schedule(opt.distill_alpha)) > 0 or
            max_schedule_value(parse_float_or_schedule(opt.distill_beta)) > 0):
        cmd.extend([
            '--teacher-weights', opt.teacher_weights,
            '--distill-alpha', opt.distill_alpha,
            '--distill-beta', opt.distill_beta,
            '--distill-conf-thres', str(opt.distill_conf_thres),
        ])
    return cmd


def main(opt):
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    save_dir.mkdir(parents=True, exist_ok=True)
    mapping_output = save_dir / 'class_mapping_check.json'
    mapping_result = validate_class_mapping(
        opt.base_data, opt.data,
        mapping_file=opt.mapping_file,
        output=mapping_output)
    if mapping_result['status'] != 'pass':
        raise SystemExit(f'class mapping check failed: {mapping_output}')

    data_yaml, train_images, replay_images, replay_manifest, replay_requested_count = build_train_data(opt, save_dir)
    opt.replay_requested_count = replay_requested_count
    dataset_manifest = save_dir / 'dataset_manifest.json'
    write_dataset_manifest(data_yaml, dataset_manifest)
    pseudo_manifest, merge_report = write_skip_artifacts(save_dir, opt)
    alpha = parse_float_or_schedule(opt.distill_alpha)
    beta = parse_float_or_schedule(opt.distill_beta)
    distill_requested = max_schedule_value(alpha) > 0 or max_schedule_value(beta) > 0
    if distill_requested and not opt.teacher_weights:
        raise SystemExit('--teacher-weights is required when distill alpha/beta is non-zero')

    stage_result = {
        'schema_version': '1.3.7',
        'save_dir': str(save_dir),
        'class_mapping_check': str(mapping_output),
        'data': str(data_yaml),
        'dataset_manifest': str(dataset_manifest),
        'replay_manifest': str(replay_manifest),
        'pseudo_label_manifest': str(pseudo_manifest),
        'merge_report': str(merge_report),
        'base_weights': opt.base_weights,
        'teacher_weights': opt.teacher_weights,
        'replay_ratio': opt.replay_ratio,
        'replay_ratio_source': opt.replay_ratio_source,
        'replay_count': opt.replay_count,
        'replay_requested_count': replay_requested_count,
        'train_images': train_images,
        'replay_images': replay_images,
        'pseudo_conf': opt.pseudo_conf,
        'pseudo_iou_dedup': opt.pseudo_iou_dedup,
        'distill_alpha': opt.distill_alpha,
        'distill_beta': opt.distill_beta,
        'bn_policy': opt.bn_policy,
        'freeze_policy': opt.freeze_policy,
        'best_val_set': opt.best_val_set,
        'save_best_only': opt.save_best_only,
        'sub_stage': opt.sub_stage,
        'dry_run': opt.dry_run,
        'status': 'dry_run' if opt.dry_run else 'pending',
    }
    command = train_command(opt, data_yaml, save_dir)
    stage_result['train_command'] = command
    save_yaml(save_dir / 'stage_result.yaml', stage_result)
    write_finetune_results(save_dir / 'finetune_results.csv', opt, train_images, replay_images, stage_result['status'])
    write_forgetting_template(save_dir / 'forgetting_report.yaml', opt, stage_result['status'])

    if opt.dry_run:
        print(yaml.safe_dump(stage_result, sort_keys=False, allow_unicode=True))
        return

    completed = subprocess.run(command, cwd=str(ROOT), check=False)
    stage_result['status'] = 'pass' if completed.returncode == 0 else 'fail'
    stage_result['returncode'] = completed.returncode
    save_yaml(save_dir / 'stage_result.yaml', stage_result)
    write_finetune_results(save_dir / 'finetune_results.csv', opt, train_images, replay_images, stage_result['status'])
    write_forgetting_template(save_dir / 'forgetting_report.yaml', opt, stage_result['status'])
    if completed.returncode:
        raise SystemExit(completed.returncode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-weights', '--weights', dest='base_weights', type=str, required=True)
    parser.add_argument('--teacher-weights', type=str, default='')
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--base-data', type=str, required=True)
    parser.add_argument('--cfg', type=str, default='')
    parser.add_argument('--hyp', type=str, default='data/hyp_finetune.yaml')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', '--batch', dest='batch_size', type=int, default=16)
    parser.add_argument('--img-size', '--img', dest='img_size', nargs='+', type=int, default=[640, 640])
    parser.add_argument('--project', default='runs/finetune')
    parser.add_argument('--name', default='exp')
    parser.add_argument('--exist-ok', action='store_true')
    parser.add_argument('--device', default='')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--mapping-file', type=str, default='')
    parser.add_argument('--replay-buffer', type=str, default='')
    parser.add_argument('--replay-ratio', type=float, default=0.3)
    parser.add_argument('--replay-ratio-source', choices=['base', 'finetune'], default='base',
                        help='when replay-ratio <= 1, choose whether the ratio is based on base or finetune train count')
    parser.add_argument('--replay-count', type=int, default=-1,
                        help='explicit replay image count; overrides --replay-ratio when >= 0')
    parser.add_argument('--pseudo-conf', type=float, default=0.5)
    parser.add_argument('--pseudo-iou-dedup', type=float, default=0.8)
    parser.add_argument('--distill-alpha', type=str, default='0.0')
    parser.add_argument('--distill-beta', type=str, default='0.0')
    parser.add_argument('--distill-conf-thres', type=float, default=0.5)
    parser.add_argument('--bn-policy', choices=['train', 'eval'], default='train')
    parser.add_argument('--freeze-policy', '--freeze', dest='freeze_policy',
                        choices=['none', 'backbone', 'partial', 'neck_lower'], default='none')
    parser.add_argument('--best-val-set', type=str, default='first',
                        help='validation set used for best.pt selection: first, last, combined, or a named val set')
    parser.add_argument('--save-best-only', action='store_true',
                        help='pass through to train.py to save only weights/best.pt')
    parser.add_argument('--sub-stage', type=str, default='1.3.7-E1')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dry-run', action='store_true')
    main(parser.parse_args())
