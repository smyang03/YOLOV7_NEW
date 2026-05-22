import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


DECISIONS = ('keep', 'keep_candidate', 'drop', 'retry_tune', 'blocker', 'defer')
DATASET_PROFILES = ('coco128_quick', 'target_full')
MODEL_FAMILIES = ('l', 'w6', 'l_only', 'w6_only')
FAILURE_CATEGORIES = (
    None, 'data', 'loader', 'loss', 'assignment', 'architecture', 'export',
    'runtime_cost', 'augmentation', 'finetune', 'train', 'artifact', 'metric')


class Decision:
    KEEP = 'keep'
    KEEP_CANDIDATE = 'keep_candidate'
    DROP = 'drop'
    RETRY_TUNE = 'retry_tune'
    BLOCKER = 'blocker'
    DEFER = 'defer'


def _read_yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding='utf-8-sig')) or {}


def _write_yaml(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding='utf-8')


def _write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


@dataclass
class StageConfig:
    stage_id: str
    stage_name: str
    dataset_profile: str
    model_family: str
    data: str
    train_type: str = ''
    start_weight: str = ''
    enabled_flags: Dict[str, Any] = field(default_factory=dict)
    disabled_flags: Dict[str, Any] = field(default_factory=dict)
    seed: int = 0
    epochs: int = 0
    output_dir: str = ''
    command: List[str] = field(default_factory=list)
    dry_run: bool = False

    @classmethod
    def from_dict(cls, data):
        return cls(
            stage_id=str(data.get('stage_id', '')),
            stage_name=str(data.get('stage_name', '')),
            dataset_profile=str(data.get('dataset_profile', '')),
            model_family=str(data.get('model_family', '')),
            data=str(data.get('data', '')),
            train_type=str(data.get('train_type', '') or ''),
            start_weight=str(data.get('start_weight', '') or ''),
            enabled_flags=dict(data.get('enabled_flags') or {}),
            disabled_flags=dict(data.get('disabled_flags') or {}),
            seed=int(data.get('seed', 0) or 0),
            epochs=int(data.get('epochs', 0) or 0),
            output_dir=str(data.get('output_dir', '')),
            command=list(data.get('command') or []),
            dry_run=bool(data.get('dry_run', False)),
        )

    @classmethod
    def load(cls, path):
        return cls.from_dict(_read_yaml(path))

    def to_dict(self):
        return asdict(self)

    def save(self, path):
        _write_yaml(path, self.to_dict())

    def validate(self, strict_paths=False):
        errors = []
        if not self.stage_id:
            errors.append('stage_id is required')
        if not self.stage_name:
            errors.append('stage_name is required')
        if not self.output_dir:
            errors.append('output_dir is required')
        if self.dataset_profile not in DATASET_PROFILES:
            errors.append(f'dataset_profile must be one of {DATASET_PROFILES}')
        if self.model_family not in MODEL_FAMILIES:
            errors.append(f'model_family must be one of {MODEL_FAMILIES}')
        if not self.data:
            errors.append('data is required')
        if self.dataset_profile == 'coco128_quick' and 'target_full' in self.output_dir.replace('\\', '/'):
            errors.append('coco128_quick output_dir must not be mixed with target_full output')
        if self.dataset_profile == 'target_full' and 'coco128_quick' in self.output_dir.replace('\\', '/'):
            errors.append('target_full output_dir must not be mixed with coco128_quick output')
        if strict_paths:
            if not Path(self.data).is_file():
                errors.append(f'data yaml not found: {self.data}')
            if self.stage_id != '00' and (not self.start_weight or not Path(self.start_weight).is_file()):
                errors.append(f'start_weight is required after Stage 00: {self.start_weight}')
        if errors:
            raise ValueError('; '.join(errors))


@dataclass
class StageResult:
    stage_id: str
    decision: str
    reason: str = ''
    best_weight: str = ''
    fallback_weight: str = ''
    metrics: Dict[str, Any] = field(default_factory=dict)
    export_status: str = 'skip'
    hard_fail: bool = False
    soft_fail: bool = False
    failed_category: Optional[str] = None
    primary_mAP: Optional[float] = None
    mAP50: Optional[float] = None
    small_AP: Optional[float] = None
    rare_recall: Optional[float] = None
    FP_per_image: Optional[float] = None
    FN_per_image: Optional[float] = None
    GFLOPs: Optional[float] = None
    GFLOPs_delta_percent: Optional[float] = None
    python_infer_ms: Optional[float] = None
    python_nms_ms: Optional[float] = None
    onnx_max_abs_diff: Optional[float] = None
    exit_code: Optional[int] = None
    stdout_path: str = ''
    stderr_path: str = ''
    command: List[str] = field(default_factory=list)
    carry_flags: Dict[str, Any] = field(default_factory=dict)
    disabled_flags: Dict[str, Any] = field(default_factory=dict)
    train_type: str = ''
    missing_artifacts: List[str] = field(default_factory=list)
    log_paths: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data):
        metrics = dict(data.get('metrics') or {})
        return cls(
            stage_id=str(data.get('stage_id', '')),
            decision=str(data.get('decision', '')),
            reason=str(data.get('reason', '') or ''),
            best_weight=str(data.get('best_weight', '') or ''),
            fallback_weight=str(data.get('fallback_weight', '') or ''),
            metrics=metrics,
            export_status=str(data.get('export_status', metrics.get('export_status', 'skip')) or 'skip'),
            hard_fail=bool(data.get('hard_fail', False)),
            soft_fail=bool(data.get('soft_fail', False)),
            failed_category=data.get('failed_category'),
            primary_mAP=data.get('primary_mAP', metrics.get('primary_mAP')),
            mAP50=data.get('mAP50', metrics.get('mAP50')),
            small_AP=data.get('small_AP', metrics.get('small_AP')),
            rare_recall=data.get('rare_recall', metrics.get('rare_recall')),
            FP_per_image=data.get('FP_per_image', metrics.get('FP_per_image')),
            FN_per_image=data.get('FN_per_image', metrics.get('FN_per_image')),
            GFLOPs=data.get('GFLOPs', metrics.get('GFLOPs')),
            GFLOPs_delta_percent=data.get('GFLOPs_delta_percent', metrics.get('GFLOPs_delta_percent')),
            python_infer_ms=data.get('python_infer_ms', metrics.get('python_infer_ms')),
            python_nms_ms=data.get('python_nms_ms', metrics.get('python_nms_ms')),
            onnx_max_abs_diff=data.get('onnx_max_abs_diff', metrics.get('onnx_max_abs_diff')),
            exit_code=data.get('exit_code'),
            stdout_path=str(data.get('stdout_path', '') or ''),
            stderr_path=str(data.get('stderr_path', '') or ''),
            command=list(data.get('command') or []),
            carry_flags=dict(data.get('carry_flags') or {}),
            disabled_flags=dict(data.get('disabled_flags') or {}),
            train_type=str(data.get('train_type', '') or ''),
            missing_artifacts=list(data.get('missing_artifacts') or []),
            log_paths=dict(data.get('log_paths') or {}),
        )

    @classmethod
    def load(cls, path):
        return cls.from_dict(_read_yaml(path))

    def to_dict(self):
        data = asdict(self)
        data['metrics'] = dict(self.metrics)
        for key in (
                'primary_mAP', 'mAP50', 'small_AP', 'rare_recall', 'FP_per_image', 'FN_per_image',
                'GFLOPs', 'GFLOPs_delta_percent', 'python_infer_ms', 'python_nms_ms', 'onnx_max_abs_diff'):
            data['metrics'].setdefault(key, data.get(key))
        data['metrics'].setdefault('export_status', self.export_status)
        return data

    def save(self, path):
        _write_yaml(path, self.to_dict())

    def save_json(self, path):
        _write_json(path, self.to_dict())

    def validate(self, dataset_profile='target_full'):
        errors = []
        if self.decision not in DECISIONS:
            errors.append(f'decision must be one of {DECISIONS}')
        if self.hard_fail and self.decision != Decision.BLOCKER:
            errors.append('hard_fail=True requires decision=blocker')
        if self.decision == Decision.DROP and not self.fallback_weight:
            errors.append('drop decision requires fallback_weight')
        if dataset_profile == 'target_full' and self.primary_mAP is None:
            errors.append('target_full stage_result requires primary_mAP')
        if self.failed_category not in FAILURE_CATEGORIES:
            errors.append(f'failed_category must be one of {FAILURE_CATEGORIES}')
        if errors:
            raise ValueError('; '.join(errors))


def result_from_metrics(stage_id, metrics, decision=Decision.KEEP, reason=''):
    return StageResult(
        stage_id=stage_id,
        decision=decision,
        reason=reason,
        metrics=dict(metrics or {}),
        export_status=(metrics or {}).get('export_status', 'skip'),
        primary_mAP=(metrics or {}).get('primary_mAP'),
        mAP50=(metrics or {}).get('mAP50'),
        small_AP=(metrics or {}).get('small_AP'),
        rare_recall=(metrics or {}).get('rare_recall'),
        FP_per_image=(metrics or {}).get('FP_per_image'),
        FN_per_image=(metrics or {}).get('FN_per_image'),
        GFLOPs=(metrics or {}).get('GFLOPs'),
        GFLOPs_delta_percent=(metrics or {}).get('GFLOPs_delta_percent'),
        python_infer_ms=(metrics or {}).get('python_infer_ms'),
        python_nms_ms=(metrics or {}).get('python_nms_ms'),
        onnx_max_abs_diff=(metrics or {}).get('onnx_max_abs_diff'),
    )
