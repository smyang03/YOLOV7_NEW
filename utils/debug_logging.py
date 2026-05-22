import json
import traceback
from datetime import datetime, timezone
from pathlib import Path


LEVEL_ORDER = {
    'off': 0,
    'error': 1,
    'debug': 2,
    'trace': 3,
}


def _now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec='milliseconds')


def _module_set(modules):
    if modules in (None, '', 'all'):
        return set()
    if isinstance(modules, str):
        return {x.strip() for x in modules.split(',') if x.strip()}
    return {str(x).strip() for x in modules if str(x).strip()}


def safe_summary(value, depth=0):
    if depth > 4:
        return str(type(value).__name__)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [safe_summary(x, depth + 1) for x in list(value)[:20]]
    if isinstance(value, dict):
        return {str(k): safe_summary(v, depth + 1) for k, v in list(value.items())[:50]}
    if hasattr(value, 'shape') and hasattr(value, 'dtype'):
        summary = {
            'type': type(value).__name__,
            'shape': [int(x) for x in getattr(value, 'shape', [])],
            'dtype': str(getattr(value, 'dtype', '')),
        }
        device = getattr(value, 'device', None)
        if device is not None:
            summary['device'] = str(device)
        return summary
    return str(value)


class DebugLogger:
    def __init__(self, save_dir, level='off', modules='', rank=0, debug_file='debug_trace.log',
                 error_file='error_trace.log'):
        self.save_dir = Path(save_dir)
        self.level = str(level or 'off')
        self.level_value = LEVEL_ORDER.get(self.level, 0)
        self.modules = _module_set(modules)
        self.rank = rank
        self.debug_path = self.save_dir / (debug_file or 'debug_trace.log')
        self.error_path = self.save_dir / error_file
        self.active = self.level_value > 0 and rank in (-1, 0)
        if self.active:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.error_path.touch(exist_ok=True)
            if self.level_value >= LEVEL_ORDER['debug']:
                self.debug_path.touch(exist_ok=True)

    def enabled(self, level='debug', module=''):
        if not self.active:
            return False
        required = LEVEL_ORDER.get(level, LEVEL_ORDER['debug'])
        if self.level_value < required:
            return False
        if level == 'error':
            return True
        return not self.modules or module in self.modules

    def _write(self, path, record):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')

    def log_event(self, level, module, function, event, message='', summary=None, **fields):
        level = str(level or 'debug')
        module = str(module or '')
        if not self.enabled(level, module):
            return
        record = {
            'time': _now_iso(),
            'level': level,
            'module': module,
            'function': str(function or ''),
            'event': str(event or ''),
            'message': str(message or ''),
            'summary': safe_summary(summary or {}),
        }
        record.update({str(k): safe_summary(v) for k, v in fields.items() if v is not None})
        self._write(self.error_path if level == 'error' else self.debug_path, record)

    def log_exception(self, module, function, exc, event='exception', message='', summary=None, **fields):
        if not self.enabled('error', module):
            return
        record = {
            'time': _now_iso(),
            'level': 'error',
            'module': str(module or ''),
            'function': str(function or ''),
            'event': str(event or 'exception'),
            'message': str(message or exc),
            'exception_type': type(exc).__name__,
            'summary': safe_summary(summary or {}),
            'traceback': ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        }
        record.update({str(k): safe_summary(v) for k, v in fields.items() if v is not None})
        self._write(self.error_path, record)


def get_debug_logger(save_dir, level='off', modules='', rank=0, debug_file='debug_trace.log'):
    return DebugLogger(save_dir, level=level, modules=modules, rank=rank, debug_file=debug_file)
