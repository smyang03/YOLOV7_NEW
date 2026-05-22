import json
from pathlib import Path

import yaml


TEXT_ENCODINGS = ('utf-8', 'cp949', 'euc-kr', 'latin1', 'utf-16')


def read_text_fallback(path):
    last_error = None
    for encoding in TEXT_ENCODINGS:
        try:
            return Path(path).read_text(encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error


def load_mapping_file(path):
    if not path:
        return {}
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'class mapping file not found: {path}')
    text = read_text_fallback(path)
    data = json.loads(text) if path.suffix.lower() == '.json' else yaml.safe_load(text)
    data = data or {}
    for key in ('old_to_new', 'mapping', 'class_mapping', 'resolved_mapping'):
        if isinstance(data, dict) and key in data:
            return data[key] or {}
    return data


def normalize_names(names):
    if isinstance(names, dict):
        try:
            items = sorted(names.items(), key=lambda kv: int(kv[0]))
        except ValueError as exc:
            raise ValueError('names dict keys must be integer-like class indexes') from exc
        indexes = [int(k) for k, _ in items]
        if indexes != list(range(len(indexes))):
            raise ValueError(f'names dict indexes must be continuous from 0, got {indexes}')
        return [str(v) for _, v in items]
    if isinstance(names, (list, tuple)):
        return [str(x) for x in names]
    raise ValueError('data yaml names must be a list or index:name dict')


def load_data_yaml(path):
    path = Path(path)
    data = yaml.safe_load(read_text_fallback(path)) or {}
    names = normalize_names(data.get('names', []))
    errors = []
    try:
        nc = int(data.get('nc', len(names)))
    except (TypeError, ValueError):
        nc = len(names)
        errors.append(f'{path}: nc must be an integer')
    if nc != len(names):
        errors.append(f'{path}: nc={nc} but names has {len(names)} entries')
    if len(set(names)) != len(names):
        errors.append(f'{path}: duplicate class names are not allowed')
    return {
        'path': str(path),
        'raw': data,
        'nc': nc,
        'names': names,
        'errors': errors,
    }


def _resolve_mapping_value(mapping_data, old_index, old_name, new_names):
    candidates = (old_index, str(old_index), old_name)
    value = None
    for key in candidates:
        if key in mapping_data:
            value = mapping_data[key]
            break
    if value is None:
        return None
    if isinstance(value, dict):
        for key in ('new_index', 'target_index', 'to', 'new'):
            if key in value:
                value = value[key]
                break
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        if value.isdigit():
            return int(value)
        if value in new_names:
            return new_names.index(value)
    raise ValueError(f'invalid mapping value for old class {old_index}:{old_name}: {value}')


def validate_class_mapping(base_data, data, mapping_file='', output=''):
    base = load_data_yaml(base_data)
    new = load_data_yaml(data)
    mapping_data = load_mapping_file(mapping_file)
    errors = list(base['errors']) + list(new['errors'])
    warnings = []
    resolved = {}

    new_name_to_index = {name: i for i, name in enumerate(new['names'])}
    for old_index, old_name in enumerate(base['names']):
        same_index = old_index < len(new['names']) and new['names'][old_index] == old_name
        explicit_index = _resolve_mapping_value(mapping_data, old_index, old_name, new['names']) if mapping_data else None

        if same_index:
            resolved[str(old_index)] = {
                'old_index': old_index,
                'old_name': old_name,
                'new_index': old_index,
                'new_name': old_name,
                'source': 'same_index',
            }
            continue

        if old_index < len(new['names']) and new['names'][old_index] != old_name and not mapping_data:
            errors.append(
                f'old index {old_index} class "{old_name}" is occupied by new class "{new["names"][old_index]}"; '
                'provide a mapping file or keep old indexes stable')

        if old_name in new_name_to_index:
            new_index = new_name_to_index[old_name]
            if not mapping_data:
                errors.append(
                    f'class "{old_name}" moved from index {old_index} to {new_index}; mapping file required')
                continue
            if explicit_index != new_index:
                errors.append(
                    f'class "{old_name}" mapping mismatch: file maps to {explicit_index}, data has {new_index}')
                continue
            resolved[str(old_index)] = {
                'old_index': old_index,
                'old_name': old_name,
                'new_index': new_index,
                'new_name': new['names'][new_index],
                'source': 'mapping_file',
            }
            continue

        if explicit_index is None:
            errors.append(f'old class "{old_name}" at index {old_index} is missing from new data yaml')
            continue
        if explicit_index < 0 or explicit_index >= len(new['names']):
            errors.append(f'class "{old_name}" maps to out-of-range new index {explicit_index}')
            continue
        warnings.append(
            f'class "{old_name}" is renamed/mapped to "{new["names"][explicit_index]}" at index {explicit_index}')
        resolved[str(old_index)] = {
            'old_index': old_index,
            'old_name': old_name,
            'new_index': explicit_index,
            'new_name': new['names'][explicit_index],
            'source': 'mapping_file_rename',
        }

    protected_indexes = {int(v['new_index']) for v in resolved.values()}
    new_only = [
        {'index': i, 'name': name}
        for i, name in enumerate(new['names'])
        if i not in protected_indexes
    ]
    result = {
        'schema_version': '1.3.7',
        'base_data': base['path'],
        'data': new['path'],
        'mapping_file': str(mapping_file) if mapping_file else '',
        'base_nc': base['nc'],
        'new_nc': new['nc'],
        'base_names': base['names'],
        'new_names': new['names'],
        'resolved_mapping': resolved,
        'new_only_classes': new_only,
        'warnings': warnings,
        'errors': errors,
        'status': 'pass' if not errors else 'fail',
    }
    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return result
