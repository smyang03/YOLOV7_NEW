import argparse
import json
import re
from pathlib import Path


NUMERIC_RE = re.compile(r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?')
DEFAULT_COLUMNS = [
    'epoch',
    'gpu_mem',
    'box',
    'obj',
    'cls',
    'total',
    'labels',
    'img_size',
    'precision',
    'recall',
    'map50',
    'map50_95',
    'val_box',
    'val_obj',
    'val_cls',
]


def parse_rows(path):
    rows = []
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        values = [float(x) for x in NUMERIC_RE.findall(line)]
        if values:
            rows.append(values)
    return rows


def row_to_metrics(row):
    metrics = {}
    for i, value in enumerate(row):
        key = DEFAULT_COLUMNS[i] if i < len(DEFAULT_COLUMNS) else f'extra_{i}'
        metrics[key] = value
    return metrics


def main(opt):
    rows = parse_rows(opt.results)
    if not rows:
        raise ValueError(f'No numeric rows found in {opt.results}')

    last = row_to_metrics(rows[-1])
    best_row = max(rows, key=lambda x: x[11] if len(x) > 11 else x[-1])
    best = row_to_metrics(best_row)

    result = {
        'schema_version': '1.3.1',
        'results': opt.results,
        'row_count': len(rows),
        'primary_metric': 'map50_95' if 'map50_95' in last else 'last_numeric',
        'last': last,
        'best': best,
    }

    output = Path(opt.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'baseline metrics saved to {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str, required=True, help='results.txt path')
    parser.add_argument('--output', type=str, required=True, help='baseline_metrics.json output path')
    main(parser.parse_args())
