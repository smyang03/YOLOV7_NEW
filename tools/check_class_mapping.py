import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.class_mapping import validate_class_mapping


def main(opt):
    result = validate_class_mapping(
        opt.base_data, opt.data,
        mapping_file=opt.mapping_file,
        output=opt.output)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if result['status'] != 'pass':
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-data', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--mapping-file', type=str, default='')
    parser.add_argument('--output', type=str, default='class_mapping_check.json')
    main(parser.parse_args())
