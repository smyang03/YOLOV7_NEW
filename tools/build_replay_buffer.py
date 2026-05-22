import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.replay_buffer import ReplayBufferBuilder


def main(opt):
    builder = ReplayBufferBuilder(
        opt.data, split=opt.split,
        replay_ratio=opt.replay_ratio,
        seed=opt.seed)
    result = builder.build(output=opt.output, copy_dir=opt.copy_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result['status'] not in ('pass', 'warn'):
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--replay-ratio', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=str, default='replay_manifest.json')
    parser.add_argument('--copy-dir', type=str, default='')
    main(parser.parse_args())
