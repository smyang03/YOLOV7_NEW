import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.experimental import attempt_load
from utils.fcos import fcos_contract
from utils.general import check_img_size, set_logging
from utils.torch_utils import select_device


def normalize_img_size(img_size):
    if len(img_size) == 1:
        return [img_size[0], img_size[0]]
    if len(img_size) == 2:
        return img_size
    raise ValueError('--img expects one square size or H W')


def flatten_tensors(output):
    if torch.is_tensor(output):
        return [output]
    if isinstance(output, (list, tuple)):
        tensors = []
        for item in output:
            tensors.extend(flatten_tensors(item))
        return tensors
    return []


def load_nc(data, fallback):
    if not data:
        return fallback
    path = Path(data)
    if not path.is_file():
        path = ROOT / data
    if not path.is_file():
        return fallback
    with open(path, encoding='utf-8') as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader) or {}
    return int(data_dict.get('nc', fallback))


def load_raw_tensor(opt, device, nc):
    if opt.raw:
        raw = torch.load(opt.raw, map_location=device)
        if isinstance(raw, dict):
            if 'fcos_raw' in raw:
                raw = raw['fcos_raw']
            elif 'raw' in raw:
                raw = raw['raw']
            else:
                raw = next(iter(raw.values()))
        return raw.to(device).float(), 'raw_file'

    if opt.weights:
        model = attempt_load(opt.weights, map_location=device).eval()
        stride = int(max(model.stride)) if hasattr(model, 'stride') else opt.stride
        img_size = [check_img_size(x, s=stride) for x in normalize_img_size(opt.img)]
        sample = torch.zeros(opt.batch, 3, img_size[0], img_size[1], device=device)
        with torch.no_grad():
            outputs = flatten_tensors(model(sample))
        candidates = [x for x in outputs if x.ndim == 4 and x.shape[1] == nc + 5]
        if candidates:
            return candidates[0].float(), 'model_output'
        if not opt.allow_synthetic:
            return None, 'no_fcos_raw_output'

    if not opt.allow_synthetic:
        return None, 'no_fcos_raw_output'

    h = max(1, normalize_img_size(opt.img)[0] // opt.stride)
    w = max(1, normalize_img_size(opt.img)[1] // opt.stride)
    return torch.randn(opt.batch, nc + 5, h, w, device=device), 'synthetic'


def main(opt):
    set_logging()
    device = select_device(opt.device)
    nc = load_nc(opt.data, opt.nc)
    raw, source = load_raw_tensor(opt, device, nc)

    result = {
        'schema_version': '1.3.6',
        'weights': opt.weights,
        'raw': opt.raw,
        'data': opt.data,
        'device': str(device),
        'source': source,
        'anchor_output_combined': False,
        'nms_input_combined': False,
        'status': 'skip' if raw is None else 'pass',
    }
    if raw is None:
        result['reason'] = 'No FCOS raw output was found. Use --raw or --allow-synthetic for decode validation.'
    else:
        result.update(fcos_contract(
            raw,
            stride=opt.stride,
            img_size=normalize_img_size(opt.img),
            conf_thres=opt.conf_thres,
            topk=opt.topk,
            score_mode=opt.score_mode,
        ))

    output = Path(opt.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'fcos decode check saved to {output}')
    if result['status'] != 'pass' and opt.require_pass:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help='optional model weights path')
    parser.add_argument('--raw', type=str, default='', help='torch tensor/dict containing FCOS raw BCHW output')
    parser.add_argument('--data', type=str, default='', help='dataset yaml used to infer nc')
    parser.add_argument('--img', nargs='+', type=int, default=[640, 640], help='input image size H W')
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--nc', type=int, default=80)
    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--conf-thres', type=float, default=0.25)
    parser.add_argument('--topk', type=int, default=1000)
    parser.add_argument('--score-mode', choices=['sqrt_cls_centerness', 'mul_cls_centerness'],
                        default='sqrt_cls_centerness')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--allow-synthetic', action='store_true')
    parser.add_argument('--require-pass', action='store_true')
    parser.add_argument('--output', type=str, required=True)
    main(parser.parse_args())
