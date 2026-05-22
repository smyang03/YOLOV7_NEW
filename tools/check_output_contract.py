import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.experimental import attempt_load
from models.yolo import Model
from utils.general import check_img_size, set_logging
from utils.torch_utils import select_device


def normalize_img_size(img_size):
    if len(img_size) == 1:
        return [img_size[0], img_size[0]]
    if len(img_size) == 2:
        return img_size
    raise ValueError('--img expects one square size or H W')


def load_model(weights, cfg, device):
    if weights:
        return attempt_load(weights, map_location=device).eval()
    if cfg:
        return Model(cfg).to(device).eval()
    raise ValueError('Either --weights or --cfg is required')


def flatten_tensors(output):
    if torch.is_tensor(output):
        return [output]
    if isinstance(output, (list, tuple)):
        tensors = []
        for item in output:
            tensors.extend(flatten_tensors(item))
        return tensors
    return []


def per_level_boxes(strides, na, img_size):
    levels = []
    total = 0
    for stride in strides:
        grid_h = int(img_size[0] // stride)
        grid_w = int(img_size[1] // stride)
        boxes = grid_h * grid_w * na
        total += boxes
        levels.append({'stride': stride, 'grid': [grid_h, grid_w], 'anchors': na, 'boxes': boxes})
    return total, levels


def main(opt):
    set_logging()
    device = select_device(opt.device)
    model = load_model(opt.weights, opt.cfg, device)
    detect = model.model[-1]
    strides = [int(x) for x in getattr(model, 'stride', [])]
    stride = max(strides) if strides else 32
    img_size = [check_img_size(x, s=stride) for x in normalize_img_size(opt.img)]
    img = torch.zeros(opt.batch, 3, img_size[0], img_size[1], device=device)

    with torch.no_grad():
        output = model(img)
    tensors = flatten_tensors(output)
    prediction = next((x for x in tensors if x.ndim == 3), None)

    total_boxes, levels = per_level_boxes(strides, int(getattr(detect, 'na', 0)), img_size)
    actual_boxes = int(prediction.shape[1]) if prediction is not None and prediction.ndim == 3 else None
    errors = []
    if opt.expect_levels is not None and len(strides) != opt.expect_levels:
        errors.append(f'expected {opt.expect_levels} levels, got {len(strides)}')
    if opt.expect_strides and strides != opt.expect_strides:
        errors.append(f'expected strides {opt.expect_strides}, got {strides}')
    if actual_boxes is not None and actual_boxes != total_boxes:
        errors.append(f'expected {total_boxes} boxes, got {actual_boxes}')

    result = {
        'schema_version': '1.3.5',
        'weights': opt.weights,
        'cfg': opt.cfg,
        'input_shape': list(img.shape),
        'device': str(device),
        'detect_module': detect.__class__.__name__,
        'strides': strides,
        'detect_layers': len(strides),
        'anchors_shape': list(getattr(detect, 'anchors', torch.empty(0)).shape),
        'main_heads': len(getattr(detect, 'm', [])),
        'aux_heads': len(getattr(detect, 'm2', [])) if hasattr(detect, 'm2') else None,
        'expected_total_boxes': int(total_boxes),
        'actual_total_boxes': actual_boxes,
        'per_level_boxes': levels,
        'output_shapes': [list(x.shape) for x in tensors],
        'status': 'pass' if not errors else 'fail',
        'errors': errors,
    }
    output_path = Path(opt.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'output contract saved to {output_path}')
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help='model weights path')
    parser.add_argument('--cfg', type=str, default='', help='model yaml path if weights are not used')
    parser.add_argument('--img', nargs='+', type=int, default=[640, 640], help='input image size H W')
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--expect-levels', type=int, default=None)
    parser.add_argument('--expect-strides', nargs='*', type=int, default=None)
    parser.add_argument('--output', type=str, required=True)
    main(parser.parse_args())
