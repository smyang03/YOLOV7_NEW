import argparse
import json
import sys
from copy import deepcopy
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
        return attempt_load(weights, map_location=device)
    if cfg:
        return Model(cfg).to(device)
    raise ValueError('Either --weights or --cfg is required')


def compute_gflops(model, img):
    try:
        from thop import profile

        return profile(deepcopy(model), inputs=(img,), verbose=False)[0] / 1E9 * 2
    except Exception as exc:
        return None, str(exc)


def flatten_tensors(output):
    if torch.is_tensor(output):
        return [output]
    if isinstance(output, (list, tuple)):
        tensors = []
        for item in output:
            tensors.extend(flatten_tensors(item))
        return tensors
    return []


def estimate_boxes(model, img_size):
    detect = model.model[-1]
    strides = [int(x) for x in getattr(detect, 'stride', getattr(model, 'stride', []))]
    na = int(getattr(detect, 'na', 0))
    per_level = []
    total = 0
    for stride in strides:
        grid_h = int(img_size[0] // stride)
        grid_w = int(img_size[1] // stride)
        boxes = grid_h * grid_w * na
        total += boxes
        per_level.append({
            'stride': stride,
            'grid': [grid_h, grid_w],
            'anchors': na,
            'boxes': boxes,
        })
    return total, per_level


def main(opt):
    set_logging()
    device = select_device(opt.device)
    model = load_model(opt.weights, opt.cfg, device).eval()

    stride = int(max(model.stride)) if hasattr(model, 'stride') else 32
    img_size = [check_img_size(x, s=stride) for x in normalize_img_size(opt.img)]
    img = torch.zeros(opt.batch, 3, img_size[0], img_size[1], device=device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    params = sum(p.numel() for p in model.parameters())
    gradients = sum(p.numel() for p in model.parameters() if p.requires_grad)
    gflops_result = compute_gflops(model, img)
    if isinstance(gflops_result, tuple):
        gflops, gflops_error = gflops_result
    else:
        gflops, gflops_error = gflops_result, None

    with torch.no_grad():
        outputs = flatten_tensors(model(img))
    output_shapes = [list(x.shape) for x in outputs]
    activation_memory_mb = sum(x.numel() * x.element_size() for x in outputs) / (1024 ** 2)
    max_cuda_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == 'cuda' else None
    total_boxes, per_level = estimate_boxes(model, img_size)
    detect = model.model[-1]
    strides = [level['stride'] for level in per_level]

    result = {
        'schema_version': '1.3.5',
        'weights': opt.weights,
        'cfg': opt.cfg,
        'input_shape': list(img.shape),
        'device': str(device),
        'stride': stride,
        'strides': strides,
        'detect_layers': len(strides),
        'anchors_shape': list(getattr(detect, 'anchors', torch.empty(0)).shape),
        'layers': len(list(model.modules())),
        'parameters': int(params),
        'gradients': int(gradients),
        'gflops': gflops,
        'gflops_error': gflops_error,
        'baseline_gflops': opt.baseline_gflops,
        'current_gflops': gflops,
        'gflops_delta_percent': ((gflops - opt.baseline_gflops) / opt.baseline_gflops * 100.0)
        if gflops is not None and opt.baseline_gflops else None,
        'total_boxes': int(total_boxes),
        'per_level_boxes': per_level,
        'output_shapes': output_shapes,
        'activation_memory_mb': activation_memory_mb,
        'max_cuda_memory_mb': max_cuda_memory_mb,
        'small_AP': None,
        'small_recall': None,
        'rare_recall': None,
    }

    output = Path(opt.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'profile saved to {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help='model weights path')
    parser.add_argument('--cfg', type=str, default='', help='model yaml path if weights are not used')
    parser.add_argument('--img', nargs='+', type=int, default=[640, 640], help='input image size H W')
    parser.add_argument('--batch', type=int, default=1, help='input batch size')
    parser.add_argument('--device', default='cpu', help='cuda device or cpu')
    parser.add_argument('--baseline-gflops', type=float, default=None, help='baseline GFLOPs for delta calculation')
    parser.add_argument('--output', type=str, required=True, help='profile.json output path')
    main(parser.parse_args())
