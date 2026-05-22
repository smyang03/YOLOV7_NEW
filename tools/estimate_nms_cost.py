import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.experimental import attempt_load
from models.yolo import Model
from utils.general import check_img_size, non_max_suppression, set_logging
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


def per_level_boxes(model, img_size):
    detect = model.model[-1]
    strides = [int(x) for x in getattr(detect, 'stride', getattr(model, 'stride', []))]
    na = int(getattr(detect, 'na', 0))
    levels = []
    total = 0
    for stride in strides:
        grid_h = int(img_size[0] // stride)
        grid_w = int(img_size[1] // stride)
        boxes = grid_h * grid_w * na
        levels.append({'stride': stride, 'grid': [grid_h, grid_w], 'anchors': na, 'boxes': boxes})
        total += boxes
    return total, levels


def random_predictions(batch, total_boxes, nc, img_size, device, candidate_ratio):
    pred = torch.rand(batch, total_boxes, nc + 5, device=device)
    pred[..., 0] *= img_size[1]
    pred[..., 1] *= img_size[0]
    pred[..., 2] = pred[..., 2] * img_size[1] * 0.15 + 2
    pred[..., 3] = pred[..., 3] * img_size[0] * 0.15 + 2
    pred[..., 4] = 0.01
    candidates = max(1, int(total_boxes * candidate_ratio))
    idx = torch.randperm(total_boxes, device=device)[:candidates]
    pred[:, idx, 4] = torch.rand(batch, candidates, device=device) * 0.5 + 0.5
    pred[:, idx, 5:] = torch.rand(batch, candidates, nc, device=device) * 0.5 + 0.5
    return pred


def main(opt):
    set_logging()
    device = select_device(opt.device)
    model = load_model(opt.weights, opt.cfg, device)
    stride = int(max(model.stride)) if hasattr(model, 'stride') else 32
    img_size = [check_img_size(x, s=stride) for x in normalize_img_size(opt.img)]
    total_boxes, levels = per_level_boxes(model, img_size)
    nc = int(getattr(model.model[-1], 'nc', getattr(model, 'nc', 80)))

    # Warmup once outside timing.
    pred = random_predictions(opt.batch, total_boxes, nc, img_size, device, opt.candidate_ratio)
    non_max_suppression(pred, opt.conf_thres, opt.iou_thres)

    times = []
    for _ in range(opt.runs):
        pred = random_predictions(opt.batch, total_boxes, nc, img_size, device, opt.candidate_ratio)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        non_max_suppression(pred, opt.conf_thres, opt.iou_thres)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        times.append((time.perf_counter() - start) * 1000.0)

    result = {
        'schema_version': '1.3.5',
        'weights': opt.weights,
        'cfg': opt.cfg,
        'input_shape': [opt.batch, 3, img_size[0], img_size[1]],
        'per_level_boxes': levels,
        'total_boxes': int(total_boxes),
        'candidate_ratio': opt.candidate_ratio,
        'conf_thres': opt.conf_thres,
        'iou_thres': opt.iou_thres,
        'mean_nms_ms': statistics.fmean(times) if times else None,
        'p95_nms_ms': sorted(times)[max(0, int(len(times) * 0.95) - 1)] if times else None,
        'runs': opt.runs,
        'device': str(device),
        'status': 'pass',
    }
    output = Path(opt.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'nms cost saved to {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help='model weights path')
    parser.add_argument('--cfg', type=str, default='', help='model yaml path if weights are not used')
    parser.add_argument('--img', nargs='+', type=int, default=[640, 640], help='input image size H W')
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--conf-thres', type=float, default=0.25)
    parser.add_argument('--iou-thres', type=float, default=0.45)
    parser.add_argument('--candidate-ratio', type=float, default=0.02)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--output', type=str, required=True)
    main(parser.parse_args())
