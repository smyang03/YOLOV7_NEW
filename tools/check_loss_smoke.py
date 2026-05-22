import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.yolo import Model
from utils.loss import ComputeLoss, ComputeLossOTA
from utils.loss_aux import ComputeLossAuxOTA
from utils.loss_components import apply_loss_options, validate_loss_options
from utils.torch_utils import select_device


def default_hyp():
    return {
        'box': 0.05,
        'cls': 0.3,
        'obj': 0.7,
        'cls_pw': 1.0,
        'obj_pw': 1.0,
        'fl_gamma': 0.0,
        'label_smoothing': 0.0,
        'anchor_t': 4.0,
        'loss_ota': 1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/training/yolov7.yaml')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--img', type=int, default=64)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--nc', type=int, default=3)
    parser.add_argument('--head', choices=['coupled', 'decoupled'], default='coupled')
    parser.add_argument('--loss-box', choices=['ciou', 'wiou_v3'], default='ciou')
    parser.add_argument('--assign', choices=['simota', 'tal'], default='simota')
    parser.add_argument('--loss-cls', choices=['bce', 'vfl'], default='bce')
    parser.add_argument('--empty-targets', action='store_true')
    parser.add_argument('--output', type=str, default='')
    opt = parser.parse_args()
    validate_loss_options(opt, parser)

    device = select_device(opt.device, batch_size=opt.batch)
    hyp = apply_loss_options(default_hyp(), opt)
    model = Model(opt.cfg, ch=3, nc=opt.nc, head=opt.head).to(device)
    model.train()
    model.hyp = hyp
    model.gr = 1.0

    imgs = torch.rand(opt.batch, 3, opt.img, opt.img, device=device)
    targets = torch.zeros((0, 6), device=device) if opt.empty_targets else \
        torch.tensor([[0, 0, 0.5, 0.5, 0.25, 0.25]], device=device)
    if opt.batch > 1 and not opt.empty_targets:
        extra = torch.tensor([[opt.batch - 1, min(opt.nc - 1, 1), 0.35, 0.35, 0.2, 0.2]], device=device)
        targets = torch.cat((targets, extra), 0)

    pred = model(imgs)
    det = model.model[-1]
    if isinstance(pred, list) and len(pred) > det.nl:
        compute_loss = ComputeLossAuxOTA(model)
        loss, items = compute_loss(pred, targets, imgs)
        loss_name = 'ComputeLossAuxOTA'
    elif hyp.get('loss_ota', 1) == 1:
        compute_loss = ComputeLossOTA(model)
        loss, items = compute_loss(pred, targets, imgs)
        loss_name = 'ComputeLossOTA'
    else:
        compute_loss = ComputeLoss(model)
        loss, items = compute_loss(pred, targets)
        loss_name = 'ComputeLoss'

    if not torch.isfinite(loss):
        raise RuntimeError(f'non-finite loss: {loss.item()}')
    loss.backward()

    result = {
        'cfg': opt.cfg,
        'head': opt.head,
        'loss_box': opt.loss_box,
        'assign': opt.assign,
        'loss_cls': opt.loss_cls,
        'loss_name': loss_name,
        'loss': float(loss.detach().cpu()),
        'loss_items': [float(x) for x in items.detach().cpu()],
        'positive_count': int(getattr(compute_loss, 'last_positive_count', 0)),
        'empty_targets': bool(opt.empty_targets),
        'status': 'pass',
    }
    text = json.dumps(result, indent=2)
    if opt.output:
        Path(opt.output).write_text(text + '\n', encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
