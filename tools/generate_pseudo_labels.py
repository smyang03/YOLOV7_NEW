import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.experimental import attempt_load
from utils.datasets import LoadImages, img2label_paths
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from utils.pseudo_label import PseudoLabelGenerator, write_manifest
from utils.torch_utils import select_device, time_synchronized


def data_nc(path):
    if not path:
        return None
    data = yaml.safe_load(Path(path).read_text(encoding='utf-8')) or {}
    return int(data['nc']) if 'nc' in data else None


def main(opt):
    device = select_device(opt.device)
    model = attempt_load(opt.weights, map_location=device).eval()
    stride = int(model.stride.max())
    imgsz = check_img_size(opt.img_size, s=stride)
    generator = PseudoLabelGenerator(
        pseudo_conf=opt.conf_thres,
        pseudo_iou_dedup=opt.gt_dedupe_iou,
        old_nc=opt.old_nc if opt.old_nc >= 0 else data_nc(opt.data))

    dataset = LoadImages(opt.source, img_size=imgsz, stride=stride)
    output = Path(opt.output)
    output.mkdir(parents=True, exist_ok=True)
    stats = []
    image_count = 0
    det_count = 0

    for path, img, im0, _ in dataset:
        if opt.max_images and image_count >= opt.max_images:
            break
        image_count += 1
        img_tensor = torch.from_numpy(img).to(device).float() / 255.0
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)
        t0 = time_synchronized()
        with torch.no_grad():
            pred = model(img_tensor)[0]
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres)
        elapsed = time_synchronized() - t0

        rows = []
        det = pred[0]
        if len(det):
            det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], im0.shape).round()
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            for *xyxy, conf, cls in det:
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                rows.append({
                    'class_id': int(cls),
                    'x': float(xywh[0]),
                    'y': float(xywh[1]),
                    'w': float(xywh[2]),
                    'h': float(xywh[3]),
                    'conf': float(conf),
                })
        label_path = img2label_paths([path])[0]
        image_stats = generator.save_image_labels(path, rows, output, gt_label=label_path)
        image_stats['inference_seconds'] = elapsed
        stats.append(image_stats)
        det_count += image_stats['kept']

    manifest = {
        'weights': opt.weights,
        'source': opt.source,
        'data': opt.data,
        'output': str(output),
        'conf_thres': opt.conf_thres,
        'iou_thres': opt.iou_thres,
        'gt_dedupe_iou': opt.gt_dedupe_iou,
        'image_count': image_count,
        'pseudo_labels': det_count,
        'images': stats if opt.verbose else [],
        'status': 'pass',
    }
    write_manifest(opt.manifest, manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--data', type=str, default='')
    parser.add_argument('--conf-thres', type=float, default=0.5)
    parser.add_argument('--iou-thres', type=float, default=0.45)
    parser.add_argument('--gt-dedupe-iou', type=float, default=0.8)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--manifest', type=str, default='pseudo_label_manifest.json')
    parser.add_argument('--img-size', '--img', dest='img_size', type=int, default=640)
    parser.add_argument('--device', default='')
    parser.add_argument('--old-nc', type=int, default=-1)
    parser.add_argument('--max-images', type=int, default=0)
    parser.add_argument('--verbose', action='store_true')
    main(parser.parse_args())
