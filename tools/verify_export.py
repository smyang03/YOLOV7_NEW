import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.experimental import attempt_load
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
        return [output.detach().float().cpu().numpy()]
    if isinstance(output, (list, tuple)):
        tensors = []
        for item in output:
            tensors.extend(flatten_tensors(item))
        return tensors
    return []


def shape_list(outputs):
    return [list(x.shape) for x in outputs]


def fcos_raw_shapes(outputs):
    shapes = []
    for output in outputs:
        if len(output.shape) == 4 and output.shape[1] >= 6:
            shapes.append(list(output.shape))
    return shapes


def run_pytorch(weights, img, device):
    model = attempt_load(weights, map_location=device).eval()
    stride = int(max(model.stride)) if hasattr(model, 'stride') else 32
    img_size = [check_img_size(x, s=stride) for x in normalize_img_size(img)]
    sample = torch.rand(1, 3, img_size[0], img_size[1], device=device)
    with torch.no_grad():
        outputs = flatten_tensors(model(sample))
    return sample.detach().cpu().numpy(), outputs, stride, img_size


def run_onnx(onnx_path, sample):
    try:
        import onnxruntime as ort
    except Exception as exc:
        return None, f'onnxruntime unavailable: {exc}'

    session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: sample.astype(np.float32)})
    return [np.asarray(x) for x in outputs], None


def compare_outputs(torch_outputs, onnx_outputs):
    if not torch_outputs or not onnx_outputs:
        return {'status': 'skip', 'reason': 'missing outputs'}
    torch_first = torch_outputs[0]
    onnx_first = onnx_outputs[0]
    if torch_first.shape != onnx_first.shape:
        return {
            'status': 'shape_mismatch',
            'torch_shape': list(torch_first.shape),
            'onnx_shape': list(onnx_first.shape),
        }
    diff = np.abs(torch_first - onnx_first)
    return {
        'status': 'ok',
        'max_abs_diff': float(diff.max()),
        'mean_abs_diff': float(diff.mean()),
    }


def main(opt):
    set_logging()
    device = select_device(opt.device)
    sample, torch_outputs, stride, img_size = run_pytorch(opt.weights, opt.img, device)
    onnx_outputs, onnx_error = run_onnx(Path(opt.onnx), sample) if Path(opt.onnx).is_file() else (None, 'onnx file missing')

    torch_fcos_shapes = fcos_raw_shapes(torch_outputs)
    onnx_fcos_shapes = fcos_raw_shapes(onnx_outputs or [])
    schema_version = '1.3.6' if torch_fcos_shapes or onnx_fcos_shapes else '1.3.1'

    comparison = compare_outputs(torch_outputs, onnx_outputs) if onnx_outputs is not None else {
        'status': 'skip',
        'reason': onnx_error,
    }
    result = {
        'schema_version': schema_version,
        'weights': opt.weights,
        'onnx': opt.onnx,
        'input_shape': list(sample.shape),
        'device': str(device),
        'stride': stride,
        'img_size': img_size,
        'torch_output_shapes': shape_list(torch_outputs),
        'onnx_output_shapes': shape_list(onnx_outputs or []),
        'torch_fcos_raw_shapes': torch_fcos_shapes,
        'onnx_fcos_raw_shapes': onnx_fcos_shapes,
        'comparison': comparison,
    }
    contract = {
        'schema_version': schema_version,
        'nms_mode': opt.nms_mode,
        'postprocess_in_graph': opt.nms_mode != 'none',
        'aux_exported': False,
        'fcos_raw_exported': bool(onnx_fcos_shapes or torch_fcos_shapes),
        'input_name': 'images',
        'input_shape': list(sample.shape),
        'output_count': len(onnx_outputs or torch_outputs),
        'output_shapes': shape_list(onnx_outputs or torch_outputs),
    }

    output = Path(opt.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')

    contract_output = Path(opt.contract_output) if opt.contract_output else output.with_name('output_contract.json')
    contract_output.write_text(json.dumps(contract, indent=2), encoding='utf-8')
    print(f'export check saved to {output}')
    print(f'output contract saved to {contract_output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='PyTorch weights path')
    parser.add_argument('--onnx', type=str, required=True, help='ONNX model path')
    parser.add_argument('--img', nargs='+', type=int, default=[640, 640], help='input image size H W')
    parser.add_argument('--device', default='cpu', help='cuda device or cpu')
    parser.add_argument('--output', type=str, required=True, help='export_check.json output path')
    parser.add_argument('--contract-output', type=str, default='', help='output_contract.json path')
    parser.add_argument('--nms-mode', choices=['none', 'end2end'], default='none', help='export NMS mode')
    main(parser.parse_args())
