import gc
from pathlib import Path

import torch

from utils.datasets import create_dataloader
from utils.general import colorstr


def phase_imgsz(state, default_imgsz):
    return state.train_imgsz or default_imgsz


def phase_rect(state, default_rect):
    return default_rect if state.rect is None else state.rect


def phase_close_mosaic(state, default_close_mosaic=False):
    if state.mosaic is None:
        return default_close_mosaic
    return not state.mosaic


def cleanup_dataloader(*objects):
    for obj in objects:
        del obj
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def save_aug_debug_samples(dataset, save_dir, samples, names=None, filename='aug_debug_samples.jpg'):
    samples = min(max(int(samples), 0), len(dataset), 16)
    if samples <= 0:
        return None
    from utils.plots import plot_images

    batch = [dataset[i] for i in range(samples)]
    imgs, labels, paths, _ = dataset.collate_fn(batch)
    output_dir = Path(save_dir) / 'aug_debug'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    plot_images(imgs, labels, paths, fname=str(output_path), names=names)
    return output_path


def build_train_dataloader(train_path, imgsz, batch_size, gs, opt, hyp, rank, rect, close_mosaic,
                           force_mosaic_off=False, allow_rect_mosaic=False, aug_phase=None):
    return create_dataloader(
        train_path, imgsz, batch_size, gs, opt,
        hyp=hyp, augment=True, cache=opt.cache_images, rect=rect, rank=rank,
        world_size=opt.world_size, workers=opt.workers,
        image_weights=opt.image_weights, quad=opt.quad, prefix=colorstr('train: '),
        close_mosaic=close_mosaic, force_mosaic_off=force_mosaic_off,
        allow_rect_mosaic=allow_rect_mosaic, aug_phase=aug_phase)


def build_val_dataloaders(val_configs, imgsz_test, batch_size, gs, opt, hyp):
    testloaders = []
    for val_cfg in val_configs:
        val_path = val_cfg['path']
        val_name = val_cfg['name']
        testloader = create_dataloader(
            val_path, imgsz_test, batch_size * 2, gs, opt,
            hyp=hyp, cache=opt.cache_images and not opt.notest, rect=True, rank=-1,
            world_size=opt.world_size, workers=opt.workers,
            pad=0.5, prefix=colorstr(f'{val_name}: '),
            close_mosaic=False)[0]
        testloaders.append((val_name, testloader))
    return testloaders
