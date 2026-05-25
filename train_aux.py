import argparse
import logging
import math
import os
import random
import time
from copy import deepcopy
from pathlib import Path
from threading import Thread

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import test  # import test.py to get mAP after each epoch
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, one_cycle, colorstr, sanitize_yaml_value
from utils.google_utils import attempt_download
from utils.early_stopping import PhaseEarlyStopping
from utils.augment_policy import ensure_aug_option_defaults, validate_aug_options
from utils.debug_logging import get_debug_logger
from utils.loss_aux import ComputeLoss, ComputeLossAuxOTA
from utils.loss_components import apply_loss_options, build_loss_state, ensure_loss_option_defaults, \
    get_loss_positive_count, load_loss_state, validate_loss_options
from utils.model_options import ensure_structure_option_defaults, validate_structure_options
from utils.phase import PhaseConfig, phase_changed, resolve_phase
from utils.plots import plot_images, plot_labels, plot_results, plot_evolution
from utils.sampler import log_sampler_stats
from utils.train_common import build_train_dataloader, build_val_dataloaders, cleanup_dataloader, phase_close_mosaic, \
    phase_imgsz, phase_rect, save_aug_debug_samples
from utils.train_logger import TrainLogger
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, is_parallel
from utils.wandb_logging.wandb_utils import WandbLogger, check_wandb_resume
import datetime
import gc
logger = logging.getLogger(__name__)


def format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f'{hours:d}h{minutes:02d}m{seconds:02d}s'
    if minutes:
        return f'{minutes:d}m{seconds:02d}s'
    return f'{seconds:d}s'


def log_epoch_start(opt, epoch, epochs, phase, batches):
    logger.info(
        '[progress] stage=%s epoch=%d/%d start phase=%s batches=%d total_batch=%s world_size=%s',
        Path(opt.save_dir).name, epoch + 1, epochs, phase, batches,
        getattr(opt, 'total_batch_size', getattr(opt, 'batch_size', '')),
        getattr(opt, 'world_size', 1))


def log_epoch_end(opt, epoch, epochs, start_epoch, run_start_time, epoch_start_time,
                  phase, mloss, results, is_best):
    now = time.time()
    epoch_seconds = now - epoch_start_time
    completed = max(1, epoch - start_epoch + 1)
    remaining = max(0, epochs - epoch - 1)
    avg_seconds = (now - run_start_time) / completed
    eta_seconds = avg_seconds * remaining
    losses = [float(x) for x in list(mloss)]
    metrics = [float(x) for x in list(results)]
    logger.info(
        '[progress] stage=%s epoch=%d/%d done phase=%s epoch_time=%s avg_epoch=%s elapsed=%s eta=%s '
        'loss(box/obj/cls/total)=%.4g/%.4g/%.4g/%.4g P=%.4g R=%.4g mAP50=%.4g mAP50-95=%.4g best=%s',
        Path(opt.save_dir).name, epoch + 1, epochs, phase,
        format_duration(epoch_seconds), format_duration(avg_seconds),
        format_duration(now - run_start_time), format_duration(eta_seconds),
        losses[0], losses[1], losses[2], losses[3],
        metrics[0], metrics[1], metrics[2], metrics[3], bool(is_best))


def log_batch_progress(opt, epoch, epochs, batch_i, total_batches, epoch_start_time, mloss, targets, imgs):
    interval = int(getattr(opt, 'progress_log_interval', 50) or 0)
    batch_num = batch_i + 1
    if interval <= 0 or (batch_num % interval and batch_num != total_batches):
        return
    elapsed = time.time() - epoch_start_time
    seconds_per_batch = elapsed / max(1, batch_num)
    eta = seconds_per_batch * max(0, total_batches - batch_num)
    losses = [float(x) for x in list(mloss)]
    mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0.0
    logger.info(
        '[progress] stage=%s epoch=%d/%d batch=%d/%d epoch_elapsed=%s epoch_eta=%s '
        'gpu_mem=%.3gG loss=%.4g/%.4g/%.4g/%.4g labels=%d img=%d',
        Path(opt.save_dir).name, epoch + 1, epochs, batch_num, total_batches,
        format_duration(elapsed), format_duration(eta), mem,
        losses[0], losses[1], losses[2], losses[3], int(targets.shape[0]), int(imgs.shape[-1]))


def stage_id_from_opt(opt):
    aug_logging = getattr(opt, 'aug_profile', 'off') != 'off' or getattr(opt, 'sampler_mode', 'off') not in ('off', 'none')
    core_logging = any((
        getattr(opt, 'head', 'coupled') != 'coupled',
        getattr(opt, 'loss_box', 'ciou') != 'ciou',
        getattr(opt, 'assign', 'simota') != 'simota',
        getattr(opt, 'loss_cls', 'bce') != 'bce',
    ))
    if aug_logging:
        return '1.3.4'
    if getattr(opt, 'phase_train', 'off') == 'on':
        return '1.3.2'
    if core_logging:
        return '1.3.3'
    return '1.3.1'


def write_failed_run_artifacts(opt, exc):
    save_dir = getattr(opt, 'save_dir', None)
    if not save_dir:
        return
    train_logger = TrainLogger(save_dir, getattr(opt, 'log_format', 'both'),
                               getattr(opt, 'per_class_log_interval', 10))
    stage_id = stage_id_from_opt(opt)
    result = train_logger.write_stage_result(
        stage=stage_id,
        stage_id=stage_id,
        decision='blocker',
        reason=f'{type(exc).__name__}: {exc}',
        status='failed',
        hard_fail=True,
        failed_category='train',
        current_run=str(save_dir),
        exception_type=type(exc).__name__)
    train_logger.write_run_summary(result)



def train(hyp, opt, device, tb_writer=None):
    apply_loss_options(hyp, opt)
    logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    save_dir, epochs, batch_size, total_batch_size, weights, rank = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank

    # Directories
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
    results_file = save_dir / 'results.txt'
    phase_config = PhaseConfig.from_opt(opt)
    aug_logging = getattr(opt, 'aug_profile', 'off') != 'off' or getattr(opt, 'sampler_mode', 'off') not in ('off', 'none')
    train_logger = TrainLogger(save_dir, getattr(opt, 'log_format', 'both'),
                               getattr(opt, 'per_class_log_interval', 10)) if rank in [-1, 0] else None
    debug_logger = get_debug_logger(save_dir,
                                    getattr(opt, 'debug_log', 'off'),
                                    getattr(opt, 'debug_log_modules', ''),
                                    rank=rank,
                                    debug_file=getattr(opt, 'debug_log_file', 'debug_trace.log'))
    debug_logger.log_event(
        'debug', 'train_aux', 'train', 'start', 'training started',
        summary={
            'save_dir': str(save_dir),
            'epochs': epochs,
            'batch_size': batch_size,
            'weights': weights,
            'phase_train': phase_config.enabled,
            'log_format': getattr(opt, 'log_format', 'both'),
        })

    # Save run settings
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(sanitize_yaml_value(vars(opt)), f, sort_keys=False)

    # Configure
    plots = not opt.evolve  # create plots
    cuda = device.type != 'cpu'
    init_seeds(2 + rank)
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    is_coco = opt.data.endswith('coco.yaml')

    # Logging- Doing this before checking the dataset. Might update data_dict
    loggers = {'wandb': None}  # loggers dict
    if rank in [-1, 0]:
        opt.hyp = hyp  # add hyperparameters
        run_id = torch.load(weights).get('wandb_id') if weights.endswith('.pt') and os.path.isfile(weights) else None
        wandb_logger = WandbLogger(opt, Path(opt.save_dir).stem, run_id, data_dict)
        loggers['wandb'] = wandb_logger.wandb
        data_dict = wandb_logger.data_dict
        if wandb_logger.wandb:
            weights, epochs, hyp = opt.weights, opt.epochs, opt.hyp  # WandbLogger might update weights, epochs if resuming

    nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes
    opt.nc = nc
    names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check

    # Model
    pretrained = weights.endswith('.pt')
    loss_state_resume = None
    if pretrained:
        with torch_distributed_zero_first(rank):
            attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, map_location=device)  # load checkpoint
        loss_state_resume = ckpt.get('loss_state')
        model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors'),
                      head=getattr(opt, 'head', 'coupled')).to(device)  # create
        exclude = ['anchor'] if (opt.cfg or hyp.get('anchors')) and not opt.resume else []  # exclude keys
        state_dict = ckpt['model'].float().state_dict()  # to FP32
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(state_dict, strict=False)  # load
        logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))  # report
    else:
        model = Model(opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors'),
                      head=getattr(opt, 'head', 'coupled')).to(device)  # create
    with torch_distributed_zero_first(rank):
        check_dataset(data_dict)  # check
    train_path = data_dict['train']

    # Support both single validation set (string) and multiple validation sets (list)
    val_config = data_dict['val']
    if isinstance(val_config, str):
        val_configs = [{'path': val_config, 'name': 'val'}]
    elif isinstance(val_config, list):
        # Check if it's a list of strings or list of dicts
        if all(isinstance(v, str) for v in val_config):
            val_configs = [{'path': v, 'name': f'val_{i}'} for i, v in enumerate(val_config)]
        elif all(isinstance(v, dict) for v in val_config):
            # List of dicts with 'path' and optional 'name' keys
            val_configs = [{'path': v.get('path', v), 'name': v.get('name', f'val_{i}')} for i, v in enumerate(val_config)]
        else:
            raise ValueError(f"Invalid val config: mixed types in list")
    else:
        raise ValueError(f"Invalid val config type: {type(val_config)}")

    logger.info(f"Validation sets: {[cfg['name'] for cfg in val_configs]}")

    # Validate that we have at least one validation set
    if len(val_configs) == 0:
        raise ValueError("No validation sets found in data.yaml. Please specify at least one validation set.")

    # Freeze
    freeze = []  # parameter names to freeze (full or partial)
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print('freezing %s' % k)
            v.requires_grad = False

    # Optimizer
    nbs = 64  # nominal batch size
    base_accumulate = int(opt.grad_accumulate) if getattr(opt, 'grad_accumulate', None) else max(round(nbs / total_batch_size), 1)
    accumulate = max(base_accumulate, 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay
    logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay
        if hasattr(v, 'im'):
            if hasattr(v.im, 'implicit'):           
                pg0.append(v.im.implicit)
            else:
                for iv in v.im:
                    pg0.append(iv.implicit)
        if hasattr(v, 'imc'):
            if hasattr(v.imc, 'implicit'):           
                pg0.append(v.imc.implicit)
            else:
                for iv in v.imc:
                    pg0.append(iv.implicit)
        if hasattr(v, 'imb'):
            if hasattr(v.imb, 'implicit'):           
                pg0.append(v.imb.implicit)
            else:
                for iv in v.imb:
                    pg0.append(iv.implicit)
        if hasattr(v, 'imo'):
            if hasattr(v.imo, 'implicit'):           
                pg0.append(v.imo.implicit)
            else:
                for iv in v.imo:
                    pg0.append(iv.implicit)
        if hasattr(v, 'ia'):
            if hasattr(v.ia, 'implicit'):           
                pg0.append(v.ia.implicit)
            else:
                for iv in v.ia:
                    pg0.append(iv.implicit)
        if hasattr(v, 'attn'):
            if hasattr(v.attn, 'logit_scale'):   
                pg0.append(v.attn.logit_scale)
            if hasattr(v.attn, 'q_bias'):   
                pg0.append(v.attn.q_bias)
            if hasattr(v.attn, 'v_bias'):  
                pg0.append(v.attn.v_bias)
            if hasattr(v.attn, 'relative_position_bias_table'):  
                pg0.append(v.attn.relative_position_bias_table)
        if hasattr(v, 'rbr_dense'):
            if hasattr(v.rbr_dense, 'weight_rbr_origin'):  
                pg0.append(v.rbr_dense.weight_rbr_origin)
            if hasattr(v.rbr_dense, 'weight_rbr_avg_conv'): 
                pg0.append(v.rbr_dense.weight_rbr_avg_conv)
            if hasattr(v.rbr_dense, 'weight_rbr_pfir_conv'):  
                pg0.append(v.rbr_dense.weight_rbr_pfir_conv)
            if hasattr(v.rbr_dense, 'weight_rbr_1x1_kxk_idconv1'): 
                pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_idconv1)
            if hasattr(v.rbr_dense, 'weight_rbr_1x1_kxk_conv2'):   
                pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_conv2)
            if hasattr(v.rbr_dense, 'weight_rbr_gconv_dw'):   
                pg0.append(v.rbr_dense.weight_rbr_gconv_dw)
            if hasattr(v.rbr_dense, 'weight_rbr_gconv_pw'):   
                pg0.append(v.rbr_dense.weight_rbr_gconv_pw)
            if hasattr(v.rbr_dense, 'vector'):   
                pg0.append(v.rbr_dense.vector)

    if opt.adam:
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    if opt.linear_lr:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    else:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # plot_lr_scheduler(optimizer, scheduler, epochs)

    # EMA
    ema = ModelEMA(model) if rank in [-1, 0] else None

    # Resume
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        # Optimizer
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        # EMA
        if ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            ema.updates = ckpt['updates']

        # Results
        if ckpt.get('training_results') is not None:
            results_file.write_text(ckpt['training_results'])  # write results.txt

        # Epochs
        start_epoch = ckpt['epoch'] + 1
        if opt.resume:
            assert start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (weights, epochs)
        if epochs < start_epoch:
            logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                        (weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt, state_dict

    # Image sizes
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples
    base_imgsz, base_imgsz_test = imgsz, imgsz_test
    phase_state = resolve_phase(start_epoch, phase_config)
    active_phase_name = phase_state.name
    imgsz, imgsz_test = phase_imgsz(phase_state, base_imgsz), phase_imgsz(phase_state, base_imgsz_test)
    if train_logger:
        train_logger.write_hyp_snapshot(start_epoch, active_phase_name, hyp)

    # DP mode
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if opt.sync_bn and cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        logger.info('Using SyncBatchNorm()')

    dataloader, dataset = build_train_dataloader(
        train_path, imgsz, batch_size, gs, opt, hyp, rank,
        rect=phase_rect(phase_state, opt.rect),
        close_mosaic=phase_close_mosaic(phase_state, opt.close_mosaic > 0),
        force_mosaic_off=phase_config.enabled and phase_close_mosaic(phase_state, False),
        allow_rect_mosaic=phase_config.enabled and phase_rect(phase_state, opt.rect) and phase_state.mosaic is True,
        aug_phase=active_phase_name)

    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    nb = len(dataloader)  # number of batches
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)
    if rank in [-1, 0] and getattr(opt, 'aug_debug_samples', 0) > 0:
        save_aug_debug_samples(
            dataset, save_dir, opt.aug_debug_samples, names,
            filename=f'aug_debug_{active_phase_name}_epoch{start_epoch}.jpg')

    # Process 0
    if rank in [-1, 0]:
        # Create dataloaders for all validation sets
        testloaders = build_val_dataloaders(val_configs, imgsz_test, batch_size, gs, opt, hyp)

        if not opt.resume:
            labels = np.concatenate(dataset.labels, 0)
            c = torch.tensor(labels[:, 0])  # classes
            # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
            # model._initialize_biases(cf.to(device))
            if plots:
                #plot_labels(labels, names, save_dir, loggers)
                if tb_writer:
                    tb_writer.add_histogram('classes', c, 0)

            # Anchors
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
            model.half().float()  # pre-reduce anchor precision

    # DDP mode
    if cuda and rank != -1:
        try:
            print(f"Global Rank: {rank}")
            print(f"Device: {device}")
            if opt.sync_bn:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            # model = model.to(device)
        
            model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank, broadcast_buffers=False,
                        # nn.MultiheadAttention incompatibility with DDP https://github.com/pytorch/pytorch/issues/26698
                        find_unused_parameters=any(isinstance(layer, nn.MultiheadAttention) for layer in model.modules()))
            torch.cuda.empty_cache()  # CUDA ĳ�� ����
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise
    # Model parameters
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names

    # Start training
    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    compute_loss_ota = ComputeLossAuxOTA(model)  # init loss class
    compute_loss = ComputeLoss(model)  # init loss class
    load_loss_state(compute_loss_ota, loss_state_resume, logger)
    load_loss_state(compute_loss, loss_state_resume, logger)
    early_stopper = PhaseEarlyStopping(
        patience=getattr(opt, 'patience', 20),
        active_phase='phase3',
        enabled=phase_config.enabled and getattr(opt, 'early_stop_phase', 'phase3') == 'phase3')
    logger.info(f'Image sizes {imgsz} train, {imgsz_test} test\n'
                f'Using {dataloader.num_workers} dataloader workers\n'
                f'Logging results to {save_dir}\n'
                f'Starting training for {epochs} epochs...')
    if not getattr(opt, 'save_best_only', False):
        torch.save(model, wdir / 'init.pt')
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        epoch_start_time = time.time()
        stop_training = False
        model.train()

        if phase_config.enabled:
            next_phase_state = resolve_phase(epoch, phase_config)
            if phase_changed(phase_state, next_phase_state):
                previous_phase = phase_state.name
                phase_state = next_phase_state
                active_phase_name = phase_state.name
                imgsz = phase_imgsz(phase_state, base_imgsz)
                imgsz_test = phase_imgsz(phase_state, base_imgsz_test)
                del dataloader, dataset
                if rank in [-1, 0]:
                    del testloaders
                cleanup_dataloader()
                dataloader, dataset = build_train_dataloader(
                    train_path, imgsz, batch_size, gs, opt, hyp, rank,
                    rect=phase_rect(phase_state, opt.rect),
                    close_mosaic=phase_close_mosaic(phase_state, False),
                    force_mosaic_off=phase_close_mosaic(phase_state, False),
                    allow_rect_mosaic=phase_rect(phase_state, opt.rect) and phase_state.mosaic is True,
                    aug_phase=active_phase_name)
                nb = len(dataloader)
                model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc
                if rank in [-1, 0]:
                    if getattr(opt, 'aug_debug_samples', 0) > 0:
                        save_aug_debug_samples(
                            dataset, save_dir, opt.aug_debug_samples, names,
                            filename=f'aug_debug_{active_phase_name}_epoch{epoch}.jpg')
                    testloaders = build_val_dataloaders(val_configs, imgsz_test, batch_size, gs, opt, hyp)
                    if train_logger:
                        train_logger.log_phase_transition(
                            epoch, previous_phase, active_phase_name, imgsz,
                            phase_rect(phase_state, opt.rect), getattr(dataset, 'mosaic', None),
                            opt.hyp, True, True, getattr(dataloader, 'persistent_workers', None),
                            'phase boundary')
                        train_logger.write_hyp_snapshot(epoch, active_phase_name, hyp)
            else:
                phase_state = next_phase_state

        if not phase_config.enabled:
            logger.info(f"current {epoch}, epochs{epochs}, close mosaic{opt.close_mosaic}, epochs - opt.close_mosaic {epochs - opt.close_mosaic}")
        if not phase_config.enabled and opt.close_mosaic > 0 and epoch == (epochs - opt.close_mosaic):
            logger.info(f"Closing mosaic augmentation at epoch {epoch}")
            dataset.mosaic = False
            if hasattr(dataloader, 'dataset'):
                dataloader.dataset.mosaic = False
        # Update image weights (optional)
        if opt.image_weights:
            # Generate indices
            if rank in [-1, 0]:
                cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
                iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
                dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx
            # Broadcast if DDP
            if rank != -1:
                indices = (torch.tensor(dataset.indices) if rank == 0 else torch.zeros(dataset.n)).int()
                dist.broadcast(indices, 0)
                if rank != 0:
                    dataset.indices = indices.cpu().numpy()

        # Update mosaic border
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        if rank in [-1, 0]:
            log_epoch_start(opt, epoch, epochs, active_phase_name, nb)

        mloss = torch.zeros(4, device=device)  # mean losses
        epoch_positive_count = 0
        if hasattr(getattr(dataloader, 'sampler', None), 'set_epoch'):
            dataloader.sampler.set_epoch(epoch)
        if hasattr(getattr(dataloader, 'batch_sampler', None), 'set_epoch'):
            dataloader.batch_sampler.set_epoch(epoch)
        pbar = enumerate(dataloader)
        logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'labels', 'img_size'))
        if rank in [-1, 0]:
            pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, base_accumulate]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward
            with amp.autocast(enabled=cuda):
                pred = model(imgs)  # forward
                loss, loss_items = compute_loss_ota(pred, targets.to(device), imgs)  # loss scaled by batch_size
                epoch_positive_count += get_loss_positive_count(compute_loss_ota)
                if rank != -1:
                    loss *= opt.world_size  # gradient averaged between devices in DDP mode
                if opt.quad:
                    loss *= 4.

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni % accumulate == 0:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            # Print
            if rank in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.4g' * 6) % (
                    '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
                pbar.set_description(s)
                log_batch_progress(opt, epoch, epochs, i, nb, epoch_start_time, mloss, targets, imgs)

                # Plot
                if plots and ni < 10:
                    f = save_dir / f'train_batch{ni}.jpg'  # filename
                    Thread(target=plot_images, args=(imgs, targets, paths, f), daemon=True).start()
                    # if tb_writer:
                    #     tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                    #     tb_writer.add_graph(torch.jit.trace(model, imgs, strict=False), [])  # add model graph
                elif plots and ni == 10 and wandb_logger.wandb:
                    wandb_logger.log({"Mosaics": [wandb_logger.wandb.Image(str(x), caption=x.name) for x in
                                                  save_dir.glob('train*.jpg') if x.exists()]})

            # end batch ------------------------------------------------------------------------------------------------
        # end epoch ----------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for tensorboard
        scheduler.step()
        if rank in [-1, 0] and getattr(opt, 'sampler_mode', 'off') == 'weighted':
            log_sampler_stats(getattr(dataloader, 'sampler', None), epoch, dataset.labels, nc, save_dir)

        # DDP process 0 or single-GPU
        if rank in [-1, 0]:
            # mAP
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
            final_epoch = epoch + 1 == epochs
            all_val_results = []
            if not opt.notest or final_epoch:  # Calculate mAP
                wandb_logger.current_epoch = epoch + 1

                # Evaluate on all validation sets
                for val_name, testloader in testloaders:
                    logger.info(f'\n{"="*60}\nEvaluating on {val_name}\n{"="*60}')
                    results, maps, times, per_class = test.test(data_dict,
                                                     batch_size=batch_size * 2,
                                                     imgsz=imgsz_test,
                                                     model=ema.ema,
                                                     single_cls=opt.single_cls,
                                                     dataloader=testloader,
                                                     save_dir=save_dir,
                                                     verbose=True,  # Always verbose for class-wise results
                                                     plots=plots and final_epoch,
                                                     wandb_logger=wandb_logger,
                                                     compute_loss=compute_loss,
                                                     is_coco=is_coco,
                                                     v5_metric=opt.v5_metric)
                    all_val_results.append({
                        'name': val_name,
                        'results': results,
                        'maps': maps,
                        'times': times,
                        'per_class': per_class,
                        'sample_count': len(testloader.dataset) if hasattr(testloader, 'dataset') else ''
                    })

                # Use first validation set's results for backward compatibility (fitness calculation)
                results = all_val_results[0]['results']
                maps = all_val_results[0]['maps']

                # Write results to file
                with open(results_file, 'a') as f:
                    f.write(s + '%10.4g' * 7 % results + '\n')

                with open(save_dir / 'results_detail.txt', 'a') as f:
                    # Write epoch and training loss
                    f.write(s)

                    # Write results for each validation set
                    for val_result in all_val_results:
                        val_name = val_result['name']
                        val_res = val_result['results']
                        per_class = val_result['per_class']

                        # Write overall metrics for this validation set
                        f.write(f"  [{val_name}] " + '%10.4g' * 7 % val_res + '\n')

                        # Write per-class results if available
                        if per_class is not None:
                            names = per_class['names']
                            for i, c in enumerate(per_class['ap_class']):
                                class_name = names[c]
                                p_i = per_class['p'][i]
                                r_i = per_class['r'][i]
                                ap50_i = per_class['ap50'][i]
                                ap_i = per_class['ap'][i]
                                nt_i = per_class['nt'][c]
                                f.write(f"    [{val_name}][{class_name}] Images: {nt_i:>5}, P: {p_i:>8.3g}, R: {r_i:>8.3g}, mAP@.5: {ap50_i:>8.3g}, mAP@.5:.95: {ap_i:>8.3g}\n")
            if len(opt.name) and opt.bucket:
                os.system('gsutil cp %s gs://%s/results/results%s.txt' % (results_file, opt.bucket, opt.name))

            # Log
            tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',  # train loss
                    'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                    'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                    'x/lr0', 'x/lr1', 'x/lr2']  # params
            for x, tag in zip(list(mloss[:-1]) + list(results) + lr, tags):
                if tb_writer:
                    tb_writer.add_scalar(tag, x, epoch)  # tensorboard
                if wandb_logger.wandb:
                    wandb_logger.log({tag: x})  # W&B

            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            is_best = fi > best_fitness
            if is_best:
                best_fitness = fi
            wandb_logger.end_epoch(best_result=is_best)
            if train_logger:
                gpu_mem_gb = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0.0
                train_logger.log_epoch(epoch, active_phase_name, mloss, results, lr, gpu_mem_gb,
                                       time.time() - epoch_start_time)
                train_logger.log_loss_detail(epoch, active_phase_name, mloss,
                                             positive_count=epoch_positive_count,
                                             assigner=getattr(opt, 'assign', 'simota'),
                                             loss_box=getattr(opt, 'loss_box', 'ciou'),
                                             loss_cls=getattr(opt, 'loss_cls', 'bce'),
                                             head=getattr(opt, 'head', 'coupled'))
                train_logger.log_scenario_metrics(epoch, active_phase_name, all_val_results)
                per_class = all_val_results[0]['per_class'] if all_val_results else None
                train_logger.log_per_class(epoch, active_phase_name, per_class, is_best=is_best)
            if early_stopper.update(epoch, active_phase_name, fi):
                logger.info(f'Early stopping at epoch {epoch} in {active_phase_name}')
                stop_training = True

            # Save model
            if (not opt.nosave) or (final_epoch and not opt.evolve):  # if save
                ckpt = {
                        'epoch': epoch,
                        'best_fitness': best_fitness,
                        'training_results': results_file.read_text(),
                        'model': deepcopy(model.module if is_parallel(model) else model).half(),
                        'ema': deepcopy(ema.ema).half(),
                        'updates': ema.updates,
                        'optimizer': optimizer.state_dict(),
                        'wandb_id': wandb_logger.wandb_run.id if wandb_logger.wandb else None,
                        'loss_state': build_loss_state(opt, compute_loss_ota),
                        'mosaic_active': dataset.mosaic}
                # Save last, best and delete
                save_best_only = getattr(opt, 'save_best_only', False)
                save_current_best = best_fitness == fi
                if save_best_only:
                    if save_current_best or (final_epoch and not best.is_file()):
                        torch.save(ckpt, best)
                        if opt.model_saveoptimizer:
                            strip_optimizer(best)
                elif opt.model_saveoptimizer:
                    # optimizer 제거하고 저장 (가벼운 모델만)
                    torch.save(ckpt, last)
                    torch.save(ckpt, wdir / 'epoch_{:03d}.pt'.format(epoch))
                    if save_current_best:
                        torch.save(ckpt, wdir / 'best_{:03d}.pt'.format(epoch))
                        torch.save(ckpt, best)
                    
                    # 저장 후 optimizer 제거
                    strip_optimizer(last)
                    strip_optimizer(wdir / 'epoch_{:03d}.pt'.format(epoch))
                    if save_current_best:
                        strip_optimizer(wdir / 'best_{:03d}.pt'.format(epoch))
                        strip_optimizer(best)
                else:
                    # optimizer 포함해서 저장 (원본 ckpt 그대로)
                    torch.save(ckpt, last)
                    torch.save(ckpt, wdir / 'epoch_{:03d}.pt'.format(epoch))
                    if save_current_best:
                        torch.save(ckpt, wdir / 'best_{:03d}.pt'.format(epoch))
                        torch.save(ckpt, best)

                del ckpt
                torch.cuda.empty_cache()
                gc.collect()
            log_epoch_end(opt, epoch, epochs, start_epoch, t0, epoch_start_time,
                          active_phase_name, mloss, results, is_best)

        if stop_training:
            break

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training
    if rank in [-1, 0]:
        # Plots
        if plots:
            plot_results(save_dir=save_dir)  # save as results.png
            if wandb_logger.wandb:
                files = ['results.png', 'confusion_matrix.png', *[f'{x}_curve.png' for x in ('F1', 'PR', 'P', 'R')]]
                wandb_logger.log({"Results": [wandb_logger.wandb.Image(str(save_dir / f), caption=f) for f in files
                                              if (save_dir / f).exists()]})
        # Test best.pt
        logger.info('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
        if opt.data.endswith('coco.yaml') and nc == 80:  # if COCO
            test_weights = [x for x in (last, best) if x.exists()]
            for m in test_weights:  # speed, mAP tests
                results, _, _, _ = test.test(opt.data,
                                             batch_size=batch_size * 2,
                                             imgsz=imgsz_test,
                                             conf_thres=0.001,
                                             iou_thres=0.7,
                                             model=attempt_load(m, device).half(),
                                             single_cls=opt.single_cls,
                                             dataloader=testloader,
                                             save_dir=save_dir,
                                             save_json=True,
                                             plots=False,
                                             is_coco=is_coco,
                                             v5_metric=opt.v5_metric)

        # Strip optimizers
        final = best if best.exists() else last  # final model
        if train_logger:
            core_logging = any((
                getattr(opt, 'head', 'coupled') != 'coupled',
                getattr(opt, 'loss_box', 'ciou') != 'ciou',
                getattr(opt, 'assign', 'simota') != 'simota',
                getattr(opt, 'loss_cls', 'bce') != 'bce',
            ))
            if aug_logging:
                stage_name = '1.3.4'
            elif phase_config.enabled:
                stage_name = '1.3.2'
            elif core_logging:
                stage_name = '1.3.3'
            else:
                stage_name = '1.3.1'
            stage_result = train_logger.write_stage_result(
                stage=stage_name,
                baseline_run=None,
                current_run=str(save_dir),
                head=getattr(opt, 'head', 'coupled'),
                loss_box=getattr(opt, 'loss_box', 'ciou'),
                assign=getattr(opt, 'assign', 'simota'),
                loss_cls=getattr(opt, 'loss_cls', 'bce'),
                aug_profile=getattr(opt, 'aug_profile', 'off'),
                sampler_mode=getattr(opt, 'sampler_mode', 'off'),
                p2_head=getattr(opt, 'p2_head', 'none'),
                neck_mod=getattr(opt, 'neck_mod', 'none'),
                psa_level=getattr(opt, 'psa_level', 'none'),
                optional_decision=getattr(opt, 'optional_decision', ''),
                phase_train=phase_config.enabled,
                phase_boundaries=phase_config.boundaries(),
                best_epoch=None,
                best_map_50_95=float(results[3]) if len(results) > 3 else None,
                profile_json=None,
                baseline_gflops=None,
                current_gflops=None,
                gflops_delta_percent=None,
                primary_mAP=float(results[3]) if len(results) > 3 else None,
                mAP_0_5=float(results[2]) if len(results) > 2 else None,
                small_AP=None,
                rare_recall=None,
                trt_latency=None,
                export_check_json=None,
                output_contract_json=None,
                status='completed')
            train_logger.write_run_summary(stage_result)
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
        if opt.bucket:
            os.system(f'gsutil cp {final} gs://{opt.bucket}/weights')  # upload
        if wandb_logger.wandb and not opt.evolve:  # Log the stripped model
            wandb_logger.wandb.log_artifact(str(final), type='model',
                                            name='run_' + wandb_logger.wandb_run.id + '_model',
                                            aliases=['last', 'best', 'stripped'])
        wandb_logger.finish_run()
    else:
        dist.destroy_process_group()
    torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolo7.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.p5.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', '--batch', dest='batch_size', type=int, default=16,
                        help='total batch size for all GPUs')
    parser.add_argument('--img-size', '--img', dest='img_size', nargs='+', type=int, default=[640, 640],
                        help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--save-best-only', action='store_true',
                        help='save/update weights/best.pt only; skip last.pt and epoch_*.pt checkpoints')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used')
    parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')
    parser.add_argument('--close-mosaic', type=int, default=0, help='close mosaic augmentation (epochs)')
    parser.add_argument('--model-saveoptimizer', action='store_true', help='Save model optimizer state')
    parser.add_argument('--phase-train', choices=['off', 'on'], default='off')
    parser.add_argument('--phase1-epochs', type=int, default=290)
    parser.add_argument('--phase2-epochs', type=int, default=70)
    parser.add_argument('--phase3-epochs', type=int, default=40)
    parser.add_argument('--phase2-img', nargs=2, type=int, default=None)
    parser.add_argument('--phase3-img', nargs=2, type=int, default=None)
    parser.add_argument('--rect-size-l', nargs=2, type=int, default=[640, 384])
    parser.add_argument('--rect-size-w6', nargs=2, type=int, default=[1280, 736])
    parser.add_argument('--phase2-rect', dest='phase2_rect', action='store_true', default=True)
    parser.add_argument('--no-phase2-rect', dest='phase2_rect', action='store_false')
    parser.add_argument('--phase2-mosaic', choices=['on', 'off'], default='on')
    parser.add_argument('--phase3-mosaic', choices=['off'], default='off')
    parser.add_argument('--aux', choices=['auto', 'on', 'off'], default='auto')
    parser.add_argument('--grad-accumulate', type=int, default=None)
    parser.add_argument('--early-stop-phase', choices=['phase3', 'off'], default='phase3')
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--profile', choices=['off', 'on'], default='off')
    parser.add_argument('--per-class-log-interval', type=int, default=10)
    parser.add_argument('--log-format', choices=['txt', 'csv', 'both'], default='both')
    parser.add_argument('--nccl-timeout', type=int, default=14400,
                        help='NCCL process group timeout in seconds (default: 14400 = 4h)')
    parser.add_argument('--debug-log', choices=['off', 'error', 'debug', 'trace'], default='off')
    parser.add_argument('--debug-log-file', type=str, default='debug_trace.log')
    parser.add_argument('--debug-log-modules', type=str,
                        default='dataset,phase,model,loss,aug,sampler,finetune,runner')
    parser.add_argument('--progress-log-interval', type=int, default=50,
                        help='print rank0 batch progress every N batches; 0 disables')
    parser.add_argument('--no-verbose', action='store_true')
    parser.add_argument('--head', choices=['coupled', 'decoupled'], default='coupled')
    parser.add_argument('--loss-box', choices=['ciou', 'wiou_v3'], default='ciou')
    parser.add_argument('--assign', choices=['simota', 'tal'], default='simota')
    parser.add_argument('--loss-cls', choices=['bce', 'vfl'], default='bce')
    parser.add_argument('--aug-profile', choices=['off', 'cctv_pixel', 'cctv_paste'], default='off')
    parser.add_argument('--sampler-mode', choices=['off', 'none', 'weighted'], default='off')
    parser.add_argument('--aug-debug-samples', type=int, default=0)
    parser.add_argument('--hard-negative-manifest', type=str, default='')
    parser.add_argument('--p2-head', choices=['none', 'anchor', 'fcos'], default='none')
    parser.add_argument('--neck-mod', choices=['none', 'scdown', 'psa', 'gelan'], default='none')
    parser.add_argument('--psa-level', choices=['none', 'p5', 'p4p5', 'p3p4p5'], default='none')
    parser.add_argument('--optional-decision', type=str, default='')

    opt = parser.parse_args()
    ensure_aug_option_defaults(opt)
    ensure_structure_option_defaults(opt)
    validate_loss_options(opt, parser)

    # Set DDP variables
    opt.total_batch_size = opt.batch_size
    opt.world_size = int(os.environ.get('WORLD_SIZE', 1))
    opt.global_rank = int(os.environ.get('RANK', -1))
    opt.local_rank = int(os.environ.get('LOCAL_RANK', -1))
    set_logging(opt.global_rank)
    validate_aug_options(opt, parser)
    validate_structure_options(opt, parser)
    #if opt.global_rank in [-1, 0]:
    #    check_git_status()
    #    check_requirements()

    # Resume
    wandb_run = check_wandb_resume(opt)
    if opt.resume and not wandb_run:  # resume an interrupted run
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        apriori = opt.global_rank, opt.local_rank
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.load(f, Loader=yaml.SafeLoader))  # replace
        ensure_loss_option_defaults(opt)
        ensure_aug_option_defaults(opt)
        ensure_structure_option_defaults(opt)
        validate_loss_options(opt, parser)
        validate_aug_options(opt, parser)
        validate_structure_options(opt, parser)
        opt.cfg, opt.weights, opt.resume, opt.batch_size, opt.global_rank, opt.local_rank = '', ckpt, True, opt.total_batch_size, *apriori  # reinstate
        logger.info('Resuming training from %s' % ckpt)
    else:
        # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
        opt.name = 'evolve' if opt.evolve else opt.name
        opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)  # increment run
    ensure_loss_option_defaults(opt)
    ensure_aug_option_defaults(opt)
    ensure_structure_option_defaults(opt)

    # DDP mode
    opt.total_batch_size = opt.batch_size
    device = select_device(opt.device, batch_size=opt.batch_size)
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        if not dist.is_initialized():
            nccl_timeout = datetime.timedelta(seconds=int(getattr(opt, 'nccl_timeout', 14400)))
            dist.init_process_group(backend='nccl', init_method='env://', timeout=nccl_timeout)
            print(f'[{opt.local_rank}] Process group initialized')
        opt.batch_size = opt.total_batch_size // opt.world_size
        print(f'[{opt.local_rank}] Setup complete - batch_size: {opt.batch_size}, device: {device}')

    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

    # Train
    logger.info(opt)
    if not opt.evolve:
        tb_writer = None  # init loggers
        if opt.global_rank in [-1, 0]:
            prefix = colorstr('tensorboard: ')
            logger.info(f"{prefix}Start with 'tensorboard --logdir {opt.project}', view at http://localhost:6006/")
            tb_writer = SummaryWriter(opt.save_dir)  # Tensorboard
        try:
            train(hyp, opt, device, tb_writer)
        except Exception as exc:
            if opt.global_rank in [-1, 0]:
                debug_logger = get_debug_logger(opt.save_dir,
                                                getattr(opt, 'debug_log', 'off'),
                                                getattr(opt, 'debug_log_modules', ''),
                                                rank=opt.global_rank,
                                                debug_file=getattr(opt, 'debug_log_file', 'debug_trace.log'))
                debug_logger.log_exception(
                    'train_aux', 'main', exc,
                    summary={'save_dir': str(opt.save_dir), 'data': opt.data, 'cfg': opt.cfg, 'weights': opt.weights})
                try:
                    write_failed_run_artifacts(opt, exc)
                except Exception as artifact_exc:
                    debug_logger.log_exception(
                        'train_aux', 'write_failed_run_artifacts', artifact_exc,
                        summary={'save_dir': str(opt.save_dir)})
            raise

    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        meta = {'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
                'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
                'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
                'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
                'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
                'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
                'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
                'box': (1, 0.02, 0.2),  # box loss gain
                'cls': (1, 0.2, 4.0),  # cls loss gain
                'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
                'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
                'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
                'iou_t': (0, 0.1, 0.7),  # IoU training threshold
                'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
                'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
                'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
                'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
                'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
                'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
                'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
                'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
                'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
                'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
                'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
                'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
                'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
                'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
                'mixup': (1, 0.0, 1.0)}  # image mixup (probability)
        
        with open(opt.hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
            if 'anchors' not in hyp:  # anchors commented in hyp.yaml
                hyp['anchors'] = 3
                
        assert opt.local_rank == -1, 'DDP mode not implemented for --evolve'
        opt.notest, opt.nosave = True, True  # only test/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        yaml_file = Path(opt.save_dir) / 'hyp_evolved.yaml'  # save best result here
        if opt.bucket:
            os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  # download evolve.txt if exists

        for _ in range(300):  # generations to evolve
            if Path('evolve.txt').exists():  # if evolve.txt exists: select best hyps and mutate
                # Select parent(s)
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt('evolve.txt', ndmin=2)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min()  # weights
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                mp, s = 0.8, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([x[0] for x in meta.values()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate

            # Constrain to limits
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits

            # Train mutation
            results = train(hyp.copy(), opt, device)

            # Write mutation results
            print_mutation(hyp.copy(), results, yaml_file, opt.bucket)

        # Plot results
        plot_evolution(yaml_file)
        print(f'Hyperparameter evolution complete. Best results saved as: {yaml_file}\n'
              f'Command to train a new model with these hyperparameters: $ python train.py --hyp {yaml_file}')
