import csv
import random
from pathlib import Path

import numpy as np
import torch

from utils.general import labels_to_class_weights, labels_to_image_weights


class ClassBalancedImageSampler(torch.utils.data.Sampler):
    def __init__(self, labels, nc, num_samples=None, seed=0, empty_weight=0.10, max_weight=10.0):
        self.labels = labels
        self.nc = int(nc)
        self.num_samples = int(num_samples or len(labels))
        self.seed = int(seed)
        self.epoch = 0
        self.class_weights = labels_to_class_weights(labels, self.nc).numpy()
        weights = labels_to_image_weights(labels, self.nc, self.class_weights)
        positive = weights[weights > 0]
        fallback = float(np.median(positive)) * float(empty_weight) if positive.size else 1.0
        weights = np.where(weights > 0, weights, fallback)
        weights = np.clip(weights, 1e-12, np.median(weights) * float(max_weight))
        self.image_weights = weights.astype(float)
        self.last_indices = []

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        population = list(range(len(self.labels)))
        self.last_indices = rng.choices(population, weights=self.image_weights.tolist(), k=self.num_samples)
        return iter(self.last_indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = int(epoch)


def build_weighted_sampler(dataset, nc, rank=-1, world_size=1, seed=0, hyp=None):
    if rank != -1 or int(world_size) > 1:
        raise ValueError('--sampler-mode weighted is single-GPU only')
    hyp = hyp or {}
    return ClassBalancedImageSampler(
        dataset.labels,
        nc=nc,
        num_samples=len(dataset),
        seed=seed,
        empty_weight=float(hyp.get('sampler_empty_weight', 0.10)),
        max_weight=float(hyp.get('sampler_max_weight', 10.0)),
    )


def log_sampler_stats(sampler, epoch, labels, nc, save_dir):
    if not isinstance(sampler, ClassBalancedImageSampler) or not sampler.last_indices:
        return
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / 'sampler_stats.csv'
    write_header = not path.exists()
    class_counts = np.zeros(int(nc), dtype=np.int64)
    empty_count = 0
    for index in sampler.last_indices:
        item = labels[index]
        if item is None or len(item) == 0:
            empty_count += 1
            continue
        classes = item[:, 0].astype(np.int64)
        class_counts += np.bincount(classes, minlength=int(nc))
    with open(path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['epoch', 'class_id', 'sampled_label_count', 'empty_image_count'])
        for class_id, count in enumerate(class_counts.tolist()):
            writer.writerow([epoch, class_id, count, empty_count])
