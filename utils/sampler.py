import csv
import math
import random
from pathlib import Path

import numpy as np
import torch

from utils.general import labels_to_class_weights, labels_to_image_weights


def _compute_image_weights(labels, nc, empty_weight, max_weight):
    """클래스 균형 이미지 가중치를 계산한다."""
    class_weights = labels_to_class_weights(labels, nc).numpy()
    weights = labels_to_image_weights(labels, nc, class_weights)
    positive = weights[weights > 0]
    fallback = float(np.median(positive)) * float(empty_weight) if positive.size else 1.0
    weights = np.where(weights > 0, weights, fallback)
    weights = np.clip(weights, 1e-12, np.median(weights) * float(max_weight))
    return weights.astype(float)


class ClassBalancedImageSampler(torch.utils.data.Sampler):
    """단일 GPU용 클래스 균형 weighted sampler."""

    def __init__(self, labels, nc, num_samples=None, seed=0, empty_weight=0.10, max_weight=10.0):
        self.labels = labels
        self.nc = int(nc)
        self.num_samples = int(num_samples or len(labels))
        self.seed = int(seed)
        self.epoch = 0
        self.image_weights = _compute_image_weights(labels, nc, empty_weight, max_weight)
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


class DistributedClassBalancedImageSampler(torch.utils.data.Sampler):
    """DDP 환경에서 동작하는 클래스 균형 weighted sampler.

    전체 인덱스를 동일한 seed로 글로벌하게 샘플링한 뒤,
    각 rank가 stride 방식으로 자신의 shard를 가져간다.
    이 방식으로 rank 간 중복 없이 클래스 균형을 유지한다.
    """

    def __init__(self, labels, nc, num_replicas=1, rank=0, num_samples=None,
                 seed=0, empty_weight=0.10, max_weight=10.0):
        self.labels = labels
        self.nc = int(nc)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        self.image_weights = _compute_image_weights(labels, nc, empty_weight, max_weight)

        # total_size는 num_replicas의 배수로 올림
        total = int(num_samples or len(labels))
        self.total_size = math.ceil(total / self.num_replicas) * self.num_replicas
        self.num_samples = self.total_size // self.num_replicas
        self.last_indices = []

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        population = list(range(len(self.labels)))
        all_indices = rng.choices(population, weights=self.image_weights.tolist(), k=self.total_size)
        # 각 rank는 stride 방식으로 자신의 shard를 가져감
        shard = all_indices[self.rank:self.total_size:self.num_replicas]
        self.last_indices = shard
        return iter(shard)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = int(epoch)


def build_weighted_sampler(dataset, nc, rank=-1, world_size=1, seed=0, hyp=None):
    """rank/world_size에 따라 적절한 weighted sampler를 반환한다.

    - 단일 GPU (rank == -1 또는 world_size == 1): ClassBalancedImageSampler
    - 멀티 GPU DDP: DistributedClassBalancedImageSampler
    """
    hyp = hyp or {}
    kwargs = dict(
        nc=nc,
        num_samples=len(dataset),
        seed=seed,
        empty_weight=float(hyp.get('sampler_empty_weight', 0.10)),
        max_weight=float(hyp.get('sampler_max_weight', 10.0)),
    )
    if rank != -1 and int(world_size) > 1:
        return DistributedClassBalancedImageSampler(
            dataset.labels,
            num_replicas=int(world_size),
            rank=rank,
            **kwargs,
        )
    return ClassBalancedImageSampler(dataset.labels, **kwargs)


def log_sampler_stats(sampler, epoch, labels, nc, save_dir):
    if not isinstance(sampler, (ClassBalancedImageSampler, DistributedClassBalancedImageSampler)) or not sampler.last_indices:
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
