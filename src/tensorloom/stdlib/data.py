"""
TensorLoom Standard Library — Data loading utilities.

Provides built-in dataset loading for common benchmarks.
"""
from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset


def load_synthetic(
    num_samples: int = 1000,
    input_dim: int = 784,
    num_classes: int = 10,
    batch_size: int = 64,
    shuffle: bool = True,
) -> DataLoader:
    """Create a synthetic dataset for testing.
    
    Generates random input tensors and random integer labels.
    Useful for verifying model architecture and training pipelines
    without downloading real datasets.
    
    Args:
        num_samples: Number of samples to generate.
        input_dim: Dimensionality of each input sample.
        num_classes: Number of classes for labels.
        batch_size: Batch size for the DataLoader.
        shuffle: Whether to shuffle the data.
    
    Returns:
        A PyTorch DataLoader.
    """
    inputs = torch.randn(num_samples, input_dim)
    labels = torch.randint(0, num_classes, (num_samples,))
    dataset = TensorDataset(inputs, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def load_dataset(
    name: str,
    batch_size: int = 64,
    shuffle: bool = True,
    **kwargs: Any,
) -> DataLoader:
    """Load a dataset by name.
    
    For Phase 1, generates synthetic data matching the expected shape.
    Future versions will download real datasets.
    
    Supported names: "mnist", "cifar10", "cifar100", "fashion_mnist"
    """
    dataset_configs: dict[str, dict[str, int]] = {
        "mnist":         {"input_dim": 784, "num_classes": 10},
        "fashion_mnist": {"input_dim": 784, "num_classes": 10},
        "cifar10":       {"input_dim": 3072, "num_classes": 10},   # 3*32*32
        "cifar100":      {"input_dim": 3072, "num_classes": 100},
    }

    config = dataset_configs.get(name.lower(), {"input_dim": 784, "num_classes": 10})
    num_samples = kwargs.get("num_samples", 1000)

    return load_synthetic(
        num_samples=num_samples,
        input_dim=config["input_dim"],
        num_classes=config["num_classes"],
        batch_size=batch_size,
        shuffle=shuffle,
    )
