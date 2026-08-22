"""
TensorLoom Standard Library — Neural Network layers.

Exposes common layers as TensorLoom builtins, wrapping torch.nn.
"""
from __future__ import annotations

import torch.nn as nn


# Layer registry — maps TensorLoom layer names to constructors
LAYER_REGISTRY: dict[str, type] = {
    "Linear": nn.Linear,
    "Conv2d": nn.Conv2d,
    "Conv1d": nn.Conv1d,
    "BatchNorm2d": nn.BatchNorm2d,
    "LayerNorm": nn.LayerNorm,
    "Dropout": nn.Dropout,
    "Embedding": nn.Embedding,
    "LSTM": nn.LSTM,
    "GRU": nn.GRU,
    "MultiHeadAttention": nn.MultiheadAttention,
    "Sequential": nn.Sequential,
    "GELU": nn.GELU,
    "ReLU": nn.ReLU,
    "Sigmoid": nn.Sigmoid,
    "Tanh": nn.Tanh,
    "SiLU": nn.SiLU,
    "MaxPool2d": nn.MaxPool2d,
    "AvgPool2d": nn.AvgPool2d,
    "AdaptiveAvgPool2d": nn.AdaptiveAvgPool2d,
    "Flatten": nn.Flatten,
}


def create_layer(name: str, *args, **kwargs) -> nn.Module:
    """Create a neural network layer by name.
    
    Args:
        name: The TensorLoom layer name (e.g., "Linear", "Conv2d").
        *args, **kwargs: Arguments passed to the layer constructor.
    
    Returns:
        An nn.Module instance.
    
    Raises:
        ValueError: If the layer name is not recognized.
    """
    if name not in LAYER_REGISTRY:
        raise ValueError(
            f"Unknown layer type: '{name}'. "
            f"Available layers: {', '.join(sorted(LAYER_REGISTRY.keys()))}"
        )
    return LAYER_REGISTRY[name](*args, **kwargs)


def register_layer(name: str, layer_class: type) -> None:
    """Register a custom layer type with TensorLoom.
    
    Args:
        name: The name to use in TensorLoom source code.
        layer_class: The nn.Module subclass.
    """
    LAYER_REGISTRY[name] = layer_class
