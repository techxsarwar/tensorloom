"""
TensorLoom Runtime — Tensor operations support.

Provides helper functions used by the generated PyTorch code for
TensorLoom-specific tensor operations.
"""
from __future__ import annotations

import torch


def tl_create_tensor(
    data: list,
    dtype: str = "float32",
    device: str = "auto",
) -> torch.Tensor:
    """Create a tensor with TensorLoom's conventions.
    
    Args:
        data: The tensor data (nested lists).
        dtype: TensorLoom dtype string (float32, float16, etc.)
        device: "gpu", "cpu", or "auto" (auto-detect).
    
    Returns:
        A torch.Tensor on the requested device.
    """
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16,
        "int32": torch.int32,
        "int64": torch.int64,
        "int8": torch.int8,
        "bool": torch.bool,
        "fp16": torch.float16,
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
    }

    torch_dtype = dtype_map.get(dtype, torch.float32)

    if device == "auto" or device == "gpu":
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        torch_device = torch.device(device)

    return torch.tensor(data, dtype=torch_dtype, device=torch_device)


def tl_memory_report() -> dict:
    """Get current GPU memory usage report.
    
    Returns:
        Dict with allocated, reserved, and free memory in MB.
    """
    if not torch.cuda.is_available():
        return {"device": "cpu", "allocated_mb": 0, "reserved_mb": 0}

    allocated = torch.cuda.memory_allocated() / (1024 * 1024)
    reserved = torch.cuda.memory_reserved() / (1024 * 1024)
    total = torch.cuda.get_device_properties(0).total_mem / (1024 * 1024)

    return {
        "device": torch.cuda.get_device_name(0),
        "allocated_mb": round(allocated, 2),
        "reserved_mb": round(reserved, 2),
        "total_mb": round(total, 2),
        "free_mb": round(total - reserved, 2),
    }


def tl_count_parameters(model: torch.nn.Module) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
