"""
TensorLoom Runtime — Training loop utilities.

Provides reusable training infrastructure used by the generated code.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn


@dataclass
class TrainingMetrics:
    """Metrics collected during a single epoch."""
    epoch: int = 0
    loss: float = 0.0
    accuracy: float = 0.0
    batches: int = 0
    elapsed_seconds: float = 0.0
    gpu_memory_mb: float = 0.0


@dataclass
class TrainingConfig:
    """Configuration for a TensorLoom training run."""
    epochs: int = 10
    precision: str = "fp32"
    checkpoint_every: int = 0  # 0 = disabled
    device: str = "auto"

    def get_device(self) -> torch.device:
        if self.device == "auto" or self.device == "gpu":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    @property
    def use_amp(self) -> bool:
        return self.precision in ("fp16", "float16", "bf16", "bfloat16")

    @property
    def amp_dtype(self) -> torch.dtype:
        mapping = {
            "fp16": torch.float16,
            "float16": torch.float16,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
        }
        return mapping.get(self.precision, torch.float16)


class TrainingLoop:
    """A reusable training loop with AMP and checkpointing support."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        config: TrainingConfig,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config
        self.device = config.get_device()
        self.callbacks: dict[str, list[Callable]] = {
            "epoch_start": [],
            "epoch_end": [],
            "batch_end": [],
        }
        self.history: list[TrainingMetrics] = []

    def on(self, event: str, callback: Callable) -> None:
        """Register a callback for a training event."""
        if event in self.callbacks:
            self.callbacks[event].append(callback)

    def train(self, dataloader) -> list[TrainingMetrics]:
        """Execute the training loop."""
        scaler = None
        if self.config.use_amp:
            scaler = torch.amp.GradScaler("cuda")

        for epoch in range(self.config.epochs):
            t0 = time.perf_counter()
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for cb in self.callbacks["epoch_start"]:
                cb(epoch)

            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                self.optimizer.zero_grad()

                if self.config.use_amp and scaler is not None:
                    with torch.amp.autocast(device_type="cuda", dtype=self.config.amp_dtype):
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    loss.backward()
                    self.optimizer.step()

                running_loss += loss.item()
                if outputs.dim() >= 2:
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

                for cb in self.callbacks["batch_end"]:
                    cb(epoch, batch_idx, loss.item())

            elapsed = time.perf_counter() - t0
            metrics = TrainingMetrics(
                epoch=epoch + 1,
                loss=running_loss / max(total, 1),
                accuracy=correct / max(total, 1),
                batches=batch_idx + 1 if total > 0 else 0,
                elapsed_seconds=elapsed,
            )

            if torch.cuda.is_available():
                metrics.gpu_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)

            self.history.append(metrics)

            # Checkpointing
            if self.config.checkpoint_every > 0 and (epoch + 1) % self.config.checkpoint_every == 0:
                path = f"checkpoint_epoch_{epoch + 1}.pt"
                torch.save(self.model.state_dict(), path)

            for cb in self.callbacks["epoch_end"]:
                cb(metrics)

        return self.history
