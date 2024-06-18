from typing import Optional

import torch
from pytorch_lightning.utilities import FLOAT32_EPSILON
from torchmetrics import Metric


class Accuracy(Metric):
    def __init__(self, dist_sync_on_step: bool = False) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state(
            "correct", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state(
            "total", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        diff = preds.argmax(dim=1).eq(targets)
        self.correct += diff.sum()
        self.total += targets.numel()

    def compute(self) -> torch.Tensor:
        return self.correct.float() / (self.total + FLOAT32_EPSILON)

