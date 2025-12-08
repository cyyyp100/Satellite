import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Type

import numpy as np
import torch
from torch import nn


def _select_device(device: Optional[str] = None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class MLPModel(nn.Module):
    """Simple MLP that operates on the last timestep of a sequence."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] | Iterable[int] = (256, 256, 128),
        dropout: float = 0.1,
        lr: float = 5e-4,
        optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adam,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.lr = lr
        self.optimizer_cls = optimizer_cls
        self.device = _select_device(device)

        layers: List[nn.Module] = []
        last_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, int(h)))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            last_dim = int(h)
        layers.append(nn.Linear(last_dim, 3))
        self.net = nn.Sequential(*layers)
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_last = x[:, -1, :]
        return self.net(x_last)

    def train_model(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int = 50,
        lr: Optional[float] = None,
        weight_decay: float = 0.0,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        optimizer = self.optimizer_cls(self.parameters(), lr=lr or self.lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        history = {"train_loss": [], "val_loss": [], "train_rmse": [], "val_rmse": [], "epoch_time": []}

        if verbose:
            print("""\nMLP Training Metrics""")
            print("Epoch | Train_MSE   Val_MSE     Train_RMSE  Val_RMSE   Time(s)")
            print("-" * 64)

        for epoch in range(1, epochs + 1):
            start = time.perf_counter()
            train_loss, train_rmse = self._run_epoch(train_loader, optimizer, criterion, training=True)
            val_loss, val_rmse = self._run_epoch(val_loader, optimizer, criterion, training=False)
            duration = time.perf_counter() - start

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_rmse"].append(train_rmse)
            history["val_rmse"].append(val_rmse)
            history["epoch_time"].append(duration)

            if verbose:
                print(
                    f"{epoch:5d} | "
                    f"{train_loss:10.6f} {val_loss:11.6f} "
                    f"{train_rmse:11.6f} {val_rmse:10.6f} "
                    f"{duration:7.2f}"
                )

        return history

    def _run_epoch(
        self,
        loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        training: bool,
    ) -> Tuple[float, float]:
        if training:
            self.train()
        else:
            self.eval()

        total_loss = 0.0
        total_rmse = 0.0
        total_samples = 0

        with torch.set_grad_enabled(training):
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                preds = self(xb)
                loss = criterion(preds, yb)

                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                batch_size = xb.size(0)
                mse_value = loss.item()
                rmse_value = torch.sqrt(torch.mean((preds - yb) ** 2)).item()

                total_loss += mse_value * batch_size
                total_rmse += rmse_value * batch_size
                total_samples += batch_size

        avg_loss = total_loss / max(total_samples, 1)
        avg_rmse = total_rmse / max(total_samples, 1)
        return avg_loss, avg_rmse

