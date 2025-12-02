import time
from typing import Dict, List, Optional, Tuple, Type

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


class EarlyStopping:
    """Early stopping utility monitoring the validation loss."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss: float = float("inf")
        self.counter = 0

    def step(self, loss: float) -> bool:
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


class GRUModel(nn.Module):
    """GRU model with a configurable training helper."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        lr: float = 1e-3,
        optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adam,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.lr = lr
        self.optimizer_cls = optimizer_cls
        self.device = _select_device(device)

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 3)
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.fc(last)

    def train_model(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int = 50,
        lr: Optional[float] = None,
        weight_decay: float = 0.0,
        verbose: bool = True,
        patience: int = 10,
        min_delta: float = 0.0,
    ) -> Dict[str, List[float]]:
        optimizer = self.optimizer_cls(self.parameters(), lr=lr or self.lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        history = {"train_loss": [], "val_loss": [], "train_rmse": [], "val_rmse": [], "epoch_time": []}
        stopper = EarlyStopping(patience=patience, min_delta=min_delta)

        if verbose:
            print("""\nGRU Training Metrics""")
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

            if stopper.step(val_loss):
                if verbose:
                    print(f"Arrêt anticipé à l'époque {epoch} (pas d'amélioration du MSE validation)")
                break

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

