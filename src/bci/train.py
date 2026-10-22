"""Training a PyTorch decoder: mini-batches, a validation split, early stopping.

See docs/DESIGN.md, "EEGNet training".
"""
import copy
import random

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int) -> None:
    """Make weight initialisation, batch order and dropout reproducible.

    Call before building the model so its initial weights are seeded too.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_train_val(
    X: np.ndarray, y: np.ndarray, val_fraction: float = 0.2, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Hold out a class-balanced validation set: (X_train, X_val, y_train, y_val)."""
    return train_test_split(X, y, test_size=val_fraction, stratify=y, random_state=seed)


def fit(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    lr: float = 1e-3,
    batch_size: int = 32,
    max_epochs: int = 300,
    patience: int = 50,
    seed: int = 0,
    device: str = "cpu",
) -> list[dict]:
    """Train in place; restore the weights from the epoch with the lowest validation loss.

    Stops early once validation loss hasn't improved for `patience` epochs.
    Returns one dict per epoch: epoch, train_loss, val_loss, val_acc.
    device: "cpu" (bit-for-bit reproducible) or "mps" (Apple GPU, much faster).
    """
    torch.manual_seed(seed)  # dropout
    model.to(device)
    loader = DataLoader(
        TensorDataset(_as_tensor(X_train), _as_labels(y_train)),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),  # batch order
    )
    X_val_t, y_val_t = _as_tensor(X_val).to(device), _as_labels(y_val).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    history = []
    best_loss, best_epoch, best_state = float("inf"), 0, None
    for epoch in range(max_epochs):
        # One pass over the training set, a batch at a time: loss -> slopes -> step.
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)

        # Check on held-out data (dropout off, no learning).
        model.eval()
        with torch.no_grad():
            logits = model(X_val_t)
            val_loss = loss_fn(logits, y_val_t).item()
            val_acc = (logits.argmax(dim=1) == y_val_t).float().mean().item()
        history.append(
            {"epoch": epoch, "train_loss": total / len(X_train), "val_loss": val_loss, "val_acc": val_acc}
        )

        if val_loss < best_loss:
            best_loss, best_epoch, best_state = val_loss, epoch, copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= patience:
            break

    model.load_state_dict(best_state)
    return history


def predict(model: nn.Module, X: np.ndarray, batch_size: int = 256, device: str = "cpu") -> np.ndarray:
    """Predicted class index for each trial in X (n_trials, n_channels, n_times)."""
    model.to(device).eval()
    loader = DataLoader(TensorDataset(_as_tensor(X)), batch_size=batch_size)
    with torch.no_grad():
        return np.concatenate([model(xb.to(device)).argmax(dim=1).cpu().numpy() for (xb,) in loader])


def _as_tensor(X: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.asarray(X), dtype=torch.float32)


def _as_labels(y: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.asarray(y), dtype=torch.long)
