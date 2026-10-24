"""Modelo de referencia (regresión logística) y perceptrón multicapa en PyTorch."""
import copy
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn

ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "gelu": nn.GELU,
}


# --------------------------------------------------------------------------- #
# Regresión logística
# --------------------------------------------------------------------------- #
def fit_logistic(X, y, C: float = 1.0, max_iter: int = 2000, seed: int = 42) -> LogisticRegression:
    model = LogisticRegression(C=C, max_iter=max_iter, random_state=seed)
    return model.fit(X, y)


def logistic_n_params(model: LogisticRegression) -> int:
    return int(model.coef_.size + model.intercept_.size)


# --------------------------------------------------------------------------- #
# Perceptrón multicapa
# --------------------------------------------------------------------------- #
class MLP(nn.Module):
    """Capas densas -> activación -> dropout, con una salida (logit de la PD)."""

    def __init__(self, n_inputs: int, hidden: list[int], activation: str = "relu", dropout: float = 0.0):
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"Activación no soportada: {activation}. Opciones: {list(ACTIVATIONS)}")
        layers, width = [], n_inputs
        for h in hidden:
            layers += [nn.Linear(width, h), ACTIVATIONS[activation]()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            width = h
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def n_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@dataclass
class MLPConfig:
    hidden: list[int] = field(default_factory=lambda: [64, 32])
    activation: str = "relu"
    dropout: float = 0.1
    l2: float = 0.0               # λ de la penalización λ·Σ||W||² (solo pesos, no sesgos)
    optimizer: str = "adam"       # adam | sgd | rmsprop
    lr: float = 1e-3
    momentum: float = 0.9         # solo para sgd
    batch_size: int = 1024
    max_epochs: int = 100
    patience: int = 10            # parada temprana sobre la pérdida de validación
    seed: int = 42


def _make_optimizer(params, cfg: MLPConfig) -> torch.optim.Optimizer:
    # La regularización L2 se suma explícitamente a la pérdida (ver _l2_penalty), así que
    # no se usa el weight_decay del optimizador.
    if cfg.optimizer == "adam":
        return torch.optim.Adam(params, lr=cfg.lr)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum)
    if cfg.optimizer == "rmsprop":
        return torch.optim.RMSprop(params, lr=cfg.lr)
    raise ValueError(f"Optimizador no soportado: {cfg.optimizer}")


def _l2_penalty(model: nn.Module) -> torch.Tensor:
    return sum((m.weight ** 2).sum() for m in model.modules() if isinstance(m, nn.Linear))


def _to_tensor(a) -> torch.Tensor:
    return torch.as_tensor(np.asarray(a, dtype=np.float32))


@torch.no_grad()
def _mean_bce(model: nn.Module, X: torch.Tensor, y: torch.Tensor, batch_size: int) -> float:
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
    total = sum(loss_fn(model(X[i:i + batch_size]), y[i:i + batch_size]).item() for i in range(0, len(X), batch_size))
    return total / len(X)


def train_mlp(X_train, y_train, X_val, y_val, cfg: MLPConfig, verbose: bool = False):
    """Entrena con entropía cruzada binaria (+ L2) y parada temprana.

    Devuelve (modelo con los mejores pesos de validación, historial).
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    Xt, yt = _to_tensor(X_train), _to_tensor(y_train)
    Xv, yv = _to_tensor(X_val), _to_tensor(y_val)

    model = MLP(Xt.shape[1], cfg.hidden, cfg.activation, cfg.dropout)
    optimizer = _make_optimizer(model.parameters(), cfg)
    loss_fn = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(cfg.seed)

    best_loss, best_state, best_epoch, wait = float("inf"), None, 0, 0
    history = {"train_loss": [], "val_loss": []}
    start = time.perf_counter()

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        order = torch.randperm(len(Xt), generator=generator)
        for i in range(0, len(Xt), cfg.batch_size):
            idx = order[i:i + cfg.batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            if cfg.l2 > 0:
                loss = loss + cfg.l2 * _l2_penalty(model)
            loss.backward()
            optimizer.step()

        train_loss = _mean_bce(model, Xt, yt, cfg.batch_size)
        val_loss = _mean_bce(model, Xv, yv, cfg.batch_size)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if verbose:
            print(f"época {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_loss - 1e-6:
            best_loss, best_state, best_epoch, wait = val_loss, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            wait += 1
            if wait >= cfg.patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    history.update(best_epoch=best_epoch, best_val_loss=best_loss, seconds=time.perf_counter() - start)
    return model, history


@torch.no_grad()
def predict_proba(model: nn.Module, X, batch_size: int = 8192) -> np.ndarray:
    """PD estimada = sigmoide(logit)."""
    model.eval()
    Xt = _to_tensor(X)
    out = [torch.sigmoid(model(Xt[i:i + batch_size])) for i in range(0, len(Xt), batch_size)]
    return torch.cat(out).numpy()
