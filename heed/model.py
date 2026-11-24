"""HEED: Hidden-state and Entropy Evidential Detector.

    alpha  = e_h + e_H + 1                         (Dirichlet over {KNOW, HALL})
    e_h    = softplus(MLP(z(h)))                   evidential MLP on the hidden representation h
    e_H    = softplus(w * z(H-bar) + b)            4-parameter evidence head on the keyword entropy H-bar
    s_FRH  = alpha_HALL / (alpha_HALL + alpha_KNOW)

Training is two-stage: e_h is trained alone with the EDL loss, frozen, and e_H is then trained on top of it
with the same loss (alpha = e_h(frozen) + e_H + 1).  Both stages use the validation split only for early
stopping.  z(.) is standardisation with train-split statistics.
"""
import random
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .metrics import monitor_value


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------------- EDL loss
def dirichlet_kl_to_uniform(alpha: torch.Tensor) -> torch.Tensor:
    k = alpha.shape[1]
    s = alpha.sum(1, keepdim=True)
    first = torch.lgamma(s) - torch.lgamma(alpha).sum(1, keepdim=True)
    first = first - torch.lgamma(torch.tensor(float(k), device=alpha.device))
    second = ((alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(s))).sum(1, keepdim=True)
    return (first + second).squeeze(1)


def edl_loss(alpha: torch.Tensor, y_onehot: torch.Tensor, class_weights: torch.Tensor, kl_weight: float):
    """Class-weighted digamma (expected cross-entropy) EDL loss + KL of the misleading evidence to Dir(1)."""
    s = alpha.sum(1, keepdim=True)
    nll = (y_onehot * (torch.digamma(s) - torch.digamma(alpha))).sum(1)
    nll = nll * (y_onehot * class_weights.unsqueeze(0)).sum(1)
    alpha_tilde = y_onehot + (1.0 - y_onehot) * alpha
    return (nll + kl_weight * dirichlet_kl_to_uniform(alpha_tilde)).mean()


def class_weights_of(labels: torch.Tensor) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=2).float()
    return counts.sum() / (counts.clamp_min(1.0) * len(counts))


def kl_schedule(epoch: int, cfg: dict) -> float:
    return min(1.0, epoch / max(1, cfg["kl_anneal_epochs"])) * cfg["kl_weight"]


def p_hall(alpha: torch.Tensor) -> np.ndarray:
    return (alpha[:, 1] / alpha.sum(1)).numpy()


# ----------------------------------------------------------------------------------- modules
class EvidentialMLP(nn.Module):
    """e_h: 3d -> hidden -> hidden/4 -> 2 (LayerNorm, GELU, dropout), softplus evidence."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 4), nn.LayerNorm(hidden_dim // 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.net(x))


class EntropyEvidence(nn.Module):
    """e_H = softplus(w * z + b), w, b in R^2 (4 parameters). w starts at 0, b at init_bias."""

    def __init__(self, input_dim: int = 1, init_bias: float = -2.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 2))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, init_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.net(x))


# ----------------------------------------------------------------------------------- training
@dataclass
class FitResult:
    model: nn.Module
    best_epoch: int
    best_val: float
    history: list = field(default_factory=list)


def _early_stopping_loop(model, step_epoch, val_score, cfg):
    best, best_score, best_epoch, patience, history = None, -1.0, 0, cfg["patience"], []
    for epoch in range(1, cfg["epochs"] + 1):
        step_epoch(epoch)
        score = val_score()
        history.append(round(float(score), 5))
        if score > best_score + 1e-4:
            best_score, best_epoch, patience = score, epoch, cfg["patience"]
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience -= 1
            if patience <= 0:
                break
    model.load_state_dict(best)
    return FitResult(model, best_epoch, best_score, history)


def fit_eh(h: torch.Tensor, labels: torch.Tensor, tr, va, cfg: dict, seed: int, device) -> tuple:
    """Stage 1.  Returns (FitResult, mean, std) with the train-split standardisation of h."""
    set_seed(seed)
    y = labels.numpy()
    cw = class_weights_of(labels).to(device)
    mean = h[torch.as_tensor(tr)].mean(0, keepdim=True)
    std = h[torch.as_tensor(tr)].std(0, keepdim=True).clamp_min(1e-6)
    x_tr, x_va = (h[torch.as_tensor(tr)] - mean) / std, (h[torch.as_tensor(va)] - mean) / std
    loader = DataLoader(TensorDataset(x_tr, labels[torch.as_tensor(tr)]), batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(TensorDataset(x_va, labels[torch.as_tensor(va)]), batch_size=cfg["batch_size"])
    model = EvidentialMLP(h.shape[1], cfg["hidden_dim"], cfg["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])

    def step_epoch(epoch):
        model.train()
        kl = kl_schedule(epoch, cfg)
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = edl_loss(model(xb) + 1.0, F.one_hot(yb, 2).float(), cw, kl)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
        sched.step()

    @torch.no_grad()
    def val_score():
        model.eval()
        alpha = torch.cat([model(xb.to(device)).cpu() + 1.0 for xb, _ in val_loader])
        return monitor_value(y[va], (alpha / alpha.sum(1, keepdim=True)).numpy(), cfg["monitor"])

    return _early_stopping_loop(model, step_epoch, val_score, cfg), mean.squeeze(0), std.squeeze(0)


@torch.no_grad()
def evidence_of(model: nn.Module, x: torch.Tensor, batch: int = 1024) -> torch.Tensor:
    """Evidence of a trained module on CPU (eval mode)."""
    model = model.cpu().eval()
    return torch.cat([model(x[i:i + batch]) for i in range(0, len(x), batch)]).float()


def fit_on_frozen(base_ev: torch.Tensor, feats: torch.Tensor, labels: torch.Tensor, tr, va, cfg: dict, seed: int,
                  device, module: nn.Module = None) -> tuple:
    """Stage 2: train an evidence module on top of frozen evidence base_ev (alpha = base_ev + module(x) + 1).
    With base_ev = 0 this trains the module alone.  Returns (FitResult, alpha for all rows)."""
    set_seed(seed)
    y = labels.numpy()
    model = (module if module is not None else EntropyEvidence(feats.shape[1], cfg["init_bias"])).to(device)
    cw = class_weights_of(labels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    idx = torch.from_numpy(tr).long()
    loader = DataLoader(TensorDataset(base_ev[idx], feats[idx], labels[idx]), batch_size=cfg["batch_size"],
                        shuffle=True)

    @torch.no_grad()
    def alpha_for(rows):
        model.eval()
        r = torch.from_numpy(rows).long()
        ev = torch.cat([model(feats[r][i:i + 2048].to(device)).cpu() for i in range(0, len(r), 2048)])
        return base_ev[r] + ev + 1.0

    def step_epoch(epoch):
        model.train()
        kl = kl_schedule(epoch, cfg)
        for b, xb, yb in loader:
            b, xb, yb = b.to(device), xb.to(device), yb.to(device)
            loss = edl_loss(b + model(xb) + 1.0, F.one_hot(yb, 2).float(), cw, kl)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
        sched.step()

    def val_score():
        a = alpha_for(va)
        return monitor_value(y[va], (a / a.sum(1, keepdim=True)).numpy(), cfg["monitor"])

    fit = _early_stopping_loop(model, step_epoch, val_score, cfg)
    return fit, alpha_for(np.arange(len(y)))
