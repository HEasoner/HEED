"""Load a trained HEED checkpoint and score new (h, H-bar) pairs."""
import torch

from .data import standardize
from .model import EntropyEvidence, EvidentialMLP


class HEEDDetector:
    def __init__(self, ckpt: dict):
        self.ckpt = ckpt
        self.eh = EvidentialMLP(ckpt["eh"]["input_dim"], ckpt["eh"]["hidden_dim"], ckpt["eh"]["dropout"])
        self.eh.load_state_dict(ckpt["eh"]["state_dict"])
        self.eH = EntropyEvidence(1)
        self.eH.load_state_dict(ckpt["eH"]["state_dict"])
        self.eh.eval()
        self.eH.eval()

    @classmethod
    def load(cls, path):
        return cls(torch.load(str(path), map_location="cpu"))

    @torch.no_grad()
    def alpha(self, h: torch.Tensor, hbar: torch.Tensor) -> torch.Tensor:
        """h: N x 3d answer-token mean hidden states (layers -4, -9, -14); hbar: N (keyword mean entropy)."""
        c = self.ckpt
        zh = standardize(h.float(), c["eh"]["mean"], c["eh"]["std"])
        zH = standardize(hbar.float().reshape(-1, 1), c["eH"]["mean"], c["eH"]["std"])
        return self.eh(zh) + self.eH(zH) + 1.0

    def score(self, h: torch.Tensor, hbar: torch.Tensor) -> torch.Tensor:
        """s_FRH = alpha_HALL / (alpha_HALL + alpha_KNOW); higher = more likely a false recall."""
        a = self.alpha(h, hbar)
        return a[:, 1] / a.sum(1)
