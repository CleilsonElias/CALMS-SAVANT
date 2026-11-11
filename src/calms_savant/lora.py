"""Analytic NumPy LoRA training, including optional defensive contrastive loss."""
import copy
import math
import numpy as np
from .numerics import matrix, nonnegative, positive_int


def allocate_ranks(inhibition, *, base_rank=16, budget=32, max_rank=16):
    base_rank, budget, max_rank = [positive_int(v, k) for v, k in
                                 zip((base_rank, budget, max_rank), ("base_rank", "budget", "max_rank"))]
    if not inhibition or budget < len(inhibition):
        raise ValueError("budget must fund at least rank one per domain")
    values = {k: nonnegative(v, "inhibition") for k, v in sorted(inhibition.items())}
    if any(v == 0 or not k for k, v in values.items()):
        raise ValueError("positive inhibition and domain IDs required")
    desired = {k: max(1, min(max_rank, int(math.floor(base_rank/v)))) for k, v in values.items()}
    result = {k: 1 for k in values}
    # Deterministic discrete allocation, never silently exceeds the rank budget.
    while sum(result.values()) < budget:
        eligible = [k for k in values if result[k] < desired[k]]
        if not eligible:
            break
        key = max(eligible, key=lambda k: (desired[k]/result[k], -values[k], k))
        result[key] += 1
    return result


def stratum_gate(inhibition, *, beta=1.0, optional_gate=1.0):
    inhibition, beta, optional_gate = [nonnegative(v, k) for v, k in
        zip((inhibition, beta, optional_gate), ("inhibition", "beta", "optional_gate"))]
    if inhibition == 0 or beta < 1 or optional_gate > 1:
        raise ValueError("invalid stratum gate")
    # Operational cap: lambda<1 must not amplify the update beyond the base LR.
    return min(1.0, inhibition**(-beta))*optional_gate


class LinearLoRA:
    def __init__(self, base, ranks, *, seed=0):
        self.base = matrix(base, "base").copy()
        out_dim, in_dim = self.base.shape
        if not ranks:
            raise ValueError("at least one adapter required")
        rng = np.random.default_rng(seed)
        self.banks = {}
        for domain, rank in sorted(ranks.items()):
            rank = positive_int(rank, "rank")
            if not domain or rank > min(in_dim, out_dim):
                raise ValueError("rank exceeds matrix dimensions or domain is empty")
            self.banks[domain] = {"A": rng.normal(0, 0.1, (rank, in_dim)),
                                  "B": np.zeros((out_dim, rank))}

    def forward(self, x):
        x = matrix(x, "inputs")
        if x.shape[1] != self.base.shape[1]:
            raise ValueError("input dimensions differ")
        # alpha/r = 1; all banks contribute at inference. Training selects one bank.
        weight = self.base + sum((b["B"] @ b["A"] for b in self.banks.values()), np.zeros_like(self.base))
        return x @ weight.T

    def loss_grad(self, x, y, domain, *, negative_targets=None, contrastive_weight=0.0, margin=0.2):
        x, y = matrix(x), matrix(y)
        if domain not in self.banks:
            raise ValueError("unknown training domain")
        output = self.forward(x)
        if output.shape != y.shape:
            raise ValueError("target dimensions differ")
        residual = output-y
        loss = float(np.mean(residual**2))
        derivative = 2*residual / residual.size
        weight = nonnegative(contrastive_weight, "contrastive_weight")
        margin = nonnegative(margin, "margin")
        if negative_targets is not None:
            neg = matrix(negative_targets)
            if neg.shape != output.shape:
                raise ValueError("negative target dimensions differ")
            positive_distance = np.mean((output-y)**2, axis=1)
            negative_distance = np.mean((output-neg)**2, axis=1)
            hinge = margin+positive_distance-negative_distance
            loss += weight*float(np.maximum(hinge, 0).mean())
            derivative += weight*(hinge > 0)[:, None]*2*(neg-y)/output.size
        elif weight:
            raise ValueError("contrastive training requires explicit rejected targets")
        dw = derivative.T @ x
        bank = self.banks[domain]
        return loss, {"A": bank["B"].T @ dw, "B": dw @ bank["A"].T}

    def apply(self, domain, gradient, rate, *, max_gradient_norm=1.0):
        rate = nonnegative(rate, "rate")
        bound = nonnegative(max_gradient_norm, "max_gradient_norm")
        if bound == 0 or domain not in self.banks or set(gradient) != {"A", "B"}:
            raise ValueError("invalid gradient configuration")
        checked = {k: matrix(v, "gradient") for k, v in gradient.items()}
        if any(checked[k].shape != self.banks[domain][k].shape for k in checked):
            raise ValueError("gradient dimensions differ")
        norm = float(np.sqrt(sum(np.sum(v*v) for v in checked.values())))
        scale = min(1.0, bound/max(norm, 1e-30))
        for key in checked:
            self.banks[domain][key] -= rate*scale*checked[key]
        return norm, min(norm, bound)

    def state_dict(self):
        return {"base": self.base.tolist(), "banks": {d: {k: v.tolist() for k, v in b.items()} for d, b in self.banks.items()}}

    def load_state_dict(self, data):
        candidate = self.from_dict(data)
        if candidate.base.shape != self.base.shape or set(candidate.banks) != set(self.banks):
            raise ValueError("incompatible model state")
        if any(candidate.banks[d][k].shape != self.banks[d][k].shape for d in self.banks for k in ("A", "B")):
            raise ValueError("incompatible adapter shape")
        if not np.array_equal(candidate.base, self.base):
            raise ValueError("frozen base changed")
        self.banks = candidate.banks

    @classmethod
    def from_dict(cls, data):
        base = matrix(data["base"])
        banks = data["banks"]
        ranks = {d: matrix(b["A"]).shape[0] for d, b in banks.items()}
        result = cls(base, ranks)
        for d, bank in banks.items():
            if set(bank) != {"A", "B"}:
                raise ValueError("invalid LoRA bank fields")
            for key, value in bank.items():
                value = matrix(value)
                if value.shape != result.banks[d][key].shape:
                    raise ValueError("invalid LoRA state shape")
                result.banks[d][key] = value.copy()
        return result
