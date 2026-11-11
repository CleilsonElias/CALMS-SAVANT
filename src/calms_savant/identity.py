"""A curated directional prior, not a truth detector or a causal graph."""
from dataclasses import dataclass
import numpy as np
from .numerics import unit, matrix, nonnegative


@dataclass
class IdentityNode:
    id: str
    direction: np.ndarray
    domain: str = "*"
    frequency: float = 0.0
    reinforcement: float = 0.0


class IdentityGraph:
    def __init__(self, nodes, adjacency=None, *, alpha=1.0, beta=0.1,
                 gamma=1.0, decay=0.05):
        self.nodes = [IdentityNode(n.id, unit(n.direction).copy(), n.domain,
                                   nonnegative(n.frequency, "frequency"),
                                   nonnegative(n.reinforcement, "reinforcement")) for n in nodes]
        if not self.nodes or any(not n.id or not n.domain for n in self.nodes):
            raise ValueError("nonempty identified nodes required")
        if len({n.id for n in self.nodes}) != len(self.nodes):
            raise ValueError("duplicate identity node IDs")
        self.directions = matrix([n.direction for n in self.nodes])
        size = len(self.nodes)
        self.adjacency = np.zeros((size, size)) if adjacency is None else matrix(adjacency)
        if self.adjacency.shape != (size, size) or (self.adjacency < 0).any() or not np.allclose(self.adjacency, self.adjacency.T):
            raise ValueError("adjacency must be symmetric, nonnegative and aligned to nodes")
        self.adjacency = self.adjacency.copy()
        self.alpha, self.beta, self.gamma, self.decay = [nonnegative(v, k) for v, k in
            zip((alpha, beta, gamma, decay), ("alpha", "beta", "gamma", "decay"))]
        # Shift by identity to remove sign alternation on bipartite graphs.
        centrality = np.ones(size) / np.sqrt(size)
        shifted = self.adjacency + np.eye(size)
        for _ in range(1000):
            candidate = unit(shifted @ centrality)
            if np.linalg.norm(candidate - centrality) < 1e-10:
                centrality = candidate
                break
            centrality = candidate
        self.centrality = centrality

    @property
    def weights(self):
        f = np.array([n.frequency for n in self.nodes])
        r = np.array([n.reinforcement for n in self.nodes])
        logits = self.alpha*self.centrality + self.beta*f + self.gamma*r
        # Clip for floating point strictly positive, strictly below one weights.
        return 1 / (1 + np.exp(-np.clip(logits, -30, 30)))

    def observe(self, node_ids, *, elapsed=1.0, trusted=False):
        if not trusted:
            raise ValueError("only externally trusted activations may update the prior weights")
        elapsed = nonnegative(elapsed, "elapsed")
        ids = set(node_ids)
        if not ids <= {n.id for n in self.nodes}:
            raise ValueError("unknown node ID")
        for n in self.nodes:
            n.frequency = n.frequency*np.exp(-self.decay*elapsed) + (n.id in ids)

    def immune_penalty(self, representations, *, domain="*", node_ids=None):
        z = unit(matrix(representations))
        if z.shape[1] != self.directions.shape[1]:
            raise ValueError("identity/representation dimensions differ")
        known = {n.id for n in self.nodes}
        if node_ids is not None and not set(node_ids) <= known:
            raise ValueError("unknown applicable node IDs")
        keep = [i for i, n in enumerate(self.nodes)
                if (domain == "*" or n.domain in ("*", domain))
                and (node_ids is None or n.id in node_ids)]
        if not keep:
            raise ValueError("no applicable prior nodes for this domain")
        penalties = np.maximum(0, -(z @ self.directions[keep].T)) @ self.weights[keep]
        return float(penalties.mean()), penalties

    def to_dict(self):
        return {"nodes": [{"id": n.id, "direction": n.direction.tolist(),
                           "domain": n.domain, "frequency": n.frequency,
                           "reinforcement": n.reinforcement} for n in self.nodes],
                "adjacency": self.adjacency.tolist(), "alpha": self.alpha,
                "beta": self.beta, "gamma": self.gamma, "decay": self.decay}

    @classmethod
    def from_dict(cls, data):
        args = dict(data)
        args["nodes"] = [IdentityNode(**n) for n in args["nodes"]]
        return cls(**args)
