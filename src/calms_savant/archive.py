"""Bounded vector signatures. A signature cannot reconstruct discarded text."""
from dataclasses import dataclass, asdict
import numpy as np
from .numerics import unit, nonnegative, positive_int


@dataclass
class Signature:
    id: str
    vector: list
    step: int
    severity: float
    domain: str
    count: int = 1


class TopologicalCompressionStore:
    def __init__(self, capacity=128, *, redundancy=0.995, decay=0.005, backend="exact"):
        self.capacity = positive_int(capacity, "capacity")
        self.redundancy = nonnegative(redundancy, "redundancy")
        self.decay = nonnegative(decay, "decay")
        if self.redundancy > 1 or backend not in ("exact", "hnsw"):
            raise ValueError("invalid redundancy threshold or backend")
        if backend == "hnsw":
            try:
                import hnswlib  # noqa: F401
            except ImportError as exc:
                raise ImportError("install calms-savant-reference[hnsw]") from exc
        self.backend = backend
        self.entries = []
        self._index = None
        self._dimension = None

    def __len__(self):
        return len(self.entries)

    def add(self, key, vector, *, step, severity, domain="*"):
        v = unit(vector)
        if v.ndim != 1 or not key or not domain or isinstance(step, bool) or int(step) != step or step < 0:
            raise ValueError("invalid signature metadata")
        severity = nonnegative(severity, "severity")
        if self._dimension is not None and len(v) != self._dimension:
            raise ValueError("archive embedding dimension changed")
        if self.entries and step < max(e.step for e in self.entries):
            raise ValueError("archive steps must be monotone")
        self._dimension = len(v)
        for old in self.entries:
            if old.id == key and (old.domain != domain or not np.allclose(old.vector, v)):
                raise ValueError("signature ID collision")
        candidates = [(i, float(np.dot(e.vector, v))) for i, e in enumerate(self.entries) if e.domain == domain]
        nearest = max(candidates, key=lambda t: t[1], default=None)
        if nearest and nearest[1] >= self.redundancy:
            entry = self.entries[nearest[0]]
            entry.count += 1
            entry.step = int(step)
            entry.severity = max(entry.severity, severity)
        else:
            self.entries.append(Signature(str(key), v.tolist(), int(step), severity, domain))
        if len(self.entries) > self.capacity:
            # Deterministic temporal/severity eviction. Redundancy merged above.
            scores = [e.severity*np.exp(-self.decay*(step-e.step)) for e in self.entries]
            del self.entries[int(np.argmin(scores))]
        self._index = None

    def query(self, vector, k=3, *, domain=None):
        v = unit(vector)
        if v.ndim != 1:
            raise ValueError("query must be a vector")
        k = positive_int(k, "k")
        if not self.entries:
            return []
        if len(v) != self._dimension:
            raise ValueError("archive/query dimensions differ")
        ids = list(range(len(self.entries)))
        if self.backend == "hnsw":
            import hnswlib
            if self._index is None:
                index = hnswlib.Index(space="cosine", dim=len(v))
                index.init_index(max_elements=self.capacity, ef_construction=100, M=16, random_seed=0)
                index.add_items(np.array([e.vector for e in self.entries], dtype=np.float32), ids)
                index.set_num_threads(1)
                index.set_ef(max(50, min(k, len(ids))))
                self._index = index
            # Domain filtering needs all candidates in this small bounded store.
            self._index.set_ef(max(50, len(ids) if domain else min(k, len(ids))))
            labels, _ = self._index.knn_query(v.astype(np.float32), k=len(ids) if domain else min(k, len(ids)))
            ids = [int(i) for i in labels[0]]
        matches = [(self.entries[i], float(np.dot(self.entries[i].vector, v)))
                   for i in ids if domain is None or self.entries[i].domain == domain]
        return sorted(matches, key=lambda item: (-item[1], item[0].id))[:k]

    def recurrence(self, representations, *, domain="*", threshold=0.9):
        threshold = nonnegative(threshold, "threshold")
        if threshold >= 1:
            raise ValueError("recurrence threshold must be below one")
        scores = []
        for v in unit(representations):
            found = self.query(v, 1, domain=domain)
            scores.append(max(0.0, (found[0][1]-threshold)/(1-threshold)) if found else 0.0)
        return float(np.mean(scores))

    def to_dict(self):
        return {"capacity": self.capacity, "redundancy": self.redundancy,
                "decay": self.decay, "backend": self.backend,
                "entries": [asdict(e) for e in self.entries]}

    @classmethod
    def from_dict(cls, data):
        args = dict(data)
        entries = args.pop("entries")
        result = cls(**args)
        if len(entries) > result.capacity:
            raise ValueError("archive exceeds capacity")
        for raw in entries:
            e = Signature(**raw)
            if e.count < 1 or int(e.count) != e.count or isinstance(e.count, bool):
                raise ValueError("invalid signature count")
            v = unit(e.vector)
            if v.ndim != 1 or not e.id or not e.domain or not isinstance(e.step, int) or isinstance(e.step, bool) or e.step < 0:
                raise ValueError("invalid saved signature")
            if result._dimension is not None and len(v) != result._dimension:
                raise ValueError("archive dimension changed")
            if any(old.id == e.id for old in result.entries):
                raise ValueError("duplicate archived signature ID")
            result._dimension = len(v)
            e.vector = v.tolist()
            e.severity = nonnegative(e.severity, "severity")
            result.entries.append(e)
        return result
