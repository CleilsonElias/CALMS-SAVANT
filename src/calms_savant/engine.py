"""Single-process transactional proposals and historical audits for NumPy LoRA."""
from collections import deque
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import numpy as np
from .archive import TopologicalCompressionStore
from .identity import IdentityGraph
from .lora import LinearLoRA, stratum_gate
from .numerics import matrix, unit, nonnegative
from .policy import RiskConfig, decision
from .transport import anchor_drift


def canonical(data):
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=".calms-", delete=False) as f:
            temporary = f.name
            f.write(canonical(data))
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


class Consolidator:
    """Use one owner per model. This lock does not implement distributed serving."""
    def __init__(self, model, graph, anchors, *, config=None, inhibition=None,
                 archive=None, learning_rate=0.5, max_gradient_norm=1.0):
        self.model, self.graph = model, graph
        self.config = config or RiskConfig()
        self.learning_rate = nonnegative(learning_rate, "learning_rate")
        self.max_gradient_norm = nonnegative(max_gradient_norm, "max_gradient_norm")
        if self.learning_rate == 0 or self.max_gradient_norm == 0:
            raise ValueError("positive learning rate and gradient bound required")
        self.anchors = {str(k): matrix(v).copy() for k, v in anchors.items()}
        if not self.anchors:
            raise ValueError("at least one anchor required")
        self.inhibition = dict(inhibition or {d: 1.0 for d in model.banks})
        if set(self.inhibition) != set(model.banks):
            raise ValueError("inhibition domains must match LoRA banks")
        for value in self.inhibition.values():
            stratum_gate(value)
        if graph.directions.shape[1] != model.base.shape[0]:
            raise ValueError("identity/output dimensions differ")
        self.archive = archive if archive is not None else TopologicalCompressionStore()
        self.baseline = self.probe()
        self.initial_model = self.model.state_dict()
        self.step_count = 0
        self.frozen = False
        self.last_event_hash = "0"*64
        self.checkpoints = deque(maxlen=self.config.checkpoint_capacity)
        self._lock = threading.RLock()

    def probe(self):
        return {key: unit(self.model.forward(value)) for key, value in self.anchors.items()}

    def _drift(self, before, after):
        return anchor_drift(before, after, **self.config.transport_kwargs)[0]

    def _event(self, data):
        result = {"step": self.step_count, "previous_hash": self.last_event_hash, **data}
        result["event_hash"] = hashlib.sha256(canonical(result)).hexdigest()
        self.last_event_hash = result["event_hash"]
        return result

    def _record_rejection(self, z, domain, severity):
        if self.config.use_archive:
            for i, vector in enumerate(z):
                self.archive.add(f"step-{self.step_count}-{i}", vector,
                                 step=self.step_count, severity=severity, domain=domain)

    def _scheduled_audit(self):
        if not self.config.use_audit or self.step_count % self.config.audit_interval:
            return None
        historic = self._drift(self.baseline, self.probe())
        if historic <= self.config.audit_threshold:
            return None
        for checkpoint_step, state in reversed([(0, self.initial_model)] + list(self.checkpoints)):
            self.model.load_state_dict(state)
            if self._drift(self.baseline, self.probe()) <= self.config.audit_threshold/2:
                self.frozen = True
                return {"status": "rolled_back_frozen", "restored_step": checkpoint_step,
                        "historical_drift_before_restore": historic, "attenuation": 0.0}
        raise ArithmeticError("no eligible checkpoint; prior/probe consistency failed")

    def consolidate(self, inputs, targets, representations, *, domain,
                    applicable_nodes=None, negative_targets=None, contrastive_weight=0.0):
        with self._lock:
            z = unit(matrix(representations))
            x, y = matrix(inputs), matrix(targets)
            if x.shape[0] != z.shape[0] or y.shape[0] != z.shape[0]:
                raise ValueError("batch lengths differ")
            self.step_count += 1
            if self.frozen:
                return self._event({"status": "frozen", "domain": domain, "attenuation": 0.0})
            original = self.model.state_dict()
            try:
                scheduled = self._scheduled_audit()
                if scheduled:
                    return self._event({"domain": domain, **scheduled})
                before = self.probe()
                loss, gradient = self.model.loss_grad(x, y, domain,
                    negative_targets=negative_targets, contrastive_weight=contrastive_weight)
                rate = self.learning_rate*stratum_gate(self.inhibition[domain])
                norm, clipped = self.model.apply(domain, gradient, rate, max_gradient_norm=self.max_gradient_norm)
                proposed = self.probe()
                drift = self._drift(before, proposed)
                self.model.load_state_dict(original)
                immune = self.graph.immune_penalty(z, domain=domain, node_ids=applicable_nodes)[0] if self.config.use_identity else 0.0
                recurrence = self.archive.recurrence(z, domain=domain) if self.config.use_archive else 0.0
                risk = decision(self.config, task=loss, drift=drift, immune=immune, recurrence=recurrence)
                if risk["reasons"]:
                    self._record_rejection(z, domain, risk["critic"])
                    return self._event({"status": "rejected", "domain": domain,
                                        "gradient_norm": norm, "clipped_gradient_norm": clipped, **risk})
                self.model.apply(domain, gradient, rate*risk["attenuation"], max_gradient_norm=self.max_gradient_norm)
                final = self.probe()
                actual_drift = self._drift(before, final)
                if self.config.use_hard_gates and actual_drift > self.config.max_local_drift:
                    self.model.load_state_dict(original)
                    risk["reasons"].append("post_update_drift")
                    risk["attenuation"] = 0.0
                    self._record_rejection(z, domain, risk["critic"])
                    return self._event({"status": "rejected", "domain": domain, **risk})
                historic = self._drift(self.baseline, final)
                audited = self.config.use_audit and self.step_count % self.config.audit_interval == 0
                status = "accepted"
                restored_step = None
                if audited and historic > self.config.audit_threshold:
                    # Re-evaluate every retained candidate; always preserve epoch 0.
                    eligible = [(0, self.initial_model)] + list(self.checkpoints)
                    restored = False
                    for checkpoint_step, state in reversed(eligible):
                        self.model.load_state_dict(state)
                        if self._drift(self.baseline, self.probe()) <= self.config.audit_threshold/2:
                            restored_step, restored = checkpoint_step, True
                            break
                    if not restored:
                        self.model.load_state_dict(original)
                        raise ArithmeticError("no eligible checkpoint; prior/probe consistency failed")
                    self.frozen = True
                    status = "rolled_back_frozen"
                elif self.step_count % self.config.checkpoint_interval == 0:
                    self.checkpoints.append((self.step_count, self.model.state_dict()))
                return self._event({"status": status, "domain": domain, "actual_local_drift": actual_drift,
                    "historical_drift_before_restore": historic, "audited": audited,
                    "restored_step": restored_step, "gradient_norm": norm,
                    "clipped_gradient_norm": clipped, **risk})
            except (ValueError, ArithmeticError, FloatingPointError, KeyError) as exc:
                self.model.load_state_dict(original)
                # Malformed/numerically unverifiable proposals cannot modify the model.
                self.frozen = True
                return self._event({"status": "invalid_frozen", "domain": domain,
                                    "error": str(exc), "attenuation": 0.0})

    def resume(self, *, review_note):
        if not isinstance(review_note, str) or not review_note.strip():
            raise ValueError("a nonempty review note is required")
        with self._lock:
            if self._drift(self.baseline, self.probe()) > self.config.audit_threshold/2:
                raise ValueError("current model is not within the restoration margin")
            self.frozen = False
            return self._event({"status": "resumed", "review_note": review_note.strip()})

    def to_dict(self):
        return {"schema_version": 1, "model": self.model.state_dict(), "graph": self.graph.to_dict(),
            "anchors": {k: v.tolist() for k, v in self.anchors.items()},
            "baseline": {k: v.tolist() for k, v in self.baseline.items()},
            "initial_model": self.initial_model, "config": asdict(self.config),
            "inhibition": self.inhibition, "archive": self.archive.to_dict(),
            "learning_rate": self.learning_rate, "max_gradient_norm": self.max_gradient_norm,
            "step_count": self.step_count, "frozen": self.frozen,
            "last_event_hash": self.last_event_hash, "checkpoints": list(self.checkpoints)}

    def save(self, path):
        with self._lock:
            payload = self.to_dict()
            atomic_json(path, {"sha256": hashlib.sha256(canonical(payload)).hexdigest(), "payload": payload})

    @classmethod
    def load(cls, path):
        envelope = json.loads(Path(path).read_text(encoding="utf-8"))
        d = envelope["payload"]
        if hashlib.sha256(canonical(d)).hexdigest() != envelope["sha256"]:
            raise ValueError("state checksum mismatch (not an authenticity signature)")
        if d["schema_version"] != 1:
            raise ValueError("unsupported state schema")
        model = LinearLoRA.from_dict(d["model"])
        result = cls(model, IdentityGraph.from_dict(d["graph"]), d["anchors"],
                     config=RiskConfig(**d["config"]), inhibition=d["inhibition"],
                     archive=TopologicalCompressionStore.from_dict(d["archive"]),
                     learning_rate=d["learning_rate"], max_gradient_norm=d["max_gradient_norm"])
        # Validate checkpoints with the frozen base and original bank shapes.
        all_states = [(0, d["initial_model"])] + d["checkpoints"]
        if len(d["checkpoints"]) > result.config.checkpoint_capacity:
            raise ValueError("too many checkpoints")
        for step, state in all_states:
            if isinstance(step, bool) or not isinstance(step, int) or not 0 <= step <= d["step_count"]:
                raise ValueError("invalid checkpoint step")
            model.load_state_dict(state)
        expected_baseline = {k: unit(matrix(v)) for k, v in d["baseline"].items()}
        model.load_state_dict(d["initial_model"])
        if set(expected_baseline) != set(result.anchors) or any(
                not np.allclose(result.probe()[k], expected_baseline[k], atol=1e-10) for k in expected_baseline):
            raise ValueError("baseline does not match epoch-0 checkpoint")
        model.load_state_dict(d["model"])
        if not isinstance(d["frozen"], bool) or isinstance(d["step_count"], bool) or not isinstance(d["step_count"], int) or d["step_count"] < 0:
            raise ValueError("invalid lifecycle state")
        if len(d["last_event_hash"]) != 64 or any(c not in "0123456789abcdef" for c in d["last_event_hash"]):
            raise ValueError("invalid event hash")
        result.baseline, result.initial_model = expected_baseline, d["initial_model"]
        result.checkpoints.extend((step, state) for step, state in d["checkpoints"])
        result.step_count, result.frozen, result.last_event_hash = d["step_count"], d["frozen"], d["last_event_hash"]
        return result
