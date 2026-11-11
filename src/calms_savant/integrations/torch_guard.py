"""Optional transactional LoRA steps using direct SGD, not Adam gradient scaling.

No torch import occurs in the core package. Real-model validation is separate.
"""
from collections import deque
import threading
import numpy as np
from ..archive import TopologicalCompressionStore
from ..numerics import matrix, unit, nonnegative
from ..policy import RiskConfig, decision
from ..transport import anchor_drift


class TorchLoRAGuard:
    def __init__(self, model, graph, anchor_probe, *, config=None, learning_rate=1e-3,
                 max_gradient_norm=1.0, archive_capacity=128):
        import torch
        self.torch = torch
        self.model, self.graph, self.anchor_probe = model, graph, anchor_probe
        self.config = config or RiskConfig(checkpoint_capacity=2)
        self.lr = nonnegative(learning_rate, "learning_rate")
        self.max_gradient_norm = nonnegative(max_gradient_norm, "max_gradient_norm")
        if self.lr == 0 or self.max_gradient_norm == 0:
            raise ValueError("positive LR and gradient bound required")
        self.parameters = {n: p for n, p in model.named_parameters() if "lora_" in n}
        if not self.parameters or any(p.requires_grad for n, p in model.named_parameters() if n not in self.parameters):
            raise ValueError("only named lora_ parameters may be trainable; freeze base and biases")
        self.archive = TopologicalCompressionStore(archive_capacity)
        self.baseline = self.probe()
        self.initial = self.snapshot()
        self.checkpoints = deque(maxlen=self.config.checkpoint_capacity)
        self.steps, self.frozen = 0, False
        self._lock = threading.RLock()

    def snapshot(self):
        return {n: p.detach().cpu().clone() for n, p in self.parameters.items()}

    def restore(self, state):
        with self.torch.no_grad():
            for name, p in self.parameters.items():
                p.copy_(state[name].to(device=p.device, dtype=p.dtype))

    def probe(self):
        previous = self.model.training
        self.model.eval()
        try:
            with self.torch.no_grad():
                raw = self.anchor_probe()
            return {k: unit(matrix(v)).copy() for k, v in raw.items()}
        finally:
            self.model.train(previous)

    def drift(self, a, b):
        return anchor_drift(a, b, **self.config.transport_kwargs)[0]

    def apply(self, original, gradients, rate):
        with self.torch.no_grad():
            for name, parameter in self.parameters.items():
                parameter.copy_(original[name].to(parameter.device, parameter.dtype))
                if name in gradients:
                    parameter.add_(gradients[name], alpha=-rate)

    def audit(self):
        drift = self.drift(self.baseline, self.probe())
        if not self.config.use_audit or drift <= self.config.audit_threshold:
            return None
        for step, state in reversed([(0, self.initial)] + list(self.checkpoints)):
            self.restore(state)
            if self.drift(self.baseline, self.probe()) <= self.config.audit_threshold/2:
                self.frozen = True
                return {"status": "rolled_back_frozen", "restored_step": step,
                        "historical_drift_before_restore": drift, "step": self.steps}
        self.frozen = True
        raise ArithmeticError("no eligible checkpoint; baseline probe no longer consistent")

    def step(self, loss_closure, representations, *, domain, inhibition=1.0, applicable_nodes=None):
        from ..lora import stratum_gate
        with self._lock:
            self.steps += 1
            if self.frozen:
                return {"status": "frozen", "step": self.steps}
            original = self.snapshot()
            training = self.model.training
            self.model.eval()  # deterministic proposals and probes; dropout disabled for this step
            try:
                if self.steps % self.config.audit_interval == 0:
                    report = self.audit()
                    if report:
                        return report
                z = unit(matrix(representations))
                before = self.probe()
                loss = loss_closure()
                scalar = float(loss.detach().cpu())
                if loss.ndim != 0 or not np.isfinite(scalar) or scalar < 0:
                    raise ValueError("loss closure must return a finite nonnegative scalar tensor")
                active = {n: p for n, p in self.parameters.items() if p.requires_grad}
                raw_grads = self.torch.autograd.grad(loss, tuple(active.values()), allow_unused=True)
                gradients = {n: g.detach() for n, g in zip(active, raw_grads) if g is not None}
                norm_sq = sum(float((g.double()**2).sum().cpu()) for g in gradients.values())
                norm = np.sqrt(norm_sq)
                if not np.isfinite(norm):
                    raise ValueError("nonfinite gradients")
                clip = min(1, self.max_gradient_norm/max(norm, 1e-30))
                gradients = {n: g*clip for n, g in gradients.items()}
                rate = self.lr*stratum_gate(inhibition)
                self.apply(original, gradients, rate)
                drift = self.drift(before, self.probe())
                self.restore(original)
                immune = self.graph.immune_penalty(z, domain=domain, node_ids=applicable_nodes)[0] if self.config.use_identity else 0.0
                recurrence = self.archive.recurrence(z, domain=domain) if self.config.use_archive else 0.0
                report = decision(self.config, task=scalar, drift=drift, immune=immune, recurrence=recurrence)
                if report["reasons"]:
                    if self.config.use_archive:
                        for i, v in enumerate(z):
                            self.archive.add(f"step-{self.steps}-{i}", v, step=self.steps,
                                             severity=report["critic"], domain=domain)
                    return {"status": "rejected", "step": self.steps, **report}
                self.apply(original, gradients, rate*report["attenuation"])
                actual = self.drift(before, self.probe())
                if self.config.use_hard_gates and actual > self.config.max_local_drift:
                    self.restore(original)
                    return {"status": "rejected", "step": self.steps, **report,
                            "attenuation": 0.0, "reasons": ["post_update_drift"]}
                if self.steps % self.config.audit_interval == 0:
                    audited = self.audit()
                    if audited:
                        return {**report, **audited}
                if self.steps % self.config.checkpoint_interval == 0:
                    self.checkpoints.append((self.steps, self.snapshot()))
                return {"status": "accepted", "step": self.steps, "actual_local_drift": actual,
                        "gradient_norm": float(norm), **report}
            except Exception as exc:
                self.restore(original)
                self.frozen = True
                return {"status": "invalid_frozen", "step": self.steps, "error": str(exc)}
            finally:
                self.model.train(training)

    def resume(self, *, review_note):
        if not isinstance(review_note, str) or not review_note.strip():
            raise ValueError("nonempty review note required")
        if self.drift(self.baseline, self.probe()) > self.config.audit_threshold/2:
            raise ValueError("restoration margin not satisfied")
        self.frozen = False
        return {"status": "resumed", "review_note": review_note.strip(), "step": self.steps}
