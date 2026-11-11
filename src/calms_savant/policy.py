"""An attenuation coefficient is not a guarantee. Hard gates are explicit."""
from dataclasses import dataclass, asdict
import math
from .numerics import nonnegative, positive_int


@dataclass(frozen=True)
class RiskConfig:
    drift_weight: float = 1.5
    immune_weight: float = 3.0
    task_weight: float = 1.0
    archive_weight: float = 0.0
    kappa: float = 2.5
    max_local_drift: float = 0.12
    max_immune: float = 0.45
    max_archive: float = 1.0
    audit_threshold: float = 0.2
    audit_interval: int = 10
    checkpoint_interval: int = 1
    checkpoint_capacity: int = 32
    epsilon: float = 0.05
    sinkhorn_iterations: int = 1000
    sinkhorn_tolerance: float = 1e-8
    transport_mode: str = "divergence"
    metric: str = "euclidean"
    use_identity: bool = True
    use_archive: bool = True
    use_audit: bool = True
    use_hard_gates: bool = True

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key.startswith("use_"):
                if not isinstance(value, bool):
                    raise ValueError(f"{key} must be boolean")
            elif key in ("audit_interval", "checkpoint_interval", "checkpoint_capacity", "sinkhorn_iterations"):
                positive_int(value, key)
            elif key not in ("metric", "transport_mode"):
                nonnegative(value, key)
        if self.checkpoint_interval > self.audit_interval:
            raise ValueError("checkpoint_interval must not exceed audit_interval")
        if self.epsilon == 0 or self.sinkhorn_tolerance == 0 or self.audit_threshold == 0:
            raise ValueError("epsilon, tolerance and audit threshold must be positive")
        if self.metric not in ("euclidean", "cosine") or self.transport_mode not in ("divergence", "transport"):
            raise ValueError("invalid transport configuration")

    @property
    def transport_kwargs(self):
        return dict(epsilon=self.epsilon, iterations=self.sinkhorn_iterations,
                    tolerance=self.sinkhorn_tolerance, mode=self.transport_mode, metric=self.metric)


def decision(config, *, task, drift, immune, recurrence):
    task, drift, immune, recurrence = [nonnegative(v, k) for v, k in
        zip((task, drift, immune, recurrence), ("task", "drift", "immune", "recurrence"))]
    critic = config.task_weight*task + config.drift_weight*drift
    critic += config.immune_weight*immune + config.archive_weight*recurrence
    reasons = []
    if config.use_hard_gates:
        if drift > config.max_local_drift:
            reasons.append("local_drift")
        if config.use_identity and immune > config.max_immune:
            reasons.append("identity_opposition")
        if config.use_archive and recurrence > config.max_archive:
            reasons.append("archived_recurrence")
    factor = 0.0 if reasons else math.exp(-config.kappa*critic)
    return {"task_loss": task, "proposed_drift": drift, "immune_penalty": immune,
            "archive_recurrence": recurrence, "critic": critic,
            "attenuation": factor, "reasons": reasons}
