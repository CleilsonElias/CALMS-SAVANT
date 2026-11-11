"""Train approved positive targets away from explicit rejected targets.

This example uses synthetic numeric evidence. Never replay rejected text as a
positive language-model target just because it was retrieved from the archive.
"""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from calms_savant.lora import LinearLoRA

model = LinearLoRA(np.eye(3), {"defensive": 2}, seed=7)
x = np.eye(3)[:1]
approved, rejected = x.copy(), -x.copy()
model.banks["defensive"]["B"][:] = 0.2
losses = []
for _ in range(60):
    loss, gradient = model.loss_grad(x, approved, "defensive", negative_targets=rejected,
                                     contrastive_weight=0.5, margin=2.0)
    losses.append(loss)
    model.apply("defensive", gradient, 0.2)
print(json.dumps({"initial_loss": losses[0], "final_loss": losses[-1],
                  "approved_target_source": "explicit synthetic fixture",
                  "output": model.forward(x).tolist()}, indent=2))
