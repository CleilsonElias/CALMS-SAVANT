import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional torch not installed")
class TorchGuardTests(unittest.TestCase):
    def test_real_autograd_update_and_rejected_rollback(self):
        import torch
        import numpy as np
        from calms_savant.integrations.torch_guard import TorchLoRAGuard
        from calms_savant.identity import IdentityGraph, IdentityNode
        class TinyLoRA(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.base = torch.nn.Parameter(torch.eye(2), requires_grad=False)
                self.lora_A = torch.nn.Parameter(torch.tensor([[0.2, 0.1]]))
                self.lora_B = torch.nn.Parameter(torch.zeros(2, 1))
            def forward(self, x):
                return x @ (self.base+self.lora_B @ self.lora_A).T
        model = TinyLoRA()
        x = torch.tensor([[1.0, 0.0]])
        graph = IdentityGraph([IdentityNode("p", [1, 0])])
        guard = TorchLoRAGuard(model, graph, lambda: {"a": model(x).detach().numpy()}, learning_rate=0.1)
        old = model.base.detach().clone()
        report = guard.step(lambda: ((model(x)-0.8*x)**2).mean(), [[1, 0]], domain="d")
        self.assertEqual(report["status"], "accepted")
        self.assertTrue(torch.equal(model.base, old))
        saved = guard.snapshot()
        report = guard.step(lambda: ((model(x)+x)**2).mean(), [[-1, 0]], domain="d")
        self.assertEqual(report["status"], "rejected")
        for name, value in guard.snapshot().items():
            self.assertTrue(torch.equal(value, saved[name]))
