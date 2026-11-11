import unittest
import numpy as np
from calms_savant.lora import LinearLoRA, allocate_ranks, stratum_gate
from calms_savant.policy import RiskConfig, decision


class LoRATests(unittest.TestCase):
    def test_analytic_gradient_matches_finite_difference(self):
        rng = np.random.default_rng(1)
        m = LinearLoRA(np.eye(3), {"d": 2})
        m.banks["d"]["B"][:] = rng.normal(0, 0.1, (3, 2))
        x, y = rng.normal(size=(4, 3)), rng.normal(size=(4, 3))
        for contrastive in (False, True):
            opts = {"negative_targets": m.forward(x), "contrastive_weight": 0.4, "margin": 1.0} if contrastive else {}
            loss, gradient = m.loss_grad(x, y, "d", **opts)
            self.assertGreaterEqual(loss, 0)
            for key in ("A", "B"):
                parameter = m.banks["d"][key]
                for index in np.ndindex(parameter.shape):
                    old = parameter[index]
                    parameter[index] = old+1e-6
                    plus = m.loss_grad(x, y, "d", **opts)[0]
                    parameter[index] = old-1e-6
                    minus = m.loss_grad(x, y, "d", **opts)[0]
                    parameter[index] = old
                    self.assertAlmostEqual((plus-minus)/2e-6, gradient[key][index], places=6)

    def test_training_improves_loss_and_keeps_base_frozen(self):
        m = LinearLoRA(np.eye(2), {"d": 2})
        original = m.base.copy()
        x, y = np.eye(2), np.eye(2)*0.5
        first = m.loss_grad(x, y, "d")[0]
        for _ in range(100):
            _, g = m.loss_grad(x, y, "d")
            m.apply("d", g, 0.3)
        self.assertLess(m.loss_grad(x, y, "d")[0], first/2)
        np.testing.assert_array_equal(m.base, original)

    def test_clip_and_zero_rate_are_exact(self):
        m = LinearLoRA(np.eye(2), {"d": 1})
        original = m.state_dict()
        g = {k: np.full_like(v, 100.0) for k, v in m.banks["d"].items()}
        norm, clipped = m.apply("d", g, 0)
        self.assertGreater(norm, 1)
        self.assertEqual(clipped, 1)
        self.assertEqual(m.state_dict(), original)

    def test_rank_budget_integer_and_inhibition(self):
        ranks = allocate_ranks({"core": 8, "style": 0.5, "facts": 2}, budget=13)
        self.assertLessEqual(sum(ranks.values()), 13)
        self.assertLessEqual(ranks["core"], ranks["style"])
        self.assertTrue(all(isinstance(v, int) and v >= 1 for v in ranks.values()))
        with self.assertRaises(ValueError):
            allocate_ranks({"a": 0})
        self.assertEqual(stratum_gate(0.1), 1.0)

    def test_attenuation_monotone_but_not_absolute_block(self):
        c = RiskConfig(use_hard_gates=False)
        a = decision(c, task=0, drift=0, immune=0.1, recurrence=0)["attenuation"]
        b = decision(c, task=0, drift=0, immune=1, recurrence=0)["attenuation"]
        self.assertGreater(a, b)
        self.assertGreater(b, 0)

    def test_subnormal_inhibition_saturates_gate_before_power(self):
        for inhibition in (float.fromhex('0x0.0000000000001p-1022'), 1e-320, 1e-200):
            for beta in (1.0, 2.0, 1e308):
                self.assertEqual(stratum_gate(inhibition, beta=beta, optional_gate=0.3), 0.3)

    def test_subnormal_inhibition_respects_rank_cap_and_budget(self):
        self.assertEqual(allocate_ranks({'tiny': 1e-320}, max_rank=8, budget=8), {'tiny': 8})
        ranks = allocate_ranks({'tiny': 1e-320, 'ordinary': 2.0}, max_rank=8, budget=9)
        self.assertLessEqual(sum(ranks.values()), 9)
        self.assertTrue(all(isinstance(v, int) and 1 <= v <= 8 for v in ranks.values()))

    def test_ordinary_gate_values_unchanged_and_zero_rejected(self):
        for inhibition in (1e-4, 0.05, 0.5, 1.0, 2.0, 10.0, 1e4):
            for beta in (1.0, 2.0, 10.0):
                self.assertEqual(stratum_gate(inhibition, beta=beta, optional_gate=0.7),
                                 min(1.0, inhibition**(-beta))*0.7)
        with self.assertRaises(ValueError):
            stratum_gate(0)

    def test_large_inhibition_gate_underflow_stays_bounded(self):
        self.assertEqual(stratum_gate(1e308, beta=1e308), 0.0)
        self.assertEqual(allocate_ranks({'large': 1e308}, budget=8, max_rank=8), {'large': 1})

    def test_invalid_states_and_configs_rejected(self):
        with self.assertRaises(ValueError):
            RiskConfig(epsilon=0)
        with self.assertRaises(ValueError):
            RiskConfig(checkpoint_interval=11, audit_interval=10)
        with self.assertRaises(ValueError):
            LinearLoRA([[np.nan]], {"d": 1})
