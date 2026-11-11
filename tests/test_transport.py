import unittest
import numpy as np
from calms_savant.transport import ground_cost, sinkhorn, distance, anchor_drift
from calms_savant.numerics import unit


class TransportTests(unittest.TestCase):
    def test_singletons_have_known_metric_distance(self):
        self.assertAlmostEqual(distance([[1, 0]], [[-1, 0]]), 1.0)
        self.assertAlmostEqual(distance([[1, 0]], [[0, 1]]), np.sqrt(2)/2)

    def test_self_divergence_is_zero_but_transport_can_be_biased(self):
        x = np.eye(3)
        self.assertAlmostEqual(distance(x, x, epsilon=0.5), 0.0, places=10)
        self.assertGreater(distance(x, x, epsilon=0.5, mode="transport"), 0)

    def test_marginals_rectangular_and_finite(self):
        rng = np.random.default_rng(2)
        r = sinkhorn(rng.normal(size=(7, 4)), rng.normal(size=(5, 4)), epsilon=0.1)
        self.assertTrue(r.converged)
        np.testing.assert_allclose(r.plan.sum(0), np.ones(5)/5, atol=1e-8)
        np.testing.assert_allclose(r.plan.sum(1), np.ones(7)/7, atol=1e-8)
        self.assertGreaterEqual(r.regularized_objective+1e-9, r.transport_cost)

    def test_permutation_cannot_hide_prompt_swaps(self):
        before = {"a": [[1, 0]], "b": [[0, 1]]}
        after = {"a": [[0, 1]], "b": [[1, 0]]}
        self.assertAlmostEqual(distance(np.eye(2), np.eye(2)[::-1]), 0.0, places=7)
        self.assertGreater(anchor_drift(before, after)[0], 0.7)

    def test_unknown_anchors_and_zero_vectors_rejected(self):
        with self.assertRaises(ValueError):
            anchor_drift({"a": [[1, 0]]}, {"b": [[1, 0]]})
        for x in ([[0, 0]], [[np.nan, 1]], []):
            with self.assertRaises(ValueError):
                distance(x, [[1, 0]])

    def test_cosine_normalized_range(self):
        self.assertEqual(ground_cost([[1, 0]], [[-1, 0]], "cosine")[0, 0], 1)

    def test_amplitude_only_changes_are_invisible(self):
        before = {"a": [[1, 0]], "b": [[0, 1]]}
        after = {"a": [[100, 0]], "b": [[0, 0.001]]}
        self.assertEqual(anchor_drift(before, after)[0], 0)

    def test_nonconvergence_is_not_a_safe_distance(self):
        rng = np.random.default_rng(3)
        with self.assertRaises(ArithmeticError):
            distance(rng.normal(size=(5, 3)), rng.normal(size=(9, 3)),
                     epsilon=0.001, iterations=1, tolerance=1e-12)

    def test_matches_independent_exact_lp_in_small_epsilon_limit(self):
        try:
            from scipy.optimize import linprog
        except ImportError:
            self.skipTest("optional independent SciPy LP oracle unavailable")
        rng = np.random.default_rng(9)
        x, y = rng.normal(size=(3, 3)), rng.normal(size=(4, 3))
        cost = ground_cost(x, y)
        equality = []
        for i in range(3):
            row = np.zeros((3, 4)); row[i, :] = 1; equality.append(row.ravel())
        for j in range(4):
            row = np.zeros((3, 4)); row[:, j] = 1; equality.append(row.ravel())
        exact = linprog(cost.ravel(), A_eq=equality, b_eq=[1/3]*3+[1/4]*4,
                        bounds=(0, None), method="highs")
        self.assertTrue(exact.success)
        approximate = sinkhorn(x, y, epsilon=0.002, iterations=20000)
        self.assertTrue(approximate.converged)
        self.assertAlmostEqual(approximate.transport_cost, exact.fun, delta=0.003)
