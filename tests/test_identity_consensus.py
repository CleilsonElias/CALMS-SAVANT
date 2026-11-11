import unittest
import numpy as np
from calms_savant.identity import IdentityGraph, IdentityNode
from calms_savant.consensus import seed_consensus


class IdentityTests(unittest.TestCase):
    def test_penalty_measures_each_sample_not_batch_argmax(self):
        graph = IdentityGraph([IdentityNode("p", [1, 0])])
        mean, each = graph.immune_penalty([[1, 0], [-1, 0]])
        self.assertGreater(mean, 0)
        self.assertEqual(each[0], 0)
        self.assertAlmostEqual(each[1], graph.weights[0])

    def test_weights_and_curated_updates(self):
        graph = IdentityGraph([IdentityNode("p", [1, 0])])
        self.assertTrue(((graph.weights > 0) & (graph.weights < 1)).all())
        with self.assertRaises(ValueError):
            graph.observe(["p"])
        graph.observe(["p"], trusted=True)
        self.assertAlmostEqual(graph.nodes[0].frequency, 1.0)
        graph.observe([], elapsed=2, trusted=True)
        self.assertAlmostEqual(graph.nodes[0].frequency, np.exp(-0.1))

    def test_scope_and_unknown_nodes(self):
        graph = IdentityGraph([IdentityNode("p", [1, 0], "privacy"), IdentityNode("f", [-1, 0], "facts")])
        self.assertEqual(graph.immune_penalty([[1, 0]], domain="privacy")[0], 0)
        with self.assertRaises(ValueError):
            graph.immune_penalty([[1, 0]], domain="absent")
        with self.assertRaises(ValueError):
            graph.immune_penalty([[1, 0]], node_ids=["absent"])

    def test_roundtrip_and_bipartite_centrality(self):
        graph = IdentityGraph([IdentityNode("a", [1, 0]), IdentityNode("b", [0, 1])], [[0, 1], [1, 0]])
        other = IdentityGraph.from_dict(graph.to_dict())
        np.testing.assert_allclose(other.weights, graph.weights)
        self.assertTrue((graph.centrality > 0).all())


class ConsensusTests(unittest.TestCase):
    def test_paper_mean_std_can_keep_strict_minority_outlier(self):
        seeds = [{"rule": [1, 0]}, {"rule": [1, 0]}, {"rule": [-1, 0]}]
        result = seed_consensus(seeds, method="paper_mean_std", alpha=2)
        self.assertEqual(result.rejected, ())
        self.assertLess(2, 0.9/np.std(result.scores))  # strict detectable shift 1 > delta=0.9

    def test_robust_heuristic_detects_this_example_without_claiming_a_theorem(self):
        seeds = [{"rule": [1, 0]}, {"rule": [1, 0]}, {"rule": [-1, 0]}]
        result = seed_consensus(seeds)
        self.assertEqual(result.rejected, (2,))
        np.testing.assert_allclose(result.directions["rule"], [1, 0])

    def test_identical_seeds_do_not_divide_by_zero(self):
        r = seed_consensus([{"r": [1, 0]}]*5)
        np.testing.assert_allclose(r.directions["r"], [1, 0])

    def test_no_silent_semantic_matching(self):
        with self.assertRaises(ValueError):
            seed_consensus([{"a": [1, 0]}, {"b": [1, 0]}, {"a": [1, 0]}])

    def test_zero_consensus_and_missing_majority_abstain(self):
        with self.assertRaises(ValueError):
            seed_consensus([{"a": [1, 0]}]*2 + [{"a": [-1, 0]}]*2, minimum_radius=2)
