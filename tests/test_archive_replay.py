import unittest
import importlib.util
import numpy as np
from calms_savant.archive import TopologicalCompressionStore
from calms_savant.replay import stratified_replay


class ArchiveTests(unittest.TestCase):
    def test_capacity_and_time_decay(self):
        store = TopologicalCompressionStore(3)
        for i, row in enumerate(np.eye(5)):
            store.add(str(i), row, step=i, severity=1)
        self.assertEqual(len(store), 3)
        self.assertEqual([x.id for x in store.entries], ["2", "3", "4"])

    def test_redundancy_is_merged(self):
        store = TopologicalCompressionStore(3)
        store.add("a", [1, 0], step=1, severity=1, domain="a")
        store.add("b", [1, 0], step=2, severity=2, domain="a")
        self.assertEqual(len(store), 1)
        self.assertEqual(store.entries[0].count, 2)
        self.assertEqual(store.entries[0].severity, 2)

    def test_query_scope_and_dimension(self):
        store = TopologicalCompressionStore(3)
        store.add("a", [1, 0], step=1, severity=1, domain="a")
        store.add("b", [0, 1], step=2, severity=1, domain="b")
        self.assertEqual(store.query([1, 0], domain="b")[0][0].id, "b")
        with self.assertRaises(ValueError):
            store.query([1, 0, 0])

    def test_roundtrip_preserves_nonchronological_storage_and_counts(self):
        store = TopologicalCompressionStore(3)
        store.add("a", [1, 0], step=1, severity=1)
        store.add("b", [0, 1], step=2, severity=1)
        store.add("a2", [1, 0], step=3, severity=2)
        recovered = TopologicalCompressionStore.from_dict(store.to_dict())
        self.assertEqual(store.to_dict(), recovered.to_dict())

    def test_normalized_nonaxis_vectors_survive_repeated_roundtrips(self):
        store = TopologicalCompressionStore(3)
        store.add("a", [0.31, 0.79, -0.51], step=1, severity=1)
        expected = store.to_dict()
        for _ in range(5):
            store = TopologicalCompressionStore.from_dict(store.to_dict())
        self.assertEqual(store.to_dict(), expected)

    def test_adversarial_replay_cap_is_strict_even_for_small_batches(self):
        for size in range(1, 33):
            result = stratified_replay(["good"], ["bad"], ["unknown"], size=size, seed=4)
            self.assertEqual(len(result), size)
            self.assertLessEqual(result.count("bad"), int(size*0.15))
        with self.assertRaises(ValueError):
            stratified_replay([], ["bad"], [], size=10)

    @unittest.skipUnless(importlib.util.find_spec("hnswlib"), "optional hnswlib not installed")
    def test_optional_hnsw_rebuilds_after_eviction(self):
        store = TopologicalCompressionStore(2, backend="hnsw")
        store.add("a", [1, 0, 0], step=1, severity=1)
        store.add("b", [0, 1, 0], step=2, severity=1)
        self.assertEqual(store.query([1, 0, 0], 1)[0][0].id, "a")
        store.add("c", [0, 0, 1], step=3, severity=1)
        self.assertEqual(store.query([0, 0, 1], 1)[0][0].id, "c")
