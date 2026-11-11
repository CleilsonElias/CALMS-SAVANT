from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from calms_savant.engine import Consolidator, canonical
from calms_savant.identity import IdentityGraph, IdentityNode
from calms_savant.lora import LinearLoRA
from calms_savant.policy import RiskConfig
from calms_savant.benchmark import make_engine, nullspace_demo, run_benchmark, stream


class EngineTests(unittest.TestCase):
    def test_accepted_updates_parameters_not_base(self):
        e = make_engine()
        before = e.model.state_dict()
        v = np.eye(8)[4:5]
        r = e.consolidate(v, v+0.2*np.eye(8)[7:8], v, domain="facts")
        self.assertEqual(r["status"], "accepted")
        self.assertEqual(e.model.state_dict()["base"], before["base"])
        self.assertNotEqual(e.model.state_dict()["banks"], before["banks"])

    def test_rejected_update_is_bitwise_unchanged_and_archived(self):
        e = make_engine()
        before = e.model.state_dict()
        v = np.eye(8)[:1]
        r = e.consolidate(v, -v, -v, domain="protected")
        self.assertEqual(r["status"], "rejected")
        self.assertEqual(e.model.state_dict(), before)
        self.assertEqual(len(e.archive), 1)

    def test_historical_audit_restores_and_remains_frozen(self):
        eye = np.eye(2)
        m = LinearLoRA(eye, {"d": 1})
        m.banks["d"]["A"][:] = [[1, 0]]
        config = RiskConfig(use_identity=False, kappa=0, max_local_drift=1,
                            audit_interval=1, audit_threshold=0.2)
        e = Consolidator(m, IdentityGraph([IdentityNode("p", eye[0])]),
            {"a": eye[:1]}, config=config, learning_rate=1, max_gradient_norm=10)
        initial = m.state_dict()
        r = e.consolidate(eye[:1], -eye[:1], eye[:1], domain="d")
        self.assertEqual(r["status"], "rolled_back_frozen")
        self.assertEqual(e.model.state_dict(), initial)
        self.assertTrue(e.frozen)
        self.assertEqual(e.consolidate(eye[:1], eye[:1], eye[:1], domain="d")["status"], "frozen")
        with self.assertRaises(ValueError):
            e.resume(review_note=" ")
        e.resume(review_note="Inspected synthetic epoch-0 fixture")
        self.assertFalse(e.frozen)

    def test_audit_is_scheduled_even_when_next_proposal_would_be_rejected(self):
        e = make_engine()
        e.config = replace(e.config, audit_interval=1)
        # Simulate a prior accepted cumulative drift; current proposal is opposing.
        e.model.banks["protected"]["A"][:] = 0
        e.model.banks["protected"]["A"][0, 0] = 1
        e.model.banks["protected"]["B"][:] = 0
        e.model.banks["protected"]["B"][0, 0] = -2
        e.config = replace(e.config, audit_threshold=0.1)
        v = np.eye(8)[:1]
        r = e.consolidate(v, -v, -v, domain="protected")
        self.assertEqual(r["status"], "rolled_back_frozen")

    def test_invalid_candidate_cannot_leave_partial_parameters(self):
        e = make_engine()
        original = e.model.state_dict()
        v = np.eye(8)[:1]
        report = e.consolidate(v, [[1, 2]], v, domain="protected")
        self.assertEqual(report["status"], "invalid_frozen")
        self.assertEqual(e.model.state_dict(), original)
        self.assertTrue(e.frozen)

    def test_save_restore_and_exact_continuation(self):
        e = make_engine()
        rows = list(stream(0, 4))
        for _, r in rows[:11]:
            e.consolidate(r["x"], r["y"], r["z"], domain=r["domain"])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"state.json"
            e.save(path)
            other = Consolidator.load(path)
            self.assertEqual(e.to_dict(), other.to_dict())
            for _, r in rows[11:]:
                a = e.consolidate(r["x"], r["y"], r["z"], domain=r["domain"])
                b = other.consolidate(r["x"], r["y"], r["z"], domain=r["domain"])
                self.assertEqual(a, b)
            self.assertEqual(e.to_dict(), other.to_dict())

    def test_checksum_and_shape_corruption_are_detected(self):
        e = make_engine()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"state.json"
            e.save(path)
            saved = json.loads(path.read_text())
            saved["payload"]["step_count"] = 2
            path.write_text(json.dumps(saved))
            with self.assertRaises(ValueError):
                Consolidator.load(path)
            saved["payload"]["model"]["banks"]["facts"]["A"] = [[1, 2]]
            saved["sha256"] = hashlib.sha256(canonical(saved["payload"])).hexdigest()
            path.write_text(json.dumps(saved))
            with self.assertRaises(ValueError):
                Consolidator.load(path)

    def test_nullspace_failure_is_preserved_not_hidden(self):
        result = nullspace_demo()
        self.assertTrue(result["behavior_reversed"])
        self.assertEqual(result["max_proposed_anchor_drift"], 0)
        self.assertFalse(result["frozen"])

    def test_surrogate_is_reproducible_and_explicit(self):
        a = run_benchmark(seeds=1, epochs=2, variants=("full", "standard_replay"))
        b = run_benchmark(seeds=1, epochs=2, variants=("full", "standard_replay"))
        self.assertEqual(a, b)
        self.assertFalse(a["is_paper_reproduction"])
        self.assertEqual(a["attack_fraction"], 0.2)

    def test_event_chain_hash_matches_content(self):
        e = make_engine()
        v = np.eye(8)[:1]
        event = e.consolidate(v, v, v, domain="protected")
        claimed = event.pop("event_hash")
        self.assertEqual(hashlib.sha256(canonical(event)).hexdigest(), claimed)
