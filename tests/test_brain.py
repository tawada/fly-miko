import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from simulation.brain import FlyBrain
from simulation.connectome import COMPILED, Connectome
from simulation.policy import MotorReadout
from simulation.train import checkpoint
from simulation.biped import BODY
import tempfile


def tiny_graph(weight=2000.):
    return SimpleNamespace(
        ids=["1", "2"],
        input_binding={"id": "fixture-input", "dataset": "flywire-783", "neurons": [{"id": "1"}]},
        output_binding={"id": "fixture-output", "dataset": "flywire-783", "neurons": [{"id": "2"}]},
        input_indices=np.array([0], dtype=np.int32), output_indices=np.array([1], dtype=np.int32),
        indptr=np.array([0, 1, 1], dtype=np.int64), posts=np.array([1], dtype=np.int32),
        weights=np.array([weight], dtype=np.float32),
    )


def stimulus(t=0., rate=0.):
    return {"bindingId": "fixture-input", "dataset": "flywire-783", "timeSeconds": t,
            "encoding": "sparse-zero", "externalRateHz": {"1": rate}}


class OneExternalEvent:
    def __init__(self):
        self.first = True

    def random(self, shape, dtype):
        values = np.ones(shape, dtype=dtype)
        if self.first:
            values[0, 0] = 0.
            self.first = False
        return values


class BrainTests(unittest.TestCase):
    def test_silent_input_keeps_entire_graph_at_rest(self):
        brain = FlyBrain(tiny_graph())
        for index in range(3):
            result = brain.advance(stimulus(index * .02))
            self.assertEqual(result["telemetry"]["spikeCount"], 0)
        np.testing.assert_array_equal(brain.v, [-52., -52.])
        self.assertEqual(brain.active_count, 0)

    def test_spike_follows_real_edge_only_after_delay(self):
        brain = FlyBrain(tiny_graph())
        brain.rng = OneExternalEvent()
        first = brain.advance(stimulus(rate=200), seconds=.0018)
        self.assertEqual(first["telemetry"]["externalEvents"], 1)
        self.assertEqual(first["telemetry"]["outputSpikeCount"], 0)
        second = brain.advance(stimulus(.0018), seconds=.0001)
        self.assertEqual(second["telemetry"]["outputSpikeCount"], 1)
        self.assertEqual(brain.active_count, 2)

    def test_inhibitory_edge_does_not_masquerade_as_excitation(self):
        brain = FlyBrain(tiny_graph(-2000.))
        brain.rng = OneExternalEvent()
        result = brain.advance(stimulus(rate=200))
        self.assertEqual(result["telemetry"]["spikeCount"], 1)
        self.assertEqual(result["telemetry"]["outputSpikeCount"], 0)
        self.assertLess(brain.v[1], -52.)

    def test_exact_linear_decay_matches_closed_form(self):
        brain = FlyBrain(tiny_graph())
        brain.v[1], brain.g[1] = -54., 1.
        brain.active_indices[0], brain.active_mask[1], brain.active_count = 1, True, 1
        brain.advance(stimulus(), seconds=.001)
        em, es = math.exp(-1 / 20), math.exp(-1 / 5)
        expected = -52. + (-54. + 52.) * em + 5 / 15 * (em - es)
        self.assertAlmostEqual(float(brain.v[1]), expected, places=4)
        self.assertAlmostEqual(float(brain.g[1]), es, places=6)

    def test_seed_reset_is_repeatable(self):
        brain = FlyBrain(tiny_graph(), seed=35)
        first = brain.advance(stimulus(rate=150))
        brain.reset(35)
        second = brain.advance(stimulus(rate=150))
        np.testing.assert_array_equal(first["ratesHz"], second["ratesHz"])
        self.assertEqual(first["telemetry"]["spikeCount"], second["telemetry"]["spikeCount"])

    def test_wrong_clock_binding_rates_and_fractional_steps_are_rejected(self):
        brain = FlyBrain(tiny_graph())
        for data in [
            {**stimulus(), "bindingId": "other"}, stimulus(.01),
            {**stimulus(), "externalRateHz": {"999": 10}},
            {**stimulus(), "externalRateHz": {"1": float("nan")}},
            stimulus(rate=201),
        ]:
            with self.assertRaises(ValueError): brain.advance(data)
        with self.assertRaises(ValueError): brain.advance(stimulus(), .00015)
        self.assertEqual(brain.time_seconds, 0)
        with self.assertRaises(ValueError): FlyBrain(tiny_graph(), dt_ms=.13)


class PolicyTests(unittest.TestCase):
    def test_dense_export_matches_pooled_actions(self):
        binding = {"id": "fixture", "neurons": [{"id": str(i)} for i in range(1, 20)]}
        policy = MotorReadout(binding, feature_count=8)
        policy.set_parameters(np.random.default_rng(3).normal(0, .1, policy.parameter_count))
        frame = {"bindingId": "fixture", "dataset": "flywire-783", "ratesHz": np.arange(19) * 5.}
        result = policy.step(frame)
        decoder = policy.export_decoder()
        expected = np.tanh(np.array(decoder["weights"]) @ (policy.filtered / 100) + decoder["bias"])
        np.testing.assert_allclose(result["actions"], expected, atol=1e-12)
        self.assertEqual(decoder["jointIds"], [j["id"] for j in BODY["joints"]])

    def test_checkpoint_roundtrip_without_pickle(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.npz"
            checkpoint(path, best=np.array([1., 2.]), generation=3, rng_state=json.dumps({"state": 7}))
            with np.load(path, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["best"], [1., 2.])
                self.assertEqual(int(saved["generation"]), 3)
                self.assertEqual(json.loads(str(saved["rng_state"])), {"state": 7})


@unittest.skipUnless((COMPILED / "manifest.json").exists(), "Run brain:build for real-data checks")
class RealGraphTests(unittest.TestCase):
    def test_pinned_full_connectivity_and_bindings(self):
        graph = Connectome()
        self.assertEqual(len(graph.ids), 138639)
        self.assertEqual(len(graph.posts), 15091983)
        self.assertEqual(int(graph.indptr[-1]), len(graph.posts))
        self.assertTrue(np.any(graph.weights < 0))
        self.assertEqual(len(graph.input_indices), 1736)
        self.assertEqual(len(graph.output_indices), 1280)
        self.assertEqual(len(set(graph.input_indices) & set(graph.output_indices)), 0)
        self.assertEqual(len(graph.output_binding["excludedWithoutConnectivity"]), 4)


if __name__ == "__main__":
    unittest.main()
