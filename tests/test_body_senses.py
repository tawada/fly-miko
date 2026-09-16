import copy
import json
import unittest

import numpy as np

from simulation.biped import Biped, BODY, CONTACT_PARTS, ROOT
from simulation.senses import SensoryEncoder, AscendingProjector
from simulation.tasks import initial_parameters, make_policy


class BodySenseTests(unittest.TestCase):
    def setUp(self):
        self.robot = Biped(initial_pose="splayed-crawl")
        self.encoder = SensoryEncoder(self.robot.mass * 9.81, "body-contact-gravity")
        self.sample = self.robot.snapshot()

    def test_twelve_parts_have_separate_load_and_event_channels(self):
        sample = copy.deepcopy(self.sample)
        sample.update(bodyContacts=[0] * 12, bodyContactForcesN=[0.] * 12)
        self.encoder.encode(sample)
        for i, part in enumerate(CONTACT_PARTS):
            sample["timeSeconds"] += .02
            sample["bodyContacts"] = [int(j == i) for j in range(12)]
            sample["bodyContactForcesN"] = [self.robot.mass * 9.81 * .5 if j == i else 0. for j in range(12)]
            signal = self.encoder.encode(sample)
            self.assertEqual(len(signal["values"]), 126)
            self.assertEqual(signal["values"][72 + i * 4:75 + i * 4], [1., .5, 1.])
            self.assertEqual(sum(signal["values"][72:120:4]), 1.)
            self.assertEqual(self.encoder.channels[72 + i * 4]["id"], part + ".contact")
            sample["timeSeconds"] += .02
            self.assertEqual(self.encoder.encode(sample)["values"][74 + i * 4], 0.)

    def test_gravity_signs_and_reset(self):
        sample = copy.deepcopy(self.sample)
        for vector, expected in [([0., -1., 0.], [0, 0, 0, 1, 0, 0]),
                                 ([1., 0., 0.], [1, 0, 0, 0, 0, 0]),
                                 ([0., 0., -1.], [0, 0, 0, 0, 0, 1])]:
            self.encoder.reset()
            sample["projectedGravity"] = vector
            signal = self.encoder.encode(sample)
            np.testing.assert_allclose(signal["values"][-6:], expected, atol=0, rtol=0)
            self.assertEqual(sum(signal["values"][74:120:4]), 0)
        self.encoder.reset()
        with self.assertRaises(ValueError):
            self.encoder.encode({**sample, "projectedGravity": [0, 0, 0]})
        self.encoder.encode(sample)

    def test_airborne_parts_do_not_report_ground_contact(self):
        sample = self.robot.reset(height=2.)
        np.testing.assert_array_equal(sample["bodyContacts"], 0)
        np.testing.assert_array_equal(sample["bodyContactForcesN"], 0)
        self.assertEqual(sample["bodyContactIds"], CONTACT_PARTS)

    @unittest.skipUnless((ROOT / "data/brain/compiled/input-binding.json").exists(), "Local input binding required")
    def test_sensor_routes_reach_real_input_neurons_and_keep_sides(self):
        binding = json.loads((ROOT / "data/brain/compiled/input-binding.json").read_text())
        projector = AscendingProjector(binding, self.encoder.channels)
        signal = self.encoder.encode(self.sample)
        stimulus = projector.project(signal)
        self.assertTrue(stimulus["externalRateHz"])
        self.assertEqual(stimulus["encoderId"], "miko-body-contact-gravity-126-v2")
        sides = {n["id"]: n["side"] for n in binding["neurons"]}
        for channel, routes in zip(self.encoder.channels, projector.routes):
            self.assertEqual(len(routes), 4)
            if channel["side"] != "both":
                self.assertTrue(all(sides[r["neuronId"]] == channel["side"] for r in routes))


class NeuralDriveTests(unittest.TestCase):
    def test_zero_initial_bias_holds_reference_and_neural_gain_remains(self):
        binding = {"id": "test", "dataset": "flywire-783", "neurons": [{"id": str(i)} for i in range(32)]}
        p = make_policy(binding, 32, initial_parameters(32, "head-height"), "head-height")
        self.assertEqual(p.parameter_count, 594)
        self.assertTrue(p.train_bias)
        self.assertTrue(np.any(p.weights))
        np.testing.assert_equal(p.bias, 0)
        frame = {"bindingId": "test", "dataset": "flywire-783", "ratesHz": np.zeros(32)}
        for _ in range(10):
            result = p.step(frame)
            np.testing.assert_array_equal(result["actions"], np.zeros(18))
            np.testing.assert_equal(result["angles"], p.reference_angles)
        frame["ratesHz"][0] = 50.
        result = p.step(frame)
        self.assertGreater(np.max(np.abs(result["actions"])), .05)
        self.assertGreater(np.max(np.abs(np.array(result["angles"]) - p.reference_angles)), .001)
        exported = p.export_decoder()
        expected = np.tanh(np.array(exported["weights"]) @ (p.filtered / 100) + exported["bias"])
        np.testing.assert_allclose(expected, result["actions"], atol=1e-12)
        for angle, joint in zip(result["angles"], BODY["joints"]):
            self.assertTrue(joint["min"] <= angle <= joint["max"])


if __name__ == "__main__":
    unittest.main()
