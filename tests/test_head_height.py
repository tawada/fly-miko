import unittest

import mujoco
import numpy as np

from simulation.biped import Biped
from simulation.closed_loop import reward, rollout
from simulation.tasks import initial_parameters
from simulation.population_search import digest as config_fingerprint
from unittest.mock import patch


class HeadHeightTaskTests(unittest.TestCase):
    def test_initial_pose_has_four_support_contacts_and_splayed_knees(self):
        robot = Biped(initial_pose="splayed-crawl")
        sample = robot.snapshot()
        self.assertFalse(robot.data.eq_active[0])
        np.testing.assert_array_equal(robot.data.qvel, 0)
        self.assertTrue(.5 < sample["headHeightM"] < .6)
        names = {mujoco.mj_id2name(robot.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
                 for c in robot.data.contact}
        self.assertEqual(names, {"left_shin_geom", "right_shin_geom",
                                 "left_hand_geom", "right_hand_geom"})
        positions = {}
        for side in ("left", "right"):
            index = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, side + "_shin")
            positions[side] = robot.data.xpos[index].copy()
        self.assertGreater(positions["left"][1] - positions["right"][1], .3)
        # Contact solver penetration is deliberately only 0.2mm.
        self.assertTrue(all(abs(c.dist) < .0003 for c in robot.data.contact))
        neck_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, "neck")
        np.testing.assert_allclose(robot.data.xmat[neck_id].reshape(3, 3), np.eye(3), atol=1e-8)
        for side in ("left", "right"):
            i = next(i for i, j in enumerate(robot.joints) if j["id"] == side + "_elbow")
            self.assertAlmostEqual(sample["jointAngles"][i], 1.)

    def test_reward_only_depends_on_world_head_height_and_keeps_low_poses(self):
        robot = Biped(initial_pose="splayed-crawl")
        sample = robot.snapshot()
        low, parts, done = reward(sample, robot, np.zeros(18), task="head-height")
        self.assertFalse(done)
        self.assertEqual(set(parts), {"headHeight"})
        sample["headHeightM"] += .5
        high, _, _ = reward(sample, robot, np.ones(18), task="head-height")
        self.assertAlmostEqual(high - low, .5 * robot.control_dt)
        sample["linearVelocity"] = [100., 100., 100.]
        sample["rootPositionM"][1] = .01
        sample["projectedGravity"] = [0., 1., 0.]
        unchanged, _, done = reward(sample, robot, np.zeros(18), task="head-height")
        self.assertEqual(high, unchanged)
        self.assertFalse(done)

    def test_ten_second_low_pose_episode_does_not_end_as_a_fall(self):
        # Exercise the complete physical rollout cheaply with a silent brain.
        binding = {"id": "test", "dataset": "flywire-783",
                   "neurons": [{"id": "1", "somaSide": "left"}]}
        class Graph:
            output_binding = binding
            input_binding = {
                "id": "test-input", "dataset": "flywire-783", "role": "ascending-input-candidates",
                "neurons": [{"id": str(i + 10), "side": "left" if i < 4 else "right"} for i in range(8)],
            }
        class SilentBrain:
            def __init__(self, *args, **kwargs): pass
            def advance(self, stimulus, dt):
                return {"bindingId": "test", "dataset": "flywire-783", "ratesHz": np.zeros(1),
                        "telemetry": {"spikeCount": 0, "outputSpikeCount": 0}}
        with patch("simulation.closed_loop.FlyBrain", SilentBrain):
            metrics, _ = rollout(Graph(), initial_parameters(32, "head-height"),
                                 seconds=10., task="head-height")
        self.assertEqual(metrics["termination"], "time_limit")
        self.assertAlmostEqual(metrics["survivalSeconds"], 10.)
        self.assertAlmostEqual(metrics["return"], metrics["meanHeadHeightM"] * 10.)

    def test_task_and_duration_change_resume_fingerprint(self):
        config = {"task": {"id": "walk-v1"}, "episodeSeconds": 2.5}
        old = config_fingerprint(config)
        self.assertNotEqual(old, config_fingerprint({**config, "task": {"id": "crawl-head-height-v1"}}))
        self.assertNotEqual(old, config_fingerprint({**config, "episodeSeconds": 10.}))


if __name__ == "__main__":
    unittest.main()
