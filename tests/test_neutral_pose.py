import unittest
from unittest.mock import patch
import mujoco
import numpy as np
from simulation.biped import BODY, Biped
from simulation.closed_loop import rollout
from simulation.tasks import TASKS, initial_parameters, make_policy


class NeutralPoseTests(unittest.TestCase):
    def test_reference_has_no_crawl_angles(self):
        binding = {'id': 'test', 'dataset': 'flywire-783', 'neurons': [{'id': '1'}]}
        policy = make_policy(binding, 32, initial_parameters(32, 'head-height'), 'head-height')
        reference = dict(zip((j['id'] for j in BODY['joints']), policy.reference_angles))
        for side in ['left', 'right']:
            self.assertEqual(reference[f'{side}_elbow'], 0.)
            self.assertEqual(reference[f'{side}_shoulder_pitch'], 0.)
            self.assertEqual(reference[f'{side}_hip_pitch'], 0.)
        self.assertEqual(policy.parameter_count, 594)

    def test_physical_neck_is_neutral_relative_to_torso(self):
        robot = Biped(initial_pose=TASKS['head-height']['initialPose'])
        neck = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, 'neck')
        np.testing.assert_allclose(robot.model.body_quat[neck], [1, 0, 0, 0], atol=1e-12)

    def test_random_replay_has_no_crawl_neck_offset(self):
        class Graph:
            output_binding = {'id': 'test', 'dataset': 'flywire-783', 'neurons': [{'id': '1'}]}
            input_binding = {'id': 'input', 'dataset': 'flywire-783', 'role': 'ascending-input-candidates',
                             'neurons': [{'id': str(i+10), 'side': 'left' if i < 4 else 'right'} for i in range(8)]}
            manifest = {'id': 'test'}
        class SilentBrain:
            def __init__(self, *args, **kwargs): pass
            def advance(self, stimulus, dt):
                return {'bindingId': 'test', 'dataset': 'flywire-783', 'ratesHz': np.zeros(1),
                        'telemetry': {'spikeCount': 0, 'outputSpikeCount': 0}}
        with patch('simulation.closed_loop.FlyBrain', SilentBrain):
            _, recording = rollout(Graph(), initial_parameters(32, 'head-height'), seconds=.1,
                                   task='head-height', record=True, initial_state_seed=3035)
        self.assertEqual(recording['rigPoseOffsets'], {})
        self.assertEqual(recording['task']['initialPose'], 'random')
        elbows = [recording['frames'][0]['jointAngles'][recording['jointIds'].index(f'{side}_elbow')]
                  for side in ['left', 'right']]
        self.assertTrue(all(value != 1. for value in elbows))
        self.assertNotEqual(elbows[0], elbows[1])
