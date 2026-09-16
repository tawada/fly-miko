import copy
import unittest
from unittest.mock import patch

from test_brain import tiny_graph, stimulus
from simulation import brain as brain_module
from simulation.activity import record_activity


class Projector:
    def __init__(self, binding, channels):
        pass

    def manifest(self):
        return {}

    def project(self, signal):
        return signal


class ActivityTests(unittest.TestCase):
    def setUp(self):
        self.graph = tiny_graph()
        self.graph.manifest = {'compiledChecksums': {'fixture': 'same'}}
        brain = brain_module.FlyBrain(self.graph, seed=35)
        self.recording = {'brainModel': self.graph.manifest, 'channels': [], 'inputRouting': {},
                          'metrics': {'seed': 35}, 'neuralParameters': {'dtMs': .1},
                          'controlDtSeconds': .02, 'durationSeconds': .06, 'policyFingerprint': 'fixture',
                          'frames': [{'timeSeconds': 0., 'sensory': stimulus(rate=200)}]}
        for i in range(3):
            result = brain.advance(stimulus(i * .02, 200))
            self.recording['frames'].append({'timeSeconds': (i + 1) * .02,
                'sensory': stimulus((i + 1) * .02, 200), 'brain': result['telemetry']})

    def test_bins_preserve_spikes_roles_and_kernel(self):
        kernel = brain_module.advance_lif
        with patch('simulation.activity.AscendingProjector', Projector):
            data = record_activity(self.recording, self.graph)
        self.assertIs(brain_module.advance_lif, kernel)
        self.assertEqual(data['roles'], [1, 2])
        self.assertEqual(data['neuronIds'], ['1', '2'])
        self.assertEqual(data['frames'][0]['counts'], [])
        for actual, expected in zip(data['frames'][1:], self.recording['frames'][1:]):
            counts = dict(actual['counts'])
            self.assertEqual(sum(counts.values()), expected['brain']['spikeCount'])
            self.assertEqual(counts.get(1, 0), expected['brain']['outputSpikeCount'])

    def test_divergence_rejected_and_hook_restored(self):
        recording = copy.deepcopy(self.recording)
        recording['frames'][1]['brain']['spikeCount'] += 1
        kernel = brain_module.advance_lif
        with patch('simulation.activity.AscendingProjector', Projector):
            with self.assertRaisesRegex(ValueError, 'Replay diverged'):
                record_activity(recording, self.graph)
        self.assertIs(brain_module.advance_lif, kernel)


if __name__ == '__main__':
    unittest.main()
