import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from simulation import population_search as search
from simulation.biped import Biped
from simulation.random_initial import randomize, lowest_surface
from simulation.tasks import make_policy, initial_parameters


class PopulationTests(unittest.TestCase):
    def test_exact_composition_and_unmodified_inheritance(self):
        previous = np.arange(30.).reshape(10, 3)
        scores = np.array([1, 4, 5, 0, 3, 6, 9, 8, 2, 7])
        best = np.array([100., 101., 102.])
        population, roles = search.next_population(best, previous, scores, np.random.default_rng(7))
        self.assertEqual(population.shape, (10, 3))
        np.testing.assert_array_equal(population[0], best)
        top = np.argsort(-scores)[:5]
        np.testing.assert_array_equal(population[3:8], previous[top])
        indices = [int(role.split(':')[1]) for role in roles[8:]]
        self.assertEqual(len(set(indices)), 2)
        self.assertFalse(set(indices) & set(top))
        np.testing.assert_array_equal(population[8:], previous[indices])
        self.assertEqual(roles[1:3], ['random', 'random'])
        boot, roles = search.next_population(best, None, None, np.random.default_rng(7))
        self.assertEqual(boot.shape, (10, 3))
        self.assertEqual(roles.count('bootstrap-random'), 9)

    def test_cache_persists_and_invalidates_by_every_evaluation_condition(self):
        calls = []
        def run(p, e):
            calls.append(e)
            return {'return': float(e['state']), 'survivalSeconds': 10.}, {'frames': []}
        with tempfile.TemporaryDirectory() as folder:
            cache = search.EvaluationCache(folder, {'duration': 10, 'code': 'A'})
            p = np.zeros(594)
            a, hit = cache.evaluate(p, {'state': 1}, run)
            self.assertFalse(hit)
            b, _ = cache.evaluate(p, {'state': 3}, run)
            self.assertEqual(search.aggregate([a, b])['return'], 2.)
            restored = search.EvaluationCache(folder, {'duration': 10, 'code': 'A'})
            self.assertTrue(restored.evaluate(p, {'state': 1}, run)[1])
            self.assertEqual(len(calls), 2)
            for ctx in [{'duration': 11, 'code': 'A'}, {'duration': 10, 'code': 'B'}]:
                self.assertFalse(search.EvaluationCache(folder, ctx).evaluate(p, {'state': 1}, run)[1])
            p[0] = .01
            self.assertFalse(cache.evaluate(p, {'state': 1}, run)[1])

    def test_random_pose_reproducible_free_finite_above_floor(self):
        robot = Biped(initial_pose='splayed-crawl')
        states = []
        for seed in range(20):
            state = randomize(robot, seed)
            states.append(state['qpos'])
            self.assertAlmostEqual(lowest_surface(robot), .001, places=10)
            self.assertAlmostEqual(np.linalg.norm(robot.data.qpos[3:7]), 1.)
            np.testing.assert_array_equal(robot.data.qvel, 0)
            for angle, joint in zip(robot.targets, robot.joints):
                self.assertTrue(joint['min'] <= angle <= joint['max'])
            robot.advance(dict(zip((j['id'] for j in robot.joints), robot.targets)))
            self.assertTrue(np.isfinite(robot.data.qpos).all())
        np.testing.assert_array_equal(randomize(robot, 0)['qpos'], states[0])
        self.assertFalse(np.array_equal(states[0], states[1]))

    def test_bias_learns_independently_and_keeps_neural_gain(self):
        binding = {'id': 'test', 'dataset': 'flywire-783', 'neurons': [{'id': '1'}]}
        p = initial_parameters(32, 'head-height')
        self.assertEqual(len(p), 594)
        p[:576] = 0
        p[-18:] = np.linspace(-.8, .8, 18)
        policy = make_policy(binding, 32, p, 'head-height')
        frame = {'bindingId': 'test', 'dataset': 'flywire-783', 'ratesHz': np.zeros(1)}
        np.testing.assert_allclose(policy.step(frame)['actions'], np.tanh(p[-18:]))
        self.assertEqual(policy.feature_gain, 20.)
        np.testing.assert_array_equal(policy.export_decoder()['bias'], p[-18:])

    def test_generations_resume_and_interrupted_episode_reuse(self):
        class Graph:
            output_binding = {'id': 'test', 'dataset': 'flywire-783', 'neurons': [{'id': '1'}]}
            input_binding = {'id': 'input'}
            manifest = {'id': 'graph', 'compiledChecksums': {'graph': 'fixed'}}
            ids = [1]; posts = [0]; input_indices = [0]; output_indices = [0]
        calls = []
        def rollout(graph, p, features, seed, seconds, **kwargs):
            calls.append((p.copy(), kwargs['initial_state_seed']))
            metrics = {'return': float(p[0] + kwargs['initial_state_seed']), 'seed': seed,
                       'survivalSeconds': seconds, 'meanHeadHeightM': .5, 'meanAbsNeuralAction': .1}
            return metrics, {'metrics': metrics, 'frames': []}
        args = SimpleNamespace(name='test', resume=False, init_from=None, population=10, task='head-height',
                               features=32, seed=35, seconds=.1, neural_dt_ms=.1, workers=1, iterations=1)
        with tempfile.TemporaryDirectory() as folder, patch.object(search, 'RUNS', Path(folder)), \
                patch.object(search, 'Connectome', Graph), patch.object(search, 'rollout', rollout):
            directory = search.train(args)
            self.assertEqual(len(calls), 20)  # baseline is candidate 0, not recomputed
            args.resume = True; args.iterations = 2
            search.train(args)
            self.assertEqual(len(calls), 24)  # only two new candidates × two episodes
            history = json.loads((directory / 'history.json').read_text())
            self.assertEqual([len(g['candidates']) for g in history], [10, 10])
            self.assertEqual(history[1]['candidates'][0]['origin'], 'best')
            args.iterations = 3
            original = rollout
            count_before = len(calls)
            def fail_after_one(*pos, **kw):
                if len(calls) > count_before:
                    raise RuntimeError('interrupted test')
                return original(*pos, **kw)
            with patch.object(search, 'rollout', fail_after_one), self.assertRaises(RuntimeError):
                search.train(args)
            self.assertEqual(len(calls), 25)
            search.train(args)
            self.assertEqual(len(calls), 28)  # cached partial generation used after restart
            best = json.loads((directory / 'best.json').read_text())
            values = best['objectiveSummary']['episodeReturns']
            self.assertEqual(best['objectiveSummary']['return'], sum(values) / 2)
            self.assertTrue((directory / 'best-second.json').exists())
