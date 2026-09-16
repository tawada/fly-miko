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
    def test_mutations_unique_elite_and_adaptive_scale(self):
        best = np.zeros(594)
        previous = np.zeros((10, 594))
        scores = np.zeros(10)
        population, roles = search.next_population(best, previous, scores, np.random.default_rng(7))
        np.testing.assert_array_equal(population[0], best)
        self.assertEqual(len({search.parameter_id(p) for p in population}), 10)
        self.assertEqual(sum(r.startswith('small-mutation:') for r in roles), 4)
        self.assertEqual(sum(r.startswith('large-mutation:') for r in roles), 3)
        self.assertEqual(roles[-2:], ['random', 'random'])
        wider, _ = search.next_population(best, previous, scores, np.random.default_rng(7), 10)
        np.testing.assert_allclose(wider[1:8], population[1:8] * 4)
        np.testing.assert_array_equal(wider[8:], population[8:])
        boot, _ = search.next_population(best, None, None, np.random.default_rng(7))
        np.testing.assert_array_equal(boot, population)
        self.assertEqual(search.mutation_settings(100), search.mutation_settings(10))

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
            self.assertEqual(len(calls), 38)  # nine new candidates × two episodes
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
            self.assertEqual(len(calls), 39)
            search.train(args)
            self.assertEqual(len(calls), 56)  # cached partial generation used after restart
            best = json.loads((directory / 'best.json').read_text())
            values = best['objectiveSummary']['episodeReturns']
            self.assertEqual(best['objectiveSummary']['return'], sum(values) / 2)
            self.assertTrue((directory / 'best-second.json').exists())
            from simulation.generation_archive import archive_available
            self.assertEqual(archive_available(directory), [1, 2, 3])
            # Evaluation condition changes must still be rejected.
            args.seconds = .2; args.iterations = 4
            with self.assertRaisesRegex(ValueError, 'conditions/code differ'):
                search.train(args)
            # A legacy checkpoint upgrades without losing evaluated episodes or
            # the ability to reconstruct generations on both sides of the switch.
            args.name = 'legacy'; args.seconds = .1; args.resume = False; args.iterations = 2
            def old_selection(best, previous, scores, rng, stagnation=0):
                return search.legacy_population(best, previous, scores, rng)
            with patch.object(search, 'ALGORITHM', search.LEGACY_ALGORITHM), \
                    patch.object(search, 'next_population', old_selection):
                legacy = search.train(args)
            original_checkpoint = (legacy / 'latest.npz').read_bytes()
            old_config = json.loads((legacy / 'config.json').read_text())
            calls_before = len(calls)
            args.resume = True; args.iterations = 3
            search.train(args)
            self.assertEqual(len(calls) - calls_before, 18)
            self.assertEqual((legacy / 'before-mutation-v2/latest.npz').read_bytes(), original_checkpoint)
            upgraded = json.loads((legacy / 'config.json').read_text())
            self.assertEqual(upgraded['fingerprint'], old_config['fingerprint'])
            self.assertEqual(upgraded['searchAlgorithm'], search.ALGORITHM)
            self.assertEqual(archive_available(legacy), [1, 2, 3])
