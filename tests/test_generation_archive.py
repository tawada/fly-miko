import json
from pathlib import Path
import shutil
import tempfile
import unittest
import numpy as np

from simulation.generation_archive import selected, archive_available, export_recording
from simulation.population_search import ALGORITHM, EvaluationCache, next_population, aggregate, stalled_generations
from simulation.train import atomic_json, checkpoint


class GenerationArchiveTests(unittest.TestCase):
    def test_schedule(self):
        self.assertEqual([g for g in range(41) if selected(g)], [1, 2, 3, 4, 5, 10, 20, 30, 40])

    def test_backfill_all_candidates_survives_cache_removal(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            context = {'seed': 7}
            episodes = [{'seed': 1}, {'seed': 2}]
            config = {'algorithm': ALGORITHM, 'evaluationContext': context, 'episodes': episodes, 'fingerprint': 'test'}
            atomic_json(directory / 'config.json', config)
            cache = EvaluationCache(directory / 'evaluation-cache', context)
            rng = np.random.default_rng(7)
            initial = np.zeros(594)
            best = initial.copy(); best_score = None
            previous = scores = None
            history = []
            populations = {}
            def run(p, e):
                # Constant scores deliberately exercise both stagnation thresholds.
                metrics = {'return': float(e['seed']), 'survivalSeconds': 10.}
                return metrics, {'frames': [{'timeSeconds': 0, 'jointAngles': [float(p[0])]}], 'metrics': metrics}
            for g in range(1, 13):
                stagnation = stalled_generations(history)
                population, roles = next_population(best, previous, scores, rng, stagnation)
                populations[g] = population.copy()
                results = [aggregate([cache.evaluate(p, e, run)[0] for e in episodes]) for p in population]
                for r, role in zip(results, roles): r['origin'] = role
                scores = np.array([r['return'] for r in results])
                if best_score is None: best_score = scores[0]
                if scores.max() > best_score:
                    best_score = scores.max(); best = population[np.argmax(scores)].copy()
                history.append({'generation': g, 'candidates': results,
                                'algorithm': ALGORITHM, 'stagnationBefore': stagnation,
                                'generationBestReturn': float(scores.max())})
                previous = population
            state = {'initial': initial.tolist(), 'history': history, 'generation': 12}
            checkpoint(directory / 'latest.npz', fingerprint='test', state=json.dumps(state))
            self.assertEqual(archive_available(directory), [1, 2, 3, 4, 5, 10])
            for g in [1, 2, 3, 4, 5, 10]:
                archived = directory / 'generations' / f'{g:04d}'
                manifest = json.loads((archived / 'manifest.json').read_text())
                self.assertEqual(len(manifest['files']), 21)
                self.assertEqual(len(list(archived.glob('candidate-*.json.gz'))), 20)
                with np.load(archived / 'parameters.npz', allow_pickle=False) as data:
                    np.testing.assert_array_equal(data['population'], populations[g])
            self.assertFalse((directory / 'generations/0006').exists())
            # Atomic completion marker makes retry idempotent.
            manifest = directory / 'generations/0001/manifest.json'
            before = manifest.read_bytes()
            archive_available(directory)
            self.assertEqual(manifest.read_bytes(), before)
            shutil.rmtree(directory / 'evaluation-cache')
            exported = directory / 'export.json'
            export_recording(directory, 1, 10, 2, exported)
            recording = json.loads(exported.read_text())
            self.assertEqual(recording['frames'][0]['jointAngles'], [populations[1][9][0]])
            # Corruption cannot silently become a video input.
            file = directory / 'generations/0001/candidate-10-state-2.json.gz'
            file.write_bytes(b'corrupt')
            with self.assertRaises(ValueError):
                export_recording(directory, 1, 10, 2, exported)
