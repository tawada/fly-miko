"""Ten-slot retained population, evaluated on two fixed random initial states."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import time

import numpy as np

from .biped import BODY, ROOT
from .closed_loop import rollout
from .connectome import Connectome
from .random_initial import VERSION
from .storage import atomic_output, check_output_directory
from .tasks import TASKS, initial_parameters, make_policy
from .train import atomic_json, checkpoint

LEGACY_ALGORITHM = 'retained-population-two-states-v1'
ALGORITHM = 'mutated-population-two-states-v2'
RUNS = ROOT / 'runs/training'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def parameter_id(parameters):
    array = np.asarray(parameters, dtype='<f8')
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError('Parameters must be a finite vector')
    return hashlib.sha256(array.tobytes()).hexdigest()


def legacy_population(best, previous, scores, rng):
    """Slots: incumbent + 2 independent random + 5 elite + 2 non-elite draws."""
    size = len(best)
    fresh = lambda n: rng.normal(0., .15, (n, size))
    if previous is None:
        return np.vstack([best, fresh(9)]), ['best'] + ['bootstrap-random'] * 9
    previous = np.asarray(previous)
    scores = np.asarray(scores)
    if previous.shape != (10, size) or scores.shape != (10,) or not np.isfinite(scores).all():
        raise ValueError('Previous generation must have ten finite evaluated candidates')
    order = np.argsort(-scores, kind='stable')
    random_indices = rng.choice(order[5:], size=2, replace=False)
    population = np.vstack([best, fresh(2), previous[order[:5]], previous[random_indices]])
    roles = ['best', 'random', 'random'] + [f'previous-top:{i}' for i in order[:5]] + [f'previous-random:{i}' for i in random_indices]
    return population, roles


def mutation_settings(stagnation):
    # Increase exploration every five unsuccessful generations, with a cap.
    scale = min(4., 2. ** (min(stagnation, 10) // 5))
    return {'smallSigma': .02 * scale, 'largeSigma': .10 * scale}


def next_population(best, previous, scores, rng, stagnation=0):
    """One elite, four small mutations, three large mutations, two immigrants."""
    best = np.asarray(best, dtype=float)
    parameter_id(best)  # validate finite vector
    parents = [best]
    seen = {parameter_id(best)}
    if previous is not None:
        previous, scores = np.asarray(previous), np.asarray(scores)
        if previous.shape != (10, len(best)) or scores.shape != (10,) or not np.isfinite(scores).all():
            raise ValueError('Previous generation must have ten finite evaluated candidates')
        for i in np.argsort(-scores, kind='stable'):
            key = parameter_id(previous[i])
            if key not in seen:
                parents.append(previous[i])
                seen.add(key)
            if len(parents) == 5:
                break
    settings = mutation_settings(stagnation)
    population, roles = [best.copy()], ['best']
    seen = {parameter_id(best)}
    for kind, count in [('small', 4), ('large', 3), ('random', 2)]:
        for i in range(count):
            parent = parents[i % len(parents)] if kind == 'small' else parents[int(rng.integers(len(parents)))]
            for attempt in range(100):
                child = (rng.normal(0., .15, len(best)) if kind == 'random' else
                         parent + rng.normal(0., settings[f'{kind}Sigma'], len(best)))
                key = parameter_id(child)
                if key not in seen:
                    break
            else:
                raise ValueError('Unable to generate distinct candidates')
            seen.add(key)
            population.append(child)
            roles.append('random' if kind == 'random' else f'{kind}-mutation:{parameter_id(parent)}')
    return np.asarray(population), roles


def stalled_generations(history):
    count = 0
    for record in reversed(history):
        # Slot zero is the incumbent before this generation.
        if record['generationBestReturn'] > record['candidates'][0]['return']:
            break
        count += 1
    return count


def evaluation_conditions(context):
    """Selection code does not affect an individual rollout's value."""
    result = {k: v for k, v in context.items() if k not in ('algorithm', 'population')}
    result['codeChecksums'] = {k: v for k, v in context['codeChecksums'].items()
                               if k != 'population_search.py'}
    return result


class EvaluationCache:
    """Each atomically committed episode includes metrics and the exact replay."""
    def __init__(self, directory, context):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.context = context
        self.context_id = digest(context)

    def key(self, parameters, episode):
        return digest({'context': self.context_id, 'parameters': parameter_id(parameters), 'episode': episode})

    def read(self, key):
        path = self.directory / f'{key}.json.gz'
        if not path.exists():
            return None
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            result = json.load(stream)
        payload = result['payload']
        if result['sha256'] != digest(payload) or payload['key'] != key or payload['contextId'] != self.context_id:
            raise ValueError(f'Invalid evaluation cache: {path}')
        return payload

    def evaluate(self, parameters, episode, run):
        key = self.key(parameters, episode)
        cached = self.read(key)
        if cached is not None:
            return cached, True
        metrics, recording = run(parameters, episode)
        if not math.isfinite(metrics['return']):
            raise ValueError('Nonfinite objective')
        payload = {'key': key, 'contextId': self.context_id, 'parameterId': parameter_id(parameters),
                   'episode': episode, 'metrics': metrics, 'recording': recording}
        data = json.dumps({'payload': payload, 'sha256': digest(payload)}, allow_nan=False).encode()
        with atomic_output(self.directory / f'{key}.json.gz') as stream:
            with gzip.GzipFile(fileobj=stream, mode='wb', mtime=0) as compressed:
                compressed.write(data)
        return payload, False


def aggregate(entries):
    if len(entries) != 2:
        raise ValueError('Exactly two initial states are required')
    # The dashboard's numeric summary fields all describe the same two episodes.
    first = entries[0]['metrics']
    result = {k: float(np.mean([e['metrics'][k] for e in entries])) for k, value in first.items()
              if isinstance(value, (int, float)) and not isinstance(value, bool) and k != 'seed'}
    result.update({'episodes': [{'cacheKey': e['key'], **e['metrics']} for e in entries],
                   'parameterId': entries[0]['parameterId'], 'objectiveEpisodeCount': 2})
    return result


def train(args):
    if args.population != 10 or args.task != 'head-height':
        raise ValueError('This scenario requires --population 10 --task head-height')
    name = args.name or datetime.now(timezone.utc).strftime('random-%Y%m%d-%H%M%S')
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', name):
        raise ValueError('Invalid run name')
    directory = RUNS / name
    if directory.exists() and not args.resume:
        raise ValueError('Run already exists; choose a new --name or use --resume')
    if args.resume and not (directory / 'latest.npz').exists():
        raise ValueError('No checkpoint to resume')
    # Refuse old CEM checkpoints before loading the large graph or altering files.
    old_config = json.loads((directory / 'config.json').read_text()) if args.resume else None
    if old_config and old_config.get('algorithm') not in (ALGORITHM, LEGACY_ALGORITHM):
        raise ValueError('Old scenario cannot resume here. Use a new --name and optionally --init-from OLD/latest.npz')
    if args.resume and args.init_from:
        raise ValueError('--init-from is only for a new run')
    graph = Connectome()
    policy = make_policy(graph.output_binding, args.features, None, args.task)
    episodes = [{'initialStateSeed': args.seed + 3000 + i, 'brainSeed': args.seed + 1000 + i} for i in range(2)]
    context = {
        'task': TASKS[args.task], 'initialStateDistribution': VERSION,
        'episodes': episodes, 'seconds': args.seconds, 'features': args.features,
        'seed': args.seed, 'neuralDtMs': args.neural_dt_ms, 'population': 10,
        'graphId': graph.manifest['id'], 'graphChecksums': graph.manifest['compiledChecksums'],
        'inputBindingHash': digest(graph.input_binding), 'outputBindingHash': digest(graph.output_binding),
        'bodyHash': digest(BODY),
        'codeChecksums': {f: hashlib.sha256((ROOT / 'simulation' / f).read_bytes()).hexdigest()
                          for f in ['brain.py', 'biped.py', 'senses.py', 'policy.py', 'closed_loop.py',
                                    'tasks.py', 'random_initial.py']},
    }
    fingerprint = digest(context)
    config = {'name': name, 'createdAt': datetime.now(timezone.utc).isoformat(),
              'algorithm': ALGORITHM, 'evaluationContext': context, 'fingerprint': fingerprint,
              'task': {**TASKS[args.task], 'initialPose': 'random'}, 'population': 10,
              'episodeSeconds': args.seconds, 'objectiveEpisodeCount': 2, 'evaluationSeedCount': 2,
              'featureCount': args.features, 'parameterCount': policy.parameter_count,
              'neuralDtMs': args.neural_dt_ms, 'brainNeuronCount': len(graph.ids),
              'brainConnectionCount': len(graph.posts), 'inputNeuronCount': len(graph.input_indices),
              'outputNeuronCount': len(graph.output_indices), 'bodyMapId': BODY['id'],
              'episodes': episodes, 'randomParameterStd': .15}
    initial = initial_parameters(args.features, args.task)
    if args.init_from:
        source = Path(args.init_from)
        with np.load(source, allow_pickle=False) as saved:
            initial = saved['best'].copy()
        # Former bias-free checkpoints migrate by appending zero biases; old scores are never reused.
        if initial.shape == (18 * args.features,):
            initial = np.concatenate([initial, np.zeros(18)])
        if initial.shape != (policy.parameter_count,) or not np.isfinite(initial).all():
            raise ValueError('Initial checkpoint parameter dimensions or values are invalid')
        source_config = json.loads((source.parent / 'config.json').read_text())
        if (source_config.get('featureCount') != args.features or source_config.get('task', {}).get('id') not in ('crawl-head-height-v2', 'crawl-head-height-v3-bias', TASKS[args.task]['id'])):
            raise ValueError('Initial checkpoint must use the same readout and head-height task')
        config['initializedFrom'] = {'path': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
    rng = np.random.default_rng(args.seed)
    state = {'generation': 0, 'best': initial.tolist(), 'initial': initial.tolist(), 'bestScore': None,
             'population': None, 'scores': None, 'history': [], 'rng': rng.bit_generator.state}
    if args.resume:
        if (digest(old_config['evaluationContext']) != old_config.get('fingerprint') or
                evaluation_conditions(old_config['evaluationContext']) != evaluation_conditions(context)):
            raise ValueError('Resume evaluation conditions/code differ; cached objectives must not be mixed')
        # Keep the old cache namespace and historical archive fingerprints.
        context = old_config['evaluationContext']
        fingerprint = old_config['fingerprint']
        with np.load(directory / 'latest.npz', allow_pickle=False) as saved:
            if str(saved['fingerprint']) != fingerprint:
                raise ValueError('Checkpoint fingerprint mismatch')
            state = json.loads(str(saved['state']))
        if args.iterations <= state['generation']:
            raise ValueError('--iterations must exceed saved generation (total generation target)')
        config = old_config
        initial = np.asarray(state['initial'])
        rng.bit_generator.state = state['rng']
    check_output_directory(directory)
    if args.resume and config.get('searchAlgorithm', config['algorithm']) != ALGORITHM:
        backup = directory / 'before-mutation-v2'
        backup.mkdir(exist_ok=True)
        for filename in ('config.json', 'latest.npz'):
            target = backup / filename
            if not target.exists():
                with atomic_output(target) as output:
                    output.write((directory / filename).read_bytes())
        config['algorithmChanges'] = config.get('algorithmChanges', []) + [
            {'fromGeneration': state['generation'] + 1, 'algorithm': ALGORITHM}]
    config['searchAlgorithm'] = ALGORITHM
    atomic_json(directory / 'config.json', config)
    cache = EvaluationCache(directory / 'evaluation-cache', context)
    started = time.perf_counter()
    new_evaluations = 0
    hits = 0

    def progress(status, **extra):
        atomic_json(directory / 'progress.json', {
            'name': name, 'status': status, 'generation': state['generation'],
            'targetGenerations': args.iterations, 'bestReturn': state['bestScore'],
            'completedEvaluations': len(list(cache.directory.glob('*.json.gz'))),
            'newEvaluationsThisSession': new_evaluations, 'cacheHitsThisSession': hits,
            'wallSecondsThisSession': time.perf_counter() - started, 'history': state['history'], **extra})

    def run(parameters, episode):
        return rollout(graph, parameters, args.features, episode['brainSeed'], args.seconds,
                       record=True, neural_dt_ms=args.neural_dt_ms, task=args.task,
                       initial_state_seed=episode['initialStateSeed'])

    def evaluate(population):
        nonlocal new_evaluations, hits
        # Deduplicate before scheduling to prevent two workers computing the same key.
        jobs = {cache.key(p, e): (p, e) for p in population for e in episodes}
        entries = {}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(cache.evaluate, p, e, run): key for key, (p, e) in jobs.items()}
            for future in as_completed(futures):
                payload, cached = future.result()
                entries[futures[future]] = payload
                hits += int(cached)
                new_evaluations += int(not cached)
                progress('training', completedEpisodeJobs=len(entries), totalEpisodeJobs=len(jobs))
        return [aggregate([entries[cache.key(p, e)] for e in episodes]) for p in population]

    def publish(parameters, label):
        entries = [cache.read(cache.key(parameters, e)) for e in episodes]
        summary = aggregate(entries)
        # Replays are recovered from cache: no extra rollout to render a winner.
        for i, entry in enumerate(entries):
            recording = dict(entry['recording'])
            recording['objectiveSummary'] = {'return': summary['return'], 'episodeCount': 2, 'episodeIndex': i,
                                              'episodeReturns': [e['metrics']['return'] for e in entries]}
            atomic_json(directory / f'{label}{"-second" if i else ""}.json', recording)
        atomic_json(directory / f'{label}-metrics.json', summary)

    def save():
        state['rng'] = rng.bit_generator.state
        checkpoint(directory / 'latest.npz', fingerprint=fingerprint, best=np.asarray(state['best']),
                   generation=state['generation'], state=json.dumps(state, allow_nan=False))

    if not args.resume:
        save()
    progress('initializing')
    try:
        # Serial first call compiles the neural kernel before parallel episode jobs.
        if state['bestScore'] is None:
            payload, cached = cache.evaluate(initial, episodes[0], run)
            new_evaluations += int(not cached)
            hits += int(cached)
            result = evaluate([initial])[0]
            state['bestScore'] = result['return']
            save()
        publish(initial, 'baseline')
        publish(np.asarray(state['best']), 'best')
        policy.set_parameters(np.asarray(state['best']))
        atomic_json(directory / 'decoder.json', policy.export_decoder())
        atomic_json(directory / 'history.json', state['history'])
        atomic_json(RUNS / 'latest.json', {'name': name})
        while state['generation'] < args.iterations:
            stagnation = stalled_generations(state['history'])
            population, roles = next_population(np.asarray(state['best']), state['population'], state['scores'], rng, stagnation)
            results = evaluate(population)
            scores = np.array([r['return'] for r in results])
            winner = int(np.argmax(scores))
            if scores[winner] > state['bestScore']:
                state['best'] = population[winner].tolist()
                state['bestScore'] = float(scores[winner])
            state['generation'] += 1
            state['population'], state['scores'] = population.tolist(), scores.tolist()
            for r, role in zip(results, roles):
                r['origin'] = role
            state['history'].append({'generation': state['generation'], 'meanReturn': float(scores.mean()),
                                     'algorithm': ALGORITHM, 'stagnationBefore': stagnation,
                                     'mutationSettings': mutation_settings(stagnation),
                                     'uniqueCandidateCount': len({parameter_id(p) for p in population}),
                                     'generationBestReturn': float(scores.max()), 'bestReturn': state['bestScore'],
                                     'winner': winner, 'candidates': results,
                                     'bestSurvivalSeconds': results[winner]['survivalSeconds']})
            save()
            # These derived files can be restored from committed checkpoint/cache after interruption.
            publish(np.asarray(state['best']), 'best')
            policy.set_parameters(np.asarray(state['best']))
            atomic_json(directory / 'decoder.json', policy.export_decoder())
            atomic_json(directory / 'history.json', state['history'])
            progress('training')
            print(f"Generation {state['generation']}: two-state mean={state['bestScore']:.4f}; "
                  f"new episodes this session={new_evaluations}, cache hits={hits}", flush=True)
        # These are the objective's two states, not an independent generalization test.
        evaluation = []
        for label, p in [('untrained', initial), ('trained', np.asarray(state['best']))]:
            evaluation.extend({'policy': label, **cache.read(cache.key(p, e))['metrics']} for e in episodes)
        atomic_json(directory / 'evaluation.json', evaluation)
        progress('complete')
        return directory
    except BaseException as error:
        progress('interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        raise
