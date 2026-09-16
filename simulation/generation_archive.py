"""Self-contained motion archives for generations 1–5 and multiples of ten."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import threading
import time

import numpy as np

from .biped import ROOT
from .population_search import ALGORITHM, EvaluationCache, next_population, parameter_id
from .storage import atomic_output
from .train import atomic_json, checkpoint


def selected(generation):
    return 1 <= generation <= 5 or generation >= 10 and generation % 10 == 0


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def reconstruct(state, config):
    """Replay parameter selection only, never neural or physical simulation."""
    rng = np.random.default_rng(config['evaluationContext']['seed'])
    best = np.asarray(state['initial'], dtype=float)
    previous = scores = None
    # The initial candidate is slot 0 of generation 1, so its score is in history.
    best_score = None
    for record in state['history']:
        population, roles = next_population(best, previous, scores, rng)
        if [parameter_id(p) for p in population] != [r['parameterId'] for r in record['candidates']]:
            raise ValueError(f"Cannot reproduce parameters of generation {record['generation']}")
        scores = np.array([r['return'] for r in record['candidates']])
        if best_score is None:
            best_score = float(scores[0])
        winner = int(np.argmax(scores))
        if scores[winner] > best_score:
            best = population[winner].copy()
            best_score = float(scores[winner])
        yield record, population, best
        previous = population


def save_generation(directory, config, cache, record, population, best):
    generation = record['generation']
    destination = directory / 'generations' / f'{generation:04d}'
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['fingerprint'] != config['fingerprint'] or manifest['candidates'] != record['candidates']:
            raise ValueError(f'Existing archive belongs to a different generation: {destination}')
        for item in manifest['files']:
            path = destination / item['name']
            if not path.is_file() or path.stat().st_size != item['bytes']:
                raise ValueError(f'Incomplete archive: {path}')
        return False
    files = []
    for index, (parameters, candidate) in enumerate(zip(population, record['candidates']), 1):
        for episode_index, episode in enumerate(config['episodes'], 1):
            key = cache.key(parameters, episode)
            payload = cache.read(key)
            if payload is None:
                raise ValueError(f'Missing motion cache for generation {generation}, candidate {index}, state {episode_index}')
            if (payload['parameterId'] != candidate['parameterId'] or
                    candidate['episodes'][episode_index - 1]['cacheKey'] != key):
                raise ValueError('Motion cache does not match generation history')
            filename = f'candidate-{index:02d}-state-{episode_index}.json.gz'
            path = destination / filename
            # Copy, not symlink: these remain usable even if the evaluation cache is removed.
            with (cache.directory / f'{key}.json.gz').open('rb') as source, atomic_output(path) as output:
                shutil.copyfileobj(source, output)
            files.append({'name': filename, 'candidate': index, 'state': episode_index,
                          'cacheKey': key, 'frames': len(payload['recording']['frames']),
                          'bytes': path.stat().st_size, 'sha256': file_hash(path)})
    parameters_path = destination / 'parameters.npz'
    checkpoint(parameters_path, population=population, scores=np.array([r['return'] for r in record['candidates']]),
               best=best, generation=generation)
    files.append({'name': 'parameters.npz', 'bytes': parameters_path.stat().st_size,
                  'sha256': file_hash(parameters_path)})
    # The manifest is the completion marker; incomplete writes are retried.
    atomic_json(manifest_path, {'version': 1, 'generation': generation, 'fingerprint': config['fingerprint'],
                               'createdAt': datetime.now(timezone.utc).isoformat(), 'config': config,
                               'candidates': record['candidates'], 'files': files,
                               'recordingLocation': 'payload.recording', 'complete': True})
    print(f'Archived generation {generation}: 10 candidates × 2 initial states at {destination}', flush=True)
    return True


def archive_available(directory):
    directory = Path(directory)
    if not (directory / 'latest.npz').exists():
        return []
    with (directory / '.generation-archive.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return []
        config = json.loads((directory / 'config.json').read_text())
        if config['algorithm'] != ALGORITHM:
            raise ValueError('Generation archives require the retained-population scenario')
        with np.load(directory / 'latest.npz', allow_pickle=False) as saved:
            if str(saved['fingerprint']) != config['fingerprint']:
                raise ValueError('Checkpoint/config mismatch')
            state = json.loads(str(saved['state']))
        cache = EvaluationCache(directory / 'evaluation-cache', config['evaluationContext'])
        generations = []
        for record, population, best in reconstruct(state, config):
            if selected(record['generation']):
                save_generation(directory, config, cache, record, population, best)
                generations.append(record['generation'])
        atomic_json(directory / 'archive-status.json', {
            'status': 'watching', 'policy': 'generations 1-5 and 10,20,30,...',
            'checkpointGeneration': state['generation'], 'archivedGenerations': generations,
            'updatedAt': datetime.now(timezone.utc).isoformat()})
        return generations


def watch(directory, stop, exit_when_complete=False):
    directory = Path(directory)
    while not stop.is_set():
        try:
            archive_available(directory)
            progress = directory / 'progress.json'
            if exit_when_complete and progress.exists() and json.loads(progress.read_text())['status'] == 'complete':
                # Status may have changed after reading the checkpoint; take a final snapshot.
                archive_available(directory)
                return
        except Exception as error:
            print(f'Generation archive retry: {error}', flush=True)
            if directory.exists():
                atomic_json(directory / 'archive-status.json', {'status': 'retrying', 'error': str(error)})
        stop.wait(5)


@contextmanager
def background_archive(directory):
    stop = threading.Event()
    worker = threading.Thread(target=watch, args=(directory, stop), daemon=True)
    worker.start()
    try:
        yield
    finally:
        stop.set()
        worker.join()
        archive_available(directory)


def export_recording(directory, generation, candidate, state, output):
    folder = Path(directory) / 'generations' / f'{generation:04d}'
    manifest = json.loads((folder / 'manifest.json').read_text())
    filename = f'candidate-{candidate:02d}-state-{state}.json.gz'
    entry = next((item for item in manifest['files'] if item['name'] == filename), None)
    path = folder / filename
    if entry is None or file_hash(path) != entry['sha256']:
        raise ValueError('Archive motion checksum mismatch')
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        packet = json.load(stream)
    from .population_search import digest
    if packet['sha256'] != digest(packet['payload']):
        raise ValueError('Archive payload checksum mismatch')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, packet['payload']['recording'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--generation', type=int)
    parser.add_argument('--candidate', type=int, choices=range(1, 11), default=1)
    parser.add_argument('--state', type=int, choices=[1, 2], default=1)
    parser.add_argument('--export', type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', args.name):
        parser.error('Invalid run name')
    directory = ROOT / 'runs/training' / args.name
    if args.export:
        if args.generation is None:
            parser.error('--export requires --generation')
        export_recording(directory, args.generation, args.candidate, args.state, args.export)
    elif args.watch:
        watch(directory, threading.Event(), exit_when_complete=True)
    else:
        archive_available(directory)


if __name__ == '__main__':
    main()
