#!/usr/bin/env python3
"""Download pinned publisher files, compile the graph, and restore the initial readout."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


@contextmanager
def atomic_file(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, filename = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    temporary = Path(filename)
    try:
        with os.fdopen(fd, 'wb') as stream:
            yield stream, temporary
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def fetch(source, root, download=False):
    path = root / source['path']
    if path.exists():
        if file_hash(path) != source['sha256']:
            raise ValueError(f'Checksum mismatch; existing file was kept: {path}')
        print(f'Verified local source: {source["path"]}', flush=True)
        return
    if not download:
        raise FileNotFoundError(f'{path} missing. Use --download after reviewing docs/data-preparation.md.')
    print(f'Downloading publisher file: {source["url"]}', flush=True)
    with atomic_file(path) as (output, temporary):
        request = urllib.request.Request(source['url'], headers={'User-Agent': 'fly-miko-reproduction/1'})
        with urllib.request.urlopen(request, timeout=120) as response:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        output.flush()
        if file_hash(temporary) != source['sha256']:
            raise ValueError(f'Download checksum mismatch: {source["path"]}')


def verify_compiled(root, spec):
    directory = root / 'data/brain/compiled'
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['id'] != spec['graphId'] or manifest['compiledChecksums'] != spec['compiledChecksums']:
        raise ValueError('Compiled graph/bindings differ from the reproduction specification')
    for filename, checksum in spec['compiledChecksums'].items():
        if file_hash(directory / filename) != checksum:
            raise ValueError(f'Compiled checksum mismatch: {filename}')


def prepare_initial(root, spec):
    import numpy as np
    path = root / spec['initialFile']
    if file_hash(path) != spec['initialFileSha256']:
        raise ValueError('Initial readout JSON checksum mismatch')
    initial = json.loads(path.read_text())
    parameters = np.asarray(initial['parameters'], dtype='<f8')
    if parameters.shape != (594,) or not np.isfinite(parameters).all():
        raise ValueError('Invalid initial readout dimensions/values')
    if hashlib.sha256(parameters.tobytes()).hexdigest() != spec['initialParameterSha256']:
        raise ValueError('Initial numerical parameters differ from the published run')
    destination = root / 'data/reproduction/random-head-bias-02'
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint = destination / 'initial.npz'
    if checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as saved:
            if not np.array_equal(saved['best'], parameters):
                raise ValueError('Existing initialization differs; file was kept')
    else:
        with atomic_file(checkpoint) as (output, _):
            np.savez_compressed(output, best=parameters)
    config = {'featureCount': initial['featureCount'], 'task': {'id': initial['taskId']},
              'parameterSha256': spec['initialParameterSha256'], 'provenance': 'config/reproduction-initial.json'}
    config_path = destination / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Existing initial readout metadata differs; file was kept')
    with atomic_file(config_path) as (output, _):
        output.write(json.dumps(config, indent=2).encode())
    return checkpoint


def prepare(root=ROOT, download=False, verify_only=False):
    spec = json.loads((root / 'config/reproduction.json').read_text())
    for filename, checksum in spec['engineCodeChecksums'].items():
        if file_hash(root / 'simulation' / filename) != checksum:
            raise ValueError(f'Reproduction engine differs: simulation/{filename}')
    body = json.loads((root / 'config/body-map.json').read_text())
    if hashlib.sha256(json.dumps(body, sort_keys=True, allow_nan=False).encode()).hexdigest() != spec['bodyHash']:
        raise ValueError('Reproduction physical body differs')
    if verify_only:
        verify_compiled(root, spec)
    else:
        for source in spec['sources']:
            fetch(source, root, download)
        # Refuse to rebuild over an existing graph; preparation never modifies training data in use.
        compiled = root / 'data/brain/compiled/manifest.json'
        if compiled.exists():
            verify_compiled(root, spec)
        else:
            subprocess.run([sys.executable, str(root / 'scripts/prepare_brain_binding.py')], cwd=root, check=True)
            subprocess.run([sys.executable, '-m', 'simulation.connectome'], cwd=root, check=True)
            verify_compiled(root, spec)
    checkpoint = prepare_initial(root, spec)
    print('Ready: 138639 neurons, 15091983 connections, 1736 inputs, 1280 outputs.', flush=True)
    print(f'Initial readout: {checkpoint}', flush=True)
    print('No training run/checkpoint was created or modified.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--download', action='store_true', help='Fetch pinned files from their public publishers')
    mode.add_argument('--verify-only', action='store_true', help='Verify existing compiled data and prepare only the initial readout')
    args = parser.parse_args()
    prepare(download=args.download, verify_only=args.verify_only)


if __name__ == '__main__':
    main()
