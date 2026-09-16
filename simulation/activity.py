"""Record whole-brain spike bins from a saved sensory replay, without retraining.

The observation wrapper reads the existing kernel's returned counts. No neural,
physical or policy code is changed, preserving checkpoint compatibility.
"""
import argparse
import hashlib
import json
import re
from unittest.mock import patch

import numpy as np

from . import brain as brain_module
from .biped import ROOT
from .connectome import Connectome
from .senses import AscendingProjector
from .train import atomic_json


def record_activity(recording, graph):
    if recording['brainModel']['compiledChecksums'] != graph.manifest['compiledChecksums']:
        raise ValueError('Recording graph differs from installed graph')
    projector = AscendingProjector(graph.input_binding, recording['channels'])
    if projector.manifest() != recording['inputRouting']:
        raise ValueError('Recording sensory routing differs from current routing')
    brain = brain_module.FlyBrain(graph, recording['metrics']['seed'],
                                 dt_ms=recording['neuralParameters']['dtMs'])
    bins = [{'timeSeconds': 0., 'counts': []}]
    kernel = brain_module.advance_lif
    observed = None

    def observe(*args):
        nonlocal observed
        result = kernel(*args)
        observed = result[1]
        return result

    # This command is deliberately single-threaded; the scoped hook is restored
    # even on failure and never runs inside the training process.
    with patch.object(brain_module, 'advance_lif', observe):
        for i, (previous, frame) in enumerate(zip(recording['frames'], recording['frames'][1:]), 1):
            result = brain.advance(projector.project(previous['sensory']), recording['controlDtSeconds'])
            expected = frame['brain']
            for field in ('spikeCount', 'outputSpikeCount', 'outputCounts', 'externalEvents'):
                if result['telemetry'][field] != expected[field]:
                    raise ValueError(f'Replay diverged at frame {i}: {field}; no activity file published')
            bins.append({'timeSeconds': frame['timeSeconds'],
                         'counts': [[int(j), int(observed[j])] for j in np.flatnonzero(observed)]})
            if i % 50 == 0:
                print(f"Verified {frame['timeSeconds']:.1f}/{recording['durationSeconds']:.1f}s", flush=True)
    roles = np.zeros(len(graph.ids), dtype=np.uint8)
    roles[graph.input_indices] |= 1
    roles[graph.output_indices] |= 2
    return {'version': 1, 'binSeconds': recording['controlDtSeconds'],
            'layout': 'graph-index-not-anatomical', 'neuronIds': list(graph.ids),
            'roles': roles.tolist(), 'frames': bins,
            'policyFingerprint': recording['policyFingerprint'], 'seed': recording['metrics']['seed'],
            'verifiedAgainstRecording': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--scenario', choices=['best', 'best-second', 'baseline', 'baseline-second', 'evaluation-replay', 'ablation'], default='best')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', args.name):
        parser.error('Invalid run name')
    path = ROOT / 'runs/training' / args.name / f'{args.scenario}.json'
    source = path.read_bytes()
    activity = record_activity(json.loads(source), Connectome())
    activity['sourceSha256'] = hashlib.sha256(source).hexdigest()
    if path.read_bytes() != source:
        raise ValueError('Source recording changed during capture; rerun the command')
    output = path.with_name(f'{args.scenario}-activity.json')
    atomic_json(output, activity)
    print(f'Saved {output}', flush=True)


if __name__ == '__main__':
    main()
