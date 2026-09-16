"""CPU spiking network on the actual signed FlyWire graph; all neurons are simulated."""
import math
import time

import numpy as np
from numba import njit

PARAMETERS = {
    "dtMs": 0.1, "restMv": -52., "resetMv": -52., "thresholdMv": -45.,
    "membraneTauMs": 20., "synapticTauMs": 5., "refractoryMs": 2.2,
    "delayMs": 1.8, "externalJumpMv": 0.275 * 250,
}


@njit(cache=True, nogil=True)
def advance_lif(v, g, refractory, pending, cursor, indptr, posts, weights,
                input_indices, input_mask, probabilities, random_values,
                em, es, coupling, refractory_ticks, delay_ticks, jump_mv,
                active_indices, active_mask, active_count):
    """Exact linear subthreshold decay; delayed impulses on a fixed-step ring buffer."""
    count = np.zeros(v.size, dtype=np.int32)
    total_spikes, external_events = 0, 0
    slots = pending.shape[0]
    for step in range(random_values.shape[0]):
        for i in range(input_indices.size):
            if random_values[step, i] < probabilities[i]:
                target = input_indices[i]
                v[target] += jump_mv
                if not active_mask[target]:
                    active_mask[target] = True
                    active_indices[active_count] = target
                    active_count += 1
                external_events += 1
        future = (cursor + delay_ticks) % slots
        # Untouched neurons are exactly at rest (v=-52, g=0); skipping their zero
        # derivatives is exact, not a subgraph approximation. Every newly reached
        # postsynaptic neuron joins this list before its delayed input can arrive.
        active_limit = active_count
        for active in range(active_limit):
            i = active_indices[active]
            g[i] += pending[cursor, i]
            pending[cursor, i] = 0.
            if refractory[i] > 0:
                refractory[i] -= 1
                continue
            old_g = g[i]
            v[i] = -52. + (v[i] + 52.) * em + old_g * coupling
            g[i] = old_g * es
            if v[i] > -45.:
                count[i] += 1
                total_spikes += 1
                v[i] = -52.
                g[i] = 0.
                refractory[i] = 0 if input_mask[i] else max(0, refractory_ticks - 1)
                for edge in range(indptr[i], indptr[i + 1]):
                    target = posts[edge]
                    pending[future, target] += weights[edge]
                    if not active_mask[target]:
                        active_mask[target] = True
                        active_indices[active_count] = target
                        active_count += 1
        cursor = (cursor + 1) % slots
    return cursor, count, total_spikes, external_events, active_count


class FlyBrain:
    def __init__(self, graph, seed=0, dt_ms=0.1):
        if not np.isfinite(dt_ms) or dt_ms <= 0 or dt_ms > 0.2:
            raise ValueError("Neural dt must be in (0, 0.2] ms")
        for duration in (PARAMETERS["delayMs"], PARAMETERS["refractoryMs"], 20.):
            if not math.isclose(duration / dt_ms, round(duration / dt_ms), abs_tol=1e-8):
                raise ValueError("Neural dt must divide delay, refractory period and 20 ms")
        self.graph = graph
        self.dt_ms = dt_ms
        self.delay_ticks = round(PARAMETERS["delayMs"] / dt_ms)
        self.refractory_ticks = round(PARAMETERS["refractoryMs"] / dt_ms)
        self.input_lookup = {n["id"]: i for i, n in enumerate(graph.input_binding["neurons"])}
        self.input_mask = np.zeros(len(graph.ids), dtype=np.bool_)
        self.input_mask[graph.input_indices] = True
        self.em = math.exp(-dt_ms / PARAMETERS["membraneTauMs"])
        self.es = math.exp(-dt_ms / PARAMETERS["synapticTauMs"])
        self.coupling = PARAMETERS["synapticTauMs"] / (
            PARAMETERS["membraneTauMs"] - PARAMETERS["synapticTauMs"]) * (self.em - self.es)
        self.reset(seed)

    def reset(self, seed=0):
        n = len(self.graph.ids)
        self.v = np.full(n, -52., dtype=np.float32)
        self.g = np.zeros(n, dtype=np.float32)
        self.refractory = np.zeros(n, dtype=np.int32)
        self.pending = np.zeros((self.delay_ticks + 1, n), dtype=np.float32)
        self.cursor = 0
        self.rng = np.random.default_rng(seed)
        self.time_seconds = 0.
        self.spike_total = 0
        self.active_indices = np.zeros(n, dtype=np.int32)
        self.active_mask = np.zeros(n, dtype=np.bool_)
        self.active_count = 0

    def advance(self, stimulus, seconds=0.02):
        if stimulus.get("bindingId") != self.graph.input_binding["id"] or stimulus.get("dataset") != "flywire-783":
            raise ValueError("Brain input binding/dataset mismatch")
        if stimulus.get("encoding") != "sparse-zero":
            raise ValueError("Declare sparse-zero external stimulation")
        timestamp = stimulus.get("timeSeconds")
        if not isinstance(timestamp, (int, float)) or not math.isclose(timestamp, self.time_seconds, abs_tol=1e-7):
            raise ValueError("Brain/body input timestamps must agree")
        if not np.isfinite(seconds) or seconds <= 0 or seconds > 0.1:
            raise ValueError("Invalid brain duration")
        steps = round(seconds * 1000 / self.dt_ms)
        if not math.isclose(steps * self.dt_ms, seconds * 1000, abs_tol=1e-8):
            raise ValueError("Duration must contain a whole number of neural steps")
        rates = np.zeros(len(self.graph.input_indices), dtype=np.float32)
        external = stimulus.get("externalRateHz")
        if not isinstance(external, dict):
            raise ValueError("externalRateHz must be a neuron-ID mapping")
        for identifier, rate in external.items():
            if identifier not in self.input_lookup:
                raise ValueError(f"Unknown brain input neuron: {identifier}")
            if not isinstance(rate, (int, float)) or not np.isfinite(rate) or not 0 <= rate <= 200:
                raise ValueError("External rates must be in [0,200] Hz")
            rates[self.input_lookup[identifier]] = rate
        started = time.perf_counter()
        random_values = self.rng.random((steps, len(rates)), dtype=np.float32)
        self.cursor, counts, total, external_count, self.active_count = advance_lif(
            self.v, self.g, self.refractory, self.pending, self.cursor,
            self.graph.indptr, self.graph.posts, self.graph.weights,
            self.graph.input_indices, self.input_mask, rates * (self.dt_ms / 1000),
            random_values, self.em, self.es, self.coupling,
            self.refractory_ticks, self.delay_ticks, PARAMETERS["externalJumpMv"],
            self.active_indices, self.active_mask, self.active_count,
        )
        self.time_seconds += seconds
        self.spike_total += int(total)
        output_counts = counts[self.graph.output_indices]
        rates_hz = output_counts.astype(np.float64) / seconds
        return {
            "bindingId": self.graph.output_binding["id"], "dataset": "flywire-783",
            "timeSeconds": self.time_seconds, "encoding": "dense",
            "ratesHz": rates_hz,
            "telemetry": {
                "neuralTimeSeconds": self.time_seconds, "spikeCount": int(total),
                "outputSpikeCount": int(output_counts.sum()), "externalEvents": int(external_count),
                "activeOutputCount": int(np.count_nonzero(output_counts)),
                "outputCounts": [[int(i), int(output_counts[i])] for i in np.flatnonzero(output_counts)],
                "inputActiveCount": int(np.count_nonzero(rates)),
                "visitedNeuronCount": int(self.active_count),
                "wallSeconds": time.perf_counter() - started,
            },
        }
