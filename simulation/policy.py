"""Trainable low-rank output readout, exportable to the existing JS bridge format."""
import hashlib
import json

import numpy as np

from .biped import BODY


class MotorReadout:
    def __init__(self, binding, feature_count=32, parameters=None, *,
                 reference_angles=None, feature_gain=1., train_bias=True):
        if not isinstance(feature_count, int) or feature_count < 1 or feature_count > 128:
            raise ValueError("Feature count must be in [1,128]")
        self.binding = binding
        self.feature_count = feature_count
        self.feature_gain = float(feature_gain)
        if not np.isfinite(self.feature_gain) or self.feature_gain <= 0:
            raise ValueError("Invalid feature gain")
        self.train_bias = bool(train_bias)
        self.reference_angles = np.array(
            [j["neutral"] for j in BODY["joints"]] if reference_angles is None else reference_angles,
            dtype=float)
        if self.reference_angles.shape != (18,) or any(
                not j["min"] <= angle <= j["max"] for angle, j in zip(self.reference_angles, BODY["joints"])):
            raise ValueError("Invalid reference pose")
        self.neuron_ids = [n["id"] for n in binding["neurons"]]
        self.groups = np.array([
            int(hashlib.sha256(f"readout-v1:{identifier}".encode()).hexdigest()[:8], 16) % feature_count
            for identifier in self.neuron_ids
        ])
        self.scales = np.sqrt(np.maximum(1, np.bincount(self.groups, minlength=feature_count)))
        self.parameter_count = 18 * (feature_count + int(self.train_bias))
        self.set_parameters(np.zeros(self.parameter_count) if parameters is None else parameters)
        self.reset()

    def set_parameters(self, parameters):
        parameters = np.asarray(parameters, dtype=np.float64)
        if parameters.shape != (self.parameter_count,) or not np.all(np.isfinite(parameters)):
            raise ValueError("Policy parameter shape/value mismatch")
        self.parameters = parameters.copy()
        self.weights = self.parameters[:18 * self.feature_count].reshape(18, self.feature_count)
        self.bias = self.parameters[18 * self.feature_count:] if self.train_bias else np.zeros(18)

    def reset(self, angles=None):
        self.filtered = np.zeros(len(self.neuron_ids))
        self.angles = np.array(self.reference_angles if angles is None else angles,
                               dtype=float)
        if self.angles.shape != (18,) or not np.all(np.isfinite(self.angles)):
            raise ValueError("Invalid initial readout angles")
        if any(not j["min"] <= angle <= j["max"] for angle, j in zip(self.angles, BODY["joints"])):
            raise ValueError("Initial readout angles outside limits")

    def step(self, brain_frame, dt=0.02):
        if brain_frame.get("bindingId") != self.binding["id"] or brain_frame.get("dataset") != "flywire-783":
            raise ValueError("Readout brain binding mismatch")
        rates = np.asarray(brain_frame.get("ratesHz"), dtype=np.float64)
        if rates.shape != self.filtered.shape or not np.all(np.isfinite(rates)) or np.any(rates < 0):
            raise ValueError("Invalid neural output rates")
        if not np.isfinite(dt) or dt <= 0 or dt > 0.1:
            raise ValueError("Invalid readout time step")
        filter_config = BODY["filter"]
        alpha = -np.expm1(-dt / filter_config["rateTimeConstantSeconds"])
        self.filtered += alpha * (np.minimum(rates, filter_config["maxRateHz"]) - self.filtered)
        normalized = self.filtered / filter_config["rateScaleHz"]
        features = self.feature_gain * np.bincount(self.groups, weights=normalized, minlength=self.feature_count) / self.scales
        actions = np.tanh(self.weights @ features + self.bias)
        neutral = self.reference_angles
        low = np.array([j["min"] for j in BODY["joints"]])
        high = np.array([j["max"] for j in BODY["joints"]])
        target = neutral + actions * np.where(actions >= 0, high - neutral, neutral - low)
        maximum_delta = np.array([j["maxSpeed"] for j in BODY["joints"]]) * dt
        self.angles += np.clip(target - self.angles, -maximum_delta, maximum_delta)
        self.angles = np.clip(self.angles, low, high)
        return {
            "jointAngles": dict(zip((j["id"] for j in BODY["joints"]), self.angles.tolist())),
            "angles": self.angles.tolist(), "actions": actions.tolist(), "features": features.tolist(),
        }

    def export_decoder(self):
        dense = self.feature_gain * self.weights[:, self.groups] / self.scales[self.groups]
        return {
            "bindingId": self.binding["id"], "bodyMapId": BODY["id"],
            "provenance": "trainable-connectome-readout",
            "neuronIds": self.neuron_ids, "jointIds": [j["id"] for j in BODY["joints"]],
            "weights": dense.tolist(), "bias": self.bias.tolist(),
            "referenceAngles": self.reference_angles.tolist(),
            "parameterization": {"kind": "fixed-hash-pooling-v2", "featureCount": self.feature_count,
                                 "featureGain": self.feature_gain, "trainBias": self.train_bias},
        }

    def fingerprint(self):
        content = json.dumps(self.export_decoder(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(content.encode()).hexdigest()
