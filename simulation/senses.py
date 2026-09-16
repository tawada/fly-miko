"""Engineered analog receptors; these rates are not measured fly-neuron activity."""
import hashlib

import numpy as np

from .biped import BODY, CONTACT_PARTS

ENCODER_ID = "miko-proprio-contact-80-v1"


class SensoryEncoder:
    max_rate_hz = 200.0

    def __init__(self, body_weight_n, mode="legacy"):
        if mode not in ("legacy", "body-contact-gravity"):
            raise ValueError("Unknown sensory mode")
        self.full_body = mode == "body-contact-gravity"
        self.encoder_id = "miko-body-contact-gravity-126-v2" if self.full_body else ENCODER_ID
        if not np.isfinite(body_weight_n) or body_weight_n <= 0:
            raise ValueError("Body weight must be positive")
        self.body_weight_n = body_weight_n
        self.channels = []
        for joint in BODY["joints"]:
            side = "left" if joint["id"].startswith("left_") else (
                "right" if joint["id"].startswith("right_") else "both")
            for kind in ("position_positive", "position_negative", "velocity_positive", "velocity_negative"):
                self.channels.append({"id": f"{joint['id']}.{kind}", "side": side,
                                      "modality": "proprioception"})
        parts = CONTACT_PARTS if self.full_body else ["left_foot", "right_foot"]
        for part in parts:
            for kind in ("contact", "load", "touchdown", "liftoff"):
                self.channels.append({"id": f"{part}.{kind}", "side": part.split("_")[0], "modality": "touch"})
        if self.full_body:
            for axis in "xyz":
                for sign in ("positive", "negative"):
                    self.channels.append({"id": f"gravity.{axis}_{sign}", "side": "both",
                                          "modality": "gravity"})
        for channel in self.channels:
            channel["encoderId"] = self.encoder_id
        self.reset()

    def reset(self):
        self.previous_contacts = None
        self.last_time = None

    def encode(self, sample):
        if sample.get("source") != "mujoco-measured":
            raise ValueError("Encoder requires measured physics feedback")
        time = sample.get("timeSeconds")
        if not isinstance(time, (float, int)) or not np.isfinite(time):
            raise ValueError("Invalid sensory timestamp")
        if self.last_time is not None and time <= self.last_time:
            raise ValueError("Sensory timestamps must advance; reset encoder between episodes")
        # Validate every field first so failed frames do not change event history.
        fields = {}
        for name, size in (("jointAngles", 18), ("jointVelocities", 18),
                           ("footContacts", 2), ("footForcesN", 2)):
            values = np.asarray(sample.get(name), dtype=float)
            if values.shape != (size,) or not np.all(np.isfinite(values)):
                raise ValueError(f"Invalid {name}")
            fields[name] = values
        if self.full_body:
            if sample.get("bodyContactIds") != CONTACT_PARTS:
                raise ValueError("Body contact order mismatch")
            for key in ("bodyContacts", "bodyContactForcesN"):
                fields[key] = np.asarray(sample.get(key), dtype=float)
                if fields[key].shape != (12,) or not np.all(np.isfinite(fields[key])):
                    raise ValueError("Invalid body contact measurements")
            gravity = np.asarray(sample.get("projectedGravity"), dtype=float)
            if gravity.shape != (3,) or not np.all(np.isfinite(gravity)) or not np.isclose(np.linalg.norm(gravity), 1., atol=1e-5):
                raise ValueError("Gravity must be a measured body-frame unit vector")
        contacts = fields["bodyContacts" if self.full_body else "footContacts"]
        forces = fields["bodyContactForcesN" if self.full_body else "footForcesN"]
        if np.any((contacts != 0) & (contacts != 1)):
            raise ValueError("Contacts must be binary")
        if np.any(forces < 0):
            raise ValueError("Negative contact force")
        if np.any(contacts != (forces > 0.5)):
            raise ValueError("Contact flags disagree with measured normal force")
        features = []
        for index, joint in enumerate(BODY["joints"]):
            delta = fields["jointAngles"][index] - joint["neutral"]
            positive = max(joint["max"] - joint["neutral"], 1e-9)
            negative = max(joint["neutral"] - joint["min"], 1e-9)
            speed = fields["jointVelocities"][index] / joint["maxSpeed"]
            features.extend((delta / positive, -delta / negative, speed, -speed))
        previous = contacts if self.previous_contacts is None else self.previous_contacts
        for side in range(len(contacts)):
            features.extend((
                contacts[side], forces[side] / self.body_weight_n,
                float(contacts[side] == 1 and previous[side] == 0),
                float(contacts[side] == 0 and previous[side] == 1),
            ))
        if self.full_body:
            for component in gravity:
                features.extend((component, -component))
        values = np.clip(features, 0, 1)
        self.previous_contacts = contacts.copy()
        self.last_time = time
        return {
            "encoderId": self.encoder_id, "timeSeconds": float(time),
            "provenance": "engineered-receptors-not-biological-firing",
            "values": values.tolist(),
            "ratesHz": (values * self.max_rate_hz).tolist(),
        }


class AscendingProjector:
    """Experimental VNC substitute: reproducible sparse routing to ascending IDs.

    Output is the rate of EXTERNAL stimulation, not the neuron's output firing rate.
    Anatomy/side labels restrict a candidate pool but do not establish sensory function.
    """
    def __init__(self, binding, channels):
        if binding.get("dataset") != "flywire-783" or binding.get("role") != "ascending-input-candidates":
            raise ValueError("Wrong sensory input binding")
        self.binding = binding
        self.encoder_id = channels[0].get("encoderId", ENCODER_ID)
        if any(c.get("encoderId", ENCODER_ID) != self.encoder_id for c in channels):
            raise ValueError("Mixed encoder versions")
        self.channel_ids = [channel["id"] for channel in channels]
        if len(set(self.channel_ids)) != len(channels):
            raise ValueError("Duplicate sensory channel")
        ids = [neuron["id"] for neuron in binding["neurons"]]
        if not ids or len(set(ids)) != len(ids) or any(not isinstance(i, str) or not i.isdigit() for i in ids):
            raise ValueError("Invalid ascending IDs")
        self.routes = []
        for channel in channels:
            candidates = [n for n in binding["neurons"]
                          if channel["side"] == "both" or n["side"] == channel["side"]]
            if len(candidates) < 4:
                raise ValueError(f"Insufficient input candidates for {channel['id']}")
            candidates.sort(key=lambda n: hashlib.sha256(
                f"{self.encoder_id}:{channel['id']}:{n['id']}".encode()).digest())
            self.routes.append([{"neuronId": n["id"], "weight": 0.25} for n in candidates[:4]])

    def project(self, signal):
        if signal.get("encoderId") != self.encoder_id:
            raise ValueError("Wrong encoder version")
        rates = np.asarray(signal.get("ratesHz"), dtype=float)
        if rates.shape != (len(self.routes),) or not np.all(np.isfinite(rates)) or np.any(rates < 0):
            raise ValueError("Invalid receptor rates")
        drive = {}
        for rate, targets in zip(rates, self.routes):
            for target in targets:
                identifier = target["neuronId"]
                drive[identifier] = min(200., drive.get(identifier, 0.) + float(rate) * target["weight"])
        return {
            "bindingId": self.binding["id"], "dataset": self.binding["dataset"],
            "encoderId": self.encoder_id, "timeSeconds": signal["timeSeconds"],
            "provenance": "experimental-external-stimulation-not-neuron-output",
            "encoding": "sparse-zero", "externalRateHz": {i: r for i, r in drive.items() if r > 0},
        }

    def manifest(self):
        return {
            "bindingId": self.binding["id"], "encoderId": self.encoder_id,
            "provenance": "engineering-routing-not-anatomical-correspondence",
            "routes": dict(zip(self.channel_ids, self.routes)),
        }
