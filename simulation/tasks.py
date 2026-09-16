"""Versioned task definitions; height is measured at the physical head centre."""
import numpy as np

from .biped import BODY

TASKS = {
    "walk": {
        "id": "walk-v1", "initialPose": "standing",
        "objective": "survival-upright-speed", "targetSpeedMps": 0.25,
    },
    "head-height": {
        "id": "neutral-head-height-v4-bias", "initialPose": "standing",
        "referencePose": "neutral", "referenceJointOverrides": {"left_elbow": 0., "right_elbow": 0.},
        "objective": "integral-head-centre-height", "targetSpeedMps": 0.,
        "rewardUnits": "metre-seconds", "lowPoseTermination": False,
        "sensoryMode": "body-contact-gravity", "featureGain": 20.,
        "trainBias": True, "initialWeightSeed": 735,
    },
}


def initial_parameters(feature_count, task):
    if task == "head-height":
        return np.concatenate([np.random.default_rng(TASKS[task]["initialWeightSeed"]).normal(0., .15, 18 * feature_count), np.zeros(18)])
    return np.zeros(18 * (feature_count + 1))


def make_policy(binding, feature_count, parameters, task):
    from .policy import MotorReadout
    options = {}
    if task == "head-height":
        pose = {j["id"]: j["neutral"] for j in BODY["joints"]}
        pose.update(TASKS[task].get("referenceJointOverrides", {}))
        options = {"reference_angles": [pose[j["id"]] for j in BODY["joints"]],
                   "feature_gain": TASKS[task]["featureGain"], "train_bias": TASKS[task]["trainBias"]}
    return MotorReadout(binding, feature_count, parameters, **options)
