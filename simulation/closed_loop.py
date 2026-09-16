"""Causal sampled loop: measured body → fly spikes → next-period motor command."""
import math
import time

import numpy as np

from .biped import Biped, BODY, CRAWL_NECK_PITCH
from .brain import FlyBrain, PARAMETERS
from .policy import MotorReadout
from .senses import SensoryEncoder, AscendingProjector
from .tasks import TASKS, initial_parameters, make_policy


def reward(sample, robot, previous_commands, target_speed=0.25, task="walk"):
    if task == "head-height":
        # No speed target, posture bonus, fall penalty, or early termination.
        # Sustaining a high head earns more than a brief peak.
        height = float(sample["headHeightM"])
        return robot.control_dt * height, {"headHeight": height}, False
    upright = float(-sample["projectedGravity"][1])
    height = float(sample["rootPositionM"][1])
    speed = float(sample["linearVelocity"][2])
    fallen = height < 0.60 or upright < 0.50
    current_commands = np.asarray(sample["commandAngles"])
    components = {
        "survival": 1.,
        "upright": max(0., upright),
        "speedTracking": math.exp(-((speed - target_speed) / 0.4) ** 2),
        "effort": -0.00003 * float(np.square(robot.data.ctrl).sum()),
        "jointSpeed": -0.002 * float(np.square(sample["jointVelocities"]).sum()),
        "commandChange": -0.1 * float(np.square(current_commands - previous_commands).sum()),
    }
    value = robot.control_dt * sum(components.values()) - (0.5 if fallen else 0.)
    return value, components, fallen


def rollout(graph, parameters, feature_count=32, seed=0, seconds=3.,
            record=False, ablate_brain=False, neural_dt_ms=0.1, task="walk", initial_state_seed=None):
    if not np.isfinite(seconds) or seconds <= 0 or seconds > 120:
        raise ValueError("Episode seconds must be in (0,120]")
    started = time.perf_counter()
    task_config = TASKS[task]
    robot = Biped(supported=False, initial_pose=task_config["initialPose"])
    initial_state = None
    if initial_state_seed is not None:
        from .random_initial import randomize
        initial_state = randomize(robot, initial_state_seed)
        task_config = {**task_config, "initialPose": "random", "initialStateDistribution": initial_state["version"]}
    encoder = SensoryEncoder(robot.mass * 9.81, mode=task_config.get("sensoryMode", "legacy"))
    projector = AscendingProjector(graph.input_binding, encoder.channels)
    brain = FlyBrain(graph, seed, dt_ms=neural_dt_ms)
    policy = make_policy(graph.output_binding, feature_count, parameters, task)
    sample = robot.snapshot()
    policy.reset(sample["jointAngles"])
    sample["targetVelocity"] = [0., 0., task_config["targetSpeedMps"]]
    signal = encoder.encode(sample)
    command = dict(zip((j["id"] for j in BODY["joints"]), sample["jointAngles"]))
    previous_commands = np.array(list(command.values()))
    frames = []
    sample["sensory"] = signal
    if record:
        frames.append(sample)
    reward_sum = 0.
    output_spikes = 0
    spike_sum = 0
    action_energy = 0.
    neural_action_sum = 0.
    neural_action_max = 0.
    action_samples = 0
    reason = "time_limit"
    start_position = sample["rootPositionM"][2]
    initial_head_height = sample["headHeightM"]
    peak_head_height = initial_head_height
    head_height_integral = 0.
    for _ in range(max(1, round(seconds / robot.control_dt))):
        stimulus = projector.project(signal)
        brain_frame = brain.advance(stimulus, robot.control_dt)
        # Ablation still runs the full neural model but removes its outputs at the readout.
        if ablate_brain:
            brain_frame["ratesHz"] = np.zeros_like(brain_frame["ratesHz"])
        next_command = policy.step(brain_frame, robot.control_dt)
        neural_delta = np.abs(np.asarray(next_command["actions"]) - np.tanh(policy.bias))
        neural_action_sum += float(neural_delta.sum())
        neural_action_max = max(neural_action_max, float(neural_delta.max()))
        action_samples += len(neural_delta)
        sample = robot.advance(command)
        sample["targetVelocity"] = [0., 0., task_config["targetSpeedMps"]]
        # u(t+dt), computed during this interval, is applied in the NEXT physical interval.
        command = next_command["jointAngles"]
        value, components, fallen = reward(sample, robot, previous_commands, task=task)
        peak_head_height = max(peak_head_height, sample["headHeightM"])
        head_height_integral += robot.control_dt * sample["headHeightM"]
        previous_commands = np.asarray(sample["commandAngles"])
        signal = encoder.encode(sample)
        reward_sum += value
        spike_sum += brain_frame["telemetry"]["spikeCount"]
        output_spikes += brain_frame["telemetry"]["outputSpikeCount"]
        action_energy += float(np.square(next_command["actions"]).sum())
        if record:
            sample.update({
                "sensory": signal, "brain": brain_frame["telemetry"],
                "nextCommandAngles": next_command["angles"],
                "decoderActions": next_command["actions"], "brainFeatures": next_command["features"],
                "reward": value, "returnSoFar": reward_sum, "rewardComponents": components,
            })
            frames.append(sample)
        if fallen:
            reason = "fall"
            break
    metrics = {
        "return": reward_sum, "survivalSeconds": sample["timeSeconds"],
        "forwardDistanceM": sample["rootPositionM"][2] - start_position,
        "finalHeightM": sample["rootPositionM"][1], "termination": reason,
        "initialHeadHeightM": initial_head_height, "peakHeadHeightM": peak_head_height,
        "finalHeadHeightM": sample["headHeightM"],
        "meanHeadHeightM": head_height_integral / sample["timeSeconds"],
        "spikeCount": spike_sum, "outputSpikeCount": output_spikes,
        "actionEnergy": action_energy, "seed": seed, "ablateBrain": ablate_brain,
        "meanAbsNeuralAction": neural_action_sum / action_samples,
        "maxAbsNeuralAction": neural_action_max,
        "wallSeconds": time.perf_counter() - started,
    }
    recording = None
    if record:
        recording = {
            "version": 1, "bodyMapId": BODY["id"], "physics": "MuJoCo",
            "controlDtSeconds": robot.control_dt, "physicsDtSeconds": robot.physics_dt,
            "durationSeconds": sample["timeSeconds"], "mode": "connectome-closed-loop",
            "supported": False,
            "learned": bool(np.any(np.asarray(parameters) != initial_parameters(feature_count, task))),
            "brainConnected": True, "task": task_config,
            "neuralDriveEnabled": bool(np.any(policy.weights)) and not ablate_brain,
            "rigPoseOffsets": {"首": [CRAWL_NECK_PITCH, 0., 0.]} if robot.initial_pose == "splayed-crawl" else {},
            "readout": policy.export_decoder()["parameterization"],
            "ablateBrain": ablate_brain, "bodyMassKg": robot.mass, "units": "metres-radians-seconds",
            "rootFrame": "three-right-handed-y-up", "jointIds": [j["id"] for j in BODY["joints"]],
            "pmxScale": 0.08, "pmxRootReferenceBone": "下半身",
            "geometrySpecs": robot.geometry_specs(), "channels": encoder.channels,
            "brainModel": graph.manifest, "neuralParameters": {**PARAMETERS, "dtMs": neural_dt_ms},
            "outputNeurons": graph.output_binding["neurons"],
            "inputRouting": projector.manifest(), "policyFingerprint": policy.fingerprint(),
            "metrics": metrics, "frames": frames, "initialState": initial_state,
        }
    return metrics, recording
