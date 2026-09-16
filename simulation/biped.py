"""A free-root 18-DOF calibration body with measured proprioception and contacts."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BODY = json.loads((ROOT / "config/body-map.json").read_text())
# MJ coordinates: x forward, y left, z up. Three.js: x left, y up, z forward.
MJ_TO_THREE = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
THREE_TO_MJ = MJ_TO_THREE.T
CONTACT_PARTS = [
    f"{side}_{part}" for side in ("left", "right")
    for part in ("hand", "upper_arm", "forearm", "foot", "thigh", "shin")
]
CRAWL_NECK_PITCH = -1.31


def text(values):
    return " ".join(f"{float(value):.9g}" for value in values)


def make_model_xml(initial_pose="standing"):
    """Approximate human proportions in metres; meshes do not define collisions."""
    root = ET.Element("mujoco", model="miko_sensor_biped")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    ET.SubElement(root, "option", timestep="0.002", gravity="0 0 -9.81",
                  integrator="implicitfast", iterations="80")
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "joint", damping="2", armature="0.02", limited="true")
    # Only body-vs-ground collisions: category 2 collides with category 1.
    ET.SubElement(default, "geom", friction="0.9 0.02 0.002", condim="3",
                  contype="2", conaffinity="1", solref="0.01 1")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "geom", name="floor", type="plane", size="20 20 0.1",
                  contype="1", conaffinity="2", rgba="0.8 0.8 0.8 1")
    pelvis = ET.SubElement(world, "body", name="pelvis", pos="0 0 0.98")
    ET.SubElement(pelvis, "freejoint", name="root")
    ET.SubElement(pelvis, "geom", name="pelvis_geom", type="box", size="0.085 0.115 0.065",
                  mass="5")

    def joints(parent, ids):
        # Compose XYZ offsets in the same order as the Three.js rig.
        for joint in sorted((j for j in BODY["joints"] if j["id"] in ids),
                            key=lambda j: "xyz".index(j["axis"])):
            axis_three = np.zeros(3)
            axis_three["xyz".index(joint["axis"])] = joint["sign"]
            ET.SubElement(parent, "joint", name=joint["id"], type="hinge",
                          axis=text(THREE_TO_MJ @ axis_three),
                          range=text([joint["min"], joint["max"]]))

    torso = ET.SubElement(pelvis, "body", name="torso", pos="0 0 0.002")
    joints(torso, ["torso_pitch", "torso_roll"])
    ET.SubElement(torso, "geom", name="torso_geom", type="capsule",
                  fromto="0 0 0.04 0 0 0.22", size="0.1", mass="8")
    neck_pitch = CRAWL_NECK_PITCH if initial_pose == "splayed-crawl" else 0.
    neck = ET.SubElement(torso, "body", name="neck", pos="0 0 0.3",
                         quat=text([np.cos(neck_pitch / 2), 0, np.sin(neck_pitch / 2), 0]))
    ET.SubElement(neck, "geom", name="head_geom", type="sphere",
                  pos="0 0 0.1", size="0.095", mass="3")
    for side, lateral in [("left", 1), ("right", -1)]:
        thigh = ET.SubElement(pelvis, "body", name=f"{side}_thigh",
                              pos=text([0, lateral * 0.071, -0.16]))
        joints(thigh, [f"{side}_hip_{axis}" for axis in ["pitch", "roll", "yaw"]])
        ET.SubElement(thigh, "geom", name=f"{side}_thigh_geom", type="capsule",
                      fromto="0 0 -0.03 0 0 -0.29", size="0.047", mass="3")
        shin = ET.SubElement(thigh, "body", name=f"{side}_shin", pos="0 0 -0.323")
        joints(shin, [f"{side}_knee"])
        ET.SubElement(shin, "geom", name=f"{side}_shin_geom", type="capsule",
                      fromto="0 0 -0.025 0 0 -0.335", size="0.032", mass="1.8")
        foot = ET.SubElement(shin, "body", name=f"{side}_foot", pos="0 0 -0.365")
        joints(foot, [f"{side}_ankle_pitch", f"{side}_ankle_roll"])
        ET.SubElement(foot, "geom", name=f"{side}_sole", type="box",
                      pos="0.035 0 -0.085", size="0.095 0.043 0.047", mass="0.8")
        arm = ET.SubElement(torso, "body", name=f"{side}_upper_arm",
                             pos=text([0, lateral * 0.13, 0.225]))
        joints(arm, [f"{side}_shoulder_pitch"])
        ET.SubElement(arm, "geom", name=f"{side}_upper_arm_geom", type="capsule",
                      fromto="0 0 -0.02 0 0 -0.21", size="0.03", mass="1")
        forearm = ET.SubElement(arm, "body", name=f"{side}_forearm", pos="0 0 -0.23")
        joints(forearm, [f"{side}_elbow"])
        ET.SubElement(forearm, "geom", name=f"{side}_forearm_geom", type="capsule",
                      fromto="0 0 -0.02 0 0 -0.16", size="0.025", mass="0.55")
        ET.SubElement(forearm, "geom", name=f"{side}_hand_geom", type="sphere",
                      pos="0 0 -0.2", size="0.025", mass="0.15")
    equality = ET.SubElement(root, "equality")
    ET.SubElement(equality, "weld", name="calibration_support", body1="pelvis", active="false")
    actuator = ET.SubElement(root, "actuator")
    for joint in BODY["joints"]:
        ET.SubElement(actuator, "motor", name=f"motor_{joint['id']}", joint=joint["id"],
                      ctrlrange="-100 100", ctrllimited="true")
    return ET.tostring(root, encoding="unicode")


class Biped:
    physics_dt = 0.002
    control_dt = 0.02

    def __init__(self, supported=False, initial_pose="standing"):
        if initial_pose not in ("standing", "splayed-crawl"):
            raise ValueError("Unknown initial pose")
        self.initial_pose = initial_pose
        self.model = mujoco.MjModel.from_xml_string(make_model_xml(initial_pose))
        self.data = mujoco.MjData(self.model)
        self.joints = BODY["joints"]
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j["id"])
                          for j in self.joints]
        self.qaddr = self.model.jnt_qposadr[self.joint_ids]
        self.vaddr = self.model.jnt_dofadr[self.joint_ids]
        self.pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.head_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "head_geom")
        self.sole_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_sole")
                         for side in ("left", "right")]
        self.contact_geom_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM,
                             part.replace("_foot", "_sole") if part.endswith("_foot") else part + "_geom")
            for part in CONTACT_PARTS
        ]
        self.contact_index = {identifier: i for i, identifier in enumerate(self.contact_geom_ids)}
        self.supported = bool(supported)
        self.mass = float(mujoco.mj_getTotalmass(self.model))
        self.reset()

    def reset(self, height=None):
        mujoco.mj_resetData(self.model, self.data)
        if height is not None:
            if not np.isfinite(height):
                raise ValueError("height must be finite")
            self.data.qpos[2] = height
        self.data.qpos[self.qaddr] = [j["neutral"] for j in self.joints]
        if self.initial_pose == "splayed-crawl":
            # Symmetric, outward hips; the knee and distal forearm collision
            # surfaces touch the ground. Free root, zero velocity, no weld.
            pitch = 1.31
            self.data.qpos[2] = 0.3397689 if height is None else height
            self.data.qpos[3:7] = [np.cos(pitch / 2), 0., np.sin(pitch / 2), 0.]
            pose = crawl_angles()
            self.data.qpos[self.qaddr] = [pose[j["id"]] for j in self.joints]
        self.data.eq_active[0] = self.supported
        self.targets = self.data.qpos[self.qaddr].copy()
        mujoco.mj_forward(self.model, self.data)
        return self.snapshot()

    def advance(self, targets):
        """One 20 ms control period. Targets are commands; feedback comes from qpos/qvel."""
        expected = {j["id"] for j in self.joints}
        if not isinstance(targets, dict) or set(targets) != expected:
            raise ValueError("Provide exactly the configured 18 joint targets")
        values = np.array([targets[j["id"]] for j in self.joints], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("Nonfinite motor target")
        low = np.array([j["min"] for j in self.joints])
        high = np.array([j["max"] for j in self.joints])
        if np.any(values < low) or np.any(values > high):
            raise ValueError("Motor target outside configured limits")
        self.targets = values
        impulses = np.zeros(2)
        for _ in range(10):
            q = self.data.qpos[self.qaddr]
            v = self.data.qvel[self.vaddr]
            # A bounded PD servo, not learned dynamics or a gait controller.
            self.data.ctrl[:] = np.clip(180 * (values - q) - 12 * v, -100, 100)
            mujoco.mj_step(self.model, self.data)
            impulses += self.contact_forces() * self.physics_dt
        if not np.all(np.isfinite(self.data.qpos)) or not np.all(np.isfinite(self.data.qvel)):
            raise RuntimeError("Nonfinite physics state")
        # mj_step integrates position after evaluating forces; forward makes returned
        # positions, velocities, contact forces and root transform describe the same state.
        mujoco.mj_forward(self.model, self.data)
        snapshot = self.snapshot()
        snapshot["footImpulseNs"] = impulses.tolist()
        return snapshot

    def contact_forces(self):
        forces = self.body_contact_forces()
        return forces[[CONTACT_PARTS.index(f"{side}_foot") for side in ("left", "right")]]

    def body_contact_forces(self):
        force = np.zeros(6)
        result = np.zeros(len(CONTACT_PARTS))
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if self.floor_id not in (contact.geom1, contact.geom2):
                continue
            other = contact.geom2 if contact.geom1 == self.floor_id else contact.geom1
            if other in self.contact_index:
                mujoco.mj_contactForce(self.model, self.data, index, force)
                result[self.contact_index[other]] += max(0.0, float(force[0]))
        return result

    def snapshot(self):
        rotation = self.data.xmat[self.pelvis_id].reshape(3, 3)
        gravity_body = rotation.T @ np.array([0., 0., -1.])
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                                self.pelvis_id, velocity, 1)
        body_forces = self.body_contact_forces()
        forces = body_forces[[CONTACT_PARTS.index(f"{side}_foot") for side in ("left", "right")]]
        contacts = (forces > 0.5).astype(int)
        root_rotation = MJ_TO_THREE @ rotation @ THREE_TO_MJ
        quat_wxyz = np.zeros(4)
        mujoco.mju_mat2Quat(quat_wxyz, root_rotation.reshape(-1))
        geometry_poses = []
        for index in range(1, self.model.ngeom):
            geometry_quat = np.zeros(4)
            # Geometries keep MuJoCo local axes; only world coordinates are converted.
            mujoco.mju_mat2Quat(geometry_quat, (MJ_TO_THREE @ self.data.geom_xmat[index].reshape(3, 3)).reshape(-1))
            geometry_poses.append({
                "position": (MJ_TO_THREE @ self.data.geom_xpos[index]).tolist(),
                "quaternion": [float(geometry_quat[i]) for i in (1, 2, 3, 0)],
            })
        return {
            "timeSeconds": float(self.data.time),
            "source": "mujoco-measured",
            "supported": self.supported,
            "jointAngles": self.data.qpos[self.qaddr].tolist(),
            "jointVelocities": self.data.qvel[self.vaddr].tolist(),
            "projectedGravity": (MJ_TO_THREE @ gravity_body).tolist(),
            "linearVelocity": (MJ_TO_THREE @ velocity[3:]).tolist(),
            "angularVelocity": (MJ_TO_THREE @ velocity[:3]).tolist(),
            "footContacts": contacts.tolist(),
            "footForcesN": forces.tolist(),
            "bodyContactIds": CONTACT_PARTS,
            "bodyContacts": (body_forces > .5).astype(int).tolist(),
            "bodyContactForcesN": body_forces.tolist(),
            "targetVelocity": [0., 0., 0.],
            "commandAngles": self.targets.tolist(),
            "rootPositionM": (MJ_TO_THREE @ self.data.xpos[self.pelvis_id]).tolist(),
            "headHeightM": float(self.data.geom_xpos[self.head_id, 2]),
            "rootQuaternion": [float(quat_wxyz[i]) for i in (1, 2, 3, 0)],
            "geometryPoses": geometry_poses,
        }

    def geometry_specs(self):
        return [
            {"name": mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, index),
             "type": int(self.model.geom_type[index]), "size": self.model.geom_size[index].tolist()}
            for index in range(1, self.model.ngeom)
        ]


def crawl_angles():
    pose = {j["id"]: j["neutral"] for j in BODY["joints"]}
    for side, yaw in (("left", -0.3), ("right", 0.3)):
        pose.update({
            f"{side}_hip_pitch": 0.79283928, f"{side}_hip_roll": 0.4,
            f"{side}_hip_yaw": yaw, f"{side}_knee": 1.5,
            f"{side}_shoulder_pitch": 0.7, f"{side}_elbow": 1.0,
        })
    return pose
