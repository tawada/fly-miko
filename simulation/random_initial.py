"""Reproducible random poses: uniform joint limits and uniform root orientation."""
import mujoco
import numpy as np

VERSION = "uniform-joints-orientation-floor-v1"


def lowest_surface(robot):
    """Exact vertical support extent for this body's sphere/capsule/box geoms."""
    bottoms = []
    for i in range(1, robot.model.ngeom):
        kind = robot.model.geom_type[i]
        size = robot.model.geom_size[i]
        rotation = robot.data.geom_xmat[i].reshape(3, 3)
        if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            extent = size[0]
        elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            extent = size[0] + size[1] * abs(rotation[2, 2])
        elif kind == mujoco.mjtGeom.mjGEOM_BOX:
            extent = np.abs(rotation[2]) @ size
        else:
            raise ValueError(f"Unsupported random-pose geometry: {kind}")
        bottoms.append(robot.data.geom_xpos[i, 2] - extent)
    return float(min(bottoms))


def randomize(robot, seed):
    if robot.supported:
        raise ValueError("Random initial state requires a free body")
    rng = np.random.default_rng(seed)
    robot.reset()
    robot.data.qpos[robot.qaddr] = rng.uniform(
        [j['min'] for j in robot.joints], [j['max'] for j in robot.joints])
    quaternion = rng.normal(size=4)
    robot.data.qpos[3:7] = quaternion / np.linalg.norm(quaternion)
    robot.data.qvel[:] = 0
    mujoco.mj_forward(robot.model, robot.data)
    # Start at rest 1 mm above the ground, with no hidden settling simulation.
    robot.data.qpos[2] += .001 - lowest_surface(robot)
    robot.targets = robot.data.qpos[robot.qaddr].copy()
    mujoco.mj_forward(robot.model, robot.data)
    return {"version": VERSION, "seed": int(seed), "qpos": robot.data.qpos.tolist(),
            "qvel": robot.data.qvel.tolist(), "clearanceM": .001}
