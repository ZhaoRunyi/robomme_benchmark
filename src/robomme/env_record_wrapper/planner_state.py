"""Restore live robot state when an expert planner joins mid-episode."""

from __future__ import annotations

from typing import Any

import numpy as np


def sync_gripper_state_from_robot(
    planner: Any,
    robot: Any,
    gripper_controller: Any | None = None,
    closed_threshold: float = 0.02,
) -> float | None:
    """Set a fresh planner's gripper command from the live finger joints.

    Stick robots expose only the seven arm joints and therefore have no
    gripper state to restore.
    """
    qpos = robot.get_qpos()
    if hasattr(qpos, "detach"):
        qpos = qpos.detach().cpu().numpy()
    qpos_array = np.asarray(qpos, dtype=np.float64).reshape(-1)
    if qpos_array.size == 7:
        return None
    if qpos_array.size < 9 or not np.isfinite(qpos_array[-2:]).all():
        raise ValueError(f"invalid Panda qpos for gripper sync: {qpos_array.shape}")

    commanded_position = None
    if gripper_controller is not None:
        target_qpos = getattr(gripper_controller, "_target_qpos", None)
        control_indices = getattr(gripper_controller, "control_joint_indices", None)
        if target_qpos is not None and control_indices is not None:
            if hasattr(target_qpos, "detach"):
                target_qpos = target_qpos.detach().cpu().numpy()
            if hasattr(control_indices, "detach"):
                control_indices = control_indices.detach().cpu().numpy()
            target_array = np.asarray(target_qpos, dtype=np.float64).reshape(-1)
            index_array = np.asarray(control_indices, dtype=np.int64).reshape(-1)
            if index_array.size and np.isfinite(target_array[index_array]).all():
                commanded_position = float(target_array[index_array].mean())

    if commanded_position is None:
        commanded_position = float(qpos_array[-2:].mean())
    gripper_state = (
        planner.OPEN
        if commanded_position > closed_threshold / 2.0
        else planner.CLOSED
    )
    planner.gripper_state = gripper_state
    return float(gripper_state)
