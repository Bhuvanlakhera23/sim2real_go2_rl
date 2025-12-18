#!/usr/bin/env python3
"""
✅ Keeps PD + policy structure identical to Unitree
✅ Adds roll/yaw/lateral stabilization & integral yaw bias correction
✅ Adds robust camera tracking of the robot base
❌ No plotting/CSV overhead
"""

import os
import time
import argparse
import numpy as np
import torch
import yaml
import mujoco
from mujoco import viewer
from legged_gym import LEGGED_GYM_ROOT_DIR


# ---------------- Path Resolver ---------------- #
def resolve_path(path: str) -> str:
    path = os.path.expandvars(os.path.expanduser(path))
    if "{LEGGED_GYM_ROOT_DIR}" in path:
        path = path.replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

    if os.path.isabs(path):
        if os.path.exists(path):
            return os.path.normpath(path)
        raise FileNotFoundError(path)

    candidates = [
        os.path.abspath(path),
        os.path.join(os.path.dirname(__file__), path),
        os.path.join(os.path.dirname(__file__), "../../../config/sim/v1", path),
    ]

    for c in candidates:
        if os.path.exists(c):
            return os.path.normpath(c)

    raise FileNotFoundError(
        "[ERROR] File not found. Tried:\n" +
        "\n".join(f"  {c}" for c in candidates)
    )


# ---------------- Math Helpers ---------------- #
def get_gravity_orientation(quaternion):
    qw, qx, qy, qz = quaternion
    g = np.zeros(3, dtype=np.float32)
    g[0] = 2 * (-qz * qx + qw * qy)
    g[1] = -2 * (qz * qy + qw * qx)
    g[2] = 1 - 2 * (qw * qw + qz * qz)
    return g


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


def quat_to_euler_wxyz(q):
    w, x, y, z = q
    roll  = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = np.arcsin(np.clip(2*(w*y - z*x), -1.0, 1.0))
    yaw   = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return roll, pitch, yaw


def find_base_body_id(m):
    for name in ["base", "trunk", "body", "base_link"]:
        try:
            return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name), name
        except ValueError:
            pass
    return 1, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, 1)


# ---------------- Main ---------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str)
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # ✅ FIXED CONFIG LOADING (MINIMAL)
    # ------------------------------------------------------------------
    cfg_path = resolve_path(args.config_file)
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    if "policy" not in cfg:
        raise KeyError("[CONFIG ERROR] Missing 'policy' in sim YAML")

    if "mujoco_version" not in cfg or "scene" not in cfg:
        raise KeyError("[CONFIG ERROR] Missing 'mujoco_version' or 'scene' in sim YAML")

    policy_path = resolve_path(
        os.path.join("../../../policies", cfg["policy"])
    )

    xml_path = resolve_path(
        os.path.join(
            os.path.dirname(__file__),
            cfg["mujoco_version"],
            cfg["scene"]
        )
    )
    # ------------------------------------------------------------------

    # --- Simulation params ---
    sim_dt = float(cfg["simulation_dt"])
    control_decimation = int(cfg["control_decimation"])
    sim_duration = float(cfg["simulation_duration"])

    kps = np.array(cfg["kps"], np.float32)
    kds = np.array(cfg["kds"], np.float32)
    default_angles = np.array(cfg["default_angles"], np.float32)

    lin_vel_scale = float(cfg.get("lin_vel_scale", 1.0))
    ang_vel_scale = float(cfg["ang_vel_scale"])
    dof_pos_scale = float(cfg["dof_pos_scale"])
    dof_vel_scale = float(cfg["dof_vel_scale"])
    action_scale = float(cfg["action_scale"])
    cmd_scale = np.array(cfg["cmd_scale"], np.float32)

    num_actions = int(cfg["num_actions"])
    num_obs = int(cfg["num_obs"])
    cmd = np.array(cfg["cmd_init"], np.float32)

    # --- State ---
    action = np.zeros(num_actions, np.float32)
    target_q = default_angles.copy()
    obs = np.zeros(num_obs, np.float32)

    # --- Stabilizers ---
    K_ROLL, K_YAWRATE, K_LATVY = 0.12, 0.045, 0.020
    MAX_HIP_DELTA = 0.08
    K_INT_YAW, YAW_INT_LIMIT = 0.15, 0.18
    yaw_I = -0.010

    # --- Load model & policy ---
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = sim_dt

    policy = torch.jit.load(policy_path).eval()
    device = next(policy.parameters()).device

    base_id, base_name = find_base_body_id(m)
    print(f"[INFO] Tracking body '{base_name}'")

    with viewer.launch_passive(m, d) as v:
        v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        v.cam.trackbodyid = base_id
        v.cam.distance = 1.2
        v.cam.elevation = -10
        v.cam.azimuth = 180

        t0 = time.time()
        yaw_ref = None
        step = 0

        while v.is_running() and time.time() - t0 < sim_duration:
            t_step = time.time()

            tau = pd_control(target_q, d.qpos[7:], kps, 0.0, d.qvel[6:], kds)
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)
            step += 1

            if step % control_decimation == 0:
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat_xyzw = d.qpos[3:7]
                quat_wxyz = np.roll(quat_xyzw, 1)

                lin_vel = d.qvel[:3]
                ang_vel = d.qvel[3:6]

                obs[:3] = lin_vel * lin_vel_scale
                obs[3:6] = ang_vel * ang_vel_scale
                obs[6:9] = get_gravity_orientation(quat_xyzw)
                obs[9:12] = cmd * cmd_scale
                obs[12:12+num_actions] = (qj - default_angles) * dof_pos_scale
                obs[12+num_actions:12+2*num_actions] = dqj * dof_vel_scale
                obs[12+2*num_actions:12+3*num_actions] = action

                with torch.no_grad():
                    action = policy(
                        torch.from_numpy(obs).to(device).unsqueeze(0)
                    ).cpu().numpy().squeeze()

                roll, _, yaw = quat_to_euler_wxyz(quat_wxyz)
                yaw_rate = ang_vel[2]
                vy = lin_vel[1]

                if yaw_ref is None:
                    yaw_ref = yaw

                yaw_I += -(yaw - yaw_ref) * K_INT_YAW * sim_dt * control_decimation
                yaw_I = np.clip(yaw_I, -YAW_INT_LIMIT, YAW_INT_LIMIT)

                corr = np.clip(
                    -(K_YAWRATE*yaw_rate + K_LATVY*vy) + yaw_I,
                    -MAX_HIP_DELTA,
                    MAX_HIP_DELTA
                )

                roll_trim = np.clip(K_ROLL * roll, -MAX_HIP_DELTA, MAX_HIP_DELTA)

                for i, s in zip([0, 3, 6, 9], [1, -1, 1, -1]):
                    action[i] += s * roll_trim + corr

                action = np.clip(action, -1, 1)
                target_q = action * action_scale + default_angles

            v.sync()
            dt = sim_dt - (time.time() - t_step)
            if dt > 0:
                time.sleep(dt)

    print("[INFO] Simulation complete.")
