#!/usr/bin/env python3
"""
sim_walk_v1.0.py — MuJoCo Unitree Go2 simulation
WITH telemetry logging + plotting
"""

# -----------------------------------------------------------------------------
# MINIMAL FIX: ensure project root is on PYTHONPATH
# -----------------------------------------------------------------------------
import sys
from pathlib import Path
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[3]   # low_level/
sys.path.insert(0, str(PROJECT_ROOT))
# -----------------------------------------------------------------------------

import os
import time
import argparse
import numpy as np
import torch
import yaml
import mujoco
from mujoco import viewer
import matplotlib.pyplot as plt
from collections import deque
from datetime import datetime
from legged_gym import LEGGED_GYM_ROOT_DIR

from common.path_utils import (
    get_policy_path,
    get_mujoco_scene_path,
    get_sim_config_path,
    get_sim_plots_dir,   # <-- ADD THIS
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------- Helpers ----------------- #
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


def ema(prev, x, alpha):
    return x if prev is None else alpha * x + (1.0 - alpha) * prev


# ----------------- Main ----------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str)
    parser.add_argument("--plot", dest="plot", action="store_true", default=True)
    parser.add_argument("--no-plot", dest="plot", action="store_false")
    parser.add_argument("--save", dest="save", action="store_true", default=True)
    parser.add_argument("--no-save", dest="save", action="store_false")
    args = parser.parse_args()

    # ---------------- CONFIG ---------------- #
    config_path = get_sim_config_path(__file__, "v1", args.config_file)
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    policy_path = get_policy_path(__file__, cfg["policy"])
    xml_path = get_mujoco_scene_path(__file__, cfg["mujoco_version"])

    sim_dt = float(cfg["simulation_dt"])
    control_decimation = int(cfg["control_decimation"])
    sim_duration = float(cfg["simulation_duration"])
    ctrl_dt = sim_dt * control_decimation

    kps = np.array(cfg["kps"], np.float32)
    kds = np.array(cfg["kds"], np.float32)
    default_angles = np.array(cfg["default_angles"], np.float32)

    lin_vel_scale = float(cfg["lin_vel_scale"])
    ang_vel_scale = float(cfg["ang_vel_scale"])
    dof_pos_scale = float(cfg["dof_pos_scale"])
    dof_vel_scale = float(cfg["dof_vel_scale"])
    action_scale = float(cfg["action_scale"])
    cmd_scale = np.array(cfg["cmd_scale"], np.float32)

    num_actions = int(cfg["num_actions"])
    num_obs = int(cfg["num_obs"])
    cmd = np.array(cfg["cmd_init"], np.float32)

    # ---------------- State ---------------- #
    action = np.zeros(num_actions, np.float32)
    target_q = default_angles.copy()
    obs = np.zeros(num_obs, np.float32)

    # ---------------- Stabilizers ---------------- #
    K_ROLL, K_YAWRATE, K_LATVY = 0.12, 0.045, 0.020
    K_INT_YAW, YAW_INT_LIMIT = 0.15, 0.18
    MAX_HIP_DELTA = 0.08
    yaw_I = -0.010

    # ---------------- Telemetry ---------------- #
    telemetry = {k: [] for k in [
        "time","heading","yaw_rate","yaw_I","vy","vx",
        "body_z","roll","pitch","action_mean"
    ]}

    # ---------------- Model ---------------- #
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = sim_dt

    policy = torch.jit.load(policy_path).eval()

    print("[INFO] Starting simulation")

    with viewer.launch_passive(m, d) as v:
        start = time.time()
        yaw_ref = None
        step = 0

        while v.is_running() and time.time() - start < sim_duration:
            tau = pd_control(target_q, d.qpos[7:], kps, 0.0, d.qvel[6:], kds)
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)
            step += 1

            if step % control_decimation != 0:
                v.sync()
                continue

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
                action = policy(torch.from_numpy(obs).unsqueeze(0)).numpy().squeeze()

            roll, pitch, yaw = quat_to_euler_wxyz(quat_wxyz)
            yaw_rate = ang_vel[2]
            vx, vy = lin_vel[0], lin_vel[1]

            if yaw_ref is None:
                yaw_ref = yaw

            yaw_I = np.clip(
                yaw_I - (yaw - yaw_ref) * K_INT_YAW * ctrl_dt,
                -YAW_INT_LIMIT,
                YAW_INT_LIMIT
            )

            roll_trim = np.clip(K_ROLL * roll, -MAX_HIP_DELTA, MAX_HIP_DELTA)
            yaw_corr = np.clip(-(K_YAWRATE*yaw_rate + K_LATVY*vy) + yaw_I,
                               -MAX_HIP_DELTA, MAX_HIP_DELTA)

            action[[0,6]] += roll_trim + yaw_corr
            action[[3,9]] += -roll_trim + yaw_corr
            action = np.clip(action, -1, 1)
            target_q = action * action_scale + default_angles

            telemetry["time"].append(time.time() - start)
            telemetry["heading"].append(np.rad2deg(yaw - yaw_ref))
            telemetry["yaw_rate"].append(np.rad2deg(yaw_rate))
            telemetry["yaw_I"].append(yaw_I)
            telemetry["vx"].append(vx)
            telemetry["vy"].append(vy)
            telemetry["body_z"].append(float(d.qpos[2]))
            telemetry["roll"].append(np.rad2deg(roll))
            telemetry["pitch"].append(np.rad2deg(pitch))
            telemetry["action_mean"].append(np.mean(np.abs(action)))

            v.sync()
    
        # ---------------- SAVE LOGS (MINIMAL, SAFE) ---------------- #
    if args.save and len(telemetry["time"]) > 0:
        log_dir = get_sim_plots_dir(__file__)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # ---- CSV ----
        csv_path = os.path.join(log_dir, f"sim_log_{ts}.csv")
        np.savetxt(
            csv_path,
            np.column_stack([
                telemetry["time"],
                telemetry["heading"],
                telemetry["yaw_rate"],
                telemetry["yaw_I"],
                telemetry["vy"],
                telemetry["vx"],
                telemetry["body_z"],
                telemetry["roll"],
                telemetry["pitch"],
                telemetry["action_mean"],
            ]),
            delimiter=",",
            header="time,heading,yaw_rate,yaw_I,vy,vx,body_z,roll,pitch,action_mean",
            comments=""
        )
        print(f"[INFO] CSV saved → {csv_path}")

        # ---- PLOT ----
        if args.plot:
            fig, ax = plt.subplots(3, 3, figsize=(12, 8))
            ax = ax.flatten()
            panels = [
                ("Heading (deg)", telemetry["heading"]),
                ("Yaw rate (deg/s)", telemetry["yaw_rate"]),
                ("Yaw integrator", telemetry["yaw_I"]),
                ("vy (m/s)", telemetry["vy"]),
                ("vx (m/s)", telemetry["vx"]),
                ("Body z (m)", telemetry["body_z"]),
                ("Roll (deg)", telemetry["roll"]),
                ("Pitch (deg)", telemetry["pitch"]),
                ("Mean |a|", telemetry["action_mean"]),
            ]

            t = telemetry["time"]
            for i, (title, data) in enumerate(panels):
                ax[i].plot(t, data)
                ax[i].set_title(title)
                ax[i].grid(True)

            plt.tight_layout()
            plot_path = os.path.join(log_dir, f"sim_plot_{ts}.png")
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            print(f"[INFO] Plot saved → {plot_path}")


    print("[INFO] Simulation finished")
