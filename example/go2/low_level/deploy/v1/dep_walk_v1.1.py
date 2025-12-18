#!/usr/bin/env python3
"""
Go2 Deployment — Raw Policy (v1.1)
-----------------------------------
python3 dep_walk_v1.1.py eno1 go2_dep_v1.1.yaml
Raw policy deployment with structured CSV logging.
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import yaml
import subprocess
import math
import csv
from datetime import datetime

# ---------------------------------------------------------------------------
# Path resolution (robust, repo-anchored)
# ---------------------------------------------------------------------------
THIS_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(THIS_FILE), "..", "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

COMMON_DIR = os.path.join(PROJECT_ROOT, "common")
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
POLICY_DIR = os.path.join(PROJECT_ROOT, "policies")
DEBUG_DIR  = os.path.join(PROJECT_ROOT, "debug")
PLOTS_DIR  = os.path.join(PROJECT_ROOT, "plots")

assert os.path.isdir(COMMON_DIR), f"Missing common/: {COMMON_DIR}"
assert os.path.isdir(CONFIG_DIR), f"Missing config/: {CONFIG_DIR}"
assert os.path.isdir(POLICY_DIR), f"Missing policies/: {POLICY_DIR}"
assert os.path.isdir(DEBUG_DIR),  f"Missing debug/: {DEBUG_DIR}"

# ---------------------------------------------------------------------------
# Unitree DDS
# ---------------------------------------------------------------------------
from unitree_sdk2py.core.channel import (
    ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
)
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.idl.default import (
    unitree_go_msg_dds__LowCmd_ as LowCmdGo,
    unitree_go_msg_dds__LowState_ as LowStateGo,
)

# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------
from common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_go
from common.remote_controller import RemoteController, KeyMap
from common.rotation_helper import get_gravity_orientation

class Go2Deployer:
    """v1.1: raw policy control + structured 50 Hz CSV logging."""

    SIM_TO_HW = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]

    def __init__(self, net_if, cfg_path):
        # ------------------- Mode Switch ------------------- #
        mode_switch_path = os.path.join(DEBUG_DIR, "mode_switch.py")
        assert os.path.isfile(mode_switch_path), f"Missing: {mode_switch_path}"
        python_exec = sys.executable

        print(f"[INFO] Preparing robot for low-level DDS via: {mode_switch_path}")
        subprocess.run([python_exec, mode_switch_path, net_if], check=True)
        print("[INFO] Mode switch successful ✅")

        # ------------------- Load Config ------------------- #
        cfg_path = os.path.abspath(cfg_path)
        if not os.path.isfile(cfg_path):
            cfg_path = os.path.join(CONFIG_DIR, "deploy", "v1", os.path.basename(cfg_path))

        assert os.path.isfile(cfg_path), f"Config not found: {cfg_path}"

        self.cfg = self._load_yaml(cfg_path)
        self.net_if = net_if
        self.control_dt = float(self.cfg.get("control_dt", 0.02))

        # ------------------- Policy Path ------------------- #
        self.policy_path = os.path.join(POLICY_DIR, "policy_v1.pt")
        assert os.path.isfile(self.policy_path), f"Policy not found: {self.policy_path}"


        self.num_actions = int(self.cfg.get("num_actions", 12))
        self.num_obs = int(self.cfg.get("num_obs", 48))

        # Gains and default pose
        self.kps = np.array(self.cfg["kps"], dtype=np.float32)
        self.kds = np.array(self.cfg["kds"], dtype=np.float32)
        self.default_angles = np.array(self.cfg["default_angles"], dtype=np.float32)
        self.default_hw = self.default_angles[self.SIM_TO_HW]

        # Scaling
        self.lin_vel_scale = float(self.cfg.get("lin_vel_scale", 2.0))
        self.ang_vel_scale = float(self.cfg.get("ang_vel_scale", 0.25))
        self.dof_pos_scale = float(self.cfg.get("dof_pos_scale", 1.0))
        self.dof_vel_scale = float(self.cfg.get("dof_vel_scale", 0.05))
        self.action_scale = float(self.cfg.get("action_scale", 0.25))
        self.cmd_scale = np.array(
            self.cfg.get("cmd_scale", [1.0, 1.0, 0.25]), dtype=np.float32
        )

        # DDS setup
        ChannelFactoryInitialize(0, self.net_if)
        self.low_cmd = LowCmdGo()
        self.low_state = LowStateGo()
        self.pub = ChannelPublisher("rt/lowcmd", type(self.low_cmd))
        self.pub.Init()
        self.sub = ChannelSubscriber("rt/lowstate", type(self.low_state))
        self.sub.Init(self._on_lowstate, 10)
        init_cmd_go(self.low_cmd, weak_motor=[])

        # Local buffers
        self.qj = np.zeros(self.num_actions, dtype=np.float32)
        self.dqj = np.zeros(self.num_actions, dtype=np.float32)
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.obs = np.zeros(self.num_obs, dtype=np.float32)
        self.remote = RemoteController()

        # ---------------- Logging Setup ---------------- #
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(PLOTS_DIR, "real_op", "go2_hw", timestamp)
        os.makedirs(
            os.path.join(PLOTS_DIR, "real_op", "go2_hw"),
            exist_ok=True
        )
        os.makedirs(log_dir, exist_ok=True)


        self.log_path = os.path.join(log_dir, f"log_{int(time.time())}.csv")
        self.log_fp = open(self.log_path, "w", newline="")
        self.csv_writer = csv.writer(self.log_fp)

        header = (
            ["t", "phase", "cmd_mag", "roll", "pitch", "yaw_rate",
             "cmd_vx", "cmd_vy", "cmd_yaw"]
            + [f"q_{i}" for i in range(12)]
            + [f"dq_{i}" for i in range(12)]
            + [f"act_{i}" for i in range(12)]
            + [f"tgt_{i}" for i in range(12)]
        )
        self.csv_writer.writerow(header)
        self.t0 = time.time()
        print(f"[INFO] Logging enabled → {self.log_path}")

        # ---------------- Policy Load ---------------- #
        print(f"[INFO] Loading policy from: {self.policy_path}")
        self.policy = torch.jit.load(self.policy_path)
        print("[INFO] Policy loaded successfully.")

        dummy_obs = np.zeros(self.num_obs, dtype=np.float32)
        with torch.no_grad():
            neutral_action = self.policy(
                torch.from_numpy(dummy_obs).unsqueeze(0)
            ).cpu().numpy().squeeze()
        self.policy_neutral_offset = neutral_action
        print(f"[DEBUG] Policy neutral offset: {np.round(self.policy_neutral_offset,3)}")

        self._wait_for_state()
        print("[INFO] Go2 ready for RAW policy control ✅")

    # ------------------------------------------------------------------- #
    def _load_yaml(self, path):
        with open(path, "r") as f:
            return yaml.load(f, Loader=yaml.FullLoader)

    def _on_lowstate(self, msg: LowStateGo):
        self.low_state = msg
        self.remote.set(self.low_state.wireless_remote)

    def _wait_for_state(self):
        print("[INFO] Waiting for robot state...")
        while self.low_state.tick == 0:
            time.sleep(self.control_dt)
        print("[INFO] Connected to Go2 state stream.")

    def _send_cmd(self):
        self.low_cmd.crc = CRC().Crc(self.low_cmd)
        self.pub.Write(self.low_cmd)

    # ------------------------------------------------------------------- #
    def zero_torque_state(self):
        print("[STATE] Zero Torque — Press START to continue.")
        while self.remote.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self._send_cmd()
            time.sleep(self.control_dt)

    def move_to_default(self):
        print("[STATE] Moving to default position (2 s ramp)...")
        steps = int(2.0 / self.control_dt)
        init_q = np.array(
            [self.low_state.motor_state[i].q for i in range(self.num_actions)],
            dtype=np.float32,
        )
        for step in range(steps):
            alpha = step / steps
            q_targets = (1 - alpha) * init_q + alpha * self.default_hw
            for i in range(self.num_actions):
                self.low_cmd.motor_cmd[i].q = q_targets[i]
                self.low_cmd.motor_cmd[i].kp = self.kps[i]
                self.low_cmd.motor_cmd[i].kd = self.kds[i]
            self._send_cmd()
            time.sleep(self.control_dt)
        print("[STATE] Default stance reached ✅")

    def hold_default(self):
        print("[STATE] Holding default position — Press A to start policy.")
        while self.remote.button[KeyMap.A] != 1:
            for i in range(self.num_actions):
                self.low_cmd.motor_cmd[i].q = self.default_hw[i]
                self.low_cmd.motor_cmd[i].kp = self.kps[i]
                self.low_cmd.motor_cmd[i].kd = self.kds[i]
            self._send_cmd()
            time.sleep(self.control_dt)
        print("[STATE] A pressed — running policy...")

    # ------------------------------------------------------------------- #
    def run_once(self):
        # ---------------- Read sensors ---------------- #
        for i in range(self.num_actions):
            self.qj[i] = self.low_state.motor_state[i].q
            self.dqj[i] = self.low_state.motor_state[i].dq
        self.qj = self.qj[self.SIM_TO_HW]
        self.dqj = self.dqj[self.SIM_TO_HW]

        quat = self.low_state.imu_state.quaternion
        ang_vel = np.array(self.low_state.imu_state.gyroscope, dtype=np.float32)
        g_vec = get_gravity_orientation(quat).astype(np.float32)

        cmd = np.array([self.remote.ly, -self.remote.lx, -self.remote.rx], dtype=np.float32)
        cmd_scaled = cmd * self.cmd_scale
        cmd_mag = float(np.linalg.norm(cmd_scaled))
        phase = "idle" if cmd_mag < 0.1 else "move"

        # ---------------- Policy inference ---------------- #
        obs = np.concatenate([
            np.zeros(3, dtype=np.float32) * self.lin_vel_scale,
            ang_vel * self.ang_vel_scale,
            g_vec,
            cmd_scaled,
            (self.qj - self.default_angles) * self.dof_pos_scale,
            self.dqj * self.dof_vel_scale,
            self.action,
        ], dtype=np.float32)
        self.obs = obs[:self.num_obs]

        with torch.no_grad():
            policy_action = self.policy(torch.from_numpy(self.obs).unsqueeze(0)).cpu().numpy().squeeze()

        policy_action -= 0.6 * self.policy_neutral_offset
        self.action = policy_action
        target_dof_pos = (self.default_angles + self.action * self.action_scale)[self.SIM_TO_HW]

        # ---------------- Send to hardware ---------------- #
        for i in range(self.num_actions):
            self.low_cmd.motor_cmd[i].q = float(target_dof_pos[i])
            self.low_cmd.motor_cmd[i].kp = float(self.kps[i])
            self.low_cmd.motor_cmd[i].kd = float(self.kds[i])
            self.low_cmd.motor_cmd[i].tau = 0.0
        self._send_cmd()

        # ---------------- Logging ---------------- #
        t = round(time.time() - self.t0, 3)
        roll = math.atan2(g_vec[1], -g_vec[2])
        pitch = math.atan2(-g_vec[0], math.sqrt(g_vec[1] ** 2 + g_vec[2] ** 2))
        row = [t, phase, round(cmd_mag,3), round(roll,3), round(pitch,3), round(ang_vel[2],3)] \
              + list(np.round(cmd_scaled,3)) \
              + list(np.round(self.qj,3)) \
              + list(np.round(self.dqj,3)) \
              + list(np.round(self.action[self.SIM_TO_HW],3)) \
              + list(np.round(target_dof_pos,3))
        self.csv_writer.writerow(row)

        time.sleep(self.control_dt)

    # ------------------------------------------------------------------- #
    def shutdown(self):
        try:
            self.log_fp.close()
        except Exception:
            pass
        create_damping_cmd(self.low_cmd)
        self._send_cmd()
        print(f"[INFO] Log saved: {self.log_path}")
        print("[INFO] Damping engaged — shutdown complete.")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("net_if", type=str)
    parser.add_argument("config", type=str)
    args = parser.parse_args()

    deploy = Go2Deployer(args.net_if, args.config)
    try:
        deploy.zero_torque_state()
        deploy.move_to_default()
        deploy.hold_default()
        print("[RUN] Running raw policy loop — press SELECT to stop.")
        while True:
            deploy.run_once()
            if deploy.remote.button[KeyMap.select] == 1:
                break
    except KeyboardInterrupt:
        pass
    finally:
        deploy.shutdown()
