#!/usr/bin/env python3
"""
Bare Bones Deployment Script for Go2 using a pre-trained policy:
- No stabilization or state estimation; direct policy inference.
- Joint position commands with PD control.
- No logging or data saving.

CLI USAGE:
python3 dep_walk_v1.0.py eno1 go2_dep_v1.0.yaml
"""

import sys, os, time, argparse, numpy as np, torch, yaml, subprocess
from unitree_sdk2py.core.channel import (
    ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
)
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.idl.default import (
    unitree_go_msg_dds__LowCmd_ as LowCmdGo,
    unitree_go_msg_dds__LowState_ as LowStateGo,
)

# ---------------------------------------------------------------------------
THIS_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(THIS_FILE), "..", "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# ---------------------------------------------------------------------------#
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
POLICY_DIR = os.path.join(PROJECT_ROOT, "policies")

# --------------------------------------------------------------------------- #
# 🧩 LOCAL HELPERS
# --------------------------------------------------------------------------- #
from common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_go
from common.remote_controller import RemoteController, KeyMap
from common.rotation_helper import get_gravity_orientation


class Go2Deployer:
    """Raw policy deployment, training-consistent."""

    # SIM → HW joint index map
    SIM_TO_HW = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]

    def __init__(self, net_if, cfg_path):
        # ---------------- Mode Switch ---------------- #
        mode_switch_path = os.path.join(PROJECT_ROOT, "debug", "mode_switch.py")
        assert os.path.isfile(mode_switch_path), f"Missing: {mode_switch_path}"

        subprocess.run(
            [sys.executable, mode_switch_path, net_if],
            check=True,
        )


        # ---------------- Load Config ---------------- #
        cfg_arg = cfg_path  # keep original CLI input

        # 1. Absolute path provided
        if os.path.isabs(cfg_arg) and os.path.isfile(cfg_arg):
            cfg_path = cfg_arg

        # 2. Relative path provided (from cwd)
        elif os.path.isfile(os.path.abspath(cfg_arg)):
            cfg_path = os.path.abspath(cfg_arg)

        # 3. Bare filename → assume config/deploy/v1
        else:
            cfg_path = os.path.join(
                CONFIG_DIR, "deploy", "v1", os.path.basename(cfg_arg)
            )

        assert os.path.isfile(cfg_path), f"Config not found: {cfg_path}"


        with open(cfg_path, "r") as f:
            self.cfg = yaml.load(f, Loader=yaml.FullLoader)

        self.control_dt = float(self.cfg.get("control_dt", 0.02))

        self.policy_path = os.path.join(POLICY_DIR, "policy_v1.pt")
        assert os.path.isfile(self.policy_path), f"Policy not found: {self.policy_path}"


        self.num_actions = int(self.cfg.get("num_actions", 12))
        self.num_obs = int(self.cfg.get("num_obs", 48))

        # ---------------- Gains / Pose ---------------- #
        self.kps = np.array(self.cfg["kps"], dtype=np.float32)
        self.kds = np.array(self.cfg["kds"], dtype=np.float32)

        self.default_angles = np.array(
            self.cfg["default_angles"], dtype=np.float32
        )
        self.default_hw = self.default_angles[self.SIM_TO_HW]

        # ---------------- Scales ---------------- #
        self.lin_vel_scale = float(self.cfg.get("lin_vel_scale", 2.0))
        self.ang_vel_scale = float(self.cfg.get("ang_vel_scale", 0.25))
        self.dof_pos_scale = float(self.cfg.get("dof_pos_scale", 1.0))
        self.dof_vel_scale = float(self.cfg.get("dof_vel_scale", 0.05))
        self.action_scale = float(self.cfg.get("action_scale", 0.25))
        self.cmd_scale = np.array(
            self.cfg.get("cmd_scale", [1.0, 1.0, 0.25]),
            dtype=np.float32,
        )

        # ---------------- DDS ---------------- #
        ChannelFactoryInitialize(0, net_if)
        self.low_cmd = LowCmdGo()
        self.low_state = LowStateGo()

        self.pub = ChannelPublisher("rt/lowcmd", type(self.low_cmd))
        self.pub.Init()

        self.sub = ChannelSubscriber("rt/lowstate", type(self.low_state))
        self.sub.Init(self._on_lowstate, 10)

        init_cmd_go(self.low_cmd, weak_motor=[])

        # ---------------- Buffers ---------------- #
        self.qj = np.zeros(self.num_actions, dtype=np.float32)
        self.dqj = np.zeros(self.num_actions, dtype=np.float32)
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.obs = np.zeros(self.num_obs, dtype=np.float32)

        self.remote = RemoteController()

        # ---------------- Policy ---------------- #
        self.policy = torch.jit.load(self.policy_path)

        dummy_obs = np.zeros(self.num_obs, dtype=np.float32)
        with torch.no_grad():
            self.policy_neutral_offset = (
                self.policy(torch.from_numpy(dummy_obs).unsqueeze(0))
                .cpu()
                .numpy()
                .squeeze()
            )

        self._wait_for_state()

    # ------------------------------------------------------------------- #
    def _on_lowstate(self, msg: LowStateGo):
        self.low_state = msg
        self.remote.set(self.low_state.wireless_remote)

    def _wait_for_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.control_dt)

    def _send_cmd(self):
        self.low_cmd.crc = CRC().Crc(self.low_cmd)
        self.pub.Write(self.low_cmd)

    # ------------------------------------------------------------------- #
    def zero_torque_state(self):
        while self.remote.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self._send_cmd()
            time.sleep(self.control_dt)

    def move_to_default(self):
        steps = int(2.0 / self.control_dt)
        init_q = np.array(
            [self.low_state.motor_state[i].q for i in range(self.num_actions)],
            dtype=np.float32,
        )

        for s in range(steps):
            a = s / steps
            q = (1 - a) * init_q + a * self.default_hw
            for i in range(self.num_actions):
                self.low_cmd.motor_cmd[i].q = float(q[i])
                self.low_cmd.motor_cmd[i].kp = float(self.kps[i])
                self.low_cmd.motor_cmd[i].kd = float(self.kds[i])
            self._send_cmd()
            time.sleep(self.control_dt)

    def hold_default(self):
        while self.remote.button[KeyMap.A] != 1:
            for i in range(self.num_actions):
                self.low_cmd.motor_cmd[i].q = float(self.default_hw[i])
                self.low_cmd.motor_cmd[i].kp = float(self.kps[i])
                self.low_cmd.motor_cmd[i].kd = float(self.kds[i])
            self._send_cmd()
            time.sleep(self.control_dt)

    # ------------------------------------------------------------------- #
    def run_once(self):
        # joints
        for i in range(self.num_actions):
            self.qj[i] = self.low_state.motor_state[i].q
            self.dqj[i] = self.low_state.motor_state[i].dq

        self.qj = self.qj[self.SIM_TO_HW]
        self.dqj = self.dqj[self.SIM_TO_HW]

        # imu
        quat = self.low_state.imu_state.quaternion
        ang_vel = np.array(self.low_state.imu_state.gyroscope, dtype=np.float32)
        g_vec = get_gravity_orientation(quat).astype(np.float32)

        # joystick → commands
        cmd = np.array(
            [self.remote.ly, -self.remote.lx, -self.remote.rx],
            dtype=np.float32,
        )
        cmd_scaled = cmd * self.cmd_scale

        # observation (MATCHES legged_gym)
        self.obs = np.concatenate(
            [
                np.zeros(3, dtype=np.float32) * self.lin_vel_scale,
                ang_vel * self.ang_vel_scale,
                g_vec,
                cmd_scaled,
                (self.qj - self.default_angles) * self.dof_pos_scale,
                self.dqj * self.dof_vel_scale,
                self.action,
            ],
            dtype=np.float32,
        )[: self.num_obs]

        # policy
        with torch.no_grad():
            act = (
                self.policy(torch.from_numpy(self.obs).unsqueeze(0))
                .cpu()
                .numpy()
                .squeeze()
            )

        act -= 0.6 * self.policy_neutral_offset
        self.action = act

        target = (self.default_angles + act * self.action_scale)[self.SIM_TO_HW]

        for i in range(self.num_actions):
            self.low_cmd.motor_cmd[i].q = float(target[i])
            self.low_cmd.motor_cmd[i].kp = float(self.kps[i])
            self.low_cmd.motor_cmd[i].kd = float(self.kds[i])
            self.low_cmd.motor_cmd[i].tau = 0.0

        self._send_cmd()
        time.sleep(self.control_dt)

    # ------------------------------------------------------------------- #
    def shutdown(self):
        create_damping_cmd(self.low_cmd)
        self._send_cmd()


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("net_if", type=str)
    parser.add_argument("config", type=str)
    args = parser.parse_args()

    d = Go2Deployer(args.net_if, args.config)

    try:
        d.zero_torque_state()
        d.move_to_default()
        d.hold_default()
        while d.remote.button[KeyMap.select] != 1:
            d.run_once()
    finally:
        d.shutdown()