#!/usr/bin/env python3
"""
Go2 Simulation with Proper Initialization (CLEAN)

✔ Deterministic policy path handling
✔ Explicit device flow (policy → env)
✔ Safe PD-based initialization
✔ Commanded velocity synced with env.commands
✔ No shadowed variables
✔ Fail-fast on configuration errors

python3 gym_walk_v1.0.py


NOTE:
This runs a torque-trained policy through PD.
That mismatch is intentional for validation only.
"""

# =============================================================================
# Imports & Path Setup
# =============================================================================
import os
import sys
import argparse
import numpy as np

# Absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))
POLICY_DIR = os.path.join(PROJECT_ROOT, "policies")

assert os.path.isdir(PROJECT_ROOT), f"Invalid project root: {PROJECT_ROOT}"
assert os.path.isdir(POLICY_DIR), f"Missing policies dir: {POLICY_DIR}"

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Isaac Gym / Legged Gym
import isaacgym  # type: ignore  # noqa
from isaacgym import gymapi, gymtorch  # type: ignore  # noqa
from legged_gym.envs import *  # noqa
from legged_gym.utils import task_registry

import torch


# =============================================================================
# Policy Wrapper
# =============================================================================
class PolicyWrapper:
    def __init__(self, policy_path: str, device: torch.device):
        assert os.path.isfile(policy_path), f"❌ Policy not found: {policy_path}"
        self.device = device
        self.policy = torch.jit.load(policy_path, map_location=device)
        self.policy.eval()
        print(f"✅ Policy loaded: {policy_path} on {device}")

    @torch.no_grad()
    def predict(self, obs: torch.Tensor, out_device: torch.device) -> torch.Tensor:
        action = self.policy(obs.to(self.device).unsqueeze(0)).squeeze(0)
        return action.to(out_device)


# =============================================================================
# PD Helpers (Initialization Only)
# =============================================================================
def compute_pd_torques(env, target_pos: torch.Tensor) -> torch.Tensor:
    device = torch.device(env.device)

    kp = torch.zeros(env.num_dof, device=device)
    kd = torch.zeros(env.num_dof, device=device)

    stiffness = env.cfg.control.stiffness
    damping = env.cfg.control.damping

    if isinstance(stiffness, dict):
        for i, name in enumerate(env.dof_names):
            for key in stiffness:
                if key in name:
                    kp[i] = stiffness[key]
                    kd[i] = damping[key]
                    break
    else:
        kp[:] = float(stiffness)
        kd[:] = float(damping)

    pos_err = target_pos.unsqueeze(0) - env.dof_pos
    torques = kp.unsqueeze(0) * pos_err - kd.unsqueeze(0) * env.dof_vel
    return torch.clamp(torques, -env.torque_limits, env.torque_limits)


def initialize_robot_with_pd_control(env, default_dof_pos: torch.Tensor) -> bool:
    print("\n⚙️ Initializing robot with PD hold...")

    env.dof_pos[:] = default_dof_pos.unsqueeze(0)
    env.dof_vel[:] = 0.0
    env.root_states[:, 0:3] = torch.tensor([0.0, 0.0, 0.42], device=env.device)
    env.root_states[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.device)
    env.root_states[:, 7:13] = 0.0

    env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(env.dof_state))
    env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.root_states))

    for i in range(200):
        torques = compute_pd_torques(env, default_dof_pos)
        env.gym.set_dof_actuation_force_tensor(
            env.sim, gymtorch.unwrap_tensor(torques)
        )
        env.gym.simulate(env.sim)
        env.gym.fetch_results(env.sim, True)
        env.gym.refresh_dof_state_tensor(env.sim)
        env.gym.refresh_actor_root_state_tensor(env.sim)

        if i % 50 == 0:
            z = float(env.root_states[0, 2])
            print(f"   settle step {i:03d} | base_z={z:.3f}")

    env.gym.refresh_net_contact_force_tensor(env.sim)
    forces = env.contact_forces[0][env.feet_indices].norm(dim=1).cpu().numpy()
    print("   Foot contact forces:", forces)

    return bool(np.all(forces > 10.0))


# =============================================================================
# Main Simulation
# =============================================================================
def run_simulation(policy_path: str, duration_s: float, command_mode: str):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️ Using device: {device}")
    print("=" * 80)

    env_cfg, _ = task_registry.get_cfgs("go2")

    # PD gains
    env_cfg.control.stiffness = dict(hip=40.0, thigh=40.0, calf=40.0, joint=40.0)
    env_cfg.control.damping = dict(hip=1.0, thigh=1.0, calf=1.0, joint=1.0)
    print("🔧 PD gains: Kp=40, Kd=1.0")

    # Environment settings
    env_cfg.env.num_envs = 1
    env_cfg.env.test = True
    env_cfg.commands.curriculum = False
    env_cfg.commands.resampling_time = 1e6
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.curriculum = False
    env_cfg.sim.gravity = [0.0, 0.0, -9.81]
    env_cfg.init_state.pos = [0.0, 0.0, 0.42]

    env, _ = task_registry.make_env("go2", None, env_cfg)

    if env.viewer is not None:
        env.gym.viewer_camera_look_at(
            env.viewer,
            None,
            gymapi.Vec3(2.0, -1.5, 1.2),
            gymapi.Vec3(0.0, 0.0, 0.5),
        )

    obs_scales = env.cfg.normalization.obs_scales
    policy = PolicyWrapper(policy_path, device)

    default_dof_pos = env.default_dof_pos[0]
    initialize_robot_with_pd_control(env, default_dof_pos)

    # Critical warm-up
    env.get_observations()

    commanded_vel = torch.tensor(
        [1.0, 0.0, 0.0] if command_mode == "forward" else [0.0, 0.0, 0.0],
        device=env.device,
    )
    env.commands[:, :3] = commanded_vel.unsqueeze(0)

    last_actions = torch.zeros(env.num_actions, device=env.device)

    control_dt = env.cfg.control.decimation * env.cfg.sim.dt
    num_steps = int(duration_s / control_dt)

    print(f"\n🏃 Running policy at {1/control_dt:.1f} Hz for {duration_s:.1f}s")
    print("=" * 80)

    for step in range(num_steps):
        q = env.dof_pos[0]
        dq = env.dof_vel[0]

        obs = torch.cat(
            (
                env.base_lin_vel[0] * obs_scales.lin_vel,
                env.base_ang_vel[0] * obs_scales.ang_vel,
                env.projected_gravity[0],
                commanded_vel
                * torch.tensor(
                    [obs_scales.lin_vel, obs_scales.lin_vel, obs_scales.ang_vel],
                    device=env.device,
                ),
                (q - default_dof_pos) * obs_scales.dof_pos,
                dq * obs_scales.dof_vel,
                last_actions,
            ),
            dim=0,
        )

        action = policy.predict(obs, out_device=env.device)
        last_actions = action.clone()
        env.step(action.unsqueeze(0))

        if step % 100 == 0:
            z = float(env.root_states[0, 2])
            vx = float(env.base_lin_vel[0, 0])
            print(f"step {step:05d} | z={z:.3f} | vx={vx:.3f}")

        if z < 0.15:
            print("❌ Robot collapsed – stopping")
            break

    print("\n✅ Simulation complete.")


# =============================================================================
# Entry Point
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        type=str,
        default=os.path.join(POLICY_DIR, "policy_v1.pt"),
    )
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument(
        "--command", choices=["forward", "stationary"], default="forward"
    )
    args = parser.parse_args()

    run_simulation(args.policy, args.duration, args.command)