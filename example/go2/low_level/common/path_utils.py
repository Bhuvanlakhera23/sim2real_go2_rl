"""
path_utils.py

Centralized, deterministic path resolution for the Go2 low_level project.

Rules:
- Pure stdlib only (NO torch, NO isaacgym, NO legged_gym)
- Paths are resolved relative to the calling file, never CWD
- Fail fast if expected directories/files are missing
"""

import os


# -----------------------------------------------------------------------------
# Core helpers
# -----------------------------------------------------------------------------
def _abspath(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def _assert_exists(path: str, kind: str = "path") -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{kind} does not exist: {path}")
    return path


# -----------------------------------------------------------------------------
# Project root
# -----------------------------------------------------------------------------
def get_project_root(start_file: str) -> str:
    """
    Resolve project root from any script under:
    example/go2/low_level/**

    Expected tree:
        low_level/
            common/
            policies/
            simulate/
            deploy/
            config/
    """
    start_file = _abspath(start_file)
    root = _abspath(os.path.join(os.path.dirname(start_file), "../../.."))
    return _assert_exists(root, "Project root")


# -----------------------------------------------------------------------------
# Canonical directories
# -----------------------------------------------------------------------------
def get_common_dir(start_file: str) -> str:
    return _assert_exists(
        os.path.join(get_project_root(start_file), "common"),
        "common dir",
    )


def get_policy_dir(start_file: str) -> str:
    return _assert_exists(
        os.path.join(get_project_root(start_file), "policies"),
        "policy dir",
    )


def get_simulate_dir(start_file: str) -> str:
    return _assert_exists(
        os.path.join(get_project_root(start_file), "simulate"),
        "simulate dir",
    )


def get_deploy_dir(start_file: str) -> str:
    return _assert_exists(
        os.path.join(get_project_root(start_file), "deploy"),
        "deploy dir",
    )


def get_config_dir(start_file: str) -> str:
    return _assert_exists(
        os.path.join(get_project_root(start_file), "config"),
        "config dir",
    )

# -----------------------------------------------------------------------------
# Files (fail fast)
# -----------------------------------------------------------------------------
def get_policy_path(start_file: str, policy_name: str) -> str:
    """
    Example:
        get_policy_path(__file__, "policy_v1.pt")
    """
    path = os.path.join(get_policy_dir(start_file), policy_name)
    return _assert_exists(path, "policy file")


def get_sim_config_path(start_file: str, version: str, name: str) -> str:
    """
    Example:
        get_sim_config_path(__file__, "v1", "go2_sim_v1.1.yaml")
    """
    path = os.path.join(
        get_config_dir(start_file),
        "sim",
        version,
        name,
    )
    return _assert_exists(path, "sim config")


def get_deploy_config_path(start_file: str, version: str, name: str) -> str:
    """
    Example:
        get_deploy_config_path(__file__, "v1", "go2_dep_v1.1.yaml")
    """
    path = os.path.join(
        get_config_dir(start_file),
        "deploy",
        version,
        name,
    )
    return _assert_exists(path, "deploy config")


# -----------------------------------------------------------------------------
# MuJoCo helpers (explicit, no guessing)
# -----------------------------------------------------------------------------
def get_mujoco_dir(start_file: str) -> str:
    return _assert_exists(
        os.path.join(get_simulate_dir(start_file), "mujoco"),
        "mujoco dir",
    )


def get_mujoco_assets_dir(start_file: str) -> str:
    return _assert_exists(
        os.path.join(get_mujoco_dir(start_file), "assets"),
        "mujoco assets dir",
    )


def get_mujoco_scene_path(start_file: str, version: str) -> str:
    """
    Example:
        get_mujoco_scene_path(__file__, "v1")
    """
    path = os.path.join(
        get_mujoco_dir(start_file),
        version,
        "scene.xml",
    )
    return _assert_exists(path, "MuJoCo scene.xml")

# -----------------------------------------------------------------------------
# Output / plots
# -----------------------------------------------------------------------------
def get_sim_plots_dir(start_file: str) -> str:
    """
    Canonical directory for MuJoCo simulation outputs.

    Expected:
        low_level/plots/sim_op/mujoco_op/
    """
    path = os.path.join(
        get_project_root(start_file),
        "plots",
        "sim_op",
        "mujoco_op",
    )
    os.makedirs(path, exist_ok=True)
    return path


# -----------------------------------------------------------------------------
# Debug / sanity
# -----------------------------------------------------------------------------
def print_tree_summary(start_file: str) -> None:
    """
    Useful for debugging path resolution over SSH.
    """
    root = get_project_root(start_file)
    print("[PATH_UTILS]")
    print(f"  project_root : {root}")
    print(f"  common       : {get_common_dir(start_file)}")
    print(f"  policies     : {get_policy_dir(start_file)}")
    print(f"  simulate     : {get_simulate_dir(start_file)}")
    print(f"  deploy       : {get_deploy_dir(start_file)}")
    print(f"  config       : {get_config_dir(start_file)}")
