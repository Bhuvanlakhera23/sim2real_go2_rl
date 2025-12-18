#!/usr/bin/env python3
"""
Universal TorchScript Policy Parser for Go2

Usage:
    python3 policyparser.py policy_1.pt
    python3 policyparser.py anydrive_v3_lstm.pt

Assumes policies are stored in:
    ../policies/

Goals:
- Auto-detect policy type (MLP vs LSTM)
- Infer input/output dimensions when possible
- Infer action semantics (heuristic)
- Provide correct deployment guidance
- Never crash on unknown policies
- Never provide misleading advice
"""

import torch
import argparse
import os
import sys
import numpy as np

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

POLICY_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../policies")
)

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def load_policy(policy_name: str):
    policy_path = os.path.join(POLICY_DIR, policy_name)
    if not os.path.isfile(policy_path):
        print(f"[ERROR] Policy not found: {policy_path}")
        sys.exit(1)

    model = torch.jit.load(policy_path, map_location="cpu")
    print(f"[INFO] Loaded policy: {policy_path}")
    return model


def is_recurrent(model) -> bool:
    """
    Robust recurrent detection for TorchScript models.
    Works for scripted, traced, and wrapped LSTMs.
    """
    try:
        graph = model.forward.graph
        inputs = list(graph.inputs())
        outputs = list(graph.outputs())

        # More than one input → likely (x, hidden)
        if len(inputs) > 2:
            return True

        # Tuple output → likely (action, hidden)
        if len(outputs) == 1 and "Tuple" in str(outputs[0].type()):
            return True

        # Explicit LSTM ops
        graph_str = str(graph)
        if "aten::lstm" in graph_str or "aten::lstm_cell" in graph_str:
            return True

    except Exception:
        pass

    return False


def infer_mlp_io_dims(model):
    """
    Infer input/output dimensions for MLP-style TorchScript models.
    """
    input_dim = None
    output_dim = None

    for name, param in model.named_parameters():
        if input_dim is None and param.dim() == 2:
            input_dim = param.shape[1]
        if param.dim() == 2:
            output_dim = param.shape[0]

    return input_dim, output_dim


def infer_lstm_dims(model):
    """
    Infer LSTM hidden size and output dim without forward pass.
    """
    hidden_size = None
    output_dim = None

    for name, param in model.named_parameters():
        if "weight_ih_l0" in name:
            hidden_size = param.shape[0] // 4
        if "linear.weight" in name:
            output_dim = param.shape[0]

    return hidden_size, output_dim


def estimate_action_semantics(model):
    """
    Heuristic action semantics estimation.
    Only valid for stateless models.
    """
    try:
        dummy = torch.randn(1, 48)
        out = model(dummy)

        if isinstance(out, tuple):
            return "unknown (recurrent output)", None, None, None

        out = out.detach().cpu().numpy()
        mean = float(out.mean())
        std = float(out.std())
        max_abs = float(np.max(np.abs(out)))

        if max_abs > 3.0:
            return "torque / unscaled action", mean, std, max_abs
        else:
            return "position / bounded action", mean, std, max_abs

    except Exception:
        return "unknown (forward requires state)", None, None, None


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TorchScript Policy Parser")
    parser.add_argument("policy", type=str, help="Policy filename (.pt)")
    args = parser.parse_args()

    model = load_policy(args.policy)

    print("\n================ MODEL SUMMARY =======================")
    print(model)

    recurrent = is_recurrent(model)
    print(f"\nRecurrent Policy: {recurrent}")

    # --------------------------------------------------
    # Stateless (MLP-style) policy
    # --------------------------------------------------
    if not recurrent:
        input_dim, output_dim = infer_mlp_io_dims(model)

        print("\n================ MODEL SHAPE ========================")
        print(f"Inferred Input Dim : {input_dim}")
        print(f"Inferred Output Dim: {output_dim}")

        semantics, mean, std, max_abs = estimate_action_semantics(model)

        print("\n================ OUTPUT STATS ========================")
        if mean is not None:
            print(f"mean    : {mean:.4f}")
            print(f"std     : {std:.4f}")
            print(f"max_abs : {max_abs:.4f}")
        else:
            print("Could not compute (no forward pass)")

        print(f"\nInferred Control Type: {semantics}")

        print("\n================ DEPLOYMENT GUIDANCE ================")
        if "torque" in semantics:
            print("- Action likely represents torque or raw command")
            print("  → Avoid PD position control")
            print("  → kp = kd ≈ 0")
            print("- Start with action_scale ∈ [0.2, 0.4]")
        else:
            print("- Action likely position delta")
            print("  → PD control acceptable")

        print("- Bias correction likely needed for standing")
        print("- Compatible with dep_walk_v1.x scripts")

    # --------------------------------------------------
    # Recurrent (LSTM-style) policy
    # --------------------------------------------------
    else:
        hidden_size, output_dim = infer_lstm_dims(model)

        print("\n================ LSTM DETAILS =======================")
        print(f"Hidden Size : {hidden_size if hidden_size else 'Unknown'}")
        print(f"Output Dim  : {output_dim if output_dim else 'Unknown'}")

        print("\n================ DEPLOYMENT GUIDANCE ================")
        print("- Recurrent (LSTM) policy detected")
        print("- Requires persistent hidden state (h, c)")
        print("- Stateless deployment WILL FAIL")
        print("- Hidden state must be reset on:")
        print("  • stand → walk transition")
        print("  • fall / emergency stop")
        print("- Likely torque-based output")
        print("- Recommended control rate ≥ 100 Hz")
        print("- NOT compatible with dep_walk_v1.x")

    # --------------------------------------------------
    print("\n================ SOURCE CODE =========================")
    try:
        print(model.code)
    except Exception:
        print("No TorchScript source available (likely traced model)")

    print("\n[INFO] Policy inspection complete.")


# ------------------------------------------------------------

if __name__ == "__main__":
    main()
