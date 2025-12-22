
```
# Go2 Low-Level Control Stack (Simulation + Deployment)

> ⚠️ **Research Sandbox — Use at Your Own Risk**  
> This directory is provided as a **research and experimentation sandbox**.  
> The author does **not take responsibility** for hardware damage, falls, or unsafe operation.  
> Always operate the robot in a **harnessed, open environment**.

---

## 1. Overview

This directory contains a **hardware-aligned low-level control pipeline** for the **Unitree Go2**, including:

- MuJoCo-based simulation
- Reinforcement learning policy inference
- Safe deployment via **joint position PD control**
- CSV-based logging for debugging and gait inspection

This stack is intended for **researchers and engineers** working on sim-to-real locomotion, not for production or consumer use.

---

## 2. Supported Scope & Limitations

### Supported
- ✅ Unitree **Go2 only**
- ✅ MuJoCo simulation
- ✅ Joint position PD control
- ✅ Pretrained locomotion policy (`policy_v1.pt`)
- ✅ CSV logging for pose and gait inspection

### Not Supported
- ❌ Policy training (IsaacLab / IsaacGym not covered here)
- ❌ Torque control on hardware
- ❌ Other robots or morphologies
- ❌ Automatic recovery or fall detection
- ❌ Safety supervisors beyond PD + damping mode

---

## 3. Directory Structure (Relevant Subset)

```

low_level/
├── common/                  # Shared helpers (paths, constants, math)
├── config/
│   ├── sim/v1/              # MuJoCo simulation YAMLs
│   └── deploy/v1/           # Hardware deployment YAMLs
├── simulate/
│   └── mujoco/v1/           # MuJoCo simulation scripts + assets
├── deploy/
│   └── v1/                  # Hardware deployment scripts
├── debug/                   # Policy inspection, mode switching
├── policies/
│   └── policy_v1.pt         # Pretrained locomotion policy
├── plots/
│   ├── sim_op/              # Simulation logs & plots
│   └── real_op/             # Hardware logs (if enabled)

```

---

## 4. Versioning Philosophy

- **v1.0**  
  First working version for a given policy.

- **v1.1**  
  Same control logic as v1.0, **adds CSV logging only**.

- **v2.x (future)**  
  Reserved for **newly trained policies**.

> All `v1.x` scripts are compatible with `policy_v1.pt`.

---

## 5. Policy Contract (Critical)

### 5.1 Policy Architecture

- Type: **MLP (no recurrence)**
- Input dimension: **48**
- Output dimension: **12**
- Output range: **unbounded (no tanh / clamp)**

### 5.2 What the Policy Outputs

The policy outputs a **12-D continuous vector** with torque-like statistics:

- Mean ≈ −0.56  
- Std ≈ 2.05  
- Max |value| ≈ 4.73  

However:

> **Torque is never directly commanded.**

---

### 5.3 Deployment-Time Interpretation

During both simulation and hardware deployment, the policy output is **reinterpreted** as:

```

target_q = default_angles + action * action_scale

```

Where:
- `action` is clipped to `[-1, 1]`
- `action_scale` is defined in YAML
- `default_angles` define the standing posture

The resulting joint targets are tracked using **joint position PD control**.

⚠️ **Important:**  
This interpretation is a **deployment safety choice**, not something the policy was trained for.

---

## 6. Control Architecture (Sim & Real)

```

Observations (48D)
↓
RL Policy (MLP)
↓
Clipping + Scaling
↓
Joint Target Positions
↓
PD Controller
↓
Motor Torques (implicit)

```

- No direct torque commands
- PD loop exists **outside** the policy
- PD gains are fully configurable via YAML

---

## 7. MuJoCo Simulation

### Primary Script
```

simulate/mujoco/v1/sim_walk_v1.0.py

```

### Experimental (Logging Only)
```

simulate/mujoco/v1/sim_walk_v1.1.py

```

### Run

```bash
cd example/go2/low_level/simulate/mujoco/v1
python3 sim_walk_v1.0.py go2_sim_v1.0.yaml
```

### Expected Behavior

* Robot stands symmetrically at default height
* Forward walking gait emerges
* Mild lateral drift to the **right** is expected
* Over long horizons, trajectory may curve rightward

This behavior reflects **policy bias**, not a bug.

---

## 8. Hardware Deployment (⚠️ Safety-Critical)

### Primary Script

```
deploy/v1/dep_walk_v1.0.py
```

### Experimental (Logging Only)

```
deploy/v1/dep_walk_v1.1.py
```

---

### 8.1 Network Interface Selection

Before deployment, identify the correct network interface:

```bash
ifconfig
```

Example (Ethernet-connected Go2):

```text
eno1: inet 192.168.123.222  netmask 255.255.255.0
```

Use the interface name (e.g. `eno1`) when running deployment scripts.

❌ Using the wrong interface will cause DDS initialization failure.

---

### 8.2 Required Robot State (MANDATORY)

> **The robot must be SITTING before deployment.**

The deployment script:

* zeros motor torques
* takes low-level control

🚫 **Never start deployment from a standing robot**
unless the robot is **fully harnessed and lifted**.

---

### 8.3 Run Deployment

```bash
cd example/go2/low_level/deploy/v1
python3 dep_walk_v1.0.py eno1 go2_dep_v1.0.yaml
```

---

### 8.4 Safe Shutdown Procedure

1. **Primary (Recommended)**
   Press **SELECT** on the Unitree controller
   → robot enters damping mode and sits down smoothly

2. **Emergency**

   ```
   CTRL+C
   ```

⚠️ Always:

* keep the robot harnessed
* operate in open space
* stand clear of legs during testing

---

## 9. Logging & Debugging

* v1.1 scripts generate CSV logs
* Used to:

  * inspect pose symmetry
  * analyze gait patterns
  * debug PD–policy interaction
* Logging does **not** change control behavior

---

## 10. Known Sim-to-Real Gaps

Be aware of the following **intentional mismatches**:

* Policy trained **without hardware awareness**
* MuJoCo has:

  * idealized contacts
  * implicit stabilizers
  * symmetric mass distribution
* Hardware requires:

  * higher PD stiffness
  * careful damping
  * compensation for asymmetry

As a result:

* Standing in sim is easier than on hardware
* Policy requires PD + stabilizer assistance to remain upright

Future versions (`v2.x`) will address this via **hardware-informed training**.

---

## 11. Final Notes

* YAML defines **control behavior**
* Python scripts **bind everything together**
* The policy itself is **fixed and opaque**
* This stack prioritizes **safety and inspectability** over performance

### Discarded / Reference Scripts

- `deploy_policy.py`  
  Deprecated and no longer used. Kept only for historical reference.

- `go2_stand_example.py`  
  Original Unitree SDK example. Not part of this control stack.
