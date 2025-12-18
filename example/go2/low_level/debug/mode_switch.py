#!/usr/bin/env python3
"""
Release high-level controller and enable low-level DDS control
for Unitree Go2 (so you can run rt/lowcmd, rt/lowstate).

Usage:
  python3 mode_switch.py [iface]
Defaults:
  iface = eno1   (tries this first, then falls back to 'lo' automatically)
"""

import sys
import time
import threading
from contextlib import suppress

# ---- Unitree SDK imports ----
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber, ChannelPublisher
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.idl.unitree_go.msg.dds_._LowState_ import LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_._LowCmd_ import LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_._MotorCmd_ import MotorCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_._BmsCmd_ import BmsCmd_
from unitree_sdk2py.utils.crc import CRC

# ---- Helpers ----
def init_dds(iface: str) -> None:
    print(f"[INIT] ChannelFactoryInitialize domain=0 iface='{iface}'")
    ChannelFactoryInitialize(0, iface)

def make_empty_lowcmd() -> LowCmd_:
    motors = [MotorCmd_(mode=0, q=0.0, dq=0.0, tau=0.0, kp=0.0, kd=0.0, reserve=[0,0,0]) for _ in range(20)]
    bms = BmsCmd_(off=0, reserve=[0,0,0])
    return LowCmd_(
        head=[0xFE,0xEF], level_flag=0xFF, frame_reserve=0,
        sn=[0,0], version=[0,0], bandwidth=0,
        motor_cmd=motors, bms_cmd=bms,
        wireless_remote=[0]*40, led=[0]*12, fan=[0]*2,
        gpio=0, reserve=0, crc=0
    )

def try_release(msc: MotionSwitcherClient) -> None:
    print("[ACTION] Releasing high-level controller...")
    with suppress(Exception):
        # Some builds use a lease. If available, obtain it so ReleaseMode sticks.
        _ = msc.GetLeaseId()
    with suppress(Exception):
        msc.ReleaseMode()
    time.sleep(0.8)

def try_select_low(msc: MotionSwitcherClient) -> None:
    # Not all firmwares expose these names; best-effort.
    candidates = ("lowstate", "low", "rt", "idle", "none")
    with suppress(Exception):
        lease = msc.GetLeaseId()
    for name in candidates:
        try:
            print(f"[ACTION] SelectMode('{name}')")
            msc.SelectMode(name)
            with suppress(Exception):
                # If lease semantics exist, let it settle.
                msc.WaitLeaseApplied(lease)
            time.sleep(0.8)
            _, mode = msc.CheckMode()
            print(f"[INFO] Mode after select: {mode.get('name','')} (ignored OK)")
            # Even if this prints empty, some firmwares still allow low-level.
        except Exception as e:
            print(f"[WARN] SelectMode('{name}') failed: {e}")

def count_lowstate_msgs(duration_s: float = 1.5) -> int:
    """Subscribe to rt/lowstate and count msgs for duration."""
    count = 0
    done = threading.Event()

    def cb(_msg: LowState_):
        nonlocal count
        count += 1

    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(cb, 50)  # up to 50 Hz callback budget; MCU is ~500 Hz but throttling is fine.

    def timer():
        time.sleep(duration_s)
        done.set()

    threading.Thread(target=timer, daemon=True).start()
    while not done.is_set():
        time.sleep(0.05)

    return count

def probe_write_lowcmd(pulse_ms: int = 300) -> bool:
    """Send a short 'no-torque' LowCmd to see if writer succeeds without exceptions."""
    pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    ok = pub.Init()
    if not ok:
        print("[WARN] lowcmd publisher Init() returned False")
    crc = CRC()
    cmd = make_empty_lowcmd()
    # Send safe 'stop' flags on first 12 joints
    for i in range(12):
        m = cmd.motor_cmd[i]
        m.mode = 0x01
        m.q = float(2.146e9)   # PosStopF
        m.dq = float(16000.0)  # VelStopF
        m.kp = 0.0
        m.kd = 0.0
        m.tau = 0.0
    cmd.crc = crc.Crc(cmd)

    t_end = time.time() + (pulse_ms / 1000.0)
    wrote_any = False
    while time.time() < t_end:
        try:
            pub.Write(cmd)
            wrote_any = True
        except Exception as e:
            print(f"[WARN] Write(lowcmd) failed: {e}")
            return False
        time.sleep(0.002)
    return wrote_any

def run_sequence(preferred_iface: str) -> bool:
    # 1) Bring up DDS
    init_dds(preferred_iface)

    # 2) Motion switcher: release, then best-effort select low
    msc = MotionSwitcherClient()
    msc.SetTimeout(5.0)
    msc.Init()

    with suppress(Exception):
        _, mode = msc.CheckMode()
        print(f"[INFO] Current mode: {mode.get('name','')}")

    try_release(msc)
    with suppress(Exception):
        _, mode = msc.CheckMode()
        print(f"[INFO] After release: {mode.get('name','')}")

    # Best-effort attempt to enter a low/rt mode name (may be no-op on some firmwares)
    try_select_low(msc)

    # 3) Verify by reading lowstate
    msg_count = count_lowstate_msgs(1.5)
    print(f"[VERIFY] lowstate messages in 1.5s: {msg_count}")
    if msg_count < 3:
        print("[HINT] Not seeing enough lowstate on this graph. Might not be on the robot's internal DDS.")
        return False

    # 4) Optional: confirm we can write to lowcmd without exception
    wrote = probe_write_lowcmd(250)
    print(f"[VERIFY] lowcmd write: {'ok' if wrote else 'failed'}")
    return True

def main():
    prefer = sys.argv[1] if len(sys.argv) > 1 else "eno1"
    print("\n=== Entering LOWSTATE (DDS) Mode for Go2 ===")

    # First try user iface; if verification fails, try 'lo'
    if run_sequence(prefer):
        print("[READY] Low-level DDS is reachable on", prefer, "— you can run dep_exp.py ✅")
        print("============================================\n")
        return

    print(f"[FALLBACK] Retrying on 'lo' (robot’s internal DDS graph)...")
    # Brief sleep to avoid DDS churn
    time.sleep(0.8)

    if run_sequence("lo"):
        print("[READY] Low-level DDS is reachable on 'lo' — use:  python3 dep_exp.py lo  ✅")
        print("============================================\n")
        return

    print("[ERROR] Could not verify low-level DDS on either iface. Possible causes:")
    print("  - High-level app still owns the lease and ignores ReleaseMode")
    print("  - Firmware doesn’t expose MotionSwitcher or mode names differ")
    print("  - Your PC is not bridged to the robot’s DDS graph (firewall/VLAN)")
    print("  - Try power-cycling the robot, and run this again.")
    print("============================================\n")

if __name__ == "__main__":
    main()
