# Reads the current state and then smoothly moves a single one joint
import time
import sys
import unitree_legged_const as go2

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient


class Custom:
    def __init__(self):
        self.low_state = None
        self.Kp = 30.0      # stiffness (affects snap)
        self.Kd = 3.0       # damping (affects smoothness)
        self.dt = 0.002     # 500 Hz
        self.alpha = 0.05   # motion interpolation rate (speed control)

        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.crc = CRC()

    def Init(self):
        self.InitLowCmd()

        # Create publisher and subscriber
        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()

        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateMessageHandler, 10)

        # Release high-level motion control
        self.sc = SportClient()
        self.sc.SetTimeout(5.0)
        self.sc.Init()

        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

        status, result = self.msc.CheckMode()
        while result["name"]:
            self.sc.StandDown()
            self.msc.ReleaseMode()
            status, result = self.msc.CheckMode()
            time.sleep(1)

    def Start(self):
        self.lowCmdWriteThreadPtr = RecurrentThread(
            interval=self.dt, target=self.LowCmdWrite, name="writebasiccmd"
        )
        self.lowCmdWriteThreadPtr.Start()

    def InitLowCmd(self):
        self.low_cmd.head[0] = 0xFE
        self.low_cmd.head[1] = 0xEF
        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0
        for i in range(12):
            self.low_cmd.motor_cmd[i].mode = 0x01
            self.low_cmd.motor_cmd[i].q = go2.PosStopF
            self.low_cmd.motor_cmd[i].dq = go2.VelStopF
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].kd = 0
            self.low_cmd.motor_cmd[i].tau = 0

    def LowStateMessageHandler(self, msg: LowState_):
        self.low_state = msg
        if not hasattr(self, "_print_counter"):
            self._print_counter = 0
        self._print_counter += 1
        if self._print_counter % 50 == 0:
            print("\n--- Joint State ---")
            for i, m in enumerate(msg.motor_state[:12]):
                print(
                    f"Joint {i}: q={m.q:.3f}, dq={m.dq:.3f}, "
                    f"tau={m.tau_est:.3f}, T={m.temperature:.1f}°C"
                )

    def LowCmdWrite(self):
        if self.low_state is None:
            return

        target_joint = 2     # FR_knee
        target_q = 0.5      # realistic knee bend (radians)

        for i in range(12):
            cmd = self.low_cmd.motor_cmd[i]
            cmd.mode = 0x01
            cmd.dq = 0
            cmd.tau = 0

            if i == target_joint:
                current_q = self.low_state.motor_state[i].q
                # gradual interpolation toward target
                cmd.q = current_q + self.alpha * (target_q - current_q)
                cmd.kp = self.Kp
                cmd.kd = self.Kd
            else:
                cmd.q = self.low_state.motor_state[i].q
                cmd.kp = self.Kp
                cmd.kd = self.Kd

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher.Write(self.low_cmd)


if __name__ == "__main__":
    print("WARNING: Robot may move. Ensure it is lifted or clear of obstacles.")
    input("Press Enter to continue...")

    # Initialize DDS
    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    custom = Custom()
    custom.Init()
    custom.Start()

    print("Running low-level control... Press Ctrl+C to stop.")
    while True:
        time.sleep(1)
