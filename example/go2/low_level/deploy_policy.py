import time
import sys
import torch
import unitree_legged_const as go2

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient


# ------------------- RL Policy Loader -------------------
class PolicyWrapper:
    def __init__(self, path):
        # Load trained PPO policy (TorchScript or .pt file)
        self.policy = torch.jit.load(path, map_location='cpu')
        self.policy.eval()

    def predict(self, obs):
        with torch.no_grad():
            action = self.policy(obs.unsqueeze(0))  # (1, 48) → (1, 12)
        return action.squeeze(0)


# ------------------- Main Controller -------------------
class Custom:
    def __init__(self, policy_path):
        self.low_state = None
        self.Kp = 30.0
        self.Kd = 10.0
        self.dt = 0.002
        self.action_scale = 0.25

        # Mapping between hardware <-> policy
        # self.to_policy = [3,4,5, 0,1,2, 9,10,11, 6,7,8]
        # self.from_policy = [3,4,5, 0,1,2, 9,10,11, 6,7,8]
        # --- Mapping between sim <-> policy (IDENTITY in sim) ---
        self.to_policy = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
        self.from_policy = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]

        # Load policy
        self.policy = PolicyWrapper(policy_path)

        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.crc = CRC()
        self.last_actions = torch.zeros(12)

        self.default_dof_pos = torch.tensor([
            0.1, 0.8, -1.5,   # FL
            -0.1, 0.8, -1.5,  # FR
            0.1, 1.0, -1.5,   # RL
            -0.1, 1.0, -1.5   # RR
        ], dtype=torch.float)

    def Init(self):
        self.InitLowCmd()
        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateMessageHandler, 10)

        # release mode
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

    # -------- receive state --------
    def LowStateMessageHandler(self, msg: LowState_):
        self.low_state = msg

    # -------- send command --------
    def LowCmdWrite(self):
        if self.low_state is None:
            return

        # === Extract state ===
        q_hw = torch.tensor([m.q for m in self.low_state.motor_state[:12]], dtype=torch.float)
        dq_hw = torch.tensor([m.dq for m in self.low_state.motor_state[:12]], dtype=torch.float)
        quat = torch.tensor(self.low_state.imu_state.quaternion, dtype=torch.float)
        gyro = torch.tensor(self.low_state.imu_state.gyroscope, dtype=torch.float)
        acc = torch.tensor(self.low_state.imu_state.accelerometer, dtype=torch.float)

        # === Compute projected gravity ===
        # Rotate gravity vector into body frame
        # Assuming quaternion format [w,x,y,z]
        w, x, y, z = quat
        g_world = torch.tensor([0, 0, -1],dtype=torch.float)
        rot_mat = torch.tensor([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z),     2*(x*z + w*y)],
            [2*(x*y + w*z),     1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x**2 + y**2)]
        ])
        projected_gravity = torch.matmul(rot_mat.T, g_world)

        # === Reorder to policy order ===
        q = q_hw[self.to_policy]
        dq = dq_hw[self.to_policy]

        # === Build observation ===
        base_lin_vel = acc * 0.0   # placeholder, use IMU integration or EKF
        base_ang_vel = gyro
        commands = torch.tensor([2.0, 0.0, 0.0])  # can later come from joystick

        obs = torch.cat((
            base_lin_vel * 2.0,
            base_ang_vel * 0.25,
            projected_gravity,
            commands,
            (q - self.default_dof_pos),
            dq * 0.05,
            self.last_actions[self.to_policy]
        ), dim=0)

        # === Policy inference ===
        policy_action = self.policy.predict(obs)
        self.last_actions[self.to_policy] = policy_action.clone()

        # === Map back to hardware ===
        action_hw = policy_action[self.from_policy]
        q_target = self.default_dof_pos + action_hw * self.action_scale


        # === Send commands ===
        for i in range(12):
            cmd = self.low_cmd.motor_cmd[i]
            cmd.mode = 0x01
            cmd.q = float(q_target[i])
            cmd.dq = 0.0
            cmd.kp = self.Kp
            cmd.kd = self.Kd
            cmd.tau = 0.0

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher.Write(self.low_cmd)

    def Start(self):
        self.lowCmdWriteThreadPtr = RecurrentThread(
            interval=self.dt, target=self.LowCmdWrite, name="policy_write_thread"
        )
        self.lowCmdWriteThreadPtr.Start()


if __name__ == "__main__":
    print("WARNING: Robot may move! Ensure it is lifted or in safe position.")
    input("Press Enter to continue...")

    if len(sys.argv) > 2:
        ChannelFactoryInitialize(0, sys.argv[2])
    else:
        ChannelFactoryInitialize(0)

    policy_path = "policy.pt"  # path to your trained TorchScript policy
    custom = Custom(policy_path)
    custom.Init()
    custom.Start()

    print("Running policy control loop... Press Ctrl+C to stop.")
    while True:
        time.sleep(1)
