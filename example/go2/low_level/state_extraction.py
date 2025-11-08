# Extracts all the state values and then prints it
import sys
import time
import unitree_legged_const as go2
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

class Custom:
    def __init__(self):
        self.low_state = None
    
    def Init(self):
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateMessageHandler, 10)

    def LowStateMessageHandler(self, msg: LowState_):
        print("\n--- Joint State ---")
        for i, m in enumerate(msg.motor_state[:12]):
            print(f"Joint {i}: q={m.q:.3f}, dq={m.dq:.3f}, tau={m.tauEst:.3f}, T={m.temperature:.1f}°C")

        print(f"IMU (rpy): {msg.imu_state.rpy}")
        print(f"Battery: {msg.power_v:.2f} V, {msg.power_a:.2f} A")


if __name__ == '__main__':
    # Initialize communication (specify interface if needed)
    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    custom = Custom()
    custom.Init()

    print("Listening for LowState messages... Press Ctrl+C to stop.")
    while True:
        time.sleep(1)
