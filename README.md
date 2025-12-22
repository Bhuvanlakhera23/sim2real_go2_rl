# Unitree Go2 – ROS2 + SDK2 Integration Guide
This documentation walks through the complete setup for controlling a Unitree robot using ROS 2, Unitree SDK2 (Python), and Isaac Gym. It covers installation, communication setup, low- and high-level control, digital-twin simulation, and wireless operation.

➡️ **Jump to the simulation & deployment guide:**  
[`example/go2/low_level/docs/README.md`](example/go2/low_level/docs/README.md)

---

## 1. Add the Repository Signing Key (Required for go2 robots which have legacy systems) (This step is to be only done only once with the robot)
```bash
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys F42ED6FBAB17C654
```

## 2. ROS 2 Installation and Setup

### Overview
- **PC:** ROS 2 Humble (22.04)  
- **Robot:** ROS 2 Foxy (20.04)

### Install ROS 2 Humble (on PC) (Do this step only if ROS 2 Humble is not there in your PC)
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install software-properties-common -y
sudo add-apt-repository universe
sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
| sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt install ros-humble-desktop -y
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### Install development tools and DDS support
```bash
sudo apt install python3-rosdep python3-colcon-common-extensions python3-argcomplete \
libcyclonedds0 ros-humble-rmw-cyclonedds-cpp -y
sudo rosdep init
rosdep update
```

### Install ROS 2 Foxy (on Robot) (Do this step only if ROS 2 Foxy is not there on your Robot)
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install software-properties-common curl -y
sudo add-apt-repository universe
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
| sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt install ros-foxy-desktop -y
echo "source /opt/ros/foxy/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### Test communication
```bash
ros2 topic list
```

## 3. Connect the Robot Wirelessly (This step has to be done each time you connect the robot to a new network)

### Configure wifi on the robot
```bash
sudo nano /etc/wpa_supplicant/wpa_supplicant.conf
```
Add the following (replace SSID and password as needed)
```bash
ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1
country=IN

network={
    ssid="Wifi Name"
    psk="Password"
}
```

### Reset connection to `wlan0`

Run the following sequence to restart and connect cleanly
```bash
sudo pkill wpa_supplicant
sudo rm -rf /var/run/wpa_supplicant/wlan0
sudo ip link set wlan0 down
sudo ip link set wlan0 up
sudo wpa_supplicant -B -i wlan0 -c /etc/wpa_supplicant/wpa_supplicant.conf
sudo dhclient -v wlan0
```

### Verify connection 

Check that you have a valid IP and internet access
```bash
ip a show wlan0
ping -c 3 8.8.8.8
```
Now you should be able to connect to the robot wirelessly

### SSH into the robot wirelessly
```bash
ssh unitree@10.40.53.233
```

## 4. Fork and Install the Custom SDK

Fork this unitree SDK onto your github account

### Dependencies

- Python ≥ 3.8  
- `cyclonedds == 0.10.2`  
- `numpy`, `opencv-python`

### Installation

```bash
cd ~
sudo apt install python3-pip
git clone [https://github.com/Project-Astrium/unitree_python_sdk.git](https://github.com/Project-Astrium/unitree_python_sdk.git)
cd unitree_sdk2_python
python -m venv .venv
source .venv/bin/activate
pip3 install -e .
```

### Cyclonedds installation 

This step is to be followed if you get an error saying could not locate cyclonedds. If you already have a previous CycloneDDS installation or build, uninstall it and remove the old directory before compiling again.

```bash
pip uninstall cyclonedds -y
rm -rf ~/cyclonedds
```

Then install the prerequisites
```bash
sudo apt update
sudo apt upgrade -y
sudo apt install -y git cmake build-essential python3-pip python3-dev libssl-dev
```

Install cyclonedds
```bash
cd ~
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
cd cyclonedds
mkdir build
mkdir install
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install
cmake --build . --target install -j$(nproc)
```

Export the required paths
```bash
export CYCLONEDDS_HOME=~/cyclonedds/install
export CMAKE_PREFIX_PATH=$CYCLONEDDS_HOME:$CMAKE_PREFIX_PATH
```

Install the sdk
```bash
cd ~/unitree_sdk2_python
pip3 install -e .
```





