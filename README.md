# Franka FR3 controlling


## 1. How to connect robot

See [connect.md](connect.md) for more details


## 2. Installation

To build the Franka ROS2 workspace, follow instructions of Franka ROS 2 repository on branch **Humble**: [https://frankarobotics.github.io/docs/franka_ros2.html](https://frankarobotics.github.io/docs/franka_ros2.html)

> [!TIP]
> Set `-DBUILD_TESTING=OFF` to avoid testing error
> ```bash
> colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
> ```


## 3. (Option 1) Run interactive control using moveit2


1. In the Franka Desk, unlock joints, switch to `execution` mode, then activate `FCI` mode.

2. In the terminal, run:
```bash

cd ~/franka_ros_ws

source install/setup.sh

ros2 launch franka_fr3_moveit_config moveit.launch.py robot_ip:=192.168.1.2

```
3. In rviz2, activate `MotionPlanning`, under this tab (bottom left), first pull the robot motion to a desired position, then press `plan` to plan the motion; then press `execute` to enable actual motion of real franka robot.


## 4. (Option 2) Run control using python script


1. In the Franka Desk, unlock joints, switch to `execution` mode, then activate `FCI` mode.

2. In the terminal 1, run moveit kernel:
```bash

cd ~/franka_ros_ws

source install/setup.sh

ros2 launch franka_fr3_moveit_config moveit.launch.py robot_ip:=192.168.1.2


```

3. In the terminal 2, run the example script controllers:

> [!NOTE]
> There are two main controller implementation in the `src/my_controller/` folder: 
>- **fr3_controller.py**, which use the default ompl for point-to-point motion planning.
>- **fr3_controller_lin.py**, which enable cartesian linear motion planning.

```bash

cd ~/franka_ros_ws

source install/setup.sh

# controller using default ompl planning
python3 src/my_controller/fr3_controller.py 

# or

# controller using cartesian linear path planning
python3 src/my_controller/fr3_controller_lin.py 

```

You can build more complex behaviour on the top of the given example controllers.

## 5. (Debug) Manually move the gripper and arm

1. In the franka desk, at right side panel: 
    - enable `free move` mode (above right); 
    - unlock the joints (middle right);
    - select `Programming` button (below right);
    - disable `FCI` mode (above right `192.168.1.2 N/A` tab)

2. When see the light turns white, press the black buttons on the grippper shoulder then you can freely move the arm by hand.


## 6. TODO: Hand-eye calibration

> [!IMPORTANT]
> TODO: Implement the robot hand-to-eye calibration (based on realsense and Apriltags/CharucoBoard)

- [ ] Debug and improve the calibrate accuracy.
- [x] Implement and optimize the code script in `src/my_conrtoller/`: **camera.py** and **robo_calibrate.py**s.
- [x] Inherite and test code from repo [RoboEXP](https://github.com/Jianghanxiao/RoboEXP/blob/master/roboexp/env/robo_calibrate.py)

