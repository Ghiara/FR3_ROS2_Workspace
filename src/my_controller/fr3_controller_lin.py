#!/usr/bin/env python3
"""
MoveIt2 Action Client Controller for Franka FR3
This implementation uses ROS2 action clients instead of moveit_py for better compatibility.

This implementation uses cartesian linear motion planner for motion planning.

Key functions:
- move_to:                  move to a pose (by default, use cartesian linear motion planner)
- move_to_cartesian:        move to a cartesian pose
- move_to_ompl:             move to a ompl pose
- rotate:                   rotate the last joint
- rotate_to_home:           rotate the last joint to home position
- set_gripper_action:       set the gripper to a specific opening width (0.0-0.08)
- grasp_object:             grasp an object (similar to set_gripper_action, but with force and tolerance)
- open_gripper:             open the gripper (default width: 0.08)
- close_gripper:            close the gripper (default width: 0.0)
- reset:                    reset the robot and gripper to home position
- reset_gripper:            reset the gripper to home position
"""
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

import numpy as np
from math import pi, cos, sin, atan2, copysign, asin
import time
from copy import deepcopy
# from typing import List, Optional

# ROS2 messages
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
import rclpy.time
from std_msgs.msg import Header
from sensor_msgs.msg import JointState

# Franka gripper actions
from franka_msgs.action import Grasp, Move, Homing
# from std_srvs.srv import Trigger

# MoveIt2 action interfaces
from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from moveit_msgs.msg import (
    MotionPlanRequest, 
    PlanningOptions,
    Constraints,
    PositionConstraint,
    OrientationConstraint,
    WorkspaceParameters,
    RobotState,
    MoveItErrorCodes,
    JointConstraint
)

from shape_msgs.msg import SolidPrimitive
from moveit_msgs.srv import GetPositionIK, GetCartesianPath

from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException


def _quat_to_rpy(x: float, y: float, z: float, w: float):
    """Convert quaternion → roll-pitch-yaw (radians)."""
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = copysign(pi / 2.0, sinp)   # use 90 deg if out of range
    else:
        pitch = asin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def _apply_offset(position, orientation, offset):
    """
    Given a geometry_msgs Point + Quaternion and an offset expressed in the
    EEF frame, return the world-frame Point after applying the offset.

    position   : geometry_msgs.msg.Vector3/Point
    orientation: geometry_msgs.msg.Quaternion
    offset     : np.ndarray shape (3,)  # in metres
    """
    # quaternion → rotation matrix
    qx, qy, qz, qw = orientation.x, orientation.y, orientation.z, orientation.w
    norm = (qx*qx + qy*qy + qz*qz + qw*qw) ** 0.5 or 1.0
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    # Rodriguez-style 3×3 rotation matrix
    R = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx*qx + qy*qy)]
    ])

    # --- transform the local offset into the world frame ---------------------
    world_offset = R @ offset.reshape(3,)
    

    return (
        position.x + world_offset[0],
        position.y + world_offset[1],
        position.z + world_offset[2],
    )


class FR3Controller(Node):
    """
    MoveIt2 controller using action clients for Franka FR3 robot
    Compatible with standard ROS2 installations without moveit_py
    """
    
    def __init__(self, init_state:list=[0.4, 0.0, 0.5, pi, 0.0, 0.0], offset:np.ndarray=None,):
        super().__init__('moveit_action_controller')
        
        # Robot configuration
        self.planning_group = "fr3_arm"  # Adjust based on your SRDF
        self.reference_frame = "fr3_link0"
        self.end_effector_link = "fr3_hand"
        # # offset from gripper tip frame to link7 frame, by default franka gripper installed
        self.offset = offset if offset is not None else np.array([0.0, 0.0, -0.095]) 
        self.init_state = init_state
        self.current_joint_state = None
        self.current_pose = None
        self.current_gripper_joint_state = None
        self.gripper_is_initialized = False
        self.use_cartesian_lin_control = True
        self.fail_count = 0
        self.joint7_home_angle = None # 45.42685 degree in z-axis
            

        # Callback groups for concurrent operations
        self.callback_group = ReentrantCallbackGroup()

        # --- TF listener for real-time EEF pose ---------------------------------
        self.tf_buffer = Buffer(cache_time=rclpy.duration.Duration(seconds=10))
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

        # 50 Hz timer to refresh self.current_pose
        self.pose_timer = self.create_timer(0.02, self._update_eef_pose,
                                            callback_group=self.callback_group)

        
        # Action client for MoveGroup
        self.move_group_client = ActionClient(
            self,
            MoveGroup,
            '/move_action',
            callback_group=self.callback_group
        )

        # Action client for ExecuteTrajectory
        self.execute_trajectory_client = ActionClient(
            self,
            ExecuteTrajectory,
            '/execute_trajectory',
            callback_group=self.callback_group
        )

        
        
        # Service client for inverse kinematics
        self.ik_client = self.create_client(
            GetPositionIK,
            '/compute_ik',
            callback_group=self.callback_group
        )

        # Service client for cartesian path planning
        self.cartesian_path_client = self.create_client(
            GetCartesianPath,
            '/compute_cartesian_path',
            callback_group=self.callback_group
        )
        
        # Joint state subscriber
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10,
            callback_group=self.callback_group
        )

        # ==================== GRIPPER CONTROL ====================
        # Multiple gripper action clients for different functionalities
        
        
        # Gripper grasp action client (available with real hardware)
        self.gripper_grasp_client = ActionClient(
            self,
            Grasp,
            '/franka_gripper/grasp',
            callback_group=self.callback_group
        )
        
        # Gripper move action client (available with real hardware)
        self.gripper_move_client = ActionClient(
            self,
            Move,
            '/franka_gripper/move',
            callback_group=self.callback_group
        )
        
        # Gripper homing action client (available with real hardware)
        self.gripper_homing_client = ActionClient(
            self,
            Homing,
            '/franka_gripper/homing',
            callback_group=self.callback_group
        )
        

        # Gripper joint state subscriber
        self.gripper_joint_state_sub = self.create_subscription(
            JointState,
            '/franka_gripper/joint_states',
            self.gripper_joint_state_callback,
            10,
            callback_group=self.callback_group
        )

        
        # Wait for services
        self.get_logger().info("Waiting for MoveIt2 services...")
        
        if not self.move_group_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("MoveGroup action server not available!")
            raise RuntimeError("MoveGroup action server not available")
        
        self.get_logger().info("✓ MoveGroup action server connected")
        
        if not self.ik_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn("IK service not available - some features may be limited")
        else:
            self.get_logger().info("✓ IK service connected")
        
        if not self.cartesian_path_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn("Cartesian path service not available - using standard planning")
        else:
            self.get_logger().info("✓ Cartesian path service connected")    
        
        # Wait for gripper services
        self.get_logger().info("Waiting for gripper services...")
        gripper_connected = 0
        
        if self.gripper_move_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().info("✓ Gripper move action server connected")
            gripper_connected += 1
        else:
            self.get_logger().warn("Gripper move action server not available")
            
        if self.gripper_homing_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().info("✓ Gripper homing action server connected")
            gripper_connected += 1
        else:
            self.get_logger().warn("Gripper homing action server not available")

            
        if gripper_connected > 0:
            self.get_logger().info(f"✓ {gripper_connected}/2 gripper services connected")
        else:
            self.get_logger().warn("No gripper services available - gripper control disabled")
        

        self.get_logger().info("MoveIt Action Controller initialized successfully!")

    
    def joint_state_callback(self, msg: JointState):
        """Update current joint state"""
        self.current_joint_state = msg

    def gripper_joint_state_callback(self, msg: JointState):
        """Update current gripper joint state"""
        self.current_gripper_joint_state = msg

    # def get_current_pose(self):
    #     """
    #     Return the latest end-effector pose as [x, y, z, roll, pitch, yaw].

    #     Returns
    #     -------
    #     list[float] or None
    #         Pose in metres / radians, or None if no pose is available yet.
    #     """
    #     if self.current_pose is None:
    #         return None
    #     else:
    #         return self.current_pose

    #     # pos = self.current_pose.position
    #     # ori = self.current_pose.orientation
    #     # roll, pitch, yaw = _quat_to_rpy(ori.x, ori.y, ori.z, ori.w)
    #     # return [pos.x, pos.y, pos.z, roll, pitch, yaw]
    
    def _update_eef_pose(self):
        """Continuously update end-effector pose from TF."""
        try:
            if self.tf_buffer.can_transform(
                'fr3_link0', 'fr3_hand', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05)
            ):
                # most recent transform: reference_frame ➜ end_effector_link
                trans = self.tf_buffer.lookup_transform(
                    self.reference_frame,          # e.g. 'fr3_link0'
                    self.end_effector_link,        # e.g. 'fr3_hand'
                    rclpy.time.Time(),             # “latest” transform
                    rclpy.duration.Duration(seconds=0.05)
                )

                # Convert geometry_msgs/TransformStamped → geometry_msgs/Pose
                pose = Pose()
                pose.position.x = trans.transform.translation.x
                pose.position.y = trans.transform.translation.y
                pose.position.z = trans.transform.translation.z
                pose.orientation = trans.transform.rotation

                # ➜ apply offset so pose reflects the TCP
                x, y, z = _apply_offset(pose.position, pose.orientation, -self.offset)
                pose.position.x, pose.position.y, pose.position.z = x, y, z

                self.current_pose = pose
        except (LookupException, ConnectivityException, ExtrapolationException):
            # Silently ignore; nothing is published yet or tree is split
            return

    # ==================== GRIPPER CONTROL METHODS ==================== 
    
    def reset_gripper(self) -> bool:
        """Initialize/home the gripper"""
        if not self.gripper_homing_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Gripper homing action server not available')
            return False
        
        goal = Homing.Goal()
        self.get_logger().info("Homing gripper...")
        
        future = self.gripper_homing_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Gripper homing goal was rejected!")
            return False
        
        # Wait for result
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        
        result = result_future.result()
        success = result.result.success
        
        if success:
            self.get_logger().info("✓ Gripper homed successfully")
            self.gripper_is_initialized = True
            return True
        else:
            self.get_logger().error("✗ Gripper homing failed")
            return False
              
    def open_gripper(self, width: float = 0.08, speed: float = 0.08) -> bool:
        """Open gripper to specified width
        
        Args:
            width: Target width in meters (default: 0.08m - fully open)
            speed: Opening speed in m/s (default: 0.08 m/s)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.gripper_move_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Gripper move action server not available')
            return False
        
        goal = Move.Goal()
        goal.width = width
        goal.speed = speed
        
        # self.get_logger().info(f"Opening gripper to width: {width:.4f}m at speed: {speed:.4f}m/s")
        
        future = self.gripper_move_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Gripper move goal was rejected!")
            return False
        
        # Wait for result
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        
        result = result_future.result()
        success = result.result.success
        
            
        return success
    
    def close_gripper(self, width: float = 0.0, speed: float = 0.1) -> bool:
        """Close gripper to specified width
        
        Args:
            width: Target width in meters (default: 0.0m - fully closed)
            speed: Closing speed in m/s (not used with GripperCommand, kept for compatibility)
            
        Returns:
            bool: True if successful, False otherwise
        """
        return self.open_gripper(width, speed)
    
    def set_gripper_action(self, grip_action: float) -> bool:
        """Set the gripper action command using normalized control signal.
        
        Args:
            grip_action: A float signal for action control between 0 and 1, where:
                - 0.0: Fully closed
                - 1.0: Fully open
                
        Returns:
            bool: True if successful, False otherwise
        """
        # Clip the grip_action value between 0 and 1
        grip_action = max(0.0, min(grip_action, 1.0))
        
        # Map grip_action to gripper width
        min_width = 0.0    # Minimum width (fully closed)
        max_width = 0.08   # Maximum width (fully open)
        target_width = min_width + (max_width - min_width) * grip_action
        
        # Use Franka move action
        return self.open_gripper(target_width, speed=0.1)
    
    def grasp_object(self, width: float = 0.02, speed: float = 0.05, force: float = 100.0, 
                    inner_tolerance: float = 0.005, outer_tolerance: float = 0.01) -> bool:
        """Grasp an object with specified parameters
        
        Args:
            width: Target grasp width in meters (default: 0.02m)
            speed: Grasping speed in m/s (default: 0.05 m/s)
            force: Grasping force in N (default: 50.0 N)
            inner_tolerance: Inner tolerance for grasp success in meters (default: 0.005m)
            outer_tolerance: Outer tolerance for grasp success in meters (default: 0.01m)
            
        Returns:
            bool: True if successful grasp, False otherwise
        """
        if not self.gripper_grasp_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Gripper grasp action server not available')
            return False
        
        goal = Grasp.Goal()
        goal.width = width
        goal.speed = speed
        goal.force = force
        goal.epsilon.inner = inner_tolerance
        goal.epsilon.outer = outer_tolerance
        
        self.get_logger().info(f"Grasping object: width={width:.4f}m, speed={speed:.4f}m/s, "
                              f"force={force:.1f}N, tolerances=[{inner_tolerance:.4f}, {outer_tolerance:.4f}]")
        
        future = self.gripper_grasp_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Gripper grasp goal was rejected!")
            return False
        
        # Wait for result
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        
        result = result_future.result()
        success = result.result.success
        
        if success:
            self.get_logger().info("✓ Object grasped successfully")
        else:
            self.get_logger().error("✗ Object grasping failed")
            
        return success
           
    def get_gripper_state(self) -> dict:
        """Get current gripper state
        
        Returns:
            dict: Dictionary containing gripper state information
        """
        if self.current_gripper_joint_state is None:
            return {
                'available': False,
                'width': None,
                'position': None,
                'joint_names': None
            }
        
        # Calculate gripper width from joint positions
        # For franka gripper, width = 2 * finger_joint_position
        if len(self.current_gripper_joint_state.position) >= 2:
            width = 2 * self.current_gripper_joint_state.position[0]  # Assuming symmetric fingers
        else:
            width = None
        
        return {
            'available': True,
            'width': width,
            'position': list(self.current_gripper_joint_state.position),
            'joint_names': list(self.current_gripper_joint_state.name)
        }
    
    def print_gripper_state(self):
        """Print current gripper state"""
        state = self.get_gripper_state()
        
        if not state['available']:
            self.get_logger().info("Gripper state not available")
            return
        
        self.get_logger().info(f"Gripper State:")
        self.get_logger().info(f"  Width: {state['width']:.4f}m" if state['width'] is not None else "  Width: Unknown")
        self.get_logger().info(f"  Joint Names: {state['joint_names']}")
        self.get_logger().info(f"  Joint Positions: {[f'{pos:.4f}' for pos in state['position']]}")

    # ==================== PLANNING METHODS ====================

    def reset(self, execute:bool=True):
        """reset to init state"""
        self.get_logger().info(f"Reset to init state")
        
        self.move_to(self.init_state[0], self.init_state[1], self.init_state[2], 
                                    self.init_state[3], self.init_state[4], self.init_state[5], execute)
        self.reset_gripper()

        idx = self.current_joint_state.name.index("fr3_joint7")
        self.joint7_home_angle = self.current_joint_state.position[idx] # rad
        self.get_logger().info(f"Robot Initialized.")

    def move_to(self, x: float, y: float, z: float, 
                         roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0,
                         execute: bool = True):
        if self.use_cartesian_lin_control:
            self.move_to_cartesian(x, y, z, roll, pitch, yaw, execute)
        else:
            self.move_to_ompl(x, y, z, roll, pitch, yaw, execute)

    def move_to_cartesian(self, x: float, y: float, z: float, 
                         roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0,
                         execute: bool = True, max_step: float = 0.005) -> bool:
        """
        Move to target pose using cartesian path planning for guaranteed straight line motion
        
        Args:
            x, y, z: Target position in meters
            roll, pitch, yaw: Target orientation in radians
            execute: Whether to execute the motion or just plan
            max_step: Maximum distance between path points (smaller = smoother)
            
        Returns:
            bool: True if successful, False otherwise
        """

        # Wait for current joint state
        timeout = 10.0
        start_time = time.time()
        while self.current_joint_state is None:
            if time.time() - start_time > timeout:
                self.get_logger().error("Timeout waiting for joint state")
                return False
            self.get_logger().info("Waiting for joint state...")
            time.sleep(0.1)
        

        # Create target pose
        target_pose = self._create_pose_target(x, y, z, roll, pitch, yaw)
        
        # Create cartesian path request
        if not self.cartesian_path_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("Cartesian path service not available")
            # Fallback to regular planning
            return self.move_to_ompl(x, y, z, roll, pitch, yaw, execute)
        

        self.get_logger().info(f"Planning cartesian path to pose: [{x:.3f}, {y:.3f}, {z:.3f}] "
                        f"orientation: [{roll:.3f}, {pitch:.3f}, {yaw:.3f}]")
        xs,ys,zs = _apply_offset(target_pose.position, target_pose.orientation, self.offset)
        target_pose.position.x, target_pose.position.y, target_pose.position.z = xs, ys, zs
        # assert self._is_pose_reachable(x, y, z)

        # Create waypoints list (just start and end for now)
        waypoints = [target_pose]

        request = GetCartesianPath.Request()
        request.header.frame_id = self.reference_frame
        request.start_state.joint_state = self.current_joint_state
        request.group_name = self.planning_group
        request.link_name = self.end_effector_link
        request.waypoints = waypoints
        request.max_step = max_step
        request.jump_threshold = 0.0  # Disable jump detection
        request.avoid_collisions = True
        
        
        # Call cartesian path service
        future = self.cartesian_path_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        response = future.result()
        
        if response.fraction < 0.9:  # Less than 90% of path achieved
            self.get_logger().warn(f"Cartesian path only {response.fraction*100:.1f}% feasible")
            # Fallback to regular planning
            self.get_logger().info("Falling back to regular motion planning...")
            return self.move_to_ompl(x, y, z, roll, pitch, yaw, execute)
        
        if not execute:
            self.get_logger().info(f"✓ Cartesian path planned successfully ({response.fraction*100:.1f}% feasible)")
            return True
        
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = response.solution
        
        self.get_logger().info("Executing cartesian trajectory...")
        
        # Send goal and wait for result
        future = self.execute_trajectory_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Cartesian trajectory execution was rejected!")
            return False
        
        # Wait for execution result
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        
        result = result_future.result()
        error_code = result.result.error_code.val
        
        if error_code == MoveItErrorCodes.SUCCESS:
            self.get_logger().info("✓ Cartesian motion completed successfully!")
            return True
        else:
            self.fail_count += 1
            self.get_logger().error(f"✗ Cartesian motion failed with error code: {error_code}")
            self.get_logger().info("Try again")
            if self.fail_count >= 4:
                return False
            self.move_to(x,y,z,roll,pitch,yaw)
            
            # return False

    def move_to_ompl(self, x: float, y: float, z: float, 
                    roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0,
                    execute: bool = True) -> bool:
        """
        Synchronously move to target pose using ompl planning
        
        Args:
            x, y, z: Target position in meters
            roll, pitch, yaw: Target orientation in radians
            execute: Whether to execute the motion or just plan
            
        Returns:
            bool: True if successful, False otherwise
        """
        assert self._is_pose_reachable(x,y,z)

        # Wait for current joint state
        timeout = 10.0
        start_time = time.time()
        while self.current_joint_state is None:
            if time.time() - start_time > timeout:
                self.get_logger().error("Timeout waiting for joint state")
                return False
            self.get_logger().info("Waiting for joint state...")
            time.sleep(0.1)
        
        # Create target pose
        target_pose = self._create_pose_target(x, y, z, roll, pitch, yaw)
        
        # Create motion plan request
        motion_request = self._create_motion_plan_request(target_pose)
        
        # Create MoveGroup goal
        goal = MoveGroup.Goal()
        goal.request = motion_request
        goal.planning_options = PlanningOptions()
        goal.planning_options.plan_only = not execute
        goal.planning_options.look_around = False
        goal.planning_options.look_around_attempts = 0
        goal.planning_options.max_safe_execution_cost = 1.0
        goal.planning_options.replan = False
        goal.planning_options.replan_attempts = 0
        
        self.get_logger().info(f"Planning motion to pose: [{x:.3f}, {y:.3f}, {z:.3f}] "
                              f"orientation: [{roll:.3f}, {pitch:.3f}, {yaw:.3f}]")
        
        # Send goal and wait for result
        future = self.move_group_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Goal was rejected!")
            return False
        
        self.get_logger().info("Goal accepted, waiting for result...")
        
        # Wait for execution result
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        
        result = result_future.result()
        error_code = result.result.error_code.val
        
        if error_code == MoveItErrorCodes.SUCCESS:
            self.get_logger().info("✓ Motion completed successfully!")

            return True
        else:
            # self.get_logger().error(f"✗ Motion failed with error code: {error_code}")
            return False

    def _create_pose_target(self, x: float, y: float, z: float, 
                          roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0) -> Pose:
        """Create target pose from position and orientation"""
        pose = Pose()
        pose.position = Point(x=x, y=y, z=z)
        
        # Convert RPY to quaternion
        cy = cos(yaw * 0.5)
        sy = sin(yaw * 0.5)
        cp = cos(pitch * 0.5)
        sp = sin(pitch * 0.5)
        cr = cos(roll * 0.5)
        sr = sin(roll * 0.5)
        
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        
        pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
        return pose
    
    def _create_motion_plan_request(self, target_pose: Pose) -> MotionPlanRequest:
        """Create motion planning request"""
        request = MotionPlanRequest()
        
        # Set planning group
        request.group_name = self.planning_group
        

        
        # Set workspace parameters
        request.workspace_parameters = WorkspaceParameters()
        request.workspace_parameters.header.frame_id = self.reference_frame
        request.workspace_parameters.min_corner.x = -2.0 
        request.workspace_parameters.min_corner.y = -2.0
        request.workspace_parameters.min_corner.z = -2.0
        request.workspace_parameters.max_corner.x = 2.0
        request.workspace_parameters.max_corner.y = 2.0
        request.workspace_parameters.max_corner.z = 2.0
        
        # Set current robot state
        if self.current_joint_state:
            request.start_state.joint_state = self.current_joint_state
        
        # Create goal constraints
        constraints = Constraints()
        
        # Position constraint
        pos_constraint = PositionConstraint()
        pos_constraint.header.frame_id = self.reference_frame
        pos_constraint.link_name = self.end_effector_link
        pos_constraint.target_point_offset.x = self.offset[0]
        pos_constraint.target_point_offset.y = self.offset[1]
        pos_constraint.target_point_offset.z = self.offset[2]
        # Create constraint region (small box around target)
        pos_constraint.constraint_region.primitives = [SolidPrimitive()]
        pos_constraint.constraint_region.primitives[0].type = SolidPrimitive.BOX
        pos_constraint.constraint_region.primitives[0].dimensions = [0.01, 0.01, 0.01]
        
        # Set constraint pose
        pos_constraint.constraint_region.primitive_poses = [target_pose]
        pos_constraint.weight = 1.0
        
        # Orientation constraint
        orient_constraint = OrientationConstraint()
        orient_constraint.header.frame_id = self.reference_frame
        orient_constraint.link_name = self.end_effector_link
        orient_constraint.orientation = target_pose.orientation
        orient_constraint.absolute_x_axis_tolerance = 0.02
        orient_constraint.absolute_y_axis_tolerance = 0.02
        orient_constraint.absolute_z_axis_tolerance = 0.02
        orient_constraint.weight = 1.0
        
        # Add constraints to goal
        constraints.position_constraints = [pos_constraint]
        constraints.orientation_constraints = [orient_constraint]
        request.goal_constraints = [constraints]
        
        # Planning parameters
        request.num_planning_attempts = 10
        request.allowed_planning_time = 10.0
        request.max_velocity_scaling_factor = 0.1
        request.max_acceleration_scaling_factor = 0.1
        
        return request

    def rotate_to_home(self, execute: bool = True) -> bool:
        """
        Return arm joint‑7 to the angle captured during the last `reset()`.

        execute  If False, only plan the motion.
        """
        if self.joint7_home_angle is None:
            self.get_logger().error("Home angle for joint‑7 is not set – call reset() first.")
            return False
        self.get_logger().info('rotate to home.')
        return self.rotate(
            self.joint7_home_angle,
            execute=execute,
            relative=False,          # absolute move back to recorded angle
        )

    def rotate(
        self,
        z: float,
        execute: bool = True,
        relative: bool = True,
        abs_tol_fixed: float = 0.002,   # 0.11° tolerance for locked joints
        abs_tol_j7:   float = 0.002,   # 0.11° tolerance for joint‑7 target
        ) -> bool:
        """
        Rotate FR3 joint‑7 (name: 'fr3_joint7').
        NOTE: This rotation could cause pose error accumulation.

        Parameters
        ----------
        z         Rotation in **radians**.
                  • relative=True  –  treat `z` as an **increment** (Δ).
                  • relative=False –  treat `z` as an **absolute** target.
                  In both cases the final angle is clamped to ± 170°.
        execute   If False, only plan.
        relative  See above.
        abs_tol_fixed  Absolute tolerance [rad] for joints 1‑6 (how much they
                       are allowed to drift – set very small to ‘freeze’ them).
        abs_tol_j7     Tolerance for the joint‑7 target.
        """
        # ── 1. wait for a JointState ─────────────────────────────────────────
        t0 = time.time()
        while self.current_joint_state is None and time.time() - t0 < 10.0:
            time.sleep(0.05)
        if self.current_joint_state is None:
            self.get_logger().error("No JointState available")
            return False
        start_js = self.current_joint_state

        # locate joint‑7 index
        try:
            j7_idx = start_js.name.index("fr3_joint7")
        except ValueError:
            self.get_logger().error("'fr3_joint7' not found in JointState")
            return False

        # ── 2. compute target angle ─────────────────────────────────────────
        current_j7 = start_js.position[j7_idx]
        target_j7  = (current_j7 + z) if relative else z
        # target_j7  = float(np.clip(target_j7, -np.pi / 2, np.pi / 2))
        max_angle = np.deg2rad(170)            # ±173 °, for safety limited to 170
        target_j7  = float(np.clip(target_j7, -max_angle, max_angle))

        # ── 3. assemble goal JointConstraints ───────────────────────────────
        req = MotionPlanRequest()
        req.group_name                     = self.planning_group          # 'fr3_arm'
        req.start_state.joint_state        = start_js
        req.num_planning_attempts          = 5
        req.allowed_planning_time          = 5.0
        req.max_velocity_scaling_factor    = 0.2
        req.max_acceleration_scaling_factor= 0.2

        arm_joint_names = [n for n in start_js.name if n.startswith("fr3_joint")]
        goal_cs = Constraints()
        for name in arm_joint_names:
            jc = JointConstraint()
            jc.joint_name = name
            jc.position   = (
                target_j7 if name == "fr3_joint7"
                else start_js.position[start_js.name.index(name)]
            )
            tol = abs_tol_j7 if name == "fr3_joint7" else abs_tol_fixed
            jc.tolerance_above = jc.tolerance_below = tol
            jc.weight = 1.0
            goal_cs.joint_constraints.append(jc)
        req.goal_constraints = [goal_cs]

        # ── 4. send to MoveGroup ────────────────────────────────────────────
        goal = MoveGroup.Goal()
        goal.request          = req
        goal.planning_options = PlanningOptions()
        goal.planning_options.plan_only = not execute

        future = self.move_group_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        gh = future.result()
        if not gh.accepted:
            self.get_logger().error("Rotate goal rejected")
            return False

        res_future = gh.get_result_async()
        rclpy.spin_until_future_complete(self, res_future)
        error_code = res_future.result().result.error_code.val

        if error_code == MoveItErrorCodes.SUCCESS:
            self.get_logger().info(
                f"✓ fr3_joint7 reached {np.rad2deg(target_j7):.1f} ° "
                f"({'relative' if relative else 'absolute'})"
            )
            return True
        else:
            self.get_logger().warn(f"MoveIt failed (code {error_code}).")
            return False

    def _get_workspace_bounds(self):
        """FR3 cylindrical workspace specification (metres)."""
        return {
            'r_min': 0.20,   # inner keep-out radius
            'r_max': 0.90,  # maximum horizontal reach of flange
            'z_min': 0.01,   # lowest useful TCP height
            'z_max': 0.85,   # highest useful TCP height
        }

    def _is_pose_reachable(self, x: float, y: float, z: float) -> bool:
        """
        True if (x, y, z) lies within the cylindrical workspace defined above.
        """
        b = self._get_workspace_bounds()
        r = (x**2 + y**2) ** 0.5        # horizontal distance from base axis
        return (b['r_min'] <= r <= b['r_max'] and
                b['z_min'] <= z <= b['z_max'])

    def move_to_named_target(self, target_name: str, execute: bool = True) -> bool:
        """
        Move to named target with predefined poses
        """
        self.get_logger().info(f"Moving to named target: {target_name}")
        
        # Predefined poses (position + orientation)
        named_poses = {
            "home": [0.4, 0.0, 0.5, pi, 0.0, 0.0],  # [x, y, z, roll, pitch, yaw]

            "look_down": [0.4, 0.0, 0.4, 0.0, pi/3, 0.0],  # Looking down
            "look_left": [0.4, 0.2, 0.5, 0.0, 0.0, pi/4],  # Looking left
            "look_right": [0.4, -0.2, 0.5, 0.0, 0.0, -pi/4], # Looking right

            "test1": [0.5, 0.0, 0.4, pi, -1/2*pi, 0.0],  # [x, y, z, roll, pitch, yaw]
            "test2": [0.6, 0.0, 0.2, pi, 0.0, 0.0],  # [x, y, z, roll, pitch, yaw]
            "test3": [0.4, 0.3, 0.4, pi, 0.0, 0.0],  # [x, y, z, roll, pitch, yaw]
            "test4": [0.4, -0.4, 0.5, pi, 0.0, 0.0],  # [x, y, z, roll, pitch, yaw]
        }
        
        if target_name in named_poses:
            pose = named_poses[target_name]
            return self.move_to(pose[0], pose[1], pose[2], 
                                        pose[3], pose[4], pose[5], execute)
        else:
            self.get_logger().error(f"Unknown named target: {target_name}")
            return False

    


def demo():
    """Demonstrate the action-based controller with pose and gripper control"""
    
    rclpy.init()
    executor = MultiThreadedExecutor()
    
    try:
        controller = FR3Controller()
        executor.add_node(controller)
        
        # Start executor in background
        import threading
        executor_thread = threading.Thread(target=executor.spin, daemon=True)
        executor_thread.start()
        
        print("\n=== MoveIt2 Action Controller Demo with Gripper Control ===")
        
        # Move to ready position first
        print("\n--- Moving to ready position ---")
        # success = controller.move_to_named_target("home", execute=True)
        controller.reset()
        
        time.sleep(2)

        # success = controller.move_to_named_target("test1", execute=True)
        # time.sleep(2)
        # success = controller.move_to_named_target("test2", execute=True)
        # time.sleep(2)
        # success = controller.move_to_named_target("test3", execute=True)
        # time.sleep(2)
        # success = controller.move_to_named_target("test4", execute=True)
        
        # # Test gripper control
        # print("\n--- Testing Gripper Control ---")
        # controller.demo_gripper_control()
        # print("\n--- Rotating last joint to +45° ---")
        # print('controller state before rotation:', controller.current_joint_state)
        controller.rotate(np.deg2rad(22))
        time.sleep(1)
        controller.rotate(np.deg2rad(-45))
        time.sleep(1)
        controller.rotate(np.deg2rad(90))
        time.sleep(1)
        controller.rotate_to_home()
        time.sleep(1)
        
        # # Return to ready
        # print("\n--- Returning to ready position ---")
        # controller.move_to_named_target("home", execute=True)

        # print('current ee pose', controller.current_pose)
        controller.reset()
        
        print("\n=== Demo completed! ===")
        
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            controller.destroy_node()
        except:
            pass
        executor.shutdown()
        rclpy.shutdown()




if __name__ == "__main__":

    demo()