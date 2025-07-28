#!/usr/bin/env python3
"""
MoveIt2 Action Client Controller for Franka FR3
This implementation uses ROS2 action clients instead of moveit_py for better compatibility.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

import numpy as np
from math import pi, cos, sin
import time
from typing import List, Optional

# ROS2 messages
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
from std_msgs.msg import Header
from sensor_msgs.msg import JointState

# Franka gripper actions
from franka_msgs.action import Grasp, Move, Homing
from control_msgs.action import GripperCommand
from std_srvs.srv import Trigger

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
    RobotTrajectory
)
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.srv import GetPositionIK, GetCartesianPath



class MoveItActionController(Node):
    """
    MoveIt2 controller using action clients for Franka FR3 robot
    Compatible with standard ROS2 installations without moveit_py
    """
    
    def __init__(self):
        super().__init__('moveit_action_controller')
        
        # Robot configuration
        self.planning_group = "fr3_arm"  # Adjust based on your SRDF
        self.reference_frame = "fr3_link0"
        self.end_effector_link = "fr3_hand"
        self.offset = np.array([0, 0, 0.095]) # offset between link7 coordinate and gripper tip 
        self.use_cartesian_lin_control = True  # Enable cartesian linear control by default
        
        # Callback groups for concurrent operations
        self.callback_group = ReentrantCallbackGroup()
        
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
        
        # Primary gripper command action client (this is what's actually available)
        # self.gripper_command_client = ActionClient(
        #     self,
        #     GripperCommand,
        #     '/franka_gripper/gripper_action',
        #     callback_group=self.callback_group
        # )
        
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
        
        # # Gripper stop service client (available with real hardware)
        # self.gripper_stop_client = self.create_client(
        #     Trigger,
        #     '/franka_gripper/stop',
        #     callback_group=self.callback_group
        # )
        
        # Gripper joint state subscriber
        self.gripper_joint_state_sub = self.create_subscription(
            JointState,
            '/franka_gripper/joint_states',
            self.gripper_joint_state_callback,
            10,
            callback_group=self.callback_group
        )
        
        self.init_state = [0.4, 0.0, 0.5, pi, 0.0, 0.0]

        # State variables
        self.current_joint_state = None
        self.current_gripper_joint_state = None
        self.gripper_is_initialized = False
        
        # Wait for services
        self.get_logger().info("Waiting for MoveIt2 services...")
        
        if not self.move_group_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("MoveGroup action server not available!")
            raise RuntimeError("MoveGroup action server not available")
        
        self.get_logger().info("✓ MoveGroup action server connected")
        
        if not self.execute_trajectory_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().warn("ExecuteTrajectory action server not available - cartesian execution may be limited")
        else:
            self.get_logger().info("✓ ExecuteTrajectory action server connected")
        
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
        
        if self.gripper_grasp_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().info("✓ Gripper grasp action server connected")
            gripper_connected += 1
        else:
            self.get_logger().warn("Gripper grasp action server not available")
            
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
        
        # if self.gripper_command_client.wait_for_server(timeout_sec=5.0):
        #     self.get_logger().info("✓ Gripper command action server connected")
        #     gripper_connected += 1
        # else:
        #     self.get_logger().warn("Gripper command action server not available")
            
        # if self.gripper_stop_client.wait_for_service(timeout_sec=5.0):
        #     self.get_logger().info("✓ Gripper stop service connected")
        #     gripper_connected += 1
        # else:
        #     self.get_logger().warn("Gripper stop service not available")
            
        if gripper_connected > 0:
            self.get_logger().info(f"✓ {gripper_connected}/5 gripper services connected")
        else:
            self.get_logger().warn("No gripper services available - gripper control disabled")
        
        self.get_logger().info("MoveIt Action Controller initialized successfully!")

    
    def joint_state_callback(self, msg: JointState):
        """Update current joint state"""
        self.current_joint_state = msg

    def gripper_joint_state_callback(self, msg: JointState):
        """Update current gripper joint state"""
        self.current_gripper_joint_state = msg

    def enable_cartesian_control(self):
        """
        Enable cartesian linear motion planning using Pilz planner.
        Requires that the Pilz planner is configured in MoveIt2.
        """
        self.use_cartesian_lin_control = True
        self.get_logger().info("🔄 Cartesian linear control ENABLED. EE will move in straight lines.")


    def disable_cartesian_control(self):
        """
        Disable cartesian linear control.
        """
        self.use_cartesian_lin_control = False
        self.get_logger().info("🔄 Cartesian linear control DISABLED. Using joint space planning.")

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
        else:
            self.get_logger().error("✗ Gripper homing failed")
            
        return success
    
    def open_gripper(self, width: float = 0.08, speed: float = 0.1) -> bool:
        """Open gripper to specified width
        
        Args:
            width: Target width in meters (default: 0.08m - fully open)
            speed: Opening speed in m/s (default: 0.1 m/s)
            
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
        
        # if success:
        #     self.get_logger().info("✓ Gripper opened successfully")
        # else:
        #     self.get_logger().error("✗ Gripper opening failed")
            
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
    
    def grasp_object(self, width: float = 0.02, speed: float = 0.1, force: float = 50.0, 
                    inner_tolerance: float = 0.005, outer_tolerance: float = 0.01) -> bool:
        """Grasp an object with specified parameters
        
        Args:
            width: Target grasp width in meters (default: 0.02m)
            speed: Grasping speed in m/s (default: 0.1 m/s)
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
        
    # def gripper_command(self, position: float, max_effort: float = 50.0) -> bool:
    #     """Control gripper using standard GripperCommand action [m]
        
    #     Args:
    #         position: Target position in meters (0.0 m = closed, 0.08 m = open)
    #         max_effort: Maximum effort in Newtons (default: 50.0)
            
    #     Returns:
    #         bool: True if successful, False otherwise
    #     """
    #     if not self.gripper_command_client.wait_for_server(timeout_sec=5.0):
    #         self.get_logger().error('Gripper command action server not available')
    #         return False
        
    #     goal = GripperCommand.Goal()
    #     goal.command.position = position
    #     goal.command.max_effort = max_effort
        
    #     self.get_logger().info(f"Sending gripper command: position={position:.4f}m, max_effort={max_effort:.1f}N")
        
    #     future = self.gripper_command_client.send_goal_async(goal)
    #     rclpy.spin_until_future_complete(self, future)
        
    #     goal_handle = future.result()
    #     if not goal_handle.accepted:
    #         self.get_logger().error("Gripper command goal was rejected!")
    #         return False
        
    #     # Wait for result
    #     result_future = goal_handle.get_result_async()
    #     rclpy.spin_until_future_complete(self, result_future)
        
    #     result = result_future.result()
    #     success = abs(result.result.position - position) < 0.005  # 1cm tolerance
        
    #     if success:
    #         self.get_logger().info("✓ Gripper command completed successfully")
    #     else:
    #         self.get_logger().error(f"✗ Gripper command failed. Target: {position:.4f}, Actual: {result.result.position:.4f}")
            
    #     return success

    # def stop_gripper(self) -> bool:
    #     """Stop current gripper action"""
    #     if not self.gripper_stop_client.wait_for_service(timeout_sec=5.0):
    #         self.get_logger().error('Gripper stop service not available')
    #         return False
        
    #     request = Trigger.Request()
    #     self.get_logger().info("Stopping gripper...")
        
    #     future = self.gripper_stop_client.call_async(request)
    #     rclpy.spin_until_future_complete(self, future)
        
    #     result = future.result()
    #     success = result.success
        
    #     if success:
    #         self.get_logger().info("✓ Gripper stopped successfully")
    #     else:
    #         self.get_logger().error(f"✗ Gripper stop failed: {result.message}")
            
    #     return success
    
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
        
        return self.move_to(self.init_state[0], self.init_state[1], self.init_state[2], 
                                    self.init_state[3], self.init_state[4], self.init_state[5], execute)


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

        # Configure planning pipeline based on cartesian control setting
        if self.use_cartesian_lin_control:
            request.pipeline_id = "pilz_industrial_motion_planner"
            request.planner_id = "LIN"  # Linear interpolation for straight line motion
            self.get_logger().info("Using Pilz LIN planner for cartesian straight line motion")
        else:
            request.pipeline_id = "ompl"  # Default OMPL planner
            request.planner_id = "RRTConnectkConfigDefault"
            self.get_logger().info("Using OMPL planner for joint space motion")
        
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
        
        # Planning parameters - more conservative for cartesian planning
        request.num_planning_attempts = 10
        request.allowed_planning_time = 15.0 if self.use_cartesian_lin_control else 10.0
        request.max_velocity_scaling_factor = 0.1
        request.max_acceleration_scaling_factor = 0.1
        
        return request

    def move_to_cartesian(self, x: float, y: float, z: float, 
                         roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0,
                         execute: bool = True, max_step: float = 0.01) -> bool:
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
        assert self._is_pose_reachable(x, y, z)

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
        
        # Create waypoints list (just start and end for now)
        waypoints = [target_pose]
        
        # Create cartesian path request
        if not self.cartesian_path_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("Cartesian path service not available")
            # Fallback to regular planning
            return self._move_to_joint_space(x, y, z, roll, pitch, yaw, execute)
        
        request = GetCartesianPath.Request()
        request.header.frame_id = self.reference_frame
        request.start_state.joint_state = self.current_joint_state
        request.group_name = self.planning_group
        request.link_name = self.end_effector_link
        request.waypoints = waypoints
        request.max_step = max_step
        request.jump_threshold = 0.0  # Disable jump detection
        request.avoid_collisions = True
        
        self.get_logger().info(f"Planning cartesian path to pose: [{x:.3f}, {y:.3f}, {z:.3f}] "
                              f"orientation: [{roll:.3f}, {pitch:.3f}, {yaw:.3f}]")
        
        # Call cartesian path service
        future = self.cartesian_path_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        response = future.result()
        
        if response.fraction < 0.9:  # Less than 90% of path achieved
            self.get_logger().warn(f"Cartesian path only {response.fraction*100:.1f}% feasible")
            # Fallback to regular planning
            self.get_logger().info("Falling back to regular motion planning...")
            return self._move_to_joint_space(x, y, z, roll, pitch, yaw, execute)
        
        if not execute:
            self.get_logger().info(f"✓ Cartesian path planned successfully ({response.fraction*100:.1f}% feasible)")
            return True
        
        # Execute the cartesian trajectory using ExecuteTrajectory action
        if not self.execute_trajectory_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("ExecuteTrajectory action server not available!")
            # Fallback to regular planning
            return self._move_to_joint_space(x, y, z, roll, pitch, yaw, execute)
        
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
            self.get_logger().error(f"✗ Cartesian motion failed with error code: {error_code}")
            return False
    
    def _move_to_joint_space(self, x: float, y: float, z: float, 
                         roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0,
                         execute: bool = True) -> bool:
        """
        Internal method for joint space planning (used as fallback for cartesian planning)
        
        Args:
            x, y, z: Target position in meters
            roll, pitch, yaw: Target orientation in radians
            execute: Whether to execute the motion or just plan
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Create target pose
        target_pose = self._create_pose_target(x, y, z, roll, pitch, yaw)
        
        # Create motion plan request with OMPL planner
        request = MotionPlanRequest()
        request.group_name = self.planning_group
        request.pipeline_id = "ompl"
        request.planner_id = "RRTConnectkConfigDefault"
        
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
        pos_constraint.constraint_region.primitives = [SolidPrimitive()]
        pos_constraint.constraint_region.primitives[0].type = SolidPrimitive.BOX
        pos_constraint.constraint_region.primitives[0].dimensions = [0.01, 0.01, 0.01]
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
        
        # Create MoveGroup goal
        goal = MoveGroup.Goal()
        goal.request = request
        goal.planning_options = PlanningOptions()
        goal.planning_options.plan_only = not execute
        goal.planning_options.look_around = False
        goal.planning_options.look_around_attempts = 0
        goal.planning_options.max_safe_execution_cost = 1.0
        goal.planning_options.replan = False
        goal.planning_options.replan_attempts = 0
        
        self.get_logger().info(f"Planning joint space motion to pose: [{x:.3f}, {y:.3f}, {z:.3f}] "
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
            self.get_logger().info("✓ Joint space motion completed successfully!")
            return True
        else:
            self.get_logger().error(f"✗ Joint space motion failed with error code: {error_code}")
            return False
    
    def move_to(self, x: float, y: float, z: float, 
                         roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0,
                         execute: bool = True) -> bool:
        """
        Synchronously move to target pose
        
        Args:
            x, y, z: Target position in meters
            roll, pitch, yaw: Target orientation in radians
            execute: Whether to execute the motion or just plan
            
        Returns:
            bool: True if successful, False otherwise
        """
        assert self._is_pose_reachable(x,y,z)

        # If cartesian control is enabled, use the cartesian path planner
        if self.use_cartesian_lin_control:
            return self.move_to_cartesian(x, y, z, roll, pitch, yaw, execute)

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
        
        planning_method = "cartesian linear" if self.use_cartesian_lin_control else "joint space"
        self.get_logger().info(f"Planning {planning_method} motion to pose: [{x:.3f}, {y:.3f}, {z:.3f}] "
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
            self.get_logger().error(f"✗ Motion failed with error code: {error_code}")
            return False

    def _get_workspace_bounds(self):
        """Get approximate workspace bounds"""
        return {
            'x_min': 0.1, 'x_max': 0.8,
            'y_min': -0.5, 'y_max': 0.5,
            'z_min': 0.1, 'z_max': 0.8  # TODO: offset between link 7 and gripper tip
        }
    
    def _is_pose_reachable(self, x: float, y: float, z: float) -> bool:
        """Check if pose is within workspace bounds"""
        bounds = self._get_workspace_bounds()
        return (bounds['x_min'] <= x <= bounds['x_max'] and
                bounds['y_min'] <= y <= bounds['y_max'] and
                bounds['z_min'] <= z <= bounds['z_max'])


    def move_to_named_target(self, target_name: str, execute: bool = True) -> bool:
        """
        Move to named target with predefined poses
        """
        self.get_logger().info(f"Moving to named target: {target_name}")
        
        # Predefined poses (position + orientation)
        named_poses = {
            # "ready": [0.4, 0.0, 0.5, 0.0, 0.0, 0.0],  # [x, y, z, roll, pitch, yaw]
            "home": [0.4, 0.0, 0.5, pi, 0.0, 0.0],  # [x, y, z, roll, pitch, yaw]
            # "home": [0.3, 0.0, 0.6, 0.0, 0.0, 0.0],
            "ready": [0.4, 0.0, 0.5, pi, 0.0, 0.0],  # [x, y, z, roll, pitch, yaw]
            "extended": [0.6, 0.0, 0.3, 0.0, 0.0, 0.0],

            "look_down": [0.4, 0.0, 0.4, 0.0, pi/3, 0.0],  # Looking down
            "look_left": [0.4, 0.2, 0.5, 0.0, 0.0, pi/4],  # Looking left
            "look_right": [0.4, -0.2, 0.5, 0.0, 0.0, -pi/4], # Looking right
            # "test1": [0.5, 0.0, 0.5, 0.0, pi/2, 0.0],  # [x, y, z, roll, pitch, yaw]
        }
        
        if target_name in named_poses:
            pose = named_poses[target_name]
            return self.move_to(pose[0], pose[1], pose[2], 
                                        pose[3], pose[4], pose[5], execute)
        else:
            self.get_logger().error(f"Unknown named target: {target_name}")
            return False


    def demo_gripper_control(self):
        """Demonstrate gripper control capabilities"""
        
        print("\n🤖 === Gripper Control Demonstration ===")
        
        # Check if gripper state is available
        gripper_state = self.get_gripper_state()
        if not gripper_state['available']:
            print("❌ Gripper joint states not available")
            return
        
        print(f"📊 Current gripper state:")
        print(f"  - Width: {gripper_state['width']:.4f}m")
        print(f"  - Joint positions: {[f'{pos:.4f}' for pos in gripper_state['position']]}")
        print(f"  - Joint names: {gripper_state['joint_names']}")
        
        # Initialize gripper
        print("\n1️⃣ Initializing gripper...")
        if not self.reset_gripper():
            print("❌ Failed to initialize gripper")
            return
        
        time.sleep(3)
        
        # Test basic movements
        print("\n2️⃣ Basic gripper movements:")
        
        # Open gripper
        print("  → Opening gripper fully")
        if self.open_gripper():
            print("    ✓ Gripper opened")
            self.print_gripper_state()
        else:
            print("    ✗ Failed to open gripper")
        
        time.sleep(3)
        
        # Close gripper
        print("  → Closing gripper")
        if self.close_gripper():
            print("    ✓ Gripper closed")
            self.print_gripper_state()
        else:
            print("    ✗ Failed to close gripper")
        
        time.sleep(3)
        
        # Test normalized control
        print("\n3️⃣ Normalized control test:")
        positions = [0.2, 0.5, 0.8, 0.5, 0.0]
        
        for pos in positions:
            print(f"  → Setting gripper to {pos*100:.0f}% open")
            if self.set_gripper_action(pos):
                print(f"    ✓ Success")
                self.print_gripper_state()
            else:
                print(f"    ✗ Failed")
            time.sleep(2)
        
        # Test grasping
        print("\n4️⃣ Grasping test:")
        print("  → Attempting to grasp object (width: 0.02m)")
        if self.grasp_object(0.02, 0.1, 30.0):
            print("    ✓ Grasp successful")
            self.print_gripper_state()
        else:
            print("    ✗ Grasp failed")
        
        time.sleep(2)
        
        # Return to open position
        print("\n5️⃣ Returning to open position...")
        self.open_gripper(0.08, 0.1)
        
        print("\n✨ Gripper demonstration completed!")
    
    def demo_pose_sequences(self):
        """Demonstrate various pose sequences with orientation changes"""
        
        print("\n🎭 === Pose Control Demonstration ===")
        
        # Sequence 1: Basic orientation changes
        print("\n1️⃣ Basic Orientation Changes:")
        basic_poses = [
            [0.4, 0.0, 0.5, 0.0, 0.0, 0.0, "Neutral pose"],
            [0.4, 0.0, 0.5, pi/6, 0.0, 0.0, "Roll +30°"],
            [0.4, 0.0, 0.5, 0.0, pi/6, 0.0, "Pitch +30°"],
            [0.4, 0.0, 0.5, 0.0, 0.0, pi/6, "Yaw +30°"],
            [0.4, 0.0, 0.5, 0.0, 0.0, 0.0, "Back to neutral"],
        ]
        
        for x, y, z, roll, pitch, yaw, description in basic_poses:
            print(f"  → {description}")
            success = self.move_to(x, y, z, roll, pitch, yaw, execute=True)
            if success:
                print(f"    ✓ Success")
                time.sleep(2)
            else:
                print(f"    ✗ Failed")
                break
        
        # Sequence 2: Functional poses
        print("\n2️⃣ Functional Poses:")
        functional_poses = [
            [0.4, 0.0, 0.4, 0.0, pi/2, 0.0, "Looking straight down"],
            [0.4, 0.2, 0.5, 0.0, 0.0, pi/3, "Looking at right side"],
            [0.4, -0.2, 0.5, 0.0, 0.0, -pi/3, "Looking at left side"],
        ]
        
        for x, y, z, roll, pitch, yaw, description in functional_poses:
            print(f"  → {description}")
            success = self.move_to(x, y, z, roll, pitch, yaw, execute=True)
            if success:
                print(f"    ✓ Success")
                time.sleep(2)
            else:
                print(f"    ✗ Failed")
                break
        
        # Sequence 3: Complex combined motions
        print("\n3️⃣ Complex Combined Motions:")
        complex_poses = [
            [0.3, 0.1, 0.6, pi/6, pi/6, pi/6, "Complex orientation 1"],
            [0.5, -0.1, 0.4, -pi/6, pi/4, -pi/4, "Complex orientation 2"],
            [0.4, 0.0, 0.5, 0.0, 0.0, 0.0, "Return to neutral"],
        ]
        
        for x, y, z, roll, pitch, yaw, description in complex_poses:
            print(f"  → {description}")
            success = self.move_to(x, y, z, roll, pitch, yaw, execute=True)
            if success:
                print(f"    ✓ Success")
                time.sleep(2)
            else:
                print(f"    ✗ Failed")
                break
        
        print("\n✨ Pose demonstration completed!")

    def demo_cartesian_vs_joint_motion(self):
        """Demonstrate the difference between cartesian linear motion and joint space motion"""
        
        print("\n🎯 === Cartesian vs Joint Space Motion Demonstration ===")
        
        # Define test poses for comparison
        test_poses = [
            [0.4, 0.0, 0.5, pi, 0.0, 0.0, "Home position"],
            [0.4, 0.2, 0.5, pi, 0.0, 0.0, "Move right 20cm"],
            [0.4, 0.2, 0.3, pi, 0.0, 0.0, "Move down 20cm"],
            [0.4, -0.2, 0.3, pi, 0.0, 0.0, "Move left 40cm"],
            [0.4, -0.2, 0.5, pi, 0.0, 0.0, "Move up 20cm"],
            [0.4, 0.0, 0.5, pi, 0.0, 0.0, "Return to home"],
        ]
        
        # Test with joint space motion first
        print("\n1️⃣ Joint Space Motion (Traditional):")
        print("   📝 Note: End effector may take curved paths, large arm movements")
        self.disable_cartesian_control()
        
        for i, (x, y, z, roll, pitch, yaw, description) in enumerate(test_poses):
            print(f"  → Step {i+1}: {description}")
            success = self.move_to(x, y, z, roll, pitch, yaw, execute=True)
            if success:
                print(f"    ✓ Success")
                time.sleep(2)  # Pause to observe motion
            else:
                print(f"    ✗ Failed")
                break
        
        time.sleep(3)
        print("\n" + "="*60)
        
        # Test with cartesian linear motion
        print("\n2️⃣ Cartesian Linear Motion (Straight Lines):")
        print("   📝 Note: End effector follows straight lines between points")
        self.enable_cartesian_control()
        
        for i, (x, y, z, roll, pitch, yaw, description) in enumerate(test_poses):
            print(f"  → Step {i+1}: {description}")
            success = self.move_to(x, y, z, roll, pitch, yaw, execute=True)
            if success:
                print(f"    ✓ Success")
                time.sleep(2)  # Pause to observe motion
            else:
                print(f"    ✗ Failed")
                break
        
        print("\n✨ Motion comparison completed!")
        print("🔍 Did you notice the difference?")
        print("   • Joint space: Arm takes curved paths, may move unexpectedly")
        print("   • Cartesian: End effector moves in straight lines")

    def demo_precise_cartesian_movements(self):
        """Demonstrate precise cartesian movements useful for manipulation tasks"""
        
        print("\n🎯 === Precise Cartesian Movement Patterns ===")
        
        # Enable cartesian control for precise movements
        self.enable_cartesian_control()
        
        # Start from home position
        print("\n📍 Moving to starting position...")
        self.move_to(0.4, 0.0, 0.4, pi, 0.0, 0.0, execute=True)
        time.sleep(2)
        
        # Pattern 1: Square movement in XY plane
        print("\n1️⃣ Square pattern in XY plane (10cm x 10cm):")
        square_points = [
            [0.4, 0.05, 0.4, pi, 0.0, 0.0, "Top right corner"],
            [0.4, 0.05, 0.35, pi, 0.0, 0.0, "Bottom right corner"],
            [0.4, -0.05, 0.35, pi, 0.0, 0.0, "Bottom left corner"],
            [0.4, -0.05, 0.4, pi, 0.0, 0.0, "Top left corner"],
            [0.4, 0.0, 0.4, pi, 0.0, 0.0, "Center (finish)"],
        ]
        
        for x, y, z, roll, pitch, yaw, description in square_points:
            print(f"  → {description}")
            success = self.move_to(x, y, z, roll, pitch, yaw, execute=True)
            if success:
                print(f"    ✓ Success")
                time.sleep(1.5)
            else:
                print(f"    ✗ Failed")
                break
        
        time.sleep(2)
        
        # Pattern 2: Vertical line movement
        print("\n2️⃣ Vertical line movement (up and down):")
        vertical_points = [
            [0.4, 0.0, 0.5, pi, 0.0, 0.0, "Move up 10cm"],
            [0.4, 0.0, 0.3, pi, 0.0, 0.0, "Move down 20cm"],
            [0.4, 0.0, 0.4, pi, 0.0, 0.0, "Return to center"],
        ]
        
        for x, y, z, roll, pitch, yaw, description in vertical_points:
            print(f"  → {description}")
            success = self.move_to(x, y, z, roll, pitch, yaw, execute=True)
            if success:
                print(f"    ✓ Success")
                time.sleep(1.5)
            else:
                print(f"    ✗ Failed")
                break
        
        time.sleep(2)
        
        # Pattern 3: Diagonal movement
        print("\n3️⃣ Diagonal movement pattern:")
        diagonal_points = [
            [0.5, 0.1, 0.5, pi, 0.0, 0.0, "Forward-right-up"],
            [0.3, -0.1, 0.3, pi, 0.0, 0.0, "Back-left-down"],
            [0.4, 0.0, 0.4, pi, 0.0, 0.0, "Return to center"],
        ]
        
        for x, y, z, roll, pitch, yaw, description in diagonal_points:
            print(f"  → {description}")
            success = self.move_to(x, y, z, roll, pitch, yaw, execute=True)
            if success:
                print(f"    ✓ Success")
                time.sleep(1.5)
            else:
                print(f"    ✗ Failed")
                break
        
        print("\n✨ Precise cartesian movement demonstration completed!")
        print("💡 These patterns are useful for:")
        print("   • Pick and place operations")
        print("   • Precise assembly tasks") 
        print("   • Drawing or tracing operations")
        print("   • Any task requiring straight-line end effector motion")


def demo_action_controller():
    """Demonstrate the action-based controller with pose and gripper control"""
    
    rclpy.init()
    executor = MultiThreadedExecutor()
    
    try:
        controller = MoveItActionController()
        executor.add_node(controller)
        
        # Start executor in background
        import threading
        executor_thread = threading.Thread(target=executor.spin, daemon=True)
        executor_thread.start()
        
        print("\n=== MoveIt2 Action Controller Demo with Gripper Control ===")
        
        # Move to ready position first
        print("\n--- Moving to ready position ---")
        success = controller.move_to_named_target("ready", execute=True)
        if success:
            print("✓ Ready position reached")
        else:
            print("✗ Failed to reach ready position")
        
        time.sleep(2)
        
        # Test gripper control
        print("\n--- Testing Gripper Control ---")
        controller.demo_gripper_control()
        
        # Advanced pose demonstration
        response = input("\nDo you want to run advanced pose sequences? (y/N): ").lower().strip()
        if response == 'y':
            controller.demo_pose_sequences()
        
        # Return to ready
        print("\n--- Returning to ready position ---")
        controller.move_to_named_target("ready", execute=True)
        
        print("\n=== Demo completed! ===")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nDemo failed: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            controller.destroy_node()
        except:
            pass
        executor.shutdown()
        rclpy.shutdown()


def test_controller():
    """Test the action-based controller with cartesian linear motion control"""
    
    rclpy.init()
    executor = MultiThreadedExecutor()
    
    try:
        controller = MoveItActionController()
        executor.add_node(controller)
        
        # Start executor in background
        import threading
        executor_thread = threading.Thread(target=executor.spin, daemon=True)
        executor_thread.start()
        
        print("\n🤖 === MoveIt2 Action Controller with Cartesian Linear Control ===")
        print("🎯 This controller now supports straight-line end effector motion!")
        
        # Move to home position first
        print("\n--- Moving to home position ---")
        success = controller.move_to_named_target("home", execute=True)
        if success:
            print("✓ Home position reached")
        else:
            print("✗ Failed to reach home position")
            return
        
        time.sleep(2)
        
        # Show current control mode
        mode_status = "ENABLED" if controller.use_cartesian_lin_control else "DISABLED"
        print(f"\n📊 Current Control Mode: Cartesian Linear Control is {mode_status}")
        
        # Test basic cartesian motion
        print("\n--- Testing Basic Cartesian Motion ---")
        print("Moving in straight lines between points...")
        
        # Simple test movements
        test_points = [
            [0.5, 0.0, 0.5, "Forward 10cm"],
            [0.5, 0.1, 0.5, "Right 10cm"],
            [0.4, 0.1, 0.4, "Back 10cm, Down 10cm"],
            [0.4, 0.0, 0.5, "Left 10cm, Up 10cm (home)"],
        ]
        
        for i, (x, y, z, description) in enumerate(test_points):
            print(f"  → Step {i+1}: {description}")
            success = controller.move_to(x, y, z, pi, 0.0, 0.0, execute=True)
            if success:
                print(f"    ✓ Success")
                time.sleep(1.5)
            else:
                print(f"    ✗ Failed")
                break
        
        print("\n--- Testing Gripper Control ---")
        controller.demo_gripper_control()
        
        # Interactive menu for additional demonstrations
        while True:
            print("\n" + "="*50)
            print("🎮 Interactive Demo Menu:")
            print("1. Compare Cartesian vs Joint Space Motion")
            print("2. Precise Cartesian Movement Patterns")
            print("3. Advanced Pose Sequences")
            print("4. Toggle Control Mode (Current: {})".format(
                "Cartesian Linear" if controller.use_cartesian_lin_control else "Joint Space"))
            print("5. Return to Home Position")
            print("0. Exit")
            
            try:
                choice = input("\nSelect option (0-5): ").strip()
                
                if choice == '0':
                    break
                elif choice == '1':
                    controller.demo_cartesian_vs_joint_motion()
                elif choice == '2':
                    controller.demo_precise_cartesian_movements()
                elif choice == '3':
                    controller.demo_pose_sequences()
                elif choice == '4':
                    if controller.use_cartesian_lin_control:
                        controller.disable_cartesian_control()
                    else:
                        controller.enable_cartesian_control()
                    print(f"   → Control mode changed to: {'Cartesian Linear' if controller.use_cartesian_lin_control else 'Joint Space'}")
                elif choice == '5':
                    print("   → Returning to home position...")
                    controller.move_to_named_target("home", execute=True)
                else:
                    print("   ❌ Invalid option. Please select 0-5.")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        # Return to home before exiting
        print("\n--- Returning to home position ---")
        controller.move_to_named_target("home", execute=True)
        
        print("\n✨ Test completed successfully!")
        print("💡 Key Benefits of Cartesian Linear Control:")
        print("   • End effector moves in predictable straight lines")
        print("   • Better for manipulation tasks requiring precision")
        print("   • Reduced unexpected arm movements")
        print("   • More intuitive motion planning")
        
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
    test_controller() 
    # demo_action_controller()