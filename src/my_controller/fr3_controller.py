#!/usr/bin/env python3
"""
MoveIt2 Action Client Controller for Franka FR3
This implementation uses ROS2 action clients instead of moveit_py for better compatibility.

This implementation uses OMPL for motion planning.
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
from std_srvs.srv import Trigger

# MoveIt2 action interfaces
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    MotionPlanRequest, 
    PlanningOptions,
    Constraints,
    PositionConstraint,
    OrientationConstraint,
    WorkspaceParameters,
    RobotState,
    MoveItErrorCodes
)
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.srv import GetPositionIK




class FR3Controller(Node):
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
        self.offset = np.array([0.0, 0.0, 0.095]) # offset between link7 coordinate and gripper tip 
        self.init_state = [0.4, 0.0, 0.5, pi, 0.0, 0.0]
        self.current_joint_state = None
        self.current_gripper_joint_state = None
        self.gripper_is_initialized = False

        
        # Callback groups for concurrent operations
        self.callback_group = ReentrantCallbackGroup()
        
        # Action client for MoveGroup
        self.move_group_client = ActionClient(
            self,
            MoveGroup,
            '/move_action',
            callback_group=self.callback_group
        )
        
        # Service client for inverse kinematics
        self.ik_client = self.create_client(
            GetPositionIK,
            '/compute_ik',
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
        
        s1 = self.move_to(self.init_state[0], self.init_state[1], self.init_state[2], 
                                    self.init_state[3], self.init_state[4], self.init_state[5], execute)
        s2 = self.reset_gripper()

        if s1 and s2:
            self.get_logger().info(f"Robot Initialized.")


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
            # self.get_logger().info("✓ Motion completed successfully!")
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
        request.planner_id = "PTP"

        
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
    

    # def _get_workspace_bounds(self):
    #     """Get approximate workspace bounds"""
    #     return {
    #         'x_min': 0.1, 'x_max': 0.8,
    #         'y_min': -0.5, 'y_max': 0.5,
    #         'z_min': 0.1, 'z_max': 0.8  # TODO: offset between link 7 and gripper tip
    #     }
    
    # def _is_pose_reachable(self, x: float, y: float, z: float) -> bool:
    #     """Check if pose is within workspace bounds"""
    #     bounds = self._get_workspace_bounds()
    #     return (bounds['x_min'] <= x <= bounds['x_max'] and
    #             bounds['y_min'] <= y <= bounds['y_max'] and
    #             bounds['z_min'] <= z <= bounds['z_max'])

    def _get_workspace_bounds(self):
        """FR3 cylindrical workspace specification (metres)."""
        return {
            'r_min': 0.30,   # inner keep-out radius
            'r_max': 0.855,  # maximum horizontal reach of flange
            'z_min': 0.02,   # lowest useful TCP height
            'z_max': 0.80,   # highest useful TCP height
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
            "test1": [0.4, 0.0, 0.2, pi, 0.0, 0.0],  # [x, y, z, roll, pitch, yaw]
            "test2": [0.6, 0.0, 0.3, pi, 0.0, 0.0],  # [x, y, z, roll, pitch, yaw]
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
        
        
        # Return to open position
        print("\n5️⃣ Returning to open position...")
        self.open_gripper(0.08, 0.1)
        
        print("\n✨ Gripper demonstration completed!")
    


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
        success = controller.move_to_named_target("home", execute=True)
        if success:
            print("✓ Ready position reached")
        else:
            print("✗ Failed to reach ready position")
        
        time.sleep(2)

        success = controller.move_to_named_target("test1", execute=True)
        time.sleep(2)
        success = controller.move_to_named_target("test2", execute=True)
        time.sleep(2)
        success = controller.move_to_named_target("test3", execute=True)
        time.sleep(2)
        success = controller.move_to_named_target("test4", execute=True)
        
        # # Test gripper control
        # print("\n--- Testing Gripper Control ---")
        # controller.demo_gripper_control()
        
        
        # Return to ready
        print("\n--- Returning to ready position ---")
        controller.move_to_named_target("home", execute=True)
        
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




if __name__ == "__main__":

    demo()