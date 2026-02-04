#!/usr/bin/env python
"""
Tool Executor Node for PRIME

Executes tool calls from the LLM executive using:
- MoveIt for APPROACH and ALIGN_YAW
- Kinova driver for GRASP and RELEASE
- User Interface for INTERACT

This node bridges the symbolic tool calls to actual robot actions.
"""

import rospy
import os
import sys
import numpy as np
from threading import Lock
from typing import Optional, Tuple

# Ensure local PRIME scripts directory is on PYTHONPATH (rosrun wrapper doesn't add it)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# MoveIt
import moveit_commander
from moveit_commander import MoveGroupCommander, PlanningSceneInterface
from moveit_msgs.msg import DisplayTrajectory

# Geometry
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from tf.transformations import quaternion_from_euler
import tf2_ros
import tf2_geometry_msgs

# Kinova
from kinova_msgs.msg import FingerPosition
from kinova_msgs.srv import HomeArm
import actionlib
from kinova_msgs.msg import SetFingersPositionAction, SetFingersPositionGoal

# PRIME messages
try:
    from prime_ros.msg import (
        ToolCall, ToolResult, SymbolicState, ObjectState,
        PRIMEQuery, PRIMEResponse
    )
    from prime_ros.srv import ExecuteTool, ExecuteToolResponse
    MSGS_AVAILABLE = True
except ImportError:
    rospy.logwarn("PRIME messages not built yet.")
    MSGS_AVAILABLE = False

from prime_memory import get_memory


class ToolExecutor:
    """
    Executes PRIME tool calls.
    
    Maps symbolic actions to robot motion and control commands.
    """
    
    # Finger positions
    FINGER_MAX = 6400
    FINGER_OPEN = 0
    FINGER_CLOSE = 5000  # Partial close for grasping
    
    def __init__(self):
        rospy.init_node('tool_executor', anonymous=False)
        
        # Parameters
        self.robot_type = rospy.get_param('robot/type', 'j2n6s300')
        self.pre_grasp_distance = rospy.get_param('tools/approach/pre_grasp_distance', 0.10)
        self.approach_speed = rospy.get_param('tools/approach/approach_speed', 0.1)
        self.align_tolerance = rospy.get_param('tools/align/tolerance', 0.1)
        
        # Thread safety
        self.lock = Lock()
        
        # State
        self.current_state: Optional[SymbolicState] = None
        self.objects: dict = {}  # object_id -> ObjectState
        self.memory = get_memory()
        
        # Initialize MoveIt
        rospy.loginfo("Initializing MoveIt...")
        moveit_commander.roscpp_initialize(sys.argv)
        
        self.robot = moveit_commander.RobotCommander()
        self.scene = PlanningSceneInterface()
        
        # Move groups
        self.arm_group = MoveGroupCommander("arm")
        self.gripper_group = MoveGroupCommander("gripper")
        
        # Configure arm group
        self.arm_group.set_planning_time(10.0)
        self.arm_group.set_num_planning_attempts(5)
        self.arm_group.set_max_velocity_scaling_factor(0.3)
        self.arm_group.set_max_acceleration_scaling_factor(0.3)
        
        # Set end effector
        self.arm_group.set_end_effector_link(f"{self.robot_type}_end_effector")
        
        # Finger action client
        self.finger_client = actionlib.SimpleActionClient(
            f'/{self.robot_type}_driver/fingers_action/finger_positions',
            SetFingersPositionAction
        )
        rospy.loginfo("Waiting for finger action server...")
        self.finger_client.wait_for_server(rospy.Duration(5.0))
        
        # TF2 for transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Subscribers
        if MSGS_AVAILABLE:
            self.state_sub = rospy.Subscriber(
                '/prime/symbolic_state',
                SymbolicState,
                self.state_callback
            )
            
            self.tool_sub = rospy.Subscriber(
                '/prime/tool_call',
                ToolCall,
                self.tool_callback
            )
            
            self.response_sub = rospy.Subscriber(
                '/prime/response',
                PRIMEResponse,
                self.response_callback
            )
        
        # Publishers
        if MSGS_AVAILABLE:
            self.result_pub = rospy.Publisher(
                '/prime/tool_result',
                ToolResult,
                queue_size=10
            )
        
        # Service
        if MSGS_AVAILABLE:
            self.execute_service = rospy.Service(
                '/prime/execute_tool',
                ExecuteTool,
                self.handle_execute_tool
            )
        
        rospy.loginfo("Tool Executor initialized")
    
    def state_callback(self, msg: SymbolicState):
        """Handle symbolic state updates."""
        with self.lock:
            self.current_state = msg
            # Build object lookup
            self.objects = {obj.object_id: obj for obj in msg.objects}
    
    def tool_callback(self, msg: ToolCall):
        """Handle incoming tool calls."""
        result = self.execute_tool(msg)
        if self.result_pub:
            self.result_pub.publish(result)
    
    def response_callback(self, msg: PRIMEResponse):
        """Handle user responses to INTERACT queries."""
        # Forward to LLM executive for processing
        pass
    
    def handle_execute_tool(self, req) -> ExecuteToolResponse:
        """Service handler for tool execution."""
        result = self.execute_tool(req.tool_call)
        return ExecuteToolResponse(result=result)
    
    def execute_tool(self, call: ToolCall) -> ToolResult:
        """
        Execute a tool call and return the result.
        
        Dispatches to the appropriate handler based on tool name.
        """
        result = ToolResult()
        result.header.stamp = rospy.Time.now()
        result.call_id = call.call_id
        result.tool_name = call.tool_name
        
        rospy.loginfo(f"Executing tool: {call.tool_name}")
        
        try:
            if call.tool_name == 'INTERACT':
                success, message = self.execute_interact(call)
            elif call.tool_name == 'APPROACH':
                success, message = self.execute_approach(call.target_object_id)
            elif call.tool_name == 'ALIGN_YAW':
                success, message = self.execute_align_yaw(call.target_object_id)
            elif call.tool_name == 'GRASP':
                success, message = self.execute_grasp()
            elif call.tool_name == 'RELEASE':
                success, message = self.execute_release()
            else:
                success = False
                message = f"Unknown tool: {call.tool_name}"
            
            result.success = success
            result.status = ToolResult.STATUS_SUCCESS if success else ToolResult.STATUS_FAILED
            result.message = message
            
            # Record in memory
            self.memory.add_tool_record(
                tool_name=call.tool_name,
                target_object=call.target_object_id if call.target_object_id else None,
                params={},
                success=success,
                error_category='execution' if not success else None,
                message=message
            )
            
        except Exception as e:
            rospy.logerr(f"Tool execution error: {e}")
            result.success = False
            result.status = ToolResult.STATUS_FAILED
            result.error_category = 'exception'
            result.message = str(e)
        
        return result
    
    def execute_interact(self, call: ToolCall) -> Tuple[bool, str]:
        """
        Execute INTERACT tool.
        
        Publishes query and waits for user response.
        The actual response handling is done asynchronously.
        """
        # Query is already published by LLM executive
        # This just acknowledges the interaction was initiated
        rospy.loginfo(f"INTERACT: {call.interact_content}")
        return True, "Query sent to user"
    
    def execute_approach(self, object_id: str) -> Tuple[bool, str]:
        """
        Execute APPROACH tool - move to pre-grasp position.
        
        Moves the gripper to a position above/near the target object.
        """
        with self.lock:
            if object_id not in self.objects:
                return False, f"Object {object_id} not found"
            
            obj = self.objects[object_id]
        
        # Compute pre-grasp pose
        target_pose = PoseStamped()
        target_pose.header.frame_id = "root"
        target_pose.header.stamp = rospy.Time.now()
        
        # Position above object
        target_pose.pose.position.x = obj.position.x
        target_pose.pose.position.y = obj.position.y
        target_pose.pose.position.z = obj.position.z + self.pre_grasp_distance
        
        # Orientation - gripper pointing down
        # Euler ZYZ: azimuth, polar, rotation
        q = quaternion_from_euler(0, np.pi, 0, 'sxyz')
        target_pose.pose.orientation = Quaternion(*q)
        
        # Plan and execute
        rospy.loginfo(f"APPROACH: Planning to {object_id} at ({obj.position.x:.2f}, {obj.position.y:.2f}, {obj.position.z + self.pre_grasp_distance:.2f})")
        
        self.arm_group.set_pose_target(target_pose)
        success = self.arm_group.go(wait=True)
        self.arm_group.stop()
        self.arm_group.clear_pose_targets()
        
        if success:
            return True, f"Approached {object_id}"
        else:
            return False, f"Failed to plan path to {object_id}"
    
    def execute_align_yaw(self, object_id: str) -> Tuple[bool, str]:
        """
        Execute ALIGN_YAW tool - align gripper orientation with object.
        
        Rotates the gripper to match the object's yaw orientation.
        """
        with self.lock:
            if object_id not in self.objects:
                return False, f"Object {object_id} not found"
            
            obj = self.objects[object_id]
        
        # Get current pose
        current_pose = self.arm_group.get_current_pose()
        
        # Create target pose with adjusted yaw
        target_pose = PoseStamped()
        target_pose.header = current_pose.header
        target_pose.pose.position = current_pose.pose.position
        
        # Compute new orientation based on object yaw
        # Keep gripper pointing down, rotate around z-axis
        target_yaw = obj.yaw_orientation
        q = quaternion_from_euler(0, np.pi, target_yaw, 'sxyz')
        target_pose.pose.orientation = Quaternion(*q)
        
        rospy.loginfo(f"ALIGN_YAW: Aligning to {object_id} with yaw {np.degrees(target_yaw):.1f}°")
        
        # Plan and execute
        self.arm_group.set_pose_target(target_pose)
        success = self.arm_group.go(wait=True)
        self.arm_group.stop()
        self.arm_group.clear_pose_targets()
        
        if success:
            return True, f"Aligned with {object_id}"
        else:
            return False, f"Failed to align with {object_id}"
    
    def execute_grasp(self) -> Tuple[bool, str]:
        """
        Execute GRASP tool - close gripper.
        """
        rospy.loginfo("GRASP: Closing gripper")
        
        goal = SetFingersPositionGoal()
        goal.fingers.finger1 = self.FINGER_CLOSE
        goal.fingers.finger2 = self.FINGER_CLOSE
        goal.fingers.finger3 = self.FINGER_CLOSE
        
        self.finger_client.send_goal(goal)
        success = self.finger_client.wait_for_result(rospy.Duration(5.0))
        
        if success:
            return True, "Gripper closed"
        else:
            self.finger_client.cancel_all_goals()
            return False, "Gripper close timeout"
    
    def execute_release(self) -> Tuple[bool, str]:
        """
        Execute RELEASE tool - open gripper.
        """
        rospy.loginfo("RELEASE: Opening gripper")
        
        goal = SetFingersPositionGoal()
        goal.fingers.finger1 = self.FINGER_OPEN
        goal.fingers.finger2 = self.FINGER_OPEN
        goal.fingers.finger3 = self.FINGER_OPEN
        
        self.finger_client.send_goal(goal)
        success = self.finger_client.wait_for_result(rospy.Duration(5.0))
        
        if success:
            return True, "Gripper opened"
        else:
            self.finger_client.cancel_all_goals()
            return False, "Gripper open timeout"
    
    def move_to_home(self) -> Tuple[bool, str]:
        """Move arm to home position."""
        rospy.loginfo("Moving to home position")
        
        self.arm_group.set_named_target("Home")
        success = self.arm_group.go(wait=True)
        self.arm_group.stop()
        
        if success:
            return True, "Moved to home"
        else:
            return False, "Failed to move to home"
    
    def run(self):
        """Run the node."""
        rospy.spin()


def main():
    try:
        executor = ToolExecutor()
        executor.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        moveit_commander.roscpp_shutdown()


if __name__ == '__main__':
    main()
