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

        # Safety bounds (in root frame by default)
        self.safety_enabled = rospy.get_param('safety_bounds/enabled', False)
        self.safety_frame = rospy.get_param('safety_bounds/frame_id', 'root')
        self.x_min = rospy.get_param('safety_bounds/x_min', rospy.get_param('workspace/x_min', -1e9))
        self.x_max = rospy.get_param('safety_bounds/x_max', rospy.get_param('workspace/x_max', 1e9))
        self.y_min = rospy.get_param('safety_bounds/y_min', rospy.get_param('workspace/y_min', -1e9))
        self.y_max = rospy.get_param('safety_bounds/y_max', rospy.get_param('workspace/y_max', 1e9))
        self.z_min = rospy.get_param('safety_bounds/z_min', rospy.get_param('workspace/z_min', -1e9))
        self.z_max = rospy.get_param('safety_bounds/z_max', rospy.get_param('workspace/z_max', 1e9))
        self.add_moveit_walls = rospy.get_param('safety_bounds/add_moveit_walls', False)
        self.wall_thickness = rospy.get_param('safety_bounds/wall_thickness', 0.02)
        
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

        # Frames
        self.planning_frame = self.arm_group.get_planning_frame()
        # Target frame for pose goals (defaults to MoveIt's planning frame)
        self.target_frame = self.planning_frame
        # Use current orientation for approach (instead of forcing straight-down which may be unreachable)
        self.use_current_orientation = rospy.get_param('~use_current_orientation', True)
        rospy.loginfo(f"MoveIt planning_frame={self.planning_frame}, target_frame={self.target_frame}, use_current_orientation={self.use_current_orientation}")
        
        # Set end effector
        self.arm_group.set_end_effector_link(f"{self.robot_type}_end_effector")

        # Add MoveIt planning-scene safety walls (optional but recommended)
        if self.safety_enabled and self.add_moveit_walls:
            try:
                self._add_safety_walls()
            except Exception as e:
                rospy.logwarn(f"Failed to add safety walls: {e}")
        
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

    def _in_bounds(self, x: float, y: float, z: float) -> bool:
        return (self.x_min <= x <= self.x_max and
                self.y_min <= y <= self.y_max and
                self.z_min <= z <= self.z_max)

    def _bounds_str(self) -> str:
        return (f"x[{self.x_min:.3f},{self.x_max:.3f}] "
                f"y[{self.y_min:.3f},{self.y_max:.3f}] "
                f"z[{self.z_min:.3f},{self.z_max:.3f}] (frame={self.safety_frame})")

    def _add_safety_walls(self):
        """
        Add 6 thin boxes around the allowed workspace.
        This helps MoveIt avoid planning outside the region.
        """
        frame = self.safety_frame
        # Center and size of allowed region
        cx = 0.5 * (self.x_min + self.x_max)
        cy = 0.5 * (self.y_min + self.y_max)
        cz = 0.5 * (self.z_min + self.z_max)
        sx = (self.x_max - self.x_min)
        sy = (self.y_max - self.y_min)
        sz = (self.z_max - self.z_min)

        t = float(self.wall_thickness)
        # walls: x-min, x-max, y-min, y-max, z-min (floor), z-max (ceiling)
        walls = [
            ("prime_wall_xmin", (self.x_min - t/2.0, cy, cz), (t, sy + 2*t, sz + 2*t)),
            ("prime_wall_xmax", (self.x_max + t/2.0, cy, cz), (t, sy + 2*t, sz + 2*t)),
            ("prime_wall_ymin", (cx, self.y_min - t/2.0, cz), (sx + 2*t, t, sz + 2*t)),
            ("prime_wall_ymax", (cx, self.y_max + t/2.0, cz), (sx + 2*t, t, sz + 2*t)),
            ("prime_wall_zmin", (cx, cy, self.z_min - t/2.0), (sx + 2*t, sy + 2*t, t)),
            ("prime_wall_zmax", (cx, cy, self.z_max + t/2.0), (sx + 2*t, sy + 2*t, t)),
        ]

        rospy.loginfo(f"Adding MoveIt safety walls with bounds: {self._bounds_str()}")
        for name, (x, y, z), (bx, by, bz) in walls:
            pose = PoseStamped()
            pose.header.frame_id = frame
            pose.header.stamp = rospy.Time.now()
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = z
            pose.pose.orientation.w = 1.0
            self.scene.add_box(name, pose, size=(bx, by, bz))

        rospy.sleep(1.0)  # allow PlanningScene to update
    
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
        Uses position-only goal (joint-space) with multiple retries and relaxed orientation
        for maximum reliability on JACO2.
        """
        with self.lock:
            if object_id not in self.objects:
                return False, f"Object {object_id} not found"
            
            obj = self.objects[object_id]

        # Fallback mapping (NO AprilTags): if state_builder provided NaN pose, use grid cell centers.
        # This keeps APPROACH usable even if object metric pose is unavailable.
        def _grid_cell_center_xy(grid_cell: int):
            try:
                grid_cell = int(grid_cell)
            except Exception:
                return None
            row = grid_cell // 3
            col = grid_cell % 3
            if not (0 <= row < 3 and 0 <= col < 3):
                return None
            use_safety_xy = bool(rospy.get_param("workspace_metric/use_safety_bounds_xy", True))
            if use_safety_xy:
                x_min = float(rospy.get_param("safety_bounds/x_min"))
                x_max = float(rospy.get_param("safety_bounds/x_max"))
                y_min = float(rospy.get_param("safety_bounds/y_min"))
                y_max = float(rospy.get_param("safety_bounds/y_max"))
            else:
                x_min = float(rospy.get_param("workspace_metric/x_min"))
                x_max = float(rospy.get_param("workspace_metric/x_max"))
                y_min = float(rospy.get_param("workspace_metric/y_min"))
                y_max = float(rospy.get_param("workspace_metric/y_max"))
            x = x_min + (col + 0.5) * (x_max - x_min) / 3.0
            y = y_min + (row + 0.5) * (y_max - y_min) / 3.0
            return float(x), float(y)

        # If object pose is invalid, fall back to grid-derived pose.
        # We use a constant object_z from workspace_metric, and tool adds pre_grasp_distance on top.
        obj_x = float(obj.position.x)
        obj_y = float(obj.position.y)
        obj_z = float(obj.position.z)
        if (not np.isfinite(obj_x)) or (not np.isfinite(obj_y)) or (not np.isfinite(obj_z)):
            xy = _grid_cell_center_xy(getattr(obj, "grid_cell", -1))
            if xy is None:
                return False, f"Invalid object pose for {object_id} and grid_cell missing; cannot APPROACH."
            obj_x, obj_y = float(xy[0]), float(xy[1])
            obj_z = float(rospy.get_param("workspace_metric/object_z", 0.0))
            rospy.logwarn(f"APPROACH: Using grid-cell center pose for {object_id} at ({obj_x:.3f},{obj_y:.3f},{obj_z:.3f}).")
        
        # Object position comes from symbolic state (state.header.frame_id). Transform if needed.
        obj_frame = None
        with self.lock:
            if self.current_state is not None and getattr(self.current_state, "header", None) is not None:
                obj_frame = self.current_state.header.frame_id
        if not obj_frame:
            obj_frame = self.target_frame

        obj_pose = PoseStamped()
        obj_pose.header.frame_id = obj_frame
        obj_pose.header.stamp = rospy.Time(0)
        obj_pose.pose.position.x = obj_x
        obj_pose.pose.position.y = obj_y
        obj_pose.pose.position.z = obj_z
        obj_pose.pose.orientation.w = 1.0

        try:
            if obj_frame != self.target_frame:
                obj_pose = self.tf_buffer.transform(obj_pose, self.target_frame, rospy.Duration(0.5))
        except Exception as e:
            return False, f"Failed TF transform {obj_frame}->{self.target_frame}: {e}"

        # Target position = above object
        tx = obj_pose.pose.position.x
        ty = obj_pose.pose.position.y
        tz = obj_pose.pose.position.z + self.pre_grasp_distance

        # Guard against invalid perception outputs
        if (not np.isfinite(tx)) or (not np.isfinite(ty)) or (not np.isfinite(tz)):
            return False, f"Invalid target position for {object_id} (nan/inf). Check state_builder object pose output."

        if self.safety_enabled:
            # Clamp to safety bounds instead of rejecting
            clamped = False
            if tx < self.x_min:
                tx = self.x_min; clamped = True
            elif tx > self.x_max:
                tx = self.x_max; clamped = True
            if ty < self.y_min:
                ty = self.y_min; clamped = True
            elif ty > self.y_max:
                ty = self.y_max; clamped = True
            if tz < self.z_min:
                tz = self.z_min; clamped = True
            elif tz > self.z_max:
                tz = self.z_max; clamped = True
            if clamped:
                rospy.logwarn(
                    f"APPROACH: Target clamped to safety bounds {self._bounds_str()}, "
                    f"now ({tx:.3f},{ty:.3f},{tz:.3f})"
                )

        # Get current pose — keep orientation, only change x,y (and z to approach height)
        try:
            current_pose = self.arm_group.get_current_pose()
        except Exception as e:
            return False, f"Failed to get current pose: {e}"

        target_pose = PoseStamped()
        target_pose.header.frame_id = self.target_frame
        target_pose.header.stamp = rospy.Time.now()
        # Keep current orientation exactly
        target_pose.pose.orientation = current_pose.pose.orientation
        # Move to object x,y with approach z
        target_pose.pose.position.x = tx
        target_pose.pose.position.y = ty
        target_pose.pose.position.z = tz

        rospy.loginfo(
            f"APPROACH: Target for {object_id}: ({tx:.3f}, {ty:.3f}, {tz:.3f}) "
            f"orientation kept from current pose, frame {self.target_frame}"
        )

        # Always plan from the latest measured state
        try:
            self.arm_group.set_start_state_to_current_state()
        except Exception:
            pass

        # Prefer a short Cartesian translation first (more stable configs, keeps orientation)
        try:
            waypoints = [current_pose.pose, target_pose.pose]
            cart_plan, fraction = self.arm_group.compute_cartesian_path(
                waypoints,
                eef_step=0.01,
                jump_threshold=0.0
            )
            if cart_plan is not None and hasattr(cart_plan, "joint_trajectory"):
                npts = len(cart_plan.joint_trajectory.points)
            else:
                npts = 0
            if fraction >= 0.95 and npts > 0:
                rospy.loginfo(f"APPROACH: Cartesian path fraction={fraction:.2f} ({npts} pts). Executing...")
                success = self.arm_group.execute(cart_plan, wait=True)
                self.arm_group.stop()
                self.arm_group.clear_pose_targets()
                if success:
                    return True, f"Approached {object_id} (cartesian)"
                return False, f"Cartesian plan found but failed to execute path to {object_id}"
            else:
                rospy.logwarn(f"APPROACH: Cartesian path fraction={fraction:.2f} ({npts} pts). Falling back to planner.")
        except Exception as e:
            rospy.logwarn(f"APPROACH: Cartesian path failed ({e}). Falling back to planner.")

        # Fallback: OMPL planner to same 6-DOF pose
        self.arm_group.set_pose_target(target_pose)

        # Increase planning time and attempts for reliability
        self.arm_group.set_planning_time(15.0)
        self.arm_group.set_num_planning_attempts(10)

        # Plan
        plan = self.arm_group.plan()
        if isinstance(plan, tuple):
            plan_success = plan[0]
            trajectory = plan[1]
            error_code = plan[3] if len(plan) > 3 else None
        else:
            trajectory = plan
            plan_success = trajectory is not None and len(trajectory.joint_trajectory.points) > 0
            error_code = None

        # Restore defaults
        self.arm_group.set_planning_time(10.0)
        self.arm_group.set_num_planning_attempts(5)

        if not plan_success:
            self.arm_group.clear_pose_targets()
            err_msg = f"Failed to plan path to {object_id} at ({tx:.3f},{ty:.3f},{tz:.3f})"
            if error_code is not None:
                err_msg += f" (MoveIt error_code={error_code})"
            rospy.logwarn(err_msg)
            return False, err_msg

        rospy.loginfo(f"APPROACH: Plan found with {len(trajectory.joint_trajectory.points)} waypoints. Executing...")
        success = self.arm_group.execute(trajectory, wait=True)
        self.arm_group.stop()
        self.arm_group.clear_pose_targets()
        
        if success:
            return True, f"Approached {object_id}"
        else:
            return False, f"Planned but failed to execute path to {object_id}"
    
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
        if self.safety_enabled:
            px = current_pose.pose.position.x
            py = current_pose.pose.position.y
            pz = current_pose.pose.position.z
            if not self._in_bounds(px, py, pz):
                return False, f"Current pose out of safety bounds ({self._bounds_str()}), got ({px:.3f},{py:.3f},{pz:.3f})"
        
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
