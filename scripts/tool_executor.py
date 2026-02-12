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
import json
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
from std_msgs.msg import String

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
        # If true, add π/2 to object yaw so gripper fingers cross the narrow dimension
        # for a more stable grasp.  Set false if you want to align parallel to major axis.
        self.align_yaw_perpendicular = bool(rospy.get_param('tools/align/perpendicular', True))
        # Extra constant yaw offset (radians) applied on top of the object yaw during ALIGN_YAW.
        # Use this to fine-tune alignment calibration.
        self.align_yaw_extra_offset = float(rospy.get_param('tools/align/extra_yaw_offset', 0.0))
        # Velocity / acceleration scaling for ALIGN_YAW to avoid hitting joint limits aggressively.
        self.align_velocity_scale = float(rospy.get_param('tools/align/velocity_scaling', 0.4))
        self.align_accel_scale = float(rospy.get_param('tools/align/acceleration_scaling', 0.4))
        # Extra clearance above object for ALIGN_YAW (added on top of pre_grasp_distance).
        self.align_extra_clearance = float(rospy.get_param('tools/align/extra_clearance', 0.0))
        # Safe wrist joint range: clamp joint-6 to [-wrist_limit, +wrist_limit] radians.
        # Kinova j2n6s300 can go continuous but hardware may flash red past ~±5.5 rad.
        self.align_wrist_limit = float(rospy.get_param('tools/align/wrist_limit', 5.0))

        # Pixel-servo refinement params (after reaching grid-cell center)
        # Default OFF: opt-in via `tools/approach/pixel_servo/enabled:=true`
        self.servo_enabled = bool(rospy.get_param("tools/approach/pixel_servo/enabled", False))
        self.servo_max_steps = int(rospy.get_param("tools/approach/pixel_servo/max_steps", 5))
        self.servo_pixel_tol = float(rospy.get_param("tools/approach/pixel_servo/pixel_tolerance", 15.0))
        self.servo_gain = float(rospy.get_param("tools/approach/pixel_servo/gain", 0.6))
        self.servo_max_step_m = float(rospy.get_param("tools/approach/pixel_servo/max_step_m", 0.01))
        self.servo_probe_step_m = float(rospy.get_param("tools/approach/pixel_servo/probe_step_m", 0.01))
        self.servo_settle_s = float(rospy.get_param("tools/approach/pixel_servo/settle_time_s", 0.2))
        self.servo_missing_stop = int(rospy.get_param("tools/approach/pixel_servo/missing_object_stop_frames", 2))

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

        # Latest YOLO detections (for pixel servoing)
        self._latest_yolo_detections = []
        self._latest_yolo_stamp = 0.0
        
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

        # YOLO detections JSON (needed for pixel-servo refinement)
        self.yolo_dets_sub = rospy.Subscriber(
            "/yolo/detections_json",
            String,
            self.yolo_detections_callback,
            queue_size=1,
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

    def yolo_detections_callback(self, msg: String):
        """Cache latest YOLO detections JSON for pixel-servo refinement."""
        try:
            payload = json.loads(msg.data)
        except Exception:
            return
        dets = payload.get("detections", []) or []
        stamp = float(payload.get("stamp", rospy.Time.now().to_sec()))
        with self.lock:
            self._latest_yolo_detections = dets
            self._latest_yolo_stamp = stamp

    def _get_best_det(self, class_name: str):
        """Return best detection dict for class_name, or None."""
        with self.lock:
            dets = list(self._latest_yolo_detections)
        best = None
        best_c = -1.0
        for d in dets:
            if d.get("class") != class_name:
                continue
            c = float(d.get("conf", 0.0))
            if c > best_c:
                best_c = c
                best = d
        return best

    @staticmethod
    def _det_pick_xy(det: dict):
        """Return (u,v) pick point for a YOLO detection dict."""
        if det is None:
            return None
        uv = det.get("pick_xy", det.get("center_xy", [None, None]))
        try:
            u, v = int(uv[0]), int(uv[1])
        except Exception:
            return None
        return float(u), float(v)

    def _get_gripper_uv(self):
        """Pixel (u,v) of gripper from YOLO 'jaco' detection."""
        det = self._get_best_det("jaco")
        return self._det_pick_xy(det)

    def _wait_for_fresh_yolo(self, timeout: float = 2.0) -> bool:
        """
        Block until a YOLO detection with a newer timestamp arrives.
        This ensures we read pixels AFTER the arm has moved, not stale cached data.
        Returns True if fresh data arrived, False on timeout.
        """
        with self.lock:
            old_stamp = self._latest_yolo_stamp
        deadline = rospy.Time.now().to_sec() + timeout
        rate = rospy.Rate(30)  # poll at 30 Hz
        while not rospy.is_shutdown():
            now = rospy.Time.now().to_sec()
            if now > deadline:
                rospy.logwarn("pixel_servo: timed out waiting for fresh YOLO frame (%.1fs)", timeout)
                return False
            with self.lock:
                if self._latest_yolo_stamp > old_stamp:
                    return True
            rate.sleep()
        return False

    def _find_object_uv(self, obj_grid_label: str, expected_uv: Tuple[float, float], max_px_dist: float = 200.0):
        """
        Find the current YOLO detection for the target object.
        Preference order:
        1) same grid cell label (e.g., 'B2') if available
        2) nearest to expected_uv
        Returns (u,v) or None if not detected.
        """
        with self.lock:
            dets = list(self._latest_yolo_detections)
        objs = [d for d in dets if d.get("class") == "object"]
        if obj_grid_label:
            in_cell = [d for d in objs if d.get("grid_cell") == obj_grid_label]
            if in_cell:
                objs = in_cell
        best = None
        best_d2 = 1e18
        eu, ev = float(expected_uv[0]), float(expected_uv[1])
        for d in objs:
            uv = self._det_pick_xy(d)
            if uv is None:
                continue
            du = float(uv[0]) - eu
            dv = float(uv[1]) - ev
            d2 = du * du + dv * dv
            if d2 < best_d2:
                best_d2 = d2
                best = uv
        if best is None:
            return None
        if best_d2 > float(max_px_dist) ** 2:
            return None
        return best

    def _execute_cartesian_delta_xy(self, dx: float, dy: float, *, min_fraction: float = 0.95) -> bool:
        """Small Cartesian translation in target_frame, keeping orientation + z."""
        try:
            current = self.arm_group.get_current_pose().pose
        except Exception:
            rospy.logwarn("pixel_servo: cannot read current pose")
            return False

        target = Pose()
        target.position.x = current.position.x + float(dx)
        target.position.y = current.position.y + float(dy)
        target.position.z = current.position.z
        target.orientation = current.orientation

        # Clamp to safety bounds if enabled (assumes safety bounds are in same frame as target_frame)
        if self.safety_enabled:
            target.position.x = float(np.clip(target.position.x, self.x_min, self.x_max))
            target.position.y = float(np.clip(target.position.y, self.y_min, self.y_max))
            target.position.z = float(np.clip(target.position.z, self.z_min, self.z_max))

        # If clamping (or tiny dx/dy) results in no-op, treat as failure (important for Jacobian probing)
        dx_eff = float(target.position.x - current.position.x)
        dy_eff = float(target.position.y - current.position.y)
        if abs(dx_eff) + abs(dy_eff) < 1e-6:
            rospy.logwarn("pixel_servo: delta_xy no-op after clamping (dx_eff=%.6f, dy_eff=%.6f)", dx_eff, dy_eff)
            return False

        rospy.loginfo("pixel_servo: cartesian delta dx=%.4f dy=%.4f -> target (%.4f, %.4f, %.4f)",
                      dx_eff, dy_eff, target.position.x, target.position.y, target.position.z)

        try:
            self.arm_group.set_start_state_to_current_state()
        except Exception:
            pass
        try:
            # NOTE: only pass [target] as waypoints — do NOT include current pose
            plan, fraction = self.arm_group.compute_cartesian_path([target], 0.005, True)
            n_pts = len(plan.joint_trajectory.points) if plan is not None else 0
            rospy.loginfo("pixel_servo: cartesian path fraction=%.3f n_pts=%d", fraction, n_pts)
            if fraction >= float(min_fraction) and plan is not None and n_pts > 0:
                ok = self.arm_group.execute(plan, wait=True)
                self.arm_group.stop()
                self.arm_group.clear_pose_targets()
                if ok:
                    rospy.loginfo("pixel_servo: cartesian move executed OK")
                else:
                    rospy.logwarn("pixel_servo: cartesian execute returned False")
                return bool(ok)
            else:
                rospy.logwarn("pixel_servo: cartesian path insufficient (fraction=%.3f), trying fallback plan", fraction)
        except Exception as ex:
            rospy.logwarn("pixel_servo: cartesian path exception: %s, trying fallback", str(ex))

        # Fallback: plan to the same pose (still keeps orientation)
        try:
            ps = PoseStamped()
            ps.header.frame_id = self.target_frame
            ps.header.stamp = rospy.Time.now()
            ps.pose = target

            try:
                self.arm_group.set_start_state_to_current_state()
            except Exception:
                pass

            self.arm_group.set_pose_target(ps)
            self.arm_group.set_planning_time(5.0)
            self.arm_group.set_num_planning_attempts(3)
            plan = self.arm_group.plan()

            if isinstance(plan, tuple):
                plan_success = bool(plan[0])
                traj = plan[1]
            else:
                traj = plan
                plan_success = traj is not None and len(traj.joint_trajectory.points) > 0

            # restore defaults
            self.arm_group.set_planning_time(10.0)
            self.arm_group.set_num_planning_attempts(5)

            if not plan_success:
                self.arm_group.clear_pose_targets()
                rospy.logwarn("pixel_servo: fallback plan also failed")
                return False

            ok = self.arm_group.execute(traj, wait=True)
            self.arm_group.stop()
            self.arm_group.clear_pose_targets()
            rospy.loginfo("pixel_servo: fallback plan executed, ok=%s", str(ok))
            return bool(ok)
        except Exception:
            try:
                self.arm_group.stop()
                self.arm_group.clear_pose_targets()
            except Exception:
                pass
            return False

    def _estimate_pixel_jacobian_inv(self) -> Tuple[Optional[np.ndarray], str]:
        """
        Estimate 2x2 Jacobian inverse mapping pixel delta -> robot (dx,dy) in target_frame.
        Uses two small probe moves and YOLO 'jaco' pixel measurements.
        Returns (J_inv, reason). If J_inv is None, reason explains why.
        """
        uv0 = self._get_gripper_uv()
        if uv0 is None:
            return None, "no gripper detection (class 'jaco')"
        u0, v0 = float(uv0[0]), float(uv0[1])
        rospy.loginfo("pixel_servo Jacobian: gripper start pixel (%.0f, %.0f)", u0, v0)

        s = float(self.servo_probe_step_m)
        if s <= 1e-6:
            return None, "probe_step_m too small"

        # Probe x (try + then -)
        rospy.loginfo("pixel_servo Jacobian: probing X with step %.4f m", s)
        sx = +s
        if not self._execute_cartesian_delta_xy(+s, 0.0):
            sx = -s
            if not self._execute_cartesian_delta_xy(-s, 0.0):
                return None, "probe move in X failed (cartesian+fallback plan)"
        rospy.sleep(self.servo_settle_s)
        self._wait_for_fresh_yolo(timeout=2.0)
        uvx = self._get_gripper_uv()
        if uvx is None:
            return None, "gripper lost during X probe"
        ux, vx = float(uvx[0]), float(uvx[1])
        rospy.loginfo("pixel_servo Jacobian: after X probe pixel (%.0f, %.0f), delta=(%.1f, %.1f) px for %.4f m",
                      ux, vx, ux - u0, vx - v0, sx)
        # Return
        if not self._execute_cartesian_delta_xy(-sx, 0.0):
            rospy.logwarn("pixel_servo: failed to return after X probe (continuing).")
        rospy.sleep(self.servo_settle_s)
        self._wait_for_fresh_yolo(timeout=2.0)

        # Re-read baseline after returning (arm might not return to exact same spot)
        uv0_new = self._get_gripper_uv()
        if uv0_new is not None:
            u0, v0 = float(uv0_new[0]), float(uv0_new[1])
            rospy.loginfo("pixel_servo Jacobian: re-baseline after X return: (%.0f, %.0f)", u0, v0)

        # Probe y (try + then -)
        rospy.loginfo("pixel_servo Jacobian: probing Y with step %.4f m", s)
        sy = +s
        if not self._execute_cartesian_delta_xy(0.0, +s):
            sy = -s
            if not self._execute_cartesian_delta_xy(0.0, -s):
                return None, "probe move in Y failed (cartesian+fallback plan)"
        rospy.sleep(self.servo_settle_s)
        self._wait_for_fresh_yolo(timeout=2.0)
        uvy = self._get_gripper_uv()
        if uvy is None:
            return None, "gripper lost during Y probe"
        uy, vy = float(uvy[0]), float(uvy[1])
        rospy.loginfo("pixel_servo Jacobian: after Y probe pixel (%.0f, %.0f), delta=(%.1f, %.1f) px for %.4f m",
                      uy, vy, uy - u0, vy - v0, sy)
        # Return
        if not self._execute_cartesian_delta_xy(0.0, -sy):
            rospy.logwarn("pixel_servo: failed to return after Y probe (continuing).")
        rospy.sleep(self.servo_settle_s)
        self._wait_for_fresh_yolo(timeout=2.0)

        # J maps [dx,dy] -> [du,dv]
        du_dx = (ux - u0) / sx
        dv_dx = (vx - v0) / sx
        du_dy = (uy - u0) / sy
        dv_dy = (vy - v0) / sy
        J = np.array([[du_dx, du_dy], [dv_dx, dv_dy]], dtype=np.float64)
        det = float(np.linalg.det(J))
        rospy.loginfo("pixel_servo Jacobian: J=[[%.1f,%.1f],[%.1f,%.1f]] det=%.3f",
                      du_dx, du_dy, dv_dx, dv_dy, det)
        if abs(det) < 1e-9:
            return None, "Jacobian singular (det=%.6f; try increasing probe_step_m or ensure not at safety bounds)" % det
        J_inv = np.linalg.inv(J)
        rospy.loginfo("pixel_servo Jacobian: J_inv=[[%.6f,%.6f],[%.6f,%.6f]]",
                      J_inv[0, 0], J_inv[0, 1], J_inv[1, 0], J_inv[1, 1])
        return J_inv, "ok"

    def _pixel_servo_refine(self, obj: "ObjectState") -> Tuple[bool, str]:
        """
        After reaching grid-cell center, take a few small XY steps to reduce pixel error.
        Stop early if the object becomes undetected (assume occluded => on top).
        """
        if not self.servo_enabled:
            return True, "pixel_servo disabled"

        # Expected object pixel from state_builder (smoothed)
        try:
            expected_uv = (float(obj.bbox_center_x), float(obj.bbox_center_y))
        except Exception:
            expected_uv = (0.0, 0.0)
        grid_label = getattr(obj, "grid_label", "")

        rospy.loginfo("pixel_servo: starting refinement for %s (expected_uv=(%.0f,%.0f), grid=%s)",
                      obj.object_id, expected_uv[0], expected_uv[1], grid_label)

        J_inv, reason = self._estimate_pixel_jacobian_inv()
        if J_inv is None:
            return True, f"pixel_servo skipped ({reason})"

        missing = 0
        for k in range(max(0, int(self.servo_max_steps))):
            uv_g = self._get_gripper_uv()
            if uv_g is None:
                rospy.logwarn("pixel_servo step %d: gripper not detected", k)
                return True, "pixel_servo stopped (gripper not detected)"

            uv_o = self._find_object_uv(grid_label, expected_uv)
            if uv_o is None:
                missing += 1
                rospy.loginfo("pixel_servo step %d: object not detected (missing=%d/%d)",
                              k, missing, int(self.servo_missing_stop))
                if missing >= int(self.servo_missing_stop):
                    return True, "pixel_servo stop: object occluded (assume on top)"
                rospy.sleep(self.servo_settle_s)
                continue

            missing = 0
            expected_uv = uv_o  # update expected

            e = np.array([float(uv_o[0] - uv_g[0]), float(uv_o[1] - uv_g[1])], dtype=np.float64)
            err_px = float(np.linalg.norm(e))
            rospy.loginfo("pixel_servo step %d: gripper=(%.0f,%.0f) object=(%.0f,%.0f) error=%.1f px",
                          k, uv_g[0], uv_g[1], uv_o[0], uv_o[1], err_px)
            if err_px <= float(self.servo_pixel_tol):
                rospy.loginfo("pixel_servo step %d: within tolerance (%.1f <= %.1f)", k, err_px, self.servo_pixel_tol)
                return True, "pixel_servo done: within tolerance"

            dxy = J_inv @ e
            dx = float(self.servo_gain) * float(dxy[0])
            dy = float(self.servo_gain) * float(dxy[1])

            # Clamp per-step motion
            step = float(np.sqrt(dx * dx + dy * dy))
            max_step = float(self.servo_max_step_m)
            if step > max_step and step > 1e-9:
                scale = max_step / step
                dx *= scale
                dy *= scale

            rospy.loginfo("pixel_servo step %d: commanding dx=%.4f dy=%.4f m (raw_step=%.4f, max=%.4f)",
                          k, dx, dy, step, max_step)

            ok = self._execute_cartesian_delta_xy(dx, dy)
            rospy.sleep(self.servo_settle_s)
            self._wait_for_fresh_yolo(timeout=2.0)
            if not ok:
                rospy.logwarn("pixel_servo step %d: Cartesian step failed", k)
                return True, "pixel_servo stopped (Cartesian step failed)"

        return True, "pixel_servo done: max_steps reached"

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
            x_min = float(rospy.get_param("workspace/x_min"))
            x_max = float(rospy.get_param("workspace/x_max"))
            y_min = float(rospy.get_param("workspace/y_min"))
            y_max = float(rospy.get_param("workspace/y_max"))
            x = x_min + (col + 0.5) * (x_max - x_min) / 3.0
            y = y_min + (row + 0.5) * (y_max - y_min) / 3.0
            return float(x), float(y)

        # If object pose is invalid, fall back to grid-derived pose.
        # We use a constant object_z from workspace config, and tool adds pre_grasp_distance on top.
        obj_x = float(obj.position.x)
        obj_y = float(obj.position.y)
        obj_z = float(obj.position.z)
        if (not np.isfinite(obj_x)) or (not np.isfinite(obj_y)) or (not np.isfinite(obj_z)):
            xy = _grid_cell_center_xy(getattr(obj, "grid_cell", -1))
            if xy is None:
                return False, f"Invalid object pose for {object_id} and grid_cell missing; cannot APPROACH."
            obj_x, obj_y = float(xy[0]), float(xy[1])
            obj_z = float(rospy.get_param("workspace/object_z", 0.0))
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
                0.01,   # eef_step
                True,   # avoid_collisions
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
                    # Pixel-servo refinement (best-effort)
                    ok, msg = self._pixel_servo_refine(obj)
                    return True, f"Approached {object_id} (cartesian); {msg}"
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
            ok, msg = self._pixel_servo_refine(obj)
            return True, f"Approached {object_id}; {msg}"
        else:
            return False, f"Planned but failed to execute path to {object_id}"
    
    def execute_align_yaw(self, object_id: str) -> Tuple[bool, str]:
        """
        Execute ALIGN_YAW tool — move above the object AND rotate the gripper
        wrist to align with the object's mask-based orientation for a stable grasp.

        This always moves to the pre-grasp position above the object (same as
        APPROACH) but also sets the target orientation from the object's mask PCA.
        If the gripper is already above the object, only the rotation changes.

        Uses configurable velocity/acceleration scaling (default 0.4) to avoid
        aggressive motions that trigger Kinova joint-limit warnings (red lights).
        Wrist joint values are clamped to a safe range to prevent hardware faults.
        """
        with self.lock:
            if object_id not in self.objects:
                return False, f"Object {object_id} not found"
            obj = self.objects[object_id]

        # ---- Apply conservative velocity/accel scaling ----
        self.arm_group.set_max_velocity_scaling_factor(self.align_velocity_scale)
        self.arm_group.set_max_acceleration_scaling_factor(self.align_accel_scale)

        try:
            return self._execute_align_yaw_inner(object_id, obj)
        finally:
            # Restore default (full speed) scaling for other tools
            self.arm_group.set_max_velocity_scaling_factor(1.0)
            self.arm_group.set_max_acceleration_scaling_factor(1.0)

    def _clamp_wrist_joint(self, joint_values: list) -> list:
        """Clamp wrist (joint 6, index 5) to safe range to avoid Kinova red-light warnings."""
        if len(joint_values) >= 6:
            wrist_idx = 5
            val = joint_values[wrist_idx]
            limit = self.align_wrist_limit
            if val > limit:
                # Wrap into [-limit, limit] by subtracting 2π multiples
                val = val - 2.0 * np.pi * np.ceil((val - limit) / (2.0 * np.pi))
                rospy.logwarn("ALIGN_YAW: Clamped wrist from %.3f to %.3f rad (limit ±%.1f)",
                              joint_values[wrist_idx], val, limit)
                joint_values[wrist_idx] = val
            elif val < -limit:
                val = val + 2.0 * np.pi * np.ceil((-limit - val) / (2.0 * np.pi))
                rospy.logwarn("ALIGN_YAW: Clamped wrist from %.3f to %.3f rad (limit ±%.1f)",
                              joint_values[wrist_idx], val, limit)
                joint_values[wrist_idx] = val
        return joint_values

    def _retime_trajectory(self, trajectory, vel_scale, acc_scale):
        """Scale the time_from_start of every trajectory point to slow the motion down.

        MoveIt's velocity scaling only affects the planner, but trajectories from
        compute_cartesian_path or the OMPL planner may still be too fast.  This
        helper linearly stretches all timestamps by 1/vel_scale.
        """
        if vel_scale <= 0 or vel_scale >= 1.0:
            return trajectory
        factor = 1.0 / vel_scale
        for pt in trajectory.joint_trajectory.points:
            pt.time_from_start = rospy.Duration(pt.time_from_start.to_sec() * factor)
            # Scale velocities and accelerations too
            pt.velocities = tuple(v * vel_scale for v in pt.velocities) if pt.velocities else pt.velocities
            pt.accelerations = tuple(a * acc_scale for a in pt.accelerations) if pt.accelerations else pt.accelerations
        return trajectory

    def _execute_align_yaw_inner(self, object_id: str, obj) -> Tuple[bool, str]:
        """Inner implementation of ALIGN_YAW (called with velocity scaling already set)."""

        # ---- Resolve object XYZ (same logic as execute_approach) ----
        def _grid_cell_center_xy(grid_cell: int):
            try:
                grid_cell = int(grid_cell)
            except Exception:
                return None
            row = grid_cell // 3
            col = grid_cell % 3
            if not (0 <= row < 3 and 0 <= col < 3):
                return None
            x_min = float(rospy.get_param("workspace/x_min"))
            x_max = float(rospy.get_param("workspace/x_max"))
            y_min = float(rospy.get_param("workspace/y_min"))
            y_max = float(rospy.get_param("workspace/y_max"))
            x = x_min + (col + 0.5) * (x_max - x_min) / 3.0
            y = y_min + (row + 0.5) * (y_max - y_min) / 3.0
            return float(x), float(y)

        obj_x = float(obj.position.x)
        obj_y = float(obj.position.y)
        obj_z = float(obj.position.z)
        if (not np.isfinite(obj_x)) or (not np.isfinite(obj_y)) or (not np.isfinite(obj_z)):
            xy = _grid_cell_center_xy(getattr(obj, "grid_cell", -1))
            if xy is None:
                return False, f"Invalid object pose for {object_id} and grid_cell missing; cannot ALIGN_YAW."
            obj_x, obj_y = float(xy[0]), float(xy[1])
            obj_z = float(rospy.get_param("workspace/object_z", 0.0))
            rospy.logwarn(f"ALIGN_YAW: Using grid-cell center pose for {object_id}.")

        # Transform frame if needed
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

        # Target XY = object, Z = above object (with optional extra clearance)
        tx = obj_pose.pose.position.x
        ty = obj_pose.pose.position.y
        tz = obj_pose.pose.position.z + self.pre_grasp_distance + self.align_extra_clearance

        if (not np.isfinite(tx)) or (not np.isfinite(ty)) or (not np.isfinite(tz)):
            return False, f"Invalid target position for {object_id} (nan/inf)."

        # Safety clamp
        if self.safety_enabled:
            tx = max(self.x_min, min(self.x_max, tx))
            ty = max(self.y_min, min(self.y_max, ty))
            tz = max(self.z_min, min(self.z_max, tz))

        # ---- Compute target yaw ----
        obj_yaw = float(obj.yaw_orientation)  # major-axis direction (robot frame)

        if self.align_yaw_perpendicular:
            target_yaw = obj_yaw + np.pi / 2.0
        else:
            target_yaw = obj_yaw

        target_yaw += self.align_yaw_extra_offset
        target_yaw = float(np.arctan2(np.sin(target_yaw), np.cos(target_yaw)))

        # Current gripper yaw (for logging)
        current_pose = self.arm_group.get_current_pose()
        cur_q = current_pose.pose.orientation
        cur_siny = 2.0 * (cur_q.w * cur_q.z + cur_q.x * cur_q.y)
        cur_cosy = 1.0 - 2.0 * (cur_q.y ** 2 + cur_q.z ** 2)
        cur_yaw = float(np.arctan2(cur_siny, cur_cosy))
        delta = float(np.arctan2(np.sin(target_yaw - cur_yaw), np.cos(target_yaw - cur_yaw)))

        clearance = self.pre_grasp_distance + self.align_extra_clearance

        if obj_yaw == 0.0:
            rospy.logwarn(
                "ALIGN_YAW: obj_yaw is exactly 0.0 for %s — mask yaw may not have been computed.",
                object_id,
            )

        rospy.loginfo(
            "ALIGN_YAW: obj=%s  pos=(%.3f,%.3f,%.3f)  obj_yaw=%.1f°  perp=%s  "
            "target_yaw=%.1f°  current_yaw=%.1f°  delta=%.1f°  clearance=%.3fm  "
            "vel_scale=%.2f  wrist_limit=±%.1frad",
            object_id, tx, ty, tz,
            np.degrees(obj_yaw),
            self.align_yaw_perpendicular,
            np.degrees(target_yaw),
            np.degrees(cur_yaw),
            np.degrees(delta),
            clearance,
            self.align_velocity_scale,
            self.align_wrist_limit,
        )

        # ---- Build target pose: position above object + aligned orientation ----
        target_pose = PoseStamped()
        target_pose.header.frame_id = self.target_frame
        target_pose.header.stamp = rospy.Time.now()
        target_pose.pose.position.x = tx
        target_pose.pose.position.y = ty
        target_pose.pose.position.z = tz

        # Gripper pointing down (pitch=π), rotated by target_yaw around z
        q = quaternion_from_euler(0, np.pi, target_yaw, 'sxyz')
        target_pose.pose.orientation = Quaternion(*q)

        # ---- Attempt 1: Cartesian path (position + orientation) ----
        self.arm_group.set_start_state_to_current_state()
        try:
            waypoints = [target_pose.pose]  # only target — MoveIt starts from current
            cart_plan, fraction = self.arm_group.compute_cartesian_path(
                waypoints,
                0.01,   # eef_step
                True,   # avoid_collisions
            )
            npts = 0
            if cart_plan is not None and hasattr(cart_plan, "joint_trajectory"):
                npts = len(cart_plan.joint_trajectory.points)
            if fraction >= 0.90 and npts > 0:
                # Retime trajectory for safety
                cart_plan = self._retime_trajectory(
                    cart_plan, self.align_velocity_scale, self.align_accel_scale
                )
                rospy.loginfo(
                    "ALIGN_YAW: Cartesian path fraction=%.2f (%d pts). Executing...",
                    fraction, npts,
                )
                success = self.arm_group.execute(cart_plan, wait=True)
                self.arm_group.stop()
                self.arm_group.clear_pose_targets()
                if success:
                    return True, (
                        f"Aligned with {object_id} — moved above object and "
                        f"gripper yaw set to {np.degrees(target_yaw):.1f}°"
                    )
                rospy.logwarn("ALIGN_YAW: Cartesian execution failed, falling back to planner.")
            else:
                rospy.logwarn(
                    "ALIGN_YAW: Cartesian fraction=%.2f (%d pts), falling back to planner.",
                    fraction, npts,
                )
        except Exception as e:
            rospy.logwarn("ALIGN_YAW: Cartesian path error (%s), falling back.", str(e))

        # ---- Attempt 2: OMPL planner (full 6-DOF pose target) ----
        prev_goal_tol = self.arm_group.get_goal_joint_tolerance()
        self.arm_group.set_goal_orientation_tolerance(0.05)  # ~2.9° — relaxed for reachability
        self.arm_group.set_planning_time(10.0)
        self.arm_group.set_num_planning_attempts(8)

        self.arm_group.set_pose_target(target_pose)
        plan = self.arm_group.plan()
        if isinstance(plan, tuple):
            plan_success = plan[0]
            trajectory = plan[1]
        else:
            trajectory = plan
            plan_success = trajectory is not None and len(trajectory.joint_trajectory.points) > 0

        if plan_success:
            rospy.loginfo("ALIGN_YAW: OMPL plan found. Executing...")
            success = self.arm_group.execute(trajectory, wait=True)
            self.arm_group.stop()
        else:
            success = False
            rospy.logwarn("ALIGN_YAW: OMPL planning failed for full pose target.")
        self.arm_group.clear_pose_targets()
        self.arm_group.set_planning_time(10.0)
        self.arm_group.set_num_planning_attempts(5)
        self.arm_group.set_goal_joint_tolerance(prev_goal_tol)

        if success:
            return True, (
                f"Aligned with {object_id} — moved above object and "
                f"gripper yaw set to {np.degrees(target_yaw):.1f}°"
            )

        # ---- Attempt 3: Two-step fallback — move above first, then rotate wrist ----
        rospy.logwarn("ALIGN_YAW: Full pose planning failed. Trying two-step: move then rotate...")

        # Step A: Move above object (keep current orientation)
        step_a_pose = PoseStamped()
        step_a_pose.header = target_pose.header
        step_a_pose.pose.position = target_pose.pose.position
        step_a_pose.pose.orientation = current_pose.pose.orientation  # keep current orientation

        self.arm_group.set_pose_target(step_a_pose)
        move_ok = self.arm_group.go(wait=True)
        self.arm_group.stop()
        self.arm_group.clear_pose_targets()

        if not move_ok:
            return False, (
                f"Failed to move above {object_id} at ({tx:.3f},{ty:.3f},{tz:.3f})"
            )

        # Step B: Rotate wrist joint to target yaw (with clamping)
        try:
            joint_values = list(self.arm_group.get_current_joint_values())
            if len(joint_values) >= 6:
                wrist_idx = 5
                current_wrist = joint_values[wrist_idx]
                # Re-compute delta from the post-move pose
                post_pose = self.arm_group.get_current_pose()
                pq = post_pose.pose.orientation
                p_siny = 2.0 * (pq.w * pq.z + pq.x * pq.y)
                p_cosy = 1.0 - 2.0 * (pq.y ** 2 + pq.z ** 2)
                post_yaw = float(np.arctan2(p_siny, p_cosy))
                wrist_delta = float(np.arctan2(
                    np.sin(target_yaw - post_yaw),
                    np.cos(target_yaw - post_yaw),
                ))
                new_wrist = current_wrist + wrist_delta
                joint_values[wrist_idx] = new_wrist
                # Clamp to safe range
                joint_values = self._clamp_wrist_joint(joint_values)

                rospy.loginfo(
                    "ALIGN_YAW step B: rotating wrist by %.1f° (%.3f -> %.3f rad)",
                    np.degrees(wrist_delta), current_wrist, joint_values[wrist_idx],
                )
                self.arm_group.set_joint_value_target(joint_values)
                rot_ok = self.arm_group.go(wait=True)
                self.arm_group.stop()
                self.arm_group.clear_pose_targets()
                if rot_ok:
                    return True, (
                        f"Aligned with {object_id} (two-step) — above object + "
                        f"wrist rotated {np.degrees(wrist_delta):.1f}°"
                    )
        except Exception as e:
            rospy.logerr("ALIGN_YAW step B error: %s", str(e))

        return False, (
            f"Moved above {object_id} but failed to rotate gripper "
            f"(target yaw {np.degrees(target_yaw):.1f}°)"
        )
    
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
