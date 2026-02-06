#!/usr/bin/env python3
"""
PRIME State Builder (fresh start, NO AprilTags)

Builds PRIME's SymbolicState from:
- YOLO detections JSON (workspace bbox + grid bbox + detections)
- Kinova tool pose (optional)
- Control mode (optional)

Outputs:
- /prime/symbolic_state (prime_ros/SymbolicState)
- /prime/candidate_objects (prime_ros/CandidateSet)

Key design (no AprilTags):
- Objects and gripper are assigned to grid cells A1..C3 purely in image space.
- Each object also gets a usable metric pose (x,y,z) by mapping the 3x3 grid onto a
  configured metric rectangle in the robot base frame.
  This makes tool calls stable and removes camera calibration dependencies.
"""

import json
from collections import deque
from threading import RLock
from typing import Dict, Optional, Tuple

import numpy as np
import rospy
from geometry_msgs.msg import Point, PoseStamped
from std_msgs.msg import Header, String
from sensor_msgs.msg import Image

try:
    from prime_ros.msg import SymbolicState, ObjectState, ControlMode, CandidateSet
    from prime_ros.srv import GetSymbolicState, GetSymbolicStateResponse
    MSGS_AVAILABLE = True
except Exception:
    rospy.logwarn("PRIME messages not built yet. Run catkin_make.")
    MSGS_AVAILABLE = False


class StateBuilder:
    def __init__(self):
        rospy.init_node("state_builder", anonymous=False)

        self.lock = RLock()

        # Robot / frames
        self.robot_type = rospy.get_param("robot/type", "j2n6s300")
        self.state_frame_id = rospy.get_param("~state_frame_id", rospy.get_param("workspace_metric/frame_id", "root"))

        # Workspace (grid only)
        self.grid_rows = int(rospy.get_param("workspace/grid_rows", 3))
        self.grid_cols = int(rospy.get_param("workspace/grid_cols", 3))

        # Metric mapping from grid -> (x,y,z)
        self.metric_enabled = bool(rospy.get_param("workspace_metric/enabled", True))
        self.metric_frame_id = rospy.get_param("workspace_metric/frame_id", "root")
        use_safety_xy = bool(rospy.get_param("workspace_metric/use_safety_bounds_xy", True))
        if use_safety_xy:
            self.metric_x_min = float(rospy.get_param("safety_bounds/x_min"))
            self.metric_x_max = float(rospy.get_param("safety_bounds/x_max"))
            self.metric_y_min = float(rospy.get_param("safety_bounds/y_min"))
            self.metric_y_max = float(rospy.get_param("safety_bounds/y_max"))
        else:
            self.metric_x_min = float(rospy.get_param("workspace_metric/x_min"))
            self.metric_x_max = float(rospy.get_param("workspace_metric/x_max"))
            self.metric_y_min = float(rospy.get_param("workspace_metric/y_min"))
            self.metric_y_max = float(rospy.get_param("workspace_metric/y_max"))
        self.metric_object_z = float(rospy.get_param("workspace_metric/object_z", 0.0))

        # State builder parameters
        self.update_rate = float(rospy.get_param("state_builder/update_rate", 10.0))
        self.history_length = int(rospy.get_param("state_builder/gripper_history_length", 10))
        self.position_threshold = float(rospy.get_param("state_builder/position_threshold", 0.02))

        # Detection filtering + tracking (stable IDs)
        self.min_object_confidence = float(rospy.get_param("state_builder/min_object_confidence", 0.5))
        self.track_max_age = float(rospy.get_param("state_builder/track_max_age", 1.0))  # seconds
        self.track_max_pixel_dist = float(rospy.get_param("state_builder/track_max_pixel_dist", 80.0))  # px
        self.track_smoothing_alpha = float(rospy.get_param("state_builder/track_smoothing_alpha", 0.35))
        self.tracks: Dict[str, dict] = {}
        self.next_track_id = 1

        # Inputs (cached)
        self.latest_detections = []
        self.latest_workspace_bbox = None
        self.latest_grid_bbox = None
        self.latest_yolo_image = None
        self.gripper_pose: Optional[PoseStamped] = None
        self.gripper_history = deque(maxlen=self.history_length)
        self.control_mode: Optional[ControlMode] = None

        # Behavior toggles
        self.use_yolo_grid = bool(rospy.get_param("~use_yolo_grid", True))
        self.gripper_cell_from_yolo = bool(rospy.get_param("~gripper_cell_from_yolo", True))

        # Subscribers
        driver_prefix = f"/{self.robot_type}_driver"
        self.pose_sub = rospy.Subscriber(
            f"{driver_prefix}/out/tool_pose",
            PoseStamped,
            self.gripper_pose_callback,
            queue_size=1,
        )
        if MSGS_AVAILABLE:
            self.mode_sub = rospy.Subscriber("/prime/control_mode", ControlMode, self.control_mode_callback, queue_size=1)

        self.yolo_img_sub = rospy.Subscriber("/yolo/image_with_bboxes", Image, self.yolo_image_callback, queue_size=1)
        self.yolo_dets_sub = rospy.Subscriber("/yolo/detections_json", String, self.yolo_detections_callback, queue_size=1)

        # Publishers / service
        if MSGS_AVAILABLE:
            self.state_pub = rospy.Publisher("/prime/symbolic_state", SymbolicState, queue_size=10)
            self.candidates_pub = rospy.Publisher("/prime/candidate_objects", CandidateSet, queue_size=10)
            self.state_service = rospy.Service("/prime/get_symbolic_state", GetSymbolicState, self.handle_get_state)

        # Timer
        self.timer = rospy.Timer(rospy.Duration(1.0 / max(1e-3, self.update_rate)), self.update_state)

        rospy.loginfo("State Builder initialized (NO AprilTags).")
        rospy.loginfo(
            "Metric grid mapping: enabled=%s frame=%s x[%.3f,%.3f] y[%.3f,%.3f] object_z=%.3f",
            str(self.metric_enabled),
            str(self.metric_frame_id),
            self.metric_x_min,
            self.metric_x_max,
            self.metric_y_min,
            self.metric_y_max,
            self.metric_object_z,
        )

    # -----------------------
    # Callbacks
    # -----------------------
    def gripper_pose_callback(self, msg: PoseStamped):
        with self.lock:
            self.gripper_pose = msg
            p = msg.pose.position
            current = Point(x=p.x, y=p.y, z=p.z)
            if not self.gripper_history:
                self.gripper_history.append(current)
                return
            last = self.gripper_history[-1]
            dist = float(np.sqrt((current.x - last.x) ** 2 + (current.y - last.y) ** 2 + (current.z - last.z) ** 2))
            if dist > self.position_threshold:
                self.gripper_history.append(current)

    def control_mode_callback(self, msg: ControlMode):
        with self.lock:
            self.control_mode = msg

    def yolo_image_callback(self, msg: Image):
        with self.lock:
            self.latest_yolo_image = msg

    def yolo_detections_callback(self, msg: String):
        try:
            payload = json.loads(msg.data)
        except Exception as e:
            rospy.logwarn_throttle(1.0, f"Failed to parse /yolo/detections_json: {e}")
            return
        with self.lock:
            self.latest_detections = payload.get("detections", []) or []
            self.latest_workspace_bbox = payload.get("workspace_bbox_xyxy", None)
            self.latest_grid_bbox = payload.get("grid_bbox_xyxy", self.latest_workspace_bbox)

    # -----------------------
    # Grid + metric mapping
    # -----------------------
    @staticmethod
    def pixel_to_grid_label(cx, cy, ws_bbox_xyxy):
        """Convert pixel coordinate to A1..C3 based on grid bbox in image space."""
        if ws_bbox_xyxy is None:
            return None, None, None, None
        x1, y1, x2, y2 = ws_bbox_xyxy
        w = max(1.0, float(x2 - x1))
        h = max(1.0, float(y2 - y1))
        col = int((float(cx) - x1) / (w / 3.0))
        row = int((float(cy) - y1) / (h / 3.0))
        col = int(np.clip(col, 0, 2))
        row = int(np.clip(row, 0, 2))
        row_letter = ["A", "B", "C"][row]
        label = f"{row_letter}{col + 1}"
        cell_index = row * 3 + col
        return label, cell_index, row, col

    @staticmethod
    def cell_index_to_label(cell_index, grid_cols=3):
        """0..8 -> A1..C3"""
        try:
            cell_index = int(cell_index)
        except Exception:
            return None
        if cell_index < 0:
            return None
        row = cell_index // grid_cols
        col = cell_index % grid_cols
        if not (0 <= row <= 2 and 0 <= col <= 2):
            return None
        return f"{['A','B','C'][row]}{col+1}"

    def grid_cell_center_xy(self, row: int, col: int) -> Tuple[float, float]:
        """Map (row,col) to metric (x,y) center within the configured rectangle."""
        row = int(np.clip(row, 0, self.grid_rows - 1))
        col = int(np.clip(col, 0, self.grid_cols - 1))
        x = self.metric_x_min + (col + 0.5) * (self.metric_x_max - self.metric_x_min) / float(self.grid_cols)
        y = self.metric_y_min + (row + 0.5) * (self.metric_y_max - self.metric_y_min) / float(self.grid_rows)
        return float(x), float(y)

    # -----------------------
    # Object tracking + state
    # -----------------------
    def update_detected_objects_from_yolo(self) -> Dict[str, ObjectState]:
        if (not MSGS_AVAILABLE) or (not self.use_yolo_grid):
            return {}
        if not self.latest_detections or self.latest_grid_bbox is None:
            return {}

        now = rospy.Time.now().to_sec()

        # Expire stale tracks
        for oid in list(self.tracks.keys()):
            if (now - float(self.tracks[oid].get("last_seen", 0.0))) > self.track_max_age:
                del self.tracks[oid]

        # Measurements
        measured = []
        for det in self.latest_detections:
            if det.get("class") != "object":
                continue
            conf = float(det.get("conf", 0.0))
            if conf < self.min_object_confidence:
                continue
            px, py = det.get("pick_xy", det.get("center_xy", [None, None]))
            if px is None or py is None:
                continue

            grid_label, cell_index, row, col = self.pixel_to_grid_label(px, py, self.latest_grid_bbox)
            if grid_label is None:
                continue

            measured.append(
                {
                    "px": int(px),
                    "py": int(py),
                    "conf": conf,
                    "grid_label": str(grid_label),
                    "grid_cell": int(cell_index),
                    "grid_row": int(row),
                    "grid_col": int(col),
                }
            )

        measured.sort(key=lambda m: m["conf"], reverse=True)

        # Greedy match by pixel distance
        unmatched_tracks = set(self.tracks.keys())
        max_d2 = float(self.track_max_pixel_dist) ** 2
        alpha = float(self.track_smoothing_alpha)

        for m in measured:
            best_oid = None
            best_d2 = 1e18
            for oid in unmatched_tracks:
                tr = self.tracks[oid]
                dx = float(m["px"]) - float(tr.get("px", 0))
                dy = float(m["py"]) - float(tr.get("py", 0))
                d2 = dx * dx + dy * dy
                if d2 < best_d2:
                    best_d2 = d2
                    best_oid = oid

            if best_oid is not None and best_d2 <= max_d2:
                tr = self.tracks[best_oid]
                tr["px"] = int(round(alpha * m["px"] + (1.0 - alpha) * float(tr.get("px", m["px"]))))
                tr["py"] = int(round(alpha * m["py"] + (1.0 - alpha) * float(tr.get("py", m["py"]))))
                tr["conf"] = float(m["conf"])
                tr["grid_label"] = m["grid_label"]
                tr["grid_cell"] = int(m["grid_cell"])
                tr["grid_row"] = int(m["grid_row"])
                tr["grid_col"] = int(m["grid_col"])
                tr["last_seen"] = now
                unmatched_tracks.remove(best_oid)
            else:
                oid = f"obj_{self.next_track_id}"
                self.next_track_id += 1
                self.tracks[oid] = {
                    "px": int(m["px"]),
                    "py": int(m["py"]),
                    "conf": float(m["conf"]),
                    "grid_label": m["grid_label"],
                    "grid_cell": int(m["grid_cell"]),
                    "grid_row": int(m["grid_row"]),
                    "grid_col": int(m["grid_col"]),
                    "last_seen": now,
                }

        # Build ObjectState outputs
        objects: Dict[str, ObjectState] = {}
        for oid, tr in self.tracks.items():
            if (now - float(tr.get("last_seen", 0.0))) > self.track_max_age:
                continue
            obj = ObjectState()
            obj.object_id = oid
            obj.label = "object"
            obj.grid_cell = int(tr.get("grid_cell", 0))
            obj.grid_row = int(tr.get("grid_row", 0))
            obj.grid_col = int(tr.get("grid_col", 0))
            obj.grid_label = str(tr.get("grid_label", ""))

            if self.metric_enabled:
                x, y = self.grid_cell_center_xy(obj.grid_row, obj.grid_col)
                obj.position = Point(x=x, y=y, z=float(self.metric_object_z))
            else:
                obj.position = Point(x=float("nan"), y=float("nan"), z=float("nan"))

            obj.yaw_orientation = 0.0
            obj.is_held = False
            obj.confidence = float(tr.get("conf", 0.0))
            obj.bbox_center_x = int(tr.get("px", 0))
            obj.bbox_center_y = int(tr.get("py", 0))
            objects[oid] = obj

        return objects

    def get_gripper_yaw(self) -> float:
        if not self.gripper_pose:
            return 0.0
        q = self.gripper_pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return float(np.arctan2(siny_cosp, cosy_cosp))

    def build_symbolic_state(self, objects: Dict[str, ObjectState]) -> Optional[SymbolicState]:
        if not MSGS_AVAILABLE:
            return None

        state = SymbolicState()
        state.header = Header(stamp=rospy.Time.now(), frame_id=str(self.state_frame_id))

        # Objects
        state.objects = list(objects.values())

        # Gripper pose (metric, from robot if available)
        gripper_cell = None
        if self.gripper_pose is not None:
            p = self.gripper_pose.pose.position
            state.gripper_position = Point(x=p.x, y=p.y, z=p.z)
            state.gripper_height = float(p.z)
            state.gripper_yaw = self.get_gripper_yaw()
        else:
            state.gripper_position = Point(x=float("nan"), y=float("nan"), z=float("nan"))
            state.gripper_height = 0.0
            state.gripper_yaw = 0.0

        # Gripper grid cell/label (prefer YOLO 'jaco')
        if self.gripper_cell_from_yolo and self.latest_detections and self.latest_grid_bbox is not None:
            jac = [d for d in self.latest_detections if d.get("class") == "jaco"]
            if jac:
                jac.sort(key=lambda d: float(d.get("conf", 0.0)), reverse=True)
                px, py = jac[0].get("pick_xy", jac[0].get("center_xy", [None, None]))
                if px is not None and py is not None:
                    _, cell2, _, _ = self.pixel_to_grid_label(px, py, self.latest_grid_bbox)
                    if cell2 is not None:
                        gripper_cell = int(cell2)

        if gripper_cell is None:
            gripper_cell = 0

        state.gripper_grid_cell = int(gripper_cell)
        state.gripper_grid_label = self.cell_index_to_label(gripper_cell, grid_cols=self.grid_cols) or ""

        # History
        state.gripper_history = list(self.gripper_history)

        # Control mode
        if self.control_mode is not None:
            state.control_mode = self.control_mode
        else:
            state.control_mode = ControlMode()
            state.control_mode.mode = ControlMode.MODE_UNKNOWN

        # Grid config
        state.grid_rows = self.grid_rows
        state.grid_cols = self.grid_cols
        # Keep existing workspace bounds (used by UI/state text, not for metric tool calls)
        state.workspace_x_min = float(rospy.get_param("workspace/x_min", 0.0))
        state.workspace_x_max = float(rospy.get_param("workspace/x_max", 0.0))
        state.workspace_y_min = float(rospy.get_param("workspace/y_min", 0.0))
        state.workspace_y_max = float(rospy.get_param("workspace/y_max", 0.0))

        return state

    def compute_candidates(self, state: SymbolicState):
        """Keep previous behavior: neighbor cells around gripper in grid space."""
        candidates = []
        labels = []
        confidences = []
        reasoning = "No gripper pose"

        if state is None:
            return candidates, labels, confidences, "No state"

        gcell = int(getattr(state, "gripper_grid_cell", 0))
        grow = gcell // self.grid_cols
        gcol = gcell % self.grid_cols

        neighbor_cells = set()
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                rr = grow + dr
                cc = gcol + dc
                if 0 <= rr < self.grid_rows and 0 <= cc < self.grid_cols:
                    neighbor_cells.add(rr * self.grid_cols + cc)

        for obj in state.objects:
            try:
                if int(obj.grid_cell) in neighbor_cells:
                    candidates.append(obj.object_id)
                    labels.append(obj.label)
                    confidences.append(float(obj.confidence))
            except Exception:
                continue

        reasoning = f"Found {len(candidates)} candidates near gripper cell {gcell}"
        return candidates, labels, confidences, reasoning

    # -----------------------
    # Timer + service
    # -----------------------
    def update_state(self, _evt):
        with self.lock:
            objects = self.update_detected_objects_from_yolo()
            state = self.build_symbolic_state(objects)
            if state is not None and hasattr(self, "state_pub"):
                self.state_pub.publish(state)
            if state is not None and hasattr(self, "candidates_pub"):
                ids, labels, confs, reason = self.compute_candidates(state)
                cand = CandidateSet()
                cand.header = Header(stamp=rospy.Time.now())
                cand.candidate_ids = ids
                cand.candidate_labels = labels
                cand.confidence_scores = confs
                cand.reasoning = reason
                self.candidates_pub.publish(cand)

    def handle_get_state(self, _req):
        with self.lock:
            objects = self.update_detected_objects_from_yolo()
            state = self.build_symbolic_state(objects)
            resp = GetSymbolicStateResponse()
            if state is not None:
                resp.state = state
                resp.success = True
                resp.message = "State retrieved successfully"
            else:
                resp.success = False
                resp.message = "Failed to build state"
            return resp

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    StateBuilder().run()

