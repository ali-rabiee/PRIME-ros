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
from cv_bridge import CvBridge
import cv2

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
        self.state_frame_id = rospy.get_param("~state_frame_id", rospy.get_param("workspace/frame_id", "root"))

        # Workspace — grid dimensions + metric rectangle for grid cell centers
        self.grid_rows = int(rospy.get_param("workspace/grid_rows", 3))
        self.grid_cols = int(rospy.get_param("workspace/grid_cols", 3))
        self.metric_frame_id = rospy.get_param("workspace/frame_id", "root")
        self.metric_x_min = float(rospy.get_param("workspace/x_min"))
        self.metric_x_max = float(rospy.get_param("workspace/x_max"))
        self.metric_y_min = float(rospy.get_param("workspace/y_min"))
        self.metric_y_max = float(rospy.get_param("workspace/y_max"))
        self.metric_object_z = float(rospy.get_param("workspace/object_z", 0.0))
        axis_signs = rospy.get_param("workspace/axis_signs", [1.0, 1.0, 1.0])
        try:
            axis_signs = list(axis_signs)
        except Exception:
            axis_signs = [1.0, 1.0, 1.0]
        while len(axis_signs) < 3:
            axis_signs.append(1.0)
        self.metric_axis_signs = [float(axis_signs[0]), float(axis_signs[1]), float(axis_signs[2])]

        # Yaw handling (purely image-based yaw coming from YOLO masks)
        # - YOLO publishes yaw in image coordinates (x right, y down) as det["mask_yaw_rad"].
        # - We map that yaw into the same metric/grid frame as objects (using axis_signs flips),
        #   then add an optional constant offset for camera-to-robot alignment.
        self.use_mask_yaw = bool(rospy.get_param("state_builder/use_mask_yaw", True))
        self.mask_yaw_field = str(rospy.get_param("state_builder/mask_yaw_field", "mask_yaw_rad")).strip()
        self.yaw_offset_rad = float(rospy.get_param("workspace/yaw_offset_rad", 0.0))

        # State builder parameters
        self.update_rate = float(rospy.get_param("state_builder/update_rate", 10.0))
        self.history_length = int(rospy.get_param("state_builder/gripper_history_length", 10))
        self.gui_event_history_length = int(rospy.get_param("state_builder/gui_event_history_length", 20))
        self.position_threshold = float(rospy.get_param("state_builder/position_threshold", 0.02))

        # Detection filtering + tracking (stable IDs)
        self.min_object_confidence = float(rospy.get_param("state_builder/min_object_confidence", 0.5))
        self.track_max_age = float(rospy.get_param("state_builder/track_max_age", 1.0))  # seconds
        self.track_max_pixel_dist = float(rospy.get_param("state_builder/track_max_pixel_dist", 80.0))  # px
        self.track_smoothing_alpha = float(rospy.get_param("state_builder/track_smoothing_alpha", 0.35))
        # If objects are mostly static during an episode, persisting tracks prevents ID churn
        # when an object is briefly occluded.
        self.persist_tracks = bool(rospy.get_param("state_builder/persist_tracks", True))
        # If >0, forget tracks not seen for this many seconds. If 0, never forget.
        self.forget_tracks_after = float(rospy.get_param("state_builder/forget_tracks_after", 0.0))
        # If true, publish tracks even if not recently seen (assumes static scene).
        self.publish_stale_tracks = bool(rospy.get_param("state_builder/publish_stale_tracks", True))
        # Prevent "one-frame false positives" from becoming permanent:
        # - New tracks start as tentative.
        # - A track becomes confirmed after N matched detections.
        self.track_min_confirmations = int(rospy.get_param("state_builder/track_min_confirmations", 3))
        self.track_tentative_ttl = float(rospy.get_param("state_builder/track_tentative_ttl", 2.0))  # seconds
        self.publish_tentative_tracks = bool(rospy.get_param("state_builder/publish_tentative_tracks", False))
        self.tracks: Dict[str, dict] = {}
        self.next_track_id = 1
        # Detection class filtering for objects.
        # - object_classes: labels considered graspable objects (default: ["object"]).
        # - ignored_detection_classes: labels to ignore when object_classes is empty or wildcard.
        # This allows richer YOLO models (e.g., class names like "mug", "can", etc.).
        raw_object_classes = rospy.get_param("state_builder/object_classes", ["object"])
        raw_ignored_classes = rospy.get_param("state_builder/ignored_detection_classes", ["workspace", "jaco", "bin"])
        try:
            self.object_classes = [str(x) for x in list(raw_object_classes)]
        except Exception:
            self.object_classes = ["object"]
        try:
            self.ignored_detection_classes = set(str(x) for x in list(raw_ignored_classes))
        except Exception:
            self.ignored_detection_classes = {"workspace", "jaco", "bin"}
        self._object_class_any = (len(self.object_classes) == 0) or ("*" in self.object_classes)

        # Inputs (cached)
        self.latest_detections = []
        self.latest_workspace_bbox = None
        self.latest_grid_bbox = None
        self.latest_yolo_image = None
        self.gripper_pose: Optional[PoseStamped] = None
        self.gripper_history = deque(maxlen=self.history_length)
        self.last_gui_teleop_event_json = ""
        self.gui_teleop_event_history = deque(maxlen=self.gui_event_history_length)
        self.control_mode: Optional[ControlMode] = None

        # Behavior toggles
        self.use_yolo_grid = bool(rospy.get_param("~use_yolo_grid", True))
        self.gripper_cell_from_yolo = bool(rospy.get_param("~gripper_cell_from_yolo", True))

        # Yaw debug visualization
        self.publish_yaw_debug = bool(rospy.get_param("state_builder/publish_yaw_debug", True))
        self.yaw_debug_pub = rospy.Publisher("/prime/yaw_debug", Image, queue_size=2)
        self.cv_bridge = CvBridge()

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
        self.gui_event_sub = rospy.Subscriber("/prime/gui_teleop_event", String, self.gui_event_callback, queue_size=50)

        self.yolo_img_sub = rospy.Subscriber("/yolo/image_with_bboxes", Image, self.yolo_image_callback, queue_size=1)
        self.yolo_dets_sub = rospy.Subscriber("/yolo/detections_json", String, self.yolo_detections_callback, queue_size=1)

        # Publishers / service
        if MSGS_AVAILABLE:
            self.state_pub = rospy.Publisher("/prime/symbolic_state", SymbolicState, queue_size=10)
            self.candidates_pub = rospy.Publisher("/prime/candidate_objects", CandidateSet, queue_size=10)
            self.state_service = rospy.Service("/prime/get_symbolic_state", GetSymbolicState, self.handle_get_state)

        # Timer
        self.timer = rospy.Timer(rospy.Duration(1.0 / max(1e-3, self.update_rate)), self.update_state)

        rospy.loginfo("State Builder initialized.")
        rospy.loginfo(
            "Workspace grid: %dx%d  frame=%s  x[%.3f,%.3f] y[%.3f,%.3f] object_z=%.3f",
            self.grid_rows, self.grid_cols,
            str(self.metric_frame_id),
            self.metric_x_min,
            self.metric_x_max,
            self.metric_y_min,
            self.metric_y_max,
            self.metric_object_z,
        )
        rospy.loginfo(
            "Metric axis_signs: [%.1f, %.1f, %.1f] (negative flips grid-to-metric direction)",
            self.metric_axis_signs[0],
            self.metric_axis_signs[1],
            self.metric_axis_signs[2],
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

    def gui_event_callback(self, msg: String):
        event_json = str(msg.data or "")
        if not event_json:
            return
        with self.lock:
            self.last_gui_teleop_event_json = event_json
            self.gui_teleop_event_history.append(event_json)

    # -----------------------
    # Grid + metric mapping
    # -----------------------
    @staticmethod
    def pixel_to_grid_label(cx, cy, ws_bbox_xyxy):
        """Convert pixel coordinate to A1..C3 based on grid bbox in image space."""
        if ws_bbox_xyxy is None:
            return None, None, None, None
        x1, y1, x2, y2 = ws_bbox_xyxy
        # If the pixel is outside the workspace/grid bbox, treat it as "not on the workspace".
        # This prevents wide-FOV cameras from mapping outside detections into edge grid cells.
        try:
            fx = float(cx)
            fy = float(cy)
        except Exception:
            return None, None, None, None
        if fx < float(x1) or fx > float(x2) or fy < float(y1) or fy > float(y2):
            return None, None, None, None
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

        # If axis_signs is negative, reverse which grid index maps to min/max along that axis.
        # This is safer than multiplying coordinates by -1 because it stays inside the same rectangle.
        if self.metric_axis_signs[0] < 0.0:
            col = (self.grid_cols - 1) - col
        if self.metric_axis_signs[1] < 0.0:
            row = (self.grid_rows - 1) - row

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

        # Expire stale tracks (optional)
        if self.persist_tracks:
            # Always prune tentative tracks that never got confirmed
            for oid in list(self.tracks.keys()):
                tr = self.tracks.get(oid, {})
                if not bool(tr.get("confirmed", False)):
                    first_seen = float(tr.get("first_seen", tr.get("last_seen", now)))
                    if (now - first_seen) > self.track_tentative_ttl and int(tr.get("hits", 0)) < self.track_min_confirmations:
                        del self.tracks[oid]
                        continue

            if self.forget_tracks_after > 0.0:
                for oid in list(self.tracks.keys()):
                    if (now - float(self.tracks[oid].get("last_seen", 0.0))) > self.forget_tracks_after:
                        del self.tracks[oid]
        else:
            for oid in list(self.tracks.keys()):
                if (now - float(self.tracks[oid].get("last_seen", 0.0))) > self.track_max_age:
                    del self.tracks[oid]

        # Measurements
        measured = []
        for det in self.latest_detections:
            det_class = str(det.get("class", "")).strip()
            if not det_class:
                continue
            if self._object_class_any:
                if det_class in self.ignored_detection_classes:
                    continue
            else:
                if det_class not in set(self.object_classes):
                    continue
                if det_class in self.ignored_detection_classes:
                    continue

            # Keep previous confidence threshold behavior
            # (applies to any accepted object class).
            if det_class in self.ignored_detection_classes:
                continue
            conf = float(det.get("conf", 0.0))
            if conf < self.min_object_confidence:
                continue
            px, py = det.get("pick_xy", det.get("center_xy", [None, None]))
            if px is None or py is None:
                continue

            yaw_img = None
            yaw_ratio = 0.0
            if self.use_mask_yaw and self.mask_yaw_field:
                try:
                    yv = det.get(self.mask_yaw_field, None)
                    if yv is not None:
                        yaw_img = float(yv)
                        if not np.isfinite(yaw_img):
                            yaw_img = None
                        else:
                            yaw_ratio = float(det.get("mask_yaw_ratio", 0.0))
                except Exception:
                    yaw_img = None

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
                    "yaw_img": yaw_img,
                    "yaw_ratio": yaw_ratio,
                    "label": det_class,
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
                tr["label"] = str(m.get("label", tr.get("label", "object")))
                tr["grid_label"] = m["grid_label"]
                tr["grid_cell"] = int(m["grid_cell"])
                tr["grid_row"] = int(m["grid_row"])
                tr["grid_col"] = int(m["grid_col"])
                # Circular smoothing for yaw (store as unit vector components)
                if m.get("yaw_img") is not None:
                    y = float(m["yaw_img"])
                    c = float(np.cos(y))
                    s = float(np.sin(y))
                    tr["yaw_c"] = float(alpha * c + (1.0 - alpha) * float(tr.get("yaw_c", c)))
                    tr["yaw_s"] = float(alpha * s + (1.0 - alpha) * float(tr.get("yaw_s", s)))
                    tr["mask_yaw_ratio"] = float(m.get("yaw_ratio", 0.0))
                tr["last_seen"] = now
                tr["hits"] = int(tr.get("hits", 1)) + 1
                if int(tr.get("hits", 0)) >= self.track_min_confirmations:
                    tr["confirmed"] = True
                unmatched_tracks.remove(best_oid)
            else:
                oid = f"obj_{self.next_track_id}"
                self.next_track_id += 1
                yaw_c = None
                yaw_s = None
                if m.get("yaw_img") is not None:
                    y = float(m["yaw_img"])
                    yaw_c = float(np.cos(y))
                    yaw_s = float(np.sin(y))
                self.tracks[oid] = {
                    "px": int(m["px"]),
                    "py": int(m["py"]),
                    "conf": float(m["conf"]),
                    "label": str(m.get("label", "object")),
                    "grid_label": m["grid_label"],
                    "grid_cell": int(m["grid_cell"]),
                    "grid_row": int(m["grid_row"]),
                    "grid_col": int(m["grid_col"]),
                    "yaw_c": yaw_c if yaw_c is not None else 1.0,
                    "yaw_s": yaw_s if yaw_s is not None else 0.0,
                    "mask_yaw_ratio": float(m.get("yaw_ratio", 0.0)),
                    "last_seen": now,
                    "first_seen": now,
                    "hits": 1,
                    "confirmed": True if self.track_min_confirmations <= 1 else False,
                }

        # Build ObjectState outputs
        objects: Dict[str, ObjectState] = {}
        for oid, tr in self.tracks.items():
            if (not self.publish_stale_tracks) and ((now - float(tr.get("last_seen", 0.0))) > self.track_max_age):
                continue
            if (not bool(tr.get("confirmed", False))) and (not self.publish_tentative_tracks):
                continue
            obj = ObjectState()
            obj.object_id = oid
            obj.label = str(tr.get("label", "object"))
            obj.grid_cell = int(tr.get("grid_cell", 0))
            obj.grid_row = int(tr.get("grid_row", 0))
            obj.grid_col = int(tr.get("grid_col", 0))
            obj.grid_label = str(tr.get("grid_label", ""))

            x, y = self.grid_cell_center_xy(obj.grid_row, obj.grid_col)
            obj.position = Point(x=x, y=y, z=float(self.metric_object_z))

            # Convert image-yaw to metric-yaw using axis_sign flips and a constant offset.
            # Image convention: x right, y down.
            # Metric convention here: x/y follow the workspace rectangle mapping.
            try:
                yaw_img = float(np.arctan2(float(tr.get("yaw_s", 0.0)), float(tr.get("yaw_c", 1.0))))
                vx = float(np.cos(yaw_img))
                vy = float(np.sin(yaw_img))
                vx_m = float(self.metric_axis_signs[0]) * vx
                vy_m = float(self.metric_axis_signs[1]) * vy
                obj.yaw_orientation = float(np.arctan2(vy_m, vx_m) + float(self.yaw_offset_rad))
            except Exception:
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
        state.last_gui_teleop_event_json = str(self.last_gui_teleop_event_json)
        state.gui_teleop_event_history_json = list(self.gui_teleop_event_history)

        # Grid config
        state.grid_rows = self.grid_rows
        state.grid_cols = self.grid_cols
        # Workspace bounds (same as grid metric rectangle)
        state.workspace_x_min = self.metric_x_min
        state.workspace_x_max = self.metric_x_max
        state.workspace_y_min = self.metric_y_min
        state.workspace_y_max = self.metric_y_max

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
    # Yaw debug visualization
    # -----------------------
    def publish_yaw_debug_image(self, objects: Dict[str, "ObjectState"]):
        """Draw an overlay on the YOLO image showing each object's believed yaw
        as an arrow, the gripper yaw, and text labels. Published on /prime/yaw_debug."""
        if not self.publish_yaw_debug:
            return
        if self.yaw_debug_pub.get_num_connections() == 0:
            return  # no subscribers, skip work

        # Use the latest YOLO image as background; if unavailable, create a blank canvas
        try:
            if self.latest_yolo_image is not None:
                canvas = self.cv_bridge.imgmsg_to_cv2(self.latest_yolo_image, "bgr8").copy()
            else:
                canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        except Exception:
            canvas = np.zeros((480, 640, 3), dtype=np.uint8)

        h, w = canvas.shape[:2]

        # Colors
        obj_arrow_color = (0, 255, 255)     # yellow: object yaw direction
        obj_minor_color = (0, 180, 180)     # dim yellow: minor axis
        gripper_color = (0, 255, 0)         # green: gripper yaw
        text_color = (255, 255, 255)        # white
        bg_color = (0, 0, 0)               # black text background
        arrow_len = 50
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1

        # Draw each tracked object's yaw
        for oid, obj in objects.items():
            tr = self.tracks.get(oid)
            if tr is None:
                continue

            cx = int(tr.get("px", 0))
            cy = int(tr.get("py", 0))
            if cx <= 0 or cy <= 0 or cx >= w or cy >= h:
                continue

            label_str = str(tr.get("label", "?"))
            yaw_c = float(tr.get("yaw_c", 1.0))
            yaw_s = float(tr.get("yaw_s", 0.0))
            yaw_img = float(np.arctan2(yaw_s, yaw_c))  # image-space yaw
            yaw_deg = float(np.degrees(yaw_img))

            # Convert to metric yaw (same transform as ObjectState)
            vx = float(np.cos(yaw_img))
            vy = float(np.sin(yaw_img))
            vx_m = float(self.metric_axis_signs[0]) * vx
            vy_m = float(self.metric_axis_signs[1]) * vy
            metric_yaw = float(np.arctan2(vy_m, vx_m) + float(self.yaw_offset_rad))
            metric_deg = float(np.degrees(metric_yaw))

            # Elongation ratio from eigenvalues (if stored)
            ratio = float(tr.get("mask_yaw_ratio", 0.0))

            # Draw major axis arrow (image-space direction so it matches visible mask)
            dx = int(round(arrow_len * np.cos(yaw_img)))
            dy = int(round(arrow_len * np.sin(yaw_img)))
            p1 = (cx - dx, cy - dy)
            p2 = (cx + dx, cy + dy)
            cv2.arrowedLine(canvas, (cx, cy), p2, obj_arrow_color, 2, tipLength=0.25)
            cv2.line(canvas, p1, (cx, cy), obj_arrow_color, 1)

            # Draw minor axis (perpendicular) as thin dashed-style
            dx_m = int(round(arrow_len * 0.5 * np.cos(yaw_img + np.pi / 2)))
            dy_m = int(round(arrow_len * 0.5 * np.sin(yaw_img + np.pi / 2)))
            cv2.line(canvas, (cx - dx_m, cy - dy_m), (cx + dx_m, cy + dy_m), obj_minor_color, 1)

            # Draw center dot
            cv2.circle(canvas, (cx, cy), 4, obj_arrow_color, -1)

            # Text: label + yaw info
            stale = (rospy.get_time() - float(tr.get("last_seen", 0.0))) > 2.0
            stale_str = " [stale]" if stale else ""
            txt1 = f"{label_str} ({oid})"
            txt2 = f"img:{yaw_deg:.0f}d  rob:{metric_deg:.0f}d{stale_str}"
            txt3 = f"ratio:{ratio:.1f}" if ratio > 0 else ""

            for i, txt in enumerate([txt1, txt2, txt3]):
                if not txt:
                    continue
                ty = cy - 15 + i * 14
                (tw, th), _ = cv2.getTextSize(txt, font, font_scale, thickness)
                cv2.rectangle(canvas, (cx + 8, ty - th - 2), (cx + 10 + tw, ty + 2), bg_color, -1)
                cv2.putText(canvas, txt, (cx + 9, ty), font, font_scale, text_color, thickness)

        # Draw gripper yaw (if known)
        if self.gripper_pose is not None:
            gripper_yaw = self.get_gripper_yaw()
            gripper_deg = float(np.degrees(gripper_yaw))

            # If we have a jaco detection in latest_detections, use its pixel coords
            jaco_cx, jaco_cy = None, None
            if self.latest_detections:
                jacs = [d for d in self.latest_detections if d.get("class") == "jaco"]
                if jacs:
                    jacs.sort(key=lambda d: float(d.get("conf", 0.0)), reverse=True)
                    jxy = jacs[0].get("pick_xy", jacs[0].get("center_xy", [None, None]))
                    if jxy[0] is not None:
                        jaco_cx, jaco_cy = int(jxy[0]), int(jxy[1])

            if jaco_cx is not None and jaco_cy is not None:
                gdx = int(round(arrow_len * np.cos(gripper_yaw)))
                gdy = int(round(arrow_len * np.sin(gripper_yaw)))
                cv2.arrowedLine(canvas, (jaco_cx, jaco_cy),
                                (jaco_cx + gdx, jaco_cy + gdy),
                                gripper_color, 2, tipLength=0.25)
                cv2.circle(canvas, (jaco_cx, jaco_cy), 5, gripper_color, -1)
                txt = f"GRIPPER yaw:{gripper_deg:.0f}d"
                (tw, th), _ = cv2.getTextSize(txt, font, font_scale, thickness)
                cv2.rectangle(canvas, (jaco_cx + 8, jaco_cy - th - 18),
                              (jaco_cx + 10 + tw, jaco_cy - 16), bg_color, -1)
                cv2.putText(canvas, txt, (jaco_cx + 9, jaco_cy - 17),
                            font, font_scale, gripper_color, thickness)

        # Legend in top-left
        legend_lines = [
            "YAW DEBUG OVERLAY",
            "Yellow arrow = object major axis (image frame)",
            "Green arrow = gripper wrist yaw (robot frame)",
            "img: image yaw | rob: robot yaw",
            "[stale] = not seen recently (using memory)",
        ]
        for i, line in enumerate(legend_lines):
            y = 18 + i * 16
            (tw, th), _ = cv2.getTextSize(line, font, font_scale, thickness)
            cv2.rectangle(canvas, (4, y - th - 2), (6 + tw, y + 3), bg_color, -1)
            color = (0, 200, 255) if i == 0 else (180, 180, 180)
            cv2.putText(canvas, line, (5, y), font, font_scale, color, thickness)

        # Publish
        try:
            msg = self.cv_bridge.cv2_to_imgmsg(canvas, "bgr8")
            msg.header.stamp = rospy.Time.now()
            self.yaw_debug_pub.publish(msg)
        except Exception:
            pass

    # -----------------------
    # Timer + service
    # -----------------------
    def update_state(self, _evt):
        try:
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
                # Publish yaw debug overlay
                self.publish_yaw_debug_image(objects)
        except Exception as e:
            rospy.logerr("state_builder update_state crashed: %s", str(e))
            import traceback
            rospy.logerr(traceback.format_exc())

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
