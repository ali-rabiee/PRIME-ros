#!/usr/bin/env python
"""
PRIME session logger.

Non-invasive logger node that subscribes to existing topics and writes:
- session metadata + metrics
- tool calls/results
- joint/pose trajectories
- optional collision checks
- optional video recording
"""

import json
import os
import threading
import time
from datetime import datetime

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import JointState, Image
from geometry_msgs.msg import PoseStamped

try:
    from prime_ros.msg import ToolCall, ToolResult
    MSGS_AVAILABLE = True
except Exception:
    MSGS_AVAILABLE = False


class PrimeLogger:
    def __init__(self):
        rospy.init_node("prime_logger", anonymous=False)

        self.node_start_wall = time.time()
        self.node_start_ros = rospy.Time.now().to_sec()

        self.log_root = os.path.expanduser(
            rospy.get_param("~log_root", "/home/tnlab/Desktop/PRIME_LOGS")
        )
        self.run_tag = rospy.get_param("~run_tag", "")
        self.session_start_mode = rospy.get_param("~session_start_mode", "first_event")
        self.robot_type = rospy.get_param("~robot_type", "j2n6s300")

        if self.session_start_mode not in (
            "node_start",
            "first_event",
            "first_tool_call",
            "first_tool_result",
        ):
            rospy.logwarn(
                "Unknown session_start_mode '%s', defaulting to 'first_event'",
                self.session_start_mode,
            )
            self.session_start_mode = "first_event"

        self.log_video = bool(rospy.get_param("~log_video", True))
        self.video_topic = rospy.get_param("~video_topic", "/camera/color/image_raw")
        self.video_fps = float(rospy.get_param("~video_fps", 30.0))

        self.log_trajectory = bool(rospy.get_param("~log_trajectory", True))
        self.trajectory_rate_hz = float(rospy.get_param("~trajectory_rate_hz", 30.0))

        self.log_collisions = bool(rospy.get_param("~log_collisions", False))
        self.collision_service = rospy.get_param("~collision_service", "/check_state_validity")
        self.collision_group = rospy.get_param("~collision_group", "arm")
        self.collision_rate_hz = float(rospy.get_param("~collision_rate_hz", 5.0))
        self.collision_log_all = bool(rospy.get_param("~collision_log_all", False))

        self._files = {}
        self._file_lock = threading.Lock()
        self._latest_joint_state = None
        self._latest_pose = None
        self._lock = threading.Lock()

        self._session_start_wall = None
        self._session_start_ros = None
        self._session_start_trigger = None

        self._tool_call_count = 0
        self._tool_result_count = 0
        self._grasp_attempts = 0
        self._grasp_successes = 0
        self._trajectory_joint_samples = 0
        self._trajectory_pose_samples = 0
        self._collision_count = 0
        self._collision_detected = False
        self._video_frames = 0

        self._last_joint_stamp = None
        self._last_pose_stamp = None
        self._last_collision_check = 0.0

        self._cv_bridge = None
        self._cv2 = None
        self._video_writer = None
        self._video_path = None

        self._collision_proxy = None
        self._collision_req_cls = None

        self.run_id, self.run_dir = self._make_run_dir(self.log_root, self.run_tag)

        if self.session_start_mode == "node_start":
            self._mark_session_start("node_start")

        self._write_json(
            "run_info.json",
            {
                "run_id": self.run_id,
                "run_dir": self.run_dir,
                "log_root": self.log_root,
                "run_tag": self.run_tag,
                "node_start_time_iso": self._iso_time(self.node_start_wall),
                "node_start_time_epoch": self.node_start_wall,
                "session_start_mode": self.session_start_mode,
                "robot_type": self.robot_type,
                "params": {
                    "log_video": self.log_video,
                    "video_topic": self.video_topic,
                    "video_fps": self.video_fps,
                    "log_trajectory": self.log_trajectory,
                    "trajectory_rate_hz": self.trajectory_rate_hz,
                    "log_collisions": self.log_collisions,
                    "collision_service": self.collision_service,
                    "collision_group": self.collision_group,
                    "collision_rate_hz": self.collision_rate_hz,
                    "collision_log_all": self.collision_log_all,
                },
            },
        )

        if MSGS_AVAILABLE:
            rospy.Subscriber(
                "/prime/tool_call", ToolCall, self._tool_call_cb, queue_size=50
            )
            rospy.Subscriber(
                "/prime/tool_result", ToolResult, self._tool_result_cb, queue_size=50
            )

        rospy.Subscriber(
            "/prime/gui_teleop_event", String, self._gui_event_cb, queue_size=50
        )

        driver_prefix = f"/{self.robot_type}_driver"
        rospy.Subscriber(
            f"{driver_prefix}/out/joint_state",
            JointState,
            self._joint_state_cb,
            queue_size=50,
        )
        rospy.Subscriber(
            f"{driver_prefix}/out/tool_pose",
            PoseStamped,
            self._pose_cb,
            queue_size=50,
        )

        if self.log_video:
            if not self._setup_video():
                self.log_video = False
            else:
                rospy.Subscriber(
                    self.video_topic, Image, self._image_cb, queue_size=1
                )

        if self.log_trajectory and self.trajectory_rate_hz > 0.0:
            self._trajectory_timer = rospy.Timer(
                rospy.Duration(1.0 / self.trajectory_rate_hz), self._trajectory_timer_cb
            )

        if self.log_collisions:
            self._setup_collision_check()
            if self._collision_proxy and self.collision_rate_hz > 0.0:
                self._collision_timer = rospy.Timer(
                    rospy.Duration(1.0 / self.collision_rate_hz),
                    self._collision_timer_cb,
                )

        rospy.on_shutdown(self._on_shutdown)
        rospy.loginfo("PRIME Logger initialized. Run dir: %s", self.run_dir)

    # -----------------------
    # Helpers
    # -----------------------
    @staticmethod
    def _iso_time(epoch_sec: float) -> str:
        return datetime.fromtimestamp(epoch_sec).isoformat()

    @staticmethod
    def _make_run_dir(log_root: str, run_tag: str):
        os.makedirs(log_root, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"{stamp}_{run_tag}" if run_tag else stamp
        run_dir = os.path.join(log_root, base)
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
            return base, run_dir
        # If the folder exists, add a numeric suffix
        idx = 1
        while True:
            candidate = f"{base}_{idx:02d}"
            run_dir = os.path.join(log_root, candidate)
            if not os.path.exists(run_dir):
                os.makedirs(run_dir)
                return candidate, run_dir
            idx += 1

    def _write_json(self, filename: str, data: dict):
        path = os.path.join(self.run_dir, filename)
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2, sort_keys=True, ensure_ascii=True)
        except Exception as exc:
            rospy.logwarn("Failed writing %s: %s", filename, exc)

    def _write_jsonl(self, filename: str, data: dict):
        path = os.path.join(self.run_dir, filename)
        try:
            with self._file_lock:
                f = self._files.get(filename)
                if f is None:
                    f = open(path, "a")
                    self._files[filename] = f
                f.write(json.dumps(data, ensure_ascii=True) + "\n")
                f.flush()
        except Exception as exc:
            rospy.logwarn("Failed appending %s: %s", filename, exc)

    @staticmethod
    def _msg_stamp(msg) -> float:
        try:
            stamp = msg.header.stamp.to_sec()
            return stamp if stamp > 0.0 else rospy.Time.now().to_sec()
        except Exception:
            return rospy.Time.now().to_sec()

    def _maybe_mark_start(self, trigger: str):
        if self._session_start_wall is not None:
            return
        mode = self.session_start_mode
        if mode == "node_start":
            return
        if mode == "first_event":
            self._mark_session_start(trigger)
        elif mode == "first_tool_call" and trigger == "tool_call":
            self._mark_session_start(trigger)
        elif mode == "first_tool_result" and trigger == "tool_result":
            self._mark_session_start(trigger)

    def _mark_session_start(self, trigger: str):
        if self._session_start_wall is None:
            self._session_start_wall = time.time()
            self._session_start_ros = rospy.Time.now().to_sec()
            self._session_start_trigger = trigger

    # -----------------------
    # Setup
    # -----------------------
    def _setup_video(self) -> bool:
        try:
            from cv_bridge import CvBridge
            import cv2
        except Exception as exc:
            rospy.logwarn("Video logging disabled (cv_bridge/cv2 unavailable): %s", exc)
            return False
        self._cv_bridge = CvBridge()
        self._cv2 = cv2
        return True

    def _setup_collision_check(self):
        try:
            from moveit_msgs.srv import GetStateValidity, GetStateValidityRequest
        except Exception as exc:
            rospy.logwarn("Collision logging disabled (moveit_msgs missing): %s", exc)
            self.log_collisions = False
            return

        try:
            rospy.wait_for_service(self.collision_service, timeout=5.0)
        except Exception as exc:
            rospy.logwarn("Collision service not available: %s (%s)", self.collision_service, exc)
            self.log_collisions = False
            return

        self._collision_proxy = rospy.ServiceProxy(self.collision_service, GetStateValidity)
        self._collision_req_cls = GetStateValidityRequest

    # -----------------------
    # Callbacks
    # -----------------------
    def _tool_call_cb(self, msg: ToolCall):
        self._maybe_mark_start("tool_call")
        self._tool_call_count += 1
        record = {
            "stamp": self._msg_stamp(msg),
            "call_id": msg.call_id,
            "tool_name": msg.tool_name,
            "target_object_id": msg.target_object_id,
            "interact_type": msg.interact_type,
            "interact_content": msg.interact_content,
            "interact_options": list(msg.interact_options),
            "reasoning": msg.reasoning,
        }
        self._write_jsonl("tool_calls.jsonl", record)

    def _tool_result_cb(self, msg: ToolResult):
        self._maybe_mark_start("tool_result")
        self._tool_result_count += 1
        if msg.tool_name == "GRASP":
            self._grasp_attempts += 1
            if msg.success:
                self._grasp_successes += 1
        record = {
            "stamp": self._msg_stamp(msg),
            "call_id": msg.call_id,
            "tool_name": msg.tool_name,
            "success": bool(msg.success),
            "status": int(msg.status),
            "error_category": msg.error_category,
            "message": msg.message,
            "user_response": msg.user_response,
            "selected_indices": list(msg.selected_indices),
        }
        self._write_jsonl("tool_results.jsonl", record)

    def _gui_event_cb(self, msg: String):
        self._maybe_mark_start("gui_event")
        record = {
            "stamp": rospy.Time.now().to_sec(),
            "event_json": msg.data,
        }
        self._write_jsonl("gui_events.jsonl", record)

    def _joint_state_cb(self, msg: JointState):
        self._maybe_mark_start("joint_state")
        with self._lock:
            self._latest_joint_state = msg

    def _pose_cb(self, msg: PoseStamped):
        self._maybe_mark_start("tool_pose")
        with self._lock:
            self._latest_pose = msg

    def _image_cb(self, msg: Image):
        self._maybe_mark_start("image")
        if not self.log_video or self._cv_bridge is None:
            return
        try:
            frame = self._cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            rospy.logwarn("Video frame conversion failed: %s", exc)
            return

        if self._video_writer is None:
            height, width = frame.shape[:2]
            self._video_path = os.path.join(self.run_dir, "video.mp4")
            fourcc = self._cv2.VideoWriter_fourcc(*"mp4v")
            self._video_writer = self._cv2.VideoWriter(
                self._video_path, fourcc, self.video_fps, (width, height)
            )
            if not self._video_writer.isOpened():
                rospy.logwarn("Failed to open video writer at %s", self._video_path)
                self._video_writer = None
                return

        self._video_writer.write(frame)
        self._video_frames += 1

    def _trajectory_timer_cb(self, _evt):
        if not self.log_trajectory:
            return
        with self._lock:
            joint_state = self._latest_joint_state
            pose = self._latest_pose

        if joint_state is not None:
            stamp = self._msg_stamp(joint_state)
            if stamp != self._last_joint_stamp:
                self._last_joint_stamp = stamp
                record = {
                    "stamp": stamp,
                    "name": list(joint_state.name),
                    "position": list(joint_state.position),
                    "velocity": list(joint_state.velocity),
                    "effort": list(joint_state.effort),
                }
                self._write_jsonl("trajectory_joint.jsonl", record)
                self._trajectory_joint_samples += 1

        if pose is not None:
            stamp = self._msg_stamp(pose)
            if stamp != self._last_pose_stamp:
                self._last_pose_stamp = stamp
                p = pose.pose.position
                o = pose.pose.orientation
                record = {
                    "stamp": stamp,
                    "position": {"x": p.x, "y": p.y, "z": p.z},
                    "orientation": {"x": o.x, "y": o.y, "z": o.z, "w": o.w},
                }
                self._write_jsonl("trajectory_pose.jsonl", record)
                self._trajectory_pose_samples += 1

    def _collision_timer_cb(self, _evt):
        if not self.log_collisions or self._collision_proxy is None:
            return
        now = time.time()
        if now - self._last_collision_check < max(0.0, 1.0 / self.collision_rate_hz):
            return
        self._last_collision_check = now
        with self._lock:
            joint_state = self._latest_joint_state
        if joint_state is None:
            return

        try:
            req = self._collision_req_cls()
            req.robot_state.joint_state = joint_state
            req.group_name = self.collision_group
            res = self._collision_proxy(req)
        except Exception as exc:
            rospy.logwarn("Collision check failed: %s", exc)
            return

        valid = bool(res.valid)
        if not valid:
            self._collision_detected = True
            self._collision_count += 1

        if self.collision_log_all or not valid:
            contacts = []
            try:
                for c in res.contacts:
                    contacts.append(
                        {
                            "body_1": c.body_name_1,
                            "body_2": c.body_name_2,
                            "depth": c.depth,
                        }
                    )
            except Exception:
                contacts = []

            record = {
                "stamp": rospy.Time.now().to_sec(),
                "valid": valid,
                "contact_count": len(contacts),
                "contacts": contacts,
            }
            self._write_jsonl("collisions.jsonl", record)

    # -----------------------
    # Shutdown
    # -----------------------
    def _on_shutdown(self):
        end_wall = time.time()
        if self._session_start_wall is None:
            self._session_start_wall = self.node_start_wall
            self._session_start_ros = self.node_start_ros
            self._session_start_trigger = "node_start"

        duration = max(0.0, end_wall - self._session_start_wall)
        grasp_rate = (
            float(self._grasp_successes) / float(self._grasp_attempts)
            if self._grasp_attempts > 0
            else None
        )

        session = {
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "session_start_time_iso": self._iso_time(self._session_start_wall),
            "session_start_time_epoch": self._session_start_wall,
            "session_end_time_iso": self._iso_time(end_wall),
            "session_end_time_epoch": end_wall,
            "session_length_sec": duration,
            "session_start_mode": self.session_start_mode,
            "session_start_trigger": self._session_start_trigger,
        }
        metrics = {
            "session_length_sec": duration,
            "tool_call_count": self._tool_call_count,
            "tool_result_count": self._tool_result_count,
            "grasp_attempts": self._grasp_attempts,
            "grasp_successes": self._grasp_successes,
            "grasp_success_rate": grasp_rate,
            "collision_detected": self._collision_detected,
            "collision_count": self._collision_count,
            "trajectory_joint_samples": self._trajectory_joint_samples,
            "trajectory_pose_samples": self._trajectory_pose_samples,
            "video_frames": self._video_frames,
            "video_path": self._video_path if self._video_path else None,
        }
        self._write_json("session.json", session)
        self._write_json("metrics.json", metrics)

        if self._video_writer is not None:
            try:
                self._video_writer.release()
            except Exception:
                pass

        with self._file_lock:
            for f in self._files.values():
                try:
                    f.close()
                except Exception:
                    pass
            self._files = {}


if __name__ == "__main__":
    logger = PrimeLogger()
    rospy.spin()
