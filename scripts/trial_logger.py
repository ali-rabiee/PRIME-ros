#!/usr/bin/env python3
"""
Trial Logger for PRIME

Logs GUI interactions, tool calls/results, queries/responses, and records the
camera feed for each trial into ~/Desktop/PRIME_LOGS/<trial_*>.
"""

import datetime
import json
import os
import threading
import time

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger, TriggerResponse
from prime_ros.srv import StartTrial, StartTrialResponse

try:
    from cv_bridge import CvBridge, CvBridgeError
    import cv2
    CV_AVAILABLE = True
except Exception:
    CV_AVAILABLE = False
    CvBridge = None
    CvBridgeError = Exception
    cv2 = None

try:
    from prime_ros.msg import ToolCall, ToolResult, PRIMEQuery, PRIMEResponse, SymbolicState, CandidateSet
    PRIME_MSGS_AVAILABLE = True
except Exception:
    PRIME_MSGS_AVAILABLE = False


class TrialLogger:
    def __init__(self):
        rospy.init_node("trial_logger", anonymous=False)

        self.log_root = str(
            rospy.get_param("~log_root", os.path.expanduser("~/Desktop/PRIME_LOGS"))
        )
        self.mode = ""
        self.subject_id = ""
        self.difficulty = ""
        self.current_log_root = self.log_root
        self.image_topic = str(rospy.get_param("~image_topic", "/camera/color/image_raw"))
        self.record_video = bool(rospy.get_param("~record_video", True))
        self.video_fps = float(rospy.get_param("~video_fps", 30.0))
        self.video_fourcc = str(rospy.get_param("~video_fourcc", "mp4v"))

        self.success_label = str(rospy.get_param("~success_label", "")).strip()
        self.success_object_id = str(rospy.get_param("~success_object_id", "")).strip()

        self.lock = threading.RLock()
        self.bridge = CvBridge() if (self.record_video and CV_AVAILABLE) else None
        if self.record_video and not CV_AVAILABLE:
            rospy.logwarn("trial_logger: cv_bridge/cv2 not available; video disabled.")
            self.record_video = False

        self.trial_active = False
        self.trial_id = ""
        self.trial_dir = ""
        self.trial_start_wall = None
        self.trial_start_ros = None
        self.trial_end_wall = None

        self.event_fp = None
        self.gui_fp = None
        self.tool_call_fp = None
        self.tool_result_fp = None
        self.query_fp = None
        self.response_fp = None
        self.llm_fp = None
        self.video_writer = None
        self.video_path = ""

        self.tool_call_count = 0
        self.tool_counts = {}
        self.tool_call_by_id = {}
        self.gui_event_count = 0
        self.tool_result_count = 0
        self.query_count = 0
        self.response_count = 0

        self.object_labels = {}
        self.object_held = {}
        self.success = False
        self.success_time = None
        self.success_reason = ""
        self.success_object = {"id": "", "label": ""}
        self.trial_token = 0
        self.first_tool_call_wall = None
        self.first_query_wall = None
        self.first_response_wall = None
        self.last_state_snapshot = None
        self.last_candidates_snapshot = None

        self._ensure_root()

        self.gui_sub = rospy.Subscriber("/prime/gui_teleop_event", String, self._on_gui_event, queue_size=200)
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self._on_image, queue_size=1)

        if PRIME_MSGS_AVAILABLE:
            self.tool_call_sub = rospy.Subscriber("/prime/tool_call", ToolCall, self._on_tool_call, queue_size=50)
            self.tool_result_sub = rospy.Subscriber("/prime/tool_result", ToolResult, self._on_tool_result, queue_size=50)
            self.query_sub = rospy.Subscriber("/prime/query", PRIMEQuery, self._on_query, queue_size=10)
            self.response_sub = rospy.Subscriber("/prime/response", PRIMEResponse, self._on_response, queue_size=10)
            self.state_sub = rospy.Subscriber("/prime/symbolic_state", SymbolicState, self._on_state, queue_size=1)
            self.candidates_sub = rospy.Subscriber("/prime/candidate_objects", CandidateSet, self._on_candidates, queue_size=1)
            self.llm_event_sub = rospy.Subscriber("/prime/llm_event", String, self._on_llm_event, queue_size=50)
        else:
            rospy.logwarn("trial_logger: PRIME messages unavailable; tool/query logging disabled.")

        self.start_srv = rospy.Service("/prime/trial_logger/start", StartTrial, self._handle_start)
        self.stop_srv = rospy.Service("/prime/trial_logger/stop", Trigger, self._handle_stop)
        self.reset_srv = rospy.Service("/prime/trial_logger/reset", Trigger, self._handle_reset)

        rospy.on_shutdown(self._on_shutdown)
        rospy.loginfo(
            "trial_logger: started (log_root=%s, image_topic=%s). Waiting for /prime/trial_logger/start.",
            self.log_root,
            self.image_topic,
        )

    def _ensure_root(self):
        try:
            os.makedirs(self.log_root, exist_ok=True)
        except Exception as exc:
            rospy.logerr("trial_logger: failed to create log_root %s: %s", self.log_root, exc)
            raise

    @staticmethod
    def _now_wall():
        return time.time()

    @staticmethod
    def _iso(ts):
        if ts is None:
            return ""
        return datetime.datetime.fromtimestamp(float(ts)).isoformat(timespec="milliseconds")

    def _next_trial_id(self, root_dir):
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"trial_{stamp}"
        trial_id = base
        idx = 1
        while os.path.exists(os.path.join(root_dir, trial_id)):
            idx += 1
            trial_id = f"{base}_{idx:02d}"
        return trial_id

    def _open_log_files(self):
        self.event_fp = open(os.path.join(self.trial_dir, "events.jsonl"), "a", encoding="utf-8")
        self.gui_fp = open(os.path.join(self.trial_dir, "gui_events.jsonl"), "a", encoding="utf-8")
        self.tool_call_fp = open(os.path.join(self.trial_dir, "tool_calls.jsonl"), "a", encoding="utf-8")
        self.tool_result_fp = open(os.path.join(self.trial_dir, "tool_results.jsonl"), "a", encoding="utf-8")
        self.query_fp = open(os.path.join(self.trial_dir, "queries.jsonl"), "a", encoding="utf-8")
        self.response_fp = open(os.path.join(self.trial_dir, "responses.jsonl"), "a", encoding="utf-8")
        self.llm_fp = open(os.path.join(self.trial_dir, "llm_events.jsonl"), "a", encoding="utf-8")

    def _close_log_files(self):
        for fp in [self.event_fp, self.gui_fp, self.tool_call_fp, self.tool_result_fp, self.query_fp, self.response_fp, self.llm_fp]:
            try:
                if fp:
                    fp.flush()
                    fp.close()
            except Exception:
                pass
        self.event_fp = None
        self.gui_fp = None
        self.tool_call_fp = None
        self.tool_result_fp = None
        self.query_fp = None
        self.response_fp = None
        self.llm_fp = None

    def _resolve_trial_root(self, mode="", subject_id="", difficulty=""):
        mode = str(mode).strip()
        subject_id = str(subject_id).strip()
        difficulty = str(difficulty).strip()
        root = self.log_root
        if mode:
            root = os.path.join(root, mode)
        if subject_id:
            root = os.path.join(root, subject_id)
        if difficulty:
            root = os.path.join(root, difficulty)
        return root, mode, subject_id, difficulty

    def _start_trial(self, reason="manual", mode="", subject_id="", difficulty=""):
        with self.lock:
            if self.trial_active:
                return False, "trial already active"

            root_dir, mode, subject_id, difficulty = self._resolve_trial_root(
                mode=mode, subject_id=subject_id, difficulty=difficulty
            )
            os.makedirs(root_dir, exist_ok=True)
            self.current_log_root = root_dir
            self.mode = mode
            self.subject_id = subject_id
            self.difficulty = difficulty

            self.trial_id = self._next_trial_id(root_dir)
            self.trial_dir = os.path.join(root_dir, self.trial_id)
            os.makedirs(self.trial_dir, exist_ok=True)
            self.trial_token += 1

            self.trial_start_wall = self._now_wall()
            self.trial_start_ros = rospy.Time.now().to_sec()
            self.trial_end_wall = None

            self.tool_call_count = 0
            self.tool_counts = {}
            self.tool_call_by_id = {}
            self.gui_event_count = 0
            self.tool_result_count = 0
            self.query_count = 0
            self.response_count = 0

            self.success = False
            self.success_time = None
            self.success_reason = ""
            self.success_object = {"id": "", "label": ""}
            self.first_tool_call_wall = None
            self.first_query_wall = None
            self.first_response_wall = None
            self.last_state_snapshot = None
            self.last_candidates_snapshot = None

            self._open_log_files()
            self.trial_active = True
            self.video_writer = None
            self.video_path = os.path.join(self.trial_dir, "camera.mp4")

            meta = {
                "trial_id": self.trial_id,
                "mode": self.mode,
                "subject_id": self.subject_id,
                "difficulty": self.difficulty,
                "log_root": self.current_log_root,
                "start_time": self._iso(self.trial_start_wall),
                "start_time_epoch": self.trial_start_wall,
                "image_topic": self.image_topic,
                "record_video": bool(self.record_video),
                "video_fps": self.video_fps,
                "success_label": self.success_label,
                "success_object_id": self.success_object_id,
                "start_reason": reason,
            }
            self._write_json(os.path.join(self.trial_dir, "trial_meta.json"), meta)
            self._log_event("trial_start", {"reason": reason})
            return True, f"trial started: {self.trial_id}"

    def _finish_trial(self, reason="manual"):
        with self.lock:
            if not self.trial_active:
                return False, "no active trial"

            self.trial_end_wall = self._now_wall()
            duration = float(self.trial_end_wall - self.trial_start_wall)
            completion = None
            if self.success_time is not None:
                completion = float(self.success_time - self.trial_start_wall)
            first_tool_call = None
            if self.first_tool_call_wall is not None:
                first_tool_call = float(self.first_tool_call_wall - self.trial_start_wall)
            time_from_first_tool_to_success = None
            if self.first_tool_call_wall is not None and self.success_time is not None:
                time_from_first_tool_to_success = float(self.success_time - self.first_tool_call_wall)
            first_query = None
            if self.first_query_wall is not None:
                first_query = float(self.first_query_wall - self.trial_start_wall)
            first_response = None
            if self.first_response_wall is not None:
                first_response = float(self.first_response_wall - self.trial_start_wall)

            if self.video_writer is not None:
                try:
                    self.video_writer.release()
                except Exception:
                    pass
                self.video_writer = None
            self.video_path = ""

            summary = {
                "trial_id": self.trial_id,
                "mode": self.mode,
                "subject_id": self.subject_id,
                "difficulty": self.difficulty,
                "log_root": self.current_log_root,
                "trial_dir": self.trial_dir,
                "start_time": self._iso(self.trial_start_wall),
                "end_time": self._iso(self.trial_end_wall),
                "duration_sec": duration,
                "completion_time_sec": completion,
                "first_tool_call_time_sec": first_tool_call,
                "time_from_first_tool_call_to_success_sec": time_from_first_tool_to_success,
                "first_query_time_sec": first_query,
                "first_response_time_sec": first_response,
                "success": bool(self.success),
                "success_time": self._iso(self.success_time) if self.success_time else "",
                "success_reason": self.success_reason,
                "success_object_id": self.success_object.get("id", ""),
                "success_object_label": self.success_object.get("label", ""),
                "tool_call_count": int(self.tool_call_count),
                "tool_calls_by_name": dict(self.tool_counts),
                "tool_result_count": int(self.tool_result_count),
                "gui_event_count": int(self.gui_event_count),
                "query_count": int(self.query_count),
                "response_count": int(self.response_count),
                "image_topic": self.image_topic,
                "video_file": "camera.mp4" if self.record_video else "",
                "events_file": "events.jsonl",
                "end_reason": reason,
            }
            self._write_json(os.path.join(self.trial_dir, "trial_summary.json"), summary)
            self._log_event("trial_end", {"reason": reason, "summary": summary})

            self.trial_active = False
            self._close_log_files()
            return True, "trial finished"

    def _write_json(self, path, payload):
        try:
            with open(path, "w", encoding="utf-8") as fp:
                json.dump(payload, fp, indent=2, sort_keys=True)
        except Exception as exc:
            rospy.logwarn("trial_logger: failed to write %s: %s", path, exc)

    def _write_jsonl(self, fp, payload):
        if fp is None:
            return
        try:
            fp.write(json.dumps(payload, sort_keys=True) + "\n")
            fp.flush()
        except Exception as exc:
            rospy.logwarn("trial_logger: jsonl write failed: %s", exc)

    def _log_event(self, event_type, payload, stamp=None):
        ts = float(stamp if stamp is not None else self._now_wall())
        data = {
            "event": str(event_type),
            "stamp": ts,
            "stamp_iso": self._iso(ts),
            "t_from_start_sec": float(ts - self.trial_start_wall) if self.trial_start_wall else None,
            "payload": payload,
        }
        self._write_jsonl(self.event_fp, data)

    def _matches_success(self, object_id="", object_label=""):
        if self.success_object_id:
            return object_id == self.success_object_id
        if self.success_label:
            return object_label.strip().lower() == self.success_label.strip().lower()
        return True

    def _mark_success(self, reason, object_id="", object_label=""):
        if self.success:
            return
        if not self._matches_success(object_id=object_id, object_label=object_label):
            return
        self.success = True
        self.success_time = self._now_wall()
        self.success_reason = str(reason)
        self.success_object = {"id": str(object_id), "label": str(object_label)}
        self._log_event(
            "success",
            {
                "reason": self.success_reason,
                "object_id": self.success_object.get("id", ""),
                "object_label": self.success_object.get("label", ""),
            },
        )

    def _on_gui_event(self, msg: String):
        with self.lock:
            if not self.trial_active:
                return
            self.gui_event_count += 1
            raw = msg.data
            parsed = None
            try:
                parsed = json.loads(raw)
            except Exception:
                parsed = None
            data = {
                "raw": raw,
                "parsed": parsed,
            }
            self._write_jsonl(self.gui_fp, data)
            self._log_event("gui_event", data)

    def _on_tool_call(self, msg: ToolCall):
        with self.lock:
            if not self.trial_active:
                return
            self.tool_call_count += 1
            if self.first_tool_call_wall is None:
                self.first_tool_call_wall = self._now_wall()
            name = str(msg.tool_name)
            self.tool_counts[name] = int(self.tool_counts.get(name, 0)) + 1
            self.tool_call_by_id[msg.call_id] = {
                "tool_name": name,
                "target_object_id": str(msg.target_object_id),
            }
            stamp = msg.header.stamp.to_sec() if msg.header else self._now_wall()
            data = {
                "call_id": msg.call_id,
                "tool_name": name,
                "target_object_id": str(msg.target_object_id),
                "interact_type": int(msg.interact_type),
                "interact_content": str(msg.interact_content),
                "interact_options": list(msg.interact_options),
                "reasoning": str(msg.reasoning),
                "stamp": stamp,
                "t_from_start_sec": float(stamp - self.trial_start_ros) if self.trial_start_ros else None,
                "state_snapshot": self.last_state_snapshot,
                "candidates_snapshot": self.last_candidates_snapshot,
            }
            self._write_jsonl(self.tool_call_fp, data)
            self._log_event("tool_call", data, stamp=data["stamp"])

    def _on_tool_result(self, msg: ToolResult):
        with self.lock:
            if not self.trial_active:
                return
            self.tool_result_count += 1
            stamp = msg.header.stamp.to_sec() if msg.header else self._now_wall()
            data = {
                "call_id": msg.call_id,
                "tool_name": str(msg.tool_name),
                "success": bool(msg.success),
                "status": int(msg.status),
                "error_category": str(msg.error_category),
                "message": str(msg.message),
                "user_response": str(msg.user_response),
                "selected_indices": list(msg.selected_indices),
                "stamp": stamp,
                "t_from_start_sec": float(stamp - self.trial_start_ros) if self.trial_start_ros else None,
            }
            self._write_jsonl(self.tool_result_fp, data)
            self._log_event("tool_result", data, stamp=data["stamp"])

            if str(msg.tool_name) == "GRASP" and bool(msg.success):
                call_meta = self.tool_call_by_id.get(msg.call_id, {})
                obj_id = call_meta.get("target_object_id", "")
                obj_label = self.object_labels.get(obj_id, "")
                self._mark_success("grasp_tool_success", object_id=obj_id, object_label=obj_label)

    def _on_query(self, msg: PRIMEQuery):
        with self.lock:
            if not self.trial_active:
                return
            self.query_count += 1
            if self.first_query_wall is None:
                self.first_query_wall = self._now_wall()
            stamp = msg.header.stamp.to_sec() if msg.header else self._now_wall()
            source = "direct"
            if msg.query_id in self.tool_call_by_id:
                source = "tool_call"
            data = {
                "query_id": msg.query_id,
                "query_type": int(msg.query_type),
                "content": str(msg.content),
                "options": list(msg.options),
                "max_selections": int(msg.max_selections),
                "timeout": float(msg.timeout),
                "stamp": stamp,
                "t_from_start_sec": float(stamp - self.trial_start_ros) if self.trial_start_ros else None,
                "source": source,
                "state_snapshot": self.last_state_snapshot,
                "candidates_snapshot": self.last_candidates_snapshot,
            }
            self._write_jsonl(self.query_fp, data)
            self._log_event("query", data, stamp=data["stamp"])

    def _on_response(self, msg: PRIMEResponse):
        with self.lock:
            if not self.trial_active:
                return
            self.response_count += 1
            if self.first_response_wall is None:
                self.first_response_wall = self._now_wall()
            stamp = msg.header.stamp.to_sec() if msg.header else self._now_wall()
            data = {
                "query_id": msg.query_id,
                "selected_indices": list(msg.selected_indices),
                "selected_labels": list(msg.selected_labels),
                "timed_out": bool(msg.timed_out),
                "response_time": float(msg.response_time),
                "stamp": stamp,
                "t_from_start_sec": float(stamp - self.trial_start_ros) if self.trial_start_ros else None,
            }
            self._write_jsonl(self.response_fp, data)
            self._log_event("response", data, stamp=data["stamp"])

    def _on_state(self, msg: SymbolicState):
        with self.lock:
            self.object_labels = {o.object_id: str(o.label) for o in msg.objects}
            self.object_held = {o.object_id: bool(o.is_held) for o in msg.objects}
            self.last_state_snapshot = self._build_state_snapshot(msg)
            if not self.trial_active:
                return
            for obj in msg.objects:
                if bool(obj.is_held):
                    self._mark_success(
                        "symbolic_state_held",
                        object_id=str(obj.object_id),
                        object_label=str(obj.label),
                    )

    def _on_candidates(self, msg: CandidateSet):
        with self.lock:
            self.last_candidates_snapshot = {
                "stamp": msg.header.stamp.to_sec() if msg.header else self._now_wall(),
                "candidate_ids": list(msg.candidate_ids),
                "candidate_labels": list(msg.candidate_labels),
                "confidence_scores": list(msg.confidence_scores),
                "reasoning": str(msg.reasoning),
            }

    def _on_llm_event(self, msg: String):
        with self.lock:
            if not self.trial_active:
                return
            raw = msg.data
            parsed = None
            try:
                parsed = json.loads(raw)
            except Exception:
                parsed = None
            data = {
                "raw": raw,
                "parsed": parsed,
            }
            self._write_jsonl(self.llm_fp, data)
            self._log_event("llm_event", data)

    def _build_state_snapshot(self, msg: SymbolicState):
        try:
            objects = []
            for obj in msg.objects:
                objects.append(
                    {
                        "id": str(obj.object_id),
                        "label": str(obj.label),
                        "grid_cell": int(obj.grid_cell),
                        "grid_label": str(obj.grid_label),
                        "is_held": bool(obj.is_held),
                    }
                )
            ctrl = msg.control_mode
            snapshot = {
                "stamp": msg.header.stamp.to_sec() if msg.header else self._now_wall(),
                "objects": objects,
                "gripper": {
                    "grid_cell": int(msg.gripper_grid_cell),
                    "grid_label": str(msg.gripper_grid_label),
                    "height": float(msg.gripper_height),
                    "yaw": float(msg.gripper_yaw),
                    "position": {
                        "x": float(getattr(msg.gripper_position, "x", 0.0)),
                        "y": float(getattr(msg.gripper_position, "y", 0.0)),
                        "z": float(getattr(msg.gripper_position, "z", 0.0)),
                    },
                },
                "control_mode": {
                    "mode": int(ctrl.mode),
                    "translation_active": bool(ctrl.translation_active),
                    "rotation_active": bool(ctrl.rotation_active),
                    "fingers_active": bool(ctrl.fingers_active),
                    "joystick_active": bool(ctrl.joystick_active),
                },
            }
            return snapshot
        except Exception:
            return None

    def _on_image(self, msg: Image):
        if not self.record_video:
            return
        with self.lock:
            if not self.trial_active:
                return
            if self.bridge is None:
                return
            token = int(self.trial_token)
            video_path = str(self.video_path)
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as exc:
            rospy.logwarn("trial_logger: cv_bridge error: %s", exc)
            return

        with self.lock:
            if not self.trial_active or int(self.trial_token) != token:
                return
            if not video_path:
                return
            if self.video_writer is None:
                height, width = cv_image.shape[:2]
                try:
                    fourcc = cv2.VideoWriter_fourcc(*self.video_fourcc)
                except Exception:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self.video_writer = cv2.VideoWriter(video_path, fourcc, self.video_fps, (width, height))
                if not self.video_writer or not self.video_writer.isOpened():
                    rospy.logerr("trial_logger: failed to open video writer at %s", video_path)
                    self.video_writer = None
                    self.record_video = False
                    return
                video_meta = {
                    "video_path": video_path,
                    "width": int(width),
                    "height": int(height),
                    "fps": float(self.video_fps),
                    "fourcc": self.video_fourcc,
                    "image_topic": self.image_topic,
                }
                self._write_json(os.path.join(self.trial_dir, "video_meta.json"), video_meta)

            if self.video_writer is not None:
                self.video_writer.write(cv_image)

    def _handle_start(self, req):
        reason = str(req.reason).strip() if req and req.reason else "service_start"
        ok, msg = self._start_trial(
            reason=reason,
            mode=req.mode,
            subject_id=req.subject_id,
            difficulty=req.difficulty,
        )
        return StartTrialResponse(success=ok, message=msg)

    def _handle_stop(self, _req):
        ok, msg = self._finish_trial(reason="service_stop")
        return TriggerResponse(success=ok, message=msg)

    def _handle_reset(self, _req):
        ok, msg = self._finish_trial(reason="service_reset")
        if ok:
            ok2, msg2 = self._start_trial(
                reason="service_reset",
                mode=self.mode,
                subject_id=self.subject_id,
                difficulty=self.difficulty,
            )
            return TriggerResponse(success=ok2, message=msg2)
        return TriggerResponse(success=False, message=msg)

    def _on_shutdown(self):
        try:
            self._finish_trial(reason="shutdown")
        except Exception:
            pass

    def run(self):
        rospy.spin()


def main():
    try:
        node = TrialLogger()
        node.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
