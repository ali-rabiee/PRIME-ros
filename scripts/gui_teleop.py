#!/usr/bin/env python3
"""GUI teleoperation for PRIME (replaces joystick pipeline)."""

import json
import os
import sys
import threading
import time
import uuid

import actionlib
import rospy
from kinova_msgs.msg import (
    FingerPosition,
    PoseVelocity,
    SetFingersPositionAction,
    SetFingersPositionGoal,
)
from std_msgs.msg import String

# Ensure local PRIME scripts directory is on PYTHONPATH (rosrun wrapper doesn't add it)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from prime_ros.msg import ControlMode

try:
    from prime_ros.msg import SymbolicState, CandidateSet, ToolCall, ToolResult, PRIMEQuery, PRIMEResponse
    PRIME_MSGS_AVAILABLE = True
except Exception:
    PRIME_MSGS_AVAILABLE = False

try:
    from oracle_assist import (
        OracleState,
        oracle_decide_tool,
        validate_tool_call,
        apply_oracle_user_reply,
        choice_to_user_content,
        strip_choice_label,
        yaw_to_bin,
        manhattan,
    )
    ORACLE_AVAILABLE = True
except Exception as e:
    ORACLE_AVAILABLE = False
    ORACLE_IMPORT_ERROR = str(e)


class TeleopCommandModel:
    """Pure-Python state model for teleop command generation and logging."""

    MODE_TRANSLATION = "translation"
    MODE_ROTATION = "rotation"
    MODE_GRIPPER = "gripper"

    _MODE_TO_ENUM = {
        MODE_TRANSLATION: ControlMode.MODE_TRANSLATION,
        MODE_ROTATION: ControlMode.MODE_ROTATION,
        MODE_GRIPPER: ControlMode.MODE_GRIPPER,
    }

    _MODE_TO_AXES = {
        MODE_TRANSLATION: {"x", "y", "z"},
        MODE_ROTATION: {"rx", "ry", "rz"},
        MODE_GRIPPER: set(),
    }

    _AXIS_TO_FIELD = {
        "x": "twist_linear_x",
        "y": "twist_linear_y",
        "z": "twist_linear_z",
        "rx": "twist_angular_x",
        "ry": "twist_angular_y",
        "rz": "twist_angular_z",
    }

    def __init__(self, linear_speed=0.02, angular_speed=5.0, gripper_activity_latch=0.0):
        self.linear_speed = float(linear_speed)
        self.angular_speed = float(angular_speed)
        self.gripper_activity_latch = float(gripper_activity_latch)

        self.mode = self.MODE_TRANSLATION
        self.active_axis = None
        self.active_direction = 0
        self.activity_until = 0.0

    @staticmethod
    def _now(stamp=None):
        return float(time.time() if stamp is None else stamp)

    @staticmethod
    def _event_json(stamp, event_type, mode, axis=None, direction=None, value=None, active=None, extra=None):
        data = {
            "stamp": float(stamp),
            "source": "gui",
            "type": str(event_type),
            "mode": str(mode),
        }
        if axis is not None:
            data["axis"] = str(axis)
        if direction is not None:
            data["direction"] = int(direction)
        if value is not None:
            data["value"] = float(value)
        if active is not None:
            data["active"] = bool(active)
        if extra:
            data.update(extra)
        return json.dumps(data, sort_keys=True)

    def is_motion_active(self):
        return self.active_axis is not None

    def set_mode(self, mode, stamp=None):
        if mode not in self._MODE_TO_ENUM:
            raise ValueError("Unknown mode: %s" % mode)
        if mode == self.mode:
            return None
        now = self._now(stamp)
        self.mode = mode
        return self._event_json(now, "mode_change", self.mode)

    def start_motion(self, axis, direction, stamp=None):
        direction = int(direction)
        if direction not in (-1, 1):
            raise ValueError("Direction must be +1 or -1")
        if axis not in self._MODE_TO_AXES[self.mode]:
            raise ValueError("Axis %s not valid in mode %s" % (axis, self.mode))

        now = self._now(stamp)
        self.active_axis = str(axis)
        self.active_direction = direction
        speed = self.linear_speed if axis in ("x", "y", "z") else self.angular_speed

        return self._event_json(
            now,
            "cartesian_velocity",
            self.mode,
            axis=self.active_axis,
            direction=self.active_direction,
            value=speed,
            active=True,
        )

    def stop_motion(self, stamp=None, reason="release", force_event=False):
        now = self._now(stamp)
        was_active = self.is_motion_active()
        axis = self.active_axis
        direction = self.active_direction if was_active else None

        self.active_axis = None
        self.active_direction = 0

        if not was_active and not force_event:
            return None

        extra = {"reason": str(reason)} if reason else None
        return self._event_json(
            now,
            "stop",
            self.mode,
            axis=axis,
            direction=direction,
            value=0.0,
            active=False,
            extra=extra,
        )

    def record_gripper_action(self, action, finger_position, stamp=None):
        if action not in ("open", "close"):
            raise ValueError("Unsupported gripper action: %s" % action)
        now = self._now(stamp)
        self.activity_until = now + max(0.0, self.gripper_activity_latch)
        return self._event_json(
            now,
            "gripper",
            self.mode,
            value=float(finger_position),
            active=True,
            extra={"action": action},
        )

    def build_velocity_command(self):
        msg = PoseVelocity()
        if self.active_axis is None:
            return msg

        # Guard against mode/axis mismatch (GUI thread can change mode while the
        # publish timer is building commands). In that case, publish zero.
        if self.active_axis not in self._MODE_TO_AXES.get(self.mode, set()):
            return msg

        field = self._AXIS_TO_FIELD[self.active_axis]
        magnitude = self.linear_speed if self.active_axis in ("x", "y", "z") else self.angular_speed
        setattr(msg, field, float(self.active_direction) * magnitude)
        return msg

    def build_control_mode(self, stamp=None):
        now = self._now(stamp)
        msg = ControlMode()
        msg.header.stamp = rospy.Time.from_sec(now)
        msg.mode = self._MODE_TO_ENUM.get(self.mode, ControlMode.MODE_UNKNOWN)
        msg.translation_active = self.mode == self.MODE_TRANSLATION
        msg.rotation_active = self.mode == self.MODE_ROTATION
        msg.fingers_active = self.mode == self.MODE_GRIPPER
        msg.joystick_active = self.is_motion_active() or (now < self.activity_until)
        return msg


class GuiTeleopNode:
    def __init__(self):
        rospy.init_node("gui_teleop", anonymous=False)

        # Prefer private params, fallback to global config.
        self.publish_rate = float(rospy.get_param("~publish_rate", rospy.get_param("teleop_gui/publish_rate", 50.0)))
        self.linear_speed = float(rospy.get_param("~linear_speed", rospy.get_param("teleop_gui/linear_speed", 0.02)))
        self.angular_speed = float(rospy.get_param("~angular_speed", rospy.get_param("teleop_gui/angular_speed", 5.0)))
        self.gripper_activity_latch = float(
            rospy.get_param("~gripper_activity_latch", rospy.get_param("teleop_gui/gripper_activity_latch", 0.2))
        )
        self.headless = bool(rospy.get_param("~headless", False))
        self.wait_for_home = bool(rospy.get_param("~wait_for_home", False))
        self.home_done_param = str(rospy.get_param("~home_done_param", "/prime/homing_done"))
        self.home_wait_timeout = float(rospy.get_param("~home_wait_timeout", 0.0))  # 0 = wait forever

        # Oracle assist config (GUI side panel)
        self.oracle_enabled = bool(rospy.get_param("~oracle_enabled", True))
        self.oracle_publish_query = bool(rospy.get_param("~oracle_publish_query", False))
        self.oracle_publish_response = bool(rospy.get_param("~oracle_publish_response", False))
        self.oracle_history_len = int(rospy.get_param("~oracle_history_len", 6))
        self.oracle_z_min = float(rospy.get_param("safety_bounds/z_min", 0.2))
        self.oracle_z_max = float(rospy.get_param("safety_bounds/z_max", 0.6))
        self.oracle_enabled_requested = self.oracle_enabled
        self.oracle_available = ORACLE_AVAILABLE
        if not self.oracle_available:
            rospy.logwarn("Oracle assist disabled: %s", ORACLE_IMPORT_ERROR)
        self.oracle_enabled = self.oracle_enabled_requested and self.oracle_available

        # Kinova velocity control is designed around ~100Hz updates.
        if self.publish_rate < 90.0:
            rospy.logwarn("GUI teleop publish_rate=%.1fHz (Kinova recommends ~100Hz for velocity control)", self.publish_rate)
        # If translation is extremely small, it can look like "nothing happens".
        if self.linear_speed < 0.05:
            rospy.logwarn("GUI teleop linear_speed=%.3f m/s is quite low; increase teleop_gui/linear_speed if translation feels unresponsive", self.linear_speed)

        if self.wait_for_home:
            self._wait_until_homed()

        self.robot_type = rospy.get_param("robot/type", rospy.get_param("~robot_type", "j2n6s300"))
        self.finger_open_position = float(rospy.get_param("tools/grasp/finger_open_position", 0.0))
        self.finger_close_position = float(rospy.get_param("tools/grasp/finger_close_position", 5000.0))

        self.driver_prefix = "/%s_driver" % self.robot_type
        self.velocity_topic = "%s/in/cartesian_velocity" % self.driver_prefix
        self.finger_action_name = "%s/fingers_action/finger_positions" % self.driver_prefix

        self.lock = threading.RLock()
        self.model = TeleopCommandModel(
            linear_speed=self.linear_speed,
            angular_speed=self.angular_speed,
            gripper_activity_latch=self.gripper_activity_latch,
        )

        self.mode_pub = rospy.Publisher("/prime/control_mode", ControlMode, queue_size=10)
        self.event_pub = rospy.Publisher("/prime/gui_teleop_event", String, queue_size=50)
        self.velocity_pub = rospy.Publisher(self.velocity_topic, PoseVelocity, queue_size=10)

        self.finger_client = actionlib.SimpleActionClient(self.finger_action_name, SetFingersPositionAction)
        if not self.finger_client.wait_for_server(rospy.Duration(2.0)):
            rospy.logwarn("Gripper action server unavailable on %s", self.finger_action_name)

        # Teleop pause gate: prevents GUI cartesian velocity from fighting MoveIt execution.
        self.teleop_paused = False
        self.teleop_pause_reason = ""
        self.teleop_pause_call_id = None

        # Oracle state + ROS plumbing
        self.oracle_state = None
        self.oracle_memory = None
        self.oracle_objects = []
        self.oracle_gripper_hist = []
        self.oracle_choices = []
        self.oracle_seen_symbolic_state = False
        self.oracle_status_var = None
        self.oracle_choices_frame = None
        # final enable already computed above

        self.oracle_tool_pub = None
        self.oracle_query_pub = None
        self.oracle_response_pub = None
        if PRIME_MSGS_AVAILABLE:
            # Listen for tool calls/results from *any* source so GUI teleop doesn't interfere.
            rospy.Subscriber("/prime/tool_call", ToolCall, self._teleop_tool_call_callback, queue_size=10)
            rospy.Subscriber("/prime/tool_result", ToolResult, self._teleop_tool_result_callback, queue_size=10)

        if self.oracle_enabled and PRIME_MSGS_AVAILABLE:
            self.oracle_tool_pub = rospy.Publisher("/prime/tool_call", ToolCall, queue_size=10)
            if self.oracle_publish_query:
                self.oracle_query_pub = rospy.Publisher("/prime/query", PRIMEQuery, queue_size=10)
            if self.oracle_publish_response:
                self.oracle_response_pub = rospy.Publisher("/prime/response", PRIMEResponse, queue_size=10)
            rospy.Subscriber("/prime/symbolic_state", SymbolicState, self._oracle_state_callback, queue_size=1)
            rospy.Subscriber("/prime/candidate_objects", CandidateSet, self._oracle_candidates_callback, queue_size=1)

        self.publish_timer = rospy.Timer(
            rospy.Duration(1.0 / max(1e-3, self.publish_rate)),
            self.publish_cycle,
        )

        self.tk = None
        self.root = None
        self.mode_var = None
        self.mode_text_var = None
        self.active_text_var = None
        self.translation_frame = None
        self.rotation_frame = None
        self.gripper_frame = None
        self.oracle_panel = None

        if not self.headless:
            self._init_gui()

        # Publish initial mode immediately so downstream state is consistent.
        self._publish_mode_and_velocity()
        rospy.loginfo("GUI teleop initialized for %s (headless=%s)", self.robot_type, self.headless)

    def _wait_until_homed(self):
        start = time.time()
        rospy.loginfo("GUI teleop waiting for homing: param %s == true", self.home_done_param)
        while not rospy.is_shutdown():
            try:
                if bool(rospy.get_param(self.home_done_param, False)):
                    rospy.loginfo("Homing complete; starting GUI teleop.")
                    return
            except Exception:
                pass

            if self.home_wait_timeout > 0 and (time.time() - start) > self.home_wait_timeout:
                rospy.logerr("Timed out waiting for homing (%ss); starting GUI teleop anyway", self.home_wait_timeout)
                return
            time.sleep(0.1)

    def publish_cycle(self, _evt):
        if rospy.is_shutdown():
            return
        with self.lock:
            cmd = self.model.build_velocity_command()
            mode_msg = self.model.build_control_mode()
            paused = bool(self.teleop_paused)
        # Always publish control mode for observability.
        self.mode_pub.publish(mode_msg)
        # Only publish cartesian velocity when not paused (prevents fighting MoveIt).
        if not paused:
            try:
                self.velocity_pub.publish(cmd)
            except Exception:
                # During shutdown, publishers can close before timers stop.
                pass

    def _publish_mode_and_velocity(self):
        with self.lock:
            cmd = self.model.build_velocity_command()
            mode_msg = self.model.build_control_mode()
        if not self.teleop_paused:
            self.velocity_pub.publish(cmd)
        self.mode_pub.publish(mode_msg)

    def _publish_event(self, event_json):
        if not event_json:
            return
        self.event_pub.publish(String(data=event_json))

    def _apply_mode_change(self, mode):
        with self.lock:
            stop_event = self.model.stop_motion(reason="mode_change")
            mode_event = self.model.set_mode(mode)
            self._oracle_set_user_mode(mode)
        self._publish_event(stop_event)
        self._publish_event(mode_event)
        self._publish_mode_and_velocity()

    def _start_motion(self, axis, direction):
        with self.lock:
            event_json = self.model.start_motion(axis, direction)
        self._publish_event(event_json)
        self._publish_mode_and_velocity()

    def _stop_motion(self, reason="release", force_event=False):
        with self.lock:
            event_json = self.model.stop_motion(reason=reason, force_event=force_event)
        self._publish_event(event_json)
        self._publish_mode_and_velocity()

    def _send_gripper_goal(self, finger_value):
        goal = SetFingersPositionGoal()
        goal.fingers = FingerPosition(
            finger1=float(finger_value),
            finger2=float(finger_value),
            finger3=float(finger_value),
        )
        self.finger_client.send_goal(goal)

    def _gripper_action(self, action):
        target = self.finger_open_position if action == "open" else self.finger_close_position

        with self.lock:
            self.model.set_mode(TeleopCommandModel.MODE_GRIPPER)
            self._oracle_set_user_mode(TeleopCommandModel.MODE_GRIPPER)
            event_json = self.model.record_gripper_action(action, target)
        self._publish_event(event_json)
        self._publish_mode_and_velocity()
        self._send_gripper_goal(target)

    def _init_gui(self):
        import tkinter as tk

        self.tk = tk
        self.root = tk.Tk()
        self.root.title("PRIME GUI Teleop")
        # Start maximized by default for better at-a-glance operation.
        try:
            self.root.state("zoomed")
        except Exception:
            # Fallback for window managers that do not support "zoomed".
            sw = int(self.root.winfo_screenwidth())
            sh = int(self.root.winfo_screenheight())
            self.root.geometry(f"{sw}x{sh}+0+0")
        self.root.minsize(1280, 800)
        self.root.configure(bg="#1E1E2E")

        # Visual theme + sizing (larger targets for robot teleop safety/usability).
        bg_main = "#1E1E2E"
        bg_panel = "#2A2D3E"
        bg_surface = "#34384A"
        fg_primary = "#F0F3FA"
        fg_muted = "#C4CADB"
        accent = "#5CC8FF"
        accent_active = "#2EA5E6"
        warn = "#FF6B6B"
        good = "#2ECC71"
        btn_w = 13
        btn_h = 3
        base_font = ("Helvetica", 16, "bold")
        mode_font = ("Helvetica", 17, "bold")
        header_font = ("Helvetica", 16, "bold")
        status_font = ("Helvetica", 14, "bold")

        main_frame = tk.Frame(self.root, bg=bg_main)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Split main window into two equal halves:
        # - left: robot control
        # - right: oracle assistance
        left_panel = tk.Frame(main_frame, bg=bg_main)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(12, 6), pady=12)

        right_panel = tk.Frame(main_frame, bg=bg_main)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 12), pady=12)

        mode_frame = tk.LabelFrame(
            left_panel,
            text="Mode Selector",
            padx=12,
            pady=10,
            bg=bg_panel,
            fg=fg_primary,
            bd=2,
            relief=tk.GROOVE,
            font=header_font,
        )
        mode_frame.pack(fill=tk.X, padx=10, pady=8)

        self.mode_var = tk.StringVar(value=TeleopCommandModel.MODE_TRANSLATION)
        self.mode_text_var = tk.StringVar(value="Mode: translation")
        self.active_text_var = tk.StringVar(value="Command Active: no")

        tk.Radiobutton(
            mode_frame,
            text="Translation",
            value=TeleopCommandModel.MODE_TRANSLATION,
            variable=self.mode_var,
            command=self._on_mode_radio,
            indicatoron=0,
            width=14,
            height=2,
            font=mode_font,
            bg=bg_surface,
            fg=fg_primary,
            activebackground=bg_surface,
            activeforeground=fg_primary,
            selectcolor=accent_active,
            relief=tk.RAISED,
            bd=2,
            highlightthickness=0,
            takefocus=False,
            cursor="hand2",
        ).pack(side=tk.LEFT, padx=6, pady=4)
        tk.Radiobutton(
            mode_frame,
            text="Rotation",
            value=TeleopCommandModel.MODE_ROTATION,
            variable=self.mode_var,
            command=self._on_mode_radio,
            indicatoron=0,
            width=14,
            height=2,
            font=mode_font,
            bg=bg_surface,
            fg=fg_primary,
            activebackground=bg_surface,
            activeforeground=fg_primary,
            selectcolor=accent_active,
            relief=tk.RAISED,
            bd=2,
            highlightthickness=0,
            takefocus=False,
            cursor="hand2",
        ).pack(side=tk.LEFT, padx=6, pady=4)
        tk.Radiobutton(
            mode_frame,
            text="Gripper",
            value=TeleopCommandModel.MODE_GRIPPER,
            variable=self.mode_var,
            command=self._on_mode_radio,
            indicatoron=0,
            width=14,
            height=2,
            font=mode_font,
            bg=bg_surface,
            fg=fg_primary,
            activebackground=bg_surface,
            activeforeground=fg_primary,
            selectcolor=accent_active,
            relief=tk.RAISED,
            bd=2,
            highlightthickness=0,
            takefocus=False,
            cursor="hand2",
        ).pack(side=tk.LEFT, padx=6, pady=4)

        status_frame = tk.Frame(left_panel, bg=bg_main)
        status_frame.pack(fill=tk.X, padx=10, pady=6)
        tk.Label(
            status_frame,
            textvariable=self.mode_text_var,
            anchor="w",
            bg=bg_main,
            fg=fg_primary,
            font=status_font,
        ).pack(fill=tk.X, pady=(0, 2))
        tk.Label(
            status_frame,
            textvariable=self.active_text_var,
            anchor="w",
            bg=bg_main,
            fg=fg_muted,
            font=status_font,
        ).pack(fill=tk.X)

        buttons_frame = tk.LabelFrame(
            left_panel,
            text="Commands",
            padx=12,
            pady=12,
            bg=bg_panel,
            fg=fg_primary,
            bd=2,
            relief=tk.GROOVE,
            font=header_font,
        )
        buttons_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        self.translation_frame = tk.Frame(buttons_frame, bg=bg_panel)
        self.rotation_frame = tk.Frame(buttons_frame, bg=bg_panel)
        self.gripper_frame = tk.Frame(buttons_frame, bg=bg_panel)

        # Translation controls (2 buttons per row).
        t_row1 = tk.Frame(self.translation_frame, bg=bg_panel)
        t_row1.pack(fill=tk.X)
        self._add_hold_button(t_row1, "Left", "x", 1, width=btn_w, height=btn_h, font=base_font, bg=bg_surface, fg=fg_primary, activebackground=accent)
        self._add_hold_button(t_row1, "Right", "x", -1, width=btn_w, height=btn_h, font=base_font, bg=bg_surface, fg=fg_primary, activebackground=accent)

        t_row2 = tk.Frame(self.translation_frame, bg=bg_panel)
        t_row2.pack(fill=tk.X)
        self._add_hold_button(t_row2, "Backward", "y", 1, width=btn_w, height=btn_h, font=base_font, bg=bg_surface, fg=fg_primary, activebackground=accent)
        self._add_hold_button(t_row2, "Forward", "y", -1, width=btn_w, height=btn_h, font=base_font, bg=bg_surface, fg=fg_primary, activebackground=accent)

        t_row3 = tk.Frame(self.translation_frame, bg=bg_panel)
        t_row3.pack(fill=tk.X)
        self._add_hold_button(t_row3, "Up", "z", 1, width=btn_w, height=btn_h, font=base_font, bg=bg_surface, fg=fg_primary, activebackground=accent)
        self._add_hold_button(t_row3, "Down", "z", -1, width=btn_w, height=btn_h, font=base_font, bg=bg_surface, fg=fg_primary, activebackground=accent)

        # Rotation controls (2 buttons per row).
        r_row1 = tk.Frame(self.rotation_frame, bg=bg_panel)
        r_row1.pack(fill=tk.X)
        self._add_hold_button(r_row1, "Down", "rx", 1, width=btn_w, height=btn_h, font=base_font, bg=bg_surface, fg=fg_primary, activebackground=accent)
        self._add_hold_button(r_row1, "Up", "rx", -1, width=btn_w, height=btn_h, font=base_font, bg=bg_surface, fg=fg_primary, activebackground=accent)

        r_row2 = tk.Frame(self.rotation_frame, bg=bg_panel)
        r_row2.pack(fill=tk.X)
        self._add_hold_button(r_row2, "Left", "ry", 1, width=btn_w, height=btn_h, font=base_font, bg=bg_surface, fg=fg_primary, activebackground=accent)
        self._add_hold_button(r_row2, "Right", "ry", -1, width=btn_w, height=btn_h, font=base_font, bg=bg_surface, fg=fg_primary, activebackground=accent)

        r_row3 = tk.Frame(self.rotation_frame, bg=bg_panel)
        r_row3.pack(fill=tk.X)
        self._add_hold_button(r_row3, "+Rot", "rz", 1, width=btn_w, height=btn_h, font=base_font, bg=bg_surface, fg=fg_primary, activebackground=accent)
        self._add_hold_button(r_row3, "-Rot", "rz", -1, width=btn_w, height=btn_h, font=base_font, bg=bg_surface, fg=fg_primary, activebackground=accent)

        # Gripper controls (same style: 2 buttons in one row).
        g_row1 = tk.Frame(self.gripper_frame, bg=bg_panel)
        g_row1.pack(fill=tk.X)
        tk.Button(
            g_row1,
            text="Open",
            width=btn_w,
            height=btn_h,
            font=base_font,
            bg=good,
            fg="#102A1C",
            activebackground=good,
            activeforeground="#102A1C",
            relief=tk.RAISED,
            bd=2,
            cursor="hand2",
            takefocus=False,
            command=lambda: self._gripper_action("open"),
        ).pack(side=tk.LEFT, padx=8, pady=8)
        tk.Button(
            g_row1,
            text="Close",
            width=btn_w,
            height=btn_h,
            font=base_font,
            bg=warn,
            fg="#3A0D0D",
            activebackground=warn,
            activeforeground="#3A0D0D",
            relief=tk.RAISED,
            bd=2,
            cursor="hand2",
            takefocus=False,
            command=lambda: self._gripper_action("close"),
        ).pack(
            side=tk.LEFT, padx=8, pady=8
        )

        stop_frame = tk.Frame(left_panel, bg=bg_main)
        stop_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        tk.Button(
            stop_frame,
            text="STOP",
            width=20,
            height=2,
            font=("Helvetica", 18, "bold"),
            bg="#D63031",
            fg=fg_primary,
            activebackground="#D63031",
            activeforeground=fg_primary,
            relief=tk.RAISED,
            bd=3,
            cursor="hand2",
            takefocus=False,
            command=self._on_stop_button,
        ).pack(side=tk.LEFT)

        # Oracle side panel (always show when requested, even if disabled)
        if self.oracle_enabled_requested:
            self.oracle_panel = tk.LabelFrame(
                right_panel,
                text="Oracle Assist",
                padx=12,
                pady=12,
                bg=bg_panel,
                fg=fg_primary,
                bd=2,
                relief=tk.GROOVE,
                font=header_font,
            )
            self.oracle_panel.pack(fill=tk.BOTH, expand=True)
            if not self.oracle_available:
                status = "Oracle unavailable (import failed). Check logs."
            elif not PRIME_MSGS_AVAILABLE:
                status = "Oracle unavailable (PRIME msgs not built)."
            else:
                status = "Oracle ready. Press 'Ask assistance'."
            self.oracle_status_var = tk.StringVar(value=status)
            tk.Label(
                self.oracle_panel,
                textvariable=self.oracle_status_var,
                anchor="w",
                justify="left",
                wraplength=520,
                bg=bg_panel,
                fg=fg_muted,
                font=("Helvetica", 16, "bold"),
            ).pack(fill=tk.X, pady=(0, 10))

            btn_row = tk.Frame(self.oracle_panel, bg=bg_panel)
            btn_row.pack(fill=tk.X, pady=(0, 8))
            ask_btn = tk.Button(
                btn_row,
                text="Ask assistance",
                width=18,
                height=2,
                font=("Helvetica", 16, "bold"),
                bg=accent,
                fg="#0E2230",
                activebackground=accent,
                activeforeground="#0E2230",
                relief=tk.RAISED,
                bd=2,
                cursor="hand2",
                takefocus=False,
                command=self._oracle_ask_assistance,
            )
            ask_btn.pack(
                side=tk.LEFT, padx=(0, 6)
            )
            reset_btn = tk.Button(
                btn_row,
                text="Reset Oracle",
                width=14,
                height=2,
                font=("Helvetica", 16, "bold"),
                bg=bg_surface,
                fg=fg_primary,
                activebackground=bg_surface,
                activeforeground=fg_primary,
                relief=tk.RAISED,
                bd=2,
                cursor="hand2",
                takefocus=False,
                command=self._oracle_reset,
            )
            reset_btn.pack(side=tk.LEFT)

            if not self.oracle_enabled:
                ask_btn.configure(state="disabled")
                reset_btn.configure(state="disabled")

            self.oracle_choices_frame = tk.Frame(self.oracle_panel, bg=bg_panel)
            self.oracle_choices_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        self.root.bind_all("<ButtonRelease-1>", self._on_any_button_release, add="+")
        self.root.protocol("WM_DELETE_WINDOW", self._on_window_close)

        self._show_mode_frame(self.mode_var.get())
        self._schedule_ui_refresh()

    def _add_hold_button(
        self,
        parent,
        label,
        axis,
        direction,
        width=8,
        height=2,
        font=None,
        bg=None,
        fg=None,
        activebackground=None,
    ):
        rest_bg = bg or "#34384A"
        pressed_bg = activebackground or "#5CC8FF"
        rest_fg = fg or "#F0F3FA"
        pressed_fg = fg or "#F0F3FA"

        button = self.tk.Button(
            parent,
            text=label,
            width=width,
            height=height,
            font=font,
            bg=rest_bg,
            fg=rest_fg,
            # Disable Tk's built-in active highlight — we handle it ourselves
            # so touchscreens don't show the blue "focus" state on first tap.
            activebackground=rest_bg,
            activeforeground=rest_fg,
            relief=self.tk.RAISED,
            bd=2,
            highlightthickness=0,
            takefocus=False,
            cursor="hand2",
        )

        def on_press(_evt, a=axis, d=direction):
            # Force focus away from any previously focused widget so the
            # press event fires immediately (critical for touchscreens).
            button.focus_set()
            button.configure(bg=pressed_bg, relief=self.tk.SUNKEN)
            self._start_motion(a, d)

        def on_release(_evt):
            button.configure(bg=rest_bg, relief=self.tk.RAISED)
            self._stop_motion(reason="release")

        button.bind("<ButtonPress-1>", on_press)
        button.bind("<ButtonRelease-1>", on_release)
        # Also stop when finger/cursor leaves button while pressed.
        button.bind("<Leave>", lambda _evt: on_release(_evt) if self.model.is_motion_active() else None)
        button.pack(side=self.tk.LEFT, padx=8, pady=8)

    def _on_mode_radio(self):
        mode = self.mode_var.get()
        self._apply_mode_change(mode)
        self._show_mode_frame(mode)

    def _on_stop_button(self):
        self._stop_motion(reason="stop_button", force_event=True)

    def _on_any_button_release(self, _evt):
        with self.lock:
            active = self.model.is_motion_active()
        if active:
            self._stop_motion(reason="release")

    def _show_mode_frame(self, mode):
        for frame in (self.translation_frame, self.rotation_frame, self.gripper_frame):
            frame.pack_forget()

        if mode == TeleopCommandModel.MODE_TRANSLATION:
            self.translation_frame.pack(fill=self.tk.X)
        elif mode == TeleopCommandModel.MODE_ROTATION:
            self.rotation_frame.pack(fill=self.tk.X)
        else:
            self.gripper_frame.pack(fill=self.tk.X)

    def _schedule_ui_refresh(self):
        if self.root is None:
            return

        with self.lock:
            mode = self.model.mode
            active = self.model.is_motion_active()
        self.mode_text_var.set("Mode: %s" % mode)
        self.active_text_var.set("Command Active: %s" % ("yes" if active else "no"))

        if not rospy.is_shutdown():
            self.root.after(100, self._schedule_ui_refresh)

    def _on_window_close(self):
        self._stop_motion(reason="window_close")
        rospy.signal_shutdown("GUI window closed")
        if self.root is not None:
            self.root.destroy()
            self.root = None

    def run(self):
        if self.root is not None:
            self.root.mainloop()
        else:
            rospy.spin()

    # -----------------------
    # Oracle assist (GUI side panel)
    # -----------------------
    def _oracle_set_user_mode(self, mode: str):
        if not self.oracle_enabled or self.oracle_memory is None:
            return
        self.oracle_memory["user_state"] = {"mode": str(mode)}

    def _oracle_state_callback(self, msg: SymbolicState):
        if not self.oracle_enabled:
            return
        self.oracle_seen_symbolic_state = True
        objects = []
        for obj in msg.objects:
            cell_label = obj.grid_label or self._grid_label_from_index(
                obj.grid_cell, msg.grid_rows, msg.grid_cols
            )
            if not cell_label:
                continue
            objects.append(
                {
                    "id": obj.object_id,
                    "label": obj.label or obj.object_id,
                    "cell": cell_label,
                    "yaw": yaw_to_bin(obj.yaw_orientation),
                    "is_held": bool(obj.is_held),
                }
            )

        gripper_cell = msg.gripper_grid_label or self._grid_label_from_index(
            msg.gripper_grid_cell, msg.grid_rows, msg.grid_cols
        )
        gripper_yaw = yaw_to_bin(msg.gripper_yaw)
        z_val = None
        try:
            z_val = float(msg.gripper_height)
        except Exception:
            try:
                z_val = float(msg.gripper_position.z)
            except Exception:
                z_val = None

        with self.lock:
            self.oracle_objects = objects
            if gripper_cell:
                self._oracle_push_gripper_sample(gripper_cell, gripper_yaw, self._quantize_z(z_val))

    def _oracle_candidates_callback(self, msg: CandidateSet):
        if not self.oracle_enabled:
            return
        with self.lock:
            if self.oracle_memory is None:
                self._oracle_init_state()
            self.oracle_memory["candidates"] = list(msg.candidate_ids or [])

    def _oracle_init_state(self):
        self.oracle_state = OracleState(intended_obj_id="")
        self.oracle_memory = {
            "n_interactions": 0,
            "past_dialogs": [],
            "candidates": [],
            "last_tool_calls": [],
            "excluded_obj_ids": [],
            "last_action": {},
            "last_prompt": {},
            "user_state": {"mode": self.model.mode},
        }

    def _oracle_reset(self):
        if not self.oracle_enabled:
            return
        with self.lock:
            self._oracle_init_state()
            self.oracle_choices = []
        self._oracle_clear_choices()
        if self.oracle_status_var is not None:
            self.oracle_status_var.set("Oracle reset. Press 'Ask assistance'.")

    def _oracle_clear_choices(self):
        if self.oracle_choices_frame is None:
            return
        for child in list(self.oracle_choices_frame.children.values()):
            child.destroy()

    def _oracle_render_choices(self, choices):
        self._oracle_clear_choices()
        if self.oracle_choices_frame is None:
            return
        for choice in choices:
            self.tk.Button(
                self.oracle_choices_frame,
                text=choice,
                width=28,
                height=2,
                font=("Helvetica", 16, "bold"),
                bg="#3B425A",
                fg="#F0F3FA",
                activebackground="#3B425A",
                activeforeground="#F0F3FA",
                relief=self.tk.RAISED,
                bd=2,
                cursor="hand2",
                takefocus=False,
                command=lambda c=choice: self._oracle_on_choice(c),
            ).pack(fill=self.tk.X, pady=6)

    def _oracle_on_choice(self, choice_str: str):
        if not self.oracle_enabled:
            return
        user_content = choice_to_user_content(choice_str)
        if self.oracle_status_var is not None:
            self.oracle_status_var.set(f"You selected: {strip_choice_label(choice_str)}")
        with self.lock:
            if self.oracle_memory is None:
                self._oracle_init_state()
            self.oracle_memory["past_dialogs"].append({"role": "user", "content": user_content})
            auto_continue = apply_oracle_user_reply(
                user_content, self.oracle_objects, self.oracle_memory, self.oracle_state
            )
        self._oracle_clear_choices()
        if self.oracle_publish_response and self.oracle_response_pub and self.oracle_choices:
            response = PRIMEResponse()
            response.header.stamp = rospy.Time.now()
            response.query_id = ""
            response.selected_labels = [strip_choice_label(choice_str)]
            try:
                response.selected_indices = [self.oracle_choices.index(choice_str)]
            except Exception:
                response.selected_indices = []
            response.timed_out = False
            response.response_time = 0.0
            self.oracle_response_pub.publish(response)

        if auto_continue:
            self._oracle_ask_assistance()

    def _oracle_ask_assistance(self):
        if not self.oracle_enabled:
            return
        with self.lock:
            if self.oracle_memory is None:
                self._oracle_init_state()

            if not self.oracle_seen_symbolic_state:
                if self.oracle_status_var is not None:
                    self.oracle_status_var.set("Waiting for /prime/symbolic_state...")
                return
            if not self.oracle_objects:
                if self.oracle_status_var is not None:
                    self.oracle_status_var.set(
                        "Symbolic state received, but no object detections. "
                        "Check YOLO class mapping for state_builder/object_classes."
                    )
                return
            if not self.oracle_gripper_hist:
                if self.oracle_status_var is not None:
                    self.oracle_status_var.set("Waiting for gripper pose/grid in symbolic state...")
                return

            # Ensure candidates are populated.
            if not self.oracle_memory.get("candidates"):
                self.oracle_memory["candidates"] = [o["id"] for o in self.oracle_objects]

            # Update intended object if missing or stale.
            if not self.oracle_state.intended_obj_id or self.oracle_state.intended_obj_id not in {
                o["id"] for o in self.oracle_objects
            }:
                self.oracle_state.intended_obj_id = self._oracle_pick_intended()

            self.oracle_memory["user_state"] = {"mode": self.model.mode}

            tool_call = oracle_decide_tool(
                self.oracle_objects,
                self.oracle_gripper_hist,
                self.oracle_memory,
                self.oracle_state,
                user_state=self.oracle_memory["user_state"],
            )
            validate_tool_call(tool_call)

        tool = tool_call["tool"]
        args = tool_call["args"]
        rospy.loginfo("Oracle decided: %s %s", tool, str(args))
        with self.lock:
            self.oracle_memory["last_tool_calls"].append(tool)
            self.oracle_memory["last_tool_calls"] = self.oracle_memory["last_tool_calls"][-3:]

        if tool == "INTERACT":
            text = args["text"]
            choices = list(args["choices"])
            if self.oracle_status_var is not None:
                self.oracle_status_var.set(text)
            with self.lock:
                self.oracle_memory["past_dialogs"].append({"role": "assistant", "content": text})
                self.oracle_memory["n_interactions"] = int(self.oracle_memory.get("n_interactions", 0)) + 1
                self.oracle_memory["last_prompt"] = {"kind": args["kind"], "text": text, "choices": list(choices)}
                self.oracle_choices = choices
            if self.oracle_publish_query and self.oracle_query_pub:
                query = PRIMEQuery()
                query.header.stamp = rospy.Time.now()
                query.query_type = {
                    "QUESTION": PRIMEQuery.TYPE_QUESTION,
                    "SUGGESTION": PRIMEQuery.TYPE_SUGGESTION,
                    "CONFIRM": PRIMEQuery.TYPE_CONFIRMATION,
                }.get(args["kind"], PRIMEQuery.TYPE_QUESTION)
                query.content = text
                query.options = choices
                query.max_selections = 1
                query.timeout = 0.0
                query.query_id = uuid.uuid4().hex
                self.oracle_query_pub.publish(query)
            self._oracle_render_choices(choices)
            return

        # Motion tools -> publish ToolCall
        if tool in {"APPROACH", "ALIGN_YAW"}:
            obj_id = args["obj"]
            if self.oracle_status_var is not None:
                self.oracle_status_var.set(f"Publishing tool call: {tool}({obj_id})")
            if not self.oracle_tool_pub:
                rospy.logerr("Oracle tool publisher not available; cannot publish /prime/tool_call")
                if self.oracle_status_var is not None:
                    self.oracle_status_var.set("ERROR: cannot publish /prime/tool_call (publisher missing)")
                return
            try:
                msg = ToolCall()
                msg.header.stamp = rospy.Time.now()
                msg.tool_name = tool
                msg.target_object_id = obj_id
                msg.reasoning = "oracle"
                msg.call_id = uuid.uuid4().hex
                # Pause GUI teleop immediately so MoveIt execution isn't interfered with.
                self._pause_teleop_for_tool(msg)
                self.oracle_tool_pub.publish(msg)
                rospy.loginfo("Published /prime/tool_call: %s target=%s call_id=%s", tool, obj_id, msg.call_id)
                if self.oracle_status_var is not None:
                    self.oracle_status_var.set(f"Published: {tool}({obj_id})")
            except Exception as e:
                rospy.logerr("Failed publishing /prime/tool_call: %s", str(e))
                if self.oracle_status_var is not None:
                    self.oracle_status_var.set(f"ERROR publishing tool call: {e}")
                return
            with self.lock:
                self.oracle_memory["last_action"] = {"tool": tool, "obj": obj_id}
            return

    # -----------------------
    # Teleop pause/resume around tool execution
    # -----------------------
    def _pause_teleop_for_tool(self, call_msg):
        """
        Pause GUI teleop velocity publishing when a motion tool starts.
        We also send a single STOP (zero velocity) to ensure no residual teleop motion.
        """
        tool = getattr(call_msg, "tool_name", "")
        if tool in ("INTERACT", "", None):
            return
        with self.lock:
            self.teleop_paused = True
            self.teleop_pause_reason = str(tool)
            self.teleop_pause_call_id = getattr(call_msg, "call_id", None)
            # Stop any held GUI motion.
            stop_event = self.model.stop_motion(reason="paused_for_tool", force_event=True)
        self._publish_event(stop_event)
        # Publish a single zero velocity (then stop publishing while paused).
        try:
            self.velocity_pub.publish(PoseVelocity())
        except Exception:
            pass
        rospy.loginfo("GUI teleop paused for tool=%s call_id=%s", str(tool), str(self.teleop_pause_call_id))

    def _resume_teleop(self, reason=""):
        with self.lock:
            self.teleop_paused = False
            self.teleop_pause_reason = ""
            self.teleop_pause_call_id = None
        rospy.loginfo("GUI teleop resumed (%s)", reason or "tool complete")

    def _teleop_tool_call_callback(self, msg):
        # Pause for any external tool call too (LLM executive, etc.).
        if getattr(msg, "tool_name", "") in ("APPROACH", "ALIGN_YAW", "GRASP", "RELEASE"):
            self._pause_teleop_for_tool(msg)

    def _teleop_tool_result_callback(self, msg):
        # Resume when the matching tool finishes.
        call_id = getattr(msg, "call_id", None)
        with self.lock:
            paused = bool(self.teleop_paused)
            paused_id = self.teleop_pause_call_id
        if not paused:
            return
        # If we don't have a call id (or caller didn't set one), be permissive and resume.
        if not paused_id or (call_id and call_id == paused_id):
            self._resume_teleop(reason=f"tool_result {getattr(msg,'tool_name','')}")

    def _oracle_pick_intended(self) -> str:
        if not self.oracle_objects:
            return ""
        candidates = self.oracle_memory.get("candidates") or [o["id"] for o in self.oracle_objects]
        objs_by_id = {o["id"]: o for o in self.oracle_objects}
        gcell = self.oracle_gripper_hist[-1]["cell"]
        best_id = None
        best_dist = None
        for cid in candidates:
            obj = objs_by_id.get(cid)
            if not obj:
                continue
            dist = manhattan(gcell, obj["cell"])
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_id = cid
        return best_id or self.oracle_objects[0]["id"]

    def _oracle_push_gripper_sample(self, cell: str, yaw: str, z: str):
        sample = {"cell": cell, "yaw": yaw, "z": z}
        if not self.oracle_gripper_hist or self.oracle_gripper_hist[-1] != sample:
            self.oracle_gripper_hist.append(sample)
        if len(self.oracle_gripper_hist) > self.oracle_history_len:
            self.oracle_gripper_hist = self.oracle_gripper_hist[-self.oracle_history_len :]

    def _grid_label_from_index(self, cell_index, rows, cols) -> str:
        try:
            cell_index = int(cell_index)
            rows = int(rows) if rows else 3
            cols = int(cols) if cols else 3
        except Exception:
            return ""
        if rows <= 0 or cols <= 0:
            return ""
        row = cell_index // cols
        col = cell_index % cols
        if not (0 <= row < rows and 0 <= col < cols):
            return ""
        row_letter = chr(ord("A") + row)
        return f"{row_letter}{col+1}"

    def _quantize_z(self, z_val):
        if z_val is None:
            return "MID"
        try:
            z_val = float(z_val)
        except Exception:
            return "MID"
        z_min = float(self.oracle_z_min)
        z_max = float(self.oracle_z_max)
        if z_max <= z_min:
            return "MID"
        span = z_max - z_min
        if z_val <= z_min + span * 0.33:
            return "LOW"
        if z_val <= z_min + span * 0.66:
            return "MID"
        return "HIGH"


def main():
    try:
        node = GuiTeleopNode()
        node.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
