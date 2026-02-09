#!/usr/bin/env python3
"""GUI teleoperation for PRIME (replaces joystick pipeline)."""

import json
import threading
import time

import actionlib
import rospy
from kinova_msgs.msg import (
    FingerPosition,
    PoseVelocity,
    SetFingersPositionAction,
    SetFingersPositionGoal,
)
from std_msgs.msg import String

from prime_ros.msg import ControlMode


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
        with self.lock:
            cmd = self.model.build_velocity_command()
            mode_msg = self.model.build_control_mode()
        self.velocity_pub.publish(cmd)
        self.mode_pub.publish(mode_msg)

    def _publish_mode_and_velocity(self):
        with self.lock:
            cmd = self.model.build_velocity_command()
            mode_msg = self.model.build_control_mode()
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
            event_json = self.model.record_gripper_action(action, target)
        self._publish_event(event_json)
        self._publish_mode_and_velocity()
        self._send_gripper_goal(target)

    def _init_gui(self):
        import tkinter as tk

        self.tk = tk
        self.root = tk.Tk()
        self.root.title("PRIME GUI Teleop")

        mode_frame = tk.LabelFrame(self.root, text="Mode Selector", padx=8, pady=6)
        mode_frame.pack(fill=tk.X, padx=10, pady=6)

        self.mode_var = tk.StringVar(value=TeleopCommandModel.MODE_TRANSLATION)
        self.mode_text_var = tk.StringVar(value="Mode: translation")
        self.active_text_var = tk.StringVar(value="Command Active: no")

        tk.Radiobutton(
            mode_frame,
            text="Translation",
            value=TeleopCommandModel.MODE_TRANSLATION,
            variable=self.mode_var,
            command=self._on_mode_radio,
        ).pack(side=tk.LEFT, padx=4)
        tk.Radiobutton(
            mode_frame,
            text="Rotation",
            value=TeleopCommandModel.MODE_ROTATION,
            variable=self.mode_var,
            command=self._on_mode_radio,
        ).pack(side=tk.LEFT, padx=4)
        tk.Radiobutton(
            mode_frame,
            text="Gripper",
            value=TeleopCommandModel.MODE_GRIPPER,
            variable=self.mode_var,
            command=self._on_mode_radio,
        ).pack(side=tk.LEFT, padx=4)

        status_frame = tk.Frame(self.root)
        status_frame.pack(fill=tk.X, padx=10, pady=4)
        tk.Label(status_frame, textvariable=self.mode_text_var, anchor="w").pack(fill=tk.X)
        tk.Label(status_frame, textvariable=self.active_text_var, anchor="w").pack(fill=tk.X)

        buttons_frame = tk.LabelFrame(self.root, text="Commands", padx=8, pady=8)
        buttons_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

        self.translation_frame = tk.Frame(buttons_frame)
        self.rotation_frame = tk.Frame(buttons_frame)
        self.gripper_frame = tk.Frame(buttons_frame)

        self._add_hold_button(self.translation_frame, "+X", "x", 1)
        self._add_hold_button(self.translation_frame, "-X", "x", -1)
        self._add_hold_button(self.translation_frame, "+Y", "y", 1)
        self._add_hold_button(self.translation_frame, "-Y", "y", -1)
        self._add_hold_button(self.translation_frame, "+Z", "z", 1)
        self._add_hold_button(self.translation_frame, "-Z", "z", -1)

        self._add_hold_button(self.rotation_frame, "+Rx", "rx", 1)
        self._add_hold_button(self.rotation_frame, "-Rx", "rx", -1)
        self._add_hold_button(self.rotation_frame, "+Ry", "ry", 1)
        self._add_hold_button(self.rotation_frame, "-Ry", "ry", -1)
        self._add_hold_button(self.rotation_frame, "+Rz", "rz", 1)
        self._add_hold_button(self.rotation_frame, "-Rz", "rz", -1)

        tk.Button(self.gripper_frame, text="Open", width=10, command=lambda: self._gripper_action("open")).pack(
            side=tk.LEFT, padx=4, pady=4
        )
        tk.Button(self.gripper_frame, text="Close", width=10, command=lambda: self._gripper_action("close")).pack(
            side=tk.LEFT, padx=4, pady=4
        )

        stop_frame = tk.Frame(self.root)
        stop_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        tk.Button(stop_frame, text="STOP", width=14, command=self._on_stop_button).pack(side=tk.LEFT)

        self.root.bind_all("<ButtonRelease-1>", self._on_any_button_release, add="+")
        self.root.protocol("WM_DELETE_WINDOW", self._on_window_close)

        self._show_mode_frame(self.mode_var.get())
        self._schedule_ui_refresh()

    def _add_hold_button(self, parent, label, axis, direction):
        button = self.tk.Button(parent, text=label, width=8)
        button.bind("<ButtonPress-1>", lambda _evt, a=axis, d=direction: self._start_motion(a, d))
        button.bind("<ButtonRelease-1>", lambda _evt: self._stop_motion(reason="release"))
        button.pack(side=self.tk.LEFT, padx=4, pady=4)

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


def main():
    try:
        node = GuiTeleopNode()
        node.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
