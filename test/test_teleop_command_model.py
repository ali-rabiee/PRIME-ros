#!/usr/bin/env python3

import json
import os
import sys
import unittest

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "scripts"))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from gui_teleop import TeleopCommandModel  # noqa: E402


class TeleopCommandModelTest(unittest.TestCase):
    def test_translation_plus_x_axis_isolated(self):
        model = TeleopCommandModel(linear_speed=0.02, angular_speed=5.0)
        event_json = model.start_motion("x", 1, stamp=10.0)
        event = json.loads(event_json)

        cmd = model.build_velocity_command()
        self.assertAlmostEqual(cmd.twist_linear_x, 0.02)
        self.assertAlmostEqual(cmd.twist_linear_y, 0.0)
        self.assertAlmostEqual(cmd.twist_linear_z, 0.0)
        self.assertAlmostEqual(cmd.twist_angular_x, 0.0)
        self.assertAlmostEqual(cmd.twist_angular_y, 0.0)
        self.assertAlmostEqual(cmd.twist_angular_z, 0.0)

        for key in ("stamp", "source", "type", "mode", "axis", "direction", "value", "active"):
            self.assertIn(key, event)
        self.assertEqual(event["type"], "cartesian_velocity")
        self.assertEqual(event["axis"], "x")
        self.assertEqual(event["direction"], 1)
        self.assertTrue(event["active"])

    def test_rotation_minus_rz_axis_isolated(self):
        model = TeleopCommandModel(linear_speed=0.02, angular_speed=5.0)
        model.set_mode(TeleopCommandModel.MODE_ROTATION, stamp=20.0)
        model.start_motion("rz", -1, stamp=21.0)

        cmd = model.build_velocity_command()
        self.assertAlmostEqual(cmd.twist_linear_x, 0.0)
        self.assertAlmostEqual(cmd.twist_linear_y, 0.0)
        self.assertAlmostEqual(cmd.twist_linear_z, 0.0)
        self.assertAlmostEqual(cmd.twist_angular_x, 0.0)
        self.assertAlmostEqual(cmd.twist_angular_y, 0.0)
        self.assertAlmostEqual(cmd.twist_angular_z, -5.0)

    def test_stop_outputs_zero_velocity(self):
        model = TeleopCommandModel(linear_speed=0.02, angular_speed=5.0)
        model.start_motion("z", 1, stamp=30.0)
        model.stop_motion(stamp=31.0)

        cmd = model.build_velocity_command()
        self.assertAlmostEqual(cmd.twist_linear_x, 0.0)
        self.assertAlmostEqual(cmd.twist_linear_y, 0.0)
        self.assertAlmostEqual(cmd.twist_linear_z, 0.0)
        self.assertAlmostEqual(cmd.twist_angular_x, 0.0)
        self.assertAlmostEqual(cmd.twist_angular_y, 0.0)
        self.assertAlmostEqual(cmd.twist_angular_z, 0.0)

    def test_mode_flags_and_joystick_active_field(self):
        model = TeleopCommandModel(linear_speed=0.02, angular_speed=5.0)

        msg = model.build_control_mode(stamp=40.0)
        self.assertTrue(msg.translation_active)
        self.assertFalse(msg.rotation_active)
        self.assertFalse(msg.fingers_active)
        self.assertFalse(msg.joystick_active)

        model.set_mode(TeleopCommandModel.MODE_ROTATION, stamp=41.0)
        msg = model.build_control_mode(stamp=41.0)
        self.assertFalse(msg.translation_active)
        self.assertTrue(msg.rotation_active)
        self.assertFalse(msg.fingers_active)
        self.assertFalse(msg.joystick_active)

        model.start_motion("rx", 1, stamp=42.0)
        msg = model.build_control_mode(stamp=42.0)
        self.assertTrue(msg.joystick_active)

        model.stop_motion(stamp=43.0)
        model.set_mode(TeleopCommandModel.MODE_GRIPPER, stamp=44.0)
        msg = model.build_control_mode(stamp=44.0)
        self.assertFalse(msg.translation_active)
        self.assertFalse(msg.rotation_active)
        self.assertTrue(msg.fingers_active)

    def test_mode_change_and_gripper_event_payloads(self):
        model = TeleopCommandModel(linear_speed=0.02, angular_speed=5.0)

        mode_change = json.loads(model.set_mode(TeleopCommandModel.MODE_GRIPPER, stamp=50.0))
        self.assertEqual(mode_change["type"], "mode_change")
        self.assertEqual(mode_change["mode"], "gripper")

        gripper_event = json.loads(model.record_gripper_action("open", 0.0, stamp=51.0))
        for key in ("stamp", "source", "type", "mode", "value", "active", "action"):
            self.assertIn(key, gripper_event)
        self.assertEqual(gripper_event["type"], "gripper")
        self.assertEqual(gripper_event["action"], "open")


if __name__ == "__main__":
    unittest.main()
