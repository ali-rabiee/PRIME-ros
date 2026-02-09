#!/usr/bin/env python
"""
Move the arm to a named MoveIt target (default: Home).

IMPORTANT:
MoveIt Python (moveit_commander) reads robot state from /joint_states.
On Kinova, real joint states are typically on:
  /j2n6s300_driver/out/joint_state

So run this with a remap (recommended):
  rosrun prime_ros go_home.py /joint_states:=/j2n6s300_driver/out/joint_state

Optional params:
  _group:=arm
  _target:=Home
  _delay:=2.0
  _retries:=3
  _retry_sleep:=2.0
"""

import os
import sys
import rospy
import moveit_commander
import actionlib
from sensor_msgs.msg import JointState
from moveit_msgs.msg import MoveGroupAction
from std_msgs.msg import Bool


def main():
    # Do NOT use anonymous=True: it breaks private params (~group/~target/~delay)
    # when launched from roslaunch because the node name gets a random suffix.
    rospy.init_node("go_home", anonymous=False)

    group_name = rospy.get_param("~group", "arm")
    target_name = rospy.get_param("~target", "Home")
    delay = float(rospy.get_param("~delay", 2.0))
    retries = int(rospy.get_param("~retries", 3))
    retry_sleep = float(rospy.get_param("~retry_sleep", 2.0))
    move_group_action = str(rospy.get_param("~move_group_action", "/move_group"))
    wait_joint_states = bool(rospy.get_param("~wait_joint_states", True))
    joint_states_timeout = float(rospy.get_param("~joint_states_timeout", 5.0))
    action_timeout = float(rospy.get_param("~move_group_timeout", 10.0))
    velocity_scaling = float(rospy.get_param("~velocity_scaling", 1.0))
    acceleration_scaling = float(rospy.get_param("~acceleration_scaling", 1.0))
    # Clamp to MoveIt expected range [0, 1]
    velocity_scaling = max(0.0, min(1.0, velocity_scaling))
    acceleration_scaling = max(0.0, min(1.0, acceleration_scaling))

    # Global flags so other nodes (e.g. GUI teleop) can wait for homing.
    done_param = str(rospy.get_param("~homing_done_param", "/prime/homing_done"))
    in_progress_param = str(rospy.get_param("~homing_in_progress_param", "/prime/homing_in_progress"))
    done_pub = rospy.Publisher(str(rospy.get_param("~homing_done_topic", "/prime/homing_done")), Bool, queue_size=1, latch=True)

    # Mark homing started
    rospy.set_param(done_param, False)
    rospy.set_param(in_progress_param, True)
    done_pub.publish(Bool(data=False))

    if delay > 0:
        rospy.loginfo(f"Waiting {delay:.1f}s before homing (let move_group start)...")
        rospy.sleep(delay)

    rospy.loginfo("Initializing MoveIt commander...")
    moveit_commander.roscpp_initialize(sys.argv)

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            # Ensure move_group action server is up before constructing commander.
            client = actionlib.SimpleActionClient(move_group_action, MoveGroupAction)
            if not client.wait_for_server(rospy.Duration(action_timeout)):
                raise RuntimeError(f"MoveIt action server not available: {move_group_action}")

            # Ensure we are receiving joint states (MoveIt python reads /joint_states).
            if wait_joint_states:
                rospy.loginfo("Waiting for /joint_states (remapped) ...")
                rospy.wait_for_message("/joint_states", JointState, timeout=joint_states_timeout)

            group = moveit_commander.MoveGroupCommander(group_name)
            group.set_planning_time(10.0)
            group.set_num_planning_attempts(5)
            # Kinova's velocity-based trajectory controller tends to under-track at very low speeds.
            # Use full speed by default for homing; override via ~velocity_scaling/~acceleration_scaling.
            group.set_max_velocity_scaling_factor(velocity_scaling)
            group.set_max_acceleration_scaling_factor(acceleration_scaling)

            rospy.loginfo(
                f"Going to named target: {group_name}.{target_name} "
                f"(attempt {attempt}/{retries}, vel_scale={velocity_scaling:.2f}, acc_scale={acceleration_scaling:.2f})"
            )
            named = []
            try:
                named = list(group.get_named_targets())
            except Exception:
                named = []
            if named and target_name not in named:
                raise RuntimeError(f"Unknown named target '{target_name}'. Available: {named}")

            group.set_named_target(target_name)
            ok = group.go(wait=True)
            group.stop()
            group.clear_pose_targets()

            if ok:
                rospy.loginfo("Reached named target successfully.")
                rospy.set_param(done_param, True)
                rospy.set_param(in_progress_param, False)
                done_pub.publish(Bool(data=True))
                return 0
            last_err = "MoveIt reported failure"
            rospy.logwarn(f"Failed to reach named target (attempt {attempt}/{retries}).")
        except Exception as e:
            last_err = str(e)
            rospy.logwarn(f"Exception during homing (attempt {attempt}/{retries}): {e}")

        if attempt < retries:
            rospy.sleep(retry_sleep)

    rospy.logerr(f"Failed to reach named target after {retries} attempts. Last error: {last_err}")
    rospy.set_param(in_progress_param, False)
    # keep done_param false
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

