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


def main():
    rospy.init_node("go_home", anonymous=True)

    group_name = rospy.get_param("~group", "arm")
    target_name = rospy.get_param("~target", "Home")
    delay = float(rospy.get_param("~delay", 2.0))
    retries = int(rospy.get_param("~retries", 3))
    retry_sleep = float(rospy.get_param("~retry_sleep", 2.0))

    if delay > 0:
        rospy.loginfo(f"Waiting {delay:.1f}s before homing (let move_group start)...")
        rospy.sleep(delay)

    rospy.loginfo("Initializing MoveIt commander...")
    moveit_commander.roscpp_initialize(sys.argv)

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            group = moveit_commander.MoveGroupCommander(group_name)
            group.set_planning_time(10.0)
            group.set_num_planning_attempts(5)
            group.set_max_velocity_scaling_factor(0.3)
            group.set_max_acceleration_scaling_factor(0.3)

            rospy.loginfo(f"Going to named target: {group_name}.{target_name} (attempt {attempt}/{retries})")
            group.set_named_target(target_name)
            ok = group.go(wait=True)
            group.stop()

            if ok:
                rospy.loginfo("Reached named target successfully.")
                return 0
            last_err = "MoveIt reported failure"
            rospy.logwarn(f"Failed to reach named target (attempt {attempt}/{retries}).")
        except Exception as e:
            last_err = str(e)
            rospy.logwarn(f"Exception during homing (attempt {attempt}/{retries}): {e}")

        if attempt < retries:
            rospy.sleep(retry_sleep)

    rospy.logerr(f"Failed to reach named target after {retries} attempts. Last error: {last_err}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

