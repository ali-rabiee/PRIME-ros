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
"""

import os
import sys
import rospy
import moveit_commander


def main():
    rospy.init_node("go_home", anonymous=True)

    group_name = rospy.get_param("~group", "arm")
    target_name = rospy.get_param("~target", "Home")

    rospy.loginfo("Initializing MoveIt commander...")
    moveit_commander.roscpp_initialize(sys.argv)

    group = moveit_commander.MoveGroupCommander(group_name)
    group.set_planning_time(10.0)
    group.set_num_planning_attempts(5)
    group.set_max_velocity_scaling_factor(0.3)
    group.set_max_acceleration_scaling_factor(0.3)

    rospy.loginfo(f"Going to named target: {group_name}.{target_name}")
    group.set_named_target(target_name)
    ok = group.go(wait=True)
    group.stop()

    if ok:
        rospy.loginfo("Reached named target successfully.")
        return 0
    rospy.logerr("Failed to reach named target.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

