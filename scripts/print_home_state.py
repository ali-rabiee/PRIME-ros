#!/usr/bin/env python
"""
Print current arm joint state in SRDF-friendly format.

Usage (real robot):
  rosrun prime_ros print_home_state.py _robot_type:=j2n6s300

This listens to:
  /<robot_type>_driver/out/joint_state

and prints:
  <joint name="..." value="..." />

for joints 1..6 of the arm.
"""

import os
import sys
import rospy
from sensor_msgs.msg import JointState


def main():
    rospy.init_node("print_home_state", anonymous=True)

    robot_type = rospy.get_param("~robot_type", "j2n6s300")
    topic = f"/{robot_type}_driver/out/joint_state"

    rospy.loginfo(f"Waiting for one JointState on {topic} ...")
    msg: JointState = rospy.wait_for_message(topic, JointState, timeout=10.0)

    wanted = [f"{robot_type}_joint_{i}" for i in range(1, 7)]
    name_to_pos = dict(zip(list(msg.name), list(msg.position)))

    missing = [n for n in wanted if n not in name_to_pos]
    if missing:
        rospy.logerr(f"Missing joints in JointState: {missing}")
        rospy.logerr("Cannot print SRDF lines. Check robot_type and topic.")
        return 2

    print("")
    print(f'<!-- Paste into {robot_type}.srdf group_state name=\"Home\" group=\"arm\" -->')
    for j in wanted:
        v = float(name_to_pos[j])
        print(f'  <joint name="{j}" value="{v:.7f}" />')
    print("")
    rospy.loginfo("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

