#!/usr/bin/env python
"""
Apply safety boundary (Kinova firmware protection zones) via kinova_driver services.

This is intended to run at startup from a launch file and then exit.

Services (provided by kinova_driver after we add them):
  /<robot_type>_driver/in/apply_protection_zones  (std_srvs/Trigger)
  /<robot_type>_driver/in/clear_protection_zones  (std_srvs/Trigger)
"""

import os
import sys
import rospy

from std_srvs.srv import Trigger


def main():
    rospy.init_node("apply_safety_boundary", anonymous=True)

    robot_type = rospy.get_param("~robot_type", "j2n6s300")
    apply_zones = bool(rospy.get_param("~apply", True))
    timeout = float(rospy.get_param("~timeout", 10.0))

    enabled = bool(rospy.get_param("/safety_bounds/enabled", False))
    if not enabled:
        rospy.loginfo("Safety bounds disabled (/safety_bounds/enabled=false). Skipping protection zones.")
        return 0

    srv_name = f"/{robot_type}_driver/in/" + ("apply_protection_zones" if apply_zones else "clear_protection_zones")
    rospy.loginfo(f"Waiting for service {srv_name} ...")
    try:
        rospy.wait_for_service(srv_name, timeout=timeout)
    except rospy.ROSException as e:
        rospy.logerr(f"Service not available: {e}")
        return 2

    try:
        proxy = rospy.ServiceProxy(srv_name, Trigger)
        resp = proxy()
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")
        return 3

    if resp.success:
        rospy.loginfo(f"Safety boundary service OK: {resp.message}")
        return 0
    rospy.logerr(f"Safety boundary service FAILED: {resp.message}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

