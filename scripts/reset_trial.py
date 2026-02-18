#!/usr/bin/env python3
"""
Reset current trial and return the robot to its home position.

Sequence:
1) Stop current trial logging (/prime/trial_logger/stop)
2) Home the robot (Kinova driver home_arm by default; MoveIt via go_home.py if ~home_method:=moveit)
3) Start new trial logging (/prime/trial_logger/start)
"""

import subprocess

import rospy
from std_srvs.srv import Trigger

try:
    from kinova_msgs.srv import HomeArm
    KINOVA_AVAILABLE = True
except Exception:
    HomeArm = None
    KINOVA_AVAILABLE = False


def _call_trigger(service_name, timeout=5.0):
    try:
        rospy.wait_for_service(service_name, timeout=timeout)
    except Exception as exc:
        rospy.logwarn("reset_trial: service %s not available: %s", service_name, exc)
        return False, str(exc)
    try:
        srv = rospy.ServiceProxy(service_name, Trigger)
        resp = srv()
        return bool(resp.success), str(resp.message)
    except Exception as exc:
        rospy.logwarn("reset_trial: failed calling %s: %s", service_name, exc)
        return False, str(exc)


def _run_go_home(robot_type, group, target, joint_states_topic, extra_args=None):
    cmd = [
        "rosrun",
        "prime_ros",
        "go_home.py",
        f"_group:={group}",
        f"_target:={target}",
        f"/joint_states:={joint_states_topic}",
    ]
    if extra_args:
        cmd.extend(extra_args)
    rospy.loginfo("reset_trial: running %s", " ".join(cmd))
    return subprocess.call(cmd)


def _call_home_arm(service_name, timeout=10.0):
    if not KINOVA_AVAILABLE:
        return False, "kinova_msgs/HomeArm not available"
    try:
        rospy.wait_for_service(service_name, timeout=timeout)
    except Exception as exc:
        rospy.logwarn("reset_trial: home_arm service %s not available: %s", service_name, exc)
        return False, str(exc)
    try:
        srv = rospy.ServiceProxy(service_name, HomeArm)
        resp = srv()
        msg = getattr(resp, "homearm_result", "") if resp is not None else ""
        return True, msg
    except Exception as exc:
        rospy.logwarn("reset_trial: failed calling home_arm %s: %s", service_name, exc)
        return False, str(exc)


def main():
    rospy.init_node("prime_trial_reset", anonymous=False)

    stop_service = str(rospy.get_param("~stop_service", "/prime/trial_logger/stop"))
    start_service = str(rospy.get_param("~start_service", "/prime/trial_logger/start"))

    robot_type = str(rospy.get_param("robot/type", rospy.get_param("~robot_type", "j2n6s300")))
    group = str(rospy.get_param("~home_group", "arm"))
    target = str(rospy.get_param("~home_target", "Home"))
    joint_states_topic = str(
        rospy.get_param("~joint_states_topic", f"/{robot_type}_driver/out/joint_state")
    )
    home_service = str(rospy.get_param("~home_service", f"/{robot_type}_driver/in/home_arm"))
    home_method = str(rospy.get_param("~home_method", "driver")).strip().lower()

    extra_args = []
    if rospy.has_param("~velocity_scaling"):
        extra_args.append(f"_velocity_scaling:={float(rospy.get_param('~velocity_scaling'))}")
    if rospy.has_param("~acceleration_scaling"):
        extra_args.append(f"_acceleration_scaling:={float(rospy.get_param('~acceleration_scaling'))}")
    if rospy.has_param("~delay"):
        extra_args.append(f"_delay:={float(rospy.get_param('~delay'))}")

    ok, msg = _call_trigger(stop_service, timeout=5.0)
    if ok:
        rospy.loginfo("reset_trial: stopped trial (%s)", msg)
    else:
        rospy.logwarn("reset_trial: stop failed (%s) - continuing", msg)

    ret = 0
    homed = False
    if home_method in ("driver", "auto"):
        ok, msg = _call_home_arm(home_service, timeout=15.0)
        if ok:
            homed = True
            rospy.loginfo("reset_trial: home_arm ok (%s)", msg or "service_call")
        else:
            rospy.logwarn("reset_trial: home_arm failed (%s)", msg)
            if home_method == "driver":
                ret = 1

    if home_method in ("moveit", "auto") and not homed:
        ret = _run_go_home(robot_type, group, target, joint_states_topic, extra_args=extra_args)
        if ret != 0:
            rospy.logerr("reset_trial: go_home failed with exit code %s", ret)

    ok, msg = _call_trigger(start_service, timeout=10.0)
    if ok:
        rospy.loginfo("reset_trial: started new trial (%s)", msg)
    else:
        rospy.logwarn("reset_trial: start failed (%s)", msg)

    return 0 if ret == 0 else ret


if __name__ == "__main__":
    raise SystemExit(main())
