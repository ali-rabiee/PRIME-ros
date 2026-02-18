#!/usr/bin/env python3
"""
Start PRIME trial logging with mode/subject/difficulty in one short command.

Examples:
  rosrun prime_ros start_trial.py manual s1 easy
  rosrun prime_ros start_trial.py assistive s2 hard --reason trial_start
"""

import argparse

import rospy

from prime_ros.srv import StartTrial


def parse_args():
    parser = argparse.ArgumentParser(description="Start PRIME trial logger.")
    parser.add_argument("mode", help="Trial mode, e.g. manual or assistive")
    parser.add_argument("subject_id", help="Subject ID, e.g. s1")
    parser.add_argument(
        "difficulty",
        nargs="?",
        default="",
        help="Task difficulty, e.g. easy or hard (optional)",
    )
    parser.add_argument(
        "--reason",
        default="manual_start",
        help="Reason string recorded in trial metadata (default: manual_start)",
    )
    parser.add_argument(
        "--service",
        default="/prime/trial_logger/start",
        help="StartTrial service name (default: /prime/trial_logger/start)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for service (default: 10.0)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rospy.init_node("prime_trial_start", anonymous=True)

    try:
        rospy.wait_for_service(args.service, timeout=args.timeout)
    except Exception as exc:
        rospy.logerr("start_trial: service %s not available: %s", args.service, exc)
        return 1

    try:
        srv = rospy.ServiceProxy(args.service, StartTrial)
        resp = srv(
            mode=args.mode,
            subject_id=args.subject_id,
            difficulty=args.difficulty,
            reason=args.reason,
        )
    except Exception as exc:
        rospy.logerr("start_trial: failed calling %s: %s", args.service, exc)
        return 1

    if resp.success:
        rospy.loginfo("start_trial: %s", resp.message)
        return 0

    rospy.logwarn("start_trial: %s", resp.message)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
