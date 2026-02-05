#!/usr/bin/env python3
"""
Calibrate planar homography for workspace (Option 2).

Goal:
- Place 4+ AprilTags on the WORKSPACE PLANE (table), visible to the camera.
- For each tag id, touch the tag center with the robot tool and press Enter.
- We will record:
    (u,v) = projected pixel of tag center (from apriltag pose + camera intrinsics)
    (x,y) = robot tool position in root frame
- Then we save a YAML containing workspace_homography/tag_xy_root which can be loaded
  once, and state_builder will compute H each episode from the observed tag pixels.

Notes:
- Tags do NOT need to stay visible during motion, only at calibration / episode start.
- This assumes your tool_pose topic is in the same root frame you want (typically "root").
"""

import os
import yaml
import rospy
import numpy as np

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo

try:
    from apriltag_ros.msg import AprilTagDetectionArray
    APRILTAG_AVAILABLE = True
except Exception:
    APRILTAG_AVAILABLE = False


class HomographyCalibrator:
    def __init__(self):
        rospy.init_node("calibrate_workspace_homography", anonymous=False)

        self.tag_ids = [int(v) for v in rospy.get_param("~tag_ids", [0, 1, 2, 3])]
        self.tag_detections_topic = rospy.get_param("~tag_detections_topic", "/tag_detections")
        self.camera_info_topic = rospy.get_param("~camera_info_topic", "/camera/color/camera_info")

        robot_type = rospy.get_param("robot/type", "j2n6s300")
        default_tool_pose = f"/{robot_type}_driver/out/tool_pose"
        self.tool_pose_topic = rospy.get_param("~tool_pose_topic", default_tool_pose)

        pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.out_yaml = rospy.get_param(
            "~out_yaml",
            os.path.join(pkg_dir, "config", "workspace_homography.yaml"),
        )
        self.median_window = int(rospy.get_param("~median_window", 15))
        self.lock_pose = bool(rospy.get_param("~lock_pose", True))
        self.disable_workspace_tag = bool(rospy.get_param("~disable_workspace_tag", True))

        # intrinsics
        self._ready = False
        self._fx = self._fy = self._cx = self._cy = None

        # latest tag translations in camera optical frame: tag_id -> np.array([X,Y,Z])
        self._t_cam_by_id = {}

        # latest tool pose
        self._tool_pose = None

        rospy.Subscriber(self.camera_info_topic, CameraInfo, self._camera_info_cb, queue_size=1)
        rospy.Subscriber(self.tool_pose_topic, PoseStamped, self._tool_pose_cb, queue_size=1)

        if not APRILTAG_AVAILABLE:
            raise RuntimeError("apriltag_ros messages not available. Install ros-noetic-apriltag-ros.")
        rospy.Subscriber(self.tag_detections_topic, AprilTagDetectionArray, self._tag_cb, queue_size=1)

    def _camera_info_cb(self, msg: CameraInfo):
        fx = float(msg.K[0])
        fy = float(msg.K[4])
        if fx <= 0 or fy <= 0:
            return
        self._fx = fx
        self._fy = fy
        self._cx = float(msg.K[2])
        self._cy = float(msg.K[5])
        self._ready = True

    def _tool_pose_cb(self, msg: PoseStamped):
        self._tool_pose = msg

    def _tag_cb(self, msg: "AprilTagDetectionArray"):
        if not msg.detections:
            return
        for det in msg.detections:
            try:
                if not det.id:
                    continue
                tid = int(det.id[0])
            except Exception:
                continue
            p = det.pose.pose.pose.position
            self._t_cam_by_id[tid] = np.array([float(p.x), float(p.y), float(p.z)], dtype=np.float64)

    def _project(self, t_cam: np.ndarray):
        if (not self._ready) or t_cam is None:
            return None
        X, Y, Z = float(t_cam[0]), float(t_cam[1]), float(t_cam[2])
        if Z <= 1e-6:
            return None
        u = self._fx * (X / Z) + self._cx
        v = self._fy * (Y / Z) + self._cy
        return float(u), float(v)

    def wait_for_inputs(self):
        rospy.loginfo("Waiting for camera_info + tool_pose + AprilTag detections...")
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self._ready and (self._tool_pose is not None):
                # require at least one tag
                if any((tid in self._t_cam_by_id) for tid in self.tag_ids):
                    return
            rate.sleep()

    def run(self):
        self.wait_for_inputs()

        tag_xy_root = {}
        zs = []

        rospy.loginfo("Starting homography tag (x,y) recording.")
        rospy.loginfo("Make sure ALL reference tags are visible to the camera now.")

        for tid in self.tag_ids:
            rospy.loginfo(f"--- Tag id {tid} ---")

            # wait until this tag is visible
            while not rospy.is_shutdown():
                t_cam = self._t_cam_by_id.get(tid)
                uv = self._project(t_cam) if t_cam is not None else None
                if uv is not None:
                    rospy.loginfo(f"Tag {tid} currently at pixel u,v ≈ ({uv[0]:.1f}, {uv[1]:.1f})")
                    break
                rospy.logwarn_throttle(1.0, f"Waiting for tag {tid} on {self.tag_detections_topic}...")
                rospy.sleep(0.1)

            rospy.loginfo("Move the robot tool tip to the CENTER of this tag.")
            try:
                input("Press Enter to RECORD this tag...")
            except EOFError:
                pass

            # snapshot
            tool = self._tool_pose
            t_cam = self._t_cam_by_id.get(tid)
            uv = self._project(t_cam) if t_cam is not None else None
            if tool is None or uv is None:
                raise RuntimeError("Missing tool_pose or tag pixel at record time.")

            x = float(tool.pose.position.x)
            y = float(tool.pose.position.y)
            z = float(tool.pose.position.z)
            tag_xy_root[str(tid)] = [x, y]
            zs.append(z)
            rospy.loginfo(f"Recorded tag {tid}: root (x,y,z)=({x:.4f},{y:.4f},{z:.4f}), pixel (u,v)=({uv[0]:.1f},{uv[1]:.1f})")

        table_z = float(np.mean(zs)) if zs else 0.0

        payload = {
            "workspace_homography": {
                "enabled": True,
                "lock_pose": self.lock_pose,
                "median_window": int(self.median_window),
                "tag_ids": [int(v) for v in self.tag_ids],
                "tag_xy_root": tag_xy_root,
                "table_z": float(table_z),
            }
        }
        if self.disable_workspace_tag:
            payload["workspace_tag"] = {"enabled": False}

        os.makedirs(os.path.dirname(self.out_yaml), exist_ok=True)
        with open(self.out_yaml, "w") as f:
            yaml.safe_dump(payload, f, default_flow_style=False, sort_keys=False)

        rospy.loginfo(f"Wrote homography calibration YAML to: {self.out_yaml}")
        rospy.loginfo("Next: rosparam load that YAML (or merge into prime_params.yaml) and restart state_builder.")


if __name__ == "__main__":
    HomographyCalibrator().run()

