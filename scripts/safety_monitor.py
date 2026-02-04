#!/usr/bin/env python
"""
Safety Monitor Node - Software fallback for workspace boundary enforcement.

Monitors the end-effector position and sends emergency stop commands
if the robot exits the defined safety bounds. This is a fallback in case
firmware protection zones don't work for joystick control.
"""

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Empty
from kinova_msgs.srv import Stop
import threading


class SafetyMonitor:
    def __init__(self):
        rospy.init_node('safety_monitor', anonymous=False)

        # Load safety bounds from parameters
        self.enabled = rospy.get_param('/safety_bounds/enabled', True)
        self.x_min = rospy.get_param('/safety_bounds/x_min', 0.2)
        self.x_max = rospy.get_param('/safety_bounds/x_max', 0.7)
        self.y_min = rospy.get_param('/safety_bounds/y_min', -0.4)
        self.y_max = rospy.get_param('/safety_bounds/y_max', 0.4)
        self.z_min = rospy.get_param('/safety_bounds/z_min', 0.0)
        self.z_max = rospy.get_param('/safety_bounds/z_max', 0.6)

        # Get robot prefix for service names
        self.robot_name = rospy.get_param('~robot_name', 'j2n6s300')
        self.stop_service_name = '/{}_driver/in/stop'.format(self.robot_name)
        self.start_service_name = '/{}_driver/in/start'.format(self.robot_name)

        # Monitoring parameters
        self.check_rate = rospy.get_param('~check_rate', 50)  # Hz
        self.margin = rospy.get_param('~margin', 0.02)  # 2cm margin before hard stop
        self.warn_margin = rospy.get_param('~warn_margin', 0.05)  # 5cm warning zone

        # State
        self.current_pose = None
        self.is_stopped = False
        self.violation_count = 0
        self.lock = threading.Lock()

        # Subscribe to robot pose
        self.pose_sub = rospy.Subscriber(
            '/{}_driver/out/tool_pose'.format(self.robot_name),
            PoseStamped,
            self.pose_callback,
            queue_size=1
        )

        # Publisher to notify violations
        self.violation_pub = rospy.Publisher(
            '~boundary_violation',
            PoseStamped,
            queue_size=1
        )

        rospy.loginfo("Safety Monitor initialized")
        rospy.loginfo("  Bounds: x[%.2f, %.2f] y[%.2f, %.2f] z[%.2f, %.2f]",
                      self.x_min, self.x_max,
                      self.y_min, self.y_max,
                      self.z_min, self.z_max)
        rospy.loginfo("  Stop service: %s", self.stop_service_name)

    def pose_callback(self, msg):
        with self.lock:
            self.current_pose = msg

    def in_bounds(self, x, y, z, margin=0.0):
        """Check if position is within bounds (with optional margin)."""
        return (self.x_min - margin <= x <= self.x_max + margin and
                self.y_min - margin <= y <= self.y_max + margin and
                self.z_min - margin <= z <= self.z_max + margin)

    def get_violation_info(self, x, y, z):
        """Get information about which bounds are violated."""
        violations = []
        if x < self.x_min:
            violations.append('x<{:.2f} (x={:.3f})'.format(self.x_min, x))
        if x > self.x_max:
            violations.append('x>{:.2f} (x={:.3f})'.format(self.x_max, x))
        if y < self.y_min:
            violations.append('y<{:.2f} (y={:.3f})'.format(self.y_min, y))
        if y > self.y_max:
            violations.append('y>{:.2f} (y={:.3f})'.format(self.y_max, y))
        if z < self.z_min:
            violations.append('z<{:.2f} (z={:.3f})'.format(self.z_min, z))
        if z > self.z_max:
            violations.append('z>{:.2f} (z={:.3f})'.format(self.z_max, z))
        return violations

    def stop_robot(self):
        """Send emergency stop command to the robot."""
        try:
            rospy.wait_for_service(self.stop_service_name, timeout=0.5)
            stop_srv = rospy.ServiceProxy(self.stop_service_name, Stop)
            stop_srv()
            rospy.logwarn("SAFETY STOP: Robot stopped due to boundary violation!")
            return True
        except rospy.ROSException as e:
            rospy.logerr("Failed to stop robot: %s", str(e))
            return False
        except Exception as e:
            rospy.logerr("Error stopping robot: %s", str(e))
            return False

    def start_robot(self):
        """Resume robot control after it's back in bounds."""
        try:
            rospy.wait_for_service(self.start_service_name, timeout=0.5)
            start_srv = rospy.ServiceProxy(self.start_service_name, Stop)
            start_srv()
            rospy.loginfo("Robot control resumed - back in bounds")
            return True
        except Exception as e:
            rospy.logwarn("Could not resume robot: %s", str(e))
            return False

    def run(self):
        """Main monitoring loop."""
        rate = rospy.Rate(self.check_rate)

        while not rospy.is_shutdown():
            if not self.enabled:
                rate.sleep()
                continue

            with self.lock:
                pose = self.current_pose

            if pose is None:
                rate.sleep()
                continue

            x = pose.pose.position.x
            y = pose.pose.position.y
            z = pose.pose.position.z

            # Check if in bounds
            in_safe_zone = self.in_bounds(x, y, z)
            in_warn_zone = self.in_bounds(x, y, z, margin=self.warn_margin)
            in_stop_zone = self.in_bounds(x, y, z, margin=-self.margin)

            if not in_stop_zone and not self.is_stopped:
                # HARD VIOLATION - Stop immediately
                violations = self.get_violation_info(x, y, z)
                rospy.logerr("BOUNDARY VIOLATION: %s", ', '.join(violations))

                if self.stop_robot():
                    self.is_stopped = True
                    self.violation_count += 1
                    self.violation_pub.publish(pose)

            elif not in_safe_zone and not self.is_stopped:
                # In margin zone - warn
                violations = self.get_violation_info(x, y, z)
                rospy.logwarn_throttle(1.0, "Near boundary: %s", ', '.join(violations))

            elif in_safe_zone and self.is_stopped:
                # Back in bounds - can resume
                rospy.loginfo("Robot back in safe zone at (%.3f, %.3f, %.3f)", x, y, z)
                self.is_stopped = False
                # Note: We don't auto-resume - user must manually restart
                rospy.logwarn("Robot stopped. Use /%s_driver/in/start to resume.", self.robot_name)

            rate.sleep()


def main():
    try:
        monitor = SafetyMonitor()
        monitor.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
