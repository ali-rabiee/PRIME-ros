#!/usr/bin/env python
"""
Joystick Monitor Node for PRIME

This node monitors the Kinova arm state to infer user input and control modes.
Since the Kinova ROS driver doesn't directly publish joystick values, we infer
control mode from velocity patterns and use joint velocity as proxy for user input.

The Kinova 3-axis joystick has these modes (toggled via mode button):
- Translation mode: Cartesian XYZ movement
- Rotation mode: Wrist rotation
- Gripper mode: Finger open/close

We detect which mode is active by observing which velocities are non-zero.
"""

import rospy
import numpy as np
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from kinova_msgs.msg import JointVelocity, FingerPosition

# Import PRIME messages (will be available after catkin build)
try:
    from prime_ros.msg import JoystickState, ControlMode
except ImportError:
    rospy.logwarn("PRIME messages not yet built. Run catkin build first.")
    JoystickState = None
    ControlMode = None


class JoystickMonitor:
    """
    Monitors Kinova arm to infer joystick input and control modes.
    
    Since we can't directly access JoystickCommand through ROS (it's only
    available through the C++ API), we infer user intent from:
    1. Joint velocities - indicates user is moving the arm
    2. Finger positions - indicates gripper mode activity
    3. Cartesian velocity patterns - infer translation vs rotation mode
    """
    
    def __init__(self):
        rospy.init_node('joystick_monitor', anonymous=False)
        
        # Parameters
        self.robot_type = rospy.get_param('~robot_type', 'j2n6s300')
        self.publish_rate = rospy.get_param('~publish_rate', 20.0)
        self.velocity_threshold = rospy.get_param('~velocity_threshold', 0.01)
        
        # State variables
        self.current_pose = None
        self.prev_pose = None
        self.current_joints = None
        self.prev_joints = None
        self.current_fingers = None
        self.prev_fingers = None
        self.last_update_time = rospy.Time.now()
        
        # Velocity tracking for mode inference
        self.linear_velocity_history = []
        self.angular_velocity_history = []
        self.finger_velocity_history = []
        self.history_length = 5
        
        # Control mode state
        self.current_mode = ControlMode.MODE_UNKNOWN if ControlMode else 255
        self.translation_active = False
        self.rotation_active = False
        self.fingers_active = False
        self.joystick_active = False
        
        # Driver topic prefix
        driver_prefix = f'/{self.robot_type}_driver'
        
        # Subscribers
        self.pose_sub = rospy.Subscriber(
            f'{driver_prefix}/out/tool_pose',
            PoseStamped,
            self.pose_callback
        )
        
        self.joint_sub = rospy.Subscriber(
            f'{driver_prefix}/out/joint_state',
            JointState,
            self.joint_callback
        )
        
        self.finger_sub = rospy.Subscriber(
            f'{driver_prefix}/out/finger_position',
            FingerPosition,
            self.finger_callback
        )
        
        # Publishers
        if JoystickState and ControlMode:
            self.joystick_pub = rospy.Publisher(
                '/prime/joystick_state',
                JoystickState,
                queue_size=10
            )
            
            self.mode_pub = rospy.Publisher(
                '/prime/control_mode',
                ControlMode,
                queue_size=10
            )
        else:
            self.joystick_pub = None
            self.mode_pub = None
            rospy.logwarn("Publishers disabled - build package first")
        
        # Timer for publishing
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / self.publish_rate),
            self.publish_state
        )
        
        rospy.loginfo(f"Joystick Monitor initialized for {self.robot_type}")
    
    def pose_callback(self, msg):
        """Handle tool pose updates."""
        self.prev_pose = self.current_pose
        self.current_pose = msg
    
    def joint_callback(self, msg):
        """Handle joint state updates."""
        self.prev_joints = self.current_joints
        self.current_joints = msg
        
        # Update velocity history from joint velocities
        if msg.velocity and len(msg.velocity) >= 6:
            # Compute approximate linear and angular velocities
            # First 3 joints contribute more to base motion (translation-like)
            # Last 3 joints contribute more to wrist motion (rotation-like)
            
            # This is a simplification - actual kinematics would be more accurate
            base_vel = np.sqrt(sum(v**2 for v in msg.velocity[:3]))
            wrist_vel = np.sqrt(sum(v**2 for v in msg.velocity[3:6]))
            
            self.linear_velocity_history.append(base_vel)
            self.angular_velocity_history.append(wrist_vel)
            
            # Keep history bounded
            if len(self.linear_velocity_history) > self.history_length:
                self.linear_velocity_history.pop(0)
            if len(self.angular_velocity_history) > self.history_length:
                self.angular_velocity_history.pop(0)
    
    def finger_callback(self, msg):
        """Handle finger position updates."""
        self.prev_fingers = self.current_fingers
        self.current_fingers = msg
        
        # Track finger velocity
        if self.prev_fingers:
            finger_delta = abs(msg.finger1 - self.prev_fingers.finger1)
            self.finger_velocity_history.append(finger_delta)
            
            if len(self.finger_velocity_history) > self.history_length:
                self.finger_velocity_history.pop(0)
    
    def infer_control_mode(self):
        """
        Infer the current control mode from velocity patterns.
        
        Returns tuple: (mode, translation_active, rotation_active, fingers_active, joystick_active)
        """
        # Compute average velocities from history
        avg_linear = np.mean(self.linear_velocity_history) if self.linear_velocity_history else 0
        avg_angular = np.mean(self.angular_velocity_history) if self.angular_velocity_history else 0
        avg_finger = np.mean(self.finger_velocity_history) if self.finger_velocity_history else 0
        
        # Determine which modes are active
        translation_active = avg_linear > self.velocity_threshold
        rotation_active = avg_angular > self.velocity_threshold
        fingers_active = avg_finger > 10  # Finger units are larger
        
        # Any activity means joystick is being used
        joystick_active = translation_active or rotation_active or fingers_active
        
        # Determine primary mode
        if fingers_active and not (translation_active or rotation_active):
            mode = ControlMode.MODE_GRIPPER if ControlMode else 2
        elif rotation_active and avg_angular > avg_linear:
            mode = ControlMode.MODE_ROTATION if ControlMode else 1
        elif translation_active:
            mode = ControlMode.MODE_TRANSLATION if ControlMode else 0
        else:
            mode = ControlMode.MODE_UNKNOWN if ControlMode else 255
        
        return mode, translation_active, rotation_active, fingers_active, joystick_active
    
    def compute_joystick_values(self):
        """
        Estimate joystick axis values from pose changes.
        
        Returns dict with estimated joystick values.
        """
        values = {
            'incline_left_right': 0.0,
            'incline_forward_backward': 0.0,
            'rotate': 0.0,
            'move_left_right': 0.0,
            'move_forward_backward': 0.0,
            'push_pull': 0.0,
        }
        
        if self.current_pose and self.prev_pose:
            # Compute position deltas
            dx = self.current_pose.pose.position.x - self.prev_pose.pose.position.x
            dy = self.current_pose.pose.position.y - self.prev_pose.pose.position.y
            dz = self.current_pose.pose.position.z - self.prev_pose.pose.position.z
            
            # Scale to -1 to 1 range (approximate)
            scale = 100.0  # Adjust based on expected velocities
            
            values['move_left_right'] = np.clip(dy * scale, -1.0, 1.0)
            values['move_forward_backward'] = np.clip(dx * scale, -1.0, 1.0)
            values['push_pull'] = np.clip(dz * scale, -1.0, 1.0)
            
            # For rotation, we'd need to compute quaternion differences
            # Simplified: use wrist joint velocity if available
            if self.current_joints and self.current_joints.velocity:
                if len(self.current_joints.velocity) >= 6:
                    wrist_vel = self.current_joints.velocity[5]  # Last joint
                    values['rotate'] = np.clip(wrist_vel / 0.5, -1.0, 1.0)
        
        return values
    
    def publish_state(self, event):
        """Publish joystick state and control mode."""
        if not self.joystick_pub or not self.mode_pub:
            return
        
        now = rospy.Time.now()
        
        # Infer control mode
        mode, trans, rot, fing, active = self.infer_control_mode()
        
        # Update internal state
        self.current_mode = mode
        self.translation_active = trans
        self.rotation_active = rot
        self.fingers_active = fing
        self.joystick_active = active
        
        # Compute joystick values
        joy_values = self.compute_joystick_values()
        
        # Publish ControlMode
        mode_msg = ControlMode()
        mode_msg.header = Header(stamp=now, frame_id='')
        mode_msg.mode = mode
        mode_msg.translation_active = trans
        mode_msg.rotation_active = rot
        mode_msg.fingers_active = fing
        mode_msg.joystick_active = active
        self.mode_pub.publish(mode_msg)
        
        # Publish JoystickState
        joy_msg = JoystickState()
        joy_msg.header = Header(stamp=now, frame_id='')
        joy_msg.button_values = [0] * 16  # We can't detect buttons this way
        joy_msg.incline_left_right = joy_values['incline_left_right']
        joy_msg.incline_forward_backward = joy_values['incline_forward_backward']
        joy_msg.rotate = joy_values['rotate']
        joy_msg.move_left_right = joy_values['move_left_right']
        joy_msg.move_forward_backward = joy_values['move_forward_backward']
        joy_msg.push_pull = joy_values['push_pull']
        joy_msg.is_active = active
        self.joystick_pub.publish(joy_msg)
    
    def run(self):
        """Run the node."""
        rospy.spin()


def main():
    try:
        monitor = JoystickMonitor()
        monitor.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
