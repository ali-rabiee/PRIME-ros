#!/usr/bin/env python
"""
State Builder Node for PRIME

Builds the symbolic state representation from:
1. YOLO object detections
2. Gripper pose from Kinova driver
3. Control mode from joystick monitor
4. Depth information from RealSense

The symbolic state discretizes the workspace into a 3x3 grid and tracks:
- Object positions (grid cell, label, pose)
- Gripper position and history
- Current control mode
"""

import rospy
import numpy as np
from collections import deque
from threading import Lock

from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped, Point
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

try:
    from prime_ros.msg import (
        SymbolicState, ObjectState, ControlMode,
        CandidateSet
    )
    from prime_ros.srv import GetSymbolicState, GetSymbolicStateResponse
    MSGS_AVAILABLE = True
except ImportError:
    rospy.logwarn("PRIME messages not built yet. Run catkin build.")
    MSGS_AVAILABLE = False


class StateBuilder:
    """
    Builds symbolic state representation for the PRIME system.
    
    Converts continuous sensor data into discrete symbolic representation
    suitable for LLM reasoning.
    """
    
    def __init__(self):
        rospy.init_node('state_builder', anonymous=False)
        
        # Parameters
        self.update_rate = rospy.get_param('~update_rate', 10.0)
        self.robot_type = rospy.get_param('robot/type', 'j2n6s300')
        
        # Workspace configuration
        self.grid_rows = rospy.get_param('workspace/grid_rows', 3)
        self.grid_cols = rospy.get_param('workspace/grid_cols', 3)
        self.x_min = rospy.get_param('workspace/x_min', 0.2)
        self.x_max = rospy.get_param('workspace/x_max', 0.7)
        self.y_min = rospy.get_param('workspace/y_min', -0.4)
        self.y_max = rospy.get_param('workspace/y_max', 0.4)
        self.z_min = rospy.get_param('workspace/z_min', 0.0)
        self.z_max = rospy.get_param('workspace/z_max', 0.5)
        
        # State history configuration
        self.history_length = rospy.get_param('state_builder/gripper_history_length', 10)
        self.position_threshold = rospy.get_param('state_builder/position_threshold', 0.02)
        
        # Thread safety
        self.lock = Lock()
        
        # State variables
        self.gripper_pose = None
        self.gripper_history = deque(maxlen=self.history_length)
        self.control_mode = None
        self.detected_objects = {}  # object_id -> ObjectState
        self.object_id_counter = 0
        
        # CV Bridge for image processing
        self.bridge = CvBridge()
        
        # YOLO detection results (from yolo_node.py)
        self.latest_detections = None
        self.latest_yolo_image = None
        
        # Subscribers
        driver_prefix = f'/{self.robot_type}_driver'
        
        self.pose_sub = rospy.Subscriber(
            f'{driver_prefix}/out/tool_pose',
            PoseStamped,
            self.gripper_pose_callback
        )
        
        if MSGS_AVAILABLE:
            self.mode_sub = rospy.Subscriber(
                '/prime/control_mode',
                ControlMode,
                self.control_mode_callback
            )
        
        # Subscribe to YOLO results
        self.yolo_sub = rospy.Subscriber(
            '/yolo/image_with_bboxes',
            Image,
            self.yolo_callback
        )
        
        # We'll also need the raw detections - for now using the annotated image
        # In a full implementation, you'd modify yolo_node to publish structured detections
        
        # Publishers
        if MSGS_AVAILABLE:
            self.state_pub = rospy.Publisher(
                '/prime/symbolic_state',
                SymbolicState,
                queue_size=10
            )
            
            self.candidates_pub = rospy.Publisher(
                '/prime/candidate_objects',
                CandidateSet,
                queue_size=10
            )
            
            # Service
            self.state_service = rospy.Service(
                '/prime/get_symbolic_state',
                GetSymbolicState,
                self.handle_get_state
            )
        
        # Timer for state updates
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / self.update_rate),
            self.update_state
        )
        
        rospy.loginfo("State Builder initialized")
    
    def gripper_pose_callback(self, msg):
        """Handle gripper pose updates."""
        with self.lock:
            self.gripper_pose = msg
            
            # Add to history if moved enough
            current_pos = Point(
                x=msg.pose.position.x,
                y=msg.pose.position.y,
                z=msg.pose.position.z
            )
            
            if len(self.gripper_history) == 0:
                self.gripper_history.append(current_pos)
            else:
                last_pos = self.gripper_history[-1]
                dist = np.sqrt(
                    (current_pos.x - last_pos.x)**2 +
                    (current_pos.y - last_pos.y)**2 +
                    (current_pos.z - last_pos.z)**2
                )
                if dist > self.position_threshold:
                    self.gripper_history.append(current_pos)
    
    def control_mode_callback(self, msg):
        """Handle control mode updates."""
        with self.lock:
            self.control_mode = msg
    
    def yolo_callback(self, msg):
        """Handle YOLO detection results."""
        with self.lock:
            self.latest_yolo_image = msg
            # TODO: Parse actual detection results
            # For now, we'll create mock objects based on the existing yolo_node structure
    
    def position_to_grid_cell(self, x, y):
        """
        Convert Cartesian position to grid cell index.
        
        Grid is organized as:
        [0, 1, 2]
        [3, 4, 5]
        [6, 7, 8]
        
        Where (0,0) is top-left in robot frame.
        """
        # Normalize to 0-1 range
        x_norm = (x - self.x_min) / (self.x_max - self.x_min)
        y_norm = (y - self.y_min) / (self.y_max - self.y_min)
        
        # Clip to valid range
        x_norm = np.clip(x_norm, 0, 0.999)
        y_norm = np.clip(y_norm, 0, 0.999)
        
        # Convert to grid indices
        row = int(x_norm * self.grid_rows)
        col = int(y_norm * self.grid_cols)
        
        # Convert to cell index
        cell = row * self.grid_cols + col
        
        return cell, row, col
    
    def get_gripper_yaw(self):
        """Extract yaw angle from gripper pose quaternion."""
        if not self.gripper_pose:
            return 0.0
        
        q = self.gripper_pose.pose.orientation
        # Simplified yaw extraction (assumes mostly vertical gripper)
        import math
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return yaw
    
    def compute_motion_trend(self):
        """
        Analyze gripper history to determine motion direction.
        
        Returns: (direction_x, direction_y, magnitude)
        """
        if len(self.gripper_history) < 2:
            return 0.0, 0.0, 0.0
        
        # Get recent positions
        positions = list(self.gripper_history)
        
        # Compute average direction from recent history
        dx_total = 0
        dy_total = 0
        for i in range(1, len(positions)):
            dx_total += positions[i].x - positions[i-1].x
            dy_total += positions[i].y - positions[i-1].y
        
        n = len(positions) - 1
        dx_avg = dx_total / n
        dy_avg = dy_total / n
        magnitude = np.sqrt(dx_avg**2 + dy_avg**2)
        
        return dx_avg, dy_avg, magnitude
    
    def compute_candidates(self, state):
        """
        Compute candidate objects based on gripper proximity and motion.
        
        As per PRIME paper:
        Ct = { ok ∈ O | gk ∈ N(gr_t) ∨ gk ∈ D(Ht) }
        
        Where:
        - N(gr_t) = gripper's grid cell and neighbors
        - D(Ht) = grid cells consistent with motion direction
        """
        candidates = []
        candidate_labels = []
        confidences = []
        
        if not self.gripper_pose:
            return candidates, candidate_labels, confidences, "No gripper pose"
        
        # Get gripper grid cell
        gripper_cell, gripper_row, gripper_col = self.position_to_grid_cell(
            self.gripper_pose.pose.position.x,
            self.gripper_pose.pose.position.y
        )
        
        # Get motion trend
        dx, dy, magnitude = self.compute_motion_trend()
        
        # Find neighboring cells (including current)
        neighbor_cells = set()
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr, nc = gripper_row + dr, gripper_col + dc
                if 0 <= nr < self.grid_rows and 0 <= nc < self.grid_cols:
                    neighbor_cells.add(nr * self.grid_cols + nc)
        
        # Find cells in motion direction
        direction_cells = set()
        if magnitude > 0.01:
            # Project motion direction to find target cells
            direction = np.arctan2(dy, dx)
            # Add cells in that general direction
            for obj_id, obj in self.detected_objects.items():
                obj_dx = obj.position.x - self.gripper_pose.pose.position.x
                obj_dy = obj.position.y - self.gripper_pose.pose.position.y
                if obj_dx == 0 and obj_dy == 0:
                    continue
                obj_direction = np.arctan2(obj_dy, obj_dx)
                angle_diff = abs(direction - obj_direction)
                if angle_diff < 0.5:  # ~30 degrees
                    direction_cells.add(obj.grid_cell)
        
        # Combine and find candidate objects
        valid_cells = neighbor_cells | direction_cells
        
        for obj_id, obj in self.detected_objects.items():
            if obj.grid_cell in valid_cells:
                candidates.append(obj_id)
                candidate_labels.append(obj.label)
                # Confidence based on proximity
                dist_to_gripper = np.sqrt(
                    (obj.position.x - self.gripper_pose.pose.position.x)**2 +
                    (obj.position.y - self.gripper_pose.pose.position.y)**2
                )
                confidence = max(0, 1 - dist_to_gripper / 0.5)
                confidences.append(confidence)
        
        reasoning = f"Found {len(candidates)} candidates near gripper cell {gripper_cell}"
        if magnitude > 0.01:
            reasoning += f" with motion toward direction {np.degrees(np.arctan2(dy, dx)):.0f}°"
        
        return candidates, candidate_labels, confidences, reasoning
    
    def update_detected_objects_from_yolo(self):
        """
        Update detected objects from YOLO.
        
        NOTE: This is a placeholder. In a full implementation, you would:
        1. Modify yolo_node.py to publish structured detection messages
        2. Use depth image to get 3D positions
        3. Track objects across frames with IDs
        
        For now, we'll create mock objects for testing.
        """
        # TODO: Implement actual YOLO integration
        # This would parse results from yolo_node and convert to ObjectState
        pass
    
    def add_mock_objects(self):
        """Add mock objects for testing (remove in production)."""
        if len(self.detected_objects) == 0:
            # Add some test objects
            mock_objects = [
                ('obj_1', 'mug', 0.4, 0.1, 0.05),
                ('obj_2', 'bottle', 0.5, -0.2, 0.08),
                ('obj_3', 'bin', 0.6, 0.3, 0.0),
            ]
            
            for obj_id, label, x, y, z in mock_objects:
                cell, row, col = self.position_to_grid_cell(x, y)
                obj = ObjectState()
                obj.object_id = obj_id
                obj.label = label
                obj.grid_cell = cell
                obj.grid_row = row
                obj.grid_col = col
                obj.position = Point(x=x, y=y, z=z)
                obj.yaw_orientation = 0.0
                obj.is_held = False
                obj.confidence = 0.9
                self.detected_objects[obj_id] = obj
    
    def build_symbolic_state(self):
        """Build the complete symbolic state message."""
        if not MSGS_AVAILABLE:
            return None
        
        state = SymbolicState()
        state.header = Header(stamp=rospy.Time.now(), frame_id='root')
        
        # Objects
        state.objects = list(self.detected_objects.values())
        
        # Gripper state
        if self.gripper_pose:
            cell, row, col = self.position_to_grid_cell(
                self.gripper_pose.pose.position.x,
                self.gripper_pose.pose.position.y
            )
            state.gripper_grid_cell = cell
            state.gripper_yaw = self.get_gripper_yaw()
            state.gripper_height = self.gripper_pose.pose.position.z
            state.gripper_position = Point(
                x=self.gripper_pose.pose.position.x,
                y=self.gripper_pose.pose.position.y,
                z=self.gripper_pose.pose.position.z
            )
        
        # Gripper history
        state.gripper_history = list(self.gripper_history)
        
        # Control mode
        if self.control_mode:
            state.control_mode = self.control_mode
        else:
            state.control_mode = ControlMode()
            state.control_mode.mode = ControlMode.MODE_UNKNOWN
        
        # Grid configuration
        state.grid_rows = self.grid_rows
        state.grid_cols = self.grid_cols
        state.workspace_x_min = self.x_min
        state.workspace_x_max = self.x_max
        state.workspace_y_min = self.y_min
        state.workspace_y_max = self.y_max
        
        return state
    
    def update_state(self, event):
        """Periodic state update and publishing."""
        with self.lock:
            # Update objects from YOLO (placeholder)
            self.update_detected_objects_from_yolo()
            
            # Add mock objects for testing
            self.add_mock_objects()
            
            # Build and publish state
            state = self.build_symbolic_state()
            if state and hasattr(self, 'state_pub'):
                self.state_pub.publish(state)
            
            # Compute and publish candidates
            if state and hasattr(self, 'candidates_pub'):
                cand_ids, cand_labels, confidences, reasoning = self.compute_candidates(state)
                
                candidates = CandidateSet()
                candidates.header = Header(stamp=rospy.Time.now())
                candidates.candidate_ids = cand_ids
                candidates.candidate_labels = cand_labels
                candidates.confidence_scores = confidences
                candidates.reasoning = reasoning
                self.candidates_pub.publish(candidates)
    
    def handle_get_state(self, req):
        """Service handler for getting current symbolic state."""
        with self.lock:
            state = self.build_symbolic_state()
            response = GetSymbolicStateResponse()
            if state:
                response.state = state
                response.success = True
                response.message = "State retrieved successfully"
            else:
                response.success = False
                response.message = "Failed to build state"
            return response
    
    def run(self):
        """Run the node."""
        rospy.spin()


def main():
    try:
        builder = StateBuilder()
        builder.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
