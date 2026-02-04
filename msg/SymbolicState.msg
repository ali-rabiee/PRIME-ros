# Symbolic State Message
# Complete symbolic representation for LLM reasoning

Header header

# Detected objects in the workspace
ObjectState[] objects

# Gripper state
uint8 gripper_grid_cell
string gripper_grid_label
float32 gripper_yaw
float32 gripper_height
geometry_msgs/Point gripper_position

# Gripper history (last N positions for motion trend analysis)
geometry_msgs/Point[] gripper_history

# Current control mode
ControlMode control_mode

# Grid dimensions (default 3x3)
uint8 grid_rows
uint8 grid_cols

# Workspace bounds in robot frame (for grid discretization)
float32 workspace_x_min
float32 workspace_x_max
float32 workspace_y_min
float32 workspace_y_max
