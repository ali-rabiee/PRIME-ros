# Object State Message
# Represents a single detected object in the workspace

# Unique identifier for tracking
string object_id

# Semantic label (e.g., "mug", "bottle", "object", "bin")
string label

# Grid cell position (0-8 for 3x3 grid)
uint8 grid_cell

# Grid row and column (0-2)
uint8 grid_row
uint8 grid_col

# Human-readable grid label (A1..C3)
string grid_label

# Object pose in robot frame
geometry_msgs/Point position
float32 yaw_orientation

# Bounding box center in image coordinates
int32 bbox_center_x
int32 bbox_center_y

# Is the object currently held by the gripper
bool is_held

# Confidence score from detection
float32 confidence
