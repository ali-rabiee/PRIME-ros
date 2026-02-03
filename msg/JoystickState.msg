# Kinova Joystick State Message
# Captures the state of the Kinova 3-axis joystick

Header header

# Button states (16 buttons, 1=pressed, 0=released)
int16[] button_values

# Stick inclination values (-1.0 to 1.0)
float32 incline_left_right
float32 incline_forward_backward
float32 rotate

# Stick movement values (-1.0 to 1.0)
float32 move_left_right
float32 move_forward_backward
float32 push_pull

# Derived: is any input active
bool is_active
