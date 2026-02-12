# Control Mode Message
# Represents the current control mode of the Kinova arm

Header header

# Control mode constants
uint8 MODE_TRANSLATION = 0
uint8 MODE_ROTATION = 1
uint8 MODE_GRIPPER = 2
uint8 MODE_UNKNOWN = 255

# Current control mode
uint8 mode

# Individual mode flags from SystemStatus
bool translation_active
bool rotation_active
bool fingers_active

# GUI active-command indicator (repurposed legacy field name)
bool joystick_active
