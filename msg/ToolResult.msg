# Tool Result Message
# Result of a tool execution

Header header

# Call ID this result is for
string call_id

# Tool name that was executed
string tool_name

# Execution success
bool success

# Result status
uint8 STATUS_SUCCESS = 0
uint8 STATUS_FAILED = 1
uint8 STATUS_CANCELLED = 2
uint8 STATUS_TIMEOUT = 3
uint8 STATUS_INVALID = 4
uint8 status

# Error category (if failed)
string error_category

# Human-readable result message
string message

# For INTERACT tool: user response
string user_response
uint8[] selected_indices
