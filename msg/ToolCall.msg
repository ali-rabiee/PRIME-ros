# Tool Call Message
# Represents a tool invocation from the LLM

Header header

# Tool names
string TOOL_INTERACT = "INTERACT"
string TOOL_APPROACH = "APPROACH"
string TOOL_ALIGN_YAW = "ALIGN_YAW"
string TOOL_GRASP = "GRASP"
string TOOL_RELEASE = "RELEASE"

# Tool being called
string tool_name

# Target object ID (for APPROACH, ALIGN_YAW, GRASP)
string target_object_id

# For INTERACT tool
uint8 interact_type  # 0=question, 1=suggestion, 2=confirmation
string interact_content
string[] interact_options

# LLM reasoning/explanation
string reasoning

# Unique call ID for tracking
string call_id
