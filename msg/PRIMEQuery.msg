# PRIME Query Message
# Represents a query from the LLM to the user

Header header

# Query types
uint8 TYPE_QUESTION = 0
uint8 TYPE_SUGGESTION = 1
uint8 TYPE_CONFIRMATION = 2

# Query type
uint8 query_type

# The question/prompt text to display
string content

# Available options (for multiple choice)
string[] options

# Maximum number of selections allowed (1 for single choice, >1 for multiple)
uint8 max_selections

# Timeout in seconds (0 for no timeout)
float32 timeout

# Query ID for tracking responses
string query_id
