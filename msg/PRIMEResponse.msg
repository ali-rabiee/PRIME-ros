# PRIME Response Message
# User's response to a PRIME query

Header header

# Query ID this response is for
string query_id

# Selected option indices (0-indexed)
uint8[] selected_indices

# Selected option labels (for convenience)
string[] selected_labels

# Was the query timed out
bool timed_out

# Response time in seconds
float32 response_time
