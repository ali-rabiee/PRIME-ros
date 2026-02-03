cmake_minimum_required(VERSION 3.0.2)
project(prime_ros)

find_package(catkin REQUIRED COMPONENTS
  rospy
  roscpp
  std_msgs
  geometry_msgs
  sensor_msgs
  kinova_msgs
  message_generation
  actionlib
  actionlib_msgs
)

# Generate messages
add_message_files(
  FILES
  JoystickState.msg
  ControlMode.msg
  ObjectState.msg
  SymbolicState.msg
  PRIMEQuery.msg
  PRIMEResponse.msg
  CandidateSet.msg
  ToolCall.msg
  ToolResult.msg
)

# Generate services
add_service_files(
  FILES
  GetSymbolicState.srv
  ExecuteTool.srv
)

# Generate actions
add_action_files(
  FILES
  PRIMEAction.action
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  sensor_msgs
  actionlib_msgs
)

catkin_package(
  CATKIN_DEPENDS 
    rospy 
    roscpp 
    std_msgs 
    geometry_msgs 
    sensor_msgs 
    kinova_msgs 
    message_runtime
    actionlib
    actionlib_msgs
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

# Install Python scripts
catkin_install_python(PROGRAMS
  scripts/prime_node.py
  scripts/state_builder.py
  scripts/llm_executive.py
  scripts/tool_executor.py
  scripts/user_interface.py
  scripts/joystick_monitor.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

# Install config files
install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)
