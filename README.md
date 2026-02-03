# PRIME ROS

**PRIME (Planning and Reasoning with Interactive Minimal-input Memory-Enhanced Executive)**

An LLM-based shared autonomy system for robotic manipulation with minimal user input, based on the IROS 2026 paper.

## Overview

PRIME enables fluent human–robot collaboration using only symbolic observations and minimal user interaction. The system:

1. **Observes** the workspace through YOLO object detection and Kinova arm state
2. **Reasons** over symbolic state using an LLM (Qwen 2.5) to infer user intent
3. **Interacts** with users through discrete choices (yes/no, multiple choice)
4. **Acts** autonomously when confidence is high, executing grasping primitives

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      PRIME System                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   │
│  │   YOLO-ROS   │──▶│State Builder │──▶│ LLM Executive│   │
│  │  (existing)  │   │  (symbolic)  │   │  (Qwen 2.5)  │   │
│  └──────────────┘   └──────────────┘   └──────┬───────┘   │
│                                               │            │
│  ┌──────────────┐   ┌──────────────┐         │            │
│  │   Joystick   │──▶│   Memory     │◀────────┤            │
│  │   Monitor    │   │   Module     │         │            │
│  └──────────────┘   └──────────────┘         ▼            │
│                                        ┌──────────────┐    │
│  ┌──────────────┐                     │Tool Executor │    │
│  │ User         │◀────────────────────│  (MoveIt)    │    │
│  │ Interface    │                     └──────────────┘    │
│  └──────────────┘                                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Prerequisites

- ROS Noetic
- Kinova ROS packages (`kinova-ros`)
- RealSense ROS packages (`realsense-ros`)
- YOLO ROS package (`yolo-ros`)
- MoveIt
- Ollama with Qwen 2.5 model (or compatible LLM server)

## Installation

1. Clone into your catkin workspace:
```bash
cd ~/catkin_ws/src
# Already in PRIME-ros directory
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Build the package:
```bash
cd ~/catkin_ws
catkin build prime_ros
source devel/setup.bash
```

4. Start Ollama with Qwen 2.5:
```bash
ollama run qwen2.5
```

## Usage

### Full System Launch

Launch everything (robot, camera, YOLO, MoveIt, PRIME):
```bash
roslaunch prime_ros prime_full.launch robot_type:=j2n6s300
```

### PRIME Only Launch

If robot and perception are already running:
```bash
roslaunch prime_ros prime.launch robot_type:=j2n6s300
```

### Configuration

Edit `config/prime_params.yaml` for:
- Workspace bounds (3x3 grid discretization)
- LLM endpoint and model settings
- Tool execution parameters
- User interface button mappings

## ROS Topics

### Published
- `/prime/symbolic_state` - Current symbolic state representation
- `/prime/candidate_objects` - Candidate target objects
- `/prime/control_mode` - Current control mode (translation/rotation/gripper)
- `/prime/joystick_state` - Joystick input state
- `/prime/query` - Queries to user (questions/confirmations)
- `/prime/tool_call` - Tool calls from LLM
- `/prime/tool_result` - Tool execution results

### Subscribed
- `/prime/response` - User responses to queries
- `/{robot}_driver/out/tool_pose` - Gripper pose
- `/{robot}_driver/out/joint_state` - Joint states
- `/yolo/image_with_bboxes` - YOLO detections

## Tools

The LLM can invoke these tools:

| Tool | Description |
|------|-------------|
| `INTERACT` | Ask user a question/confirmation |
| `APPROACH(obj)` | Move gripper toward object |
| `ALIGN_YAW(obj)` | Align gripper orientation |
| `GRASP` | Close gripper |
| `RELEASE` | Open gripper |

## User Input

Users can respond to PRIME queries via:

1. **Keyboard** (terminal):
   - `y` / `1` = Yes / Option 1
   - `n` / `2` = No / Option 2
   - `3-5` = Options 3-5
   - `q` = Cancel query

2. **Kinova Joystick** (button mapping configurable in params)

## Symbolic State

The workspace is discretized into a 3x3 grid:
```
[0, 1, 2]
[3, 4, 5]
[6, 7, 8]
```

Objects and gripper positions are tracked by grid cell, enabling efficient LLM reasoning.

## Memory System

PRIME maintains structured memory for multi-step reasoning:
- **Dialog history**: Past interactions and user responses
- **Candidate set**: Plausible target objects
- **Tool history**: Recent actions and outcomes

## Paper Reference

This implementation is based on:

> PRIME: An LLM-Based Executive for Interactive Manipulation Planning with Minimal User Effort
> Ali Rabiee et al., IROS 2026

## License

MIT License
