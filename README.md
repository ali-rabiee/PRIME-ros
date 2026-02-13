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
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐     │
│  │   YOLO-ROS   │──▶│State Builder │──▶│ LLM Executive│     │
│  │  (existing)  │   │  (symbolic)  │   │  (Qwen 2.5)  │     │
│  └──────────────┘   └──────────────┘   └──────┬───────┘     │
│                                               │             │
│  ┌──────────────┐   ┌──────────────┐          │             │
│  │ GUI Teleop   │──▶│   Memory     │◀─────────┤             │
│  │ (mode+cmd)   │   │   Module     │          │             │
│  └──────────────┘   └──────────────┘          ▼             │
│                                        ┌──────────────┐     │
│  ┌──────────────┐                      │Tool Executor │     │
│  │ User         │◀──────────────────── │  (MoveIt)    │     │
│  │ Interface    │                      └──────────────┘     │
│  └──────────────┘                                           │
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

### Evaluation Logging (prime_full.launch)

PRIME includes an optional, non-invasive logger node that records session metrics,
trajectories, tool calls/results, and video into an external folder. Each run
creates a time-stamped subdirectory inside the log root:

```
/home/tnlab/Desktop/PRIME_LOGS/YYYYMMDD_HHMMSS[_run_tag]/
```

Example:
```bash
roslaunch prime_ros prime_full.launch \
  robot_type:=j2n6s300 \
  enable_logging:=true \
  log_root:=/home/tnlab/Desktop/PRIME_LOGS \
  run_tag:=pilot01 \
  log_video:=true \
  log_collisions:=true
```

**Logging outputs per run:**
- `run_info.json` - run metadata + parameters
- `session.json` - start/end time + session length
- `metrics.json` - session length, grasp success, collisions, sample counts
- `tool_calls.jsonl` - tool call stream
- `tool_results.jsonl` - tool results (used for grasp success)
- `trajectory_joint.jsonl` - joint trajectory samples
- `trajectory_pose.jsonl` - end-effector pose samples
- `gui_events.jsonl` - GUI event stream
- `collisions.jsonl` - collision checks (if enabled)
- `video.mp4` - run video (if enabled)

**prime_full.launch logging args:**
- `enable_logging` (bool, default `false`): enable the logger node
- `log_root` (string, default `/home/tnlab/Desktop/PRIME_LOGS`): output root
- `run_tag` (string, default empty): optional tag appended to run folder name
- `session_start_mode` (string, default `first_event`): one of `node_start`, `first_event`, `first_tool_call`, `first_tool_result`
- `log_video` (bool, default `true`): record `video.mp4`
- `video_topic` (string, default `/camera/color/image_raw`): image source
- `video_fps` (float, default `30.0`): video writer FPS
- `log_trajectory` (bool, default `true`): record joint + pose trajectories
- `trajectory_rate_hz` (float, default `30.0`): trajectory sampling rate
- `log_collisions` (bool, default `false`): run MoveIt collision checks
- `collision_service` (string, default `/check_state_validity`): MoveIt validity service
- `collision_group` (string, default `arm`): MoveIt planning group
- `collision_rate_hz` (float, default `5.0`): collision check rate
- `collision_log_all` (bool, default `false`): log all checks (not just collisions)

**Note:** logging uses MoveIt’s state-validity service for collision checks. If
the service name differs in your setup, override `collision_service`.

### Configuration

Edit `config/prime_params.yaml` for:
- Workspace bounds (3x3 grid discretization)
- LLM endpoint and model settings
- Tool execution parameters
- GUI teleop speeds/publish rate

## ROS Topics

### Published
- `/prime/symbolic_state` - Current symbolic state representation
- `/prime/candidate_objects` - Candidate target objects
- `/prime/control_mode` - GUI-selected control mode and active-command flag
- `/prime/gui_teleop_event` - GUI action logs as JSON strings
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

Teleoperation and responses are split:

1. **GUI Teleop (`gui_teleop.py`)**
   - Modes: `Translation`, `Rotation`, `Gripper`
   - Press-and-hold axis buttons publish cartesian velocity on one axis only
   - `STOP` immediately zeroes velocity
   - `Open/Close` send finger action goals
   - Publishes `/prime/control_mode` and `/prime/gui_teleop_event`

2. **Keyboard Query Responses (`user_interface.py`)**
   - `y` / `1` = Yes / Option 1
   - `n` / `2` = No / Option 2
   - `3-5` = Options 3-5
   - `q` = Cancel query

## Testing

### Automated Test

```bash
cd ~/catkin_ws
catkin build prime_ros
catkin run_tests prime_ros
catkin_test_results build/prime_ros
```

Unit coverage is in `test/test_teleop_command_model.py` for:
- Translation/rotation axis-isolated velocity mapping
- Stop-to-zero behavior
- Control mode flags and `joystick_active` semantics
- GUI event JSON payload fields

### Manual Teleop Smoke Test

```bash
cd ~/catkin_ws
source devel/setup.bash
roslaunch prime_ros prime.launch
```

In separate terminals:

```bash
rostopic echo /prime/control_mode
rostopic echo /prime/gui_teleop_event
rostopic echo /<robot_type>_driver/in/cartesian_velocity
```

Validation steps:
1. Switch modes in the GUI and confirm `/prime/control_mode.mode` changes and motion is stopped.
2. Press/hold `+X` in Translation mode and confirm only `twist_linear_x` is non-zero.
3. Release the button and confirm velocity returns to all zeros immediately.
4. Press/hold `-Rz` in Rotation mode and confirm only `twist_angular_z` is non-zero and negative.
5. Click `Open`/`Close` in Gripper mode and confirm `/prime/gui_teleop_event` emits `type:\"gripper\"` entries.

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
