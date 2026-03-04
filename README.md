# PRIME ROS

**PRIME (Planning and Reasoning with Interactive Minimal-input Memory-Enhanced Executive)**

An LLM-based shared autonomy system for robotic manipulation with minimal user input, based on the IROS 2026 paper.

## Overview

PRIME enables fluent human–robot collaboration using symbolic observations and minimal user interaction. The system:

1. **Observes** the workspace through YOLO object detection and Kinova arm state
2. **Reasons** over symbolic state (optionally using an LLM) to infer user intent
3. **Interacts** with users through discrete choices (yes/no, multiple choice)
4. **Acts** via tool primitives: APPROACH, ALIGN_YAW, GRASP, RELEASE (MoveIt + Kinova driver)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      PRIME System                           │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐    │
│  │   YOLO-ROS   │──▶│State Builder │──▶│ LLM Executive│    │
│  │  (detection) │   │  (symbolic)   │   │  (optional)  │    │
│  └──────────────┘   └──────────────┘   └──────┬───────┘    │
│                                                │            │
│  ┌──────────────┐   ┌──────────────┐           │            │
│  │ GUI Teleop   │──▶│   Memory     │◀──────────┤            │
│  │ (mode+cmd)   │   │   Module     │           │            │
│  └──────────────┘   └──────────────┘           ▼            │
│                                         ┌──────────────┐    │
│  ┌──────────────┐   ┌──────────────┐   │Tool Executor │    │
│  │ User         │◀──│ Trial Logger │   │  (MoveIt)    │    │
│  │ Interface    │   │ (start/stop)  │   └──────────────┘    │
│  └──────────────┘   └──────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

## Package structure

| Path | Description |
|------|-------------|
| `launch/` | `prime_full.launch` (robot + camera + YOLO + MoveIt + PRIME), `prime.launch` (PRIME only) |
| `scripts/` | Nodes: `prime_node.py`, `state_builder.py`, `tool_executor.py`, `gui_teleop.py`, `user_interface.py`, `llm_executive.py`, `go_home.py`, `trial_logger.py`, `start_trial.py`, `reset_trial.py` |
| `config/` | `prime_params.yaml`, `llm_prompts.yaml` |
| `srv/` | `StartTrial.srv` (mode, subject_id, difficulty, reason → success, message) |
| `msg/` | PRIME messages (ToolCall, ToolResult, SymbolicState, etc.) |

## Prerequisites

- ROS Noetic
- Kinova ROS packages (`kinova-ros`, `kinova_bringup`)
- RealSense ROS (`realsense2_camera`)
- YOLO ROS (`yolo-ros`)
- MoveIt (e.g. `j2n6s300_moveit_config` for Jaco2)
- Optional: Ollama with Qwen 2.5 (or compatible LLM) for `llm_executive`

## Installation

1. Clone or place the package in your catkin workspace:
   ```bash
   cd ~/catkin_ws/src
   # package lives at src/PRIME-ros (prime_ros)
   ```

2. Install Python dependencies (if any):
   ```bash
   pip install -r requirements.txt
   ```

3. Build:
   ```bash
   cd ~/catkin_ws
   catkin_make
   # or: catkin build prime_ros
   source devel/setup.bash
   ```

4. Optional — for LLM-driven decisions, start Ollama:
   ```bash
   ollama run qwen2.5
   ```

## Usage

### Full stack (robot + camera + YOLO + MoveIt + PRIME)

```bash
roslaunch prime_ros prime_full.launch robot_type:=j2n6s300
```

Launch arguments (examples):
- `robot_type:=j2n6s300`
- `yolo_model:=yolo26` or `yolov8`
- `color_width`, `color_height`, `color_fps` (RealSense)
- `yolo_conf_threshold` (default `0.25`)
- `yolo_iou_threshold` (default `0.45`)
- `yolo_agnostic_nms` (default `false`)
- `yolo_max_det` (default `100`)

Example duplicate-box suppression tuning:
```bash
roslaunch prime_ros prime_full.launch \
  yolo_conf_threshold:=0.4 \
  yolo_iou_threshold:=0.35 \
  yolo_agnostic_nms:=true
```

### PRIME only (robot and perception already running)

```bash
roslaunch prime_ros prime.launch robot_type:=j2n6s300
```

Launch arguments for `prime.launch`:
- `robot_type`, `auto_home`, `home_target`, `home_delay`
- `enable_gui_teleop`, `enable_trial_logger`
- `trial_log_root` (default: `$(env HOME)/Desktop/PRIME_LOGS`)
- `trial_image_topic`, `trial_video_fps`, `trial_success_label`
- `start_llm`, `llm_endpoint`

## Configuration

Main config: `config/prime_params.yaml`.

| Section | Purpose |
|--------|---------|
| `workspace` | 3×3 grid bounds (`x_min`/`x_max`, `y_min`/`y_max`), `object_z`, `axis_signs` |
| `safety_bounds` | Motion limits; `add_moveit_walls` for planning scene |
| `state_builder` | Tracking, YOLO filtering, mask yaw, publish rates |
| `tools/approach` | `pre_grasp_distance`, `approach_speed`; pixel_servo options |
| `tools/align` | `velocity_scaling`, `acceleration_scaling` (ALIGN_YAW speed), `wrist_limit`, `down_pitch_deg` |
| `tools/grasp` | Finger positions (open/close) |
| `teleop_gui` | `publish_rate`, `linear_speed`, `angular_speed` |
| `llm` | `endpoint`, `model`, `temperature`, `timeout` |

Tuning rotation speed: increase `tools/align/velocity_scaling` and `acceleration_scaling` (e.g. 1.0–1.2). Lower values (e.g. 0.4) reduce joint-limit warnings.

## Trial logging

The trial logger records GUI events, tool calls/results, queries/responses, and optional camera video **per trial**. It does **not** start a trial at launch — you start and stop explicitly.

### Log layout

- **Root:** `~/Desktop/PRIME_LOGS` (overridable via `trial_log_root` in `prime.launch`).
- **Per run:** When you start a trial with mode, subject, and difficulty, logs go under:
  `PRIME_LOGS/<mode>/<subject_id>/<difficulty>/trial_<timestamp>/`
- If fields are omitted in the start call, missing path levels are skipped.

Each trial folder contains:
- `events.jsonl` — all events
- `gui_events.jsonl`, `tool_calls.jsonl`, `tool_results.jsonl`, `queries.jsonl`, `responses.jsonl`, `llm_events.jsonl`
- `trial_meta.json`, `trial_summary.json`
- `camera.mp4` (if recording enabled), `video_meta.json`

### Start / stop trial

**Start** (logging and video begin here):

```bash
rosrun prime_ros start_trial.py <mode> <subject_id> <difficulty>
# e.g.
rosrun prime_ros start_trial.py manual s1 easy
rosrun prime_ros start_trial.py assistive s2 hard --reason participant_ready
```

Or call the service directly:

```bash
rosservice call /prime/trial_logger/start "{mode: 'manual', subject_id: 's1', difficulty: 'easy', reason: 'trial_start'}"
```

**Stop** (writes summary and closes files):

```bash
rosservice call /prime/trial_logger/stop
```

Do **not** rely on terminating `roslaunch` to stop cleanly; use the stop service so files are finalized.

### Reset trial (stop + home + start)

Stop current trial, home the arm, then start a new trial with optional mode/subject:

```bash
rosrun prime_ros reset_trial.py _mode:=manual _sub:=s1 _difficulty:=easy
```

Mode/subject/difficulty for the *new* trial come from private params
`_mode`, `_sub` (or `_subject_id`), and `_difficulty` (or `_task_difficulty`).

### Success label (optional)

To mark success for a specific object label in trial summaries, set at launch:

```bash
roslaunch prime_ros prime.launch trial_success_label:=cup
```

## ROS services

| Service | Type | Description |
|--------|------|-------------|
| `/prime/trial_logger/start` | `prime_ros/StartTrial` | Start a new trial (args: `mode`, `subject_id`, `difficulty`, `reason`) |
| `/prime/trial_logger/stop` | `std_srvs/Trigger` | End current trial and write summary |
| `/prime/trial_logger/reset` | `std_srvs/Trigger` | End current trial and start a new one (same mode/subject as last start) |

## ROS topics

### Published

- `/prime/symbolic_state` — symbolic state (grid, objects, gripper)
- `/prime/candidate_objects` — candidate targets
- `/prime/control_mode` — GUI control mode and active-command flag
- `/prime/gui_teleop_event` — GUI action logs (JSON)
- `/prime/query` — queries to user
- `/prime/tool_call` — tool calls (from LLM or executive)
- `/prime/tool_result` — tool execution results

### Subscribed

- `/prime/response` — user responses to queries
- `/<robot>_driver/out/tool_pose`, `/<robot>_driver/out/joint_state`
- `/yolo/...` (detections); camera topic set via `trial_image_topic` for logger

## Tools

| Tool | Description |
|------|-------------|
| `INTERACT` | Ask user a question/confirmation |
| `APPROACH(obj)` | Move gripper toward object (MoveIt) |
| `ALIGN_YAW(obj)` | Align gripper yaw to object (MoveIt; speed via `tools/align` config) |
| `GRASP` | Close gripper (Kinova) |
| `RELEASE` | Open gripper (Kinova) |

## User input

1. **GUI Teleop (`gui_teleop.py`)**  
   Modes: Translation, Rotation, Gripper. Axis buttons publish Cartesian velocity; STOP zeroes it; Open/Close send finger goals. Publishes `/prime/control_mode` and `/prime/gui_teleop_event`.

2. **Keyboard query responses (`user_interface.py`)**  
   `y`/`1` = Yes/Option 1, `n`/`2` = No/Option 2, `3`–`5` = Options 3–5, `q` = Cancel.

## Symbolic state

Workspace is a 3×3 grid; objects and gripper are tracked by grid cell for efficient reasoning.

## Memory

PRIME maintains: dialog history, candidate set, and tool history for multi-step behavior.

## Testing

```bash
cd ~/catkin_ws
source devel/setup.bash
catkin_make
catkin run_tests prime_ros
# or: catkin_test_results build/prime_ros
```

Unit tests: `test/test_teleop_command_model.py` (teleop command model, mode/axis behavior).

Manual smoke test: launch `prime.launch`, then inspect `/prime/control_mode`, `/prime/gui_teleop_event`, and `/<robot>_driver/in/cartesian_velocity` while using the GUI.

## Paper reference

> PRIME: An LLM-Based Executive for Interactive Manipulation Planning with Minimal User Effort  
> Ali Rabiee et al., IROS 2026

## License

MIT License
