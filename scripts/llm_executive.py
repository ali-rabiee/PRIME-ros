#!/usr/bin/env python
"""
LLM Executive Node for PRIME

This node interfaces with the Qwen 2.5 model (via Ollama or similar)
to make decisions about tool calls based on symbolic state and memory.

The LLM receives:
- Current symbolic state (objects, gripper, control mode)
- Memory context (dialog history, candidates, recent actions)

And outputs exactly ONE tool call per invocation:
- INTERACT: Ask user a question/confirmation
- APPROACH: Move toward an object
- ALIGN_YAW: Align gripper orientation
- GRASP: Close gripper
- RELEASE: Open gripper
"""

import rospy
import os
import sys
import json
import requests
from typing import Optional, Dict, Any, Tuple
from threading import Lock
from std_msgs.msg import String

# Ensure local PRIME scripts directory is on PYTHONPATH (rosrun wrapper doesn't add it)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

try:
    from prime_ros.msg import (
        SymbolicState, ToolCall, ToolResult,
        PRIMEQuery, CandidateSet
    )
    MSGS_AVAILABLE = True
except ImportError:
    rospy.logwarn("PRIME messages not built yet. Run catkin build.")
    MSGS_AVAILABLE = False

# Import memory module
from prime_memory import PRIMEMemory, get_memory


class LLMExecutive:
    """
    LLM-based executive for PRIME decision making.
    
    Uses Qwen 2.5 (or compatible model) to reason over symbolic state
    and produce tool calls.
    """
    
    def __init__(self):
        rospy.init_node('llm_executive', anonymous=False)
        
        # Parameters
        self.model = rospy.get_param('llm/model', 'qwen2.5')
        self.endpoint = rospy.get_param('llm/endpoint', 'http://localhost:11434/api/generate')
        self.temperature = rospy.get_param('llm/temperature', 0.3)
        self.max_tokens = rospy.get_param('llm/max_tokens', 500)
        self.timeout = rospy.get_param('llm/timeout', 30.0)
        
        # Load prompts
        self.system_prompt = rospy.get_param('llm/system_prompt', self._default_system_prompt())
        self.interact_template = rospy.get_param('llm/interact_template', '')
        
        # Thread safety
        self.lock = Lock()
        
        # State
        self.current_state: Optional[SymbolicState] = None
        self.current_candidates: Optional[CandidateSet] = None
        self.memory = get_memory()
        
        # Subscribers
        if MSGS_AVAILABLE:
            self.state_sub = rospy.Subscriber(
                '/prime/symbolic_state',
                SymbolicState,
                self.state_callback
            )
            
            self.candidates_sub = rospy.Subscriber(
                '/prime/candidate_objects',
                CandidateSet,
                self.candidates_callback
            )
        
        # Publishers
        if MSGS_AVAILABLE:
            self.tool_pub = rospy.Publisher(
                '/prime/tool_call',
                ToolCall,
                queue_size=10
            )
            
            self.query_pub = rospy.Publisher(
                '/prime/query',
                PRIMEQuery,
                queue_size=10
            )
        self.llm_event_pub = rospy.Publisher('/prime/llm_event', String, queue_size=50)
        self.session_id = f"llm_{rospy.Time.now().to_nsec()}"
        self._publish_llm_event("session_start", {"session_id": self.session_id, "model": self.model})
        
        rospy.loginfo(f"LLM Executive initialized with model: {self.model}")
    
    def _default_system_prompt(self) -> str:
        """Default system prompt if not loaded from config."""
        return """You are PRIME, an intelligent robotic manipulation assistant.

You receive symbolic state about objects and the gripper, and must output ONE tool call.

Available tools:
1. INTERACT(type, content, options) - Ask user a question/confirmation
2. APPROACH(object_id) - Move toward object (requires confirmation)
3. ALIGN_YAW(object_id) - Align gripper with object
4. GRASP() - Close gripper
5. RELEASE() - Open gripper

Rules:
- Output exactly ONE tool per response
- Before APPROACH/ALIGN_YAW/GRASP, get user confirmation via INTERACT
- When multiple candidates exist, use INTERACT to disambiguate
- Analyze motion trends to infer intent

Output format (JSON only):
{
  "reasoning": "brief explanation",
  "tool": "TOOL_NAME",
  "params": { ... }
}"""
    
    def state_callback(self, msg: SymbolicState):
        """Handle symbolic state updates."""
        with self.lock:
            self.current_state = msg
    
    def candidates_callback(self, msg: CandidateSet):
        """Handle candidate set updates."""
        with self.lock:
            self.current_candidates = msg
            
            # Update memory with candidates
            labels = dict(zip(msg.candidate_ids, msg.candidate_labels))
            self.memory.set_candidates(list(msg.candidate_ids), labels)
    
    def state_to_text(self, state: SymbolicState) -> str:
        """Convert symbolic state to text for LLM."""
        lines = []
        
        # Objects
        if state.objects:
            obj_strs = []
            for obj in state.objects:
                held = " [HELD]" if obj.is_held else ""
                grid_lbl = getattr(obj, "grid_label", "")
                if grid_lbl:
                    obj_strs.append(f"{obj.object_id} ({obj.label}) at {grid_lbl}{held}")
                else:
                    obj_strs.append(f"{obj.object_id} ({obj.label}) at grid cell {obj.grid_cell}{held}")
            lines.append(f"Objects: {'; '.join(obj_strs)}")
        else:
            lines.append("Objects: none detected")
        
        # Gripper
        gr_lbl = getattr(state, "gripper_grid_label", "")
        if gr_lbl:
            lines.append(f"Gripper: {gr_lbl}, height {state.gripper_height:.2f}m, yaw {state.gripper_yaw:.2f}rad")
        else:
            lines.append(f"Gripper: cell {state.gripper_grid_cell}, height {state.gripper_height:.2f}m, yaw {state.gripper_yaw:.2f}rad")
        
        # Motion trend
        if len(state.gripper_history) >= 2:
            hist = state.gripper_history
            dx = hist[-1].x - hist[0].x
            dy = hist[-1].y - hist[0].y
            import math
            if dx != 0 or dy != 0:
                direction = math.degrees(math.atan2(dy, dx))
                lines.append(f"Motion trend: moving toward {direction:.0f}° direction")
            else:
                lines.append("Motion trend: stationary")
        else:
            lines.append("Motion trend: insufficient history")
        
        # Control mode
        mode_names = {0: 'translation', 1: 'rotation', 2: 'gripper', 255: 'unknown'}
        mode_name = mode_names.get(state.control_mode.mode, 'unknown')
        lines.append(f"Control mode: {mode_name}")
        
        return "\n".join(lines)
    
    def build_prompt(self, state: SymbolicState) -> str:
        """Build the full prompt for the LLM."""
        # State description
        state_text = self.state_to_text(state)
        
        # Memory context
        memory_ctx = self.memory.get_memory_context()
        
        # Build prompt
        prompt = f"""{self.system_prompt}

Current State:
{state_text}

Candidates: {memory_ctx['candidates']}
Confirmed target: {memory_ctx['confirmed_object'] or 'None'}

Recent interactions:
{memory_ctx['recent_interactions']}

Recent actions:
{memory_ctx['recent_actions']}

Based on this information, decide your next action. Output ONLY valid JSON."""
        
        return prompt

    @staticmethod
    def _truncate_text(text: str, limit: int = 2000) -> str:
        if text is None:
            return ""
        if len(text) <= limit:
            return text
        return text[:limit] + "…"

    def _publish_llm_event(self, event_type: str, payload: Dict[str, Any]):
        try:
            data = {
                "event": str(event_type),
                "stamp": rospy.Time.now().to_sec(),
                "session_id": self.session_id,
                "payload": payload,
            }
            self.llm_event_pub.publish(String(data=json.dumps(data, sort_keys=True)))
        except Exception:
            pass
    
    def call_llm(self, prompt: str) -> Tuple[Optional[Dict], str]:
        """
        Call the LLM API and parse the response.
        
        Returns: (parsed_response, raw_text)
        """
        try:
            # Ollama API format
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
            }
            
            response = requests.post(
                self.endpoint,
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                rospy.logerr(f"LLM API error: {response.status_code}")
                self._publish_llm_event("api_error", {"status_code": int(response.status_code)})
                return None, f"API error: {response.status_code}"
            
            result = response.json()
            raw_text = result.get('response', '')
            self._publish_llm_event("response_received", {"raw_text": self._truncate_text(raw_text)})
            
            # Parse JSON from response
            # Try to extract JSON from the response
            try:
                # Find JSON in response
                start = raw_text.find('{')
                end = raw_text.rfind('}') + 1
                if start >= 0 and end > start:
                    json_str = raw_text[start:end]
                    parsed = json.loads(json_str)
                    return parsed, raw_text
                else:
                    rospy.logwarn(f"No JSON found in response: {raw_text}")
                    self._publish_llm_event("no_json", {"raw_text": self._truncate_text(raw_text)})
                    return None, raw_text
            except json.JSONDecodeError as e:
                rospy.logwarn(f"JSON parse error: {e}")
                self._publish_llm_event("parse_error", {"error": str(e), "raw_text": self._truncate_text(raw_text)})
                return None, raw_text
                
        except requests.exceptions.Timeout:
            rospy.logerr("LLM API timeout")
            self._publish_llm_event("api_timeout", {"timeout_sec": float(self.timeout)})
            return None, "Timeout"
        except requests.exceptions.RequestException as e:
            rospy.logerr(f"LLM API request error: {e}")
            self._publish_llm_event("api_request_error", {"error": str(e)})
            return None, str(e)
    
    def parse_tool_call(self, response: Dict) -> Optional[ToolCall]:
        """Parse LLM response into a ToolCall message."""
        if not MSGS_AVAILABLE:
            return None
        
        tool_name = response.get('tool', '').upper()
        params = response.get('params', {})
        reasoning = response.get('reasoning', '')
        if not tool_name:
            self._publish_llm_event("tool_missing", {"response": response})
        
        call = ToolCall()
        call.header.stamp = rospy.Time.now()
        call.tool_name = tool_name
        call.reasoning = reasoning
        call.call_id = f"call_{rospy.Time.now().to_nsec()}"
        
        if tool_name == 'INTERACT':
            call.interact_type = params.get('type', 0)  # 0=question, 1=suggestion, 2=confirmation
            call.interact_content = params.get('content', '')
            call.interact_options = params.get('options', [])
        elif tool_name in ['APPROACH', 'ALIGN_YAW']:
            call.target_object_id = params.get('object_id', '')
        elif tool_name in ['GRASP', 'RELEASE']:
            pass  # No additional params needed
        else:
            rospy.logwarn(f"Unknown tool: {tool_name}")
            self._publish_llm_event("unknown_tool", {"tool_name": tool_name, "response": response})
            return None
        
        return call
    
    def validate_tool_call(self, call: ToolCall) -> Tuple[bool, str]:
        """
        Validate a tool call against PRIME rules.
        
        Returns: (is_valid, error_message)
        """
        tool = call.tool_name
        
        # INTERACT is always allowed
        if tool == 'INTERACT':
            return True, ""
        
        # Action tools require confirmation
        if tool in ['APPROACH', 'ALIGN_YAW', 'GRASP']:
            if not self.memory.confirmed_object:
                return False, "No confirmed target. Must use INTERACT to confirm first."
            
            if tool in ['APPROACH', 'ALIGN_YAW']:
                if call.target_object_id != self.memory.confirmed_object:
                    return False, f"Target {call.target_object_id} doesn't match confirmed {self.memory.confirmed_object}"
        
        # Check candidate set for APPROACH/ALIGN_YAW
        if tool in ['APPROACH', 'ALIGN_YAW']:
            if len(self.memory.candidate_ids) > 1:
                return False, "Multiple candidates remain. Must disambiguate first."
            if call.target_object_id not in self.memory.candidate_ids:
                return False, f"Target {call.target_object_id} not in candidate set"
        
        return True, ""
    
    def decide(self) -> Optional[ToolCall]:
        """
        Make a decision based on current state.
        
        This is the main entry point called by the PRIME node.
        """
        with self.lock:
            if not self.current_state:
                rospy.logwarn("No state available for decision")
                return None
            
            # Build prompt
            prompt = self.build_prompt(self.current_state)
            try:
                memory_ctx = self.memory.get_memory_context()
            except Exception:
                memory_ctx = {"candidates": [], "confirmed_object": None}
            self._publish_llm_event(
                "call_start",
                {
                    "state_summary": self._truncate_text(self.state_to_text(self.current_state), limit=500),
                    "candidates": memory_ctx.get("candidates", []),
                    "confirmed_object": memory_ctx.get("confirmed_object"),
                },
            )
            
            # Call LLM
            rospy.loginfo("Calling LLM for decision...")
            response, raw_text = self.call_llm(prompt)
            
            if not response:
                rospy.logerr(f"LLM call failed: {raw_text}")
                return None
            
            rospy.loginfo(f"LLM response: {response}")
            
            # Parse tool call
            call = self.parse_tool_call(response)
            if not call:
                rospy.logerr("Failed to parse tool call")
                self._publish_llm_event("parse_tool_call_failed", {"response": response})
                return None
            
            # Validate
            is_valid, error = self.validate_tool_call(call)
            if not is_valid:
                rospy.logwarn(f"Invalid tool call: {error}")
                self._publish_llm_event(
                    "invalid_tool_call",
                    {
                        "error": error,
                        "tool_name": call.tool_name,
                        "target_object_id": call.target_object_id,
                        "confirmed_object": self.memory.confirmed_object,
                        "candidates": list(self.memory.candidate_ids),
                    },
                )
                # LLM made invalid call - could retry or return error
                # For now, generate a fallback INTERACT
                call = self._make_fallback_interact(error)
            
            # Publish
            if self.tool_pub:
                self.tool_pub.publish(call)
            self._publish_llm_event(
                "tool_call_published",
                {
                    "call_id": call.call_id,
                    "tool_name": call.tool_name,
                    "target_object_id": call.target_object_id,
                    "interact_type": int(call.interact_type) if call.tool_name == "INTERACT" else None,
                },
            )
            
            # If INTERACT, also publish as query
            if call.tool_name == 'INTERACT' and self.query_pub:
                query = PRIMEQuery()
                query.header = call.header
                query.query_type = call.interact_type
                query.content = call.interact_content
                query.options = call.interact_options
                query.max_selections = 1
                query.timeout = rospy.get_param('ui/query_timeout', 30.0)
                query.query_id = call.call_id
                self.query_pub.publish(query)
            
            return call
    
    def _make_fallback_interact(self, error: str) -> ToolCall:
        """Create a fallback INTERACT tool call when LLM makes invalid call."""
        call = ToolCall()
        call.header.stamp = rospy.Time.now()
        call.tool_name = 'INTERACT'
        call.interact_type = 0  # question
        call.interact_content = "I need more information. What would you like me to do?"
        call.interact_options = ["Pick up object", "Release object", "Cancel"]
        call.reasoning = f"Fallback due to invalid call: {error}"
        call.call_id = f"call_{rospy.Time.now().to_nsec()}"
        self._publish_llm_event(
            "fallback_interact",
            {
                "error": error,
                "call_id": call.call_id,
            },
        )
        return call
    
    def handle_user_response(self, query_id: str, selected_indices: list, response_text: str):
        """
        Handle user response to an INTERACT query.
        
        Updates memory and potentially confirms targets.
        """
        # Add to dialog history
        # Find the original query content (would need to track this)
        self.memory.add_dialog_turn(
            query_type='question',
            content='',  # Would need to track original query
            options=[],
            user_response=response_text,
            selected_indices=selected_indices,
            response_time=0.0
        )
        
        # Check if this was a confirmation
        if 'yes' in response_text.lower() or selected_indices == [0]:
            # User confirmed - if we have single candidate, set as confirmed
            single_candidate = self.memory.get_single_candidate()
            if single_candidate:
                self.memory.set_confirmed(single_candidate)
                rospy.loginfo(f"Target confirmed: {single_candidate}")
        
        # Check if this was disambiguation
        if len(selected_indices) == 1 and len(self.memory.candidate_ids) > 1:
            # User selected one option - prune candidates
            # This assumes options were ordered same as candidates
            candidates_list = list(self.memory.candidate_ids)
            if selected_indices[0] < len(candidates_list):
                selected_id = candidates_list[selected_indices[0]]
                self.memory.prune_candidates([selected_id])
                rospy.loginfo(f"Candidates pruned to: {selected_id}")
    
    def run(self):
        """Run the node."""
        rospy.spin()


def main():
    try:
        executive = LLMExecutive()
        executive.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
