#!/usr/bin/env python
"""
Main PRIME Executive Node

This is the main orchestration node that coordinates:
1. State monitoring (from state_builder)
2. LLM decision making (from llm_executive)
3. Tool execution (from tool_executor)
4. User interaction (from user_interface)

The PRIME loop:
1. Monitor symbolic state and control mode
2. When user is actively controlling (GUI motion command active):
   - Update candidate set based on proximity and motion
   - Call LLM for decision
3. Execute LLM's tool call
4. Handle user responses
5. Repeat

PRIME is designed for shared autonomy - it assists users with limited
input capabilities by intelligently deciding when to ask questions,
when to suggest actions, and when to execute autonomously.
"""

import rospy
import os
import sys
from threading import Lock
from enum import Enum

# Ensure local PRIME scripts directory is on PYTHONPATH (rosrun wrapper doesn't add it)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

try:
    from prime_ros.msg import (
        SymbolicState, ControlMode, CandidateSet,
        ToolCall, ToolResult, PRIMEQuery, PRIMEResponse
    )
    MSGS_AVAILABLE = True
except ImportError:
    rospy.logwarn("PRIME messages not built yet. Run catkin build.")
    MSGS_AVAILABLE = False

from prime_memory import get_memory


class PRIMEState(Enum):
    """States of the PRIME executive."""
    IDLE = 0              # Waiting for user activity
    MONITORING = 1        # User is moving, monitoring intent
    DECIDING = 2          # LLM is making a decision
    AWAITING_RESPONSE = 3 # Waiting for user response to query
    EXECUTING = 4         # Executing a tool
    ERROR = 5             # Error state


class PRIMENode:
    """
    Main PRIME executive node.
    
    Implements the closed-loop LLM reasoning system for shared autonomy.
    """
    
    def __init__(self):
        rospy.init_node('prime_executive', anonymous=False)
        
        # Parameters
        self.update_rate = rospy.get_param('~update_rate', 5.0)  # Hz
        self.activity_timeout = rospy.get_param('~activity_timeout', 2.0)  # seconds
        self.min_candidates_for_decision = rospy.get_param('~min_candidates', 1)
        # Disable LLM decisions by default (LLM runs in its own node)
        self.enable_decisions = rospy.get_param('~enable_decisions', False)
        
        # State
        self.lock = Lock()
        self.state = PRIMEState.IDLE
        self.current_symbolic_state: SymbolicState = None
        self.current_candidates: CandidateSet = None
        self.last_activity_time = rospy.Time.now()
        self.pending_query_id: str = None
        self.executing_call_id: str = None
        
        # Memory
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
            
            self.response_sub = rospy.Subscriber(
                '/prime/response',
                PRIMEResponse,
                self.response_callback
            )
            
            self.tool_result_sub = rospy.Subscriber(
                '/prime/tool_result',
                ToolResult,
                self.tool_result_callback
            )
        
        # Publishers
        if MSGS_AVAILABLE:
            self.tool_pub = rospy.Publisher(
                '/prime/tool_call',
                ToolCall,
                queue_size=10
            )
        
        # LLM Executive reference (import here to avoid circular imports)
        self.llm_executive = None
        
        # Main loop timer
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / self.update_rate),
            self.main_loop
        )
        
        rospy.loginfo("PRIME Executive initialized")
        rospy.loginfo(f"State: {self.state.name}")
    
    def state_callback(self, msg: SymbolicState):
        """Handle symbolic state updates."""
        with self.lock:
            self.current_symbolic_state = msg
            
            # Check for user activity (GUI motion command active)
            if msg.control_mode.joystick_active:
                self.last_activity_time = rospy.Time.now()
    
    def candidates_callback(self, msg: CandidateSet):
        """Handle candidate set updates."""
        with self.lock:
            self.current_candidates = msg
            
            # Update memory
            labels = dict(zip(msg.candidate_ids, msg.candidate_labels))
            self.memory.set_candidates(list(msg.candidate_ids), labels)
    
    def response_callback(self, msg: PRIMEResponse):
        """Handle user response to query."""
        with self.lock:
            if self.state != PRIMEState.AWAITING_RESPONSE:
                rospy.logwarn("Received response but not awaiting one")
                return
            
            if msg.query_id != self.pending_query_id:
                rospy.logwarn(f"Response query_id mismatch: {msg.query_id} vs {self.pending_query_id}")
                return
            
            rospy.loginfo(f"User response received: {msg.selected_labels}")
            
            # Process response
            self.process_user_response(msg)
            
            # Return to monitoring state
            self.state = PRIMEState.MONITORING
            self.pending_query_id = None
    
    def tool_result_callback(self, msg: ToolResult):
        """Handle tool execution result."""
        with self.lock:
            if self.state != PRIMEState.EXECUTING:
                return
            
            if msg.call_id != self.executing_call_id:
                rospy.logwarn(f"Result call_id mismatch")
                return
            
            rospy.loginfo(f"Tool result: {msg.tool_name} - {'SUCCESS' if msg.success else 'FAILED'}")
            
            # Return to monitoring state
            self.state = PRIMEState.MONITORING
            self.executing_call_id = None
    
    def process_user_response(self, response: PRIMEResponse):
        """Process user's response and update state accordingly."""
        if response.timed_out:
            rospy.logwarn("Query timed out, no action taken")
            return
        
        # Add to dialog history
        self.memory.add_dialog_turn(
            query_type='question',
            content='',  # Would need to track original query
            options=[],
            user_response=', '.join(response.selected_labels),
            selected_indices=list(response.selected_indices),
            response_time=response.response_time
        )
        
        # Handle based on response content
        labels = response.selected_labels
        indices = list(response.selected_indices)
        
        # Check for confirmation (Yes/No)
        if labels and labels[0].lower() in ['yes', 'y']:
            # User confirmed - set confirmed object
            single_candidate = self.memory.get_single_candidate()
            if single_candidate:
                self.memory.set_confirmed(single_candidate)
                rospy.loginfo(f"User confirmed target: {single_candidate}")
        
        elif labels and labels[0].lower() in ['no', 'n']:
            # User rejected - clear confirmation
            self.memory.confirmed_object = None
            rospy.loginfo("User rejected, confirmation cleared")
        
        # Check for disambiguation (object selection)
        elif len(indices) == 1 and len(self.memory.candidate_ids) > 1:
            # User selected from multiple candidates
            candidates_list = list(self.memory.candidate_ids)
            if indices[0] < len(candidates_list):
                selected_id = candidates_list[indices[0]]
                self.memory.prune_candidates([selected_id])
                rospy.loginfo(f"User selected: {selected_id}")
    
    def should_make_decision(self) -> bool:
        """Determine if PRIME should make a decision now."""
        with self.lock:
            # Need state and candidates
            if not self.current_symbolic_state or not self.current_candidates:
                return False
            
            # Check for recent user activity
            time_since_activity = (rospy.Time.now() - self.last_activity_time).to_sec()
            
            # If user was recently active and has stopped, might be a good time to assist
            if time_since_activity > 0.5 and time_since_activity < self.activity_timeout:
                # User paused - good time for decision
                return True
            
            # If we have candidates and user is in gripper mode near an object
            if (self.current_candidates.candidate_ids and 
                self.current_symbolic_state.control_mode.fingers_active):
                # User might be trying to grasp
                return True
            
            return False
    
    def make_decision(self):
        """Call LLM to make a decision."""
        # The LLM executive is intended to run as a separate ROS node (`llm_executive.py`).
        # Creating it here would call rospy.init_node() again and crash.
        rospy.logwarn("Decision-making is disabled in prime_node (enable with ~enable_decisions:=true after refactor).")
        return
    
    def handle_tool_call(self, call: ToolCall):
        """Handle a tool call from the LLM."""
        with self.lock:
            rospy.loginfo(f"LLM decision: {call.tool_name}")
            rospy.loginfo(f"Reasoning: {call.reasoning}")
            
            if call.tool_name == 'INTERACT':
                # Transition to awaiting response
                self.state = PRIMEState.AWAITING_RESPONSE
                self.pending_query_id = call.call_id
                rospy.loginfo(f"Awaiting user response to: {call.interact_content}")
            
            else:
                # Execute action tool
                self.state = PRIMEState.EXECUTING
                self.executing_call_id = call.call_id
                
                # Publish tool call (tool_executor will handle it)
                if self.tool_pub:
                    self.tool_pub.publish(call)
    
    def main_loop(self, event):
        """Main PRIME control loop."""
        with self.lock:
            current_state = self.state
        
        if current_state == PRIMEState.IDLE:
            # Check for user activity to start monitoring
            time_since_activity = (rospy.Time.now() - self.last_activity_time).to_sec()
            if time_since_activity < 1.0:
                with self.lock:
                    self.state = PRIMEState.MONITORING
                rospy.loginfo("User activity detected, starting monitoring")
        
        elif current_state == PRIMEState.MONITORING:
            # Check if we should make a decision
            if self.enable_decisions and self.should_make_decision():
                with self.lock:
                    self.state = PRIMEState.DECIDING
                rospy.loginfo("Making decision...")
                self.make_decision()
        
        elif current_state == PRIMEState.DECIDING:
            # LLM is deciding, wait for result
            pass
        
        elif current_state == PRIMEState.AWAITING_RESPONSE:
            # Waiting for user, check for timeout
            pass
        
        elif current_state == PRIMEState.EXECUTING:
            # Tool is executing, wait for result
            pass
        
        elif current_state == PRIMEState.ERROR:
            # Error state - try to recover
            rospy.logwarn("In error state, attempting recovery")
            with self.lock:
                self.state = PRIMEState.IDLE
                self.memory.reset_task()
    
    def run(self):
        """Run the node."""
        rospy.loginfo("PRIME Executive running")
        rospy.spin()


def main():
    try:
        node = PRIMENode()
        node.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
