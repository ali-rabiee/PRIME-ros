#!/usr/bin/env python
"""
Memory Module for PRIME

Maintains structured memory for multi-step interactive reasoning:
1. Dialog history - past interactions and user responses
2. Candidate set - plausible target objects
3. Recent tool calls and outcomes
4. Task context

This module is imported by other PRIME nodes rather than running standalone.
"""

import rospy
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from datetime import datetime


@dataclass
class DialogTurn:
    """Represents a single dialog turn in the interaction history."""
    timestamp: datetime
    query_type: str  # 'question', 'suggestion', 'confirmation'
    content: str
    options: List[str]
    user_response: Optional[str]
    selected_indices: List[int]
    response_time: float  # seconds
    
    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'query_type': self.query_type,
            'content': self.content,
            'options': self.options,
            'user_response': self.user_response,
            'selected_indices': self.selected_indices,
            'response_time': self.response_time
        }
    
    def to_text(self) -> str:
        """Convert to human-readable text for LLM context."""
        if self.options:
            options_str = ", ".join([f"{i+1}) {opt}" for i, opt in enumerate(self.options)])
            return f"System: {self.content} [{options_str}] -> User selected: {self.user_response}"
        else:
            return f"System: {self.content} -> User: {self.user_response}"


@dataclass
class ToolRecord:
    """Records a tool call and its outcome."""
    timestamp: datetime
    tool_name: str
    target_object: Optional[str]
    params: dict
    success: bool
    error_category: Optional[str]
    message: str
    
    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'tool_name': self.tool_name,
            'target_object': self.target_object,
            'params': self.params,
            'success': self.success,
            'error_category': self.error_category,
            'message': self.message
        }
    
    def to_text(self) -> str:
        """Convert to human-readable text for LLM context."""
        status = "SUCCESS" if self.success else f"FAILED ({self.error_category})"
        if self.target_object:
            return f"{self.tool_name}({self.target_object}): {status}"
        else:
            return f"{self.tool_name}: {status}"


class PRIMEMemory:
    """
    Manages memory for the PRIME system.
    
    As per the paper, memory consists of:
    - Dialog history: ordered history of past interactions
    - Candidate set: current plausible target objects
    - Last interaction: most recent prompt/query
    - Tool history: recent actions and outcomes
    """
    
    def __init__(self, max_dialog_history: int = 20, max_tool_history: int = 10):
        """
        Initialize PRIME memory.
        
        Args:
            max_dialog_history: Maximum number of dialog turns to keep
            max_tool_history: Maximum number of tool records to keep
        """
        self.max_dialog_history = max_dialog_history
        self.max_tool_history = max_tool_history
        
        # Dialog history
        self.dialog_history: deque = deque(maxlen=max_dialog_history)
        
        # Current candidate set
        self.candidate_ids: Set[str] = set()
        self.candidate_labels: Dict[str, str] = {}  # id -> label
        
        # Last interaction (for avoiding redundant queries)
        self.last_interaction: Optional[DialogTurn] = None
        
        # Tool call history
        self.tool_history: deque = deque(maxlen=max_tool_history)
        
        # Current task context
        self.current_task: Optional[str] = None
        self.task_start_time: Optional[datetime] = None
        
        # Confirmation state
        self.pending_confirmation: bool = False
        self.confirmed_object: Optional[str] = None
    
    def add_dialog_turn(self, 
                        query_type: str,
                        content: str,
                        options: List[str],
                        user_response: Optional[str],
                        selected_indices: List[int],
                        response_time: float):
        """Add a new dialog turn to history."""
        turn = DialogTurn(
            timestamp=datetime.now(),
            query_type=query_type,
            content=content,
            options=options,
            user_response=user_response,
            selected_indices=selected_indices,
            response_time=response_time
        )
        self.dialog_history.append(turn)
        self.last_interaction = turn
        
        return turn
    
    def add_tool_record(self,
                        tool_name: str,
                        target_object: Optional[str],
                        params: dict,
                        success: bool,
                        error_category: Optional[str] = None,
                        message: str = ""):
        """Record a tool execution."""
        record = ToolRecord(
            timestamp=datetime.now(),
            tool_name=tool_name,
            target_object=target_object,
            params=params,
            success=success,
            error_category=error_category,
            message=message
        )
        self.tool_history.append(record)
        
        # Reset confirmation after action
        if tool_name in ['APPROACH', 'ALIGN_YAW', 'GRASP', 'RELEASE']:
            self.pending_confirmation = False
            if success:
                pass  # Keep confirmed_object for reference
            else:
                self.confirmed_object = None  # Reset on failure
        
        return record
    
    def set_candidates(self, 
                       candidate_ids: List[str], 
                       candidate_labels: Dict[str, str]):
        """Update the current candidate set."""
        self.candidate_ids = set(candidate_ids)
        self.candidate_labels = candidate_labels
    
    def prune_candidates(self, keep_ids: List[str]):
        """
        Prune candidate set to only keep specified IDs.
        
        Used after user disambiguates between options.
        """
        self.candidate_ids = self.candidate_ids.intersection(set(keep_ids))
        self.candidate_labels = {
            k: v for k, v in self.candidate_labels.items() 
            if k in self.candidate_ids
        }
    
    def remove_candidate(self, object_id: str):
        """Remove a single candidate from the set."""
        self.candidate_ids.discard(object_id)
        self.candidate_labels.pop(object_id, None)
    
    def get_single_candidate(self) -> Optional[str]:
        """
        Get the target object ID if only one candidate remains.
        
        Returns None if multiple candidates or no candidates.
        """
        if len(self.candidate_ids) == 1:
            return list(self.candidate_ids)[0]
        return None
    
    def set_confirmed(self, object_id: str):
        """Mark an object as confirmed by user."""
        self.pending_confirmation = False
        self.confirmed_object = object_id
    
    def is_action_allowed(self, tool_name: str, target_object: Optional[str]) -> bool:
        """
        Check if an action tool is allowed based on PRIME rules.
        
        Rules:
        - Action tools only valid when candidate set has single element
        - Confirmation required before action execution
        """
        if tool_name == 'INTERACT':
            return True
        
        if tool_name in ['APPROACH', 'ALIGN_YAW', 'GRASP']:
            # Must have single confirmed target
            if not self.confirmed_object:
                return False
            if target_object and target_object != self.confirmed_object:
                return False
            return True
        
        if tool_name == 'RELEASE':
            return True  # Release doesn't need target confirmation
        
        return False
    
    def get_recent_dialog_text(self, n: int = 5) -> str:
        """Get recent dialog history as text for LLM."""
        recent = list(self.dialog_history)[-n:]
        if not recent:
            return "No recent interactions."
        return "\n".join([turn.to_text() for turn in recent])
    
    def get_recent_tools_text(self, n: int = 3) -> str:
        """Get recent tool history as text for LLM."""
        recent = list(self.tool_history)[-n:]
        if not recent:
            return "No recent actions."
        return "\n".join([record.to_text() for record in recent])
    
    def get_candidates_text(self) -> str:
        """Get candidate set as text for LLM."""
        if not self.candidate_ids:
            return "No candidate objects."
        
        candidates = []
        for cid in self.candidate_ids:
            label = self.candidate_labels.get(cid, "unknown")
            candidates.append(f"{cid} ({label})")
        
        return ", ".join(candidates)
    
    def get_memory_context(self) -> dict:
        """
        Get complete memory context for LLM prompting.
        
        Returns a dictionary with all relevant memory information.
        """
        return {
            'recent_interactions': self.get_recent_dialog_text(),
            'recent_actions': self.get_recent_tools_text(),
            'candidates': self.get_candidates_text(),
            'candidate_count': len(self.candidate_ids),
            'confirmed_object': self.confirmed_object,
            'pending_confirmation': self.pending_confirmation,
            'last_interaction': self.last_interaction.to_text() if self.last_interaction else None
        }
    
    def reset_task(self):
        """Reset memory for a new task."""
        self.candidate_ids = set()
        self.candidate_labels = {}
        self.last_interaction = None
        self.pending_confirmation = False
        self.confirmed_object = None
        self.current_task = None
        self.task_start_time = None
    
    def start_task(self, task_description: str):
        """Start a new task."""
        self.reset_task()
        self.current_task = task_description
        self.task_start_time = datetime.now()


# Singleton instance for use across nodes
_memory_instance: Optional[PRIMEMemory] = None


def get_memory() -> PRIMEMemory:
    """Get the singleton memory instance."""
    global _memory_instance
    if _memory_instance is None:
        max_dialog = rospy.get_param('memory/max_dialog_history', 20)
        max_tools = rospy.get_param('memory/max_tool_history', 10)
        _memory_instance = PRIMEMemory(max_dialog, max_tools)
    return _memory_instance
