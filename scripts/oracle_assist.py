#!/usr/bin/env python3
"""
Oracle logic adapted from grasp-copilot for PRIME GUI assistance.

This module mirrors the original oracle state machine and decision rules,
but keeps dependencies self-contained (no external grid/yaw modules).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from math import pi
from typing import Dict, List, Optional, Sequence, Tuple

UserState = Dict[str, str]  # expects {"mode": "translation"|"rotation"|"gripper"}

# Hard constraint used for both dataset generation and inference-time validation.
MAX_INTERACT_CHOICES = 5

YAW_BINS: Tuple[str, ...] = ("E", "NE", "N", "NW", "W", "SW", "S", "SE")


def yaw_to_bin(yaw_rad: Optional[float]) -> str:
    """Quantize yaw (radians) to one of 8 bins."""
    if yaw_rad is None:
        return "N"
    try:
        yaw = float(yaw_rad)
    except Exception:
        return "N"
    # Normalize to [0, 2pi)
    yaw = (yaw + 2 * pi) % (2 * pi)
    step = 2 * pi / len(YAW_BINS)
    idx = int(round(yaw / step)) % len(YAW_BINS)
    return YAW_BINS[idx]


def _cell_from_label(label: str) -> Tuple[int, int]:
    if len(label) != 2:
        raise ValueError(f"Invalid cell label: {label!r}")
    row, col = label[0].upper(), label[1]
    if row not in {"A", "B", "C"} or col not in {"1", "2", "3"}:
        raise ValueError(f"Invalid cell label: {label!r}")
    return ord(row) - ord("A"), int(col) - 1


def manhattan(a: str, b: str) -> int:
    ra, ca = _cell_from_label(a)
    rb, cb = _cell_from_label(b)
    return abs(ra - rb) + abs(ca - cb)


@dataclass
class OracleState:
    intended_obj_id: str
    selected_obj_id: Optional[str] = None
    pending_action_obj_id: Optional[str] = None
    pending_mode: Optional[str] = None  # "APPROACH" | "ALIGN_YAW"
    awaiting_confirmation: bool = False
    awaiting_help: bool = False
    awaiting_choice: bool = False
    awaiting_intent_gate: bool = False
    awaiting_anything_else: bool = False
    awaiting_mode_select: bool = False
    last_prompt_context: Optional[Dict] = None
    last_declined_obj_id: Optional[str] = None
    last_tool_calls: List[str] = field(default_factory=list)
    terminate_episode: bool = False


def _tool(tool: str, args: Dict) -> Dict:
    return {"tool": tool, "args": args}


def _interact(kind: str, text: str, choices: List[str], context: Dict, state: OracleState) -> Dict:
    state.last_prompt_context = context
    return _tool("INTERACT", {"kind": kind, "text": text, "choices": choices})


def _rank_candidates(objects: Sequence[Dict], candidates: Sequence[str], gripper_cell: str) -> List[Dict]:
    available = [o for o in objects if o["id"] in set(candidates) and not o.get("is_held", False)]
    scored = [(o, manhattan(gripper_cell, o["cell"])) for o in available]
    scored.sort(key=lambda x: (x[1], x[0]["id"]))
    return [o for o, _ in scored]


def _top_two_candidates(objects: Sequence[Dict], candidates: Sequence[str], gripper_cell: str) -> Optional[List[Dict]]:
    ranked = _rank_candidates(objects, candidates, gripper_cell)
    if len(ranked) < 2:
        return None
    return [ranked[0], ranked[1]]


def _has_yaw_oscillation(gripper_hist: Sequence[Dict]) -> Tuple[bool, Optional[str], Optional[str], Optional[str]]:
    if len(gripper_hist) < 6:
        return False, None, None, None
    cells = [g["cell"] for g in gripper_hist]
    yaws = [g["yaw"] for g in gripper_hist]
    cell_counts = Counter(cells)
    dominant_cell, count = cell_counts.most_common(1)[0]
    if count < 4:
        return False, None, None, None

    unique_order: List[str] = []
    for y in yaws:
        if not unique_order or unique_order[-1] != y:
            unique_order.append(y)
    if len(set(unique_order)) != 2:
        return False, None, None, None
    switches = sum(1 for i in range(1, len(yaws)) if yaws[i] != yaws[i - 1])
    if switches < 3:
        return False, None, None, None
    yaw1, yaw2 = unique_order[0], unique_order[1]
    return True, dominant_cell, yaw1, yaw2


def _has_cell_oscillation(gripper_hist: Sequence[Dict], cell_a: str, cell_b: str) -> bool:
    if len(gripper_hist) < 6:
        return False
    cells = [g["cell"] for g in gripper_hist]
    allowed = {cell_a, cell_b}
    if cells.count(cell_a) < 2 or cells.count(cell_b) < 2:
        return False
    transitions = 0
    for i in range(1, len(cells)):
        if cells[i] != cells[i - 1] and {cells[i], cells[i - 1]} <= allowed:
            transitions += 1
    return transitions >= 2


def _effective_mode(user_state: Optional[UserState], gripper_hist: Sequence[Dict], memory: Dict) -> str:
    mode = str((user_state or {}).get("mode") or "translation")
    if mode not in {"translation", "rotation", "gripper"}:
        mode = "translation"
    if mode != "gripper":
        return mode
    cur = gripper_hist[-1]
    cell = str(cur.get("cell", "A1"))
    yaw = str(cur.get("yaw", "N"))
    n = int(memory.get("n_interactions", 0))
    score = sum(ord(ch) for ch in (cell + yaw)) + n
    return "translation" if (score % 2 == 0) else "rotation"


def _emit_intent_gate(
    objects: Sequence[Dict],
    gripper_hist: Sequence[Dict],
    memory: Dict,
    state: OracleState,
    *,
    user_state: Optional[UserState],
) -> Optional[Dict]:
    current_cell = gripper_hist[-1]["cell"]
    candidates = list(memory.get("candidates", []))
    excluded_obj_ids = set(memory.get("excluded_obj_ids") or [])
    if excluded_obj_ids:
        candidates = [c for c in candidates if c not in excluded_obj_ids]
    mode = _effective_mode(user_state, gripper_hist, memory)

    if mode == "rotation":
        triggered, dom_cell, yaw1, yaw2 = _has_yaw_oscillation(gripper_hist)
        if triggered:
            target_obj = next((o for o in objects if o["cell"] == dom_cell), None)
            if target_obj:
                text = (
                    f"I notice you are struggling aligning the gripper yaw while near the {target_obj['label']}. "
                    f"Is that what you are trying to do?"
                )
                choices = ["1) YES", "2) NO"]
                context = {
                    "type": "intent_gate_yaw",
                    "obj_id": target_obj["id"],
                    "label": target_obj["label"],
                    "action": "ALIGN_YAW",
                }
                return _interact("QUESTION", text, choices, context, state)

    ranked = _rank_candidates(objects, candidates, current_cell)
    if len(ranked) >= 2:
        k = min(3, len(ranked))
        a = ranked[0]
        others = ranked[1:k]
        other_labels = ", ".join(o["label"] for o in others)
        if mode == "rotation":
            text = (
                f"I notice you are rotating the gripper near the {a['label']}. However, {other_labels} "
                f"{'is' if len(others)==1 else 'are'} also close. Are you trying to align yaw to one of these?"
            )
            action = "ALIGN_YAW"
        else:
            text = (
                f"I notice you are approaching the {a['label']}. However, {other_labels} "
                f"{'is' if len(others)==1 else 'are'} also close. Are you trying to grasp one of these?"
            )
            action = "APPROACH"
        choices = ["1) YES", "2) NO"]
        context = {
            "type": "intent_gate_candidates",
            "labels": [o["label"] for o in ranked[:k]],
            "action": action,
        }
        return _interact("QUESTION", text, choices, context, state)

    triggered, dom_cell, yaw1, yaw2 = _has_yaw_oscillation(gripper_hist)
    if triggered:
        target_obj = next((o for o in objects if o["cell"] == dom_cell), None)
        if target_obj:
            text = (
                f"I notice you are struggling aligning the gripper yaw while near the {target_obj['label']}. "
                f"Is that what you are trying to do?"
            )
            choices = ["1) YES", "2) NO"]
            context = {"type": "intent_gate_yaw", "obj_id": target_obj["id"], "label": target_obj["label"], "action": "ALIGN_YAW"}
            return _interact("QUESTION", text, choices, context, state)

    return None


def oracle_decide_tool(
    objects: Sequence[Dict],
    gripper_hist: Sequence[Dict],
    memory: Dict,
    state: OracleState,
    user_state: Optional[UserState] = None,
) -> Dict:
    current_cell = gripper_hist[-1]["cell"]
    candidates = list(memory.get("candidates", []))
    excluded_obj_ids = set(memory.get("excluded_obj_ids") or [])
    if excluded_obj_ids:
        candidates = [c for c in candidates if c not in excluded_obj_ids]
    objects_by_id = {o["id"]: o for o in objects}
    current_yaw = gripper_hist[-1]["yaw"]
    mode = _effective_mode(user_state, gripper_hist, memory)

    if state.terminate_episode:
        return _interact(
            "SUGGESTION",
            "Okay. I'll stay out of the way.",
            ["1) OK"],
            {"type": "terminal_ack"},
            state,
        )

    last_action = memory.get("last_action") or {}
    if (
        isinstance(last_action, dict)
        and last_action.get("tool") == "APPROACH"
        and isinstance(last_action.get("obj"), str)
        and state.pending_action_obj_id is None
        and state.selected_obj_id is None
        and not (state.awaiting_confirmation or state.awaiting_choice or state.awaiting_help or state.awaiting_anything_else or state.awaiting_mode_select or state.awaiting_intent_gate)
    ):
        obj_id = last_action["obj"]
        obj = objects_by_id.get(obj_id)
        if obj and current_cell == obj["cell"] and current_yaw != obj["yaw"]:
            state.selected_obj_id = obj_id
            state.awaiting_confirmation = True
            question = f"Do you want me to also align yaw to the {obj['label']}?"
            context = {"type": "confirm", "obj_id": obj_id, "label": obj["label"], "action": "ALIGN_YAW"}
            choices = ["1) YES", "2) NO"]
            return _interact("CONFIRM", question, choices, context, state)

    if state.pending_action_obj_id is not None and state.pending_action_obj_id in objects_by_id:
        target = objects_by_id[state.pending_action_obj_id]

        def clear_pending() -> None:
            state.pending_action_obj_id = None
            state.selected_obj_id = None
            state.pending_mode = None
            state.awaiting_confirmation = False
            state.awaiting_choice = False
            state.awaiting_help = False
            state.awaiting_intent_gate = False
            state.awaiting_anything_else = False
            state.awaiting_mode_select = False

        if state.pending_mode == "APPROACH":
            if current_cell != target["cell"]:
                tool = _tool("APPROACH", {"obj": target["id"]})
                clear_pending()
                return tool
            clear_pending()
        elif state.pending_mode == "ALIGN_YAW":
            if current_yaw != target["yaw"]:
                tool = _tool("ALIGN_YAW", {"obj": target["id"]})
                clear_pending()
                return tool
            clear_pending()
        else:
            if current_cell != target["cell"]:
                tool = _tool("APPROACH", {"obj": target["id"]})
                clear_pending()
                return tool
            if current_yaw != target["yaw"]:
                tool = _tool("ALIGN_YAW", {"obj": target["id"]})
                clear_pending()
                return tool
            clear_pending()

    if state.awaiting_confirmation:
        obj_id = state.selected_obj_id or state.intended_obj_id
        obj = objects_by_id.get(obj_id)
        if obj:
            action = state.last_prompt_context.get("action") if state.last_prompt_context else None
            if action == "ALIGN_YAW":
                question = f"Do you want me to align yaw to the {obj['label']}?"
            else:
                question = f"Do you want me to approach the {obj['label']}?"
            choices = ["1) YES", "2) NO"]
            context = {"type": "confirm", "obj_id": obj["id"], "label": obj["label"], "action": action or "APPROACH"}
            return _interact("CONFIRM", question, choices, context, state)
        state.awaiting_confirmation = False

    if state.awaiting_anything_else:
        text = "Uh, I should have misunderstood. Is there anything else I can help with?"
        choices = ["1) YES", "2) NO"]
        context = {"type": "anything_else"}
        return _interact("QUESTION", text, choices, context, state)

    if state.awaiting_mode_select:
        text = "Do you want help with approaching an object or aligning the gripper yaw to an object?"
        choices = ["1) APPROACH", "2) ALIGN_YAW"]
        context = {"type": "mode_select"}
        return _interact("SUGGESTION", text, choices, context, state)

    if state.awaiting_choice:
        ranked = _rank_candidates(objects, candidates, current_cell)
        if ranked:
            k = min(4, len(ranked))
            labels = [ranked[i]["label"] for i in range(k)]
            obj_ids = [ranked[i]["id"] for i in range(k)]
            choices = [f"{i+1}) {labels[i]}" for i in range(k)]
            none_idx = k + 1
            choices.append(f"{none_idx}) None of them")
            context = {"type": "candidate_choice", "labels": labels, "obj_ids": obj_ids, "none_index": none_idx}
            if state.pending_mode == "ALIGN_YAW":
                prompt = "Which object do you want me to align yaw to?"
            elif state.pending_mode == "APPROACH":
                prompt = "Which object do you want me to help you approach?"
            else:
                prompt = "Uh, which one do you want?"
            return _interact("QUESTION", prompt, choices, context, state)
        state.awaiting_choice = False
        state.awaiting_anything_else = True
        text = "Okay — none of those. Is there anything else I can help with?"
        choices = ["1) YES", "2) NO"]
        context = {"type": "anything_else"}
        return _interact("QUESTION", text, choices, context, state)

    if state.awaiting_help:
        triggered, dom_cell, yaw1, yaw2 = _has_yaw_oscillation(gripper_hist)
        if triggered:
            target_obj = next((o for o in objects if o["cell"] == dom_cell), None)
            if target_obj and target_obj["yaw"] not in {yaw1, yaw2}:
                text = f"Do you want me to help you align yaw to the {target_obj['label']}?"
                choices = ["1) YES", "2) NO"]
                context = {"type": "help", "obj_id": target_obj["id"], "yaws": (yaw1, yaw2, target_obj["yaw"])}
                return _interact("SUGGESTION", text, choices, context, state)
        state.awaiting_help = False

    if state.awaiting_intent_gate:
        gate = _emit_intent_gate(objects, gripper_hist, memory, state, user_state=user_state)
        if gate is not None:
            return gate
        state.awaiting_intent_gate = False

    if int(memory.get("n_interactions", 0)) == 0 and not (memory.get("past_dialogs") or []):
        if not (state.awaiting_confirmation or state.awaiting_choice or state.awaiting_help):
            state.awaiting_intent_gate = True
            gate = _emit_intent_gate(objects, gripper_hist, memory, state, user_state=user_state)
            if gate is not None:
                return gate
            state.awaiting_intent_gate = False

    if state.selected_obj_id is not None and not state.awaiting_confirmation:
        obj = objects_by_id.get(state.selected_obj_id)
        if obj:
            if state.pending_mode == "ALIGN_YAW":
                question = f"Do you want me to align yaw to the {obj['label']}?"
                context = {"type": "confirm", "obj_id": obj["id"], "label": obj["label"], "action": "ALIGN_YAW"}
            elif state.pending_mode == "APPROACH":
                question = f"Do you want me to approach the {obj['label']}?"
                context = {"type": "confirm", "obj_id": obj["id"], "label": obj["label"], "action": "APPROACH"}
            elif current_cell != obj["cell"]:
                question = f"Do you want me to approach the {obj['label']}?"
                context = {"type": "confirm", "obj_id": obj["id"], "label": obj["label"], "action": "APPROACH"}
            elif current_yaw != obj["yaw"]:
                question = f"Do you want me to align yaw to the {obj['label']}?"
                context = {"type": "confirm", "obj_id": obj["id"], "label": obj["label"], "action": "ALIGN_YAW"}
            else:
                state.selected_obj_id = None
                state.awaiting_confirmation = False
                obj = None
            if obj is not None:
                state.awaiting_confirmation = True
                choices = ["1) YES", "2) NO"]
                return _interact("CONFIRM", question, choices, context, state)

    top_two = _top_two_candidates(objects, candidates, current_cell)
    last_calls = list(memory.get("last_tool_calls", []))
    just_moved = bool(last_calls and last_calls[-1] == "APPROACH")
    if top_two and not state.awaiting_choice and not state.awaiting_confirmation:
        a, b = top_two
        osc = _has_cell_oscillation(gripper_hist, a["cell"], b["cell"])
        allow = just_moved or osc
        if allow:
            dist_a = manhattan(current_cell, a["cell"])
            dist_b = manhattan(current_cell, b["cell"])
            if abs(dist_a - dist_b) <= 1:
                state.awaiting_intent_gate = True
                ranked = _rank_candidates(objects, candidates, current_cell)
                k = min(3, len(ranked))
                a0 = ranked[0]
                others = ranked[1:k]
                other_labels = ", ".join(o["label"] for o in others)
                text = (
                    f"I notice you are approaching the {a0['label']}. However, {other_labels} "
                    f"{'is' if len(others)==1 else 'are'} also close. Are you trying to grasp one of these?"
                )
                choices = ["1) YES", "2) NO"]
                context = {"type": "intent_gate_candidates", "labels": [o["label"] for o in ranked[:k]]}
                return _interact("QUESTION", text, choices, context, state)

    triggered, dom_cell, yaw1, yaw2 = _has_yaw_oscillation(gripper_hist)
    if triggered and not state.awaiting_help:
        target_obj = next((o for o in objects if o["cell"] == dom_cell), None)
        if target_obj and target_obj["yaw"] not in {yaw1, yaw2}:
            state.awaiting_intent_gate = True
            text = (
                f"I notice you are struggling aligning the gripper yaw while near the {target_obj['label']}. "
                f"Is that what you are trying to do?"
            )
            choices = ["1) YES", "2) NO"]
            context = {"type": "intent_gate_yaw", "obj_id": target_obj["id"], "label": target_obj["label"]}
            return _interact("QUESTION", text, choices, context, state)

    intended = objects_by_id[state.intended_obj_id]
    if current_cell != intended["cell"]:
        return _tool("APPROACH", {"obj": intended["id"]})
    if current_yaw != intended["yaw"]:
        return _tool("ALIGN_YAW", {"obj": intended["id"]})

    if not state.awaiting_confirmation:
        state.awaiting_confirmation = True
        text = f"Do you want me to approach the {intended['label']}?"
        choices = ["1) YES", "2) NO"]
        context = {"type": "confirm", "obj_id": intended["id"], "label": intended["label"], "action": "APPROACH"}
        return _interact("CONFIRM", text, choices, context, state)

    text = f"Do you want me to approach the {intended['label']}?"
    choices = ["1) YES", "2) NO"]
    context = {"type": "confirm", "obj_id": intended["id"], "label": intended["label"], "action": "APPROACH"}
    return _interact("CONFIRM", text, choices, context, state)


def validate_tool_call(tool_call: Dict) -> None:
    if not isinstance(tool_call, dict) or set(tool_call.keys()) != {"tool", "args"}:
        raise ValueError("Tool call must be {tool, args}")
    tool = tool_call["tool"]
    args = tool_call["args"]
    if tool not in {"INTERACT", "APPROACH", "ALIGN_YAW"}:
        raise ValueError(f"Invalid tool: {tool}")
    if not isinstance(args, dict):
        raise ValueError("args must be an object")
    if tool == "INTERACT":
        required_keys = {"kind", "text", "choices"}
        if set(args.keys()) != required_keys:
            raise ValueError("INTERACT args must be {kind,text,choices}")
        if args["kind"] not in {"QUESTION", "SUGGESTION", "CONFIRM"}:
            raise ValueError("Invalid INTERACT.kind")
        if not isinstance(args["text"], str):
            raise ValueError("INTERACT.text must be a string")
        choices = args["choices"]
        if not isinstance(choices, list) or not choices or not all(isinstance(c, str) for c in choices):
            raise ValueError("INTERACT.choices must be a non-empty list of strings")
        if len(choices) > MAX_INTERACT_CHOICES:
            raise ValueError(f"INTERACT.choices must have <= {MAX_INTERACT_CHOICES} items")
        for c in choices:
            prefix = c.split(")", 1)[0]
            if not prefix.isdigit():
                raise ValueError("INTERACT.choices must start with numbered prefixes like '1)'")
    elif tool in {"APPROACH", "ALIGN_YAW"}:
        if set(args.keys()) != {"obj"} or not isinstance(args["obj"], str):
            raise ValueError(f"{tool} args must be {{obj}}")


def strip_choice_label(choice: str) -> str:
    if ")" in choice:
        return choice.split(")", 1)[1].strip()
    return choice.strip()


def choice_to_user_content(choice: str) -> str:
    semantic = strip_choice_label(choice).strip().upper()
    if semantic in {"YES", "NO"}:
        return semantic
    return strip_choice_label(choice).strip()


def apply_oracle_user_reply(user_content: str, objects: Sequence[Dict], memory: Dict, state: OracleState) -> bool:
    ctx = state.last_prompt_context or {}
    t = ctx.get("type")

    def reset_conversation_only() -> None:
        memory["n_interactions"] = 0
        memory["past_dialogs"] = []
        memory["last_tool_calls"] = []
        memory["excluded_obj_ids"] = []
        memory["last_action"] = {}
        state.selected_obj_id = None
        state.pending_action_obj_id = None
        state.pending_mode = None
        state.awaiting_confirmation = False
        state.awaiting_help = False
        state.awaiting_choice = False
        state.awaiting_intent_gate = False
        state.awaiting_anything_else = False
        state.awaiting_mode_select = False
        state.terminate_episode = False
        state.last_prompt_context = None

    def set_selected_by_label(label: str) -> None:
        for o in objects:
            if o["label"] == label:
                state.selected_obj_id = o["id"]
                state.intended_obj_id = o["id"]
                return

    auto_continue = True

    if t == "intent_gate_candidates":
        if user_content.upper() == "YES":
            state.awaiting_choice = True
            state.awaiting_intent_gate = False
            action = str(ctx.get("action") or "APPROACH").upper()
            state.pending_mode = action if action in {"APPROACH", "ALIGN_YAW"} else "APPROACH"
        else:
            state.awaiting_intent_gate = False
            state.awaiting_choice = False
            state.awaiting_anything_else = True
            state.pending_mode = None
            state.selected_obj_id = None
    elif t == "intent_gate_yaw":
        if user_content.upper() == "YES":
            state.awaiting_help = True
            state.awaiting_intent_gate = False
            state.pending_mode = "ALIGN_YAW"
            obj_id = ctx.get("obj_id")
            if isinstance(obj_id, str):
                state.selected_obj_id = obj_id
        else:
            state.awaiting_intent_gate = False
            state.awaiting_help = False
            state.awaiting_anything_else = True
            state.pending_mode = None
            state.selected_obj_id = None
    elif t == "candidate_choice":
        labels: List[str] = list(ctx.get("labels") or [])
        obj_ids: List[str] = list(ctx.get("obj_ids") or [])
        none_index = int(ctx.get("none_index") or (len(labels) + 1))
        if user_content.strip().lower() == "none of them":
            ex = set(memory.get("excluded_obj_ids") or [])
            for oid in obj_ids:
                ex.add(oid)
            memory["excluded_obj_ids"] = sorted(ex)
            state.selected_obj_id = None
            state.awaiting_choice = True
            state.awaiting_confirmation = False
        elif user_content in labels:
            set_selected_by_label(user_content)
            state.awaiting_choice = False
            state.awaiting_confirmation = False
        elif user_content.isdigit():
            idx = int(user_content) - 1
            if int(user_content) == none_index:
                ex = set(memory.get("excluded_obj_ids") or [])
                for oid in obj_ids:
                    ex.add(oid)
                memory["excluded_obj_ids"] = sorted(ex)
                state.selected_obj_id = None
                state.awaiting_choice = True
                state.awaiting_confirmation = False
            else:
                if 0 <= idx < len(labels):
                    set_selected_by_label(labels[idx])
                state.awaiting_choice = False
                state.awaiting_confirmation = False
    elif t == "confirm":
        obj_id = ctx.get("obj_id")
        action = str(ctx.get("action") or "").upper()
        if user_content.upper() == "YES" and isinstance(obj_id, str):
            state.pending_action_obj_id = obj_id
            state.selected_obj_id = obj_id
            if action in {"APPROACH", "ALIGN_YAW"}:
                state.pending_mode = action
        else:
            state.pending_action_obj_id = None
            state.pending_mode = None
            state.selected_obj_id = None
            state.awaiting_anything_else = True
        state.awaiting_confirmation = False
    elif t == "help":
        obj_id = ctx.get("obj_id")
        if user_content.upper() == "YES" and isinstance(obj_id, str):
            state.pending_action_obj_id = obj_id
            state.selected_obj_id = obj_id
            state.pending_mode = "ALIGN_YAW"
        else:
            state.pending_action_obj_id = None
            state.pending_mode = None
            state.selected_obj_id = None
            state.awaiting_anything_else = True
        state.awaiting_help = False
    elif t == "anything_else":
        if user_content.upper() == "YES":
            memory["excluded_obj_ids"] = []
            state.awaiting_mode_select = True
            state.awaiting_anything_else = False
        else:
            reset_conversation_only()
            auto_continue = False
    elif t == "mode_select":
        uc = user_content.strip().upper()
        if uc in {"APPROACH", "ALIGN_YAW"}:
            state.pending_mode = uc
        elif user_content == "1":
            state.pending_mode = "APPROACH"
        elif user_content == "2":
            state.pending_mode = "ALIGN_YAW"
        state.awaiting_mode_select = False
        state.awaiting_choice = True
    elif t == "terminal_ack":
        reset_conversation_only()
        auto_continue = False

    state.last_prompt_context = None
    return auto_continue
