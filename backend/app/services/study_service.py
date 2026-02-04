"""Study session management and interaction logging for PRISM user studies."""
from __future__ import annotations

import csv
import io
import json
import random
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from app.config import settings

# Lazy imports to avoid circular deps and optional model dependency
def _get_model_decision_for_row_index(row_index: int) -> str | None:
    """Get model decision (Approve/Reject) for default dataset row. Returns None if unavailable."""
    try:
        from app.services.data_service import data_service
        from app.services.model_service import model_service
        df = data_service.load_default_dataset()
        if row_index < 0 or row_index >= len(df):
            return None
        row = df.iloc[row_index: row_index + 1]
        X = data_service.transform(row)
        dec, _, _ = model_service.predict_single(X[0])
        return "Approve" if dec == "+" else "Reject"
    except Exception:
        return None

# Study data storage paths
STUDY_DIR = Path(settings.base_dir) / "study_data"
SESSIONS_FILE = STUDY_DIR / "sessions.jsonl"
INTERACTIONS_FILE = STUDY_DIR / "interactions.jsonl"
TASKS_FILE = STUDY_DIR / "task_responses.jsonl"
PRE_QUESTIONNAIRES_FILE = STUDY_DIR / "pre_questionnaires.jsonl"
POST_QUESTIONNAIRES_FILE = STUDY_DIR / "post_questionnaires.jsonl"

# Default study configuration
DEFAULT_CONFIG = {
    "study_id": "prism_study_v1",
    "conditions": ["interactive", "static", "minimal"],
    "tasks_per_session": 4,
    "include_attention_checks": True,
    "attention_check_frequency": 2,  # Attention check every 2 tasks
    "randomize_task_order": True,
    "random_seed": 42,
    "within_subjects": False,  # If True, same participant sees multiple conditions (blocks)
    "within_subjects_blocks": [{"condition": "static", "tasks": 2}, {"condition": "interactive", "tasks": 2}],
}

# Predefined evaluation tasks
EVALUATION_TASKS = [
    {
        "task_id": "decision_1",
        "task_type": "decision",
        "row_index": 0,
        "question": "Based on the information provided, would you approve this credit application?",
        "options": ["Approve", "Reject", "Uncertain"],
        "correct_answer": None,  # Depends on model output
    },
    {
        "task_id": "comprehension_1",
        "task_type": "comprehension",
        "row_index": 0,
        "question": "What is the MOST important factor influencing this decision?",
        "options": None,  # Free text
        "correct_answer": None,
    },
    {
        "task_id": "counterfactual_1",
        "task_type": "counterfactual",
        "row_index": 0,
        "question": "What single change would most likely flip this decision?",
        "options": None,
        "correct_answer": None,
    },
    {
        "task_id": "attention_1",
        "task_type": "attention_check",
        "row_index": -1,  # No specific row
        "question": "To verify you're paying attention, please select 'Blue' from the options below.",
        "options": ["Red", "Blue", "Green", "Yellow"],
        "correct_answer": "Blue",
    },
    {
        "task_id": "attention_2",
        "task_type": "attention_check",
        "row_index": -1,
        "question": "Please select the number that equals 2+2.",
        "options": ["3", "4", "5", "6"],
        "correct_answer": "4",
    },
    {
        "task_id": "attention_3",
        "task_type": "attention_check",
        "row_index": -1,
        "question": "Which of these is a color?",
        "options": ["Triangle", "Seven", "Purple", "Monday"],
        "correct_answer": "Purple",
    },
]


class StudyService:
    """Manages user study sessions, tasks, and interaction logging."""

    def __init__(self) -> None:
        STUDY_DIR.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, dict[str, Any]] = {}
        self._task_responses: dict[str, list[dict[str, Any]]] = {}  # session_id -> responses
        self._pre_questionnaires: dict[str, dict[str, Any]] = {}
        self._post_questionnaires: dict[str, dict[str, Any]] = {}
        self._config = DEFAULT_CONFIG.copy()
        self._load_data()

    def _load_data(self) -> None:
        """Load all existing data from files."""
        # Load sessions
        if SESSIONS_FILE.exists():
            with open(SESSIONS_FILE, "r") as f:
                for line in f:
                    if line.strip():
                        session = json.loads(line)
                        self._sessions[session["session_id"]] = session

        # Load task responses
        if TASKS_FILE.exists():
            with open(TASKS_FILE, "r") as f:
                for line in f:
                    if line.strip():
                        resp = json.loads(line)
                        sid = resp.get("session_id")
                        if sid:
                            if sid not in self._task_responses:
                                self._task_responses[sid] = []
                            self._task_responses[sid].append(resp)

        # Load questionnaires
        if PRE_QUESTIONNAIRES_FILE.exists():
            with open(PRE_QUESTIONNAIRES_FILE, "r") as f:
                for line in f:
                    if line.strip():
                        q = json.loads(line)
                        self._pre_questionnaires[q.get("session_id")] = q

        if POST_QUESTIONNAIRES_FILE.exists():
            with open(POST_QUESTIONNAIRES_FILE, "r") as f:
                for line in f:
                    if line.strip():
                        q = json.loads(line)
                        self._post_questionnaires[q.get("session_id")] = q

    def _save_session(self, session: dict[str, Any]) -> None:
        """Append session to file."""
        with open(SESSIONS_FILE, "a") as f:
            f.write(json.dumps(session, default=str) + "\n")

    def _log_interaction(self, interaction: dict[str, Any]) -> None:
        """Append interaction to file."""
        with open(INTERACTIONS_FILE, "a") as f:
            f.write(json.dumps(interaction, default=str) + "\n")

    def _save_task_response(self, response: dict[str, Any]) -> None:
        """Save task response to file."""
        with open(TASKS_FILE, "a") as f:
            f.write(json.dumps(response, default=str) + "\n")

    def _save_pre_questionnaire(self, data: dict[str, Any]) -> None:
        """Save pre-study questionnaire."""
        with open(PRE_QUESTIONNAIRES_FILE, "a") as f:
            f.write(json.dumps(data, default=str) + "\n")

    def _save_post_questionnaire(self, data: dict[str, Any]) -> None:
        """Save post-study questionnaire."""
        with open(POST_QUESTIONNAIRES_FILE, "a") as f:
            f.write(json.dumps(data, default=str) + "\n")

    def _rewrite_sessions(self) -> None:
        """Rewrite all sessions to file (for updates)."""
        with open(SESSIONS_FILE, "w") as f:
            for session in self._sessions.values():
                f.write(json.dumps(session, default=str) + "\n")

    def _parse_datetime(self, dt_str: str) -> datetime:
        """Parse datetime string, handling timezone-aware and naive formats."""
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        return dt

    # ============== SESSION MANAGEMENT ==============

    def create_session(
        self,
        participant_id: str,
        condition: str,
        pre_questionnaire: dict[str, Any] | None = None,
        within_subjects: bool | None = None,
    ) -> dict[str, Any]:
        """Create a new study session. If within_subjects=True, tasks span multiple conditions (blocks)."""
        session_id = str(uuid.uuid4())[:8]
        now = datetime.utcnow().isoformat()
        use_within = within_subjects if within_subjects is not None else self._config.get("within_subjects", False)

        # Generate task sequence for this session
        if use_within:
            tasks = self._generate_task_sequence_within_subjects()
        else:
            tasks = self._generate_task_sequence(condition)

        session = {
            "session_id": session_id,
            "participant_id": participant_id,
            "condition": condition,
            "started_at": now,
            "ended_at": None,
            "is_active": True,
            # Counters
            "total_interactions": 0,
            "total_decisions": 0,
            "total_whatif_explorations": 0,
            "first_decision_at": None,
            "decision_times": [],
            "features_explored": [],
            "mode_switches": 0,
            "exports": 0,
            # Task tracking
            "tasks": tasks,
            "current_task_index": 0,
            "tasks_completed": 0,
            "tasks_correct": 0,
            "attention_checks_passed": 0,
            "attention_checks_total": 0,
            # Enhanced tracking
            "hover_times": [],
            "scroll_depths": [],
            "mode_preferences": {},
            "decision_reversals": 0,
        }

        self._sessions[session_id] = session
        self._save_session(session)

        # Save pre-questionnaire if provided
        if pre_questionnaire:
            pre_data = {
                "session_id": session_id,
                "participant_id": participant_id,
                "timestamp": now,
                **pre_questionnaire,
            }
            self._pre_questionnaires[session_id] = pre_data
            self._save_pre_questionnaire(pre_data)

        # Log session start
        self._log_interaction({
            "session_id": session_id,
            "participant_id": participant_id,
            "action": "session_start",
            "condition": condition,
            "timestamp": now,
            "details": {"has_pre_questionnaire": pre_questionnaire is not None},
        })

        return session

    def _generate_task_sequence(self, condition: str) -> list[dict[str, Any]]:
        """Generate a sequence of tasks for a session."""
        tasks = []
        task_count = self._config["tasks_per_session"]
        attention_freq = self._config["attention_check_frequency"]

        # Get non-attention-check tasks
        regular_tasks = [t for t in EVALUATION_TASKS if t["task_type"] != "attention_check"]
        attention_checks = [t for t in EVALUATION_TASKS if t["task_type"] == "attention_check"]

        # Seed for reproducibility
        rng = random.Random(self._config["random_seed"])

        # Generate tasks
        row_indices = list(range(50))  # Assume up to 50 rows
        if self._config["randomize_task_order"]:
            rng.shuffle(row_indices)

        for i in range(task_count):
            # Insert attention check at intervals
            if self._config["include_attention_checks"] and (i + 1) % attention_freq == 0:
                attn = rng.choice(attention_checks).copy()
                attn["sequence_index"] = len(tasks)
                tasks.append(attn)

            # Add regular task
            if regular_tasks:
                task_template = regular_tasks[i % len(regular_tasks)].copy()
                task_template["task_id"] = f"{task_template['task_id']}_{i}"
                row_idx = row_indices[i % len(row_indices)]
                task_template["row_index"] = row_idx
                task_template["sequence_index"] = len(tasks)
                if task_template["task_type"] == "decision":
                    task_template["correct_answer"] = _get_model_decision_for_row_index(row_idx)
                tasks.append(task_template)

        return tasks

    def _generate_task_sequence_within_subjects(self) -> list[dict[str, Any]]:
        """Generate tasks across multiple conditions (within-subjects). Each task has a 'condition' field."""
        tasks = []
        blocks = self._config.get("within_subjects_blocks") or [
            {"condition": "static", "tasks": 2},
            {"condition": "interactive", "tasks": 2},
        ]
        regular_tasks = [t for t in EVALUATION_TASKS if t["task_type"] != "attention_check"]
        attention_checks = [t for t in EVALUATION_TASKS if t["task_type"] == "attention_check"]
        rng = random.Random(self._config["random_seed"])
        row_indices = list(range(50))
        if self._config["randomize_task_order"]:
            rng.shuffle(row_indices)
        attention_freq = self._config["attention_check_frequency"]
        seq_idx = 0
        for block in blocks:
            block_condition = block.get("condition", "interactive")
            n_slots = block.get("tasks", 2)
            for slot in range(n_slots):
                if self._config["include_attention_checks"] and (seq_idx + 1) % attention_freq == 0:
                    attn = rng.choice(attention_checks).copy()
                    attn["sequence_index"] = seq_idx
                    attn["condition"] = block_condition
                    tasks.append(attn)
                else:
                    task_template = regular_tasks[seq_idx % len(regular_tasks)].copy()
                    task_template["task_id"] = f"{task_template['task_id']}_{seq_idx}_{block_condition}"
                    row_idx = row_indices[seq_idx % len(row_indices)]
                    task_template["row_index"] = row_idx
                    task_template["sequence_index"] = seq_idx
                    task_template["condition"] = block_condition
                    if task_template["task_type"] == "decision":
                        task_template["correct_answer"] = _get_model_decision_for_row_index(row_idx)
                    tasks.append(task_template)
                seq_idx += 1
        return tasks

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Get session by ID."""
        return self._sessions.get(session_id)

    def end_session(self, session_id: str) -> dict[str, Any] | None:
        """End a study session."""
        session = self._sessions.get(session_id)
        if not session:
            return None

        now = datetime.utcnow().isoformat()
        session["ended_at"] = now
        session["is_active"] = False

        self._log_interaction({
            "session_id": session_id,
            "participant_id": session["participant_id"],
            "action": "session_end",
            "timestamp": now,
            "details": {},
        })

        self._rewrite_sessions()
        return session

    def get_all_sessions(self) -> list[dict[str, Any]]:
        """Get all study sessions."""
        return list(self._sessions.values())

    # ============== INTERACTION LOGGING ==============

    def log_interaction(
        self,
        session_id: str,
        action: str,
        details: dict[str, Any] | None = None,
        timestamp: str | None = None,
    ) -> bool:
        """Log a user interaction with enhanced tracking."""
        session = self._sessions.get(session_id)
        if not session or not session["is_active"]:
            return False

        now = timestamp or datetime.utcnow().isoformat()
        details = details or {}

        interaction = {
            "session_id": session_id,
            "participant_id": session["participant_id"],
            "condition": session["condition"],
            "action": action,
            "timestamp": now,
            "details": details,
        }

        self._log_interaction(interaction)

        # Update session counters
        session["total_interactions"] += 1

        if action == "view_decision":
            session["total_decisions"] += 1
            if session["first_decision_at"] is None:
                session["first_decision_at"] = now
            session["decision_times"].append(now)

        elif action == "whatif_adjust":
            session["total_whatif_explorations"] += 1
            feature = details.get("feature")
            if feature and feature not in session["features_explored"]:
                session["features_explored"].append(feature)

        elif action == "change_mode":
            session["mode_switches"] += 1
            mode = details.get("to")
            if mode:
                session["mode_preferences"][mode] = session["mode_preferences"].get(mode, 0) + 1

        elif action == "export":
            session["exports"] += 1

        elif action == "decision_reversal":
            session["decision_reversals"] += 1

        # Enhanced tracking
        if "hover_time_ms" in details:
            session["hover_times"].append(details["hover_time_ms"])
        if "scroll_depth_percent" in details:
            session["scroll_depths"].append(details["scroll_depth_percent"])

        return True

    # ============== TASK MANAGEMENT ==============

    def get_current_task(self, session_id: str) -> dict[str, Any] | None:
        """Get the current task for a session."""
        session = self._sessions.get(session_id)
        if not session or not session["is_active"]:
            return None

        idx = session.get("current_task_index", 0)
        tasks = session.get("tasks", [])

        if idx < len(tasks):
            task = tasks[idx].copy()
            task["total_tasks"] = len(tasks)
            task["current_index"] = idx
            # Don't send correct answer to client
            task.pop("correct_answer", None)
            return task

        return None  # All tasks completed

    def submit_task_response(
        self,
        session_id: str,
        task_id: str,
        response: str,
        confidence: int = 3,
        time_taken_seconds: float = 0,
    ) -> dict[str, Any]:
        """Submit a response to a task."""
        session = self._sessions.get(session_id)
        if not session:
            return {"error": "Session not found"}

        # Find the task
        task = None
        for t in session.get("tasks", []):
            if t["task_id"] == task_id:
                task = t
                break

        if not task:
            return {"error": "Task not found"}

        # Score the response
        correct_answer = task.get("correct_answer")
        is_correct = None
        if correct_answer:
            is_correct = response.strip().lower() == correct_answer.strip().lower()

        result = {
            "session_id": session_id,
            "participant_id": session["participant_id"],
            "task_id": task_id,
            "task_type": task["task_type"],
            "response": response,
            "correct_answer": correct_answer,
            "is_correct": is_correct,
            "confidence": confidence,
            "time_taken_seconds": time_taken_seconds,
            "timestamp": datetime.utcnow().isoformat(),
        }
        if task.get("condition") is not None:
            result["task_condition"] = task["condition"]

        # Save response
        if session_id not in self._task_responses:
            self._task_responses[session_id] = []
        self._task_responses[session_id].append(result)
        self._save_task_response(result)

        # Update session counters
        session["tasks_completed"] += 1
        if is_correct is not None:
            if task["task_type"] == "attention_check":
                session["attention_checks_total"] += 1
                if is_correct:
                    session["attention_checks_passed"] += 1
            else:
                if is_correct:
                    session["tasks_correct"] += 1

        # Advance to next task
        session["current_task_index"] += 1

        # Log interaction
        details = {
            "task_id": task_id,
            "task_type": task["task_type"],
            "is_correct": is_correct,
            "time_taken_seconds": time_taken_seconds,
        }
        if task.get("condition") is not None:
            details["task_condition"] = task["condition"]
        self._log_interaction({
            "session_id": session_id,
            "participant_id": session["participant_id"],
            "action": "task_submit",
            "timestamp": result["timestamp"],
            "details": details,
        })

        return {
            "task_id": task_id,
            "is_correct": is_correct,
            "tasks_remaining": len(session["tasks"]) - session["current_task_index"],
        }

    # ============== QUESTIONNAIRES ==============

    def submit_post_questionnaire(
        self,
        session_id: str,
        questionnaire: dict[str, Any],
    ) -> dict[str, str]:
        """Submit post-study questionnaire."""
        session = self._sessions.get(session_id)
        if not session:
            return {"error": "Session not found"}

        post_data = {
            "session_id": session_id,
            "participant_id": session["participant_id"],
            "condition": session["condition"],
            "timestamp": datetime.utcnow().isoformat(),
            **questionnaire,
        }

        self._post_questionnaires[session_id] = post_data
        self._save_post_questionnaire(post_data)

        self._log_interaction({
            "session_id": session_id,
            "participant_id": session["participant_id"],
            "action": "post_questionnaire_submit",
            "timestamp": post_data["timestamp"],
            "details": {},
        })

        return {"status": "recorded"}

    # ============== METRICS & EXPORT ==============

    def get_session_metrics(self, session_id: str) -> dict[str, Any] | None:
        """Compute comprehensive metrics for a session."""
        session = self._sessions.get(session_id)
        if not session:
            return None

        started = self._parse_datetime(session["started_at"])
        ended = self._parse_datetime(session["ended_at"]) if session["ended_at"] else datetime.utcnow()
        duration = (ended - started).total_seconds()

        # Time metrics
        time_to_first = None
        if session.get("first_decision_at"):
            first_decision = self._parse_datetime(session["first_decision_at"])
            time_to_first = (first_decision - started).total_seconds()

        avg_time_per_decision = None
        if len(session.get("decision_times", [])) > 1:
            times = [self._parse_datetime(t) for t in session["decision_times"]]
            deltas = [(times[i+1] - times[i]).total_seconds() for i in range(len(times)-1)]
            avg_time_per_decision = sum(deltas) / len(deltas) if deltas else None

        # Task accuracy
        tasks_completed = session.get("tasks_completed", 0)
        tasks_correct = session.get("tasks_correct", 0)
        task_accuracy = (tasks_correct / tasks_completed * 100) if tasks_completed > 0 else None

        # Enhanced metrics
        hover_times = session.get("hover_times", [])
        avg_hover = sum(hover_times) / len(hover_times) if hover_times else None

        scroll_depths = session.get("scroll_depths", [])
        avg_scroll = sum(scroll_depths) / len(scroll_depths) if scroll_depths else None

        # Mode preference
        mode_prefs = session.get("mode_preferences", {})
        preferred_mode = max(mode_prefs, key=mode_prefs.get) if mode_prefs else None

        return {
            "session_id": session_id,
            "participant_id": session["participant_id"],
            "condition": session["condition"],
            "duration_seconds": duration,
            "total_interactions": session.get("total_interactions", 0),
            "decisions_viewed": session.get("total_decisions", 0),
            "whatif_explorations": session.get("total_whatif_explorations", 0),
            "mode_switches": session.get("mode_switches", 0),
            "exports": session.get("exports", 0),
            "time_to_first_decision_seconds": time_to_first,
            "avg_time_per_decision_seconds": avg_time_per_decision,
            "features_explored": session.get("features_explored", []),
            "feedback_submitted": session_id in self._post_questionnaires,
            # Enhanced metrics
            "tasks_completed": tasks_completed,
            "tasks_correct": tasks_correct,
            "task_accuracy": task_accuracy,
            "attention_checks_passed": session.get("attention_checks_passed", 0),
            "attention_checks_total": session.get("attention_checks_total", 0),
            "avg_hover_time_ms": avg_hover,
            "avg_scroll_depth": avg_scroll,
            "preferred_explanation_mode": preferred_mode,
            "decision_reversals": session.get("decision_reversals", 0),
        }

    def export_study_data(self) -> dict[str, Any]:
        """Export all study data for analysis."""
        sessions = []
        for session in self._sessions.values():
            metrics = self.get_session_metrics(session["session_id"])
            if metrics:
                sessions.append(metrics)

        interactions = []
        if INTERACTIONS_FILE.exists():
            with open(INTERACTIONS_FILE, "r") as f:
                for line in f:
                    if line.strip():
                        interactions.append(json.loads(line))

        task_results = []
        for session_responses in self._task_responses.values():
            task_results.extend(session_responses)

        # Compute summary statistics
        conditions = {}
        for s in sessions:
            cond = s["condition"]
            if cond not in conditions:
                conditions[cond] = {"count": 0, "total_duration": 0, "total_accuracy": 0}
            conditions[cond]["count"] += 1
            conditions[cond]["total_duration"] += s["duration_seconds"]
            if s["task_accuracy"] is not None:
                conditions[cond]["total_accuracy"] += s["task_accuracy"]

        condition_summary = {}
        for cond, data in conditions.items():
            condition_summary[cond] = {
                "sessions": data["count"],
                "avg_duration_seconds": data["total_duration"] / data["count"] if data["count"] > 0 else 0,
                "avg_task_accuracy": data["total_accuracy"] / data["count"] if data["count"] > 0 else 0,
            }

        return {
            "sessions": sessions,
            "interactions": interactions,
            "task_results": task_results,
            "pre_questionnaires": list(self._pre_questionnaires.values()),
            "post_questionnaires": list(self._post_questionnaires.values()),
            "summary": {
                "total_sessions": len(sessions),
                "by_condition": condition_summary,
                "total_interactions": sum(s["total_interactions"] for s in sessions),
                "total_tasks_completed": sum(s["tasks_completed"] for s in sessions),
                "avg_decisions_per_session": sum(s["decisions_viewed"] for s in sessions) / len(sessions) if sessions else 0,
            },
        }

    def export_for_r(self) -> str:
        """Export data in R-friendly CSV format (long format)."""
        data = self.export_study_data()
        output = io.StringIO()

        # Sessions CSV
        if data["sessions"]:
            writer = csv.DictWriter(output, fieldnames=[
                "session_id", "participant_id", "condition", "duration_seconds",
                "total_interactions", "decisions_viewed", "whatif_explorations",
                "mode_switches", "tasks_completed", "tasks_correct", "task_accuracy",
                "attention_checks_passed", "attention_checks_total",
                "avg_hover_time_ms", "avg_scroll_depth", "decision_reversals",
                "preferred_explanation_mode", "time_to_first_decision_seconds",
                "avg_time_per_decision_seconds"
            ])
            writer.writeheader()
            for s in data["sessions"]:
                writer.writerow({k: s.get(k, "") for k in writer.fieldnames})

        return output.getvalue()

    def export_interactions_long(self) -> str:
        """Export interactions in long format for statistical analysis."""
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["session_id", "participant_id", "condition", "action", "timestamp", "details_json"])

        if INTERACTIONS_FILE.exists():
            with open(INTERACTIONS_FILE, "r") as f:
                for line in f:
                    if line.strip():
                        i = json.loads(line)
                        writer.writerow([
                            i.get("session_id", ""),
                            i.get("participant_id", ""),
                            i.get("condition", ""),
                            i.get("action", ""),
                            i.get("timestamp", ""),
                            json.dumps(i.get("details", {})),
                        ])

        return output.getvalue()

    def export_questionnaires(self) -> dict[str, str]:
        """Export questionnaires as separate CSVs."""
        pre_output = io.StringIO()
        post_output = io.StringIO()

        # Pre-questionnaire
        if self._pre_questionnaires:
            first = list(self._pre_questionnaires.values())[0]
            writer = csv.DictWriter(pre_output, fieldnames=list(first.keys()))
            writer.writeheader()
            for q in self._pre_questionnaires.values():
                writer.writerow(q)

        # Post-questionnaire (flatten nested structures)
        if self._post_questionnaires:
            flat_posts = []
            for q in self._post_questionnaires.values():
                flat = {"session_id": q.get("session_id"), "participant_id": q.get("participant_id")}
                # Flatten NASA-TLX
                if "nasa_tlx" in q:
                    for k, v in q["nasa_tlx"].items():
                        flat[f"nasa_{k}"] = v
                # Flatten Trust
                if "trust" in q:
                    for k, v in q["trust"].items():
                        flat[f"trust_{k}"] = v
                # Flatten SUS (full 10-item)
                if "sus" in q:
                    for k, v in q["sus"].items():
                        flat[f"sus_{k}"] = v
                # Flatten legacy usability (2-item) if present
                if "usability" in q:
                    for k, v in q["usability"].items():
                        flat[f"usability_{k}"] = v
                # Open-ended
                for k in ["most_helpful_feature", "most_confusing_aspect", "improvement_suggestions", "additional_comments"]:
                    flat[k] = q.get(k, "")
                flat_posts.append(flat)

            if flat_posts:
                writer = csv.DictWriter(post_output, fieldnames=list(flat_posts[0].keys()))
                writer.writeheader()
                for fp in flat_posts:
                    writer.writerow(fp)

        return {
            "pre_questionnaires": pre_output.getvalue(),
            "post_questionnaires": post_output.getvalue(),
        }


# Singleton instance
study_service = StudyService()
