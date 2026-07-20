from typing import Dict, Any, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from simulation.models import Intersection
    from simulation.dqn_agent import DQNAgent


VALID_STATES = {"NORTH_SOUTH", "EAST_WEST"}
VALID_MODES = {"FIXED", "RL"}


class TrafficLightAgent:
    """
    Autonomous agent controlling signal phases for an intersection.
    Supports FIXED timing baseline mode and RL (Deep Q-Network) control mode.
    """

    def __init__(
        self,
        agent_id: str,
        intersection_id: str,
        initial_state: str = "NORTH_SOUTH",
        phase_duration: int = 10,
        mode: str = "FIXED",
    ):
        self.id: str = agent_id
        self.intersection_id: str = intersection_id
        self.phase_duration: int = phase_duration
        self.timer: int = 0
        self.mode: str = mode.upper() if mode.upper() in VALID_MODES else "FIXED"

        self.change_state(initial_state)

    def set_mode(self, mode: str) -> None:
        """Sets control mode to 'FIXED' or 'RL'."""
        mode_upper = mode.upper()
        if mode_upper not in VALID_MODES:
            raise ValueError(f"Invalid agent mode '{mode}'. Must be one of {VALID_MODES}")
        self.mode = mode_upper

    def change_state(self, new_state: str) -> None:
        """
        Sets the active signal phase with state validation.
        Ensures strict safety constraints (only valid phases allowed).
        """
        state_upper = new_state.upper()
        if state_upper not in VALID_STATES:
            raise ValueError(
                f"Invalid light state '{new_state}'. State must be one of {VALID_STATES}"
            )
        self.state: str = state_upper
        self.timer = 0

    def get_state(self) -> str:
        """Returns the current signal phase."""
        return self.state

    def step(
        self,
        intersection: Optional["Intersection"] = None,
        dqn_agent: Optional["DQNAgent"] = None,
        eval_mode: bool = False,
    ) -> bool:
        """
        Advances agent step logic.
        In FIXED mode: toggles phase when timer reaches phase_duration.
        In RL mode: queries DQNAgent act() for action decision (0: Hold, 1: Switch).
        """
        self.timer += 1

        if self.mode == "RL" and dqn_agent is not None and intersection is not None:
            obs_data = self.observe(intersection)
            queue_ns = sum(
                len(q)
                for r_id, q in intersection.lane_queues.items()
                if "EW" not in r_id
            )
            queue_ew = obs_data["total_queue"] - queue_ns
            phase_idx = 0.0 if self.state == "NORTH_SOUTH" else 1.0
            timer_norm = self.timer / max(1, self.phase_duration)

            state_vec = np.array(
                [float(queue_ns), float(queue_ew), phase_idx, timer_norm],
                dtype=np.float32,
            )

            action = dqn_agent.act(state_vec, eval_mode=eval_mode)
            if action == 1:
                next_state = (
                    "EAST_WEST" if self.state == "NORTH_SOUTH" else "NORTH_SOUTH"
                )
                self.change_state(next_state)
                return True
            return False

        # Default FIXED timing logic
        if self.timer >= self.phase_duration:
            next_state = (
                "EAST_WEST" if self.state == "NORTH_SOUTH" else "NORTH_SOUTH"
            )
            self.change_state(next_state)
            return True

        return False

    def observe(self, intersection: "Intersection") -> Dict[str, Any]:
        """
        Returns real-time intersection state observation for RL agents.
        """
        queue_lengths = {
            road_id: len(q) for road_id, q in intersection.lane_queues.items()
        }
        total_queue = sum(queue_lengths.values())

        return {
            "agent_id": self.id,
            "intersection_id": self.intersection_id,
            "state": self.state,
            "timer": self.timer,
            "phase_duration": self.phase_duration,
            "queue_lengths": queue_lengths,
            "total_queue": total_queue,
            "mode": self.mode,
        }
