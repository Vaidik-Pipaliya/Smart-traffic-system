import numpy as np
from typing import Tuple, Dict, Any, Optional
from simulation.models import Intersection, Road
from simulation.graph import TrafficGraph
from simulation.engine import SimulationEngine


class TrafficEnv:
    """
    Reinforcement Learning Environment for Single/Multi Intersection Traffic Signal Control.
    Converts simulation state into state vector, discrete action space, and penalty-based reward.
    """

    def __init__(
        self,
        target_intersection_id: str = "I_1_1",
        episode_length: int = 100,
        spawn_rate: float = 0.4,
        seed: Optional[int] = None,
    ):
        self.target_intersection_id: str = target_intersection_id
        self.episode_length: int = episode_length
        self.spawn_rate: float = spawn_rate
        self.seed: Optional[int] = seed

        self.state_dim: int = 4  # [queue_NS, queue_EW, phase_idx, timer_norm]
        self.action_dim: int = 2  # 0: Hold phase, 1: Switch phase

        self.engine: Optional[SimulationEngine] = None
        self.reset()

    def _build_network(self) -> TrafficGraph:
        """Constructs a standard 3x3 grid network."""
        graph = TrafficGraph()
        for r in range(3):
            for c in range(3):
                node_id = f"I_{r}_{c}"
                graph.add_intersection(Intersection(node_id, phase_duration=10))

        for r in range(3):
            for c in range(3):
                u_id = f"I_{r}_{c}"
                if c + 1 < 3:
                    v_id = f"I_{r}_{c+1}"
                    graph.add_road(Road(f"R_{u_id}->{v_id}", u_id, v_id, length=50.0, speed_limit=10.0, direction="EW"))
                    graph.add_road(Road(f"R_{v_id}->{u_id}", v_id, u_id, length=50.0, speed_limit=10.0, direction="EW"))
                if r + 1 < 3:
                    v_id = f"I_{r+1}_{c}"
                    graph.add_road(Road(f"R_{u_id}->{v_id}", u_id, v_id, length=50.0, speed_limit=10.0, direction="NS"))
                    graph.add_road(Road(f"R_{v_id}->{u_id}", v_id, u_id, length=50.0, speed_limit=10.0, direction="NS"))
        return graph

    def _get_observation(self) -> np.ndarray:
        """Constructs state feature vector for target intersection."""
        target_node = self.engine.graph.intersections[self.target_intersection_id]

        queue_ns = 0
        queue_ew = 0

        for road_id, q in target_node.lane_queues.items():
            road = self.engine.graph.roads.get(road_id)
            if road:
                if road.direction == "NS":
                    queue_ns += len(q)
                else:
                    queue_ew += len(q)

        # Enforce non-negative state validation
        queue_ns = max(0, queue_ns)
        queue_ew = max(0, queue_ew)

        phase_idx = 0.0 if target_node.signal_phase == "NORTH_SOUTH" else 1.0
        timer_norm = target_node.timer / max(1, target_node.phase_duration)

        return np.array([float(queue_ns), float(queue_ew), phase_idx, timer_norm], dtype=np.float32)

    def reset(self) -> np.ndarray:
        """Resets simulation environment and returns initial state vector."""
        graph = self._build_network()
        self.engine = SimulationEngine(graph, spawn_rate=self.spawn_rate, seed=self.seed)
        return self._get_observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Applies action (0: Hold, 1: Switch), steps simulation, and returns (next_state, reward, done, info).
        """
        target_node = self.engine.graph.intersections[self.target_intersection_id]

        # Action 1: Force switch signal phase
        if action == 1:
            target_node.toggle_phase()

        # Step engine simulation forward
        stats = self.engine.step()

        # Calculate observation, reward, and completion status
        next_state = self._get_observation()

        obs_data = target_node.observe()
        total_queue = obs_data["total_queue"]

        # Reward = -(waiting time + queue penalty)
        reward = -(float(stats["total_wait_time"]) + float(total_queue) * 2.0)

        done = self.engine.step_count >= self.episode_length

        info = {
            "step": self.engine.step_count,
            "active_vehicles": stats["active_vehicles"],
            "arrived_vehicles": stats["arrived_vehicles"],
            "avg_wait_time": stats["avg_wait_time"],
            "total_queue": total_queue,
        }

        return next_state, reward, done, info
