from typing import List, Dict, Any, Optional
from simulation.agent import TrafficLightAgent


class Intersection:
    """
    Represents an intersection node managing traffic signal state via TrafficLightAgent.
    """

    def __init__(self, node_id: str, phase_duration: int = 10):
        self.id: str = node_id
        self.lane_queues: Dict[str, List[str]] = {}  # road_id -> list of vehicle_ids
        self.agent: TrafficLightAgent = TrafficLightAgent(
            agent_id=f"agent_{node_id}",
            intersection_id=node_id,
            initial_state="NORTH_SOUTH",
            phase_duration=phase_duration,
        )

    @property
    def signal_phase(self) -> str:
        """Returns the current signal phase managed by the agent."""
        return self.agent.get_state()

    @signal_phase.setter
    def signal_phase(self, new_phase: str) -> None:
        """Sets the current signal phase managed by the agent."""
        self.agent.change_state(new_phase)

    @property
    def timer(self) -> int:
        return self.agent.timer

    @property
    def phase_duration(self) -> int:
        return self.agent.phase_duration

    def step_timer(self) -> bool:
        """Triggers agent step timer. Returns True if signal phase toggled."""
        return self.agent.step(self)

    def toggle_phase(self) -> None:
        """Switches traffic light signal phase via agent."""
        next_state = (
            "EAST_WEST" if self.agent.state == "NORTH_SOUTH" else "NORTH_SOUTH"
        )
        self.agent.change_state(next_state)

    def is_green(self, direction: str) -> bool:
        """Checks if green light for given movement direction ('NS' or 'EW')."""
        if direction.upper() in ("NS", "NORTH_SOUTH", "NORTH", "SOUTH"):
            return self.signal_phase == "NORTH_SOUTH"
        return self.signal_phase == "EAST_WEST"

    def observe(self) -> Dict[str, Any]:
        """Returns agent observation space data for this intersection."""
        return self.agent.observe(self)

    def add_to_queue(self, road_id: str, vehicle_id: str) -> None:
        if road_id not in self.lane_queues:
            self.lane_queues[road_id] = []
        if vehicle_id not in self.lane_queues[road_id]:
            self.lane_queues[road_id].append(vehicle_id)

    def remove_from_queue(self, road_id: str, vehicle_id: str) -> None:
        if road_id in self.lane_queues and vehicle_id in self.lane_queues[road_id]:
            self.lane_queues[road_id].remove(vehicle_id)


class Road:
    """
    Represents a directed road edge connecting two intersections.
    """

    def __init__(
        self,
        road_id: str,
        u: str,
        v: str,
        length: float = 100.0,
        capacity: int = 10,
        speed_limit: float = 10.0,
        direction: str = "NS",
    ):
        self.id: str = road_id
        self.u: str = u
        self.v: str = v
        self.length: float = length
        self.capacity: int = capacity
        self.speed_limit: float = speed_limit
        self.direction: str = direction  # "NS" or "EW"
        self.vehicles: List["Vehicle"] = []

    @property
    def is_full(self) -> bool:
        return len(self.vehicles) >= self.capacity

    @property
    def vehicle_count(self) -> int:
        return len(self.vehicles)

    def add_vehicle(self, vehicle: "Vehicle") -> bool:
        if self.is_full:
            return False
        self.vehicles.append(vehicle)
        return True

    def remove_vehicle(self, vehicle: "Vehicle") -> None:
        if vehicle in self.vehicles:
            self.vehicles.remove(vehicle)


class Vehicle:
    """
    Represents an autonomous vehicle moving through the graph network.
    """

    def __init__(
        self,
        vehicle_id: str,
        origin: str,
        destination: str,
        route: List[str],
        speed: float = 10.0,
    ):
        self.id: str = vehicle_id
        self.origin: str = origin
        self.destination: str = destination
        self.route: List[str] = route  # List of node IDs, e.g. ['A', 'B', 'C']
        self.route_index: int = 0  # Index of current origin node in route
        self.distance_on_road: float = 0.0
        self.speed: float = speed
        self.wait_time: int = 0
        self.status: str = "en_route"  # "en_route", "waiting", "arrived"

    @property
    def current_node(self) -> str:
        return self.route[self.route_index]

    @property
    def next_node(self) -> Optional[str]:
        if self.route_index + 1 < len(self.route):
            return self.route[self.route_index + 1]
        return None

    def increment_wait(self) -> None:
        self.wait_time += 1
        self.status = "waiting"
