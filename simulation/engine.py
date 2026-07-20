import random
from typing import Dict, List, Optional
from simulation.graph import TrafficGraph
from simulation.models import Vehicle, Intersection, Road


class SimulationEngine:
    """
    Core simulation engine managing vehicle dynamics, signal timing, and step iteration.
    """

    def __init__(
        self,
        traffic_graph: TrafficGraph,
        spawn_rate: float = 0.3,
        seed: Optional[int] = None,
    ):
        self.graph: TrafficGraph = traffic_graph
        self.spawn_rate: float = spawn_rate
        self.active_vehicles: Dict[str, Vehicle] = {}
        self.arrived_vehicles: List[Vehicle] = []
        self.step_count: int = 0
        self.vehicle_counter: int = 0

        if seed is not None:
            random.seed(seed)

    def spawn_vehicle(
        self, origin: Optional[str] = None, destination: Optional[str] = None
    ) -> Optional[Vehicle]:
        """Spawns a new vehicle with shortest path routing."""
        nodes = list(self.graph.intersections.keys())
        if len(nodes) < 2:
            return None

        if origin is None or destination is None:
            origin, destination = random.sample(nodes, 2)

        try:
            route = self.graph.get_shortest_path(origin, destination)
        except ValueError:
            return None

        self.vehicle_counter += 1
        vehicle_id = f"veh_{self.vehicle_counter}"
        vehicle = Vehicle(vehicle_id, origin, destination, route)

        # Place vehicle on first road segment if space available
        first_u, first_v = route[0], route[1]
        road = self.graph.get_road(first_u, first_v)
        if road and not road.is_full:
            road.add_vehicle(vehicle)
            self.active_vehicles[vehicle_id] = vehicle
            return vehicle
        return None

    def step(self) -> Dict[str, any]:
        """
        Advances the simulation by one discrete time step.
        """
        self.step_count += 1

        # 1. Update intersection signal timers
        for intersection in self.graph.intersections.values():
            intersection.step_timer()

        # 2. Attempt vehicle spawn based on spawn_rate
        if random.random() < self.spawn_rate:
            self.spawn_vehicle()

        # 3. Process vehicle movements
        # Copy values to safely modify active_vehicles during iteration
        for vehicle in list(self.active_vehicles.values()):
            if vehicle.status == "arrived":
                continue

            u = vehicle.route[vehicle.route_index]
            v = vehicle.route[vehicle.route_index + 1]
            current_road = self.graph.get_road(u, v)

            if not current_road:
                continue

            # Check if vehicle has reached end of road (approaching intersection v)
            if vehicle.distance_on_road >= current_road.length:
                target_intersection = self.graph.intersections.get(v)

                # Case A: Vehicle reached final destination
                if v == vehicle.destination:
                    current_road.remove_vehicle(vehicle)
                    vehicle.status = "arrived"
                    if target_intersection:
                        target_intersection.remove_from_queue(current_road.id, vehicle.id)
                    self.active_vehicles.pop(vehicle.id, None)
                    self.arrived_vehicles.append(vehicle)
                    continue

                # Case B: Intermediate intersection; attempt move to next road segment (v -> w)
                next_w = vehicle.route[vehicle.route_index + 2]
                next_road = self.graph.get_road(v, next_w)

                is_green = (
                    target_intersection.is_green(current_road.direction)
                    if target_intersection
                    else True
                )
                can_enter_next_road = next_road and not next_road.is_full

                if is_green and can_enter_next_road:
                    # Clear from current queue & road, transition to next road
                    if target_intersection:
                        target_intersection.remove_from_queue(current_road.id, vehicle.id)
                    current_road.remove_vehicle(vehicle)
                    next_road.add_vehicle(vehicle)

                    vehicle.route_index += 1
                    vehicle.distance_on_road = 0.0
                    vehicle.status = "en_route"
                else:
                    # Stopped at red light or congestion queue
                    vehicle.increment_wait()
                    if target_intersection:
                        target_intersection.add_to_queue(current_road.id, vehicle.id)
            else:
                # Vehicle advances along current road segment
                move_speed = min(vehicle.speed, current_road.speed_limit)
                vehicle.distance_on_road += move_speed
                if vehicle.distance_on_road > current_road.length:
                    vehicle.distance_on_road = current_road.length
                vehicle.status = "en_route"

        return self.get_stats()

    def get_stats(self) -> Dict[str, any]:
        """Returns statistical metrics for current simulation state."""
        active_count = len(self.active_vehicles)
        arrived_count = len(self.arrived_vehicles)
        total_wait = sum(v.wait_time for v in self.active_vehicles.values()) + sum(
            v.wait_time for v in self.arrived_vehicles
        )
        total_vehicles = active_count + arrived_count
        avg_wait = total_wait / total_vehicles if total_vehicles > 0 else 0.0

        return {
            "step": self.step_count,
            "active_vehicles": active_count,
            "arrived_vehicles": arrived_count,
            "total_wait_time": total_wait,
            "avg_wait_time": round(avg_wait, 2),
        }
