import pytest
from simulation.models import Intersection, Road, Vehicle
from simulation.graph import TrafficGraph
from simulation.engine import SimulationEngine


def test_graph_initialization():
    """Verify graph initialization node & edge count matches input."""
    graph = TrafficGraph()
    n1 = Intersection("N1")
    n2 = Intersection("N2")
    n3 = Intersection("N3")

    graph.add_intersection(n1)
    graph.add_intersection(n2)
    graph.add_intersection(n3)

    r1 = Road("R1", "N1", "N2", length=100.0)
    r2 = Road("R2", "N2", "N3", length=100.0)

    graph.add_road(r1)
    graph.add_road(r2)

    assert graph.node_count == 3
    assert graph.edge_count == 2
    assert graph.get_shortest_path("N1", "N3") == ["N1", "N2", "N3"]


def test_invalid_node_connection_error():
    """Verify error handling when connecting non-existent nodes."""
    graph = TrafficGraph()
    graph.add_intersection(Intersection("N1"))

    # Connecting N1 to non-existent N2 should raise ValueError
    invalid_road = Road("R_ERR", "N1", "N_NONEXISTENT")
    with pytest.raises(ValueError, match="does not exist"):
        graph.add_road(invalid_road)


def test_unreachable_route_error():
    """Verify error handling for unreachable pathfinding targets."""
    graph = TrafficGraph()
    graph.add_intersection(Intersection("A"))
    graph.add_intersection(Intersection("B"))
    # No road edge added between A and B

    with pytest.raises(ValueError, match="No valid route found"):
        graph.get_shortest_path("A", "B")


def test_vehicle_spawn_and_movement():
    """Verify step() progresses simulation state and updates vehicle distance."""
    graph = TrafficGraph()
    graph.add_intersection(Intersection("A"))
    graph.add_intersection(Intersection("B"))
    graph.add_road(Road("R_AB", "A", "B", length=100.0, speed_limit=20.0))

    engine = SimulationEngine(graph, spawn_rate=0.0)
    vehicle = engine.spawn_vehicle("A", "B")

    assert vehicle is not None
    assert vehicle.origin == "A"
    assert vehicle.destination == "B"

    # Step simulation (vehicle speed 10.0 on road with limit 20.0 moves by 10.0)
    engine.step()
    assert vehicle.distance_on_road == 10.0
    assert vehicle.status == "en_route"


def test_wait_time_accumulation_at_red_light():
    """Verify vehicle wait_time increases when held at an intersection red light."""
    graph = TrafficGraph()
    # Intersection B starts in NORTH_SOUTH phase. Edge R_AB direction is EW (Red light at B)
    node_a = Intersection("A")
    node_b = Intersection("B", phase_duration=20)
    node_b.signal_phase = "NORTH_SOUTH"
    node_c = Intersection("C")

    graph.add_intersection(node_a)
    graph.add_intersection(node_b)
    graph.add_intersection(node_c)

    r1 = Road("R_AB", "A", "B", length=10.0, speed_limit=10.0, direction="EW")
    r2 = Road("R_BC", "B", "C", length=100.0, speed_limit=10.0, direction="EW")
    graph.add_road(r1)
    graph.add_road(r2)

    engine = SimulationEngine(graph, spawn_rate=0.0)
    vehicle = engine.spawn_vehicle("A", "C")
    assert vehicle is not None

    # Step 1: Vehicle reaches end of road R_AB (distance_on_road = 10.0 = length)
    engine.step()
    assert vehicle.distance_on_road == 10.0

    # Step 2: Vehicle is at end of road R_AB, facing RED light at B (direction EW vs phase NS)
    initial_wait = vehicle.wait_time
    engine.step()

    # Wait time should increment because light is RED
    assert vehicle.wait_time > initial_wait
    assert vehicle.status == "waiting"
