from simulation.models import Intersection, Road
from simulation.graph import TrafficGraph
from simulation.engine import SimulationEngine


def build_sample_grid() -> TrafficGraph:
    """Builds a 3x3 grid network of intersections and roads."""
    graph = TrafficGraph()

    # 1. Add 9 Intersections (3x3 grid)
    for r in range(3):
        for c in range(3):
            node_id = f"I_{r}_{c}"
            graph.add_intersection(Intersection(node_id, phase_duration=5))

    # 2. Add Horizontal (EW) and Vertical (NS) bidirectional Roads
    for r in range(3):
        for c in range(3):
            u_id = f"I_{r}_{c}"

            # Horizontal edge (East)
            if c + 1 < 3:
                v_id = f"I_{r}_{c+1}"
                r1 = Road(f"R_{u_id}->{v_id}", u_id, v_id, length=50.0, speed_limit=10.0, direction="EW")
                r2 = Road(f"R_{v_id}->{u_id}", v_id, u_id, length=50.0, speed_limit=10.0, direction="EW")
                graph.add_road(r1)
                graph.add_road(r2)

            # Vertical edge (South)
            if r + 1 < 3:
                v_id = f"I_{r+1}_{c}"
                r1 = Road(f"R_{u_id}->{v_id}", u_id, v_id, length=50.0, speed_limit=10.0, direction="NS")
                r2 = Road(f"R_{v_id}->{u_id}", v_id, u_id, length=50.0, speed_limit=10.0, direction="NS")
                graph.add_road(r1)
                graph.add_road(r2)

    return graph


def run_simulation(steps: int = 100):
    print("=" * 60)
    print("      SMART TRAFFIC SYSTEM - BASE SIMULATION ENGINE DEMO      ")
    print("=" * 60)

    graph = build_sample_grid()
    print(f"Network Initialized: {graph.node_count} Intersections, {graph.edge_count} Roads")

    engine = SimulationEngine(graph, spawn_rate=0.5, seed=42)

    target_intersection_id = "I_1_1"

    for s in range(1, steps + 1):
        stats = engine.step()

        if s == 1 or s % 20 == 0 or s == steps:
            target_node = graph.intersections[target_intersection_id]
            obs = target_node.observe()

            print(
                f"[Step {s:03d}] Active Vehicles: {stats['active_vehicles']:2d} | "
                f"Arrived: {stats['arrived_vehicles']:2d} | "
                f"Avg Wait: {stats['avg_wait_time']:5.2f}s | "
                f"Agent ({obs['agent_id']}): State={obs['state']}, Queue={obs['total_queue']}"
            )

    print("=" * 60)
    final_stats = engine.get_stats()
    print("Simulation Complete!")
    print(f"Total Steps: {final_stats['step']}")
    print(f"Active Vehicles Remaining: {final_stats['active_vehicles']}")
    print(f"Total Vehicles Arrived: {final_stats['arrived_vehicles']}")
    print(f"Final Average Wait Time: {final_stats['avg_wait_time']} steps")
    print("=" * 60)


if __name__ == "__main__":
    run_simulation(100)
