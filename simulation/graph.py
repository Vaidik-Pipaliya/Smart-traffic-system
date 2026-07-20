from typing import Dict, List, Optional
import networkx as nx
from simulation.models import Intersection, Road


class TrafficGraph:
    """
    Graph manager for the traffic network built on networkx.DiGraph.
    Nodes represent Intersections; Edges represent directed Roads.
    """

    def __init__(self):
        self.graph: nx.DiGraph = nx.DiGraph()
        self.intersections: Dict[str, Intersection] = {}
        self.roads: Dict[str, Road] = {}  # road_id -> Road
        self._edge_to_road: Dict[tuple, str] = {}  # (u, v) -> road_id

    def add_intersection(self, intersection: Intersection) -> None:
        """Adds an intersection node to the network."""
        self.intersections[intersection.id] = intersection
        self.graph.add_node(intersection.id, data=intersection)

    def add_road(self, road: Road) -> None:
        """
        Adds a directed road edge connecting origin u to destination v.
        Raises ValueError if origin or destination nodes do not exist.
        """
        if road.u not in self.intersections:
            raise ValueError(f"Origin node '{road.u}' does not exist in graph.")
        if road.v not in self.intersections:
            raise ValueError(f"Destination node '{road.v}' does not exist in graph.")

        self.roads[road.id] = road
        self._edge_to_road[(road.u, road.v)] = road.id
        self.graph.add_edge(
            road.u,
            road.v,
            weight=road.length / road.speed_limit,
            data=road,
        )

    def get_road(self, u: str, v: str) -> Optional[Road]:
        """Returns the Road object between node u and node v if it exists."""
        road_id = self._edge_to_road.get((u, v))
        if road_id:
            return self.roads.get(road_id)
        return None

    def get_shortest_path(self, origin: str, destination: str) -> List[str]:
        """
        Computes the shortest path route (list of node IDs) between origin and destination.
        Raises ValueError if nodes are invalid or no path exists.
        """
        if origin not in self.intersections:
            raise ValueError(f"Origin node '{origin}' does not exist in graph.")
        if destination not in self.intersections:
            raise ValueError(f"Destination node '{destination}' does not exist in graph.")

        try:
            path = nx.shortest_path(self.graph, source=origin, target=destination, weight="weight")
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound) as err:
            raise ValueError(f"No valid route found between '{origin}' and '{destination}': {err}")

    @property
    def node_count(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self.graph.number_of_edges()
