# Informed Search Algorithm Visualizer
# This script loads a graph from a text file and runs three informed search algorithms:
# Greedy Best-First Search (GBFS), A* Search, and Dijkstra's Algorithm (used as CUS2).
# It prints detailed results and visualizes the path for each algorithm using matplotlib and networkx.

import heapq
import matplotlib.pyplot as plt
import networkx as nx

# Graph class to load nodes, edges, origin, and destination from a formatted text file
class Graph:
    def __init__(self):
        self.nodes = {}           # Node coordinates: {node_id: (x, y)}
        self.edges = {}           # Graph edges: {node_id: [(neighbor, cost)]}
        self.origin = None        # Starting node
        self.destinations = set() # Goal nodes

    def add_node(self, node_id, coordinates):
        self.nodes[node_id] = coordinates
        self.edges[node_id] = []

    def add_edge(self, node1, node2, cost):
        self.edges[node1].append((node2, cost))

    def load_from_file(self, filename):
        # Reads nodes, edges, origin, and destinations from a text file
        with open(filename, 'r') as file:
            section = None
            for line in file:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("Nodes:"):
                    section = "nodes"
                elif line.startswith("Edges:"):
                    section = "edges"
                elif line.startswith("Origin:"):
                    parts = line.split(":")
                    if len(parts) > 1 and parts[1].strip():
                        self.origin = int(parts[1].strip())
                    else:
                        section = "origin"
                elif line.startswith("Destinations:"):
                    parts = line.split(":")
                    if len(parts) > 1 and parts[1].strip():
                        self.destinations = set(map(int, parts[1].strip().split(";")))
                    else:
                        section = "destinations"
                else:
                    if section == "nodes":
                        node_id, coords = line.split(":")
                        x, y = map(int, coords.strip().strip("() ").split(","))
                        self.add_node(int(node_id), (x, y))
                    elif section == "edges":
                        nodes, cost = line.split(":")
                        node1, node2 = map(int, nodes.strip("() ").split(","))
                        self.add_edge(node1, node2, int(cost))
                    elif section == "origin":
                        self.origin = int(line)
                        section = None
                    elif section == "destinations":
                        self.destinations = set(map(int, line.split(";")))
                        section = None

    def heuristic(self, node, goal):
        # Euclidean distance heuristic used in GBFS and A*
        x1, y1 = self.nodes[node]
        x2, y2 = self.nodes[goal]
        return ((x1 - x2)**2 + (y1 - y2)**2) ** 0.5

# Greedy Best-First Search (uses only heuristic to pick next best node)
def greedy_best_first_search(graph):
    h = graph.heuristic
    dest = min(graph.destinations)
    pq = [(h(graph.origin, dest), graph.origin, [graph.origin])]
    visited = set()
    nodes_expanded = 0
    while pq:
        _, node, path = heapq.heappop(pq)
        nodes_expanded += 1
        if node in graph.destinations:
            return node, nodes_expanded, path
        if node not in visited:
            visited.add(node)
            for neighbor, _ in sorted(graph.edges[node]):
                if neighbor not in visited:
                    heapq.heappush(pq, (h(neighbor, dest), neighbor, path + [neighbor]))
    return None, nodes_expanded, []

# A* Search (uses cost + heuristic for optimal pathfinding)
def a_star_search(graph):
    h = graph.heuristic
    dest = min(graph.destinations)
    pq = [(0, graph.origin, [graph.origin], 0)]  # f, node, path, g
    visited = set()
    nodes_expanded = 0
    while pq:
        _, node, path, g = heapq.heappop(pq)
        nodes_expanded += 1
        if node in graph.destinations:
            return node, nodes_expanded, path
        if node not in visited:
            visited.add(node)
            for neighbor, cost in sorted(graph.edges[node]):
                if neighbor not in visited:
                    g_new = g + cost
                    f = g_new + h(neighbor, dest)
                    heapq.heappush(pq, (f, neighbor, path + [neighbor], g_new))
    return None, nodes_expanded, []

# Dijkstra's Algorithm (used as CUS2, considers cost only)
def custom_informed_search(graph):
    pq = [(0, graph.origin, [graph.origin])]
    visited = set()
    nodes_expanded = 0
    while pq:
        cost, node, path = heapq.heappop(pq)
        nodes_expanded += 1
        if node in graph.destinations:
            return node, nodes_expanded, path
        if node not in visited:
            visited.add(node)
            for neighbor, edge_cost in sorted(graph.edges[node]):
                if neighbor not in visited:
                    heapq.heappush(pq, (cost + edge_cost, neighbor, path + [neighbor]))
    return None, nodes_expanded, []

# Function to display the graph and highlight the path visually
def visualize_path(path, graph, title):
    G = nx.DiGraph()
    for node in graph.edges:
        for neighbor, cost in graph.edges[node]:
            G.add_edge(node, neighbor, weight=cost)

    pos = nx.spring_layout(G, seed=42)  # Layout for consistent node positions
    nx.draw(G, pos, with_labels=True, node_color='lightgray', edge_color='gray', node_size=800, arrows=True)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    path_edges = list(zip(path, path[1:]))
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2)
    nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='orange')
    plt.title(title)
    plt.pause(3)  # Display graph for 3 seconds then auto-close
    plt.close()

# Runs one search algorithm and prints results with visualization
def run_and_display(name, search_function, graph_file):
    graph = Graph()
    graph.load_from_file(graph_file)
    goal, expanded, path_taken = search_function(graph)

    print(f"\n{graph_file} {name}")
    if goal is None:
        print(f"No path found using {name}")
        return

    print(f"{goal} {expanded}")  # Final node reached and number of expanded nodes

    # Show each step in the path with its travel cost
    detailed_path = ""
    total_cost = 0
    for i in range(len(path_taken) - 1):
        a, b = path_taken[i], path_taken[i + 1]
        for neighbor, cost in graph.edges[a]:
            if neighbor == b:
                detailed_path += f"{a} →{b}(cost: {cost}) "
                total_cost += cost
                break

    print(detailed_path.strip())
    print(f"Total Path Cost: {total_cost}")

    # Show visual graph for the current algorithm
    visualize_path(path_taken, graph, f"{name} Path Visualization")

# Main function to run all three searches and display their results
def main():
    graph_file = "PathFinder-test.txt"
    print("======= Informed Search Algorithm Comparison =======")
    run_and_display("GBFS", greedy_best_first_search, graph_file)
    run_and_display("A*", a_star_search, graph_file)
    run_and_display("CUS2", custom_informed_search, graph_file)

if __name__ == "__main__":
    main()
