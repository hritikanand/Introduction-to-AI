# Group Project - COS30019 Intro to AI Assignment 2
# Algorithms included: DFS, BFS, GBFS, Astar, CUS1 (UCS), CUS2 (Dijkstra)

import sys
import heapq
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import networkx as nx

# Main program start
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python searches.py <filename> <method>")
    else:
        input_file = sys.argv[1]
        method = sys.argv[2]
        # Run the search function later after everything is defined
        # (We have called this at the bottom of the script!)

# Graph class to store nodes, edges, origin and goals
class Graph:
    def __init__(self):
        self.nodes = {}  
        self.edges = defaultdict(list)
        self.origin = None
        self.destinations = set()

    def load_from_file(self, filename):
        # Load graph details from text file
        section = None
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if line == "Nodes:":
                    section = "nodes"
                elif line == "Edges:":
                    section = "edges"
                elif line == "Origin:":
                    section = "origin"
                elif line == "Destinations:":
                    section = "destinations"
                else:
                    if section == "nodes":
                        node_id, coords = line.split(":")
                        node_id = int(node_id.strip())
                        x, y = map(int, coords.strip(" ()").split(","))
                        self.nodes[node_id] = (x, y)

                    elif section == "edges":
                        edge_info, cost = line.split(":")
                        start, end = map(int, edge_info.strip("()").split(","))
                        cost = int(cost.strip())
                        self.edges[start].append((end, cost))

                    elif section == "origin":
                        self.origin = int(line.strip())

                    elif section == "destinations":
                        self.destinations = set(map(int, line.split(";")))

    def heuristic(self, node, goal):
        # Euclidean distance as heuristic
        x1, y1 = self.nodes[node]
        x2, y2 = self.nodes[goal]
        return ((x1 - x2)**2 + (y1 - y2)**2)**0.5

# DFS function
def dfs(graph):
    stack = [(graph.origin, [graph.origin])]
    visited = set()

    while stack:
        current, path = stack.pop()

        if current in graph.destinations:
            return current, len(path), path

        if current not in visited:
            visited.add(current)
            neighbors = sorted(graph.edges.get(current, []))
            for neighbor, _ in reversed(neighbors):
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor]))

    return None, 0, []

# BFS function
def bfs(graph):
    queue = deque([(graph.origin, [graph.origin])])
    visited = set()

    while queue:
        current, path = queue.popleft()

        if current in graph.destinations:
            return current, len(path), path

        if current not in visited:
            visited.add(current)
            for neighbor, _ in sorted(graph.edges.get(current, [])):
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))

    return None, 0, []

# GBFS function
def gbfs(graph):
    h = graph.heuristic
    goal = min(graph.destinations)
    heap = [(h(graph.origin, goal), graph.origin, [graph.origin])]
    visited = set()
    nodes_expanded = 0

    while heap:
        _, current, path = heapq.heappop(heap)
        nodes_expanded += 1

        if current in graph.destinations:
            return current, nodes_expanded, path

        if current not in visited:
            visited.add(current)
            for neighbor, _ in sorted(graph.edges[current]):
                if neighbor not in visited:
                    heapq.heappush(heap, (h(neighbor, goal), neighbor, path + [neighbor]))

    return None, nodes_expanded, []

# A* function
def astar(graph):
    h = graph.heuristic
    goal = min(graph.destinations)
    heap = [(0, graph.origin, [graph.origin], 0)]
    visited = set()
    nodes_expanded = 0

    while heap:
        _, current, path, g_cost = heapq.heappop(heap)
        nodes_expanded += 1

        if current in graph.destinations:
            return current, nodes_expanded, path

        if current not in visited:
            visited.add(current)
            for neighbor, cost in sorted(graph.edges[current]):
                if neighbor not in visited:
                    new_g = g_cost + cost
                    f = new_g + h(neighbor, goal)
                    heapq.heappush(heap, (f, neighbor, path + [neighbor], new_g))

    return None, nodes_expanded, []

# CUS1 - Uniform Cost Search
def cus1(graph):
    heap = [(0, graph.origin, [graph.origin])]
    visited = set()
    nodes_expanded = 0

    while heap:
        cost, current, path = heapq.heappop(heap)
        nodes_expanded += 1

        if current in graph.destinations:
            return current, nodes_expanded, path

        if current not in visited:
            visited.add(current)
            for neighbor, edge_cost in sorted(graph.edges[current]):
                if neighbor not in visited:
                    heapq.heappush(heap, (cost + edge_cost, neighbor, path + [neighbor]))

    return None, nodes_expanded, []

# CUS2 - Dijkstra
def cus2(graph):
    heap = [(0, graph.origin, [graph.origin])]
    visited = set()
    nodes_expanded = 0

    while heap:
        cost, current, path = heapq.heappop(heap)
        nodes_expanded += 1

        if current in graph.destinations:
            return current, nodes_expanded, path

        if current not in visited:
            visited.add(current)
            for neighbor, edge_cost in sorted(graph.edges[current]):
                if neighbor not in visited:
                    heapq.heappush(heap, (cost + edge_cost, neighbor, path + [neighbor]))

    return None, nodes_expanded, []

# Visualisation of the graph
def show_graph(graph, path, method):
    G = nx.DiGraph()

    # Add edges and weights
    for node in graph.edges:
        for neighbor, cost in graph.edges[node]:
            G.add_edge(node, neighbor, weight=cost)

    pos = {node: graph.nodes[node] for node in graph.nodes}

    plt.figure(figsize=(8, 6))
    fig = plt.gcf()
    fig.canvas.manager.set_window_title(f"{method} Visualisation")

    nx.draw(G, pos, with_labels=True, node_size=700, node_color='lightgrey',
            arrows=True, edge_color='grey')
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    if path:
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2)
        nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='orange')

        path_text = " > ".join(map(str, path))
        plt.text(0.5, -0.1, f"Path: {path_text}", horizontalalignment='center',
                 verticalalignment='center', transform=plt.gca().transAxes, fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.5))

    plt.title(f"{method} Path Visualisation", fontsize=14, color='darkblue')
    plt.text(0.5, 1.02, f"Search Method: {method}", horizontalalignment='center',
             verticalalignment='center', transform=plt.gca().transAxes, fontsize=11, color='green')

    plt.axis('off')
    plt.show()

# Running the chosen search algorithm
def run_search(graph_file, method):
    graph = Graph()
    graph.load_from_file(graph_file)

    algorithms = {
        "DFS": dfs,
        "BFS": bfs,
        "GBFS": gbfs,
        "Astar": astar,
        "CUS1": cus1,
        "CUS2": cus2
    }

    if method not in algorithms:
        print(f"Error: Unknown method '{method}'")
        return

    goal, expanded, path = algorithms[method](graph)

    print(f"{graph_file} {method}")
    if goal is None:
        print("No valid path found.")
        return

    print(f"{goal} {expanded}")
    print(" > ".join(map(str, path)))

    # Draw the graph with path
    show_graph(graph, path, method)

# Main program call at the end
if __name__ == "__main__":
    run_search(input_file, method)
