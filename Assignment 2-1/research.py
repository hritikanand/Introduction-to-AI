"""
Research Prototype

This script implements a basic working prototype for solving the multi-goal routing problem.
Given an origin and multiple destinations, it calculates all possible visiting sequences
and determines the total cost using Dijkstra’s algorithm to find shortest paths.

Author: Hritik
"""

import itertools
import heapq

# Sample graph (Adjacency list with costs)
graph = {
    1: [(2, 4), (3, 2)],
    2: [(3, 5), (4, 10)],
    3: [(4, 3)],
    4: [(1, 7)]
}

origin = 1
destinations = [2, 3, 4]

# Dijkstra's algorithm to compute shortest path cost
def dijkstra(start, goal, graph):
    heap = [(0, start)]
    visited = set()

    while heap:
        cost, node = heapq.heappop(heap)
        if node == goal:
            return cost
        if node in visited:
            continue
        visited.add(node)
        for neighbor, edge_cost in graph.get(node, []):
            if neighbor not in visited:
                heapq.heappush(heap, (cost + edge_cost, neighbor))
    return float('inf')  # No path found

# Calculate total cost for a sequence of destinations
def calculate_total_cost(sequence, graph, origin):
    total_cost = 0
    current = origin
    for dest in sequence:
        cost = dijkstra(current, dest, graph)
        if cost == float('inf'):
            return float('inf')  # Invalid path
        total_cost += cost
        current = dest
    return total_cost

# Generate all sequences of destinations
def generate_sequences(destinations):
    return list(itertools.permutations(destinations))

# Evaluate all sequences
def evaluate_sequences(sequences, graph, origin):
    best_sequence = None
    best_cost = float('inf')

    print("Evaluating sequences for multi-goal optimisation:\n")

    for seq in sequences:
        cost = calculate_total_cost(seq, graph, origin)
        path_str = f"{origin} -> " + " -> ".join(map(str, seq))
        if cost == float('inf'):
            print(f"Sequence: {path_str} | No valid path")
        else:
            print(f"Sequence: {path_str} | Total Cost: {cost}")
            if cost < best_cost:
                best_cost = cost
                best_sequence = seq

    if best_sequence:
        best_path = f"{origin} -> " + " -> ".join(map(str, best_sequence))
        print(f"\nBest sequence found: {best_path} | Total Cost: {best_cost}")
    else:
        print("\nNo valid sequence found.")

# Run the prototype
if __name__ == "__main__":
    sequences = generate_sequences(destinations)
    evaluate_sequences(sequences, graph, origin)
