import heapq
import sys

class Graph:

    def __init__(self, nodes, edges, origin, destinations):
        self.nodes = nodes  
        self.edges = edges   
        self.origin = origin   
        self.destinations = set(destinations)  


    def uniform_cost_search(self):  
        priority_queue = []  
        heapq.heappush(priority_queue, (0, self.origin, []))   
        visited = set()
        nodes_expanded = 0 


        while priority_queue:
            cost, current, path = heapq.heappop(priority_queue)

            if current in visited:
                continue 

            visited.add(current)
            nodes_expanded += 1
            path = path + [current]
 
            if current in self.destinations:
                return current, nodes_expanded, path  

            for (start, end), edge_cost in self.edges.items():
                if start == current and end not in visited:
                    heapq.heappush(priority_queue, (cost + edge_cost, end, path))

        return None, nodes_expanded, []   

def parse_file(filename):
    nodes = {}
    edges = {}
    origin = None
    destinations = []

    with open(filename, "r") as file:
        lines = file.readlines()

    section = None
    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("Nodes:"):
            section = "nodes"
            continue
        elif line.startswith("Edges:"):
            section = "edges"
            continue
        elif line.startswith("Origin:"):
            section = "origin"
            continue
        elif line.startswith("Destinations:"):
            section = "destinations"
            continue

        if section == "nodes":
            node_id, coords = line.split(":")
            node_id = int(node_id.strip())
            x, y = map(int, coords.strip(" ()").split(","))  
            nodes[node_id] = (x, y)

        elif section == "edges":
            edge_info, cost = line.split(":")
            start, end = map(int, edge_info.strip("()").split(","))
            cost = int(cost.strip())
            edges[(start, end)] = cost

        elif section == "origin":
            origin = int(line.strip())

        elif section == "destinations":
            destinations = list(map(int, line.strip().split(";")))

    return Graph(nodes, edges, origin, destinations)   

def main():
    if len(sys.argv) != 3:
        print("Usage: python search.py <filename> <method>")
        return

    filename = sys.argv[1]
    method = sys.argv[2]

    graph = parse_file(filename)

    if method == "CUS1":
        goal, nodes_expanded, path = graph.uniform_cost_search()
        if goal is not None:
            print(f"{filename} {method}")
            print(f"{goal} {nodes_expanded}")
            print(" -> ".join(map(str, path)))
        else:
            
            print("No path found.")
    else:
        print(f"Error: Unknown method '{method}'")

if __name__ == "__main__": 
    main()
