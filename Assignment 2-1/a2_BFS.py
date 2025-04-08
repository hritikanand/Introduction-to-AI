import sys
from collections import defaultdict, deque

if len(sys.argv) != 2:
    print("Usage: python script.py <filename>")
    sys.exit(1)

filename = sys.argv[1]

try:
    with open(filename, 'r') as file:
        lines = file.readlines()
except FileNotFoundError:
    print(f"Error: File '{filename}' not found.")
    sys.exit(1)

graph = defaultdict(list)
nodes = {}
origin = None
destinations = set()

readnodes = False
readedges = False
readorigin = False
readdestinations = False

for line in lines:
    line = line.strip()
    if line == "Nodes:":
        readnodes = True
        continue
    elif line == "Edges:":
        readnodes = False
        readedges = True
        continue
    elif line == "Origin:":
        readedges = False
        readorigin = True
        continue
    elif line == "Destinations:":
        readorigin = False
        readdestinations = True
        continue
    
    if readnodes and line:
        node, coord = line.split(": ")
        node = int(node)
        x, y = map(int, coord.strip('()').split(','))
        nodes[node] = (x, y)
    
    if readedges and line:
        edge, weight = line.split(": ")
        start, end = map(int, edge.strip('()').split(','))
        weight = int(weight)
        graph[start].append((end, weight))
    
    if readorigin and line:
        origin = int(line)
    
    if readdestinations and line:
        destinations.update(int(d) for d in line.split(';'))

def bfs_path_finder(graph, start, goals):
    queue = deque([(start, [start])])
    nodesvisited = set()
    
    while queue:
        currentnode, path = queue.popleft()
        
        if currentnode in goals:
            return path, currentnode
        
        if currentnode not in nodesvisited:
            nodesvisited.add(currentnode)
            for neighbour, _ in graph.get(currentnode, []):
                if neighbour not in nodesvisited:
                    new_path = path + [neighbour]
                    queue.append((neighbour, new_path))
    
    return None, None

path, goal = bfs_path_finder(graph, origin, destinations)

if path:
    print(f"{filename} BFS")
    print(f"{goal}, {len(path)}")
    print(" > ".join(map(str, path)))
else:
    print(f"{filename} BFS")
    print("No valid path found.")
