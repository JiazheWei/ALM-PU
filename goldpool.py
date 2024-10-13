import igraph as ig

# Sample data - This represents connections between community members
edges = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 5), (3, 4), (3, 6), (4, 5), (5, 6)] 

# Create the igraph graph object
graph = ig.Graph(edges=edges)

# Apply the layout_auto function
layout = graph.layout_auto()

# Get the coordinates for each node in the layout
x_coords = [layout[i][0] for i in range(len(graph.vs))]
y_coords = [layout[i][1] for i in range(len(graph.vs))]

# Print the coordinates for verification (you can use these to plot the graph)
print(f"X-coordinates: {x_coords}")
print(f"Y-coordinates: {y_coords}")