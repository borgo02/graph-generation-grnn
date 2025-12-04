import networkx as nx
import pickle
import numpy as np

def has_transitive_edges(G):
    """
    Checks if a graph has transitive edges.
    A transitive edge is an edge (u, v) where there is also a path u -> ... -> v of length > 1.
    Since GraphRNN generates undirected graphs, we check for triangles in a specific way or just general redundancy?
    User example: 0->1, 1->2, 0->2. This is a triangle.
    In a DAG context, 0->2 is transitive.
    
    Let's check for triangles in the undirected graph as a proxy for this "redundancy".
    """
    # Check for triangles
    triangles = nx.triangles(G)
    num_triangles = sum(triangles.values()) // 3
    return num_triangles > 0

def load_graphs(filename):
    print('Loading graph dataset from file: ' + filename)
    graphs = []
    with open(filename, 'r') as f:
        content = f.read()

    graph_blocks = content.split('XP')
    
    for block in graph_blocks:
        block = block.strip()
        if not block:
            continue
            
        G = nx.Graph()
        lines = block.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if parts[0] == 'v':
                node_id = int(float(parts[1]))
                label = parts[2]
                G.add_node(node_id, label=label)
            elif parts[0] == 'e':
                u = int(float(parts[1]))
                v = int(float(parts[2]))
                G.add_edge(u, v)
        
        if G.number_of_nodes() > 0:
            G = nx.convert_node_labels_to_integers(G, first_label=0)
            graphs.append(G)
    return graphs

def check_dataset():
    # Load dataset
    graphs = load_graphs('dataset_v2/Helpdesk_igs_complete.g')
    print(f"Loaded {len(graphs)} graphs.")
    
    count = 0
    for i, G in enumerate(graphs):
        if has_transitive_edges(G):
            count += 1
            if count <= 5:
                print(f"Graph {i} has triangles (potential transitive edges).")
                print(G.edges())
    
    print(f"Total graphs with triangles: {count} out of {len(graphs)} ({count/len(graphs)*100:.2f}%)")

if __name__ == '__main__':
    check_dataset()
