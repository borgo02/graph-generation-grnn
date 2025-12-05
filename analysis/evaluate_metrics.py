import networkx as nx
import numpy as np
import pickle
import argparse
import os
import glob
from collections import defaultdict

def load_graphs(path, file_type='dat'):
    graphs = []
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, f"*.{file_type}"))
        for f in files:
            graphs.extend(load_graph_file(f, file_type))
    else:
        graphs = load_graph_file(path, file_type)
    return graphs

def load_graph_file(fname, file_type):
    if file_type == 'dat':
        with open(fname, "rb") as f:
            return pickle.load(f)
    elif file_type == 'g':
        # Custom loader for .g files (reused from check_transitive.py)
        graphs = []
        with open(fname, 'r') as f:
            content = f.read()
        blocks = content.split('XP')
        for block in blocks:
            block = block.strip()
            if not block: continue
            G = nx.Graph()
            lines = block.split('\n')
            for line in lines:
                parts = line.strip().split()
                if not parts: continue
                if parts[0] == 'v':
                    G.add_node(int(float(parts[1])), label=parts[2])
                elif parts[0] == 'e':
                    G.add_edge(int(float(parts[1])), int(float(parts[2])))
            if G.number_of_nodes() > 0:
                G = nx.convert_node_labels_to_integers(G, first_label=0)
                graphs.append(G)
        return graphs
    return []

def compute_metrics(graphs):
    """
    Computes Uniqueness and Variability for a list of graphs.
    """
    if not graphs:
        return 0.0, 0.0
    
    # 1. Uniqueness
    # Count non-isomorphic graphs
    unique_graphs = []
    for G in graphs:
        is_unique = True
        for G_unique in unique_graphs:
            if nx.is_isomorphic(G, G_unique, node_match=lambda n1, n2: n1.get('label') == n2.get('label')):
                is_unique = False
                break
        if is_unique:
            unique_graphs.append(G)
            
    uniqueness = len(unique_graphs) / len(graphs)
    
    # 2. Variability (Diversity)
    # Average pairwise Graph Edit Distance (GED)
    # GED is expensive, so we use a sample if N is large
    N = len(graphs)
    if N > 50:
        indices = np.random.choice(N, 50, replace=False)
        sample_graphs = [graphs[i] for i in indices]
    else:
        sample_graphs = graphs
        
    ged_sum = 0
    count = 0
    for i in range(len(sample_graphs)):
        for j in range(i + 1, len(sample_graphs)):
            # Use optimized GED or approximation if available
            # For small graphs, exact GED is feasible
            dist = nx.graph_edit_distance(sample_graphs[i], sample_graphs[j], 
                                          node_match=lambda n1, n2: n1.get('label') == n2.get('label'))
            ged_sum += dist
            count += 1
            
    variability = ged_sum / count if count > 0 else 0.0
    
    return uniqueness, variability

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path', type=str, required=True, help='Path to generated graphs (.dat)')
    parser.add_argument('--real_path', type=str, required=True, help='Path to real graphs (.g)')
    args = parser.parse_args()
    
    print(f"Loading generated graphs from {args.pred_path}...")
    gen_graphs = load_graphs(args.pred_path, 'dat')
    print(f"Loaded {len(gen_graphs)} generated graphs.")
    
    print(f"Loading real graphs from {args.real_path}...")
    real_graphs = load_graphs(args.real_path, 'g')
    print(f"Loaded {len(real_graphs)} real graphs.")
    
    # Group by length
    gen_by_len = defaultdict(list)
    for G in gen_graphs:
        gen_by_len[G.number_of_nodes()].append(G)
        
    real_by_len = defaultdict(list)
    for G in real_graphs:
        real_by_len[G.number_of_nodes()].append(G)
        
    # Compare per length
    all_lengths = sorted(set(gen_by_len.keys()) | set(real_by_len.keys()))
    
    print("\n" + "="*60)
    print(f"{'Length':<10} | {'Type':<10} | {'Count':<10} | {'Uniqueness':<10} | {'Variability (GED)':<15}")
    print("="*60)
    
    for length in all_lengths:
        # Real
        real_g = real_by_len[length]
        if real_g:
            u_real, v_real = compute_metrics(real_g)
            print(f"{length:<10} | {'Real':<10} | {len(real_g):<10} | {u_real:.4f}     | {v_real:.4f}")
        
        # Generated
        gen_g = gen_by_len[length]
        if gen_g:
            u_gen, v_gen = compute_metrics(gen_g)
            print(f"{length:<10} | {'Gen':<10}  | {len(gen_g):<10} | {u_gen:.4f}     | {v_gen:.4f}")
            
        if real_g or gen_g:
            print("-" * 60)

if __name__ == '__main__':
    main()
