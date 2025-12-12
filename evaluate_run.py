"""
Evaluate generated graphs against ground truth for a specific training run.

Usage:
    python evaluate_run.py --run_id <run_folder_name> --epoch <epoch_to_evaluate> --with-ig-metrics
    
Example:
    python evaluate_run.py --run_id GraphRNN_RNN_helpdesk_4_128_2025-12-07_11-44-02 --epoch 300 --with-ig-metrics --ground-truth
"""
import argparse
import os
import pickle
import warnings

# Suppress pyemd's pkg_resources deprecation warnings
warnings.filterwarnings('ignore', message='pkg_resources is deprecated')

import numpy as np
from random import shuffle

import eval.stats
import eval.ig_metrics


def load_graphs(fname):
    """Load graphs from pickle file."""
    with open(fname, 'rb') as f:
        return pickle.load(f)


def evaluate_graphs(graph_test, graph_pred, name=""):
    """Compute MMD metrics between test and predicted graphs."""
    print(f"\n{'='*60}")
    print(f"EVALUATION: {name}")
    print(f"{'='*60}")
    print(f"Test graphs: {len(graph_test)}")
    print(f"Predicted graphs: {len(graph_pred)}")
    
    # Average sizes
    test_avg = np.mean([g.number_of_nodes() for g in graph_test])
    pred_avg = np.mean([g.number_of_nodes() for g in graph_pred])
    print(f"Test avg nodes: {test_avg:.2f}")
    print(f"Pred avg nodes: {pred_avg:.2f}")
    
    # Compute MMD metrics
    print("\nComputing MMD metrics...")
    
    mmd_degree = eval.stats.degree_stats(graph_test, graph_pred)
    print(f"  Degree MMD: {mmd_degree:.6f}")
    
    mmd_clustering = eval.stats.clustering_stats(graph_test, graph_pred)
    print(f"  Clustering MMD: {mmd_clustering:.6f}")
    
    try:
        mmd_orbits = eval.stats.orbit_stats_all(graph_test, graph_pred)
        print(f"  Orbit MMD: {mmd_orbits:.6f}")
    except Exception as e:
        print(f"  Orbit MMD: Error - {e}")
        mmd_orbits = -1
    
    return {
        'degree': mmd_degree,
        'clustering': mmd_clustering,
        'orbits': mmd_orbits
    }


def compute_internal_mmd(graphs, name="Ground Truth"):
    """Compute MMD within a set of graphs (split in half)."""
    print(f"\n--- Internal MMD ({name}) ---")
    mid = len(graphs) // 2
    shuffle(graphs)
    
    mmd_degree = eval.stats.degree_stats(graphs[:mid], graphs[mid:])
    mmd_clustering = eval.stats.clustering_stats(graphs[:mid], graphs[mid:])
    
    print(f"  Degree MMD: {mmd_degree:.6f}")
    print(f"  Clustering MMD: {mmd_clustering:.6f}")
    
    return {'degree': mmd_degree, 'clustering': mmd_clustering}


def analyze_uniqueness(graph_pred, graph_train):
    """Analyze how many generated graphs are unique vs copies from training."""
    print("\n--- Uniqueness Analysis ---")
    
    def graph_hash(g):
        """Simple hash based on sorted edge list and node count."""
        edges = tuple(sorted(g.edges()))
        return (g.number_of_nodes(), g.number_of_edges(), edges)
    
    train_hashes = set(graph_hash(g) for g in graph_train)
    pred_hashes = [graph_hash(g) for g in graph_pred]
    
    # Check for copies
    copies = sum(1 for h in pred_hashes if h in train_hashes)
    unique_pred = len(set(pred_hashes))
    
    print(f"  Generated graphs: {len(graph_pred)}")
    print(f"  Unique generated: {unique_pred} ({100*unique_pred/len(graph_pred):.1f}%)")
    print(f"  Copies from training: {copies} ({100*copies/len(graph_pred):.1f}%)")
    print(f"  Novel graphs: {len(graph_pred) - copies} ({100*(len(graph_pred)-copies)/len(graph_pred):.1f}%)")
    
    return {
        'total': len(graph_pred),
        'unique': unique_pred,
        'copies': copies,
        'novel': len(graph_pred) - copies
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate a training run')
    parser.add_argument('--run_id', type=str, required=True,
                        help='Run folder name')
    parser.add_argument('--epoch', type=int, default=None,
                        help='Specific epoch to evaluate (default: latest)')
    parser.add_argument('--sample_time', type=int, default=1,
                        help='Sample time suffix (default: 1)')
    parser.add_argument('--with-ig-metrics', action='store_true',
                        help='Compute Instance Graph metrics (Avg Generalization)')
    parser.add_argument('--ground-truth', type=str, default=None,
                        help='Path to ground truth graphs file (for Accuracy/MC metrics)')
    parser.add_argument('--mc-timeout', type=float, default=5.0,
                        help='Timeout per graph for Matching Cost computation (default: 5.0s)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("GRAPH EVALUATION")
    print("=" * 60)
    print(f"Run: {args.run_id}")
    
    # Build paths
    graphs_dir = os.path.join('./graphs/', args.run_id)
    fname_prefix = 'GraphRNN_RNN_helpdesk_4_128_'
    
    if not os.path.exists(graphs_dir):
        print(f"ERROR: Directory not found: {graphs_dir}")
        return
    
    # Find available epochs
    available_epochs = []
    for f in os.listdir(graphs_dir):
        if 'pred_' in f and f.endswith('.dat'):
            # Extract epoch: GraphRNN_RNN_helpdesk_4_128_pred_300_1.dat
            parts = f.replace('.dat', '').split('_')
            try:
                epoch_idx = parts.index('pred') + 1
                if epoch_idx < len(parts):
                    available_epochs.append(int(parts[epoch_idx]))
            except (ValueError, IndexError):
                pass
    
    available_epochs = sorted(set(available_epochs))
    print(f"Available epochs: {available_epochs}")
    
    if not available_epochs:
        print("No prediction files found!")
        return
    
    # Select epoch
    if args.epoch is None:
        epoch = available_epochs[-1]
        print(f"Using latest epoch: {epoch}")
    else:
        epoch = args.epoch
        if epoch not in available_epochs:
            print(f"Epoch {epoch} not available!")
            return
    
    # Load ground truth (training data)
    fname_train = os.path.join(graphs_dir, fname_prefix + 'train_0.dat')
    fname_test = os.path.join(graphs_dir, fname_prefix + 'test_0.dat')
    
    if os.path.exists(fname_train):
        graph_all = load_graphs(fname_train)
        print(f"Loaded {len(graph_all)} graphs from training data")
    elif os.path.exists(fname_test):
        graph_all = load_graphs(fname_test)
        print(f"Loaded {len(graph_all)} graphs from test data")
    else:
        print("ERROR: No ground truth file found!")
        return
    
    # Split into train/test (80/20)
    n = len(graph_all)
    graph_train = graph_all[:int(0.8 * n)]
    graph_test = graph_all[int(0.8 * n):]
    print(f"Train set: {len(graph_train)}, Test set: {len(graph_test)}")
    
    # Load predictions
    fname_pred = os.path.join(graphs_dir, fname_prefix + f'pred_{epoch}_{args.sample_time}.dat')
    if not os.path.exists(fname_pred):
        print(f"ERROR: Prediction file not found: {fname_pred}")
        return
    
    graph_pred = load_graphs(fname_pred)
    print(f"Loaded {len(graph_pred)} predicted graphs from epoch {epoch}")
    
    # Evaluate
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    # Internal MMD (baseline - how similar are real graphs to each other?)
    compute_internal_mmd(graph_test.copy(), "Test Set")
    
    # Main evaluation: predicted vs test
    results = evaluate_graphs(graph_test, graph_pred, f"Epoch {epoch}")
    
    # Uniqueness analysis
    uniqueness = analyze_uniqueness(graph_pred, graph_train)
    
    # Instance Graph metrics (optional)
    ig_results = None
    if args.with_ig_metrics:
        print("\n" + "=" * 60)
        print("INSTANCE GRAPH METRICS")
        print("=" * 60)
        
        # Load ground truth for Accuracy and Matching Cost
        true_graphs = None
        if args.ground_truth and os.path.exists(args.ground_truth):
            true_graphs = load_graphs(args.ground_truth)
            print(f"Loaded {len(true_graphs)} ground truth graphs from file")
        else:
            # Use test set as ground truth (sample to match prediction count)
            n_pred = len(graph_pred)
            if len(graph_test) >= n_pred:
                true_graphs = graph_test[:n_pred]
                print(f"Using {n_pred} test graphs as ground truth")
            else:
                true_graphs = graph_test
                print(f"Using all {len(graph_test)} test graphs as ground truth")
        
        ig_results = eval.ig_metrics.evaluate_instance_graphs(
            pred_graphs=graph_pred,
            true_graphs=true_graphs,
            compute_ag=True,
            mc_timeout=args.mc_timeout
        )
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Epoch: {epoch}")
    print(f"Degree MMD: {results['degree']:.6f}")
    print(f"Clustering MMD: {results['clustering']:.6f}")
    print(f"Orbits MMD: {results['orbits']:.6f}" if results['orbits'] >= 0 else "Orbits MMD: N/A")
    print(f"Novelty: {100*uniqueness['novel']/uniqueness['total']:.1f}%")
    
    if ig_results:
        ag = ig_results.get('avg_generalization', {})
        if ag.get('num_dags', 0) > 0:
            print(f"Avg Generalization: {ag.get('dag_mean', 'N/A'):.2f} (DAGs: {ag.get('num_dags')}, Cyclic: {ag.get('num_cyclic')})")
        else:
            print(f"Avg Generalization: N/A (all {ag.get('num_cyclic', 0)} graphs contain cycles)")
        if 'accuracy' in ig_results:
            acc = ig_results['accuracy']
            print(f"Accuracy: {acc['count']}/{ig_results['num_true']} ({acc['percentage']:.1f}%)")
        if 'matching_cost' in ig_results:
            mc = ig_results['matching_cost']
            print(f"Matching Cost: mean={mc['mean']:.2f}, median={mc['median']:.2f}")
    
    print("=" * 60)


if __name__ == '__main__':
    main()
