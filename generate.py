"""
Generate graphs from a trained checkpoint.
This script loads a trained model and generates graphs without training.

Usage:
    python generate.py --run_id <run_folder_name> --epoch <checkpoint_epoch> --num_graphs <number_to_generate>
    
Example:
    python generate.py --run_id GraphRNN_RNN_helpdesk_4_128_2025-12-06_19-58-27 --epoch 450 --num_graphs 10
"""
import argparse
import os
import pickle
import torch
import torch.nn as nn
import yaml
import networkx as nx
from model import GRU_plain, MLP_plain
from train import test_rnn_epoch
from utils import save_graph_list, draw_graph_list


def load_label_mapping(graphs_dir, fname_prefix):
    """Load label mapping from saved ground truth graphs."""
    
    # Look for train or test .dat file to extract labels
    train_file = os.path.join(graphs_dir, fname_prefix + 'train_0.dat')
    test_file = os.path.join(graphs_dir, fname_prefix + 'test_0.dat')
    
    data_file = None
    if os.path.exists(train_file):
        data_file = train_file
    elif os.path.exists(test_file):
        data_file = test_file
    
    if data_file is None:
        print(f"  ⚠ No ground truth graphs found to extract labels from")
        return None
    
    print(f"  Loading labels from: {data_file}")
    
    with open(data_file, 'rb') as f:
        graphs = pickle.load(f)
    
    # Extract all unique labels from graphs
    all_labels = set()
    for g in graphs:
        for n in g.nodes():
            if 'label' in g.nodes[n]:
                all_labels.add(g.nodes[n]['label'])
    
    # Build label_to_id (sorted for consistency with training)
    label_to_id = {l: i for i, l in enumerate(sorted(list(all_labels)))}
    id_to_label = {v: k for k, v in label_to_id.items()}
    
    print(f"  Found {len(id_to_label)} labels: {label_to_id}")
    
    return id_to_label


def load_model_from_checkpoint(model_dir, fname_prefix, epoch, config, device='cpu'):
    """Load all model components from a checkpoint."""
    
    # Model parameters from config
    max_prev_node = config.get('max_prev_node', 6)
    label_embedding_size = config.get('label_embedding_size', 8)
    embedding_size_rnn = config.get('embedding_size_rnn', 64)
    hidden_size_rnn = config.get('hidden_size_rnn', 128)
    hidden_size_rnn_output = config.get('hidden_size_rnn_output', 16)
    num_layers = config.get('num_layers', 4)
    embedding_size_rnn_output = config.get('embedding_size_rnn_output', 8)
    embedding_size_output = config.get('embedding_size_output', 64)
    num_node_labels = config.get('num_node_labels', 12)
    
    print(f"\nCreating model architecture...")
    print(f"  max_prev_node: {max_prev_node}")
    print(f"  label_embedding_size: {label_embedding_size}")
    print(f"  hidden_size_rnn: {hidden_size_rnn}")
    print(f"  hidden_size_rnn_output: {hidden_size_rnn_output}")
    print(f"  num_layers: {num_layers}")
    print(f"  num_node_labels: {num_node_labels}")
    
    # Create models
    rnn = GRU_plain(
        input_size=max_prev_node + label_embedding_size + 3, 
        embedding_size=embedding_size_rnn,
        hidden_size=hidden_size_rnn, 
        num_layers=num_layers, 
        has_input=True,
        has_output=True, 
        output_size=hidden_size_rnn_output
    )
    
    output = GRU_plain(
        input_size=1, 
        embedding_size=embedding_size_rnn_output,
        hidden_size=hidden_size_rnn_output, 
        num_layers=num_layers, 
        has_input=True,
        has_output=True, 
        output_size=1
    )
    
    label_embedding = nn.Embedding(num_node_labels, label_embedding_size)
    label_head = MLP_plain(h_size=hidden_size_rnn_output, embedding_size=embedding_size_output, y_size=num_node_labels)
    time_head = MLP_plain(h_size=hidden_size_rnn_output, embedding_size=embedding_size_output, y_size=3)
    
    # Build checkpoint paths
    fname_rnn = os.path.join(model_dir, fname_prefix + 'lstm_' + str(epoch) + '.dat')
    fname_output = os.path.join(model_dir, fname_prefix + 'output_' + str(epoch) + '.dat')
    fname_label_embed = os.path.join(model_dir, fname_prefix + 'label_embedding_' + str(epoch) + '.dat')
    fname_label_head = os.path.join(model_dir, fname_prefix + 'label_head_' + str(epoch) + '.dat')
    fname_time_head = os.path.join(model_dir, fname_prefix + 'time_head_' + str(epoch) + '.dat')
    
    print(f"\nLoading checkpoints from epoch {epoch}...")
    
    # Load RNN and Output (required)
    if not os.path.exists(fname_rnn) or not os.path.exists(fname_output):
        raise FileNotFoundError(f"Required checkpoint files not found:\n  RNN: {fname_rnn}\n  Output: {fname_output}")
    
    rnn.load_state_dict(torch.load(fname_rnn, map_location=device))
    print(f"  ✓ Loaded RNN: {fname_rnn}")
    
    output.load_state_dict(torch.load(fname_output, map_location=device))
    print(f"  ✓ Loaded Output: {fname_output}")
    
    # Load label embedding, label head, time head (optional for old checkpoints)
    if os.path.exists(fname_label_embed):
        label_embedding.load_state_dict(torch.load(fname_label_embed, map_location=device))
        print(f"  ✓ Loaded Label Embedding: {fname_label_embed}")
    else:
        print(f"  ⚠ Label embedding not found, using random weights")
    
    if os.path.exists(fname_label_head):
        label_head.load_state_dict(torch.load(fname_label_head, map_location=device))
        print(f"  ✓ Loaded Label Head: {fname_label_head}")
    else:
        print(f"  ⚠ Label head not found, using random weights")
    
    if os.path.exists(fname_time_head):
        time_head.load_state_dict(torch.load(fname_time_head, map_location=device))
        print(f"  ✓ Loaded Time Head: {fname_time_head}")
    else:
        print(f"  ⚠ Time head not found, using random weights")
    
    # Move to device
    if device == 'cuda':
        rnn.cuda()
        output.cuda()
        label_embedding.cuda()
        label_head.cuda()
        time_head.cuda()
    
    return rnn, output, label_embedding, label_head, time_head


def generate_graphs(rnn, output, label_embedding, label_head, time_head, 
                    config, id_to_label, num_graphs=10, batch_size=16):
    """Generate graphs using the loaded model."""
    
    # Create a simple args-like object for test_rnn_epoch
    class Args:
        def __init__(self, config):
            self.max_num_node = config.get('max_num_node', 7)
            self.max_prev_node = config.get('max_prev_node', 6)
            self.num_node_labels = config.get('num_node_labels', 12)
            self.num_layers = config.get('num_layers', 4)
            self.hidden_size_rnn_output = config.get('hidden_size_rnn_output', 16)
            self.cuda = config.get('cuda', False)
            self.min_gen_node_count = config.get('min_gen_node_count', 5)
            self.max_gen_node_count = config.get('max_gen_node_count', 7)
            self.config = config
    
    args = Args(config)
    
    print(f"\nGenerating {num_graphs} graphs...")
    
    all_graphs = []
    while len(all_graphs) < num_graphs:
        # Generate a batch
        current_batch_size = min(batch_size, num_graphs - len(all_graphs))
        
        G_pred = test_rnn_epoch(
            epoch=0,  # Not used for generation
            args=args,
            rnn=rnn,
            output=output,
            test_batch_size=current_batch_size,
            label_embedding=label_embedding,
            label_head=label_head,
            time_head=time_head,
            id_to_label=id_to_label
        )
        
        # Filter graphs by node count
        G_pred = [g for g in G_pred 
                  if args.min_gen_node_count <= g.number_of_nodes() <= args.max_gen_node_count]
        
        all_graphs.extend(G_pred)
        print(f"  Generated {len(all_graphs)}/{num_graphs} valid graphs...")
    
    return all_graphs[:num_graphs]


def print_graph_info(graphs, id_to_label=None):
    """Print information about generated graphs."""
    print("\n" + "=" * 60)
    print("GENERATED GRAPHS")
    print("=" * 60)
    
    for i, G in enumerate(graphs):
        print(f"\nGraph {i+1}:")
        print(f"  Nodes: {G.number_of_nodes()}")
        print(f"  Edges: {G.number_of_edges()}")
        
        # Print node details
        print("  Node details:")
        for node in sorted(G.nodes()):
            label_id = G.nodes[node].get('label', 'N/A')
            label_name = id_to_label.get(label_id, f"ID:{label_id}") if id_to_label else str(label_id)
            
            norm_time = G.nodes[node].get('norm_time', 'N/A')
            trace_time = G.nodes[node].get('trace_time', 'N/A')
            prev_event_time = G.nodes[node].get('prev_event_time', 'N/A')
            
            if isinstance(norm_time, float):
                print(f"    Node {node}: {label_name:15} | norm_time={norm_time:.4f}, trace_time={trace_time:.4f}, prev_event={prev_event_time:.4f}")
            else:
                print(f"    Node {node}: {label_name}")
        
        # Print edges
        print(f"  Edges: {list(G.edges())}")


def main():
    parser = argparse.ArgumentParser(description='Generate graphs from a trained checkpoint')
    parser.add_argument('--run_id', type=str, required=True, 
                        help='Run folder name (e.g., GraphRNN_RNN_helpdesk_4_128_2025-12-06_19-58-27)')
    parser.add_argument('--epoch', type=int, required=True,
                        help='Checkpoint epoch to load (e.g., 450)')
    parser.add_argument('--num_graphs', type=int, default=10,
                        help='Number of graphs to generate')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for generation')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save generated graphs (default: graphs/<run_id>/)')
    parser.add_argument('--save_figures', action='store_true',
                        help='Save visualizations of generated graphs')
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA for generation')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("GRAPH GENERATOR")
    print("=" * 60)
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Override cuda setting
    if args.cuda and torch.cuda.is_available():
        config['cuda'] = True
        device = 'cuda'
    else:
        config['cuda'] = False
        device = 'cpu'
    
    # Build paths
    model_dir = os.path.join('./model_save/', args.run_id)
    graphs_dir = os.path.join('./graphs/', args.run_id)
    fname_prefix = 'GraphRNN_RNN_helpdesk_4_128_'  # Standard prefix
    
    if not os.path.exists(model_dir):
        print(f"\nERROR: Model directory not found: {model_dir}")
        print("\nAvailable runs:")
        for run in sorted(os.listdir('./model_save/')):
            if os.path.isdir(os.path.join('./model_save/', run)):
                print(f"  - {run}")
        return
    
    # Load label mapping from saved graphs
    print("\nLoading label mapping...")
    id_to_label = load_label_mapping(graphs_dir, fname_prefix)
    
    if id_to_label is None:
        print("  Using default label mapping")
        id_to_label = {i: str(i) for i in range(config.get('num_node_labels', 12))}
    
    # Load model
    rnn, output, label_embedding, label_head, time_head = load_model_from_checkpoint(
        model_dir, fname_prefix, args.epoch, config, device
    )
    
    # Generate graphs
    graphs = generate_graphs(
        rnn, output, label_embedding, label_head, time_head,
        config, id_to_label, 
        num_graphs=args.num_graphs,
        batch_size=args.batch_size
    )
    
    # Print graph info
    print_graph_info(graphs, id_to_label)
    
    # Save graphs
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = graphs_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    fname = os.path.join(output_dir, f'generated_epoch{args.epoch}_n{args.num_graphs}.dat')
    save_graph_list(graphs, fname)
    print(f"\n✓ Saved graphs to: {fname}")
    
    # Save figures
    if args.save_figures:
        fig_dir = os.path.join('./figures_prediction/', args.run_id)
        os.makedirs(fig_dir, exist_ok=True)
        fname_fig = os.path.join(fig_dir, f'generated_epoch{args.epoch}')
        draw_graph_list(graphs[:16], 4, 4, fname=fname_fig)
        print(f"✓ Saved figures to: {fname_fig}.png")
    
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE!")
    print("=" * 60)


if __name__ == '__main__':
    main()

#sample command