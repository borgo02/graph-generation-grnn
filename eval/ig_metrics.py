"""
Instance Graph (IG) Evaluation Metrics.

Implements metrics from the paper for evaluating Instance Graphs:
- Accuracy (Acc): Number of correctly reconstructed graphs
- Matching Cost (MC): Graph edit distance between generated and true IGs
- Average Generalization (AG): Number of occurrence sequences per IG


"""

import networkx as nx
import numpy as np
from typing import List, Tuple, Optional, Callable
from functools import lru_cache
import warnings


def node_match(n1_attrs: dict, n2_attrs: dict) -> bool:
    """Check if two nodes have the same label."""
    label1 = n1_attrs.get('label', n1_attrs.get('concept:name', None))
    label2 = n2_attrs.get('label', n2_attrs.get('concept:name', None))
    return label1 == label2


def edge_match(e1_attrs: dict, e2_attrs: dict) -> bool:
    """Check if two edges have the same label (if labels exist)."""
    label1 = e1_attrs.get('label', None)
    label2 = e2_attrs.get('label', None)
    # If edges don't have labels, consider them matching
    if label1 is None and label2 is None:
        return True
    return label1 == label2


# ============================================================================
# ACCURACY METRICS
# ============================================================================

def is_labeled_isomorphic(g1: nx.DiGraph, g2: nx.DiGraph) -> bool:
    """
    Check if two graphs are isomorphic considering node and edge labels.
    
    Args:
        g1: First graph
        g2: Second graph
        
    Returns:
        True if graphs are labeled-isomorphic, False otherwise
    """
    return nx.is_isomorphic(g1, g2, node_match=node_match, edge_match=edge_match)


def compute_accuracy(true_graphs: List[nx.DiGraph], 
                     pred_graphs: List[nx.DiGraph],
                     match_by_index: bool = True) -> Tuple[int, float]:
    """
    Compute accuracy - count and percentage of exactly matching graphs.
    
    Args:
        true_graphs: List of ground truth graphs
        pred_graphs: List of predicted/generated graphs
        match_by_index: If True, compare graphs at same index.
                       If False, find best match for each true graph.
    
    Returns:
        Tuple of (correct_count, percentage)
    """
    if len(true_graphs) == 0:
        return 0, 0.0
    
    correct = 0
    
    if match_by_index:
        # Compare graphs at same index
        n = min(len(true_graphs), len(pred_graphs))
        for i in range(n):
            if is_labeled_isomorphic(true_graphs[i], pred_graphs[i]):
                correct += 1
    else:
        # For each true graph, check if any predicted graph matches
        # This is more expensive but allows for unordered comparison
        used_pred = set()
        for true_g in true_graphs:
            for j, pred_g in enumerate(pred_graphs):
                if j not in used_pred and is_labeled_isomorphic(true_g, pred_g):
                    correct += 1
                    used_pred.add(j)
                    break
    
    percentage = (correct / len(true_graphs)) * 100
    return correct, percentage


# ============================================================================
# MATCHING COST (GRAPH EDIT DISTANCE)
# ============================================================================

def compute_matching_cost(true_graph: nx.DiGraph, 
                          pred_graph: nx.DiGraph,
                          timeout: float = 5.0) -> Optional[int]:
    """
    Compute graph edit distance (matching cost) between true and predicted graphs.
    
    Operations (each with cost 1):
    - Add/delete a node
    - Add/delete an edge
    - Change a node/edge label
    - Reverse an edge direction
    
    Args:
        true_graph: Ground truth graph
        pred_graph: Predicted/generated graph
        timeout: Maximum time in seconds for GED computation
        
    Returns:
        Integer matching cost, or None if timeout exceeded
    """
    # Define cost functions (all operations have cost 1)
    def node_subst_cost(n1, n2):
        """Cost to substitute n1 with n2 (0 if same label, 1 otherwise)."""
        return 0 if node_match(n1, n2) else 1
    
    def node_del_cost(n):
        """Cost to delete a node."""
        return 1
    
    def node_ins_cost(n):
        """Cost to insert a node."""
        return 1
    
    def edge_subst_cost(e1, e2):
        """Cost to substitute e1 with e2."""
        return 0 if edge_match(e1, e2) else 1
    
    def edge_del_cost(e):
        """Cost to delete an edge."""
        return 1
    
    def edge_ins_cost(e):
        """Cost to insert an edge."""
        return 1
    
    try:
        ged = nx.graph_edit_distance(
            true_graph, pred_graph,
            node_subst_cost=node_subst_cost,
            node_del_cost=node_del_cost,
            node_ins_cost=node_ins_cost,
            edge_subst_cost=edge_subst_cost,
            edge_del_cost=edge_del_cost,
            edge_ins_cost=edge_ins_cost,
            timeout=timeout
        )
        return int(ged) if ged is not None else None
    except Exception as e:
        warnings.warn(f"Error computing GED: {e}")
        return None


def compute_matching_cost_batch(true_graphs: List[nx.DiGraph],
                                pred_graphs: List[nx.DiGraph],
                                timeout_per_pair: float = 5.0) -> dict:
    """
    Compute matching cost statistics for paired graph lists.
    
    Args:
        true_graphs: List of ground truth graphs
        pred_graphs: List of predicted graphs (same length as true_graphs)
        timeout_per_pair: Timeout for each GED computation
        
    Returns:
        Dictionary with: mean, median, std, total, count, failed
    """
    n = min(len(true_graphs), len(pred_graphs))
    costs = []
    failed = 0
    
    for i in range(n):
        mc = compute_matching_cost(
            true_graphs[i], pred_graphs[i], 
            timeout=timeout_per_pair
        )
        if mc is not None:
            costs.append(mc)
        else:
            failed += 1
    
    if not costs:
        return {
            'mean': float('nan'),
            'median': float('nan'),
            'std': float('nan'),
            'total': 0,
            'count': 0,
            'failed': failed
        }
    
    return {
        'mean': np.mean(costs),
        'median': np.median(costs),
        'std': np.std(costs),
        'total': sum(costs),
        'count': len(costs),
        'failed': failed,
        'costs': costs  # Include individual costs for analysis
    }


def compute_matching_cost_best(true_graphs: List[nx.DiGraph],
                               pred_graphs: List[nx.DiGraph],
                               timeout_per_pair: float = 5.0) -> dict:
    """
    Compute matching cost statistics using a 'Best Match' strategy.
    
    For each predicted graph, we find the ground truth graph that minimizes
    the graph edit distance. This is O(N*M) and can be very slow.
    
    Args:
        true_graphs: List of ground truth graphs
        pred_graphs: List of predicted graphs
        timeout_per_pair: Timeout for each GED computation
        
    Returns:
        Dictionary with: mean, median, std, total, count, failed
    """
    costs = []
    failed = 0
    
    print(f"  Computing Best Match MC for {len(pred_graphs)} graphs against {len(true_graphs)} true graphs...")
    
    for i, pred_g in enumerate(pred_graphs):
        best_cost = float('inf')
        found_match = False
        
        # Optimization: Filter potential matches by simple properties first?
        # For now, we do brute force but with early exit if cost is 0
        
        for true_g in true_graphs:
            # Quick check for isomorphism (cost 0)
            if is_labeled_isomorphic(pred_g, true_g):
                best_cost = 0
                found_match = True
                break
                
            # Compute GED
            mc = compute_matching_cost(true_g, pred_g, timeout=timeout_per_pair)
            
            if mc is not None:
                if mc < best_cost:
                    best_cost = mc
                    found_match = True
            
        if found_match and best_cost != float('inf'):
            costs.append(best_cost)
        else:
            # If we couldn't compute specific costs due to timeouts using all candidates,
            # or if the lists are empty.
            failed += 1
            
        if (i + 1) % 10 == 0:
            print(f"    Processed {i + 1}/{len(pred_graphs)}...")

    if not costs:
        return {
            'mean': float('nan'),
            'median': float('nan'),
            'std': float('nan'),
            'total': 0,
            'count': 0,
            'failed': failed
        }
    
    return {
        'mean': np.mean(costs),
        'median': np.median(costs),
        'std': np.std(costs),
        'total': sum(costs),
        'count': len(costs),
        'failed': failed,
        'costs': costs
    }


# ============================================================================
# AVERAGE GENERALIZATION
# ============================================================================

def convert_to_dag(graph: nx.Graph) -> nx.DiGraph:
    """
    Convert an undirected graph to a directed acyclic graph (DAG).
    
    Uses the node ordering (node IDs) to determine edge direction.
    Assumes nodes are numbered in BFS/generation order where lower IDs
    come before higher IDs in the process flow.
    
    Args:
        graph: An undirected graph (nx.Graph)
        
    Returns:
        A directed graph (nx.DiGraph) with edges pointing from lower to higher node IDs
    """
    if graph.is_directed():
        return graph
    
    dag = nx.DiGraph()
    
    # Copy nodes with their attributes
    for node in graph.nodes():
        dag.add_node(node, **graph.nodes[node])
    
    # Add edges directed from lower to higher node ID
    for u, v in graph.edges():
        if u < v:
            dag.add_edge(u, v, **graph.edges[u, v])
        else:
            dag.add_edge(v, u, **graph.edges[u, v])
    
    return dag

def find_start_end_nodes(graph: nx.DiGraph) -> Tuple[Optional[int], Optional[int]]:
    """
    Find START and END nodes in the graph.
    
    Looks for nodes with labels 'START'/'END' or nodes with 
    in-degree 0 (start) and out-degree 0 (end).
    """
    start_node = None
    end_node = None
    
    for node in graph.nodes():
        label = graph.nodes[node].get('label', 
                graph.nodes[node].get('concept:name', str(node)))
        if label == 'START' or str(label).upper() == 'START':
            start_node = node
        elif label == 'END' or str(label).upper() == 'END':
            end_node = node
    
    # Fallback: use in/out degree
    if start_node is None:
        for node in graph.nodes():
            if graph.in_degree(node) == 0:
                start_node = node
                break
    
    if end_node is None:
        for node in graph.nodes():
            if graph.out_degree(node) == 0:
                end_node = node
                break
    
    return start_node, end_node


def count_topological_orderings(graph: nx.DiGraph, 
                                 start_node: Optional[int] = None,
                                 end_node: Optional[int] = None,
                                 max_count: int = 10000) -> int:
    """
    Count the number of valid topological orderings (occurrence sequences).
    
    This counts how many different linear sequences of nodes can be generated
    while respecting the partial order defined by the graph edges.
    
    Args:
        graph: A directed acyclic graph (DAG)
        start_node: Optional start node (must be first in sequence)
        end_node: Optional end node (must be last in sequence)
        max_count: Maximum count before returning (to prevent hanging)
        
    Returns:
        Number of valid topological orderings (capped at max_count)
    """
    if not nx.is_directed_acyclic_graph(graph):
        # Return 0 silently - will be tracked as cyclic in batch computation
        return 0
    
    nodes = list(graph.nodes())
    n = len(nodes)
    
    if n == 0:
        return 0
    if n == 1:
        return 1
    
    # Use dynamic programming with memoization
    # State: set of remaining nodes to order
    
    def count_orderings(remaining: frozenset, last_placed: frozenset) -> int:
        """
        Count orderings for remaining nodes given which have been placed.
        A node can be placed if all its predecessors have been placed.
        """
        if len(remaining) == 0:
            return 1
        
        # Early termination if we've exceeded max count
        if count_orderings.total >= max_count:
            return 0
        
        total = 0
        for node in remaining:
            # Check if all predecessors are placed
            preds = set(graph.predecessors(node))
            if preds.issubset(last_placed):
                # This node can be placed next
                new_remaining = remaining - {node}
                new_placed = last_placed | {node}
                total += count_orderings(new_remaining, new_placed)
                
                if total >= max_count:
                    count_orderings.total = total
                    return total
        
        count_orderings.total = max(count_orderings.total, total)
        return total
    
    count_orderings.total = 0
    
    # Handle start/end constraints
    if start_node is None:
        start_node, end_node = find_start_end_nodes(graph)
    
    remaining = frozenset(nodes)
    placed = frozenset()
    
    # If start node specified, it must be first
    if start_node is not None and start_node in remaining:
        remaining = remaining - {start_node}
        placed = frozenset({start_node})
    
    # Count orderings (end node constraint is implicit - it has no successors)
    count = count_orderings(remaining, placed)
    
    return min(count, max_count)


def compute_generalization(graph: nx.DiGraph, max_count: int = 10000) -> int:
    """
    Compute the generalization of an Instance Graph.
    
    Generalization is the number of occurrence sequences (valid execution traces)
    that the IG can represent.
    
    Args:
        graph: The Instance Graph (should be a DAG)
        max_count: Maximum count (to prevent hanging on highly parallel graphs)
        
    Returns:
        Number of occurrence sequences (capped at max_count)
    """
    return count_topological_orderings(graph, max_count=max_count)


def compute_avg_generalization(graphs: List[nx.DiGraph], 
                                max_count_per_graph: int = 10000) -> dict:
    """
    Compute average generalization across a set of Instance Graphs.
    
    Args:
        graphs: List of Instance Graphs
        max_count_per_graph: Max count per graph before capping
        
    Returns:
        Dictionary with: mean, median, std, min, max, counts
    """
    if not graphs:
        return {
            'mean': float('nan'),
            'median': float('nan'),
            'std': float('nan'),
            'min': 0,
            'max': 0,
            'counts': []
        }
    
    counts = []
    num_dags = 0
    num_cyclic = 0
    num_converted = 0
    
    for g in graphs:
        # Convert undirected graphs to directed using node ordering
        if not g.is_directed():
            g = convert_to_dag(g)
            num_converted += 1
        
        is_dag = nx.is_directed_acyclic_graph(g)
        if is_dag:
            num_dags += 1
            ag = compute_generalization(g, max_count=max_count_per_graph)
        else:
            num_cyclic += 1
            ag = 0  # Cyclic graphs have undefined AG
        counts.append(ag)
    
    # Compute statistics only for valid DAGs
    dag_counts = [c for c in counts if c > 0]
    
    if not dag_counts:
        dag_mean = float('nan')
        dag_median = float('nan')
        dag_std = float('nan')
    else:
        dag_mean = np.mean(dag_counts)
        dag_median = np.median(dag_counts)
        dag_std = np.std(dag_counts)
    
    return {
        'mean': np.mean(counts),  # Overall mean (includes 0 for cyclic)
        'median': np.median(counts),
        'std': np.std(counts),
        'min': min(counts),
        'max': max(counts),
        'dag_mean': dag_mean,  # Mean only for DAGs
        'dag_median': dag_median,
        'dag_std': dag_std,
        'num_dags': num_dags,
        'num_cyclic': num_cyclic,
        'ag_1': sum(1 for c in counts if c == 1),  # Tailored graphs (AG=1)
        'ag_capped': sum(1 for c in counts if c >= max_count_per_graph),
        'counts': counts
    }


# ============================================================================
# SUMMARY FUNCTION
# ============================================================================

def evaluate_instance_graphs(pred_graphs: List[nx.DiGraph],
                              true_graphs: Optional[List[nx.DiGraph]] = None,
                              compute_ag: bool = True,
                              ag_max_count: int = 10000,
                              mc_timeout: float = 5.0,
                              match_strategy: str = 'index') -> dict:
    """
    Comprehensive evaluation of Instance Graphs.
    
    Args:
        pred_graphs: List of predicted/generated graphs
        true_graphs: Optional list of ground truth graphs (for Acc/MC)
        compute_ag: Whether to compute Average Generalization
        ag_max_count: Max count for AG computation
        mc_timeout: Timeout for MC (GED) computation per pair
        match_strategy: 'index' (compare i-th with i-th) or 'best' (find closest match)
        
    Returns:
        Dictionary with all evaluation metrics
    """
    results = {
        'num_predicted': len(pred_graphs)
    }
    
    # Average Generalization (can always be computed)
    if compute_ag:
        print("Computing Average Generalization...")
        ag_results = compute_avg_generalization(pred_graphs, ag_max_count)
        results['avg_generalization'] = ag_results
        print(f"  DAGs: {ag_results['num_dags']}/{len(pred_graphs)}, Cyclic: {ag_results['num_cyclic']}")
        if ag_results['num_dags'] > 0:
            print(f"  AG Mean (DAGs only): {ag_results['dag_mean']:.2f}")
            print(f"  AG Median (DAGs only): {ag_results['dag_median']:.2f}")
            print(f"  AG Min/Max: {ag_results['min']}/{ag_results['max']}")
            print(f"  Tailored (AG=1): {ag_results['ag_1']}, Capped: {ag_results['ag_capped']}")
        else:
            print("  (All graphs contain cycles - AG undefined)")
    
    # Accuracy and Matching Cost (require ground truth)
    if true_graphs is not None:
        results['num_true'] = len(true_graphs)
        
        # Accuracy
        print(f"\nComputing Accuracy (Strategy: {match_strategy})...")
        match_by_index = (match_strategy == 'index')
        acc_count, acc_pct = compute_accuracy(true_graphs, pred_graphs, match_by_index=match_by_index)
        results['accuracy'] = {
            'count': acc_count,
            'percentage': acc_pct
        }
        print(f"  Correct: {acc_count}/{len(true_graphs)} ({acc_pct:.1f}%)")
        
        # Matching Cost
        print(f"\nComputing Matching Cost (Strategy: {match_strategy})...")
        print("Note: 'best' strategy is computationally expensive (O(NxM)).")
        
        if match_strategy == 'best':
            mc_results = compute_matching_cost_best(
                true_graphs, pred_graphs,
                timeout_per_pair=mc_timeout
            )
        else:
            mc_results = compute_matching_cost_batch(
                true_graphs, pred_graphs, 
                timeout_per_pair=mc_timeout
            )
        results['matching_cost'] = {
            'mean': mc_results['mean'],
            'median': mc_results['median'],
            'std': mc_results['std'],
            'total': mc_results['total'],
            'computed': mc_results['count'],
            'failed': mc_results['failed']
        }
        print(f"  MC Mean: {mc_results['mean']:.2f}")
        print(f"  MC Median: {mc_results['median']:.2f}")
        print(f"  MC Total: {mc_results['total']}")
        if mc_results['failed'] > 0:
            print(f"  Failed (timeout): {mc_results['failed']}")
    
    return results


# ============================================================================
# TESTING
# ============================================================================

if __name__ == '__main__':
    # Simple tests
    print("Testing IG Metrics...")
    
    # Test 1: Simple linear chain
    g1 = nx.DiGraph()
    g1.add_node(0, label='START')
    g1.add_node(1, label='A')
    g1.add_node(2, label='END')
    g1.add_edge(0, 1)
    g1.add_edge(1, 2)
    
    ag1 = compute_generalization(g1)
    print(f"Linear chain (START->A->END): AG = {ag1}")  # Should be 1
    assert ag1 == 1, f"Expected 1, got {ag1}"
    
    # Test 2: Fork-join
    g2 = nx.DiGraph()
    g2.add_node(0, label='START')
    g2.add_node(1, label='A')
    g2.add_node(2, label='B')
    g2.add_node(3, label='END')
    g2.add_edge(0, 1)
    g2.add_edge(0, 2)
    g2.add_edge(1, 3)
    g2.add_edge(2, 3)
    
    ag2 = compute_generalization(g2)
    print(f"Fork-join (START->{'{A,B}'}->END): AG = {ag2}")  # Should be 2
    assert ag2 == 2, f"Expected 2, got {ag2}"
    
    # Test 3: Accuracy
    g3 = g1.copy()  # Same as g1
    acc, pct = compute_accuracy([g1], [g3])
    print(f"Accuracy (same graph): {acc}/{1} = {pct}%")
    assert acc == 1
    
    # Test 4: Matching cost
    mc = compute_matching_cost(g1, g1)
    print(f"Matching cost (same graph): {mc}")
    assert mc == 0, f"Expected 0, got {mc}"
    
    # Test 5: Different graphs
    g4 = nx.DiGraph()
    g4.add_node(0, label='START')
    g4.add_node(1, label='X')  # Different label
    g4.add_node(2, label='END')
    g4.add_edge(0, 1)
    g4.add_edge(1, 2)
    
    mc2 = compute_matching_cost(g1, g4)
    print(f"Matching cost (one label diff): {mc2}")
    assert mc2 == 1, f"Expected 1, got {mc2}"
    
    print("\n✓ All tests passed!")
