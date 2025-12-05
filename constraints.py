import torch
import torch.nn.functional as F

class GraphConstraint:
    """Base class for graph constraints."""
    def compute_loss(self, logits, targets, step, **kwargs):
        """
        Compute the loss penalty for this constraint.
        Args:
            logits: Predicted logits (batch_size, num_classes)
            targets: Target labels (batch_size)
            step: Current generation step (0 to max_num_node-1)
        Returns:
            loss: Scalar tensor
        """
        return 0.0

class StartNodeConstraint(GraphConstraint):
    """Ensures the first node is 'START'"""
    def __init__(self, target_label, label_to_id):
        self.target_label = target_label
        self.target_id = label_to_id.get(target_label)
        if self.target_id is None:
            print(f"Warning: StartNodeConstraint target '{target_label}' not found in label map.")

    def compute_loss(self, logits, targets, step, **kwargs):
        if step == 0 and self.target_id is not None:
            # Force START at step 0
            batch_size = logits.size(0)
            target = torch.full((batch_size,), self.target_id, dtype=torch.long, device=logits.device)
            return F.cross_entropy(logits, target)
        return 0.0

class UniqueStartConstraint(GraphConstraint):
    """Ensures START appears only at step 0"""
    def __init__(self, target_label, label_to_id):
        self.target_label = target_label
        self.target_id = label_to_id.get(target_label)
        if self.target_id is None:
            print(f"Warning: UniqueStartConstraint target '{target_label}' not found in label map.")

    def compute_loss(self, logits, targets, step, **kwargs):
        if step > 0 and self.target_id is not None:
            # Penalize START appearing after step 0
            probs = F.softmax(logits, dim=1)
            p_start = probs[:, self.target_id]
            # Minimize probability of START
            return p_start.mean()
        return 0.0

class UniqueEndConstraint(GraphConstraint):
    """Ensures END appears only at the final step"""
    def __init__(self, target_label, label_to_id):
        self.target_label = target_label
        self.target_id = label_to_id.get(target_label)
        if self.target_id is None:
            print(f"Warning: UniqueEndConstraint target '{target_label}' not found in label map.")

    def compute_loss(self, logits, targets, step, is_final_step=False, **kwargs):
        if self.target_id is not None and not is_final_step:
            # Penalize END appearing before the final step
            probs = F.softmax(logits, dim=1)
            p_end = probs[:, self.target_id]
            # Minimize probability of END
            return p_end.mean()
        return 0.0

class FinalNodeConstraint(GraphConstraint):
    """Ensures the last node of every sequence is 'END'"""
    def __init__(self, target_label, label_to_id):
        self.target_label = target_label
        self.target_id = label_to_id.get(target_label)
        if self.target_id is None:
            print(f"Warning: FinalNodeConstraint target '{target_label}' not found in label map.")

    def compute_loss(self, logits, targets, step, is_final_step=False, **kwargs):
        """
        Penalizes sequences that don't end with 'END'.
        Args:
            is_final_step: True if this is the last step of the sequence
        """
        if is_final_step and self.target_id is not None:
            # Encourage 'END' prediction at the final step
            batch_size = logits.size(0)
            target = torch.full((batch_size,), self.target_id, dtype=torch.long, device=logits.device)
            return F.cross_entropy(logits, target)
        return 0.0

class ParallelNodeConstraint(GraphConstraint):
    """
    Ensures that parallel nodes (siblings sharing a parent) have similar time attributes.
    """
    def __init__(self):
        pass

    def compute_loss(self, logits, targets, step, **kwargs):
        """
        Args:
            step: Current node index (i)
            time_pred: Predicted times for current node (batch, 3)
            adj: Adjacency matrix (batch, max_len, max_prev_node)
            time_gt: Ground truth times (batch, max_len, 3)
        """
        time_pred = kwargs.get('time_pred')
        adj = kwargs.get('adj')
        time_gt = kwargs.get('time_gt')
        
        if time_pred is None or adj is None or time_gt is None:
            return 0.0
            
        # adj is (batch, max_len, max_prev_node)
        # adj[b, i, k] = 1 means node i connects to node i-k-1
        
        batch_size = time_pred.size(0)
        loss = 0.0
        count = 0
        
        # Iterate over batch
        for b in range(batch_size):
            # Current node i = step
            # Find parents of node i
            # Edges are in adj[b, step, :]
            # If adj[b, step, k] == 1, parent is p = step - k - 1
            
            parents_i = []
            if step < adj.size(1):
                row = adj[b, step]
                for k in range(len(row)):
                    if row[k] == 1:
                        p = step - k - 1
                        if p >= 0:
                            parents_i.append(p)
            
            if not parents_i:
                continue
                
            # Find siblings j < i
            # Sibling j must share at least one parent with i
            siblings = []
            for j in range(step):
                parents_j = []
                row_j = adj[b, j]
                for k in range(len(row_j)):
                    if row_j[k] == 1:
                        p = j - k - 1
                        if p >= 0:
                            parents_j.append(p)
                            
                # Check intersection
                if not set(parents_i).isdisjoint(parents_j):
                    siblings.append(j)
            
            # Compute loss against siblings
            if siblings:
                # Average time of siblings? Or minimize distance to each?
                # Let's minimize distance to the closest sibling or average?
                # User said "same numbers", so they should be close.
                # Let's take the mean of siblings' times
                sibling_times = time_gt[b, siblings, :] # (num_siblings, 3)
                target_time = sibling_times.mean(dim=0)
                
                loss += F.mse_loss(time_pred[b], target_time)
                count += 1
                
        if count > 0:
            return loss / count
        return 0.0
        if count > 0:
            return loss / count
        return 0.0


class StartTimeConstraint(GraphConstraint):
    """
    Ensures that the START node (step 0) has trace_time and prev_event_time close to 0.0.
    """
    def __init__(self):
        pass

    def compute_loss(self, logits, targets, step, **kwargs):
        time_pred = kwargs.get('time_pred') # (batch, 3)
        
        if step == 0 and time_pred is not None:
            # time_pred is [norm_time, trace_time, prev_event_time]
            # We want trace_time (idx 1) and prev_event_time (idx 2) to be 0.0
            
            # Target is 0.0 for indices 1 and 2
            target = torch.zeros_like(time_pred[:, 1:])
            loss = F.mse_loss(time_pred[:, 1:], target)
            return loss
            
        return 0.0

class GlobalConnectivityConstraint(GraphConstraint):
    """
    Ensures all nodes are reachable from Node 0 using differentiable matrix multiplication.
    Constructs full adjacency matrix A from y_pred and computes (I+A)^N.
    """
    def __init__(self, max_prev_node):
        self.max_prev_node = max_prev_node

    def compute_loss(self, logits, targets, step, **kwargs):
        # This constraint is computed once per batch, not per step.
        
        y_pred = kwargs.get('y_pred_full')
        if y_pred is None:
            return 0.0
            
        # y_pred: (batch, N-1, max_prev_node)
        # Node 0 is START.
        # y_pred[b, i, :] corresponds to edges for Node i+1.
        
        batch_size, seq_len, _ = y_pred.size()
        num_nodes = seq_len + 1
        
        # Construct Adjacency Matrix A (batch, N, N)
        A = torch.eye(num_nodes).unsqueeze(0).repeat(batch_size, 1, 1).to(y_pred.device)
        
        for i in range(seq_len):
            u = i + 1
            
            # Edges to previous nodes
            # y_pred[b, i, j] is prob of edge (u, u - 1 - j)
            # j goes from 0 to max_prev_node-1
            
            valid_j = min(self.max_prev_node, u)
            
            probs = y_pred[:, i, :valid_j] # (batch, valid_j)
            
            for j in range(valid_j):
                v = u - 1 - j
                A[:, u, v] = probs[:, j]
                A[:, v, u] = probs[:, j]
        
        R = A
        for _ in range(4):
            R = torch.bmm(R, R)            
            pass
        
        reachability = R[:, 0, :] # (batch, N)
        
        threshold = 1e-2
        loss = F.relu(threshold - reachability)
        
        return torch.mean(loss)


class ConstraintManager:
    def __init__(self, config, label_to_id):
        self.constraints = []
        self.weights = {}
        
        c_config = config.get('constraints', {})
        
        # Start Node Constraints
        if 'force_start_node' in c_config:
            label = c_config['force_start_node']
            # Ensure START is at position 0
            self.constraints.append(StartNodeConstraint(label, label_to_id))
            self.weights[StartNodeConstraint] = c_config.get('start_node_weight', 1.0)
            
            # Ensure START appears only once
            self.constraints.append(UniqueStartConstraint(label, label_to_id))
            self.weights[UniqueStartConstraint] = c_config.get('unique_start_weight', 1.0)
            
        # End Node Constraints
        if 'force_end_node' in c_config:
            label = c_config['force_end_node']
            
            # Ensure END appears only at the final step
            self.constraints.append(UniqueEndConstraint(label, label_to_id))
            self.weights[UniqueEndConstraint] = c_config.get('unique_end_weight', 1.0)
            
            # Ensure the last node is END
            self.constraints.append(FinalNodeConstraint(label, label_to_id))
            self.weights[FinalNodeConstraint] = c_config.get('final_node_weight', 1.0)
            
        # Parallel Node Constraint
        if 'parallel_node_weight' in c_config:
            self.constraints.append(ParallelNodeConstraint())
            self.weights[ParallelNodeConstraint] = c_config.get('parallel_node_weight', 1.0)
            
        # Start Time Constraint
        if 'start_time_weight' in c_config:
            self.constraints.append(StartTimeConstraint())
            self.weights[StartTimeConstraint] = c_config.get('start_time_weight', 1.0)
            
        # Connectivity Constraint
        if 'connectivity_weight' in c_config:
            max_prev_node = config.get('max_prev_node', 50)
            self.constraints.append(GlobalConnectivityConstraint(max_prev_node=max_prev_node))
            self.weights[GlobalConnectivityConstraint] = c_config.get('connectivity_weight', 1.0)

    def compute_loss(self, logits, targets, step, **kwargs):
        total_loss = 0.0
        for constraint in self.constraints:
            weight = self.weights.get(type(constraint), 1.0)
            
            # Special handling for GlobalConnectivityConstraint (step -1)
            if step == -1:
                if isinstance(constraint, GlobalConnectivityConstraint):
                    loss = constraint.compute_loss(logits, targets, step, **kwargs)
                    total_loss += weight * loss
                continue
                
            # Skip GlobalConnectivityConstraint for regular steps
            if isinstance(constraint, GlobalConnectivityConstraint):
                continue
                
            loss = constraint.compute_loss(logits, targets, step, **kwargs)
            total_loss += weight * loss

        return total_loss
