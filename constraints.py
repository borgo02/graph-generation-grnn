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


    def compute_loss(self, logits, targets, step, **kwargs):
        total_loss = 0.0
        for constraint in self.constraints:
            weight = self.weights.get(type(constraint), 1.0)
            loss = constraint.compute_loss(logits, targets, step, **kwargs)
            total_loss += weight * loss
        return total_loss
