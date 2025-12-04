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

    def apply_mask(self, logits, step, **kwargs):
        """
        Apply masking to logits to enforce hard constraints during generation.
        Args:
            logits: Predicted logits (batch_size, num_classes)
            step: Current generation step
        Returns:
            masked_logits: Logits with invalid actions set to -inf
        """
        return logits

class StartNodeConstraint(GraphConstraint):
    def __init__(self, target_label, label_to_id):
        self.target_label = target_label
        self.target_id = label_to_id.get(target_label)
        if self.target_id is None:
            print(f"Warning: StartNodeConstraint target '{target_label}' not found in label map.")

    def compute_loss(self, logits, targets, step, **kwargs):
        if step == 0 and self.target_id is not None:
            # Create a target tensor filled with the start node ID
            batch_size = logits.size(0)
            target = torch.full((batch_size,), self.target_id, dtype=torch.long, device=logits.device)
            return F.cross_entropy(logits, target)
        return 0.0

    def apply_mask(self, logits, step, **kwargs):
        if step == 0 and self.target_id is not None:
            # Mask everything except the target_id
            mask = torch.ones_like(logits) * float('-inf')
            mask[:, self.target_id] = 0
            return logits + mask
        return logits

class EndNodeConstraint(GraphConstraint):
    def __init__(self, target_label, label_to_id, min_nodes=None):
        self.target_label = target_label
        self.target_id = label_to_id.get(target_label)
        self.min_nodes = min_nodes
        if self.target_id is None:
            print(f"Warning: EndNodeConstraint target '{target_label}' not found in label map.")

    def compute_loss(self, logits, targets, step, **kwargs):
        # Penalize predicting END too early.
        if self.min_nodes and step < self.min_nodes and self.target_id is not None:
             probs = F.softmax(logits, dim=1)
             p_end = probs[:, self.target_id]
             # Minimize the probability of predicting the END label
             return p_end.mean()
        return 0.0

    def apply_mask(self, logits, step, **kwargs):
        if self.target_id is not None:
            # Prevent END before min_nodes
            if self.min_nodes and step < self.min_nodes:
                logits[:, self.target_id] = float('-inf')
        return logits

class ConstraintManager:
    def __init__(self, config, label_to_id):
        self.constraints = []
        self.weights = {}
        
        c_config = config.get('constraints', {})
        
        # Start Node
        if 'force_start_node' in c_config:
            label = c_config['force_start_node']
            self.constraints.append(StartNodeConstraint(label, label_to_id))
            self.weights[StartNodeConstraint] = c_config.get('start_node_weight', 1.0)
            
        # End Node
        if 'force_end_node' in c_config:
            label = c_config['force_end_node']
            min_nodes = config.get('min_gen_node_count', 0)
            self.constraints.append(EndNodeConstraint(label, label_to_id, min_nodes))
            self.weights[EndNodeConstraint] = c_config.get('end_node_weight', 1.0)

    def compute_loss(self, logits, targets, step):
        total_loss = 0.0
        for constraint in self.constraints:
            weight = self.weights.get(type(constraint), 1.0)
            loss = constraint.compute_loss(logits, targets, step)
            total_loss += weight * loss
        return total_loss

    def apply_mask(self, logits, step):
        for constraint in self.constraints:
            logits = constraint.apply_mask(logits, step)
        return logits
