# ultralytics/yolo/loss.py

import torch
import torch.nn.functional as F

class ComputeLoss:
    def __init__(self, model, class_weights=None):
        self.model = model
        self.class_weights = class_weights  # Tensor of class weights
        # Initialize other loss components if necessary

    def __call__(self, predictions, targets):
        # Extract components from predictions
        cls_logits = predictions['cls']  # Shape: [batch_size, num_classes, ...]
        bbox_preds = predictions['bbox']  # Shape: [batch_size, 4, ...]

        # Extract class targets
        cls_targets = targets[:, 0].long()  # Assuming first column is class index
        
        # Classification loss (Cross-Entropy Loss)
        cls_loss = F.cross_entropy(cls_logits, cls_targets, reduction='none')  # Per-sample loss
        
        if self.class_weights is not None:
            # Apply class weights
            cls_loss = cls_loss * self.class_weights[cls_targets]
        
        # Average classification loss over the batch
        cls_loss = cls_loss.mean()
        
        # Compute bounding box loss (assuming you use L1 or GIoU loss)
        bbox_loss = F.l1_loss(bbox_preds, targets[:, 1:], reduction='mean')
        
        # Compute objectness loss (if applicable)
        # Example:
        # obj_loss = F.binary_cross_entropy_with_logits(predictions['obj'], targets[:, 5], reduction='mean')
        
        # Total loss (sum or weighted sum of different components)
        total_loss = cls_loss + bbox_loss  # + obj_loss if applicable
        
        # Collect individual loss components for logging
        loss_items = {
            'cls_loss': cls_loss,
            'bbox_loss': bbox_loss,
            # 'obj_loss': obj_loss
        }

        return total_loss, loss_items

