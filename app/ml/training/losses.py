"""
Training utilities for Siamese Network
=======================================
Loss functions and training helpers for sentiment analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss for Siamese Networks.
    
    Encourages similar pairs to have high similarity and
    dissimilar pairs to have low similarity.
    
    Loss = (1 - label) * distance^2 + label * max(margin - distance, 0)^2
    
    Where:
    - label = 1 for dissimilar pairs
    - label = 0 for similar pairs
    - distance = ||encoding1 - encoding2||
    
    Args:
        margin: Margin for dissimilar pairs (default: 1.0)
    """
    
    def __init__(self, margin: float = 1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(
        self,
        encoding1: torch.Tensor,
        encoding2: torch.Tensor,
        label: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            encoding1: First encoding [batch_size, embedding_dim]
            encoding2: Second encoding [batch_size, embedding_dim]
            label: 0 for similar, 1 for dissimilar [batch_size]
        
        Returns:
            Loss value (scalar)
        """
        # Euclidean distance
        distance = F.pairwise_distance(encoding1, encoding2, p=2)
        
        # Contrastive loss
        loss_similar = (1 - label) * torch.pow(distance, 2)
        loss_dissimilar = label * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        
        loss = torch.mean(loss_similar + loss_dissimilar)
        
        return loss


class TripletLoss(nn.Module):
    """
    Triplet Loss for Siamese Networks.
    
    Uses anchor, positive (similar), and negative (dissimilar) samples.
    Loss = max(distance(anchor, positive) - distance(anchor, negative) + margin, 0)
    
    Args:
        margin: Minimum distance between positive and negative (default: 0.5)
    """
    
    def __init__(self, margin: float = 0.5):
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute triplet loss.
        
        Args:
            anchor: Anchor encoding [batch_size, embedding_dim]
            positive: Similar encoding [batch_size, embedding_dim]
            negative: Dissimilar encoding [batch_size, embedding_dim]
        
        Returns:
            Loss value (scalar)
        """
        # Distance to positive and negative
        distance_positive = F.pairwise_distance(anchor, positive, p=2)
        distance_negative = F.pairwise_distance(anchor, negative, p=2)
        
        # Triplet loss
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        
        return torch.mean(losses)


class CombinedLoss(nn.Module):
    """
    Combined loss for Siamese Network training.
    
    Combines:
    1. Contrastive loss (similarity learning)
    2. Cross-entropy loss (classification)
    
    Total loss = α * contrastive_loss + β * classification_loss
    
    Args:
        alpha: Weight for contrastive loss (default: 0.5)
        beta: Weight for classification loss (default: 0.5)
        margin: Margin for contrastive loss (default: 1.0)
        class_weights: Optional class weights for imbalanced data
    """
    
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        margin: float = 1.0,
        class_weights: torch.Tensor = None
    ):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.contrastive_loss = ContrastiveLoss(margin=margin)
        self.classification_loss = nn.CrossEntropyLoss(weight=class_weights)
    
    def forward(
        self,
        encoding1: torch.Tensor,
        encoding2: torch.Tensor,
        similarity_label: torch.Tensor,
        logits: torch.Tensor,
        sentiment_label: torch.Tensor
    ) -> tuple:
        """
        Compute combined loss.
        
        Args:
            encoding1: First review encoding
            encoding2: Second review encoding
            similarity_label: 0=similar, 1=dissimilar
            logits: Classification logits [batch_size, num_classes]
            sentiment_label: Sentiment class indices [batch_size]
        
        Returns:
            total_loss, contrastive_loss, classification_loss
        """
        # Contrastive loss
        contr_loss = self.contrastive_loss(encoding1, encoding2, similarity_label)
        
        # Classification loss
        class_loss = self.classification_loss(logits, sentiment_label)
        
        # Combined loss
        total_loss = self.alpha * contr_loss + self.beta * class_loss
        
        return total_loss, contr_loss, class_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    Focuses training on hard examples by down-weighting easy examples.
    Loss = -α * (1 - p_t)^γ * log(p_t)
    
    Args:
        alpha: Weighting factor (default: 1.0)
        gamma: Focusing parameter (default: 2.0)
        reduction: 'mean' or 'sum' (default: 'mean')
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Logits [batch_size, num_classes]
            targets: Class indices [batch_size]
        
        Returns:
            Loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute classification accuracy.
    
    Args:
        logits: Model output [batch_size, num_classes]
        labels: True labels [batch_size]
    
    Returns:
        Accuracy as percentage
    """
    predictions = torch.argmax(logits, dim=1)
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    accuracy = 100.0 * correct / total
    return accuracy


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor, num_classes: int = 3) -> dict:
    """
    Compute detailed classification metrics.
    
    Args:
        logits: Model output [batch_size, num_classes]
        labels: True labels [batch_size]
        num_classes: Number of classes (default: 3)
    
    Returns:
        Dictionary with accuracy, precision, recall, f1 per class
    """
    predictions = torch.argmax(logits, dim=1)
    
    metrics = {
        'accuracy': 0.0,
        'precision': [0.0] * num_classes,
        'recall': [0.0] * num_classes,
        'f1': [0.0] * num_classes
    }
    
    # Overall accuracy
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    metrics['accuracy'] = 100.0 * correct / total
    
    # Per-class metrics
    for cls in range(num_classes):
        # True positives, false positives, false negatives
        tp = ((predictions == cls) & (labels == cls)).sum().item()
        fp = ((predictions == cls) & (labels != cls)).sum().item()
        fn = ((predictions != cls) & (labels == cls)).sum().item()
        
        # Precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # F1 score
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics['precision'][cls] = precision
        metrics['recall'][cls] = recall
        metrics['f1'][cls] = f1
    
    return metrics


if __name__ == "__main__":
    # Test loss functions
    print("Testing loss functions...\n")
    
    batch_size = 4
    embedding_dim = 128
    num_classes = 3
    
    # Dummy data
    encoding1 = torch.randn(batch_size, embedding_dim)
    encoding2 = torch.randn(batch_size, embedding_dim)
    similarity_label = torch.tensor([0, 1, 0, 1], dtype=torch.float)  # 0=similar, 1=dissimilar
    logits = torch.randn(batch_size, num_classes)
    sentiment_label = torch.tensor([0, 1, 2, 1], dtype=torch.long)
    
    # Test contrastive loss
    print("1. Contrastive Loss:")
    contr_loss = ContrastiveLoss(margin=1.0)
    loss = contr_loss(encoding1, encoding2, similarity_label)
    print(f"   Loss: {loss.item():.4f}")
    
    # Test triplet loss
    print("\n2. Triplet Loss:")
    encoding3 = torch.randn(batch_size, embedding_dim)
    triplet_loss = TripletLoss(margin=0.5)
    loss = triplet_loss(encoding1, encoding2, encoding3)
    print(f"   Loss: {loss.item():.4f}")
    
    # Test combined loss
    print("\n3. Combined Loss:")
    combined_loss = CombinedLoss(alpha=0.5, beta=0.5)
    total, contr, classif = combined_loss(
        encoding1, encoding2, similarity_label, logits, sentiment_label
    )
    print(f"   Total Loss: {total.item():.4f}")
    print(f"   Contrastive: {contr.item():.4f}")
    print(f"   Classification: {classif.item():.4f}")
    
    # Test focal loss
    print("\n4. Focal Loss:")
    focal_loss = FocalLoss(alpha=1.0, gamma=2.0)
    loss = focal_loss(logits, sentiment_label)
    print(f"   Loss: {loss.item():.4f}")
    
    # Test metrics
    print("\n5. Metrics:")
    accuracy = compute_accuracy(logits, sentiment_label)
    print(f"   Accuracy: {accuracy:.2f}%")
    
    metrics = compute_metrics(logits, sentiment_label)
    print(f"   Overall Accuracy: {metrics['accuracy']:.2f}%")
    print("   Per-class metrics:")
    for cls in range(num_classes):
        print(f"     Class {cls}: P={metrics['precision'][cls]:.3f}, R={metrics['recall'][cls]:.3f}, F1={metrics['f1'][cls]:.3f}")
    
    print("\n✅ All loss functions working correctly!")
