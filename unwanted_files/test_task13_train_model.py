"""
Task 13: Train Siamese Network on Telugu Movie Reviews
======================================================
Complete training pipeline with validation and model saving
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
import sys
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# Add app to path
sys.path.append('.')

from app.ml.models.siamese_network import SiameseNetwork, create_siamese_model
from app.ml.training.losses import CombinedLoss, compute_accuracy, compute_metrics
from app.ml.training.data_loader import create_data_loaders


class TeluguTokenizer:
    """Telugu text tokenizer with vocabulary building."""
    
    def __init__(self, vocab_size: int = 5000):
        self.vocab_size = vocab_size
        self.word2idx = {"<PAD>": 0, "<UNK>": 1, "<START>": 2, "<END>": 3}
        self.idx2word = {0: "<PAD>", 1: "<UNK>", 2: "<START>", 3: "<END>"}
        self.vocab_built = False
    
    def build_vocab(self, texts: List[str]):
        """Build vocabulary from texts."""
        from collections import Counter
        word_counts = Counter()
        for text in texts:
            tokens = text.split()
            word_counts.update(tokens)
        
        most_common = word_counts.most_common(self.vocab_size - 4)
        
        for word, _ in most_common:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        
        self.vocab_built = True
    
    def encode(self, text: str, max_length: int = 50):
        """Encode text to token indices."""
        words = text.split()
        tokens = [self.word2idx.get(word, self.word2idx["<UNK>"]) for word in words]
        length = min(len(tokens), max_length)
        
        if len(tokens) < max_length:
            tokens = tokens + [self.word2idx["<PAD>"]] * (max_length - len(tokens))
        else:
            tokens = tokens[:max_length]
        
        return tokens, length
    
    def save(self, filepath: str):
        """Save tokenizer vocabulary."""
        data = {
            'vocab_size': self.vocab_size,
            'word2idx': self.word2idx,
            'idx2word': {int(k): v for k, v in self.idx2word.items()}
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, filepath: str):
        """Load tokenizer from file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tokenizer = cls(vocab_size=data['vocab_size'])
        tokenizer.word2idx = data['word2idx']
        tokenizer.idx2word = {int(k): v for k, v in data['idx2word'].items()}
        tokenizer.vocab_built = True
        return tokenizer


class Trainer:
    """Trainer for Siamese Network."""
    
    def __init__(
        self,
        model: SiameseNetwork,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        device: str = 'cpu',
        checkpoint_dir: str = 'checkpoints'
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_contr_loss = 0.0
        total_class_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        
        for batch in pbar:
            # Move to device
            review1 = batch['review1'].to(self.device)
            review2 = batch['review2'].to(self.device)
            length1 = batch['length1'].to(self.device)
            length2 = batch['length2'].to(self.device)
            similarity_label = batch['similarity_label'].to(self.device)
            sentiment_label = batch['sentiment_label'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Get encodings
            encoding1 = self.model.forward_once(review1, length1)
            encoding2 = self.model.forward_once(review2, length2)
            
            # Get logits
            logits = self.model.forward(review1, review2, length1, length2, return_similarity=False)
            
            # Compute loss
            loss, contr_loss, class_loss = self.loss_fn(
                encoding1, encoding2, similarity_label, logits, sentiment_label
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Metrics
            batch_size = review1.size(0)
            total_loss += loss.item() * batch_size
            total_contr_loss += contr_loss.item() * batch_size
            total_class_loss += class_loss.item() * batch_size
            
            predictions = torch.argmax(logits, dim=1)
            total_correct += (predictions == sentiment_label).sum().item()
            total_samples += batch_size
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * total_correct / total_samples:.2f}%'
            })
        
        # Average metrics
        avg_loss = total_loss / total_samples
        avg_contr_loss = total_contr_loss / total_samples
        avg_class_loss = total_class_loss / total_samples
        avg_acc = 100.0 * total_correct / total_samples
        
        return {
            'loss': avg_loss,
            'contr_loss': avg_contr_loss,
            'class_loss': avg_class_loss,
            'accuracy': avg_acc
        }
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        all_logits = []
        all_labels = []
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')
        
        with torch.no_grad():
            for batch in pbar:
                # Move to device
                review1 = batch['review1'].to(self.device)
                review2 = batch['review2'].to(self.device)
                length1 = batch['length1'].to(self.device)
                length2 = batch['length2'].to(self.device)
                similarity_label = batch['similarity_label'].to(self.device)
                sentiment_label = batch['sentiment_label'].to(self.device)
                
                # Forward pass
                encoding1 = self.model.forward_once(review1, length1)
                encoding2 = self.model.forward_once(review2, length2)
                logits = self.model.forward(review1, review2, length1, length2, return_similarity=False)
                
                # Compute loss
                loss, _, _ = self.loss_fn(
                    encoding1, encoding2, similarity_label, logits, sentiment_label
                )
                
                # Metrics
                batch_size = review1.size(0)
                total_loss += loss.item() * batch_size
                
                predictions = torch.argmax(logits, dim=1)
                total_correct += (predictions == sentiment_label).sum().item()
                total_samples += batch_size
                
                # Store for detailed metrics
                all_logits.append(logits.cpu())
                all_labels.append(sentiment_label.cpu())
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.0 * total_correct / total_samples:.2f}%'
                })
        
        # Average metrics
        avg_loss = total_loss / total_samples
        avg_acc = 100.0 * total_correct / total_samples
        
        # Detailed metrics
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        detailed_metrics = compute_metrics(all_logits, all_labels, num_classes=3)
        
        return {
            'loss': avg_loss,
            'accuracy': avg_acc,
            'detailed_metrics': detailed_metrics
        }
    
    def train(self, num_epochs: int, patience: int = 5):
        """Complete training loop."""
        print("\n" + "=" * 70)
        print("TRAINING SIAMESE NETWORK")
        print("=" * 70)
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(epoch)
            
            # Update learning rate
            self.scheduler.step(val_metrics['loss'])
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Store history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['learning_rate'].append(current_lr)
            
            # Print epoch summary
            print(f"\nEpoch {epoch}/{num_epochs}:")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%")
            print(f"  LR: {current_lr:.6f}")
            
            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_val_acc = val_metrics['accuracy']
                self.save_checkpoint(epoch, is_best=True)
                print(f"  ✓ Best model saved! Val Loss: {val_metrics['loss']:.4f}")
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                print(f"  No improvement ({self.patience_counter}/{patience})")
            
            # Early stopping
            if self.patience_counter >= patience:
                print(f"\n⚠ Early stopping triggered after {epoch} epochs")
                break
            
            # Regular checkpoint
            if epoch % 5 == 0:
                self.save_checkpoint(epoch, is_best=False)
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETED!")
        print("=" * 70)
        print(f"Best Val Loss: {self.best_val_loss:.4f}")
        print(f"Best Val Acc: {self.best_val_acc:.2f}%")
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }
        
        if is_best:
            filepath = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, filepath)
        else:
            filepath = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save(checkpoint, filepath)
    
    def plot_training_history(self, save_path: str = 'training_history.png'):
        """Plot training curves."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        axes[0].plot(self.history['train_loss'], label='Train Loss', marker='o')
        axes[0].plot(self.history['val_loss'], label='Val Loss', marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy plot
        axes[1].plot(self.history['train_acc'], label='Train Acc', marker='o')
        axes[1].plot(self.history['val_acc'], label='Val Acc', marker='s')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Training history plot saved to {save_path}")


def main():
    """Main training function."""
    print("=" * 70)
    print("TASK 13: SIAMESE NETWORK TRAINING")
    print("=" * 70)
    
    # Configuration
    config = {
        'vocab_size': 5000,
        'embedding_dim': 128,
        'hidden_dim': 256,
        'num_layers': 2,
        'dropout': 0.3,
        'batch_size': 16,
        'learning_rate': 0.001,
        'num_epochs': 20,
        'patience': 5,
        'max_length': 50,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"\nDevice: {config['device']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Max epochs: {config['num_epochs']}")
    
    # Load dataset
    print("\n" + "=" * 70)
    print("LOADING DATASET")
    print("=" * 70)
    
    data_dir = Path("data/telugu_reviews")
    with open(data_dir / "train.json", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(data_dir / "validation.json", encoding="utf-8") as f:
        val_data = json.load(f)
    
    print(f"✓ Train reviews: {len(train_data)}")
    print(f"✓ Val reviews: {len(val_data)}")
    
    # Build tokenizer
    print("\n" + "=" * 70)
    print("BUILDING VOCABULARY")
    print("=" * 70)
    
    tokenizer = TeluguTokenizer(vocab_size=config['vocab_size'])
    train_texts = [r['text'] for r in train_data]
    tokenizer.build_vocab(train_texts)
    
    print(f"✓ Vocabulary size: {len(tokenizer.word2idx)}")
    
    # Save tokenizer
    tokenizer_path = "checkpoints/tokenizer.json"
    Path("checkpoints").mkdir(exist_ok=True)
    tokenizer.save(tokenizer_path)
    print(f"✓ Tokenizer saved to {tokenizer_path}")
    
    # Create data loaders
    print("\n" + "=" * 70)
    print("CREATING DATA LOADERS")
    print("=" * 70)
    
    train_loader, val_loader = create_data_loaders(
        train_data,
        val_data,
        tokenizer,
        batch_size=config['batch_size'],
        max_length=config['max_length']
    )
    
    # Create model
    print("\n" + "=" * 70)
    print("CREATING MODEL")
    print("=" * 70)
    
    model = create_siamese_model(
        vocab_size=len(tokenizer.word2idx),
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        device=config['device']
    )
    
    # Create optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # Create loss function
    loss_fn = CombinedLoss(alpha=0.5, beta=0.5, margin=1.0)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=config['device'],
        checkpoint_dir='checkpoints'
    )
    
    # Train
    trainer.train(num_epochs=config['num_epochs'], patience=config['patience'])
    
    # Plot training history
    trainer.plot_training_history('checkpoints/training_history.png')
    
    # Save final training info
    training_info = {
        'config': config,
        'best_val_loss': trainer.best_val_loss,
        'best_val_acc': trainer.best_val_acc,
        'total_epochs': len(trainer.history['train_loss']),
        'vocab_size': len(tokenizer.word2idx)
    }
    
    with open('checkpoints/training_info.json', 'w') as f:
        json.dump(training_info, f, indent=2)
    
    print("\n✅ Task 13 completed successfully!")
    print(f"   Best model: checkpoints/best_model.pt")
    print(f"   Tokenizer: checkpoints/tokenizer.json")
    print(f"   Training history: checkpoints/training_history.png")


if __name__ == "__main__":
    main()
