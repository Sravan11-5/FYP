"""
Task 14: Evaluate Siamese Network Model Performance
====================================================
Comprehensive evaluation on test dataset
"""

import torch
import json
import sys
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score,
    confusion_matrix,
    classification_report
)

# Add app to path
sys.path.append('.')

from app.ml.models.siamese_network import SiameseNetwork, create_siamese_model


class TeluguTokenizer:
    """Telugu text tokenizer."""
    
    @classmethod
    def load(cls, filepath: str):
        """Load tokenizer from file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tokenizer = cls()
        tokenizer.vocab_size = data['vocab_size']
        tokenizer.word2idx = data['word2idx']
        tokenizer.idx2word = {int(k): v for k, v in data['idx2word'].items()}
        return tokenizer
    
    def encode(self, text: str, max_length: int = 50):
        """Encode text to token indices."""
        words = text.split()
        tokens = [self.word2idx.get(word, self.word2idx.get("<UNK>", 1)) for word in words]
        length = min(len(tokens), max_length)
        
        if len(tokens) < max_length:
            tokens = tokens + [self.word2idx.get("<PAD>", 0)] * (max_length - len(tokens))
        else:
            tokens = tokens[:max_length]
        
        return tokens, length


class ModelEvaluator:
    """Evaluate trained Siamese Network."""
    
    def __init__(
        self,
        model: SiameseNetwork,
        tokenizer: TeluguTokenizer,
        device: str = 'cpu'
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Metrics storage
        self.predictions = []
        self.true_labels = []
        self.inference_times = []
        self.confidences = []
    
    def evaluate_dataset(
        self,
        test_data: List[Dict],
        max_length: int = 50
    ) -> Dict:
        """Evaluate model on test dataset."""
        print("\n" + "=" * 70)
        print("EVALUATING MODEL ON TEST DATASET")
        print("=" * 70)
        print(f"Test samples: {len(test_data)}")
        
        self.predictions = []
        self.true_labels = []
        self.inference_times = []
        self.confidences = []
        
        # Sentiment mapping (MUST match training mapping!)
        sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        
        # Process each review
        for i, review in enumerate(test_data):
            # Encode review
            tokens, length = self.tokenizer.encode(review['text'], max_length)
            
            # Convert to tensor
            review_tensor = torch.tensor([tokens], dtype=torch.long).to(self.device)
            length_tensor = torch.tensor([length], dtype=torch.long).to(self.device)
            
            # Measure inference time
            start_time = time.time()
            
            with torch.no_grad():
                # Get prediction using the model's predict_sentiment method
                logits = self.model.predict_sentiment(review_tensor, length_tensor)
                
                probabilities = torch.softmax(logits, dim=1)
                predicted_class = torch.argmax(logits, dim=1).item()
                confidence = probabilities[0, predicted_class].item()
            
            inference_time = (time.time() - start_time) * 1000  # ms
            
            # Convert true label to integer
            true_label = sentiment_map.get(review['sentiment'].lower(), review['sentiment'])
            
            # Store results
            self.predictions.append(predicted_class)
            self.true_labels.append(true_label)
            self.inference_times.append(inference_time)
            self.confidences.append(confidence)
            
            # Progress
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(test_data)} reviews...")
        
        print(f"✓ Evaluation complete!")
        
        # Calculate metrics
        metrics = self._calculate_metrics()
        
        return metrics
    
    def _calculate_metrics(self) -> Dict:
        """Calculate comprehensive metrics."""
        print("\n" + "=" * 70)
        print("CALCULATING METRICS")
        print("=" * 70)
        
        # Basic metrics
        accuracy = accuracy_score(self.true_labels, self.predictions)
        
        # Per-class metrics
        precision_macro = precision_score(self.true_labels, self.predictions, average='macro', zero_division=0)
        recall_macro = recall_score(self.true_labels, self.predictions, average='macro', zero_division=0)
        f1_macro = f1_score(self.true_labels, self.predictions, average='macro', zero_division=0)
        
        precision_weighted = precision_score(self.true_labels, self.predictions, average='weighted', zero_division=0)
        recall_weighted = recall_score(self.true_labels, self.predictions, average='weighted', zero_division=0)
        f1_weighted = f1_score(self.true_labels, self.predictions, average='weighted', zero_division=0)
        
        # Per-class detailed metrics
        precision_per_class = precision_score(self.true_labels, self.predictions, average=None, zero_division=0)
        recall_per_class = recall_score(self.true_labels, self.predictions, average=None, zero_division=0)
        f1_per_class = f1_score(self.true_labels, self.predictions, average=None, zero_division=0)
        
        # Inference time statistics
        avg_inference_time = np.mean(self.inference_times)
        median_inference_time = np.median(self.inference_times)
        max_inference_time = np.max(self.inference_times)
        min_inference_time = np.min(self.inference_times)
        
        # Confidence statistics
        avg_confidence = np.mean(self.confidences)
        
        metrics = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'avg_inference_time_ms': avg_inference_time,
            'median_inference_time_ms': median_inference_time,
            'max_inference_time_ms': max_inference_time,
            'min_inference_time_ms': min_inference_time,
            'avg_confidence': avg_confidence,
            'confusion_matrix': confusion_matrix(self.true_labels, self.predictions).tolist()
        }
        
        # Print summary
        print(f"\n{'Metric':<30} {'Value':<15}")
        print("-" * 45)
        print(f"{'Accuracy':<30} {accuracy * 100:.2f}%")
        print(f"{'F1-Score (Macro)':<30} {f1_macro:.4f}")
        print(f"{'F1-Score (Weighted)':<30} {f1_weighted:.4f}")
        print(f"{'Precision (Macro)':<30} {precision_macro:.4f}")
        print(f"{'Recall (Macro)':<30} {recall_macro:.4f}")
        print(f"{'Avg Inference Time':<30} {avg_inference_time:.2f} ms")
        print(f"{'Avg Confidence':<30} {avg_confidence * 100:.2f}%")
        
        # Per-class metrics
        print(f"\n{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-" * 51)
        class_names = ['Negative', 'Neutral', 'Positive']
        for i, name in enumerate(class_names):
            if i < len(precision_per_class):
                print(f"{name:<15} {precision_per_class[i]:.4f}       {recall_per_class[i]:.4f}       {f1_per_class[i]:.4f}")
        
        return metrics
    
    def plot_confusion_matrix(self, save_path: str = 'evaluation_results/confusion_matrix.png'):
        """Plot confusion matrix."""
        cm = confusion_matrix(self.true_labels, self.predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['Negative', 'Neutral', 'Positive'],
            yticklabels=['Negative', 'Neutral', 'Positive'],
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix - Test Set', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to {save_path}")
        plt.close()
    
    def plot_metrics_comparison(self, metrics: Dict, save_path: str = 'evaluation_results/metrics_comparison.png'):
        """Plot metrics comparison."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Per-class metrics
        class_names = ['Negative', 'Neutral', 'Positive']
        x = np.arange(len(class_names))
        width = 0.25
        
        axes[0, 0].bar(x - width, metrics['precision_per_class'], width, label='Precision', color='steelblue')
        axes[0, 0].bar(x, metrics['recall_per_class'], width, label='Recall', color='coral')
        axes[0, 0].bar(x + width, metrics['f1_per_class'], width, label='F1-Score', color='lightgreen')
        axes[0, 0].set_xlabel('Sentiment Class')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Per-Class Metrics', fontweight='bold')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(class_names)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim([0, 1.1])
        
        # 2. Overall metrics
        overall_metrics = {
            'Accuracy': metrics['accuracy'],
            'F1 (Macro)': metrics['f1_macro'],
            'F1 (Weighted)': metrics['f1_weighted'],
            'Precision': metrics['precision_macro'],
            'Recall': metrics['recall_macro']
        }
        
        axes[0, 1].barh(list(overall_metrics.keys()), list(overall_metrics.values()), color='mediumpurple')
        axes[0, 1].set_xlabel('Score')
        axes[0, 1].set_title('Overall Performance Metrics', fontweight='bold')
        axes[0, 1].set_xlim([0, 1.1])
        axes[0, 1].grid(True, alpha=0.3, axis='x')
        
        for i, v in enumerate(overall_metrics.values()):
            axes[0, 1].text(v + 0.02, i, f'{v:.3f}', va='center')
        
        # 3. Inference time distribution
        axes[1, 0].hist(self.inference_times, bins=20, color='teal', alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(metrics['avg_inference_time_ms'], color='red', linestyle='--', linewidth=2, label='Mean')
        axes[1, 0].axvline(metrics['median_inference_time_ms'], color='orange', linestyle='--', linewidth=2, label='Median')
        axes[1, 0].set_xlabel('Inference Time (ms)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Inference Time Distribution', fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Confidence distribution
        axes[1, 1].hist(self.confidences, bins=20, color='gold', alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(metrics['avg_confidence'], color='red', linestyle='--', linewidth=2, label=f'Mean: {metrics["avg_confidence"]:.3f}')
        axes[1, 1].set_xlabel('Prediction Confidence')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Prediction Confidence Distribution', fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xlim([0, 1])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Metrics comparison saved to {save_path}")
        plt.close()
    
    def analyze_errors(self, test_data: List[Dict], save_path: str = 'evaluation_results/error_analysis.txt'):
        """Analyze misclassified samples."""
        print("\n" + "=" * 70)
        print("ERROR ANALYSIS")
        print("=" * 70)
        
        errors = []
        class_names = ['Negative', 'Neutral', 'Positive']
        
        for i, (pred, true) in enumerate(zip(self.predictions, self.true_labels)):
            if pred != true:
                errors.append({
                    'index': i,
                    'text': test_data[i]['text'],
                    'true_label': class_names[true],
                    'predicted_label': class_names[pred],
                    'confidence': self.confidences[i],
                    'inference_time': self.inference_times[i]
                })
        
        print(f"Total errors: {len(errors)}/{len(test_data)} ({len(errors)/len(test_data)*100:.2f}%)")
        
        # Save error analysis
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("ERROR ANALYSIS\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Total Errors: {len(errors)}/{len(test_data)} ({len(errors)/len(test_data)*100:.2f}%)\n\n")
            
            for i, error in enumerate(errors, 1):
                f.write(f"\nError {i}:\n")
                f.write("-" * 70 + "\n")
                f.write(f"Review: {error['text'][:100]}...\n")
                f.write(f"True Label: {error['true_label']}\n")
                f.write(f"Predicted: {error['predicted_label']}\n")
                f.write(f"Confidence: {error['confidence']:.4f}\n")
                f.write(f"Inference Time: {error['inference_time']:.2f} ms\n")
        
        print(f"✓ Error analysis saved to {save_path}")
        
        return errors
    
    def generate_classification_report(self, save_path: str = 'evaluation_results/classification_report.txt'):
        """Generate detailed classification report."""
        class_names = ['Negative', 'Neutral', 'Positive']
        report = classification_report(
            self.true_labels, 
            self.predictions, 
            target_names=class_names,
            digits=4
        )
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("CLASSIFICATION REPORT\n")
            f.write("=" * 70 + "\n\n")
            f.write(report)
        
        print(f"✓ Classification report saved to {save_path}")
        
        return report


def main():
    """Main evaluation function."""
    print("=" * 70)
    print("TASK 14: MODEL EVALUATION")
    print("=" * 70)
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Load tokenizer
    print("\n" + "=" * 70)
    print("LOADING TOKENIZER")
    print("=" * 70)
    tokenizer = TeluguTokenizer.load('checkpoints/tokenizer.json')
    print(f"✓ Vocabulary size: {len(tokenizer.word2idx)}")
    
    # Load test data
    print("\n" + "=" * 70)
    print("LOADING TEST DATASET")
    print("=" * 70)
    with open('data/telugu_reviews/test.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    print(f"✓ Test samples: {len(test_data)}")
    
    # Create model
    print("\n" + "=" * 70)
    print("CREATING MODEL")
    print("=" * 70)
    model = create_siamese_model(
        vocab_size=len(tokenizer.word2idx),
        embedding_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.3,
        device=device
    )
    
    # Load trained weights
    print("\n" + "=" * 70)
    print("LOADING TRAINED MODEL")
    print("=" * 70)
    checkpoint = torch.load('checkpoints/best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Model loaded from epoch {checkpoint['epoch']}")
    print(f"✓ Best validation loss: {checkpoint['best_val_loss']:.6f}")
    print(f"✓ Best validation accuracy: {checkpoint['best_val_acc']:.2f}%")
    
    # Create evaluator
    evaluator = ModelEvaluator(model, tokenizer, device)
    
    # Evaluate
    metrics = evaluator.evaluate_dataset(test_data, max_length=50)
    
    # Generate visualizations and reports
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    evaluator.plot_confusion_matrix()
    evaluator.plot_metrics_comparison(metrics)
    errors = evaluator.analyze_errors(test_data)
    report = evaluator.generate_classification_report()
    
    # Save metrics
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    with open('evaluation_results/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("✓ Metrics saved to evaluation_results/metrics.json")
    
    # Performance summary
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"✓ Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"✓ F1-Score (Macro): {metrics['f1_macro']:.4f}")
    print(f"✓ F1-Score (Weighted): {metrics['f1_weighted']:.4f}")
    print(f"✓ Avg Inference Time: {metrics['avg_inference_time_ms']:.2f} ms")
    print(f"✓ Errors: {len(errors)}/{len(test_data)}")
    
    # Check targets
    print("\n" + "=" * 70)
    print("TARGET METRICS CHECK")
    print("=" * 70)
    
    target_accuracy = 0.85
    target_f1 = 0.80
    target_inference_time = 10.0  # ms
    
    accuracy_met = metrics['accuracy'] >= target_accuracy
    f1_met = metrics['f1_weighted'] >= target_f1
    time_met = metrics['avg_inference_time_ms'] <= target_inference_time
    
    print(f"{'Metric':<30} {'Target':<15} {'Actual':<15} {'Status':<10}")
    print("-" * 70)
    
    acc_target = f">= {target_accuracy * 100}%"
    acc_actual = f"{metrics['accuracy'] * 100:.2f}%"
    acc_status = "✓ PASS" if accuracy_met else "✗ FAIL"
    print(f"{'Accuracy':<30} {acc_target:<15} {acc_actual:<15} {acc_status:<10}")
    
    f1_target = f">= {target_f1}"
    f1_actual = f"{metrics['f1_weighted']:.4f}"
    f1_status = "✓ PASS" if f1_met else "✗ FAIL"
    print(f"{'F1-Score':<30} {f1_target:<15} {f1_actual:<15} {f1_status:<10}")
    
    time_target = f"< {target_inference_time} ms"
    time_actual = f"{metrics['avg_inference_time_ms']:.2f} ms"
    time_status = "✓ PASS" if time_met else "✗ FAIL"
    print(f"{'Inference Time':<30} {time_target:<15} {time_actual:<15} {time_status:<10}")
    
    print("\n" + "=" * 70)
    if accuracy_met and f1_met and time_met:
        print("✅ ALL TARGETS MET! MODEL IS READY FOR DEPLOYMENT!")
    else:
        print("⚠️  SOME TARGETS NOT MET. SEE IMPROVEMENT SUGGESTIONS.")
    print("=" * 70)
    
    print("\n✅ Task 14 completed successfully!")
    print("   Results saved to evaluation_results/")


if __name__ == "__main__":
    main()
