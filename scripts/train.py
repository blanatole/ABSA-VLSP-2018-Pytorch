#!/usr/bin/env python3
"""
Training script for PyTorch ABSA VLSP 2018
Reproduces results from the original TensorFlow implementation
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score
import json
from pathlib import Path
import logging
from datetime import datetime

# Add src and utils to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))

from data_processing import (
    VLSP2018Dataset, create_dataloaders, 
    VietnameseTextPreprocessor, PolarityMapping
)
from models import VLSP2018Model, VLSP2018Loss, count_parameters
from config_utils import load_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AspectTrainer:
    """Trainer for ABSA VLSP 2018 models"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Setup domain-specific paths
        self.domain = config['data'].get('domain', 'hotel')
        self.approach = config['model']['approach']
        self.model_name = f"{self.domain}_{self.approach}"
        
        # Create domain-specific directories
        self.model_dir = f"models/{self.domain}"
        self.results_dir = f"results/{self.domain}"
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        logger.info(f"Domain: {self.domain}")
        logger.info(f"Approach: {self.approach}")
        logger.info(f"Model will be saved to: {self.model_dir}")
        logger.info(f"Results will be saved to: {self.results_dir}")
        
        # Create model
        self.model = VLSP2018Model(
            pretrained_model_name=config['model']['pretrained_model_name'],
            aspect_categories=config['aspect_categories'],
            approach=config['model']['approach'],
            num_last_layers=config['model'].get('num_last_layers', 4),
            dropout_rate=config['model'].get('dropout_rate', 0.2)
        ).to(self.device)
        
        # Loss function
        self.loss_fn = VLSP2018Loss(
            approach=config['model']['approach'],
            num_aspects=len(config['aspect_categories'])
        )
        
        # Optimizer and scheduler
        learning_rate = float(config['training']['learning_rate'])
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=config['training'].get('weight_decay', 0.01)
        )
        
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_f1_acd = 0.0
        self.best_f1_acd_spc = 0.0
        
        logger.info(f"Model created with {count_parameters(self.model)['total_parameters']:,} parameters")
    
    def train_epoch(self, dataloader):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc="Training")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(input_ids, attention_mask)
            loss = self.loss_fn(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def evaluate(self, dataloader, split_name="validation"):
        """Evaluate model and compute metrics"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc=f"Evaluating {split_name}")
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask)
                loss = self.loss_fn(outputs, labels)
                
                total_loss += loss.item()
                num_batches += 1
                
                # Collect predictions and targets
                if self.config['model']['approach'] == 'multitask':
                    # For multitask: outputs shape [batch_size, num_aspects * 4]
                    batch_preds = self._extract_multitask_predictions(outputs)
                    batch_targets = self._extract_multitask_targets(labels)
                else:
                    # For multibranch: outputs is list of [batch_size, 4] tensors
                    batch_preds = self._extract_multibranch_predictions(outputs)
                    batch_targets = self._extract_multibranch_targets(labels)
                
                all_predictions.extend(batch_preds)
                all_targets.extend(batch_targets)
        
        avg_loss = total_loss / num_batches
        
        # Compute metrics
        metrics = self._compute_metrics(all_predictions, all_targets)
        
        return avg_loss, metrics
    
    def _extract_multitask_predictions(self, outputs):
        """Extract predictions from multitask outputs"""
        predictions = []
        num_aspects = len(self.config['aspect_categories'])
        
        # Reshape outputs: [batch_size, num_aspects * 4] -> [batch_size, num_aspects, 4]
        outputs_reshaped = outputs.view(-1, num_aspects, 4)
        
        for batch_idx in range(outputs.shape[0]):
            batch_pred = {}
            for aspect_idx, aspect in enumerate(self.config['aspect_categories']):
                # Get probabilities for this aspect
                aspect_logits = outputs_reshaped[batch_idx, aspect_idx]
                aspect_probs = torch.softmax(aspect_logits, dim=0)
                
                # Find predicted class (0: negative, 1: neutral, 2: positive, 3: none)
                pred_class = torch.argmax(aspect_probs).item()
                confidence = aspect_probs[pred_class].item()
                
                batch_pred[aspect] = {
                    'predicted_class': pred_class,
                    'confidence': confidence
                }
            
            predictions.append(batch_pred)
        
        return predictions
    
    def _extract_multibranch_predictions(self, outputs):
        """Extract predictions from multibranch outputs"""
        predictions = []
        batch_size = outputs[0].shape[0]
        
        for batch_idx in range(batch_size):
            batch_pred = {}
            for aspect_idx, aspect in enumerate(self.config['aspect_categories']):
                # Get probabilities for this aspect
                aspect_logits = outputs[aspect_idx][batch_idx]
                aspect_probs = torch.softmax(aspect_logits, dim=0)
                
                # Find predicted class
                pred_class = torch.argmax(aspect_probs).item()
                confidence = aspect_probs[pred_class].item()
                
                batch_pred[aspect] = {
                    'predicted_class': pred_class,
                    'confidence': confidence
                }
            
            predictions.append(batch_pred)
        
        return predictions
    
    def _extract_multitask_targets(self, labels):
        """Extract targets from multitask labels"""
        targets = []
        num_aspects = len(self.config['aspect_categories'])
        
        # labels shape: [batch_size, num_aspects * 4]
        labels_reshaped = labels.view(-1, num_aspects, 4)
        
        for batch_idx in range(labels.shape[0]):
            batch_target = {}
            for aspect_idx, aspect in enumerate(self.config['aspect_categories']):
                # Find true class from one-hot encoding
                aspect_labels = labels_reshaped[batch_idx, aspect_idx]
                true_class = torch.argmax(aspect_labels).item()
                
                batch_target[aspect] = true_class
            
            targets.append(batch_target)
        
        return targets
    
    def _extract_multibranch_targets(self, labels):
        """Extract targets from multibranch labels"""
        targets = []
        batch_size = labels.shape[0]
        
        for batch_idx in range(batch_size):
            batch_target = {}
            for aspect_idx, aspect in enumerate(self.config['aspect_categories']):
                # labels shape: [batch_size, num_aspects]
                true_class = labels[batch_idx, aspect_idx].item()
                batch_target[aspect] = true_class
            
            targets.append(batch_target)
        
        return targets
    
    def _compute_metrics(self, predictions, targets):
        """Compute ACD and ACD+SPC metrics like original paper"""
        aspect_categories = self.config['aspect_categories']
        
        # Initialize metrics storage
        acd_metrics = {'y_true': [], 'y_pred': []}  # Aspect Category Detection
        acd_spc_metrics = {'y_true': [], 'y_pred': []}  # ACD + Sentiment Polarity Classification
        
        for pred_sample, target_sample in zip(predictions, targets):
            for aspect in aspect_categories:
                pred_info = pred_sample[aspect]
                true_class = target_sample[aspect]
                pred_class = pred_info['predicted_class']
                
                # ACD: Check if aspect is mentioned (class != 3 means mentioned)
                acd_true = 1 if true_class != 3 else 0  # 1 if mentioned, 0 if not
                acd_pred = 1 if pred_class != 3 else 0
                
                acd_metrics['y_true'].append(acd_true)
                acd_metrics['y_pred'].append(acd_pred)
                
                # ACD+SPC: Only consider mentioned aspects for sentiment classification
                if true_class != 3:  # If aspect is actually mentioned
                    acd_spc_metrics['y_true'].append(true_class)
                    acd_spc_metrics['y_pred'].append(pred_class if pred_class != 3 else 0)  # Default to negative if wrongly predicted as none
        
        # Compute F1 scores
        acd_f1 = f1_score(acd_metrics['y_true'], acd_metrics['y_pred'], average='weighted')
        
        if len(acd_spc_metrics['y_true']) > 0:
            acd_spc_f1 = f1_score(acd_spc_metrics['y_true'], acd_spc_metrics['y_pred'], average='weighted')
        else:
            acd_spc_f1 = 0.0
        
        # Additional detailed metrics
        aspect_f1_scores = {}
        for aspect in aspect_categories:
            aspect_y_true = []
            aspect_y_pred = []
            
            for pred_sample, target_sample in zip(predictions, targets):
                true_class = target_sample[aspect]
                pred_class = pred_sample[aspect]['predicted_class']
                
                aspect_y_true.append(true_class)
                aspect_y_pred.append(pred_class)
            
            try:
                aspect_f1 = f1_score(aspect_y_true, aspect_y_pred, average='weighted')
                aspect_f1_scores[aspect] = aspect_f1
            except:
                aspect_f1_scores[aspect] = 0.0
        
        metrics = {
            'acd_f1': acd_f1,
            'acd_spc_f1': acd_spc_f1,
            'aspect_f1_scores': aspect_f1_scores,
            'average_aspect_f1': np.mean(list(aspect_f1_scores.values()))
        }
        
        return metrics
    
    def train(self, train_loader, val_loader, test_loader=None):
        """Main training loop"""
        num_epochs = self.config['training']['num_epochs']
        patience = self.config['training'].get('early_stopping_patience', 5)
        
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Batch size: {self.config['training']['batch_size']}")
        logger.info(f"Learning rate: {self.config['training']['learning_rate']}")
        
        best_epoch = 0
        patience_counter = 0
        
        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Training
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss, val_metrics = self.evaluate(val_loader, "validation")
            self.val_losses.append(val_loss)
            
            # Scheduler step
            self.scheduler.step(val_loss)
            
            # Logging
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}")
            logger.info(f"Val ACD F1: {val_metrics['acd_f1']:.4f}")
            logger.info(f"Val ACD+SPC F1: {val_metrics['acd_spc_f1']:.4f}")
            logger.info(f"Average Aspect F1: {val_metrics['average_aspect_f1']:.4f}")
            
            # Save best model
            is_best = val_metrics['acd_f1'] > self.best_f1_acd
            if is_best:
                self.best_f1_acd = val_metrics['acd_f1']
                self.best_f1_acd_spc = val_metrics['acd_spc_f1']
                best_epoch = epoch + 1
                patience_counter = 0
                
                # Save best model
                model_filename = f"best_{self.model_name}_model.pt"
                self.save_model(model_filename, epoch, val_metrics)
                logger.info(f"New best model saved! ACD F1: {self.best_f1_acd:.4f}")
                logger.info(f"Model saved as: {self.model_dir}/{model_filename}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {patience} epochs without improvement")
                break
        
        # Test evaluation
        if test_loader is not None:
            logger.info("\nEvaluating on test set...")
            model_filename = f"best_{self.model_name}_model.pt"
            self.load_model(model_filename)
            test_loss, test_metrics = self.evaluate(test_loader, "test")
            
            logger.info(f"\n=== FINAL TEST RESULTS ===")
            logger.info(f"Test Loss: {test_loss:.4f}")
            logger.info(f"Test ACD F1: {test_metrics['acd_f1']:.4f}")
            logger.info(f"Test ACD+SPC F1: {test_metrics['acd_spc_f1']:.4f}")
            logger.info(f"Average Aspect F1: {test_metrics['average_aspect_f1']:.4f}")
            
            # Save detailed results with timestamp and domain
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            results = {
                'domain': self.domain,
                'approach': self.approach,
                'model_name': self.model_name,
                'timestamp': timestamp,
                'best_epoch': best_epoch,
                'test_metrics': test_metrics,
                'training_history': {
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses
                },
                'config': self.config,
                'model_parameters': count_parameters(self.model)
            }
            
            # Save detailed results
            results_filename = f"{self.results_dir}/test_results_{self.model_name}_{timestamp}.json"
            with open(results_filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Detailed results saved to: {results_filename}")
            
            # Also save a latest results file for easy access
            latest_results_filename = f"{self.results_dir}/latest_results_{self.model_name}.json"
            with open(latest_results_filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Latest results saved to: {latest_results_filename}")
            
            return test_metrics
        
        return {'acd_f1': self.best_f1_acd, 'acd_spc_f1': self.best_f1_acd_spc}
    
    def save_model(self, filepath, epoch, metrics):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'domain': self.domain,
            'approach': self.approach,
            'model_name': self.model_name
        }
        
        torch.save(checkpoint, f'{self.model_dir}/{filepath}')
    
    def load_model(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(f'{self.model_dir}/{filepath}', map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])


def main():
    parser = argparse.ArgumentParser(description='Train ABSA VLSP 2018 model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    
    # Create tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model']['pretrained_model_name'])
    
    # CSV file paths - dynamic based on domain
    data_dir = config['data']['data_dir']
    domain = config['data'].get('domain', 'hotel')
    domain_name = domain.title()  # Hotel or Restaurant
    
    train_csv = f"{data_dir}/1-VLSP2018-SA-{domain_name}-train.csv"
    val_csv = f"{data_dir}/2-VLSP2018-SA-{domain_name}-dev.csv"
    test_csv = f"{data_dir}/3-VLSP2018-SA-{domain_name}-test.csv"
    
    train_loader, val_loader, test_loader = create_dataloaders(
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        tokenizer=tokenizer,
        aspect_categories=config['aspect_categories'],
        preprocessor=None,  # Will create default preprocessor
        batch_size=config['training']['batch_size'],
        max_length=config['model'].get('max_length', 128),
        approach=config['model']['approach'],
        num_workers=config['training'].get('num_workers', 4)
    )
    
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")
    
    # Create trainer
    trainer = AspectTrainer(config)
    
    # Resume training if specified
    if args.resume:
        logger.info(f"Resuming training from {args.resume}")
        trainer.load_model(args.resume)
    
    # Train model
    final_metrics = trainer.train(train_loader, val_loader, test_loader)
    
    # Print comparison with original paper
    logger.info("\n" + "="*60)
    logger.info("COMPARISON WITH ORIGINAL PAPER (TensorFlow)")
    logger.info("="*60)
    logger.info("Original Paper Results (Hotel Domain):")
    logger.info("  ACD F1-score: 82.55%")
    logger.info("  ACD+SPC F1-score: 77.32%")
    logger.info("-"*60)
    logger.info("Our PyTorch Results:")
    logger.info(f"  ACD F1-score: {final_metrics['acd_f1']*100:.2f}%")
    logger.info(f"  ACD+SPC F1-score: {final_metrics['acd_spc_f1']*100:.2f}%")
    logger.info("-"*60)
    
    acd_diff = (final_metrics['acd_f1'] - 0.8255) * 100
    acd_spc_diff = (final_metrics['acd_spc_f1'] - 0.7732) * 100
    
    logger.info("Performance Difference:")
    logger.info(f"  ACD: {acd_diff:+.2f}% {'(Better)' if acd_diff > 0 else '(Worse)'}")
    logger.info(f"  ACD+SPC: {acd_spc_diff:+.2f}% {'(Better)' if acd_spc_diff > 0 else '(Worse)'}")
    logger.info("="*60)


if __name__ == '__main__':
    main() 