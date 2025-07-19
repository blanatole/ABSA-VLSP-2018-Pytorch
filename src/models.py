"""
PyTorch Model Definitions for ABSA VLSP 2018
Multi-task and Multi-branch approaches using PhoBERT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from typing import List, Dict, Optional, Tuple, Union


class PhoBERTEncoder(nn.Module):
    """PhoBERT encoder with concatenated last 4 layers"""
    
    def __init__(self, pretrained_model_name: str = "vinai/phobert-base", 
                 num_last_layers: int = 4, dropout_rate: float = 0.2):
        super(PhoBERTEncoder, self).__init__()
        
        self.pretrained_model_name = pretrained_model_name
        self.num_last_layers = num_last_layers
        self.dropout_rate = dropout_rate
        
        # Load PhoBERT with output_hidden_states=True
        self.config = AutoConfig.from_pretrained(pretrained_model_name)
        self.config.output_hidden_states = True
        self.phobert = AutoModel.from_pretrained(pretrained_model_name, config=self.config)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Calculate concatenated hidden size
        self.hidden_size = self.config.hidden_size * num_last_layers
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through PhoBERT encoder
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Concatenated hidden states [batch_size, hidden_size * num_last_layers]
        """
        # Get PhoBERT outputs
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states  # All layer outputs
        
        # Concatenate last num_last_layers layers
        last_layers = hidden_states[-self.num_last_layers:]  # Get last N layers
        concatenated = torch.cat(last_layers, dim=-1)  # [batch, seq_len, hidden*N]
        
        # Use [CLS] token representation
        cls_embedding = concatenated[:, 0, :]  # [batch, hidden*N]
        
        # Apply dropout
        cls_embedding = self.dropout(cls_embedding)
        
        return cls_embedding


class VLSP2018MultiTaskModel(nn.Module):
    """Multi-task approach: Concatenated outputs with binary cross-entropy"""
    
    def __init__(self, pretrained_model_name: str, aspect_categories: List[str],
                 num_last_layers: int = 4, dropout_rate: float = 0.2):
        super(VLSP2018MultiTaskModel, self).__init__()
        
        self.aspect_categories = aspect_categories
        self.num_aspects = len(aspect_categories)
        self.num_polarities = 4  # None, Positive, Negative, Neutral
        
        # PhoBERT encoder
        self.encoder = PhoBERTEncoder(
            pretrained_model_name=pretrained_model_name,
            num_last_layers=num_last_layers,
            dropout_rate=dropout_rate
        )
        
        # Individual dense layers for each aspect category
        self.aspect_classifiers = nn.ModuleList([
            nn.Linear(self.encoder.hidden_size, self.num_polarities)
            for _ in range(self.num_aspects)
        ])
        
        # Total output size for flattened one-hot labels
        self.total_output_size = self.num_aspects * self.num_polarities
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for multi-task approach
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Flattened predictions [batch_size, num_aspects * 4]
        """
        # Get encoded representation
        encoded = self.encoder(input_ids, attention_mask)  # [batch, hidden_size]
        
        # Get predictions for each aspect
        aspect_outputs = []
        for classifier in self.aspect_classifiers:
            output = classifier(encoded)  # [batch, 4]
            output = F.softmax(output, dim=-1)  # Apply softmax
            aspect_outputs.append(output)
        
        # Concatenate all outputs
        concatenated_output = torch.cat(aspect_outputs, dim=-1)  # [batch, num_aspects * 4]
        
        return concatenated_output
    
    def predict_aspects(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> List[Dict]:
        """
        Predict aspect categories and sentiments
        
        Returns:
            List of predictions for each sample in batch
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            batch_size = outputs.size(0)
            
            predictions = []
            for i in range(batch_size):
                sample_pred = {}
                output = outputs[i].reshape(self.num_aspects, self.num_polarities)
                
                for j, aspect_category in enumerate(self.aspect_categories):
                    probs = output[j]
                    predicted_class = torch.argmax(probs).item()
                    
                    # Map to polarity
                    polarity_mapping = {0: None, 1: 'positive', 2: 'negative', 3: 'neutral'}
                    polarity = polarity_mapping[predicted_class]
                    
                    if polarity is not None:  # Only include non-None predictions
                        sample_pred[aspect_category] = {
                            'polarity': polarity,
                            'confidence': probs[predicted_class].item()
                        }
                
                predictions.append(sample_pred)
            
            return predictions


class VLSP2018MultiBranchModel(nn.Module):
    """Multi-branch approach: Separate branches with sparse categorical cross-entropy"""
    
    def __init__(self, pretrained_model_name: str, aspect_categories: List[str],
                 num_last_layers: int = 4, dropout_rate: float = 0.2):
        super(VLSP2018MultiBranchModel, self).__init__()
        
        self.aspect_categories = aspect_categories
        self.num_aspects = len(aspect_categories)
        self.num_polarities = 4  # None, Positive, Negative, Neutral
        
        # PhoBERT encoder
        self.encoder = PhoBERTEncoder(
            pretrained_model_name=pretrained_model_name,
            num_last_layers=num_last_layers,
            dropout_rate=dropout_rate
        )
        
        # Separate branch for each aspect category
        self.aspect_branches = nn.ModuleList([
            nn.Linear(self.encoder.hidden_size, self.num_polarities)
            for _ in range(self.num_aspects)
        ])
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass for multi-branch approach
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            List of predictions for each aspect [num_aspects, batch_size, 4]
        """
        # Get encoded representation
        encoded = self.encoder(input_ids, attention_mask)  # [batch, hidden_size]
        
        # Get predictions for each aspect branch
        branch_outputs = []
        for branch in self.aspect_branches:
            output = branch(encoded)  # [batch, 4]
            # Note: Don't apply softmax here as it's handled in loss function
            branch_outputs.append(output)
        
        return branch_outputs
    
    def predict_aspects(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> List[Dict]:
        """
        Predict aspect categories and sentiments
        
        Returns:
            List of predictions for each sample in batch
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            batch_size = outputs[0].size(0)
            
            predictions = []
            for i in range(batch_size):
                sample_pred = {}
                
                for j, aspect_category in enumerate(self.aspect_categories):
                    logits = outputs[j][i]  # [4]
                    probs = F.softmax(logits, dim=-1)
                    predicted_class = torch.argmax(probs).item()
                    
                    # Map to polarity
                    polarity_mapping = {0: None, 1: 'positive', 2: 'negative', 3: 'neutral'}
                    polarity = polarity_mapping[predicted_class]
                    
                    if polarity is not None:  # Only include non-None predictions
                        sample_pred[aspect_category] = {
                            'polarity': polarity,
                            'confidence': probs[predicted_class].item()
                        }
                
                predictions.append(sample_pred)
            
            return predictions


class VLSP2018Model(nn.Module):
    """Unified model class that can handle both approaches"""
    
    def __init__(self, pretrained_model_name: str, aspect_categories: List[str],
                 approach: str = 'multitask', num_last_layers: int = 4, 
                 dropout_rate: float = 0.2):
        super(VLSP2018Model, self).__init__()
        
        self.approach = approach
        self.aspect_categories = aspect_categories
        
        if approach == 'multitask':
            self.model = VLSP2018MultiTaskModel(
                pretrained_model_name=pretrained_model_name,
                aspect_categories=aspect_categories,
                num_last_layers=num_last_layers,
                dropout_rate=dropout_rate
            )
        elif approach == 'multibranch':
            self.model = VLSP2018MultiBranchModel(
                pretrained_model_name=pretrained_model_name,
                aspect_categories=aspect_categories,
                num_last_layers=num_last_layers,
                dropout_rate=dropout_rate
            )
        else:
            raise ValueError(f"Unsupported approach: {approach}. Use 'multitask' or 'multibranch'")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        return self.model(input_ids, attention_mask)
    
    def predict_aspects(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> List[Dict]:
        return self.model.predict_aspects(input_ids, attention_mask)


class VLSP2018Loss(nn.Module):
    """Custom loss function for VLSP 2018 ABSA models"""
    
    def __init__(self, approach: str = 'multitask', num_aspects: int = 34):
        super(VLSP2018Loss, self).__init__()
        
        self.approach = approach
        self.num_aspects = num_aspects
        
        if approach == 'multitask':
            # Binary cross-entropy for flattened one-hot labels
            self.criterion = nn.BCELoss()
        elif approach == 'multibranch':
            # Sparse categorical cross-entropy for each branch
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported approach: {approach}")
    
    def forward(self, outputs: Union[torch.Tensor, List[torch.Tensor]], 
                targets: torch.Tensor) -> torch.Tensor:
        """
        Compute loss based on approach
        
        Args:
            outputs: Model outputs
            targets: Ground truth labels
            
        Returns:
            Loss value
        """
        if self.approach == 'multitask':
            # outputs: [batch, num_aspects * 4]
            # targets: [batch, num_aspects * 4] (flattened one-hot)
            return self.criterion(outputs, targets)
        
        elif self.approach == 'multibranch':
            # outputs: List[Tensor] of length num_aspects, each [batch, 4]
            # targets: [batch, num_aspects] (class indices)
            total_loss = 0.0
            
            for i, branch_output in enumerate(outputs):
                branch_target = targets[:, i]  # [batch]
                branch_loss = self.criterion(branch_output, branch_target)
                total_loss += branch_loss
            
            return total_loss / len(outputs)  # Average loss across branches


def create_model(config: Dict) -> Tuple[VLSP2018Model, VLSP2018Loss]:
    """
    Factory function to create model and loss based on configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (model, loss_function)
    """
    model_config = config['model']
    
    model = VLSP2018Model(
        pretrained_model_name=model_config['pretrained_model'],
        aspect_categories=config['aspect_categories'],
        approach=model_config['approach'],
        num_last_layers=model_config.get('num_last_layers', 4),
        dropout_rate=model_config.get('dropout_rate', 0.2)
    )
    
    loss_fn = VLSP2018Loss(
        approach=model_config['approach'],
        num_aspects=len(config['aspect_categories'])
    )
    
    return model, loss_fn


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }


if __name__ == '__main__':
    # Test model creation
    print("Testing VLSP2018 PyTorch models...")
    
    # Test data
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Hotel aspect categories (simplified)
    aspect_categories = [
        "HOTEL#CLEANLINESS", "HOTEL#COMFORT", "HOTEL#DESIGN&FEATURES",
        "ROOMS#CLEANLINESS", "ROOMS#COMFORT", "SERVICE#GENERAL"
    ]
    
    # Test Multi-task model
    print("\n=== Testing Multi-task Model ===")
    multitask_model = VLSP2018Model(
        pretrained_model_name="vinai/phobert-base",
        aspect_categories=aspect_categories,
        approach='multitask'
    )
    
    multitask_outputs = multitask_model(input_ids, attention_mask)
    print(f"Multi-task output shape: {multitask_outputs.shape}")
    print(f"Expected shape: [batch_size={batch_size}, num_aspects*4={len(aspect_categories)*4}]")
    
    # Test Multi-branch model
    print("\n=== Testing Multi-branch Model ===")
    multibranch_model = VLSP2018Model(
        pretrained_model_name="vinai/phobert-base",
        aspect_categories=aspect_categories,
        approach='multibranch'
    )
    
    multibranch_outputs = multibranch_model(input_ids, attention_mask)
    print(f"Multi-branch outputs: {len(multibranch_outputs)} branches")
    print(f"Each branch shape: {multibranch_outputs[0].shape}")
    print(f"Expected: {len(aspect_categories)} branches, each [batch_size={batch_size}, 4]")
    
    # Test predictions
    print("\n=== Testing Predictions ===")
    multitask_predictions = multitask_model.predict_aspects(input_ids, attention_mask)
    print(f"Multi-task predictions for sample 0: {multitask_predictions[0]}")
    
    multibranch_predictions = multibranch_model.predict_aspects(input_ids, attention_mask)
    print(f"Multi-branch predictions for sample 0: {multibranch_predictions[0]}")
    
    # Test parameter counting
    print("\n=== Model Parameters ===")
    multitask_params = count_parameters(multitask_model)
    print(f"Multi-task model parameters: {multitask_params}")
    
    multibranch_params = count_parameters(multibranch_model)
    print(f"Multi-branch model parameters: {multibranch_params}")
    
    print("\nModel testing completed!") 