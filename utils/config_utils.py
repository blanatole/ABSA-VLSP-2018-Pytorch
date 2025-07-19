"""
Configuration utilities for PyTorch ABSA VLSP 2018
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Union


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    # Handle defaults/inheritance
    if 'defaults' in config:
        base_configs = config['defaults']
        if isinstance(base_configs, list):
            merged_config = {}
            for base_config in base_configs:
                if isinstance(base_config, str):
                    base_path = config_path.parent / f"{base_config}.yaml"
                    base_cfg = load_config(base_path)
                    merged_config = merge_configs(merged_config, base_cfg)
            
            # Merge current config on top
            config = merge_configs(merged_config, {k: v for k, v in config.items() if k != 'defaults'})
    
    return config


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """
    Save configuration to file
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        elif config_path.suffix.lower() == '.json':
            json.dump(config, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries
    
    Args:
        base: Base configuration
        override: Override configuration
        
    Returns:
        Merged configuration
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration structure
    
    Args:
        config: Configuration to validate
        
    Returns:
        True if valid, raises exception otherwise
    """
    required_sections = ['model', 'training', 'data']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate model section
    model_config = config['model']
    required_model_keys = ['pretrained_model', 'approach']
    for key in required_model_keys:
        if key not in model_config:
            raise ValueError(f"Missing required model configuration: {key}")
    
    # Validate approach
    if model_config['approach'] not in ['multitask', 'multibranch']:
        raise ValueError(f"Invalid approach: {model_config['approach']}")
    
    # Validate training section
    training_config = config['training']
    required_training_keys = ['batch_size', 'learning_rate', 'num_epochs']
    for key in required_training_keys:
        if key not in training_config:
            raise ValueError(f"Missing required training configuration: {key}")
    
    # Validate aspect categories
    if 'aspect_categories' not in config:
        raise ValueError("Missing aspect_categories in configuration")
    
    if not isinstance(config['aspect_categories'], list) or len(config['aspect_categories']) == 0:
        raise ValueError("aspect_categories must be a non-empty list")
    
    return True


def get_config_summary(config: Dict[str, Any]) -> str:
    """
    Get a summary string of the configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Summary string
    """
    model_config = config.get('model', {})
    training_config = config.get('training', {})
    
    summary = f"""Configuration Summary:
    Model: {model_config.get('pretrained_model', 'Unknown')}
    Approach: {model_config.get('approach', 'Unknown')}
    Domain: {model_config.get('domain', 'Unknown')}
    Aspect Categories: {len(config.get('aspect_categories', []))}
    Batch Size: {training_config.get('batch_size', 'Unknown')}
    Learning Rate: {training_config.get('learning_rate', 'Unknown')}
    Epochs: {training_config.get('num_epochs', 'Unknown')}
    """
    
    return summary


if __name__ == '__main__':
    # Test configuration utilities
    print("Testing configuration utilities...")
    
    # Test config creation
    test_config = {
        'model': {
            'pretrained_model': 'vinai/phobert-base',
            'approach': 'multitask',
            'domain': 'hotel'
        },
        'training': {
            'batch_size': 32,
            'learning_rate': 2e-4,
            'num_epochs': 20
        },
        'aspect_categories': ['HOTEL#CLEANLINESS', 'ROOMS#COMFORT']
    }
    
    # Test validation
    try:
        validate_config(test_config)
        print("✓ Configuration validation passed")
    except Exception as e:
        print(f"✗ Configuration validation failed: {e}")
    
    # Test summary
    summary = get_config_summary(test_config)
    print("Configuration summary:")
    print(summary)
    
    print("Configuration utilities test completed!") 