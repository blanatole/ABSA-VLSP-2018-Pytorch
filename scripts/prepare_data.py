#!/usr/bin/env python3
"""
Data preparation script for VLSP 2018 ABSA dataset
Converts TXT format to CSV and preprocesses Vietnamese text
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_processing import VLSP2018Parser, VietnameseTextPreprocessor


def prepare_hotel_data(data_dir: str, output_dir: str = None):
    """Prepare Hotel domain data"""
    print("=== Preparing Hotel Domain Data ===")
    
    if output_dir is None:
        output_dir = data_dir
    
    # Hotel data paths
    train_path = os.path.join(data_dir, 'datasets/vlsp2018_hotel/1-VLSP2018-SA-Hotel-train.txt')
    val_path = os.path.join(data_dir, 'datasets/vlsp2018_hotel/2-VLSP2018-SA-Hotel-dev.txt')
    test_path = os.path.join(data_dir, 'datasets/vlsp2018_hotel/3-VLSP2018-SA-Hotel-test.txt')
    
    # Check if files exist
    for path in [train_path, val_path, test_path]:
        if not os.path.exists(path):
            print(f"Warning: {path} not found!")
            return None
    
    # Parse and convert
    parser = VLSP2018Parser(train_path, val_path, test_path)
    parser.txt2csv()
    
    print(f"Hotel data preparation completed!")
    print(f"Found {len(parser.aspect_categories)} aspect categories:")
    for i, category in enumerate(parser.aspect_categories):
        print(f"  {i+1:2d}. {category}")
    
    return parser.aspect_categories


def prepare_restaurant_data(data_dir: str, output_dir: str = None):
    """Prepare Restaurant domain data"""
    print("\n=== Preparing Restaurant Domain Data ===")
    
    if output_dir is None:
        output_dir = data_dir
    
    # Restaurant data paths
    train_path = os.path.join(data_dir, 'datasets/vlsp2018_restaurant/1-VLSP2018-SA-Restaurant-train.txt')
    val_path = os.path.join(data_dir, 'datasets/vlsp2018_restaurant/2-VLSP2018-SA-Restaurant-dev.txt')
    test_path = os.path.join(data_dir, 'datasets/vlsp2018_restaurant/3-VLSP2018-SA-Restaurant-test.txt')
    
    # Check if files exist
    for path in [train_path, val_path, test_path]:
        if not os.path.exists(path):
            print(f"Warning: {path} not found!")
            return None
    
    # Parse and convert
    parser = VLSP2018Parser(train_path, val_path, test_path)
    parser.txt2csv()
    
    print(f"Restaurant data preparation completed!")
    print(f"Found {len(parser.aspect_categories)} aspect categories:")
    for i, category in enumerate(parser.aspect_categories):
        print(f"  {i+1:2d}. {category}")
    
    return parser.aspect_categories


def setup_vncorenlp(vncorenlp_dir: str = 'VnCoreNLP'):
    """Setup VnCoreNLP for Vietnamese text processing"""
    print("=== Setting up VnCoreNLP ===")
    
    # Create VnCoreNLP directory in utils
    utils_dir = Path(__file__).parent.parent / 'utils'
    utils_dir.mkdir(exist_ok=True)
    vncorenlp_path = utils_dir / vncorenlp_dir
    
    # Test Vietnamese preprocessor (this will download VnCoreNLP if needed)
    try:
        preprocessor = VietnameseTextPreprocessor(
            vncorenlp_dir=str(vncorenlp_path),
            extra_teencodes={
                'khách sạn': ['ks'],
                'nhà hàng': ['nhahang', 'nha hang'],
                'nhân viên': ['nv'],
                'dịch vụ': ['dv'],
                'phòng tắm': ['pt'],
            }
        )
        
        # Test preprocessing
        test_text = "Ks này rất đẹp và sạch sẽ. Nv thân thiện."
        processed = preprocessor.process_text(test_text)
        print(f"Test preprocessing:")
        print(f"  Input:  {test_text}")
        print(f"  Output: {processed}")
        
        print("VnCoreNLP setup completed!")
        return True
        
    except Exception as e:
        print(f"Error setting up VnCoreNLP: {e}")
        return False


def create_config_files(hotel_aspects: list, restaurant_aspects: list):
    """Create configuration files with discovered aspect categories"""
    print("\n=== Creating Configuration Files ===")
    
    config_dir = Path(__file__).parent.parent / 'configs'
    
    # Update hotel config
    hotel_config_path = config_dir / 'hotel_multitask.yaml'
    if hotel_config_path.exists():
        # Read current config
        with open(hotel_config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find and replace aspect categories section
        import re
        pattern = r'aspect_categories:\s*\n(.*?)(?=\n[a-zA-Z_]|\n#|\Z)'
        
        # Create new aspect categories section
        new_aspects = 'aspect_categories:\n'
        for aspect in hotel_aspects:
            new_aspects += f'  - "{aspect}"\n'
        
        # Replace in content
        updated_content = re.sub(pattern, new_aspects.rstrip(), content, flags=re.DOTALL)
        
        # Write back
        with open(hotel_config_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        print(f"Updated {hotel_config_path}")
    
    # Create restaurant config (copy from hotel and modify)
    restaurant_config_path = config_dir / 'restaurant_multitask.yaml'
    restaurant_config_content = f"""# Restaurant Domain - Multi-task Approach Configuration

# Inherit from base config
defaults:
  - base_config

# Override specific settings
model:
  approach: "multitask"  # multitask vs multibranch
  domain: "restaurant"
  num_aspect_categories: {len(restaurant_aspects)}
  
# Restaurant-specific aspect categories
aspect_categories:
"""
    
    for aspect in restaurant_aspects:
        restaurant_config_content += f'  - "{aspect}"\n'
    
    restaurant_config_content += f"""
# Data paths
data:
  train_path: "data/datasets/vlsp2018_restaurant/1-VLSP2018-SA-Restaurant-train.txt"
  val_path: "data/datasets/vlsp2018_restaurant/2-VLSP2018-SA-Restaurant-dev.txt"
  test_path: "data/datasets/vlsp2018_restaurant/3-VLSP2018-SA-Restaurant-test.txt"

# Training specific to Restaurant domain
training:
  batch_size: 20  # As used in original paper
  
# Output settings
output:
  model_name: "restaurant_multitask"
  save_dir: "models/restaurant_multitask"
  results_dir: "results/restaurant_multitask"
"""
    
    with open(restaurant_config_path, 'w', encoding='utf-8') as f:
        f.write(restaurant_config_content)
    
    print(f"Created {restaurant_config_path}")


def main():
    parser = argparse.ArgumentParser(description='Prepare VLSP 2018 ABSA data for PyTorch training')
    parser.add_argument('--data_dir', type=str, default='data', 
                       help='Directory containing VLSP 2018 dataset')
    parser.add_argument('--domain', type=str, choices=['hotel', 'restaurant', 'both'], 
                       default='both', help='Which domain to prepare')
    parser.add_argument('--setup_vncorenlp', action='store_true', 
                       help='Setup VnCoreNLP for Vietnamese preprocessing')
    parser.add_argument('--skip_preprocessing', action='store_true',
                       help='Skip Vietnamese text preprocessing setup')
    
    args = parser.parse_args()
    
    print("PyTorch ABSA VLSP 2018 - Data Preparation")
    print("=" * 50)
    
    # Check data directory
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory {args.data_dir} not found!")
        print("Please copy the VLSP 2018 dataset to the data directory first.")
        return
    
    hotel_aspects = None
    restaurant_aspects = None
    
    # Prepare data based on domain
    if args.domain in ['hotel', 'both']:
        hotel_aspects = prepare_hotel_data(args.data_dir)
    
    if args.domain in ['restaurant', 'both']:
        restaurant_aspects = prepare_restaurant_data(args.data_dir)
    
    # Setup Vietnamese preprocessing
    if not args.skip_preprocessing:
        if args.setup_vncorenlp or args.domain == 'both':
            setup_vncorenlp()
    
    # Create/update config files
    if hotel_aspects or restaurant_aspects:
        # Use default aspects if not loaded
        if hotel_aspects is None:
            hotel_aspects = [
                "FACILITIES#CLEANLINESS", "FACILITIES#COMFORT", "FACILITIES#DESIGN&FEATURES",
                "FACILITIES#GENERAL", "FACILITIES#MISCELLANEOUS", "FACILITIES#PRICES",
                "FACILITIES#QUALITY", "FOOD&DRINKS#MISCELLANEOUS", "FOOD&DRINKS#PRICES",
                "FOOD&DRINKS#QUALITY", "FOOD&DRINKS#STYLE&OPTIONS", "HOTEL#CLEANLINESS",
                "HOTEL#COMFORT", "HOTEL#DESIGN&FEATURES", "HOTEL#GENERAL", "HOTEL#MISCELLANEOUS",
                "HOTEL#PRICES", "HOTEL#QUALITY", "LOCATION#GENERAL", "ROOMS#CLEANLINESS",
                "ROOMS#COMFORT", "ROOMS#DESIGN&FEATURES", "ROOMS#GENERAL", "ROOMS#MISCELLANEOUS",
                "ROOMS#PRICES", "ROOMS#QUALITY", "ROOM_AMENITIES#CLEANLINESS",
                "ROOM_AMENITIES#COMFORT", "ROOM_AMENITIES#DESIGN&FEATURES",
                "ROOM_AMENITIES#GENERAL", "ROOM_AMENITIES#MISCELLANEOUS",
                "ROOM_AMENITIES#PRICES", "ROOM_AMENITIES#QUALITY", "SERVICE#GENERAL"
            ]
        
        if restaurant_aspects is None:
            restaurant_aspects = [
                "AMBIENCE#GENERAL", "DRINKS#PRICES", "DRINKS#QUALITY", "DRINKS#STYLE&OPTIONS",
                "FOOD#PRICES", "FOOD#QUALITY", "FOOD#STYLE&OPTIONS", "LOCATION#GENERAL",
                "RESTAURANT#GENERAL", "RESTAURANT#MISCELLANEOUS", "RESTAURANT#PRICES",
                "SERVICE#GENERAL"
            ]
        
        create_config_files(hotel_aspects, restaurant_aspects)
    
    print("\n" + "=" * 50)
    print("Data preparation completed!")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Train a model: python scripts/train.py --config configs/hotel_multitask.yaml")
    print("3. Evaluate model: python scripts/evaluate.py --model_path models/hotel_multitask.pth")


if __name__ == '__main__':
    main() 