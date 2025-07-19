#!/usr/bin/env python3
"""
Demo script for PyTorch ABSA VLSP 2018
Quick test of data processing and model creation
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def test_imports():
    """Test if all modules can be imported"""
    print("=== Testing Imports ===")
    
    try:
        from data_processing import (
            PolarityMapping, VietnameseTextCleaner, 
            VietnameseToneNormalizer, VLSP2018Parser
        )
        print("✓ Data processing modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import data processing modules: {e}")
        return False
    
    try:
        from models import (
            VLSP2018Model, VLSP2018Loss, PhoBERTEncoder,
            create_model, count_parameters
        )
        print("✓ Model modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import model modules: {e}")
        return False
    
    try:
        import torch
        import torch.nn as nn
        print(f"✓ PyTorch {torch.__version__} imported successfully")
        
        # Check CUDA availability
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name()}")
        else:
            print("⚠ CUDA not available, will use CPU")
            
    except ImportError as e:
        print(f"✗ Failed to import PyTorch: {e}")
        return False
    
    return True


def test_text_processing():
    """Test Vietnamese text processing"""
    print("\n=== Testing Text Processing ===")
    
    try:
        from data_processing import VietnameseTextCleaner, VietnameseToneNormalizer
        
        # Test text cleaner
        test_texts = [
            "Khách sạn rất đẹp và sạch sẽ! 😊",
            "<p>Phòng thoáng mát</p> http://example.com",
            "Dịch vụ tốt, nhân viên thân thiện ⭐⭐⭐",
            "KS này ok, giá cả hợp lý!!!"
        ]
        
        success_count = 0
        for text in test_texts:
            try:
                cleaned = VietnameseTextCleaner.process_text(text)
                normalized = VietnameseToneNormalizer.normalize_unicode(cleaned)
                print(f"Original: {text}")
                print(f"Cleaned:  {cleaned}")
                print(f"Normalized: {normalized}")
                print()
                success_count += 1
            except Exception as e:
                print(f"✗ Failed to process text: {text}")
                print(f"  Error: {e}")
                return False
        
        print(f"✓ Successfully processed {success_count}/{len(test_texts)} text samples")
        return True
        
    except ImportError as e:
        print(f"✗ Failed to import text processing modules: {e}")
        return False
    except Exception as e:
        print(f"✗ Text processing test failed: {e}")
        return False


def test_model_creation():
    """Test model creation and basic operations"""
    print("=== Testing Model Creation ===")
    
    import torch
    from models import VLSP2018Model, VLSP2018Loss, count_parameters
    
    # Simple test aspect categories
    aspect_categories = [
        "HOTEL#CLEANLINESS", "HOTEL#COMFORT", "HOTEL#DESIGN&FEATURES",
        "ROOMS#CLEANLINESS", "ROOMS#COMFORT", "SERVICE#GENERAL"
    ]
    
    print(f"Testing with {len(aspect_categories)} aspect categories")
    
    # Test Multi-task model
    print("\n--- Multi-task Model ---")
    try:
        multitask_model = VLSP2018Model(
            pretrained_model_name="vinai/phobert-base",
            aspect_categories=aspect_categories,
            approach='multitask'
        )
        
        # Count parameters
        params = count_parameters(multitask_model)
        print(f"✓ Multi-task model created successfully")
        print(f"  Total parameters: {params['total_parameters']:,}")
        print(f"  Trainable parameters: {params['trainable_parameters']:,}")
        
        # Test forward pass
        batch_size = 2
        seq_len = 64
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        outputs = multitask_model(input_ids, attention_mask)
        expected_size = len(aspect_categories) * 4  # 4 polarities per aspect
        print(f"✓ Forward pass successful")
        print(f"  Output shape: {outputs.shape}")
        print(f"  Expected: [batch_size={batch_size}, features={expected_size}]")
        
        # Test loss function
        loss_fn = VLSP2018Loss(approach='multitask', num_aspects=len(aspect_categories))
        targets = torch.rand(batch_size, expected_size)  # Random targets for testing
        loss = loss_fn(outputs, targets)
        print(f"✓ Loss computation successful: {loss.item():.4f}")
        
    except Exception as e:
        print(f"✗ Multi-task model test failed: {e}")
        return False
    
    # Test Multi-branch model
    print("\n--- Multi-branch Model ---")
    try:
        multibranch_model = VLSP2018Model(
            pretrained_model_name="vinai/phobert-base",
            aspect_categories=aspect_categories,
            approach='multibranch'
        )
        
        params = count_parameters(multibranch_model)
        print(f"✓ Multi-branch model created successfully")
        print(f"  Total parameters: {params['total_parameters']:,}")
        print(f"  Trainable parameters: {params['trainable_parameters']:,}")
        
        # Test forward pass
        outputs = multibranch_model(input_ids, attention_mask)
        print(f"✓ Forward pass successful")
        print(f"  Number of branches: {len(outputs)}")
        print(f"  Each branch shape: {outputs[0].shape}")
        print(f"  Expected: {len(aspect_categories)} branches, each [batch_size={batch_size}, 4]")
        
        # Test loss function
        loss_fn = VLSP2018Loss(approach='multibranch', num_aspects=len(aspect_categories))
        targets = torch.randint(0, 4, (batch_size, len(aspect_categories)))  # Random class indices
        loss = loss_fn(outputs, targets)
        print(f"✓ Loss computation successful: {loss.item():.4f}")
        
    except Exception as e:
        print(f"✗ Multi-branch model test failed: {e}")
        return False
    
    return True


def test_prediction():
    """Test model prediction functionality"""
    print("\n=== Testing Prediction ===")
    
    import torch
    from models import VLSP2018Model
    
    aspect_categories = [
        "HOTEL#CLEANLINESS", "HOTEL#COMFORT", "SERVICE#GENERAL"
    ]
    
    try:
        model = VLSP2018Model(
            pretrained_model_name="vinai/phobert-base",
            aspect_categories=aspect_categories,
            approach='multitask'
        )
        
        # Create dummy input
        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        # Test prediction
        predictions = model.predict_aspects(input_ids, attention_mask)
        print(f"✓ Prediction successful")
        print(f"  Batch size: {len(predictions)}")
        
        for i, pred in enumerate(predictions):
            print(f"  Sample {i+1} predictions: {pred}")
        
    except Exception as e:
        print(f"✗ Prediction test failed: {e}")
        return False
    
    return True


def test_data_parsing():
    """Test data parsing if data files exist"""
    print("\n=== Testing Data Parsing ===")
    
    data_dir = Path(__file__).parent.parent / 'data'
    hotel_train = data_dir / 'datasets/vlsp2018_hotel/1-VLSP2018-SA-Hotel-train.txt'
    
    if hotel_train.exists():
        print(f"✓ Found hotel training data: {hotel_train}")
        
        try:
            from data_processing import VLSP2018Parser
            
            # Parse just training data for testing
            parser = VLSP2018Parser(str(hotel_train))
            print(f"✓ Data parsing successful")
            print(f"  Found {len(parser.aspect_categories)} aspect categories")
            print(f"  Training samples: {len(parser.reviews['train'])}")
            
            # Show first few aspect categories
            print("  Sample aspect categories:")
            for i, category in enumerate(parser.aspect_categories[:5]):
                print(f"    {i+1}. {category}")
            
            if len(parser.aspect_categories) > 5:
                print(f"    ... and {len(parser.aspect_categories) - 5} more")
                
        except Exception as e:
            print(f"✗ Data parsing failed: {e}")
            return False
    else:
        print(f"⚠ Hotel training data not found at {hotel_train}")
        print("  To test data parsing, copy VLSP 2018 dataset to data/ directory")
        print("  Then run: python scripts/prepare_data.py")
    
    return True


def test_config_loading():
    """Test configuration loading"""
    print("\n=== Testing Configuration ===")
    
    # Test config utilities without importing (since we might not have PyYAML)
    config_path = Path(__file__).parent.parent / 'configs/hotel_multitask.yaml'
    
    if config_path.exists():
        try:
            # Try to read the config file manually
            with open(config_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            print(f"✓ Configuration file found: {config_path.name}")
            print(f"  File size: {len(content)} characters")
            
            # Check if it contains expected sections
            expected_sections = ['model:', 'aspect_categories:', 'data:', 'training:']
            found_sections = []
            
            for section in expected_sections:
                if section in content:
                    found_sections.append(section.rstrip(':'))
            
            print(f"  Found sections: {found_sections}")
            
            if len(found_sections) == len(expected_sections):
                print("✓ Configuration structure looks correct")
            else:
                print("⚠ Some expected sections might be missing")
                
        except Exception as e:
            print(f"✗ Failed to read configuration: {e}")
            return False
    else:
        print(f"⚠ Configuration file not found: {config_path}")
        print("  Run prepare_data.py to generate configuration files")
    
    return True


def main():
    """Run all tests"""
    print("PyTorch ABSA VLSP 2018 - Demo Script")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Text Processing", test_text_processing),
        ("Model Creation", test_model_creation),
        ("Prediction", test_prediction),
        ("Data Parsing", test_data_parsing),
        ("Configuration", test_config_loading),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Results Summary:")
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Project is ready for use.")
    elif passed >= total * 0.8:
        print("⚠ Most tests passed. Minor issues may need attention.")
    else:
        print("❌ Several tests failed. Please check dependencies and setup.")
    
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Prepare data: python scripts/prepare_data.py") 
    print("3. Train model: python scripts/train.py --config configs/hotel_multitask.yaml")


if __name__ == '__main__':
    main() 