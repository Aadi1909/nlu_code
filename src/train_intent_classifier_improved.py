#!/usr/bin/env python3
"""
Improved training script with better hyperparameters
Uses the balanced dataset created by quick_fix_data.py
"""

import sys
sys.path.append('..')

from train_intent_classifier import IntentClassifierTrainer, ModelConfig, TrainingConfig
import torch

def main():
    print("🚀 Starting Improved Intent Classifier Training")
    print("=" * 70)
    
    # Check if augmented v2 dataset exists, fallback to balanced
    from pathlib import Path
    augmented_v2_path = Path("../data/processed/train_augmented_v2.json")
    balanced_path = Path("../data/processed/train_balanced.json")
    
    if augmented_v2_path.exists():
        train_data_path = augmented_v2_path
        print(f"✅ Using diverse augmented dataset: {train_data_path}")
    elif balanced_path.exists():
        train_data_path = balanced_path
        print(f"✅ Using balanced dataset: {train_data_path}")
    else:
        print("❌ No augmented dataset found!")
        print("   Run: python generate_diverse_data.py first")
        return
    
    # Initialize trainer with improved hyperparameters
    trainer = IntentClassifierTrainer(
        model_config=ModelConfig(
            model_name="google/muril-base-cased",
            max_length=128,
            output_dir="/mnt/sagemaker-nvme/intent_models_improved" if Path("/mnt/sagemaker-nvme").exists() else "../models/intent_classifier_improved"
        ),
        training_config=TrainingConfig(
            num_epochs=20,  # More epochs
            batch_size=16 if torch.cuda.is_available() else 8,
            learning_rate=2e-5,  # Lower learning rate for better convergence
            weight_decay=0.01,
            warmup_ratio=0.1,
            eval_steps=25,  # More frequent evaluation
            save_steps=25,
            logging_steps=10,
            early_stopping_patience=7,  # More patience
            fp16=True,
            use_augmented=False  # We'll manually specify the balanced dataset
        ),
        data_dir="../data/processed"
    )
    
    # Temporarily modify the trainer to use balanced dataset
    print("\n📊 Loading balanced training data...")
    
    # Override the load_datasets method to use balanced data
    original_load = trainer.load_datasets
    
    def load_balanced_datasets():
        from datasets import load_dataset
        
        # Load augmented/balanced training data
        train_dataset = load_dataset('json', data_files=str(train_data_path), split='train')
        
        # Use regular val and test
        val_dataset = load_dataset('json', data_files=str(Path("../data/processed/val_augmented.json")), split='train')
        test_dataset = load_dataset('json', data_files=str(Path("../data/processed/test_augmented.json")), split='train')
        
        print(f"✅ Loaded - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset
    
    trainer.load_datasets = load_balanced_datasets
    
    # Train
    print("\n🎯 Starting training with improved hyperparameters...")
    print("   - More epochs: 20")
    print("   - Lower learning rate: 2e-5")
    print("   - Balanced dataset")
    print("   - More patience: 7")
    print()
    
    results = trainer.train()
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nModel saved to: {trainer.model_config.output_dir}")
    print("\nTest the improved model with:")
    print("  python test_model_quick.py")
    print()
    
    return results


if __name__ == "__main__":
    main()
