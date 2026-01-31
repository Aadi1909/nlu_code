#!/usr/bin/env python3
"""
Optimized Intent Classifier Training Script for Large Datasets (1.6M+ rows)

Key optimizations:
1. Streaming/chunked data loading to manage memory
2. Gradient accumulation for effective larger batch sizes
3. Mixed precision (FP16/BF16) training
4. Efficient data collation with dynamic padding
5. Memory-mapped datasets for large files
6. Multi-worker data loading
7. Gradient checkpointing for memory efficiency
"""

import os
import gc
import json
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from loguru import logger

from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import warnings
warnings.filterwarnings("ignore")


@dataclass
class ModelConfig:
    """Model configuration for large-scale training"""
    model_name: str = "google/muril-base-cased"  # Best for Indian languages
    max_length: int = 128
    num_labels: int = 29  # Will be updated from data
    output_dir: str = "models/intent_classifier"
    

@dataclass
class LargeScaleTrainingConfig:
    """Training configuration optimized for large datasets (1.6M+ rows)"""
    # Epochs - fewer needed for large datasets
    num_epochs: int = 3  # Reduced for large data - 1-3 epochs is enough
    
    # Batch sizes - use gradient accumulation for effective larger batches
    per_device_batch_size: int = 32  # Adjust based on GPU memory
    gradient_accumulation_steps: int = 4  # Effective batch = 32 * 4 = 128
    
    # Learning rate - slightly lower for stability with large data
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06  # Lower warmup for large datasets
    
    # Evaluation strategy - less frequent for large datasets
    eval_strategy: str = "steps"
    eval_steps: int = 2000  # Evaluate every 2000 steps
    save_steps: int = 2000
    logging_steps: int = 500
    
    # Early stopping
    early_stopping_patience: int = 3
    
    # Memory optimizations
    fp16: bool = True  # Use mixed precision
    bf16: bool = False  # Use BF16 if available (better for newer GPUs)
    gradient_checkpointing: bool = True  # Save memory at cost of speed
    
    # Data loading
    dataloader_num_workers: int = 4  # Parallel data loading
    dataloader_pin_memory: bool = True  # Faster GPU transfers
    
    # Save limits
    save_total_limit: int = 2  # Keep only 2 best checkpoints
    
    # Use augmented data
    use_augmented: bool = False


class LargeDatasetIntentTrainer:
    """
    Optimized trainer for large-scale Intent Classification
    Designed to handle 1.6M+ training samples efficiently
    """
    
    def __init__(
        self, 
        model_config: ModelConfig = None,
        training_config: LargeScaleTrainingConfig = None,
        data_dir: str = "data/processed"
    ):
        self.model_config = model_config or ModelConfig()
        self.training_config = training_config or LargeScaleTrainingConfig()
        self.data_dir = Path(data_dir)
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._log_system_info()
        
        # Load label mapping
        self.label_mapping = self._load_label_mapping()
        self.model_config.num_labels = len(self.label_mapping["label2id"])
        
        # Initialize tokenizer with memory efficiency
        logger.info(f"Loading tokenizer: {self.model_config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.model_name,
            use_fast=True  # Use fast tokenizer for speed
        )
    
    def _log_system_info(self):
        """Log system and GPU information"""
        logger.info(f"Using device: {self.device}")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU: {gpu_name}")
            logger.info(f"GPU Memory: {gpu_memory:.2f} GB")
            
            # Check for BF16 support
            if torch.cuda.is_bf16_supported():
                logger.info("BF16 supported - enabling for better precision")
                self.training_config.bf16 = True
                self.training_config.fp16 = False
        else:
            logger.warning("No GPU available - training will be very slow for large datasets!")
            # Reduce batch size for CPU
            self.training_config.per_device_batch_size = 8
            self.training_config.fp16 = False
            self.training_config.gradient_checkpointing = False
            self.training_config.dataloader_num_workers = 0
        
    def _load_label_mapping(self) -> Dict:
        """Load label mapping from data directory"""
        mapping_path = self.data_dir / "label_mapping.json"
        with open(mapping_path, 'r') as f:
            mapping = json.load(f)
        
        # Convert string keys to int for id2label
        mapping["id2label"] = {int(k): v for k, v in mapping["id2label"].items()}
        logger.info(f"Loaded {len(mapping['label2id'])} intent labels")
        
        return mapping
    
    def load_datasets(self) -> DatasetDict:
        """
        Load datasets efficiently for large-scale training
        Uses streaming when files are very large
        """
        logger.info("Loading datasets...")
        
        file_suffix = "_augmented" if self.training_config.use_augmented else ""
        
        train_path = str(self.data_dir / f"train{file_suffix}.json")
        val_path = str(self.data_dir / f"val{file_suffix}.json")
        test_path = str(self.data_dir / f"test{file_suffix}.json")
        
        # Check file sizes to decide loading strategy
        train_size_mb = os.path.getsize(train_path) / (1024 * 1024)
        logger.info(f"Train file size: {train_size_mb:.2f} MB")
        
        # Load datasets
        train_dataset = load_dataset('json', data_files=train_path, split='train')
        val_dataset = load_dataset('json', data_files=val_path, split='train')
        test_dataset = load_dataset('json', data_files=test_path, split='train')
        
        # Recompute labels from intent to ensure consistency
        def _add_label(batch):
            return {"label": self.label_mapping["label2id"][batch["intent"]]}
        
        # Add labels (using multiprocessing for large datasets)
        num_proc = 4 if len(train_dataset) > 100000 else 1
        
        train_dataset = train_dataset.map(_add_label, num_proc=num_proc)
        val_dataset = val_dataset.map(_add_label)
        test_dataset = test_dataset.map(_add_label)
        
        logger.info(f"Dataset sizes - Train: {len(train_dataset):,}, Val: {len(val_dataset):,}, Test: {len(test_dataset):,}")
        
        return DatasetDict({
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset
        })
    
    def tokenize_function(self, examples):
        """Tokenize examples with truncation"""
        return self.tokenizer(
            examples["text"],
            truncation=True,
            max_length=self.model_config.max_length,
            padding=False  # Dynamic padding via data collator
        )
    
    def prepare_datasets(self, datasets: DatasetDict) -> DatasetDict:
        """Tokenize all datasets efficiently"""
        logger.info("Tokenizing datasets...")
        
        # Determine number of processes based on dataset size
        train_size = len(datasets['train'])
        num_proc = min(8, os.cpu_count() or 1) if train_size > 100000 else 1
        
        logger.info(f"Using {num_proc} processes for tokenization")
        
        # Columns to remove after tokenization
        columns_to_remove = ["text", "intent", "language"]
        if "entities" in datasets['train'].column_names:
            columns_to_remove.append("entities")
        if "metadata" in datasets['train'].column_names:
            columns_to_remove.append("metadata")
        
        tokenized = datasets.map(
            self.tokenize_function,
            batched=True,
            batch_size=1000,  # Process in batches for memory efficiency
            num_proc=num_proc,
            remove_columns=columns_to_remove,
            desc="Tokenizing"
        )
        
        return tokenized
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted', zero_division=0
        )
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    def train(self):
        """Main training function optimized for large datasets"""
        logger.info("=" * 70)
        logger.info("Starting Large-Scale Intent Classifier Training")
        logger.info(f"Expected dataset size: 1.6M+ samples")
        logger.info("=" * 70)
        
        # 1. Load and prepare datasets
        datasets = self.load_datasets()
        tokenized_datasets = self.prepare_datasets(datasets)
        
        # Clear memory
        del datasets
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 2. Load model with memory optimizations
        logger.info(f"Loading model: {self.model_config.model_name}")
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_config.model_name,
            num_labels=self.model_config.num_labels,
            id2label=self.label_mapping["id2label"],
            label2id=self.label_mapping["label2id"]
        )
        
        # Enable gradient checkpointing for memory efficiency
        if self.training_config.gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        
        model.to(self.device)
        
        # 3. Data collator with dynamic padding
        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding='longest',  # Pad to longest in batch (more efficient)
            max_length=self.model_config.max_length
        )
        
        # 4. Determine output directory
        if os.path.exists("/mnt/sagemaker-nvme"):
            output_dir = "/mnt/sagemaker-nvme/intent_models"
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Using NVMe storage: {output_dir}")
        else:
            output_dir = self.model_config.output_dir
            os.makedirs(output_dir, exist_ok=True)
        
        # 5. Calculate training steps for logging
        train_size = len(tokenized_datasets['train'])
        effective_batch_size = (
            self.training_config.per_device_batch_size * 
            self.training_config.gradient_accumulation_steps
        )
        steps_per_epoch = train_size // effective_batch_size
        total_steps = steps_per_epoch * self.training_config.num_epochs
        
        logger.info(f"Training samples: {train_size:,}")
        logger.info(f"Effective batch size: {effective_batch_size}")
        logger.info(f"Steps per epoch: {steps_per_epoch:,}")
        logger.info(f"Total training steps: {total_steps:,}")
        
        # 6. Training arguments optimized for large datasets
        training_args = TrainingArguments(
            output_dir=output_dir,
            
            # Epochs
            num_train_epochs=self.training_config.num_epochs,
            
            # Batch sizes
            per_device_train_batch_size=self.training_config.per_device_batch_size,
            per_device_eval_batch_size=self.training_config.per_device_batch_size * 2,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            
            # Learning rate
            learning_rate=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
            warmup_ratio=self.training_config.warmup_ratio,
            
            # Evaluation
            eval_strategy=self.training_config.eval_strategy,
            eval_steps=self.training_config.eval_steps,
            save_strategy="steps",
            save_steps=self.training_config.save_steps,
            logging_steps=self.training_config.logging_steps,
            
            # Best model tracking
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=self.training_config.save_total_limit,
            
            # Mixed precision
            fp16=self.training_config.fp16 and torch.cuda.is_available(),
            bf16=self.training_config.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            
            # Data loading optimization
            dataloader_num_workers=self.training_config.dataloader_num_workers if torch.cuda.is_available() else 0,
            dataloader_pin_memory=self.training_config.dataloader_pin_memory and torch.cuda.is_available(),
            
            # Memory optimization
            gradient_checkpointing=self.training_config.gradient_checkpointing,
            optim="adamw_torch",  # Use PyTorch AdamW
            
            # Logging
            report_to="none",
            logging_first_step=True,
            
            # For very large datasets, skip eval on start
            eval_on_start=False,
        )
        
        self.model_config.output_dir = output_dir
        
        # 7. Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets['train'],
            eval_dataset=tokenized_datasets['validation'],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=self.training_config.early_stopping_patience
                )
            ]
        )
        
        # 8. Train
        logger.info("Starting training...")
        train_result = trainer.train()
        
        # 9. Save final model
        logger.info("Saving model...")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save label mapping
        with open(Path(output_dir) / "label_mapping.json", 'w') as f:
            json.dump(self.label_mapping, f, indent=2)
        
        # 10. Evaluate on test set
        logger.info("Evaluating on test set...")
        test_results = trainer.evaluate(tokenized_datasets['test'])
        logger.info(f"Test Results: {test_results}")
        
        # 11. Generate classification report (on a sample if test set is huge)
        logger.info("Generating classification report...")
        test_dataset = tokenized_datasets['test']
        
        # If test set is very large, sample it for the report
        if len(test_dataset) > 50000:
            logger.info(f"Sampling 50,000 from {len(test_dataset):,} test samples for detailed report")
            test_dataset = test_dataset.shuffle(seed=42).select(range(50000))
        
        predictions = trainer.predict(test_dataset)
        pred_labels = np.argmax(predictions.predictions, axis=1)
        true_labels = test_dataset["label"]
        
        # Get all label IDs in sorted order
        all_labels = sorted(self.label_mapping["id2label"].keys())
        label_names = [self.label_mapping["id2label"][i] for i in all_labels]
        
        report = classification_report(
            true_labels,
            pred_labels,
            labels=all_labels,
            target_names=label_names,
            output_dict=True,
            zero_division=0
        )
        
        # Save report
        with open(Path(output_dir) / "classification_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print report
        print("\n" + "=" * 70)
        print("CLASSIFICATION REPORT")
        print("=" * 70)
        print(classification_report(
            true_labels,
            pred_labels,
            labels=all_labels,
            target_names=label_names,
            zero_division=0
        ))
        
        logger.info("=" * 70)
        logger.info("Training Complete!")
        logger.info(f"Model saved to: {output_dir}")
        logger.info("=" * 70)
        
        return {
            "train_result": train_result,
            "test_results": test_results,
            "classification_report": report
        }


def main():
    """Main entry point for large-scale training"""
    
    # Log GPU info
    if torch.cuda.is_available():
        logger.info(f"GPU Available: {torch.cuda.get_device_name(0)}")
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU Memory: {gpu_mem:.2f} GB")
        
        # Adjust batch size based on GPU memory
        if gpu_mem >= 40:  # A100 or similar
            batch_size = 64
            grad_accum = 2
        elif gpu_mem >= 16:  # V100, T4, etc
            batch_size = 32
            grad_accum = 4
        elif gpu_mem >= 8:  # RTX 3070, etc
            batch_size = 16
            grad_accum = 8
        else:
            batch_size = 8
            grad_accum = 16
    else:
        logger.warning("No GPU available - training on CPU (will be extremely slow!)")
        batch_size = 8
        grad_accum = 1
    
    # Initialize trainer with optimized config
    trainer = LargeDatasetIntentTrainer(
        model_config=ModelConfig(
            model_name="google/muril-base-cased",
            max_length=128,
            output_dir="../models/intent_classifier"
        ),
        training_config=LargeScaleTrainingConfig(
            num_epochs=3,  # 3 epochs is usually enough for 1.6M samples
            per_device_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=2e-5,
            early_stopping_patience=3,
            eval_steps=2000,
            save_steps=2000,
            logging_steps=500,
            use_augmented=False
        ),
        data_dir="../data/processed"
    )
    
    # Train
    results = trainer.train()
    
    return results


if __name__ == "__main__":
    main()
