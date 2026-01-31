#!/usr/bin/env python3
"""
Train intent classifier using the FULL training_data.json (6000+ examples)
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from loguru import logger

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import warnings
warnings.filterwarnings("ignore")


@dataclass
class ModelConfig:
    model_name: str = "google/muril-base-cased"
    max_length: int = 128
    num_labels: int = 10  # 10 intents in training_data.json


@dataclass  
class TrainingConfig:
    num_epochs: int = 10
    batch_size: int = 32  # Larger batch for more data
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    eval_steps: int = 100
    save_steps: int = 100
    logging_steps: int = 50
    early_stopping_patience: int = 3


def main():
    print("🚀 Training Intent Classifier with FULL Dataset")
    print("=" * 70)
    
    # Paths
    data_dir = Path("../data/processed")
    
    # Determine output directory
    if os.path.exists("/mnt/sagemaker-nvme"):
        output_dir = "/mnt/sagemaker-nvme/intent_models_full"
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Using NVMe disk: {output_dir}")
    else:
        output_dir = "../models/intent_classifier_full"
        os.makedirs(output_dir, exist_ok=True)
    
    # Check for full data files
    train_path = data_dir / "train_full.json"
    val_path = data_dir / "val_full.json"
    test_path = data_dir / "test_full.json"
    label_path = data_dir / "label_mapping_full.json"
    
    if not train_path.exists():
        print("❌ Full dataset not found!")
        print("   Run: python process_raw_training_data.py first")
        return
    
    # Load label mapping
    with open(label_path, 'r') as f:
        label_mapping = json.load(f)
    
    id2label = {int(k): v for k, v in label_mapping["id2label"].items()}
    label2id = label_mapping["label2id"]
    num_labels = len(label2id)
    
    print(f"📊 Number of intents: {num_labels}")
    
    # Load tokenizer
    model_config = ModelConfig()
    logger.info(f"Loading tokenizer: {model_config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
    
    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = load_dataset('json', data_files=str(train_path), split='train')
    val_dataset = load_dataset('json', data_files=str(val_path), split='train')
    test_dataset = load_dataset('json', data_files=str(test_path), split='train')
    
    print(f"✅ Loaded - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=model_config.max_length,
            padding=False
        )
    
    logger.info("Tokenizing datasets...")
    train_tokenized = train_dataset.map(tokenize_function, batched=True, remove_columns=["text", "intent"])
    val_tokenized = val_dataset.map(tokenize_function, batched=True, remove_columns=["text", "intent"])
    test_tokenized = test_dataset.map(tokenize_function, batched=True, remove_columns=["text", "intent"])
    
    # Remove extra columns that might cause issues
    columns_to_remove = [col for col in train_tokenized.column_names if col not in ['input_ids', 'attention_mask', 'label']]
    train_tokenized = train_tokenized.remove_columns(columns_to_remove)
    val_tokenized = val_tokenized.remove_columns([col for col in val_tokenized.column_names if col not in ['input_ids', 'attention_mask', 'label']])
    test_tokenized = test_tokenized.remove_columns([col for col in test_tokenized.column_names if col not in ['input_ids', 'attention_mask', 'label']])
    
    # Load model
    logger.info(f"Loading model: {model_config.model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Using device: {device}")
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Compute metrics
    def compute_metrics(eval_pred):
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
    
    # Training arguments
    training_config = TrainingConfig()
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_config.num_epochs,
        per_device_train_batch_size=training_config.batch_size,
        per_device_eval_batch_size=training_config.batch_size * 2,
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
        warmup_ratio=training_config.warmup_ratio,
        eval_strategy="steps",
        eval_steps=training_config.eval_steps,
        save_strategy="steps",
        save_steps=training_config.save_steps,
        logging_steps=training_config.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none",
        dataloader_num_workers=2 if torch.cuda.is_available() else 0,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=training_config.early_stopping_patience)]
    )
    
    # Train
    logger.info("Starting training...")
    train_result = trainer.train()
    
    # Save
    logger.info("Saving model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    with open(Path(output_dir) / "label_mapping.json", 'w') as f:
        json.dump(label_mapping, f, indent=2)
    
    # Evaluate
    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate(test_tokenized)
    logger.info(f"Test Results: {test_results}")
    
    # Classification report
    predictions = trainer.predict(test_tokenized)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = test_tokenized["label"]
    
    all_labels = sorted(id2label.keys())
    label_names = [id2label[i] for i in all_labels]
    
    report = classification_report(
        true_labels,
        pred_labels,
        labels=all_labels,
        target_names=label_names,
        output_dict=True,
        zero_division=0
    )
    
    with open(Path(output_dir) / "classification_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
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
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print(f"   Accuracy: {test_results['eval_accuracy']:.2%}")
    print(f"   F1 Score: {test_results['eval_f1']:.2%}")
    print(f"   Model saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
