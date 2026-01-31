import os
import json
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict
from loguru import logger

from datasets import load_dataset, Dataset
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
    """Model configuration"""
    model_name: str = "google/muril-base-cased"  # Best for Indian languages
    max_length: int = 128
    num_labels: int = 20  # Will be updated from data
    output_dir: str = "models/intent_classifier"
    

@dataclass
class TrainingConfig:
    """Training configuration"""
    num_epochs: int = 15  # Increased from 10
    batch_size: int = 16
    learning_rate: float = 3e-5  # Slightly increased
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    eval_steps: int = 50  # More frequent evaluation
    save_steps: int = 50
    logging_steps: int = 25  # More frequent logging
    early_stopping_patience: int = 5  # More patience
    fp16: bool = True  # Use mixed precision if GPU available
    use_augmented: bool = False  # Use augmented data


class IntentClassifierTrainer:
    """
    Trainer for Intent Classification using MuRIL
    """
    
    def __init__(
        self, 
        model_config: ModelConfig = None,
        training_config: TrainingConfig = None,
        data_dir: str = "data/processed"
    ):
        self.model_config = model_config or ModelConfig()
        self.training_config = training_config or TrainingConfig()
        self.data_dir = Path(data_dir)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load label mapping
        self.label_mapping = self._load_label_mapping()
        self.model_config.num_labels = len(self.label_mapping["label2id"])
        
        # Initialize tokenizer
        logger.info(f"Loading tokenizer: {self.model_config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.model_name)
        
    def _load_label_mapping(self) -> Dict:
        """Load label mapping from data directory"""
        mapping_path = self.data_dir / "label_mapping.json"
        with open(mapping_path, 'r') as f:
            mapping = json.load(f)
        # Convert string keys to int for id2label
        mapping["id2label"] = {int(k): v for k, v in mapping["id2label"].items()}
        logger.info(f"Loaded {len(mapping['label2id'])} labels")
        return mapping
    
    def load_datasets(self):
        """Load train, validation, and test datasets"""
        logger.info("Loading datasets...")
        
        # Check if augmented data should be used
        file_suffix = "_augmented" if self.training_config.use_augmented else ""
        
        # Load from JSON files
        train_dataset = load_dataset('json', data_files=str(self.data_dir / f"train{file_suffix}.json"), split='train')
        val_dataset = load_dataset('json', data_files=str(self.data_dir / f"val{file_suffix}.json"), split='train')
        test_dataset = load_dataset('json', data_files=str(self.data_dir / f"test{file_suffix}.json"), split='train')

        # 🔒 Recompute labels from intent to avoid mismatched label ids in augmented data
        def _add_label(batch):
            return {"label": self.label_mapping["label2id"][batch["intent"]]}

        train_dataset = train_dataset.map(_add_label)
        val_dataset = val_dataset.map(_add_label)
        test_dataset = test_dataset.map(_add_label)
        
        logger.info(f"Loaded - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset
    
    def tokenize_function(self, examples):
        """Tokenize examples"""
        return self.tokenizer(
            examples["text"],
            truncation=True,
            max_length=self.model_config.max_length,
            padding=False  # Will be done by data collator
        )
    
    def prepare_datasets(self, train_dataset, val_dataset, test_dataset):
        """Tokenize all datasets"""
        logger.info("Tokenizing datasets...")
        
        train_tokenized = train_dataset.map(
            self.tokenize_function, 
            batched=True,
            remove_columns=["text", "intent", "language"]
        )
        
        val_tokenized = val_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=["text", "intent", "language"]
        )
        
        test_tokenized = test_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=["text", "intent", "language"]
        )
        
        return train_tokenized, val_tokenized, test_tokenized
    
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
        """Main training function"""
        logger.info("=" * 60)
        logger.info("Starting Intent Classifier Training")
        logger.info("=" * 60)
        
        # 1. Load datasets
        train_dataset, val_dataset, test_dataset = self.load_datasets()
        
        # 2. Tokenize
        train_tokenized, val_tokenized, test_tokenized = self.prepare_datasets(
            train_dataset, val_dataset, test_dataset
        )
        
        # 3. Load model
        logger.info(f"Loading model: {self.model_config.model_name}")
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_config.model_name,
            num_labels=self.model_config.num_labels,
            id2label=self.label_mapping["id2label"],
            label2id=self.label_mapping["label2id"]
        )
        model.to(self.device)
        
        # 4. Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # 5. Determine output directory (use NVMe on SageMaker if available)
        if os.path.exists("/mnt/sagemaker-nvme"):
            output_dir = "/mnt/sagemaker-nvme/intent_models"
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Using NVMe disk for checkpoints and model save: {output_dir}")
        else:
            output_dir = self.model_config.output_dir
            logger.info(f"Using default output directory: {output_dir}")
        
        # 6. Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.training_config.num_epochs,
            per_device_train_batch_size=self.training_config.batch_size,
            per_device_eval_batch_size=self.training_config.batch_size * 2,
            learning_rate=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
            warmup_ratio=self.training_config.warmup_ratio,
            eval_strategy="steps",  # Changed from evaluation_strategy in transformers 4.30+
            eval_steps=self.training_config.eval_steps,
            save_strategy="steps",
            save_steps=self.training_config.save_steps,
            logging_steps=self.training_config.logging_steps,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=3,
            fp16=self.training_config.fp16 and torch.cuda.is_available(),  # Enable FP16 if GPU available
            report_to="none",
            dataloader_num_workers=2 if torch.cuda.is_available() else 0,  # Use workers on GPU
        )
        
        # Update model config to use the same output directory
        self.model_config.output_dir = output_dir

        # 7. Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=val_tokenized,
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
        trainer.save_model(self.model_config.output_dir)
        self.tokenizer.save_pretrained(self.model_config.output_dir)
        
        # Save label mapping with model
        with open(Path(self.model_config.output_dir) / "label_mapping.json", 'w') as f:
            json.dump(self.label_mapping, f, indent=2)
        
        # 10. Evaluate on test set
        logger.info("Evaluating on test set...")
        test_results = trainer.evaluate(test_tokenized)
        logger.info(f"Test Results: {test_results}")
        
        # 11. Generate detailed classification report
        predictions = trainer.predict(test_tokenized)
        pred_labels = np.argmax(predictions.predictions, axis=1)
        true_labels = test_tokenized["label"]
        
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
        with open(Path(self.model_config.output_dir) / "classification_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print report
        print("\n" + "=" * 60)
        print("CLASSIFICATION REPORT")
        print("=" * 60)
        print(classification_report(
            true_labels,
            pred_labels,
            labels=all_labels,
            target_names=label_names,
            zero_division=0
        ))
        
        logger.info("=" * 60)
        logger.info("Training Complete!")
        logger.info(f"Model saved to: {self.model_config.output_dir}")
        logger.info("=" * 60)
        
        return {
            "train_result": train_result,
            "test_results": test_results,
            "classification_report": report
        }


def main():
    """Main entry point"""
    
    # Check GPU
    if torch.cuda.is_available():
        logger.info(f"GPU Available: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.warning("No GPU available, training on CPU (will be slow)")
    
    # Initialize trainer
    trainer = IntentClassifierTrainer(
        model_config=ModelConfig(
            model_name="google/muril-base-cased",
            max_length=128,
            output_dir="../models/intent_classifier"
        ),
        training_config=TrainingConfig(
            num_epochs=15,  # Increased
            batch_size=16 if torch.cuda.is_available() else 8,
            learning_rate=3e-5,  # Slightly higher
            early_stopping_patience=5,  # More patience
            use_augmented=True  # Use augmented data
        ),
        data_dir="../data/processed"
    )
    
    # Train
    results = trainer.train()
    
    return results


if __name__ == "__main__":
    main()