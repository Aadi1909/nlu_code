import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

class IntentDataset(Dataset):
    """PyTorch Dataset for intent classification"""
    
    def __init__(self, data: List[Dict], tokenizer, intent_to_id: Dict):
        self.data = data
        self.tokenizer = tokenizer
        self.intent_to_id = intent_to_id
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            item['text'],
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        # Get intent label
        label = self.intent_to_id[item['intent']]
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class IntentClassifierTrainer:
    """Train intent classification model"""
    
    def __init__(self, 
                 model_name: str = "microsoft/mdeberta-v3-base",
                 output_dir: str = "models/intent_classifier"):
        self.model_name = model_name
        self.output_dir = output_dir
        self.tokenizer = None
        self.model = None
        self.intent_to_id = {}
        self.id_to_intent = {}
        
    def prepare_data(self, train_path: str, val_path: str):
        """Load and prepare datasets"""
        # Load data
        with open(train_path, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        
        with open(val_path, 'r', encoding='utf-8') as f:
            val_data = json.load(f)
        
        # Create intent mappings
        all_intents = sorted(set([item['intent'] for item in train_data]))
        self.intent_to_id = {intent: idx for idx, intent in enumerate(all_intents)}
        self.id_to_intent = {idx: intent for intent, idx in self.intent_to_id.items()}
        
        print(f"Found {len(all_intents)} intents:")
        for intent, idx in self.intent_to_id.items():
            print(f"  {idx}: {intent}")
        
        # Save mappings
        with open(f'{self.output_dir}/intent_mapping.json', 'w') as f:
            json.dump({
                'intent_to_id': self.intent_to_id,
                'id_to_intent': self.id_to_intent
            }, f, indent=2)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Create datasets
        train_dataset = IntentDataset(train_data, self.tokenizer, self.intent_to_id)
        val_dataset = IntentDataset(val_data, self.tokenizer, self.intent_to_id)
        
        return train_dataset, val_dataset
    
    def train(self, train_dataset, val_dataset, epochs: int = 10):
        """Train the model"""
        # Initialize model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.intent_to_id),
            problem_type="single_label_classification"
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            learning_rate=2e-5,
            weight_decay=0.01,
            warmup_steps=500,
            logging_dir=f'{self.output_dir}/logs',
            logging_steps=50,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=3,
            report_to="none"
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train
        print("\n🚀 Starting training...")
        trainer.train()
        
        # Save model
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        print(f"\n✅ Model saved to {self.output_dir}")
        
        return trainer
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted', zero_division=0
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def evaluate_detailed(self, test_path: str):
        """Detailed evaluation on test set"""
        # Load test data
        with open(test_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        test_dataset = IntentDataset(test_data, self.tokenizer, self.intent_to_id)
        
        # Create trainer for evaluation
        trainer = Trainer(
            model=self.model,
            compute_metrics=self.compute_metrics
        )
        
        # Evaluate
        results = trainer.evaluate(test_dataset)
        
        print("\n=== Test Set Results ===")
        for key, value in results.items():
            print(f"{key}: {value:.4f}")
        
        # Get predictions
        predictions = trainer.predict(test_dataset)
        pred_labels = np.argmax(predictions.predictions, axis=1)
        true_labels = predictions.label_ids
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, pred_labels)
        
        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=list(self.intent_to_id.keys()),
            yticklabels=list(self.intent_to_id.keys())
        )
        plt.title('Intent Classification Confusion Matrix')
        plt.ylabel('True Intent')
        plt.xlabel('Predicted Intent')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/confusion_matrix.png')
        print(f"\n📊 Confusion matrix saved to {self.output_dir}/confusion_matrix.png")
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, pred_labels, zero_division=0
        )
        
        print("\n=== Per-Intent Metrics ===")
        for idx, intent in self.id_to_intent.items():
            print(f"{intent}:")
            print(f"  Precision: {precision[idx]:.3f}")
            print(f"  Recall: {recall[idx]:.3f}")
            print(f"  F1: {f1[idx]:.3f}")
            print(f"  Support: {support[idx]}")
        
        return results


# Training Script
if __name__ == "__main__":
    import os
    
    # Create output directory
    os.makedirs('models/intent_classifier', exist_ok=True)
    
    # Initialize trainer
    trainer = IntentClassifierTrainer(
        model_name="microsoft/mdeberta-v3-base",  # Good for multilingual
        output_dir="models/intent_classifier"
    )
    
    # Prepare data
    print("📂 Loading data...")
    train_dataset, val_dataset = trainer.prepare_data(
        train_path='data/processed/train.json',
        val_path='data/processed/val.json'
    )
    
    # Train
    trainer.train(train_dataset, val_dataset, epochs=10)
    
    # Evaluate
    print("\n📊 Evaluating on test set...")
    trainer.evaluate_detailed(test_path='data/processed/test.json')
    
    print("\n✅ Intent classifier training complete!")