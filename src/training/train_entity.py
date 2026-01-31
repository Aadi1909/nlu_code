# src/training/train_entity.py

import json
import torch
from typing import List, Dict
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from seqeval.metrics import classification_report, f1_score
import numpy as np

class EntityDataset(Dataset):
    """Dataset for NER training"""
    
    def __init__(self, data: List[Dict], tokenizer, entity_to_id: Dict):
        self.data = data
        self.tokenizer = tokenizer
        self.entity_to_id = entity_to_id
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize
        tokenized = self.tokenizer(
            item['text'],
            truncation=True,
            max_length=128,
            padding='max_length',
            return_offsets_mapping=True
        )
        
        # Create labels
        labels = self._create_labels(
            item['text'],
            item.get('entities', []),
            tokenized['offset_mapping']
        )
        
        return {
            'input_ids': torch.tensor(tokenized['input_ids']),
            'attention_mask': torch.tensor(tokenized['attention_mask']),
            'labels': torch.tensor(labels)
        }
    
    def _create_labels(self, text, entities, offset_mapping):
        """Create BIO labels for tokens"""
        labels = [self.entity_to_id['O']] * len(offset_mapping)
        
        for entity in entities:
            entity_type = entity['entity']
            start = entity['start']
            end = entity['end']
            
            # Find tokens that overlap with entity span
            is_first_token = True
            for token_idx, (token_start, token_end) in enumerate(offset_mapping):
                if token_start >= start and token_end <= end:
                    if is_first_token:
                        labels[token_idx] = self.entity_to_id[f'B-{entity_type}']
                        is_first_token = False
                    else:
                        labels[token_idx] = self.entity_to_id[f'I-{entity_type}']
        
        return labels


class EntityExtractorTrainer:
    """Train entity extraction model"""
    
    def __init__(self,
                 model_name: str = "xlm-roberta-base",
                 output_dir: str = "models/entity_extractor"):
        self.model_name = model_name
        self.output_dir = output_dir
        self.tokenizer = None
        self.model = None
        self.entity_to_id = {}
        self.id_to_entity = {}
    
    def prepare_data(self, train_path: str, val_path: str):
        """Prepare NER datasets"""
        # Load data (JSON Lines format)
        train_data = []
        with open(train_path, 'r', encoding='utf-8') as f:
            for line in f:
                train_data.append(json.loads(line.strip()))
        
        val_data = []
        with open(val_path, 'r', encoding='utf-8') as f:
            for line in f:
                val_data.append(json.loads(line.strip()))
        
        # Collect all entity types
        entity_types = set()
        for item in train_data:
            if 'entities' in item:
                for entity in item['entities']:
                    entity_types.add(entity['entity'])
        
        # Create BIO tag mapping
        tags = ['O']  # Outside
        for entity_type in sorted(entity_types):
            tags.append(f'B-{entity_type}')  # Beginning
            tags.append(f'I-{entity_type}')  # Inside
        
        self.entity_to_id = {tag: idx for idx, tag in enumerate(tags)}
        self.id_to_entity = {idx: tag for tag, idx in self.entity_to_id.items()}
        
        print(f"Found {len(entity_types)} entity types:")
        print(tags)
        
        # Save mappings
        with open(f'{self.output_dir}/entity_mapping.json', 'w') as f:
            json.dump({
                'entity_to_id': self.entity_to_id,
                'id_to_entity': self.id_to_entity
            }, f, indent=2)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Create datasets
        train_dataset = EntityDataset(train_data, self.tokenizer, self.entity_to_id)
        val_dataset = EntityDataset(val_data, self.tokenizer, self.entity_to_id)
        
        return train_dataset, val_dataset
    
    def train(self, train_dataset, val_dataset, epochs: int = 10):
        """Train NER model"""
        # Initialize model
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.entity_to_id)
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            learning_rate=5e-5,
            weight_decay=0.01,
            warmup_steps=500,
            logging_steps=50,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            save_total_limit=3,
            report_to="none"
        )
        
        # Data collator
        data_collator = DataCollatorForTokenClassification(self.tokenizer)
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )
        
        # Train
        print("\n🚀 Starting NER training...")
        trainer.train()
        
        # Save
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        print(f"\n✅ NER model saved to {self.output_dir}")
        
        return trainer
    
    def compute_metrics(self, eval_pred):
        """Compute NER metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        
        # Convert to tag sequences
        true_labels = []
        pred_labels = []
        
        for pred_seq, label_seq in zip(predictions, labels):
            true_tags = []
            pred_tags = []
            
            for pred, label in zip(pred_seq, label_seq):
                if label != -100:  # Ignore padding
                    true_tags.append(self.id_to_entity[label])
                    pred_tags.append(self.id_to_entity[pred])
            
            true_labels.append(true_tags)
            pred_labels.append(pred_tags)
        
        # Calculate F1
        f1 = f1_score(true_labels, pred_labels)
        
        return {'f1': f1}


# Training script
if __name__ == "__main__":
    import os
    
    os.makedirs('../../models/entity_extractor', exist_ok=True)
    
    trainer = EntityExtractorTrainer(
        model_name="xlm-roberta-base",
        output_dir="../../models/entity_extractor"
    )
    
    train_dataset, val_dataset = trainer.prepare_data(
        train_path='../../data/processed/train.json',
        val_path='../../data/processed/val.json'
    )
    
    trainer.train(train_dataset, val_dataset, epochs=10)
    
    print("\n✅ Entity extractor training complete!")