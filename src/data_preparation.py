import yaml
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split
from loguru import logger
import random

class DataPreparator:
    """
    Prepare training data from YAML configs for NLU model training
    """
    
    def __init__(self, config_dir: str = "config", output_dir: str = "data/processed"):
        self.config_dir = Path(config_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_yaml(self, filename: str) -> Dict:
        """Load YAML file"""
        filepath = self.config_dir / filename
        with open(filepath, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def prepare_intent_data(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Prepare intent classification training data
        Returns DataFrame and label mapping
        """
        logger.info("Loading intents from YAML...")
        intents_config = self.load_yaml("intents.yaml")
        
        data = []
        intent_list = []
        
        for intent_config in intents_config.get("intents", []):
            intent_name = intent_config.get("intent")
            intent_list.append(intent_name)
            examples = intent_config.get("examples", {})
            
            # Collect examples from all languages
            for language, texts in examples.items():
                for text in texts:
                    data.append({
                        "text": text,
                        "intent": intent_name,
                        "language": language
                    })
        
        # Create label mapping
        label2id = {label: idx for idx, label in enumerate(sorted(set(intent_list)))}
        id2label = {idx: label for label, idx in label2id.items()}
        
        df = pd.DataFrame(data)
        
        # Add label IDs
        df["label"] = df["intent"].map(label2id)
        
        logger.info(f"Prepared {len(df)} examples for {len(label2id)} intents")
        
        return df, {"label2id": label2id, "id2label": id2label}
    
    def augment_data(self, df: pd.DataFrame, augmentation_factor: int = 2) -> pd.DataFrame:
        """
        Simple data augmentation for training
        """
        logger.info("Augmenting data...")
        augmented_data = []
        
        for _, row in df.iterrows():
            text = row["text"]
            
            # Original
            augmented_data.append(row.to_dict())
            
            # Augmentation 1: Random word dropout (10% of words)
            words = text.split()
            if len(words) > 3:
                dropout_text = ' '.join([w for w in words if random.random() > 0.1])
                if dropout_text.strip():
                    new_row = row.to_dict()
                    new_row["text"] = dropout_text
                    augmented_data.append(new_row)
            
            # Augmentation 2: Add common filler words (for Hinglish)
            fillers = ["bhai", "yaar", "please", "zara", "thoda"]
            if row["language"] == "hinglish" and random.random() > 0.5:
                filler = random.choice(fillers)
                positions = [f"{filler} {text}", f"{text} {filler}"]
                new_row = row.to_dict()
                new_row["text"] = random.choice(positions)
                augmented_data.append(new_row)
        
        augmented_df = pd.DataFrame(augmented_data)
        logger.info(f"Augmented from {len(df)} to {len(augmented_df)} examples")
        
        return augmented_df
    
    def create_train_test_split(
        self, 
        df: pd.DataFrame, 
        test_size: float = 0.15,
        val_size: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create train/validation/test splits with stratification
        """
        logger.info("Creating train/val/test splits...")
        
        # First split: train+val vs test
        train_val_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            stratify=df["intent"],
            random_state=42
        )
        
        # Second split: train vs val
        adjusted_val_size = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=adjusted_val_size,
            stratify=train_val_df["intent"],
            random_state=42
        )
        
        logger.info(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def save_datasets(
        self, 
        train_df: pd.DataFrame, 
        val_df: pd.DataFrame, 
        test_df: pd.DataFrame,
        label_mapping: Dict
    ):
        """Save processed datasets"""
        
        # Save as CSV
        train_df.to_csv(self.output_dir / "train.csv", index=False)
        val_df.to_csv(self.output_dir / "val.csv", index=False)
        test_df.to_csv(self.output_dir / "test.csv", index=False)
        
        # Save as JSON (for Hugging Face datasets)
        train_df.to_json(self.output_dir / "train.json", orient="records", lines=True, force_ascii=False)
        val_df.to_json(self.output_dir / "val.json", orient="records", lines=True, force_ascii=False)
        test_df.to_json(self.output_dir / "test.json", orient="records", lines=True, force_ascii=False)
        
        # Save label mapping
        with open(self.output_dir / "label_mapping.json", 'w') as f:
            json.dump(label_mapping, f, indent=2)
        
        logger.info(f"Saved all datasets to {self.output_dir}")
    
    def prepare_entity_data(self) -> pd.DataFrame:
        """
        Prepare entity extraction training data (for NER)
        """
        logger.info("Preparing entity data...")
        entities_config = self.load_yaml("entities.yaml")
        
        entity_examples = []
        
        for entity in entities_config.get("entities", []):
            entity_name = entity.get("entity")
            examples = entity.get("examples", [])
            
            for example in examples:
                entity_examples.append({
                    "entity_type": entity_name,
                    "example": example
                })
        
        df = pd.DataFrame(entity_examples)
        df.to_csv(self.output_dir / "entity_examples.csv", index=False)
        
        logger.info(f"Prepared {len(df)} entity examples")
        return df
    
    def run_full_preparation(self) -> Dict:
        """
        Run the complete data preparation pipeline
        """
        logger.info("=" * 50)
        logger.info("Starting Data Preparation Pipeline")
        logger.info("=" * 50)
        
        # 1. Prepare intent data
        intent_df, label_mapping = self.prepare_intent_data()
        
        # 2. Augment data
        augmented_df = self.augment_data(intent_df)
        
        # 3. Create splits
        train_df, val_df, test_df = self.create_train_test_split(augmented_df)
        
        # 4. Save datasets
        self.save_datasets(train_df, val_df, test_df, label_mapping)
        
        # 5. Prepare entity data
        entity_df = self.prepare_entity_data()
        
        # 6. Generate statistics
        stats = {
            "total_examples": len(augmented_df),
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df),
            "num_intents": len(label_mapping["label2id"]),
            "intents": list(label_mapping["label2id"].keys()),
            "languages": list(augmented_df["language"].unique()),
            "examples_per_intent": augmented_df.groupby("intent").size().to_dict()
        }
        
        # Save stats
        with open(self.output_dir / "dataset_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info("=" * 50)
        logger.info("Data Preparation Complete!")
        logger.info(f"Stats: {json.dumps(stats, indent=2)}")
        logger.info("=" * 50)
        
        return stats


def main():
    """Main entry point"""
    preparator = DataPreparator(
        config_dir="config",
        output_dir="data/processed"
    )
    stats = preparator.run_full_preparation()
    return stats


if __name__ == "__main__":
    main()