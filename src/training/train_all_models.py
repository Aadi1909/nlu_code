import os
import sys
import json
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from training.train_intent import IntentClassifierTrainer
from training.train_entity import EntityExtractorTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_all_models():
    """Train both intent classifier and entity extractor"""
    
    logger.info("=" * 80)
    logger.info("BATTERY SMART NLU - COMPLETE TRAINING PIPELINE")
    logger.info("=" * 80)
    
    # Check if data exists
    data_dir = Path('data/processed')
    if not (data_dir / 'train.json').exists():
        logger.error("Training data not found!")
        logger.error("Please run data preprocessing first:")
        logger.error("  python src/data_preprocessing/data_preparation_pipeline.py")
        sys.exit(1)
    
    # Create model directories
    os.makedirs('models/intent_classifier', exist_ok=True)
    os.makedirs('models/entity_extractor', exist_ok=True)
    
    logger.info("\n" + "=" * 80)
    logger.info("PART 1: TRAINING INTENT CLASSIFIER")
    logger.info("=" * 80)
    
    try:
        intent_trainer = IntentClassifierTrainer(
            model_name="microsoft/mdeberta-v3-base",
            output_dir="models/intent_classifier"
        )
        
        logger.info("\n📂 Loading data for intent classification...")
        train_dataset, val_dataset = intent_trainer.prepare_data(
            train_path='data/processed/train.json',
            val_path='data/processed/val.json'
        )
        
        logger.info("\n🚀 Training intent classifier...")
        intent_trainer.train(train_dataset, val_dataset, epochs=10)
        
        logger.info("\n📊 Evaluating intent classifier on test set...")
        intent_results = intent_trainer.evaluate_detailed(
            test_path='data/processed/test.json'
        )
        
        logger.info("\n✅ Intent classifier training complete!")
        
    except Exception as e:
        logger.error(f"❌ Intent classifier training failed: {str(e)}")
        raise
    
    # ==========================================
    # PART 2: TRAIN ENTITY EXTRACTOR
    # ==========================================
    
    logger.info("\n" + "=" * 80)
    logger.info("PART 2: TRAINING ENTITY EXTRACTOR")
    logger.info("=" * 80)
    
    try:
        entity_trainer = EntityExtractorTrainer(
            model_name="xlm-roberta-base",
            output_dir="models/entity_extractor"
        )
        
        logger.info("\n📂 Loading data for entity extraction...")
        train_dataset, val_dataset = entity_trainer.prepare_data(
            train_path='data/processed/train.json',
            val_path='data/processed/val.json'
        )
        
        logger.info("\n🚀 Training entity extractor...")
        entity_trainer.train(train_dataset, val_dataset, epochs=10)
        
        logger.info("\n✅ Entity extractor training complete!")
        
    except Exception as e:
        logger.error(f"❌ Entity extractor training failed: {str(e)}")
        raise
    
    logger.info("\n" + "=" * 80)
    logger.info("🎉 TRAINING COMPLETE!")
    logger.info("=" * 80)
    logger.info("\nModels saved to:")
    logger.info("  - models/intent_classifier/")
    logger.info("  - models/entity_extractor/")
    logger.info("\nNext steps:")
    logger.info("  1. Test the NLU pipeline: python src/nlu/test_nlu_pipeline.py")
    logger.info("  2. Start the API server: python src/api/main.py")
    logger.info("=" * 80)


if __name__ == "__main__":
    train_all_models()