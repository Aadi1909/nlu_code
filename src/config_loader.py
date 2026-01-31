import yaml
from pathlib import Path
from typing import Dict, Any

class ConfigLoader:
    """Load and manage YAML configurations"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self._cache = {}
    
    def load(self, filename: str) -> Dict[str, Any]:
        """Load a YAML config file"""
        if filename in self._cache:
            return self._cache[filename]
        
        filepath = self.config_dir / filename
        with open(filepath, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        self._cache[filename] = config
        return config
    
    def get_intents(self) -> Dict:
        return self.load("intents.yaml")
    
    def get_entities(self) -> Dict:
        return self.load("entities.yaml")
    
    def get_slots(self) -> Dict:
        return self.load("slots.yml")
    
    def get_responses(self) -> Dict:
        return self.load("responses.yaml")
    
    def get_domain(self) -> Dict:
        return self.load("domain.yaml")
    
    def get_intent_examples(self, intent_name: str, language: str = None) -> list:
        """Get training examples for a specific intent"""
        intents = self.get_intents()
        for intent in intents.get("intents", []):
            if intent.get("intent") == intent_name:
                examples = intent.get("examples", {})
                if language:
                    return examples.get(language, [])
                # Return all examples flattened
                all_examples = []
                for lang_examples in examples.values():
                    all_examples.extend(lang_examples)
                return all_examples
        return []


# Usage
config = ConfigLoader()
intents = config.get_intents()
battery_examples = config.get_intent_examples("check_battery_status", "hinglish")