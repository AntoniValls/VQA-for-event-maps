import json
from typing import Dict, List


class PromptManager:
    """Manages VQA prompts from JSON configuration file."""
    
    def __init__(self, json_path: str = "../inout/vqa_prompts.json", preset: str = "safety_critical"):
        """
        Initialize PromptManager.
        
        Args:
            json_path: Path to JSON file with prompts
            preset: Name of preset configuration to use
        """
        with open(json_path, 'r') as f:
            self.config = json.load(f)
        
        self.base_context = self.config['metadata']['base_context']
        self.preset = preset
        self.prompts = self._load_prompts()
        
    def _load_prompts(self) -> List[Dict]:
        """Load enabled prompts based on preset."""
        preset_config = self.config['prompt_presets'].get(self.preset)
        if not preset_config:
            raise ValueError(f"Preset '{self.preset}' not found")
        
        enabled_ids = preset_config['enabled_questions']
        
        # Collect all prompts
        all_prompts = []
        for category_name, category_data in self.config['prompt_categories'].items():
            for question in category_data['questions']:
                if question.get('enabled', True):
                    all_prompts.append(question)
        
        # Filter by preset
        if enabled_ids == "all":
            return all_prompts
        else:
            return [q for q in all_prompts if q['id'] in enabled_ids]
    
    def get_prompts(self) -> List[Dict]:
        """Get list of enabled prompts."""
        return self.prompts
    
    def get_full_prompt(self, question_data: Dict) -> str:
        """Get full prompt with base context."""
        return f"{self.base_context} {question_data['text']}"
    
    def get_short_label(self, question_data: Dict) -> str:
        """Get short label for display."""
        return question_data.get('short_label', question_data['id'])
    
    