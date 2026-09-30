import json
from typing import Dict, List, Optional

from common.paths import PROMPTS_PATH


class PromptManager:
    """Manages hierarchical VQA prompts from JSON configuration file."""

    def __init__(self, json_path: str = PROMPTS_PATH, preset: str = "level_1_only"):
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
        
        # Load all prompts and build dependency map
        self.all_prompts = self._load_all_prompts()
        self.dependency_map = self._build_dependency_map()
        
        # Get initial prompts based on preset
        self.initial_prompts = self._load_initial_prompts()
        
        # Track answers for dependency resolution
        self.answer_history = {}
        
    def _load_all_prompts(self) -> Dict[str, Dict]:
        """Load all prompts into a dictionary keyed by question ID."""
        all_prompts = {}
        for category_name, category_data in self.config['prompt_categories'].items():
            for question in category_data['questions']:
                if question.get('enabled', True):
                    all_prompts[question['id']] = question
        return all_prompts
    
    def _build_dependency_map(self) -> Dict[str, List[str]]:
        """Build a map of parent question -> list of dependent questions."""
        dep_map = {}
        for q_id, question in self.all_prompts.items():
            if 'dependency' in question:
                parent = question['dependency']
                if parent not in dep_map:
                    dep_map[parent] = []
                dep_map[parent].append(q_id)
        return dep_map
    
    def _load_initial_prompts(self) -> List[Dict]:
        """Load initial prompts (Level 1 questions without dependencies) based on preset."""
        preset_config = self.config['prompt_presets'].get(self.preset)
        if not preset_config:
            raise ValueError(f"Preset '{self.preset}' not found")
        
        enabled_ids = preset_config['enabled_questions']
        
        # Get questions without dependencies (Level 1)
        initial = []
        for q_id, question in self.all_prompts.items():
            # Check if question is in enabled list
            if enabled_ids != "all" and q_id not in enabled_ids:
                continue
            
            # Only include questions without dependencies
            if 'dependency' not in question:
                initial.append(question)
        
        return initial
    
    def get_initial_prompts(self) -> List[Dict]:
        """Get Level 1 prompts to ask first."""
        return self.initial_prompts
    
    def get_followup_prompts(self, question_id: str, answer: str) -> List[Dict]:
        """
        Get follow-up prompts based on a parent question's answer.
        
        Args:
            question_id: The ID of the parent question
            answer: The answer to the parent question (should contain 'yes' or 'no')
        
        Returns:
            List of follow-up questions if answer is affirmative, empty list otherwise
        """
        # Store answer for potential nested dependencies
        self.answer_history[question_id] = answer
        
        # Only trigger follow-ups if answer is affirmative (contains 'yes')
        if 'yes' not in answer.lower():
            return []
        
        # Get preset configuration
        preset_config = self.config['prompt_presets'].get(self.preset)
        enabled_ids = preset_config['enabled_questions']
        
        # Get dependent questions
        followups = []
        if question_id in self.dependency_map:
            for dep_q_id in self.dependency_map[question_id]:
                # Check if question is in enabled list
                if enabled_ids != "all" and dep_q_id not in enabled_ids:
                    continue
                
                followups.append(self.all_prompts[dep_q_id])
        
        return followups
    
    def reset_answer_history(self):
        """Reset answer history for new frame/image."""
        self.answer_history = {}
    
    def get_prompts(self) -> List[Dict]:
        """
        Get list of initial prompts (for backward compatibility).
        Use get_initial_prompts() and get_followup_prompts() for hierarchical processing.
        """
        return self.initial_prompts
    
    def get_full_prompt(self, question_data: Dict) -> str:
        """Get full prompt with base context."""
        return f"{self.base_context} {question_data['text']}"
    
    def get_short_label(self, question_data: Dict) -> str:
        """Get short label for display."""
        return question_data.get('short_label', question_data['id'])
    
    def get_question_by_id(self, question_id: str) -> Optional[Dict]:
        """Get a specific question by its ID."""
        return self.all_prompts.get(question_id)

    def get_level1_ancestor(self, question_id: str) -> str:
        """Follow the dependency chain up to the Level-1 question (the hazard category)."""
        seen = set()
        while question_id in self.all_prompts and 'dependency' in self.all_prompts[question_id]:
            if question_id in seen:
                break
            seen.add(question_id)
            question_id = self.all_prompts[question_id]['dependency']
        return question_id

    def get_level(self, question_id: str) -> int:
        """1 for primary questions, 2 for their follow-ups, 3 for follow-ups of follow-ups."""
        level = 1
        while question_id in self.all_prompts and 'dependency' in self.all_prompts[question_id] and level < 10:
            question_id = self.all_prompts[question_id]['dependency']
            level += 1
        return level
    
    def print_hierarchy_info(self):
        """Print information about the hierarchical structure."""
        print("\n" + "="*70)
        print(f"PROMPT HIERARCHY - Preset: {self.preset}")
        print("="*70)
        
        print(f"\nLevel 1 Questions ({len(self.initial_prompts)}):")
        for q in self.initial_prompts:
            q_id = q['id']
            label = self.get_short_label(q)
            num_deps = len(self.dependency_map.get(q_id, []))
            print(f"  • {label} ({q_id})")
            if num_deps > 0:
                print(f"    → Triggers {num_deps} follow-up question(s)")
        
        print(f"\nTotal questions in hierarchy: {len(self.all_prompts)}")
        print(f"Questions with dependencies: {sum(1 for q in self.all_prompts.values() if 'dependency' in q)}")
        print("="*70 + "\n")