#!/usr/bin/env python3
"""
Utility script for managing VQA prompts.

Usage:
    python prompt_utils.py list-presets
    python prompt_utils.py list-questions [--preset PRESET]
    python prompt_utils.py show-question QUESTION_ID
    python prompt_utils.py create-preset NEW_PRESET --questions q1,q2,q3
    python prompt_utils.py stats
"""

import json
import argparse
from pathlib import Path
from typing import List


class PromptUtils:
    """Utility class for managing VQA prompts."""
    
    def __init__(self, json_path: str):
        self.json_path = Path(json_path)
        with open(self.json_path, 'r') as f:
            self.config = json.load(f)
    
    def list_presets(self):
        """List all available presets."""
        print("\n" + "="*70)
        print("AVAILABLE PRESETS")
        print("="*70)
        
        for preset_name, preset_data in self.config['prompt_presets'].items():
            enabled = preset_data.get('enabled_questions', [])
            count = len(enabled) if enabled != "all" else "all"
            print(f"\n{preset_name}")
            print(f"  Description: {preset_data['description']}")
            print(f"  Questions: {count}")
    
    def list_questions(self, preset: str = None):
        """List all questions, optionally filtered by preset."""
        print("\n" + "="*70)
        if preset:
            print(f"QUESTIONS IN PRESET: {preset}")
        else:
            print("ALL QUESTIONS")
        print("="*70)
        
        # Get enabled question IDs for preset
        enabled_ids = None
        if preset:
            preset_config = self.config['prompt_presets'].get(preset)
            if preset_config:
                enabled_ids = preset_config['enabled_questions']
                if enabled_ids == "all":
                    enabled_ids = None
        
        # List questions by category
        for category_name, category_data in self.config['prompt_categories'].items():
            print(f"\n{category_data['name']}")
            print("-" * 70)
            
            for question in category_data['questions']:
                q_id = question['id']
                
                # Skip if preset filter active and question not in preset
                if enabled_ids and q_id not in enabled_ids:
                    continue
                
                enabled = "✓" if question.get('enabled', True) else "✗"
                short = question.get('short_label', 'N/A')
                
                print(f"  [{enabled}] {q_id}")
                print(f"      Label: {short}")
                print(f"      Type: {question['type']}")
                
                # Show dependency if exists
                if 'dependency' in question:
                    print(f"      Depends on: {question['dependency']}")
    
    def show_question(self, question_id: str):
        """Show detailed information about a specific question."""
        # Find the question
        found = None
        for category_name, category_data in self.config['prompt_categories'].items():
            for question in category_data['questions']:
                if question['id'] == question_id:
                    found = (category_data['name'], question)
                    break
            if found:
                break
        
        if not found:
            print(f"Question '{question_id}' not found!")
            return
        
        category_name, question = found
        
        print("\n" + "="*70)
        print(f"QUESTION: {question_id}")
        print("="*70)
        print(f"Category: {category_name}")
        print(f"Short Label: {question.get('short_label', 'N/A')}")
        print(f"Topic: {question['topic']}")
        print(f"Type: {question['type']}")
        print(f"Enabled: {question.get('enabled', True)}")
        
        if 'dependency' in question:
            print(f"Dependency: {question['dependency']}")
        
        print(f"\nQuestion Text:")
        print(f"  {question['text']}")
        
        if 'choices' in question:
            print(f"\nChoices:")
            for choice in question['choices']:
                print(f"  - {choice}")
        
        if 'instructions' in question:
            print(f"\nInstructions:")
            print(f"  {question['instructions']}")
    
    def show_stats(self):
        """Show statistics about the prompt configuration."""
        total_questions = 0
        enabled_questions = 0
        by_type = {}
        by_category = {}
        
        for category_name, category_data in self.config['prompt_categories'].items():
            cat_count = len(category_data['questions'])
            by_category[category_name] = cat_count
            
            for question in category_data['questions']:
                total_questions += 1
                if question.get('enabled', True):
                    enabled_questions += 1
                
                q_type = question['type']
                by_type[q_type] = by_type.get(q_type, 0) + 1
        
        print("\n" + "="*70)
        print("PROMPT STATISTICS")
        print("="*70)
        print(f"\nTotal Questions: {total_questions}")
        print(f"Enabled Questions: {enabled_questions}")
        print(f"Disabled Questions: {total_questions - enabled_questions}")
        print(f"\nPresets: {len(self.config['prompt_presets'])}")
        print(f"Categories: {len(self.config['prompt_categories'])}")
        
        print(f"\nQuestions by Type:")
        for q_type, count in sorted(by_type.items()):
            print(f"  {q_type:20s}: {count}")
        
        print(f"\nQuestions by Category:")
        for cat_name, count in by_category.items():
            print(f"  {cat_name:30s}: {count}")
    
    def create_preset(self, preset_name: str, description: str, question_ids: List[str]):
        """Create a new preset configuration."""
        # Validate question IDs
        all_ids = []
        for category_data in self.config['prompt_categories'].values():
            for question in category_data['questions']:
                all_ids.append(question['id'])
        
        invalid_ids = [qid for qid in question_ids if qid not in all_ids]
        if invalid_ids:
            print(f"Error: Invalid question IDs: {invalid_ids}")
            return
        
        # Create preset
        self.config['prompt_presets'][preset_name] = {
            "name": preset_name,
            "description": description,
            "enabled_questions": question_ids
        }
        
        # Save
        with open(self.json_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print(f"✓ Created preset '{preset_name}' with {len(question_ids)} questions")


def main():
    parser = argparse.ArgumentParser(description="VQA Prompt Management Utility")
    parser.add_argument('command', choices=['list-presets', 'list-questions', 'show-question', 
                                           'create-preset', 'stats'],
                       help="Command to execute")
    parser.add_argument('args', nargs='*', help="Command arguments")
    parser.add_argument('--preset', help="Preset name (for list-questions)")
    parser.add_argument('--description', help="Preset description (for create-preset)")
    parser.add_argument('--questions', help="Comma-separated question IDs (for create-preset)")
    parser.add_argument('--json', default='../inout/vqa_prompts.json', help="Path to prompts JSON file")
    
    args = parser.parse_args()
    
    utils = PromptUtils(args.json)
    
    if args.command == 'list-presets':
        utils.list_presets()
    
    elif args.command == 'list-questions':
        utils.list_questions(preset=args.preset)
    
    elif args.command == 'show-question':
        if not args.args:
            print("Error: Please provide a question ID")
            return
        utils.show_question(args.args[0])
    
    elif args.command == 'create-preset':
        if not args.args or not args.description or not args.questions:
            print("Error: create-preset requires: preset_name --description DESC --questions q1,q2,q3")
            return
        
        preset_name = args.args[0]
        question_ids = [q.strip() for q in args.questions.split(',')]
        utils.create_preset(preset_name, args.description, question_ids)
    
    elif args.command == 'stats':
        utils.show_stats()
    
    print()  # Final newline


if __name__ == "__main__":
    main()
