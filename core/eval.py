"""
Metrics evaluation module for VQA Pedestrian Navigation System
Calculates F1, Accuracy, Specificity, and Recall for binary classification tasks
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from datetime import datetime
import string

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.paths import MODELS, list_sequences, sequence_dir
from core.promptManager import PromptManager

class MetricsEvaluator:
    """Evaluates VQA model predictions against ground truth labels including Risk Score Error."""
    
    # Risk Score Configuration
    HAZARD_CONFIG = {
        "CRITICAL": {"weight": 1.0, "ids": ["q_construction_visible", "q_surface_hazardous", "q_pedestrian_not_on_sidewalk"]},
        "HIGH":     {"weight": 0.6, "ids": ["q_crossing_nearby", "q_stairs_visible", "q_obstacle_blocking"]},
        "LOW":      {"weight": 0.3, "ids": ["q_pedestrians_present", "q_vehicle_nearby"]}
    }
    SAFETY_REWARD_RATIO = 1/8

    def __init__(self, predictions_path: str, ground_truth_path: str):
        self.predictions_path = predictions_path
        self.ground_truth_path = ground_truth_path
        self.predictions = []
        self.ground_truth = []
        # Map IDs to weights for risk calculation
        self.weight_lookup = {qid: cfg["weight"] for cfg in self.HAZARD_CONFIG.values() for qid in cfg["ids"]}
        
    def load_data(self) -> Tuple[List[Dict], List[Dict]]:
        def parse_mixed_json(path):
            data = []
            if not os.path.exists(path): return []
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read().replace('}{', '}\n{')
                for line in content.splitlines():
                    if line.strip():
                        try: data.append(json.loads(line))
                        except json.JSONDecodeError: continue
            return data

        self.predictions = parse_mixed_json(self.predictions_path)
        self.ground_truth = parse_mixed_json(self.ground_truth_path)
        self.prompt_manager = PromptManager(preset="full_hierarchical")
        return self.predictions, self.ground_truth
    
    def calculate_risk_score(self, questions_dict: Dict) -> float:
        """Calculates a normalized risk score [0, 1] for a set of answers for one image."""
        primary_ids = list(self.weight_lookup.keys())
        active_weights = [self.weight_lookup.get(qid, 0) for qid in primary_ids if qid in questions_dict]
        max_theoretical = sum(active_weights)
        
        if max_theoretical <= 0:
            return 0.0

        net_score = 0.0
        for q_id in primary_ids:
            if q_id not in questions_dict:
                continue

            ans = questions_dict[q_id].get('answer', '').lower().strip().translate(str.maketrans('', '', string.punctuation))
            weight = self.weight_lookup.get(q_id, 0)
            
            is_yes = any(pos in ans for pos in ['yes', 'true', 'hazard'])
            is_no = any(neg in ans for neg in ['no', 'false', 'safe'])
            
            if is_yes:
                net_score += weight
            elif is_no:
                net_score -= (weight * self.SAFETY_REWARD_RATIO)

        return min(max(0, net_score) / max_theoretical, 1.0)

    def normalize_yes_no(self, answer: str) -> str:
        if not isinstance(answer, str): return "no"
        a = answer.strip().lower()
        if a.startswith("yes"): return "yes"
        if a.startswith("no"): return "no"
        return a

    def calculate_binary_metrics(self, y_true: List[str], y_pred: List[str]) -> Dict:
        y_true_binary = [1 if self.normalize_yes_no(l) == "yes" else 0 for l in y_true]
        y_pred_binary = [1 if self.normalize_yes_no(l) == "yes" else 0 for l in y_pred]
        
        TP = sum(1 for t, p in zip(y_true_binary, y_pred_binary) if t == 1 and p == 1)
        TN = sum(1 for t, p in zip(y_true_binary, y_pred_binary) if t == 0 and p == 0)
        FP = sum(1 for t, p in zip(y_true_binary, y_pred_binary) if t == 0 and p == 1)
        FN = sum(1 for t, p in zip(y_true_binary, y_pred_binary) if t == 1 and p == 0)
        
        total = TP + TN + FP + FN
        accuracy = (TP + TN) / total if total > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN, 'Total': total, 
                'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 
                'Specificity': specificity, 'F1': f1}

    def evaluate(self, save_detailed_results: bool = True) -> Dict:
        self.load_data()
        
        # 1. Group data by image for risk calculation
        img_gt_map = {} # {image_name: {q_id: data}}
        for gt in self.ground_truth:
            img_gt_map.setdefault(gt['image_name'], {})[gt['question_id']] = gt
            
        img_pred_map = {}
        for pred in self.predictions:
            img_pred_map.setdefault(pred['image_name'], {})[pred['question_id']] = pred

        # 2. Match pairs for binary metrics
        matched_pairs = []
        for img_name, preds in img_pred_map.items():
            if img_name in img_gt_map:
                for q_id, pred_data in preds.items():
                    if q_id in img_gt_map[img_name]:
                        matched_pairs.append({
                            'image_name': img_name,
                            'question_id': q_id,
                            'pred_answer': pred_data.get('answer'),
                            'gt_answer': img_gt_map[img_name][q_id].get('answer'),
                            # Level from the question hierarchy (GT and older model outputs may mislabel Level 3 as 2)
                            'level': self.prompt_manager.get_level(q_id),
                            'ground_truth': img_gt_map[img_name][q_id]
                        })

        if not matched_pairs: return None

        # 3. Calculate Risk Errors
        risk_errors = []
        for img_name in img_gt_map:
            if img_name in img_pred_map:
                gt_risk = self.calculate_risk_score(img_gt_map[img_name])
                pred_risk = self.calculate_risk_score(img_pred_map[img_name])
                risk_errors.append(abs(gt_risk - pred_risk))
        
        avg_risk_error = sum(risk_errors) / len(risk_errors) if risk_errors else 0

        # 4. Standard Metrics grouping (Level, Topic, etc.)
        y_true_all = [pair['gt_answer'] for pair in matched_pairs]
        y_pred_all = [pair['pred_answer'] for pair in matched_pairs]
        overall_metrics = self.calculate_binary_metrics(y_true_all, y_pred_all)
        overall_metrics['Risk_MAE'] = avg_risk_error # Add new metric here
    
        # 5. Grouping Logic ---
        question_groups = {}
        level_groups = {}
        topic_groups = {} 
        for pair in matched_pairs:
            # Level Grouping
            lvl = pair.get('level', 1)
            level_groups.setdefault(lvl, {'y_true': [], 'y_pred': []})
            level_groups[lvl]['y_true'].append(pair['gt_answer'])
            level_groups[lvl]['y_pred'].append(pair['pred_answer'])

            # Question ID Grouping
            q_id = pair['question_id']
            question_groups.setdefault(q_id, {'y_true': [], 'y_pred': [], 'label': pair.get('short_label', q_id)})
            question_groups[q_id]['y_true'].append(pair['gt_answer'])
            question_groups[q_id]['y_pred'].append(pair['pred_answer'])
            
            # Topic Grouping: the Level-1 question (hazard category) the question descends from
            topic = self.prompt_manager.get_level1_ancestor(pair['question_id'])

            topic_groups.setdefault(topic, {'y_true': [], 'y_pred': []})
            topic_groups[topic]['y_true'].append(pair['gt_answer'])
            topic_groups[topic]['y_pred'].append(pair['pred_answer'])

        # --- Calculate Metrics ---
        metrics_by_level = {f"Level_{k}": {**self.calculate_binary_metrics(v['y_true'], v['y_pred']), 'sample_count': len(v['y_true'])} 
                            for k, v in level_groups.items()}
        
        metrics_by_question = {k: {**self.calculate_binary_metrics(v['y_true'], v['y_pred']), 'short_label': v['label'], 'sample_count': len(v['y_true'])} 
                               for k, v in question_groups.items()}
        
        metrics_by_topic = {k: {**self.calculate_binary_metrics(v['y_true'], v['y_pred']), 'sample_count': len(v['y_true'])} 
                            for k, v in topic_groups.items()}

        results = {
            'overall_metrics': overall_metrics,
            'risk_statistics': {
                'total_images_evaluated': len(risk_errors),
                'mean_absolute_error': avg_risk_error
            }
        }
        results = {
            'model_info': {'predictions_file': self.predictions_path, 'evaluation_timestamp': datetime.now().isoformat()},
            'risk_statistics': {
                'total_images_evaluated': len(risk_errors),
                'mean_absolute_error': avg_risk_error
            },
            'overall_metrics': overall_metrics,
            'metrics_by_level': metrics_by_level,
            'metrics_by_question': metrics_by_question,
            'metrics_by_topic': metrics_by_topic # Added to results
        }
        
        if save_detailed_results:
            results['detailed_matches'] = matched_pairs[:100]
        
        return results
    
    def save_results(self, results: Dict, output_path: str):
        """
        Save evaluation results to JSON file.
        
        Args:
            results: Dictionary containing evaluation results
            output_path: Path to save the results
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
    
    
    def print_summary(self, results: Dict):
        """
        Print a formatted summary of the evaluation results including topic analysis.
        
        Args:
            results: Dictionary containing evaluation results
        """
        print("\n" + "="*70)
        print("EVALUATION RESULTS SUMMARY")
        print("="*70)
        
        # Overall metrics
        overall = results['overall_metrics']
        print("\nOVERALL METRICS:")
        print(f"  Accuracy:    {overall['Accuracy']:.3f}")
        print(f"  F1 Score:    {overall['F1']:.3f}")
        print(f"  Recall:      {overall['Recall']:.3f}")
        print(f"  Specificity: {overall['Specificity']:.3f}")
        print(f"  Precision:   {overall['Precision']:.3f}")
        print(f"  Risk MAE (Error): {overall.get('Risk_MAE', 0.0):.4f}")
        
        print(f"\n  Confusion Matrix:")
        print(f"    TP: {overall['TP']:4d}  FP: {overall['FP']:4d}")
        print(f"    FN: {overall['FN']:4d}  TN: {overall['TN']:4d}")
        
        # Metrics by level
        if 'metrics_by_level' in results:
            print("\n" + "-"*70)
            print("METRICS BY LEVEL:")
            for level, metrics in sorted(results['metrics_by_level'].items()):
                print(f"\n  {level} (n={metrics['sample_count']}):")
                print(f"    Accuracy: {metrics['Accuracy']:.3f}  F1: {metrics['F1']:.3f}")
                print(f"    Recall: {metrics['Recall']:.3f}  Specificity: {metrics['Specificity']:.3f}")

        # Metrics by topic (The new section)
        if 'metrics_by_topic' in results:
            print("\n" + "-"*70)
            print("METRICS BY TOPIC (sorted by F1):")
            sorted_topics = sorted(
                results['metrics_by_topic'].items(),
                key=lambda x: x[1]['F1'],
                reverse=True
            )
            for topic, metrics in sorted_topics:
                print(f"\n  Topic: {topic} (n={metrics['sample_count']})")
                print(f"    F1: {metrics['F1']:.3f} | Acc: {metrics['Accuracy']:.3f} | Rec: {metrics['Recall']:.3f}")
        
        # Top/Bottom performing questions
        if 'metrics_by_question' in results:
            print("\n" + "-"*70)
            print("TOP PERFORMING QUESTIONS (by F1 score):")
            sorted_questions = sorted(
                results['metrics_by_question'].items(),
                key=lambda x: x[1]['F1'],
                reverse=True
            )
            for q_id, metrics in sorted_questions[:5]:
                label = metrics.get('short_label', q_id)
                print(f"  {label}: F1={metrics['F1']:.3f}, Acc={metrics['Accuracy']:.3f} (n={metrics['sample_count']})")
        
            print("\nBOTTOM PERFORMING QUESTIONS (by F1 score):")
            for q_id, metrics in sorted_questions[-5:]:
                label = metrics.get('short_label', q_id)
                print(f"  {label}: F1={metrics['F1']:.3f}, Acc={metrics['Accuracy']:.3f} (n={metrics['sample_count']})")
        
        print("\n" + "="*70)


def evaluate_model_performance(predictions_path: str, ground_truth_path: str):
    """
    Convenience function to evaluate a model and save results.
    
    Args:
        predictions_path: Path to predictions JSONL file
        ground_truth_path: Path to ground truth JSONL file  
    """
    # Initialize evaluator
    evaluator = MetricsEvaluator(predictions_path, ground_truth_path)
    
    # Perform evaluation
    results = evaluator.evaluate(save_detailed_results=True)
    
    if results:
        # Calculate counts for the report
        pred_answers = [evaluator.normalize_yes_no(p.get('answer')) for p in evaluator.predictions]
        gt_answers = [evaluator.normalize_yes_no(g.get('answer')) for g in evaluator.ground_truth]

        def get_stats(answers):
            total = len(answers)
            yes_c = answers.count('yes')
            no_c = answers.count('no')
            others = total - (yes_c + no_c)
            return yes_c, no_c, others, total

        p_yes, p_no, p_others, p_total = get_stats(pred_answers)
        g_yes, g_no, g_others, g_total = get_stats(gt_answers)

        # Write simple text report
        report_path = os.path.join(Path(predictions_path).parent, "answer_characteristics.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"ANSWER CHARACTERISTICS REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*30}\n\n")
            f.write(f"PREDICTIONS (Total: {p_total})\n")
            f.write(f"- Yes: {p_yes} ({ (p_yes/p_total*100) if p_total > 0 else 0:.1f}%)\n")
            f.write(f"- No:  {p_no} ({ (p_no/p_total*100) if p_total > 0 else 0:.1f}%)\n")
            f.write(f"- Other/Invalid: {p_others}\n\n")
            f.write(f"GROUND TRUTH (Total: {g_total})\n")
            f.write(f"- Yes: {g_yes} ({ (g_yes/g_total*100) if g_total > 0 else 0:.1f}%)\n")
            f.write(f"- No:  {g_no} ({ (g_no/g_total*100) if g_total > 0 else 0:.1f}%)\n")
            f.write(f"- Other/Invalid: {g_others}\n")
        
        print(f"Characteristics report saved to: {report_path}")

        # Print summary
        evaluator.print_summary(results)
        
        # Save results
        output_path = os.path.join(Path(predictions_path).parent, f"metrics.json")
        evaluator.save_results(results, output_path)
        
        # Also save a simplified CSV for easy comparison
        csv_path = os.path.join(Path(predictions_path).parent, f"metrics_summary.csv")
        save_summary_csv(results, csv_path)
        
        return results
    else:
        print("Evaluation failed - no matched pairs found")
        return None


def save_summary_csv(results: Dict, csv_path: str):
    """Save a simplified CSV summary including Topics."""
    summary_data = []
    
    # helper to append rows
    def add_to_summary(category, sub_name, metrics):
        summary_data.append({
            'Category': category,
            'Subcategory': sub_name,
            'Samples': metrics.get('sample_count', metrics.get('Total', 0)),
            'Accuracy': f"{metrics['Accuracy']:.3f}",
            'F1': f"{metrics['F1']:.3f}",
            'Recall': f"{metrics['Recall']:.3f}",
            'Specificity': f"{metrics['Specificity']:.3f}",
            'Precision': f"{metrics['Precision']:.3f}",
            'Risk_MAE': f"{metrics.get('Risk_MAE', 0.0):.4f}"
        })

    # Add Overall
    add_to_summary('Overall', 'All', results['overall_metrics'])
    
    # Add Levels
    for name, m in sorted(results.get('metrics_by_level', {}).items()):
        add_to_summary('Level', name, m)
        
    # Add Topics
    for name, m in sorted(results.get('metrics_by_topic', {}).items()):
        add_to_summary('Topic', name, m)
    
    pd.DataFrame(summary_data).to_csv(csv_path, index=False)


if __name__ == "__main__":
    # Re-evaluates existing model answers against the GT (no model is run).
    #   python core/eval.py                                  -> all sequences, all models
    #   python core/eval.py --continent Asia --city Tokio1 --models qwen-vl
    parser = argparse.ArgumentParser(description="Evaluate model answers against the ground truth")
    parser.add_argument("--continent", help="Only this continent (default: all)")
    parser.add_argument("--city", help="Only this sequence folder (default: all)")
    parser.add_argument("--models", nargs="+", default=MODELS, help=f"Models to evaluate (default: {' '.join(MODELS)})")
    args = parser.parse_args()

    for continent, city in list_sequences(args.continent, args.city):
        for model in args.models:
            seq_dir = sequence_dir(continent, city)
            predictions_path = seq_dir / "results" / model / "answers.jsonl"
            if not predictions_path.exists():
                print(f"Skipping {continent}/{city} [{model}]: no answers.jsonl")
                continue
            print(f"\n### {continent}/{city} [{model}]")
            evaluate_model_performance(str(predictions_path), str(seq_dir / "ground_truth_labels.jsonl"))
