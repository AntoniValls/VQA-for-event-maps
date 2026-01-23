"""
Metrics evaluation module for VQA Pedestrian Navigation System
Calculates F1, Accuracy, Specificity, and Recall for binary classification tasks
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from datetime import datetime


class MetricsEvaluator:
    """Evaluates VQA model predictions against ground truth labels."""
    
    def __init__(self, predictions_path: str, ground_truth_path: str):
        """
        Initialize the evaluator with paths to predictions and ground truth.
        
        Args:
            predictions_path: Path to the predictions JSONL file
            ground_truth_path: Path to the ground truth JSONL file
        """
        self.predictions_path = predictions_path
        self.ground_truth_path = ground_truth_path
        self.predictions = []
        self.ground_truth = []
        
    def load_data(self) -> Tuple[List[Dict], List[Dict]]:
        """Load predictions and ground truth from JSONL files, handling missing newlines."""
        
        def parse_mixed_json(path):
            data = []
            with open(path, 'r', encoding='utf-8') as f:
                # Read the whole content to handle cases where newlines are missing
                content = f.read()
                # This regex splits by newlines OR identifies the boundary between }{ 
                # It replaces }{ with }\n{ so we can use splitlines()
                normalized_content = content.replace('}{', '}\n{')
                
                for line in normalized_content.splitlines():
                    if line.strip():
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            # This will help you identify if there's actual corruption 
                            # (like the terminal prompt (.venv) we saw earlier)
                            print(f"Skipping invalid JSON entry in {path}: {e}")
            return data

        # Load predictions
        self.predictions = parse_mixed_json(self.predictions_path)
        
        # Load ground truth
        self.ground_truth = parse_mixed_json(self.ground_truth_path)
        
        print(f"Loaded {len(self.predictions)} predictions")
        print(f"Loaded {len(self.ground_truth)} ground truth labels")
        
        return self.predictions, self.ground_truth
    
    def match_predictions_to_ground_truth(self) -> Dict:
        """
        Match predictions to ground truth based on image and question ID.
        If a ground truth pair appears multiple times, the last one is taken as true (this could happen because of a bug!)
        
        Returns:
            Dictionary with matched pairs and statistics
        """
        # Create lookup dictionary for ground truth
        gt_lookup = {}
        duplicates_count = 0

        for gt in self.ground_truth:
            key = (gt.get('image_name'), gt.get('question_id'))
            if key in gt_lookup:
                duplicates_count += 1
            gt_lookup[key] = gt
            
        if duplicates_count > 0:
            print(f"Found {duplicates_count} duplicate GT entries. Using the most recent (last) entries.")
        
        # Match predictions to ground truth
        matched_pairs = []
        unmatched_predictions = []
        
        # Similarly for predictions: if a model answered twice, we take the last answer
        pred_lookup = {}
        for pred in self.predictions:
            key = (pred.get('image_name'), pred.get('question_id'))
            pred_lookup[key] = pred

        # Now iterate through the de-duplicated predictions to match with GT
        for key, pred in pred_lookup.items():
            if key in gt_lookup:
                matched_pairs.append({
                    'prediction': pred,
                    'ground_truth': gt_lookup[key],
                    'image_name': pred.get('image_name'),
                    'question_id': pred.get('question_id'),
                    'pred_answer': pred.get('answer'),
                    'gt_answer': gt_lookup[key].get('answer'),
                    'level': pred.get('level'),
                    'short_label': pred.get('short_label', gt_lookup[key].get('short_label'))
                })
            else:
                unmatched_predictions.append(pred)
        
        return {
            'matched_pairs': matched_pairs,
            'unmatched_predictions': unmatched_predictions,
            'total_predictions': len(self.predictions),
            'total_ground_truth': len(self.ground_truth),
            'matched_count': len(matched_pairs),
            'unmatched_count': len(unmatched_predictions)
        }
    
    def normalize_yes_no(self, answer: str) -> str:
        if not isinstance(answer, str):
            return answer

        a = answer.strip().lower()

        if a.startswith("yes"):
            return "yes"
        if a.startswith("no"):
            return "no"

        return a  # fallback for unexpected values
    
    def calculate_binary_metrics(self, y_true: List[str], y_pred: List[str]) -> Dict:
        """
        Calculate binary classification metrics.
        
        Args:
            y_true: List of ground truth labels ('yes' or 'no')
            y_pred: List of predicted labels ('yes' or 'no')
        
        Returns:
            Dictionary containing TP, TN, FP, FN, Accuracy, Precision, Recall, Specificity, F1
        """
        # Convert to binary (yes=1, no=0)
        y_true_binary = [1 if label.lower() == 'yes' else 0 for label in y_true]
        y_pred_binary = [1 if label.lower() == 'yes' else 0 for label in y_pred]
        
        # Calculate confusion matrix components
        TP = sum(1 for t, p in zip(y_true_binary, y_pred_binary) if t == 1 and p == 1)
        TN = sum(1 for t, p in zip(y_true_binary, y_pred_binary) if t == 0 and p == 0)
        FP = sum(1 for t, p in zip(y_true_binary, y_pred_binary) if t == 0 and p == 1)
        FN = sum(1 for t, p in zip(y_true_binary, y_pred_binary) if t == 1 and p == 0)
        
        # Calculate metrics
        total = TP + TN + FP + FN
        
        # Accuracy: (TP + TN) / Total
        accuracy = (TP + TN) / total if total > 0 else 0
        
        # Precision: TP / (TP + FP)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        
        # Recall (Sensitivity): TP / (TP + FN)
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        
        # Specificity: TN / (TN + FP)
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        
        # F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'TP': TP,
            'TN': TN,
            'FP': FP,
            'FN': FN,
            'Total': total,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'Specificity': specificity,
            'F1': f1
        }
    
    def evaluate(self, save_detailed_results: bool = True) -> Dict:
        """
        Main evaluation function that calculates all metrics.
        
        Args:
            save_detailed_results: Whether to save detailed per-question results
        
        Returns:
            Dictionary containing all evaluation results
        """
        # Load data
        self.load_data()
        
        # Match predictions to ground truth
        matching_results = self.match_predictions_to_ground_truth()
        matched_pairs = matching_results['matched_pairs']
        
        if not matched_pairs:
            print("No matched pairs found between predictions and ground truth!")
            return None
        
        print(f"\nMatched {len(matched_pairs)} prediction-ground truth pairs")
        print(f"Unmatched predictions: {matching_results['unmatched_count']}")
        
        # Extract + normalize labels for overall metrics
        y_true_all = [self.normalize_yes_no(pair['gt_answer']) for pair in matched_pairs]
        y_pred_all = [self.normalize_yes_no(pair['pred_answer']) for pair in matched_pairs]
        
        # Calculate overall metrics
        overall_metrics = self.calculate_binary_metrics(y_true_all, y_pred_all)
        
        # Calculate metrics by question type
        metrics_by_question = {}
        question_groups = {}
        
        # Group by question_id
        for pair in matched_pairs:
            q_id = pair['question_id']
            if q_id not in question_groups:
                question_groups[q_id] = {'y_true': [], 'y_pred': [], 'short_label': pair.get('short_label', q_id)}
            question_groups[q_id]['y_true'].append(pair['gt_answer'])
            question_groups[q_id]['y_pred'].append(pair['pred_answer'])
        
        # Calculate metrics for each question type
        for q_id, data in question_groups.items():
            metrics_by_question[q_id] = {
                'short_label': data['short_label'],
                'sample_count': len(data['y_true']),
                **self.calculate_binary_metrics(data['y_true'], data['y_pred'])
            }
        
        # Calculate metrics by level
        metrics_by_level = {}
        level_groups = {}
        
        # Group by level
        for pair in matched_pairs:
            level = pair.get('level', 1)
            if level not in level_groups:
                level_groups[level] = {'y_true': [], 'y_pred': []}
            level_groups[level]['y_true'].append(pair['gt_answer'])
            level_groups[level]['y_pred'].append(pair['pred_answer'])
        
        # Calculate metrics for each level
        for level, data in level_groups.items():
            metrics_by_level[f'Level_{level}'] = {
                'sample_count': len(data['y_true']),
                **self.calculate_binary_metrics(data['y_true'], data['y_pred'])
            }
        
        # Compile results
        results = {
            'model_info': {
                'predictions_file': self.predictions_path,
                'ground_truth_file': self.ground_truth_path,
                'evaluation_timestamp': datetime.now().isoformat()
            },
            'data_statistics': matching_results,
            'overall_metrics': overall_metrics,
            'metrics_by_level': metrics_by_level,
            'metrics_by_question': metrics_by_question
        }
        
        # Save detailed results if requested
        if save_detailed_results:
            results['detailed_matches'] = matched_pairs[:100]  # Save first 100 for inspection
        
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
        Print a formatted summary of the evaluation results.
        
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
        
        # Top performing questions
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
    """Save a simplified CSV summary of the metrics."""
    summary_data = []
    
    # Overall metrics
    overall = results['overall_metrics']
    summary_data.append({
        'Category': 'Overall',
        'Subcategory': 'All',
        'Samples': overall['Total'],
        'Accuracy': f"{overall['Accuracy']:.3f}",
        'F1': f"{overall['F1']:.3f}",
        'Recall': f"{overall['Recall']:.3f}",
        'Specificity': f"{overall['Specificity']:.3f}",
        'Precision': f"{overall['Precision']:.3f}"
    })
    
    # Metrics by level
    for level, metrics in sorted(results.get('metrics_by_level', {}).items()):
        summary_data.append({
            'Category': 'Level',
            'Subcategory': level,
            'Samples': metrics['sample_count'],
            'Accuracy': f"{metrics['Accuracy']:.3f}",
            'F1': f"{metrics['F1']:.3f}",
            'Recall': f"{metrics['Recall']:.3f}",
            'Specificity': f"{metrics['Specificity']:.3f}",
            'Precision': f"{metrics['Precision']:.3f}"
        })
    
    # Save to CSV
    df = pd.DataFrame(summary_data)
    df.to_csv(csv_path, index=False)
    print(f"Summary CSV saved to: {csv_path}")


if __name__ == "__main__":
    # Example usage
    models = ["llava", "instructblip", "qwen-vl", "vilt"]
    """ continent_city = {
                    #"America": ["BuenosAires", "NewYork", "SanFrancisco", "Ushuaia"],
                    #"Europe": ["London1", "Oslo", "00", "01", "02", "03", "04", "05", "06", "07", "08"],
                    #"Asia": ["Bombai", "Singapore", "Tokio1", "Tokio2"],
                    "Africa": [#"Kampala", "Lusaka",
                         "Marrakesh"]
                      }"""
    
    continent_city = {"America": ["Ushuaia"]}
    for continent, cities in continent_city.items():
        for city in cities:
            for model in models:
                MODEL = model  # Options: vilt, blip2, blip2-large, llava, instructblip
                PROMPT_PRESET = "full_hierarchical"  # Options: level_1_only, crossing, stairs, construction, obstacle, crowding, vehicle, surface, visibility, full_hierarchical
                CONTINENT = continent
                CITY = city  # Use the current city in the list

    predictions_path = f"../data/{CONTINENT}/{CITY}/results/{MODEL}/answers.jsonl"
    ground_truth_path = f"../data/{CONTINENT}/{CITY}/ground_truth_labels.jsonl"
    
    evaluate_model_performance(predictions_path, ground_truth_path)