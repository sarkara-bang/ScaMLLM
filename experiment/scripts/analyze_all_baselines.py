#!/usr/bin/env python3
"""
Comprehensive Analysis of All Baselines
"""
import json
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd

PROPERTIES = ['MW', 'LogP', 'HBD', 'HBA']

def load_results(filepath):
    """Load evaluation results"""
    with open(filepath, 'r') as f:
        return json.load(f)

def calculate_property_metrics(results_data):
    """Calculate metrics for each property"""
    property_metrics = {}

    detailed_results = results_data.get('detailed_results', [])

    for prop in PROPERTIES:
        targets = []
        predictions = []

        for sample in detailed_results:
            target_props = sample.get('target_properties', {})
            actual_props = sample.get('actual_properties', {})

            if prop in target_props and prop in actual_props:
                targets.append(target_props[prop])
                predictions.append(actual_props[prop])

        if len(targets) > 0:
            targets = np.array(targets)
            predictions = np.array(predictions)

            # Calculate metrics
            mae = mean_absolute_error(targets, predictions)
            rmse = np.sqrt(mean_squared_error(targets, predictions))
            std = np.std(predictions - targets)

            # R² score (can be negative for poor models)
            try:
                r2 = r2_score(targets, predictions)
            except:
                r2 = -999

            # MAPE: avoid division by zero
            mask = targets != 0
            if mask.sum() > 0:
                mape = np.mean(np.abs((targets[mask] - predictions[mask]) / targets[mask])) * 100
            else:
                mape = 999

            property_metrics[prop] = {
                'count': len(targets),
                'MAE': round(mae, 3),
                'RMSE': round(rmse, 3),
                'std': round(std, 3),
                'R2': round(r2, 4),
                'MAPE': round(mape, 2) if mape != 999 else 'N/A',
                'mean_target': round(np.mean(targets), 2),
                'mean_pred': round(np.mean(predictions), 2)
            }
        else:
            property_metrics[prop] = {
                'count': 0,
                'MAE': 'N/A',
                'RMSE': 'N/A',
                'std': 'N/A',
                'R2': 'N/A',
                'MAPE': 'N/A',
                'mean_target': 'N/A',
                'mean_pred': 'N/A'
            }

    return property_metrics

def generate_comparison_table(all_metrics):
    """Generate comparison table"""
    print("\n" + "="*120)
    print("Baseline Models Comparison")
    print("="*120)

    # 1. Generation Quality Metrics
    print("\n[1] Generation Quality")
    print("-"*120)
    print(f"{'Model':<20} {'Validity':<18} {'Uniqueness':<20} {'Diversity':<20}")
    print("-"*120)

    for model_name, metrics in all_metrics.items():
        gen_metrics = metrics['generation_metrics']
        validity = gen_metrics.get('validity', 'N/A')
        uniqueness = gen_metrics.get('uniqueness', 'N/A')
        diversity = gen_metrics.get('diversity', 'N/A')

        print(f"{model_name:<20} {validity:>15}%  {uniqueness:>17}%  {diversity:>20}")

    # 2. Task-Specific Metrics
    print("\n[2] Task-Specific Metrics")
    print("-"*120)
    print(f"{'Model':<20} {'Scaffold Preservation':<25} {'Property Satisfaction':<25} {'Overall Success':<20}")
    print("-"*120)

    for model_name, metrics in all_metrics.items():
        task_metrics = metrics['task_metrics']
        scaffold = task_metrics.get('scaffold_preservation_rate', 'N/A')
        property_sat = task_metrics.get('property_satisfaction_rate', 'N/A')
        overall = task_metrics.get('overall_success_rate', 'N/A')

        print(f"{model_name:<20} {scaffold:>20}%  {property_sat:>22}%  {overall:>20}%")

    # 3. Property-wise Metrics
    for prop in PROPERTIES:
        print(f"\n[3.{PROPERTIES.index(prop)+1}] Property: {prop}")
        print("-"*120)
        print(f"{'Model':<20} {'Count':<10} {'MAE':<12} {'RMSE':<12} {'Std':<12} {'R²':<12} {'MAPE (%)':<12}")
        print("-"*120)

        for model_name, metrics in all_metrics.items():
            prop_metrics = metrics['property_metrics'].get(prop, {})
            count = prop_metrics.get('count', 0)
            mae = prop_metrics.get('MAE', 'N/A')
            rmse = prop_metrics.get('RMSE', 'N/A')
            std = prop_metrics.get('std', 'N/A')
            r2 = prop_metrics.get('R2', 'N/A')
            mape = prop_metrics.get('MAPE', 'N/A')

            print(f"{model_name:<20} {count:<10} {str(mae):<12} {str(rmse):<12} {str(std):<12} {str(r2):<12} {str(mape):<12}")

    print("\n" + "="*120)

def save_summary(all_metrics, output_path):
    """Save summary to JSON"""
    with open(output_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nResults saved to: {output_path}")

# Main execution
if __name__ == "__main__":
    print("="*120)
    print("Baseline Models Analysis")
    print("="*120)

    # Load all results
    models = {
        'MolT5': '/root/autodl-tmp/multimodel/experiment/results/molt5_results.json',
        'DrugAssist-7B': '/root/autodl-tmp/multimodel/experiment/results/drugassist_results.json',
        'LLaMA2-7B-chat': '/root/autodl-tmp/multimodel/experiment/results/llama2_results.json'
    }

    all_metrics = {}

    for model_name, filepath in models.items():
        print(f"\nProcessing {model_name}...")
        results = load_results(filepath)

        # Extract metrics
        all_metrics[model_name] = {
            'generation_metrics': results.get('generation_metrics', {}),
            'task_metrics': results.get('task_metrics', {}),
            'property_metrics': calculate_property_metrics(results)
        }

    # Generate comparison table
    generate_comparison_table(all_metrics)

    # Save summary
    output_path = '/root/autodl-tmp/multimodel/experiment/results/all_baselines_summary.json'
    save_summary(all_metrics, output_path)

    print("\nAnalysis completed")
    print("="*120)
