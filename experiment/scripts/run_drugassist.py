#!/usr/bin/env python3
"""
Run DrugAssist-7B baseline evaluation - 2000 samples
Text-only input 
"""
import sys
sys.path.append('/root/autodl-tmp/multimodel/experiment/baselines')

from drugassist_evaluator import DrugAssistEvaluator

print("="*80)
print("DrugAssist-7B Baseline Evaluation - 2000 samples")
print("="*80)
print()
print("Configuration:")
print("  - Text-only input (no scaffold images/SMILES)")
print("  - Property constraints only (MW, LogP, HBD, HBA, TPSA, QED)")
print("  - Save all generation results")
print()

# 2000 samples full evaluation
evaluator = DrugAssistEvaluator(
    test_data_path="/root/autodl-tmp/multimodel/data/2000_test.json",
    scaffold_mapping_path="/root/autodl-tmp/multimodel/data/scaffold_mapping.json",
    output_path="/root/autodl-tmp/multimodel/experiment/results/drugassist_results.json",
    num_samples=2000,
    device="cuda:0"
)

print("Starting 2000 samples evaluation...")
print("Estimated time: 1-2 hours")
print()

try:
    results = evaluator.run_evaluation()

    print("\n" + "="*80)
    print(" DrugAssist evaluation completed!")
    print("="*80)
    print(f"\n Final results:")
    print(f"  Number of test samples:    {results['num_samples_tested']}")
    print(f"  Valid SMILES:    {results['generation_metrics']['validity']:.1f}%")
    print(f"  Uniqueness:        {results['generation_metrics']['uniqueness']:.1f}%")
    print(f"  Diversity:        {results['generation_metrics']['diversity']:.4f}")
    print(f"\n  Scaffold preservation rate:    {results['task_metrics']['scaffold_preservation_rate']:.1f}%")
    print(f"  Property satisfaction rate:    {results['task_metrics']['property_satisfaction_rate']:.1f}%")
    print(f"  Overall success rate:    {results['task_metrics']['overall_success_rate']:.1f}%")
    print(f"\n Results saved to:")
    print(f"  {evaluator.output_path}")
    print(f"\n Saved for each sample:")
    print(f"  - generated_output (full model output)")
    print(f"  - generated_smiles (extracted SMILES)")
    print(f"  - canonical_smiles (canonicalized SMILES)")
    print(f"  - target_properties vs actual_properties")
    print("="*80)

except Exception as e:
    print(f"\n Evaluation failed: {e}")
    import traceback
    traceback.print_exc()