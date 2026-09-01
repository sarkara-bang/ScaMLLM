#!/usr/bin/env python3
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
sys.path.append('/root/autodl-tmp/multimodel/experiment/baselines')
from molt5_evaluator import MolT5Evaluator
print("="*80)
print("MolT5-caption2smiles-2000samples")
print("="*80)
print()

evaluator = MolT5Evaluator(
    test_data_path="/root/autodl-tmp/multimodel/data/2000_test.json",
    scaffold_mapping_path="/root/autodl-tmp/multimodel/data/scaffold_mapping.json",
    output_path="/root/autodl-tmp/multimodel/experiment/results/molt5_results.json",
    num_samples=2000,
    device="cuda:0"
)

print("\nstarting MolT5 evaluation...")
print()

try:
    results = evaluator.run_evaluation()

    print("\n" + "="*80)
    print(" MolT5 evaluation completed!")
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
    print("="*80)

except Exception as e:
    print(f"\n Evaluation failed: {e}")
    import traceback
    traceback.print_exc()