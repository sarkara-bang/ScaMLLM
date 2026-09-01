#!/usr/bin/env python3
"""
Universal Regression Metrics Calculator for All Baselines
Calculates MAE, STD, RMSE, R² for property prediction tasks
"""
import json
import numpy as np
import argparse
import re
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski


def calculate_properties(smiles):
    """Calculate molecular properties"""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None

    return {
        'MW': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'HBD': Lipinski.NumHDonors(mol),
        'HBA': Lipinski.NumHAcceptors(mol)
    }


def parse_target_properties(input_text):
    """Parse target properties from input text"""
    properties = {}

    mw_match = re.search(r'molecular weight (?:near|around) (\d+(?:\.\d+)?)', input_text, re.IGNORECASE)
    if mw_match:
        properties['MW'] = float(mw_match.group(1))

    logp_match = re.search(r'lipophilicity (?:near|around) ([-\d]+(?:\.\d+)?)', input_text, re.IGNORECASE)
    if logp_match:
        properties['LogP'] = float(logp_match.group(1))

    hbd_match = re.search(r'with (\d+) HBD groups?', input_text, re.IGNORECASE)
    if hbd_match:
        properties['HBD'] = int(hbd_match.group(1))

    hba_match = re.search(r'with (\d+) HBA groups?', input_text, re.IGNORECASE)
    if hba_match:
        properties['HBA'] = int(hba_match.group(1))

    return properties


def calculate_regression_metrics(results_path, test_data_path, output_path):
    """
    Calculate regression metrics for property prediction

    Args:
        results_path: Path to model results JSON file
        test_data_path: Path to test data JSON file
        output_path: Path to save results with regression metrics
    """
    print(f"Loading results from {results_path}...")
    with open(results_path, 'r') as f:
        data = json.load(f)

    print(f"Loading test data from {test_data_path}...")
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)

    sample_inputs = {}
    for i, sample in enumerate(test_data['samples']):
        sample_inputs[i] = sample['input']

    property_data = {
        'MW': {'targets': [], 'predictions': []},
        'LogP': {'targets': [], 'predictions': []},
        'HBD': {'targets': [], 'predictions': []},
        'HBA': {'targets': [], 'predictions': []}
    }

    valid_samples = 0

    detailed_results = data.get('detailed_results', [])

    for sample in detailed_results:
        # Get generated SMILES (handle different field names)
        gen_smiles = sample.get('generated_smiles') or sample.get('smiles') or sample.get('canonical_smiles')

        if not gen_smiles:
            continue

        # Calculate actual properties
        props = calculate_properties(gen_smiles)
        if not props:
            continue

        # Get target properties
        sample_id = sample.get('sample_id')
        if sample_id is None or sample_id not in sample_inputs:
            continue

        target_props = parse_target_properties(sample_inputs[sample_id])
        if not target_props:
            continue

        valid_samples += 1

        # Collect data for each property
        for prop in ['MW', 'LogP', 'HBD', 'HBA']:
            if prop in target_props:
                property_data[prop]['targets'].append(target_props[prop])
                property_data[prop]['predictions'].append(props[prop])

    print(f"Valid samples for regression: {valid_samples}")

    # Calculate metrics for each property
    regression_metrics = {}
    for prop, values in property_data.items():
        if len(values['targets']) > 0:
            targets = np.array(values['targets'])
            predictions = np.array(values['predictions'])

            mae = mean_absolute_error(targets, predictions)
            rmse = np.sqrt(mean_squared_error(targets, predictions))
            r2 = r2_score(targets, predictions)
            std = np.std(predictions - targets)

            regression_metrics[prop] = {
                'count': len(targets),
                'MAE': round(mae, 3),
                'RMSE': round(rmse, 3),
                'STD': round(std, 3),
                'R2': round(r2, 4),
                'mean_target': round(np.mean(targets), 2),
                'mean_pred': round(np.mean(predictions), 2)
            }

            print(f"\n{prop}:")
            print(f"  Count: {len(targets)}")
            print(f"  MAE:   {mae:.3f}")
            print(f"  STD:   {std:.3f}")
            print(f"  RMSE:  {rmse:.3f}")
            print(f"  R²:    {r2:.4f}")

    # Update results with regression metrics
    data['property_regression_metrics'] = regression_metrics

    # Save updated results
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\n Results with regression metrics saved to {output_path}")

    return regression_metrics


def main():
    parser = argparse.ArgumentParser(description='Calculate regression metrics for baseline models')
    parser.add_argument('--results', required=True, help='Path to model results JSON file')
    parser.add_argument('--test-data', required=True, help='Path to test data JSON file')
    parser.add_argument('--output', required=True, help='Path to save results with regression metrics')

    args = parser.parse_args()

    print("="*70)
    print("Universal Regression Metrics Calculator")
    print("="*70)

    metrics = calculate_regression_metrics(args.results, args.test_data, args.output)

    print("\n" + "="*70)
    print("Property Regression Metrics Summary")
    print("="*70)
    for prop, values in metrics.items():
        print(f"\n{prop}:")
        for key, val in values.items():
            print(f"  {key}: {val}")


if __name__ == "__main__":
    main()
