#!/usr/bin/env python3
"""
Create LogP-focused dataset from original test data
Extracts LogP from output SMILES and creates simplified input format
"""
import json
import re
from rdkit import Chem
from rdkit.Chem import Descriptors

# Paths
INPUT_JSON = "/root/autodl-tmp/multimodel/data/2000_test.json"
OUTPUT_JSON = "/root/autodl-tmp/multimodel/data/logp_test.json"

print("="*80)
print("Creating LogP-focused Dataset")
print("="*80)

# Load original data
print(f"\n[1/3] Loading original dataset: {INPUT_JSON}")
with open(INPUT_JSON, 'r') as f:
    data = json.load(f)

samples = data.get('samples', [])
print(f"Total samples: {len(samples)}")

print(f"\n[2/3] Processing samples and extracting LogP...")
new_samples = []
logp_values = []
failed_count = 0

for idx, sample in enumerate(samples):
    if (idx + 1) % 100 == 0:
        print(f"  Processed: {idx + 1}/{len(samples)}", end='\r')

    output_text = sample.get('output', '')
    scaffold_image = sample.get('scaffold_image', '')

    # Extract SMILES from output using multiple patterns
    smiles = None
    patterns = [
        r'"([^"]+)"',
        r'is:\s*"?([A-Za-z0-9@\[\]\(\)=#\-\+\\/\.]+)"?',
        r'molecule is\s+"?([A-Za-z0-9@\[\]\(\)=#\-\+\\/\.]+)"?'
    ]

    for pattern in patterns:
        match = re.search(pattern, output_text)
        if match:
            candidate_smiles = match.group(1)
            mol = Chem.MolFromSmiles(candidate_smiles)
            if mol is not None:
                smiles = candidate_smiles
                break

    if not smiles:
        failed_count += 1
        continue

    # Calculate LogP
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        failed_count += 1
        continue

    logp = Descriptors.MolLogP(mol)
    logp_values.append(logp)

    # Create new sample with LogP-only input
    new_sample = {
        "instruction": "Design a chemical compound based on the scaffold structure.",
        "input": f"Using the scaffold structure in the image, create a molecule with lipophilicity around {logp:.1f}.",
        "output": output_text,
        "original_smiles": smiles,
        "target_logp": round(logp, 2),
        "scaffold_image": scaffold_image
    }

    new_samples.append(new_sample)

print(f"\n  Successfully processed: {len(new_samples)}")
print(f"  Failed to extract SMILES: {failed_count}")

# Save new dataset
print(f"\n[3/3] Saving new dataset: {OUTPUT_JSON}")
output_data = {
    "metadata": {
        "source": INPUT_JSON,
        "total_samples": len(new_samples),
        "property": "LogP (lipophilicity)",
        "description": "Simplified dataset with only LogP constraint",
        "logp_statistics": {
            "min": round(min(logp_values), 2) if logp_values else 0,
            "max": round(max(logp_values), 2) if logp_values else 0,
            "mean": round(sum(logp_values) / len(logp_values), 2) if logp_values else 0
        }
    },
    "samples": new_samples,
    "reference_logp_values": [round(lp, 2) for lp in logp_values]
}

with open(OUTPUT_JSON, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"\n Dataset created successfully!")
print(f"   Total samples: {len(new_samples)}")
print(f"   LogP range: [{output_data['metadata']['logp_statistics']['min']:.2f}, {output_data['metadata']['logp_statistics']['max']:.2f}]")
print(f"   LogP mean: {output_data['metadata']['logp_statistics']['mean']:.2f}")
print("="*80)
