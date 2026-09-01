#!/usr/bin/env python3
import json
import torch
import numpy as np
from PIL import Image
from transformers import AutoTokenizer
from peft import PeftModel
import sys
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
import re
from collections import Counter
sys.path.append('/root/autodl-tmp/multimodel')
sys.path.append('/root/autodl-tmp/multimodel/experiment')
from baselines.image import MultiModalDrugAssist_ImageMol

def calculate_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        'MW': round(Descriptors.MolWt(mol), 2),
        'LogP': round(Descriptors.MolLogP(mol), 2),
        'HBD': Lipinski.NumHDonors(mol),
        'HBA': Lipinski.NumHAcceptors(mol)
    }

def evaluate_property_match(props, targets, tolerances):
    """Evaluate if properties match target constraints"""
    satisfied = {}
    for key, target in targets.items():
        if key not in props or key not in tolerances:
            continue
        tolerance = tolerances[key]
        if isinstance(tolerance, (int, float)):
            satisfied[key] = abs(props[key] - target) <= tolerance
        else:
            satisfied[key] = True
    return satisfied


# Configuration
BASE_DIR = "/root/autodl-tmp/multimodel"
TEST_JSON = f"{BASE_DIR}/data/2000_test.json"
SCAFFOLD_MAPPING = f"{BASE_DIR}/data/scaffold_mapping.json"
BASE_MODEL_PATH = f"{BASE_DIR}/models/DrugAssist-7B"
LORA_MODEL_PATH = f"{BASE_DIR}/models/lora_finetuned"
IMAGEMOL_CHECKPOINT = f"{BASE_DIR}/models/ImageMol/ImageMol.pth.tar"
OUTPUT_JSON = f"{BASE_DIR}/experiment/results/multimodal_textonly_results.json"

# Evaluation settings
NUM_SAMPLES = 2000  # Evaluate 2000 samples

print("[1/5] Loading model...")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Load multimodal model
model = MultiModalDrugAssist_ImageMol(
    llm_model_name_or_path=BASE_MODEL_PATH,
    imagemol_checkpoint_path=IMAGEMOL_CHECKPOINT,
    device_map="auto"
)

device = next(model.llm.parameters()).device
model.vision_model = model.vision_model.to(device)
model.projection = model.projection.to(device)
if hasattr(model, 'fusion_norm'):
    model.fusion_norm = model.fusion_norm.to(device)

# Load LoRA weights
model = PeftModel.from_pretrained(model, LORA_MODEL_PATH)
model.eval()

print(" Model loaded successfully")

print(f"\n[2/5] Loading test data...")
with open(TEST_JSON, 'r') as f:
    data_file = json.load(f)

if isinstance(data_file, dict) and 'samples' in data_file:
    all_test_data = data_file['samples']
elif isinstance(data_file, list):
    all_test_data = data_file
else:
    raise ValueError(f"Unexpected data format in {TEST_JSON}")

test_data = all_test_data[:NUM_SAMPLES]

with open(SCAFFOLD_MAPPING, 'r') as f:
    reference_structures = json.load(f)

print(f"Testing on: {len(test_data)} samples")

# Property tolerances
tolerances = {
    'LogP': 0.5,
    'MW': 50,
    'HBD': 1,
    'HBA': 1
}
results = []
structure_match_count = 0
property_match_count = 0
overall_success_count = 0

all_generated_smiles = []
all_valid_smiles = []

for idx, item in enumerate(test_data):
    print(f"\rProgress: {idx+1}/{len(test_data)}", end='', flush=True)

    instruction = item.get('instruction', 'Design a molecule based on the scaffold.')
    input_text = item.get('input', '')

    # Parse target properties
    target_props = {}
    logp_match = re.search(r'lipophilicity around ([\d\.]+)', input_text)
    if logp_match:
        target_props['LogP'] = float(logp_match.group(1).rstrip('.'))

    mw_match = re.search(r'molecular weight near ([\d\.]+)', input_text)
    if mw_match:
        target_props['MW'] = float(mw_match.group(1).rstrip('.'))

    hbd_match = re.search(r'(\d+) HBD', input_text)
    if hbd_match:
        target_props['HBD'] = int(hbd_match.group(1))

    hba_match = re.search(r'(\d+) HBA', input_text)
    if hba_match:
        target_props['HBA'] = int(hba_match.group(1))

    image_key = item.get('scaffold_image', '')
    reference_structure = reference_structures.get(image_key)

    if not reference_structure:
        output_text = item.get('output', '')
        output_patterns = [r'"([^"]+)"', r'is:\s*"?([A-Za-z0-9@\[\]\(\)=#\-\+\\/\.]+)"?']
        for pattern in output_patterns:
            match = re.search(pattern, output_text)
            if match:
                output_smiles = match.group(1)
                output_mol = Chem.MolFromSmiles(output_smiles)
                if output_mol:
                    try:
                        reference_structure = MurckoScaffold.GetScaffoldForMol(output_mol)
                        if reference_structure:
                            break
                    except:
                        pass

    if not reference_structure:
        continue

    if isinstance(reference_structure, Chem.Mol):
        reference_structure = Chem.MolToSmiles(reference_structure)
    elif not isinstance(reference_structure, str):
        continue

    pixel_values = torch.zeros(1, 3, 224, 224).to(device)

    prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,  
            max_new_tokens=128,
            num_beams=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = generated_text[len(prompt):].strip() if len(generated_text) > len(prompt) else generated_text

    patterns = [r'"([^"]+)"', r'is\s+"?([A-Za-z0-9@\[\]\(\)=#\-\+\\/\.]+)"?']
    raw_smiles = None
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            smiles = match.group(1)
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                raw_smiles = smiles
                break

    if not raw_smiles:
        continue

    all_generated_smiles.append(raw_smiles)

    try:
        final_mol = Chem.MolFromSmiles(raw_smiles)
        if final_mol is None:
            continue
        final_smiles = Chem.MolToSmiles(final_mol)
    except:
        continue

    all_valid_smiles.append(final_smiles)

    # Calculate properties
    final_props = calculate_properties(final_smiles)
    if not final_props:
        continue

    # Evaluate scaffold preservation
    ref_mol = Chem.MolFromSmiles(reference_structure)
    if ref_mol:
        structure_match = final_mol.HasSubstructMatch(ref_mol)

        if structure_match:
            structure_match_count += 1

        satisfied = evaluate_property_match(final_props, target_props, tolerances)
        all_satisfied = all(satisfied.values()) if satisfied else False

        if all_satisfied:
            property_match_count += 1

        if structure_match and all_satisfied:
            overall_success_count += 1

        results.append({
            'sample_id': idx,
            'reference_scaffold': reference_structure,
            'generated_output': response,
            'generated_smiles': raw_smiles,
            'canonical_smiles': final_smiles,
            'target_properties': target_props,
            'actual_properties': final_props,
            'scaffold_preserved': structure_match,
            'properties_satisfied': satisfied,
            'all_satisfied': all_satisfied
        })


total_generated = len(all_generated_smiles)
total_valid = len(all_valid_smiles)
unique_smiles = list(set(all_valid_smiles))
total_unique = len(unique_smiles)

validity = (total_valid / total_generated * 100) if total_generated > 0 else 0.0
uniqueness = (total_unique / total_valid * 100) if total_valid > 0 else 0.0

# Diversity
from rdkit.Chem import AllChem, DataStructs
fingerprints = []
for smiles in unique_smiles:
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        fingerprints.append(fp)

if len(fingerprints) >= 2:
    similarities = []
    for i in range(len(fingerprints)):
        for j in range(i + 1, len(fingerprints)):
            sim = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
            similarities.append(sim)
    diversity = 1.0 - np.mean(similarities) if similarities else 0.0
else:
    diversity = 0.0

print(f"\n Generation Quality:")
print(f"  Validity:    {total_valid}/{total_generated} ({validity:.2f}%)")
print(f"  Uniqueness:  {total_unique}/{total_valid} ({uniqueness:.2f}%)")
print(f"  Diversity:   {diversity:.4f}")

print(f"\nTask-specific metrics")
print("="*80)

total = len(results)
structure_rate = 0.0
overall_rate = 0.0

if total > 0:
    structure_rate = (structure_match_count / total) * 100
    overall_rate = (overall_success_count / total) * 100

    print(f"\n Task Performance:")
    print(f"  Total samples:           {total}")
    print(f"  Scaffold preservation:   {structure_match_count}/{total} ({structure_rate:.1f}%)")
else:
    print(f"\n  No valid results generated.")

# Save results
output_data = {
    'model': 'MultiModal (Text-Only Mode)',
    'architecture': LORA_MODEL_PATH,
    'test_file': TEST_JSON,
    'num_samples_tested': len(test_data),
    'mode': 'text_only_ablation',
    'description': 'Same model architecture but with zero image input',

    'generation_metrics': {
        'total_generated': total_generated,
        'total_valid': total_valid,
        'total_unique': total_unique,
        'validity': round(validity, 2),
        'uniqueness': round(uniqueness, 2),
        'diversity': round(diversity, 4)
    },

    'task_metrics': {
        'total_evaluated': total,
        'scaffold_preserved': structure_match_count,
        'scaffold_preservation_rate': round(structure_rate, 2),
    },

    'detailed_results': results
}

with open(OUTPUT_JSON, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"\n Results saved to: {OUTPUT_JSON}")
