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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import re

sys.path.append('/root/autodl-tmp/multimodel')
sys.path.append('/root/autodl-tmp/multimodel/experiment')
from baselines.image import MultiModalDrugAssist_ImageMol
from data.preprocessing import chem_utils

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


BASE_DIR = "/root/autodl-tmp/multimodel"
TEST_JSON = f"{BASE_DIR}/data/2000_test.json"
SCAFFOLD_MAPPING = f"{BASE_DIR}/data/scaffold_mapping.json"
IMAGE_DIR = f"{BASE_DIR}/datasets"
BASE_MODEL_PATH = f"{BASE_DIR}/models/DrugAssist-7B"
LORA_MODEL_PATH = f"{BASE_DIR}/models/lora_finetuned"
IMAGEMOL_CHECKPOINT = f"{BASE_DIR}/models/ImageMol/ImageMol.pth.tar"
OUTPUT_JSON = f"{BASE_DIR}/experiment/results/multimodal_full_with_regression.json"

NUM_SAMPLES = 2000

print("Loading model...")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

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

model = PeftModel.from_pretrained(model, LORA_MODEL_PATH)
model.eval()

print("Model loaded successfully")

print("Loading test data...")
with open(TEST_JSON, 'r') as f:
    data_file = json.load(f)

if isinstance(data_file, dict) and 'samples' in data_file:
    all_test_data = data_file['samples']
elif isinstance(data_file, list):
    all_test_data = data_file
else:
    raise ValueError(f"Unexpected data format")

test_data = all_test_data[:NUM_SAMPLES]

with open(SCAFFOLD_MAPPING, 'r') as f:
    reference_structures = json.load(f)

print(f"Testing on: {len(test_data)} samples")

tolerances = {
    'LogP': 0.5, 'MW': 50, 'HBD': 1, 'HBA': 1
}

print("Running inference...")
print("="*80)

structure_match_count = 0
valid_smiles_list = []
total_evaluated = 0
property_pairs = {
    'MW': {'targets': [], 'predictions': []},
    'LogP': {'targets': [], 'predictions': []},
    'HBD': {'targets': [], 'predictions': []},
    'HBA': {'targets': [], 'predictions': []}
}

for idx, item in enumerate(test_data):
    print(f"\rProgress: {idx+1}/{len(test_data)}", end='', flush=True)

    instruction = item.get('instruction', 'Design a molecule based on the scaffold.')
    input_text = item.get('input', '')

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
        for pattern in [r'"([^"]+)"', r'is:\s*"?([A-Za-z0-9@\[\]\(\)=#\-\+\\/\.]+)"?']:
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

    image_path = f"{IMAGE_DIR}/{image_key}"
    try:
        image = Image.open(image_path).convert('RGB')
        pixel_values = model.image_processor(image, return_tensors="pt")['pixel_values'].to(device)
    except:
        continue

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

    smiles = None
    for pattern in [r'"([^"]+)"', r'is\s+"?([A-Za-z0-9@\[\]\(\)=#\-\+\\/\.]+)"?']:
        match = re.search(pattern, response)
        if match:
            candidate = match.group(1)
            mol = Chem.MolFromSmiles(candidate)
            if mol is not None:
                smiles = candidate
                break

    if not smiles:
        continue

    valid_smiles_list.append(smiles)

    props = calculate_properties(smiles)
    if not props:
        continue

    for prop_name in ['MW', 'LogP', 'HBD', 'HBA']:
        if prop_name in target_props and prop_name in props:
            property_pairs[prop_name]['targets'].append(target_props[prop_name])
            property_pairs[prop_name]['predictions'].append(props[prop_name])

    ref_mol = Chem.MolFromSmiles(reference_structure)
    if ref_mol:
        eval_smiles = chem_utils.canonicalize(smiles, scaffold=reference_structure, properties=target_props, verbose=False)
        eval_mol = Chem.MolFromSmiles(eval_smiles)
        if eval_mol is None:
            eval_mol = Chem.MolFromSmiles(smiles)

        structure_match = eval_mol.HasSubstructMatch(ref_mol) if eval_mol else False

        if structure_match:
            structure_match_count += 1

        total_evaluated += 1

print("\n" + "="*80)
print("Calculating metrics...")

total_samples = len(test_data)
total_valid = len(valid_smiles_list)
unique_smiles = list(set(valid_smiles_list))
total_unique = len(unique_smiles)

validity = (total_valid / total_samples * 100) if total_samples > 0 else 0.0
uniqueness = (total_unique / total_valid * 100) if total_valid > 0 else 0.0

fingerprints = []
for smi in unique_smiles:
    mol = Chem.MolFromSmiles(smi)
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        fingerprints.append(fp)

if len(fingerprints) >= 2:
    similarities = []
    for i in range(len(fingerprints)):
        for j in range(i + 1, len(fingerprints)):
            similarities.append(DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j]))
    diversity = 1.0 - np.mean(similarities) if similarities else 0.0
else:
    diversity = 0.0

print(f"\nGeneration Quality:")
print(f"  Validity:    {total_valid}/{total_samples} ({validity:.2f}%)")
print(f"  Uniqueness:  {total_unique}/{total_valid} ({uniqueness:.2f}%)")
print(f"  Diversity:   {diversity:.4f}")

property_metrics = {}
for prop in ['MW', 'LogP', 'HBD', 'HBA']:
    targets = np.array(property_pairs[prop]['targets'])
    predictions = np.array(property_pairs[prop]['predictions'])
    
    if len(targets) > 0:
        mae = mean_absolute_error(targets, predictions)
        rmse = np.sqrt(mean_squared_error(targets, predictions))
        r2 = r2_score(targets, predictions)
        std = np.std(predictions - targets)
        
        property_metrics[prop] = {
            'count': len(targets),
            'MAE': round(mae, 3),
            'RMSE': round(rmse, 3),
            'STD': round(std, 3),
            'R2': round(r2, 4),
            'mean_target': round(np.mean(targets), 2),
            'mean_pred': round(np.mean(predictions), 2)
        }
        
        print(f"\n{prop}:")
        print(f"  Samples: {len(targets)}")
        print(f"  MAE:     {mae:.3f}")
        print(f"  RMSE:    {rmse:.3f}")
        print(f"  STD:     {std:.3f}")
        print(f"  R²:      {r2:.4f}")
    else:
        property_metrics[prop] = {
            'count': 0,
            'MAE': 'N/A',
            'RMSE': 'N/A',
            'STD': 'N/A',
            'R2': 'N/A'
        }

print("\n" + "="*80)
print("Task-specific metrics")
print("="*80)

if total_evaluated > 0:
    structure_rate = (structure_match_count / total_evaluated) * 100

    print(f"\nTask Performance:")
    print(f"  Total samples:           {total_evaluated}")
    print(f"  Scaffold preservation:   {structure_match_count}/{total_evaluated} ({structure_rate:.1f}%)")
else:
    structure_rate = 0.0
    print(f"\nNo valid results.")

output_data = {
    'model': 'MultiModal-ImageMol-LLaMA-7B',
    'architecture': LORA_MODEL_PATH,
    'test_file': TEST_JSON,
    'num_samples_tested': len(test_data),
    'generation_metrics': {
        'total_samples': total_samples,
        'total_valid': total_valid,
        'total_unique': total_unique,
        'validity': round(validity, 2),
        'uniqueness': round(uniqueness, 2),
        'diversity': round(diversity, 4)
    },
    'task_metrics': {
        'total_evaluated': total_evaluated,
        'scaffold_preserved': structure_match_count,
        'scaffold_preservation_rate': round(structure_rate, 2),
    },
    'property_regression_metrics': property_metrics
}

with open(OUTPUT_JSON, 'w') as f:
    json.dump(output_data, f, indent=2)
print(f"\nResults saved to: {OUTPUT_JSON}")

