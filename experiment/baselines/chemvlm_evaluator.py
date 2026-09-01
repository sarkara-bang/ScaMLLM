#!/usr/bin/env python3
"""
ChemVLM Baseline Evaluator
Uses ChemVLM (Chemistry Vision-Language Model) for scaffold-based molecule generation.
ChemVLM is a multimodal model that can process both molecular images and text.
"""
import re
import sys
import os
import json
sys.path.append('/root/autodl-tmp/multimodel/experiment/baselines')

from base_evaluator import BaseEvaluator
from transformers import AutoTokenizer, AutoModel
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


class ChemVLMEvaluator(BaseEvaluator):
    """Evaluator for ChemVLM multimodal baseline"""

    def __init__(self, data_root="/root/autodl-tmp/DrugAssist-main/datasets", **kwargs):
        super().__init__(**kwargs)
        self.model_path = "/root/autodl-tmp/models/ChemVLM-8B"
        self.data_root = data_root
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load ChemVLM model"""
        print(f"Loading ChemVLM from {self.model_path}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

        self.model = AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        ).eval()

        print("ChemVLM model loaded successfully")

    def load_image(self, image_path: str, input_size=448, max_num=6):
        """Load and preprocess image for ChemVLM"""
        full_path = os.path.join(self.data_root, image_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Image not found: {full_path}")

        image = Image.open(full_path).convert('RGB')
        transform = build_transform(input_size=input_size)
        images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(img) for img in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

    def format_prompt(self, instruction: str, input_text: str) -> str:
        """Format prompt for ChemVLM with strong scaffold emphasis"""
        question = (
            f"You are a chemistry expert. Your task is to generate a molecule that MUST contain the scaffold structure shown in the image.\n\n"
            f"{instruction}\n\n"
            f"{input_text}\n\n"
            f"IMPORTANT REQUIREMENTS:\n"
            f"1. The generated molecule MUST preserve the complete scaffold structure from the image\n"
            f"2. You can only add substituents or functional groups to the scaffold\n"
            f"3. Do NOT modify or remove any part of the scaffold core structure\n"
            f"4. Output ONLY the SMILES string of the final molecule, nothing else\n\n"
            f"Generated SMILES:"
        )
        return question

    def clean_atom_map(self, smiles: str) -> str:
        """Remove atom mapping numbers"""
        cleaned = re.sub(r'\[([A-Za-z@+\-]+):\d+\]', r'[\1]', smiles)
        cleaned = re.sub(r':\d+', '', cleaned)
        return cleaned

    def extract_smiles_from_output(self, output: str) -> str:
        """Extract SMILES from ChemVLM output"""
        candidates = []

        # Pattern 1: SMILES: xxx
        m = re.search(r'SMILES:\s*(\S+)', output, re.IGNORECASE)
        if m:
            candidates.append(m.group(1).strip())

        # Pattern 2: <SMILES>xxx</SMILES>
        m = re.search(r'<SMILES>\s*([^<]+?)\s*</SMILES>', output, re.IGNORECASE)
        if m:
            candidates.append(m.group(1).strip())

        # Pattern 3: quoted "xxx"
        m = re.search(r'"([A-Za-z0-9@\[\]\(\)=#\-\+\\/\.\%:]+)"', output)
        if m:
            candidates.append(m.group(1).strip())

        # Pattern 4: Molecule: xxx or Generated: xxx
        m = re.search(r'(?:Molecule|Generated|Output):\s*([A-Za-z0-9@\[\]\(\)=#\-\+\\/\.\%:]+)', output, re.IGNORECASE)
        if m:
            candidates.append(m.group(1).strip())

        # Pattern 5: longest SMILES-like token
        tokens = re.findall(r'[A-Za-z0-9@\[\]\(\)=#\-\+\\/\.\%:]{8,}', output)
        candidates.extend(tokens)

        for raw in candidates:
            if Chem.MolFromSmiles(raw):
                return raw
            cleaned = self.clean_atom_map(raw)
            if Chem.MolFromSmiles(cleaned):
                return cleaned

        return None

    def generate_smiles(self, instruction: str, input_text: str, image_path: str = None) -> str:
        """Generate SMILES using ChemVLM with image input"""
        if not image_path:
            raise ValueError("ChemVLM requires image input")

        # Load and preprocess image
        try:
            pixel_values = self.load_image(image_path, input_size=448, max_num=6)
            pixel_values = pixel_values.to(torch.float16).to(self.model.device)
        except Exception as e:
            raise ValueError(f"Failed to load image {image_path}: {e}")

        # Format question
        question = self.format_prompt(instruction, input_text)

        # Generate using ChemVLM's chat method
        try:
            gen_kwargs = {
                'max_new_tokens': 256,
                'do_sample': True,
                'temperature': 0.7,
                'top_p': 0.9,
            }
            response = self.model.chat(self.tokenizer, pixel_values, question, gen_kwargs)
            return response
        except Exception as e:
            raise RuntimeError(f"Generation failed: {e}")

    def run_evaluation(self):
        """Override to pass image_path to generate_smiles"""
        print("="*80)
        print(f"Running {self.__class__.__name__}")
        print("="*80)

        self.load_model()

        results = []
        all_generated_smiles = []
        all_valid_smiles = []
        ref_smiles_paired = []
        scaffold_match_count = 0

        for idx, item in enumerate(self.test_data):
            print(f"\rProgress: {idx+1}/{len(self.test_data)}", end='', flush=True)

            image_path = item.get('scaffold_image', '')
            if not image_path:
                results.append({'sample_id': idx, 'error': 'No scaffold image'})
                continue

            image_key = item.get('scaffold_image', '')
            reference_scaffold = self.scaffold_mapping.get(image_key)

            if not reference_scaffold:
                output_text = item.get('output', '')
                output_smiles = self.extract_smiles_from_output(output_text)
                if output_smiles:
                    mol = Chem.MolFromSmiles(output_smiles)
                    if mol:
                        try:
                            scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
                            reference_scaffold = Chem.MolToSmiles(scaffold_mol)
                        except:
                            pass

            if not reference_scaffold:
                continue

            if isinstance(reference_scaffold, Chem.Mol):
                reference_scaffold = Chem.MolToSmiles(reference_scaffold)

            gt_smiles = self.extract_smiles_from_output(item.get('output', ''))
            instruction = item.get('instruction', '')
            input_text = item.get('input', '')

            try:
                generated_output = self.generate_smiles(instruction, input_text, image_path=image_path)
                generated_smiles = self.extract_smiles_from_output(generated_output)
            except Exception as e:
                print(f"\nError generating for sample {idx}: {e}")
                results.append({
                    'sample_id': idx,
                    'reference_scaffold': reference_scaffold,
                    'generated_smiles': None,
                    'error': str(e)
                })
                continue

            if not generated_smiles:
                results.append({
                    'sample_id': idx,
                    'reference_scaffold': reference_scaffold,
                    'generated_smiles': None,
                    'error': 'Failed to extract SMILES'
                })
                continue

            all_generated_smiles.append(generated_smiles)

            mol = Chem.MolFromSmiles(generated_smiles)
            if not mol:
                results.append({
                    'sample_id': idx,
                    'reference_scaffold': reference_scaffold,
                    'generated_smiles': generated_smiles,
                    'canonical_smiles': None,
                    'error': 'Invalid SMILES'
                })
                continue

            canonical_smiles = Chem.MolToSmiles(mol)
            all_valid_smiles.append(canonical_smiles)

            if gt_smiles and Chem.MolFromSmiles(gt_smiles):
                ref_smiles_paired.append(Chem.MolToSmiles(Chem.MolFromSmiles(gt_smiles)))
            else:
                ref_smiles_paired.append(None)

            scaffold_match = self.check_scaffold_preservation(canonical_smiles, reference_scaffold)
            if scaffold_match:
                scaffold_match_count += 1

            results.append({
                'sample_id': idx,
                'reference_scaffold': reference_scaffold,
                'generated_smiles': generated_smiles,
                'canonical_smiles': canonical_smiles,
                'scaffold_preserved': scaffold_match,
            })

        print()

        total_attempts = len(self.test_data)
        total_generated = len(all_generated_smiles)
        total_valid = len(all_valid_smiles)
        unique_smiles = list(set(all_valid_smiles))
        total_unique = len(unique_smiles)
        total_evaluated = len(results)

        validity = round(total_valid / total_attempts * 100, 2) if total_attempts > 0 else 0.0
        uniqueness = round(total_unique / total_valid * 100, 2) if total_valid > 0 else 0.0
        diversity = self.calculate_diversity(unique_smiles)
        novelty = self.calculate_novelty(unique_smiles)
        scaffold_rate = round(scaffold_match_count / total_evaluated * 100, 2) if total_evaluated > 0 else 0.0

        paired_gen = []
        paired_ref = []
        for smi, ref in zip(all_valid_smiles, ref_smiles_paired[:len(all_valid_smiles)]):
            if ref is not None:
                paired_gen.append(smi)
                paired_ref.append(ref)
        fp_sims = self.calculate_fingerprint_similarities(paired_gen, paired_ref)

        print(f"\n Results:")
        print(f"  Validity:              {total_valid}/{total_attempts} ({validity:.2f}%)")
        print(f"  Uniqueness:            {total_unique}/{total_valid} ({uniqueness:.2f}%)")
        print(f"  Diversity:             {diversity:.4f}")
        print(f"  Novelty:               {novelty:.2f}%  (over {total_unique} unique valid mols)")
        print(f"  Scaffold preservation: {scaffold_match_count}/{total_evaluated} ({scaffold_rate:.2f}%)")
        print(f"  Morgan similarity:     {fp_sims['morgan_similarity']:.4f}")
        print(f"  MACCS similarity:      {fp_sims['maccs_similarity']:.4f}")
        print(f"  RDKit similarity:      {fp_sims['rdkit_similarity']:.4f}")

        output_data = {
            'model': self.__class__.__name__,
            'num_samples_tested': len(self.test_data),
            'metrics': {
                'validity': validity,
                'uniqueness': uniqueness,
                'diversity': diversity,
                'novelty': novelty,
                'scaffold_preservation_rate': scaffold_rate,
                'morgan_similarity': fp_sims['morgan_similarity'],
                'maccs_similarity': fp_sims['maccs_similarity'],
                'rdkit_similarity': fp_sims['rdkit_similarity'],
            },
            'counts': {
                'total_attempts': total_attempts,
                'total_generated': total_generated,
                'total_valid': total_valid,
                'total_unique': total_unique,
                'scaffold_matches': scaffold_match_count,
                'fp_pairs': fp_sims['num_pairs'],
            },
            'detailed_results': results
        }

        with open(self.output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n Results saved to: {self.output_path}")
        print("="*80)

        return output_data


if __name__ == "__main__":
    evaluator = ChemVLMEvaluator(
        test_data_path="/root/autodl-tmp/multimodel/data/2000_test.json",
        scaffold_mapping_path="/root/autodl-tmp/multimodel/data/scaffold_mapping.json",
        output_path="/root/autodl-tmp/multimodel/experiment/results/chemvlm_results.json",
        num_samples=2000,
        device="cuda:0"
    )

    results = evaluator.run_evaluation()

