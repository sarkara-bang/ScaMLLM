#!/usr/bin/env python3
"""
Provides common evaluation logic for all baseline models
"""
import json
import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
import re
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, QED, DataStructs, MACCSkeys
from rdkit.Chem.Scaffolds import MurckoScaffold


class BaseEvaluator(ABC):
    """Base class for all baseline evaluators"""

    def __init__(
        self,
        test_data_path: str,
        scaffold_mapping_path: str,
        output_path: str,
        train_data_path: str = "/root/autodl-tmp/multimodel/data/train.json",
        num_samples: int = 2000,
        device: str = "cuda:0"
    ):
        self.test_data_path = test_data_path
        self.scaffold_mapping_path = scaffold_mapping_path
        self.output_path = output_path
        self.train_data_path = train_data_path
        self.num_samples = num_samples
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load data
        self.test_data = self.load_test_data()
        self.scaffold_mapping = self.load_scaffold_mapping()
        self.train_smiles_set = self.load_train_smiles()

    def load_test_data(self) -> List[Dict]:
        """Load test data"""
        with open(self.test_data_path, 'r') as f:
            data_file = json.load(f)

        if isinstance(data_file, dict) and 'samples' in data_file:
            all_data = data_file['samples']
        elif isinstance(data_file, list):
            all_data = data_file
        else:
            raise ValueError(f"Unexpected data format")

        return all_data[:self.num_samples]

    def load_scaffold_mapping(self) -> Dict:
        """Load scaffold SMILES mapping"""
        with open(self.scaffold_mapping_path, 'r') as f:
            return json.load(f)

    def load_train_smiles(self) -> set:
        """Load training SMILES set for novelty calculation"""
        print("Loading training SMILES for novelty evaluation...")
        import re
        train_smiles = set()
        try:
            with open(self.train_data_path, 'r') as f:
                data = json.load(f)
            samples = data['samples'] if isinstance(data, dict) and 'samples' in data else data
            for item in samples:
                output = item.get('output', '')
                m = re.search(r'"([^"]+)"', output)
                if m:
                    smi = m.group(1)
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        train_smiles.add(Chem.MolToSmiles(mol))
            print(f"  Loaded {len(train_smiles)} training SMILES")
        except Exception as e:
            print(f"  Warning: could not load training SMILES ({e})")
        return train_smiles

    @abstractmethod
    def load_model(self):
        """Load the baseline model (to be implemented by subclasses)"""
        pass

    @abstractmethod
    def generate_smiles(self, instruction: str, input_text: str) -> str:
        """Generate SMILES from text input (to be implemented by subclasses)"""
        pass

    def convert_to_text_only(self, item: Dict, scaffold_smiles: str = None) -> Tuple[str, str]:
        """
        Convert multimodal data to text-only format.
        When scaffold_smiles is provided, replaces image references with the
        actual scaffold SMILES string so baselines receive equivalent information.

        Returns:
            instruction: Modified instruction
            input_text: Modified input with scaffold SMILES injected
        """
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')

        if scaffold_smiles:
            scaffold_placeholder = f"the scaffold SMILES: {scaffold_smiles}"

            # Replace all image-referencing preambles in input_text
            image_preambles = [
                r'Using the scaffold structure in the image',
                r'Using the scaffold depicted',
                r'Using the provided scaffold structure',
                r'Based on the scaffold image provided',
                r'Based on the scaffold in the image',
                r'Based on the scaffold shown in the image',
                r'From the given scaffold image',
                r'From the scaffold framework in the image',
                r'Given the scaffold structure shown',
                r'With the scaffold framework shown',
            ]
            pattern = '(' + '|'.join(re.escape(p) for p in image_preambles) + ')'
            input_text = re.sub(pattern, lambda m: f'Using {scaffold_placeholder}', input_text)

            # Update instructions that reference an image or scaffold implicitly
            instruction = re.sub(
                r'based on the scaffold structure',
                lambda m, s=scaffold_placeholder: f'based on {s}',
                instruction, flags=re.IGNORECASE
            )
            instruction = re.sub(
                r'using the (given |provided )?scaffold (framework|structure)?',
                lambda m, s=scaffold_placeholder: f'using {s}',
                instruction, flags=re.IGNORECASE
            )
            instruction = re.sub(
                r'from the (given )?scaffold',
                lambda m, s=scaffold_placeholder: f'from {s}',
                instruction, flags=re.IGNORECASE
            )
        else:
            # Fallback: just strip image-referencing phrases (original behaviour)
            instruction = instruction.replace('based on the scaffold structure', 'with specific properties')
            input_text = re.sub(
                r'(Using the scaffold structure in the image|Using the scaffold depicted'
                r'|Based on the scaffold(?: image)? (?:shown|provided|in the image)?'
                r'|From the (?:given )?scaffold(?: framework)? in the image'
                r'|Given the scaffold structure shown|With the scaffold framework shown'
                r'|Using the provided scaffold structure),?\s*',
                '', input_text
            )

        return instruction, input_text

    def parse_target_properties(self, input_text: str) -> Dict:
        """Parse target properties from input text"""
        target_props = {}

        # LogP
        logp_match = re.search(r'lipophilicity around ([\d\.]+)', input_text)
        if logp_match:
            target_props['LogP'] = float(logp_match.group(1).rstrip('.'))

        # MW
        mw_match = re.search(r'molecular weight near ([\d\.]+)', input_text)
        if mw_match:
            target_props['MW'] = float(mw_match.group(1).rstrip('.'))

        # HBD
        hbd_match = re.search(r'(\d+) HBD', input_text)
        if hbd_match:
            target_props['HBD'] = int(hbd_match.group(1))

        # HBA
        hba_match = re.search(r'(\d+) HBA', input_text)
        if hba_match:
            target_props['HBA'] = int(hba_match.group(1))

        # TPSA
        tpsa_match = re.search(r'TPSA of around ([\d\.]+)', input_text)
        if tpsa_match:
            target_props['TPSA'] = float(tpsa_match.group(1).rstrip('.'))

        # QED
        qed_match = re.search(r'QED around ([\d\.]+)', input_text)
        if qed_match:
            target_props['QED'] = float(qed_match.group(1).rstrip('.'))

        return target_props

    def extract_smiles_from_output(self, output: str) -> str:
        """Extract SMILES from model output"""
        # Try different patterns
        patterns = [
            r'"([^"]+)"',
            r'is\s+"?([A-Za-z0-9@\[\]\(\)=#\-\+\\/\.]+)"?',
            r'([A-Za-z0-9@\[\]\(\)=#\-\+\\/\.]+)'
        ]

        for pattern in patterns:
            match = re.search(pattern, output)
            if match:
                smiles = match.group(1)
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    return smiles

        return None

    def calculate_properties(self, smiles: str) -> Dict:
        """Calculate molecular properties"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        return {
            'MW': round(Descriptors.MolWt(mol), 2),
            'LogP': round(Descriptors.MolLogP(mol), 2),
            'HBD': Lipinski.NumHDonors(mol),
            'HBA': Lipinski.NumHAcceptors(mol),
            'TPSA': round(Descriptors.TPSA(mol), 2),
            'QED': round(QED.qed(mol), 3)
        }

    def check_scaffold_preservation(self, generated_smiles: str, reference_smiles: str) -> bool:
        """Check if generated molecule contains reference scaffold"""
        try:
            gen_mol = Chem.MolFromSmiles(generated_smiles)
            ref_mol = Chem.MolFromSmiles(reference_smiles)

            if gen_mol is None or ref_mol is None:
                return False

            return gen_mol.HasSubstructMatch(ref_mol)
        except:
            return False

    def evaluate_property_match(self, actual_props: Dict, target_props: Dict) -> Dict:
        """Evaluate if properties match targets within tolerance"""
        satisfied = {}

        for key, target in target_props.items():
            if key not in actual_props:
                continue

            tolerance = self.tolerances.get(key, 0)
            satisfied[key] = abs(actual_props[key] - target) <= tolerance

        return satisfied

    def calculate_diversity(self, smiles_list: List[str]) -> float:
        """Internal diversity: 1 - mean pairwise Tanimoto (Morgan fp)"""
        fps = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))

        if len(fps) < 2:
            return 0.0

        sims = []
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))

        return round(1.0 - float(np.mean(sims)), 4)

    def calculate_novelty(self, unique_valid_smiles: List[str]) -> float:
        """Novelty: fraction of unique valid molecules not seen in training set"""
        if not unique_valid_smiles or not self.train_smiles_set:
            return 0.0
        novel = sum(1 for smi in unique_valid_smiles if smi not in self.train_smiles_set)
        return round(novel / len(unique_valid_smiles) * 100, 2)

    def calculate_fingerprint_similarities(
        self, generated_smiles: List[str], reference_smiles: List[str]
    ) -> Dict:
        """
        Three fingerprint similarities between generated and reference (ground truth) molecules.
        Returns mean Morgan (ECFP4), MACCS, and RDKit fingerprint Tanimoto similarities.
        """
        morgan_sims, maccs_sims, rdkit_sims = [], [], []

        for gen, ref in zip(generated_smiles, reference_smiles):
            gen_mol = Chem.MolFromSmiles(gen)
            ref_mol = Chem.MolFromSmiles(ref)
            if gen_mol is None or ref_mol is None:
                continue

            # Morgan (ECFP4)
            gen_morgan = AllChem.GetMorganFingerprintAsBitVect(gen_mol, 2, nBits=2048)
            ref_morgan = AllChem.GetMorganFingerprintAsBitVect(ref_mol, 2, nBits=2048)
            morgan_sims.append(DataStructs.TanimotoSimilarity(gen_morgan, ref_morgan))

            # MACCS
            gen_maccs = MACCSkeys.GenMACCSKeys(gen_mol)
            ref_maccs = MACCSkeys.GenMACCSKeys(ref_mol)
            maccs_sims.append(DataStructs.TanimotoSimilarity(gen_maccs, ref_maccs))

            # RDKit
            gen_rdkit = Chem.RDKFingerprint(gen_mol)
            ref_rdkit = Chem.RDKFingerprint(ref_mol)
            rdkit_sims.append(DataStructs.TanimotoSimilarity(gen_rdkit, ref_rdkit))

        def safe_mean(lst):
            return round(float(np.mean(lst)), 4) if lst else 0.0

        return {
            'morgan_similarity': safe_mean(morgan_sims),
            'maccs_similarity':  safe_mean(maccs_sims),
            'rdkit_similarity':  safe_mean(rdkit_sims),
            'num_pairs':         len(morgan_sims),
        }

    def run_evaluation(self) -> Dict:
        """Run full evaluation pipeline"""
        print("="*80)
        print(f"Running {self.__class__.__name__}")
        print("="*80)

        # Load model
        print("[1/3] Loading model...")
        self.load_model()

        # Run inference
        print(f"[2/3] Running inference on {len(self.test_data)} samples...")

        results = []
        all_generated_smiles = []   # raw (may be invalid)
        all_valid_smiles = []       # canonical valid SMILES
        ref_smiles_paired = []      # ground-truth SMILES paired with valid generated
        scaffold_match_count = 0

        for idx, item in enumerate(self.test_data):
            print(f"\rProgress: {idx+1}/{len(self.test_data)}", end='', flush=True)

            # Get reference scaffold — needed for prompt injection and evaluation
            image_key = item.get('scaffold_image', '')
            reference_scaffold = self.scaffold_mapping.get(image_key)

            if not reference_scaffold:
                # Fallback: derive scaffold from ground-truth output
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

            # Ground-truth SMILES for fingerprint similarity
            gt_smiles = self.extract_smiles_from_output(item.get('output', ''))

            # Convert to text-only format, injecting scaffold SMILES into the prompt
            instruction, input_text = self.convert_to_text_only(item, scaffold_smiles=reference_scaffold)

            # Generate
            try:
                generated_output = self.generate_smiles(instruction, input_text)
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
                    'error': 'Failed to extract SMILES from output'
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

            # Paired ground-truth for fingerprint similarity
            if gt_smiles and Chem.MolFromSmiles(gt_smiles):
                ref_smiles_paired.append(Chem.MolToSmiles(Chem.MolFromSmiles(gt_smiles)))
            else:
                ref_smiles_paired.append(None)

            # Scaffold preservation
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


        total_attempts  = len(self.test_data)  
        total_generated = len(all_generated_smiles)
        total_valid     = len(all_valid_smiles)
        unique_smiles   = list(set(all_valid_smiles))
        total_unique    = len(unique_smiles)
        total_evaluated = len(results)

        validity    = round(total_valid  / total_attempts * 100, 2) if total_attempts > 0 else 0.0
        uniqueness  = round(total_unique / total_valid    * 100, 2) if total_valid    > 0 else 0.0
        diversity   = self.calculate_diversity(unique_smiles)
        novelty     = self.calculate_novelty(unique_smiles)
        scaffold_rate = round(scaffold_match_count / total_evaluated * 100, 2) if total_evaluated > 0 else 0.0

        # Fingerprint similarity: only pair valid generated vs their ground truth
        paired_gen = []
        paired_ref = []
        for smi, ref in zip(all_valid_smiles, ref_smiles_paired[:len(all_valid_smiles)]):
            if ref is not None:
                paired_gen.append(smi)
                paired_ref.append(ref)
        fp_sims = self.calculate_fingerprint_similarities(paired_gen, paired_ref)

        # ── Print ─────────────────────────────────────────────────────────────
        print(f"\n Results:")
        print(f"  Validity:              {total_valid}/{total_attempts} ({validity:.2f}%)")
        print(f"  Uniqueness:            {total_unique}/{total_valid} ({uniqueness:.2f}%)")
        print(f"  Diversity:             {diversity:.4f}")
        print(f"  Novelty:               {novelty:.2f}%  (over {total_unique} unique valid mols)")
        print(f"  Scaffold preservation: {scaffold_match_count}/{total_evaluated} ({scaffold_rate:.2f}%)")
        print(f"  Morgan similarity:     {fp_sims['morgan_similarity']:.4f}")
        print(f"  MACCS similarity:      {fp_sims['maccs_similarity']:.4f}")
        print(f"  RDKit similarity:      {fp_sims['rdkit_similarity']:.4f}")

        # ── Save ──────────────────────────────────────────────────────────────
        output_data = {
            'model': self.__class__.__name__,
            'num_samples_tested': len(self.test_data),
            'metrics': {
                'validity':             validity,
                'uniqueness':           uniqueness,
                'diversity':            diversity,
                'novelty':              novelty,
                'scaffold_preservation_rate': scaffold_rate,
                'morgan_similarity':    fp_sims['morgan_similarity'],
                'maccs_similarity':     fp_sims['maccs_similarity'],
                'rdkit_similarity':     fp_sims['rdkit_similarity'],
            },
            'counts': {
                'total_attempts':   total_attempts,
                'total_generated':  total_generated,
                'total_valid':      total_valid,
                'total_unique':     total_unique,
                'scaffold_matches': scaffold_match_count,
                'fp_pairs':         fp_sims['num_pairs'],
            },
            'detailed_results': results
        }

        with open(self.output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n Results saved to: {self.output_path}")
        print("="*80)

        return output_data
