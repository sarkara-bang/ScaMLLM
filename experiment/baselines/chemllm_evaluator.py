#!/usr/bin/env python3
"""
ChemLLM Baseline Evaluator
Uses AI4Chem/ChemLLM-7B-Chat for scaffold-based molecule generation
ChemLLM is based on InternLM2, fine-tuned on chemistry datasets.
"""
import re
import sys
sys.path.append('/root/autodl-tmp/multimodel/experiment/baselines')

from base_evaluator import BaseEvaluator
from transformers import AutoTokenizer, AutoModelForCausalLM
from rdkit import Chem
import torch


class ChemLLMEvaluator(BaseEvaluator):
    """Evaluator for ChemLLM-7B-Chat baseline"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_path = "/root/autodl-tmp/models/ChemLLM-7B-Chat"
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load ChemLLM model"""
        print(f"Loading ChemLLM from {self.model_path}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            use_fast=False
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        self.model.eval()
        print("ChemLLM model loaded successfully")

    def format_prompt(self, instruction: str, input_text: str) -> str:
        """
        Format prompt for ChemLLM (InternLM2 chat format).
        """
        system = (
            "You are a chemistry expert. Your task is to generate a valid SMILES string "
            "for a molecule satisfying the given conditions. "
            "Output only the SMILES string directly, nothing else."
        )
        user_msg = (
            f"{instruction}\n\n{input_text}\n\n"
            "Please output only the SMILES string of the generated molecule."
        )

        prompt = (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        return prompt

    def clean_atom_map(self, smiles: str) -> str:
        """Remove atom mapping numbers like [CH2:1] -> [CH2]"""
        cleaned = re.sub(r'\[([A-Za-z@+\-]+):\d+\]', r'[\1]', smiles)
        # also strip lone :num inside brackets
        cleaned = re.sub(r':\d+', '', cleaned)
        return cleaned

    def extract_smiles_from_output(self, output: str) -> str:
        """Override to handle ChemLLM output format (may contain atom-mapped SMILES)"""

        candidates = []

        # Pattern 1: SMILES: xxx
        m = re.search(r'SMILES:\s*(\S+)', output)
        if m:
            candidates.append(m.group(1).strip())

        # Pattern 2: <SMILES>xxx</SMILES>
        m = re.search(r'<SMILES>\s*([^<]+?)\s*</SMILES>', output)
        if m:
            candidates.append(m.group(1).strip())

        # Pattern 3: quoted "xxx"
        m = re.search(r'"([A-Za-z0-9@\[\]\(\)=#\-\+\\/\.\%:]+)"', output)
        if m:
            candidates.append(m.group(1).strip())

        # Pattern 4: longest SMILES-like token in output
        tokens = re.findall(r'[A-Za-z0-9@\[\]\(\)=#\-\+\\/\.\%:]{6,}', output)
        candidates.extend(tokens)

        for raw in candidates:
            # try as-is
            if Chem.MolFromSmiles(raw):
                return raw
            # try after stripping atom map numbers
            cleaned = self.clean_atom_map(raw)
            if Chem.MolFromSmiles(cleaned):
                return cleaned

        return None

    def generate_smiles(self, instruction: str, input_text: str) -> str:
        """Generate SMILES using ChemLLM"""
        prompt = self.format_prompt(instruction, input_text)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        return generated_text


if __name__ == "__main__":
    evaluator = ChemLLMEvaluator(
        test_data_path="/root/autodl-tmp/multimodel/data/2000_test.json",
        scaffold_mapping_path="/root/autodl-tmp/multimodel/data/scaffold_mapping.json",
        output_path="/root/autodl-tmp/multimodel/experiment/results/chemllm_results.json",
        num_samples=2000,
        device="cuda:0"
    )

    results = evaluator.run_evaluation()

