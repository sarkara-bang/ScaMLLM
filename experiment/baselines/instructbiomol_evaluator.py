#!/usr/bin/env python3
"""
InstructBioMol Baseline Evaluator
Uses hicai-zju/InstructBioMol-base for scaffold-based molecule generation
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import re
import sys
sys.path.append('/root/autodl-tmp/multimodel/experiment/baselines')

from base_evaluator import BaseEvaluator
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class InstructBioMolEvaluator(BaseEvaluator):
    """Evaluator for InstructBioMol baseline"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_path = "/root/autodl-tmp/models/InstructBioMol-instruct"
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load InstructBioMol model"""
        print(f"Loading InstructBioMol from {self.model_path}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            use_fast=False
        )

        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        self.model.eval()
        print("InstructBioMol model loaded successfully")

    def format_instructbiomol_prompt(self, instruction: str, input_text: str) -> str:
        """
        Format input for InstructBioMol.
        InstructBioMol expects instruction-following format for molecule generation.
        """
        # Combine instruction and input into a clear task description
        prompt = f"{instruction}\n\n{input_text}"

        # InstructBioMol uses instruction format
        # Adjust based on the model's training format
        formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"

        return formatted_prompt

    def extract_smiles_from_output(self, output: str) -> str:
        """Extract SMILES from InstructBioMol output"""
        from rdkit import Chem

        m = re.search(r'SMILES:\s*([A-Za-z0-9@\[\]\(\)=#\-\+\\/\.\%]+)', output)
        if m:
            smi = m.group(1).strip()
            if Chem.MolFromSmiles(smi):
                return smi

        m = re.search(r'"([A-Za-z0-9@\[\]\(\)=#\-\+\\/\.\%]+)"', output)
        if m:
            smi = m.group(1).strip()
            if Chem.MolFromSmiles(smi):
                return smi

        m = re.search(r'(?:molecule|SMILES|structure) is:?\s*([A-Za-z0-9@\[\]\(\)=#\-\+\\/\.\%]+)', output, re.IGNORECASE)
        if m:
            smi = m.group(1).strip()
            if Chem.MolFromSmiles(smi):
                return smi

        for token in re.findall(r'[A-Za-z0-9@\[\]\(\)=#\-\+\\/\.\%]{6,}', output):
            if Chem.MolFromSmiles(token):
                return token

        return None

    def generate_smiles(self, instruction: str, input_text: str) -> str:
        """Generate SMILES using InstructBioMol"""
        # Format input
        prompt = self.format_instructbiomol_prompt(instruction, input_text)

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode only the generated part (exclude input prompt)
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return generated_text


if __name__ == "__main__":
    evaluator = InstructBioMolEvaluator(
        test_data_path="/root/autodl-tmp/multimodel/data/2000_test.json",
        scaffold_mapping_path="/root/autodl-tmp/multimodel/data/scaffold_mapping.json",
        output_path="/root/autodl-tmp/multimodel/experiment/results/instructbiomol_results.json",
        num_samples=2000,
        device="cuda:0"
    )

    results = evaluator.run_evaluation()

