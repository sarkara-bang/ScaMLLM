#!/usr/bin/env python3
"""
LlaSMol Baseline Evaluator
Uses osunlp/LlaSMol-Mistral-7B for scaffold-based molecule generation
"""
import os
import re
import sys
sys.path.append('/root/autodl-tmp/multimodel/experiment/baselines')

from base_evaluator import BaseEvaluator
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch


class LlaSMolEvaluator(BaseEvaluator):
    """Evaluator for LlaSMol-Mistral-7B baseline (LoRA adapter)"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_model_path = "/root/autodl-tmp/models/Mistral-7B-v0.1"
        self.adapter_path = "/root/autodl-tmp/models/LlaSMol-Mistral-7B"
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load LlaSMol model (base model + LoRA adapter)"""
        print(f"Loading base model from {self.base_model_path}...")

        # Load tokenizer from base model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path,
            trust_remote_code=True,
            use_fast=False
        )

        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        # Load LoRA adapter
        print(f"Loading LoRA adapter from {self.adapter_path}...")
        self.model = PeftModel.from_pretrained(
            base_model,
            self.adapter_path,
            torch_dtype=torch.float16
        )

        self.model.eval()
        print("LlaSMol model (base + adapter) loaded successfully")

    def format_llasmol_prompt(self, instruction: str, input_text: str) -> str:
        """
        Format input for LlaSMol using instruction-following format.
        LlaSMol expects prompts in a conversational format.
        """
        # Combine instruction and input into a clear task description
        prompt = f"{instruction}\n\n{input_text}"

        # LlaSMol uses Mistral format (or similar instruction format)
        # Adjust based on the model's training format
        formatted_prompt = f"[INST] {prompt} [/INST]"

        return formatted_prompt

    def extract_smiles_from_output(self, output: str) -> str:
        """Override to handle LlaSMol output format: 'SMILES: xxx' or '<SMILES>xxx</SMILES>'"""
        from rdkit import Chem

        # Pattern 1: <SMILES>xxx</SMILES>
        m = re.search(r'<SMILES>\s*([^<]+?)\s*</SMILES>', output)
        if m:
            smi = m.group(1).strip()
            if Chem.MolFromSmiles(smi):
                return smi

        # Pattern 2: SMILES: xxx
        m = re.search(r'SMILES:\s*([A-Za-z0-9@\[\]\(\)=#\-\+\\/\.\%]+)', output)
        if m:
            smi = m.group(1).strip()
            if Chem.MolFromSmiles(smi):
                return smi

        # Pattern 3: quoted SMILES "xxx"
        m = re.search(r'"([A-Za-z0-9@\[\]\(\)=#\-\+\\/\.\%]+)"', output)
        if m:
            smi = m.group(1).strip()
            if Chem.MolFromSmiles(smi):
                return smi

        # Pattern 4: fallback — any SMILES-like token
        for token in re.findall(r'[A-Za-z0-9@\[\]\(\)=#\-\+\\/\.\%]{6,}', output):
            if Chem.MolFromSmiles(token):
                return token

        return None

    def generate_smiles(self, instruction: str, input_text: str) -> str:
        """Generate SMILES using LlaSMol"""
        # Format input
        prompt = self.format_llasmol_prompt(instruction, input_text)

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
    # Configuration
    evaluator = LlaSMolEvaluator(
        test_data_path="/root/autodl-tmp/multimodel/data/2000_test.json",
        scaffold_mapping_path="/root/autodl-tmp/multimodel/data/scaffold_mapping.json",
        output_path="/root/autodl-tmp/multimodel/experiment/results/llasmol_results.json",
        num_samples=2000,
        device="cuda:0"
    )

    results = evaluator.run_evaluation()

