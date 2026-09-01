#!/usr/bin/env python3
"""
MolT5 Baseline Evaluator
Uses microsoft/molt5-large-caption2smiles for text-only molecule generation
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import re
import sys
sys.path.append('/root/autodl-tmp/multimodel/experiment/baselines')

from base_evaluator import BaseEvaluator
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch


class MolT5Evaluator(BaseEvaluator):
    """Evaluator for MolT5-caption2smiles baseline"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = "laituan245/molt5-large-caption2smiles"
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load MolT5 model"""
        print(f"Loading MolT5 from {self.model_name}...")

        self.tokenizer = T5Tokenizer.from_pretrained(
            self.model_name,
            model_max_length=512
        )

        self.model = T5ForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        self.model.eval()
        print("MolT5 model loaded successfully")

    def format_molt5_input(self, instruction: str, input_text: str) -> str:
        """
        Format input for MolT5.
        Scaffold SMILES is already embedded in input_text by convert_to_text_only(),
        so we extract properties and pass the full enriched context to the model.
        """
        prompt_parts = []

        # Extract scaffold SMILES if present
        scaffold_match = re.search(r'scaffold SMILES:\s*([^\s,\.]+)', input_text)
        if scaffold_match:
            prompt_parts.append(f"scaffold={scaffold_match.group(1)}")

        # LogP
        logp_match = re.search(r'lipophilicity around ([\d\.]+)', input_text)
        if logp_match:
            prompt_parts.append(f"LogP={logp_match.group(1)}")

        # MW
        mw_match = re.search(r'molecular weight near ([\d\.]+)', input_text)
        if mw_match:
            prompt_parts.append(f"MW={mw_match.group(1)}")

        # HBD
        hbd_match = re.search(r'(\d+) HBD', input_text)
        if hbd_match:
            prompt_parts.append(f"HBD={hbd_match.group(1)}")

        # HBA
        hba_match = re.search(r'(\d+) HBA', input_text)
        if hba_match:
            prompt_parts.append(f"HBA={hba_match.group(1)}")

        # TPSA
        tpsa_match = re.search(r'TPSA (?:of |)around ([\d\.]+)', input_text)
        if tpsa_match:
            prompt_parts.append(f"TPSA={tpsa_match.group(1)}")

        # QED
        qed_match = re.search(r'QED (?:score of |)(?:around )?([\d\.]+)', input_text)
        if qed_match:
            prompt_parts.append(f"QED={qed_match.group(1)}")

        if prompt_parts:
            prompt = f"Generate molecule with {', '.join(prompt_parts)}"
        else:
            prompt = f"Generate molecule. {input_text}"

        return prompt

    def generate_smiles(self, instruction: str, input_text: str) -> str:
        """Generate SMILES using MolT5"""
        # Format input
        prompt = self.format_molt5_input(instruction, input_text)

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=512,
                num_beams=5,
                early_stopping=True,
                do_sample=False
            )

        # Decode
        generated_smiles = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return generated_smiles


if __name__ == "__main__":
    # Configuration
    evaluator = MolT5Evaluator(
        test_data_path="/root/autodl-tmp/multimodel/data/2000_test.json",
        scaffold_mapping_path="/root/autodl-tmp/multimodel/data/scaffold_mapping.json",
        output_path="/root/autodl-tmp/multimodel/experiment/results/molt5_results.json",
        num_samples=2000,
        device="cuda:0"
    )

    results = evaluator.run_evaluation()

