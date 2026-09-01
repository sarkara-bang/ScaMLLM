#!/usr/bin/env python3
"""
DrugAssist-7B Baseline Evaluator
"""
import sys
sys.path.append('/root/autodl-tmp/multimodel/experiment/baselines')

from base_evaluator import BaseEvaluator
from transformers import AutoTokenizer, LlamaForCausalLM
import torch


class DrugAssistEvaluator(BaseEvaluator):
    """Evaluator for DrugAssist-7B baseline"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_path = "/root/autodl-tmp/multimodel/models/DrugAssist-7B"
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load DrugAssist-7B model"""
        print(f"Loading DrugAssist-7B from {self.model_path}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = LlamaForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        self.model.eval()
        print("DrugAssist-7B model loaded successfully")

    def format_drugassist_prompt(self, instruction: str, input_text: str) -> str:
        prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
        return prompt

    def generate_smiles(self, instruction: str, input_text: str) -> str:
        """Generate SMILES using DrugAssist"""
        prompt = self.format_drugassist_prompt(instruction, input_text)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                num_beams=1,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "### Response:" in generated_text:
            response = generated_text.split("### Response:")[-1].strip()
        else:
            response = generated_text

        return response


if __name__ == "__main__":
    evaluator = DrugAssistEvaluator(
        test_data_path="/root/autodl-tmp/multimodel/data/2000_test.json",
        scaffold_mapping_path="/root/autodl-tmp/multimodel/data/scaffold_mapping.json",
        output_path="/root/autodl-tmp/multimodel/experiment/results/drugassist_results.json",
        num_samples=2000, 
        device="cuda:0"
    )

    results = evaluator.run_evaluation()
    print("\n DrugAssist-7B evaluation completed!")
