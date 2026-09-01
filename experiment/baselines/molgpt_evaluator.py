#!/usr/bin/env python3
"""
MolGPT Baseline Evaluator
Uses MolGPT scaffold+logp conditioned model for scaffold-aware molecule generation.
Checkpoint: moses_scaf_wholeseq_logp_newtokens.pt
"""
import os
import sys

sys.path.append('/root/autodl-tmp/multimodel/experiment/baselines')
sys.path.append('/root/autodl-tmp/molgpt/generate')

from base_evaluator import BaseEvaluator
from model import GPT, GPTConfig
import torch
import torch.nn.functional as F
import re
import json


class MolGPTEvaluator(BaseEvaluator):
    """Evaluator for MolGPT — scaffold + LogP conditioned generation (MOSES)"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.checkpoint_path = "/root/autodl-tmp/molgpt/checkpoints/moses_scaf_wholeseq_logp_newtokens.pt"
        self.vocab_file = "/root/autodl-tmp/molgpt/moses2_stoi.json"
        self.block_size = 54
        self.scaffold_maxlen = 48   
        self.n_layer = 8
        self.n_head = 8
        self.n_embd = 256
        self.num_props = 1          

        self.model = None
        self.stoi = None
        self.itos = None

    def load_model(self):
        """Load scaffold-conditioned MolGPT model"""
        print(f"Loading MolGPT (scaffold+LogP) from {self.checkpoint_path}...")

        with open(self.vocab_file, 'r') as f:
            self.stoi = json.load(f)
        self.itos = {v: k for k, v in self.stoi.items()}
        vocab_size = len(self.stoi)

        mconf = GPTConfig(
            vocab_size=vocab_size,
            block_size=self.block_size,
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_embd=self.n_embd,
            num_props=self.num_props,
            scaffold_maxlen=self.scaffold_maxlen,
            lstm=False,
            lstm_layers=0,
            scaffold=True
        )

        self.model = GPT(mconf)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model = self.model.to(self.device)
        self.model.eval()
        print("MolGPT model loaded successfully")

    def tokenize(self, smiles: str) -> list:
        """Tokenize SMILES string into vocab indices"""
        pattern = r"(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        tokens = re.findall(pattern, smiles)
        unk = self.stoi.get('<', 0)
        return [self.stoi.get(t, unk) for t in tokens]

    def detokenize(self, tokens: list) -> str:
        return ''.join([self.itos.get(t, '') for t in tokens])

    def encode_scaffold(self, scaffold_smiles: str) -> torch.Tensor:
        """
        Encode scaffold to fixed-length tensor (scaffold_maxlen),
        padded with '<' token — same convention as MolGPT training.
        """
        tokens = self.tokenize(scaffold_smiles)
        pad_id = self.stoi.get('<', 0)


        tokens = tokens[:self.scaffold_maxlen]
        tokens = tokens + [pad_id] * (self.scaffold_maxlen - len(tokens))

        return torch.tensor([tokens], dtype=torch.long).to(self.device)

    def sample_from_model(self, x, steps, prop, scaffold_tensor,
                          temperature=1.0, top_k=None):
        """Auto-regressive sampling with scaffold and property conditioning"""
        self.model.eval()
        for _ in range(steps):
            x_cond = x if x.size(1) <= self.block_size else x[:, -self.block_size:]
            logits, _, _ = self.model(x_cond, prop=prop, scaffold=scaffold_tensor)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            ix = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, ix), dim=1)
        return x

    def generate_smiles(self, instruction: str, input_text: str) -> str:
        """
        Generate SMILES conditioned on scaffold SMILES + LogP.
        Scaffold is extracted from input_text (injected by convert_to_text_only).
        """
        
        scaffold_match = re.search(
            r'scaffold SMILES:\s*([A-Za-z0-9@\[\]\(\)=#\-\+\\/\.\%]+)', input_text
        )
        scaffold_smiles = scaffold_match.group(1) if scaffold_match else 'C'
        scaffold_tensor = self.encode_scaffold(scaffold_smiles)

        logp_match = re.search(r'lipophilicity around ([\d\.\-]+)', input_text)
        if logp_match:
            logp_raw = float(logp_match.group(1).rstrip('.'))
            logp_norm = max(0.0, min(1.0, (logp_raw + 5.0) / 15.0))
        else:
            logp_norm = 0.5
        prop = torch.tensor([[logp_norm]], dtype=torch.float).to(self.device)

        start_tokens = self.tokenize('C')
        x = torch.tensor([start_tokens], dtype=torch.long).to(self.device)

        with torch.no_grad():
            samples = self.sample_from_model(
                x,
                steps=self.block_size - len(start_tokens),
                prop=prop,
                scaffold_tensor=scaffold_tensor,
                temperature=1.0,
                top_k=None
            )

        generated_tokens = samples[0].cpu().tolist()
        generated_smiles = self.detokenize(generated_tokens)
        generated_smiles = generated_smiles.replace('<', '').strip()
        return generated_smiles


if __name__ == "__main__":
    evaluator = MolGPTEvaluator(
        test_data_path="/root/autodl-tmp/multimodel/data/2000_test.json",
        scaffold_mapping_path="/root/autodl-tmp/multimodel/data/scaffold_mapping.json",
        output_path="/root/autodl-tmp/multimodel/experiment/results/molgpt_results.json",
        num_samples=2000,
        device="cuda:0"
    )

    results = evaluator.run_evaluation()
   

