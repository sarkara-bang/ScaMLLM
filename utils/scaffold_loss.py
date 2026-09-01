import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
import re
from typing import Optional, List, Tuple

def extract_smiles_from_generated(text: str) -> Optional[str]:
    patterns = [
        r'"([^"]+)"',
        r'is\s+"?([A-Za-z0-9@\[\]\(\)=#\-\+\\/\.]+)"?',
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            candidate = match.group(1)
            mol = Chem.MolFromSmiles(candidate)
            if mol is not None:
                return candidate
    return None


class ScaffoldEvaluator:

    def __init__(
        self,
        use_hybrid: bool = True,
        substructure_weight: float = 0.7,
        fingerprint_weight: float = 0.3,
        invalid_penalty: float = -1.0
    ):
        self.use_hybrid = use_hybrid
        self.substructure_weight = substructure_weight
        self.fingerprint_weight = fingerprint_weight
        self.invalid_penalty = invalid_penalty

        assert abs(substructure_weight + fingerprint_weight - 1.0) < 1e-6, \
            "Weights must sum to 1.0"

    def compute_substructure_match(self, gen_smiles: str, scaffold_smiles: str) -> float:
        try:
            gen_mol = Chem.MolFromSmiles(gen_smiles)
            scaffold_mol = Chem.MolFromSmiles(scaffold_smiles)

            if gen_mol is None or scaffold_mol is None:
                return 0.0

            has_scaffold = gen_mol.HasSubstructMatch(scaffold_mol)
            return 1.0 if has_scaffold else 0.0

        except Exception as e:
            return 0.0

    def compute_fingerprint_similarity(self, gen_smiles: str, scaffold_smiles: str) -> float:
        try:
            gen_mol = Chem.MolFromSmiles(gen_smiles)
            scaffold_mol = Chem.MolFromSmiles(scaffold_smiles)

            if gen_mol is None or scaffold_mol is None:
                return 0.0

            if gen_mol.HasSubstructMatch(scaffold_mol):
                try:
                    gen_scaffold = MurckoScaffold.GetScaffoldForMol(gen_mol)
                    gen_fp = AllChem.GetMorganFingerprintAsBitVect(gen_scaffold, radius=2, nBits=2048)
                except Exception:
                    gen_fp = AllChem.GetMorganFingerprintAsBitVect(gen_mol, radius=2, nBits=2048)
            else:
                gen_fp = AllChem.GetMorganFingerprintAsBitVect(gen_mol, radius=2, nBits=2048)

            scaffold_fp = AllChem.GetMorganFingerprintAsBitVect(scaffold_mol, radius=2, nBits=2048)

            similarity = DataStructs.TanimotoSimilarity(gen_fp, scaffold_fp)
            return similarity

        except Exception as e:
            return 0.0

    def compute_reward(self, gen_smiles: str, scaffold_smiles: str) -> float:
        if gen_smiles is None or scaffold_smiles is None:
            return self.invalid_penalty

        try:
            gen_mol = Chem.MolFromSmiles(gen_smiles)
            if gen_mol is None:
                return self.invalid_penalty

            if self.use_hybrid:
                substructure_score = self.compute_substructure_match(gen_smiles, scaffold_smiles)
                fingerprint_score = self.compute_fingerprint_similarity(gen_smiles, scaffold_smiles)

                reward = (self.substructure_weight * substructure_score +
                         self.fingerprint_weight * fingerprint_score)
            else:
                reward = self.compute_substructure_match(gen_smiles, scaffold_smiles)

            return reward

        except Exception as e:
            return self.invalid_penalty

    def evaluate_batch(
        self,
        generated_texts: List[str],
        scaffold_smiles_list: List[str]
    ) -> dict:
        substructure_matches = []
        fingerprint_similarities = []
        rewards = []
        valid_molecules = 0

        for gen_text, scaffold_smiles in zip(generated_texts, scaffold_smiles_list):
            if gen_text is None or scaffold_smiles is None:
                continue

            if isinstance(gen_text, str):
                mol = Chem.MolFromSmiles(gen_text)
                gen_smiles = gen_text if mol else extract_smiles_from_generated(gen_text)
            else:
                gen_smiles = None

            if gen_smiles is None:
                continue

            valid_molecules += 1

            substructure_match = self.compute_substructure_match(gen_smiles, scaffold_smiles)
            fingerprint_sim = self.compute_fingerprint_similarity(gen_smiles, scaffold_smiles)
            reward = self.compute_reward(gen_smiles, scaffold_smiles)

            substructure_matches.append(substructure_match)
            fingerprint_similarities.append(fingerprint_sim)
            rewards.append(reward)

        if len(rewards) == 0:
            return {
                'scaffold_match_rate': 0.0,
                'avg_fingerprint_similarity': 0.0,
                'avg_reward': 0.0,
                'valid_molecule_rate': 0.0,
                'num_valid': 0
            }

        return {
            'scaffold_match_rate': sum(substructure_matches) / len(substructure_matches),
            'avg_fingerprint_similarity': sum(fingerprint_similarities) / len(fingerprint_similarities),
            'avg_reward': sum(rewards) / len(rewards),
            'valid_molecule_rate': valid_molecules / len(generated_texts),
            'num_valid': valid_molecules
        }


class ScaffoldReinforceLoss(nn.Module):

    def __init__(
        self,
        weight: float = 0.5,
        use_hybrid: bool = True,
        substructure_weight: float = 0.7,
        baseline_alpha: float = 0.9,
        invalid_penalty: float = -1.0,
        normalize_rewards: bool = True
    ):
        super().__init__()
        self.weight = weight
        self.baseline_alpha = baseline_alpha
        self.baseline = 0.0
        self.normalize_rewards = normalize_rewards

        self.evaluator = ScaffoldEvaluator(
            use_hybrid=use_hybrid,
            substructure_weight=substructure_weight,
            fingerprint_weight=1.0 - substructure_weight,
            invalid_penalty=invalid_penalty
        )

    def forward(
        self,
        logits: torch.Tensor,
        generated_ids: torch.Tensor,
        scaffold_smiles_list: List[str],
        tokenizer,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        batch_size, seq_len, vocab_size = logits.shape

        log_probs = F.log_softmax(logits, dim=-1)

        selected_log_probs = log_probs.gather(
            dim=2,
            index=generated_ids.unsqueeze(-1)
        ).squeeze(-1)

        if attention_mask is not None:
            selected_log_probs = selected_log_probs * attention_mask
            sequence_log_probs = selected_log_probs.sum(dim=1)
        else:
            sequence_log_probs = selected_log_probs.sum(dim=1)

        rewards = []
        generated_smiles = []

        for i in range(batch_size):
            gen_text = tokenizer.decode(generated_ids[i], skip_special_tokens=True)

            if "### Response:" in gen_text:
                response = gen_text.split("### Response:")[-1].strip()
            else:
                response = gen_text

            mol = Chem.MolFromSmiles(response)
            gen_smiles = response if mol else extract_smiles_from_generated(response)
            generated_smiles.append(gen_smiles)

            reward = self.evaluator.compute_reward(gen_smiles, scaffold_smiles_list[i])
            rewards.append(reward)

        rewards_tensor = torch.tensor(rewards, device=logits.device, dtype=torch.float32)

        current_mean_reward = rewards_tensor.mean().item()
        self.baseline = self.baseline_alpha * self.baseline + (1 - self.baseline_alpha) * current_mean_reward

        reward_std = rewards_tensor.std()
        reward_range = rewards_tensor.max() - rewards_tensor.min()

        if reward_std < 1e-6 or reward_range < 1e-6 or torch.all(rewards_tensor == rewards_tensor[0]):
            weighted_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
            metrics = {
                'scaffold_loss': 0.0,
                'avg_reward': current_mean_reward,
                'baseline': self.baseline,
                'avg_advantage': 0.0,
                'reward_std': reward_std.item(),
            }
            eval_metrics = self.evaluator.evaluate_batch(generated_smiles, scaffold_smiles_list)
            metrics.update(eval_metrics)
            return weighted_loss, metrics

        if self.normalize_rewards and batch_size > 1:
            if reward_std > 1e-6:
                advantages = (rewards_tensor - rewards_tensor.mean()) / (reward_std + 1e-8)
            else:
                advantages = rewards_tensor - self.baseline
        else:
            advantages = rewards_tensor - self.baseline

        sequence_log_probs = torch.clamp(sequence_log_probs, min=-100.0, max=0.0)
        advantages = torch.clamp(advantages, min=-10.0, max=10.0)

        if torch.any(torch.isinf(sequence_log_probs)) or torch.any(torch.isnan(sequence_log_probs)):
            weighted_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
            policy_loss_value = 0.0
        else:
            policy_loss = -(sequence_log_probs * advantages.detach()).mean()

            if torch.isnan(policy_loss) or torch.isinf(policy_loss) or policy_loss.item() < -1e6 or policy_loss.item() > 1e6:
                weighted_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
                policy_loss_value = 0.0
            else:
                weighted_loss = policy_loss * self.weight
                policy_loss_value = weighted_loss.item()

        metrics = {
            'scaffold_loss': policy_loss_value,
            'avg_reward': current_mean_reward,
            'baseline': self.baseline,
            'avg_advantage': advantages.mean().item(),
            'reward_std': rewards_tensor.std().item(),
        }

        eval_metrics = self.evaluator.evaluate_batch(generated_smiles, scaffold_smiles_list)
        metrics.update(eval_metrics)

        return weighted_loss, metrics


def training_step_with_reinforeinrce(
    model,
    tokenizer,
    batch,
    scaffold_smiles_batch: List[str],
    reinforce_loss_fn: ScaffoldReinforceLoss,
    device,
    compute_scaffold_loss: bool = True,
    generation_config: dict = None
):
    if generation_config is None:
        generation_config = {
            'max_new_tokens': 64,
            'do_sample': True,
            'temperature': 1.0,
            'num_beams': 1,
        }

    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    pixel_values = batch['pixel_values'].to(device) if 'pixel_values' in batch else None
    labels = batch['labels'].to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        labels=labels,
        return_dict=True
    )

    lm_loss = outputs.loss

    if compute_scaffold_loss:
        generated_outputs = model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            return_dict_in_generate=True,
            output_scores=True,
            **generation_config
        )

        generated_ids = generated_outputs.sequences
        scores = generated_outputs.scores

        logits = torch.stack(scores, dim=1)

        prompt_length = input_ids.shape[1]
        new_generated_ids = generated_ids[:, prompt_length:]

        gen_attention_mask = torch.ones_like(new_generated_ids, dtype=torch.float32)

        scaffold_loss, scaffold_metrics = reinforce_loss_fn(
            logits=logits,
            generated_ids=new_generated_ids,
            scaffold_smiles_list=scaffold_smiles_batch,
            tokenizer=tokenizer,
            attention_mask=gen_attention_mask
        )

        total_loss = lm_loss + scaffold_loss

        metrics = {
            'total_loss': total_loss.item(),
            'lm_loss': lm_loss.item(),
            **scaffold_metrics
        }

    else:
        total_loss = lm_loss
        metrics = {
            'total_loss': total_loss.item(),
            'lm_loss': lm_loss.item(),
            'scaffold_loss': 0.0,
            'avg_reward': 0.0
        }

    return total_loss, metrics


def validation_step_with_scaffold_metrics(
    model,
    tokenizer,
    batch,
    scaffold_smiles_batch: List[str],
    evaluator: ScaffoldEvaluator,
    device,
    generation_config: dict = None
):
    if generation_config is None:
        generation_config = {
            'max_new_tokens': 64,
            'num_beams': 1,
            'do_sample': False,
        }

    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    pixel_values = batch['pixel_values'].to(device) if 'pixel_values' in batch else None
    labels = batch['labels'].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels
        )
        lm_loss = outputs.loss

        generated_ids = model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            **generation_config
        )

    generated_texts = []
    for gen_ids in generated_ids:
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        if "### Response:" in text:
            response = text.split("### Response:")[-1].strip()
        else:
            response = text
        generated_texts.append(response)

    metrics = evaluator.evaluate_batch(generated_texts, scaffold_smiles_batch)
    metrics['lm_loss'] = lm_loss.item()

    return metrics
