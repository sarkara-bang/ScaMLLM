import torch
from transformers import Trainer
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


class ScaffoldAwareTrainer(Trainer):

    def __init__(
        self,
        scaffold_loss_fn=None,
        scaffold_loss_freq: int = 5,
        scaffold_loss_start_step: int = 0,
        generation_config: Optional[Dict] = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.scaffold_loss_fn = scaffold_loss_fn
        self.scaffold_loss_freq = scaffold_loss_freq
        self.scaffold_loss_start_step = scaffold_loss_start_step

        self.generation_config = generation_config or {
            'max_new_tokens': 64,
            'do_sample': True,
            'temperature': 1.0,
            'num_beams': 1,
        }

        logger.info(f"ScaffoldAwareTrainer initialized")
        logger.info(f"   Scaffold loss frequency: every {scaffold_loss_freq} steps")
        logger.info(f"   Scaffold loss starts at step: {scaffold_loss_start_step}")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        scaffold_smiles_batch = inputs.pop('scaffold_smiles', None)

        should_compute_scaffold = (
            self.scaffold_loss_fn is not None and
            scaffold_smiles_batch is not None and
            self.state.global_step >= self.scaffold_loss_start_step and
            self.state.global_step % self.scaffold_loss_freq == 0
        )

        if not should_compute_scaffold:
            outputs = model(**inputs)
            loss = outputs.loss

            return (loss, outputs) if return_outputs else loss

        outputs = model(**inputs)
        lm_loss = outputs.loss

        input_ids = inputs.get('input_ids')
        attention_mask = inputs.get('attention_mask')

        try:
            base_model = model
            if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
                base_model = model.base_model.model
            elif hasattr(model, 'module'):
                base_model = model.module

            generated_outputs = base_model.generate(
                input_ids=input_ids,
                pixel_values=None,
                attention_mask=attention_mask,
                return_dict_in_generate=True,
                output_scores=True,
                **self.generation_config
            )

            generated_ids = generated_outputs.sequences
            scores = generated_outputs.scores

            logits = torch.stack(scores, dim=1)

            prompt_length = input_ids.shape[1]
            new_generated_ids = generated_ids[:, prompt_length:]

            gen_attention_mask = torch.ones_like(new_generated_ids, dtype=torch.float32)

            scaffold_loss, scaffold_metrics = self.scaffold_loss_fn(
                logits=logits,
                generated_ids=new_generated_ids,
                scaffold_smiles_list=scaffold_smiles_batch,
                tokenizer=self.tokenizer,
                attention_mask=gen_attention_mask
            )

            total_loss = lm_loss + scaffold_loss

            self.log({
                'lm_loss': lm_loss.item(),
                'scaffold_loss': scaffold_loss.item(),
                'avg_reward': scaffold_metrics.get('avg_reward', 0.0),
                'scaffold_match_rate': scaffold_metrics.get('scaffold_match_rate', 0.0),
            })

            if self.state.global_step % 100 == 0:
                logger.info(
                    f"\nStep {self.state.global_step}: "
                    f"LM Loss={lm_loss.item():.4f}, "
                    f"Scaffold Loss={scaffold_loss.item():.4f}"
                )

        except Exception as e:
            logger.warning(f"Failed to compute scaffold loss: {e}")
            logger.warning("Falling back to LM loss only")
            total_loss = lm_loss

        return (total_loss, outputs) if return_outputs else total_loss

    def _prepare_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        scaffold_smiles = inputs.get('scaffold_smiles', None)

        inputs = super()._prepare_inputs(inputs)

        if scaffold_smiles is not None:
            inputs['scaffold_smiles'] = scaffold_smiles

        return inputs
