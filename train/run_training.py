import logging
import math
import os

os.environ["HF_DATASETS_CACHE"] = "/root/autodl-tmp/multimodel/.hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/root/autodl-tmp/multimodel/.hf_cache"
os.environ["HF_HOME"] = "/root/autodl-tmp/multimodel/.hf_cache"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass, field
from typing import Optional, Dict, List
import datasets
import torch

from utils.multimodal_dataset import build_multimodal_dataset, MultiModalDataCollator
from utils.multimodal_model import EnhancedMultiModalDrugAssist as MultiModalDrugAssist
from utils.scaffold_loss import ScaffoldReinforceLoss, ScaffoldEvaluator
from utils.trainer import ScaffoldAwareTrainer

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, PREFIX_CHECKPOINT_DIR
from transformers.utils import send_example_telemetry
from transformers.utils.versions import require_version

from peft import LoraConfig, get_peft_model

IGNORE_INDEX = -100

require_version("datasets>=1.8.0")


class KeepVisionModelEvalCallback(transformers.TrainerCallback):
    """Keep vision model in eval mode"""
    def on_step_begin(self, args, state, control, model=None, **kwargs):
        if model is not None:
            vision_model = None
            if hasattr(model, 'vision_model'):
                vision_model = model.vision_model
            elif hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
                if hasattr(model.base_model.model, 'vision_model'):
                    vision_model = model.base_model.model.vision_model
            elif hasattr(model, 'module') and hasattr(model.module, 'vision_model'):
                vision_model = model.module.vision_model

            if vision_model is not None:
                vision_model.eval()
                for module in vision_model.modules():
                    if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                        module.eval()
                        module.track_running_stats = False


class SavePeftModelCallback(transformers.TrainerCallback):
    """Save PEFT model checkpoint"""
    def save_model(self, args, state, kwargs):
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "enhanced_lora_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "enhanced_lora_model")

        try:
            model = kwargs["model"]
            final_state_dict = {}

            for name, param in model.named_parameters():
                if param.requires_grad and 'lora' in name.lower():
                    clean_name = name.replace('base_model.model.', '')
                    final_state_dict[clean_name] = param.data.clone()

            for name, param in model.named_parameters():
                if param.requires_grad:
                    if 'projection' in name and 'vision_model' not in name:
                        clean_name = name.replace('base_model.model.', '')
                        final_state_dict[clean_name] = param.data.clone()
                    elif 'fusion_norm' in name:
                        clean_name = name.replace('base_model.model.', '')
                        final_state_dict[clean_name] = param.data.clone()
                    elif 'cross_attention_fusion' in name:
                        clean_name = name.replace('base_model.model.', '')
                        final_state_dict[clean_name] = param.data.clone()

            os.makedirs(peft_model_path, exist_ok=True)
            torch.save(final_state_dict, os.path.join(peft_model_path, "adapter_model.bin"))
            kwargs["tokenizer"].save_pretrained(peft_model_path)

            if hasattr(kwargs["model"], "peft_config"):
                import json
                config = kwargs["model"].peft_config
                if isinstance(config, dict):
                    config = list(config.values())[0]
                elif isinstance(config, list):
                    config = config[0]
                with open(os.path.join(peft_model_path, "adapter_config.json"), 'w') as f:
                    json.dump(config.to_dict(), f, indent=2)

            print(f"Model saved to {peft_model_path}")

        except Exception as e:
            print(f"Error saving model: {e}")

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 1000 == 0 and state.global_step > 0:
            print(f"\n{'='*50}")
            print(f"Step {state.global_step}: Saving checkpoint...")
            print(f"{'='*50}")
            self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        print(f"\n{'='*50}")
        print(f"Training ended. Saving final model...")
        print(f"{'='*50}")
        self.save_model(args, state, kwargs)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    tokenizer_name_or_path: Optional[str] = field(default=None)
    imagemol_checkpoint_path: Optional[str] = field(
        default="models/ImageMol/ImageMol.pth.tar",
        metadata={"help": "ImageMol checkpoint path"}
    )
    num_image_tokens: int = field(default=4, metadata={"help": "Number of image tokens"})
    projection_gain: float = field(default=1.0, metadata={"help": "Projection initialization gain"})
    use_cross_attention: bool = field(default=True, metadata={"help": "Use cross-attention fusion"})
    image_feature_weight: float = field(default=1.5, metadata={"help": "Image feature scaling weight"})

    use_scaffold_loss: bool = field(default=True, metadata={"help": "Enable REINFORCE scaffold loss"})
    scaffold_loss_weight: float = field(default=0.3, metadata={"help": "Weight for scaffold loss (recommended: 0.1-0.5)"})
    substructure_weight: float = field(default=0.7, metadata={"help": "Weight for substructure match (recommended: 0.6-0.8)"})
    scaffold_loss_freq: int = field(default=5, metadata={"help": "Compute scaffold loss every N steps (recommended: 3-10)"})
    scaffold_loss_start_step: int = field(default=0, metadata={"help": "Start scaffold loss at step N"})


@dataclass
class DataArguments:
    train_file: str = field(metadata={"help": "Training data file path"})
    validation_file: Optional[str] = field(default=None)
    image_dir: str = field(default="datasets", metadata={"help": "Image directory"})
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)
    preprocessing_num_workers: Optional[int] = field(default=None)
    scaffold_mapping_file: Optional[str] = field(
        default="data/scaffold_mapping.json",
        metadata={"help": "Scaffold SMILES mapping file"}
    )


@dataclass
class MyTrainingArguments(TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, MyTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info(f"=" * 80)
    logger.info(f"REINFORCE Scaffold-Aware Training Configuration")
    logger.info(f"=" * 80)
    logger.info(f"  Image tokens: {model_args.num_image_tokens}")
    logger.info(f"  Projection gain: {model_args.projection_gain}")
    logger.info(f"  Cross-attention: {model_args.use_cross_attention}")
    logger.info(f"  Image weight: {model_args.image_feature_weight}")
    logger.info(f"")
    logger.info(f"  REINFORCE Scaffold Loss: {model_args.use_scaffold_loss}")
    logger.info(f"     - Weight: {model_args.scaffold_loss_weight}")
    logger.info(f"     - Substructure weight: {model_args.substructure_weight}")
    logger.info(f"     - Frequency: every {model_args.scaffold_loss_freq} steps")
    logger.info(f"     - Start step: {model_args.scaffold_loss_start_step}")
    logger.info(f"=" * 80)

    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name_or_path or model_args.model_name_or_path,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading multi-modal model...")
    model = MultiModalDrugAssist(
        llm_model_name_or_path=model_args.model_name_or_path,
        imagemol_checkpoint_path=model_args.imagemol_checkpoint_path,
        num_image_tokens=model_args.num_image_tokens,
        projection_gain=model_args.projection_gain,
        use_cross_attention=model_args.use_cross_attention,
        image_feature_weight=model_args.image_feature_weight,
        freeze_vision_model=True,
        freeze_llm=True,
        device_map="auto"
    )

    logger.info("Applying LoRA...")

    if hasattr(model, 'peft_config'):
        logger.warning("Model already has peft_config, removing it before applying LoRA")
        delattr(model, 'peft_config')

    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["projection", "fusion_norm", "cross_attention_fusion"] if model_args.use_cross_attention else ["projection", "fusion_norm"]
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    logger.info("Loading dataset...")
    train_dataset = build_multimodal_dataset(
        data_path=data_args.train_file,
        image_dir=data_args.image_dir,
        tokenizer=tokenizer,
        image_processor=model.image_processor,
        model_max_length=training_args.model_max_length,
        scaffold_mapping_file=data_args.scaffold_mapping_file
    )

    eval_dataset = None
    if data_args.validation_file:
        eval_dataset = build_multimodal_dataset(
            data_path=data_args.validation_file,
            image_dir=data_args.image_dir,
            tokenizer=tokenizer,
            image_processor=model.image_processor,
            model_max_length=training_args.model_max_length,
            scaffold_mapping_file=data_args.scaffold_mapping_file
        )

    data_collator = MultiModalDataCollator(
        tokenizer=tokenizer,
        image_processor=model.image_processor
    )

    scaffold_loss_fn = None
    if model_args.use_scaffold_loss:
        scaffold_loss_fn = ScaffoldReinforceLoss(
            weight=model_args.scaffold_loss_weight,
            use_hybrid=True,
            substructure_weight=model_args.substructure_weight,
            baseline_alpha=0.9,
            normalize_rewards=True
        )
        logger.info("REINFORCE scaffold loss initialized")

    trainer = ScaffoldAwareTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        scaffold_loss_fn=scaffold_loss_fn,
        scaffold_loss_freq=model_args.scaffold_loss_freq,
        scaffold_loss_start_step=model_args.scaffold_loss_start_step,
        generation_config={
            'max_new_tokens': 64,
            'do_sample': True,
            'temperature': 1.0,
            'num_beams': 1,
        },
        callbacks=[
            KeepVisionModelEvalCallback(),
            SavePeftModelCallback()
        ]
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Training completed!")


if __name__ == "__main__":
    main()
