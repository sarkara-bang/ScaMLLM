import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaConfig
from typing import Optional, Tuple, List, Dict, Any, Union
import sys
import os
sys.path.insert(0, '/root/autodl-tmp/multimodel')
from utils.imagemol_vision import ImageMolVisionModel, ImageMolImageProcessor, load_imagemol_vision_model

class MultiModalDrugAssist_ImageMol(nn.Module):
    
    def __init__(
        self,
        llm_model_name_or_path: Optional[str] = None,
        imagemol_checkpoint_path: str = None,
        normalize_features: bool = True,  
        projection_dim: int = 512, 
        use_multiscale_vision: bool = False,
        freeze_vision_model: bool = True,
        freeze_llm: bool = True,
        device_map: Optional[Dict[str, int]] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        if llm_model_name_or_path is not None:
            self.llm = LlamaForCausalLM.from_pretrained(
                llm_model_name_or_path,
                device_map=device_map,
                torch_dtype=torch_dtype,
            )
            self.config = self.llm.config
            self.config.model_type = "multimodal_llama"

            if hasattr(self.llm, 'generation_config'):
                self.generation_config = self.llm.generation_config
            else:
                from transformers import GenerationConfig
                self.generation_config = GenerationConfig.from_model_config(self.config)

            llm_hidden_size = self.llm.config.hidden_size
        else:
            llm_hidden_size = 4096 
            self.llm = None 


        if imagemol_checkpoint_path is None:
            imagemol_checkpoint_path = '/root/autodl-tmp/DrugAssist-main/model/ImageMol/ImageMol.pth.tar'

        self.vision_model = ImageMolVisionModel(normalize_features=normalize_features)
        self.vision_model.load_pretrained_weights(imagemol_checkpoint_path)

        if use_multiscale_vision:
            from utils.multiscale_vision import upgrade_to_multiscale
            self.vision_model = upgrade_to_multiscale(self.vision_model)

        self.image_processor = ImageMolImageProcessor()

        self.peft_config = {
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "modules_to_save": ["projection", "fusion_norm"] 
        }
        vision_hidden_size = 512 

        self.projection = nn.Linear(vision_hidden_size, llm_hidden_size)

        self.fusion_norm = nn.LayerNorm(llm_hidden_size)
        nn.init.xavier_uniform_(self.projection.weight, gain=0.1)
        nn.init.zeros_(self.projection.bias)

        if freeze_vision_model:
            for param in self.vision_model.parameters():
                param.requires_grad = False

        if freeze_llm and self.llm is not None:
            for param in self.llm.parameters():
                param.requires_grad = False

        for param in self.projection.parameters():
            param.requires_grad = True
        for param in self.fusion_norm.parameters():
            param.requires_grad = True

    def encode_images(self, pixel_values: torch.FloatTensor) -> torch.FloatTensor:
    
        vision_outputs = self.vision_model(pixel_values)
        image_features = vision_outputs.last_hidden_state 
        projected_features = self.projection(image_features) 
        return projected_features

    def process_images(self, images):
        if isinstance(images, torch.Tensor):
            return images

        return self.image_processor(images=images, return_tensors="pt")['pixel_values']

    def prepare_inputs_for_generation(
        self, input_ids, attention_mask=None, image_features=None, **kwargs
    ):
        model_inputs = self.llm.prepare_inputs_for_generation(
            input_ids, attention_mask=attention_mask, **kwargs
        )
        if image_features is not None:
            model_inputs["image_features"] = image_features
        return model_inputs

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_features: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        if not hasattr(self, 'llm') or self.llm is None:
            raise ValueError("LLM model not found. Please provide a valid model checkpoint path.")
        if pixel_values is not None and image_features is None:
            image_features = self.encode_images(pixel_values)

        if image_features is not None:
            inputs_embeds = self.llm.get_input_embeddings()(input_ids)
            batch_size = inputs_embeds.shape[0]
            img_seq_len = image_features.shape[1]

            if image_features.dtype != inputs_embeds.dtype:
                image_features = image_features.to(inputs_embeds.dtype)
            if torch.isnan(image_features).any() or torch.isinf(image_features).any():
                print(f" Image features (before norm) mean: {image_features.mean().item():.6f}")
                print(f"   Image features (before norm) std: {image_features.std().item():.6f}")

            image_features = self.fusion_norm(image_features)
            new_embeds = torch.cat([image_features, inputs_embeds], dim=1)
            if attention_mask is not None:
                image_attention_mask = torch.ones(
                    (batch_size, img_seq_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )
                new_attention_mask = torch.cat([image_attention_mask, attention_mask], dim=1)
            else:
                new_attention_mask = None
            if labels is not None:
                image_labels = torch.full(
                    (batch_size, img_seq_len),
                    fill_value=-100,
                    dtype=labels.dtype,
                    device=labels.device
                )
                new_labels = torch.cat([image_labels, labels], dim=1)
            else:
                new_labels = None
            filtered_kwargs = {k: v for k, v in kwargs.items()
                             if k not in ['inputs_embeds', 'attention_mask', 'labels', 'input_ids']}
            outputs = self.llm(
                inputs_embeds=new_embeds,
                attention_mask=new_attention_mask,
                labels=new_labels,
                **filtered_kwargs
            )
        else:
            filtered_kwargs = {k: v for k, v in kwargs.items()
                             if k not in ['input_ids', 'attention_mask', 'labels']}
            outputs = self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **filtered_kwargs
            )
        return outputs

    def generate(
        self,
        input_ids: torch.LongTensor,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **generate_kwargs
    ):
        if pixel_values is not None and image_features is None:
            image_features = self.encode_images(pixel_values)
        if image_features is not None:
            inputs_embeds = self.llm.get_input_embeddings()(input_ids)
            batch_size = inputs_embeds.shape[0]
            img_seq_len = image_features.shape[1]

            if image_features.dtype != inputs_embeds.dtype:
                image_features = image_features.to(inputs_embeds.dtype)
            image_features = self.fusion_norm(image_features)

            new_embeds = torch.cat([image_features, inputs_embeds], dim=1)

            if attention_mask is not None:
                image_attention_mask = torch.ones(
                    (batch_size, img_seq_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )
                new_attention_mask = torch.cat([image_attention_mask, attention_mask], dim=1)
            else:
                new_attention_mask = None

            filtered_kwargs = {k: v for k, v in generate_kwargs.items()
                             if k not in ['inputs_embeds', 'attention_mask', 'input_ids']}

            generated_ids = self.llm.generate(
                inputs_embeds=new_embeds,
                attention_mask=new_attention_mask,
                **filtered_kwargs
            )

            full_output = torch.cat([input_ids, generated_ids], dim=1)
            return full_output
        else:
            filtered_kwargs = {k: v for k, v in generate_kwargs.items()
                             if k not in ['input_ids', 'attention_mask']}
            return self.llm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **filtered_kwargs
            )
    @classmethod
    def from_pretrained(cls, pretrained_model_path, *args, **kwargs):
        llm_model_path = pretrained_model_path
        imagemol_checkpoint = kwargs.pop('imagemol_checkpoint_path', None)
        use_multiscale_vision = kwargs.pop('use_multiscale_vision', False)

        config_kwargs = kwargs.pop('config_kwargs', {})
        ignore_mismatched_sizes = kwargs.pop('ignore_mismatched_sizes', False)
        local_files_only = kwargs.pop('local_files_only', False)

        try:
            try:
                llm = LlamaForCausalLM.from_pretrained(
                    llm_model_path,
                    device_map=kwargs.get('device_map'),
                    torch_dtype=kwargs.get('torch_dtype'),
                    ignore_mismatched_sizes=ignore_mismatched_sizes,
                    local_files_only=local_files_only,
                    **config_kwargs
                )
            except Exception as e:
                print(f"erro: {e}")
                if "config.json" in str(e):
                    config = LlamaConfig(
                        vocab_size=32000,
                        hidden_size=4096,
                        intermediate_size=11008,
                        num_hidden_layers=32,
                        num_attention_heads=32,
                        max_position_embeddings=2048,
                        **config_kwargs
                    )
                    llm = LlamaForCausalLM.from_pretrained(
                        llm_model_path,
                        config=config,
                        device_map=kwargs.get('device_map'),
                        torch_dtype=kwargs.get('torch_dtype'),
                        ignore_mismatched_sizes=ignore_mismatched_sizes,
                        local_files_only=local_files_only
                    )
                else:
                    raise e

            model = cls(
                llm_model_name_or_path=None,
                imagemol_checkpoint_path=imagemol_checkpoint,
                use_multiscale_vision=use_multiscale_vision,
                *args, **kwargs
            )
            model.llm = llm
            model.config = llm.config

            if hasattr(llm, 'generation_config'):
                model.generation_config = llm.generation_config
            else:
                from transformers import GenerationConfig
                model.generation_config = GenerationConfig.from_model_config(model.config)

        except Exception as e:
            print(f"lose: {e}")
            model = cls(
                llm_model_name_or_path=llm_model_path,
                imagemol_checkpoint_path=imagemol_checkpoint,
                use_multiscale_vision=use_multiscale_vision,
                *args, **kwargs
            )
        return model

    # PEFT compatibility methods
    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.llm.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.llm.get_output_embeddings()

    def set_output_embeddings(self, value):
        self.llm.set_output_embeddings(value)

    def resize_token_embeddings(self, new_num_tokens=None):
        return self.llm.resize_token_embeddings(new_num_tokens)

    @property
    def dtype(self):
        return self.llm.dtype

    def to(self, *args, **kwargs):
        self.llm = self.llm.to(*args, **kwargs)
        self.vision_model = self.vision_model.to(*args, **kwargs)
        self.projection = self.projection.to(*args, **kwargs)
        self.fusion_norm = self.fusion_norm.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def half(self):
        self.llm = self.llm.half()
        self.vision_model = self.vision_model.half()
        self.projection = self.projection.half()
        self.fusion_norm = self.fusion_norm.half()
        return self

    def float(self):
        self.llm = self.llm.float()
        self.vision_model = self.vision_model.float()
        self.projection = self.projection.float()
        self.fusion_norm = self.fusion_norm.float()
        return self
