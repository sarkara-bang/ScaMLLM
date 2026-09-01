"""
Enhanced MultiModal Model
"""
import torch
import torch.nn as nn
from transformers import LlamaForCausalLM
from typing import Optional, Dict

from utils.imagemol_vision import ImageMolVisionModel, ImageMolImageProcessor


class ScaffoldAwareFusion(nn.Module):
    """Cross-attention fusion module"""
    def __init__(self, hidden_size=4096, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )

        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, text_embeds, image_features):
        """Fuse text and image features"""
        attended, attn_weights = self.cross_attention(
            query=text_embeds,
            key=image_features,
            value=image_features
        )

        concat = torch.cat([text_embeds, attended], dim=-1)
        gate_values = self.gate(concat)

        fused = text_embeds * (1 - gate_values) + attended * gate_values

        return self.norm(fused), attn_weights


class EnhancedMultiModalDrugAssist(nn.Module):
    """Multi-modal model with image-text fusion"""
    def __init__(
        self,
        llm_model_name_or_path: Optional[str] = None,
        imagemol_checkpoint_path: str = None,
        num_image_tokens: int = 4,
        projection_gain: float = 1.0,
        use_cross_attention: bool = True,
        image_feature_weight: float = 1.0,
        freeze_vision_model: bool = True,
        freeze_llm: bool = True,
        device_map: Optional[Dict[str, int]] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.num_image_tokens = num_image_tokens
        self.use_cross_attention = use_cross_attention
        self.image_feature_weight = image_feature_weight

        # Load LLM
        if llm_model_name_or_path is not None:
            self.llm = LlamaForCausalLM.from_pretrained(
                llm_model_name_or_path,
                device_map=device_map,
                torch_dtype=torch_dtype,
            )
            self.config = self.llm.config
            self.config.model_type = "enhanced_multimodal_llama"

            if hasattr(self.llm, 'generation_config'):
                self.generation_config = self.llm.generation_config
            else:
                from transformers import GenerationConfig
                self.generation_config = GenerationConfig.from_model_config(self.config)

            llm_hidden_size = self.llm.config.hidden_size
        else:
            llm_hidden_size = 4096
            self.llm = None
        print("\n[Enhanced Model] Loading molecular vision encoder...")

        if imagemol_checkpoint_path is None:
            imagemol_checkpoint_path = '/root/autodl-tmp/multimodel/models/ImageMol/ImageMol.pth.tar'

        self.vision_model = ImageMolVisionModel(normalize_features=True)
        self.vision_model.load_pretrained_weights(imagemol_checkpoint_path)
        self.image_processor = ImageMolImageProcessor()

        print(f"[Enhanced Model] Vision encoder loaded")

        # PEFT configuration
        modules_to_save = ["projection", "fusion_norm"]
        if use_cross_attention:
            modules_to_save.append("cross_attention_fusion")

        self.peft_config = {
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "modules_to_save": modules_to_save
        }

        # Projection layer
        vision_hidden_size = 512
        self.projection = nn.Linear(vision_hidden_size, llm_hidden_size)

        nn.init.xavier_uniform_(self.projection.weight, gain=projection_gain)
        nn.init.zeros_(self.projection.bias)
        print(f"[Projection] Initialized with gain={projection_gain}")

        self.fusion_norm = nn.LayerNorm(llm_hidden_size)

        # Cross-attention fusion
        if use_cross_attention:
            self.cross_attention_fusion = ScaffoldAwareFusion(
                hidden_size=llm_hidden_size,
                num_heads=8
            )
            print(f"[Fusion] Added cross-attention fusion module")
        else:
            self.cross_attention_fusion = None

        self.image_scale = nn.Parameter(torch.tensor(image_feature_weight))
        print(f"[Fusion] Image feature weight: {image_feature_weight}")
        print(f"[Fusion] Number of image tokens: {num_image_tokens}")

        # Freeze/unfreeze
        if freeze_vision_model:
            for param in self.vision_model.parameters():
                param.requires_grad = False
            print(f"[Vision] Vision encoder frozen")

        if freeze_llm and self.llm is not None:
            for param in self.llm.parameters():
                param.requires_grad = False
            print(f"[LLM] Language model frozen")

        for param in self.projection.parameters():
            param.requires_grad = True
        for param in self.fusion_norm.parameters():
            param.requires_grad = True
        if self.cross_attention_fusion is not None:
            for param in self.cross_attention_fusion.parameters():
                param.requires_grad = True

        print(f"\n{'='*60}")
        print(f"Multi-Modal Architecture:")
        print(f"  Vision: ImageMol ResNet18 (512d)")
        print(f"  Projection: Linear (512 -> {llm_hidden_size})")
        print(f"  Image Tokens: {num_image_tokens}")
        print(f"  Fusion: {'Cross-Attention' if use_cross_attention else 'Concat'}")
        print(f"  Language: LLaMA ({llm_hidden_size}d)")
        print(f"{'='*60}\n")

    def encode_images(self, pixel_values: torch.FloatTensor) -> torch.FloatTensor:
        """Encode images to feature vectors"""
        vision_outputs = self.vision_model(pixel_values)
        image_features = vision_outputs.last_hidden_state

        projected = self.projection(image_features)

        if self.num_image_tokens > 1:
            projected = projected.repeat(1, self.num_image_tokens, 1)

        projected = projected * self.image_scale

        return projected

    def process_images(self, images):
        """Process images using ImageMol processor"""
        if isinstance(images, torch.Tensor):
            return images
        return self.image_processor(images=images, return_tensors="pt")['pixel_values']

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_features: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_attention_weights: bool = False,
        **kwargs
    ):
        """Forward pass"""
        if not hasattr(self, 'llm') or self.llm is None:
            raise ValueError("LLM not initialized")

        if pixel_values is not None and image_features is None:
            image_features = self.encode_images(pixel_values)

        attn_weights = None

        if image_features is not None:
            inputs_embeds = self.llm.get_input_embeddings()(input_ids)

            batch_size = inputs_embeds.shape[0]
            img_seq_len = image_features.shape[1]

            if image_features.dtype != inputs_embeds.dtype:
                image_features = image_features.to(inputs_embeds.dtype)

            image_features = self.fusion_norm(image_features)

            if self.use_cross_attention and self.cross_attention_fusion is not None:
                fused_text, attn_weights = self.cross_attention_fusion(
                    inputs_embeds, image_features
                )
                new_embeds = torch.cat([image_features, fused_text], dim=1)
            else:
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

        if return_attention_weights:
            outputs.attention_weights = attn_weights

        return outputs

    def prepare_inputs_for_generation(
        self, input_ids, attention_mask=None, image_features=None, **kwargs
    ):
        """Prepare inputs for generation"""
        model_inputs = self.llm.prepare_inputs_for_generation(
            input_ids, attention_mask=attention_mask, **kwargs
        )

        if image_features is not None:
            model_inputs["image_features"] = image_features

        return model_inputs

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing"""
        if hasattr(self.llm, 'gradient_checkpointing_enable'):
            self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing"""
        if hasattr(self.llm, 'gradient_checkpointing_disable'):
            self.llm.gradient_checkpointing_disable()

    def generate(
        self,
        input_ids: torch.LongTensor,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **generate_kwargs
    ):
        """Generate text"""
        if pixel_values is not None and image_features is None:
            image_features = self.encode_images(pixel_values)

        if image_features is not None:
            inputs_embeds = self.llm.get_input_embeddings()(input_ids)
            batch_size = inputs_embeds.shape[0]
            img_seq_len = image_features.shape[1]

            if image_features.dtype != inputs_embeds.dtype:
                image_features = image_features.to(inputs_embeds.dtype)

            image_features = self.fusion_norm(image_features)

            if self.use_cross_attention and self.cross_attention_fusion is not None:
                fused_text, _ = self.cross_attention_fusion(
                    inputs_embeds, image_features
                )
                new_embeds = torch.cat([image_features, fused_text], dim=1)
            else:
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
        if self.cross_attention_fusion is not None:
            self.cross_attention_fusion = self.cross_attention_fusion.to(*args, **kwargs)
        return super().to(*args, **kwargs)
