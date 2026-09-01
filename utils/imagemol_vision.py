"""
ImageMol-based vision encoder
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np


class BasicBlock(nn.Module):
    """ResNet BasicBlock for ImageMol"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ImageMolEncoder(nn.Module):
    """
    ImageMol ResNet18 encoder for molecular images
    Output: [batch_size, 512] feature vector
    """
    def __init__(self):
        super(ImageMolEncoder, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.hidden_size = 512

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: [batch_size, 3, 224, 224] image tensor
        Returns:
            [batch_size, 512] feature vector
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return out


class ImageMolVisionModel(nn.Module):
    """
    Wrapper to make ImageMol compatible with ViT interface
    Includes feature normalization for stable multimodal fusion
    """
    def __init__(self, normalize_features=True):
        super().__init__()
        self.encoder = ImageMolEncoder()

        # Feature normalization layer (optional but recommended)
        self.normalize_features = normalize_features
        if normalize_features:
            self.feature_norm = nn.LayerNorm(512)

        # Config for compatibility with transformers API
        from types import SimpleNamespace
        self.config = SimpleNamespace(hidden_size=512)

        def _frozen_train(self, mode=True):
            """Override train() to always stay in eval mode"""
            return super(type(self), self).train(False)  

        # Apply to all BatchNorm layers
        for module in self.encoder.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.train = lambda mode=True, m=module: _frozen_train(m, mode)
                module.eval()  # Initially set to eval mode
                module.track_running_stats = False  # Disable running stats update

                if module.running_var is not None:
                    module.running_var.clamp_(min=0.001, max=100.0)
                if module.running_mean is not None:
                    module.running_mean.clamp_(min=-10.0, max=10.0)

    def forward(self, pixel_values):
        """
        Args:
            pixel_values: [batch_size, 3, 224, 224]
        Returns:
            Output object with last_hidden_state attribute
        """
        # Extract features
        features = self.encoder(pixel_values)  # [batch_size, 512]

        # Normalize features if enabled
        if self.normalize_features:
            features = self.feature_norm(features)

        # Reshape to [batch_size, 1, 512] to match ViT's sequence format
        # ViT outputs [batch_size, num_patches, hidden_size]
        features = features.unsqueeze(1)  # [batch_size, 1, 512]

        # Return in ViT-compatible format
        from types import SimpleNamespace
        return SimpleNamespace(last_hidden_state=features)

    def load_pretrained_weights(self, checkpoint_path):
        """Load ImageMol pretrained weights"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['state_dict']

        # Map checkpoint keys to model keys
        model_dict = self.encoder.state_dict()
        mapped_dict = {}

        for ckpt_key, ckpt_value in state_dict.items():
            if ckpt_key.startswith('embedding_layer.'):
                suffix = ckpt_key.replace('embedding_layer.', '')

                # Map layer indices
                if suffix.startswith('0.'):
                    new_key = 'conv1.' + suffix[2:]
                elif suffix.startswith('1.'):
                    new_key = 'bn1.' + suffix[2:]
                elif suffix.startswith('4.'):
                    layer_suffix = suffix[2:]
                    if 'downsample' in layer_suffix:
                        layer_suffix = layer_suffix.replace('downsample', 'shortcut')
                    new_key = 'layer1.' + layer_suffix
                elif suffix.startswith('5.'):
                    layer_suffix = suffix[2:]
                    if 'downsample' in layer_suffix:
                        layer_suffix = layer_suffix.replace('downsample', 'shortcut')
                    new_key = 'layer2.' + layer_suffix
                elif suffix.startswith('6.'):
                    layer_suffix = suffix[2:]
                    if 'downsample' in layer_suffix:
                        layer_suffix = layer_suffix.replace('downsample', 'shortcut')
                    new_key = 'layer3.' + layer_suffix
                elif suffix.startswith('7.'):
                    layer_suffix = suffix[2:]
                    if 'downsample' in layer_suffix:
                        layer_suffix = layer_suffix.replace('downsample', 'shortcut')
                    new_key = 'layer4.' + layer_suffix
                else:
                    continue

                if new_key in model_dict:
                    mapped_dict[new_key] = ckpt_value

        # Load weights
        self.encoder.load_state_dict(mapped_dict, strict=False)
        print(f"Loaded {len(mapped_dict)}/{len(model_dict)} parameters from ImageMol checkpoint")

        for module in self.encoder.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                if module.running_var is not None:
                    module.running_var.clamp_(min=0.001, max=100.0)
                if module.running_mean is not None:
                    module.running_mean.clamp_(min=-10.0, max=10.0)

        return len(mapped_dict)


class ImageMolImageProcessor:
    """
    Image processor for ImageMol (compatible with ViT processor interface)
    Includes proper normalization - NO torchvision dependency
    """
    def __init__(self):
        # ImageNet normalization constants
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def _resize_and_crop(self, img, size=224):
        """Resize to 256 and center crop to 224"""
        # Resize
        w, h = img.size
        if w < h:
            new_w = 256
            new_h = int(256 * h / w)
        else:
            new_h = 256
            new_w = int(256 * w / h)

        img = img.resize((new_w, new_h), Image.BILINEAR)

        # Center crop
        w, h = img.size
        left = (w - size) // 2
        top = (h - size) // 2
        img = img.crop((left, top, left + size, top + size))

        return img

    def _normalize(self, img_array):
        """Apply ImageNet normalization"""
        # img_array is [H, W, 3] in range [0, 1]
        img_array = (img_array - self.mean) / self.std
        return img_array

    def __call__(self, images, return_tensors="pt"):
        """
        Process images
        Args:
            images: PIL Image or list of PIL Images
            return_tensors: "pt" for PyTorch tensors
        Returns:
            Dictionary with 'pixel_values' key
        """
        if not isinstance(images, list):
            images = [images]

        # Convert to PIL if needed
        processed = []
        for img in images:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            elif not isinstance(img, Image.Image):
                raise ValueError(f"Unsupported image type: {type(img)}")

            # Ensure RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Resize and crop
            img = self._resize_and_crop(img, size=224)

            # Convert to numpy array [0, 1]
            img_array = np.array(img).astype(np.float32) / 255.0

            # Normalize
            img_array = self._normalize(img_array)

            # Convert to tensor [C, H, W]
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()

            processed.append(img_tensor)

        # Stack into batch
        if return_tensors == "pt":
            pixel_values = torch.stack(processed)
            return {"pixel_values": pixel_values}
        else:
            return {"pixel_values": processed}


def load_imagemol_vision_model(checkpoint_path, normalize_features=True, device='cpu'):
    """
    Load ImageMol vision model with pretrained weights

    Args:
        checkpoint_path: Path to ImageMol.pth.tar
        normalize_features: Whether to normalize features before projection
        device: Device to load model on

    Returns:
        vision_model: ImageMolVisionModel instance
        image_processor: ImageMolImageProcessor instance
    """
    # Create model
    vision_model = ImageMolVisionModel(normalize_features=normalize_features)

    # Load pretrained weights
    num_loaded = vision_model.load_pretrained_weights(checkpoint_path)

    if num_loaded < 100:
        print(f"Warning: Only loaded {num_loaded} parameters. Expected ~120.")

    # Move to device
    vision_model = vision_model.to(device)
    vision_model.eval()

    # Create processor
    image_processor = ImageMolImageProcessor()

    print(f"ImageMol vision model loaded successfully:")
    print(f"  - Parameters loaded: {num_loaded}/120")
    print(f"  - Feature normalization: {normalize_features}")
    print(f"  - Output dimension: 512")
    print(f"  - Device: {device}")

    return vision_model, image_processor


if __name__ == '__main__':
    # Test the model
    print("Testing ImageMol vision encoder...")

    # Create model
    vision_model = ImageMolVisionModel(normalize_features=True)

    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    output = vision_model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.last_hidden_state.shape}")
    print(f"Output mean: {output.last_hidden_state.mean():.4f}")
    print(f"Output std: {output.last_hidden_state.std():.4f}")

    # Test image processor
    processor = ImageMolImageProcessor()
    test_img = Image.new('RGB', (256, 256), color='white')
    processed = processor([test_img], return_tensors="pt")

    print(f"\nProcessor test:")
    print(f"  Input: PIL Image (256x256)")
    print(f"  Output: {processed['pixel_values'].shape}")
    print(f"  Value range: [{processed['pixel_values'].min():.2f}, {processed['pixel_values'].max():.2f}]")

