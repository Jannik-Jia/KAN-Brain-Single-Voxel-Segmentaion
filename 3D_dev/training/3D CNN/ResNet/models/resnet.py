"""
ResNet-50 Architecture for MRI Brain Voxel Classification

Based on the detailed specification:
- ~50M parameters ResNet for 7×7 patch classification
- 351 input channels (multi-modal MRI features) 
- 102 output classes (brain regions)
- 3-4-6-3 Bottleneck structure with base_width=104
- Expand-first Stem (3×3/s1/p1, out=512)
- ResNet-B downsampling (stride handled by 3×3 conv)

Spatial progression: 7 → 7 → 4 → 2 → 1
Channel progression: 512 → 256 → 512 → 1024 → 2048 → 102
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Any, Callable, Union, List, Optional


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    """
    Bottleneck block for ResNet with configurable width
    
    Architecture:
    1×1 (in → width) → 3×3 (width → width, stride) → 1×1 (width → expansion*planes)
    
    For ResNet-B: stride is applied to the 3×3 conv (not the 1×1)
    """
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        # Calculate width based on base_width scaling
        width = int(planes * (base_width / 64.0)) * groups
        
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)  # ResNet-B: stride here
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        # 1×1 conv: channel mixing
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 3×3 conv: spatial processing with potential stride
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # 1×1 conv: expansion to final channels
        out = self.conv3(out)
        out = self.bn3(out)

        # Shortcut connection
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class MRIResNet(nn.Module):
    """
    ResNet for MRI Brain Voxel Classification
    
    Optimized for 7×7 patch input with 351 channels
    Target: ~50M parameters for better handling of imbalanced classes
    """

    def __init__(
        self,
        block: Type[Bottleneck],
        layers: List[int],
        num_classes: int = 102,
        input_channels: int = 351,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        stem_channels: int = 512,  # Expand-first stem
    ) -> None:
        super().__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = stem_channels  # 512 for expand-first
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # Each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group

        # Expand-first Stem: 3×3/s1/p1, 351 → 512, NO MaxPool
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, stem_channels, kernel_size=3, 
                     stride=1, padding=1, bias=False),
            norm_layer(stem_channels),
            nn.ReLU(inplace=True),
        )

        # Stage 1: 3 blocks, stride=1, output=256
        self.layer1 = self._make_layer(block, 64, layers[0])
        
        # Stage 2: 4 blocks, stride=2, output=512  
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                      dilate=replace_stride_with_dilation[0])
        
        # Stage 3: 6 blocks, stride=2, output=1024
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                      dilate=replace_stride_with_dilation[1])
        
        # Stage 4: 3 blocks, stride=2, output=2048
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                      dilate=replace_stride_with_dilation[2])

        # Global Average Pooling + Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Bottleneck],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        
        # Downsample if stride != 1 or channel mismatch
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        # First block handles stride/downsample
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, 
                self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        # Stem: 351 × 7 × 7 → 512 × 7 × 7
        x = self.stem(x)

        # Stage 1: 512 × 7 × 7 → 256 × 7 × 7
        x = self.layer1(x)
        
        # Stage 2: 256 × 7 × 7 → 512 × 4 × 4 (stride=2)
        x = self.layer2(x)
        
        # Stage 3: 512 × 4 × 4 → 1024 × 2 × 2 (stride=2)
        x = self.layer3(x)
        
        # Stage 4: 1024 × 2 × 2 → 2048 × 1 × 1 (stride=2)
        x = self.layer4(x)

        # Global Average Pooling: 2048 × 1 × 1 → 2048
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Classification: 2048 → 102
        x = self.fc(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)


def _mri_resnet(
    block: Type[Bottleneck],
    layers: List[int],
    **kwargs: Any,
) -> MRIResNet:
    """Generic ResNet builder"""
    model = MRIResNet(block, layers, **kwargs)
    return model


def mri_resnet50(
    input_channels: int = 351,
    num_classes: int = 102,
    base_width: int = 104,  # Key parameter for ~50M params
    **kwargs: Any,
) -> MRIResNet:
    """
    MRI-optimized ResNet-50 with ~50M parameters
    
    Architecture: 3-4-6-3 Bottleneck blocks
    Base width: 104 (vs standard 64) for increased channel capacity
    Stem: 3×3/s1/p1, 351→512 channels (expand-first)
    
    Args:
        input_channels: Number of input feature channels (351 for MRI)
        num_classes: Number of output classes (102 brain regions)
        base_width: Width scaling factor (104 for ~50M params)
        **kwargs: Additional arguments
    
    Returns:
        MRIResNet model with ~49.9M parameters
    """
    return _mri_resnet(
        Bottleneck, 
        [3, 4, 6, 3], 
        input_channels=input_channels,
        num_classes=num_classes,
        width_per_group=base_width,
        **kwargs
    )


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_info(model: nn.Module, input_size: tuple = (1, 351, 7, 7)) -> dict:
    """Get model information including parameter count and output shapes"""
    
    # Count parameters
    total_params = count_parameters(model)
    
    # Get intermediate shapes (requires forward pass)
    model.eval()
    with torch.no_grad():
        x = torch.randn(input_size)
        
        shapes = {}
        shapes['input'] = x.shape
        
        # Stem
        x = model.stem(x)
        shapes['stem'] = x.shape
        
        # Stages
        x = model.layer1(x)
        shapes['stage1'] = x.shape
        
        x = model.layer2(x)
        shapes['stage2'] = x.shape
        
        x = model.layer3(x)
        shapes['stage3'] = x.shape
        
        x = model.layer4(x)
        shapes['stage4'] = x.shape
        
        # GAP + FC
        x = model.avgpool(x)
        x = torch.flatten(x, 1)
        shapes['gap'] = x.shape
        
        x = model.fc(x)
        shapes['output'] = x.shape
    
    return {
        'total_parameters': total_params,
        'shapes': shapes,
        'parameter_size_mb': total_params * 4 / (1024**2),  # Assuming float32
    }


if __name__ == "__main__":
    # Test the model
    model = mri_resnet50()
    info = get_model_info(model)
    
    print("=== MRI ResNet-50 Model Information ===")
    print(f"Total Parameters: {info['total_parameters']:,}")
    print(f"Model Size: {info['parameter_size_mb']:.2f} MB")
    print(f"Target: ~50M parameters")
    
    print("\n=== Shape Flow ===")
    for stage, shape in info['shapes'].items():
        print(f"{stage:>8}: {str(tuple(shape)):>20}")
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        x = torch.randn(4, 351, 7, 7)  # Batch of 4
        output = model(x)
        print(f"\nForward pass successful!")
        print(f"Input shape: {x.shape}")  
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")