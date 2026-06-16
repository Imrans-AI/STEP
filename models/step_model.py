"""
STEP Model with Your Innovations
- ResNet backbone (from DAISY)
- Dual Attention (from DAISY)
- Structure-aware weighting (YOUR innovation!)
- Global features guide attention (from DAISY)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# ============================================================================
# ResNet Components (from DAISY)
# ============================================================================

class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=False)
    
    def forward(self, x):
        identity = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = self.relu(out + identity)
        return out

class ResNet(nn.Module):
    """ResNet backbone for physicochemical maps"""
    def __init__(self, block, num_blocks, in_channels=5):
        super().__init__()
        self.in_channels = 64
        
        # Initial conv
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, 
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        
        # ResNet layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=1)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=1)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=1)
        
    def _make_layer(self, block, out_channels, num_blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

# ============================================================================
# Attention Components (from DAISY)
# ============================================================================

class PAM_Module(nn.Module):
    """Position Attention Module"""
    def __init__(self, in_channels):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        batch, C, H, W = x.size()
        
        # Query, Key, Value
        proj_query = self.query_conv(x).view(batch, -1, H * W).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch, -1, H * W)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        
        proj_value = self.value_conv(x).view(batch, -1, H * W)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch, C, H, W)
        
        out = self.gamma * out + x
        return out

class CAM_Module(nn.Module):
    """Channel Attention Module"""
    def __init__(self, in_channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        batch, C, H, W = x.size()
        
        proj_query = x.view(batch, C, -1)
        proj_key = x.view(batch, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        
        attention = self.softmax(energy)
        proj_value = x.view(batch, C, -1)
        
        out = torch.bmm(attention, proj_value)
        out = out.view(batch, C, H, W)
        
        out = self.gamma * out + x
        return out

class DANetHead(nn.Module):
    """Dual Attention Network Head"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        inter_channels = in_channels // 4
        
        # Position attention branch
        self.conv_pa = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=False)
        )
        self.pa = PAM_Module(inter_channels)
        self.conv_pa_out = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=False)
        )
        
        # Channel attention branch
        self.conv_ca = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=False)
        )
        self.ca = CAM_Module(inter_channels)
        self.conv_ca_out = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=False)
        )
        
        # Final output
        self.conv_out = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(inter_channels, out_channels, 1)
        )
    
    def forward(self, x):
        # Position attention
        feat_pa = self.conv_pa(x)
        feat_pa = self.pa(feat_pa)
        feat_pa = self.conv_pa_out(feat_pa)
        
        # Channel attention
        feat_ca = self.conv_ca(x)
        feat_ca = self.ca(feat_ca)
        feat_ca = self.conv_ca_out(feat_ca)
        
        # Combine
        feat_sum = feat_pa + feat_ca
        out = self.conv_out(feat_sum)
        return out

# ============================================================================
# Global Feature Attention (from DAISY)
# ============================================================================

class GlobalFeatureAttention(nn.Module):
    """Attention mechanism for global features"""
    def __init__(self, feature_dim=7):
        super().__init__()
        self.layer_norm = nn.LayerNorm(feature_dim)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=feature_dim, 
            num_heads=1,
            batch_first=True
        )
    
    def forward(self, x):
        x = x.float()
        x = self.layer_norm(x)
        x = x.unsqueeze(1)
        attention_output, _ = self.self_attention(x, x, x)
        output = attention_output.squeeze(1) + x.squeeze(1)
        return output

# ============================================================================
# STEP Model with YOUR innovations
# ============================================================================

class STEPModel(nn.Module):
    """
    STEP Architecture with Your Innovations:
    - ResNet + DANet (from DAISY)
    - Structure-aware physchem weighting (YOUR innovation!)
    - Global features guide learning (from DAISY)
    """
    def __init__(self, num_classes=2, use_structure_priors=True):
        super().__init__()
        self.use_structure_priors = use_structure_priors
        
        # YOUR INNOVATION: Structure priors
        if use_structure_priors:
            try:
                priors_path = Path(__file__).parent.parent / 'data' / 'embeddings' / 'structure_priors.npz'
                priors_data = np.load(priors_path)
                contact_weights = torch.from_numpy(priors_data['contact_weights'])
                self.register_buffer('contact_weights', contact_weights.unsqueeze(0))
                print(f"✅ Loaded structure priors from {priors_data['n_structures']} PDB structures")
            except Exception as e:
                print(f"⚠️  Could not load structure priors: {e}")
                self.register_buffer('contact_weights', torch.ones(1, 20, 15))
                self.use_structure_priors = False
        else:
            self.register_buffer('contact_weights', torch.ones(1, 20, 15))
        
        # ResNet backbone (from DAISY)
        self.resnet = ResNet(BasicBlock, [2, 2, 2, 2], in_channels=5)
        
        # Global feature attention (from DAISY)
        self.global_attention = GlobalFeatureAttention(feature_dim=7)
        
        # Transform global features to match ResNet output
        self.global_transform = nn.Conv2d(7, 512, kernel_size=1)
        
        # DANet head (from DAISY)
        self.danet_head = DANetHead(512, num_classes)
        
        # Final pooling and classification
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, physchem_map, global_features):
        # YOUR INNOVATION: Apply structure-aware weighting
        if self.use_structure_priors:
            weighted_map = physchem_map * self.contact_weights.unsqueeze(1)
        else:
            weighted_map = physchem_map
        
        # ResNet feature extraction
        resnet_features = self.resnet(weighted_map)  # [B, 512, H, W]
        
        # Global feature attention
        global_feat = self.global_attention(global_features)  # [B, 7]
        
        # Transform and expand global features
        global_feat_expanded = global_feat.unsqueeze(-1).unsqueeze(-1)  # [B, 7, 1, 1]
        global_feat_expanded = self.global_transform(global_feat_expanded)  # [B, 512, 1, 1]
        
        # Expand to match spatial dimensions
        _, _, H, W = resnet_features.size()
        global_feat_expanded = global_feat_expanded.expand(-1, -1, H, W)
        
        # Combine with ResNet features
        combined_features = resnet_features + global_feat_expanded
        
        # Dual attention
        out = self.danet_head(combined_features)  # [B, num_classes, H, W]
        
        # Pool and output
        out = self.avgpool(out)  # [B, num_classes, 1, 1]
        out = torch.flatten(out, 1)  # [B, num_classes]
        
        # Binary classification
        out = torch.softmax(out, dim=1)[:, 1]  # Get probability of class 1
        
        return out

def step_model(use_structure_priors=True):
    """Factory function"""
    return STEPModel(num_classes=2, use_structure_priors=use_structure_priors)

if __name__ == '__main__':
    # Test
    model = step_model()
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test forward pass
    batch = 4
    physchem = torch.randn(batch, 5, 20, 15)
    global_feat = torch.randn(batch, 7)
    
    out = model(physchem, global_feat)
    print(f"Output shape: {out.shape}")
    print("✅ Model test passed")
