import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureEnhancementModule(nn.Module):
     

     
    def __init__(self, in_channels=8, hidden_channels=64, out_channels=8):
        super(FeatureEnhancementModule, self).__init__()
        
         
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(hidden_channels)
        
        self.conv4 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(hidden_channels)
        
        self.conv5 = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)
        
         
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_channels, hidden_channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 8, hidden_channels, 1),
            nn.Sigmoid()
        )
        
         
        self.residual_proj = None
        if in_channels != out_channels:
            self.residual_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x, use_enhancement=True, bypass_enhancement=False, enhancement_weight=1.0):
         

        identity = x
        
         
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))
        
         
        attention_weights = self.attention(out)
        out = out * attention_weights
        
        out = F.relu(self.bn4(self.conv4(out)))
        out = self.conv5(out)
        
         
        if self.residual_proj is not None:
            identity = self.residual_proj(identity)
        
         
        enhanced_features = identity + enhancement_weight * out
        
        return enhanced_features

class LightweightFeatureEnhancement(nn.Module):

     
    def __init__(self, in_channels=8, hidden_channels=32, out_channels=8):
        super(LightweightFeatureEnhancement, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)
        
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        
         
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_channels, hidden_channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels // 4, hidden_channels, 1),
            nn.Sigmoid()
        )
        
        self.residual_proj = None
        if in_channels != out_channels:
            self.residual_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x, use_enhancement=True, bypass_enhancement=False, enhancement_weight=1.0):
        if not use_enhancement or bypass_enhancement or enhancement_weight == 0:
            return x
        
        identity = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = out * self.se(out)   
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.conv3(out)
        
        if self.residual_proj is not None:
            identity = self.residual_proj(identity)
        
        return identity + enhancement_weight * out