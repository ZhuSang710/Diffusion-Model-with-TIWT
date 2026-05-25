import torch
import torch.nn as nn
import pywt
import numpy as np

class NonSubsampledWaveletTransform(nn.Module):
     
    
    def __init__(self, wavelet='db4', mode='symmetric'):
        super().__init__()
        self.wavelet = wavelet
        self.mode = mode
        
    def forward_transform(self, x):
         
        batch_size, channels, height, width = x.shape
        
         
        coeffs_list = []
        for b in range(batch_size):
            batch_coeffs = []
            for c in range(channels):
                img = x[b, c].cpu().numpy()
                 
                cA, (cH, cV, cD) = pywt.dwt2(img, self.wavelet, mode=self.mode)
                batch_coeffs.append([cA, cH, cV, cD])
            coeffs_list.append(batch_coeffs)
        
         
        cA_batch = torch.stack([torch.tensor(coeffs_list[b][c][0], dtype=x.dtype, device=x.device) 
                               for b in range(batch_size) for c in range(channels)])
        cH_batch = torch.stack([torch.tensor(coeffs_list[b][c][1], dtype=x.dtype, device=x.device) 
                               for b in range(batch_size) for c in range(channels)])
        cV_batch = torch.stack([torch.tensor(coeffs_list[b][c][2], dtype=x.dtype, device=x.device) 
                               for b in range(batch_size) for c in range(channels)])
        cD_batch = torch.stack([torch.tensor(coeffs_list[b][c][3], dtype=x.dtype, device=x.device) 
                               for b in range(batch_size) for c in range(channels)])
        
         
        cA_h, cA_w = cA_batch[0].shape
        
         
        cA_batch = cA_batch.view(batch_size, channels, cA_h, cA_w)
        cH_batch = cH_batch.view(batch_size, channels, cA_h, cA_w)
        cV_batch = cV_batch.view(batch_size, channels, cA_h, cA_w)
        cD_batch = cD_batch.view(batch_size, channels, cA_h, cA_w)
        
         
        wavelet_coeffs = torch.cat([cA_batch, cH_batch, cV_batch, cD_batch], dim=1)
        
         
        if cA_h != 128 or cA_w != 128:
            import torch.nn.functional as F
            wavelet_coeffs = F.interpolate(wavelet_coeffs, size=(128, 128), mode='bilinear', align_corners=False)
        
        return wavelet_coeffs
    
    def inverse_transform(self, wavelet_coeffs):
         
        batch_size, total_channels, height, width = wavelet_coeffs.shape
        channels = total_channels // 4
        
         
        cA_batch = wavelet_coeffs[:, :channels]
        cH_batch = wavelet_coeffs[:, channels:2*channels]
        cV_batch = wavelet_coeffs[:, 2*channels:3*channels]
        cD_batch = wavelet_coeffs[:, 3*channels:]
        
         
        reconstructed = []
        for b in range(batch_size):
            batch_imgs = []
            for c in range(channels):
                cA = cA_batch[b, c].cpu().numpy()
                cH = cH_batch[b, c].cpu().numpy()
                cV = cV_batch[b, c].cpu().numpy()
                cD = cD_batch[b, c].cpu().numpy()
                
                 
                img = pywt.idwt2((cA, (cH, cV, cD)), self.wavelet, mode=self.mode)
                batch_imgs.append(torch.tensor(img, dtype=wavelet_coeffs.dtype, device=wavelet_coeffs.device))
            
            reconstructed.append(torch.stack(batch_imgs))
        
        return torch.stack(reconstructed)

 
from .feature_enhancement import FeatureEnhancementModule

 
class WaveletDiffusionModel(nn.Module):
    def __init__(self, model, use_enhancement=False):
        super(WaveletDiffusionModel, self).__init__()
        self.model = model
        self.base_model = model
        self.wavelet_transform = NonSubsampledWaveletTransform()
        
         
        self.use_enhancement = use_enhancement
        self.enhancement_enabled = False   
        if use_enhancement:
            self.feature_enhancement = FeatureEnhancementModule(
                in_channels=8, 
                hidden_channels=64, 
                out_channels=8
            )
        else:
            self.feature_enhancement = None
    
    def enable_feature_enhancement(self, enable=True):
         
        self.enhancement_enabled = enable
        if enable and self.feature_enhancement is None:
            print("Warning: Feature enhancement module not initialized. Creating default module.")
            self.feature_enhancement = FeatureEnhancementModule(
                in_channels=8, 
                hidden_channels=64, 
                out_channels=8
            )
        print(f"Feature Enhancement Module: {'Enabled' if enable else 'Disabled'}")
    
    def forward(self, x, t, use_enhancement=None, bypass_enhancement=True, enhancement_weight=0):
         
        x_wavelet = self.input_transform(x)
        
         
        if use_enhancement is None:
            use_enhancement = self.enhancement_enabled
        
         
        if self.feature_enhancement is not None and use_enhancement:
            x_wavelet = self.feature_enhancement(
                x_wavelet, 
                use_enhancement=use_enhancement,
                bypass_enhancement=bypass_enhancement,
                enhancement_weight=enhancement_weight
            )
        
         
        output = self.model(x_wavelet, t)
        
         
        return self.output_inverse_transform(output)
    
    def input_transform(self, x):
         
        return self.wavelet_transform.forward_transform(x)
    
    def output_inverse_transform(self, wavelet_output):
         
        return self.wavelet_transform.inverse_transform(wavelet_output)
    
    def transform_inputs(self, x_img, x_gt):
         
        x_img_wavelet = self.wavelet_transform.forward_transform(x_img)
        x_gt_wavelet = self.wavelet_transform.forward_transform(x_gt)
        return x_img_wavelet, x_gt_wavelet
    
    def inverse_transform_output(self, wavelet_output):
         
        return self.wavelet_transform.inverse_transform(wavelet_output)
