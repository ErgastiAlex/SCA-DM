import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleEffStyleEncoder(nn.Module):
    def __init__(self, input_channels = 3, num_mask_channels = 19, num_downsample = 4, num_upsample = 3, 
        num_feat = 4, output_dim = 256, kernel_dim = 3):
        super(MultiScaleEffStyleEncoder, self).__init__()

        self.nmc = num_mask_channels
        self.num_downsample = num_downsample
        self.num_upsample = num_upsample

        self.kernels = []
        for i in range(0,num_downsample):
            self.kernels += [(num_mask_channels*num_feat*(2**(i)), (num_mask_channels*num_feat*(2**(i+1))))]
        
        self.Encoder = nn.ModuleDict()
        self.Decoder = nn.ModuleDict()
        self.out = nn.ModuleDict()

        # input layer
        self.Encoder['first_layer'] = nn.Sequential(nn.Conv2d(input_channels, self.kernels[0][0], kernel_dim, padding=1),
                    nn.GroupNorm(self.nmc, self.kernels[0][0]),
                    nn.ReLU())
        
        # Encoding
        for i, (in_kernel, out_kernel) in enumerate(self.kernels):
            
            self.Encoder[f'enc_layer_{i}'] = nn.Sequential(nn.Conv2d(in_kernel,out_kernel, 3, 
                        stride = 2, padding=1, groups=self.nmc), 
                        nn.GroupNorm(self.nmc, out_kernel),
                        nn.ReLU())
        
        # Upsampling
        for i, (in_kernel, out_kernel) in reversed(list(enumerate(self.kernels))):
            prev_kernel = out_kernel
            if i == num_downsample - 1:
                prev_kernel = 0

            if i == (num_downsample - 1 - num_upsample):
                break
            self.Decoder[f'dec_layer_{num_downsample-1-i}'] = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                        nn.Conv2d(out_kernel + prev_kernel, in_kernel, 3, 
                        stride = 1, padding=1, groups=self.nmc),
                        nn.GroupNorm(self.nmc, in_kernel),
                        nn.ReLU())

        for i, (in_kernel, out_kernel) in reversed(list(enumerate(self.kernels))):
            if i == (num_downsample - 1 - num_upsample):
                break
            self.out[f'out_{num_downsample-1-i}'] = nn.Sequential(nn.Conv1d(in_kernel, output_dim*self.nmc, 1, 
                                groups=self.nmc), nn.Tanh())
                
        self.eps = 1e-5
    
    def forward(self, x, mask):
        x = self.Encoder['first_layer'](x)

        enc_feat = []
        for i in range(self.num_downsample):
            x = self.Encoder[f'enc_layer_{i}'](x)
            enc_feat.append(x)

        dec_style_feat = []
        for i in range(self.num_upsample):
            x = self.Decoder[f'dec_layer_{i}'](x)

            _,_,side_h,side_w = x.shape
            mask_int = nn.functional.interpolate(mask, size=(side_h, side_w), mode='nearest')   
            repetitions = th.tensor([self.kernels[self.num_downsample-1-i][0]//self.nmc]*self.nmc).to(mask.device) # G*19  
            mask_int = th.repeat_interleave(mask_int, repeats=repetitions, dim=1)

            h = x * mask_int # B, G*19, H, W

            # pooling
            h = th.sum(h, dim=(2,3))  # B, G*19     
            div = th.sum(mask_int, dim=(2,3)) # B, G*19    
            h = h / (div + self.eps) 

            h = self.out[f'out_{i}'](h.unsqueeze(-1)) # B, 256*19, 1

            h = h.reshape((h.shape[0], self.nmc, h.shape[1]//self.nmc))
            dec_style_feat.append(h)

            # prepare skip connection
            x = th.cat((x, enc_feat[self.num_upsample-1-i]), dim = 1)
        
        #[s1,s2,s3,s4,s5]
        return th.cat(dec_style_feat, dim = 2)

