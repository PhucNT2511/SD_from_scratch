import torch
import torch.nn as nn
import torch.nn.functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

# VAE_ResidualBlock: just a CNN block with residual mechanism, not skip connection
# VAE_AttentionBlock: just a CNN block with QKV attention to attain global info, beside local info

class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(

            # (B,3,H,W) --> (B,128,H,W)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),

            # (B,128,H,W) --> (B,128,H,W)
            VAE_ResidualBlock(128,128),

            # (B,128,H,W) --> (B,128,H,W)
            VAE_ResidualBlock(128,128),

            # (B,128,H,W) --> (B,128,H//2,W//2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),

            # (B,128,H//2,W//2) --> (B,256,H//2,W//2)
            VAE_ResidualBlock(128,256),

            # (B,256,H//2,W//2) --> (B,256,H//2,W//2)
            VAE_ResidualBlock(256,256),

            # (B,256,H//2,W//2) --> (B,256,H//4,W//4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),

            # (B,256,H//4,W//4) --> (B,512,H//4,W//4)
            VAE_ResidualBlock(256,512),

            # (B,512,H//4,W//4) --> (B,512,H//4,W//4)
            VAE_ResidualBlock(512,512),

            # (B,512,H//2,W//2) --> (B,512,H//8,W//8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),

            # (B,512,H//8,W//8) --> (B,512,H//8,W//8)
            VAE_ResidualBlock(512,512),

            # (B,512,H//8,W//8) --> (B,512,H//8,W//8)
            VAE_ResidualBlock(512,512),

            # (B,512,H//8,W//8) --> (B,512,H//8,W//8)
            VAE_ResidualBlock(512,512),

            # (B,512,H//8,W//8) --> (B,512,H//8,W//8)
            VAE_AttentionBlock(512),

            # (B,512,H//8,W//8) --> (B,512,H//8,W//8)
            VAE_ResidualBlock(512,512),

            # (B,512,H//8,W//8) --> (B,512,H//8,W//8)
            nn.GroupNorm(32,512),

            # (B,512,H//8,W//8) --> (B,512,H//8,W//8)
            nn.SiLU(),

            # (B,512,H//8,W//8) --> (B,8,H//8,W//8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),

            ## (B,8,H//8,W//8) --> (B,8,H//8,W//8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0)
        )
    def forward(self, x: torch.Tensor, noise: torch.Tensor):
        # x.shape: (B,3,H,W)
        # noise.shape: (B,C,H/8,W/8)

        for module in self:
            if getattr(module, "stride", None) == (2,2):
                # (pad_left, pad_right, pad_top, pad_bottom)
                x = F.padding(x, (0,1,0,1))
            x = module(x)
        
        ### (B, 8, H/8, W/8) --> 2 tensors: (B, 4, H/8, W/8)
        mean, log_var = torch.chunk(x, 2, dim = 1)

        log_var = torch.clamp(log_var, -30, 20)
        variance = log_var.exp()
        std = variance.sqrt()

        x = mean + std * noise ### Because VAE need sample from a norm distribution. Not autoencoder.
        x = x * 0.18215 ## scale?? --> don't know why to choose

        return x


