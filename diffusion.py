import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention, CrossAttention

class TimeEmbedding(nn.Module):
    '''
    I don't really know why they choose 1280 to be the dim of time_emb space
    '''
    def __init__(self, n_embd: int):
        super().__init__()
        self.linear_1 = nn.Linear(n_embd, 4*n_embd)
        self.linear_2 = nn.Linear(4*n_embd, 4*n_embd)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (1,320) -> (1,1280)
        x = self.linear_1(x)

        x = F.silu(x)

        # (1,1280) -> (1,1280)
        x = self.linear_2(x)

        return x

class SwitchSequential(nn.Sequential):
    '''
    Like a rule-based method to identify particular inputs to each layer
    '''
    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, Unet_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, Unet_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x
    

class Upsample (nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode = "nearest")
        return self.conv(x) 
    
class Unet_OutputLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self,x):
        # (B,320,H/8,W/8)
        x = self.groupnorm(x)
        x = F.silu(x)
        x = self.conv(x)
        return x

class Unet_ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_time = 1280):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32,in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels) ### 1280 --> out_channels
        
        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        
    def forward(self, feature, time):
        # feature.shape (B,inchannels, H, W)
        # time (1,1280)

        residue = feature

        feature = self.groupnorm_feature(feature)
        feature = F.silu(feature)
        feature = self.conv_feature(feature)

        time = F.silu(time)
        time = self.linear_time(time) #(1,out_channels)

        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        merged = self.groupnorm_merged(merged)
        merged = F.silu(merged)
        merged = self.conv_merged(merged)

        return merged + self.residual_layer(residue)
    
class Unet_AttentionBlock(nn.Module):
    def __init__(self, n_head: int, n_embd: int, d_context=768):
        super().__init__()
        channels = n_head * n_embd

        self.groupnorm = nn.GroupNorm(32, channels, eps = 1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False) ## d_context used for crossAttn
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, 4*channels*2)
        self.linear_geglu_2 = nn.Linear(4*channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
    
    def forward(self, x, context):
        # x: (B, C, H, W)
        # context: (B, seq_len, dim)
        residue_long = x

        x = self.groupnorm(x)
        x = self.conv_input(x)

        n,c,h,w = x.shape

        x = x.view((n,c,h*w))
        x = x.transpose(-1,-2) 

        ### self.attention
        residue_short = x
        x = self.layernorm_1(x)
        x = self.attention_1(x)
        x += residue_short

        ### cross_attention
        residue_short = x
        x = self.layernorm_2(x)
        x = self.attention_2(x, context)
        x += residue_short

        ### Normalization + FF in Transformer
        residue_short = x
        x = self.layernorm_3(x)
        x, gate = self.linear_geglu_1(x).chunk(2, dim = -1) ### gate here to control the amount of info from x_cur
        x = x * F.gelu(gate)
        x = self.linear_geglu_2(x)
        x += residue_short

        x = x.transpose(-1,-2)
        x = x.view((n,c,h,w))

        return self.conv_output(x) + residue_long



class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module([
            #(B,4,H/8,W/8) -> (B,320,H/8,W/8)
            SwitchSequential(nn.Conv2d(4,320,kernel_size=3,padding=1)),
            SwitchSequential(Unet_ResidualBlock(320,320), Unet_AttentionBlock(8,40)),
            SwitchSequential(Unet_ResidualBlock(320,320), Unet_AttentionBlock(8,40)),

            #(B,320,H/8,W/8) -> (B,640,H/16,W/16)
            SwitchSequential(nn.Conv2d(320,320,kernel_size=3,stride=2,padding=1)),
            SwitchSequential(Unet_ResidualBlock(320,640), Unet_AttentionBlock(8,80)), # 8x80=640
            SwitchSequential(Unet_ResidualBlock(640,640), Unet_AttentionBlock(8,80)),

            #(B,640,H/16,W/16) -> (B,1280,H/32,W/32)
            SwitchSequential(nn.Conv2d(640,640,kernel_size=3,stride=2,padding=1)),
            SwitchSequential(Unet_ResidualBlock(640,1280), Unet_AttentionBlock(8,160)),
            SwitchSequential(Unet_ResidualBlock(1280,1280), Unet_AttentionBlock(8,160)),

            #(B,1280,H/32,W/32) -> (B,1280,H/64,W/64)
            SwitchSequential(nn.Conv2d(1280,1280,kernel_size=3,stride=2,padding=1)),
            SwitchSequential(Unet_ResidualBlock(1280,1280)),
            SwitchSequential(Unet_ResidualBlock(1280,1280)),
        ])

        self.bottle_neck = SwitchSequential(
            Unet_ResidualBlock(1280,1280),
            Unet_AttentionBlock(8,160),
            Unet_ResidualBlock(1280,1280),
        )

        self.decoder = nn.ModuleList([

            ### skip connection here
            SwitchSequential(Unet_ResidualBlock(2560,1280)),
            SwitchSequential(Unet_ResidualBlock(2560,1280)),
            SwitchSequential(Unet_ResidualBlock(2560,1280), Upsample(1280)),

            SwitchSequential(Unet_ResidualBlock(2560,1280), Unet_AttentionBlock(8,160)),
            SwitchSequential(Unet_ResidualBlock(2560,1280), Unet_AttentionBlock(8,160)),
            SwitchSequential(Unet_ResidualBlock(1920,1280), Unet_AttentionBlock(8,160), Upsample(1280)),

            SwitchSequential(Unet_ResidualBlock(1920,640), Unet_AttentionBlock(8,80)),
            SwitchSequential(Unet_ResidualBlock(1280,640), Unet_AttentionBlock(8,80)),
            SwitchSequential(Unet_ResidualBlock(960,640), Unet_AttentionBlock(8,80), Upsample(640)),

            SwitchSequential(Unet_ResidualBlock(960,320), Unet_AttentionBlock(8,40)),
            SwitchSequential(Unet_ResidualBlock(640,320), Unet_AttentionBlock(8,40)),
            SwitchSequential(Unet_ResidualBlock(640,320), Unet_AttentionBlock(8,40)),           

        ])

    def forward(self, x, context, time):
        # x: (Batch_Size, 4, Height / 8, Width / 8)
        # context: (Batch_Size, Seq_Len, Dim) 
        # time: (1, 1280)
        skip_connections = []
        for layers in self.encoders:
            x = layers(x, context, time)
            skip_connections.append(x)

        x = self.bottleneck(x, context, time)

        for layers in self.decoders:
            x = torch.cat((x, skip_connections.pop()), dim=1) 
            x = layers(x, context, time)
        
        return x
    
class Diffusion(nn.Module):
    '''
    I don't really know why they choose 320 is the number of channel/dim.
    '''
    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = Unet()
        self.final = Unet_OutputLayer(320,4) # from C = 320 --> C = 4 

    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        # latent: (B,4,H/8,W/8)
        # context: (B, seq_len, dim)
        # time: (1,320)

        # (1,320) -> (1,1280)
        time = self.time_embedding(time)

        # (B,4,H/8,W/8) -> (B,320,H/8,W/8)
        output = self.unet(latent, context, time)

        # (B,320,H/8,W/8) -> (B,4,H/8,W/8)
        output = self.final(output)

        return output
    
