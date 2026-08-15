import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.nn.init import _calculate_fan_in_and_fan_out
from timm.models.layers import to_2tuple, trunc_normal_
from pytorch_wavelets import DWT, IDWT

class PatchEmbed(nn.Module):

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        if kernel_size is None:
            kernel_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size - patch_size + 1) // 2, padding_mode='reflect')

    def forward(self, x):
        x = self.proj(x)
        return x


class PatchUnEmbed(nn.Module):

    def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = embed_dim
        if kernel_size is None:
            kernel_size = 1
        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans * patch_size ** 2, kernel_size=kernel_size,
                      padding=kernel_size // 2, padding_mode='reflect'),
            nn.PixelShuffle(patch_size)
        )

    def forward(self, x):
        x = self.proj(x)
        return x

class FHSD(nn.Module):
   
    def __init__(self, in_chans, embed_dim):
        super().__init__()
        self.dwt = DWT(J=1, wave='haar', mode='reflect')
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=1, padding=1, padding_mode='reflect')

    def forward(self, x):
        LL, Yh = self.dwt(x)
        hi_bands = Yh[0].reshape(LL.shape[0], -1, LL.shape[2], LL.shape[3])
        x_downsampled = self.proj(LL)
        return x_downsampled, hi_bands

class FHSU(nn.Module):
    def __init__(self, in_dim, out_chans):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, out_chans, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.idwt = IDWT(wave='haar', mode='reflect')

    def forward(self, x_low, hi_bands):
        x_ll = self.proj(x_low)
        B, C, H, W = x_ll.shape
        x_hi = hi_bands.view(B, C, 3, H, W)
        x_reconstructed = self.idwt((x_ll, [x_hi]))
        return x_reconstructed


class RLN(nn.Module):
    def __init__(self, dim, eps=1e-5, detach_grad=False):
        super(RLN, self).__init__()
        self.eps = eps
        self.detach_grad = detach_grad

        self.weight = nn.Parameter(torch.ones((1, dim, 1, 1)))
        self.bias = nn.Parameter(torch.zeros((1, dim, 1, 1)))

        self.meta1 = nn.Conv2d(1, dim, 1)
        self.meta2 = nn.Conv2d(1, dim, 1)

        trunc_normal_(self.meta1.weight, std=.02)
        nn.init.constant_(self.meta1.bias, 1)

        trunc_normal_(self.meta2.weight, std=.02)
        nn.init.constant_(self.meta2.bias, 0)

    def forward(self, input):
        mean = torch.mean(input, dim=(1, 2, 3), keepdim=True)
        std = torch.sqrt((input - mean).pow(2).mean(dim=(1, 2, 3), keepdim=True) + self.eps)

        normalized_input = (input - mean) / std

        if self.detach_grad:
            rescale = self.meta1(std.detach())
            rebias = self.meta2(mean.detach())
        else:
            rescale = self.meta1(std)
            rebias = self.meta2(mean)

        out = normalized_input * self.weight + self.bias
        return out, rescale, rebias




class AttentionGate(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv_se = nn.Sequential(
        nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False),
        nn.Sigmoid()
    )
    def forward(self, x):
        attention_weights = self.conv_se(self.pool(x))
        return attention_weights

class AAFFN(nn.Module):
    def __init__(self, network_depth, in_features, hidden_features=None, out_features=None,
            act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.network_depth = network_depth

        self.proj_in = nn.Conv2d(in_features, hidden_features, 1)
        self.act1 = act_layer(True)
    
        self.attention_gate = AttentionGate(hidden_features)
        self.dconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3,
                           stride=1, padding=1, groups=hidden_features)


        self.act2 = act_layer(True)
        self.proj_out = nn.Conv2d(hidden_features, out_features, 1)

        self.drop = nn.Dropout(drop) if drop > 0. else nn.Identity()
    
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            gain = (8 * self.network_depth) ** (-1 / 4)
            fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            trunc_normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.proj_in(x)
        x = self.act1(x)
     
        identity_path = x
    
        attention = self.attention_gate(x) 
        gated_path = self.dconv(x)
    
        gated_path = gated_path * attention
    
        x = identity_path + gated_path
    
        x = self.act2(x)
        x = self.proj_out(x)
        x = self.drop(x)
    
        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size ** 2, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def get_relative_positions(window_size):
    coords_h = torch.arange(window_size)
    coords_w = torch.arange(window_size)
    coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
    coords_flatten = torch.flatten(coords, 1)
    relative_positions = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_positions = relative_positions.permute(1, 2, 0).contiguous()
    relative_positions_log = torch.sign(relative_positions) * torch.log(1. + relative_positions.abs())
    return relative_positions_log


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        relative_positions = get_relative_positions(self.window_size)
        self.register_buffer("relative_positions", relative_positions)

        self.meta = nn.Sequential(
            nn.Linear(2, 256, bias=True),
            nn.ReLU(True),
            nn.Linear(256, num_heads, bias=True)
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv):
        B_, N, _ = qkv.shape
        qkv = qkv.reshape(B_, N, 3, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.meta(self.relative_positions)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, self.dim)
        return x


class Attention(nn.Module):
    def __init__(self, network_depth, dim, num_heads, window_size, shift_size, use_attn=False, conv_type=None):
        super().__init__()
        self.dim = dim
        self.head_dim = int(dim // num_heads)
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.network_depth = network_depth
        self.use_attn = use_attn
        self.conv_type = conv_type

        if self.conv_type == 'Conv':
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect'),
                nn.ReLU(True),
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect')
            )

        if self.conv_type == 'DWConv':
            self.conv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, padding_mode='reflect')

        if self.conv_type == 'DWConv' or self.use_attn:
            self.V = nn.Conv2d(dim, dim, 1)
            self.proj = nn.Conv2d(dim, dim, 1)

        if self.use_attn:
            self.QK = nn.Conv2d(dim, dim * 2, 1)
            self.attn = WindowAttention(dim, window_size, num_heads)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            w_shape = m.weight.shape
            if w_shape[0] == self.dim * 2:
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
                std = math.sqrt(2.0 / float(fan_in + fan_out))
                trunc_normal_(m.weight, std=std)
            else:
                gain = (8 * self.network_depth) ** (-1 / 4)
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
                std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
                trunc_normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def check_size(self, x, shift=False):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        if shift:
            x = F.pad(x, (self.shift_size, (self.window_size - self.shift_size + mod_pad_w) % self.window_size,
                          self.shift_size, (self.window_size - self.shift_size + mod_pad_h) % self.window_size),
                      mode='reflect')
        else:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, X):
        B, C, H, W = X.shape
        if self.conv_type == 'DWConv' or self.use_attn:
            V = self.V(X)

        if self.use_attn:
            QK = self.QK(X)
            QKV = torch.cat([QK, V], dim=1)
            shifted_QKV = self.check_size(QKV, self.shift_size > 0)
            Ht, Wt = shifted_QKV.shape[2:]
            shifted_QKV = shifted_QKV.permute(0, 2, 3, 1)
            qkv = window_partition(shifted_QKV, self.window_size)
            attn_windows = self.attn(qkv)
            shifted_out = window_reverse(attn_windows, self.window_size, Ht, Wt)
            if self.shift_size > 0:
                out = shifted_out[:, self.shift_size:(self.shift_size + H), self.shift_size:(self.shift_size + W), :]
            else:
                out = shifted_out[:, :H, :W, :]
            attn_out = out.permute(0, 3, 1, 2)
            if self.conv_type in ['Conv', 'DWConv']:
                conv_out = self.conv(V)
                out = self.proj(conv_out + attn_out)
            else:
                out = self.proj(attn_out)
        else:
            if self.conv_type == 'Conv':
                out = self.conv(X)
            elif self.conv_type == 'DWConv':
                out = self.proj(self.conv(V))
            else:
                out = X
        return out


class TransformerBlock(nn.Module):
    def __init__(self, network_depth, dim, num_heads, ffn_ratio=4.,
                 norm_layer=nn.LayerNorm, ffn_norm=False,
                 window_size=8, shift_size=0, use_attn=True, conv_type=None
                 ):
        super().__init__()
        self.use_attn = use_attn
        self.ffn_norm = ffn_norm
        self.norm1 = norm_layer(dim) if use_attn else nn.Identity()
        self.attn = Attention(network_depth, dim, num_heads=num_heads, window_size=window_size,
                              shift_size=shift_size, use_attn=use_attn, conv_type=conv_type)
        self.norm2 = norm_layer(dim) if use_attn and ffn_norm else nn.Identity()

    
        self.ffn = AAFFN(network_depth, dim, hidden_features=int(dim * ffn_ratio))

    def forward(self, x):
        identity = x
        if self.use_attn:
            norm_output = self.norm1(x)
            if isinstance(self.norm1, RLN):
                x_normed, rescale, rebias = norm_output
            else:
                x_normed = norm_output
                rescale, rebias = 1.0, 0.0
            attn_output = self.attn(x_normed)
            if isinstance(self.norm1, RLN):
                x = identity + attn_output * rescale + rebias
            else:
                x = identity + attn_output
        else:
            x = identity + self.attn(x)

        identity = x
        if self.use_attn and self.ffn_norm:
            norm_output = self.norm2(x)
            if isinstance(self.norm2, RLN):
                x_normed, rescale, rebias = norm_output
            else:
                x_normed = norm_output
                rescale, rebias = 1.0, 0.0
            ffn_output = self.ffn(x_normed)
            if isinstance(self.norm2, RLN):
                x = identity + ffn_output * rescale + rebias
            else:
                x = identity + ffn_output
        else:
            x = identity + self.ffn(self.norm2(x))
        return x


class BasicLayer(nn.Module):
    def __init__(self, network_depth, dim, depth, num_heads,ffn_ratio=4.,
                 norm_layer=nn.LayerNorm, window_size=8,
                 attn_ratio=0., attn_loc='last', conv_type=None):
        super().__init__()
        self.dim = dim
        self.depth = depth
        attn_depth = attn_ratio * depth
        if attn_loc == 'last':
            use_attns = [i >= depth - attn_depth for i in range(depth)]
        elif attn_loc == 'first':
            use_attns = [i < attn_depth for i in range(depth)]
        elif attn_loc == 'middle':
            use_attns = [i >= (depth - attn_depth) // 2 and i < (depth + attn_depth) // 2 for i in range(depth)]
        else:
            use_attns = [False for i in range(depth)]

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = TransformerBlock(network_depth=network_depth,
                                     dim=dim,
                                     num_heads=num_heads,
                                     ffn_ratio=ffn_ratio,
                                     norm_layer=norm_layer,
                                     window_size=window_size,
                                     shift_size=0 if (i % 2 == 0) else window_size // 2,
                                     use_attn=use_attns[i],
                                     conv_type=conv_type)
            self.blocks.append(block)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

class SKFusion(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super(SKFusion, self).__init__()
        self.height = height
        d = max(int(dim / reduction), 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, dim * height, 1, bias=False)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape
        in_feats_cat = torch.cat(in_feats, dim=1)
        in_feats_view = in_feats_cat.view(B, self.height, C, H, W)
        feats_sum = torch.sum(in_feats_view, dim=1)
        attn = self.ffn(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))
        out = torch.sum(in_feats_view * attn, dim=1)
        return out


class ED_Former(nn.Module):
    def __init__(self, in_chans=3, out_chans=4, window_size=8,
                 embed_dims=[24, 48, 96, 48, 24],
                 ffn_ratios=[2., 4., 4., 2., 2.],
                 depths=[4, 4, 4, 2, 2],
                 num_heads=[2, 4, 6, 1, 1],
                 attn_ratio=[0, 1 / 2, 1, 0, 0],
                 conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'],
                 norm_layer=[RLN, RLN, RLN, RLN, RLN]):
        super(ED_Former, self).__init__()
        self.patch_size = 4
        self.window_size = window_size
        self.ffn_ratios = ffn_ratios
        self.patch_embed = PatchEmbed(
            patch_size=1, in_chans=in_chans, embed_dim=embed_dims[0], kernel_size=3)
        self.layer1 = BasicLayer(network_depth=sum(depths), dim=embed_dims[0], depth=depths[0], num_heads=num_heads[0],
                                 ffn_ratio=ffn_ratios[0], norm_layer=norm_layer[0], window_size=window_size,
                                 attn_ratio=attn_ratio[0], attn_loc='last', conv_type=conv_type[0])
        self.fhsd1 = FHSD(in_chans=embed_dims[0], embed_dim=embed_dims[1])

        self.layer2 = BasicLayer(network_depth=sum(depths), dim=embed_dims[1], depth=depths[1], num_heads=num_heads[1],
                                 ffn_ratio=ffn_ratios[1], norm_layer=norm_layer[1], window_size=window_size,
                                 attn_ratio=attn_ratio[1], attn_loc='last', conv_type=conv_type[1])
        self.fhsd2 = FHSD(in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.layer3 = BasicLayer(network_depth=sum(depths), dim=embed_dims[2], depth=depths[2], num_heads=num_heads[2],
                                 ffn_ratio=ffn_ratios[2], norm_layer=norm_layer[2], window_size=window_size,
                                 attn_ratio=attn_ratio[2], attn_loc='last', conv_type=conv_type[2])

        self.fhsu1 = FHSU(in_dim=embed_dims[2], out_chans=embed_dims[3])
        assert embed_dims[1] == embed_dims[3], "Channel mismatch for skip connection 1"
        self.fusion1 = SKFusion(embed_dims[3])
        self.layer4 = BasicLayer(network_depth=sum(depths), dim=embed_dims[3], depth=depths[3], num_heads=num_heads[3],
                                 ffn_ratio=ffn_ratios[3], norm_layer=norm_layer[3], window_size=window_size,
                                 attn_ratio=attn_ratio[3], attn_loc='last', conv_type=conv_type[3])

        self.fhsu2 = FHSU(in_dim=embed_dims[3], out_chans=embed_dims[4])
        assert embed_dims[0] == embed_dims[4], "Channel mismatch for skip connection 2"
        self.fusion2 = SKFusion(embed_dims[4])
        self.layer5 = BasicLayer(network_depth=sum(depths), dim=embed_dims[4], depth=depths[4], num_heads=num_heads[4],
                                 ffn_ratio=ffn_ratios[4], norm_layer=norm_layer[4], window_size=window_size,
                                 attn_ratio=attn_ratio[4], attn_loc='last', conv_type=conv_type[4])

        self.patch_unembed = PatchUnEmbed(
            patch_size=1, out_chans=out_chans, embed_dim=embed_dims[4], kernel_size=3)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        # DWT(J=1) x 2, so image size must be divisible by 4
        mod_pad_h = (4 - h % 4) % 4
        mod_pad_w = (4 - w % 4) % 4
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x):

        x = self.patch_embed(x)
        x_layer1_out = self.layer1(x)

        x, hi_bands1 = self.fhsd1(x_layer1_out)
        x_layer2_out = self.layer2(x)

        x, hi_bands2 = self.fhsd2(x_layer2_out)


        x = self.layer3(x)

        x = self.fhsu1(x, hi_bands2)
        x = self.fusion1([x, x_layer2_out])
        x = self.layer4(x)

        x = self.fhsu2(x, hi_bands1)
        x = self.fusion2([x, x_layer1_out])
        x = self.layer5(x)

        x = self.patch_unembed(x)
        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x_checked = self.check_image_size(x)
        feat = self.forward_features(x_checked)
        K, B_res = torch.split(feat, (1, 3), dim=1)
        restored_image = K * x_checked - B_res + x_checked
        return restored_image[:, :, :H, :W]

