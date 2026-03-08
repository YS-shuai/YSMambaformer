from timm.layers import DropPath, trunc_normal_, to_2tuple
import torch
from torch import nn
from einops.layers.torch import Rearrange
from models.Mambaformer.mamba2 import Mamba2
from models.Mambaformer.swin_atten import SwinTransformerBlock
from models.layers.blocks import DRF, Gate, Mlp
from models.layers.layer_utils import ACTIVATIONS
from models.layers.attention import Attention


class Early_feature(nn.Module):
    def __init__(self, encoder,in_channels, out_channels, input_resolution, patch_size, act='silu', field_size=None, bias=True):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.input_resolution = input_resolution
        self.encoder =encoder
        self.reshape = Rearrange('b t c (h p1) (w p2) -> (b t) (p1 p2 c) h w', p1=self.patch_size, p2=self.patch_size)
        if self.encoder == "Conv2d":
            field_size = [1, 3, 5] if field_size is None else field_size
            self.local = DRF(n_images=in_channels, out_channels=out_channels, patch_sizes=self.patch_size,
                             field_sizes=field_size, input_resolution=input_resolution, bias=bias)
            self.act = ACTIVATIONS[act]()
            proj_channels = out_channels * len(field_size)

            self.project = nn.Conv2d(in_channels=proj_channels, out_channels=out_channels, kernel_size=1, )
            self.apply(self._init_weights)

        elif self.encoder == "Linear":
            self.mlp = Mlp(in_channels*self.patch_size**2 , out_channels, out_channels)



    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, T, C, H, W = x.shape

        x = self.reshape(x)

        if self.encoder == "Conv2d":
            map = self.act(self.local(x))
            x = self.project(map)

        x = x.view(B * T, -1, self.input_resolution*self.input_resolution).permute(0, 2, 1).contiguous()

        if self.encoder == "Linear":
            x = self.mlp(x)

        x_patches = x.view(B, T, self.input_resolution*self.input_resolution, self.out_channels)

        return x_patches  # (B, T, N, C)

class SwiGLU(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.SiLU,
            norm_layer=None,
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1_g = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.fc1_x = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def init_weights(self):
        nn.init.ones_(self.fc1_g.bias)
        nn.init.normal_(self.fc1_g.weight, std=1e-6)

    def forward(self, x):
        x_gate = self.fc1_g(x)
        x = self.fc1_x(x)
        x = self.act(x_gate) * x
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
class Mid_feature(nn.Module):
    def __init__(
        self,
        dim, input_resolution, num_heads, mlp_ratio=4, drop=0.0, attn_drop=0.0,
        norm_layer=nn.LayerNorm, fast_attn=True, use_swin=False, window_size=4,
        drop_path=0.1, i=0, Gate_act="silu", Space=True,
    ):
        super().__init__()
        if Gate_act == "silu":
            gate = Gate
        elif Gate_act == "gelu":
            gate = Mlp

        self.Tem_attn = Mamba2(
            d_model=dim,
            device="cuda:0",
            dt_min=0.001,
            dt_max=0.1,
            d_state=256,
            d_conv=4,
            expand=2,
            headdim=64,
            D_has_hdim=True,
            norm_before_gate=True
        )
        self.Sem_attn = Mamba2(
            d_model=dim,
            device="cuda:0",
            dt_min=0.001,
            dt_max=0.1,
            d_state=256,
            d_conv=4,
            expand=2,
            headdim=64,
            D_has_hdim=True,
            norm_before_gate=True
        )
        self.Tem_drop1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.Tem_gate = gate(dim, mlp_ratio * dim, drop=drop)
        self.Tem_drop2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.Tem_norm1 = norm_layer(dim)
        self.Tem_norm2 = norm_layer(dim)
        self.Tem_norm3 = norm_layer(dim)

        self.Space = Space
        if self.Space:
            if use_swin:
                shift_size = 0 if (i % 2 == 0) else window_size // 2
                self.Spa_attn = SwinTransformerBlock(
                    dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size,
                    shift_size=shift_size, drop=drop, attn_drop=attn_drop,
                )
            else:
                self.Spa_attn = Attention(
                    dim, num_heads=num_heads,
                    attn_drop=attn_drop, proj_drop=drop, norm_layer=norm_layer, fast_attn=fast_attn,
                )

            self.Spa_drop1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
            self.Spa_gate = gate(dim, mlp_ratio * dim, drop=drop)
            self.Spa_drop2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
            self.Spa_norm1 = norm_layer(dim)
            self.Spa_norm2 = norm_layer(dim)
            self.Spa_norm3 = norm_layer(dim)
            self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
           )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        b, t, n, c = x.shape

        x = x.transpose(1, 2).reshape(-1, t, c)
        x_forward = self.Tem_attn(self.Tem_norm1(x))
        x_backward = self.Sem_attn(self.Tem_norm1(x).flip(1)).flip(1)
        combined = torch.cat([x_forward, x_backward], dim=-1)
        gate_weights = self.gate(combined)
        output = gate_weights * x_forward + (1 - gate_weights) * x_backward
        x = x + self.Tem_drop1(output)
        x = x + self.Tem_drop2(self.Tem_gate(self.Tem_norm2(x)))
        x = self.Tem_norm3(x)

        x = x.view(b, n, t, c).transpose(1, 2).reshape(-1, n, c)

        if self.Space:
            x = x + self.Spa_drop1(self.Spa_attn(self.Spa_norm1(x)))
            x = x + self.Spa_drop2(self.Spa_gate(self.Spa_norm2(x)))
            x = self.Spa_norm3(x)

        return x.view(b, t, n, c)

