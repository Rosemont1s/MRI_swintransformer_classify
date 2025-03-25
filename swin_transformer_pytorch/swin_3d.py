import torch
from torch import nn, einsum
import numpy as np
from einops import rearrange, repeat


class CyclicShift3D(nn.Module):
    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement

    def forward(self, x):
        return torch.roll(x, shifts=(self.displacement, self.displacement, self.displacement), dims=(1, 2, 3))


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


def create_mask_3d(window_size, displacement, directions):
    """
    Create 3D window attention masks for different spatial directions
    """
    mask = torch.zeros(window_size ** 3, window_size ** 3)

    for direction, is_active in directions.items():
        if is_active:
            if direction == 'upper_lower':
                mask[-displacement * window_size**2:, :-displacement * window_size**2] = float('-inf')
                mask[:-displacement * window_size**2, -displacement * window_size**2:] = float('-inf')
            elif direction == 'left_right':
                mask = rearrange(mask, '(h1 w1 d1) (h2 w2 d2) -> h1 w1 d1 h2 w2 d2',
                                 h1=window_size, w1=window_size, d1=window_size)
                mask[:, -displacement:, :, :, :-displacement, :] = float('-inf')
                mask[:, :-displacement, :, :, -displacement:, :] = float('-inf')
                mask = rearrange(mask, 'h1 w1 d1 h2 w2 d2 -> (h1 w1 d1) (h2 w2 d2)')
            elif direction == 'depth':
                mask = rearrange(mask, '(h1 w1 d1) (h2 w2 d2) -> h1 w1 d1 h2 w2 d2',
                                 h1=window_size, w1=window_size, d1=window_size)
                mask[:, :, -displacement:, :, :, :-displacement] = float('-inf')
                mask[:, :, :-displacement, :, :, -displacement:] = float('-inf')
                mask = rearrange(mask, 'h1 w1 d1 h2 w2 d2 -> (h1 w1 d1) (h2 w2 d2)')

    return mask


def get_relative_distances_3d(window_size):
    """
    Generate relative distances for 3D window
    """
    indices = torch.tensor(np.array([[x, y, z]
                                     for x in range(window_size)
                                     for y in range(window_size)
                                     for z in range(window_size)]))
    distances = indices[None, :, :] - indices[:, None, :]
    return distances


class WindowAttention3D(nn.Module):
    def __init__(self, dim, heads, head_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        inner_dim = head_dim * heads

        self.heads = heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted

        if self.shifted:
            displacement = window_size // 2
            self.cyclic_shift = CyclicShift3D(-displacement)
            self.cyclic_back_shift = CyclicShift3D(displacement)
            self.masks = nn.ParameterDict({
                'upper_lower': nn.Parameter(create_mask_3d(window_size, displacement,
                                                           {'upper_lower': True, 'left_right': False, 'depth': False}),
                                            requires_grad=False),
                'left_right': nn.Parameter(create_mask_3d(window_size, displacement,
                                                          {'upper_lower': False, 'left_right': True, 'depth': False}),
                                           requires_grad=False),
                'depth': nn.Parameter(create_mask_3d(window_size, displacement,
                                                     {'upper_lower': False, 'left_right': False, 'depth': True}),
                                      requires_grad=False)
            })

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances_3d(window_size) + window_size - 1
            self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1, 2 * window_size - 1))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(window_size ** 3, window_size ** 3))

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        if self.shifted:
            x = self.cyclic_shift(x)

        b, n_h, n_w, n_d, _, h = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        nw_h = n_h // self.window_size
        nw_w = n_w // self.window_size
        nw_d = n_d // self.window_size

        q, k, v = map(
            lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (nw_d w_d) (h d) -> b h (nw_h nw_w nw_d) (w_h w_w w_d) d',
                                h=h, w_h=self.window_size, w_w=self.window_size, w_d=self.window_size), qkv)

        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale

        if self.relative_pos_embedding:
            rel_pos_index_1 = self.relative_indices[:, :, 0]
            rel_pos_index_2 = self.relative_indices[:, :, 1]
            rel_pos_index_3 = self.relative_indices[:, :, 2]
            dots += self.pos_embedding[rel_pos_index_1, rel_pos_index_2, rel_pos_index_3]
        else:
            dots += self.pos_embedding

        if self.shifted:
            # Add masks for different spatial directions
            dots[:, :, -nw_w * nw_d:] += self.masks['upper_lower']
            dots[:, :, nw_w * nw_d - 1::nw_w * nw_d] += self.masks['left_right']
            dots[:, :, nw_w - 1::nw_w] += self.masks['depth']

        attn = dots.softmax(dim=-1)

        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)
        out = rearrange(out, 'b h (nw_h nw_w nw_d) (w_h w_w w_d) d -> b (nw_h w_h) (nw_w w_w) (nw_d w_d) (h d)',
                        h=h, w_h=self.window_size, w_w=self.window_size, w_d=self.window_size,
                        nw_h=nw_h, nw_w=nw_w, nw_d=nw_d)
        out = self.to_out(out)

        if self.shifted:
            out = self.cyclic_back_shift(out)
        return out


class SwinBlock3D(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        self.attention_block = Residual(PreNorm(dim, WindowAttention3D(dim=dim,
                                                                       heads=heads,
                                                                       head_dim=head_dim,
                                                                       shifted=shifted,
                                                                       window_size=window_size,
                                                                       relative_pos_embedding=relative_pos_embedding)))
        self.mlp_block = Residual(PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim)))

    def forward(self, x):
        x = self.attention_block(x)
        x = self.mlp_block(x)
        return x


class PatchMerging3D(nn.Module):
    def __init__(self, in_channels, out_channels, downscaling_factor):
        super().__init__()
        self.downscaling_factor = downscaling_factor
        self.patch_merge = nn.Unfold(kernel_size=downscaling_factor, stride=downscaling_factor, padding=0)
        self.linear = nn.Linear(in_channels * downscaling_factor ** 3, out_channels)

    def forward(self, x):
        b, c, h, w, d = x.shape
        new_h, new_w, new_d = h // self.downscaling_factor, w // self.downscaling_factor, d // self.downscaling_factor
        x = self.patch_merge(x.view(b * c, 1, h, w, d)).view(b, -1, new_h, new_w, new_d).permute(0, 2, 3, 4, 1)
        x = self.linear(x)
        return x


class StageModule3D(nn.Module):
    def __init__(self, in_channels, hidden_dimension, layers, downscaling_factor, num_heads, head_dim, window_size,
                 relative_pos_embedding):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

        self.patch_partition = PatchMerging3D(in_channels=in_channels, out_channels=hidden_dimension,
                                              downscaling_factor=downscaling_factor)

        self.layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.layers.append(nn.ModuleList([
                SwinBlock3D(dim=hidden_dimension, heads=num_heads, head_dim=head_dim,
                            mlp_dim=hidden_dimension * 4, shifted=False, window_size=window_size,
                            relative_pos_embedding=relative_pos_embedding),
                SwinBlock3D(dim=hidden_dimension, heads=num_heads, head_dim=head_dim,
                            mlp_dim=hidden_dimension * 4, shifted=True, window_size=window_size,
                            relative_pos_embedding=relative_pos_embedding),
            ]))

    def forward(self, x):
        x = self.patch_partition(x)
        for regular_block, shifted_block in self.layers:
            x = regular_block(x)
            x = shifted_block(x)
        return x.permute(0, 4, 1, 2, 3)


class SwinTransformer3D(nn.Module):
    def __init__(self, *, hidden_dim, layers, heads, channels=1, num_classes=1000, head_dim=32,
                 window_size=7, downscaling_factors=(4, 2, 2, 2), relative_pos_embedding=True):
        super().__init__()

        self.stage1 = StageModule3D(in_channels=channels, hidden_dimension=hidden_dim, layers=layers[0],
                                    downscaling_factor=downscaling_factors[0], num_heads=heads[0],
                                    head_dim=head_dim, window_size=window_size,
                                    relative_pos_embedding=relative_pos_embedding)
        self.stage2 = StageModule3D(in_channels=hidden_dim, hidden_dimension=hidden_dim * 2, layers=layers[1],
                                    downscaling_factor=downscaling_factors[1], num_heads=heads[1],
                                    head_dim=head_dim, window_size=window_size,
                                    relative_pos_embedding=relative_pos_embedding)
        self.stage3 = StageModule3D(in_channels=hidden_dim * 2, hidden_dimension=hidden_dim * 4, layers=layers[2],
                                    downscaling_factor=downscaling_factors[2], num_heads=heads[2],
                                    head_dim=head_dim, window_size=window_size,
                                    relative_pos_embedding=relative_pos_embedding)
        self.stage4 = StageModule3D(in_channels=hidden_dim * 4, hidden_dimension=hidden_dim * 8, layers=layers[3],
                                    downscaling_factor=downscaling_factors[3], num_heads=heads[3],
                                    head_dim=head_dim, window_size=window_size,
                                    relative_pos_embedding=relative_pos_embedding)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 8),
            nn.Linear(hidden_dim * 8, num_classes)
        )

    def forward(self, vol):
        x = self.stage1(vol)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = x.mean(dim=[2, 3, 4])
        return self.mlp_head(x)


def swin_3d_t(hidden_dim=96, layers=(2, 2, 6, 2), heads=(3, 6, 12, 24), **kwargs):
    return SwinTransformer3D(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)


def swin_3d_s(hidden_dim=96, layers=(2, 2, 18, 2), heads=(3, 6, 12, 24), **kwargs):
    return SwinTransformer3D(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)


def swin_3d_b(hidden_dim=128, layers=(2, 2, 18, 2), heads=(4, 8, 16, 32), **kwargs):
    return SwinTransformer3D(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)


def swin_3d_l(hidden_dim=192, layers=(2, 2, 18, 2), heads=(6, 12, 24, 48), **kwargs):
    return SwinTransformer3D(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)