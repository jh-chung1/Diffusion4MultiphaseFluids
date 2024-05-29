import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
from torch import nn, einsum
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam

from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator

from denoising_diffusion_pytorch.attend import Attend
from denoising_diffusion_pytorch.fid_evaluation import FIDEvaluation

from denoising_diffusion_pytorch.version import __version__

# added packages
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import subprocess

# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cast_tuple(t, length = 1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

def divisible_by(numer, denom):
    return (numer % denom) == 0

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# gradient calculation function to emphasize fracture geometry in loss fn
def gradient_difference_loss(pred, true):
    # Compute the gradient in x direction
    pred_grad_x = pred - F.pad(pred[:, :, :, :-1], (1, 0))
    true_grad_x = true - F.pad(true[:, :, :, :-1], (1, 0))

    # Compute the gradient in y direction
    pred_grad_y = pred - F.pad(pred[:, :, :-1, :], (0, 0, 1, 0))
    true_grad_y = true - F.pad(true[:, :, :-1, :], (0, 0, 1, 0))

    # Combine the gradients for x and y directions
    pred_grad = torch.abs(pred_grad_x) + torch.abs(pred_grad_y)
    true_grad = torch.abs(true_grad_x) + torch.abs(true_grad_y)

    # Return the mean squared error of the gradients
    return F.mse_loss(pred_grad, true_grad)

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

def normalize_to_range(img, min_val, max_val):
    """
    Normalize image tensor values to [min_val, max_val]
    """
    img_normalized = (img - img.min()) / (img.max() - img.min())
    return img_normalized * (max_val - min_val) + min_val


# small helper modules

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)       

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        flash = False
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.attend = Attend(flash = flash)

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.heads), qkv)

        out = self.attend(q, k, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

# model

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults = (1, 2, 4, 8),
        channels = 3,
        self_condition = False,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        attn_dim_head = 32,
        attn_heads = 4,
        full_attn = (False, False, False, True),
        flash_attn = False
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # attention

        num_stages = len(dim_mults)
        full_attn  = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        assert len(full_attn) == len(dim_mults)

        FullAttention = partial(Attention, flash = flash_attn)

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = ind >= (num_resolutions - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                attn_klass(dim_in, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = FullAttention(mid_dim, heads = attn_heads[-1], dim_head = attn_dim_head[-1])
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            is_last = ind == (len(in_out) - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                attn_klass(dim_out, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    def forward(self, x, time, x_self_cond = None):
        assert all([divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]), f'your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet'

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x) + x
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x) + x

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps = 1000,
        sampling_timesteps = None,
        objective = 'pred_v',
        beta_schedule = 'sigmoid',
        loss = 'l2',
        schedule_fn_kwargs = dict(),
        ddim_sampling_eta = 0.,
        auto_normalize = True,
        offset_noise_strength = 0.,  # https://www.crosslabs.org/blog/diffusion-with-offset-noise
        min_snr_loss_weight = False, # https://arxiv.org/abs/2303.09556
        min_snr_gamma = 5
    ):
        super().__init__()
        #assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        assert not model.random_or_learned_sinusoidal_cond

        self.model = model

        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        self.image_size = image_size

        self.objective = objective
        self.loss = loss

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # offset noise strength - in blogpost, they claimed 0.1 was ideal

        self.offset_noise_strength = offset_noise_strength

        # derive loss weight
        # snr - signal noise ratio

        snr = alphas_cumprod / (1 - alphas_cumprod)

        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        if objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    @property
    def device(self):
        return self.betas.device

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, mask, t, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False):
        #print('x input', x.shape, 'mask shape', mask.shape, 'model input shape', torch.cat([x, mask], dim=1).shape)
        #x = torch.cat([x, mask], dim=1)
        
        #print(f'x.shape {x.shape}, mask.shape {mask.shape}')
        
        model_output = self.model(torch.cat([x, mask], dim=1), t, x_self_cond)
        #print('model output shape 1', model_output.shape)
        #print(f't: {t[0]}')
#         if t[0] % 10 == 0:
#             plt.figure(dpi=100)
#             plt.subplot(1,3,1)
#             plt.imshow(x[0,0,].cpu().detach().numpy())
#             plt.axis('off')
#             plt.title(f'x at {t[0]}')
#             plt.subplot(1,3,2)
#             plt.imshow(mask[0,0,].cpu().detach().numpy())
#             plt.title('mask')
#             plt.axis('off')
#             plt.subplot(1,3,3)
#             plt.imshow(model_output[0,0,].cpu().detach().numpy())
#             plt.axis('off')
#             plt.title(f'pred {self.objective}')
#                 #plt.close()
#             plt.show()
        
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            # print(f'2. In pred_noise, x.shape {x.shape}, {x[:,:-1].shape}, pred_noise, {pred_noise.shape}, t {t}')
            x_start = self.predict_start_from_noise(x, t, pred_noise)           
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, mask, t, x_self_cond = None, clip_denoised = True):
        # print('**p_mean_variance, x_start.shape', x.shape, 'mask shape', mask.shape, 't', t)
        preds = self.model_predictions(x, mask, t, x_self_cond)
        x_start = preds.pred_x_start


        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.inference_mode()
    def p_sample(self, x, mask, t: int, x_self_cond = None):
        # print(f'p_sample x.shape {x.shape}, self.device {self.device}')
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device = device, dtype = torch.long)
        
        # concatenate the input and the binary mask before getting predictions
        # x = torch.cat([x, mask], dim=1)
        
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, mask = mask, t = batched_times, x_self_cond = x_self_cond, clip_denoised = True)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        
        # print('noise shape', noise.shape)
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise

        ## 2D fracture & 2D porous medium
        # Ensure that pred_img has only one channel.
        #pred_img = pred_img[:, :1, :, :]        
        
        ## 
        # Ensure that pred_img has only 8 channels.
        pred_img = pred_img[:, :8, :, :]   
        
        # print('pred_img.shape', pred_img.shape, 'x_start.shape', x_start.shape)
        return pred_img, x_start


#    def p_sample_loop(self, shape, return_all_timesteps = False):
    @torch.inference_mode()
    def p_sample_loop(self, start_image, return_all_timesteps = False):        
        batch, device = start_image.shape[0], self.device

        # add binary channel
## 2D fracture
#         mask = (start_image != 0).float() 
#         img = torch.randn_like(start_image, device = device) #torch.randn(shape, device = device)
#         imgs = [img]

## 2D porous media
        sampling_size, h, w = start_image.shape  # start_image has shape (sampling_size, 128, 256)

        # Create the mask channel from start_image
        mask = (start_image != 0).float()  # Makes it binary, shape will be (sampling_size, 128, 256)
        mask = mask.unsqueeze(1)  # Adds the channel dimension back, resulting in (sampling_size, 1, 128, 256)

        # Create random noise tensor with 8 channels
        noise_shape = (sampling_size, 8, h, w)
        img = torch.randn(noise_shape, device=self.device)  # Shape will be (sampling_size, 8, 128, 256)

        # Concatenate mask and noise to form img with 9 channels
        #img = torch.cat([img_noise, mask], dim=1)  # Shape will be (sampling_size, 9, 128, 256)
        #print(f'start_image shape {start_image.shape}, mask shape {mask.shape}, img shape {img.shape}')
        imgs = [img]
       

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            # print('In p_sample loop, t', t, 'img.shape', img.shape, 'mask shape', mask.shape)
            img, x_start = self.p_sample(img, mask, t, self_cond)
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def ddim_sample(self, shape, return_all_timesteps = False):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start = True, rederive_pred_noise = True)

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def sample(self, start_image, return_all_timesteps = False): #batch_size = 16, return_all_timesteps = False):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop #if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn(start_image, return_all_timesteps = return_all_timesteps)
        #return sample_fn((batch_size, channels, image_size, image_size), return_all_timesteps = return_all_timesteps)

    @torch.inference_mode()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    @autocast(enabled = False)
    def q_sample(self, x_start, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, noise = None, offset_noise_strength = None):
        b, c, h, w = x_start.shape

        noise = default(noise, lambda: torch.randn_like(x_start))

        # offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise

        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)

        if offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start.shape[:2], device = self.device)
            noise += offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')

        # noise sample
        x = self.q_sample(x_start = x_start, t = t, noise = noise)
        
        # add binary channel
        ## 2D fracture
        #mask = (x_start != 0).float() 
        
        ## 2D porous media
        mask_channel = x_start[:, 0, :, :]  # Picks the first channel from every image in the batch
        mask = (mask_channel != 0).float().unsqueeze(1)  # Makes it binary and adds the channel dimension back, resulting in [20, 1, 128, 256]
        
        x = torch.cat([x, mask], dim = 1)
        

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.inference_mode():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step
        model_out = self.model(x, t, x_self_cond)
        #print(f'1. Original model output shape: {model_out.shape}')


        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')
            
         # Restricting loss computation to only the original channels
        model_out = model_out[:, :c]
        target = target[:, :c]    
        #loss = F.mse_loss(model_out, target, reduction = 'none') #mse_loss(model_out, target, reduction = 'none')
        
        if self.loss == 'l2':
            fluid_loss = mask*F.mse_loss(model_out, target, reduction = 'none')
            solid_loss = (1-mask)*F.mse_loss(model_out, target, reduction = 'none')
        else:
            fluid_loss = mask*F.l1_loss(model_out, target, reduction = 'none')
            solid_loss = (1-mask)*F.l1_loss(model_out, target, reduction = 'none')
            
        loss = fluid_loss + solid_loss

        
        # trial 1) adding more loss term for geometry
#         # compute mask loss
#         mask_out = model_out[:, c-1]  # assuming last channel is the mask channel
#         mask_loss = F.mse_loss(mask_out, mask.squeeze(1), reduction='none')
#         mask_loss = reduce(mask_loss, 'b ... -> b (...)', 'mean')
#         lambda_mask = 10  # for example, adjust this value based on your needs
#         loss = loss + lambda_mask * mask_loss

        # trial 2) adding graident loss term for geometry
        # Extract only the mask part from the model_out for edge-aware losses
#         model_out_mask = model_out[:, c-1:c]
#         mask = (x_start != 0).float()#.unsqueeze(1)
#         #print('model out mask shape', model_out_mask.shape, 'mask', mask.shape)
#         # Compute the edge-aware losses
#         gdl_loss = gradient_difference_loss(model_out, target)
#         loss = loss + gdl_loss
        
        
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss * extract(self.loss_weight, t, loss.shape)
    
        return loss.mean()

#     def forward(self, img, *args, **kwargs):
#         print(f' img.shape {*img.shape}, img.device {img.device}, self.image_size {self.image_size}')
#         b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
#         #assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
#         t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

#         img = self.normalize(img)
#         return self.p_losses(img, t, *args, **kwargs)

    def forward(self, img, *args, **kwargs):
        #print(f'img.shape {img.shape}, img.device {img.device}, self.image_size {self.image_size}')

        b, c, h, w = img.shape
        device = img.device
        img_size = self.image_size

        #assert (h, w) == torch.Tensor(img_size), f'Height and width of image must be {img_size}'

        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        return self.p_losses(img, t, *args, **kwargs)


# dataset classes

class Dataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts = ['jpg', 'jpeg', 'png', 'tiff', 'npy'],
        augment_horizontal_flip = False,
        convert_image_to = None
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        ext = path.suffix.lower()[1:]  # get the file extension
        
        if ext == 'npy':
            img = np.load(path)
            #img = torch.Tensor(np.expand_dims(img, axis=0)) ## 2D fracture
            img = torch.Tensor(img) ## 2D porous media
        else:
            img = Image.open(path)
            img = self.transform(img)        
        return img

# trainer class

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        folder,
        *,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        augment_horizontal_flip = True,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        sampling_dataset_folder = '/scratch/users/jhchung1/denoising-diffusion-pytorch_v2/diffusion_sampling_dataset/mean_aperture_25',
        save_and_sample_every = 1000,
        num_samples = 25,
        results_folder = './results',
        amp = False,
        mixed_precision_type = 'fp16',
        split_batches = True,
        convert_image_to = None,
        calculate_fid = True,
        inception_block_idx = 2048,
        max_grad_norm = 1.,
        num_fid_samples = 50000,
        save_best_and_latest_only = False
    ):
        super().__init__()

        # accelerator

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no'
        )

        # model

        self.model = diffusion_model
        self.channels = diffusion_model.channels
        is_ddim_sampling = diffusion_model.is_ddim_sampling

        # sampling and training hyperparameters

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        assert (train_batch_size * gradient_accumulate_every) >= 16, f'your effective batch size (train_batch_size x gradient_accumulate_every) should be at least 16 or above'

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        self.max_grad_norm = max_grad_norm

        # dataset and dataloader

        self.ds = Dataset(folder, self.image_size, augment_horizontal_flip = augment_horizontal_flip, convert_image_to = convert_image_to)

        #assert len(self.ds) >= 100, 'you should have at least 100 images in your folder. at least 10k images recommended'

        dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = 1)# cpu_count())
        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        ## load test dataset (binary geometry)
        self.sampling_dataset_folder = sampling_dataset_folder
        ds_test = Dataset(self.sampling_dataset_folder, self.image_size, augment_horizontal_flip = augment_horizontal_flip, convert_image_to = convert_image_to)
        dl_test = DataLoader(ds_test, batch_size = self.num_samples, shuffle = True, pin_memory = True, num_workers = 1)
        dl_test = self.accelerator.prepare(dl_test)
        self.dl_test_cycle = cycle(dl_test)
        # print('test load done')
        
        # optimizer
        #self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)
        
        # optimizer with learning rate adjuster
        self.opt = Adam(diffusion_model.parameters(), lr=train_lr, betas=adam_betas)
        self.scheduler = ReduceLROnPlateau(self.opt, mode='min', factor=0.95, patience=50, verbose=True)




        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        # FID-score computation

        self.calculate_fid = calculate_fid and self.accelerator.is_main_process

        if self.calculate_fid:
            if not is_ddim_sampling:
                self.accelerator.print(
                    "WARNING: Robust FID computation requires a lot of generated samples and can therefore be very time consuming."\
                    "Consider using DDIM sampling to save time."
                )
            self.fid_scorer = FIDEvaluation(
                batch_size=self.batch_size,
                dl=self.dl,
                sampler=self.ema.ema_model,
                channels=self.channels - 1, # minus geometry channel
                accelerator=self.accelerator,
                stats_dir=results_folder,
                device=self.device,
                num_fid_samples=num_fid_samples,
                inception_block_idx=inception_block_idx
            )

        if save_best_and_latest_only:
            assert calculate_fid, "`calculate_fid` must be True to provide a means for model evaluation for `save_best_and_latest_only`."
            self.best_fid = 1e10 # infinite

        self.save_best_and_latest_only = save_best_and_latest_only

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])
            
    def create_and_submit_slurm(self, sample_img_dir, num_samples):
        # function for submitting simulation jobs 

        ## 2D porous media
        sim_num = num_samples - 1
        sample_dir = f'{sample_img_dir}/'
        output_dir = f'{sample_dir}DNS_results/'

        os.makedirs(output_dir, exist_ok=True)

        # Create/open bash file
        file = open(f'{output_dir}job_sampling_sims.sh', 'w+')
        file.write('#!/bin/bash \n\n')

        file.write('#SBATCH --time=48:00:00 \n')
        file.write("#SBATCH --job-name='D2DNS' \n")
        file.write('#SBATCH -p serc \n')
        file.write('#SBATCH -n 50 \n')
        file.write(f'#SBATCH --output={output_dir}job-%A.out \n')
        file.write(f'#SBATCH --error={output_dir}job-%A.err \n\n')

        file.write('declare -a fluid_init_configurations=("Diffusion" "geometric" "leftfill" "random") \n\n') #"leftfill"

        file.write('ml openmpi py-numba/0.54.1_py39 \n')
        file.write(f'sample_dir={sample_dir}\n')

        file.write('for fluid_init in "${fluid_init_configurations[@]}"; do \n')        
        file.write(f'    for i in {{0..{sim_num}}}; do \n')
        file.write('        sample_file="sample_${i}.npy" \n')
        file.write('        input_dir="${sample_dir}sample_${i}/" \n')
        file.write('        output_dir="${sample_dir}DNS_results/sample_${i}_Fluid_init_${fluid_init}/" \n')

        # Condition to run different simulation files
        file.write('        if [ "$fluid_init" == "Diffusion" ]; then \n')
        # fracture
        file.write('        /home/groups/mukerji/jhchung1/miniconda3/envs/MPLBM-UT/bin/python /scratch/users/jhchung1/MPLBM-UT/examples/steady_state_relperm/2_phase_sim_Diffusion_Output2DNS_sim_fracture.py --input_dir "$input_dir" --output_dir "$output_dir" --file_name "$sample_file" --num_procs 50 --fluid_init_configuration "$fluid_init" --convergence_relperm 1e-4 --convergence_iter 100 --raw_geometry_size 128 5 128 --sim_geometry_size 128 5 128 --DNS2DNS True --input_dir_4_DNS2DNS "$input_dir" \n')
#         file.write('            /home/groups/mukerji/jhchung1/miniconda3/envs/MPLBM-UT/bin/python /scratch/users/jhchung1/MPLBM-UT/examples/steady_state_relperm/2_phase_sim_JC_v5.py --input_dir "$input_dir" --output_dir "$output_dir" --file_name "geometry.raw" --num_procs 50 --raw_geometry_size 128 1 128 --sim_geometry_size 128 5 128 --DNS2DNS True --input_dir_4_DNS2DNS "$input_dir" \n')        
        # porous media
        #file.write('            /home/groups/mukerji/jhchung1/miniconda3/envs/MPLBM-UT/bin/python /scratch/users/jhchung1/MPLBM-UT/examples/steady_state_relperm/2_phase_sim_JC_v5.py --input_dir "$input_dir" --output_dir "$output_dir" --file_name "geometry.raw" --num_procs 50 --raw_geometry_size 128 1 256 --sim_geometry_size 128 5 256 --DNS2DNS True --input_dir_4_DNS2DNS "$input_dir" \n')
        file.write('        else \n')
        file.write('        /home/groups/mukerji/jhchung1/miniconda3/envs/MPLBM-UT/bin/python /scratch/users/jhchung1/MPLBM-UT/examples/steady_state_relperm/2_phase_sim_Diffusion_Output2DNS_sim_fracture.py --input_dir "$input_dir" --output_dir "$output_dir" --file_name "$sample_file" --num_procs 50 --fluid_init_configuration "$fluid_init" --convergence_relperm 1e-4 --convergence_iter 100 --raw_geometry_size 128 5 128 --sim_geometry_size 128 5 128 \n')     
        #file.write('            /home/groups/mukerji/jhchung1/miniconda3/envs/MPLBM-UT/bin/python /scratch/users/jhchung1/MPLBM-UT/examples/steady_state_relperm/2_phase_sim_Diffusion_Output2DNS_sim.py --input_dir "$input_dir" --output_dir "$output_dir" --file_name "$sample_file" --num_procs 50 --fluid_init_configuration "$fluid_init"  --sim_geometry_size 128 5 256 \n')
        file.write('        fi \n')

        file.write('        # Check if fluid_init is Diffusion and run DNS2DNS\n')
        file.write('        if [ "$fluid_init" == "Diffusion" ]; then\n')
        file.write('            output_DNS2DNS_dir=${sample_dir}DNS_results/sample_${i}_DNS2DNS/ \n')
        file.write('            /home/groups/mukerji/jhchung1/miniconda3/envs/MPLBM-UT/bin/python /scratch/users/jhchung1/MPLBM-UT/examples/steady_state_relperm/2_phase_sim_JC_v5.py --input_dir "$input_dir" --output_dir "$output_DNS2DNS_dir" --file_name "geometry.raw" --num_procs 50 --raw_geometry_size 128 5 128 --sim_geometry_size 128 5 128 --DNS2DNS True --input_dir_4_DNS2DNS "$output_dir" \n')        
#        file.write('            /home/groups/mukerji/jhchung1/miniconda3/envs/MPLBM-UT/bin/python /scratch/users/jhchung1/MPLBM-UT/examples/steady_state_relperm/2_phase_sim_JC_v5.py --input_dir "$input_dir" --output_dir "$output_DNS2DNS_dir" --file_name "geometry.raw" --num_procs 50 --raw_geometry_size 128 1 256 --sim_geometry_size 128 5 256 --DNS2DNS True --input_dir_4_DNS2DNS "$output_dir" \n')
        file.write('        fi\n')

        file.write('    done \n')
        file.write('done')

        # Close the file
        file.close()

        
        
        ## 2D fracture
#         sim_num = num_samples - 1
#         sample_dir = f'{sample_img_dir}/'
#         output_dir = f'{sample_dir}DNS_results/'
        
#         os.makedirs(output_dir, exist_ok=True)


# #         # Create/open bash file
#         file = open(f'{output_dir}job_sampling_sims.sh', 'w+')
#         file.write('#!/bin/bash \n\n')
        
#         file.write('#SBATCH --time=48:00:00 \n')
#         file.write("#SBATCH --job-name='D2DNS' \n")
#         file.write('#SBATCH -p serc \n')
#         file.write('#SBATCH -n 50 \n')
#         file.write(f'#SBATCH --output={output_dir}job-%A.out \n')
#         file.write(f'#SBATCH --error={output_dir}job-%A.err \n\n')
        
#         file.write('declare -a fluid_init_configurations=("Diffusion" "geometric" "random") \n\n') #"leftfill" 
 
#         file.write('ml openmpi py-numba/0.54.1_py39 \n')
#         file.write(f'input_dir={sample_dir}\n')

#         file.write('for fluid_init in "${fluid_init_configurations[@]}"; do \n')        
#         file.write(f'    for i in {{0..{sim_num}}}; do \n')
#         file.write('        sample_file="sample_${i}.npy" \n')
#         file.write('        output_dir=${input_dir}DNS_results/sample_${i}_Fluid_init_${fluid_init}/ \n')
        
#         file.write('        /home/groups/mukerji/jhchung1/miniconda3/envs/MPLBM-UT/bin/python /scratch/users/jhchung1/MPLBM-UT/examples/steady_state_relperm/2_phase_sim_Diffusion_Output2DNS_sim.py --input_dir "$input_dir" --output_dir "$output_dir" --file_name "$sample_file" --num_procs 50 --fluid_init_configuration "$fluid_init" \n')

#         file.write('        # Check if fluid_init is Diffusion and run DNS2DNS\n')
#         file.write('        if [ "$fluid_init" == "Diffusion" ]; then\n')
#         file.write('            output_DNS2DNS_dir=${input_dir}DNS_results/sample_${i}_DNS2DNS/ \n')
#         file.write('            /home/groups/mukerji/jhchung1/miniconda3/envs/MPLBM-UT/bin/python /scratch/users/jhchung1/MPLBM-UT/examples/steady_state_relperm/2_phase_sim_DNS2DNS.py --input_dir "$output_dir" --output_dir "$output_DNS2DNS_dir" --frac_num ${i} --num_procs 50\n')
#         file.write('        fi\n')

        
#         file.write('    done \n')
#         file.write('done')
        
#         # save DNS info. python file
          #file.write(f'    /home/groups/mukerji/jhchung1/miniconda3/envs/MPLBM-UT/bin/python /scratch/users/jhchung1/denoising-diffusion-pytorch_v2/Save_Diffusion2DNS.py --sim_dir "$input_dir" --num_sim {sim_num}')
        
        
#         # input num_sim, input_dir = output/sim_output.txt for each sim case, output_dir = sample_img_dir or sample_dir
       
        file.close()
        
        # Modify script permissions to make it executable
        subprocess.run(["chmod", "+x", f'{output_dir}job_sampling_sims.sh'])

# # #     # Submit the script
        subprocess.run(["sbatch", f'{output_dir}job_sampling_sims.sh'])

    # Define a custom formatter
    def custom_formatter(x):
        # Check if x is an integer or a float with non-zero decimals
        if x % 1:
            # If x has trailing zeros, format to remove them
            return '{:g}'.format(float('{:.3f}'.format(x)))
        else:
            # If x is an integer, format without decimal points
            return '{:.0f}'.format(x) 

    
    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        losses_per_timestep = [] # loss for each timestep

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(device)

                    with self.accelerator.autocast():
                        loss = self.model(data)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()
                        losses_per_timestep.append(loss.item())

                    self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()

                self.opt.step()
                self.scheduler.step(total_loss)  # Update the scheduler with the recent loss
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and divisible_by(self.step, self.save_and_sample_every):
                        self.ema.ema_model.eval()

                        with torch.inference_mode():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            
                            #print(f'milestone: {milestone}, batches: {batches}') 

                            test_batch = next(self.dl_test_cycle).to(device)
                            
                          
                            all_images_list = [self.ema.ema_model.sample(start_image=test_batch[:n]) for n in batches]

#                             all_images_list = list(map(lambda n: self.ema.ema_model.sample(start_image=test_batch[:self.num_samples]), batches))
                            
                            #all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))
                            
                            #print(f'all_images_list: {all_images_list}')

                        all_images = torch.cat(all_images_list, dim = 0)
                        #combined_start_sample_images = torch.stack([test_batch[:self.num_samples], all_images[:self.num_samples]], dim=1).view(-1, *a.shape[1:])
#                         combined_start_sample_images = torch.stack([test_batch[:self.num_samples], all_images[:self.num_samples]], dim=1).view(-1, *test_batch.shape[1:])

                        # ---------- save generated images and sim. info. for DNS input ----------
                        # print(f'all_images.shape {all_images.shape}')
                        # multiply with geometry to ensure the generated images have identical geometries even in early traning timestep

                        min_samples = min(test_batch.shape[0], all_images.shape[0])
           
                        sample_img = test_batch[:min_samples]*all_images[:min_samples,0]
                                
                        # Loop through samples
                        for sample_idx in range(all_images.shape[0]):
                            # make directory
                            sample_img_dir = f'{self.results_folder}/sample_img_timestep_{self.step}/sample_{sample_idx}/'
                            if not os.path.exists(sample_img_dir):
                                os.makedirs(sample_img_dir)
                                
                            # save raw file
                            geometry = test_batch[sample_idx].cpu().numpy()
                            np.save(f'{sample_img_dir}/sample_{sample_idx}_geometry.npy', geometry)
                            geometry_stack = np.repeat(geometry[np.newaxis, :, :], 5, axis=0)
                            geometry_stack = np.transpose(geometry_stack, (1, 0, 2))  # shape (128, 5, 256)

                            geometry_stack.flatten().astype('uint8').tofile(f'{sample_img_dir}/geometry.raw')

                            # Get the image tensor for this sample and move it to CPU
                            sample_tensor = all_images[sample_idx].cpu()

                            # For rho_1 and rho_2 (i == 0 or i == 4)
                            for i in [0, 4]:
                                img_slice = sample_tensor[i].numpy()*test_batch[sample_idx].cpu().numpy()#.astype('float32')  # shape (128, 256)

                                # Unnormalize
                                if i == 0:
                                    img_slice = -0.4 + ((img_slice + 1) / 2) * (2.1 - (-0.4))
                                elif i == 4:
                                    img_slice = 0 + ((img_slice + 1) / 2) * (2.5 - 0)
                                    
                                img_slice = img_slice*test_batch[sample_idx].cpu().numpy()#.astype('float32')  # shape (128, 256)                                    
                                np.save(f'{sample_img_dir}/sample_{sample_idx}.npy', img_slice*test_batch[sample_idx].cpu().numpy())

                                img_stack = np.repeat(img_slice[np.newaxis, :, :], 5, axis=0)  # shape (5, 128, 256)
                                img_stack = np.transpose(img_stack, (1, 0, 2))  # shape (128, 5, 256)
                                #np.savetxt(f'{sample_img_dir}/rho_f{1 if i == 0 else 2}_000.dat', img_stack.flatten().astype(np.float64), fmt='%.3e', delimiter=' ', newline=' ')
                                flattened_img_stack = img_stack.flatten().astype(np.float64)
                                formatted_elements = ['0' if el == 0 else ('{:.0e}'.format(el) if abs(el) < 1e-3 else '{:.3f}'.format(el).rstrip('0').rstrip('.')) for el in flattened_img_stack]
                                formatted_str = ' '.join(formatted_elements)

                                with open(f'{sample_img_dir}/rho_f{1 if i == 0 else 2}_000.dat', 'w') as f:
                                    f.write(formatted_str)
                                
                                # For saving rho
                                #np.savetxt(f'{sample_img_dir}/rho_f{1 if i == 0 else 2}_000.dat', img_stack.flatten().astype(np.float64), fmt=self.custom_formatter, delimiter=' ', newline=' ')

                                #np.savetxt(f'{sample_img_dir}/rho_f{1 if i == 0 else 2}_000.dat', img_stack.flatten())#.astype('float32'))
                                #img_stack.flatten().astype('float32').tofile(f'{sample_img_dir}/rho_f{1 if i == 0 else 2}_000.dat')

                                
                            # Define the ranges for unnormalizing
                            ranges = {
                                1: (0.1, -0.1),
                                5: (0.1, -0.1),
                                2: (1e-15, -1e-15),
                                6: (1e-15, -1e-15),
                            }

                            # For vel_1 and vel_2 (1 <= i <= 3 and 5 <= i <= 7)
                            for i_range, vel_type in [([1, 2, 3], '1'), ([5, 6, 7], '2')]:
                                img_slices = []
                                for i in i_range:
                                    normalized = sample_tensor[i].numpy() * test_batch[sample_idx].cpu().numpy()

                                    # Unnormalize the values based on the condition
                                    max_val, min_val = ranges.get(i, (0.3, -0.3))  # Default to (0.3, -0.3) if i not in ranges
                                    unnormalized = (normalized * (max_val - min_val)) + min_val
                                    unnormalized = unnormalized* test_batch[sample_idx].cpu().numpy()

                                    img_slices.append(unnormalized)

                                img_stack = np.stack(img_slices, axis=-1)  # shape (128, 256, 3)
                                img_stack = np.repeat(img_stack[np.newaxis, :, :, :], 5, axis=0)  # shape (5, 128, 256, 3)
                                img_stack = np.transpose(img_stack, (1, 0, 2, 3))  # shape (128, 5, 256, 3)
                                #np.savetxt(f'{sample_img_dir}/vel_f{vel_type}_000.dat', img_stack.flatten().astype(np.float64), fmt='%.18e', delimiter=' ', newline=' ') 


                                #Saving with reduced precision and without scientific notation
                                flattened_img_stack = img_stack.flatten().astype(np.float64)
                                formatted_elements = ['0' if el == 0 else ('{:.0e}'.format(el) if abs(el) < 1e-3 else '{:.3f}'.format(el).rstrip('0').rstrip('.')) for el in flattened_img_stack]
                                formatted_str = ' '.join(formatted_elements)

                                with open(f'{sample_img_dir}/vel_f{vel_type}_000.dat', 'w') as f:
                                    f.write(formatted_str)

                            
                        # Write a bash file and submit a job to run simulations in sampling geometries
                        sample_img_dir = f'{self.results_folder}/sample_img_timestep_{self.step}'
                        if not os.path.exists(sample_img_dir):
                            os.makedirs(sample_img_dir)                    
                        self.create_and_submit_slurm(sample_img_dir, all_images.shape[0])
                            
                        # -------------------------------------------------------------------------    
                        
                        # normalize image 
                        min_val_start = test_batch[:self.num_samples].min()
                        max_val_start = test_batch[:self.num_samples].max()
                        all_images = normalize_to_range(all_images, min_val_start, max_val_start)
                    

                        # 2D fracture
                        # combined_start_sample_images = torch.stack([test_batch[:min_samples],all_images[:min_samples,0]], dim=1).view(-1, *test_batch.shape[1:])
                        # utils.save_image(combined_start_sample_images, str(self.results_folder / f'sample-{milestone}.png'), nrow=int(math.sqrt(self.num_samples) * 2))
                        
                        # 2D porous media
                        # Select the first channel (or whichever channel you're interested in) from all_images
                        selected_all_images = all_images[:min_samples, 0]  # This should have a shape of [4, 128, 256], assuming min_samples is 4
                        
                        # Make sure the data type is float32
                        test_batch = test_batch.type(torch.FloatTensor)
                        selected_all_images = selected_all_images.type(torch.FloatTensor)

                        # Ensure the value range is [0, 1] 
                        test_batch = test_batch / 255.0 if test_batch.max().item() > 1.0 else test_batch
                        selected_all_images = selected_all_images / 255.0 if selected_all_images.max().item() > 1.0 else selected_all_images
                        
                        # Your stacking and reshaping code
                        combined_start_sample_images = torch.stack([test_batch[:min_samples], selected_all_images], dim=1)
                        flattened_images = combined_start_sample_images.view(-1, 1, *test_batch.shape[1:])

                        # Save the image
                        utils.save_image(flattened_images, str(self.results_folder / f'sample-{milestone}.png'), nrow=2)

#                         # Stack along a new dimension to create pairs
#                         combined_start_sample_images = torch.stack([test_batch[:min_samples], selected_all_images], dim=1)  # Shape will be [4, 2, 128, 256]

#                         # Flatten the tensor to have single images for each row, ready for save_image
#                         flattened_images = combined_start_sample_images.view(-1, *test_batch.shape[1:])  # Should have shape [8, 128, 256]

#                         # Save the image using torchvision's save_image function
#                         utils.save_image(flattened_images, str(self.results_folder / f'sample-{milestone}.png'), nrow=2)

                        
                        
                        #print(f'generated image shape: {all_images[:min_samples].shape}')


                        #combined_start_sample_images = torch.stack([test_batch[:self.num_samples], all_images[:self.num_samples]], dim=1).view(-1, *test_batch.shape[1:])

                        #utils.save_image(combined_start_sample_images, str(self.results_folder / f'sample-{milestone}.png'), nrow=int(math.sqrt(self.num_samples) * 2))
                        
                        plt.figure(figsize=(10, 5))
                        plt.plot(losses_per_timestep)
                        plt.xlabel('Time step')
                        plt.ylabel('Loss')
                        plt.title(f'Loss per Time Step')
                        plt.yscale('log')
                        plt.grid(True, which="both", ls="--", c='0.7')  # This ensures the grid appears properly on log scale
                        #plt.grid(True)
                        plt.tight_layout()
                        plt.savefig(str(self.results_folder / "loss_plot.png"))
                        plt.close()
                        #plt.show()
                        np.save(str(self.results_folder / "losses_per_timestep.npy"), np.array(losses_per_timestep))
                        

                        #utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow = int(math.sqrt(self.num_samples)))

                        # whether to calculate fid

                        if self.calculate_fid:
                            fid_score = self.fid_scorer.fid_score()
                            accelerator.print(f'fid_score: {fid_score}')
                        if self.save_best_and_latest_only:
                            if self.best_fid > fid_score:
                                self.best_fid = fid_score
                                self.save("best")
                            self.save("latest")
                        else:
                            self.save(milestone)

                pbar.update(1)
                
        np.save(str(self.results_folder / "losses_per_timestep.npy"), np.array(losses_per_timestep))
        accelerator.print('training complete')
