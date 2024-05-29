from denoising_diffusion_pytorch.denoising_diffusion_pytorch_fracture import Unet_f, GaussianDiffusion_f, Trainer_f
import argparse
import os

parser = argparse.ArgumentParser(description="Model configurations and paths")
# For UNET
parser.add_argument('--dim', type=int, default=16, help='Dim for UNET')
# For diffusion_model
parser.add_argument('--timesteps', type=int, default=49, help='Timesteps for GaussianDiffusion')
parser.add_argument('--objective', type=str, default='pred_noise', choices=['pred_v', 'pred_x0', 'pred_noise'], help='Objective for GaussianDiffusion')
parser.add_argument('--beta_schedule', type=str, default='sigmoid', choices=['cosine', 'sigmoid'], help='Beta schedule for GaussianDiffusion')
parser.add_argument('--loss_fn', type=str, default='l2', choices=['l2', 'l1'], help='Loss function')
# Paths
parser.add_argument('--data_dir', type=str, default='/diffusion_input/MultiphaseFluids_in_Fractures', help='Directory containing the data')
parser.add_argument('--results_dir', type=str, default='/diffusion_output/test/', help='Directory containing the data')
parser.add_argument('--sampling_data_dir', type=str, default='/diffusion_sampling_dataset/fractures', help='Directory containing the data')

parser.add_argument('--batch_size', type=int, default=16, help='Batch size for processing')
parser.add_argument('--sample_num', type=int, default=4, help='number of sampling images')
args = parser.parse_args([])

args.data_dir = os.getcwd() + args.data_dir
args.results_dir = os.getcwd() + args.results_dir
args.sampling_data_dir = os.getcwd() + args.sampling_data_dir

model = Unet_f(
    dim = args.dim,
    init_dim = None,
    out_dim = 1,    
    dim_mults = (1, 2, 4, 8),
    flash_attn = False, #True, #True,
    channels = 2, #2,
    self_condition = False,
    resnet_block_groups = 8,
    learned_variance = False,
    learned_sinusoidal_cond = False,
    random_fourier_features = False,
    learned_sinusoidal_dim = 16,
    full_attn = (False, False, False, True),
)

diffusion = GaussianDiffusion_f(
    model,
    image_size = 128,
    timesteps = args.timesteps, # number of steps
    sampling_timesteps = args.timesteps//2, # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    objective = args.objective,
    beta_schedule = args.beta_schedule,
    loss = args.loss_fn,
    schedule_fn_kwargs = dict(),
    ddim_sampling_eta = 0.,
    auto_normalize = True,
    offset_noise_strength = 0.,  # https://www.crosslabs.org/blog/diffusion-with-offset-noise
    min_snr_loss_weight = False, # https://arxiv.org/abs/2303.09556
    min_snr_gamma = 5    
)
