from denoising_diffusion_pytorch.denoising_diffusion_pytorch_fracture import Unet_f, GaussianDiffusion_f, Trainer_f
import argparse

parser = argparse.ArgumentParser(description="Model configurations and paths")

# For UNET
parser.add_argument('--dim', type=int, default=64, help='Dim for UNET')

# For diffusion_model
parser.add_argument('--timesteps', type=int, default=100, help='Timesteps for GaussianDiffusion')
parser.add_argument('--train_num_steps', type=int, default=10000, help='total training steps')
parser.add_argument('--save_and_sample_every', type=int, default=1000, help='Timesteps for GaussianDiffusion')
parser.add_argument('--num_samples', type=int, default=25, help='number of sampling., simulation number')
parser.add_argument('--objective', type=str, default='pred_noise', choices=['pred_v', 'pred_x0', 'pred_noise'], help='Objective for GaussianDiffusion')
parser.add_argument('--beta_schedule', type=str, default='cosine', choices=['cosine', 'sigmoid', 'linear'], help='Beta schedule for GaussianDiffusion')
parser.add_argument('--loss_fn', type=str, default='l2', choices=['l2', 'l1'], help='Loss function')

# Paths
## fracture
parser.add_argument('--data_dir', type=str, default='/scratch/users/jhchung1/denoising-diffusion-pytorch_v2/diffusion_input/fracture_ap25_sat_0.2_0.6', help='Directory containing the data')
parser.add_argument('--sampling_data_dir', type=str, default='/scratch/users/jhchung1/denoising-diffusion-pytorch_v2/diffusion_sampling_dataset/fracture_ap25_connected_sat_0.2_0.6', help='Directory containing the data')
parser.add_argument('--results_dir', type=str, default='/scratch/users/jhchung1/denoising-diffusion-pytorch_v2/diffusion_output/dfn4/', help='Directory containing the data')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for processing')
args = parser.parse_args()


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
    image_size = 128,  # 128,
    timesteps = args.timesteps, #500, #1000,           # number of steps
    sampling_timesteps = args.timesteps//2,    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    objective = args.objective, #'pred_v',
    beta_schedule = args.beta_schedule, #'sigmoid',
    loss = args.loss_fn,
    schedule_fn_kwargs = dict(),
    ddim_sampling_eta = 0.,
    auto_normalize = False,
    offset_noise_strength = 0.,  # https://www.crosslabs.org/blog/diffusion-with-offset-noise
    min_snr_loss_weight = False, # https://arxiv.org/abs/2303.09556
    min_snr_gamma = 5    
)


trainer = Trainer_f(
    diffusion,
    args.data_dir, #fracture_ap25_filtered_disconnected_nonWf',
   # '/scratch/users/jhchung1/denoising-diffusion-pytorch/diffusion_input/mean_apertures_32',
    train_batch_size = args.batch_size,
    train_lr = 1e-4, #8e-5,
    train_num_steps = args.train_num_steps,#0 # 150000         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_update_every = 10,
    ema_decay = 0.995,                # exponential moving average decay
    amp =True,                       # turn on mixed precision
    calculate_fid = False, #True,              # whether to calculate fid during training
    augment_horizontal_flip = True,  
    adam_betas = (0.9, 0.99),
    sampling_dataset_folder = args.sampling_data_dir,
    save_and_sample_every = args.save_and_sample_every, # 10000
    num_samples = args.num_samples, # 30? 50? 
    results_folder = args.results_dir + f'Unet_dim_{args.dim}_{args.objective}_beta_sche_{args.beta_schedule}_Loss_fn{args.loss_fn}_batch_{args.batch_size}_T_{args.timesteps}_Total_timesteps_{args.train_num_steps}_save_and_sample_every_{args.save_and_sample_every}', 
    mixed_precision_type = 'fp16',
    split_batches = True,
    convert_image_to = None,
    inception_block_idx = 2048,
    max_grad_norm = 1.,
    num_fid_samples = 50000,
    save_best_and_latest_only = False
)

trainer.train()
