import torch
import numpy as np
import argparse
import os
from denoising_diffusion_pytorch.denoising_diffusion_pytorch_fracture import Unet_f, GaussianDiffusion_f

# Argument parser for model configurations and paths
parser = argparse.ArgumentParser(description="Model configurations and paths")
parser.add_argument('--dim', type=int, default=16, help='Dim for UNET')
parser.add_argument('--timesteps', type=int, default=99, help='Timesteps for GaussianDiffusion')
parser.add_argument('--objective', type=str, default='pred_noise', choices=['pred_v', 'pred_x0', 'pred_noise'], help='Objective for GaussianDiffusion')
parser.add_argument('--beta_schedule', type=str, default='cosine', choices=['cosine', 'sigmoid'], help='Beta schedule for GaussianDiffusion')
parser.add_argument('--loss_fn', type=str, default='l2', choices=['l2', 'l1'], help='Loss function')
parser.add_argument('--saved_model_dir', type=str, default='/diffusion_output/test/', help='Path to the trained model')
parser.add_argument('--model_details', type=str, default='Unet_dim_16_pred_pred_noise_beta_sche_sigmoid_Loss_fnl2_T_99/model-5.pt', help='Path to the trained model')
parser.add_argument('--sampling_data_dir', type=str, default='/diffusion_sampling_dataset/fractures', help='Directory containing the data')
parser.add_argument('--batch_size', type=int, default=2, help='Batch size for processing')

args = parser.parse_args([])

args.saved_model_dir = os.path.join(os.getcwd(), args.saved_model_dir)
args.sampling_data_dir = os.path.join(os.getcwd(), args.sampling_data_dir)

# Function to load the trained model
def load_model(model_path):
    model = Unet_f(
        dim=args.dim,
        out_dim=1,
        dim_mults=(1, 2, 4, 8),
        channels=2,
        resnet_block_groups=8,
    )

    diffusion_model = GaussianDiffusion_f(
        model,
        image_size=128,
        timesteps=args.timesteps,
        sampling_timesteps=args.timesteps // 2,
        objective=args.objective,
        beta_schedule=args.beta_schedule,
        loss=args.loss_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    diffusion_model.load_state_dict(checkpoint['model'])
    diffusion_model.eval()

    return diffusion_model, model

# Function to load test geometry
def load_geometry(path):
    img = np.load(path)
    img[np.isnan(img)] = 0
    img[img != 0] = 1
    img = torch.Tensor(np.expand_dims(img, axis=0)).unsqueeze(0)
    return img

# Function to run the model for testing
def run_model_test(model, start_image, return_all_timesteps):
    sample = model.p_sample_loop(start_image, return_all_timesteps=return_all_timesteps)
    return sample

# Load model
model, _ = load_model(os.path.join(args.saved_model_dir, args.model_details))

# Load and process geometries
data_dir = args.sampling_data_dir
masks = []
batch_size = args.batch_size

for frac_num in range(10, 10 + batch_size):
    geometry = load_geometry(f'{data_dir}/fracture_ap25_{frac_num:04d}.npy')
    mask = torch.tensor(geometry) if isinstance(geometry, np.ndarray) else geometry.clone().detach()
    masks.append(mask)

# Concatenate masks into a single tensor
masks_tensor = torch.cat(masks, dim=0)

# Run model
sample = run_model_test(model, masks_tensor, True)

print('Sample generated.')
