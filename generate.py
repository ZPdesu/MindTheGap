import torch
import numpy as np
import os
import dlib
from PIL import Image
import json
from pathlib import Path
import sys

from models.II2S import II2S
from e4e.e4e_projection import e4e_projection_im_path
from options.face_embed_options import face_opts
from utils.model_utils import google_drive_paths, download_weight
from utils.shape_predictor import align_face
from models.stylegan2.model import Generator
import torchvision
from torchvision.utils import save_image
from argparse import Namespace
from argparse import ArgumentParser
toPIL = torchvision.transforms.ToPILImage()



def main(args):

    # Download pre_trained models
    print('Download pre_trained models')
    os.makedirs('pretrained_models', exist_ok=True)
    style_ckpt = Path(args.style_img).stem + '.pt'
    if not os.path.exists(os.path.join('pretrained_models', style_ckpt)):
        if style_ckpt in google_drive_paths:
            download_weight(os.path.join('pretrained_models', style_ckpt))
        else:
            sys.exit('{} does not exist'.format(style_ckpt))


    # Load finetuned generator
    print('Load finetuned generator')
    generator = Generator(args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier).to(args.device)
    style_ckpt = torch.load(os.path.join('pretrained_models', style_ckpt), map_location=args.device)
    generator.load_state_dict(style_ckpt["g_ema"], strict=True)
    generator.eval()
    style_latent = style_ckpt["style_latent"]


    # Generate random latents
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    latent_avg = style_ckpt["latent_avg"]
    random_z = torch.randn(args.n_sample, 512, device=args.device)

    w_styles = generator.style(random_z)

    output_latents = args.truc * (w_styles - latent_avg) + latent_avg
    output_latents = output_latents.unsqueeze(1).repeat(1, 18, 1)
    output_latents[:, 7:, :] = style_latent[:, 7:, :]

    # Save generated images
    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)
    with torch.no_grad():
        outputs = generator([output_latents], input_is_latent=True)[0]
        nrows = int(args.n_sample ** 0.5)
        # nrows = args.n_sample
        save_image(
            outputs,
            os.path.join(output_folder, 'style_{}_sample_{}.png'.format(Path(args.style_img).stem, args.n_sample)),
            nrow=nrows,
            normalize=True,
            range=(-1, 1),
        )



if __name__ == "__main__":

    parser = ArgumentParser()

    # I/O arguments
    parser.add_argument('--style_img', type=str, default='titan_erwin.png',
                        help='Style image')
    parser.add_argument('--n_sample', type=int, default=4,
                        help='Number of generated images')
    parser.add_argument('--truc', type=float, default=0.5,
                        help='Truncation')
    parser.add_argument('--output_folder', type=str, default='output/generate')
    args = parser.parse_args()
    args = Namespace(**vars(args), **vars(face_opts))

    main(args)