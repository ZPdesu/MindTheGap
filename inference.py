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
from argparse import Namespace
from argparse import ArgumentParser

toPIL = torchvision.transforms.ToPILImage()



def main(args):

    # Download pre_trained models
    print('Load/Download pre_trained models')
    os.makedirs('pretrained_models', exist_ok=True)
    style_ckpt = Path(args.style_img).stem + '.pt'
    if not os.path.exists(os.path.join('pretrained_models', style_ckpt)):
        if style_ckpt in google_drive_paths:
            download_weight(os.path.join('pretrained_models', style_ckpt))
        else:
            sys.exit('{} does not exist'.format(style_ckpt))


    # Check the input image/latent
    print('Check the input image/latent')
    os.makedirs('Inversions/{}'.format(args.embedding_method), exist_ok=True)
    latent_path = os.path.join('Inversions', args.embedding_method, Path(args.input_img).stem + '.npy')
    if not os.path.exists(latent_path):
        aligned_im_path = os.path.join('face_images', 'Aligned', Path(args.input_img).stem + '.png')
        unaligned_im_path = os.path.join('face_images', 'Unaligned', args.input_img)
        if os.path.exists(aligned_im_path) or os.path.exists(unaligned_im_path):
            if not os.path.exists(aligned_im_path):
                predictor_weight = os.path.join('pretrained_models', 'shape_predictor_68_face_landmarks.dat')
                download_weight(predictor_weight)
                predictor = dlib.shape_predictor(predictor_weight)

                face = align_face(unaligned_im_path, predictor, output_size=args.output_size)
                face.save(aligned_im_path)

            if args.embedding_method == 'II2S':
                if not os.path.exists(args.ckpt):
                    download_weight(args.ckpt)
                ii2s = II2S(args)
                latents = ii2s.invert_images(image_path=aligned_im_path, output_dir=None,
                                             return_latents=True, align_input=False, save_output=False)[0]
            elif args.embedding_method == 'e4e':
                if not os.path.exists(os.path.join('pretrained_models', 'e4e_ffhq_encode.pt')):
                    download_weight(os.path.join('pretrained_models', 'e4e_ffhq_encode.pt'))
                latents = e4e_projection_im_path(im_path=aligned_im_path, device=args.device)

            os.makedirs(os.path.join('Inversions', args.embedding_method), exist_ok=True)
            np.save(latent_path, latents.detach().cpu().numpy())

        else:
            sys.exit('Image {} does not exist'.format(args.input_img))
    else:
        latents = torch.from_numpy(np.load(latent_path)).to(args.device)


    # Load finetuned generator
    print('Load finetuned generator')
    generator = Generator(args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier).to(args.device)
    style_ckpt = torch.load(os.path.join('pretrained_models', style_ckpt), map_location=args.device)
    generator.load_state_dict(style_ckpt["g_ema"], strict=True)
    generator.eval()
    style_latent = style_ckpt["style_latent"]

    # Generate output image
    output_latent = latents.clone()
    output_latent[:,  7:, :] = style_latent[:,  7:, :]
    with torch.no_grad():
        output = generator([output_latent], input_is_latent=True)[0][0]


    # Save result
    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)
    toPIL(((output + 1) / 2).cpu().detach().clamp(0, 1)).save(os.path.join(output_folder, 'style_{}_input_{}_{}.png'.format(
                                            Path(args.style_img).stem, Path(args.input_img).stem, args.embedding_method)))
    output_latent_path = os.path.join(output_folder, 'style_{}_input_{}_{}.npy'.format(Path(args.style_img).stem,
                                            Path(args.input_img).stem, args.embedding_method))
    np.save(output_latent_path, output_latent.detach().cpu().numpy())




if __name__ == "__main__":

    parser = ArgumentParser()

    # I/O arguments
    parser.add_argument('--input_img', type=str, default='Yui.jpg',
                        help='Input image')
    parser.add_argument('--style_img', type=str, default='titan_erwin.png',
                        help='Style image')
    parser.add_argument('--embedding_method', default='II2S', choices=['II2S', 'e4e'],
                        help='Embedding method during inference')
    parser.add_argument('--output_folder', type=str, default='output/inference')

    args = parser.parse_args()

    args = Namespace(**vars(args), **vars(face_opts))

    main(args)