
import os
import json
import numpy as np
import torch
from tqdm import tqdm
import torchvision
from pathlib import Path
from models.MTG import MTG
from utils.file_utils import save_images
from options.MTG_options import MTGOptions
toPIL = torchvision.transforms.ToPILImage()

def train(args, output_dir):
    # Set up networks, optimizers.
    print("Initializing networks...")
    net = MTG(args)
    style_latent = net.embed_style_img(args.style_img)
    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    g_optim = torch.optim.Adam(
        net.generator_trainable.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    sample_dir = os.path.join(output_dir, "sample")
    ckpt_dir = os.path.join(output_dir, "checkpoint")

    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Set fixed sample
    fixed_z = torch.randn(args.n_sample, 512, device=args.device)
    save_image(net, fixed_z, args, sample_dir, -1)


    # Training loop
    pbar = tqdm(range(args.iter))
    for i in pbar:

        net.train()

        sample_z = torch.randn(args.batch, 512, device=args.device)

        if args.use_truncation_in_training:
            [_, _, _, _], loss = net([sample_z], truncation=args.sample_truncation)
        else:
            [_, _, _, _], loss = net([sample_z])

            # print('args.use_truncation_in_training', args.use_truncation_in_training)

        net.zero_grad()
        loss.backward()
        g_optim.step()
        pbar.set_description(f"Finetuning Generator | Total loss: {loss}")

        if (i + 1) % args.vis_interval == 0 or (i + 1) == args.iter:
            save_image(net, fixed_z, args, sample_dir, i)

        if args.save_interval is not None and ((i + 1) % args.save_interval == 0 or (i + 1) == args.iter):
            ckpt_name = '{}/{}.pt'.format(ckpt_dir, str(i + 1).zfill(6))
            save_checkpoint(net, g_optim, style_latent, ckpt_name)

    ckpt_name = '{}/{}.pt'.format(ckpt_dir, Path(args.style_img).stem)
    save_checkpoint(net, g_optim, style_latent, ckpt_name)




def save_image(net, fixed_z, args, sample_dir, i):
    net.eval()
    with torch.no_grad():
        [sampled_src, sampled_dst, rec_dst, without_color_dst], loss = net([fixed_z],
                                                                           truncation=args.sample_truncation,
                                                                           inference=True)
        grid_rows = int(args.n_sample ** 0.5)
        save_images(sampled_dst, sample_dir, "dst", grid_rows, i+1)
        save_images(without_color_dst, sample_dir, "without_color", grid_rows, i+1)
        toPIL(((rec_dst[0] + 1) / 2).cpu().detach().clamp(0, 1)).save(f"{sample_dir}/rec_{i + 1}.png")

def save_checkpoint(net, g_optim, style_latent, ckpt_name):
    torch.save(
        {
            "g_ema": net.generator_trainable.generator.state_dict(),
            "g_optim": g_optim.state_dict(),
            "latent_avg": net.generator_trainable.mean_latent,
            "style_latent": style_latent

        },
        ckpt_name
    )






if __name__ == "__main__":

    option = MTGOptions()
    parser = option.parser

    # I/O arguments
    parser.add_argument('--style_img', type=str, default='moana.jpg',
                        help='Style image')
    parser.add_argument('--output_dir', type=str, default='output/train')
    args = option.parse()
    output_dir = os.path.join(args.output_dir, Path(args.style_img).stem)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "args.json"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    train(args, output_dir)