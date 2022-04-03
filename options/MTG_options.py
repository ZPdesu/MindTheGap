from argparse import ArgumentParser
class MTGOptions(object):


    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):

        self.parser.add_argument(
            "--frozen_gen_ckpt",
            type=str,
            default="pretrained_models/ffhq.pt",
            help="Path to a pre-trained StyleGAN2 generator for use as the initial frozen network. " \
                 "If train_gen_ckpt is not provided, will also be used for the trainable generator initialization.",
        )

        self.parser.add_argument(
            "--train_gen_ckpt",
            type=str,
            help="Path to a pre-trained StyleGAN2 generator for use as the initial trainable network."
        )
        self.parser.add_argument(
            "--device",
            type=str,
            default="cuda",
        )

        self.parser.add_argument(
            '--seed',
            type=int,
            default=2)
        #################################################################
        # Loss Setting
        #################################################################
        self.parser.add_argument(
            "--clip_across_lambda",
            type=float,
            default=1.0,
        )
        self.parser.add_argument(
            "--ref_clip_lambda",
            type=float,
            default=30,
        )
        self.parser.add_argument(
            "--l2_lambda",
            type=float,
            default=10,
        )
        self.parser.add_argument(
            "--lpips_lambda",
            type=float,
            default=10,
        )

        self.parser.add_argument(
            "--clip_within_lambda",
            type=float,
            default=0.5,
        )
        #################################################################
        # Clip Setting
        #################################################################
        self.parser.add_argument(
            "--lambda_direction",
            type=float,
            default=1.0,
            help="Strength of directional clip loss",
        )

        self.parser.add_argument(
            "--lambda_patch",
            type=float,
            default=0.0,
            help="Strength of patch-based clip loss",
        )

        self.parser.add_argument(
            "--lambda_global",
            type=float,
            default=0.0,
            help="Strength of global clip loss",
        )

        self.parser.add_argument(
            "--lambda_texture",
            type=float,
            default=0.0,
            help="Strength of texture preserving loss",
        )

        self.parser.add_argument(
            "--lambda_manifold",
            type=float,
            default=0.0,
            help="Strength of manifold constraint term"
        )
        self.parser.add_argument(
            "--clip_models",
            nargs="+",
            type=str,
            default=["ViT-B/32", "ViT-B/16"],
            help="Names of CLIP models to use for losses"
        )

        self.parser.add_argument(
            "--clip_model_weights",
            nargs="+",
            type=float,
            default=[1.0, 1.0],
            help="Relative loss weights of the clip models"
        )
        ##############################################################
        ##############################################################

        self.parser.add_argument(
            "--save_interval",
            type=int,
            default=300,
            help="How often to save a model checkpoint. No checkpoints will be saved if not set.",

        )

        self.parser.add_argument(
            "--vis_interval",
            type=int,
            default=50,
            help="How often to save an output image",
        )

        # Used for manual layer choices. Leave as None to use paper layers.
        self.parser.add_argument(
            "--phase",
            help="Training phase flag"
        )

        self.parser.add_argument(
            "--num_grid_outputs",
            type=int,
            default=0,
            help="Number of paper-style grid images to generate after training."
        )

        self.parser.add_argument(
            "--iter", type=int, default=600, help="total training iterations"
        )
        self.parser.add_argument(
            "--batch", type=int, default=4, help="batch sizes for each gpus"
        )

        self.parser.add_argument(
            "--n_sample",
            type=int,
            default=4,
            help="number of the samples generated during training",
        )

        self.parser.add_argument(
            "--g_reg_every",
            type=int,
            default=4,
            help="interval of the applying path length regularization",
        )
        self.parser.add_argument(
            "--size", type=int, default=1024, help="image sizes for the model"
        )

        self.parser.add_argument("--lr", type=float, default=0.002, help="learning rate")

        self.parser.add_argument(
            "--channel_multiplier",
            type=int,
            default=2,
            help="channel multiplier factor for the model. config-f = 2, else = 1",
        )

        self.parser.add_argument(
            "--sample_truncation",
            default=0.5,
            type=float,
            help="Truncation value for sampled test images."
        )

        self.parser.add_argument(
            "--use_truncation_in_training",
            action='store_true'
        )

        self.parser.add_argument(
            "--verbose",
            action='store_true'
        )

    def parse(self):
        opts = self.parser.parse_args()

        if len(opts.clip_models) != len(opts.clip_model_weights):
            raise ValueError("Number of clip model names must match number of model weights")

        opts.train_gen_ckpt = opts.train_gen_ckpt or opts.frozen_gen_ckpt

        return opts



