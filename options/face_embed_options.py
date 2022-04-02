from argparse import Namespace


opts = Namespace()

# StyleGAN2 setting
opts.size = 1024
opts.ckpt = "pretrained_models/ffhq.pt"
opts.channel_multiplier = 2
opts.latent = 512
opts.n_mlp = 8

# loss options
opts.percept_lambda = 1.0
opts.l2_lambda = 1.0
opts.p_norm_lambda = 1e-3

# arguments
opts.device = 'cuda'
opts.seed = 2
opts.tile_latent = False
opts.opt_name = 'adam'
opts.learning_rate = 0.01
opts.lr_schedule = 'fixed'
# opts.steps = 1300
opts.steps = 1000
opts.save_intermediate = False
opts.save_interval = 300
opts.verbose = False

face_opts = opts