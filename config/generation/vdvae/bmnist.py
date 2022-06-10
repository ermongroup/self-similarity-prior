from config.default_bmnist import get_baseline_bmnist_config

def get_config():
    config = get_baseline_bmnist_config()

    ##### Experiment handling ####
    config.seed = 123
    config.n_devices = 2
    config.wandb_dir = './checkpoints/wandb/'
    config.project_name = 'cifar'
    config.user = 'user' # wandb user
    config.exp_name = "vdvae_cifar" # log name to wandb. if placeholder `config.exp_name` is used instead.
    config.run_id = ""

    ##### Checkpointing #####
    config.restore_from_checkpoint = True
    config.checkpoint_dir = f'./checkpoints/'
    config.load_step = 0
    config.ckpt_steps = 10000    

    ##### Model #####
    config.model = "VDVAE"
    config.loss_fn =  'logistic' 
    config.no_bias_above = 64
    config.enc_blocks = "28x1,28d4,7x1,7d7,1x1" # "28x2,28d2,14x3,14d2,7x5"
    config.dec_blocks = "1x1,7m1,7x2,28m7,28x1"
    
    config.enc_widths = None
    config.dec_widths = None
    config.enc_default_width = 64
    config.dec_default_width = 64
    config.n_fourier_enc = ()
    config.use_fourier_enc = False
    config.n_fourier_dec = ()
    config.use_fourier_dec = False 
    config.enc_bottleneck_factor = 1.
    config.dec_bottleneck_factor = 1.
    config.dec_latent_dim = 16
    config.no_bias_above = 64
    config.residual_posteriors = True

    config.superres_eval_factors = []

    return config
